from typing import Type, Any, Optional, Set, Sequence, Mapping, Dict, cast
from .exceptions import MatchError
from .path import Index, Path
from .symbol import Symbol, SymbolRef
from .scope import Scope, Value
from .expression import Expression
from .symbol_types import ExpressionValueT

class TryScope(Scope):
    """
    Scope that will commit its changes atomically, enabling things to
    be tried and to only take effect if all of a sequence of operations
    is successful.
    """
    typename = 'try_scope'
    tp = Type['TryScope']
    
    def __init__(self, owner: Scope, name: str):
        if not isinstance(owner, Scope):
            raise TypeError('TryScope scope argument is not Scope: {} of type {}'
                            .format(owner, type(owner)))
        super().__init__(name, owner)
        self._owner: Scope = owner
        self._name: str = name
        self._values: Dict[Path, Value] = {}
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type: type, exc_val: Any, exc_tb):
        pass

    def mutate(self, transaction_name: str) -> 'TryScope':
        """
        Return a scope which can mutate the current scope.
        """
        return TryScope(self, transaction_name)
        
    def symbol(self, tp: Type[Symbol], idx: Index) -> Symbol:
        """
        Add a symbol with the given name.
        """
        s = tp(idx, self)

        p = s.path()
        if p in self._values:
            v = self._values[p]
            #symboltype = cast(Symbol, type(v.symbol))
            symboltype = type(v.symbol)
            if symboltype != tp:
                raise MatchError('Attempt to redefine symbol {} from type {} to {}'
                                 .format(idx, symboltype.typename, tp.typename))
        else:
            self._values[p] = Value(s, None, {s.ref()})

        return s

    def _my_path(self, sym: Symbol) -> Optional[Path]:
        """
        Determine if the symbol is in our scope; if so return its
        local path.  Returns None if not.
        """
        mine = self.path()
        theirs = sym.path()

        if len(mine) > len(theirs):
            return None

        for i in range(len(mine)):
            if mine[i] != theirs[i]:
                return None

        return theirs

    def _is_mine(self, sym: Symbol) -> bool:
        """
        Is this symbol within the scope of the try block?
        """
        # TODO: this is about the slowest possible way to do it...
        return str(sym.path()).startswith(str(self.path()))
    
    def _resolve(self, symbol: Symbol, value: Any,
                 resolving_in: Optional[Set[SymbolRef]],
                 final: bool = None) -> None:
        """
        Internal method for resolve.  The default symbol method forwards
        to this one.  This will recursively resolve all of the references
        within a system and atomically update the output, or alternatively
        raise an exception and change nothing.

        The final parameter tells us whether the given value is fully
        resolved or not.
        """
        # Resolve flow:
        # 1.  If already resolving, no-op (return)
        # 2.  If it already has a value, verify the value, and return
        # 3.  Resolve the value via the symbol's resolve method
        # 4.  Resolve the values of all aliases

        if final is None:
            final = not symbol.is_structured

        # In resolving, we keep track of what we're already resolving, to avoid
        # infinite recursion.
        resolving: Set[SymbolRef] = set() if resolving_in is None else resolving_in

        values_before = dict(self._values)
            
        # Resolve one of the symbols, keeping a list of which ones to resolve
        # recursively afterwards.
        def resolve_one(symbol: Symbol, value: Any):
            """
            Implementation of a single resolve.
            """
            assert isinstance(symbol, Symbol)
            assert not isinstance(value, SymbolRef)
            
            #print('try resolve one', symbol, 'to value', value, 'in scope', self._name,
            #      'final', final, 'is_structured', symbol.is_structured, 'resolving', resolving)

            if symbol.ref() in resolving:
                # Already in the process of doing this one; we can skip
                return

            # We're now currently doing this one
            resolving.add(symbol.ref())
            
            # Resolve to a symbol is an alias operation
            #print('value', value)
            if isinstance(value, Symbol):
                return self._alias(symbol, value)

            p = symbol.path()
            #print('p', p)
            
            if p in self._values:
                # It already has a value; ensure that it's the same one
                current = self._values[p]
                symboltype = cast(Symbol, type(current.symbol))
                #print('already known with value', current)
                if (current.value is not None
                    and symboltype.incompatible(current.value, value)):
                    raise MatchError('Attempt to redefine {} type {} from {} to {}'
                                     .format(str(symbol), symboltype.typename,
                                             current.value, value))

                # We have resolved our value.  We also need to resolve anything
                # that we alias.  Save the aliases we're about to remove
                aliases = current.aliases
                
                # Since all aliases will be resolved, we now set the aliases to
                # be null
                self._values[p] = Value(symbol, value, set())

                for alias in aliases:
                    #print('resolving alias', alias, 'to', value)
                    self.resolve(alias.sym(), value, resolving)

            elif self._owner.resolved(symbol):
                #print('owner resolved', self._owner.values())
                #o = self._owner
                #while hasattr(o, '_owner'):
                #    o = o._owner
                #    if hasattr(o, 'values'):
                #        print('  ', o.values())
                
                # If it's already resolved by our owner, then we need to simply
                # verify the old and new values are compatible
                current = self._owner.value(symbol)
                symboltype = cast(Symbol, type(symbol))

                if symboltype.incompatible(current, value):
                    raise MatchError('Attempt to redefine {} type {} from {} to {}'
                                     .format(str(symbol), symboltype.typename,
                                             current, value))

            else:
                # Not resolved.  Record the aliases
                #print('not resolved')
                
                mine = self._my_path(symbol)

                # Aliases may come from the owner
                if mine is None:
                    aliases = set(self._owner.aliases(symbol))
                    
                    for a in aliases:
                        assert isinstance(a, SymbolRef)
                else:
                    aliases = set()

                #print('not resolved:', symbol, 'mine', mine, 'value', value, 'aliases', aliases, 'final', final)

                if value is None:
                    self._values[p] = Value(symbol, None, aliases)
                else:
                    self._values[p] = Value(symbol, value if final else None, set())
                #print('value', self._values[nm])

                # Do any symbol-specific post resolution stuff so that child
                # symbols are also resolved.
                symbol._post_resolve(self, value, resolving)

                #print('doing aliases', symbol, aliases)
                for a in aliases:
                    #print('alias', a, type(a), 'mine', symbol.ref(), 'eq', a == symbol.ref())
                    if a != symbol.ref():
                        resolve_one(a.sym(), value)
                #print('done aliases')
                #print('value', self._values[nm])

            #print('doing parent')
            # And tell the parent that its child was resolved, if the value is
            # fully resolved
            if (symbol.parent is not None
                and p in self._values
                and self._values[p].value is not None):
                self.resolve_parent(symbol, self._values[p].value, resolving)
            #print('done parent')

        try:
            resolve_one(symbol, value)
        except:
            # On an exception, clean back up.  This is a bit messy with multiple
            # nested values resolves, but at least gets the job done.
            self._values = values_before
            raise
                
    def _alias(self, sym1: Symbol, sym2: Symbol) -> None:
        """
        The symbol resolves to the value of another symbol.  This is a more
        complex situation, as it creates a dependency graph that needs to
        be resolved if possible, or it can create a match failure if the
        different values already resolved to different things.
        """

        assert isinstance(sym1, Symbol)
        assert isinstance(sym2, Symbol)
        
        #print('alias', sym1, sym2)
        tp = type(sym1).compatible_type(type(sym1), type(sym2))
        if tp is None:
            tp = type(sym2).compatible_type(type(sym2), type(sym1))
        if tp is None:
            raise MatchError('Attempt to alias symbols of different types: {} and {}'
                             .format(sym1, sym2))
        
        resolved1 = self.resolved(sym1)
        resolved2 = self.resolved(sym2)

        #print('resolved1', resolved1, 'resolved2', resolved2)
        
        if resolved1 and resolved2:
            # Verify that the values match
            value1 = self.value(sym1)
            value2 = self.value(sym2)

            #print('value1', value1, 'value2', value2)
            
            if sym1.incompatible(value1, value2):
                raise MatchError('Attempt to alias to different values {} val {} and {} val {}'
                                 .format(sym1, value1, sym2, value2))
        elif resolved1:
            value1 = self.value(sym1)
            self.resolve(sym2, value1)
        elif resolved2:
            value2 = self.value(sym2)
            self.resolve(sym1, value2)
        else:
            # Neither side is resolved, we need to record the alias
            # Create combined list of aliases
            aliases = { sym1.ref(), sym2.ref() }

            if not self._is_mine(sym1):
                aliases |= self._owner.aliases(sym1)
                
            if not self._is_mine(sym2):
                aliases |= self._owner.aliases(sym2)

            p1,p2 = sym1.path(),sym2.path()
            if p1 in self._values:
                aliases |= self._values[p1].aliases

            if p2 in self._values:
                aliases |= self._values[p2].aliases

            #print('aliases', aliases)
            for ref in aliases:
                assert isinstance(ref, SymbolRef)
                sym = ref.sym()
                # Strictly, this check is redundant... but wny not check the
                # invariant while we're at it?
                p = sym.path()
                if (p in self._values
                    and self._values[p].value is not None):
                    raise RuntimeError('Logic error in alias: '
                                       +'simultaneous value and aliases')

                self._values[p] = Value(sym, None, aliases)

    def _expression(self, symbol: Symbol, value: Expression, resolving: Optional[Set[SymbolRef]]):
        """
        This tells us that the given symbol is equal to the result of
        the given expression.
        """

        assert isinstance(symbol, Symbol)
        assert isinstance(value, Expression)
        
        expr = cast(ExpressionValueT, self.symbol(ExpressionValueT, '$' + str(symbol) + '$__expr'))

        #print('expression with values', self.values())
        with self.mutate('try_expr') as inner:
            inner._resolve(symbol, expr, resolving)
            inner._resolve(expr.expr, value, resolving)
            #print('inner values', inner.values())
            inner._resolve(expr.args, value.inputs(), resolving)
            #print('commiting', inner.values())
            inner.commit()

    def _condition(self, name: str, condition: Expression):
        """
        Internal method to implement a condition.
        """

        print('condition', name, condition)
        
        def check_condition(b: bool) -> bool:
            if not b:
                raise MatchError("Condition didn't match: {} {}"
                                  .format(name, condition))
            return True

        all_resolved = all([self.resolved(sym) for sym in condition.inputs()])

        print('all_resolved', all_resolved)
        
        # If there are no inputs, we will never check the expression later so we need
        # to do it now
        if all_resolved:
            check_condition(condition.evaluate([self.value(sym) for sym in condition.inputs()]))
            return
            
        resolving: Set[SymbolRef] = set()
        
        equalities = condition.equalities()

        print('equalities', equalities, condition)
        
        # If we can reduce to a set of equations symbol=expression, then we can decompose
        # into sub-resolutions.
        if equalities is not None:
            with self.mutate('condition ' + name) as inner:
                for lhs,rhs in equalities:
                    inner.resolve(lhs, rhs)
                inner.commit()
        
        expr = cast(ExpressionValueT, self.symbol(ExpressionValueT, '$' + name + '$__expr'))

        check = Expression('check', check_condition, [condition])
        
        with self.mutate('cond_expr') as inner:
            inner._resolve(expr.expr, check, resolving)
            inner._resolve(expr.args, check.inputs(), resolving)

            # TODO: Add these as dependencies of the aliases... how?
            #inner._resolve(symbol, expr, resolving)
            inner.commit()
        
        
    def resolved(self, symbol: Symbol) -> bool:
        """
        Tell us if the given symbol is resolved (has a concrete value) in
        the scope of the graph.
        """
        #print('try resolved', symbol)
        assert isinstance(symbol, Symbol)

        p = symbol.path()
        
        if p in self._values:
            if self._values[p].value is not None:
                return True

        if not self._is_mine(symbol):
            return self._owner.resolved(symbol)
            
        return False

    def value(self, symbol: Symbol) -> Any:
        """
        Return the current value of the given symbol.
        """
        assert isinstance(symbol, Symbol)

        p = symbol.path()
        
        if p in self._values:
            if self._values[p].value is not None:
                return self._values[p].value

        if not self._is_mine(symbol):
            return self._owner.value(symbol)
            
        return None
        
    def commit(self, resolving: Optional[Set[SymbolRef]] = None):
        """
        Take all of the values that are not in our scope and push them down into
        the parent object as a resolution.
        """
        #print('committing', self._name, self.values())

        def filter_value(v: Value) -> Value:
            # Filter aliases to not include anything that's internal to this
            # block.
            new_aliases = { a for a in v.aliases if not self._is_mine(a.sym()) }
            return Value(v.symbol, v.value, new_aliases)

        to_commit = [filter_value(v) for _,v in self._values.items() if not self._is_mine(v.symbol)]
        self._owner._commit_from(to_commit)

    def _commit_from(self, values: Sequence[Value]):
        """
        Update the given committed values.  Must be atomic; either:
        a) no exception is raised, and all values are committed, OR
        b) an exception is raised, and the operation is a no-op (no state
           is modified).
        """
        for value in values:
            self._values[value.symbol.path()] = value

    def aliases(self, s: Symbol) -> Set[SymbolRef]:

        if not isinstance(s, Symbol):
            raise TypeError('aliases argument must be a symbol')

        p = s.path()
        
        if p in self._values:
            #print('aliases 1')
            return self._values[p].aliases
        elif not self._my_path(s):
            #print('aliases 2', s, self.values())
            return self._owner.aliases(s)
        else:
            return set()
                
    def values(self) -> Mapping[SymbolRef, Value]:
        return { v.symbol.ref(): v for k,v in self._values.items() }

    def children(self, s: Symbol) -> Set[SymbolRef]:
        """
        Return all child symbols of the symbol s.
        """
        if not isinstance(s, Symbol):
            raise TypeError('children argument must be a symbol')

        result = self._owner.children(s)

        #print('owner children for', self._name, 'are', result)

        p = s.path()
        
        for k,v in self._values.items():

            #print('matching child', k, 'with path', p)
            
            if k.startswith(p):
                #print('    k', k, '=', v, 'parent', v.symbol.parent)
                if (v.symbol.parent is not None
                    and v.symbol.parent.ref() == s.ref()
                    and not isinstance(v.symbol, Scope)):
                    result.add(v.symbol.ref())
                smb: Optional[Symbol] = v.symbol

                # Some are only implicitly there via their children.  Go
                # back up the parent chain to add any which are missing.
                while (smb is not None):
                    if (smb.parent is not None
                        and smb.parent.ref() == s.ref()
                        and not isinstance(smb, Scope)):
                        result.add(smb.ref())

                        break
                    smb = smb.parent

        #print('children for', self._name, 'returning', result)
                    
        return result


