from typing import Type, Optional, Set, Any, Union, List, Dict, TYPE_CHECKING
from .exceptions import MatchError
from .path import Index, Path, _private
from .symbol import Symbol, SymbolRef
from .operators import _make_logical, _make_arithmetic, EqualityComparable
from .expression import Expression
from functools import reduce

if TYPE_CHECKING:
    from .scope import Scope


Logical: Type[Symbol] = _make_logical('Logical', Symbol)
Arithmetic: Type[Symbol] = _make_arithmetic('Arithmetic', Symbol)


class BoolT(Logical):
    """
    Type specialization for a boolean (either True or False).
    """
    typename='bool'
    tp = bool
    
class DimT(Arithmetic):
    """
    Type specialization for a dimension (an integer that's zero or higher).
    """
    typename = 'dim'
    tp = int

class RealT(Arithmetic):
    """
    Type specialization for a symbol that represents a real number.
    """
    typename = 'real'
    tp = float

class SymbolArray(Symbol):
    """
    This models an array of symbols, where rather than the whole array being
    resolvable or not, each of the elements (and the length) are individually
    symbolic.
    """
    is_structured = True
    tp = list
    inner: Type[Symbol]
    LEN_IDX = _private('len')
    
    def __init__(self, 
                 idx: Index,
                 parent: Optional[Symbol] = None):
        super().__init__(idx, parent)
        self._len = DimT(self.LEN_IDX, self)

    def _resolve(self, scope: 'Scope', value: list, resolving: Optional[Set[SymbolRef]] = None):
        """
        Resolve overide for array.  This is needed because it's possible to
        resolve with an array of symbols and values, which may not resolve
        the array itself if there are unresolved symbols in it.
        """
        if not isinstance(value, list):
            value = list(value)

        self._post_resolve(scope, value, resolving)
            
        return self

    def _post_resolve(self, scope: 'Scope', value: Any, resolving: Optional[Set[SymbolRef]]) -> None:
        """
        Post-resolution for the array.  We ensure that it's possible to resolve
        all of the constituents.
        """
        #print('_post_resolve array', self, value, scope.values())

        ends_with_ellipsis = len(value) > 0 and value[-1] is ...

        with scope.mutate('post_resolve_array') as inner:
            # First resolve the number of dimensions
            if ends_with_ellipsis:
                value.pop()
            else:
                inner.resolve(self._len, len(value), resolving)
            
            # If this passed, the length is valid
            for i,v in zip(range(len(value)),value):
                #print('array resolving', i, 'to', v)
                if v is ...:
                    raise MatchError('Array ellipsis can only occur at last position: {}'
                                     .format(value))
                inner.resolve(self[i], v)

            # If it all succeeds, we can commit the changes
            inner.commit(resolving)

    def _resolve_child(self, scope: 'Scope', child: Symbol, value: Any,
                       resolving: Optional[Set[SymbolRef]]):
        #print('** array', self.name, 'resolved child', child.name, child.idx, value)
        #if scope.resolved(self):
        #    print('self resolved', scope.value(self))
            
        
        with scope.mutate('array_child') as inner:
            if child.idx == self.LEN_IDX:
                # resolved length
                # Nothing to do, as length is resolved by the child
                mylen = int(value)
            elif scope.resolved(self._len):
                mylen = scope.value(self._len)

                assert isinstance(child.idx, int)
                assert child.idx >= 0
                assert child.idx < mylen

                if not inner.resolved(self[child.idx]):
                    inner.resolve(self[child.idx], value, resolving)
            else:
                # length is not resolved; we're not done.  This normally
                # happens when we use an ellipsis to indicate that some
                # values are set but the length is not yet resolved.
                return
                    
            if inner.resolved(self):
                #print('self resolved', inner.value(self))
                return
            #print('self not resolved')
            
            for i in range(mylen):
                if not inner.resolved(self[i]):
                    return

            # The whole thing is resolved; resolve to the final value
            inner._resolve(self, [inner.value(self[i]) for i in range(mylen)],
                           resolving, True)

            inner.commit()
            
    def _partial_value(self, scope: 'Scope') -> list:
        """
        An array's partial_value implementation is particularly complicated, as
        depending upon what is resolved (length versus some items versus all
        items), it needs to include Free values, ellipsis, etc.
        """
        children = scope.children(self)
        indexes: List[Index] = [ch.sym().idx for ch in children]

        #print('children of', self, children)
        
        #for ch in children:
        #    print('  ', ch.sym(), ch.sym().parent, ch.sym().idx)

        # We only have Private and int indexes in an array, so this simply selects
        # the maximum index
        max_idx = reduce(max, [i for i in indexes if isinstance(i, int)], -1)
        result: List[Union[None,Any]] = [None] * (max_idx + 1)
        
        for ref in children:

            sym = ref.sym()

            if not isinstance(sym.idx, int):
                # equivalent to `if sym.idx == self.LEN_IDX:` but type-checks better
                continue

            val = scope.partial_value(sym)

            #print('child', sym, 'val', val)
            
            result[sym.idx] = scope.partial_value(val)

        if scope.resolved(self._len):
            length = scope.value(self._len)
            if len(result) > length:
                raise RuntimeError('length of array is not congruent with values')
            result += [None] * (length - len(result))
        else:
            result += [...]    

        # Replace any free variables with something from their alias
        # set, so that they will resolve properly when attempted.
        # TODO: if there is more than one alias, choose the simplest one.
        for i in range(len(result)):
            if result[i] is None:
                result[i] = self[i]
            
        #print('result', result)
            
        return result
        
    def debug_str(self, value: Any) -> str:
        def pr(v: Any) -> str:
            return repr(v) if v is not ... else '...'
        return super().debug_str('[' + ','.join([pr(v) for v in value]) + ']')

    def __getitem__(self, key: Union[int,str]) -> Symbol:
        if key == 'len' or key == self.LEN_IDX:
            return self._len
        if not isinstance(key, int):
            raise TypeError(("attempt to index array symbol on non-int {} "
                            + "(only 'len' or integers 0..len valid)")
                            .format(key))
        return self.inner(key, self)

    def __len__(self):
        return self.len

    def __iter__(self):
        return NotImplemented


class TypeT(Symbol):
    """
    Symbolic type that represents the contained type of an Any value.
    """
    typename='type'
    tp=type


class AnyT(EqualityComparable):
    """
    Symbolic type that represents a parameter value for an operation.
    """
    is_structured = True
    typename='any'
    tp=object
    TYPE_IDX=_private('type')
    LEN_IDX=_private('len')
    
    def __init__(self, idx: Index, parent: Symbol = None):
        super().__init__(idx, parent)
        self.type = TypeT(self.TYPE_IDX, self)
        self.len = DimT(self.LEN_IDX, self)
        
    def _resolve(self, scope: 'Scope', value: Any, resolving: Optional[Set[SymbolRef]]):

        with scope.mutate('any_resolve') as inner:
            inner.resolve(self.type, type(value))

            if isinstance(value, list):
                # Array resolve; look for all of the values
                #print('array resolve', self, value)

                inner.resolve(self.len, len(value))
                
                val = []
                all_resolved = True
                for i in range(len(value)):
                    #print('resolving any array element', i, 'to', value[i])
                    inner.resolve(self[i], value[i])
                    if inner.resolved(self[i]):
                        val.append(inner.value(self[i]))
                    else:
                        val.append(None)
                        all_resolved = False

                #print('array resolve return', all_resolved, val)
                #print(inner.values())
                inner.commit()
                return

            elif isinstance(value, dict):
                # Structured resolve; assume ParameterMap
                print('dict resolve', self, value)
                raise RuntimeError("dict resolve not done")
                pass




            elif isinstance(value, int) and value >= 0:
                #print('dim resolve')
                pass
            elif isinstance(value, float) or isinstance(value, int):
                #print('float resolve')
                pass
            elif isinstance(value, str):
                #print('string resolve')
                pass

            self._post_resolve(inner, value, resolving)

            inner.commit()
        
        return self

    def _post_resolve(self, scope, value: Any, resolving: Optional[Set[SymbolRef]]) -> None:
        """
        Post-resolution for the array.  We ensure that it's possible to resolve
        all of the constituents.
        """
        #print('any', self, 'post resolve', value)
        with scope.mutate('any_post_resolve') as inner:
            inner._resolve(self, value, resolving, True)
            inner.commit(resolving)
    
    def _resolve_child(self, scope: 'Scope', child: Symbol, value: Any,
                       resolving: Optional[Set[SymbolRef]]) -> None:
        """
        One of our child fields (dtype or shape) was resolved.  Since there
        is nothing we need to do in response to this, it's a no-op.

        To make this work, we need to know if the contained value is complete
        or not.  We do so making reference to the length or complete properties.
        """
        my_type = scope.value(self.type)
        #print('my_type', my_type)

        if my_type == list:
            if not scope.resolved(self.len):
                return

            my_len = scope.value(self.len)
            #print('my_len', my_len)
            vals = []

            for i in range(my_len):
                if not scope.resolved(self[i]):
                    return
                vals.append(scope.value(self[i]))

            with scope.mutate('any_resolve_child') as inner:
                inner._resolve(self, vals, resolving, True)
                inner.commit(resolving)

            return
            
        elif my_type == dict:
            pass
        else:
            if child != self.type:
                raise RuntimeError('resolve_child for non-structured Any type {} of type {}'
                                   .format(child, my_type))
    
    @staticmethod
    def compatible_type(type1: Type[Symbol], type2: Type[Symbol]) -> Optional[type]:
        """
        Are these two types compatible?  Normally they are only compatible
        if the types are equal, but some types (like 'any') are compatible
        with more than just themselves.
        """
        return type2

    def __getitem__(self, key: Union[str,int]) -> 'AnyT':
        return AnyT(key, self)


class UnitT(Symbol):
    """
    Unit type that can only hold True.  Used to represent resolved-ness
    without containing a value.
    """
    typename='unit'
    tp=bool
    
    def coerce_value(self, value: Any) -> Any:
        """
        Coerce the value into the correct type for the value.  Only the
        True value works for UnitT, so here we in fact verify it coerces
        to True in a truthfullness context.
        """
        if not value:
            raise TypeError('attempt to resolve UnitT to non-True {}'
                            .format(value))
        return True


class ParameterMapT(Symbol):
    """
    Symbolic version of a map of (string, symbol) parameters.  Enables
    us to match on a set of key/value parameters.
    """
    typename='params'
    is_structured = True
    tp = dict
    COMPLETE_IDX=_private('complete')
    
    def __init__(self, idx: Index,
                 parent: Optional[Symbol] = None):
                 
        super().__init__(idx, parent)
        self.complete = UnitT(self.COMPLETE_IDX, self)

    def _resolve(self, scope: 'Scope', value: dict, resolving: Optional[Set[SymbolRef]]):

        is_complete = scope.resolved(self.complete)

        if resolving is None:
            resolving = {self.ref()}
        else:
            resolving.add(self.ref())
        
        #print('ParameterMapT resolve', value, 'is_complete', is_complete,
        #      'resolving', resolving)

        if is_complete:
            # If it's complete, we can't add or remove any children.  Thus, we
            # need to make sure that the structure of the children is right
            # for those that are still there.
            known_children = { ch.sym().idx for ch in scope.children(self) if ch != self.complete.ref() }
            for k in value.keys():
                if k not in known_children:
                    raise MatchError(('Attempt to add new ParameterMapT key {}={} ' +
                                      'to closed value')
                                     .format(k, value[k]))

            for k in known_children:
                if k not in value:
                    raise MatchError('ParameterMapT key {} redefined to no value'
                                     .format(k))
                    
        with scope.mutate('parameter_map_resolve') as inner:
        
            for k,v in value.items():
                inner.resolve(self[k], v)

            if not is_complete:
                inner.resolve(self.complete, True)

            inner.commit()
            
        return self

    def _resolve_child(self, scope: 'Scope', child: Symbol, value: Any,
                       resolving: Optional[Set[SymbolRef]]):
        if resolving is not None and self in resolving:
            return

        # We don't have a value if not complete
        #print('parameter map resolve_child', self, child, value, resolving)
        if child != self.complete and not scope.resolved(self.complete):
            return

        resolved_values: Dict[Index, Any] = {}

        #print('resolved children', self, type(scope), scope.values(), scope.children(self))
        #for k,v in scope._values.items():
        #    print('  ', k, v.symbol, v.symbol.parent, v.symbol.idx, v)
        
        for ref in scope.children(self):
            if ref == self.complete.ref():
                continue
            sym = ref.sym()
            if not scope.resolved(sym):
                #print('not resolved', k)
                return
            v = scope.value(sym)
            resolved_values[sym.idx] = v

        #print('resolved_values', resolved_values)
            
        # We have all of our values; set our value
        with scope.mutate('param_map_child') as inner:
            # The whole thing is resolved; resolve to the final value
            inner._resolve(self, resolved_values, resolving, True)
            inner.commit()

        
    def __getitem__(self, key: str) -> AnyT:
        return AnyT(key, self)

    
class ExpressionT(Symbol):
    """
    Type representing an expression as a first-class function.
    """
    typename='expr'
    tp=Expression


class ArgumentListT(SymbolArray):
    """
    Type representing a list of symbols to hold the arguments to an
    expression.
    """
    typename = 'args'
    inner = AnyT


class ExpressionValueT(Symbol):
    """
    Internal symbol that represents the value of an expression.  This
    operates by a) collecting the arguments, b) evaluating the expression,
    c) setting the result.
    """
    typename='expr_value'
    is_structured = True
    tp = Type['ExpressionValueT']
    
    def __init__(self, idx: Index, parent: Optional['Scope'] = None):
        super().__init__(idx, parent)
        self.expr = ExpressionT("expr", self)
        self.args = ArgumentListT("args", self)
        
    def _resolve(self, scope: 'Scope', value: list, resolving: Optional[Set[SymbolRef]] = None):
        """
        Resolve overide for ExpressionValue.  Should never be called as
        it should only be resolved once it's children (expression and arguments)
        are resolved.
        """
        pass
        
    def _post_resolve(self, scope, value: Any, resolving: Optional[Set[SymbolRef]]) -> None:
        """
        Post-resolution for the array.  Nothing needs to be done as the resolution
        itself can only happen from _resolve_child.
        """
        pass
        
    def _resolve_child(self, scope: 'Scope', child: Symbol, value: Any,
                       resolving: Optional[Set[SymbolRef]]):
        #print('expr value resolve_child', self, child, value, scope.values())
        
        #if scope.resolved(self):
        #    print('resolved to', scope.value(self))
        
        #if self in resolving:
        #    return

        # We don't have a value if not complete
        def resolved_args():
            return scope.resolved(self.args)
        
        def resolved_expr():
            return child == self.expr or scope.resolved(self.expr)

        if not resolved_args() or not resolved_expr():
            return

        expr = value if child == self.expr else scope.value(self.expr)
        args = value if child == self.args else scope.value(self.args)

        #print('expr', expr, 'args', args)
        
        result = expr.evaluate(args)

        #print('result', result)
        
        # We have all of our values; set our value
        with scope.mutate('expression_value_child') as inner:
            # The whole thing is resolved; resolve to the final value
            inner._resolve(self, result, resolving, True)
            inner.commit()

    @staticmethod
    def compatible_type(type1: Type[Symbol], type2: Type[Symbol]) -> Optional[type]:
        """
        Are these two types compatible?  The output of an expression
        can be any type, which may even depend on the arguments, and
        and so is (theoretically) compatible with any other type.  So
        we return that type here, at least until the expression is
        capable of knowing its return type.
        """
        return type2

class Structure(Symbol):
    """
    A Structure is a symbol into which we can descend to find sub-fields
    (which should all be symbols themselves).
    """
    is_structured = True
