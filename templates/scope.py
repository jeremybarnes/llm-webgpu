from typing import NamedTuple, Any, Set, Dict, Optional, Mapping, Type, Sequence, Union, cast, List, TYPE_CHECKING
from .path import Index, Path
from .symbol import Symbol, SymbolRef
from .expression import Expression
from .symbol_types import DimT, RealT
from .tensor_types import Tensor, DimListT
from abc import abstractmethod

if TYPE_CHECKING:
    from .try_scope import TryScope
    from .named_scope import NamedScope
    

class Value(NamedTuple):
    """
    Internal representation of the value of a symbol.  Includes the
    symbol, its type, the value, and the aliases.
    """
    symbol: Symbol
    value: Any
    aliases: Set[SymbolRef]

    def __repr__(self) -> str:
        return (type(self.symbol).typename + "="
                + ('?' if self.value is None else self.symbol.debug_str(self.value))
                + (str(self.aliases) if len(self.aliases) > 0 else ''))

    def __str__(self) -> str:
        return self.__repr__()

class MappedValueNode(NamedTuple):
    children: Dict[Index, Any] = {}
    value: Optional[Value] = None
    
# Type of the return of mapped_values()
MappedValues = Mapping[Index, MappedValueNode]
    
class Scope(Symbol):
    """
    Base class for a lexical scope in which symbols have values.  Scopes
    do contain state, which is the only exception to the rule that Symbols
    don't contain state (if this wasn't the case, it would not be possible
    for any state to exist anywhere in the system at all!)
    """

    is_structured = True
    
    def __init__(self, idx: Index, parent: Optional[Symbol] = None):
        super().__init__(idx, parent)
        
    @abstractmethod
    def symbol(self, tp: Type[Symbol], idx: Index) -> Symbol:
        """
        Create and register a new symbol in this scope, of the given type,
        with the given (local) name, and the given index.  This method will
        set up the parent properly and make sure the name is adjusted so
        that it's inside the current scope.
        """
        pass

    def tensor(self, idx: Index) -> 'Tensor':
        """
        Add a symbolic tensor with the given name, data type and shape
        (which can be unknown).
        """
        return cast(Tensor, self.symbol(Tensor, idx))

    def dim(self, idx: Index) -> 'DimT':
        """
        Add a symbolic dimension with the given name.
        """
        return cast(DimT, self.symbol(DimT, idx))

    def shape(self, idx: Index) -> 'DimListT':
        """
        Add a symbolic dimension with the given name.
        """
        return cast(DimListT, self.symbol(DimListT, idx))

    def real(self, idx: Index) -> 'RealT':
        """
        Add a symbolic real number to the graph.
        """
        return cast(RealT, self.symbol(RealT, idx))
        
    def enter(self, name: str) -> 'NamedScope':
        """
        Enter into a named scope, with all changes relayed through the
        scope object.
        """
        from .named_scope import NamedScope
        return NamedScope(self, name)

    def resolve(self, symbol: Symbol, value: Any, resolving: Optional[Set[SymbolRef]] = None) -> None:
        """
        We now know the value of the given symbol.  Return the new value.
        It will throw an exception if the resolution can't happen.
        """
        assert isinstance(symbol, Symbol)

        # We are already resolving this; cut off the recursive loop
        if resolving is not None and symbol.ref() in resolving:
            return

        if isinstance(value, Symbol):
            # Value resolves to the value of another symbol
            # An alias implies x = y and y = x
            self._alias(symbol, value)
            return

        if isinstance(value, Expression):
            # Value resolves to an expression over the values of other
            # symbols.
            self._expression(symbol, value, resolving)
            return

        if not isinstance(value, symbol.tp):
            value = symbol.coerce_value(value)

        assert isinstance(value, symbol.tp), "Attempt to resolve symbol of wrong type"

        # Although the public interface to resolve is on the scope, we
        # implement it per-value as it needs to be overriden for each of
        # the possible Symbol sub-classes to deal with structures.
        symbol._resolve(self, value, resolving)

    @abstractmethod
    def value(self, symbol: Symbol) -> Any:
        """
        Return the current value of the given symbol.
        """
        pass

    @abstractmethod
    def _alias(self, sym1: Symbol, sym2: Symbol) -> None:
        """
        The symbol resolves to the value of another symbol.  This is a more
        complex situation, as it creates a dependency graph that needs to
        be resolved if possible, or it can create a match failure if the
        different values already resolved to different things.
        """
        pass

    @abstractmethod
    def aliases(self, s: Symbol) -> Set[SymbolRef]:
        pass

    @abstractmethod
    def children(self, s: Symbol) -> Set[SymbolRef]:
        """
        Return all child symbols of the symbol s.
        """
        pass
    
    def _expression(self, symbol: Symbol, value: Expression, resolving: Optional[Set[SymbolRef]]):
        """
        This tells us that the given symbol is equal to the result of
        the given expression.
        """
        raise TypeError('Expressions may only be added in a mutate block')
    
    @abstractmethod
    def resolved(self, symbol: Symbol) -> bool:
        """
        Tell us if the given symbol is resolved (has a concrete value) in
        the scope of the graph.
        """
        pass
        
    def condition(self, name: str, expr: Expression):
        """
        Add the given condition to the system, such that it won't be resolved
        unless the condition is true.
        """
        self._condition(name, expr)

    def require(self, pred: Expression):
        """
        Require that the predicate be true.
        """
        self._condition('require_' + str(pred), pred)
        
    def _condition(self, name: str, expr: Expression):
        """
        Internal implementation method for condition.
        """
        raise TypeError('Conditions may only be added in a mutate block')
        
    def resolve_parent(self, symbol: Symbol, value, resolving: Optional[Set[SymbolRef]]):
        """
        One of the child fields was resolved of the given symbol.
        """
        #print('resolve_parent', symbol, value)
        if symbol.parent is not None:
            symbol.parent._resolve_child(self, symbol, value, resolving)
        
    def _resolve_child(self, scope, child, value: Any, resolving: Optional[Set[SymbolRef]]) -> None:
        """
        A scope does nothing when its child is resolved.
        """
        pass
        
    def mutate(self, transaction_name: str) -> 'TryScope':
        """
        Return a scope that can be used to modify values inside this scope
        atomically.
        """
        raise TypeError('Scope of type {} cannot be mutated'.format(str(type(self))))
        
    def _commit_from(self, values: Sequence['Value']):
        """
        Update the internal values from the given set of committed values.
        """
        raise TypeError("Can't commit to Scope of type {}"
                        .format(type(self)))

    def partial_value(self, symbol: Union[Symbol,Any]) -> Any:
        """
        Returns a partial value of the given symbol: that is, the value of
        the symbol as a structure with any of its resolved components
        (recursively) given by their own partial value.

        Something that is resolved will return the resolved value.

        Something that is not a symbol will return its argument.
        """

        if not isinstance(symbol, Symbol):
            return symbol
            raise TypeError('Non-symbol passed to partial_value: {}', symbol)        

        # If it's resolved, return the value
        if self.resolved(symbol):
            return self.value(symbol)

        # Let the symbol return its own partial value
        return symbol._partial_value(self)
        
    def values(self) -> Mapping[SymbolRef, Value]:
        """
        Return all fully-resolved values for this symbol.
        """
        return {}

    def mapped_values(self) -> MappedValues:
        """
        Mapped version of values with symbols resolved.
        """

        result: Dict[Index, MappedValueNode] = {}

        def add_value(sym: SymbolRef, val: Value):
            p = sym.path()
            d: Dict[Index, MappedValueNode] = result
            for i in range(len(p)):
                idx = p[i]
                if idx not in d:
                    d[idx] = MappedValueNode({}, None)

                node = d[idx]

                if i == len(p) - 1:
                    # insert the value
                    d[idx] = MappedValueNode(node.children, val)
                else:
                    d = cast(Dict[Index, MappedValueNode], node.children)

        for sym,val in self.values().items():
            add_value(sym, val)

        return result
    
    def dump(self, focus_in: Optional['Scope'] = None) -> int:
        """
        Dump the current state all the way down to the root scope.
        """

        focus: Scope = self if focus_in is None else focus_in
            
        if self.parent is not None:
            indent = cast(Scope, self.parent).dump(focus)
        else:
            indent = 0

        spaces = ' ' * indent
        print(spaces + 'SCOPE ' + str(self.idx))
        #for k,v in self.values().items():
        #    print(spaces + ' ' + str(k) + ': ' + str(v))

        def print_val(k: Index, v: MappedValueNode, i: int, prefix: List[Index], current: Path):
            spaces = ' ' * i

            if v.value is None and len(v.children) == 1:
                # Can print by adding a prefix
                k2 = list(v.children.keys())[0]
                print_val(k2, v.children[k2], i, prefix + [k], current + k2)
                return

            pstr = '' if len(prefix) == 0 else (','.join(repr(p) for p in prefix)) + ','

            if v.value is not None:
                if v.value.value is None:
                    if focus.resolved(v.value.symbol):
                        val_repr = '...'
                    else:
                        val_repr = '?'
                else:
                    val_repr = str(v.value.value)

                print(spaces + '  VAL ' + pstr + repr(k) + ': ' + type(v.value.symbol).typename + '=' + val_repr)

                for a in v.value.aliases:
                    if a.path() != current:
                        print(spaces + '    ALIAS ' + repr(a.path()))
            else:
                print(spaces + '  PREFIX ' + pstr + repr(k))

            for k2,v2 in v.children.items():
                print_val(k2,v2,i+2, [], current + k2)
            
        vals = self.mapped_values()

        for k2,v2 in vals.items():
            print_val(k2,v2,indent,[], Path([k2]))
            
        return indent + 2

