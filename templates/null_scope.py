from typing import Type, Any, Optional, Set, Sequence, Mapping
from .exceptions import MatchError
from .path import Index, Path
from .symbol import Symbol, SymbolRef
from .scope import Scope, Value
from .try_scope import TryScope
from .expression import Expression

class NullScope(Scope):
    """
    Scope in which nothing is defined.  Useful for testing.
    """
    typename = 'null_scope'
    tp = Type['NullScope']
    
    def __init__(self, name: Index = "<<root>>"):
        super().__init__(name)
    
    def resolve(self, symbol: Symbol, value: Any, resolving: Optional[Set[SymbolRef]] = None) -> None:
        """
        We now know the value of the given symbol.  Return the new value.
        It will throw an exception if the resolution can't happen.
        """
        raise MatchError("no resolution in NullScope")

    def resolved(self, symbol: Symbol) -> bool:
        """
        Tell us if the given symbol is resolved (has a concrete value) in
        the scope of the graph.
        """
        return False
    
    def aliases(self, symbol: Symbol) -> Set[SymbolRef]:
        return {symbol.ref()}

    def children(self, s: Symbol) -> Set[SymbolRef]:
        return set()

    def _alias(self, sym1: Symbol, sym2: Symbol) -> None:
        raise NotImplemented('NullScope cannot alias')

    def symbol(self, tp: Type[Symbol], idx: Index) -> Symbol:
        raise NotImplemented('NullScope cannot have symbols')
        
    def value(self, symbol: Symbol) -> Any:
        raise NotImplemented('NullScope symbols have no value')
