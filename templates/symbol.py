from typing import Type, Optional, Any, Set, List, TYPE_CHECKING
from .path import Index, Path

if TYPE_CHECKING:
    from .scope import Scope

class Symbol(object):
    """
    A template symbol that can eventually be resolved to hold a
    concrete value of a given type, for example a data type or a
    tensor.

    This never actually stores its own value, simply a name.  That name
    is resolved by the scope to identify its actual current value.
    """
    typename: str
    is_structured: bool = False
    tp: Type
    
    def __init__(self, idx: Index,
                 parent: Optional['Symbol'] = None):
        if parent is not None and not isinstance(parent, Symbol):
            raise TypeError('Non-symbol parent name {} of type {} passed to Symbol'
                            .format(parent, type(parent)))
        
        self.parent = parent
        self.idx = idx

    def path(self, within: Optional['Scope'] = None) -> Path:
        """
        Return the path of this symbol within the given scope (or the global path
        if the within argument is None:
        """
        current: Symbol = self
        result: List[Index] = [self.idx]
        while current.parent is not None and current.parent is not within:
            if current.idx is None:
                raise TypeError('wrong index')
            result = [current.parent.idx] + result
            current = current.parent

        return Path(result)
        
    def __repr__(self) -> str:
        return '[' + ','.join([repr(p) for p in self.path()]) + ']:' + self.typename

    def __str__(self) -> str:
        return '.'.join([str(p) for p in self.path()]) + ':' + self.typename

    def __key(self) -> tuple:
        return (self.path(), self.typename)

    def ref(self) -> 'SymbolRef':
        """
        Return a reference to this symbol.  That's the object that is used to
        perform internal operations rather than build up expressions around
        symbols (which Symbol does).

        For example, sym1 == sym2 returns an Expression testing equality of the
        two symbols, whereas sym1.ref() == sym2.ref() returns True if the two
        symbols are one and the same and False if not.
        """
        return SymbolRef(self)

    def debug_str(self, value: Any) -> str:
        """
        Truncated string for debugging.
        """
        return repr(value).replace('\n',' ')
    
    def coerce_value(self, value: Any) -> Any:
        """
        Coerce the value into the correct type for the value.
        """
        # Don't create a new object unnecessarily
        if not isinstance(value, self.tp):
            value = self.tp(value)
        return value

    def _resolve(self, scope, value: Any, resolving: Optional[Set['SymbolRef']]):
        """
        Default resolve method for Symbol.  Forwards to the scope.
        """
        return scope._resolve(self, value, resolving)

    def _post_resolve(self, scope, value: Any, resolving: Optional[Set['SymbolRef']]) -> None:
        """
        Default post-resolution for a symbol.  This is called once the
        value is set for this symbol, to enable resolution of the child
        values.  No-op here; it needs to be overridden for structured
        symbols.
        """
        pass
    
    def _resolve_child(self, scope, child, value: Any, resolving: Optional[Set['SymbolRef']]):
        """
        Default method called when a parent symbol has its children resolved.
        """
        raise TypeError('_resolve_child called on non-structured type {}'
                        .format(type(self)))

    def _partial_value(self, scope):
        """
        Return the partially resolved value of the symbol within the scope.
        The default works for non-structured symbols, simply returning the
        symbol is fine (the caller takes care of returning a value if there
        already is one).

        The postcondition on this method is that
        scope.resolve(sym, scope.partial_value(sym))
        is a no-op.
        """
        
        return self
    
    @staticmethod
    def incompatible(val1: Any, val2: Any) -> bool:
        """
        Generic incompatible: two things are incompatible if their values
        are not equal.
        """
        if val1 is None or val2 is None:
            return False
        return val1 != val2

    @staticmethod
    def compatible_type(type1: type, type2: type) -> Optional[type]:
        """
        Are these two types compatible?  Normally they are only compatible
        if the types are equal, but some types (like 'any') are compatible
        with more than just themselves.
        """
        return type1 if type1 == type2 else None


class SymbolRef(object):
    """
    Class that we use when we want to refer to a symbol as a symbol, not as
    a value, particularly when we want operators like equality to mean "the
    same symbol" and not "an expression that compares the equality of the
    values of the two symbols".
    """

    def __init__(self, symbol: Symbol):
        self._symbol: Symbol = symbol

    def sym(self) -> Symbol:
        return self._symbol

    def name(self) -> str:
        return str(self._symbol.path())
    
    def __key(self):
        #print('key: path', self._symbol.path(), 'typename', self._symbol.typename)
        return (self._symbol.path(), self._symbol.typename)

    def __hash__(self) -> int:
        return hash(self.__key())

    def __repr__(self) -> str:
        return 'SymbolRef(' + repr(self._symbol) + ')'

    def __str__(self) -> str:
        return '&' + str(self._symbol)
    
    def __eq__(self, other) -> bool:
        return self.__key() == other.__key()

    def __ne__(self, other) -> bool:
        return self.__key() != other.__key()

    def path(self) -> Path:
        return self._symbol.path()

    
