from typing import Type, Any, Optional, Set, Sequence, Mapping, Dict, List
from .exceptions import MatchError
from .path import Index, Path
from .symbol import Symbol, SymbolRef
from .scope import Scope, Value
from .try_scope import TryScope
from .expression import Expression

class NamedScope(Scope):
    """
    Scope manager that is a named scope inside another scope.
    """
    typename = 'named_scope'
    tp = Type['NamedScope']
    
    def __init__(self, owner: Scope, name: Index):
        super().__init__(name, owner)
        self._owner: Scope = owner
        self._name = name
        self._values: Dict[Path, Value] = {}
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def mutate(self, transaction_name: str) -> TryScope:
        """
        Return a scope which can mutate the current scope.
        """
        return TryScope(self, transaction_name)

    def symbol(self, tp: Type[Symbol], idx: Index) -> Symbol:
        """
        Add a symbol with the given name.
        """
        s = tp(idx, self)

        #print('symbol: tp', tp, 'name', name, 's', s)
        p = Path([idx])
        
        if p in self._values:
            v = self._values[p]
            if type(v.symbol) != tp:
                raise MatchError('Attempt to redefine symbol {} from type {} to {}'
                                 .format(idx, type(v.symbol).typename, tp))
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

        return theirs[len(mine):]        
        
    def _is_mine(self, sym: Symbol) -> bool:
        """
        Tell us if this is our symbol or not.
        """
        return self._my_path(sym) is not None
    
    def resolved(self, s: Symbol) -> bool:
        """
        Tell us if the given symbol is resolved (has a concrete value) in
        the scope of the graph.
        """

        path = self._my_path(s)

        if path is None:
            return self._owner.resolved(s)

        if path not in self._values:
            return False

        return self._values[path].value is not None

    def _resolve(self, symbol: Symbol, value: Any, resolving: Optional[Set[SymbolRef]]) -> Symbol:
        """
        Internal method for resolve.  The default symbol method forwards
        to this one.
        """
        raise RuntimeError("NamedScope can only resolve via TryScope")

    def _alias(self, symbol: Symbol, value: Symbol) -> None:
        """
        The symbol resolves to the value of another symbol.  This is a more
        complex situation, as it creates a dependency graph that needs to
        be resolved if possible, or it can create a match failure if the
        different values already resolved to different things.
        """
        raise RuntimeError("NamedScope can only alias via TryScope")

    def value(self, s: Symbol) -> Any:
        """
        Return the value of the given symbol.
        """

        path = self._my_path(s)

        if path is None:
            return self._owner.value(s)

        return self._values[path].value

    def aliases(self, s: Symbol) -> Set[SymbolRef]:
        """
        Returns all known aliases of the given symbol.
        """
        path = self._my_path(s)

        if path is None:
            return self._owner.aliases(s)

        if path in self._values:
            return self._values[path].aliases

        return {s.ref()}
    
    def values(self) -> Mapping[SymbolRef, Value]:
        return { value.symbol.ref(): value for subpath,value in self._values.items() }

    def children(self, s: Symbol) -> Set[SymbolRef]:
        """
        Return all child symbols of the symbol s.
        """

        if not self._is_mine(s):
            return self._owner.children(s)

        result: Set[SymbolRef] = set()
        
        for k,v in self._values.items():
            #print('child', k, v, v.symbol.parent)
            if (v.symbol.parent is not None
                and v.symbol.parent.ref() == s.ref()):
                result.add(v.symbol.ref())

        return result

    def _commit_from(self, values: Sequence[Value]):
        """
        Update the given committed values.  Must be atomic; either:
        a) no exception is raised, and all values are committed, OR
        b) an exception is raised, and the operation is a no-op (no state
           is modified).
        """

        # First, split into things to be committed into the owner vs things
        # we take into this scope

        owner_commits: List[Value] = []
        my_commits: Dict[Path, Value] = {}
        
        for value in values:
            path = self._my_path(value.symbol)
             
            if path is None:
                owner_commits.append(value)
            else:
                # TOOD: verify that aliases are correct, new value is
                # compatible with the old one, etc (internal consistency check;
                # these things should be true anyway).
                my_commits[path] = value

        # Start with the owner commits, as it's atomic, so if it fails we can
        # unwind with no effect.
        if len(owner_commits) > 0:
            self._owner._commit_from(owner_commits)

        if len(my_commits) > 0:
            for path,value in my_commits.items():
                self._values[path] = value


