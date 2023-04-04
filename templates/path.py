from typing import Union, Sequence, overload, Any, Iterator, List

class Private(object):
    """
    Indicates that an index is private, creating a different namespace.
    """
    def __init__(self, idx: Union[int, str, 'Private']):
        self._idx: Union[int, str] = idx if not isinstance(idx, Private) else idx._idx

    def __repr__(self) -> str:
        return '$' + repr(self._idx)

    def __str__(self) -> str:
        return '$' + str(self._idx)
        
    def __hash__(self):
        return hash(('$', self._idx))

    def __eq__(self, other) -> bool:
        return isinstance(other, Private) and self._idx == other._idx

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


# These types may be used as an index into an object or scope
Index = Union[Private,int,str]

def _private(idx: Union[int, str]) -> Private:
    """
    Create a private version of the indexed value.
    """
    return Private(idx)


class Path(object):
    """
    A path defines how we address a symbol within its scope.  It's a list of
    keys we can use to recursively look up the contents.
    """
    def __init__(self, path: Sequence[Index]):
        assert not isinstance(path, Path)
        self._path: List[Index] = list(path)

    def startswith(self, p: 'Path') -> bool:
        """
        Returns True iff p is a prefix of self.
        """
        if len(p) > len(self):
            return False

        for i in range(len(p)):
            if self._path[i] != p[i]:
                return False

        return True

    def __len__(self) -> int:
        return len(self._path)

    def __list__(self) -> Sequence[Index]:
        return self._path

    @overload
    def __getitem__(self, i: int) -> Index: ...

    @overload
    def __getitem__(self, i: slice) -> 'Path': ...

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Path(self._path[i])
        else:
            return self._path[i]

    @overload
    def __add__(self, other: 'Path') -> 'Path': ...
        
    @overload
    def __add__(self, other: Index) -> 'Path': ...
        
    def __add__(self, other):
        if isinstance(other, Path):
            return Path(self._path + other._path)
        else:
            return Path(self._path + [other])

    # @overload
    # def __iadd__(self, other: 'Path') -> None: ...
        
    # @overload
    # def __iadd__(self, other: Index) -> None: ...
        
    # def __iadd__(self, other):
    #     if isinstance(Path, other):
    #         self._path += other._path
    #     else:
    #         self._path.append(other)

    def __hash__(self) -> int:
        return hash(tuple(self._path))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Path):
            return NotImplemented
        return self._path == other._path

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        return repr(self._path)

    def __iter__(self) -> Iterator[Index]:
        return self._path.__iter__()
