from typing import Type, Optional, Set, Any, TYPE_CHECKING
from .path import Index
from .symbol import Symbol, SymbolRef
from .symbol_types import SymbolArray, DimT
import numpy

if TYPE_CHECKING:
    from .scope import Scope


class DimListT(SymbolArray):
    """
    Symbolic variable-length list of dimensions, used as a shape for
    tensors.
    """
    typename: str ='dim_list'
    inner: Type[Symbol] = DimT


class DTypeT(Symbol):
    """
    Symbolic data type, used as a data type for tensors.
    """
    typename = 'd_type'
    tp = numpy.dtype


class Tensor(Symbol):
    """
    A tensor, to be fully specified, has both a data type and a shape
    that need to be determined (as well as its immediate value).
    """
    is_structured = True
    typename = 'tensor'
    tp = numpy.ndarray

    def __init__(self, idx: Index, parent: Symbol = None):
        super().__init__(idx, parent)
        self.dtype = DTypeT('dtype', self)
        self.shape = DimListT('shape', self)
        
    def _resolve(self, scope: 'Scope', value: numpy.ndarray,
                 resolving: Optional[Set[SymbolRef]]):
        # A tensor takes a numpy.ndarray
        if not isinstance(value, numpy.ndarray):
            raise TypeError(('attempt to resolve Tensor with {} of type {} '
                            +' (expected numpy.ndarray)')
                            .format(value, type(value)))

        self._post_resolve(scope, value, resolving)
        
        return self

    def _post_resolve(self, scope, value: Any, resolving: Optional[Set[SymbolRef]]) -> None:
        """
        Post-resolution for the array.  We ensure that it's possible to resolve
        all of the constituents.
        """
        with scope.mutate('tensor_resolve') as inner:
            inner.resolve(self.dtype, value.dtype, resolving)
            inner.resolve(self.shape, value.shape, resolving)
            inner._resolve(self, value, resolving, True) # should not be here...
            inner.commit(resolving)

    def _resolve_child(self, scope: 'Scope', child: Symbol, value: Any,
                       resolving: Optional[Set[SymbolRef]]) -> None:
        """
        One of our child fields (dtype or shape) was resolved.  Since there
        is nothing we need to do in response to this, it's a no-op.
        """
        #print('tensor', self.name, 'resolved child', child.name)
        pass
    
    def _partial_value(self, scope: 'Scope') -> dict:
        """
        Partial values of tensors are structures, until we are fully
        resolved and we can return an actual Tensor object.
        """

        return {
            'dtype': scope.partial_value(self.dtype),
            'shape': scope.partial_value(self.shape)
        }
    
    def __getitem__(self, key: str) -> Symbol:
        if key == 'dtype':
            return self.dtype
        elif key == 'shape':
            return self.shape
        return NotImplemented

    @staticmethod
    def incompatible(val1: Any, val2: Any) -> bool:
        """
        Tensor incompatible override: two tensors are incompatible if their
        elements are not all equal.
        """
        if val1 is None or val2 is None:
            return False
        return (val1 != val2).any()

    
