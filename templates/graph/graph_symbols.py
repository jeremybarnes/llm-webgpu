from typing import Optional, Any, Set
from ..symbol import Symbol, SymbolRef
from ..symbol_types import SymbolArray, Structure, ParameterMapT
from ..tensor_types import Tensor
from ..operators import EqualityComparable
from .graph import Node, Graph
from ..path import Index
from ..scope import Scope

class ValueNameT(Symbol):
    """
    Type representing an argument to a graph operation, as the name of a value.
    """
    typename="arg"
    tp=str

class OperationArgsT(SymbolArray):
    """
    Type representing the symbolic names of a set of input or output arguments
    in a graph.
    """
    typename='op_args'
    inner = ValueNameT
    
class TensorListT(SymbolArray):
    """
    Type representing a list of tensors that represents a list of arguments
    to an operator.
    """
    typename = 'tensors'
    inner = Tensor


class OpNameT(EqualityComparable):
    """
    Symbolic string type that represents the name of an operation.
    """
    typename='opname'
    tp=str

    
class NodeNameT(Symbol):
    """
    Symbolic string type that represents the name of an operation node.
    """
    typename='nodename'
    tp=str

class OperationT(Structure):
    """
    Symbolic object that represents an operation (a node in an expression
    graph).
    """
    typename='op'
    tp = Node
    
    def __init__(self, idx: Index, parent: Optional[Scope] = None):
        super().__init__(idx, parent)
        self.node_name = NodeNameT("node_name", self)
        self.op = OpNameT("op", self)
        self.inputs = OperationArgsT("inputs", self)
        self.outputs = OperationArgsT("outputs", self)
        self.params = ParameterMapT("params", self)

    def _resolve_child(self, scope: Scope, child: Symbol, value: Any,
                       resolving: Optional[Set[SymbolRef]]):
        # There is no resolved type (it would normally be a node) so we don't
        # do anything here
        if (scope.resolved(self.node_name)
            and scope.resolved(self.op)
            and scope.resolved(self.inputs)
            and scope.resolved(self.outputs)
            and scope.resolved(self.params)):

            value = self._partial_value(scope)
            with scope.mutate('array_child') as inner:
                inner._resolve(self, value, resolving, True)
                inner.commit()
            
        
        
    def _partial_value(self, scope: Scope) -> dict:
        """
        Partial values of tensors are structures, until we are fully
        resolved and we can return an actual Tensor object.
        """

        return {
            'node_name': scope.partial_value(self.node_name),
            'op': scope.partial_value(self.op),
            'inputs': scope.partial_value(self.inputs),
            'outputs': scope.partial_value(self.outputs),
            'params': scope.partial_value(self.params),
        }

class GraphT(Symbol):
    """
    This is an entire graph.  Currently it can only be resolved all at once;
    eventually we will allow indexing.
    """
    typename='graph'
    tp=Graph
