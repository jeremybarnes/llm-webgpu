from templates import *
from templates.graph import *
from typing import Iterator, List, Sequence
import unittest
from collections import OrderedDict

@predicate
def is_square_convolution_1(scope: Scope, conv: OperationT) -> Iterator[Scope]:
    with scope.mutate('is_square_convolution_1') as inner:
        side = scope.dim('side')
        inner.condition('is_conv', conv.op == 'Conv')
        inner.condition('is_square', conv.params['kernel_shape'] == [side,side])
        yield inner


@predicate
def node_in_graph(scope: Scope, graph: GraphT, node: OperationT) -> Iterator[Scope]:
    return match_operation_in_graph(scope, graph, node)

    
_graph = Argument(GraphT, 'graph')
_conv1 = Argument(OperationT, 'conv1')
_bn    = Argument(OperationT, 'bn')
_conv2 = Argument(OperationT, 'conv2')
_node1 = Argument(OperationT, 'node1')
_node2 = Argument(OperationT, 'node2')

square_convolution_in_graph = pred_all([_graph, _conv1],
                                       is_square_convolution_1(_conv1),
                                       node_in_graph(_graph, _conv1))

        
class TestGraphPredicates(unittest.TestCase):
    def test_predicate(self):
        pred = is_square_convolution_1
        print('pred', pred, repr(pred))

        print('args', pred.arguments(), 'bound', pred.bound())

        assert pred.bound() == OrderedDict()
        assert list(pred.arguments().values()) == [ArgumentRef(OperationT, 'conv')]

        print('\n\n')
        print('-----------------')
        bound = is_square_convolution_1(_conv1)
        print('bound', bound)
        print('bound args', bound.arguments().values())
        assert list(bound.arguments().values()) == [ArgumentRef(OperationT, 'conv1')]

    def test_args(self):
        pred = square_convolution_in_graph

        assert list(pred.arguments().values()) == [ArgumentRef(GraphT, 'graph'), ArgumentRef(OperationT, 'conv1')]






