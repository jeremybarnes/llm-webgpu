from templates import *
from templates.graph import *
from typing import Iterator, List, Sequence, Union, cast
import unittest
from collections import OrderedDict
import onnx
import pytest

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


filename = 'resnet18v1.onnx'

model = onnx.load(filename)
graph = Graph()
graph.load_onnx(filename)

# A predicate has two operations:
#
# 1.  Bind, which is calling with Arguments to partially bind the
#     free arguments.  When all of the arguments are bound, it's fully
#     bound.  Note that the output of binding a predicate is still a
#     predicate.
#
#     Syntax: bind(Predicate, vars) -> Predicate
#
# 2.  Apply, which is calling with a Scope.  Any unbound arguments
#     become free variables, and it will begin to generate scopes.
#
#     Syntax: apply(Scope, Predicate) -> Iterator[Scope]

def bind(pred: Predicate, *args: Union[Argument,Symbol,Any]) -> Predicate:
    return pred.bind(*args)

def apply(pred: Predicate, scope: Scope, *syms: Symbol) -> Iterator[Scope]:
    return pred.apply(scope, *syms)


class TestGraphOnnxPredicates(unittest.TestCase):
    def test_bind_then_find(self):
        root = NullScope()
        inner = NamedScope(root, 'test_bind_then_find')

        # First bind, then find
        bound = bind(is_square_convolution_1, _conv1)

        print('bound', bound)

        print('-------------------------- is_square_convolution')
        # This one does not match a graph, so we don't expect to actually get the convolutions

        num_found = 0
        
        for found in apply(bound, inner):
            num_found += 1
            #print('found', found.values())
            to_find = cast(OperationT,_conv1._in(found))
            print('to_find', to_find)
            print('found', found.partial_value(to_find))
            #found.dump()
            assert found.partial_value(to_find.op) == 'Conv', 'Expected to constrain Conv operation'
            assert found.partial_value(to_find)['op'] == 'Conv', 'Expected to constrain Conv operation'

        assert num_found == 1, "Expected to find one generic convolution"

    def test_direct_bind(self):
        root = NullScope()
        inner = NamedScope(root, 'test_direct_bind')
        conv_op = cast(OperationT, inner.symbol(OperationT, 'conv_op'))
        
        # Try again binding the symbol directly
        bound = bind(is_square_convolution_1, conv_op)

        print('bound_a', bound)

        print('-------------------------- is_square_convolution direct bind')
        # This one does not match a graph, so we don't expect to actually get the convolutions

        num_found = 0
        
        for found in apply(bound, inner):
            num_found += 1
            #print('found', found.values())
            to_find = conv_op
            print('to_find', to_find)
            print('found', found.partial_value(to_find))
            found.dump()
            #for k,v in found.values().items():
            #    print(k,'=',v)
            assert found.partial_value(to_find.op) == 'Conv', 'Expected to constrain Conv operation'
            assert found.partial_value(to_find)['op'] == 'Conv', 'Expected to constrain Conv operation'

    def test_direct_apply(self):
        root = NullScope()
        inner = NamedScope(root, 'test_direct_apply')
        conv_op = cast(OperationT, inner.symbol(OperationT, 'conv_op'))

        # Try again without binding
        print('-------------------------- is_square_convolution no bind')
        # This one does not match a graph, so we don't expect to actually get the convolutions

        for found in apply(is_square_convolution_1, inner, conv_op):
            #print('found', found.values())
            to_find = conv_op
            print('to_find', to_find)
            print('found', found.partial_value(to_find))
            found.dump()
            #for k,v in found.values().items():
            #    print(k,'=',v)
            assert found.partial_value(to_find.op) == 'Conv', 'Expected to constrain Conv operation'
            assert found.partial_value(to_find)['op'] == 'Conv', 'Expected to constrain Conv operation'

    def test_extract_all(self):
        root = NullScope()
        inner = NamedScope(root, 'test_extract_all')
        conv_op = cast(OperationT, inner.symbol(OperationT, 'conv_op'))

        bound2 = bind(square_convolution_in_graph, graph, conv_op)

        print('bound2', bound2, repr(bound2))

        num_found = 0
        
        for found in apply(bound2, inner):
            num_found += 1
            print('found', found.partial_value(conv_op), num_found)
            #found.dump()

        self.assertEqual(num_found, 20, "Expected to find twenty convolutions in ResNet19")
