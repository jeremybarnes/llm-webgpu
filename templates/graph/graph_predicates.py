from typing import Iterator
from ..predicate import *
from .graph import *
from .graph_symbols import *
from ..exceptions import MatchError
from natsort import natsorted

def match_operation_in_graph(scope: Scope, graph: GraphT, op: OperationT) -> Iterator[Scope]:
    # Figure out which nodes could match by successive filtering

    print('doing operaation', scope.partial_value(op), scope.values())
    scope.dump()
    
    if scope.resolved(graph):
        graph_value: Graph = scope.value(graph)
        possible_nodes: Set[str]
        partial_inputs = scope.partial_value(op.inputs)
        known_inputs = { i for i in partial_inputs if isinstance(i, str) }

        partial_outputs = scope.partial_value(op.outputs)
        known_outputs = { o for o in partial_outputs if isinstance(o, str) }

        res_name = scope.resolved(op.node_name)
        res_op = scope.resolved(op.op)

        if scope.resolved(op.node_name):
            possible_nodes = { scope.value(op.node_name) }
        elif len(known_inputs) > 0:
            possible_nodes = known_inputs
        elif len(known_outputs) > 0:
            possible_nodes = known_outputs
        else:
            resop = scope.resolved(op.op)
            opnm = scope.partial_value(op.op)
            possible_nodes = { nm for nm,nd in graph_value.nodes.items() if not resop or nd.operation.operation == opnm }

        def keep_node(nm: str) -> bool:
            return (
                nm in graph_value.nodes
                and (not res_op or scope.value(op.op) == graph_value.nodes[nm].operation.operation))
        # TODO: filter better on inputs and outputs, especially by position

        print('possible_nodes', natsorted(possible_nodes))

        for n in natsorted(possible_nodes):
            if keep_node(n):
                try:
                    with scope.mutate('match_operation_in_graph') as inner:
                        node = graph_value.nodes[n]
                        node_params = { k:v for (ns,k),v in node.attributes.attributes.items()}

                        inner.resolve(op.node_name, n)
                        inner.resolve(op.op, node.operation.operation)
                        inner.resolve(op.inputs, node.inputs)
                        inner.resolve(op.outputs, node.outputs)
                        inner.resolve(op.params, node_params)

                        yield inner
                except MatchError as e:
                    pass
    else:
        # TOOD later: we can always add a node to the graph and return it
        # since the graph is not closed
        raise MatchError('Graph is not closed')
        
@predicate
def node_in_graph(scope: Scope, graph: GraphT, node: OperationT) -> Iterator[Scope]:
    return match_operation_in_graph(scope, graph, node)

