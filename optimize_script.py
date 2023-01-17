from torch.nn import Module
from torch import Tensor, ScriptModule, ScriptFunction, Value, Size, Block, dtype, memory_format, device, scalar_tensor, add, tanh
import torch.jit as jit
import torch.fx as fx
import torch
from typing import Type, Tuple, Any, Dict, List, Optional, Generator, Callable, OrderedDict, Union, TypeVar, Iterator, Iterable
from dataclasses import dataclass, field
from introspect import introspect_model, record_invocations, SummaryData, Invocation, Invocations
from graphs import Scope, default_find_operation, Operation, _print_value
from torch._C import Graph, Node, dtype as cdtype
from enum import Enum
import inspect

@dataclass
class ModuleOptimizationInfo:
    invocations: List[Invocations] = field(default_factory=list, repr=False)
    summary: SummaryData = field(default_factory=SummaryData)

    def total_runtime(self) -> float:
        return sum([inv.total_runtime() for inv in self.invocations], 0.0)

    def add(self, i: Invocations):
        self.invocations.append(i)
        self.summary.add(i.summarize())


@dataclass
class OptimizeModelData:
    modules: Dict[str, Module] = field(default_factory=dict)
    optinfo: Dict[str, ModuleOptimizationInfo] = field(default_factory=dict)

class Origin(Enum):
    UNKNOWN = 0
    SELF = 1
    ARG = 2
    DEFAULT_ARG = 3
    LOCAL = 4
    CONST_PROP = 5

@dataclass
class VariableInfo:
    name: str = ""
    origin: Origin = Origin.UNKNOWN
    const: bool = False
    const_value: Optional[Any] = None
    produced: int = 0
    first_consumed: int = 10000
    last_consumed: int = 0

@dataclass
class Variables:
    vars: OrderedDict[str, VariableInfo] = field(default_factory=OrderedDict)

    def add(self, v: VariableInfo):
        assert len(v.name) > 0
        assert v.name not in self.vars
        self.vars[v.name] = v

    def add_constant(self, n: str, v: Any, i: int):
        info = VariableInfo(n, Origin.CONST_PROP, True, v, i)
        self.add(info)

    def get(self, n: str, i: int) -> VariableInfo:
        result = self.vars[n]
        if i < result.first_consumed:
            result.first_consumed = i
        if i > result.last_consumed:
            result.last_consumed = i
        return result

    def print_vars(self, indent: str = ''):
        for name,info in self.vars.items():
            print(indent,f"{name:30} {info.origin.name:10} {info.const:6} {info.produced:5} {info.first_consumed:5} {info.last_consumed:5} {_print_value(info.const_value)}")

VT = TypeVar("VT")
def first(x: Iterable[VT]) -> Optional[VT]:
    for v in x:
        return v
    return None

def optimize_script(script: ScriptModule, invocations: Invocations, info: ModuleOptimizationInfo) -> Optional[ScriptModule]:
    optimized = script #jit.optimize_for_inference(script)
    signature = inspect.signature(invocations.m.forward)

    #print('optimizing script', script)
    try:
        graph: Graph = optimized.inlined_graph
    except:
        print("can't be optimized as it has no inlined_graph")
        return None
    print("doing graph", graph)

    graph_inputs = list(graph.inputs())
    print("graph_inputs", graph_inputs)
    invocations.summarize().print_args()
    info.summary.print_args()

    vars = Variables()

    scope = Scope(default_find_operation)

    self_info = VariableInfo("self", Origin.SELF, const=True, const_value=invocations.m)
    vars.add(self_info)
    scope.add_var("self", invocations.m)

    for i,arg in enumerate(invocations.summarize().args):
        graph_input = graph_inputs[i+1]  #+1 for self
        name = graph_input.debugName()
        is_const = len(arg.values) == 1
        const_value: Optional[Any] = None

        if is_const:
            const_value = first(arg.values.keys())
            scope.add_var(name, const_value)

        var_info = VariableInfo(name, Origin.ARG, is_const, const_value)
        vars.add(var_info)
        print("got input", i, var_info)

    for i in range(len(info.summary.args) + 1, len(graph_inputs)):
        graph_input = graph_inputs[i]
        print("default arg", graph_input)

    def do_node(i: int, node: Node, indent: str) -> Tuple[Optional[Any|Tuple[Any]], bool]:

        print(indent, "executing node", i, node)
        all_constant_inputs = True
        input_values = []
        result = None
        for input in node.inputs():
            name = input.debugName()
            var = vars.get(name, i)
            #print(indent, "    input", name, var)
            if not var.const:
                all_constant_inputs = False
            input_values.append(var.const_value)

        print(indent, "all_constant_inputs:", all_constant_inputs)
        is_constant_output = False

        if all_constant_inputs:
            if node.kind() == "prim::If":
                # If with constant condition, there is only one block, so we can
                # propagate constants through it
                block_index = 0 if input_values[0] else 1
                blocks = list(node.blocks())
                block = blocks[block_index]
                for block_node in block.nodes():
                    do_node(i, block_node, indent + ' ' * 8)

                return_node = block.returnNode()
                result,is_constant_output = do_node(i, return_node, indent + ' ' * 8)
                print(indent, "constant if: result", result)
                scope.print_vars()
            else:
                #scope.print_vars()
                result = scope.exec_node(node)
                is_constant_output = True

            print(indent, "constant node, returned output", _print_value(result))
            print(indent, "outputs size", node.outputsSize(), list(node.outputs()))
        else:
            # Find what the op does, and figure out the output
            # - type
            # - dtype
            # - device
            # - shape
            print(indent, "non constant node", node.kind())
            if node.kind() == "prim::If":
                results: List[Any] = []
                for block in node.blocks():
                    for block_node in block.nodes():
                        do_node(i, block_node, indent + ' ' * 8)

                    return_node = block.returnNode()
                    print(indent, "return node", return_node)
                    results.append(do_node(i, return_node, indent + ' ' * 8))
                # TODO: collapse across the branches

        if is_constant_output:
            if node.outputsSize() == 0:
                pass
            elif node.outputsSize() == 1:
                info = node.outputsAt(0)
                vars.add_constant(info.debugName(), result, i)
            else:
                assert isinstance(result, tuple)
                for val,info in zip(result, node.outputs()):
                    vars.add_constant(info.debugName(), val, i)
        else:
            for output in node.outputs():
                info = VariableInfo(output.debugName(), Origin.LOCAL, False, None, i)
                vars.add(info)


        vars.print_vars(indent)

        return result,is_constant_output

    for i,node in enumerate(graph.nodes()):
        do_node(i, node, '')

    print("optimizing", graph)
    print("type", type(graph))
    print(f"this summary: {invocations.summarize()}")
    print(f"module summary: {info.summary}")
    return script
