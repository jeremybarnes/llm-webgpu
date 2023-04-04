from torch.nn import Module
from torch import Tensor, ScriptModule, ScriptFunction, Value, Size, Block, dtype, memory_format, device, scalar_tensor, add, tanh
import torch.jit as jit
import torch.fx as fx
import torch
from typing import Type, Tuple, Any, Dict, List, Optional, Generator, Callable, OrderedDict, Union, TypeVar, Iterator, Iterable, Sequence
from dataclasses import dataclass, field
from introspect import (introspect_model, record_invocations, _short_dtype,
                        SummaryData, Invocation, Invocations)
from variables import (VariableInfo, Variables, Origin, _to_torch_type)
from graphs import Scope, default_find_operation, Operation
from utils import _print_value
from torch._C import Graph, Node, dtype as cdtype
import inspect
from operators import const_prop_graph, first

@dataclass
class ModuleOptimizationInfo:
    invocations: List[Invocations] = field(default_factory=list, repr=False)
    #summary: SummaryData = field(default_factory=SummaryData)

    def total_runtime(self) -> float:
        return sum([inv.total_runtime() for inv in self.invocations], 0.0)

    def max_num_args(self) -> int:
        res = 0
        for i in self.invocations:
            for call in i.calls:
                al = len(call.args)
                res = max(res, al)
        return res

    def add(self, i: Invocations):
        self.invocations.append(i)
        #self.summary.add(i.summarize())

    def summarize(self) -> SummaryData:
        res = SummaryData()
        vars = Variables()

        arg_samples: List[List[Any]] = []

        def record_arg(i: int, a: Any):
            if i >= len(arg_samples):
                arg_samples.append([a])
            else:
                arg_samples[i].append(a)

        kwarg_samples: Dict[str, List[Any]] = {}

        def record_kwarg(k: str, a: Any): 
            if k in kwarg_samples:
                kwarg_samples[k].append(a)
            else:
                kwarg_samples[k] = [a]

        for i in self.invocations:
            for call in i.calls:
                al = len(call.args)
                if al in res.arg_lengths:
                    res.arg_lengths[al] += 1
                else:
                    res.arg_lengths[al] = 1

                for i,a in enumerate(call.args):
                    record_arg(i,a)
                for k,a in call.kwargs.items():
                    record_kwarg(k,a)

        for i,samples in enumerate(arg_samples):
            annotation = None
            torch_type = _to_torch_type(annotation, samples)
            res.args.append(vars.sampled("#arg#" + str(i),torch_type,samples,None,-1))

        for k,samples in kwarg_samples.items():
            annotation = None
            torch_type = _to_torch_type(annotation, samples)
            res.kwargs[k] = vars.sampled("#kwarg#" + k,torch_type,samples,None,-1)

        return res

@dataclass
class OptimizeModelData:
    modules: Dict[str, Module] = field(default_factory=dict)
    optinfo: Dict[str, ModuleOptimizationInfo] = field(default_factory=dict)

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
    for i,input in enumerate(graph_inputs):
        print("  input",i,input)
    invocations.summarize().print_args()
    #info.summary.print_args()

    vars = Variables()

    scope = Scope(default_find_operation)

    self_info = vars.constant(name="self", origin=Origin.SELF, value=invocations.m, produced=-1, produced_by=graph)
    vars.add(self_info)
    scope.add_var("self", invocations.m)

    summary: SummaryData = invocations.summarize()

    def add_input(name: str, graph_input: Value, var_info: VariableInfo, tp: 'torch._C.JitType'):
        var_info = var_info.renamed(name)
        vars.add(var_info)
        print("got input", i, var_info)

    for i,arg in enumerate(summary.args):
        graph_input = graph_inputs[i+1]  #+1 for self
        print("default input", i, graph_input, graph_input.type(), type(graph_input.type()))
        name = graph_input.debugName()
        add_input(name, graph_input, arg, graph_input.type())

    other_args: Dict[str, Value] = {}
    for i in range(info.max_num_args() + 1, len(graph_inputs)):
        graph_input = graph_inputs[i]
        print("default arg", graph_input, graph_input.type(), type(graph_input.type()), type(graph_input.type()).__bases__)
        other_args[graph_input.debugName()] = graph_input

    for name,arg in summary.kwargs.items():
        print("kwarg", name, arg)
        input_name = name + ".1"
        graph_input = other_args[input_name]
        add_input(input_name, graph_input, arg, graph_input.type())

    const_prop_graph(graph, vars)

    #print("optimizing", graph)
    #print("type", type(graph))
    #print(f"this summary: {invocations.summarize()}")
    #print(f"module summary: {info.summary}")

    print()
    vars.dump_vars()
    return script
