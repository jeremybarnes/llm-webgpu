import torch
import torch.jit as jit
from torch.jit import RecursiveScriptModule, ScriptModule
from torch.nn import Module
from torch.fx import symbolic_trace
from typing import Dict, List, Any, Tuple, Optional, Callable, Sequence, SupportsInt, Union, Iterator, get_origin, get_args, Set, overload
import inspect
from dataclasses import dataclass, field
import time
import copy
from runtimes import print_elapsed
from collections import defaultdict, OrderedDict
from torch.utils.hooks import RemovableHandle
from utils import _short_dtype, _print_value, _print_param, _print_value, typeify
from variables import ArgumentData, _to_torch_type


def introspect_model(m: Module):
    """
    Introspect the model, looking for what is scriptable and getting information on the
    operations used.
    """


    moduleCounts: Dict[str, int] = {}

    def recurse_module(m: Module, recursion: int, path: str):
        indent: str = ' ' * recursion * 4
        #print(f'{indent}got {m._get_name()} at {path} recursion {recursion}')
        print(f'{indent}{path} {m._get_name()}{inspect.signature(m.forward)}')
        for name,buffer in m.named_buffers(path, recurse=False):
            print(f'{indent}    buffer {name} is {_print_param(buffer.shape, buffer.dtype, buffer.device)}')
        for name,param in m.named_parameters(path, recurse=False):
            print(f'{indent}    parameter {name} is {_print_param(param.shape, param.dtype, param.device)}')

        gotScripted = None
        print_details = m._get_name() == "Embedding"

        try:
            scripted: RecursiveScriptModule = jit.script(m)
            gotScripted = scripted
            print(f"{indent}    (scripted)")
            if print_details:
                print(scripted.code)
                optimized: ScriptModule = jit.optimize_for_inference(scripted)
                print(optimized.graph)
                gotScripted = optimized
        except Exception as e:
            print(f"{indent}    (not scripted): {e}")
            pass

        n = m._get_name()
        if n in moduleCounts:
            moduleCounts[n] += 1
        else:
            moduleCounts[n] = 1

        if gotScripted is not None:
            return

        for name,child in m.named_children():
            recurse_module(child, recursion + 1, path + "." + name)

    recurse_module(m, 0, '')

    print("module counts")
    for name,count in sorted(moduleCounts.items()):
        print(f"{name:>40}:{count:5}")

@dataclass
class Invocation:
    args: Tuple
    kwargs: OrderedDict[str, Any] = field(default_factory = OrderedDict)
    output: Tuple = field(default_factory = tuple)
    elapsed: float = 0

    def __str__(self) -> str:
        def summarize_arg(arg: Any) -> str:
            if isinstance(arg, dict):
                return str(dict({k: summarize_arg(v) for k,v in arg.items()}))
            elif isinstance(arg, tuple):
                return str(tuple(summarize_arg(v) for v in arg))
            elif isinstance(arg, list):
                return str(list([summarize_arg(v) for v in arg]))
            elif isinstance(arg, torch.Tensor):
                if arg.numel() < 10:
                    return str(arg)
                else:
                    return _print_param(arg.size(), arg.dtype, arg.device) + " " + str(arg.device)
            else:
                return str(arg)

        summarized_args = list(map(summarize_arg, self.args))
        summarized_kwargs = {k: summarize_arg(v) for k,v in self.kwargs.items()}

        return f"Invocation(elapsed={print_elapsed(self.elapsed)} args={summarized_args} kwargs={summarized_kwargs})"

@dataclass
class SummaryData:
    arg_lengths: Dict[int, int] = field(default_factory = dict)
    args: List[ArgumentData] = field(default_factory = list)
    kwargs: Dict[str, ArgumentData] = field(default_factory = dict)

    def print_args(self, indent: int = 0):
        ind = ' ' * indent
        for i in range(len(self.args)):
            arg = self.args[i]
            print(f"{ind}{i}: {arg.summarize()}")

        for kw,arg in self.kwargs.items():
            print(f"{ind}{kw}: {arg.summarize()}")

    def add(self, other: 'SummaryData'):
        """
        Add another summary to make a combined summary.
        """
        for l,n in other.arg_lengths.items():
            self.arg_lengths[l] = self.arg_lengths.get(l, 0) + n

        for i,ad in enumerate(other.args):
            if i < len(self.args):
                self.args[i].combine(ad)
            else:
                self.args.append(ad)

        for n,ad in other.kwargs.items():
            if n in self.kwargs:
                self.kwargs[n].combine(ad)
            else:
                self.kwargs[n] = ad

@dataclass
class Invocations:
    m: Module
    path: str
    sig: inspect.Signature
    calls: List[Invocation] = field(default_factory = list)
    children: Dict[str, 'Invocations'] = field(default_factory = dict)

    def __init__(self, m: Module, path: str, *,
                 sig: Optional[inspect.Signature] = None,
                 calls: Optional[List[Invocation]] = None,
                 children: Optional[Dict[str, 'Invocations']] = None):
        self.m = m
        self.path = path
        self.sig = inspect.signature(m.forward) if sig is None else sig
        self.calls = [] if calls is None else calls
        self.children = {} if children is None else children

    def total_runtime(self) -> float:
        return sum([c.elapsed for c in self.calls])

    def __str__(self) -> str:
        return f"Invocations(path={self.path} module={self.m._get_name()} runtime={print_elapsed(self.total_runtime())} ncalls={len(self.calls)} nchildren={len(self.children)})" # sig={inspect.signature(self.m.forward)})"

    def summarize(self) -> SummaryData:
        result = SummaryData()

        max_nargs = 0
        all_kwargs: Set[str] = set()
        for c in self.calls:
            nargs = len(c.args)
            result.arg_lengths[nargs] = result.arg_lengths.get(nargs, 0) + 1
            max_nargs = max(max_nargs, nargs)
            all_kwargs.update(c.kwargs.keys())

        print("max_nargs", max_nargs)
        print("parameters=", self.sig.parameters)
        ordered_params: List[Tuple[str, inspect.Parameter]] = list(self.sig.parameters.items())

        for i in range(max_nargs):
            name,param = ordered_params[i]
            print("param", name, param)
            samples = [c.args[i] for c in self.calls if i < len(c.args[i])]
            torch_type = _to_torch_type(param.annotation, samples)
            result.args.append(ArgumentData(name=name,torch_type=torch_type))

        for k in all_kwargs:
            param = self.sig.parameters[k]
            samples = [c.kwargs.get(k) for c in self.calls if k in c.kwargs]
            torch_type = _to_torch_type(param.annotation, samples)
            result.kwargs[k] = ArgumentData(name=k,torch_type=torch_type)

        for c in self.calls:
            for i in range(len(c.args)):
                a = c.args[i]
                ad = result.args[i]
                name,param = ordered_params[i]
                #print("param", name, param)
                torch_type = _to_torch_type(param.annotation, [a])
                ad.add(a, torch_type)

            for k,a in c.kwargs.items():
                #print("parameter", k)
                param = self.sig.parameters[k]
                torch_type = _to_torch_type(param.annotation, [a])
                ad = result.kwargs[k]
                ad.add(a, torch_type)

        return result

def record_invocations(model: Module) -> Tuple[Invocations, Callable]:
    """
    Records the calls to the model and the arguments that are passed.
    """

    handles: List[RemovableHandle] = []

    def recurse_module(m: Module, recursion: int, path: str) -> Invocations:
        invs = Invocations(m, path)

        def pre_hook(module: Module, args: Tuple, kwargs: OrderedDict[str,Any]):
            #inv = Invocation(copy.deepcopy(args), copy.deepcopy(kwargs), (), time.time())
            inv = Invocation(args, kwargs, (), time.time())
            invs.calls.append(inv)
            return None

        def post_hook(module: Module, args: Tuple[Any], kwargs: Dict[str, Any], output: Tuple):
            inv = invs.calls[-1]
            inv.output = output
            inv.elapsed = time.time() - inv.elapsed
            return None

        handle1 = m.register_forward_pre_hook(pre_hook, with_kwargs=True)
        handle2 = m.register_forward_hook(post_hook, with_kwargs = True)

        for name,child in m.named_children():
            invs.children[name] = recurse_module(child, recursion + 1, path + "." + name)

        handles.append(handle1)
        handles.append(handle2)

        return invs

    def remove_hooks():
        for h in handles:
            try:
                h.remove()
            except:
                pass

    try:
        return recurse_module(model, 0, ''), remove_hooks
    except:
        remove_hooks()
        raise