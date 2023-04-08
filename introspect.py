import torch
import torch.jit as jit
from torch.jit import RecursiveScriptModule, ScriptModule # pyright: ignore
from torch.nn import Module
from typing import Dict, List, Any, Tuple, Optional, Callable, Sequence, SupportsInt, Union, Iterator, get_origin, get_args, Set, overload
import inspect
from dataclasses import dataclass, field
import time
import copy
from collections import defaultdict, OrderedDict
from torch.utils.hooks import RemovableHandle
from utils import _short_dtype, _print_value, _print_param, _print_value, typeify
from variables import VariableInfo, Variables, _to_torch_type, Invocations, SummaryData, Invocation


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
        print_details = False #m._get_name() == "Embedding"

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