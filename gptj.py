# Load the model

from transformers import GPTJForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast, GPTNeoModel
import torch
from typing import Type, Tuple, Any, Dict, List, Optional, Generator, Callable, OrderedDict, Union
from types import ModuleType

from torch.nn import Module
from torch import Tensor, ScriptModule, ScriptFunction, Value, Size, Block, dtype, memory_format, device, scalar_tensor, add, tanh
import torch.jit as jit
import torch.fx as fx
import torch
import inspect
import time
import copy
import struct
import sys
from scriptable_gpt_neo import GPTNeoForCausalLM

from graphs import Scope, Operation, default_find_operation
from utils import _print_value as print_value
from torch._C import Graph, Node, dtype as cdtype
from dataclasses import dataclass, field


from ansi.color import bg, fg
from ansi.color.fx import reset # pyright: ignore

def process_args(_, device_in='cpu', dtype_in='float32'):
    global device, dtype
    device,dtype = torch.device(device_in),getattr(torch, dtype_in)

process_args(*sys.argv)

if torch.has_mps and device == torch.device('mps'):
    import mps_fixups
    mps_fixups.fixup_mps()

def init_model(klass: Type[PreTrainedModel], model: str, *args, **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    if str(device) == 'cpu':
        kwargs['torch_dtype'] = torch.float32
        kwargs['low_cpu_mem_usage'] = True

    elif str(device) == 'mps':
        kwargs['torch_dtype'] = dtype

    else:
        raise RuntimeError(f'unknown device {str(device)}')

    res: PreTrainedModel = klass.from_pretrained(model, *args, **kwargs)
    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model, *args)

    if str(device) == 'cpu':
        result = res.to(device, torch.float32), tok
        result[0]._dtype = torch.float32

    elif str(device) == 'mps':
        result = res.to(device, dtype), tok
        result[0]._dtype = dtype

    else:
        raise RuntimeError(f'unknown device {str(device)}')

    result[0]._device = device
    return result


llm: str = ''
model: Module
tokenizer: PreTrainedTokenizerFast

llm = 'gpt-neo-125M'
#llm = 'gpt-neox-20B'
#llm = 'gpt-j-6B'

if llm == 'gpt-neo-125M':
    model,tokenizer = init_model(GPTNeoForCausalLM, "EleutherAI/gpt-neo-125M")
elif llm == 'gpt-neox-20B':
    model,tokenizer = init_model(GPTNeoXForCausalLM, "EleutherAI/gpt-neox-20b")
elif llm == 'gpt-j-6B':
    model,tokenizer = init_model(GPTJForCausalLM, "EleutherAI/gpt-j-6B")
else:
    raise RuntimeError(f'unknown llm {llm}')

#print("tok=", tokenizer)
#print("model=", model)

from introspect import introspect_model, record_invocations, SummaryData
#introspect_model(model)


#runtimes: Dict[str, float] = instrument_runtimes(model, runtimes)



from device_comparison import device_comparison_mode

#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float32), (torch.device('mps'), torch.float16)])
#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float16)])
#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float32)])
#device_comparison_mode(model, device, dtype, [(torch.device('cpu'), torch.float32)])
#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float32)])

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)


input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

#print(input_ids)

invocations,remove_invocations = record_invocations(model)

gen_tokens = model.generate(
    input_ids,
    do_sample=False,  # make it deterministic for debugging
    temperature=0.9,
    max_length=50,
)

gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)

#print(invocations)
#for call in invocations.calls:
#    print(f"  {call}")

remove_invocations()

print(inspect.signature(invocations.m.forward))

sd = invocations.summarize()
sd.print_args()


from introspect import Invocations
from runtimes import print_elapsed
from introspect import Invocations

from optimize_script import OptimizeModelData, optimize_script, ModuleOptimizationInfo




def _optimize_module(invocations: Invocations, dtype: torch.dtype, device: torch.device, cache: OptimizeModelData) -> Module:
    # See if the graph can be scripted
    module = invocations.m
    script: Optional[ScriptModule] = None
    optimized: Optional[Module] = None

    mod_name = module._get_name()
    if mod_name in cache.modules:
        return cache.modules[mod_name]

    try:
        pass
        script = torch.jit.script(module)
        #script = symbolic_trace(module)
    except Exception as e:
        print(f"{invocations.path} is not scriptable: {e}")
        pass

    if script is not None:
        info = cache.optinfo[mod_name]
        try:
            optimized = optimize_script(script, invocations, info)
            #optimized.dtype = script.dtype

            #print(f"dtype is {script.dtype}")
            #print(f"new dtype is {optimized.dtype}")

            # Some of the sanity checks use the parameters to know which device the
            # model is on.  So we play along by adding them back in.
            #for name,param in module.named_parameters(recurse=False):
            #    print(f"parameter {name}={print_value(param)}")
            #    optimized.register_parameter(name, param)
            #print("successfully optimized", invocations.path)
            print(f"{invocations.path} was optimized to {optimized}")

        except Exception as e:
            raise
            print(f"{invocations.path} is not optimizable: {e}")
            pass

    if optimized is None:
        # try to optimize each of the children, as we couldn't optimize the parent
        optimized = copy.deepcopy(module)

        for name,child in module.named_children():
            child_invocations = invocations.children[name]
            assert child_invocations.m is child
            optimized_child = _optimize_module(child_invocations, dtype, device, cache)
            optimized.add_module(name, optimized_child)
            #setattr(optimized, name, optimized_child)


    def try_set(attr: str, val: Any):
        try:
            setattr(optimized, attr, val)
        except:
            pass

    try_set('dtype', dtype)
    try_set('_dtype', dtype)
    try_set('devicer', device)
    try_set('_device', device)

    #print(f"params before {len(list(module.parameters()))}")
    #print(f"params after {len(list(optimized.parameters()))}")
    #assert len(list(module.named_parameters())) == len(list(optimized.named_parameters()))
    #print("module.dtype", module.dtype)
    #assert optimized.dtype == module.dtype

    cache.modules[mod_name] = optimized
    return optimized

def optimize_module(invocations: Invocations, dtype: torch.dtype, device: torch.device):
    cache = OptimizeModelData()

    # Find the full set of invocations per module
    def recurse_invocations(i: Invocations):
        model_name = i.m._get_name()
        if model_name not in cache.optinfo:
            cache.optinfo[model_name] = ModuleOptimizationInfo()
        info = cache.optinfo[model_name]
        info.add(i)

        for n,ch in i.children.items():
            recurse_invocations(ch)

    recurse_invocations(invocations)

    for name,info in sorted(cache.optinfo.items()):
        print(f"module {name} has {len(info.invocations)} invocations with runtime {print_elapsed(info.total_runtime())}")
        #info.summary.print_args(8)
        print()

    return _optimize_module(invocations, dtype, device, cache)

compiled_mod = torch.compile(model)

optimized_mod = optimize_module(invocations, dtype, device)

cpu_mod = copy.deepcopy(model).to('cpu')

#print("compiled_mod", compiled_mod)

from runtimes import print_elapsed

for m,name in [(model,"base"),(compiled_mod,"compiled"),(optimized_mod,"optimized"),(optimized_mod,"optimized2"),(cpu_mod,"cpu"),]:
    before = time.time()
    model_inputs = input_ids if name != "cpu" else input_ids.to('cpu')
    gen_tokens = m.generate(
        model_inputs,
        do_sample=False,  # make it deterministic for debugging
        temperature=0.9,
        max_length=50,
    )
    after = time.time()
    print(type(m))
    print(f"{name:30} {m.device} {m.dtype} {print_elapsed(after-before)}")

exit(1)

from graph_comparison import make_compare_operation

for name,ch in invocations.children.items():
    print('  child1: ', ch)
    print('      sig:', inspect.signature(ch.m.forward))
    ch.summarize().print_args(8);
    try:
        script = jit.script(ch.m)
        print('script', script)

        optimized: ScriptModule = jit.optimize_for_inference(script)
        #print("code", optimized.graph)
        graph: Graph = optimized.inlined_graph
        #print(graph)

        for call in ch.calls:
            scope = Scope(find_operation=make_compare_operation('cpu'))
            scope.exec_graph(graph, ch.m, call.args, call.kwargs)
            print("done\n")

    except:
        pass

    for name2,ch2 in ch.children.items():
        print('      child2: ', ch2)
        print('          sig:', inspect.signature(ch2.m.forward))
        ch2.summarize().print_args(16);
        #script = jit.script(ch2.m)
        #print('script', script)
        #optimized: ScriptModule = jit.optimize_for_inference(script)
        #print("code", optimized.graph)
        #graph: Graph = optimized.inlined_graph

        #if len(ch2.calls) > 0:
        #    trace = jit.trace(ch2.m.forward, example_inputs=ch2.calls[0].args, example_kwarg_inputs=ch2.calls[0].kwargs)

        for name3,ch3 in ch2.children.items():
            print('          child3: ', ch3)
            print('              sig:', inspect.signature(ch3.m.forward))
            ch3.summarize().print_args(24)
            script = jit.script(ch3.m)
            #print('script', script)
            optimized: ScriptModule = jit.optimize_for_inference(script)
            #print("code", optimized.graph)
            graph: Graph = optimized.inlined_graph
            #print(graph)

            for call in ch3.calls:
                scope = Scope(find_operation=make_compare_operation('cpu'))
                scope.exec_graph(graph, ch3.m, call.args, call.kwargs)
                print("done\n")

            #trace = jit.trace(ch3.m.forward, example_inputs=ch3.calls[0].args, example_kwarg_inputs=ch3.calls[0].kwargs)
            #print('trace', trace.inlined_graph)
