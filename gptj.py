# Load the model

from transformers import GPTJForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast, GPTNeoModel
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

from graphs import exec_graph
from torch._C import Graph, Node, dtype as cdtype

from ansi.color import bg, fg
from ansi.color.fx import reset


if torch.has_mps:
    import mps_fixups
    mps_fixups.fixup_mps()


#device,dtype = torch.device("mps"), torch.float16
#device,dtype = torch.device("mps"), torch.float32
device,dtype = torch.device("cpu"), torch.float32

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
        return res.to(device, torch.float32), tok

    elif str(device) == 'mps':
        return res.to(device, dtype), tok

    else:
        raise RuntimeError(f'unknown device {str(device)}')

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

from introspect import introspect_model, record_invocations
#introspect_model(model)
from runtimes import instrument_runtimes


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
    max_length=70,
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

for name,ch in invocations.children.items():
    print('  child: ', ch)
    print('      sig:', inspect.signature(ch.m.forward))
    ch.summarize().print_args(8);

    for name2,ch2 in ch.children.items():
        print('      child: ', ch2)
        print('          sig:', inspect.signature(ch2.m.forward))
        ch2.summarize().print_args(16);

        #if len(ch2.calls) > 0:
        #    trace = jit.trace(ch2.m.forward, example_inputs=ch2.calls[0].args, example_kwarg_inputs=ch2.calls[0].kwargs)

        for name3,ch3 in ch2.children.items():
            print('          child: ', ch3)
            print('              sig:', inspect.signature(ch3.m.forward))
            ch3.summarize().print_args(24);
            script = jit.script(ch3.m)
            print('script', script)
            optimized: ScriptModule = jit.optimize_for_inference(script)
            #print("code", optimized.graph)
            graph: Graph = optimized.inlined_graph
            print(graph)
            exec_graph(graph, ch3.m, ch3.calls[0].args, ch3.calls[0].kwargs)

            #trace = jit.trace(ch3.m.forward, example_inputs=ch3.calls[0].args, example_kwarg_inputs=ch3.calls[0].kwargs)
            #print('trace', trace.inlined_graph)
