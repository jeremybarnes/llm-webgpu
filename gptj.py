# Load the model

from transformers import GPTJForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast, GPTNeoModel
import torch
from typing import Type, Tuple, Any, Dict, List, Optional, Generator
from types import ModuleType

from torch.nn import Module
from torch import Tensor

import inspect
import time
import copy
import struct
import sys


if torch.has_mps:
    import mps_fixups
    mps_fixups.fixup_mps()


#device,dtype = torch.device("mps"), torch.float16
device,dtype = torch.device("mps"), torch.float32
#device,dtype = torch.device("cpu"), torch.float32

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

print("tok=", tokenizer)
print("model=", model)

from introspect import introspect_model
introspect_model(model)
from runtimes import instrument_runtimes


runtimes: Dict[str, float] = {}
instrument_runtimes(model, runtimes)



from device_comparison import device_comparison_mode

#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float32), (torch.device('mps'), torch.float16)])
#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float16)])
#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float32)])
device_comparison_mode(model, device, dtype, [(torch.device('cpu'), torch.float32)])
#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float32)])

prompt = (

    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "

    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "

    "researchers was the fact that the unicorns spoke perfect English."

)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)



#print(input_ids)

gen_tokens = model.generate(
    input_ids,
    do_sample=False,  # make it deterministic for debugging
    temperature=0.9,
    max_length=100,
)

gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
