# Load the model

from transformers import GPTJForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast, GPTNeoModel
import torch
from typing import Type, Tuple, Any, Dict, List, Optional, Generator
from types import ModuleType

from torch.library import Library as _Library
from torch.nn import Module
from torch import Tensor
_lib = _Library("aten", "IMPL")

import torch.jit as jit
from torch.jit import RecursiveScriptModule, ScriptModule
import inspect
import time
import copy
import struct
import sys

if True:
    for n,m in sys.modules.items():
        #print(f"module {n} is {m}")
        for n2,v in m.__dict__.items():
            if n2.find('var_mean') != -1 and callable(v):
                print(f"got {n2} in ", m.__name__)

from dataclasses import dataclass

def _topk(self: torch.Tensor, k: int, dim:int=-1, largest:bool=True, sorted:bool=True):
    res, indices = torch.topk(self.to('cpu', torch.float32), k, dim, largest, sorted)
    return res.to(self), indices.to('mps')

_lib.impl("topk", _topk, "MPS")

def _max(self: torch.Tensor) -> torch.Tensor:
    return torch.max(self.to('cpu')).to('mps')

_lib.impl("max", _max, "MPS")

#def _embedding(weight: torch.Tensor, input: torch.Tensor, padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False):
#    return torch.embedding(weight.to('cpu'), input.to('cpu'), padding_idx, scale_grad_by_freq, sparse).to('mps')

#_lib.impl("embedding", _embedding, "MPS")


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
model: PreTrainedModel
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

def printParam(size: torch.Size, dtype: torch.dtype):
    dtypes = {
        torch.float32: 'f32',
        torch.float16: 'f16',
        torch.int8: 'i8',
        torch.uint8: 'u8',
        torch.int32: 'i32',
        torch.bool: 'b'
    }
    dsizeof = {
        torch.float32: 4,
        torch.float16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int32: 4,
        torch.bool: 1,
    }

    totalBytes = size.numel() * dsizeof[dtype]
    if totalBytes < 1024:
        totalSize = str(totalBytes) + " bytes"
    elif totalBytes < 1024*1024:
        totalSize = str(totalBytes / 1024) + " kb"
    else:
        totalSize = str(totalBytes / 1024 / 1024) + " mb"

    return dtypes[dtype] + '[' + ','.join([str(s) for s in size]) + "] (" + totalSize + ")"

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
            print(f'{indent}    buffer {name} is {printParam(buffer.shape, buffer.dtype)}')
        for name,param in m.named_parameters(path, recurse=False):
            print(f'{indent}    parameter {name} is {printParam(param.shape, param.dtype)}')

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
        except:
            print(f"{indent}    (not scripted)")
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

#introspect_model(model)

def instrument_runtimes(model: Module, output: Dict[str, float]):
    """
    Instrument the given model to collect runtimes of each type of layer in the given dict
    """

    def recurse_module(m: Module, recursion: int, path: str):

        n: str = m._get_name()
        indent: str = ' ' * recursion * 4
        start: List[float] = [0]

        def pre_hook(module: Module, args: Tuple, kwargs: Dict):
            #print(f"{indent}{n} ENTER")
            start[0] = time.time()
            return None

        def post_hook(module: Module, args: Tuple, kwargs: Dict, output: Tuple):
            finish = time.time()
            elapsed = finish - start[0]
            if elapsed > 0.001:
                s = path + " " + n
                print(f"    {s:60} {int(elapsed*100000)*10:10}us")
            return None

        m.register_forward_pre_hook(pre_hook, with_kwargs=True)
        m.register_forward_hook(post_hook, with_kwargs = True)

        for name,child in m.named_children():
            recurse_module(child, recursion + 1, path + "." + name)

    recurse_module(model, 0, '')

runtimes: Dict[str, float] = {}
#instrument_runtimes(model, runtimes)

def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def float16_to_int(i: Tensor) -> Tensor:
    assert i.storage_offset() == 0
    #print(f"stride {i.stride()} storage offset {i.storage_offset()}")
    if i.dtype != torch.float16:
        raise RuntimeError(f"expected fp16 tensor; got {i.dtype}")
    #print(f"shape in {i.shape} shape out {torch.ShortTensor(i.to('cpu').untyped_storage()).shape}")
    return torch.ShortTensor(i.to('cpu').untyped_storage()).reshape_as(i)

def int_to_float16(i: Tensor) -> Tensor:
    assert i.storage_offset() == 0
    if not isinstance(i, torch.ShortTensor):
        raise RuntimeError(f"expected i16 tensor; got {type(i)}")
    return torch.HalfTensor(i.to('cpu').untyped_storage()).reshape_as(i)

def bfloat16_to_int(i: torch.Tensor) -> torch.ShortTensor:
    if not isinstance(i.storage(), torch.BFloat16Storage):
        raise RuntimeError(f"expected bf16 tensor; got {type(i)}")
    return torch.ShortTensor(i.untyped_storage())

def int_to_bfloat16(i: torch.ShortTensor) -> Tensor:
    if not isinstance(i, torch.ShortTensor):
        raise RuntimeError(f"expected i16 tensor; got {type(i)}")
    return torch.Tensor(torch.BFloat16Storage(i.untyped_storage()))

def float32_to_int(i: torch.FloatTensor) -> torch.IntTensor:
    return torch.IntTensor(i.untyped_storage())

def int_to_float32(i: torch.IntTensor) -> torch.FloatTensor:
    return torch.FloatTensor(i.untyped_storage())

def float64_to_int(i: torch.DoubleTensor) -> torch.LongTensor:
    return torch.LongTensor(i.untyped_storage())

def int_to_float64(i: torch.IntTensor) -> torch.DoubleTensor:
    return torch.DoubleTensor(i.untyped_storage())

# Maximum number of floating point ULPS that we accept as a difference in a computation
MAX_ULPS=300

# Maximum absolute difference we accept as a difference in a computation
MAX_DIFF=1e-6



def ulps_difference(i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
    """
    Returns the number of ulps in difference between the two tensors, taking into account overflow
    """

    return torch.minimum(torch.abs(i1 - i2), 32768 - torch.abs(i2 - i1))

    diff1 = i1 - i2
    diff2 = i2 - i1

    return torch.minimum(diff1, diff2)

@dataclass
class TensorDifference:
    expected: Tensor
    received: Tensor
    
    index: int = 0
    v1: Optional[Tensor] = None
    v2: Optional[Tensor] = None

    i1: int = 0
    i2: int = 0

    difference: float = 0.
    ulps: int = 0

    message: str = ''

    def __bool__(self) -> bool:
        return self.message != '' or self.difference != 0

    def is_significant(self) -> bool:
        """
        Returns true if and only if the difference between the values is considered
        significant.
        """

        return abs(self.i1 - self.i2) > MAX_ULPS

    @staticmethod
    def between(expected: Tensor, received: Tensor) -> 'TensorDifference':
        """
        Compare the two tensors, first converting to a common representation to ensure that different
        data types are taken into account.

        Returns None if there is no difference, or a string describing it if there is one.
        """

        result = TensorDifference(expected, received)

        sh1,dt1 = expected.shape, expected.dtype
        sh2,dt2 = received.shape, received.dtype

        if sh1 != sh2:
            result.message = "tensor shapes differ: {sh1} vs {sh2}"
            return result

        if expected.eq(received).all():
            return result

        flat1 = expected.to('cpu').flatten()
        flat2 = received.to('cpu').flatten()

        if not dt1.is_floating_point:
            diff = torch.abs(flat2 - flat1)
            result.difference = diff.max().item()
            result.index = int(diff.argmax().item())

        else:
            if False:
                # floating point

                tinfo = {
                    torch.float16:   (2, torch.int16,  float16_to_int, int_to_float16),
                    torch.bfloat16:  (2, torch.int16,  bfloat16_to_int, int_to_bfloat16),
                    torch.float32:   (4, torch.int32,  float32_to_int, int_to_float32),
                    torch.float64:   (8, torch.int64,  float64_to_int, int_to_float64),
                }

                w1,it1,to_int1,from_int1 = tinfo[dt1]
                w2,it2,to_int2,from_int2 = tinfo[dt2]

                if w2 < w1:
                    dt,w,it,to_int,from_int = dt2,w2,it2,to_int2,from_int2
                else:
                    dt,w,it,to_int,from_int = dt1,w1,it1,to_int1,from_int1

            # Temporary
            dt = torch.float16
            to_int = float16_to_int
            from_int = int_to_float16

            #print(f'dt = {dt} dt1 = {dt1} dt2 = {dt2}')

            conv1 = flat1.to(dt)
            conv2 = flat2.to(dt)

            if conv1.eq(conv2).all():
                return result

            asint1 = to_int(conv1)
            asint2 = to_int(conv2)

            assert from_int(asint1).eq(conv1).all()
            assert from_int(asint2).eq(conv2).all()

            diff = ulps_difference(asint1, asint2)

            max_diff = diff.max()
            max_diff_el = torch.argmax(diff)
            result.ulps = int(max_diff.item())
            result.index = int(max_diff_el.item())
            result.i1 = int(asint1[result.index])
            result.i2 = int(asint2[result.index])
            result.difference = float(conv2[max_diff_el] - conv1[max_diff_el].item())

            assert torch.flatten(diff)[max_diff_el] == max_diff

            #el1 = torch.flatten(expected)[max_diff_el]
            #el2 = torch.flatten(received)[max_diff_el]
            #eli1 = torch.flatten(asint1)[max_diff_el]
            #eli2 = torch.flatten(asint2)[max_diff_el]

            #print(f'max_diff_fp = {max_diff_fp} max_diff = {max_diff} max_diff_el = {max_diff_el} dt1 = {dt1} dt2 = {dt2} dt = {dt} shape = {sh1} el1 = {el1} el2 = {el2} eli1 = {eli1} eli2 = {eli2}')
            #
            #if max_diff <= MAX_ULPS or max_diff >= 32768 - MAX_ULPS:
            #    return None

            #return f"tensors differ; max diff is {max_diff} ulps"

        result.v1 = flat1[result.index]
        result.v2 = flat2[result.index]
        return result



def device_comparison_mode(model: Module, master_device: torch.device, master_dtype: torch.dtype, scenarios: List[Tuple[torch.device, torch.dtype]]):
    """
    Instrument the given model to run on multiple devices/datatypes, comparing the results
    between the (canonical) CPU version and the others, to identify where accuracy errors
    are coming from.
    """

    def recurse_module(m: Module, recursion: int, path: str):

        n: str = m._get_name()
        indent: str = ' ' * recursion * 4

        slave_modules: List[Tuple[Module,torch.device,torch.dtype]] = []

        for device,dtype in scenarios:
            slave_modules.append((copy.deepcopy(m).to(torch.device(device), dtype), device, dtype))

        def pre_hook(module: Module, args: Tuple, kwargs: Dict):
            #print(f"    {path:60} {n} ENTER")
            return None

        def post_hook(module: Module, args: Tuple, kwargs: Dict, output: Tuple):
            master_output = output
            #master_output = module.forward(*args, **kwargs)

            debug: bool = n == "Embedding"
            if debug:
                #args[0].contiguous()
                #while True:
                #    master_output = module.forward(*args, **kwargs)

                print(f"Embedding: input {args[0].shape}", args[0])
                print(f"Embedding: output {master_output[0].shape}", master_output[0][0][0:10])
                assert isinstance(module, torch.nn.Embedding)
                print("weights are", module.weight.shape)
                idx = args[0][0][0].item()
                print("index weights are", module.weight[idx][0:10])
                index: Tensor = args[0]
                print(f"index.storage = { index.storage() } offset = { index.storage_offset() })")
                if module.weight[idx].ne(master_output[0][0]).any():
                    for i in range(module.weight.shape[0]):
                        if module.weight[i].eq(master_output[0][0]).all():
                            print('equal i = ', i)

                    raise RuntimeError("embedding got the wrong thing")

            # jit._get_trace_graph
            #graph = jit.trace(m.forward, example_inputs=args, example_kwarg_inputs=kwargs)

            for slave,device,dtype in slave_modules:
                prefix = f"    {path:50} {n:20} {device}"

                if debug:
                    assert isinstance(slave, torch.nn.Embedding)
                    assert isinstance(module, torch.nn.Embedding)
                    assert module.weight.to('cpu').eq(slave.weight.to('cpu')).all()


                try:
                    def fixup_argument(arg, device, dtype):
                        #print('fixup argument', type(arg), isinstance(arg, Generator))
                        if isinstance(arg, torch.Tensor):
                            # dtype is only changed for floating point types, otherwise we just shift device
                            if arg.dtype.is_floating_point:
                                return arg.to(device, dtype)
                            else:
                                return arg.to(device)
                        elif isinstance(arg, List) or isinstance(arg, Generator):
                            return list([fixup_argument(a, device, dtype) for a in arg])
                        elif isinstance(arg, Tuple):
                            return tuple((fixup_argument(a, device, dtype) for a in arg))
                        elif isinstance(arg, Dict):
                            return dict({k: fixup_argument(v, device, dtype) for k,v in arg.items()})
                        else:
                            return arg

                    def fixup_to_slave(arg):
                        return fixup_argument(arg, device, dtype)

                    def fixup_to_master(arg):
                        return fixup_argument(arg, master_device, master_dtype)

                    if False:
                        for a in args:
                            print("arg ", type(a))
                        for k,v in kwargs.items():
                            print("kw", k, type(v))
                        if 'layer_past' in kwargs:
                            print('layer_past', kwargs['layer_past'])

                    slave_output = slave.forward(*fixup_to_slave(args), **{k: fixup_to_slave(v) for k,v in kwargs.items()})

                    if debug:
                        print(f"Embedding: slave output {slave_output[0].shape}", slave_output[0][0][0:10])
                        print("index weights are", slave.weight[idx][0:10])

                    slave_output = fixup_to_master(slave_output)

                except Exception as e:
                    print(prefix + f" FAILED {e}")
                    raise
                    return None

                biggest: List[Optional[TensorDifference]] = [None]

                def compare_output(expected, received, name: str):
                    #print("compare: expected ", expected, "received", received)
                    if type(expected) != type(received):
                        # Coerce to tuple if one is a tuple
                        if isinstance(expected, tuple) or isinstance(received, tuple):
                            return compare_output(tuple(expected), tuple(received), name)
                        # Coerce to dict if one is a dict
                        if isinstance(expected, dict) or isinstance(received, dict):
                            return compare_output(dict(expected), dict(received), name)
                        raise RuntimeError(f'{name} types differ: {type(expected)} vs {type(received)}')

                    if isinstance(expected, torch.Tensor):
                        if expected.shape != received.shape:
                            raise RuntimeError(f'{name} tensor shapes differ: {expected.shape} vs {received.shape}')
                        res = TensorDifference.between(expected, received)
                        if biggest[0] is None or res.ulps > biggest[0].ulps:
                            biggest[0] = res

                        #if res.is_significant():
                        #    print(res)
                        #if res:
                        #    print(res)
                            #raise RuntimeError(f'{name} tensors differ: {res}')
                        return

                    elif isinstance(expected, list) or isinstance(expected, tuple):
                        if len(expected) != len(received):
                            raise RuntimeError("{name} lengths differ: {len(expected)} vs {len(received})")
                        for i in range(len(expected)):
                            compare_output(expected[i], received[i], name + "[" + str(i) + "]")
                    elif isinstance(expected, dict):
                        items1 = sorted(expected.items())
                        items2 = sorted(received.items())

                        keys1 = [k for k,_ in items1]
                        keys2 = [k for k,_ in items2]

                        if keys1 != keys2:
                            raise RuntimeError(f"{name} dict keys differ: {keys1} vs {keys2}")

                        # Compare the values for each key
                        for i in range(len(keys1)):
                            k,v1 = items1[i]
                            _,v2 = items2[i]
                            compare_output(v1, v2, name + "{" + k + "}")
                    else:
                        raise RuntimeError(f'{name} outputs differ: {expected} vs {received}')

                compare_output(master_output, slave_output, str(device) + " " + str(dtype) + " " + path)
                print(prefix + f" {biggest[0].ulps:>5} ulps")
                if biggest[0].ulps > 1000:
                    raise RuntimeError("too big an ULPS difference")

            return None

        m.register_forward_pre_hook(pre_hook, with_kwargs=True)
        m.register_forward_hook(post_hook, with_kwargs = True)

        for name,child in m.named_children():
            recurse_module(child, recursion + 1, path + "." + name)

    recurse_module(model, 0, '')

#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float32), (torch.device('mps'), torch.float16)])
#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float16)])
#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float32)])
device_comparison_mode(model, device, dtype, [(torch.device('cpu'), torch.float32)])
#device_comparison_mode(model, device, dtype, [(torch.device('mps'), torch.float32)])

#def printModule(m: Module):
#    print('got module', m)

#transformer.apply(printModule)

#scripted = jit.script(model)
#print("scripted=", scripted)

#weights = init_model(),klass = "EleutherAI/gpt-j-6B",GPTJForCausalLM
#weights,klass = "EleutherAI/gpt-neox-20b",GPTNeoXForCausalLM

#model = klass.from_pretrained(
#   weights,
#    revision="float16",
#    torch_dtype=torch.float16,
#    low_cpu_mem_usage=True
#
#).to(device, torch.float32)

#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = (

    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "

    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "

    "researchers was the fact that the unicorns spoke perfect English."

)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)



print(input_ids)

#for name,module in model.named_children():
#    print(name)

#traced = jit.trace(model.generate, example_inputs=(input_ids,), example_kwarg_inputs={'do_sample': True, 'temperature': 0.9, 'max_length': 100})

def forward_hook(module: Module, args: Tuple, kwargs: Dict):
    return None
    def print_arg(arg: Any):
        if isinstance(arg, torch.Tensor):
            return f"Tensor {arg.dtype} {arg.shape}"
        return str(arg)[0:100]
    #print(f"forward: {module._get_name()} with {len(args)} args and {len(kwargs)} kwargs: {[print_arg(arg) for arg in args]}")
    print(f"forward: {module._get_name()}")
    if False:
        for i in range(len(args)):
            print(f"  arg {i:>30}: {print_arg(args[i])}")
        for k,v in kwargs.items():
            print(f"  {k:>30}: {print_arg(v)}")
        #for  { {name: print_arg(val) for name,val in kwargs.items()}}")
    return None

#model.register_forward_pre_hook(forward_hook, with_kwargs=True)

gen_tokens = model.generate(
    input_ids,
    do_sample=False,
    temperature=0.9,
    max_length=100,
)

gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
