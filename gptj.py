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
from torch._C import Graph, Node, dtype as cdtype

from ansi.color import bg, fg
from ansi.color.fx import reset
from device_comparison import fixup_argument, compare_output
from tensor_comparisons import TensorDifference

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
from runtimes import instrument_runtimes, print_elapsed


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

exit(0)

#print(invocations)
#for call in invocations.calls:
#    print(f"  {call}")

remove_invocations()

print(inspect.signature(invocations.m.forward))

sd = invocations.summarize()
sd.print_args()

def find_operation(n: Node) -> Operation:
    return default_find_operation(n)

def compare_operation(n: Node) -> Operation:
    op = default_find_operation(n)

    def new_interpreter(s: Scope, n: Node) -> Any:
        return op.interpreter(s, n)

    op.interpreter = new_interpreter
    return op

def time_operation(n: Node) -> Operation:
    old_op = default_find_operation(n)
    new_op = Operation(old_op.name + '.timed', old_op.source)

    def new_interpreter(s: Scope, n: Node) -> Any:
        before = time.time()
        res = old_op.interpret(s, n)
        after = time.time()
        print(n.kind(), print_elapsed(after-before))
        return res

    new_op._interpreter = new_interpreter
    return new_op

def make_compare_operation(device: torch.device|str) -> Callable[[Node],Operation]:

    def to_device(val: Any) -> Any:
        return fixup_argument(val, torch.device(device))

    def update_scope(s1: Scope, s2: Scope):
        for i in range(len(s2.vars), len(s1.vars)):
            name,val=s1.vars[i]
            s2.add_var(name, to_device(val))

    def compare_operation(n: Node) -> Operation:
        old_op = default_find_operation(n)
        new_op = Operation(old_op.name + '.compare', old_op.source)

        scopes: Dict[Scope, Scope] = {}

        def new_interpreter(s1: Scope, n: Node) -> Any:
            if s1 not in scopes:
                scopes[s1] = Scope()

            s2: Scope = scopes[s1]
            update_scope(s1, s2)
            len_before = len(s1.vars)

            before1 = time.time()
            res1 = old_op.interpret(s1, n)
            after1 = time.time()

            before2 = time.time()
            res2 = old_op.interpret(s2, n)
            after2 = time.time()

            max_difference = TensorDifference()
            for i in range(len_before, len(s1.vars)):
                name1,val1 = s1.vars[i]
                name2,val2 = s2.vars[i]

                assert name1 == name2

                try:
                    difference = compare_output(val1, val2, name1)
                    if max_difference.ulps < difference.ulps:
                        max_difference = difference
                except Exception as e:
                    if str(e).find('cpu') == -1:
                        print(f"{fg.red}{e}{reset}")
                        raise

                #s2.vars[i] = (name2, to_device(val1))

            print(f"{n.kind():30} {print_elapsed(after1-before1):8} {print_elapsed(after2-before2):8} {max_difference.ulps:6} ulps")
            return res1

        new_op._interpreter = new_interpreter
        return new_op

    return compare_operation

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
