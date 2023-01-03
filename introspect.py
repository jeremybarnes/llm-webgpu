import torch
import torch.jit as jit
from torch.jit import RecursiveScriptModule, ScriptModule
from torch.nn import Module
from typing import Dict
import inspect

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
