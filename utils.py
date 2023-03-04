import torch
from typing import Any

def _short_dtype(dtype: torch.dtype):
    dtypes = {
        torch.float32: 'f32',
        torch.float16: 'f16',
        torch.int8: 'i8',
        torch.uint8: 'u8',
        torch.int32: 'i32',
        torch.int64: 'i64',
        torch.bool: 'b'
    }

    return dtypes[dtype]

def _print_param(size: torch.Size, dtype: torch.dtype, device: torch.device):
    dsizeof = {
        torch.float32: 4,
        torch.float16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int32: 4,
        torch.int64: 8,
        torch.bool: 1,
    }

    totalBytes = size.numel() * dsizeof[dtype]
    if totalBytes < 1024:
        totalSize = str(totalBytes) + " bytes"
    elif totalBytes < 1024*1024:
        totalSize = str(totalBytes / 1024) + " kb"
    else:
        totalSize = str(totalBytes / 1024 / 1024) + " mb"

    return _short_dtype(dtype) + '[' + ','.join([str(s) for s in size]) + "] (" + totalSize + ")" + " " + str(device)

def _print_type(v: Any) -> str:
    return str(type(v))

def _print_value(v: Any) -> str:
    def shorten(s: str) -> str:
        if len(s) > 55:
            s = s[0:50] + "..."
        return s

    if isinstance(v, (bool,int,float,type(None))):
        return str(v)
    elif isinstance(v, str):
        return shorten('"' + str(v) + '"')
    elif isinstance(v, list):
        return shorten(str(v))
    elif isinstance(v, torch.Tensor):
        return summarize_tensor(v)
    elif isinstance(v, tuple):
        return shorten('(' + ','.join([_print_value(v2) for v2 in v]) + ')')

    else:
        result = str(v).replace('\n,','')
        return _print_type(v) + " " + result[0:50]

def summarize_tensor(t: torch.Tensor) -> str:
    result = _short_dtype(t.dtype) + '[' + ']['.join([str(s) for s in t.shape]) + ']'
    nel = t.numel()
    strt = str(t).replace('Parameter containing:\n', '').replace('tensor(', '').replace('\n', ' ')
    if nel > 10:
        return result + "=" + strt[0:50] + "..."
    else:
        return result + "=" + strt

def typeify(v: Any) -> Any:
    if isinstance(v, tuple):
        return tuple(typeify(e) for e in v)
    elif isinstance(v, list):
        return list([typeify(e) for e in v])
    else:
        return type(v)


