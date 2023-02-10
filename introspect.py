import torch
import torch.jit as jit
from torch.jit import RecursiveScriptModule, ScriptModule
from torch.nn import Module
from typing import Dict, List, Any, Tuple, Optional, Callable, Sequence, SupportsInt, Union, Iterator
import inspect
from dataclasses import dataclass, field
import time
import copy
from runtimes import print_elapsed
from collections import defaultdict, OrderedDict
from torch.utils.hooks import RemovableHandle

def short_dtype(dtype: torch.dtype):
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

def printParam(size: torch.Size, dtype: torch.dtype, device: torch.device):
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

    return short_dtype(dtype) + '[' + ','.join([str(s) for s in size]) + "] (" + totalSize + ")" + " " + str(device)

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
            print(f'{indent}    buffer {name} is {printParam(buffer.shape, buffer.dtype, buffer.device)}')
        for name,param in m.named_parameters(path, recurse=False):
            print(f'{indent}    parameter {name} is {printParam(param.shape, param.dtype, param.device)}')

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


class Arg:
    """
    Base class for information about an argument to a function.
    """

    def get_type(self) -> type:
        """
        Return the type of this argument, or a superclass if the type can vary.
        This should not return NoneType unless the argument is a constant None;
        instead, is_optional() should return true.
        """
        raise RuntimeError(f"Class {self.__class__.__name__} doesn't override get_type()")

    def is_optional(self) -> bool:
        """
        Return true if this is optional; in other words, if None is one of the possible
        values for the argument.  The other methods return information for the non-optional
        case.
        """
        return False

    def non_optional(self) -> 'Arg':
        """
        Returns the non-optional version of this type.  Default checks that is_optional() is
        false and returns self (which works for all non-optional types).
        """
        assert not self.is_optional()
        return self

    def get_dtype(self) -> Optional[torch.dtype]:
        """
        Return the dtype of this argument, None if it's not a tensor.
        """
        return None

    def get_device(self) -> Optional[torch.device]:
        """
        Return the device of this argument, None if it's not a tensor.
        """
        return None

    def get_shape(self) -> Optional['TensorShapes']:
        """
        Return the shape of this argument, None if it's not a tensor.
        """
        return None

class UnknownArg(Arg):
    def __str__(self) -> str:
        return "Unknown()"

    def get_type(self) -> type: return object

class ConstantArg(Arg):
    value: Any

    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        return f"Constant({self.value})"

    def get_type(self) -> type: return type(self.value)

@dataclass
class ShapeRange:
    min: int = 10000000000
    max: int = 0

    def is_const(self) -> bool:
        """
        Is this shape a constant int value?
        """
        return self.min == self.max

    def const_value(self) -> int:
        """
        Return the constant shape for this dimension.

        PRE: is_const() is true.
        """
        if self.min != self.max:
            raise RuntimeError("asked for constant value with is_const() false")
        return self.min

    def do(self, val: int):
        self.min = min(self.min, val)
        self.max = max(self.max, val)

    def add(self, val: 'ShapeRange'):
        self.do(val.min)
        self.do(val.max)

    def __init__(self, val:Optional[Union[SupportsInt,'ShapeRange']] = None):
        if isinstance(val, ShapeRange):
            self.min = val.min
            self.max = val.max
        elif val is None:
            pass
        else:
            self.min = self.max = int(val)

    def __repr__(self) -> str:
        if self.max < self.min:
            return "[*]"
        elif self.max == self.min:
            return f"[{self.max}]"
        else:
            return f"[{self.min}-{self.max}]"

@dataclass
class TensorShape:
    """
    Shape range for a fixed number of dimensions
    """
    dims: List[ShapeRange] = field(default_factory=list)

    def __init__(self, dims: Sequence[ShapeRange|SupportsInt]):
        self.dims = [ShapeRange(s) for s in dims]

    def __len__(self) -> int:
        return len(self.dims)

    def __getitem__(self, item: int) -> ShapeRange:
        return self.dims[item]

    def __iter__(self) -> Iterator[ShapeRange]:
        return iter(self.dims)

    def __repr__(self) -> str:
        return ''.join([str(s) for s in self.dims])

    def do(self, shape: List[int]):
        assert len(shape) == len(self)
        for dim,sh in zip(self.dims, shape):
            dim.do(sh)

    def add(self, shape: 'TensorShape'):
        assert len(shape) == len(self)
        for dim,sh in zip(self.dims, shape.dims):
            dim.add(sh)

    @staticmethod
    def from_tensor(t: torch.Tensor) -> 'TensorShape':
        """
        Return a TensorShape object from a single tensor.
        """
        return TensorShape(t.size())

@dataclass
class TensorShapes:
    """
    Shape range for a variable number of dimensions.  For when something is called
    with multiple tensor dimensions.
    """

    lengths: Dict[int, TensorShape] = field(default_factory=dict)

    def add(self, shape: TensorShape):
        l = len(shape.dims)
        if l not in self.lengths:
            self.lengths[l] = shape
        else:
            self.lengths[l].add(shape)

    def do(self, size: Sequence[SupportsInt]):
        shape = TensorShape(size)
        l = len(shape)
        if l in self.lengths:
            self.lengths[l].add(shape)
        else:
            self.lengths[l] = shape

    @staticmethod
    def from_tensor(t: torch.Tensor) -> 'TensorShapes':
        """
        Return a TensorShapes object from a single tensor.
        """
        result = TensorShapes()
        result.add(TensorShape.from_tensor(t))
        return result

    def __repr__(self) -> str:
        return ' | '.join(str(shape) for len,shape in sorted(self.lengths.items()))


class TensorArg(Arg):
    """
    Describes a tensor-valued argument.
    """
    dtype: torch.dtype
    device: torch.device
    shape: TensorShapes

    def __init__(self, dtype: torch.dtype, device: torch.device, shape: TensorShapes):
        self.dtype = dtype
        self.device = device
        self.shape = shape

    def __repr__(self) -> str:
        return f"Tensor({self.dtype}{self.shape}{self.device})"

    def get_type(self) -> type: return torch.Tensor
    def get_dtype(self) -> Optional[torch.dtype]: return self.dtype
    def get_device(self) -> Optional[torch.device]: return self.device
    def get_shape(self) -> Optional['TensorShapes']: return self.shape


class OptionalArg(Arg):
    """
    Describes an argument that can be either None or another value, representing
    optional or defaulted values.
    """
    value: Arg

    def __init__(self, value: Arg):
        self.value = value

    def __repr__(self) -> str:
        return f"Optional({self.value})"

    def is_optional(self) -> bool: return True
    def non_optional(self) -> 'Arg': return self.value
    def get_type(self) -> type: return self.value.get_type()
    def get_dtype(self) -> Optional[torch.dtype]: return self.value.get_dtype()
    def get_device(self) -> Optional[torch.device]: return self.value.get_device()
    def get_shape(self) -> Optional['TensorShapes']: return self.value.get_shape()

class TupleArg(Arg):
    """
    Fixed-length, non-homogeneous tuple.
    """

    values: List[Arg]

    def __init__(self, values: List[Arg]):
        self.values = values

    def __repr__(self) -> str:
        return f"Tuple({self.values})"

    def get_type(self) -> type: return tuple

class ListTupleArg(Arg):
    """
    Variable length, homogeneous tuple.
    """

    length: ShapeRange
    value: Arg

    def __init__(self, length: ShapeRange, value: Arg):
        self.length = length
        self.value = value

    def __repr__(self) -> str:
        return f"ListTuple({self.value}{self.length})"

    def get_type(self) -> type: return tuple

@dataclass
class Invocation:
    args: Tuple
    kwargs: OrderedDict[str, Any] = field(default_factory = dict)
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
                    return printParam(arg.size(), arg.dtype, arg.device) + " " + str(arg.device)
            else:
                return str(arg)

        summarized_args = list(map(summarize_arg, self.args))
        summarized_kwargs = {k: summarize_arg(v) for k,v in self.kwargs.items()}

        return f"Invocation(elapsed={print_elapsed(self.elapsed)} args={summarized_args} kwargs={summarized_kwargs})"

@dataclass
class ArgumentData:
    count: int = 0
    types: Dict[type, int] = field(default_factory = dict)
    tensor_dtypes: Dict[torch.dtype, int] = field(default_factory = dict)
    tensor_devices: Dict[torch.device, int] = field(default_factory = dict)
    tensor_shapes: Dict[torch.Size, int] = field(default_factory = dict)
    tensor_values: Dict[torch.Tensor, int] = field(default_factory = dict, repr=False)
    tuple_lengths: Dict[int, int] = field(default_factory=dict)
    tuple_args: List['ArgumentData'] = field(default_factory=list)
    values: Dict[Any, int] = field(default_factory = dict, repr=False)

    def is_homogeneous(self, other: 'ArgumentData'):
        """
        Tells us whether this and the other type are homogeneous and can be combined without
        much loss of generality.
        """

        #print("is_homogeneous", str(self.summarize()), str(other.summarize()))

        return str(self.summarize()) == str(other.summarize())

    def make_homogeneous(self, other: 'ArgumentData') -> 'ArgumentData':
        """
        Make a version that maps onto all of the others.
        """
        v = ArgumentData()
        v.combine(self)
        v.combine(other)
        return v

    def combine(self, other: 'ArgumentData'):
        """
        Combine the two arguments, producing one which can accept either of the inputs.
        """
        self.count += other.count
        
        def update_counts(v1: Dict[Any, int], v2: Dict[Any, int]):
            for k,v in v2.items():
                v1[k] = v1.get(k, 0) + v

        update_counts(self.types, other.types)
        update_counts(self.tensor_dtypes, other.tensor_dtypes)
        update_counts(self.tensor_devices, other.tensor_devices)
        update_counts(self.tensor_shapes, other.tensor_shapes)
        update_counts(self.tensor_values, other.tensor_values)
        update_counts(self.tuple_lengths, other.tuple_lengths)
        update_counts(self.values, other.values)

        while len(self.tuple_args) < len(other.tuple_args):
            self.tuple_args.append(ArgumentData())

        for i in range(len(other.tuple_args)):
            self.tuple_args[i].combine(other.tuple_args[i])

    def add(self, a: Any):
        """
        Add the given argument instance to the analysis.
        """
        at = type(a)

        self.count += 1
        self.types[at] = self.types.get(at, 0) + 1

        if isinstance(a, torch.Tensor):
            sh = a.shape
            dt = a.dtype
            dv = a.device
            self.tensor_shapes[sh] = self.tensor_shapes.get(sh, 0) + 1
            self.tensor_dtypes[dt] = self.tensor_dtypes.get(dt, 0) + 1
            self.tensor_devices[dv] = self.tensor_devices.get(dv, 0) + 1
            self.tensor_values[a] = self.tensor_values.get(a, 0) + 1

            if len(sh) == 0:  # scalar
                pass
        elif isinstance(a, dict):
            pass
        elif isinstance(a, list):
            pass
        elif isinstance(a, tuple):
            tl = len(a)
            self.tuple_lengths[tl] = self.tuple_lengths.get(tl, 0) + 1
            while len(self.tuple_args) < tl:
                self.tuple_args.append(ArgumentData())

            for i in range(tl):
                self.tuple_args[i].add(a[i])
        elif isinstance(a, (float, int, str, bool)):
            self.values[a] = self.values.get(a, 0) + 1
            pass

    def summarize(self) -> Arg:
        if len(self.types) == 0:
            return UnknownArg()

        non_optional_types = self.types.copy()

        def identity(x: Arg) -> Arg:
            return x

        def make_optional(x: Arg) -> Arg:
            return OptionalArg(x)

        wrapper = identity

        if type(None) in self.types:
            if len(self.types) == 1:
                return ConstantArg(None)
            del non_optional_types[type(None)]
            wrapper = make_optional
        
        if len(non_optional_types) > 1:
            raise NotImplementedError(f"Can't handle multiple types {self.types} yet")
        first_type: type = list(non_optional_types.keys())[0]

        if len(self.values) == 1:
            first_value: Any = list(self.values.keys())[0]
            return ConstantArg(first_value)

        if issubclass(first_type, torch.Tensor):
            if len(self.tensor_dtypes) > 1:
                raise NotImplementedError("Can't handle multiple tensor types yet")
            tensor_dtype = list(self.tensor_dtypes)[0]

            if len(self.tensor_devices) > 1:
                raise NotImplementedError("Can't handle multiple tensor devices yet")
            tensor_device = list(self.tensor_devices)[0]

            shapes = TensorShapes()

            for sh,_ in self.tensor_shapes.items():
                shapes.add(TensorShape(sh))

            return wrapper(TensorArg(tensor_dtype, tensor_device, shapes))

        elif issubclass(first_type, tuple):
            if len(self.tuple_lengths) > 1:

                # Assume the tuple is like a list
                lens = ShapeRange()
                for k,l in self.tuple_lengths.items():
                    lens.do(l)

                # Homogenize the tuple length data
                arg = ArgumentData()
                for a in self.tuple_args:
                    arg.combine(a)

                return wrapper(ListTupleArg(lens, arg.summarize()))
            else:
                if len(self.tuple_args) == 0:
                    return wrapper(TupleArg([]))

                        
                try:
                    def get_homogeneous(a1: ArgumentData, a2: ArgumentData):
                        if a1.is_homogeneous(a2):
                            return a1.make_homogeneous(a2)
                        else:
                            raise RuntimeError("can't make homogeneous")

                    # Try to get a homogenous type for the tuple arguments
                    common = self.tuple_args[0]
                    for a in self.tuple_args:
                        common = get_homogeneous(common, a)
                    lens = ShapeRange()
                    lens.do(len(self.tuple_args))
                    return wrapper(ListTupleArg(lens, common.summarize()))
                except:
                    pass

                # Homogeneous length tuple, assume it's differently typed
                args = [a.summarize() for a in self.tuple_args]
                return wrapper(TupleArg(args))

        raise NotImplementedError(f"Summarize of argument of type {first_type}")

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
    calls: List[Invocation] = field(default_factory = list)
    children: Dict[str, 'Invocations'] = field(default_factory = dict)

    def total_runtime(self) -> float:
        return sum([c.elapsed for c in self.calls])

    def __str__(self) -> str:
        return f"Invocations(path={self.path} module={self.m._get_name()} runtime={print_elapsed(self.total_runtime())} ncalls={len(self.calls)} nchildren={len(self.children)})" # sig={inspect.signature(self.m.forward)})"

    def summarize(self) -> SummaryData:
        result = SummaryData()

        max_nargs = 0
        for c in self.calls:
            nargs = len(c.args)
            result.arg_lengths[nargs] = result.arg_lengths.get(nargs, 0) + 1
            max_nargs = max(max_nargs, nargs)

        result.args = [ArgumentData() for _ in range(max_nargs)]

        for c in self.calls:
            for i in range(len(c.args)):
                a = c.args[i]
                ad = result.args[i]
                ad.add(a)

            for k,a in c.kwargs.items():
                if k not in result.kwargs:
                    result.kwargs[k] = ArgumentData()
                ad = result.kwargs[k]
                ad.add(a)

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