from aten import int_to_dtype, dtype_to_int, int_to_memory_format, summarize_tensor

import torch
from typing import Type, Tuple, Any, Dict, List, Optional, Generator, Callable, OrderedDict, Sequence, Mapping

from torch.nn import Module
from torch import Tensor, ScriptModule, ScriptFunction, Value, Size, Block, dtype, memory_format, device, scalar_tensor, add, tanh
import torch.jit as jit

from torch._C import Graph, Node, dtype as cdtype
import inspect
from dataclasses import dataclass, field
from ansi.color import bg, fg
from ansi.color.fx import reset
import traceback

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
    elif isinstance(v, Tensor):
        return summarize_tensor(v)
    elif isinstance(v, tuple):
        return shorten('(' + ','.join([_print_value(v2) for v2 in v]) + ')')

    else:
        result = str(v).replace('\n,','')
        return _print_type(v) + " " + result[0:50]

Inputs = Tuple[Any, ...]
NodeInterpreter = Callable[['Scope', Node],Any]
NodeOperator = Callable[..., Tuple[Any,...]|Any]
NodeCompiler = Callable[[Node],NodeOperator]

@dataclass
class Operation:
    name: str
    source: str
    _interpreter: Optional[NodeInterpreter] = None
    _compiler: Optional[NodeCompiler] = None
    _operator: Optional[NodeOperator] = None

    def interpret(self, s: 'Scope', n: Node) -> Any:
        if self._interpreter is not None:
            return self._interpreter(s, n)
        else:
            op: NodeOperator = self.compile(n)
            return s.exec_node_op(n, op)

    def compile(self, n: Node) -> NodeOperator:
        if self._compiler is not None:
            return self._compiler(n)
        elif self._operator is not None:
            return self._operator
        else:
            raise RuntimeError(f"node {self.name} has no compiler at {self.source}")

_ops: Dict[str, List[Operation]] = {}
_num_op_impls: int = 0

def _add_op(op: Operation):
    global _num_op_impls
    if op.name in _ops:
        _ops[op.name].append(op)
    else:
        _ops[op.name] = [op]
    _num_op_impls += 1


def _add_builtin_ops():
    import torch.jit._builtins
    #import torch.jit.supported_ops
    #print(torch.jit.supported_ops._get_torchscript_builtins())
    #print(torch.jit.supported_ops._get_tensor_ops())
    ops = torch.jit._builtins._builtin_ops
    for method,name in ops:
        _add_op(Operation(name, "torch.jit._builtins", _operator=method))
    print(f"added {len(_ops)} builtin ops with {_num_op_impls} implementations")

_add_builtin_ops()

def interpret_op(name: str):
    def do_overrive(interpreter: NodeInterpreter) -> NodeInterpreter:
        frame = next(traceback.walk_stack(None))[0]
        source = str(frame.f_code)
        _add_op(Operation(name, source, _interpreter=interpreter))
        return interpreter
    return do_overrive

def compile_op(name: str):
    def do_overrive(compiler: NodeCompiler) -> NodeCompiler:
        frame = next(traceback.walk_stack(None))[0]
        source = str(frame.f_code)
        _add_op(Operation(name, source, _compiler=compiler))
        return compiler
    return do_overrive

def exec_op(name: str):
    def do_overrive(op: NodeOperator) -> NodeOperator:
        def interpret(scope: 'Scope', node: Node):
            return scope.exec_node_op(node, op)
        def compiler(n: Node) -> NodeOperator:
            return op
        frame = next(traceback.walk_stack(None))[0]
        source = str(frame.f_code)
        _add_op(Operation(name, source, interpret, compiler, op))
        return op
    return do_overrive

@compile_op("prim::Constant")
def prim_constant(n: Node) -> NodeOperator:
    val = n.output()
    def exec_prim_constant():
        return val.toIValue()
    return exec_prim_constant

@exec_op("prim::Uninitialized")
def prim_uninitialized():
    return None

@exec_op("prim::RaiseException")
def prim_raise_exception(msg: str, exctype: str):
    exctypes: Dict[str, type] = {
        'builtins.RuntimeError': RuntimeError
    }
    raise exctypes[exctype](msg)

@exec_op("prim::Print")
def prim_print(*args):
    print(fg.red,*args,reset,sep='')

@exec_op("prim::dtype")
def prim_dtype(t: Tensor) -> int:
    return dtype_to_int(t.dtype)

@exec_op("prim::device")
def prim_device(t: Tensor) -> torch.device:
    return t.device

@exec_op("prim::ListConstruct")
def prim_list_construct(*args: Any) -> List[Any]:
    return list([a for a in args])

@exec_op("prim::TupleConstruct")
def prim_tuple_construct(*args: Any) -> Tuple[Any, ...]:
    return tuple([a for a in args])

@exec_op("prim::TupleIndex")
def prim_tuple_index(t: Tuple[Any, ...], i: int) -> Any:
    return t[i]

@compile_op("prim::GetAttr")
def compile_prim_get_attr(n: Node) -> NodeOperator:
    a: str = n.s("name")
    def prim_get_attr(o: object) -> Any:
        return getattr(o, a)
    return prim_get_attr

@exec_op("prim::Return")
def do_return(*args) -> Any:
    ta = tuple((a for a in args))
    if len(ta) == 1:
        return ta[0]
    else:
        return ta

@exec_op("prim::unchecked_cast")
def prim_unchecked_cast(x):
    return x

@exec_op("prim::TupleUnpack")
def prim_tuple_unpack(x):
    return x

@exec_op("aten::format")
def aten_format(fmt: str, *args) -> str:
    return fmt.format(*args)

@exec_op("aten::scalar_tensor")
def aten_scalar_tensor(val, dtype=None, layout=None, device=None, pin_memory=None) -> Tensor:
    return scalar_tensor(val, dtype=int_to_dtype(dtype), layout=layout, device=device, pin_memory=pin_memory)

@compile_op("aten::size")
def compile_aten_size(n: Node) -> NodeOperator:
    def get_sizes(t: Tensor) -> List[int]:
        return list([int(s) for s in t.size()])

    def get_size(t: Tensor, dim: int) -> int:
        return t.size(dim)

    ninputs = n.inputsSize()
    if ninputs == 1:
        return get_sizes
    elif ninputs == 2:
        return get_size
    else:
        raise RuntimeError(f"couldn't get aten::size for {ninputs} inputs")

@exec_op("aten::__getitem__")
def get_item(l: Any, item: Any) -> Any:
    return l[item]

@exec_op("aten::view")
def aten_view(t: Tensor, v: List[int]):
    return t.view(v)

@exec_op("aten::contiguous")
def aten_contiguous(t: Tensor, m: int):
    return t.contiguous(memory_format=int_to_memory_format(m))

@compile_op("aten::slice")
def compile_aten_slice(n: Node) -> NodeOperator:
    #for arg in n.inputs():
    #    print("input", arg)
    def aten_slice_python(x: Any, start: Optional[int] = None, end: Optional[int] = None, step: int = 1):
        start = 0 if start is None else start
        end = len(x) if end is None else end
        step = 1 if step is None else step
        return x[start:end:step]

    def aten_slice_tensor(t: Tensor, dim: Optional[int] = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1):
        start = 0 if start is None else start
        dim = 0 if dim is None else dim
        end = t.size(dim) if end is None else end
        step = 1 if step is None else step

        #print(f"{fg.blue}do_slice (dim {dim}): {start}:{end}:{step}{reset}")

        if dim == 0:
            return t[start:end:step]
        elif dim == 1:
            return t[:,start:end:step]
        elif dim == 2:
            return t[:,:,start:end:step]
        elif dim == 3:
            return t[:,:,:,start:end:step]
        elif dim == 4:
            return t[:,:,:,:,start:end:step]
        elif dim == 5:
            return t[:,:,:,:,:,start:end:step]
        else:
            raise RuntimeError("only 5d slices currently handled")

    input1 = n.inputsAt(0)
    if input1.type().kind() == "TensorType":
        return aten_slice_tensor
    else:
        return aten_slice_python

@exec_op("aten::__isnot__")
def is_not(v: Any, o: Any) -> bool:
    return v is not o
    
@exec_op("aten::__is__")
def aten_is(v: Any, o: Any) -> bool:
    return v is o

@exec_op("aten::eq")
def aten_eq(v1: Any, v2: Any) -> Any:
    if isinstance(v1, Tensor):
        return v1.eq(v2)
    else:
        return v1 == v2

@exec_op("aten::pow")
def aten_pow(v1: Tensor, v2: Tensor|float) -> Tensor:
    return v1.pow(v2)

@exec_op("aten::tanh")
def aten_tanh(v1: Tensor) -> Tensor:
    return tanh(v1)

@exec_op("aten::to")
def aten_to(v1: Tensor, a1: Any, *rest) -> Any:
    def option1(v1: Tensor, idt: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[memory_format] = None):
        dt = int_to_dtype(idt)
        #print(f"{fg.green}trying to option 1 dt {dt}{reset}")
        assert memory_format is None
        return v1.to(dt, non_blocking, copy)
    def option2(v1: Tensor, dev: device, dt: dtype, non_blocking: bool = False, copy: bool = False, memory_format: Optional[memory_format] = None):
        #print(f"{fg.green}trying to option 2{reset}")
        assert memory_format is None
        return v1.to(dev, dt, non_blocking, copy)
    def option3(v1: Tensor, v2: Tensor, non_blocking: bool = False, copy: bool = False, memory_format: Optional[memory_format] = None):
        #print(f"{fg.green}trying to option 3{reset}")
        assert memory_format is None
        return v1.to(v2, non_blocking, copy)

    return option1(v1, a1, *rest)

@compile_op("aten::add")
def compile_aten_add(n: Node) -> NodeOperator:
    def aten_add_tensors(t1: Tensor, t2: Tensor, alpha = 1):
        return add(t1, t2, alpha=alpha)
    def aten_add(t1: Any, t2: Any):
        #print(f"aten_add: {type(t1)} + {type(t2)}")
        return t1 + t2

    # Choose which one based on the type of the first argument
    input1 = n.inputsAt(0)
    input2 = n.inputsAt(1)
    #print(input1.type(), input2.type())
    if n.inputsSize() == 3 or (input1.type().kind() == "TensorType" and input2.type().kind() == "TensorType"):
        return aten_add_tensors
    else:
        return aten_add

@interpret_op("prim::If")
def exec_prim_if(s: 'Scope', n: Node):
    #print(n)
    input = s.collect_inputs(n)
    cond: bool = input[0]
    blocks = list(n.blocks())
    block: Block
    if cond:
        block = blocks[0]
    else:
        block = blocks[1]

    for block_node in block.nodes():
        s.exec_node(block_node)

    return_node: Node = block.returnNode()

    res = s.exec_node(return_node)

    #print("adding outputs", res, n)

    s.add_outputs(res, n)

    return res

def default_find_operation(n: Node) -> Operation:
    k: str = n.kind()
    #print("finding op", k, _ops[k])
    res = _ops[k][-1]
    assert res.name == k
    return res

class Scope:
    vars: List[Tuple[str, Any]]
    var_names: Dict[str, int]
    _find_operation: Callable[[Node],Operation]

    def __init__(self, find_operation: Callable[[Node],Operation] = default_find_operation):
        self.vars = []
        self.var_names = {}
        self._find_operation = find_operation

    def add_var(self, name: str, val: Any):
        assert name not in self.var_names
        self.var_names[name] = len(self.vars)
        self.vars.append((name, val))

    def get_var(self, name: str) -> Any:
        return self.vars[self.var_names[name]][1]

    def print_var_index(self, i: int) -> str:
        n,v = self.vars[i]

        return f"{i:3} {n:30} {_print_value(v)}"

    def print_var_named(self, sn: str) -> str:
        return self.print_var_index(self.var_names[sn])

    def print_vars(self):
        for i in range(len(self.vars)):
            print(self.print_var_index(i))

    def collect_inputs(self, n: Node) -> Tuple[Any,...]:
        def get_input(input: Value) -> Any:
            n = input.debugName()
            #print("  ", self.print_var_named(n))
            return self.get_var(n)

        return tuple((get_input(input) for input in n.inputs()))

    def add_outputs(self, result: Tuple[Any,...] | Any, n: Node):
        #print(n.outputsSize(), "outputs")

        def do_output(o: Value, v: Any):
            self.add_var(o.debugName(), v)
            #print("  ", self.print_var_named(o.debugName()))

        if n.outputsSize() == 0:
            pass
        elif n.outputsSize() == 1:
            do_output(n.outputsAt(0), result)
        else:
            #print("result is", type(result))
            assert isinstance(result, tuple)
            for i in range(n.outputsSize()):
                o = n.outputsAt(i)
                do_output(o, result[o.offset()])

    def exec_node_op(self, n: Node, op: NodeOperator):
        #print(n)
        #print("invoking", n)
        inputs = self.collect_inputs(n)
        result = op(*inputs)
        self.add_outputs(result, n)
        return result

    def exec_node(self, n: Node) -> Any:
        found: Operation = self._find_operation(n)
        #print(f"found {n} {found}")
        return found.interpret(self, n)

    def exec_graph(self, g: Graph|Block, module: Module, args: Tuple, kwargs: OrderedDict[str, Any]):
        # Seed variables with inputs
        self.add_var("self", module)

        sig = inspect.signature(module.forward)
        params = list(sig.parameters.items())

        for v,(n,p) in zip(args,params):
            #print_vars()
            #print('doing name', n + ".1")
            self.add_var(n+'.1', v)

        for n,v in kwargs.items():
            # TODO: check the name is right
            self.add_var(n+".1",v)

        result = None
        for node in g.nodes():
            result = self.exec_node(node)

        #print("finished")
        #self.print_vars()
        return result

