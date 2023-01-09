from aten import int_to_dtype, dtype_to_int, int_to_memory_format, summarize_tensor

import torch
from typing import Type, Tuple, Any, Dict, List, Optional, Generator, Callable, OrderedDict, Union

from torch.nn import Module
from torch import Tensor, ScriptModule, ScriptFunction, Value, Size, Block, dtype, memory_format, device, scalar_tensor, add, tanh
import torch.jit as jit

from torch._C import Graph, Node, dtype as cdtype
import inspect
from ansi.color import bg, fg
from ansi.color.fx import reset

def exec_graph(g: Graph, self: Module, args: Tuple, kwargs: OrderedDict[str, Any]):
    kinds: Dict[str, int] = {}
    for node in g.nodes():
        print(node.kind(), node.f)
        kinds[node.kind()] = kinds.get(node.kind(), 0) + 1
    
    for k,v in sorted(kinds.items()):
        print(f"{k:60} {v}")

    print("graph is of type", type(g))

    print("calling with", len(args), " args and ", len(kwargs), " kwargs", [k for k in kwargs.keys()])

    # Seed variables with inputs
    vars: List[Tuple[str, Any]] = []
    var_names: Dict[str, int] = {}

    def add_var(name: str, val: Any):
        assert name not in var_names
        var_names[name] = len(vars)
        vars.append((name, val))


    def get_var(name: str) -> Any:
        return vars[var_names[name]][1]

    def print_var_index(i: int) -> str:
        n,v = vars[i]
        def print_type(v: Any) -> str:
            return str(type(v))

        def print_value(v: Any) -> str:
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
                return shorten('(' + ','.join([print_value(v2) for v2 in v]) + ')')

            else:
                result = str(v).replace('\n,','')
                return print_type(v) + " " + result[0:50]

        return f"{i:3} {n:30} {print_value(v)}"

    def print_var_named(n: str) -> str:
        return print_var_index(var_names[n])

    def print_vars():
        for i in range(len(vars)):
            print(print_var_index(i))

    add_var("self", self)

    sig = inspect.signature(self.forward)
    params = list(sig.parameters.items())

    for v,(n,p) in zip(args,params):
        #print_vars()
        print('doing name', n + ".1")
        add_var(n+'.1', v)

    for n,v in kwargs.items():
        # TODO: check the name is right
        add_var(n+".1",v)

    #for name,param in list(sig.parameters.items()):
    #    print("parameter", name, param, param.kind)

    print_vars()

    #for i in g.inputs():
    #    val = None
    #    if i.debugName() == "self":
    #        val = self
    #    else:
    #        val = kwargs[i]

    #    print("input", i.debugName(), i.type(), i.offset())
    #    num += 1
    
    def exec_node(n: Node) -> Any:

        print("\n\nexecuting node", n)
        #print("node is of type", type(n))

        print(n.inputsSize(),"inputs")
        def get_input(input: Value) -> Any:
            n = input.debugName()
            print("  ", print_var_named(n))
            return get_var(n)

        inputs = tuple((get_input(input) for input in n.inputs()))

        found: Optional[Callable] = None

        if n.kind() == "prim::Constant":
            val = n.output()
            found = lambda: val.toIValue()
        elif n.kind() == "prim::Uninitialized":
            found = lambda: None
        elif n.kind() == "prim::RaiseException":
            def raise_exception(msg: str, exctype: str):
                exctypes: Dict[str, type] = {
                    'builtins.RuntimeError': RuntimeError
                }
                raise exctypes[exctype](msg)
            found = raise_exception
        elif n.kind() == "prim::Print":
            def do_print(*args):
                print(fg.red,*args,reset,sep='')
            found = do_print
        elif n.kind() == "prim::dtype":
            found = lambda t: dtype_to_int(t.dtype)
        elif n.kind() == "prim::device":
            found = lambda t: t.device
        elif n.kind() == "prim::ListConstruct":
            def list_construct(*args: Any) -> List[Any]:
                return list([a for a in args])
            found = list_construct
        elif n.kind() == "prim::TupleConstruct":
            def tuple_construct(*args: Any) -> Tuple[Any, ...]:
                return tuple([a for a in args])
            found = tuple_construct
        elif n.kind() == "prim::TupleIndex":
            def tuple_index(t: Tuple[Any, ...], i: int) -> Any:
                return t[i]
            found = tuple_index
        elif n.kind() == "prim::GetAttr":
            a: str = n.s("name")
            def get_attr(o: object) -> Any:
                return getattr(o, a)
            found = get_attr
        elif n.kind() == "prim::Return":
            def do_return(*args) -> Any:
                ta = tuple((a for a in args))
                if len(ta) == 1:
                    return ta[0]
                else:
                    return ta
            found = do_return
        elif n.kind() == "prim::unchecked_cast":
            found = lambda x: x
        elif n.kind() == "prim::TupleUnpack":
            found = lambda x: x
        elif n.kind() == "prim::If":
            def exec_if(cond: bool):
                blocks = list(n.blocks())
                block: Block
                if cond:
                    block = blocks[0]
                else:
                    block = blocks[1]

                for block_node in block.nodes():
                    exec_node(block_node)

                return_node: Node = block.returnNode()

                #print("return node", return_node)

                res = exec_node(return_node)

                return res
                #print("res", res)

            found = exec_if
        elif n.kind() == "aten::format":
            def aten_format(fmt: str, *args) -> str:
                return fmt.format(*args)
            found = aten_format
        elif n.kind() == "aten::scalar_tensor":
            def aten_scalar_tensor(val, dtype=None, layout=None, device=None, pin_memory=None) -> Tensor:
                return scalar_tensor(val, dtype=int_to_dtype(dtype), layout=layout, device=device, pin_memory=pin_memory)
            found = aten_scalar_tensor
        elif n.kind() == "aten::size":
            def get_sizes(t: Any) -> List[int]:
                assert isinstance(t, Tensor)
                return list([int(s) for s in t.size()])
            def get_size(t: Any, dim: int) -> int:
                assert isinstance(t, Tensor)
                return t.size(dim)
            if len(inputs) == 1:
                found = get_sizes
            elif len(inputs) == 2:
                found = get_size
            else:
                raise RuntimeError(f"couldn't get aten::size for {len(inputs)} inputs")
        elif n.kind() == "aten::__getitem__":
            def get_item(l: Any, item: Any) -> Any:
                return l[item]
            found = get_item
        elif n.kind() == "aten::view":
            def aten_view(t: Tensor, v: List[int]):
                return t.view(v)
            found = aten_view
        elif n.kind() == "aten::contiguous":
            def aten_contiguous(t: Tensor, m: int):
                return t.contiguous(memory_format=int_to_memory_format(m))
            found = aten_contiguous
        elif n.kind() == "aten::slice":
            for arg in n.inputs():
                print("input", arg)
            def aten_slice_python(x: Any, start: Optional[int] = None, end: Optional[int] = None, step: int = 1):
                start = 0 if start is None else start
                end = len(x) if end is None else end
                step = 1 if step is None else step
                return x[start:end:step]

            def aten_slice_tensor(t: Tensor, dim: Optional[int] = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1):
                start = 0 if start is None else start
                end = t.size(dim) if end is None else end
                step = 1 if step is None else step
                dim = 0 if dim is None else dim

                print(f"{fg.blue}do_slice (dim {dim}): {start}:{end}:{step}{reset}")

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
                found = aten_slice_tensor
            else:
                found = aten_slice_python
        elif n.kind() == "aten::__isnot__":
            def is_not(v: Any, o: Any) -> bool:
                return v is not o
            found = is_not
        elif n.kind() == "aten::__is__":
            def aten_is(v: Any, o: Any) -> bool:
                return v is o
            found = aten_is
        elif n.kind() == "aten::eq":
            def aten_eq(v1: Any, v2: Any) -> Any:
                if isinstance(v1, Tensor):
                    return v1.eq(v2)
                else:
                    return v1 == v2
            found = aten_eq
        elif n.kind() == "aten::pow":
            def aten_pow(v1: Tensor, v2: Union[Tensor,float]) -> Tensor:
                return v1.pow(v2)
            found = aten_pow
        elif n.kind() == "aten::tanh":
            def aten_tanh(v1: Tensor) -> Tensor:
                return tanh(v1)
            found = aten_tanh
        elif n.kind() == "aten::to":
            def aten_to(v1: Tensor, a1: Any, *rest) -> Any:
                def option1(v1: Tensor, idt: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[memory_format] = None):
                    dt = int_to_dtype(idt)
                    print(f"{fg.green}trying to option 1 dt {dt}{reset}")
                    assert memory_format is None
                    return v1.to(dt, non_blocking, copy)
                def option2(v1: Tensor, dev: device, dt: dtype, non_blocking: bool = False, copy: bool = False, memory_format: Optional[memory_format] = None):
                    print(f"{fg.green}trying to option 2{reset}")
                    assert memory_format is None
                    return v1.to(dev, dt, non_blocking, copy)
                def option3(v1: Tensor, v2: Tensor, non_blocking: bool = False, copy: bool = False, memory_format: Optional[memory_format] = None):
                    print(f"{fg.green}trying to option 3{reset}")
                    assert memory_format is None
                    return v1.to(v2, non_blocking, copy)

                return option1(v1, a1, *rest)
            found = aten_to
        elif n.kind() == "aten::add":
            def aten_add_tensors(t1: Tensor, t2: Tensor, alpha = 1):
                return add(t1, t2, alpha=alpha)
            def aten_add(t1: Any, t2: Any):
                print(f"aten_add: {type(t1)} + {type(t2)}")
                return t1 + t2

            # Choose which one based on the type of the first argument
            input1 = n.inputsAt(0)
            input2 = n.inputsAt(1)
            print(input1.type(), input2.type())
            if n.inputsSize() == 3 or (input1.type().kind() == "TensorType" and input2.type().kind() == "TensorType"):
                found = aten_add_tensors
            else:
                found = aten_add
        else:
            import torch.jit._builtins
            ops = torch.jit._builtins._builtin_ops
            for method,name in ops:
                if name == n.kind():
                    found = method
                    break

        if found is None:
            raise RuntimeError(f"Couldn't find op for PyTorch op {n.kind()}")

        print("executing found node", found)

        try:
            print(inspect.signature(found))
        except:
            pass

        #for i in inputs:
        #    print(type(i), i)
        result = found(*inputs)
        #print("returned value", result)

        print(n.outputsSize(), "outputs")

        def do_output(o: Value, v: Any):
            add_var(o.debugName(), v)
            print("  ", print_var_named(o.debugName()))

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

        #print_vars()

        #print("executing node", n)
        #for input in n.inputs():
        #    print("input", input)
        #    input(inputs)

        return result

    for node in g.nodes():
        exec_node(node)

    print("finished")
    print_vars()

