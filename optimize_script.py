from torch.nn import Module
from torch import Tensor, ScriptModule, ScriptFunction, Value, Size, Block, dtype, memory_format, device, scalar_tensor, add, tanh
import torch.jit as jit
import torch.fx as fx
import torch
from typing import Type, Tuple, Any, Dict, List, Optional, Generator, Callable, OrderedDict, Union, TypeVar, Iterator, Iterable, Sequence
from dataclasses import dataclass, field
from introspect import (introspect_model, record_invocations, short_dtype,
                        SummaryData, Invocation, Invocations,
                        ArgumentData, Arg, TensorArg, TensorShape, TensorShapes, ShapeRange,
                        TupleArg, ListTupleArg, UnknownArg, ConstantArg, OptionalArg)
from graphs import Scope, default_find_operation, Operation, _print_value
from torch._C import Graph, Node, dtype as cdtype
from enum import Enum
import inspect
import copy
from ansi.color import bg, fg
from ansi.color.fx import reset


def _unify_types(summary: Arg, torch_type: 'torch._C.JitType') -> Tuple[Type, Optional[torch.dtype], Optional[torch.device], Optional[TensorShapes]]:

    print("torch_type", type(torch_type), torch_type, dir(torch_type), torch_type.annotation_str)
    print("summary", summary)

    if isinstance(torch_type, torch.TensorType):
        tp: Optional[Type] = summary.get_type()
        dtype: Optional[torch.dtype] = summary.get_dtype()
        device: Optional[torch.device] = summary.get_device()
        shape: Optional[TensorShapes] = summary.get_shape()
        return tp,dtype,device,shape
    elif isinstance(torch_type, torch.OptionalType):
        contained = torch_type.getElementType()

        print("contained", torch_type.getElementType())
        tp,dtype,device,shape = _unify_types(summary.non_optional(), contained)
        if summary.is_optional():
            pass
        else:
            raise RuntimeError("TODO: Optional")
    elif isinstance(torch_type, torch.TupleType):
        contained = torch_type.containedTypes()
        print("contained", contained)
        print("contained dir", dir(contained))
        if isinstance(summary, ListTupleArg):
            # Unify tuple with each argument
            for i in range(len(contained)):
                st = contained[i]
                tp,dtype,device,shape = _unify_types(summary.value, st)
                # TODO: return something meaningful here...
            return tuple,None,None,None
        elif isinstance(summary, TupleArg):
            # Unify each element of the tuple
            raise RuntimeError("TODO: TupleArg")
            pass
        else:
            raise RuntimeError(f"Torch is tuple but argument is {summary}")
    else:
        raise RuntimeError(f"Unknown torch_type {torch_type}")




    pass

@dataclass
class ModuleOptimizationInfo:
    invocations: List[Invocations] = field(default_factory=list, repr=False)
    summary: SummaryData = field(default_factory=SummaryData)

    def total_runtime(self) -> float:
        return sum([inv.total_runtime() for inv in self.invocations], 0.0)

    def add(self, i: Invocations):
        self.invocations.append(i)
        self.summary.add(i.summarize())

@dataclass
class OptimizeModelData:
    modules: Dict[str, Module] = field(default_factory=dict)
    optinfo: Dict[str, ModuleOptimizationInfo] = field(default_factory=dict)

class Origin(Enum):
    UNKNOWN = 0
    SELF = 1
    ARG = 2
    DEFAULT_ARG = 3
    LOCAL = 4
    CONST_PROP = 5

class Constness(Enum):
    UNKNOWN = 0
    VAR = 1
    CONST_TYPE = 2

@dataclass
class VariableInfo:
    """
    Holds the static information known about a variable.
    """
    name: str = ""
    origin: Origin = Origin.UNKNOWN

    is_const: bool = False
    is_optional: bool = False
    const_type: Optional[type] = None

    # The following are for Tensors.  We model the data type, device and shape separately.
    const_dtype: Optional[torch.dtype] = None
    const_device: Optional[torch.device] = None
    tensor_shape: Optional[TensorShapes] = None

    # The following are for sequences (lists and tuples).  These are modelled
    # with:
    # 1. seq_length, which describes the length of the sequence (it can be a range, or fixed)
    # 2. seq_els, which describes the VariableInfo for the first elements
    # 3. seq_el, which descrives the VariableInfo for any which aren't covered by seq_els
    seq_length: Optional[ShapeRange] = None
    seq_els: Optional[List['VariableInfo']] = None
    seq_el: Optional['VariableInfo'] = None
    const_value: Optional[Any] = None

    # Which node produced, and which nodes read, this value.  The integers are the sequence
    # of nodes in the graph.
    produced_by: Optional[Node] = None
    produced: int = 0
    first_consumed: int = 10000
    last_consumed: int = 0

    def renamed(self, new_name: str) -> 'VariableInfo':
        res = copy.copy(self)
        res.name = new_name
        return res

    def is_homogeneous(self, other: 'VariableInfo'):
        """
        Are these homogeneous, meaning that they can be combined into a single
        VariableInfo that covers both of them.
        """
        if self.const_type != other.const_type:
            return False
        if self.is_const and other.is_const:
            return self.const_value == other.const_value
        if self.const_type == torch.Tensor:
            raise RuntimeError("is_homogeneous for tensor")
        elif self.const_type == tuple or self.const_type == list:
            raise RuntimeError("is_homogeneous for sequence")

        return True

    def combine(self, other: 'VariableInfo'):
        """
        Combine the two VariableInfos together to create one that covers both.
        """

        if other.is_optional:
            self.is_optional = True

        if other.const_type != self.const_type:
            raise RuntimeError("TODO: combine with two different types")

        if self.is_const and not other.is_const:
            self.const_value = None
            self.is_const = False

        if self.const_type == torch.Tensor:
            raise RuntimeError("TODO: combine tensors")
        elif self.const_type == tuple or self.const_type == list:
            raise RuntimeError("TODO: combine sequences")

        self.produced = min(self.produced, other.produced)
        self.first_consumed = min(self.first_consumed, other.first_consumed)
        self.last_consumed = max(self.last_consumed, other.last_consumed)

    @staticmethod
    def constant(*, name: str, origin: Origin, value: Any, produced_by: Node, produced: int) -> 'VariableInfo':
        """
        Return the variable info for a constant.  This will fill in all of the ancilliary
        information.
        """
        dtype: Optional[torch.dtype] = None
        device: Optional[torch.device] = None
        shape: Optional[TensorShapes] = None

        if isinstance(value, torch.Tensor):
            dtype = value.dtype
            device = value.device
            shape = TensorShapes.from_tensor(value)

        return VariableInfo(name=name, origin=origin, is_const=True,
                            const_type=type(value), const_value=value,
                            const_dtype=dtype, const_device=device, tensor_shape=shape,
                            produced_by=produced_by, produced=produced)

    @staticmethod
    def argument(*, name: str, produced_by: Node, observed: ArgumentData, torch_type: 'torch._C.JitType') -> 'VariableInfo':
        """
        Return the variable info for an argument.  This will specialize based on the observed
        values.
        """

        summary = observed.summarize()

        dtype: Optional[torch.dtype] = summary.get_dtype()
        device: Optional[torch.device] = summary.get_device()
        shape: Optional[TensorShapes] = summary.get_shape()
        tp: Optional[Type] = summary.get_type()

        tp,dtype,device,shape = _unify_types(summary, torch_type)

        result = VariableInfo(name=name, origin=Origin.ARG, is_const=False,
                              const_type=tp,
                              const_dtype=dtype, const_device=device, tensor_shape=shape,
                              produced_by=produced_by, produced=-1)

        return result

    @staticmethod
    def local(*, name: str, origin: Origin, tp: Type, produced_by: Node, produced: int) -> 'VariableInfo':
        """
        Return the variable info for a local variable.  This should not be used for
        tensors.
        """
        assert not issubclass(tp, torch.Tensor), "Tensors should use the tensor method, not local"

        return VariableInfo(name=name, origin=origin, is_const=False, const_type=tp,
                            produced_by=produced_by, produced=produced)

    @staticmethod
    def tensor(*, name: str, origin: Origin,
               dtype: Optional[torch.dtype], device: Optional[torch.device], shape: Optional[List[Optional[int]]],
               produced_by: Node, produced: int) -> 'VariableInfo':
        """
        Return the variable info for a tensor valued variable.
        """
        return VariableInfo(name=name, origin=origin, is_const=False, const_type=torch.Tensor,
                            const_dtype=dtype, const_device=device, tensor_shape=shape,
                            produced_by=produced_by, produced=produced)

    @staticmethod
    def any(*, name: str, origin: Origin, produced_by: Node, produced: int) -> 'VariableInfo':
        """
        Return the variable info for something that could be any type (nothing static is known
        about it).
        """
        return VariableInfo(name=name, origin=origin, is_const=False,
                            produced_by=produced_by, produced=produced)

    @staticmethod
    def homogeneous_sequence(*, name: str, origin: Origin, tp: Type[List|Tuple], produced_by: Node, produced: int,
                             length: ShapeRange, values: 'VariableInfo') -> 'VariableInfo':
        """
        Create the VariableInfo for a homogeneous sequence (tuple or list) with a fixed or
        variable length content of an instance of a single type.
        """
        return VariableInfo(name=name, origin=origin, is_const=False, const_type=tp, seq_length=length, seq_el=values,
                            produced_by=produced_by, produced=produced)

    @staticmethod
    def inhomogeneous_sequence(*, name: str, origin: Origin, tp: Type[List|Tuple], produced_by: Node, produced: int,
                               values: List['VariableInfo']) -> 'VariableInfo':
        """
        Create the VariableInfo for an inhomogeneous sequence (tuple or list) with a fixed
        length and each element having a different type.
        """
        print("inhomogeneous sequence")
        for i,v in enumerate(values):
            print(_print_var(str(i), v))
        seq_length = ShapeRange(len(values))
        print("seq_length", seq_length)
        return VariableInfo(name=name, origin=origin, is_const=False, const_type=tp, seq_length=seq_length, seq_els=values,
                            produced_by=produced_by, produced=produced)

def _print_var_fields(name: str, origin: Any, const: Any, produced: Any, first_consumed: Any, last_consumed: Any, produced_by: Any, tp: Any, const_value: Any) -> str:
    return f"{name:20} {origin:10} {const:6} {produced:5} {first_consumed:5} {last_consumed:5} {produced_by:12} {tp:12} {const_value}"

def _print_var_names() -> str:
    return _print_var_fields("name", "origin", "const", "prod", "first", "last", "node", "type", "value")

def _short_info_str(info: VariableInfo) -> str:
    if info.is_const:
        return _print_value(info.const_value)
    else:
        assert info.const_type is not None
        return "<" + info.const_type.__name__ + ">"

def _print_var(name: str, info: VariableInfo) -> str:
    produced_kind = ''
    if info.produced_by is not None:
        produced_kind = info.produced_by.kind().replace("aten::", "").replace("prim::", "")
    type_str = ''
    if info.const_type is not None:
        type_str = info.const_type.__name__

    value_str: str = ""
    if info.is_const:
        value_str = _print_value(info.const_value)
    elif info.const_type == torch.Tensor:
        if info.const_dtype is None:
            value_str = "<dtype?>"
        else:
            value_str = short_dtype(info.const_dtype)

        if info.tensor_shape is None:
            value_str += "<shape?>"
        else:
            value_str += str(info.tensor_shape)

        if info.const_device is None:
            value_str += "<device?>"
        else:
            value_str += str(info.const_device)
    elif info.const_type == list or info.const_type == tuple:
        open = "[" if info.const_type == list else '('
        close = "]" if info.const_type == list else ')'
        if info.seq_els is not None:
            el_strs: List[str] = []
            for el in info.seq_els:
                el_strs.append(_short_info_str(el))
            value_str = open + ", ".join(el_strs) + close
        elif info.seq_length is not None:
            assert info.seq_el is not None
            if info.const_type == list:
                value_str = _short_info_str(info.seq_el) + str(info.seq_length)
            else:
                value_str = "(" + ",".join([_short_info_str(info.seq_el)] * info.seq_length.min)
                if info.seq_length.max > info.seq_length.min:
                    value_str += "..." + ")" + str(info.seq_length)
                else:
                    value_str += ")"
        else:
            print("unhandled list/sequence: info", info)
            raise RuntimeError("TODO: print unhandled list/sequence case")

    else:
        pass

    return _print_var_fields(name, info.origin.name, info.is_const, info.produced, info.first_consumed,
                             info.last_consumed, produced_kind, type_str, value_str)

@dataclass
class Variables:
    vars: OrderedDict[str, VariableInfo] = field(default_factory=OrderedDict)

    def __len__(self) -> int:
        return len(self.vars)

    def add(self, v: VariableInfo):
        assert v.produced_by is None or isinstance(v.produced_by, Node)
        assert len(v.name) > 0
        assert v.name not in self.vars
        self.vars[v.name] = v

    def add_constant(self, n: str, v: Any, node: Node, i: int):
        assert isinstance(node, Node)
        info = VariableInfo(n, Origin.CONST_PROP, True, v, node, i)
        self.add(info)

    def get(self, n: str, i: int) -> VariableInfo:
        result = self.vars[n]
        if i < result.first_consumed:
            result.first_consumed = i
        if i > result.last_consumed:
            result.last_consumed = i
        return result

    def dump_vars(self, indent: str = '', start_at: int = 0):
        print(f"{indent} {fg.boldblack}{_print_var_names()}{reset}")
        for i,(name,info) in enumerate(self.vars.items()):
            if i < start_at:
                continue
            print(indent, _print_var(name, info))

VT = TypeVar("VT")
def first(x: Iterable[VT]) -> Optional[VT]:
    for v in x:
        return v
    return None

class TorchOperator:

    def is_block(self) -> bool: ...

    # Is this a constant operation?  In other words, will it always return the same value
    # given the same inputs, and doesn't have any side-effects?
    def is_const(self) -> bool: ...

    # Perform constant propagation for the operation
    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo: ...



class TorchBlockOperator(TorchOperator):
    """
    Base class for operators which deal with blocks (principally Prim::If)
    """
    node: Node
    blocks: List[Block]

    def __init__(self, node: Node):
        self.node = node
        self.blocks = list(node.blocks())
        assert len(self.blocks) > 0

    def is_block(self) -> bool:
        return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert len(inputs) == 1, "block selection happens with one input only"
        input = inputs[0]
        num_outputs = self.node.outputsSize()

        results: List[VariableInfo] = []
        
        def do_result(results: List[VariableInfo], this_result: List[VariableInfo]):
            print("do_result", this_result)
            print("results", results)
            if len(results) == 0:
                results.extend(this_result)
            else:
                assert len(results) == len(this_result)
                for res,this_res in zip(results, this_result):
                    res.union(this_res)

        def process_block(block: Block) -> List[VariableInfo]:
            new_vars = copy.copy(vars)
            const_prop_graph(block, new_vars)
            return_node = block.returnNode()
            assert return_node.kind() == "prim::Return"
            assert return_node.inputsSize() == num_outputs
            return [new_vars.get(input.debugName(), i).renamed(output.debugName()) for input,output in zip(return_node.inputs(),self.node.outputs())]

        if not input.is_const:
            # It's not a constant.  Perform constant propagation for each block and take the
            # union of the outputs.

            for block in self.blocks:
                do_result(results, process_block(block))

        else:
            # It's a constant.  We do only one block.
            block_num = 0 if input.const_value else 1
            do_result(results, process_block(self.blocks[block_num]))

        if num_outputs == 0:
            return tuple()
        elif num_outputs == 1:
            assert results[0] is not None
            return results[0]
        else:
            return tuple(results)

class TorchFunctionOperator(TorchOperator):
    """
    Base class for operators which execute a function
    """

    node: Node

    def __init__(self, node: Node):
        self.node = node

    def is_block(self) -> bool:
        return False

    def collect_const_inputs(self, inputs: List[VariableInfo]) -> Optional[Scope]:
        """
        Collect all of the constant values for the inputs.  Returns None if
        they aren't all constant.
        """

        assert len(inputs) == self.node.inputsSize()
        if not self.is_const():
            return None

        scope = Scope()

        for var,input in zip(inputs, self.node.inputs()):
            if var.is_const:
                scope.add_var(input.debugName(), var.const_value)
            else:
                return None

        return scope        

_torch_operators: Dict[str, Type[TorchOperator]] = {}

def get_torch_operator(node: Node) -> TorchOperator:
    return _torch_operators[node.kind()](node)

def torch_operator(kind: str):
    def do_operator(klass):
        assert kind not in _torch_operators
        _torch_operators[kind] = klass
        return klass
    return do_operator


@torch_operator('aten::Param')
class ParamOperator(TorchOperator):
    """
    Operator for a simple parameter that is passed in.  Single variant.
    """

    node: Node

    def __init__(self, node: Node):
        self.node = node

    def is_block(self) -> bool:
        return False

    def is_const(self) -> bool:
        raise RuntimeError("is_const")

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        raise RuntimeError("paramoperator")


@torch_operator('prim::Constant')
class ConstantOperator(TorchOperator):
    """
    Operator for a simple parameter that is passed in.  Single variant.
    """

    node: Node

    def __init__(self, node: Node):
        assert node.inputsSize() == 0
        assert node.outputsSize() == 1
        self.node = node

    def is_block(self) -> bool:
        return False

    def is_const(self) -> bool:
        return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        This is a constant, so it produces a constant that can be propagated.  Single output with the
        provided value.
        """
        assert len(inputs) == 0
        output = self.node.outputsAt(0)
        result = VariableInfo.constant(name=output.debugName(), origin=Origin.CONST_PROP,
                                       value=output.toIValue(), produced_by=self.node, produced=i)
        return result

@torch_operator('prim::GetAttr')
class GetAttrOperator(TorchFunctionOperator):
    """
    Operator for a simple parameter that is passed in.  Single variant.
    """

    def __init__(self, node: Node):
        assert node.inputsSize() == 1
        assert node.outputsSize() == 1
        super().__init__(node)
        self.node = node

    def is_block(self) -> bool:
        return False

    def is_const(self) -> bool:
        return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert len(inputs) == 1
        attr_name = self.node.s("name")

        #Const prop for getattr depends on constness of the argument; the attribute name is always
        #a constant.
        if inputs[0].is_const:
            val = getattr(inputs[0].const_value, attr_name)
            result = VariableInfo.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP,
                                           value=val, produced_by=self.node, produced=i)
        else:
            # Potentially we could look up the type and see if there are any type annotations on the
            # field we're retrieving.
            # (TODO)x
            # For now, return an any
            result = VariableInfo.any(name=self.node.outputsAt(0).debugName(), origin=Origin.LOCAL,
                                        produced_by=self.node, produced=i)

        return result

@torch_operator('aten::__is__')
class AtenIsOperator(TorchFunctionOperator):
    """
    Base class for operators that only depend on comparing two types, such as x is y or x is not y
    """

    invert: bool  # Do we invert the polarity (changes is to is not)

    def __init__(self, node: Node, invert: bool = False):
        super().__init__(node)
        self.node = node
        self.invert = invert

    def is_block(self) -> bool:
        return False

    def is_const(self) -> bool:
        return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert len(inputs) == 2
        assert self.node.outputsSize() == 1

        val: Optional[bool] = None
        # Otherwise if values are known...
        if inputs[0].is_const and inputs[1].is_const:
            val = inputs[0].const_value is inputs[1].const_value
            print(f"const is: {inputs[0].const_value} is {inputs[1].const_value} = {val}")
        elif inputs[1].const_type == type(None) and inputs[1].const_type == type(None):
            # x is None can be short circuited based on types only
            val = True
            print(f"const typed is: {inputs[0].const_type} is {inputs[1].const_type} = {val}")

        if val is not None:
            print("invert is", self.invert, "val is", val)
            if self.invert:
                val = not val
            return VariableInfo.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP,
                                         value=val, produced_by=self.node, produced=i)

        # Otherwise, it's a boolean valued unknown
        return VariableInfo.local(name=self.node.outputsAt(0).debugName(), origin=Origin.LOCAL, tp=bool,
                                  produced_by=self.node, produced=i)

@torch_operator('aten::__isnot__')
class AtenIsNotOperator(AtenIsOperator):
    def __init__(self, node: Node):
        super().__init__(node, True)


@torch_operator('prim::If')
class IfOperator(TorchBlockOperator):
    """
    Block-based operator that handles the "if" statement.
    """
    pass

@torch_operator('prim::ListConstruct')
class ListConstructOperator(TorchFunctionOperator):
    """
    Block-based operator that handles the "if" statement.
    """

    def is_const(self) -> bool: return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert self.node.outputsSize() == 1
        output_name = self.node.outputsAt(0).debugName()

        collected = self.collect_const_inputs(inputs)
        if collected is not None:
            # All inputs are constant, so we return a constant list
            result = [value for name,value in collected.vars]
            return VariableInfo.constant(name=output_name, origin=Origin.CONST_PROP, value=result,
                                         produced_by=self.node, produced=i)

        else:
            # Propagate what we know about the list
            common_info: VariableInfo = copy.copy(inputs[0])
            has_common_info = True
            length = ShapeRange(len(inputs))

            for j in range(1, len(inputs)):
                other = inputs[j]    
                if common_info.is_homogeneous(other):
                    common_info.combine(other)
                else:
                    has_common_info = False
                    break

            if has_common_info:
                return VariableInfo.homogeneous_sequence(name=output_name, origin=Origin.CONST_PROP, tp=list, length=length,
                                                         values=common_info, produced_by=self.node, produced=i)
            else:
                return VariableInfo.inhomogeneous_sequence(name=output_name, origin=Origin.CONST_PROP, tp=list,
                                                           values=inputs, produced_by=self.node, produced=i)

@torch_operator('aten::size')
class AtenSizeOperator(TorchFunctionOperator):
    """
    Handles the "size" operator.  Often has a const output even from a non-const input.
    """

    def is_const(self) -> bool: return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert self.node.outputsSize() == 1
        assert self.node.inputsSize() == 1
        output_name = self.node.outputsAt(0).debugName()
        input = inputs[0]

        if input.const_type == torch.Tensor:
            sizes = inputs[0].tensor_shape
            assert sizes is not None
            if len(sizes.lengths) != 1:
                raise RuntimeError("TODO: multiple tensor sizes")
            assert len(sizes.lengths) == 1


            for length,shape in sizes.lengths.items():
                # This loop called exactly once until we handle multiple tensor sizes

                values: List[int] = []
                all_const_values: bool = True
                infos: List[VariableInfo] = []

                for j,dim in enumerate(shape):
                    dim_name=output_name + f".{str(j)}"
                    if dim.is_const():
                        values.append(dim.const_value())
                        infos.append(VariableInfo.constant(name=dim_name, origin=Origin.CONST_PROP,
                                                           value=dim.const_value(), produced_by=self.node, produced=i))
                    else:
                        all_const_values = False
                        values.append(-1)
                        infos.append(VariableInfo.local(name=dim_name, origin=Origin.CONST_PROP, tp=int,
                                                        produced_by=self.node, produced=i))

                if all_const_values:
                    return VariableInfo.constant(name=output_name, origin=Origin.CONST_PROP, value=values,
                                                 produced_by=self.node, produced=i)
                else:
                    print("values", values)
                    print("infos", infos)
                    return VariableInfo.inhomogeneous_sequence(name=output_name, origin=Origin.CONST_PROP, tp=list,
                                                               values=infos, produced_by=self.node, produced=i)

        else:
            print("input", input)
            raise RuntimeError("TODO: non-tensor aten::size")


        collected = self.collect_const_inputs(inputs)
        if collected is not None:
            # All inputs are constant, so we return a constant list
            result = [value for name,value in collected.vars]
            return VariableInfo.constant(name=output_name, origin=Origin.CONST_PROP, value=result,
                                         produced_by=self.node, produced=i)

        else:
            # Propagate what we know about the list
            common_info: VariableInfo = copy.copy(inputs[0])
            has_common_info = True
            length = ShapeRange(len(inputs))

            for j in range(1, len(inputs)):
                other = inputs[j]    
                if common_info.is_homogeneous(other):
                    common_info.combine(other)
                else:
                    has_common_info = False
                    break

            if has_common_info:
                return VariableInfo.homogeneous_sequence(name=output_name, origin=Origin.CONST_PROP, tp=list, length=length,
                                                         values=common_info, produced_by=self.node, produced=i)
            else:
                return VariableInfo.inhomogeneous_sequence(name=output_name, origin=Origin.CONST_PROP, tp=list,
                                                           values=inputs, produced_by=self.node, produced=i)

@torch_operator('aten::__getitem__')
class AtenGetItemOperator(TorchFunctionOperator):
    """
    Handles the "getitem" operator.  Often used to manipulate sizes, so we make special
    efforts to ensure that const propagation happens properly.
    """

    def is_const(self) -> bool: return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert self.node.outputsSize() == 1
        assert self.node.inputsSize() == 2
        output_name = self.node.outputsAt(0).debugName()
        input = inputs[0]
        item = inputs[1]

        def get_from_seq(item_info: VariableInfo) -> VariableInfo:
            "Return a variable info from a known VariableInfo which may contain a constant"
            if item_info.is_const:
                # This value is constant in the sequence even if others aren't
                return VariableInfo.constant(name=output_name, origin=Origin.CONST_PROP, value=item_info.const_value,
                                                produced_by=self.node, produced=i)
            else:
                return item_info.renamed(output_name)

        if item.is_const:
            n = item.const_value
            assert n is not None

            if input.is_const:
                # Simple const value propagation
                assert input.const_value is not None
                value = input.const_value[item.const_value]
                return VariableInfo.constant(name=output_name, origin=Origin.CONST_PROP, value=value, produced_by=self.node,
                                             produced=i)
            elif input.seq_els is not None and len(input.seq_els) > n:
                # Const value propagation through an inhomogeneous sequence
                item_info = input.seq_els[n]
                return get_from_seq(item_info)
            elif input.seq_length is not None and input.seq_el is not None:
                # Const value propagation through an homogeneous sequence
                item_info = input.seq_el
                return get_from_seq(item_info)
            else:
                raise RuntimeError("TODO: const item non-const sequence unhandled case")

        # Non-const item
        if input.seq_els is not None:
            # Const value propagation through an inhomogeneous sequence
            raise RuntimeError("TODO: non-const item from inhomogeneous sequence")
        elif input.seq_length is not None and input.seq_el is not None:
            # Const value propagation through an homogeneous sequence
            item_info = input.seq_el
            return get_from_seq(item_info)
        else:
            raise RuntimeError("TODO: non-const item unhandled case")

NodeOperator = Callable[..., Tuple[Any,...]|Any]
_ops: Dict[str, List[Operation]] = {}


class TorchTensorOperator(TorchFunctionOperator):
    """
    Operator that handles a Torch tensor operator
    """

    def is_const(self) -> bool:
        return True

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        """
        Child classes override this method to implement shape calculations.
        """
        return None

    def specialize_dtype(self, output: Value, input_dtype: List[Optional[torch.dtype]]) -> Optional[torch.dtype]:
        """
        Child classes override this method to implement dtype calculations.
        """
        result = None
        for dtype in input_dtype:
            if dtype is None:
                continue
            if result is None:
                result = dtype
            elif result != dtype:
                raise RuntimeError(f"Inconsistent dtypes; need to override {self.__class__.__name__}.specialize_dtype")
        if result is None:
            raise RuntimeError(f"No tensors to infer dtype; need to override {self.__class__.__name__}.specialize_dtype")
        return result

    def specialize_device(self, output: Value, input_device: List[Optional[torch.device]]) -> Optional[torch.device]:
        """
        Child classes override this method to implement device calculations.
        """
        result = None
        for device in input_device:
            if device is None:
                continue
            if result is None:
                result = device
            elif result != device:
                raise RuntimeError(f"Inconsistent devices {result} and {device}; need to override {self.__class__.__name__}.specialize_device")
        if result is None:
            raise RuntimeError(f"No tensors to infer device; need to override {self.__class__.__name__}.specialize_device")
        return result

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[TensorShapes]]) -> Optional[TensorShapes]:
        def get_shape(shapes: Optional[TensorShapes]) -> Optional[TensorShape]:
            if shapes is None:
                return None
            if len(shapes.lengths) > 1:
                raise RuntimeError("TODO: more than one shape")
            for length,shape in shapes.lengths.items():
                return shape

        input_shape = [get_shape(input) for input in input_shapes]

        specialized = self.specialize_shape(output, inputs, input_shape)
        if specialized is None:
            return None
        return TensorShapes({len(specialized): specialized})

    def specialize_tensor(self, output: Value, inputs: List[VariableInfo]) -> Tuple[Optional[torch.dtype], Optional[TensorShapes], Optional[torch.device]]:
        """
        Return the tensor size and attributes of a returned tensor.  Needs to be specialized per operation as
        there is no way to generically know.
        """
        def get_dtype(input: VariableInfo) -> Optional[torch.dtype]:
            if input.const_dtype:
                return input.const_dtype
            return None

        dtype: Optional[torch.dtype] = self.specialize_dtype(output, [get_dtype(input) for input in inputs])

        def get_device(input: VariableInfo) -> Optional[torch.device]:
            if input.const_device:
                return input.const_device
            return None

        device: Optional[torch.device] = self.specialize_device(output, [get_device(input) for input in inputs])

        def get_shape(input: VariableInfo) -> Optional[TensorShapes]:
            if input.tensor_shape:
                return input.tensor_shape
            return None

        input_shapes = [get_shape(input) for input in inputs]

        shapes: Optional[TensorShapes] = self.specialize_shapes(output, inputs, input_shapes)

        return (dtype, shapes, device)

    def fallback_const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        Const prop for when inputs are not all known.  We do out best from the prototype
        and annotations.
        """

        def do_output(output: Value) -> VariableInfo:
            jit_type = output.type()
            print("jit_type", jit_type, jit_type.kind())
            kinds: Dict[str, type] = {
                "TensorType": torch.Tensor,
            }
            tp: type = kinds[jit_type.kind()]

            if jit_type.kind() == "TensorType":
                dtype,shape,device = self.specialize_tensor(output, inputs)
                info = VariableInfo.tensor(name=output.debugName(), origin=Origin.LOCAL, produced_by=self.node, produced=i,
                                           dtype=dtype, shape=shape, device=device)
            else:
                info = VariableInfo.local(name=output.debugName(), tp=tp, origin=Origin.LOCAL, produced_by=self.node, produced=i)
            return info

        if self.node.outputsSize() == 0:
            return tuple()
        elif self.node.outputsSize() == 1:
            return do_output(self.node.outputsAt(0))
        else:
            return tuple(do_output(output) for output in self.node.outputs())

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        Const prop for a torch operator is trivial if all inputs are constant: we can simply run
        the function.
        """
        assert len(inputs) == self.node.inputsSize()

        scope: Optional[Scope] = self.collect_const_inputs(inputs)
        result = None

        if scope is not None:
            # We can execute this node
            # Create the scope in which to execute it
            result = scope.exec_node(self.node)

            if self.node.outputsSize() == 0:
                assert result is None or len(result) == 0
            elif self.node.outputsSize() == 1:
                info = self.node.outputsAt(0)
                result = VariableInfo.constant(name=info.debugName(), origin=Origin.CONST_PROP, value=result,
                                               produced_by=self.node, produced=i)
            else:
                assert len(result) == self.node.outputsSize
                assert isinstance(result, tuple)
                result = tuple(VariableInfo.constant(name=info.debugName(), origin=Origin.CONST_PROP, value=val,
                                                produced_by=self.node, produced=i) for val,info in zip(result,self.node.outputs()))
        else:
            result = self.fallback_const_prop(i, inputs, vars)

        return result

class TorchBuiltinOperator(TorchTensorOperator):
    """
    Operator that handles a Torch builtin (which may have multiple implementations).
    """

    ops: List[Operation]

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation] = None):
        super().__init__(node)
        if ops is None:
            self.ops = _ops[node.kind()]
        elif isinstance(ops, list):
            self.ops = ops
        else:
            self.ops = [ops]

@torch_operator('aten::layer_norm')
class LayerNormOperator(TorchBuiltinOperator):
    """
    Layer norm operator.  Normalizes the statistics of layer activations to ensure good
    numerical properaties.
    """
    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        return input_shape[0]

@torch_operator('aten::linear')
class LinearOperator(TorchBuiltinOperator):
    """
    Linear ax + b operator.
    """
    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        i_shape,w_shape,b_shape = input_shape
        assert i_shape is not None
        assert w_shape is not None
        b,m,n1=i_shape
        n2,o=w_shape
        assert n1==n2
        return TensorShape([b,m,o])

@torch_operator('aten::view')
class AtenViewOperator(TorchBuiltinOperator):
    """
    Handles the "view" operator.  This manipulates the sizes and strides without touching
    the data.
    """

    #@staticmethod
    #def do_tensor_view(input: Tensor, arg: Sequence[int]|torch.dtype) -> Tensor:
    #    return input.view(arg)

    def __init__(self, node: Node):
        super().__init__(node, Tensor.view)

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[TensorShapes]]) -> Optional[TensorShapes]:
        print("specialize_shapes")
        print("inputs", inputs)
        print("input_shapes", input_shapes)
        raise RuntimeError("TODO")

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        assert len(inputs) == 2
        if inputs[1].is_const:
            # Constant shape, we can return it
            val = inputs[1].const_value
            assert isinstance(val, list)
            return TensorShape(val)
        else:
            # We return what we know
            ndims = len(input_shape)
            shape: List[TensorShape] = []
            ish = input_shape[1]
            return ish

class TorchComparisonOperator(TorchBuiltinOperator):
    """
    Base class for Torch comparison operators.
    """
    pass

@torch_operator('aten::lt')
class LessThanOperator(TorchComparisonOperator):
    pass

@torch_operator('aten::le')
class LessEqualOperator(TorchBuiltinOperator):
    pass

@torch_operator('aten::eq')
class EqualOperator(TorchBuiltinOperator):
    pass

@torch_operator('aten::ne')
class NotEqualOperator(TorchBuiltinOperator):
    pass

@torch_operator('aten::gt')
class GreaterThanOperator(TorchBuiltinOperator):
    pass

@torch_operator('aten::ge')
class GreaterEqualOperator(TorchBuiltinOperator):
    pass

@torch_operator('aten::embedding')
class EmbeddingOperator(TorchBuiltinOperator):
    def specialize_dtype(self, output: Value, input_dtype: List[Optional[torch.dtype]]) -> Optional[torch.dtype]:
        weights_dtype,_,_,_,_ = input_dtype
        return weights_dtype

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        weights_shape,indexes_shape,_,_,_ = input_shape

        assert weights_shape is not None
        assert indexes_shape is not None

        _,w_dim = weights_shape
        i_batch,i_len = indexes_shape

        return TensorShape([i_batch, i_len, w_dim])

@torch_operator('aten::dropout')
class DropoutOperator(TorchBuiltinOperator):
    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        i_shape,_,_ = input_shape
        return i_shape

def _add_builtin_ops():
    import torch.jit._builtins
    #import torch.jit.supported_ops
    #print(torch.jit.supported_ops._get_torchscript_builtins())
    #print(torch.jit.supported_ops._get_tensor_ops())
    ops = torch.jit._builtins._builtin_ops
    for op,name in ops:
        if name in _ops:
            _ops[name].append(op)
        else:
            _ops[name] = [op]

    #for name in _ops.keys():
    #    _torch_operators[name] = TorchBuiltinOperator

_add_builtin_ops()


def const_prop_graph(graph: Graph|Block, vars: Variables):

    start_at = [0]
    def do_node(i: int, node: Node, indent: str):  # -> Tuple[Optional[Any|Tuple[Any]], bool]:
        try:
            print(indent, "executing node", i, node)

            inputs: List[VariableInfo] = []
            for input in node.inputs():
                name = input.debugName()
                var = vars.get(name, i)
                inputs.append(var)

            op = get_torch_operator(node)

            print("got op", op)

            result = op.const_prop(i, inputs, vars)

            def add_output(info: Value, var: VariableInfo):
                is_constant_output = op.is_const and var.is_const
                var.name = info.debugName()
                if is_constant_output:
                    var.origin = Origin.CONST_PROP
                else:
                    var.is_const = False
                    var.origin = Origin.LOCAL
                    var.const_value = None
                vars.add(var)

            if node.outputsSize() == 0:
                assert node.outputsSize() == 0
                pass
            elif node.outputsSize() == 1:
                assert isinstance(result, VariableInfo)
                assert node.outputsSize() == 1
                info = node.outputsAt(0)
                add_output(info, result)
            else:
                assert isinstance(result, tuple)
                assert node.outputsSize() == len(result)
                for val,info in zip(result, node.outputs()):
                    add_output(info, val)

            vars.dump_vars(indent, start_at[0])
            start_at[0] = len(vars)
        except:
            print("exception executing node", node)
            vars.dump_vars(indent)
            raise


    for i,node in enumerate(graph.nodes()):
        do_node(i, node, '')

    return vars

def optimize_script(script: ScriptModule, invocations: Invocations, info: ModuleOptimizationInfo) -> Optional[ScriptModule]:
    optimized = script #jit.optimize_for_inference(script)
    signature = inspect.signature(invocations.m.forward)

    #print('optimizing script', script)
    try:
        graph: Graph = optimized.inlined_graph
    except:
        print("can't be optimized as it has no inlined_graph")
        return None
    print("doing graph", graph)

    graph_inputs = list(graph.inputs())
    for i,input in enumerate(graph_inputs):
        print("  input",i,input)
    invocations.summarize().print_args()
    info.summary.print_args()

    vars = Variables()

    scope = Scope(default_find_operation)

    self_info = VariableInfo("self", Origin.SELF, is_const=True, const_value=invocations.m, const_type=type(invocations.m), produced=-1)
    vars.add(self_info)
    scope.add_var("self", invocations.m)

    summary: SummaryData = invocations.summarize()

    def add_input(name: str, graph_input: Value, arg: ArgumentData, tp: 'torch._C.JitType'):
        is_const = len(arg.values) == 1
        const_value: Optional[Any] = None

        if is_const:
            const_value = first(arg.values.keys())
            scope.add_var(name, const_value)
            var_info = VariableInfo.constant(name=name, origin=Origin.ARG, value=const_value, produced_by=graph_input.node(), produced=-1)            
        else:
            var_info = VariableInfo.argument(name=name, produced_by=graph_input.node(), observed=arg, torch_type=tp)
        vars.add(var_info)
        print("got input", i, var_info)


    for i,arg in enumerate(summary.args):
        graph_input = graph_inputs[i+1]  #+1 for self
        print("default input", i, graph_input, graph_input.type(), type(graph_input.type()))
        name = graph_input.debugName()
        add_input(name, graph_input, arg, graph_input.type())

    other_args: Dict[str, Value] = {}
    for i in range(len(info.summary.args) + 1, len(graph_inputs)):
        graph_input = graph_inputs[i]
        print("default arg", graph_input, graph_input.type(), type(graph_input.type()), type(graph_input.type()).__bases__)
        other_args[graph_input.debugName()] = graph_input

    for name,arg in summary.kwargs.items():
        print("kwarg", name, arg)
        input_name = name + ".1"
        graph_input = other_args[input_name]
        add_input(input_name, graph_input, arg, graph_input.type())

    const_prop_graph(graph, vars)

    #print("optimizing", graph)
    #print("type", type(graph))
    #print(f"this summary: {invocations.summarize()}")
    #print(f"module summary: {info.summary}")

    print()
    vars.dump_vars()
    return script
