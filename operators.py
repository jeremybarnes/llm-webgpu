from typing import List, Tuple, Optional, Dict, Callable, Type, Sequence, Any, Iterable, TypeVar
from variables import VariableInfo, Variables, Origin, TensorShape, TensorShapes, ShapeRange
from torch._C import Graph, Node
from torch import Tensor, Value, Block
from graphs import Scope, Operation
import copy
import torch
import operator
import math

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
                if input.debugName() not in scope.var_names:
                    scope.add_var(input.debugName(), var.const_value)
            else:
                return None

        return scope        

_torch_operators: Dict[Tuple[str,int], Callable[[Node],TorchOperator]] = {}

def get_torch_operator(node: Node) -> TorchOperator:
    arity = node.inputsSize()
    print(f"getting operator {node.kind()} with arity {arity}")
    key = (node.kind(), int(arity))
    if key in _torch_operators:
        return _torch_operators[key](node)
    key = (node.kind(), -1)
    return _torch_operators[key](node)

def torch_operator(kind: str, arity: int = -1):
    def do_operator(klass):
        key = (kind,arity)
        assert key not in _torch_operators
        _torch_operators[key] = klass
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

class SequenceConstructOperator(TorchFunctionOperator):
    """
    Constructs a sequence (list or tuple)
    """

    constructor: Type[Sequence]

    def __init__(self, node: Node, constructor: Type[Sequence]):
        super().__init__(node)
        self.constructor = constructor

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
            common_info: VariableInfo = inputs[0].deepcopy()
            has_common_info = True
            length = ShapeRange(len(inputs))

            # Are any constant?
            num_const = 0
            for input in inputs:
                if input.is_const:
                    num_const += 1
            
            for j in range(1, len(inputs)):
                other = inputs[j]    
                if common_info.is_homogeneous(other):
                    common_info.combine(other)
                else:
                    has_common_info = False
                    break

            if has_common_info and num_const == 0:
                return VariableInfo.homogeneous_sequence(name=output_name, origin=Origin.CONST_PROP, tp=self.constructor, length=length,
                                                         values=common_info, produced_by=self.node, produced=i)
            else:
                return VariableInfo.inhomogeneous_sequence(name=output_name, origin=Origin.CONST_PROP, tp=self.constructor,
                                                           values=inputs, produced_by=self.node, produced=i)

@torch_operator('prim::ListConstruct')
class ListConstructOperator(SequenceConstructOperator):
    """
    Construction of a fixed length list
    """

    def __init__(self, node: Node):
        super().__init__(node, list)


@torch_operator('prim::TupleConstruct')
class TupleConstructOperator(SequenceConstructOperator):
    """
    Construction of a fixed length tuple
    """

    def __init__(self, node: Node):
        super().__init__(node, tuple)


class SequenceIndexOperator(TorchFunctionOperator):
    """
    Indexs a sequence (list or tuple)
    """

    def is_const(self) -> bool: return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        sequence,index = inputs

        # If both are constant, default behavior is fine
        if sequence.is_const and index.is_const:
            return super().const_prop(i, inputs, vars)

        if index.is_const:
            # We know the index.  Find the element.
            el = index.typed_const_nonnull_value(int)
            if sequence.seq_els is not None and sequence.seq_el is None:
                return sequence.seq_els[el]
            elif sequence.seq_els is None and sequence.seq_el is not None:
                return sequence.seq_el
            else:
                raise RuntimeError("TODO: sequence index: const index {index.const_value} with either no or both sequences")
        else:
            raise RuntimeError("TODO: sequence index: non-const index {index}")


@torch_operator('prim::ListIndex')
class ListIndexOperator(SequenceIndexOperator):
    """
    Indexation of a list
    """

@torch_operator('prim::TupleIndex')
class TupleIndexOperator(SequenceIndexOperator):
    """
    Indexation of a tuple
    """


class SequenceUnpackOperator(TorchFunctionOperator):
    """
    Unpacks a sequence (list or tuple)
    """

    def is_const(self) -> bool: return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        numoutputs = self.node.outputsSize()
        assert len(inputs) == 1
        input = inputs[0]

        if input.is_const:
            # Unpack is easy... a constant for each entry
            val = input.const_value
            assert isinstance(val, tuple) or isinstance(val, list)
            assert len(val) == numoutputs
            def do_output(n: int) -> VariableInfo:
                output = self.node.outputsAt(n)
                valn = val[n]
                return VariableInfo.constant(name=output.debugName(), origin=Origin.CONST_PROP, value=valn,produced_by=self.node,produced=i)

            return tuple((do_output(n) for n in range(numoutputs)))
        elif input.seq_length is not None and input.seq_length.is_const():
            assert input.seq_length.const_value() == numoutputs
            def do_output(n: int) -> VariableInfo:
                output = self.node.outputsAt(n)
                el = input.seq_els[n] if input.seq_els is not None and n < len(input.seq_els) else input.seq_el
                assert el is not None
                return el.renamed(output.debugName())

            return tuple((do_output(n) for n in range(numoutputs)))

        else:
            raise RuntimeError("TODO SequenceUnpack const prop")

@torch_operator('prim::ListUnpack')
class ListUnpackOperator(SequenceUnpackOperator):
    """
    Unpack a fixed length list
    """


@torch_operator('prim::TupleUnpack')
class TupleUnpackOperator(SequenceUnpackOperator):
    """
    Unpack a fixed length tuple
    """


@torch_operator('aten::size')
class AtenSizeOperator(TorchFunctionOperator):
    """
    Handles the "size" operator.  Often has a const output even from a non-const input.
    """

    def is_const(self) -> bool: return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        print(f"size with {self.node.inputsSize()} inputs and {self.node.outputsSize()} outputs")
        assert self.node.outputsSize() == 1
        assert self.node.inputsSize() == 1 or self.node.inputsSize() == 2
        one_dim: bool = self.node.inputsSize() == 2
        output_name = self.node.outputsAt(0).debugName()
        input = inputs[0]

        known_dim: Optional[int] = None
        if one_dim:
            dim_input = inputs[1]
            if dim_input.is_const:
                assert isinstance(dim_input.const_value, int)
                known_dim = dim_input.const_value

        if input.const_type == torch.Tensor:
            sizes = inputs[0].tensor_shape
            assert sizes is not None
            if len(sizes.lengths) != 1:
                raise RuntimeError("TODO: multiple tensor sizes")
            assert len(sizes.lengths) == 1

            for length,shape in sizes.lengths.items():
                # This loop called exactly once until we handle multiple tensor sizes

                # fix up known_dim when it's negative (from the end)
                if known_dim is not None and known_dim < 0:
                    known_dim += length

                values: List[int] = []
                all_const_values: bool = True
                infos: List[VariableInfo] = []

                for j,dim in enumerate(shape):
                    # If we know the dimension, extract it
                    if known_dim is not None and j != known_dim:
                        continue
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

                if one_dim:
                    print("known_dim", known_dim)
                    print("shape", shape)
                    print("infos", infos)
                    return infos[0]

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

@torch_operator('prim::Uninitialized')
class UninitializedOperator(TorchFunctionOperator):
    """
    Handles the "uninitialized" operator, which creates an uninitialized value of the given type.
    """

    def is_const(self) -> bool: return True

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        return_type = self.node.outputsAt(0).type().kind()
        result_types = {
            'FloatType': float,
        }
        result = result_types[return_type]()
        return VariableInfo.constant(name="<<<unknown>>>", origin=Origin.CONST_PROP, value=result, produced_by=self.node, produced=i)

NodeOperator = Callable[..., Tuple[Any,...]|Any]
_ops: Dict[str, List[Operation]] = {}

_jit_type_kinds: Dict[str, type] = {
    "TensorType": torch.Tensor,
    "IntType": int,
    "ListType": list,
    "TupleType": tuple,
}

def _jit_type_to_type(jit_type: 'torch._C.JitType') -> type:
    return _jit_type_kinds[jit_type.kind()]


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

    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[torch.dtype]]) -> Optional[torch.dtype]:
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
        """
        Child classes override this method to pass on extra information about the shapes.
        This one is used where there is more than one shape dimension possible; use
        specialize_shape() where there is only one shape dimension possible.
        """
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
            #elif issubclass(input.const_type, int):
            #    return torch.int32
            #elif issubclass(input.const_type, float):
            #    return torch.float32
            return None

        dtype: Optional[torch.dtype] = self.specialize_dtype(output, inputs, [get_dtype(input) for input in inputs])

        def get_device(input: VariableInfo) -> Optional[torch.device]:
            if input.const_device:
                return input.const_device
            elif issubclass(input.const_type, (int,float)):
                return torch.device("cpu")
            return None

        device: Optional[torch.device] = self.specialize_device(output, [get_device(input) for input in inputs])

        def get_shape(input: VariableInfo) -> Optional[TensorShapes]:
            if input.tensor_shape:
                return input.tensor_shape
            elif issubclass(input.const_type, (int,float)):
                return TensorShapes.scalar()
            return None

        input_shapes = [get_shape(input) for input in inputs]

        shapes: Optional[TensorShapes] = self.specialize_shapes(output, inputs, input_shapes)

        return (dtype, shapes, device)

    def infer_variable_info(self, i: int, output: Value, inputs: List[VariableInfo]) -> VariableInfo:
        """
        Infers the variable info from the output spec.
        """
        jit_type = output.type()
        tp = _jit_type_to_type(jit_type)

        if jit_type.kind() == "TensorType":
            dtype,shape,device = self.specialize_tensor(output, inputs)
            info = VariableInfo.tensor(name=output.debugName(), origin=Origin.LOCAL, produced_by=self.node, produced=i,
                                        dtype=dtype, shape=shape, device=device)
        elif jit_type.kind() == "ListType" or jit_type.kind() == "TupleType":
            elinfo,elsinfo,elen = self.specialize_sequence(i, output, inputs)
            if elsinfo is None:
                assert elinfo is not None
                info = VariableInfo.homogeneous_sequence(name=output.debugName(), origin=Origin.LOCAL, produced_by=self.node,
                                                            produced=i,tp=tp,values=elinfo,length=elen)
            else:
                info = VariableInfo.inhomogeneous_sequence(name=output.debugName(), origin=Origin.LOCAL, produced_by=self.node,
                                                            produced=i,tp=tp,values=elsinfo)
        else:
            info = VariableInfo.local(name=output.debugName(), tp=tp, origin=Origin.LOCAL, produced_by=self.node, produced=i)
        return info

    def specialize_sequence(self, i: int, output: Value, inputs: List[VariableInfo]) -> Tuple[Optional[VariableInfo], Optional[List[VariableInfo]], ShapeRange]:
        """
        Return the sequence attributes.  Looks at the output to see what we can learn.
        """
        jtp = output.type()

        if isinstance(jtp, torch.ListType):
            el = jtp.getElementType()
            eltp = _jit_type_to_type(el)
            elinfo = VariableInfo.local(name=output.debugName() + ".<element>", origin=Origin.CONST_PROP, tp=eltp,
                                        produced_by=self.node, produced=i)
            sh = ShapeRange()
            return elinfo,None,sh

        elif isinstance(jtp, torch.TupleType):
            els = jtp.elements()
            raise RuntimeError("TODO: specialize_sequence TupleType")
        else:
            raise RuntimeError("Unknown sequence type to specialize")

    def fallback_const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        Const prop for when inputs are not all known.  We do out best from the prototype
        and annotations.
        """
        if self.node.outputsSize() == 0:
            return tuple()
        elif self.node.outputsSize() == 1:
            return self.infer_variable_info(i, self.node.outputsAt(0), inputs)
        else:
            return tuple(self.infer_variable_info(i, output, inputs) for output in self.node.outputs())

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
        if i_shape is None or w_shape is None:
            return None
        
        print("i_shape", i_shape)
        print("w_shape", w_shape)

        #mul_shape = _matmul_shape(i_shape,w_shape)
        assert i_shape is not None
        assert w_shape is not None
        b,m,n1=i_shape
        o,n2=w_shape
        print("n1 = ", n1)
        print("n2 = ", n2)
        assert n1==n2
        return TensorShape([b,m,o])

class PythonBuiltinBinaryOperator(TorchBuiltinOperator):
    """
    This is the base class for operators which are builtin to Python for non-Tensor data
    types, but are also overridden by PyTorch for Tensors.
    """

    python_op: Callable[[Any, Any], Any]
    op_str: str
    builtin_cases: List[Tuple[type, type, type]]

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation], python_op: Callable[[Any,Any], Any], op_str: str):
        super().__init__(node, ops)
        self.python_op = python_op
        self.op_str = op_str
        self.builtin_cases = [
            (float, float, float),
            (float, int,   float),
            (int,   float, float),
            (int,   int,   int),
        ]

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        # For tensor operations only.  Does the broadcasting.
        a_shape,b_shape = input_shape
        print(f"specialize_shape: {a_shape} {self.op_str} {b_shape}")
        if a_shape is None or b_shape is None:
            # Can't know the shape of the output as broadcasting can cause all kinds of things to happen
            return None  
        else:
            return broadcast_shapes(a_shape, b_shape)

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        Constant propagation needs to handle both Tensor types (by calling aten) and Python types (which use the
        default operators).
        """
        a = inputs[0]
        b = inputs[1]
        a_type,b_type = a.const_type,b.const_type

        # Unknown types?  Nothing to do, we just have a generic variable
        if a_type is None or b_type is None:
            return super().const_prop(i, inputs, vars)

        # Tensor operation?  Existing implementation works fine
        if issubclass(a_type, torch.Tensor) or issubclass(b_type, torch.Tensor):
            return super().const_prop(i, inputs, vars)

        if len(inputs) != 2:
            raise RuntimeError(f"const_prop operator {self.op_str}: TODO: more than two arguments")

        # Constant operation?  Use the Python builtin logic
        if a.is_const and b.is_const:
            result = self.python_op(a.const_value, b.const_value)
            return VariableInfo.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP, value=result,
                                         produced_by=self.node,produced=i)

        # Otherwise it's complicated... we need to emulate all of the Python inbuilt cases
        def match_case(case: Tuple[type, type, type]) -> Optional[type]:
            a_type,b_type,res_type = case
            assert a.const_type is not None
            assert b.const_type is not None
            if issubclass(a.const_type, a_type) and issubclass(b.const_type, b_type):
                return res_type
            return None

        for case in self.builtin_cases:
            res = match_case(case)
            if res is not None:
                return VariableInfo.local(name=self.node.outputsAt(0).debugName(), origin=Origin.LOCAL,
                                          tp=res,produced_by=self.node,produced=i)

        print("a", a)
        print("b", b)
        raise RuntimeError(f"TODO: complex unmatched case for non-Tensor {type(self)} {a_type} {b_type}")

@torch_operator('aten::sub')
class SubtractOperator(PythonBuiltinBinaryOperator):
    """
    Linear a - b operator.
    """

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation] = None):
        super().__init__(node, ops, operator.sub, "-")
        self.builtin_cases.extend([
        ])

@torch_operator('aten::mul')
class MultiplyOperator(PythonBuiltinBinaryOperator):
    """
    Linear a * b operator.
    """

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation] = None):
        super().__init__(node, ops, operator.mul, "*")
        self.builtin_cases.extend([
            (str, int, str),
            (list, int, list),       
        ])

@torch_operator('aten::pow')
class PowerOperator(PythonBuiltinBinaryOperator):
    """
    a ** b operator.
    """

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation] = None):
        super().__init__(node, ops, operator.pow, "**")
        self.builtin_cases.extend([
        ])


@torch_operator('aten::add', 2)
class AdditionOperator(PythonBuiltinBinaryOperator):
    """
    Linear a + b operator.
    """

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation] = None):
        super().__init__(node, ops, operator.add, "+")
        self.builtin_cases.extend([            
            (str, str, str),
            (list, list, list),       
            (tuple, tuple, tuple),       
        ])

@torch_operator('aten::add', 3)
class ScaledAdditionOperator(TorchBuiltinOperator):
    """
    Linear a + alpha*b operator.
    """

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        a_shape,b_shape,_ = input_shape
        if a_shape is None or b_shape is None:
            # Can't know the shape of the output as broadcasting can cause all kinds of things to happen
            return None  
        else:
            return broadcast_shapes(a_shape, b_shape)

class PythonBuiltinUnaryOperator(TorchBuiltinOperator):
    """
    This is the base class for operators which are builtin to Python for non-Tensor data
    types, but are also overridden by PyTorch for Tensors.
    """

    python_op: Callable[[Any], Any]
    op_str: str
    builtin_cases: List[Tuple[type, type]]

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation], python_op: Callable[[Any], Any], op_str: str):
        super().__init__(node, ops)
        self.python_op = python_op
        self.op_str = op_str
        self.builtin_cases = [
            (float, float),
            (int,   float),
        ]

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        # For tensor operations only.  Does the broadcasting.
        a_shape, = input_shape
        print(f"specialize_shape: {self.op_str} {a_shape}")
        if a_shape is None:
            # Can't know the shape of the output as broadcasting can cause all kinds of things to happen
            return None  
        else:
            return a_shape

    def const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        Constant propagation needs to handle both Tensor types (by calling aten) and Python types (which use the
        default operators).
        """
        a, = inputs
        a_type = a.const_type

        # Unknown types?  Nothing to do, we just have a generic variable
        if a_type is None:
            return super().const_prop(i, inputs, vars)

        # Tensor operation?  Existing implementation works fine
        if issubclass(a_type, torch.Tensor):
            return super().const_prop(i, inputs, vars)

        if len(inputs) != 1:
            raise RuntimeError(f"const_prop operator {self.op_str}: TODO: more than one argument")

        # Constant operation?  Use the Python builtin logic
        if a.is_const:
            result = self.python_op(a.const_value)
            return VariableInfo.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP, value=result,
                                         produced_by=self.node,produced=i)

        # Otherwise it's complicated... we need to emulate all of the Python inbuilt cases
        def match_case(case: Tuple[type, type]) -> Optional[type]:
            a_type,res_type = case
            assert a.const_type is not None
            if issubclass(a.const_type, a_type):
                return res_type
            return None

        for case in self.builtin_cases:
            res = match_case(case)
            if res is not None:
                return VariableInfo.local(name=self.node.outputsAt(0).debugName(), origin=Origin.LOCAL,
                                          tp=res,produced_by=self.node,produced=i)

        print("a", a)
        raise RuntimeError(f"TODO: complex unmatched case for non-Tensor binary {type(self)} {a_type} {b_type}")

@torch_operator('aten::tanh')
class TahnhOperator(PythonBuiltinUnaryOperator):
    """
    Unary tanh(x) operator.
    """

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation] = None):
        super().__init__(node, ops, math.tanh, "tanh")
        self.builtin_cases.extend([
        ])

def _matmul_shape(a_shape: TensorShape, b_shape: TensorShape) -> TensorShape:
    """
    Returns the output shape of a matrix multiply between tensors of the two shapes,
    including broadcasting.
    """
    la = len(a_shape)
    lb = len(b_shape)

    # From https://pytorch.org/docs/stable/generated/torch.matmul.html

    # The behavior depends on the dimensionality of the tensors as follows:

    #    If both tensors are 1-dimensional, the dot product (scalar) is returned.
    if la == 1 and lb == 1:
        return TensorShape([])
    
    #    If both arguments are 2-dimensional, the matrix-matrix product is returned.
    elif la == 2 and lb == 2:
        m,n1 = a_shape
        n2,o = b_shape
        assert(n1 == n2)
        return TensorShape([m,o])

    #    If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1
    #    is prepended to its dimension for the purpose of the matrix multiply. After the matrix
    #    multiply, the prepended dimension is removed.
    elif la == 1 and lb == 2:
        n1 = a_shape[0]
        n2,o = b_shape
        assert(n1 == n2)
        return TensorShape([o])

    #    If the first argument is 2-dimensional and the second argument is 1-dimensional, the
    #    matrix-vector product is returned.
    elif la == 2 and lb == 1:
        m,n1 = a_shape
        n2 = b_shape[0]
        assert n1 == n2
        return TensorShape([m])

    #    If both arguments are at least 1-dimensional and at least one argument is N-dimensional
    #    (where N > 2), then a batched matrix multiply is returned.

    else:
        #    If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose
        #    of the batched matrix multiply and removed after.
        if la == 1:
            m1,n1,b1 = ShapeRange(1),a_shape[0],[]
        else:
            m1,n1,b1 = a_shape[-2],a_shape[-1],list(a_shape[0:-2])

        #    If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of
        #    the batched matrix multiple and removed after. 
        if lb == 1:
            m2,n2,b2 = b_shape[0],ShapeRange(1),[]
        else:
            m2,n2,b2 = b_shape[-2],b_shape[-1],list(b_shape[0:-2])

        print(f"a: {b1} x {m1} x{n1}")
        print(f"b: {b2} x {m2} x{n2}")

        n = broadcast_dim(n1, m2)

        print(f"out: {b2} x {m1} x {n2}")

        #    The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be
        #    broadcastable). For example, if input is a (j×1×n×n)(j×1×n×n) tensor and other is a
        #    (k×n×n)(k×n×n) tensor, out will be a (j×k×n×n)(j×k×n×n) tensor.

        #    Note that the broadcasting logic only looks at the batch dimensions when determining if the
        #    inputs are broadcastable, and not the matrix dimensions. For example, if input is a
        #    (j×1×n×m)(j×1×n×m) tensor and other is a (k×m×p)(k×m×p) tensor, these inputs are valid for
        #    broadcasting even though the final two dimensions (i.e. the matrix dimensions) are different.
        #    out will be a (j×k×n×p)(j×k×n×p) tensor.

        batch_dims = broadcast_shapes(TensorShape(b1), TensorShape(b2))
        dims = batch_dims.dims
        dims.extend([m1,n2])
        print(f"returning {dims}")
        return TensorShape(dims)

@torch_operator('aten::matmul')
class MatMulOperator(TorchBuiltinOperator):
    """
    Linear ax operator.
    """
    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        print("input_shape", input_shape)
        a_shape,b_shape = input_shape
        if a_shape is None or b_shape is None:
            return None
        return _matmul_shape(a_shape, b_shape)


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

        print("input_shape", input_shapes[0])
        print("new_shape", inputs[1])

        new_shape = inputs[1]

        if new_shape.is_const:
            # Constant shape... we know the output shape
            assert isinstance(new_shape.const_value, list)
            print("new_shape", new_shape.const_type, new_shape.const_value)
            return TensorShapes.from_shape(new_shape.const_value)
        else:
            seq_len = new_shape.seq_length
            assert seq_len is not None
            if seq_len.is_const():
                l = seq_len.const_value()
                if new_shape.seq_el is not None:
                    # Homogeneous
                    raise RuntimeError("TODO: homogeneous")
                else:
                    assert new_shape.seq_els is not None
                    assert len(new_shape.seq_els) == l

                    def get_dim(input: VariableInfo) -> ShapeRange:
                        if input.is_const:
                            return ShapeRange(input.const_value)
                        else:
                            return ShapeRange()

                    dims = [get_dim(input) for input in new_shape.seq_els]
                    dim_dict = { l: TensorShape(dims) }

                    return TensorShapes(dim_dict)
            else:
                # Nothing known
                return None

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

@torch_operator('aten::permute')
class AtenPermuteOperator(TorchBuiltinOperator):
    """
    Handles the "permite" operator.  This manipulates the order of dimensions without touching
    the data.
    """

    def __init__(self, node: Node):
        super().__init__(node, Tensor.permute)

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[TensorShapes]]) -> Optional[TensorShapes]:

        old_shape = input_shapes[0]
        new_order = inputs[1]

        print("old_shape", old_shape)
        print("new_order", new_order)

        if old_shape is not None and new_order.is_const:
            new_order_val = new_order.const_value
            assert isinstance(new_order_val, list) or isinstance(new_order_val, tuple)
            new_shape_lens: Dict[int, TensorShape] = {}
            for l in old_shape.lengths.values():
                new_shape_lens[len(l)] = TensorShape([l[v] for v in new_order_val])
            return TensorShapes(new_shape_lens)

        else:
            # Non-const order; shape propagation terminates
            return None


@torch_operator('aten::transpose')
class AtenTransposeOperator(TorchBuiltinOperator):
    """
    Handles the "transpose" operator.
    """

    def __init__(self, node: Node):
        super().__init__(node, Tensor.transpose)

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[TensorShapes]]) -> Optional[TensorShapes]:

        old_shape = input_shapes[0]
        if old_shape is None or not inputs[1].is_const or not inputs[2].is_const:
            return None

        assert isinstance(inputs[1].const_value, int)
        assert isinstance(inputs[2].const_value, int)
        dim1: int = inputs[1].const_value
        dim2: int = inputs[2].const_value

        new_shape_lens: Dict[int, TensorShape] = {}
        for l,len_shape in old_shape.lengths.items():

            new_order = list([x for x in range(l)])
            tmp = new_order[dim1]
            new_order[dim1] = new_order[dim2]
            new_order[dim2] = tmp

            print("old_shape", old_shape)
            print("new_order", new_order)

            new_shape_lens[l] = TensorShape([len_shape[v] for v in new_order])

        return TensorShapes(new_shape_lens)

@torch_operator('aten::contiguous')
class AtenContiguousOperator(TorchBuiltinOperator):
    """
    Handles the "contiguous" operation, which copies a tensor if necessary so that
    its data is contiguous.
    """

    def __init__(self, node: Node):
        super().__init__(node, Tensor.contiguous)

    def is_const(self) -> bool: return True

    def fallback_const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        input,memfmt = inputs
        if memfmt.is_const and (memfmt.const_value is None or memfmt.const_value in {0,1}):
            return input
        # TODO: does a different memory format matter
        # Contiguous = 0
        # Preserve = 1
        # ChannelsLast = 2
        # ChannelsLast3d = 3
        return input
        raise RuntimeError(f"TODO: contiguous: non-default memory format {memfmt}")
        

@torch_operator('aten::to')
class AtenToOperator(TorchBuiltinOperator):
    """
    Handles the "to" operator.  Copies a tensor to a different dtype or device.
    """

    def __init__(self, node: Node):
        super().__init__(node, Tensor.to)

    def is_const(self) -> bool: return True

    def fallback_const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        from aten import int_to_dtype, dtype_to_int, int_to_memory_format
        for input in inputs:
            print(input.name, input.const_type, input.const_value)
        tensor = inputs[0]
        shape = inputs[0].tensor_shape
        dtype: Optional[torch.dtype] = inputs[0].const_dtype
        device: Optional[torch.device] = inputs[0].const_device
        assert inputs[1].const_type is not None
        if issubclass(inputs[1].const_type, torch.Tensor):
            # First variant: tensor.to(tensor, ...)
            dtype = inputs[1].const_dtype
            device = inputs[1].const_device
            offset = 2
        else:
            if inputs[1].is_const:
                print("to const value", inputs[1].const_value)
                if isinstance(inputs[1].const_value, torch.device):
                    device = inputs[1].const_value
                else:
                    assert isinstance(inputs[1].const_value, int)
                    dtype = int_to_dtype(inputs[1].const_value)
            offset = 2
            if device is None and inputs[2].const_type != bool:
                if inputs[2].is_const:
                    assert isinstance(inputs[2].const_value, torch.device)
                    device = inputs[2].const_value
                offset = 3

        # Rest are non_blocking, copy and memory_format which don't affect output tensor

        return VariableInfo.tensor(name="<<<unknown>>>", origin=Origin.CONST_PROP, dtype=dtype, device=device,
                                   shape=shape, produced_by=self.node,produced=i)

@torch_operator('aten::slice')
class AtenSliceOperator(TorchBuiltinOperator):
    """
    Handles the "slice" operator.  Manipulates only strides, offsets and sizes.
    """

    def __init__(self, node: Node):
        super().__init__(node, torch.slice_copy)

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[TensorShapes]]) -> Optional[TensorShapes]:

        for input in inputs:
            print(f"input {input.name} {input.const_type} {input.const_value}")

        input,dim_number,start,end,step = inputs

        if not dim_number.is_const or input_shapes[0] is None:
            # TODO: could specialize more
            return None
        assert isinstance(dim_number.const_value, int)

        input_shape = input_shapes[0]

        assert len(input_shape.lengths) == 1

        input_dims = first(input_shape.lengths.values())
        if input_dims is None:
            return input_shape

        sliced_dim = input_dims.dims[dim_number.const_value]
        if sliced_dim.is_const():
            if start.is_const and end.is_const and step.is_const:
                raise RuntimeError(f"slice of known dimension {sliced_dim} {start.const_value}:{end.const_value}:{step.const_value}")
            input_dims.dims[dim_number.const_value] = ShapeRange()
            # TODO: could specialize more
            return input_shape
    
    def fallback_const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        result = super().fallback_const_prop(i, inputs, vars)
        assert isinstance(result, VariableInfo)
        input = inputs[0]

        if input.const_type is not None and issubclass(input.const_type, torch.Tensor):
            return result

        if not result.is_const and input.seq_els is not None and input.seq_el is None and input.const_type is not None and issubclass(input.const_type, list):
            # Specializing sequence... slicing a list with constant indices returns a slice of the VariableInfos
            start = inputs[1].typed_default_value(0)
            end = inputs[2].typed_const_value(int)
            step = inputs[3].typed_default_value(1)

            #print("start", start, "end", end, "step", step)

            if start is None or end is None or step is None:
                return result

            new_els = list(input.seq_els[start:end:step])

            all_const = all(map(lambda x: x.is_const, new_els))
            if all_const:
                const_values = list([x.const_value for x in new_els])
                return VariableInfo.constant(name=result.name, origin=Origin.CONST_PROP, value=const_values, produced_by=self.node, produced=i)
            else:
                return VariableInfo.inhomogeneous_sequence(name=result.name, origin=Origin.CONST_PROP, tp=list, values=new_els, produced_by=self.node, produced=i)

            raise RuntimeError(f"TODO slice fallback_const_prop sequence {inputs[0]} {dim} {start} {end} {step} {result}")

        raise RuntimeError(f"TODO slice fallback_const_prop sequence {result}")
        return result

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
    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[torch.dtype]]) -> Optional[torch.dtype]:
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

@torch_operator('prim::device')
class DeviceOperator(TorchBuiltinOperator):
    """device() operator that returns the device of a tensor"""

    def __init__(self, node: Node):
        super().__init__(node, Tensor.get_device)

    def is_const(self) -> bool: return True  # later... different per device

    def fallback_const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        if inputs[0].const_device is None:
            return VariableInfo.local(name="<<<unknown>>>", origin=Origin.LOCAL, tp=torch.device, produced_by=self.node,
                                      produced=i)
        else:
            return VariableInfo.constant(name="<<<unknown>>>", origin=Origin.LOCAL, value=inputs[0].const_device, produced_by=self.node,
                                         produced=i)

def get_dtype(t: Tensor) -> int: #torch.dtype:
    from aten import dtype_to_int
    return dtype_to_int(t.dtype)

@torch_operator('prim::dtype')
class DataTypeOperator(TorchBuiltinOperator):
    """dtype() operator that returns the data type of a tensor"""

    def __init__(self, node: Node):
        super().__init__(node, get_dtype)

    def is_const(self) -> bool: return True  # later... different per device

    def fallback_const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        from aten import dtype_to_int
        if inputs[0].const_dtype is None:
            return VariableInfo.local(name="<<<unknown>>>", origin=Origin.LOCAL, tp=int, produced_by=self.node,
                                      produced=i)
        else:
            return VariableInfo.constant(name="<<<unknown>>>", origin=Origin.LOCAL, value=dtype_to_int(inputs[0].const_dtype),
                                         produced_by=self.node, produced=i)

@torch_operator('aten::scalar_tensor')
class ScalarTensorOperator(TorchBuiltinOperator):
    """create a tensor from a scalar"""

    def __init__(self, node: Node):
        super().__init__(node, torch.scalar_tensor)

    def is_const(self) -> bool: return True

    def fallback_const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        raise RuntimeError("scalar_tensor fallback_const_prop")

    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[torch.dtype]]) -> Optional[torch.dtype]:
        raise RuntimeError("scalar_tensor specialize_dtype")

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        return TensorShape([])

def broadcast_dim(dim1: Optional[ShapeRange], dim2: Optional[ShapeRange]) -> Optional[ShapeRange]:
    """
    Return the dimension of the broadcast two dimensions, or throw if they
    are not compatible.
    """

    def broadcast_one(dim: ShapeRange) -> Optional[ShapeRange]:
        # If it's dimension 1, it's broadcastable so we don't know the dimension
        if dim.is_const():
            if dim.const_value == 1:
                return None
            else:
                return dim
        elif dim.min <= 1 and dim.max >= 1:
            return None
        else:
            return dim

    if dim1 is None:
        if dim2 is None:
            return None
        else:
            return broadcast_one(dim2)
    else:
        if dim2 is None:
            return broadcast_one(dim1)
        
        # Both are non-null
        if dim1.is_const() and dim1.const_value() == 1:
            return dim2
        elif dim2.is_const() and dim2.const_value() == 1:
            return dim1
        else:
            result = copy.deepcopy(dim1)
            result.broadcast_from(dim2)
            return result


def broadcast_shape(shapes: Sequence[TensorShape]) -> TensorShape:
    """
    Return the shape of the list of input tensors broadcast to the same shape.
    """
    #https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics

    # Two tensors are “broadcastable” if the following rules hold:
    #
    # -  Each tensor has at least one dimension.
    # -  When iterating over the dimension sizes, starting at the trailing dimension, the
    #    dimension sizes must either be equal, one of them is 1, or one of them does not exist.
    #
    # If two tensors x, y are “broadcastable”, the resulting tensor size is calculated as follows:
    #
    # -  If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the
    #    tensor with fewer dimensions to make them equal length.
    # -  Then, for each dimension size, the resulting dimension size is the max of the sizes of x
    #    and y along that dimension.

    if len(shapes) == 0:
        raise RuntimeError("Cannot broadcast an empty set of shapes")

    max_len = max([len(s.dims) for s in shapes])
    print("max_len = ", max_len)
    dims = [ShapeRange(1) for _ in range(max_len)]

    def do_shape(sh: TensorShape):
        newsh = [ShapeRange(1) for _ in range(max_len - len(sh.dims))]
        newsh.extend(sh.dims)

        print("do_shape", sh.dims, newsh)

        for outsh,insh in zip(dims,newsh):
            outsh.broadcast_from(insh)
    
    for sh in shapes:
        do_shape(sh)

    print("broadcast_shape returning", dims)

    return TensorShape(dims)

def broadcast_shapes(*shapes: TensorShape) -> TensorShape:
    """
    Return the shape of the arguments broadcast to the same shape.
    """
    return broadcast_shape(shapes)

def broadcast_dtypes(*dtypes: Optional[torch.dtype]) -> Optional[torch.dtype]:
    """
    Return the dtype that covers the input dtypes.
    """

    result: Optional[torch.dtype] = None

    precedence: Dict[torch.dtype, int] = {
        torch.float64: 26,
        torch.float32: 23,
        torch.float16: 20,
        torch.int64: 16,
        torch.int32: 13,
        torch.int16: 12,
        torch.int8: 11,
        torch.uint8: 10,
        torch.bool: 5
    }

    def accum(result: Optional[torch.dtype], t: Optional[torch.dtype]):
        if t is None:
            return result
        if result is None:
            return t
        p1 = precedence[result]
        p2 = precedence[t]

        return t if p2 > p1 else result

    for t in dtypes:
        result = accum(result, t)
    return result

@torch_operator('aten::where')
class WhereOperator(TorchBuiltinOperator):
    """Select elements from one tensor based on a condition tensor"""

    def is_const(self) -> bool: return True

    #def fallback_const_prop(self, i: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
    #    for input in inputs:
    #        print(input.name, input.const_type, input.const_dtype, input.tensor_shape, input.const_value)
    #    raise RuntimeError("where fallback_const_prop")

    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[torch.dtype]]) -> Optional[torch.dtype]:
        _,left,right = input_dtype
        return broadcast_dtypes(left, right)

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        cond,left,right = input_shape
        assert cond is not None
        assert left is not None
        assert right is not None
        sh = broadcast_shapes(cond, left, right)

        print("cond", cond)
        print("left", left)
        print("right", right, right.dims)
        print("shape", sh)

        return sh

@torch_operator('aten::softmax')
class SoftMaxOperator(TorchBuiltinOperator):
    """Differentiable max function"""

    def is_const(self) -> bool: return True

    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[torch.dtype]]) -> Optional[torch.dtype]:
        input,dim,dtype = inputs
        if dtype.is_const and dtype.const_value is None:
            # Take the input dtype
            dtype,_,_ = input_dtype
            return dtype
        elif dtype.is_const:
            # Take the passed dtype
            return dtype.const_value
        else:
            # Can't determine dtype
            return None

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[TensorShape]]) -> Optional[TensorShape]:
        shape,_,_ = input_shape
        return shape

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

            #print("before const prop", op)
            #vars.dump_vars(indent)

            print("const_prop is", op.const_prop)

            result = op.const_prop(i, inputs, vars)

            #print("after const prop", op)
            #vars.dump_vars(indent)

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
