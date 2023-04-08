from typing import List, Tuple, Optional, Dict, Callable, Type, Sequence, Any, Iterable, TypeVar
from variables import VariableInfo, Variables, Origin, TensorShapeVariable, TorchOperator, get_torch_operator, torch_operator
from utils import first
from torch._C import Graph, Node
from torch import Tensor, Value, Block
from graphs import Scope, Operation
import copy
import torch
import operator
import math
import traceback
from collections import OrderedDict


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

    def is_const(self) -> bool:
        print("WARNING: TOOD: really implement is_const() for block")
        return True

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert len(inputs) == 1, "block selection happens with one input only"
        input = inputs[0]
        num_outputs = self.node.outputsSize()

        results: List[List[VariableInfo]] = []
        all_vars: List[Variables] = []
        outputs: List[VariableInfo]

        def process_block(block: Block):
            new_vars = vars.new_frame()
            print("exedcuting block", list(block.nodes()))
            const_prop_graph(block, new_vars)
            #print("new_vars", new_vars)
            new_vars.dump_vars('        ')
            return_node = block.returnNode()
            assert return_node.kind() == "prim::Return"
            assert return_node.inputsSize() == num_outputs

            this_result = [new_vars.renamed(new_vars.get(input.debugName(), produced), output.debugName()) for input,output in zip(return_node.inputs(),self.node.outputs())]
            results.append(this_result)
            all_vars.append(new_vars)

        if not input.is_const():
            # It's not a constant.  Perform constant propagation for each block and take the
            # union of the outputs.

            for block in self.blocks:
                process_block(block)

            all_keys: OrderedDict[str, List[VariableInfo]] = OrderedDict()

            for block_vars in all_vars:
                for name,info in block_vars.items():
                    if name not in all_keys:
                        all_keys[name] = []
                    all_keys[name].append(info)

            for key,infos in all_keys.items():
                # Is this variable info in all of the branches?  If so, it's not optional
                in_all = len(infos) == len(all_vars)
                print("adding if variable", key, " from ", len(infos), " blocks", infos)
                if in_all:
                    vars.add_covering(infos, key, self.node, produced)
                else:
                    vars.add_optional(infos, key, self.node, produced)

            # Deal with the outputs
            combined_results = []
            num_outputs = self.node.outputsSize()
            for i in range(num_outputs):
                this_res = [res[i] for res in results]
                #print("output", i, "combining", this_res)
                combined_results.append(vars.covering(this_res, self.node.outputsAt(i).debugName(), self.node, produced))
                #print("got", combined_results[-1])

            outputs = combined_results

        else:
            # It's a constant.  We do only one block.
            block_num = 0 if input.const_value() else 1
            process_block(self.blocks[block_num])
            outputs = results[0]

        if num_outputs == 0:
            return tuple()
        elif num_outputs == 1:
            return outputs[0]
        else:
            return tuple(outputs)

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
            if var.is_const():
                if input.debugName() not in scope.var_names:
                    scope.add_var(input.debugName(), var.const_value())
            else:
                return None

        return scope        

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

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
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

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        This is a constant, so it produces a constant that can be propagated.  Single output with the
        provided value.
        """
        assert len(inputs) == 0
        output = self.node.outputsAt(0)
        result = vars.constant(name=output.debugName(), origin=Origin.CONST_PROP,
                                       value=output.toIValue(), produced_by=self.node, produced=produced)
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

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert len(inputs) == 1
        attr_name = self.node.s("name")

        #Const prop for getattr depends on constness of the argument; the attribute name is always
        #a constant.
        if inputs[0].is_const():
            val = getattr(inputs[0].const_value(), attr_name)
            result = vars.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP,
                                           value=val, produced_by=self.node, produced=produced)
        else:
            # Potentially we could look up the type and see if there are any type annotations on the
            # field we're retrieving.
            # (TODO)x
            # For now, return an any
            result = vars.any(name=self.node.outputsAt(0).debugName(), origin=Origin.LOCAL,
                                        produced_by=self.node, produced=produced)

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

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert len(inputs) == 2
        assert self.node.outputsSize() == 1

        val: Optional[bool] = None
        # Otherwise if values are known...
        if inputs[0].is_const() and inputs[1].is_const():
            val = inputs[0].const_value() is inputs[1].const_value()
            #print(f"const is: {inputs[0].const_value()} is {inputs[1].const_value()} = {val}")
        elif inputs[0].const_type() == type(None) and inputs[1].const_type() == type(None):
            # x is None can be short circuited based on types only
            val = True
            #print("inputs[0] = ", inputs[0], inputs[0].const_type())
            #print("inputs[1] = ", inputs[1], inputs[1].const_type())
            #print(f"const none typed is: {inputs[0].const_type()} is {inputs[1].const_type()} = {val}")

        if val is not None:
            #print("invert is", self.invert, "val is", val)
            if self.invert:
                val = not val
            return vars.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP,
                                         value=val, produced_by=self.node, produced=produced)

        # Otherwise, it's a boolean valued unknown
        return vars.local(name=self.node.outputsAt(0).debugName(), origin=Origin.LOCAL, tp=bool,
                          produced_by=self.node, produced=produced)

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

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert self.node.outputsSize() == 1
        output_name = self.node.outputsAt(0).debugName()

        collected = self.collect_const_inputs(inputs)
        if collected is not None:
            # All inputs are constant, so we return a constant list
            result = [value for name,value in collected.vars]
            return vars.constant(name=output_name, origin=Origin.CONST_PROP, value=result,
                                 produced_by=self.node, produced=produced)

        else:
            # Propagate what we know about the list
            return vars.inhomogeneous_sequence(name=output_name, origin=Origin.CONST_PROP, tp=self.constructor,
                                                           values=inputs, produced_by=self.node, produced=produced)

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

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        sequence,index = inputs
        sequence = sequence.as_sequence()
        index = index.as_int()

        # If both are constant, default behavior is fine
        if sequence.is_const() and index.is_const():
            return super().const_prop(produced, inputs, vars)

        if index.is_const() and sequence.sequence_length_is_const():
            return sequence[index.const_value()]
            # We know the index.  Find the element.
            el = index.typed_const_nonnull_value(int)
            if sequence.seq_els is not None:
                return sequence.seq_els[el]
            elif sequence.seq_els is None and sequence.seq_el is not None:
                return sequence.seq_el
            else:
                raise RuntimeError("TODO: sequence index: const index {index.const_value()} with either no or both sequences")
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

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        numoutputs = self.node.outputsSize()
        assert len(inputs) == 1
        input = inputs[0]

        if input.is_const():
            # Unpack is easy... a constant for each entry
            val = input.const_value
            assert isinstance(val, tuple) or isinstance(val, list)
            assert len(val) == numoutputs
            def do_output(n: int) -> VariableInfo:
                output = self.node.outputsAt(n)
                valn = val[n]
                return vars.constant(name=output.debugName(), origin=Origin.CONST_PROP, value=valn,produced_by=self.node,produced=produced)

            return tuple((do_output(n) for n in range(numoutputs)))
        elif input.sequence_length_is_const():
            assert input.sequence_const_length() == numoutputs
            return tuple(input)

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

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        #print(f"size with {self.node.inputsSize()} inputs and {self.node.outputsSize()} outputs")
        assert self.node.outputsSize() == 1
        assert self.node.inputsSize() == 1 or self.node.inputsSize() == 2
        one_dim: bool = self.node.inputsSize() == 2
        output_name = self.node.outputsAt(0).debugName()
        input = inputs[0]

        assert input.is_tensor()
        sh = inputs[0].tensor_shape()
        assert isinstance(sh, TensorShapeVariable)

        if one_dim:
            dim_input = inputs[1]
            if dim_input.is_const():
                assert isinstance(dim_input.const_value(), int)
                return sh[dim_input.const_value()]
            else:
                dim_name=output_name + f".#size_element"
                return vars.local(name=dim_name, origin=Origin.CONST_PROP, tp=int, produced_by=self.node, produced=produced)

        else:
            return sh

@torch_operator('aten::__getitem__')
class AtenGetItemOperator(TorchFunctionOperator):
    """
    Handles the "getitem" operator.  Often used to manipulate sizes, so we make special
    efforts to ensure that const propagation happens properly.
    """

    def is_const(self) -> bool: return True

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        assert self.node.outputsSize() == 1
        assert self.node.inputsSize() == 2
        output_name = self.node.outputsAt(0).debugName()
        input = inputs[0]
        item = inputs[1]

        def get_from_seq(item_info: VariableInfo) -> VariableInfo:
            "Return a variable info from a known VariableInfo which may contain a constant"
            if item_info.is_const():
                # This value is constant in the sequence even if others aren't
                return vars.constant(name=output_name, origin=Origin.CONST_PROP, value=item_info.const_value(),
                                     produced_by=self.node, produced=produced)
            else:
                return item_info.renamed(output_name)

        if item.is_const():
            n = item.const_value()
            assert n is not None

            if input.is_const():
                # Simple const value propagation
                assert input.const_value() is not None
                value = input.const_value()[item.const_value()]
                return vars.constant(name=output_name, origin=Origin.CONST_PROP, value=value, produced_by=self.node,
                                     produced=produced)
            elif input.is_sequence():
                return input.sequence_element_at_index(n)
            else:
                raise RuntimeError("TODO: const item non-const sequence unhandled case")

        raise NotImplementedError()
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

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        return_type = self.node.outputsAt(0).type().kind()
        result_types = {
            'FloatType': float,
        }
        result = result_types[return_type]()
        return vars.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP, value=result, produced_by=self.node, produced=produced)

@torch_operator('prim::unchecked_cast')
class UninitializedOperator(TorchFunctionOperator):
    """
    Casts a value to a different type.
    """

    def is_const(self) -> bool: return True

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        input, = inputs

        input_type = input.torch_type()
        output_type = self.node.outputsAt(0).type()

        if input.is_optional() and not isinstance(output_type, torch.OptionalType):
            # Normally used to remove optional
            input_str = input_type.getElementType().kind()
            output_str = output_type.kind()

            if input_str != output_str:
                raise RuntimeError(f"Attempt to unchecked_cast optional {input_type} to incompatible {output_type}")
            return input.non_optional().renamed(self.node.outputsAt(0).debugName())

        if input_type.kind() == output_type.kind():
            # Types are the same, we can just return it
            return input.renamed(self.node.outputsAt(0).debugName())

        # Check the type
        print("input", input_type)
        print("output", output_type)

        raise NotImplementedError()
        return_type = self.node.outputsAt(0).type()

        result_types = {
            'FloatType': float,
        }
        result = result_types[return_type]()
        return vars.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP, value=result, produced_by=self.node, produced=produced)

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

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> Optional[VariableInfo]:
        """
        Child classes override this method to implement shape calculations.
        """
        return None

    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        """
        Child classes override this method to implement dtype calculations.
        """
        variables: List[VariableInfo] = []
        fixed: Optional[torch.device] = None
        for dtype in input_dtype:
            if dtype is None:
                continue
            if dtype.is_const():
                if fixed is None:
                    fixed = dtype.const_value()
                elif fixed != dtype.const_value():
                    raise RuntimeError(f"Inconsistent dtypes {fixed} and {dtype.const_value()}; need to override {self.__class__.__name__}.specialize_dtype")
                # fall through
            else:
                variables.append(dtype)

        if fixed is not None:
            if len(variables) > 0:
                vars.unify(fixed, *variables)
        else:
            vars.alias(*variables)
        
        if fixed is not None:
            return vars.constant(name=output.debugName(), origin=Origin.CONST_PROP, value=fixed, produced_by=self.node, produced=produced)
        else:
            return variables[0]

    def specialize_device(self, output: Value, input_device: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        """
        Child classes override this method to implement device calculations.
        """
        result = None
        variables: List[VariableInfo] = []
        fixed: Optional[torch.device] = None
        for device in input_device:
            if device is None:
                continue
            if device.is_const():
                if fixed is None:
                    fixed = device.const_value()
                elif fixed != device.const_value():
                    raise RuntimeError(f"Inconsistent devices {fixed} and {device.const_value()}; need to override {self.__class__.__name__}.specialize_device")
                # fall through
            else:
                variables.append(device)

        if fixed is not None:
            if len(variables) > 0:
                vars.unify(fixed, *variables)
        else:
            vars.alias(*variables)
        
        if fixed is not None:
            return vars.constant(name=output.debugName(), origin=Origin.CONST_PROP, value=fixed, produced_by=self.node, produced=produced)
        else:
            return variables[0]

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        """
        Child classes override this method to pass on extra information about the shapes.
        This one is used where there is more than one shape dimension possible; use
        specialize_shape() where there is only one shape dimension possible.
        """
        specialized = self.specialize_shape(output, inputs, input_shapes, produced, vars)
        if not isinstance(specialized, VariableInfo):
            raise RuntimeError(f"specialize_shape() for {type(self).__name__} returned wrong type {type(specialized).__name__}")
        return specialized

    def specialize_tensor(self, output: Value, inputs: List[VariableInfo], produced: int, vars: Variables) -> Tuple[VariableInfo, VariableInfo, VariableInfo]:
        """
        Return the tensor size and attributes of a returned tensor.  Needs to be specialized per operation as
        there is no way to generically know.
        """
        def get_dtype(input: VariableInfo) -> Optional[VariableInfo]:
            if not input.is_tensor():
                return None
            return input.tensor_dtype()

        dtype: VariableInfo = self.specialize_dtype(output, inputs, [get_dtype(input) for input in inputs], produced, vars)

        def get_device(input: VariableInfo) -> Optional[VariableInfo]:
            if input.is_tensor():
                return input.tensor_device()
            elif issubclass(input.const_type(), (int,float)):
                return vars.constant(name='#cpu_device', origin=Origin.CONST_PROP,value=torch.device("cpu"), produced_by=self.node, produced=produced)
            return None

        device: VariableInfo = self.specialize_device(output, [get_device(input) for input in inputs], produced, vars)

        def get_shape(input: VariableInfo) -> Optional[VariableInfo]:
            #print(f"get_shape for {input} is_tensor {input.is_tensor()}")
            if input.is_tensor():
                return input.tensor_shape()
            elif issubclass(input.const_type(), (int,float)):
                return input.owner.constant(name="#scalar_shape", origin=Origin.CONST_PROP, value=[],produced_by=input.produced_by,produced=input.produced)
            return None

        input_shapes = [get_shape(input) for input in inputs]

        shapes: VariableInfo = self.specialize_shapes(output, inputs, input_shapes, produced, vars)

        return (dtype, shapes, device)

    def infer_variable_info(self, produced: int, output: Value, inputs: List[VariableInfo], vars: Variables) -> VariableInfo:
        """
        Infers the variable info from the output spec.
        """
        jit_type = output.type()
        tp = _jit_type_to_type(jit_type)

        if jit_type.kind() == "TensorType":
            dtype,shape,device = self.specialize_tensor(output, inputs, produced, vars)
            #print("dtype", dtype)
            #print("shape", shape)
            #print("device", device)
            info = vars.tensor(name=output.debugName(), origin=Origin.LOCAL, produced_by=self.node, produced=produced,
                               dtype=dtype, shape=shape, device=device)
        elif jit_type.kind() == "ListType" or jit_type.kind() == "TupleType":
            elinfo,elsinfo,elen = self.specialize_sequence(produced, output, inputs, vars)
            #print("infer_variable_info for list")
            #print("inputs", inputs)
            #print("elinfo", elinfo)
            #print("elsinfo", elsinfo)
            #print("elen", elen)
            if elsinfo is None:
                assert elinfo is not None
                info = vars.homogeneous_sequence(name=output.debugName(), origin=Origin.LOCAL, produced_by=self.node,
                                                            produced=produced,tp=tp,values=elinfo,length=elen)
            else:
                info = vars.inhomogeneous_sequence(name=output.debugName(), origin=Origin.LOCAL, produced_by=self.node,
                                                            produced=produced,tp=tp,values=elsinfo)
        else:
            info = vars.local(name=output.debugName(), tp=tp, origin=Origin.LOCAL, produced_by=self.node, produced=produced)
        return info

    def specialize_sequence(self, produced: int, output: Value, inputs: List[VariableInfo], vars: Variables) -> Tuple[Optional[VariableInfo], Optional[List[VariableInfo]], VariableInfo]:
        """
        Return the sequence attributes.  Looks at the output to see what we can learn.
        """
        jtp = output.type()

        if isinstance(jtp, torch.ListType):
            el = jtp.getElementType()
            eltp = _jit_type_to_type(el)
            elinfo = vars.local(name=output.debugName() + ".#el", origin=Origin.CONST_PROP, tp=eltp,
                                        produced_by=self.node, produced=produced)
            len = vars.local(name=output.debugName() + ".#length", origin=Origin.CONST_PROP, tp=int,
                                        produced_by=self.node, produced=produced)
            return elinfo,None,len

        elif isinstance(jtp, torch.TupleType):
            els = jtp.elements()
            raise RuntimeError("TODO: specialize_sequence TupleType")
        else:
            raise RuntimeError("Unknown sequence type to specialize")

    def fallback_const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        Const prop for when inputs are not all known.  We do out best from the prototype
        and annotations.
        """
        if self.node.outputsSize() == 0:
            return tuple()
        elif self.node.outputsSize() == 1:
            return self.infer_variable_info(produced, self.node.outputsAt(0), inputs, vars)
        else:
            return tuple(self.infer_variable_info(produced, output, inputs, vars) for output in self.node.outputs())

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
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
                result = vars.constant(name=info.debugName(), origin=Origin.CONST_PROP, value=result,
                                    produced_by=self.node, produced=produced)
            else:
                assert len(result) == self.node.outputsSize
                assert isinstance(result, tuple)
                result = tuple(VariableInfo.constant(name=info.debugName(), origin=Origin.CONST_PROP, value=val,
                                                produced_by=self.node, produced=produced) for val,info in zip(result,self.node.outputs()))
        else:
            result = self.fallback_const_prop(produced, inputs, vars)

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
    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        assert input_shape[0] is not None
        return input_shape[0]

@torch_operator('aten::linear')
class LinearOperator(TorchBuiltinOperator):
    """
    Linear ax + b operator.
    """
    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        i_shape,w_shape,b_shape = input_shape
        if i_shape is None or w_shape is None:
            return None
        
        #print("i_shape", i_shape)
        #print("w_shape", w_shape)

        #mul_shape = _matmul_shape(i_shape,w_shape)
        assert i_shape is not None
        assert w_shape is not None
        b,m,n1=i_shape
        o,n2=w_shape
        #print("n1 = ", n1)
        #print("n2 = ", n2)
        vars.alias(n1, n2)
        #assert n1==n2
        return vars.tensor_shape(name=output.debugName()+"#shape", origin=Origin.CONST_PROP, dims=[b,m,o], produced=produced, produced_by=self.node)

@torch_operator('aten::cat')
class LinearOperator(TorchBuiltinOperator):
    """
    Concatenation operator.  Joins tensors together.
    """
    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        input,dim = inputs

        if dim.is_const():
            d = dim.const_value()

            if False:  # later, when we can do expressions
                dim_width = vars.constant(name=output.debugName()+".#accum", origin=Origin.CONST_PROP, value=0, produced_by=self.node, produced=produced)
                dims: List[List[VariableInfo]] = []

                # First extract the shapes, and accumulate the given variable
                for chunk in input.sequence_chunks():
                    if chunk.is_sequence_chunk():
                        shape = chunk.sequence_chunk_schema().tensor_shape()
                        dim_width += shape[d] * chunk.sequence_length()
                    else:
                        shape = chunk.tensor_shape()
                        dim_width += shape[d]
                    if len(dims) == 0:
                        dims = [[] for _ in range(len(shape))]
                    assert len(shape) == len(dims)
                    for i in range(len(shape)):
                        dims[i].append(shape[i])
            else:
                dim_width: int = 0
                all_const = True
                dims: List[List[VariableInfo]] = []

                # First extract the shapes, and accumulate the given variable
                for chunk in input.sequence_chunks():
                    if chunk.is_sequence_chunk():
                        shape = chunk.sequence_chunk_schema().tensor_shape()
                        this_dim = shape[d]
                        if this_dim.is_const() and chunk.sequence_length_is_const():
                            dim_width += this_dim.const_value() * chunk.sequence_const_length()
                        else:
                            all_const = False
                    else:
                        shape = chunk.tensor_shape()
                        this_dim = shape[d]
                        if this_dim.is_const():
                            dim_width += this_dim.const_value()
                        else:
                            all_const = False

                    if d < 0:
                        d = len(shape) + d
                    assert d >= 0

                    if len(dims) == 0:
                        dims = [[] for _ in range(len(shape))]
                    assert len(shape) == len(dims)
                    for i in range(len(shape)):
                        dims[i].append(shape[i])

                result: List[VariableInfo] = []

                for i in range(len(dims)):
                    if i == d:
                        if all_const:
                            print("all const")
                            result.append(vars.constant(name=output.debugName()+".#dim"+str(i), origin=Origin.CONST_PROP, value=dim_width, produced_by=self.node, produced=produced))
                        else:
                            print("not all const")
                            result.append(vars.local(name=output.debugName()+".#dim"+str(i), origin=Origin.CONST_PROP, tp=int, produced_by=self.node, produced=produced))
                    else:
                        # All of these dimensions must be equal
                        print("wrong dim")
                        result.append(vars.alias(*dims[i]))

                    print("  doing dim", i, "(looking for", d, i==d, ") res", result[-1], "dim_width", dim_width)

                print("dim", dim)
                print("d", d)
                print("dims", dims)
                print("result", result)

                shape_res = vars.tensor_shape(name=output.debugName(), origin=Origin.CONST_PROP, dims=result, produced_by=self.node, produced=produced)
                print(shape_res)
                return shape_res

            raise NotImplementedError()
        else:
            raise NotImplementedError()

        return vars.tensor_shape(name=output.debugName()+"#shape", origin=Origin.CONST_PROP, dims=[b,m,o], produced=produced, produced_by=self.node)

    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        input,_ = inputs
        assert input.is_sequence()

        # dtype must all be the same
        dtypes: List[VariableInfo] = []

        # First extract the dtypes
        for chunk in input.sequence_chunks():
            if chunk.is_sequence_chunk():
                dtype = chunk.sequence_chunk_schema().tensor_dtype()
            else:
                dtype = chunk.tensor_dtype()
            dtypes.append(dtype)

        # And now unify them as they all must have the same value
        vars.unify(*dtypes)

        return dtypes[0]

PythonBuiltinBinarySpecialization = Callable[['PythonBuiltinBinaryOperator',int,VariableInfo,VariableInfo,Variables],VariableInfo]

class PythonBuiltinBinaryOperator(TorchBuiltinOperator):
    """
    This is the base class for operators which are builtin to Python for non-Tensor data
    types, but are also overridden by PyTorch for Tensors.
    """

    python_op: Callable[[Any, Any], Any]
    op_str: str
    builtin_cases: List[Tuple[type, type, type, Optional[PythonBuiltinBinarySpecialization]]]

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation], python_op: Callable[[Any,Any], Any], op_str: str):
        super().__init__(node, ops)
        self.python_op = python_op
        self.op_str = op_str
        self.builtin_cases = [
            (float, float, float, None),
            (float, int,   float, None),
            (int,   float, float, None),
            (int,   int,   int,   None),
        ]

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        # For tensor operations only.  Does the broadcasting.
        a_shape,b_shape = input_shape
        #print(f"specialize_shape: {a_shape} {self.op_str} {b_shape}")
        assert a_shape is not None
        assert b_shape is not None
        return broadcast_shapes(a_shape, b_shape, name=output.debugName(), produced_by=self.node, produced=produced,vars=vars)

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        Constant propagation needs to handle both Tensor types (by calling aten) and Python types (which use the
        default operators).
        """
        a = inputs[0]
        b = inputs[1]
        a_type,b_type = a.const_type(),b.const_type()

        # Unknown types?  Nothing to do, we just have a generic variable
        if a_type is None or b_type is None:
            return super().const_prop(produced, inputs, vars)

        # Tensor operation?  Existing implementation works fine
        if issubclass(a_type, torch.Tensor) or issubclass(b_type, torch.Tensor):
            return super().const_prop(produced, inputs, vars)

        if len(inputs) != 2:
            raise RuntimeError(f"const_prop operator {self.op_str}: TODO: more than two arguments")

        # Constant operation?  Use the Python builtin logic
        if a.is_const() and b.is_const():
            result = self.python_op(a.const_value(), b.const_value())
            return vars.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP, value=result,
                                         produced_by=self.node,produced=produced)

        # Otherwise it's complicated... we need to emulate all of the Python inbuilt cases
        def match_case(case: Tuple[type, type, type, Optional[PythonBuiltinBinarySpecialization]]) -> Optional[VariableInfo]:
            a_type,b_type,res_type,specialization = case
            if specialization:
                return specialization(self, produced, a, b, vars)
            assert a.const_type() is not None
            assert b.const_type() is not None
            if issubclass(a.const_type(), a_type) and issubclass(b.const_type(), b_type):
                return vars.local(name=self.node.outputsAt(0).debugName(), origin=Origin.LOCAL,
                                          tp=res_type,produced_by=self.node,produced=produced)
            return None

        for case in self.builtin_cases:
            res = match_case(case)
            if res is not None:
                return res

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
            (str, int, str, None),
            (list, int, list, None),       
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

    @staticmethod
    def handle_seq_add(self_: PythonBuiltinBinaryOperator, produced: int, arg1: VariableInfo, arg2: VariableInfo, vars: Variables) -> VariableInfo:
        assert arg1.is_sequence()
        assert arg2.is_sequence()
        name=self_.node.outputsAt(0).debugName()

        if arg1.sequence_length_is_const() and arg2.sequence_length_is_const():
            res = list(arg1)
            res.extend(arg2)
            return vars.inhomogeneous_sequence(name=name, origin=Origin.CONST_PROP, tp=arg1.const_type(), produced_by=self_.node, produced=produced, values=res)

        raise NotImplementedError()

    def __init__(self, node: Node, ops: Optional[List[Operation]|Operation] = None):
        super().__init__(node, ops, operator.add, "+")
        self.builtin_cases.extend([            
            (str, str, str, None),
            (list, list, list, AdditionOperator.handle_seq_add),       
            (tuple, tuple, tuple, None),       
        ])

@torch_operator('aten::add', 3)
class ScaledAdditionOperator(TorchBuiltinOperator):
    """
    Linear a + alpha*b operator.
    """

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        a_shape,b_shape,_ = input_shape
        assert a_shape is not None
        assert b_shape is not None
        return broadcast_shapes(a_shape, b_shape, name=output.debugName(),produced=produced,produced_by=self.node,vars=vars)

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

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        # For tensor operations only.  Does the broadcasting.
        a_shape, = input_shape
        assert a_shape is not None
        return a_shape

    def const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        """
        Constant propagation needs to handle both Tensor types (by calling aten) and Python types (which use the
        default operators).
        """
        a, = inputs
        a_type = a.const_type()

        # Unknown types?  Nothing to do, we just have a generic variable
        if a_type is None:
            return super().const_prop(produced, inputs, vars)

        # Tensor operation?  Existing implementation works fine
        if issubclass(a_type, torch.Tensor):
            return super().const_prop(produced, inputs, vars)

        if len(inputs) != 1:
            raise RuntimeError(f"const_prop operator {self.op_str}: TODO: more than one argument")

        # Constant operation?  Use the Python builtin logic
        if a.is_const():
            result = self.python_op(a.const_value())
            return VariableInfo.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.CONST_PROP, value=result,
                                         produced_by=self.node,produced=produced)

        # Otherwise it's complicated... we need to emulate all of the Python inbuilt cases
        def match_case(case: Tuple[type, type]) -> Optional[type]:
            a_type,res_type = case
            assert a.const_type() is not None
            if issubclass(a.const_type(), a_type):
                return res_type
            return None

        for case in self.builtin_cases:
            res = match_case(case)
            if res is not None:
                return VariableInfo.local(name=self.node.outputsAt(0).debugName(), origin=Origin.LOCAL,
                                          tp=res,produced_by=self.node,produced=produced)

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

def _matmul_shape(a_shape: VariableInfo, b_shape: VariableInfo, name: str, produced_by: Optional[Node|Graph], produced: int, vars: Variables) -> VariableInfo:
    """
    Returns the output shape of a matrix multiply between tensors of the two shapes,
    including broadcasting.
    """
    la = len(a_shape)
    lb = len(b_shape)

    # From https://pytorch.org/docs/stable/generated/torch.matmul.html

    # The behavior depends on the dimensionality of the tensors as follows:

    def result(dims: List[VariableInfo]) -> VariableInfo:
        return vars.tensor_shape(name="#scalar_shape", origin=Origin.CONST_PROP, dims=dims, produced_by=produced_by, produced=produced)

    #    If both tensors are 1-dimensional, the dot product (scalar) is returned.
    if la == 1 and lb == 1:
        return result([])
    
    #    If both arguments are 2-dimensional, the matrix-matrix product is returned.
    elif la == 2 and lb == 2:
        m,n1 = a_shape
        n2,o = b_shape
        vars.unify(n1, n2)
        return result([m,o])

    #    If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1
    #    is prepended to its dimension for the purpose of the matrix multiply. After the matrix
    #    multiply, the prepended dimension is removed.
    elif la == 1 and lb == 2:
        n1 = a_shape[0]
        n2,o = b_shape
        vars.unify(n1, n2)
        return result([o])

    #    If the first argument is 2-dimensional and the second argument is 1-dimensional, the
    #    matrix-vector product is returned.
    elif la == 2 and lb == 1:
        m,n1 = a_shape
        n2 = b_shape[0]
        vars.unify(n1, n2)
        return result([m])

    #    If both arguments are at least 1-dimensional and at least one argument is N-dimensional
    #    (where N > 2), then a batched matrix multiply is returned.

    else:
        one_range = vars.constant(name="#one_dim", origin=Origin.CONST_PROP, value=1, produced_by=produced_by, produced=produced)
        #    If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose
        #    of the batched matrix multiply and removed after.
        if la == 1:
            m1,n1,b1 = one_range,a_shape[0],[]
        else:
            m1,n1,b1 = a_shape[-2],a_shape[-1],list(a_shape[0:-2])

        #    If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of
        #    the batched matrix multiple and removed after. 
        if lb == 1:
            m2,n2,b2 = b_shape[0],one_range,[]
        else:
            m2,n2,b2 = b_shape[-2],b_shape[-1],list(b_shape[0:-2])

        #print(f"a: {b1} x {m1} x{n1}")
        #print(f"b: {b2} x {m2} x{n2}")

        n = broadcast_dim(n1, m2, name, vars)

        #print(f"out: {b2} x {m1} x {n}")

        #    The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be
        #    broadcastable). For example, if input is a (j×1×n×n)(j×1×n×n) tensor and other is a
        #    (k×n×n)(k×n×n) tensor, out will be a (j×k×n×n)(j×k×n×n) tensor.

        #    Note that the broadcasting logic only looks at the batch dimensions when determining if the
        #    inputs are broadcastable, and not the matrix dimensions. For example, if input is a
        #    (j×1×n×m)(j×1×n×m) tensor and other is a (k×m×p)(k×m×p) tensor, these inputs are valid for
        #    broadcasting even though the final two dimensions (i.e. the matrix dimensions) are different.
        #    out will be a (j×k×n×p)(j×k×n×p) tensor.

        batch_dims = broadcast_shapes(result(b1), result(b2), name=name, produced_by=produced_by, produced=produced, vars=vars)
        dims = list(batch_dims)
        dims.extend([m1,n2])
        #print(f"returning {dims}")
        return result(dims)

@torch_operator('aten::matmul')
class MatMulOperator(TorchBuiltinOperator):
    """
    Linear ax operator.
    """
    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        #print("input_shape", input_shape)
        a_shape,b_shape = input_shape
        assert a_shape is not None
        assert b_shape is not None
        return _matmul_shape(a_shape, b_shape, output.debugName(), self.node, produced, vars)


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

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:

        #print("input_shape", input_shapes[0])
        #print("new_shape", inputs[1])

        new_shape = inputs[1]
        return new_shape

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        raise NotImplementedError()
        assert len(inputs) == 2
        if inputs[1].is_const():
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

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:

        old_shape = input_shapes[0]
        new_order = inputs[1]

        #print("old_shape", old_shape)
        #print("new_order", new_order)

        if old_shape is not None and new_order.is_const():
            new_order_val = new_order.const_value()
            assert isinstance(new_order_val, list) or isinstance(new_order_val, tuple)
            new_shape = [old_shape[v] for v in new_order_val]
            return vars.tensor_shape(name=output.debugName()+".#shape", origin=Origin.CONST_PROP, dims=new_shape, produced_by=self.node, produced=produced)

        else:
            # Non-const order; shape propagation terminates
            raise NotImplementedError()


@torch_operator('aten::transpose')
class AtenTransposeOperator(TorchBuiltinOperator):
    """
    Handles the "transpose" operator.
    """

    def __init__(self, node: Node):
        super().__init__(node, Tensor.transpose)

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:

        old_shape = input_shapes[0]
        if old_shape is None or not inputs[1].is_const() or not inputs[2].is_const():
            raise NotImplementedError()

        assert isinstance(inputs[1].const_value(), int)
        assert isinstance(inputs[2].const_value(), int)
        dim1: int = inputs[1].const_value()
        dim2: int = inputs[2].const_value()

        l = old_shape.sequence_const_length()

        new_order = list([x for x in range(l)])
        tmp = new_order[dim1]
        new_order[dim1] = new_order[dim2]
        new_order[dim2] = tmp

        #"old_shape", old_shape)
        #print("new_order", new_order)

        return vars.tensor_shape(name=output.debugName()+".#shape", origin=Origin.CONST_PROP, dims=[old_shape[v] for v in new_order], produced_by=self.node, produced=produced)

@torch_operator('aten::contiguous')
class AtenContiguousOperator(TorchBuiltinOperator):
    """
    Handles the "contiguous" operation, which copies a tensor if necessary so that
    its data is contiguous.
    """

    def __init__(self, node: Node):
        super().__init__(node, Tensor.contiguous)

    def is_const(self) -> bool: return True

    def fallback_const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        input,memfmt = inputs
        if memfmt.is_const() and (memfmt.const_value() is None or memfmt.const_value() in {0,1}):
            return input
        # TODO: does a different memory format matter
        # Contiguous = 0
        # Preserve = 1
        # ChannelsLast = 2
        # ChannelsLast3d = 3
        return input
        

@torch_operator('aten::to')
class AtenToOperator(TorchBuiltinOperator):
    """
    Handles the "to" operator.  Copies a tensor to a different dtype or device.
    """

    def __init__(self, node: Node):
        super().__init__(node, Tensor.to)

    def is_const(self) -> bool: return True

    def fallback_const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        from aten import int_to_dtype, dtype_to_int, int_to_memory_format
        for input in inputs:
            print(input.name, input)

        name=self.node.outputsAt(0).debugName()

        tensor = inputs[0]
        shape = tensor.tensor_shape()
        dtype: VariableInfo = inputs[0].tensor_dtype()
        device: VariableInfo = inputs[0].tensor_device()
        assert inputs[1].const_type() is not None
        if issubclass(inputs[1].const_type(), torch.Tensor):
            # First variant: tensor.to(tensor, ...)
            dtype = inputs[1].tensor_dtype()
            device = inputs[1].tensor_device()
            offset = 2
        else:
            if inputs[1].is_const():
                #print("to const value", inputs[1].const_value())
                if isinstance(inputs[1].const_value(), torch.device):
                    device = inputs[1]
                else:
                    # We get passed an int, we need to convert to a datatype
                    assert isinstance(inputs[1].const_value(), int)
                    dtype = vars.constant(name=name, origin=Origin.CONST_PROP, value=int_to_dtype(inputs[1].const_value()),
                                            produced_by=self.node, produced=produced)
            offset = 2
            if device is None and inputs[2].const_type() != bool:
                device = inputs[2]
                offset = 3

        # Rest are non_blocking, copy and memory_format which don't affect output tensor

        return vars.tensor(name=name, origin=Origin.CONST_PROP, dtype=dtype, device=device,
                                   shape=shape, produced_by=self.node,produced=produced)

@torch_operator('aten::slice')
class AtenSliceOperator(TorchBuiltinOperator):
    """
    Handles the "slice" operator.  Manipulates only strides, offsets and sizes.
    """

    def __init__(self, node: Node):
        super().__init__(node, torch.slice_copy)

    def specialize_shapes(self, output: Value, inputs: List[VariableInfo], input_shapes: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:

        #for input in inputs:
        #    print(f"input {input.name} {input.const_type()} {input}")

        input,dim_number,start,end,step = inputs

        if not dim_number.is_const():
            raise NotImplementedError()

        n = dim_number.const_value()
        assert isinstance(n, int)

        input_shape = input_shapes[0]
        assert input_shape is not None

        input_dims = len(input_shape)

        sliced_dim: VariableInfo = input_shape[n]

        #print("sliced_dim: ", sliced_dim, "slice:", start, ":", end, ":", step)
        if sliced_dim.is_const():
            if start.is_const() and end.is_const() and step.is_const():
                raise NotImplementedError(f"slice of known dimension {sliced_dim} {start.const_value()}:{end.const_value()}:{step.const_value()}")

            output_dims: List[VariableInfo] = list(input_shape)

            # If step is 1 and start is None, then the length is equal to the end
            if start.is_none() and step.is_const() and step.const_value() == 1:
                output_dims[n] = end
            else:
                output_dims[n] = vars.local(name=output.debugName()+".#el"+str(n), origin=Origin.CONST_PROP, tp=int, produced_by=self.node, produced=produced)

            return vars.tensor_shape(name=output.debugName(), origin=Origin.CONST_PROP, dims=output_dims, produced_by=self.node, produced=produced)

        print("slice_copy: input_shapes", input_shapes)
        raise NotImplementedError()

    def fallback_const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        result = super().fallback_const_prop(produced, inputs, vars)
        assert isinstance(result, VariableInfo)
        input = inputs[0]

        if input.const_type() is not None and issubclass(input.const_type(), torch.Tensor):
            return result

        print("input", input)

        if not result.is_const() and input.is_sequence():
            input = input.as_sequence()
            # Specializing sequence... slicing a list with constant indices returns a slice of the VariableInfos
            start = inputs[1].typed_default_value(0)
            end = inputs[2].typed_const_value(int)
            step = inputs[3].typed_default_value(1)

            #print("start", start, "end", end, "step", step)

            if start is None or end is None or step is None:
                return result

            old_els = list(input)
            new_els = list(old_els[start:end:step])

            #print("new_els", new_els)

            all_const = all(map(lambda x: x.is_const(), new_els))
            
            if all_const:
                const_values = list([x.const_value() for x in new_els])
                return vars.constant(name=result.name, origin=Origin.CONST_PROP, value=const_values, produced_by=self.node, produced=produced)
            else:
                return vars.inhomogeneous_sequence(name=result.name, origin=Origin.CONST_PROP, tp=list, values=new_els, produced_by=self.node, produced=produced)

        raise NotImplementedError(f"TODO slice fallback_const_prop sequence {result}")

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
    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[torch.dtype]], produced: int, vars: Variables) -> Optional[torch.dtype]:
        weights_dtype,_,_,_,_ = input_dtype
        return weights_dtype

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[VariableInfo], produced: int, vars: Variables) -> VariableInfo:
        #print("inputs", inputs)
        #print("input_shape", input_shape)

        weights_shape,indexes_shape,_,_,_ = input_shape

        #print("weights_shape", weights_shape)
        #print("indexes_shape", indexes_shape)

        _,w_dim = weights_shape
        i_batch,i_len = indexes_shape

        vars = weights_shape.owner
        return vars.tensor_shape(name=output.debugName(), origin=Origin.CONST_PROP, produced_by=self.node, produced=produced, dims=[i_batch, i_len, w_dim])

        return TensorShape([i_batch, i_len, w_dim])

@torch_operator('aten::dropout')
class DropoutOperator(TorchBuiltinOperator):
    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        i_shape,_,_ = input_shape
        return i_shape

@torch_operator('prim::device')
class DeviceOperator(TorchBuiltinOperator):
    """device() operator that returns the device of a tensor"""

    def __init__(self, node: Node):
        super().__init__(node, Tensor.get_device)

    def is_const(self) -> bool: return True  # later... different per device

    def fallback_const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        return inputs[0].tensor_device()

def get_dtype(t: Tensor) -> int: #torch.dtype:
    from aten import dtype_to_int
    return dtype_to_int(t.dtype)

@torch_operator('prim::dtype')
class DataTypeOperator(TorchBuiltinOperator):
    """dtype() operator that returns the data type of a tensor"""

    def __init__(self, node: Node):
        super().__init__(node, get_dtype)

    def is_const(self) -> bool: return True  # later... different per device

    def fallback_const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        from aten import dtype_to_int
        if inputs[0].tensor_dtype_is_const():
            dt = inputs[0].tensor_const_dtype()
            assert dt is not None
            return vars.constant(name=self.node.outputsAt(0).debugName(), origin=Origin.LOCAL, value=dtype_to_int(dt),
                                         produced_by=self.node, produced=produced)

        # TODO: convert to int?
        return inputs[0].tensor_dtype()

@torch_operator('aten::scalar_tensor')
class ScalarTensorOperator(TorchBuiltinOperator):
    """create a tensor from a scalar"""

    def __init__(self, node: Node):
        super().__init__(node, torch.scalar_tensor)

    def is_const(self) -> bool: return True

    def fallback_const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
        raise RuntimeError("scalar_tensor fallback_const_prop")

    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[torch.dtype]]) -> Optional[torch.dtype]:
        raise NotImplementedError("scalar_tensor specialize_dtype")

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        raise NotImplementedError("scalar_tensor specialize_shape")

def broadcast_from(dim1: VariableInfo, dim2: VariableInfo, name: str, vars: Variables) -> VariableInfo:
    raise NotImplementedError()

def broadcast_dim(dim1: VariableInfo, dim2: VariableInfo, name: str, vars: Variables) -> VariableInfo:
    """
    Return the dimension of the broadcast two dimensions, or throw if they
    are not compatible.
    """

    #print("broadcast_dim", name, dim1, dim2)

    if dim1.is_const() and dim2.is_const():
        val1 = dim1.const_value()
        assert isinstance(val1, int)
        val2 = dim2.const_value()
        assert isinstance(val2, int)

        if val1 == 1:
            return dim2
        if val2 == 1:
            return dim1

        vars.unify(dim1, dim2)

        if val1 > val2:
            return dim1
        else:
            return dim2
    elif dim1.is_const() and dim1.const_value() == 1:
        return dim2
    elif dim2.is_const() and dim2.const_value() == 1:
        return dim1

    # Typically, non-constant dimensions will not be 1 and so we won't get strange broadcasting behavior

    #print("dim1", dim1)
    #print("dim2", dim2)

    # if dim1 is 1, then dim2 can be whatever it wants
    # if dim2 is 1, then dim1 can be whatever it wants
    # here we assume neither
    #vars.condition(dim1 != 1)
    #vars.condition(dim2 != 1)
    vars.unify(dim1, dim2)

    return dim1

def broadcast_shape2(shape1: List[int], shape2: List[int]) -> List[int]:
    max_len = max(len(shape1), len(shape2))
    #print("max_len = ", max_len)
    dims = [1] * max_len


    def do_shape(sh: VariableInfo):
        newsh = [one_dim for _ in range(max_len - len(sh))]
        newsh.extend(sh)

        #print("do_shape", sh, newsh)

        for outsh,insh in zip(dims,newsh):
            outsh = broadcast_dim(outsh, insh, name, vars)
    
    for sh in shapes:
        do_shape(sh)

    #print("broadcast_shape returning", dims)

    return vars.tensor_shape(name=name, origin=Origin.CONST_PROP, dims=dims, produced_by=produced_by, produced=produced)


def broadcast_shape(shapes: Sequence[VariableInfo], name: str, produced_by: Optional[Node|Graph], produced: int, vars: Variables) -> VariableInfo:
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

    max_len = max([len(s) for s in shapes])
    #print("max_len = ", max_len)
    one_dim = vars.constant(name="#dimpadding", origin=Origin.CONST_PROP, value=1, produced_by=produced_by, produced=produced)
    dims = [one_dim for _ in range(max_len)]

    def do_shape(sh: VariableInfo):
        newsh = [one_dim for _ in range(max_len - len(sh))]
        newsh.extend(sh)

        #print("do_shape", sh, newsh)

        for i,(outsh,insh) in enumerate(zip(dims,newsh)):
            dims[i] = broadcast_dim(outsh, insh, name, vars)
    
    for sh in shapes:
        do_shape(sh)

    #print("broadcast_shape returning", dims)

    return vars.tensor_shape(name=name, origin=Origin.CONST_PROP, dims=dims, produced_by=produced_by, produced=produced)

def broadcast_shapes(*shapes: VariableInfo, name: str, produced_by: Optional[Node|Graph], produced: int, vars: Variables) -> VariableInfo:
    """
    Return the shape of the arguments broadcast to the same shape.
    """
    return broadcast_shape(shapes, name, produced_by, produced, vars)

def broadcast_dtypes(*dtypes: Optional[VariableInfo], name: str, produced_by: Optional[Node|Graph], produced: int, vars: Variables) -> VariableInfo:
    """
    Return the dtype that covers the input dtypes.
    """

    if len(dtypes) == 0:
        raise RuntimeError("Can't broadcast no dtypes")

    result: List[torch.dtype] = []
    nonconst: List[VariableInfo] = []

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

    def accum(t: VariableInfo) -> None:
        assert t.const_type() == torch.dtype
        if t.is_const():
            dt = t.const_value()
            assert isinstance(dt, torch.dtype)
            p = precedence[dt]

            if len(result) == 0:
                result.append(dt)
            else:
                pr = precedence[result[0]]
                if pr < p:
                    result[0] = dt
        else:
            # Not constant
            nonconst.append(t)

    for t in dtypes:
        if t is None:
            continue
        accum(t)

    if len(result) == 0 or len(nonconst) > 0:
        # No constant result
        if len(result) == 0 and len(nonconst) == 1:
            return nonconst[0]
        else:
            if len(nonconst) > 0:
                print(f"TODO: constrain nonconst dtype")
            return vars.local(name=name,origin=Origin.CONST_PROP,tp=torch.dtype,produced_by=produced_by,produced=produced)
    else:
        return vars.constant(name=name,origin=Origin.CONST_PROP,value=result[0],produced_by=produced_by,produced=produced)

@torch_operator('aten::where')
class WhereOperator(TorchBuiltinOperator):
    """Select elements from one tensor based on a condition tensor"""

    def is_const(self) -> bool: return True

    #def fallback_const_prop(self, produced: int, inputs: List[VariableInfo], vars: Variables) -> Tuple[VariableInfo] | VariableInfo:
    #    for input in inputs:
    #        print(input.name, input.const_type(), input.const_dtype, input.tensor_shape, input.const_value())
    #    raise RuntimeError("where fallback_const_prop")

    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        _,left,right = input_dtype
        assert left is not None
        assert right is not None
        return broadcast_dtypes(left, right, name=output.debugName()+".#dtype",produced_by=self.node,produced=produced,vars=vars)

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        cond,left,right = input_shape
        assert cond is not None
        assert left is not None
        assert right is not None
        sh = broadcast_shapes(cond, left, right, name=output.debugName()+".#shape",produced_by=self.node,produced=produced,vars=vars)

        #print("cond", cond)
        #print("left", left)
        #print("right", right)
        #print("shape", sh)

        return sh

@torch_operator('aten::softmax')
class SoftMaxOperator(TorchBuiltinOperator):
    """Differentiable max function"""

    def is_const(self) -> bool: return True

    def specialize_dtype(self, output: Value, inputs: List[VariableInfo], input_dtype: List[VariableInfo], produced: int, vars: Variables) -> VariableInfo:
        input,dim,dtype = inputs
        if dtype.is_const() and dtype.const_value() is None:
            # Take the input dtype
            dtype,_,_ = input_dtype
            return dtype
        elif dtype.is_const():
            # Take the passed dtype
            return dtype
        else:
            # Can't determine dtype
            return vars.local(name=output.debugName()+".#dtype", origin=Origin.CONST_PROP, tp=torch.dtype, produced_by=self.node, produced=produced )

    def specialize_shape(self, output: Value, inputs: List[VariableInfo], input_shape: List[Optional[VariableInfo]], produced: int, vars: Variables) -> VariableInfo:
        shape,_,_ = input_shape
        assert shape is not None
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
    def do_node(produced: int, node: Node, indent: str):  # -> Tuple[Optional[Any|Tuple[Any]], bool]:
        try:
            #print(indent, "executing node", i, node)

            inputs: List[VariableInfo] = []
            for input in node.inputs():
                name = input.debugName()
                var = vars.get(name, i)
                inputs.append(var)

            op = get_torch_operator(node)

            #print("got op", op)

            #print("before const prop", op)
            #vars.dump_vars(indent)

            #print("const_prop is", op.const_prop)

            result = op.const_prop(produced, inputs, vars)

            #print("after const prop", op)
            #vars.dump_vars(indent)

            def add_output(info: Value, var: VariableInfo):
                is_constant_output = op.is_const() and var.is_const()
                var2 = var.renamed(info.debugName())
                if is_constant_output:
                    var2.origin = Origin.CONST_PROP
                else:
                    var2.origin = Origin.LOCAL

                #print("name before->after", var.name, var2.name)
                vars.add(var2)

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
        except Exception as e:
            print("exception executing node", node)
            vars.dump_vars(indent)
            #traceback.print_exception(e)
            raise


    for i,node in enumerate(graph.nodes()):
        do_node(i, node, '')

    return vars
