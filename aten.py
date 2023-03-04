import torch
import json
import yaml
import numpy as np
from collections import namedtuple
from typing import Any, List, Dict, Optional, Sequence, Tuple
from torch import Tensor
from utils import _short_dtype

ATenValue = torch.Value

def dtype_to_numpy(t):
    return torch.empty(size=[], dtype=t).numpy().dtype

def numpy_to_dtype(t):
    return torch.from_numpy(np.empty([], dtype=t)).dtype

Loader: Any

try:
    # use faster C loader if available
    #from yaml import CLoader as Loader
    from yaml import CLoader as CLoader
    Loader=CLoader
except ImportError:
    from yaml import Loader as NativeLoader
    Loader=NativeLoader

def parse_native_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=Loader)

# Declarations.yaml comes from pytorch
# cd <path_to_pytorch_source>/aten/src/ATen
# python ./gen.py native/native_functions.yaml  -d ./output-native-functions
# file is at output-native-functions/Declarations.yaml
#aten_declarations = parse_native_yaml('Declarations.yaml')


def get_aten_declarations(cache=[]):
    # Get the declarations from pytorch so we can understand the
    # inputs
    if cache == []:
        # TODO: ugly ugly hack hack self modifying code!
        # put this in a Makefile
        # it's done this way because the YAML parser can be horribly
        # slow, and the Python class loader is much faster
        try:
            import aten_declarations
            cache.append(aten_declarations.declarations)
        except:
            declarations_raw = parse_native_yaml('Declarations.yaml')
            declarations = {}
            for d in declarations_raw:
                declarations.setdefault(d['name'], []).append(d)
            with open('aten_declarations.py', 'w', encoding='utf-8') as f:
                f.write('declarations='+repr(declarations))
            cache.append(declarations)

    return cache[0]

class ATenOverrides(object):
    def add(t, v, o=None):
        #print(get_aten_declarations()['add'])
        #print(getattr(t.add, '__doc__'))
        #print(type(t), type(v), type(o))
        #print(t.shape, v.shape, o.shape if torch.is_tensor(o) else None)
        if o is None:
            return torch.add(t, v)
        else:
            return torch.add(t, o, v)

def find_aten_operator(overload, overrides = [ATenOverrides]):
    operator = overload['name']

    import torch

    mod = torch
    #submod = declarations[operator]['python_module']

    #if submod is not '':
    #    mod = getattr(torch, submod)

    mods = overrides + [torch._C._nn, torch._C._TensorBase, torch._C._VariableFunctions, torch, torch.nn, torch.nn.functional]
    op = None
    for mod in mods:
        if hasattr(mod, operator):
            #print('getting', operator, 'from', mod)
            op = getattr(mod, operator)
            break

    done = set()

    # This is a debugging function that will help us to find where we
    # should be looking for a function we can't find elsewhere.
    def find_function(where, path):
        try:
            if where in done:
                return
            done.add(where)

            for f in dir(where):
                if f.find('__') != -1:
                    continue
                #if f == operator:
                #    print('found in', where, 'at path', path)
                attr = getattr(where, f)
                find_function(attr, path + [f])
        except:
            return


    if op is None:
        find_function(torch, [])

    assert op is not None, "Could not find operator " + operator + " anywhere"
    #print('op is', op, 'in module', mod)

    return op, mod

def package_arg_for_aten(type, value, arginfo, input_name='', use_sample_inputs=False, as_variables=False):
    import torch

    #print('arg', arginfo, 'nargs', 't', type, 'v', str(value)[0:40])

    if type is None:
        return (None, None)

    def maybe_wrap_variable(var):
        if as_variables:
            #print('  --> returning as variable')
            return (torch.jit.Variable(var), input_name)
        else:
            return (var, None)

    if arginfo['dynamic_type'] in ['Tensor', 'TensorList']:
        if value is not None:
            return (torch.from_numpy(value), input_name)
        if not use_sample_inputs:
            raise AssertionError('Null tensor argument passed with use_sample_inputs=False')
        return maybe_wrap_variable(torch.from_numpy(np.zeros(dtype=type.dtype, shape=type.shape)))
    else:
        if value is None:
            return (value, None)
        elif len(value.shape) == 0:
            out = np.asscalar(value)
            if arginfo['type'] == 'bool':
                out = bool(out)
            return (out, None)
        else:
            return (value.tolist(), None)


def package_args_for_aten(overload, inputs, input_names, use_sample_inputs=False, as_variables=False):
    # Any TensorList args will be expanded into simple tensors, which
    # means that we may have more arguments than there are in the
    # declaration.  In that case, we need to copy the argument information
    # multiple times, but gather them back into a single array to be
    # passed onwards.
    args = overload['arguments']
    tensorlists = [i for i in range(0, len(args)) if args[i]['dynamic_type'] == 'TensorList']

    if len(tensorlists) >= 1:

        # More than one tensorlist is ambiguous, so we don't even try
        # to handle (unknown if this actually exists or not)
        assert len(tensorlists) == 1, "Cannot handle multiple tensorlist arguments for ATen operations"
        tl = tensorlists[0] # which arg number it's at
        num_tensors_in_list = len(inputs) - len(args) + 1
        #print('op', overload['name'], 'tl', tl, 'ntil', num_tensors_in_list)
        assert num_tensors_in_list > 0, "Cannot have tensorlist with negative or zero length"
        args = list(args)
        for i in range(0, num_tensors_in_list-1):
            args.insert(tl, args[tl])
        assert len(args) == len(inputs), "Wrong arguments for input"

    outputs = []
    tensorlist_arg = []
    var_names = []

    for argnum in range(0, len(inputs)):
        arg_info = args[argnum]
        type, value = inputs[argnum]
        input_name = '__i' + str(input_names[argnum])
        packaged_arg, var_name = package_arg_for_aten(type, value, arg_info, input_name, use_sample_inputs, as_variables)
        if var_name is not None:
            var_names.append(var_name)
        if arg_info['dynamic_type'] == 'TensorList':
            tensorlist_arg.append(packaged_arg)
        else:
            if len(tensorlist_arg) > 0:
                outputs.append(tensorlist_arg)
                tensorlist_arg = []
            outputs.append(packaged_arg)

    if len(tensorlist_arg) > 0:
        outputs.append(tensorlist_arg)

    assert len(outputs) == len(overload['arguments'])

    #print('outputs', outputs)
    #print('var_names', var_names)

    return (outputs, var_names)

def choose_overload_for_aten(operator: str,
                             inputs: List[ATenValue],
                             num_outputs: int,
                             attr: Dict[str,Any] = None,
                             use_sample_inputs=False,
                             as_variables=False,
                             symbolic_mode=False) \
                             -> Tuple[Any, Tuple[List[Any], List[int], List[str]]]:
    """
    Given the operator name, the set of inputs, attributes, and the number
    of outputs, choose the appropriate ATen declaration for the given
    arguments and return the arguments packaged in a way in which they can
    be passed to the ATen function.

    Where use_sample_inputs is True, an input which has no value but does
    have a type will be replaced with a concrete zero-filled value.  This
    is useful for tracing.

    Where as_variables is True, any Tensor inputs (sample or not) will be
    wrapped in a torch.jit.Variable.  Again, this is useful for tracing.

    Where symbolic_mode is True, inputs which are passed in with a name
    (either in inputs or as an ATenValue in as an ATenValue in attr) will
    be passed back out as an ATenValue, so that they can be called
    symbolically.

    The outputs is (overload, (args, arg_is_set, arg_names)) where:

    - overload is the ATen declaration of the function to be called
    - args is a list of the arguments it should be called with, each of
      which is wrapped in an ATenValue.  To pass them to ATen, the
      .value field should be extracted.
    - arg_is_set is a per-argument a list of values, with 0 meaning not
      set (it won't happen), 1 meaning set to the default and 2 meaning
      set explicitly.
    - arg_names is simply the names corresponding to the arguments.
    """
    if attr is None:
        attr = {}

    for i in inputs:
        if i is None:
            continue
        if not isinstance(i, ATenValue):
            raise AttributeError('ATen overload passed non ATenValue {}' \
                                 .format(i))
    
    declarations = get_aten_declarations()
    op_declaration = declarations[operator]
    #if len(op_declaration) == 1:
    #    return op_declaration[0]

    # We have multiple overloads for the given function
    #print('multiple overloads for operator', operator)

    # A function is viable if:
    # 1.  Every passed in input and attribute is viable for the function
    # 2.  Every non-optional input is covered for the function

    def is_viable(declaration):
        
        #print('trying declaration', declaration)
        
        args = declaration['arguments']

        if len(declaration['returns']) != num_outputs:
            #raise AttributeError('returns wrong number of results')
            return None
        
        tensorlists = [i for i in range(0, len(args)) if args[i]['dynamic_type'] == 'TensorList']
        num_optional_args = len([i for i in range(0, len(args)) if 'default' in args[i]])

        tl = None
        min_tensors_in_list = 0
        max_tensors_in_list = 0

        if len(tensorlists) >= 1:

            # More than one tensorlist is ambiguous, so we don't even try
            # to handle (unknown if this actually exists or not)
            # (It does exist, eg for LSTM, and we handle by requiring that
            # the arguments be explicitly packaged up)
            #assert len(tensorlists) == 1, "Cannot handle multiple tensorlist arguments for ATen operations"
            tl = tensorlists # which arg number it's at
            #print('op', overload['name'], 'tl', tl, 'ntil', num_tensors_in_list)
            min_tensors_in_list = len(inputs) + len(attr) - len(args)
            max_tensors_in_list = min_tensors_in_list + num_optional_args


        #print('testing declaration with', len(args), 'arguments,',
        #      len(inputs), 'inputs and', len(attr), 'attributes',
        #      'tl', tl, 'min_tensors', min_tensors_in_list, 'max_tensors',
        #      max_tensors_in_list)

        NOT_SET = 0
        DEFAULT = 1
        SET = 2

        def test_scenario(num_tensors_in_list):

            def get_default(arg):
                if 'default' not in arg:
                    return None
                # Defaults are parsed from C++, {} is basically a default
                # constructor in C++, but this is used for integer lists
                if arg['default'] == '{}':
                    return []
                return arg['default']

            arg_values = [get_default(arg) for arg in args]
            arg_is_set = [DEFAULT if 'default' in arg else NOT_SET for arg in args]
            arg_names = [None for arg in args]
        
            try:
                def set_arg(pos: int, value, tp: Optional[TensorType]=None, name: Optional[str]=None):
                    """
                    Set the argument at the given position to the given value.
                    """
                    arginfo = args[pos]
                    
                    #print('input', pos, args[pos]['name'], tp, value, arginfo['dynamic_type'], name)

                    # TODO: all args should be ATenValue
                    if isinstance(value, ATenValue):
                        name = value.name
                    
                    def verify_tensor(value):
                        #print('verify_tensor value', value, 'tp', tp, 'name', name)
                        if value is None and tp is not None:
                            if use_sample_inputs:
                                value = torch.from_numpy(np.zeros(dtype=tp.dtype, shape=tp.shape))
                            if as_variables:
                                return torch.jit.Variable(value)
                        if isinstance(value, np.ndarray):
                            value = torch.from_numpy(value)
                        return value
                
                    def verify_scalar(value, dtype):
                        #print('verify_scalar', value, dtype)
                        if tp is not None and len(tp.shape) != 0:
                            raise AttributeError('argument {} of type {} doesn\'t match dynamic type {}'.format(i, tp, dynamic_type))
                        if dtype is None:
                            dtype = lambda x: x
                        if isinstance(value, ATenValue):
                            if value.value is None:
                                return value
                            return dtype(value.value.item())
                        if isinstance(value, np.ndarray):
                            return dtype(np.asscalar(value))

                        #print('dtype', dtype, 'value', value)

                        return dtype(value)

                    def verify_constant_array(value, dtype):
                        if isinstance(value, list):
                            return value
                        if isinstance(value, np.ndarray):
                            return value.tolist()
                        if isinstance(value, torch.Tensor):
                            return verify_constant_array(value.numpy(), dtype)
                        if isinstance(value, ATenValue):
                            return verify_constant_array(value.value, dtype)
                        raise NotImplementedError('value {} of type {} is not a list'.format(value, value.__class__))
                        return value
                    
                    dynamic_type = arginfo['dynamic_type']
                    if dynamic_type == 'Tensor':
                        value = verify_tensor(value)
                    elif dynamic_type == 'Scalar':
                        value = verify_scalar(value, None)
                    elif dynamic_type == 'bool':
                        value = verify_scalar(value, bool)
                    elif dynamic_type == 'double':
                        value = verify_scalar(value, float)
                    elif dynamic_type == 'IntArrayRef':
                        value = verify_constant_array(value, int)
                    elif dynamic_type == 'int64_t':
                        value = verify_scalar(value, int)
                    elif dynamic_type == 'TensorList':
                        value = verify_tensor(value)
                    elif dynamic_type == 'ScalarType':
                        raise AttributeError('ScalarType values not supported')
                    else:
                        raise NotImplementedError('overload resolution for type {}'.format(arginfo['dynamic_type']))

                    # Re-wrap those that have a name with the ATenValue if we
                    # have asked for values out.
                    if name is not None \
                       and symbolic_mode \
                       and dynamic_type in { 'Tensor', 'Scalar', 'TensorList' } \
                       and not isinstance(value, ATenValue):
                        dtype = numpy_to_dtype(tp.dtype) if tp is not None else value.dtype
                        shape = tp.shape if tp is not None else value.shape
                        value = ATenValue(name=name, dtype=dtype, shape=shape, value=value)
                    
                    #print('set_arg at', pos, 'with value', value)
                    if tl is not None and pos in tl:
                        if len(tl) == 1:
                            if arg_is_set[pos] != SET:
                                arg_values[pos] = []
                            arg_values[pos].append(value)
                        else:
                            # For multiple tensorlists, the arguments
                            # have to be lists
                            if not isinstance(value, list) and not isinstance(value, tuple):
                                raise AttributeError('multiple tensorlists must be explicitly packed into lists')
                            if arg_is_set[pos] == SET:
                                raise AttributeError('tensorlist argument {} set twice'.format(pos))
                            arg_values[pos] = list(value)
                    else:
                        if arg_is_set[pos] == SET:
                            raise AttributeError('argument {} set twice'.format(pos))
                        arg_values[pos] = value

                    arg_is_set[pos] = SET

                def process_attribute(name, value):
                    pos = None
                    for i in range(0, len(args)):
                        if args[i]['name'] == name:
                            pos = i
                            break
                    if pos is None:
                        raise AttributeError('attribute {} not found'.format(name))

                    # We know the value; set it
                    set_arg(pos, value)

                for n,v in attr.items():
                    process_attribute(n,v)

                def process_input(i):
                    #print('input', inputs[i])
                    if inputs[i] is None:
                        name,type,value = (None,None,None)
                    else:
                        name,type,value = (inputs[i].name, inputs[i].to_type(), inputs[i].value)

                    argindex = i
                    if tl is None or len(tl) > 1 or i < tl[0]:
                        argindex = i
                    elif i < tl[0] + num_tensors_in_list:
                        argindex = tl[0]
                    else:
                        argindex = i - num_tensors_in_list
                        
                    set_arg(argindex, value, type, name)

                for i in range(0, len(inputs)):
                    process_input(i)

                #print('arg_is_set', arg_is_set)

                if not all([arg_is_set[i] != NOT_SET for i in range(len(args))]):
                    for i in range(len(args)):
                        if arg_is_set[i] == NOT_SET:
                            print('arg not set:', args[i])
                    raise AttributeError('Not all required parameters were set')

                return arg_values, arg_is_set, arg_names

            except AttributeError as e:
                print('overload failed: ' + str(e))
                return None
            except Exception as e:
                print('overload failed: ' + str(e))
                raise
            
        for nt in range(min_tensors_in_list, max_tensors_in_list+1):
            #print('trying scenario with', nt, 'tensors in list')
            out = test_scenario(nt)
            if out is not None:
                return out

        return None
            
    viable = [d for d in op_declaration if is_viable(d) is not None]

    if len(viable) == 0:
        #print('inputs', inputs)
        #print('ops', op_declaration)
        raise Exception('no viable overloads for operator ' + operator)
    elif len(viable) == 1:
        #print('viable', viable[0])
        return (viable[0], is_viable(viable[0]))
    else:
        return (viable[0], is_viable(viable[0]))
        for m in viable:
            print(json.dumps(m, indent=4))
        raise Exception('not finished: multiple viable overloads')
    
int_to_dtype_table = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.complex32,
    torch.complex64,
    torch.complex128,
    torch.bool,
    None, # quantized int8
    None, # quantized uint8
    None, # quantized int32
    torch.bfloat16,
]

dtype_to_int_table = {dt: i for dt,i in zip(int_to_dtype_table,range(100000)) if dt is not None}

def int_to_dtype(i: int) -> torch.dtype:
    res = int_to_dtype_table[i]
    assert res is not None
    return res

def optional_int_to_dtype(i: Optional[int]) -> Optional[torch.dtype]:
    if i is None:
        return None
    res = int_to_dtype_table[i]
    assert res is not None
    return res

def dtype_to_int(dt: torch.dtype) -> int:
    return dtype_to_int_table[dt]


int_to_memory_format_table = [
    torch.contiguous_format,
    torch.preserve_format,
    torch.channels_last,
    torch.channels_last_3d
]

memory_format_to_int_table = {dt: i for dt,i in zip(int_to_memory_format_table,range(100000)) if dt is not None}

def int_to_memory_format(i: int) -> torch.memory_format:
    res = int_to_memory_format_table[i]
    assert res is not None
    return res

def optional_int_to_memory_format(i: Optional[int]) -> Optional[torch.memory_format]:
    if i is None:
        return None
    res = int_to_memory_format_table[i]
    assert res is not None
    return res

def memory_format_to_int(dt: torch.memory_format) -> int:
    return memory_format_to_int_table[dt]
