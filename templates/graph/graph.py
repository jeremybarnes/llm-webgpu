import collections
#import onnx
#from onnx_utils import decode_type, decode_tensor, decode_attrs
from natsort import natsorted
import numpy
from typing import Callable, Any, Set, Optional, List, Mapping, Dict
from random import randrange

class TensorType(object):
    """
    Describes the type of a Tensor.
    """
    def __init__(self, dtype=None, shape=None, denotation=None, **kwargs):
        if dtype is not None and not isinstance(dtype, numpy.dtype):
            raise AttributeError('numpy.dtype or None expected but {} supplied'
                                 .format(dtype))
            
        self.dtype = dtype
        if shape is not None:
            shape = list(shape)
        self.shape = shape
        self.denotation = denotation
            
        for (arg,val) in kwargs.items():
            setattr(self, arg, val)

    def from_value(value):
        return TensorType(value.dtype, list(value.shape))

    def merge(self, other) -> None:
        def merged_value(key: str, first, second):
            if first is None:
                return second
            if second is None:
                return first
            if first == second:
                return first
            raise Exception('Attempt to merge ambiguous values in context {}: {} vs {}'.format(key, first, second))

        self.dtype = merged_value('dtype', self.dtype, other.dtype)
        self.shape = merged_value('shape', self.shape, other.shape)
        self.denotation = merged_value('denotation', self.denotation, other.denotation)
        
    def __str__(self):
        if self.denotation is None:
            return 'TensorType({} {})'.format(self.dtype, self.shape)
        else:
            return 'TensorType({} {}: {})'.format(self.dtype, self.shape, self.denotation)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.dtype == other.dtype and self.shape == other.shape and self.denotation == other.denotation
    
    def shape_fully_determined(self) -> bool:
        """
        Tells us if the shape is fully determined, in other words that each
        of the dimensions is known.
        """
        return self.shape is not None and all([isinstance(s, int) for s in self.shape])
    
    def to_onnx(self, t = None):
        if t is None:
            t = onnx.TypeProto()
        if self.dtype is not None:
            if self.dtype == numpy.int64:
                t.tensor_type.elem_type = onnx.TensorProto.INT64
            else:
                t.tensor_type.elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[self.dtype]

        #print('to_onnx shape', self.shape)
            
        if self.shape is not None and len(self.shape) > 0:
            def convert_dim(dim):
                #print('converting dim', dim, 'of type', type(dim))
                d = onnx.TensorShapeProto.Dimension()
                try:
                    d.dim_value = int(dim)
                except:
                    d.dim_param = dim
                return d
                    
            t.tensor_type.shape.dim.extend([convert_dim(d) for d in self.shape])

        return t

    def sizes(self):
        return self.shape
    
class Operation(object):
    def __init__(self, operation=None, namespace=None, **kwargs):
        if namespace is None and isinstance(operation, str) and operation.find('::'):
            namespace, operation = operation.split('::', 1)

        self.operation = operation
        self.namespace = namespace

        for (arg,val) in kwargs.items():
            setattr(self, arg, val)

    def __str__(self):
        if self.namespace is None:
            return 'Op({})'.format(self.operation)
        else:
            return 'Op({}::{})'.format(self.namespace, self.operation)
            
class Attribute(object):
    """
    Attribute of a graph.  This encodes an attribute with a value, modelling
    an ONNX attribute.
    """
    @staticmethod
    def _fix_namespace(name: Optional[str], namespace: str):
        if namespace == '' and name is not None and name.find('::') != -1:
            (namespace, name) = name.split('::', 1)
        return (namespace, name)
        
    def __init__(self, name: str=None, value: Any=None, namespace: str='', **kwargs):
        namespace, name = Attribute._fix_namespace(name, namespace)

        self.name = name
        self.value = value
        self.namespace = namespace

        for (arg,val) in kwargs.items():
            setattr(self, arg, val)
            
    def __lt__(self, other) -> bool:
        return (self.namespace, self.name, self.value) < (other.namespace, other.name, other.value) 

    def __str__(self) -> str:
        if self.namespace == '':
            return '{}:{}'.format(self.name, self.value)
        else:
            return '{}::{}={}'.format(self.namespace, self.name, self.value)

    def __repr__(self) -> str:
        return self.__str__()

    def to_tensorproto(self, v: numpy.ndarray, tp = None):
        if tp is None:
            tp = onnx.TensorProto()
        tp.dims.extend(list(v.shape))
        tp.data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[v.dtype]
        return tp
        
    def to_onnx(self):
        a = onnx.AttributeProto()
        a.name = self.name
        if isinstance(self.value, list):
            if isinstance(self.value[0], float):
                a.type = onnx.AttributeProto.FLOATS
                a.floats.extend(self.value)
            elif isinstance(self.value[0], int):
                a.type = onnx.AttributeProto.INTS
                a.ints.extend(self.value)
            elif isinstance(self.value[0], str):
                a.type = onnx.AttributeProto.STRINGS
                a.strings.extend([v.encode('utf-8') for v in self.value])
            elif isinstance(self.value[0], numpy.ndarray):
                a.type = onnx.AttributeProto.TENSORS
                a.tensors.extend([Attribute.to_tensorproto(v) for v in self.value])
            else:
                assert False, "Unknown attribute value type"

        else:
            if isinstance(self.value, float):
                a.type = onnx.AttributeProto.FLOAT
                a.f = self.value
            elif isinstance(self.value, int):
                a.type = onnx.AttributeProto.INT
                a.i = self.value
            elif isinstance(self.value, str):
                a.type = onnx.AttributeProto.STRING
                a.s = self.value.encode('utf-8')
            elif isinstance(self.value, numpy.ndarray):
                a.type = onnx.AttributeProto.TENSOR
                Attribute.to_tensorproto(self.value, a.t)
            else:
                assert False, "Unknown attribute value type"

        return a
    
class Attributes(object):
    def __init__(self, attributes: List[Attribute] = None):
        if attributes is None:
            attributes = []
        
        self.attributes = { (a.namespace, a.name): a.value for a in attributes }

    @staticmethod
    def _fix_namespace(name: str, namespace: str):
        if namespace == '' and name.find('::') != -1:
            (namespace, name) = name.split('::', 1)
        return (namespace, name)
        
    def get(self, name: str, default: Any = None, namespace: str = ''):
        return self.attributes.get(Attributes._fix_namespace(name, namespace),
                                   default)

    def set(self, name: str, value: Any, namespace: str = ''):
        self.attributes[Attributes._fix_namespace(name, namespace)] = value

    def remove(self, name: str, namespace: str=''):
        key = Attributes._fix_namespace(name, namespace)
        if key in self.attributes:
            del self.attributes[key]

    def merge(self, other) -> None:
        for key,value in other.attributes.items():
            if key not in self.attributes:
                self.attributes[key] = value
            elif value == self.attributes[key]:
                pass
            else:
                raise Exception('ambiguity in value of attribute {}: {} vs {} in merge'.format(key, value, self.attributes[key]))

    def __str__(self) -> str:
        return str(sorted([Attribute(nm,v,ns) for (ns,nm),v in self.attributes.items()]))

    def to_onnx(self):
        return [Attribute(nm,v,ns).to_onnx() for (ns,nm),v in self.attributes.items() if ns == 'onnx']
    
    
class WithAttributes(object):
    def __init__(self, attributes: Optional[Attributes] = None):
        if attributes is None:
            attributes = Attributes()
        if isinstance(attributes, dict):
            print('is dict')
            attributes = Attributes([Attribute(k,v) for k,v in attributes.items()])
        assert isinstance(attributes, Attributes), "Need to pass attributes to WithAttributes(); got " + str(attributes) + " of type " + str(type(attributes))
            
        self.attributes = attributes
    
    def set_attribute(self, name, value, namespace=''):
        self.attributes.set(name, value, namespace)
        
    def get_attribute(self, name: str, default: Any = None, namespace: str = '') -> Any:
        return self.attributes.get(name, default, namespace)

    def remove_attribute(self, name: str, namespace: str = ''):
        self.attributes.remove(name, namespace)
        
    
class Node(WithAttributes):
    def __init__(self,
                 operation: Operation=Operation(),
                 inputs=None,
                 outputs=None,
                 attr: Attributes=None,
                 **kwargs):
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []

        super().__init__(attr)

        self.operation = operation
        self.inputs = inputs
        self.outputs = outputs
        
        for arg, val in kwargs.items():
            setattr(self, arg, val)
        
    def set_onnx(self, onnx):
        self.onnx = onnx

    def to_onnx(self, name: str = None, onode: onnx.NodeProto =None):
        if onode is None:
            onode = onnx.NodeProto()

        if name is not None:
            onode.name = name

        onode.op_type = self.operation.operation
        onode.attribute.extend(self.attributes.to_onnx())
        onode.input.extend([i if i is not None else '' for i in self.inputs])
        onode.output.extend([o if o is not None else '' for o in self.outputs])

        return onode
        
    def set_type(self, type, **kwargs):
        self.type = TensorType(type, **kwargs)

    def initialize_from_onnx(self, node):
        result = Node()
        result.set_onnx(node)
        return result

    def from_onnx(node):
        inputs = [i if i != '' else None for i in node.input]
        outputs = [o if o != '' else None for o in node.output]
        attrs = Attributes([Attribute(a,v,'onnx') for (a,v) in decode_attrs(node.attribute).items()])
            
        return Node(Operation(node.op_type, 'onnx'), inputs, outputs, attrs, onnx=node)
    
    def get_attribute(self, name: str, defvalue: Any = None, namespace: str = '') -> List[Any]:
        return self.attributes.get(name, defvalue, namespace)

    def is_operation(self, op: str, namespace: Optional[str] = None):
        if namespace is None:
            (namespace, op) = op.split('::', 1)

        return self.operation.operation == op and self.operation.namespace == namespace

    # For compatibility with Torch ONNX nodes
    def kind(self):
        return self.operation.namespace + "::" + self.operation.operation
        
    def __str__(self):
        return "Node({},{},{},{})".format(self.operation, self.inputs, self.outputs, self.attributes)

    
class Value(WithAttributes):
    def __init__(self,
                 attributes: Attributes = None,
                 type: TensorType = None,
                 value: Any = None):
        super().__init__(attributes)

        self.type = type if type is not None else TensorType()
        self.value = value

    def set_parameter(self, value):
        self.value = value
        self.set_input(False)

    def set_input(self, is_input=True):
        self.set_attribute('input', is_input)
        
    def set_output(self, is_output=True):
        self.set_attribute('output', is_output)

    def set_type(self, type, **kwargs):
        if isinstance(type, TensorType):
            self.type = type
        else:
            self.type = TensorType(type, **kwargs)

    def summary_value(self):
        if self.value is None:
            return None

        sv = str(self.value).replace('\n', ' ')
        if len(sv) <= 40:
            return sv

        return sv[0:40] + '...'
            
    def __str__(self):
        if self.value is not None:
            return 'Value({},{}={})'.format(self.type, self.attributes, self.summary_value())
        else:
            return 'Value({},{})'.format(self.type, self.attributes)

    def __repr__(self):
        return self.__str__()
        
def ignore_nodes(name: str, node: Node, depth: int) -> bool:
    pass

def ignore_values(name: str, value: Value, depth: int) -> bool:
    pass

class Graph(object):
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.values: Dict[str, Value] = {}
        #self.inputs: Dict[] = {}
        #self.outputs = {}
        self.temporary_node_number = 0
        self.producer: Dict[str, Optional[str]] = {}
        self.consumers: Dict[Str, Set[str]] = {}
        
    def load_onnx(self, filename: str):
        model = onnx.load(filename)
        print(model.ir_version)
        print(model.opset_import)
        self.initialize_from_onnx(model.graph)

    def add_node(self, name, operation, inputs, outputs, attr, **kwargs):
        #print('adding node', name, operation, inputs, outputs, attr)
        if name in self.nodes:
            raise AttributeError('Duplicate node name ' + name)
        self.nodes[name] = Node(operation, inputs, outputs, attr, **kwargs)

        for i in inputs:
            if i is not None:
                self.consumers.setdefault(i, set()).add(name)

        for o in outputs:
            if o is not None:
                assert o not in self.producer or self.producer[o] is None, "Multiple producers for value " + o
                self.producer[o] = name
                if o not in self.values:
                    self.values[o] = Value()

    def copy_node(self, name, node, **kwargs):
        assert name is not None, 'node names cannot be None'
        self.add_node(name, node.operation, node.inputs, node.outputs, node.attributes, **kwargs)

    def copy_value(self, name: str, value: Value):
        assert name is not None, 'value names cannot be None'
        self.values[name] = value
        if value.value is not None:
            self.producer[name] = None
    
    def merge_value(self, name: str, value: Value):
        assert name is not None, 'value names cannot be None'
        if name not in self.values:
            return self.copy_value(name, value)

        val = self.values[name]
        if value.value is not None:
            val.value = value.value
            self.producer[name] = None
        val.type.merge(value.type)
        val.attributes.merge(value.attributes)
    
    def get_node(self, name):
        return self.nodes[name]
        
    def get_value(self, name):
        assert name is not None, 'value names cannot be None'
        if name not in self.values:
            self.values[name] = Value()
            self.producer[name] = None
            self.consumers[name] = set()
        return self.values[name]
        
    def get_temporary_node_name(self):
        self.temporary_node_number += 1
        return 'n' + str(self.temporary_node_number)

    def set_value_type(self, name: str, type: TensorType, **kwargs):
        value = self.get_value(name)
        value.set_type(type, **kwargs)

    def set_input(self, name: str, type: TensorType, **kwargs):
        value = self.get_value(name)
        value.set_type(type, **kwargs)
        value.set_input()

    def set_output(self, name: str, type: TensorType, **kwargs):
        value = self.get_value(name)
        value.set_type(type, **kwargs)
        value.set_output()

    def set_parameter(self, name: str, type: TensorType, constant: numpy.ndarray, **kwargs):
        value = self.get_value(name)
        value.set_type(type, **kwargs)
        value.set_parameter(constant)
        self.producer[name] = None
        
    def select_nodes_where(self, selector: Callable[[Node], bool]):
        return [name for (name, node) in self.nodes.items() if selector(node)]

    def select_nodes_where_attribute(self, attribute: str, value: Any, defvalue: Any=None, namespace: str=''):
        def selector(node: Node) -> bool:
            val = node.get_attribute(attribute, defvalue=defvalue, namespace=namespace)
            return val == value
        
        return self.select_nodes_where(selector)

    def select_values_where(self, selector: Callable[[Value], bool]):
        return [name for (name, value) in self.values.items() if selector(value)]
        
    def select_values_where_attribute(self, attribute: str, testvalue: Any, default: Any=None, namespace: str=''):
        def selector(value: Value) -> bool:
            return value.get_attribute(attribute, default, namespace) == testvalue
        
        return self.select_values_where(selector)

    def get_inputs(self) -> List[str]:
        return self.select_values_where_attribute('input', True);
    
    def get_outputs(self) -> List[str]:
        return self.select_values_where_attribute('output', True);
    
    def initialize_from_onnx(self, graph: onnx.GraphProto):
        self.onnx = graph
        
        for input in graph.input:
            #print('input', input)
            (dtype, shape, denotation) = decode_type(input.type)
            self.set_input(input.name, TensorType(dtype, shape, denotation), onnx=input)

        for output in graph.output:
            #print('output', output)
            (dtype, shape, denotation) = decode_type(output.type)
            self.set_output(output.name, TensorType(dtype, shape, denotation), onnx=output)

        for info in graph.value_info:
            self.set_value_type(info.name, decode_type(info.type),
                                onnx=info)

        for val in graph.initializer:
            value = decode_tensor(val)
            type = TensorType.from_value(value)
            self.set_parameter(val.name, type, value, onnx=val)

        for node in graph.node:
            name = node.name if node.name != '' else self.get_temporary_node_name()
            self.copy_node(name, Node.from_onnx(node))
            #print(node)
            #print(str(n), node.op_type, inputs, '->', outputs, attrs)
        

        #print(len(self.values), 'values')
        #for (n,v) in natsorted(self.values.items()):
        #    print(n, v)

        #print(len(self.nodes), 'nodes')
        #for (n,v) in natsorted(self.nodes.items()):
        #    print(n, v)

    def get_producer(self, name):
        if name in self.producer:
            return self.producer[name]
        return None

    def get_consumers(self, name):
        if name in self.consumers:
            return self.consumers[name]
        return set()
            
    def graft_subgraph(self, g, renames: Mapping[str, str], namespace: str = None):
        """
        Graft the other subgraph onto this one, which consists of
        renaming and copying both nodes and variables.

        This takes the given graph, renames any values in the
        rename_from list to the correponding name in the rename_to list
        (which should match values in the current graph), renames all
        other values and nodes to internal values that won't clash,
        and then copies nodes and values in.

        The renames variable is the only way to make the subgraph refer
        to values in the main graph.
        """

        node_renames: Dict[str, str] = {}

        if namespace is None:
            namespace = str(randrange(1000000))
        
        def create_unique_name(name: str) -> str:
            return (namespace if namespace is not None else '') + '__' + name
        
        def rename_value(name: str) -> str:
            if name not in renames:
                node_renames[name] = create_unique_name(name)
            return node_renames[name]

        def rename_node(name: str) -> str:
            if name not in node_renames:
                node_renames[name] = create_unique_name(name)
            return node_renames[name]
        
        def on_node(name: str, node: Node, depth: int):
            name = create_unique_name(name)

            for i in range(0, len(node.inputs)):
                node.inputs[i] = rename_value(node.inputs[i])

            for i in range(0, len(node.outputs)):
                node.outputs[i] = rename_value(node.outputs[i])
                
            self.copy_node(name, node)
            
        def on_value(name: str, value: Value, depth: int):
            value.remove_attribute('input')
            value.remove_attribute('output')
            self.merge_value(rename_value(name), value)
                
        g.visit(g.get_inputs(), g.get_outputs(), on_node, on_value)
        
        
    
    def visit(self, inputs: List[str], outputs: List[str],
              on_node: Callable[[str, Node, int], bool] = ignore_nodes,
              on_value: Callable[[str, Value, int], bool] = ignore_values):
        """
        In-order visit of nodes of the graph, moving from inputs to outputs
        """

        done_nodes: Set[str] = set()
        input_set = set(inputs)
        done_values = set(input_set)
        output_set = set(outputs)

        #print('producers', natsorted(self.producer.items()))
        
        def bottom_up(values: Set[str], depth: int):
            new_nodes = set([self.producer[v] for v in values if self.producer[v] is not None])# - done_nodes
            #done_nodes.update(new_nodes)

            new_values = set([i for n in new_nodes if n in self.nodes for i in self.nodes[n].inputs if i is not None]) - input_set# - done_values
            #done_values.update(new_values)

            #print('at depth', depth, 'outputs', values, 'nodes', new_nodes, 'inputs', new_values)
            
            if len(new_values) > 0:
                # produce the new inputs
                bottom_up(new_values, depth + 1)

            # Now the inputs are there, we can run the nodes
            for n in natsorted(new_nodes):
                if n not in done_nodes:
                    on_node(n, self.nodes[n], depth)
                    done_nodes.add(n)

            # And those produced the values
            for v in natsorted(values):
                if v not in done_values:
                    on_value(v, self.values[v], depth)
                    done_values.add(v)

        # We know about the inputs
        for i in inputs:
            on_value(i, self.values[i], -1)
        
        # First, go bottom up (from output to inputs) to identify what nodes
        # we need to run
        bottom_up(set(outputs), 0)
