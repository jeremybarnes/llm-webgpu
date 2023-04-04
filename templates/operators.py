from typing import Callable, Any, cast, Dict, TYPE_CHECKING
from .symbol import Symbol, SymbolRef
import operator

if TYPE_CHECKING:
    from .expression import Expression

# Handle circular import issues by late-importing the expression library
def _make_expression(name, fn, args) -> 'Expression':
    from .expression import Expression
    return Expression(name, fn, args)
    
def _find_op(nm: str) -> Callable:
    """
    Find the operator with the given name and return a callable.
    """
    if hasattr(operator, nm):
        return getattr(operator, nm)
    return getattr(operator, '__' + nm + '__')

def _unary_operator(nm: str) -> Callable[[Any],'Expression']:
    fn = _find_op(nm)
    return lambda x: _make_expression(nm, fn, [x])

def _binary_operator(nm: str) -> Callable[[Any, Any],'Expression']:
    fn = _find_op(nm)
    return lambda x,y: _make_expression(nm, fn, [x,y])

def _right_operator(nm: str) -> Callable[[Any, Any],'Expression']:
    # Right operators we handle by reversing the arguments
    # (needs to be tested...)
    fn = _find_op(nm)
    return lambda x,y: _make_expression(nm, fn, [y, x])

def _inplace_operator(nm: str) -> Callable[[Any, Any],'Expression']:
    fn = _find_op('i' + nm)
    return lambda x,y: _make_expression(nm, fn, [x,y])

def _nm(op: str) -> str:
    if op.startswith('__'):
        return op
    return '__' + op + '__'

def _special_operator(nm: str) -> Callable:

    # Define those operators that can't be easily imported using their name
    # from the operator module, usually due to unusual arguments or name
    # clashes with builtins.
    
    #TODO: how to handle round?  It's also done below (and needs tests...)
    #def __round__(self, ndigits = None):
    #    return _make_expression('round', operator.round, [self, ndigits])

    def __complex__(self):
        return _make_expression('complex', complex, [self])

    def __int__(self):
        return _make_expression('int', int, [self])

    def __float__(self):
        return _make_expression('float', float, [self])

    def __round__(self):
        return _make_expression('round', round, [self])
    
    def __trunc__(self):
        return _make_expression('trunc', trunc, [self])

    def __floor__(self):
        return _make_expression('floor', floor, [self])

    def __ceil__(self):
        return _make_expression('ceil', ceil, [self])
    
    def __divmod__(self, other):
        return _make_expression('divmod', divmod, [self,other])

    def __getitem__(self, item):
        return _make_expression('getitem', operator.__getitem__, [self,item]) 

    ops_list = [ __complex__, __int__, __float__, __round__, __trunc__, __floor__,
                 __ceil__, __divmod__, __getitem__ ]

    ops = { op.__name__.replace('__', ''): op for op in ops_list }

    return cast(Callable, ops[nm])


def _make_logical(typename: str, base: type) -> type:
    """
    Creates a version of the given base class with the logical operators
    defined, which enables it to be used in expressions.  These expressions
    will eventually generate an Expression object.
    """

    unary_operators = ['invert', 'truth' ]
    comparison_operators = ['gt', 'ge', 'lt', 'le', 'eq', 'ne' ] 
    binary_operators = ['and', 'xor', 'or']
    
    # Here are the methods defined for the class, which are all generated
    # from those in the operator module.
    methods: Dict[str, Any] = {
        **{_nm(op): _unary_operator(op) for op in unary_operators },
        **{_nm(op): _binary_operator(op) for op in comparison_operators },
        **{_nm(op): _binary_operator(op) for op in binary_operators },
        **{_nm('r'+op): _right_operator(op) for op in binary_operators + ['pow'] },
        **{_nm('i'+op): _inplace_operator(op) for op in binary_operators },
        'typename': typename,
    }

    return type(typename, (base,), methods)

def _make_arithmetic(typename: str, base: type) -> type:
    """
    Creates a version of the given base class with the logical operators
    defined, which enables it to be used in expressions.  These expressions
    will eventually generate an Expression object.
    """

    unary_operators = ['neg', 'pos', 'abs', 'invert', 'truth' ]
    comparison_operators = ['gt', 'ge', 'lt', 'le', 'eq', 'ne' ] 
    binary_operators = ['add', 'sub', 'mul', 'matmul',
                        'truediv', 'floordiv', 'mod', 'pow',
                        'lshift', 'rshift', 'and', 'xor', 'or']
    special_operators = [ 'complex', 'int', 'float', 'round', 'trunc', 'floor',
                          'ceil', 'divmod' ]
    
    # Here are the methods defined for the class, which are all generated
    # from those in the operator module.
    methods: Dict[str, Any] = {
        **{_nm(op): _unary_operator(op) for op in unary_operators },
        **{_nm(op): _binary_operator(op) for op in comparison_operators },
        **{_nm(op): _binary_operator(op) for op in binary_operators },
        **{_nm('r'+op): _right_operator(op) for op in binary_operators + ['pow'] },
        **{_nm('i'+op): _inplace_operator(op) for op in binary_operators },
        **{_nm(op): _special_operator(op) for op in special_operators },
        'typename': typename,
    }

    return type(typename, (base,), methods)


class EqualityComparable(Symbol):
    """
    Represents a symbol on which the basic comparison operators are
    defined.
    """

    def __eq__(self, other: Any) -> 'Expression': # type: ignore
        return _make_expression('eq', operator.eq, [self, other])

    def __ne__(self, other: Any) -> 'Expression': # type: ignore
        return _make_expression('ne', operator.ne, [self, other])
