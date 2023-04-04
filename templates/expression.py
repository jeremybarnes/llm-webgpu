from typing import Callable, Sequence, Any, List, Optional, Tuple, Union, Set, Dict
from .symbol import Symbol, SymbolRef
from .operators import _make_arithmetic
import operator

class ExpressionBase(object):
    """
    Represents an expression over one or more input symbols.
    """

    def __init__(self, name: str, fn: Callable, args: Sequence[Any]):
        self._name = name
        self._fn = fn
        self._args = list(args)

    def __repr__(self) -> str:
        return self._name + '(' + ','.join([repr(arg) for arg in self._args]) + ')'

    def __str__(self) -> str:
        return self.__repr__()
    
    def __truth__(self) -> NotImplemented:
        return NotImplemented
    
    def inputs(self) -> List[Symbol]:
        """
        Returns the set of input symbols for the expression.  Once all
        of these have values, it is possible to evaluate the expression
        itself.
        """

        result = []
        done: Set[SymbolRef] = set()

        def scan_inputs(arg):
            if isinstance(arg, Expression):
                for subarg in arg._args:
                    scan_inputs(subarg)
            elif isinstance(arg, Symbol):
                if arg.ref() not in done:
                    result.append(arg)
                    done.add(arg.ref())

        scan_inputs(self)

        return result

    def evaluate(self, args: List[Any]) -> Any:
        """
        Evaluate the expression over the given set of arguments.  Args is a list of
        values the same length as inputs() and in the same order, each of which will
        be bound to the appropriate symbol.
        """
        #print('evaluate with args', args)
        
        symbol_map = {k.ref(): v for k,v in zip(self.inputs(), args) }

        def eval_recursive(arg):
            if isinstance(arg, Expression):
                vals = [eval_recursive(a) for a in arg._args]
                return arg._fn(*vals)
            elif isinstance(arg, Symbol):
                return symbol_map[arg.ref()]
            else:
                return arg
            
        return eval_recursive(self)

    def substitute(self,
                   subst_symbols: Callable[[SymbolRef], Any] = None,
                   ) -> 'ExpressionBase':
        """
        Substitute arguments from the given mapping, returning a new Expression
        of the same type as self with each symbol replaced by the result of
        calling subst on it.
        """

        symbol_map: Dict[SymbolRef, Any] = {}

        def get_symbol(sym: SymbolRef) -> SymbolRef:
            assert not isinstance(sym, Symbol), "get_symbol passed Symbol not SymbolRef"

            if subst_symbols is None:
                return sym
            elif sym in symbol_map:
                return symbol_map[sym]
            else:
                res = subst_symbols(sym)
                symbol_map[sym] = res
                return res
        
        def subst_recursive(arg):
            if isinstance(arg, Expression):
                args = [subst_recursive(a) for a in arg._args]
                fn = arg._fn
                name = arg._name

                # Construct a result of the same type so that subclasses retain
                # their specialized behavior
                return type(arg)(name, fn, args)
            elif isinstance(arg, Symbol):
                return get_symbol(arg.ref())
            else:
                return arg
            
        return subst_recursive(self)

        
    
    def equalities(self) -> Optional[List[Tuple[Symbol, Union[Symbol,'Expression']]]]:
        """
        If possible, reduce the given expression into a set of simple equations

        var1=expr(other vars apart from var1)
        var2=expr(other vars apart from var2)
        ...

        This can be used to convert conditions into aliasing and more rapidly
        resolve values that come from expressions.

        Returns None if the expression is not decomposable under the analysis
        that can be easily done.
        """

        # sym == expression is already in terms of equalities
        if len(self._args) == 2 and self._fn == operator.eq:
            def check(arg1: Any, arg2: Any) -> Optional[Tuple[Symbol, Union[Symbol,Expression]]]:
                if isinstance(arg1, Symbol):
                    return (arg1,arg2)
                return None

            arg1,arg2 = self._args
            v1 = check(arg1, arg2)
            if v1 is not None:
                return [v1]
            v2 = check(arg2, arg1)
            if v2 is not None:
                return [v2]
        elif self._name == 'all':
            # An all function means that the equalities of each term can be
            # extracted
            result: List[Tuple[Symbol, Union[Symbol,'Expression']]] = []
            for arg in self._args:
                if isinstance(arg, Symbol):
                    # TODO: this could be relaxed if the symbol is a boolean...
                    raise TypeError("Arguments to expr_all should not be Symbols")
                eq = arg.equalities()
                #print('arg', arg, type(arg), eq)
                if eq is None:
                    return None
                else:
                    result += eq
            return result
            
        # We could do more, but for now that's enough
        return None
    
# We add the arithmetic operators to both Symbol and Expression.  This is
# to avoid multiple inheritance.  We don't want Expression to derive from
# Symbol because we need to define Expression first, nor can Symbol derive
# from Expression as we don't want all kinds of symbols to have access to
# all kinds of operators.

Expression: Any = _make_arithmetic('Expression', ExpressionBase)

def equals(a1: Union[Expression,Symbol,Any],
           a2: Union[Expression,Symbol,Any]) -> Expression:
    """
    Easier to type version of a1 == a2 (since MyPy assumes that all equality
    comparisons result in a boolean type)
    """
    return Expression('eq', operator.eq, [a1,a2])

def expr_all(*exprs: Union[ExpressionBase, bool], name: str = None) -> Expression:
    """
    Create an expression where all of the given sub-expressions have to be
    true.
    """
    fixed: List[bool] = [e for e in exprs if isinstance(e, bool)]
    var:   List[ExpressionBase] = [e for e in exprs if not isinstance(e, bool)]

    print('expr_all: fixed', fixed, 'var', var)
    
    def static_true() -> bool: return True
    def static_false() -> bool: return False
    
    # Short circuit false
    if len(fixed) > 0 and not all(fixed):
        print('short circuit false')
        return Expression('short_circuit_false', static_false, [])

    # Short circuit true
    if (len(exprs) == 0
        or (len(var) == 0 and all(fixed))):
        print('short circuit true')
        return Expression('short_circuit_true', static_true, [])

    def tracing_all(*args) -> bool:
        print('tracing_all with args', list(args))
        print('result', all(args))
        return all(args)
    
    # Otherwise, we execute with the all function
    print('not short circuit')
    return Expression('all', tracing_all, exprs)

