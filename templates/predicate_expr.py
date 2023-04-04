from typing import Sequence, Optional, Callable, List, Iterator, Dict

from .predicate import *
from .named_scope import NamedScope
from .null_scope import NullScope


class PredicateExpr(Predicate):
    """
    A predicate that matches an underlying expression.  This class takes care of the
    arguments and application.
    """

    def __init__(self,
                 result: ArgumentRef,
                 args: Sequence[ArgumentRef],
                 expr: Expression,
                 name: Optional[str] = None):
        """
        Initialize from the given expression.  The given set of arguments are those
        that are exposed; the others are bound to free temporaries.  The result of
        the expression is unified with the result variable, or if None, then is
        assigned to a free variable and discarded.

        Note that all arguments passed must be arguments of the expression, apart
        from result.
        """

        self.name = expr._name if name is None else name
        self._expr = expr
        self._args: BindArguments = OrderedDict()
        self._result = result
        
        print('expr', expr, expr.inputs())
        
        expr_args: OrderedDict[Index, ArgumentRef] = OrderedDict([(i.idx, ArgumentRef(i.tp, i.idx)) for i in expr.inputs()])

        # The result is an argument
        self._args[result._get_identifier()] = result._ref()
        
        for a in args:
            aid = a._get_identifier()
            if aid not in expr_args:
                raise MatchError('pred_expr {} argument {} not in inputs to expression'
                                 .format(self.name, a._get_identifier()))
            # don't allow the result to be an argument (for now)
            if aid == result._get_identifier():
                raise MatchError('pred_expr {} argument {} has same name as result'
                                 .format(self.name, a._get_identifier()))
            self._args[aid] = expr_args[aid]

        # Index the args by name
        arg_nums: Dict[Index, int] = { aid: idx for idx,(aid,_) in zip(range(len(self._args)),self._args.items()) }
        
        # for each input: if None, then free variable, or otherwise from arg number n
        self._inputs: List[Tuple[Optional[int], ArgumentRef]] = []

        # Create a list of how to generate each input to the expression: either
        # - get it from arg x, or
        # - create a free variable for it
        has_free = False
        for sym in expr.inputs():
            aid = sym.idx
            idx = arg_nums[aid] if aid in self._args else None
            if aid not in self._args:
                has_free = True
            self._inputs.append((idx, expr_args[aid]))

        self._has_free = has_free

        self._verify_arguments()
        
    def arguments(self) -> BindArguments:
        return self._args

    def _apply_resolved(self, scope: Scope, inputs: Sequence[Symbol]) -> Iterator[Scope]:
        # Construct the argument list.  This involves:

        with scope.mutate('predicate_expr free vars ' + self.name) as inner:
            def get_arg(i: int) -> Symbol:
                idx,msym = self._inputs[i]
                if idx is None:
                    # free symbol, create it
                    return inner.symbol(msym._get_type(), msym._get_identifier())
                else:
                    # bound to an argument, use it
                    return inputs[idx]
        
            call_args: List[Symbol] = [get_arg(i) for i in range(len(self._inputs))]

            res = self._expr.evaluate(call_args)

            if self._result is not None:
                # We evaluate and assign to the given result variable
                res_sym = inner.symbol(self._result._get_type(), self._result._get_identifier())
                inner.resolve(res_sym, res)

            yield inner


def pred_expr(result: ArgumentRef,
              args: List[ArgumentRef],
              expr: Expression,
              name: Optional[str] = None) -> Predicate:
    """
    "Decorate" the given predicate into an expression.
    """

    return PredicateExpr(result, args, expr, name)


