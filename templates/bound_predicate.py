from typing import Type
from .predicate import *
from .exceptions import BindError

class BoundPredicate(Predicate):
    """
    Predicate with bound arguments.  This is the default implementation
    of the bind() function.
    """
    name = 'bind'

    def _verify_bind_compatible(self, arg: ArgumentRef, b: BindArg) -> BindArg:
        """
        Verify that the given argument can be bound into the given argument.
        """
        # Depending upon what we've bound, we either:
        # 1.  If it's a metasymbol, then it becomes a new argument
        # 2.  If it's a symbol, then it's bound symbolically
        # 3.  If it's a value, then it's bound by value

        print('verify_bind_compatible', arg, b, type(arg), type(b))
        
        tp = arg._get_type()
        name = arg._get_identifier()
        
        if isinstance(b, ArgumentRef):
            if not issubclass(b._get_type(), tp):
                raise BindError('Unbindable argument {}: parameter {} is not instance of {}'
                                .format(name, b._get_type().__name__, tp.__name__))
            # It's a new argument
            return b

        elif isinstance(b, Symbol):
            if not isinstance(b, tp):
                raise BindError('Bound parameter {}: {} is not instance of {}'
                                .format(name, b, tp))
            return b
        else:
            # Try coercing to determine compatibility; this will raise
            # an exception if it's not usable
            print('coercing', b, 'to', tp)
            coerced = tp(0).coerce_value(b)
            return coerced
    
    def __init__(self, pred: Predicate, *to_bind: BindArg):
        """
        Return a bound version of the given predicate, with the arguments
        replaced by the given values.
        """

        print('bound predicate', pred, to_bind)

        bound: OrderedDict[Index, BindArg] = OrderedDict()
        
        # Don't nest BoundPredicates; keep it to one single bind operation
        if isinstance(pred, BoundPredicate) and False:
            # Later: we can avoid nesting calls by resolving them
            self._pred: Predicate = pred._pred
            self._args: BindArguments = pred._args
        else:
            self._pred = pred

            pred_args = pred.arguments()
            pred_args_list = list(pred_args.items())

            if len(to_bind) > len(pred_args):
                raise MatchError('too many bound arguments ({}) for unbound args ({})'
                                 .format(len(bound), len(pred_args)))

            #bound_args = []
            args: BindArguments = OrderedDict()

            for i in range(len(to_bind)):
                b = to_bind[i]
                nm,arg = pred_args_list[i]

                print('binding arg', arg, 'to value', b)
                bound[nm] = self._verify_bind_compatible(arg, b)

                if isinstance(b, ArgumentRef):
                    # This is also an argument to bound predicate
                    new_name = b._get_identifier()
                    args[new_name] = b._ref()
            
            for i in range(len(to_bind), len(pred_args_list)):
                nm,arg = pred_args_list[i]
                print('unbound arg', arg)
                args[nm] = arg._ref()
                
            self._args = args

        self._bound: OrderedDict[Index, BindArg] = bound
        self._verify_invariants()
        
    def bind(self, *bound: BindArg) -> Predicate:
        # TODO: don't need to nest here...
        return BoundPredicate(self._pred, *self._bound.values(), *bound)

    def _verify_invariants(self) -> None:
        """
        Verify invariants:
        - the total arity matches that of the underlying predicate
        - bound arguments match the argument types of the underlying predicate
        """
        #print('verify pred', self._pred, 'bound', self._bound, 'args', self._args)
        self._verify_arguments()
        pass
        
    def arguments(self) -> BindArguments:
        return self._args
        
    def _apply_resolved(self, scope: Scope,
                        inputs: Sequence[Symbol]) -> Iterator[Scope]:
        # When this is called, we have the same number of inputs as we
        # have arguments.  We now have to assemble a set of arguments to
        # pass to the underlying predicate.  These come from the bound
        # arguments:

        print('bound apply_resolved with inputs', self, inputs)

        # We need to know our arguments.  These correspond to the inputs
        # and have the same length.  We mostly use this to know which name
        # corresponds with which position
        args = self.arguments()

        assert len(args) == len(inputs), "Logic error: _apply_resolved wrong num inputs"

        # What position is each arg in?  TODO: pre-cache in __init__
        arg_positions = { nm: index for nm,index in zip(args.keys(), range(len(args))) }

        # We also need to know the predicate arguments, so we know what type
        # the symbols are
        pred_args = self._pred.arguments()

        assert len(pred_args) >= len(self._bound)
        
        pred_inputs: List[Symbol] = []

        print('self._bound', self._bound)
        
        with scope.mutate(self.name + ' _apply_resolved') as inner:

            # Bound ones first
            for bound_var,index,arg in zip(self._bound.values(), range(len(self._bound)), pred_args.values()):

                arg_value: Symbol

                if isinstance(bound_var, ArgumentRef):
                    # It's a reference to the named argument
                    arg_name = bound_var._get_identifier()
                    arg_position = arg_positions[arg_name]
                    arg_value = inputs[arg_position]
                elif isinstance(bound_var, Symbol):
                    arg_value = bound_var
                else:
                    arg_value = arg._in(inner)
                    print('arg_value', arg_value)
                    print('bound_var', bound_var)
                    inner.resolve(arg_value, bound_var)

                pred_inputs.append(arg_value)

            # Unbound ones correspond position for position
            for index,arg in zip(range(len(pred_args)), pred_args.values()):
                if index < len(self._bound):
                    continue
                ...

            print('calling pred with inputs', pred_inputs)
                
            return self._pred.apply(inner, *pred_inputs)
            
    def __str__(self) -> str:
        res = 'bind(' + str(self._pred)
        for n,v in self._bound.items():
            res += ', ' + str(n) + '=' + str(v)
        res += ')' + self._print_args()
        return res
        
