from typing import Optional, Dict, Callable, Tuple, Sequence, Iterator
from .exceptions import BindError
from .path import Index
from .argument import ArgumentRef
from .predicate import Predicate, BindArguments, _get_arg_index
from .scope import Scope
from .symbol import Symbol


class ReorderedPredicate(Predicate):
    """
    Predicate with reordered arguments.  This is the default implementation
    of the reorder_args() function.
    """
    name = 'reordered'

    def __init__(self, pred: Predicate,
                 new_args: BindArguments,
                 name: Optional[str] = None):
        if name is not None:
            self.name = name
        
        self._pred = pred
        old_args = pred.arguments()
        arg_types: Dict[Index, ArgumentRef] = {}

        def collect_arg(name: Index, arg: ArgumentRef) -> Callable[[Scope, Sequence[Symbol]], Symbol]:
            """
            Returns a function that will collect the argument of the given name and type
            from either the inputs or the scope.
            
            This is used in constructing argument lists for predicates.
            """
            if name in new_args:
                # This is simply reordered.  First make sure the type is
                # compatible.
                old_tp = arg._get_type()
                new_tp = new_args[name]._get_type()

                new_arg_number = _get_arg_index(new_args, name)
                
                resolved_tp = old_tp.compatible_type(old_tp, new_tp)
                if resolved_tp is None:
                    raise BindError('incompatible types in reordered_args: old {} new {}'
                                    .format(old_tp, new_tp))
                arg_types[name] = ArgumentRef(resolved_tp, name)

                return lambda sc,syms: syms[new_arg_number]
            else:
                # This argument is free; create it
                return lambda sc,syms: sc.symbol(arg._get_type(), arg._get_identifier())

        # Go through the old args, figuring out how to collect it from the new
        # ones
        self._collect_args = [ collect_arg(name, arg) for name,arg in old_args.items() ]

        def get_arg(name: Index, arg: ArgumentRef) -> Tuple[Index,ArgumentRef]:
            if name in arg_types:
                return (name, arg_types[name])
            else:
                return (name, arg)
        
        # Go through new arguments
        self._args = BindArguments([get_arg(n,a) for n,a in new_args.items()])

    def arguments(self) -> BindArguments:
        return self._args
        
    def _apply_resolved(self, scope: Scope,
                        inputs: Sequence[Symbol]) -> Iterator[Scope]:
        with scope.mutate('reorder ' + self.name) as inner:
            # Collect each of the inputs to the pred
            collected = [collect(inner, inputs) for collect in self._collect_args]
            return self._pred.apply(inner, *collected)

