from typing import Union, Any, TYPE_CHECKING, Optional, Set, Iterator, Callable, Tuple, overload, Sequence, List, Optional
from .exceptions import MatchError
from .path import Index
from .scope import Scope
from .symbol import Symbol
from .argument import Argument, ArgumentRef
from .expression import Expression

from collections import OrderedDict
from itertools import count
from abc import abstractmethod

BindArg = Union[Symbol,ArgumentRef,Any]
if TYPE_CHECKING:
    BindArguments = OrderedDict[Index, ArgumentRef]
else:
    BindArguments = OrderedDict # [str, ArgumentRef]

def _get_arg_index(args: BindArguments, arg: Index) -> int:
    for index,name in zip(count(), args):
        if name == arg:
            return index
    raise RuntimeError('could not find arg {} in args {}'
                       .format(arg, args))
    
    
class Predicate(object):
    """
    A predicate is an unbound match generator.  It exposes a single
    operation (bind) which returns a match generator.
    """
    name: str
    
    def bind(self, *args: BindArg) -> 'Predicate':
        """
        Bind the given symbols to the arguments of the predicate.  This
        function should not mutate self; it should return a new Predicate
        which is self with the arguments bound.
        """
        from .bound_predicate import BoundPredicate
        return BoundPredicate(self, *args)

    def reorder_args(self, new_args: BindArguments,
                     name: Optional[str] = None) -> 'Predicate':
        """
        Change the arguments of the predicate to be those in new_args.

        Postcondition: the result.arguments() == new_args (possibly with
                       some of the types specialized)

        This is different to bind() in that the arguments are not renamed;
        the arguments are in the same namespace.  This function can simply
        reorder arguments, add new (unused) arguments, or make existing
        arguments into free variables.

        For each argument:
        - If it's present in both the original arguments and new_args, it
          is kept (possibly in a new position).
        - If it's present in new_args but not in the original arguments,
          then the returned predicate will take (but ignore) the argument
        - If it's present in the original arguments but not in new_args,
          the argument will be replaced by a free variable.
        """
        from .reordered_predicate import ReorderedPredicate
        return ReorderedPredicate(self, new_args, self.name if name is None else name)
    
    @abstractmethod
    def arguments(self) -> BindArguments:
        """
        Return the arguments for the predicate (those symbols which are
        exposed to the outside and may be bound to external symbols).
        """
        ...

    def _verify_arguments(self) -> None:
        """
        Internal method that will verify the integrity of the arguments:
        a) They must be ArgumentRefs and not Arguments
        b) Their names must be unique
        """
        names: Set[Index] = set()
        for nm,a in self.arguments().items():
            if not isinstance(a, ArgumentRef):
                raise TypeError('arg {} ({}) of {} is not ArgumentRef'
                                .format(nm, a, self.name))
            if isinstance(a, Argument):
                raise TypeError('arg {} ({}) of {} is Argument; use arg._ref() to make ArgumentRef'
                                .format(nm, a, self.name))
            if nm in names:
                raise TypeError('two arguments of {} have the same name {}'
                                .format(self.name, nm))
            names.add(nm)
        
    #@abstractmethod
    def _apply_resolved(self, scope: Scope, inputs: Sequence[Symbol]) -> Iterator[Scope]:
        """
        Apply with all of the arguments resolved.  This is called by the
        default apply method implementation; either this method or apply
        needs to be overridden.  The length of inputs will be equal to
        the length of self.arguments(), and each input will correspond
        directly to the given argument.
        """
        ...
        
    #@abstractmethod
    def apply(self, scope: Scope, *inputs: Symbol) -> Iterator[Scope]:
        """
        Apply the predicate to the given scope, returning the possible
        match options.
        """
        print('apply with inputs', self, scope, inputs)

        args = self.arguments()

        # Verify we have the right number of symbols passed in for the
        # arity of the predicate
        if len(args) < len(inputs):
            raise MatchError('predicate {} passed too many args: expected {} args but got {}'
                             .format(self, len(args), len(inputs)))

        # Verify that each of the inputs has the right type for its
        # argument
        for arg,input,index in zip(args.values(), inputs, range(len(inputs))):
            if not isinstance(input, arg._get_type()):
                raise MatchError('predicate {} input {} wrong type: expected {} got {}'
                                 .format(self, index, arg._get_type(), input))
        
        if len(args) == len(inputs):
            return self._apply_resolved(scope, inputs)
        
        # Find out the value of each of our input arguments.
        # This is a combination of:
        # - Those which are passed from inputs get passed directly
        # - The rest are considered free, and have a free variable created
        arg_values: List[Symbol] = list(inputs)

        with scope.mutate(self.name + ' apply') as inner:

            # Those afterwards are free variables
            for arg,index in zip(args.values(), range(len(args))):
                if index < len(inputs):
                    continue

                print('index', index, 'arg', arg, 'scope', scope)
                
                # Free metasymbol within the current scope
                sym = arg._in(scope)

                # Append it
                arg_values.append(sym)

            return self._apply_resolved(inner, arg_values)

    # It can be called as a function to bind...
    def __call__(self, *args: BindArg):
        return self.bind(*args)

    # Or indexed to apply (the first argument is a scope)
    @overload
    def __getitem__(self, args: Tuple[Scope]) -> Iterator[Scope]: ...

    @overload
    def __getitem__(self, args: Tuple[Scope, Symbol]) -> Iterator[Scope]: ...

    @overload
    def __getitem__(self, args: Tuple[Scope, Symbol, Symbol]) -> Iterator[Scope]: ...

    def __getitem__(self, args):
        if isinstance(args, tuple):
            return self.apply(*args)
        return self.apply(args)

    def _print_args(self) -> str:
        def print_arg(a: ArgumentRef) -> str:
            return str(a._get_identifier()) + ': ' + a._get_type().__name__

        return '(' + ', '.join(print_arg(a) for a in self.arguments().values()) + ')'
    
    # Pretty-printing
    def __str__(self) -> str:

        return self.name + self._print_args()

    def __repr__(self) -> str:
        return type(self).__name__ + self._print_args()



class PredicateFn(Predicate):
    """
    A predicate that forwards to a Python function.  It introspects via
    the type annotations to figure out how to implement the various
    Predicate functions.

    The function passed in must take a Scope as the first argument, and
    symbols as its other arguments, and return a generator of scopes
    when called.
    """

    def __init__(self,
                 fn: Callable[..., Iterator[Scope]]):
        """
        Initialize from the given function, and a list of pre-bound arguments.
        """

        self.name = fn.__name__
        ann = fn.__annotations__

        if ann is None:
            raise TypeError('Generators require type annotations to be bound')

        ann_list = list(ann.items())

        if ann_list[0][1] != Scope:
            raise TypeError('First input to function {} wrapped as a generator must be a Scope'
                            .format(fn.__name__))

        # Other arguments must be a subclass of symbols.  They are turned into
        # meta symbols.  This can't currently be expressed in the annotations,
        # so we check it here.

        symbol_params = ann_list[1:-1]

        args: BindArguments = OrderedDict()

        for (name,tp),i in zip(symbol_params, range(len(symbol_params))):
            if not issubclass(tp, Symbol):
                raise TypeError('Parameters of generator functions must be symbols')
            meta = ArgumentRef(tp, name)
            args[name] = meta
                
        print('args', args)
                    
        self._args: BindArguments = args
        self._fn = fn

        self._verify_arguments()
        
    #def bind(self, *args: BindArg) -> Predicate:
    #    print('bind', args, self._bound)
    #    return PredicateFn(self._fn, self._bound + list(args))

    def arguments(self) -> BindArguments:
        return self._args

    def bound(self) -> 'OrderedDict[str, BindArg]':
        # Nothing is bound
        return OrderedDict()
    
    def _apply_resolved(self, scope: Scope, inputs: Sequence[Symbol]) -> Iterator[Scope]:
        # Construct the argument list.  This involves:
        # - Bound arguments are resolved in their scope
        # - Unbound arguments are considered free and created

        print('inputs', inputs)
        
        return self._fn(scope, *inputs)


def predicate(fn: Callable[..., Iterator[Scope]]) -> Predicate:
    """
    Decorate the given function, turning it into a Predicate (and hence a
    Generator once bound).
    """
    return PredicateFn(fn)

