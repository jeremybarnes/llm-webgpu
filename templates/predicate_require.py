from typing import Sequence, Optional, Callable, List, Iterator

from .path import Private
from .symbol import Symbol, SymbolRef
from .scope import Scope
from .predicate import Predicate, BindArguments, _get_arg_index
from .argument import Argument, ArgumentRef
from .expression import Expression
from .named_scope import NamedScope
from .null_scope import NullScope


class PredicateRequire(Predicate):
    """
    Implements a predicate that requires that a given condition be true.  This
    has the effect of strictly reducing the set of possible solutions to those
    for which the passed predicate returns True.

    Arguments:
    args: the arguments (in order) that are parameters of the expression.  Each
          argument must be referred to in the expression.
    require: the expression that is required to be True in order for the
          predicate to match
    name: the name of the expression (used for tracing and debugging).  By
          default, the name of the require argument is used.
    """

    def __init__(self,
                 args: Sequence[ArgumentRef],
                 require: Expression,
                 name: Optional[str] = None):
        self.name = require._name if name is None else name
        self._require = require
        self._args = BindArguments([(a._get_identifier(),a) for a in args])

        print('require', require)
        
        # TODO: we should really pass in a path here to avoid name clashes within this
        # single scope for the require predicate... currently two different predicates
        # with the same name and the same argument name would create the same symbol
        # for that argument, which is probably not what we want.  Later...
        outer = NullScope(Private('bind'))
        scope = NamedScope(outer, self.name)

        def map_input(ref: SymbolRef) -> Symbol:
            sym = ref.sym()
            if not isinstance(sym, ArgumentRef):
                raise TypeError('Predicate input {} is {} not ArgumentRef'
                                .format(sym, type(sym)))

            # Create a bind-time symbol
            res = scope.symbol(sym._get_type(), sym._get_identifier())
            print('input', ref, 'is', res, 'with path', res.path())
            return res

        # Substitute in concrete symbols for the arguments in the input.
        subst = require.substitute(map_input)

        print('--> require', require, '\n-->subst', subst)
        
        # We get our inputs as per the substituted version
        inputs = subst.inputs()

        print('\n\n\nsubst', subst, subst.inputs())
        print('args', self._args)
        self._subst = subst

        # Get the input from the list of input symbols
        def collect_input(inp: SymbolRef) -> Callable[[Scope, Sequence[Symbol]], Symbol]:
            print('  collect input', inp._symbol.idx, 'in?', self._args.keys(), inp._symbol.idx in self._args)
            if inp._symbol.idx in self._args:
                index = _get_arg_index(self._args, inp._symbol.idx)
                def get_sym_from_input(scope: Scope, syms: Sequence[Symbol]):
                    return syms[index]
                return get_sym_from_input
            else:
                def create_free_sym(scope: Scope, syms: Sequence[Symbol]):
                    return scope.symbol(type(inp._symbol), inp._symbol.idx)
                return create_free_sym

        self._collect_inputs = [ collect_input(inp.ref()) for inp in subst.inputs() ]
        
    def arguments(self) -> BindArguments:
        return self._args

    def _apply_resolved(self, scope: Scope, inputs: Sequence[Symbol]) -> Iterator[Scope]:
        # We simply yeild a scope with the condition added and the condition's
        # inputs resolved to the expression's inputs.

        print('require apply_resolved: inputs', list(inputs))
        print('subst inputs', self._subst.inputs())
        print('args', self._args )
        
        with scope.mutate('predicate_require ' + self.name) as inner:
            # The condition must be true
            inner.condition(self.name, self._subst)

            # Bind the inputs into the system
            for lhs,collect in zip(self._subst.inputs(), self._collect_inputs):
                rhs = collect(inner, inputs)
                print('resolving', lhs, 'to', rhs, 'using', collect)
                inner.resolve(lhs, rhs)
            yield inner
            
def pred_require(args: List[ArgumentRef],
                 expr: Expression,
                 name: Optional[str] = None) -> Predicate:
    """
    "Decorate" the given expression into a predicate.
    """

    return PredicateRequire(args, expr, name)

