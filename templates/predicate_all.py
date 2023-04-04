from typing import Type
from .predicate import *


class PredicateAll(Predicate):

    name = 'pred_all'
    
    def __init__(self,
                 arguments: Sequence[ArgumentRef],
                 preds: Sequence[Predicate],
                 name: Optional[str] = None):

        our_args: Set[Tuple[Index, Type[Symbol]]] = set()

        def get_key(a: ArgumentRef) -> Tuple[Index, Type[Symbol]]:
            return (a._get_identifier(), a._get_type())
        
        for a in arguments:
            key = get_key(a)
            if key in our_args:
                raise MatchError('Predicate argument specified twice')
            our_args.add(key)
            

        self._args: 'BindArguments' = OrderedDict([(arg._get_identifier(), arg) for arg in arguments])
        self._preds = list(preds)
        if name is not None:
            self.name = name


        print('\n\n')

        print('preds', preds)
        
        # Each sub-expression gets its arguments reordered
        self._reordered = [pred.reorder_args(self._args) for pred in preds]

        print('reordered_exprs', self._reordered)

    def arguments(self) -> BindArguments:
        return self._args

    def _apply_resolved(self, scope: Scope, inputs: Sequence[Symbol]) -> Iterator[Scope]:

        #if len(self._preds) == 1:
        #    return self._reordered[0].apply(scope, *inputs)
        
        print('pred_all apply with inputs', inputs)

        def apply_i(prev: Scope, i: int) -> Iterator[Scope]:
            print('apply_i', i, 'of', len(self._preds), 'prev', prev)
            pred = self._reordered[i]
            gen = pred.apply(prev, *inputs)

            for outer in gen:
                if i == len(self._preds) - 1:
                    yield outer
                else:
                    for inner in apply_i(outer, i + 1):
                        yield inner
                    
        return apply_i(scope, 0)
                

def pred_all(arguments: List[ArgumentRef],
             *preds: Predicate,
             name: Optional[str] = None) -> Predicate:
    print('pred_all: arguments', arguments)
    return PredicateAll(arguments, preds, name)

