from typing import Type, Union, Optional
from operator import getitem

from .path import Index
from .symbol import Symbol
from .expression import Expression
from .operators import _make_arithmetic
from .symbol_types import AnyT
from .scope import Scope

class ArgumentRef(Symbol):
    typename='meta'
    tp=Symbol
    
    def __init__(self, tp: Type[Symbol], identifier: Index, parent: Optional['Argument'] = None):
        super().__init__(identifier, parent)
        self._type: Type[Symbol] = tp
        self.tp = tp
        self._identifier: Index = identifier

    def __repr__(self) -> str:
        return 'Argument(' + self._type.__name__ + ',' + repr(self._identifier) + ')'

    def __key(self):
        return (self._identifier,self._type)

    def __hash__(self):
        return hash(self.__key())
    
    def __eq__(self, other):
        return self.__key() == other.__key()

    def __ne__(self, other):
        return self.__key() != other.__key()

    def _get_type(self) -> Type[Symbol]:
        return self.__dict__['_type']

    def _get_identifier(self) -> Index:
        return self.__dict__['_identifier']

    def _in(self, scope: Scope) -> Symbol:
        print('in for', self, self._get_type())
        return scope.symbol(self._get_type(), self._get_identifier())

    def _ref(self) -> 'ArgumentRef':
        if type(self) == ArgumentRef:
            return self
        return ArgumentRef(self._get_type(), self._get_identifier())


ArgumentBase = _make_arithmetic('ArgumentBase', ArgumentRef)
    
class Argument(ArgumentBase):
    def __getattr__(self, name: Union[Expression,Symbol,Index]) -> Union[Expression, 'Argument']:
        if isinstance(name, Expression) or isinstance(name, Symbol) or True:
            # it's an expression
            return Expression('getattr', getattr, [self, name])
        else:
            # it's a pure lookup; we take a child
            return Argument(AnyT, name, self)

    def __getitem__(self, name: Union[Expression,Symbol,Index]) -> Expression:
        return Expression('getitem', getitem, [self, name])


_1 = Argument(AnyT, 1)
_2 = Argument(AnyT, 2)
_3 = Argument(AnyT, 3)
_4 = Argument(AnyT, 4)
_5 = Argument(AnyT, 5)
_6 = Argument(AnyT, 6)
_7 = Argument(AnyT, 7)
_8 = Argument(AnyT, 8)
_9 = Argument(AnyT, 9)
