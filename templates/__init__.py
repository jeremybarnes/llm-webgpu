# We want:
# - (done) Backtracking (as we may have more than one way to match something)
# - (done) yield (so we don't need to be as careful with state)
# - (done) Resolving to a value
# - (done) Resolving to the same value as another symbol eventually resolves to
# - (done) Resolving to an expression over another symbol's value
# - (done) Assertions / postconditions that block a match
# - (done) type annotations & checking

# TODOs:
# - graph operations as predicates not methods
# - (done) no more names (paths and indexes)
# - (done) commit lifts names out of the try space
# - systems separate from everything else
# - (done) equality expression equivalent to assign when possible
# - ExpressionBase as abstract root of XXXExpression hierarchy

from typing import Type, Any

# Exception types
from .exceptions import MatchError, BindError

# Base abstractions
from .path import Index, Private, Path, _private
from .symbol import Symbol, SymbolRef
from .operators import _make_logical, _make_arithmetic, EqualityComparable
from .expression import ExpressionBase, Expression, equals, expr_all
from .scope import Scope

# Data types
from .symbol_types import Logical, Arithmetic, BoolT, DimT, RealT, SymbolArray, TypeT, AnyT, UnitT, ParameterMapT, ExpressionT, ArgumentListT, ExpressionValueT, Structure
from .tensor_types import Tensor, DimListT

# Concrete scope implementations
from .try_scope import TryScope
from .named_scope import NamedScope
from .null_scope import NullScope

# Predicates
from .argument import Argument, ArgumentRef, _1, _2, _3, _4, _5, _6, _7, _8, _9
from .predicate import Predicate, PredicateFn, predicate, BindArg, BindArguments, _get_arg_index
from .predicate_require import PredicateRequire, pred_require
from .predicate_expr import PredicateExpr, pred_expr
from .predicate_all import PredicateAll, pred_all
from .reordered_predicate import ReorderedPredicate
