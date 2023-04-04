from templates import *
from typing import Iterator, List, Sequence
import unittest

class TestPredicates(unittest.TestCase):

    def test_pred_expr(self):

        arg1 = Argument(DimT, 'arg1')
        arg2 = Argument(DimT, 'arg2')
        res = Argument(BoolT, 'result')

        expr = equals(arg1, arg2)

        pred = pred_expr(res, [arg1, arg2], expr)

        print('pred args', pred.arguments())
        assert list(pred.arguments().values()) == [Argument(BoolT, 'result'), Argument(DimT, 'arg1'), Argument(DimT, 'arg2')]

        # No particular reason why this shouldn't work, but currently the deduplication
        # of result and an argument in arguments() isn't implemented so it doesn't
        try:
            pred2 = pred_expr(arg1, [arg1, arg2], expr)
            assert False, "binding result to an argument shouldn't work"
        except AssertionError as e:
            raise
        except:
            pass

    def test_reordered_predicate(self):

        # Simple testing predicate that asserts the third argument is an array of
        # the first two
        @predicate
        def my_pred(scope: Scope, arg1: DimT, arg2: DimT, arg3: DimListT) -> Iterator[Scope]:
            with scope.mutate('my_pred') as inner:
                inner.resolve(arg3, [arg1, arg2])
                yield inner

        assert str(my_pred) == 'my_pred(arg1: DimT, arg2: DimT, arg3: DimListT)'

        root = NullScope()

        def test_can_bind(pred: Predicate,
                          vals: List[Any]):
            assert len(pred.arguments()) == len(vals)

            outer = NamedScope(root, 'test_can_bind')
            print('outer scope')
            print('-----------')
            outer.dump()

            with outer.mutate('can_bind') as inner:
                args: List[Symbol] = []

                # Each argument gets instantiated and resolved
                for a,v in zip(pred.arguments().values(), vals):
                    print('resolving', a, 'to', v)
                    sym = a._in(inner)
                    inner.resolve(sym, v)
                    args.append(sym)

                print('inner scope')
                print('-----------')
                inner.dump()


                print('args')
                print('----')
                print(repr(args))

                # Finally, apply it
                has_applied = False
                for applied in pred.apply(inner, *args):
                    # here we should check that it matches
                    assert not has_applied
                    has_applied = True
                    #print('applied', applied.values())
                    applied.dump()
                    ...

                assert has_applied

        # check that the predicate works
        test_can_bind(my_pred, [1, 2, [1,2]])

        _arg1 = ArgumentRef(DimT, 'arg1')
        __arg1 = Argument(DimT, 'arg1')
        _arg2 = ArgumentRef(DimT, 'arg2')
        __arg2 = Argument(DimT, 'arg2')
        _arg3 = ArgumentRef(DimListT, 'arg3')
        _arg4 = ArgumentRef(DimT, 'arg4')  # extra (missing) argument

        old_args = my_pred.arguments()

        print('old_args', list(old_args.values()))

        assert list(old_args.values()) == [ _arg1, _arg2, _arg3 ]

        reordered1 = my_pred.reorder_args(old_args)

        assert list(reordered1.arguments().values()) == [ _arg1, _arg2, _arg3 ]

        test_can_bind(reordered1, [1, 2, [1,2]])

        new_args = BindArguments([('arg2', _arg2), ('arg1', _arg1), ('arg3', _arg3)])

        reordered2 = my_pred.reorder_args(new_args)

        print(str(reordered2))

        assert str(reordered2) == 'my_pred(arg2: DimT, arg1: DimT, arg3: DimListT)'

        test_can_bind(reordered2, [1, 2, [2,1]])

        def reorder(pred: Predicate, args: Sequence[ArgumentRef]) -> Predicate:
            bind_args = [ (a._get_identifier(), a) for a in args ]
            return pred.reorder_args(BindArguments(bind_args))

        # Check that we can't reorder with the same argument twice
        # (disabled as the OrderedDict silently accepts it and it's undetectable
        # once we're in an OrderedDict).
        try:
            reorder(my_pred, [_arg1, _arg1, _arg2])
            #assert False, "Shouldn't be able to reorder with same argument twice"
        except BindError as b:
            pass
        except:
            raise

        # Check that we can't change the type of an argument
        try:
            reorder(my_pred, [ArgumentRef(DimListT, 'arg1'), _arg2, _arg3])
            assert False, "Shouldn't be able to change argument type"
        except BindError as b:
            pass
        except:
            raise

        # Check that we can handle free variables
        __free = Argument(DimT, 'free')
        free1 = pred_require([], __free == __free)

        print('free1', free1)

        test_can_bind(free1, [])

        # Check that an unresolvable system can't be bound
        free2 = pred_require([_arg1], expr_all(__arg1 == 1, __arg1 == 2))

        print('free2', free2)

        try:
            test_can_bind(free2, [6])
            assert False, "free2 shouldn't bind"
        except MatchError as e:
            print('free2 error:', e)
            pass
        except:
            raise


        # Check that an unresolvable system with free variables isn't bindable
        # Here, the value of free (which is a free variable) must be both 1 and 2
        # We link it to the argument to ensure it's right
        free2a = pred_require([_arg1, _arg2], expr_all(__arg1 == __arg2, __arg2 == 1))
        print('free2a', free2a)

        # Should be able to bind this one...
        test_can_bind(free2a, [1, 1])

        # ... but not this one
        try:
            test_can_bind(free2a, [6, 3])
            assert False, "free2a shouldn't bind"
        except MatchError as e:
            print('free2a error:', e)
            pass
        except:
            raise


        # Check that an unresolvable system with free variables isn't bindable
        # Here, the value of free (which is a free variable) must be both 1 and 2
        # We link it to the argument to ensure it's right
        free3 = pred_require([_arg1], expr_all(__free == __arg1, __free == 1))
        print('free3', free3)

        # Should be able to bind this one...
        test_can_bind(free3, [1])

        # ... but not this one
        try:
            test_can_bind(free3, [6])
            assert False, "free3 shouldn't bind"
        except MatchError as e:
            print('free3 error:', e)
            pass
        except:
            raise

        # Check that an unresolvable system with free variables isn't bindable
        # Here, the value of free (which is a free variable) must be both 1 and 2
        # We link it to the argument to ensure it's right
        free4 = pred_require([_arg1], expr_all(__free == __arg1, __free == 1, __free == 2))

        print('\n\n')
        print('free4', free4)

        try:
            test_can_bind(free4, [6])
            assert False, "free4 shouldn't bind"
        except MatchError as e:
            pass
        except:
            raise

    def verify_is_square(self, pred: Predicate):
        print('pred', str(pred), repr(pred))
        assert list(pred.arguments().values()) == [ArgumentRef(DimT, 'left'), ArgumentRef(DimT, 'top')]

        # Check that is_square works for square
        outer = NullScope()

        with NamedScope(outer, 'verify_is_square ' + pred.name) as scope:
            got_inner = False

            # If neither are bound, we should get one match which is generic
            for inner in pred[scope]:
                print('got inner', inner.values())
                assert not got_inner
                got_inner = True
            assert got_inner

            left = scope.dim('left')
            top = scope.dim('top')

            got_inner = False
            for inner in pred[scope, left, top]:
                assert not got_inner
                got_inner = True

                with inner.mutate('set left') as nested:
                    nested.resolve(left, 2)
                    nested.dump()
                    print('value of top', nested.value(top))
                    assert nested.value(top) == 2
            assert got_inner

            # now we bind to left, leaving one argument (top)
            got_inner = False
            bound = pred(left, top)

            assert len(bound.arguments()) == 0

            print('\n\n')
            print('bound', bound)

            for inner in bound[scope]:
                assert not got_inner
                got_inner = True

                with inner.mutate('set left') as nested:
                    nested.resolve(left, 2)
                    assert nested.value(top) == 2
            assert got_inner

            # We bind left to 3, and check that top gets the right value
            got_inner = False
            bound = pred(3, top)
            for inner in bound[scope]:
                assert not got_inner
                got_inner = True

                assert inner.value(top) == 3
            assert got_inner

            # We bind left to 3, and check that top gets the right value
            got_inner = False
            bound = pred(3, top)
            for inner in bound[scope]:
                assert not got_inner
                got_inner = True

                assert inner.value(top) == 3
            assert got_inner

            # We bind top to 3, and check that left gets the right value
            got_inner = False
            bound = pred(left, 5)
            for inner in bound[scope]:
                assert not got_inner
                got_inner = True

                assert inner.value(left) == 5
            assert got_inner

            # We bind top to 3, and check that left gets the right value
            got_inner = False
            bound = pred(left, 5)
            for inner in bound[scope]:
                assert not got_inner
                got_inner = True

                assert inner.value(left) == 5
            assert got_inner

            # We bind them to incompatible values, and check that there is an
            # exception and no match
            bound = pred(3, 5)
            try:
                for inner in bound[scope]:
                    assert False, "should be no matches for incompatible bind"
                assert False, "should throw for incompatible bind"
            except MatchError as e:
                pass

            # Attempt to partially bind should fail (at least for now)
            if False:
                try:
                    bound = pred(3)
                    assert False, "Partial bind should fail (for now)"
                except MatchError as e:
                    print('exception', e)
                    pass
                assert False

        # Bind with two differently named arguments, that are reversed

        outer = NullScope()

        print('\n\n')
        with NamedScope(outer, 'test1') as scope:
            left = scope.dim('left')
            top = scope.dim('top')
            bound0 = pred(left)
            print('bound0', bound0)
            bound1 = pred(left)(top)
            print('bound1', bound1)
            bound2 = pred(left,top)
            print('bound2', bound1)

            assert bound1.arguments() == bound2.arguments()



    def test_binding(self):

        # Basic predicate that tests for squareness, for the test
        @predicate
        def is_square(scope: Scope, left: DimT, top: DimT) -> Iterator[Scope]:
            with scope.mutate('is_square') as inner:
                inner.resolve(left, top)
                yield inner

        assert str(is_square) == 'is_square(left: DimT, top: DimT)'
        self.verify_is_square(is_square)

        # Another way of doing it, via a condition
        @predicate
        def is_square2(scope: Scope, left: DimT, top: DimT) -> Iterator[Scope]:
            with scope.mutate('is_square') as inner:
                inner.condition("equal_sides", equals(left, top))
                yield inner

        assert str(is_square2) == 'is_square2(left: DimT, top: DimT)'
        self.verify_is_square(is_square2)

        _left = Argument(DimT, 'left')
        _top = Argument(DimT, 'top')

        is_square3 = pred_require([_left, _top], _left == _top, name='is_square3')
        print('is_square3', str(is_square3))
        assert str(is_square3) == 'is_square3(left: DimT, top: DimT)'

        self.verify_is_square(is_square3)

    def test_pred_all(self):

        _left = Argument(DimT, 'left')
        _top = Argument(DimT, 'top')
        _free = Argument(DimT, 'free')

        # Make sure a pred_all with a single predicate matches the same as the
        # underlying predicate
        is_square4 = pred_all([_left, _top],
                              pred_require([_left, _top],_left == _top),
                              name='is_square4')
        print('is_square4', str(is_square4))
        assert str(is_square4) == 'is_square4(left: DimT, top: DimT)'

        self.verify_is_square(is_square4)

        # Twice the same predicate should be equivalent to once; this part tests
        # the recursion
        is_square5 = pred_all([_left, _top],
                              pred_require([_left, _top], _left == _top),
                              pred_require([_top, _left], _top == _left),
                              name='is_square5')

        print('is_square5', str(is_square5))
        assert str(is_square5) == 'is_square5(left: DimT, top: DimT)'

        self.verify_is_square(is_square5)

        print('-------------------')
        # Add a free variable which doesn't affect the outcome
        is_square6 = pred_all([_left, _top],
                              pred_require([_left, _top], _left == _top),
                              pred_require([_free], _free == _free),
                              name='is_square6')

        print('is_square6', str(is_square6))
        assert str(is_square6) == 'is_square6(left: DimT, top: DimT)'

        self.verify_is_square(is_square6)

        # Now we introduce a free variable in the middle so that the left and top
        # are not directly assigned
        is_square7 = pred_all([_left, _top],
                              pred_require([_left, _top], _left == _top),
                              pred_require([_free], _free == _free),
                              name='is_square7')

        print('is_square7', str(is_square7))
        assert str(is_square7) == 'is_square7(left: DimT, top: DimT)'

        self.verify_is_square(is_square7)

if __name__ == '__main__':
    unittest.main()

