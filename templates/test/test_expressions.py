from templates import *
import numpy as np
import unittest

class TestExpressions(unittest.TestCase):
    def test_basics(self):
        scope1 = NullScope()

        with scope1.enter('test1') as outer:
            x = outer.dim('x')
            y = outer.shape('sz')

            with outer.mutate('inner1') as inner:
                inner.resolve(y, [x])

                assert not inner.resolved(y)

                inner.resolve(x, 1)

                assert inner.resolved(y)
                assert inner.resolved(x)

        with scope1.enter('test2') as outer:
            x = outer.dim('x')
            y = outer.symbol(ArgumentListT, 'sz')

            with outer.mutate('inner1') as inner:
                inner.resolve(y, [x])

                assert not inner.resolved(y)

                inner.resolve(x, 1)

                assert inner.resolved(y)
                assert inner.resolved(x)

                assert inner.value(x) == 1
                assert inner.value(y) == [1]

    def test_expression(self):
        scope1 = NullScope()

        with scope1.enter('test1') as outer:
            x = outer.dim('x')
            y = outer.dim('y')

            with outer.mutate('inner1') as inner:
                print('')
                print('')

                inner.resolve(y, pow((2 * x + 3) // 2, 3))
                assert not inner.resolved(y)

                #assert 'test1.inner1.$expr.args' not in inner.values()

                print('')
                print('')
                print(inner.values())

                inner.resolve(x, 1)

                print(inner.values())

                assert inner.resolved(x)
                assert inner.resolved(y)

                assert inner.value(x) == 1
                assert inner.value(y) == 8

            with outer.mutate('inner3') as inner:
                # test right operators
                inner.resolve(x, 1)
                inner.resolve(y, 2 / x)
                print(inner.values())
                assert inner.value(y) == 2.0

            with outer.mutate('inner2') as inner:
                # test left operators
                inner.resolve(x, 1)
                inner.resolve(y, x / 2)
                assert inner.value(y) == 0.5

    def test_condition(self):
        scope1 = NullScope()

        # Ensure that condition works when the condition is defined before the inputs
        # are
        with scope1.enter('test1') as outer:
            x = outer.dim('x')

            with outer.mutate('inner1') as inner:

                inner.condition('c1', x > 2)
                print(inner.values())

                try:
                    inner.resolve(x, 1)
                    assert False
                except MatchError as e:
                    pass

                print(inner.values())
                assert not inner.resolved(x)

                inner.resolve(x, 3)

                assert inner.resolved(x)

        # Ensure that condition works when the condition is defined with all of the
        # inputs already resolved
        with scope1.enter('test2') as outer:
            x = outer.dim('x')

            with outer.mutate('inner1') as inner:

                inner.resolve(x, 1)

                try:
                    inner.condition('c1', x > 2)
                    assert False, "Condition should have failed immediately when added"
                except MatchError as e:
                    pass

    def test_equalities(self):

        scope1 = NullScope()

        # test basic equalities
        with scope1.enter('test1') as outer:
            x = outer.dim('x')
            y = outer.dim('y')

            expr = equals(x, y)

            assert expr.equalities() == [(x,y)]

        # test basic equalities
        with scope1.enter('test1') as outer:
            x = outer.dim('x')

            expr = equals(x, 1)

            print('expr.equalities()', expr.equalities())
            assert expr.equalities() == [(x,1)]


        # test expr_all's equalities
        with scope1.enter('test2') as outer:
            x = outer.dim('x')
            y = outer.dim('y')
            z = outer.dim('z')

            expr = expr_all(equals(x, y), equals(y, z))

            assert expr.equalities() == [(x,y), (y,z)]

        with scope1.enter('test2') as outer:
            x = outer.dim('x')
            y = outer.dim('y')
            z = outer.dim('z')

            expr = expr_all(equals(x, y), equals(y, z), equals(z, 1))

            assert expr.equalities() == [(x,y), (y,z), (z,1)]

    def test_expr_all(self):
        scope1 = NullScope()

        with scope1.enter('test1') as outer:
            x = outer.dim('x')

            with outer.mutate('inner1') as inner:

                inner.condition('c1', expr_all(x > 2, x < 2))
                print(inner.values())

                try:
                    inner.resolve(x, 1)
                    assert False
                except MatchError as e:
                    pass

                print(inner.values())
                assert not inner.resolved(x)

            # test short circuiting of true values
            with outer.mutate('inner2') as inner:

                inner.condition('c1', expr_all(True, True, True))
                print(inner.values())

                inner.resolve(x, 1)
                assert inner.resolved(x)

            # test short circuiting of false values with no inputs
            with outer.mutate('inner3') as inner:

                try:
                    inner.condition('c1', expr_all(False, True, True))
                    inner.resolve(x, 1)
                    assert False
                except MatchError as e:
                    pass

                assert not inner.resolved(x)


            # test short circuiting of false values with inputs
            with outer.mutate('inner4') as inner:

                try:
                    inner.condition('c1', expr_all(False, True, x == 1))
                    assert False
                except MatchError as e:
                    pass

                assert not inner.resolved(x)

if __name__ == '__main__':
    unittest.main()
