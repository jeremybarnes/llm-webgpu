from templates import *
import numpy as np
import unittest

class TestScopes(unittest.TestCase):
    
    def test_path(self):
        p = Path('x')

        assert str(p) == "['x']"

    #test_path()

    def test_aliases_try_dim(self):
        scope1 = NullScope()

        with scope1.enter('test1_dim') as outer:
            x = outer.dim('x')
            y = outer.dim('y')

            with outer.mutate('inner1') as inner:
                inner.resolve(x, y)

                assert inner.aliases(x) == {x.ref(),y.ref()}
                assert inner.aliases(y) == {x.ref(),y.ref()}
                assert not inner.resolved(x)
                assert not inner.resolved(y)

                print(inner.values())

                inner.commit()

            print(outer.values())
            assert outer.aliases(x) == {x.ref(),y.ref()}
            assert outer.aliases(y) == {x.ref(),y.ref()}
            assert not outer.resolved(x)
            assert not outer.resolved(y)

            with outer.mutate('inner2') as inner:
                inner.resolve(x, 1)

                print(inner.values())

                assert inner.aliases(x) == set()
                assert inner.aliases(y) == set()
                assert inner.resolved(x)
                assert inner.resolved(y)
                assert inner.value(x) == 1
                assert inner.value(y) == 1

                assert outer.aliases(x) == {x.ref(),y.ref()}
                assert outer.aliases(y) == {x.ref(),y.ref()}
                assert not outer.resolved(x)
                assert not outer.resolved(y)

                inner.commit()

            assert outer.aliases(x) == set()
            assert outer.aliases(y) == set()
            assert outer.resolved(x)
            assert outer.resolved(y)
            assert outer.value(x) == 1
            assert outer.value(y) == 1

    #test_aliases_try_dim()

    def test_aliases_try_compounds(self):
        scope1 = NullScope()

        with scope1.enter('test1_shape') as outer:
            x = outer.shape('x')
            y = outer.shape('y')

            with outer.mutate('inner1') as inner:
                inner.resolve(x, y)

                assert inner.aliases(x) == {x.ref(),y.ref()}
                assert inner.aliases(y) == {x.ref(),y.ref()}
                assert not inner.resolved(x)
                assert not inner.resolved(y)

                print(inner.values())

                inner.commit()

            print(outer.values())
            assert outer.aliases(x) == {x.ref(),y.ref()}
            assert outer.aliases(y) == {x.ref(),y.ref()}
            assert not outer.resolved(x)
            assert not outer.resolved(y)

            with outer.mutate('inner2') as inner:
                inner.resolve(x, [1])

                print(inner.values())

                assert inner.aliases(x) == set()
                assert inner.aliases(y) == set()
                assert inner.resolved(x)
                assert inner.resolved(y)
                assert inner.value(x) == [1]
                assert inner.value(y) == [1]

                assert outer.aliases(x) == {x.ref(),y.ref()}
                assert outer.aliases(y) == {x.ref(),y.ref()}
                assert not outer.resolved(x)
                assert not outer.resolved(y)

                inner.commit()

            assert outer.aliases(x) == set()
            assert outer.aliases(y) == set()
            assert outer.resolved(x)
            assert outer.resolved(y)
            assert outer.value(x) == [1]
            assert outer.value(y) == [1]

        scope1t = NullScope()

        with scope1t.enter('test1_tensor') as outer:
            x = outer.tensor('x')
            y = outer.tensor('y')

            with outer.mutate('inner1') as inner:
                inner.resolve(x, y)

                assert inner.aliases(x) == {x.ref(),y.ref()}
                assert inner.aliases(y) == {x.ref(),y.ref()}
                assert not inner.resolved(x)
                assert not inner.resolved(y)

                inner.commit()

            assert outer.aliases(x) == {x.ref(),y.ref()}
            assert outer.aliases(y) == {x.ref(),y.ref()}
            assert not outer.resolved(x)
            assert not outer.resolved(y)

            with outer.mutate('inner2') as inner:
                print('')
                print('')
                print('outer 1', outer.values())

                inner.resolve(x, np.array(1))

                print('inner 2', inner.values())

                assert inner.aliases(x) == set()
                assert inner.aliases(y) == set()
                assert inner.resolved(x)
                assert inner.resolved(y)
                assert inner.value(x) == np.array(1)
                assert inner.value(y) == np.array(1)

                print('outer 2', outer.values())
                assert outer.aliases(x) == {x.ref(),y.ref()}
                assert outer.aliases(y) == {x.ref(),y.ref()}
                assert not outer.resolved(x)
                assert not outer.resolved(y)

                inner.commit()

            assert outer.aliases(x) == set()
            assert outer.aliases(y) == set()
            assert outer.resolved(x)
            assert outer.resolved(y)
            assert outer.value(x) == np.array(1)
            assert outer.value(y) == np.array(1)


    #test_aliases_try_compounds()

    def test_aliases_named(self):
        scope = NullScope()

        with scope.enter('test1') as outer:
            x = outer.tensor('x')
            y = outer.tensor('y')

            with outer.mutate('try') as inner:
                inner.resolve(x, y)

                assert inner.aliases(x) == {x.ref(),y.ref()}
                assert inner.aliases(y) == {x.ref(),y.ref()}
                assert not inner.resolved(x)
                assert not inner.resolved(y)

                inner.resolve(x, np.array(1))

                assert inner.aliases(x) == set()
                assert inner.aliases(y) == set()
                assert inner.resolved(x)
                assert inner.resolved(y)
                assert inner.value(x) == np.array(1)
                assert inner.value(y) == np.array(1)

                inner.commit()

            assert outer.aliases(x) == set()
            assert outer.aliases(y) == set()
            assert outer.resolved(x)
            assert outer.resolved(y)
            assert outer.value(x) == np.array(1)
            assert outer.value(y) == np.array(1)

        with scope.enter('test2') as outer:
            x = outer.tensor('x')
            y = outer.dim('y')

            with outer.mutate('try') as inner:
                try:
                    inner.resolve(x, y)
                    print(inner.values())
                    assert not "modification of symbol type should throw"
                except MatchError as e:
                    pass

                print(inner.values())

                assert inner.aliases(x) == {x.ref()}
                assert inner.aliases(y) == {y.ref()}
                assert not inner.resolved(x)
                assert not inner.resolved(y)

                inner.commit()

            assert outer.aliases(x) == {x.ref()}
            assert outer.aliases(y) == {y.ref()}
            assert not outer.resolved(x)
            assert not outer.resolved(y)

        with scope.enter('test3') as outer:
            x = outer.tensor('x')
            y = outer.tensor('y')
            z = outer.tensor('z')

            with outer.mutate('try') as inner:
                inner.resolve(x,y)
                print('resolve1', inner.values())
                print('')
                print('')
                inner.resolve(y,z)
                print('')
                print('')

                print('resolve2', inner.values())

                assert inner.aliases(x) == {x.ref(),y.ref(),z.ref()}
                assert inner.aliases(y) == {x.ref(),y.ref(),z.ref()}
                assert inner.aliases(z) == {x.ref(),y.ref(),z.ref()}
                assert not inner.resolved(x)
                assert not inner.resolved(y)
                assert not inner.resolved(z)

                inner.commit()

            assert outer.aliases(x) == {x.ref(),y.ref(),z.ref()}
            assert outer.aliases(y) == {x.ref(),y.ref(),z.ref()}
            assert outer.aliases(z) == {x.ref(),y.ref(),z.ref()}
            assert not outer.resolved(x)
            assert not outer.resolved(y)
            assert not outer.resolved(z)


    #test_aliases_named()


    def test_tryscope(self):
        scope = NullScope()

        with scope.enter('test1') as inner:
            x = inner.tensor('x')
            with inner.mutate('inner') as trying:
                trying.resolve(x, np.array(1))
                print(trying.values())
                assert trying.resolved(x)
                assert not inner.resolved(x)
                trying.commit()
            assert inner.resolved(x)

        with scope.enter('test2') as inner:
            x = inner.tensor('x')
            with inner.mutate('inner') as trying:
                trying.resolve(x, np.array(1))
                assert trying.resolved(x)
                assert not inner.resolved(x)
            assert not inner.resolved(x)

    #test_tryscope()

    def test_resolve(self):
        scope = NullScope()

        print('resolve test 1')
        with scope.enter('test1') as outer:
            x = outer.tensor('x')
            y = outer.tensor('y')

            with outer.mutate('try') as inner:
                inner.resolve(x, np.array(1))
                inner.resolve(y, np.array(2))

                assert inner.resolved(x)
                assert inner.resolved(y)
                assert inner.value(x) == np.array(1)
                assert inner.value(y) == np.array(2)

                inner.commit()

            assert outer.resolved(x)
            assert outer.resolved(y)
            assert outer.value(x) == np.array(1)
            assert outer.value(y) == np.array(2)

        print('resolve test 2')
        with scope.enter('test2') as outer:
            x = outer.tensor('x')
            y = outer.tensor('y')

            with outer.mutate('try') as inner:
                print('---- 1')
                inner.resolve(x, y)
                print(inner.values())

                print('---- 2')
                inner.resolve(y, np.array(2))
                print(inner.values())

                assert inner.resolved(x)
                assert inner.resolved(y)

                assert inner.value(x) == np.array(2)
                assert inner.value(y) == np.array(2)

                inner.commit()

            assert outer.resolved(x)
            assert outer.resolved(y)

            assert outer.value(x) == np.array(2)
            assert outer.value(y) == np.array(2)

        print('resolve test 3')
        with scope.enter('test3') as outer:
            x = outer.tensor('x')
            y = outer.tensor('y')

            with outer.mutate('try') as inner:
                inner.resolve(y, np.array(2))
                inner.resolve(x, y)

                print(inner.values())

                assert inner.resolved(x)
                assert inner.resolved(y)
                assert inner.value(x) == np.array(2)
                assert inner.value(y) == np.array(2)

                inner.commit()

            assert outer.resolved(x)
            assert outer.resolved(y)
            assert outer.value(x) == np.array(2)
            assert outer.value(y) == np.array(2)

        print('resolve test 4')
        with scope.enter('test4') as outer:
            x = outer.tensor('x')
            y = outer.tensor('y')

            with outer.mutate('try') as inner:
                inner.resolve(x, np.array(2))
                inner.resolve(x, y)

                print(inner.values())

                assert inner.resolved(x)
                assert inner.resolved(y)
                assert inner.value(x) == np.array(2)
                assert inner.value(y) == np.array(2)

                inner.commit()

            assert outer.resolved(x)
            assert outer.resolved(y)
            assert outer.value(x) == np.array(2)
            assert outer.value(y) == np.array(2)

        print('resolve test 5')
        with scope.enter('test5') as outer:
            x = outer.tensor('x')
            y = outer.tensor('y')

            with outer.mutate('try') as inner:
                inner.resolve(x, np.array(2))
                inner.resolve(y, x)

                assert inner.resolved(x)
                assert inner.resolved(y)
                assert inner.value(x) == np.array(2)
                assert inner.value(y) == np.array(2)

                inner.commit()

            assert outer.resolved(x)
            assert outer.resolved(y)
            assert outer.value(x) == np.array(2)
            assert outer.value(y) == np.array(2)


    #test_resolve()        

    def test_commit_aliases(self):
        scope = NullScope()

        with scope.enter('test1') as outer:
            x = outer.dim('x')
            y = outer.dim('y')

            with outer.mutate('try') as inner:
                # Alias them within the scope
                inner.resolve(x, y)

                # Aliases should be visible inside the scope
                assert inner.aliases(x) == {x.ref(),y.ref()}
                assert inner.aliases(y) == {x.ref(),y.ref()}

                # But not in the outer scope
                assert outer.aliases(x) == {x.ref()}
                assert outer.aliases(y) == {y.ref()}

                # Commit; they should be visible everywhere
                inner.commit()

            # Aliases should now be visible outside the scope
            assert outer.aliases(x) == {x.ref(),y.ref()}
            assert outer.aliases(y) == {x.ref(),y.ref()}

        print('---------------------------------')

        # Double nested; ensure that aliases are properly transmitted in
        # multiply nested try scopes
        with scope.enter('test1') as outer:
            x = outer.dim('x')
            y = outer.dim('y')

            with outer.mutate('try1') as middle:
                with middle.mutate('try2') as inner:
                    # Alias them within the scope
                    inner.resolve(x, y)

                    # Aliases should be visible inside the scope
                    assert inner.aliases(x) == {x.ref(),y.ref()}
                    assert inner.aliases(y) == {x.ref(),y.ref()}

                    # But not in the outer scopes
                    assert middle.aliases(x) == {x.ref()}
                    assert middle.aliases(y) == {y.ref()}
                    assert outer.aliases(x) == {x.ref()}
                    assert outer.aliases(y) == {y.ref()}

                    print('inner values', inner.values())
                    print('middle values', middle.values())

                    # Commit; they should be visible in the middle now
                    inner.commit()

                    print('committed middle', middle.values())

                # Aliases should now be visible inside the middle scope
                assert middle.aliases(x) == {x.ref(),y.ref()}
                assert middle.aliases(y) == {x.ref(),y.ref()}

                # But not in the outer scope
                assert outer.aliases(x) == {x.ref()}
                assert outer.aliases(y) == {y.ref()}

                # Commit from the middle to the outer
                middle.commit()

            # Aliases should now be visible in the outer scope too
            assert outer.aliases(x) == {x.ref(),y.ref()}
            assert outer.aliases(y) == {x.ref(),y.ref()}

    #test_commit_aliases()

    def test_structured_resolve(self):
        scope = NullScope()

        with scope.enter('test1') as outer:
            # Square is two sides the same length
            side = outer.dim('s')
            square = outer.shape('sq')

            with outer.mutate('try') as inner:
                print('')
                print('')
                inner.resolve(square, [side, side])
                print('')
                print('')
                print(inner.values())

                # Not resolved, as the sides haven't been resolved yet
                assert not inner.resolved(square)
                assert not inner.resolved(square[0])
                assert not inner.resolved(square[1])

                # But the length is
                assert inner.resolved(square._len)

                aliases = {side.ref(), square[0].ref(), square[1].ref()}

                print('aliases', inner.aliases(side), aliases)
                assert inner.aliases(side) == aliases
                assert inner.aliases(square[0]) == aliases
                assert inner.aliases(square[1]) == aliases

                print('')
                print('')
                inner.resolve(side, 3)
                print('')
                print('')

                print(inner.values())

                assert inner.resolved(square)
                assert inner.value(square) == [3,3]

        with scope.enter('test2') as outer:
            # Square is two sides the same length
            side = outer.dim('s')
            square = outer.shape('sq')

            with outer.mutate('try') as inner:
                inner.resolve(square, [side, side])

                inner.resolve(square, [3, side])

                assert inner.resolved(side)
                assert inner.resolved(square)
                assert inner.value(square) == [3,3]

        with scope.enter('test3') as outer:
            # Square is two sides the same length
            side = outer.dim('s')
            square = outer.shape('sq')

            with outer.mutate('try') as inner:
                inner.resolve(square, [side, side])

                try:
                    # Shouldn't work because we've constrained it
                    print('------ before resolve')
                    inner.resolve(square, [3, 4])
                    assert False
                except MatchError as e:
                    print('got MatchError as expected')
                    pass

                print(inner.values())

                # Make sure that we didn't keep the non-resolved values
                assert not inner.resolved(side)
                assert not inner.resolved(square)

    #test_structured_resolve()

    # Test the children() function, ensuring that all of the child values
    # are found.
    def test_children(self):
        scope = NullScope()

        # Verify basics
        with scope.enter('test1') as outer:
            x = outer.dim('x')
            y = outer.dim('y')

            print('outer.children(outer)', outer.children(outer))
            assert outer.children(outer) == {x.ref(), y.ref()}
            print('outer.children(x)', outer.children(x))
            assert outer.children(x) == set()

            with outer.mutate('trans') as inner:
                assert inner.children(outer) == {x.ref(), y.ref()}
                assert inner.children(x) == set()

                z = inner.dim('z')

                assert inner.children(outer) == {x.ref(), y.ref()}
                assert inner.children(inner) == {z.ref()}

                inner.commit()

            assert outer.children(outer) == {x.ref(), y.ref()}
        

if __name__ == '__main__':
    unittest.main()
