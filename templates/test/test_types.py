from templates import *
import numpy as np
import unittest

class TestTypes(unittest.TestCase):
    
    # Verify that a shape can be resolved twice with a different length
    # in a different part of the context.  This means that there can be
    # no state stored in the symbol itself; the state must all be in
    # the context.
    def test_shape_multiple_lengths(self):
        scope1 = NullScope()

        with scope1.enter('test1') as outer:
            x = outer.shape('x')
            with outer.mutate('inner1') as inner:
                inner.resolve(x, [1,2])

                print(outer.values())
                print(inner.values())

                assert inner.resolved(x)
                assert inner.value(x) == [1,2]
                assert not outer.resolved(x)

                # no commit, so the effect of the resolution should be null

            with outer.mutate('inner2') as inner:
                inner.resolve(x, [1,2,3])

                assert inner.resolved(x)
                assert not outer.resolved(x)
                assert inner.value(x) == [1,2,3]

    #test_shape_multiple_lengths()

    # Test the ellipsis operator for shapes

    def test_shape_ellipsis(self):
        scope1 = NullScope()

        # Ellipsis in middle is an error (they can only be at the end to avoid
        # an open parameter)
        with scope1.enter('test0') as outer:
            x = outer.shape('x')
            with outer.mutate('inner1') as inner:
                try:
                    inner.resolve(x, [1,...,2])
                except MatchError as e:
                    print('got error', e)
                    pass

        with scope1.enter('test1') as outer:
            x = outer.shape('x')
            with outer.mutate('inner1') as inner:
                inner.resolve(x, [1,2,...])

                assert not inner.resolved(x)
                assert inner.resolved(x[0])
                assert inner.value(x[0]) == 1
                assert inner.resolved(x[1])
                assert inner.value(x[1]) == 2
                assert not inner.resolved(x._len)

                # Should also work for simply 1,2
                inner.resolve(x, [1,2])

                assert inner.resolved(x)
                assert inner.value(x) == [1,2]

    #test_shape_ellipsis()

    # Test the parameter map type
    def test_parameter_map(self):
        scope1 = NullScope()

        with scope1.enter('test1') as outer:
            params = outer.symbol(ParameterMapT, 'p')
            d = outer.dim('d')

            with outer.mutate('inner1') as inner:
                inner.resolve(params['x'], 3)
                print(inner.values())
                assert inner.resolved(params['x'])
                assert inner.value(params['x']) == 3

                inner.resolve(params['y'], 4)
                assert inner.resolved(params['y'])
                assert inner.value(params['y']) == 4

                assert not inner.resolved(params)

                inner.resolve(params.complete, True)

                print(inner.values())
                assert inner.resolved(params)
                print(inner.value(params))
                print(inner._values)
                assert inner.value(params) == {'x': 3, 'y': 4}

        with scope1.enter('test2') as outer:
            params = outer.symbol(ParameterMapT, 'p')
            d = outer.dim('d')

            with outer.mutate('inner1') as inner:
                inner.resolve(params['x'], 3)
                inner.resolve(params['y'], d)
                inner.resolve(params.complete, True)
                assert inner.resolved(params.complete)

                # d doesn't have a value, so shouldn't be resolved
                assert not inner.resolved(params)

                # now give it a value, it should be resolved
                inner.resolve(d, 4)

                assert inner.resolved(params.complete)
                print(inner.values())
                assert inner.resolved(params)
                assert inner.value(params) == {'x': 3, 'y': 4}

        with scope1.enter('test3') as outer:
            params = outer.symbol(ParameterMapT, 'p')
            d = outer.dim('d')

            with outer.mutate('inner1') as inner:
                inner.resolve(params, {'x': 3, 'y':4})
                assert inner.resolved(params.complete)

                assert inner.resolved(params)
                assert inner.value(params) == {'x': 3, 'y': 4}

        with scope1.enter('test4') as outer:
            params = outer.symbol(ParameterMapT, 'p')
            d = outer.dim('d')

            with outer.mutate('inner1') as inner:
                inner.resolve(params, {'x': 3, 'y':d})
                assert inner.resolved(params.complete)

                # d doesn't have a value, so shouldn't be resolved
                assert not inner.resolved(params)

                # now give it a value, it should be resolved
                inner.resolve(d, 4)

                assert inner.resolved(params)
                assert inner.value(params) == {'x': 3, 'y': 4}

                inner.resolve(params, {'x': 3, 'y': 4})

                assert inner.resolved(params)
                assert inner.value(params) == {'x': 3, 'y': 4}


        # Verify that we can't modify the keys over multiple resolutions
        with scope1.enter('test5') as outer:
            params = outer.symbol(ParameterMapT, 'p')
            d = outer.dim('d')

            with outer.mutate('inner1') as inner:
                inner.resolve(params, {'x': 3, 'y':d})
                assert inner.resolved(params.complete)

                # Resolve with a new key should fail
                try:
                    inner.resolve(params, {'z': 3, 'y':d})
                    assert False
                except MatchError as e:
                    print('exception 1', e)

                # Resolve with a missing key should fail
                try:
                    inner.resolve(params, {'x': 3})
                    assert False
                except MatchError as e:
                    print('exception 2', e)

                print(inner.values())
                print('hello')

        # Verify that complex resolution works
        with scope1.enter('test6') as outer:
            params = outer.symbol(ParameterMapT, 'p')
            d = outer.dim('d')

            with outer.mutate('inner1') as inner:
                inner.resolve(params, {'x': 3, 'y': d})
                assert inner.resolved(params.complete)

                # This should resolve x and y to 3, as that's the only way that the
                # whole thing can be resolved.
                inner.resolve(params, {'x': d, 'y': d})

                assert inner.resolved(d)
                assert inner.value(d) == 3
                assert inner.resolved(params)
                assert inner.value(params) == {'x': 3, 'y': 3}

        # Verify that we can resolve a structured parameter map
        with scope1.enter('test7') as outer:
            params = outer.symbol(ParameterMapT, 'p')
            d = outer.dim('d')

            with outer.mutate('inner1') as inner:
                print('')
                print('')
                inner.resolve(params['square'], [d,d])
                print(inner.values())
                assert not inner.resolved(params['square'])
                assert not inner.resolved(params.complete)
                assert not inner.resolved(params)
                assert not inner.resolved(d)

                print('')
                print('')
                inner.resolve(d, 3)

                print(inner.values())
                assert inner.resolved(params['square'])
                assert inner.value(params['square']) ==  [3,3]

                print('')
                print('')
                inner.resolve(params.complete, True)

                print(inner.values())
                assert inner.resolved(d)
                assert inner.resolved(params)
                assert inner.value(params) == {'square': [3,3]}

            with outer.mutate('inner2') as inner:
                print('')
                print('')
                inner.resolve(params, {'square': [d,d]})
                print('')
                print(inner.values())
                assert inner.resolved(params.complete)
                assert not inner.resolved(params)
                assert not inner.resolved(d)

                inner.resolve(d, 3)
                assert inner.resolved(d)
                assert inner.resolved(params)
                assert inner.value(params['square']) ==  [3,3]
                assert inner.value(params) == {'square': [3,3]}

    # Verify that when a tensor is fully resolved, its value is properly
    # extracted
    def test_tensor_gradual_resolve(self):
        scope1 = NullScope()

        with scope1.enter('test1') as outer:
            x = outer.tensor('x')

            with outer.mutate('inner1') as inner:

                inner.resolve(x['dtype'], np.int32)

                assert inner.resolved(x['dtype'])
                assert not inner.resolved(x)

                inner.resolve(x['shape']['len'], 2)

                assert inner.resolved(x['shape']['len'])
                assert not inner.resolved(x['shape'])

                inner.resolve(x['shape'][0], 2)

                assert inner.resolved(x['shape'][0])
                assert not inner.resolved(x['shape'])

                inner.resolve(x['shape'][1], 2)

                assert inner.resolved(x['shape'][1])
                assert inner.resolved(x['shape'])

                assert not inner.resolved(x)

                try:
                    inner.resolve(x, np.zeros([2,2], dtype=np.float64))
                    assert False
                except MatchError as e:
                    print('got error', e)
                    pass

                print('')
                print('')
                print(inner.values())
                inner.resolve(x, np.zeros([2,2], dtype=np.int32))
                print(inner.values())

                assert inner.resolved(x)

        with scope1.enter('test2') as outer:
            x = outer.tensor('x')

            with outer.mutate('inner1') as inner:

                inner.resolve(x['dtype'], np.int32)

                assert inner.resolved(x['dtype'])
                assert not inner.resolved(x)

                try:
                    inner.resolve(x, np.zeros([2,2], dtype=np.float64))
                    assert False
                except MatchError as e:
                    print('got error', e)
                    pass

                assert not inner.resolved(x)
                assert not inner.resolved(x['shape'])

        with scope1.enter('test3') as outer:
            x = outer.tensor('x')

            with outer.mutate('inner1') as inner:

                inner.resolve(x['shape']['len'], 3)

                assert inner.resolved(x['shape']['len'])
                assert not inner.resolved(x)
                assert not inner.resolved(x['shape'])

                try:
                    inner.resolve(x, np.zeros([2,2], dtype=np.float64))
                    assert False
                except MatchError as e:
                    print('got error', e)
                    pass

                assert not inner.resolved(x)
                assert not inner.resolved(x['dtype'])
                assert not inner.resolved(x['shape'])

        with scope1.enter('test4') as outer:
            x = outer.tensor('x')

            with outer.mutate('inner1') as inner:

                inner.resolve(x['shape']['len'], 3)

                assert inner.resolved(x['shape']['len'])
                assert not inner.resolved(x)
                assert not inner.resolved(x['shape'])

                inner.resolve(x, np.zeros([2,2,2], dtype=np.float64))

                assert inner.resolved(x)
                assert inner.resolved(x['dtype'])
                assert inner.resolved(x['shape'])

    def test_shape_partial_resolve(self):
        scope1 = NullScope()

        with scope1.enter('test') as outer:

            x = outer.shape('x')
            y = outer.shape('y')

            with outer.mutate('inner1') as inner:
                inner.resolve(x, y)

                inner.resolve(x['len'], 0)
                assert inner.resolved(x['len'])
                assert inner.value(x['len']) == 0
                assert inner.resolved(x)

                assert inner.resolved(y['len'])
                assert inner.value(y['len']) == 0
                assert inner.resolved(y)


        with scope1.enter('test0') as outer:
            x = outer.shape('x')
            y = outer.shape('y')

            with outer.mutate('inner1') as inner:
                inner.resolve(x, y)

                inner.resolve(x['len'], 0)

                assert inner.resolved(x['len'])
                assert inner.resolved(y['len'])
                assert inner.value(x['len']) == 0
                assert inner.value(y['len']) == 0


        with scope1.enter('test1') as outer:
            x = outer.shape('x')
            y = outer.shape('y')

            with outer.mutate('inner1') as inner:

                inner.resolve(x['len'], 0)
                inner.resolve(y['len'], 1)

                # Should not work; the shapes have different lengths
                try:
                    inner.resolve(x,y)
                    assert False
                except MatchError as e:
                    print('got error', e)


    def test_tensor_partial_resolve(self):
        scope1 = NullScope()

        with scope1.enter('test1_dim') as outer:
            x = outer.tensor('x')
            y = outer.tensor('y')

            with outer.mutate('inner1') as inner:
                inner.resolve(x['dtype'], np.int32)
                inner.resolve(y['dtype'], np.float32)

                # Make sure the alias doesn't work (it would require that they have
                # the same dtype, and we've already established that they are
                # different).

                # NOTE: maybe at some point we won't want this test to fail here;
                # although it's provably impossible that the system is consistent,
                # we may not want to pay the price to always determine this at the
                # earliest possible moment.  Any attempt to fully resolve x or y
                # will inevitably lead to failure.  For the moment, for the
                # ergonomics, we will require that once it's provably impossible,
                # it fail, hence this test is valid.

                inner.resolve(x, y)
                print(inner.values())

                vals_before = inner.values()
                try:
                    inner.resolve(x, np.zeros([2,2], np.int32))
                    print(inner.values())
                    assert False
                except MatchError as e:
                    print('got error', e)

                vals_after = inner.values()
                print('before', vals_before)
                print('after', vals_after)

    def test_partial_values(self):
        scope1 = NullScope()

        with scope1.enter('test1') as outer:
            x = outer.dim('x')

            assert outer.partial_value(x).ref() == x.ref()

            with outer.mutate('inner1') as inner:
                inner.resolve(x, 1)
                assert inner.partial_value(x) == 1

        with scope1.enter('test2') as outer:
            x = outer.shape('x')
            assert outer.partial_value(x) == [...]

        with scope1.enter('test3') as outer:
            x = outer.shape('x')
            y = outer.dim('y')

            with outer.mutate('inner1') as inner:

                assert inner.partial_value(x) == [...]

                inner.resolve(x, [y,...])

                assert inner.partial_value(x) == [y,...]

                inner.resolve(x, [y,y])

                assert inner.partial_value(x) == [y,y]

                inner.resolve(y, 1)

                assert inner.partial_value(x) == [1,1]

        with scope1.enter('test4') as outer:
            x = outer.tensor('x')

            with outer.mutate('inner1') as inner:

                print(inner.partial_value(x))

                inner.resolve(x.dtype, np.int32)

                print(inner.partial_value(x))

                inner.resolve(x.shape._len, 3)

                print(inner.partial_value(x))

                inner.resolve(x.shape[2], 3)

                print(inner.partial_value(x))

                
if __name__ == '__main__':
    unittest.main()
    
