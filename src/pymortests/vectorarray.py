# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import product, chain
from numbers import Number

import pytest
import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.vectorarrays.interfaces import VectorSpaceInterface
from pymortests.fixtures.vectorarray import \
    (vector_array_without_reserve, vector_array, compatible_vector_array_pair_without_reserve,
     compatible_vector_array_pair, incompatible_vector_array_pair,
     picklable_vector_array_without_reserve, picklable_vector_array)
from pymortests.pickling import assert_picklable_without_dumps_function



def ind_complement(v, ind):
    if isinstance(ind, Number):
        ind = [ind]
    elif type(ind) is slice:
        ind = range(*ind.indices(len(v)))
    l = len(v)
    return sorted(set(range(l)) - {i if i >= 0 else l+i for i in ind})


def indexed(v, ind):
    if ind is None:
        return v
    elif type(ind) is slice:
        return v[ind]
    elif isinstance(ind, Number):
        return v[[ind]]
    elif len(ind) == 0:
        return np.empty((0, v.shape[1]), dtype=v.dtype)
    else:
        return v[ind]


def invalid_inds(v, length=None):
    yield None
    if length is None:
        yield len(v)
        yield [len(v)]
        yield -len(v)-1
        yield [-len(v)-1]
        yield [0, len(v)]
        length = 42
    if length > 0:
        yield [-len(v)-1] + [0, ] * (length - 1)
        yield list(range(length - 1)) + [len(v)]


def valid_inds(v, length=None):
    if length is None:
        yield []
        yield slice(None)
        yield slice(0, len(v))
        yield slice(0, 0)
        yield slice(-3)
        yield slice(0, len(v), 3)
        yield slice(0, len(v)//2, 2)
        yield list(range(-len(v), len(v)))
        yield list(range(int(len(v)/2)))
        yield list(range(len(v))) * 2
        length = 32
    if len(v) > 0:
        for ind in [-len(v), 0, len(v) - 1]:
            yield ind
        if len(v) == length:
            yield slice(None)
        np.random.seed(len(v) * length)
        yield list(np.random.randint(-len(v), len(v), size=length))
    else:
        if len(v) == 0:
            yield slice(0, 0)
        yield []


def valid_inds_of_same_length(v1, v2):
    if len(v1) == len(v2):
        yield slice(None), slice(None)
        yield list(range(len(v1))), list(range(len(v1)))
        yield (slice(0, len(v1)),) * 2
        yield (slice(0, 0),) * 2
        yield (slice(-3),) * 2
        yield (slice(0, len(v1), 3),) * 2
        yield (slice(0, len(v1)//2, 2),) * 2
    yield [], []
    if len(v1) > 0 and len(v2) > 0:
        yield 0, 0
        yield len(v1) - 1, len(v2) - 1
        yield -len(v1), -len(v2)
        yield [0], 0
        yield (list(range(min(len(v1), len(v2))//2)),) * 2
        np.random.seed(len(v1) * len(v2))
        for count in np.linspace(0, min(len(v1), len(v2)), 3).astype(int):
            yield (list(np.random.randint(-len(v1), len(v1), size=count)),
                   list(np.random.randint(-len(v2), len(v2), size=count)))
        yield slice(None), np.random.randint(-len(v2), len(v2), size=len(v1))
        yield np.random.randint(-len(v1), len(v1), size=len(v2)), slice(None)


def valid_inds_of_different_length(v1, v2):
    if len(v1) != len(v2):
        yield slice(None), slice(None)
        yield list(range(len(v1))), list(range(len(v2)))
    if len(v1) > 0 and len(v2) > 0:
        if len(v1) > 1:
            yield [0, 1], 0
            yield [0, 1], [0]
            yield [-1, 0, 1], [0]
            yield slice(0, -1), []
        if len(v2) > 1:
            yield 0, [0, 1]
            yield [0], [0, 1]
        np.random.seed(len(v1) * len(v2))
        for count1 in np.linspace(0, len(v1), 3).astype(int):
            count2 = np.random.randint(0, len(v2))
            if count2 == count1:
                count2 += 1
                if count2 == len(v2):
                    count2 -= 2
            if count2 >= 0:
                yield (list(np.random.randint(-len(v1), len(v1), size=count1)),
                       list(np.random.randint(-len(v2), len(v2), size=count2)))


def invalid_ind_pairs(v1, v2):
    for inds in valid_inds_of_different_length(v1, v2):
        yield inds
    for ind1 in valid_inds(v1):
        for ind2 in invalid_inds(v2, length=v1.len_ind(ind1)):
            yield ind1, ind2
    for ind2 in valid_inds(v2):
        for ind1 in invalid_inds(v1, length=v2.len_ind(ind2)):
            yield ind1, ind2


def ind_to_list(v, ind):
    if type(ind) is slice:
        return list(range(*ind.indices(len(v))))
    elif not hasattr(ind, '__len__'):
        return [ind]
    else:
        return ind


def test_empty(vector_array):
    with pytest.raises(Exception):
        vector_array.empty(-1)
    for r in (0, 1, 100):
        v = vector_array.empty(reserve=r)
        assert v.space == vector_array.space
        assert len(v) == 0
        try:
            assert v.to_numpy().shape == (0, v.dim)
        except NotImplementedError:
            pass


def test_zeros(vector_array):
    with pytest.raises(Exception):
        vector_array.zeros(-1)
    for c in (0, 1, 2, 30):
        v = vector_array.zeros(count=c)
        assert v.space == vector_array.space
        assert len(v) == c
        if min(v.dim, c) > 0:
            assert max(v.sup_norm()) == 0
            assert max(v.l2_norm()) == 0
        try:
            assert v.to_numpy().shape == (c, v.dim)
            assert np.allclose(v.to_numpy(), np.zeros((c, v.dim)))
        except NotImplementedError:
            pass


def test_ones(vector_array):
    with pytest.raises(Exception):
        vector_array.ones(-1)
    for c in (0, 1, 2, 30):
        v = vector_array.ones(count=c)
        assert v.space == vector_array.space
        assert len(v) == c
        if min(v.dim, c) > 0:
            assert np.allclose(v.sup_norm(), np.ones(c))
            assert np.allclose(v.l2_norm(), np.full(c, np.sqrt(v.dim)))
        try:
            assert v.to_numpy().shape == (c, v.dim)
            assert np.allclose(v.to_numpy(), np.ones((c, v.dim)))
        except NotImplementedError:
            pass


def test_full(vector_array):
    with pytest.raises(Exception):
        vector_array.full(9, -1)
    for c in (0, 1, 2, 30):
        for val in (-1e-3,0,7):
            v = vector_array.full(val, count=c)
            assert v.space == vector_array.space
            assert len(v) == c
            if min(v.dim, c) > 0:
                assert np.allclose(v.sup_norm(), np.full(c, abs(val)))
                assert np.allclose(v.l2_norm(), np.full(c, np.sqrt(val**2 * v.dim)))
            try:
                assert v.to_numpy().shape == (c, v.dim)
                assert np.allclose(v.to_numpy(), np.full((c, v.dim), val))
            except NotImplementedError:
                pass


def test_random_uniform(vector_array):
    with pytest.raises(Exception):
        vector_array.random(-1)
    for c in (0, 1, 2, 30):
        for low in (-1e-3, 0, 7):
            for high in (0.5, 7):
                if c > 0 and high <= low:
                    with pytest.raises(ValueError):
                        vector_array.random(c, low=low, high=high)
                    continue
                seed = 123
                try:
                    v = vector_array.random(c, low=low, high=high, seed=seed)
                except ValueError as e:
                    if high <= low:
                        continue
                    raise e
                assert v.space == vector_array.space
                assert len(v) == c
                if min(v.dim, c) > 0:
                    assert np.all(v.sup_norm() < max(abs(low), abs(high)))
                try:
                    x = v.to_numpy()
                    assert x.shape == (c, v.dim)
                    assert np.all(x < high)
                    assert np.all(x >= low)
                except NotImplementedError:
                    pass
                vv = vector_array.random(c, distribution='uniform', low=low, high=high, seed=seed)
                assert np.allclose((v - vv).sup_norm(), 0.)


def test_random_normal(vector_array):
    with pytest.raises(Exception):
        vector_array.random(-1)
    for c in (0, 1, 2, 30):
        for loc in (-1e-3, 0, 7):
            for scale in (-1, 0.5, 7):
                if c > 0 and scale <= 0:
                    with pytest.raises(ValueError):
                        vector_array.random(c, 'normal', loc=loc, scale=scale)
                    continue
                seed = 123
                try:
                    v = vector_array.random(c, 'normal', loc=loc, scale=scale, seed=seed)
                except ValueError as e:
                    if scale <= 0:
                        continue
                    raise e
                assert v.space == vector_array.space
                assert len(v) == c
                try:
                    x = v.to_numpy()
                    assert x.shape == (c, v.dim)
                    import scipy.stats
                    n = x.size
                    if n == 0:
                        continue
                    # test for expected value
                    norm = scipy.stats.norm()
                    gamma = 1 - 1e-7
                    alpha = 1 - gamma
                    lower = np.sum(x)/n - norm.ppf(1 - alpha/2) * scale / np.sqrt(n)
                    upper = np.sum(x)/n + norm.ppf(1 - alpha/2) * scale / np.sqrt(n)
                    assert lower <= loc <= upper
                except NotImplementedError:
                    pass
                vv = vector_array.random(c, 'normal', loc=loc, scale=scale, seed=seed)
                assert np.allclose((v - vv).sup_norm(), 0.)


def test_from_numpy(vector_array):
    try:
        d = vector_array.to_numpy()
    except NotImplementedError:
        return
    try:
        v = vector_array.space.from_numpy(d)
        assert np.allclose(d, v.to_numpy())
    except NotImplementedError:
        pass


def test_shape(vector_array):
    v = vector_array
    assert len(vector_array) >= 0
    assert vector_array.dim >= 0
    try:
        assert v.to_numpy().shape == (len(v), v.dim)
    except NotImplementedError:
        pass


def test_space(vector_array):
    v = vector_array
    assert isinstance(v.space, VectorSpaceInterface)
    assert v in v.space


def test_getitem_repeated(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        v_ind = v[ind]
        v_ind_copy = v_ind.copy()
        assert not v_ind_copy.is_view
        for ind_ind in valid_inds(v_ind):
            v_ind_ind = v_ind[ind_ind]
            assert np.all(almost_equal(v_ind_ind, v_ind_copy[ind_ind]))


def test_copy(vector_array):
    v = vector_array
    for ind in chain(valid_inds(v), [None]):
        for deep in (True, False):
            if ind is None:
                c = v.copy(deep)
                assert len(c) == len(v)
            else:
                c = v[ind].copy(deep)
                assert len(c) == v.len_ind(ind)
            assert c.space == v.space
            if ind is None:
                assert np.all(almost_equal(c, v))
            else:
                assert np.all(almost_equal(c, v[ind]))
            try:
                assert np.allclose(c.to_numpy(), indexed(v.to_numpy(), ind))
            except NotImplementedError:
                pass


def test_copy_repeated_index(vector_array):
    v = vector_array
    if len(v) == 0:
        return
    ind = [int(len(vector_array) * 3 / 4)] * 2
    for deep in (True, False):
        c = v[ind].copy(deep)
        assert almost_equal(c[0], v[ind[0]])
        assert almost_equal(c[1], v[ind[0]])
        try:
            assert indexed(v.to_numpy(), ind).shape == c.to_numpy().shape
        except NotImplementedError:
            pass
        c[0].scal(2.)
        assert almost_equal(c[1], v[ind[0]])
        assert c[0].l2_norm() == 2 * v[ind[0]].l2_norm()
        try:
            assert indexed(v.to_numpy(), ind).shape == c.to_numpy().shape
        except NotImplementedError:
            pass


def test_append(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    len_v1, len_v2 = len(v1), len(v2)
    for ind in valid_inds(v2):
        c1, c2 = v1.copy(), v2.copy()
        c1.append(c2[ind])
        len_ind = v2.len_ind(ind)
        ind_complement_ = ind_complement(v2, ind)
        assert len(c1) == len_v1 + len_ind
        assert np.all(almost_equal(c1[len_v1:len(c1)], c2[ind]))
        try:
            assert np.allclose(c1.to_numpy(), np.vstack((v1.to_numpy(), indexed(v2.to_numpy(), ind))))
        except NotImplementedError:
            pass
        c1.append(c2[ind], remove_from_other=True)
        assert len(c2) == len(ind_complement_)
        assert c2.space == c1.space
        assert len(c1) == len_v1 + 2 * len_ind
        assert np.all(almost_equal(c1[len_v1:len_v1 + len_ind], c1[len_v1 + len_ind:len(c1)]))
        assert np.all(almost_equal(c2, v2[ind_complement_]))
        try:
            assert np.allclose(c2.to_numpy(), indexed(v2.to_numpy(), ind_complement_))
        except NotImplementedError:
            pass


def test_append_self(vector_array):
    v = vector_array
    c = v.copy()
    len_v = len(v)
    c.append(c)
    assert len(c) == 2 * len_v
    assert np.all(almost_equal(c[:len_v], c[len_v:len(c)]))
    try:
        assert np.allclose(c.to_numpy(), np.vstack((v.to_numpy(), v.to_numpy())))
    except NotImplementedError:
        pass
    c = v.copy()
    with pytest.raises(Exception):
        v.append(v, remove_from_other=True)


def test_del(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        ind_complement_ = ind_complement(v, ind)
        c = v.copy()
        del c[ind]
        assert c.space == v.space
        assert len(c) == len(ind_complement_)
        assert np.all(almost_equal(v[ind_complement_], c))
        try:
            assert np.allclose(c.to_numpy(), indexed(v.to_numpy(), ind_complement_))
        except NotImplementedError:
            pass
        del c[:]
        assert len(c) == 0


def test_scal(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        if v.len_ind(ind) != v.len_ind_unique(ind):
            with pytest.raises(Exception):
                c = v.copy()
                c[ind].scal(1.)
            continue
        ind_complement_ = ind_complement(v, ind)
        c = v.copy()
        c[ind].scal(1.)
        assert len(c) == len(v)
        assert np.all(almost_equal(c, v))

        c = v.copy()
        c[ind].scal(0.)
        assert np.all(almost_equal(c[ind], v.zeros(v.len_ind(ind))))
        assert np.all(almost_equal(c[ind_complement_], v[ind_complement_]))

        for x in (1., 1.4, np.random.random(v.len_ind(ind))):
            c = v.copy()
            c[ind].scal(x)
            assert np.all(almost_equal(c[ind_complement_], v[ind_complement_]))
            assert np.allclose(c[ind].sup_norm(), v[ind].sup_norm() * abs(x))
            assert np.allclose(c[ind].l2_norm(), v[ind].l2_norm() * abs(x))
            try:
                y = v.to_numpy(True)
                if isinstance(x, np.ndarray) and not isinstance(ind, Number):
                    x = x[:, np.newaxis]
                y[ind] *= x
                assert np.allclose(c.to_numpy(), y)
            except NotImplementedError:
                pass


def test_axpy(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair

    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        if v1.len_ind(ind1) != v1.len_ind_unique(ind1):
            with pytest.raises(Exception):
                c1, c2 = v1.copy(), v2.copy()
                c1[ind1].axpy(0., c2[ind2])
            continue

        ind1_complement = ind_complement(v1, ind1)
        c1, c2 = v1.copy(), v2.copy()
        c1[ind1].axpy(0., c2[ind2])
        assert len(c1) == len(v1)
        assert np.all(almost_equal(c1, v1))
        assert np.all(almost_equal(c2, v2))

        np.random.seed(len(v1) + 39)
        for a in (1., 1.4, np.random.random(v1.len_ind(ind1))):
            c1, c2 = v1.copy(), v2.copy()
            c1[ind1].axpy(a, c2[ind2])
            assert len(c1) == len(v1)
            assert np.all(almost_equal(c1[ind1_complement], v1[ind1_complement]))
            assert np.all(almost_equal(c2, v2))
            assert np.all(c1[ind1].sup_norm() <= v1[ind1].sup_norm() + abs(a) * v2[ind2].sup_norm() * (1. + 1e-10))
            assert np.all(c1[ind1].l1_norm() <= (v1[ind1].l1_norm() + abs(a) * v2[ind2].l1_norm()) * (1. + 1e-10))
            assert np.all(c1[ind1].l2_norm() <= (v1[ind1].l2_norm() + abs(a) * v2[ind2].l2_norm()) * (1. + 1e-10))
            try:
                x = v1.to_numpy(True)
                if isinstance(ind1, Number):
                    x[[ind1]] += indexed(v2.to_numpy(), ind2) * a
                else:
                    if isinstance(a, np.ndarray):
                        aa = a[:, np.newaxis]
                    else:
                        aa = a
                    x[ind1] += indexed(v2.to_numpy(), ind2) * aa
                assert np.allclose(c1.to_numpy(), x)
            except NotImplementedError:
                pass
            c1[ind1].axpy(-a, c2[ind2])
            assert len(c1) == len(v1)
            assert np.all(almost_equal(c1, v1))


def test_axpy_one_x(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair

    for ind1, ind2 in product(valid_inds(v1), valid_inds(v2, 1)):
        if v1.len_ind(ind1) != v1.len_ind_unique(ind1):
            with pytest.raises(Exception):
                c1, c2 = v1.copy(), v2.copy()
                c1[ind1].axpy(0., c2[ind2])
            continue

        ind1_complement = ind_complement(v1, ind1)
        c1, c2 = v1.copy(), v2.copy()
        c1[ind1].axpy(0., c2[ind2])
        assert len(c1) == len(v1)
        assert np.all(almost_equal(c1, v1))
        assert np.all(almost_equal(c2, v2))

        np.random.seed(len(v1) + 39)
        for a in (1., 1.4, np.random.random(v1.len_ind(ind1))):
            c1, c2 = v1.copy(), v2.copy()
            c1[ind1].axpy(a, c2[ind2])
            assert len(c1) == len(v1)
            assert np.all(almost_equal(c1[ind1_complement], v1[ind1_complement]))
            assert np.all(almost_equal(c2, v2))
            assert np.all(c1[ind1].sup_norm() <= v1[ind1].sup_norm() + abs(a) * v2[ind2].sup_norm() * (1. + 1e-10))
            assert np.all(c1[ind1].l1_norm() <= (v1[ind1].l1_norm() + abs(a) * v2[ind2].l1_norm()) * (1. + 1e-10))
            assert np.all(c1[ind1].l2_norm() <= (v1[ind1].l2_norm() + abs(a) * v2[ind2].l2_norm()) * (1. + 1e-10))
            try:
                x = v1.to_numpy(True)
                if isinstance(ind1, Number):
                    x[[ind1]] += indexed(v2.to_numpy(), ind2) * a
                else:
                    if isinstance(a, np.ndarray):
                        aa = a[:, np.newaxis]
                    else:
                        aa = a
                    x[ind1] += indexed(v2.to_numpy(), ind2) * aa
                assert np.allclose(c1.to_numpy(), x)
            except NotImplementedError:
                pass
            c1[ind1].axpy(-a, c2[ind2])
            assert len(c1) == len(v1)
            assert np.all(almost_equal(c1, v1))


def test_axpy_self(vector_array):
    v = vector_array

    for ind1, ind2 in valid_inds_of_same_length(v, v):
        if v.len_ind(ind1) != v.len_ind_unique(ind1):
            with pytest.raises(Exception):
                c, = v.copy()
                c[ind1].axpy(0., c[ind2])
            continue

        ind1_complement = ind_complement(v, ind1)
        c = v.copy()
        c[ind1].axpy(0., c[ind2])
        assert len(c) == len(v)
        assert np.all(almost_equal(c, v))

        np.random.seed(len(v) + 8)
        for a in (1., 1.4, np.random.random(v.len_ind(ind1))):
            c = v.copy()
            c[ind1].axpy(a, c[ind2])
            assert len(c) == len(v)
            assert np.all(almost_equal(c[ind1_complement], v[ind1_complement]))
            assert np.all(c[ind1].sup_norm() <= v[ind1].sup_norm() + abs(a) * v[ind2].sup_norm() * (1. + 1e-10))
            assert np.all(c[ind1].l1_norm() <= (v[ind1].l1_norm() + abs(a) * v[ind2].l1_norm()) * (1. + 1e-10))
            try:
                x = v.to_numpy(True)
                if isinstance(ind1, Number):
                    x[[ind1]] += indexed(v.to_numpy(), ind2) * a
                else:
                    if isinstance(a, np.ndarray):
                        aa = a[:, np.newaxis]
                    else:
                        aa = a
                    x[ind1] += indexed(v.to_numpy(), ind2) * aa
                assert np.allclose(c.to_numpy(), x)
            except NotImplementedError:
                pass
            c[ind1].axpy(-a, v[ind2])
            assert len(c) == len(v)
            assert np.all(almost_equal(c, v))

    for ind in valid_inds(v):
        if v.len_ind(ind) != v.len_ind_unique(ind):
            continue

        for x in (1., 23., -4):
            c = v.copy()
            cc = v.copy()
            c[ind].axpy(x, c[ind])
            cc[ind].scal(1 + x)
            assert np.all(almost_equal(c, cc))


def test_pairwise_dot(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        r = v1[ind1].pairwise_dot(v2[ind2])
        assert isinstance(r, np.ndarray)
        assert r.shape == (v1.len_ind(ind1),)
        r2 = v2[ind2].pairwise_dot(v1[ind1])
        assert np.allclose, (r, r2)
        assert np.all(r <= v1[ind1].l2_norm() * v2[ind2].l2_norm() * (1. + 1e-10))
        try:
            assert np.allclose(r, np.sum(indexed(v1.to_numpy(), ind1) * indexed(v2.to_numpy(), ind2), axis=1))
        except NotImplementedError:
            pass


def test_pairwise_dot_self(vector_array):
    v = vector_array
    for ind1, ind2 in valid_inds_of_same_length(v, v):
        r = v[ind1].pairwise_dot(v[ind2])
        assert isinstance(r, np.ndarray)
        assert r.shape == (v.len_ind(ind1),)
        r2 = v[ind2].pairwise_dot(v[ind1])
        assert np.allclose(r, r2)
        assert np.all(r <= v[ind1].l2_norm() * v[ind2].l2_norm() * (1. + 1e-10))
        try:
            assert np.allclose(r, np.sum(indexed(v.to_numpy(), ind1) * indexed(v.to_numpy(), ind2), axis=1))
        except NotImplementedError:
            pass
    for ind in valid_inds(v):
        r = v[ind].pairwise_dot(v[ind])
        assert np.allclose(r, v[ind].l2_norm() ** 2)


def test_dot(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in chain(valid_inds_of_different_length(v1, v2), valid_inds_of_same_length(v1, v2)):
        r = v1[ind1].dot(v2[ind2])
        assert isinstance(r, np.ndarray)
        assert r.shape == (v1.len_ind(ind1), v2.len_ind(ind2))
        r2 = v2[ind2].dot(v1[ind1])
        assert np.allclose(r, r2.T)
        assert np.all(r <= v1[ind1].l2_norm()[:, np.newaxis] * v2[ind2].l2_norm()[np.newaxis, :] * (1. + 1e-10))
        try:
            assert np.allclose(r, indexed(v1.to_numpy(), ind1).dot(indexed(v2.to_numpy(), ind2).T))
        except NotImplementedError:
            pass


def test_dot_self(vector_array):
    v = vector_array
    for ind1, ind2 in chain(valid_inds_of_different_length(v, v), valid_inds_of_same_length(v, v)):
        r = v[ind1].dot(v[ind2])
        assert isinstance(r, np.ndarray)
        assert r.shape == (v.len_ind(ind1), v.len_ind(ind2))
        r2 = v[ind2].dot(v[ind1])
        assert np.allclose(r, r2.T)
        assert np.all(r <= v[ind1].l2_norm()[:, np.newaxis] * v[ind2].l2_norm()[np.newaxis, :] * (1. + 1e-10))
        try:
            assert np.allclose(r, indexed(v.to_numpy(), ind1).dot(indexed(v.to_numpy(), ind2).T))
        except NotImplementedError:
            pass
    for ind in valid_inds(v):
        r = v[ind].dot(v[ind])
        assert np.allclose(r, r.T)


def test_lincomb_1d(vector_array):
    v = vector_array
    np.random.seed(len(v) + 42 + v.dim)
    for ind in valid_inds(v):
        coeffs = np.random.random(v.len_ind(ind))
        lc = v[ind].lincomb(coeffs)
        assert lc.space == v.space
        assert len(lc) == 1
        lc2 = v.zeros()
        for coeff, i in zip(coeffs, ind_to_list(v, ind)):
            lc2.axpy(coeff, v[i])
        assert np.all(almost_equal(lc, lc2))


def test_lincomb_2d(vector_array):
    v = vector_array
    np.random.seed(len(v) + 42 + v.dim)
    for ind in valid_inds(v):
        for count in (0, 1, 5):
            coeffs = np.random.random((count, v.len_ind(ind)))
            lc = v[ind].lincomb(coeffs)
            assert lc.space == v.space
            assert len(lc) == count
            lc2 = v.empty(reserve=count)
            for coeffs_1d in coeffs:
                lc2.append(v[ind].lincomb(coeffs_1d))
            assert np.all(almost_equal(lc, lc2))


def test_lincomb_wrong_coefficients(vector_array):
    v = vector_array
    np.random.seed(len(v) + 42 + v.dim)
    for ind in valid_inds(v):
        coeffs = np.random.random(v.len_ind(ind) + 1)
        with pytest.raises(Exception):
            v[ind].lincomb(coeffs)
        coeffs = np.random.random(v.len_ind(ind)).reshape((1, 1, -1))
        with pytest.raises(Exception):
            v[ind].lincomb(coeffs)
        if v.len_ind(ind) > 0:
            coeffs = np.random.random(v.len_ind(ind) - 1)
            with pytest.raises(Exception):
                v[ind].lincomb(coeffs)
            coeffs = np.array([])
            with pytest.raises(Exception):
                v[ind].lincomb(coeffs)


def test_l1_norm(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        c = v.copy()
        norm = c[ind].l1_norm()
        assert isinstance(norm, np.ndarray)
        assert norm.shape == (v.len_ind(ind),)
        assert np.all(norm >= 0)
        if v.dim == 0:
            assert np.all(norm == 0)
        try:
            assert np.allclose(norm, np.sum(np.abs(indexed(v.to_numpy(), ind)), axis=1))
        except NotImplementedError:
            pass
        c.scal(4.)
        assert np.allclose(c[ind].l1_norm(), norm * 4)
        c.scal(-4.)
        assert np.allclose(c[ind].l1_norm(), norm * 16)
        c.scal(0.)
        assert np.allclose(c[ind].l1_norm(), 0)


def test_l2_norm(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        c = v.copy()
        norm = c[ind].l2_norm()
        assert isinstance(norm, np.ndarray)
        assert norm.shape == (v.len_ind(ind),)
        assert np.all(norm >= 0)
        if v.dim == 0:
            assert np.all(norm == 0)
        try:
            assert np.allclose(norm, np.sqrt(np.sum(np.power(indexed(v.to_numpy(), ind), 2), axis=1)))
        except NotImplementedError:
            pass
        c.scal(4.)
        assert np.allclose(c[ind].l2_norm(), norm * 4)
        c.scal(-4.)
        assert np.allclose(c[ind].l2_norm(), norm * 16)
        c.scal(0.)
        assert np.allclose(c[ind].l2_norm(), 0)


def test_l2_norm2(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        c = v.copy()
        norm = c[ind].l2_norm2()
        assert isinstance(norm, np.ndarray)
        assert norm.shape == (v.len_ind(ind),)
        assert np.all(norm >= 0)
        if v.dim == 0:
            assert np.all(norm == 0)
        try:
            assert np.allclose(norm, np.sum(np.power(indexed(v.to_numpy(), ind), 2), axis=1))
        except NotImplementedError:
            pass
        c.scal(4.)
        assert np.allclose(c[ind].l2_norm2(), norm * 16)
        c.scal(-4.)
        assert np.allclose(c[ind].l2_norm2(), norm * 256)
        c.scal(0.)
        assert np.allclose(c[ind].l2_norm2(), 0)


def test_sup_norm(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        c = v.copy()
        norm = c[ind].sup_norm()
        assert isinstance(norm, np.ndarray)
        assert norm.shape == (v.len_ind(ind),)
        assert np.all(norm >= 0)
        if v.dim == 0:
            assert np.all(norm == 0)
        if v.dim > 0:
            try:
                assert np.allclose(norm, np.max(np.abs(indexed(v.to_numpy(), ind)), axis=1))
            except NotImplementedError:
                pass
        c.scal(4.)
        assert np.allclose(c[ind].sup_norm(), norm * 4)
        c.scal(-4.)
        assert np.allclose(c[ind].sup_norm(), norm * 16)
        c.scal(0.)
        assert np.allclose(c[ind].sup_norm(), 0)


def test_dofs(vector_array):
    v = vector_array
    np.random.seed(len(v) + 24 + v.dim)
    for ind in valid_inds(v):
        c = v.copy()
        dofs = c[ind].dofs(np.array([], dtype=np.int))
        assert isinstance(dofs, np.ndarray)
        assert dofs.shape == (v.len_ind(ind), 0)

        c = v.copy()
        dofs = c[ind].dofs([])
        assert isinstance(dofs, np.ndarray)
        assert dofs.shape == (v.len_ind(ind), 0)

        if v.dim > 0:
            for count in (1, 5, 10):
                c_ind = np.random.randint(0, v.dim, count)
                c = v.copy()
                dofs = c[ind].dofs(c_ind)
                assert dofs.shape == (v.len_ind(ind), count)
                c = v.copy()
                dofs2 = c[ind].dofs(list(c_ind))
                assert np.all(dofs == dofs2)
                c = v.copy()
                c.scal(3.)
                dofs2 = c[ind].dofs(c_ind)
                assert np.allclose(dofs * 3, dofs2)
                c = v.copy()
                dofs2 = c[ind].dofs(np.hstack((c_ind, c_ind)))
                assert np.all(dofs2 == np.hstack((dofs, dofs)))
                try:
                    assert np.all(dofs == indexed(v.to_numpy(), ind)[:, c_ind])
                except NotImplementedError:
                    pass


def test_components_wrong_dof_indices(vector_array):
    v = vector_array
    np.random.seed(len(v) + 24 + v.dim)
    for ind in valid_inds(v):
        with pytest.raises(Exception):
            v[ind].dofs(None)
        with pytest.raises(Exception):
            v[ind].dofs(1)
        with pytest.raises(Exception):
            v[ind].dofs(np.array([-1]))
        with pytest.raises(Exception):
            v[ind].dofs(np.array([v.dim]))


def test_amax(vector_array):
    v = vector_array
    if v.dim == 0:
        return
    for ind in valid_inds(v):
        max_inds, max_vals = v[ind].amax()
        assert np.allclose(np.abs(max_vals), v[ind].sup_norm())
        for i, max_ind, max_val in zip(ind_to_list(v, ind), max_inds, max_vals):
            assert np.allclose(max_val, v[[i]].dofs([max_ind]))


# def test_amax_zero_dim(zero_dimensional_vector_space):
#     for count in (0, 10):
#         v = zero_dimensional_vector_space.zeros(count=count)
#         for ind in valid_inds(v):
#             with pytest.raises(Exception):
#                 v.amax(ind)


def test_gramian(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        assert np.allclose(v[ind].gramian(), v[ind].dot(v[ind]))


def test_add(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if len(v2) < len(v1):
        v2.append(v2[np.zeros(len(v1) - len(v2), dtype=np.int)])
    elif len(v2) > len(v1):
        del v2[:len(v2)-len(v1)]
    c1 = v1.copy()
    cc1 = v1.copy()
    c1.axpy(1, v2)
    assert np.all(almost_equal(v1 + v2, c1))
    assert np.all(almost_equal(v1, cc1))


def test_iadd(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if len(v2) < len(v1):
        v2.append(v2[np.zeros(len(v1) - len(v2), dtype=np.int)])
    elif len(v2) > len(v1):
        del v2[:len(v2)-len(v1)]
    c1 = v1.copy()
    c1.axpy(1, v2)
    v1 += v2
    assert np.all(almost_equal(v1, c1))


def test_sub(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if len(v2) < len(v1):
        v2.append(v2[np.zeros(len(v1) - len(v2), dtype=np.int)])
    elif len(v2) > len(v1):
        del v2[list(range(len(v2)-len(v1)))]
    c1 = v1.copy()
    cc1 = v1.copy()
    c1.axpy(-1, v2)
    assert np.all(almost_equal((v1 - v2), c1))
    assert np.all(almost_equal(v1, cc1))


def test_isub(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if len(v2) < len(v1):
        v2.append(v2[np.zeros(len(v1) - len(v2), dtype=np.int)])
    elif len(v2) > len(v1):
        del v2[:len(v2)-len(v1)]
    c1 = v1.copy()
    c1.axpy(-1, v2)
    v1 -= v2
    assert np.all(almost_equal(v1, c1))


def test_neg(vector_array):
    v = vector_array
    c = v.copy()
    cc = v.copy()
    c.scal(-1)
    assert np.all(almost_equal(c, -v))
    assert np.all(almost_equal(v, cc))


def test_mul(vector_array):
    v = vector_array
    c = v.copy()
    for a in (-1, -3, 0, 1, 23):
        cc = v.copy()
        cc.scal(a)
        assert np.all(almost_equal((v * a), cc))
        assert np.all(almost_equal(v, c))


def test_mul_wrong_factor(vector_array):
    v = vector_array
    with pytest.raises(Exception):
        _ = v * v


def test_rmul(vector_array):
    v = vector_array
    c = v.copy()
    for a in (-1, -3, 0, 1, 23):
        cc = v.copy()
        cc.scal(a)
        assert np.all(almost_equal((a * v), cc))
        assert np.all(almost_equal(v, c))


def test_imul(vector_array):
    v = vector_array
    for a in (-1, -3, 0, 1, 23):
        c = v.copy()
        cc = v.copy()
        c.scal(a)
        cc *= a
        assert np.all(almost_equal(c, cc))


def test_imul_wrong_factor(vector_array):
    v = vector_array
    with pytest.raises(Exception):
        v *= v


########################################################################################################################


def test_append_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, remove_from_other=False)
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, remove_from_other=True)
    c1, c2 = v1.copy(), v2.copy()


def test_axpy_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(0., c2[ind2])
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(1., c2[ind2])
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(-1., c2[ind2])
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(1.42, c2[ind2])


def test_dot_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].dot(c2[ind2])


def test_pairwise_dot_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].pairwise_dot(c2[ind2])


def test_add_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    with pytest.raises(Exception):
        _ = v1 + v2


def test_iadd_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    with pytest.raises(Exception):
        v1 += v2


def test_sub_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    with pytest.raises(Exception):
        _ = v1 - v2


def test_isub_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    with pytest.raises(Exception):
        v1 -= v2


########################################################################################################################


def test_copy_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v[ind].copy()


def test_remove_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            del c[ind]


def test_scal_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            c[ind].scal(0.)
        c = v.copy()
        with pytest.raises(Exception):
            c[ind].scal(1.)
        c = v.copy()
        with pytest.raises(Exception):
            c[ind].scal(-1.)
        c = v.copy()
        with pytest.raises(Exception):
            c[ind].scal(1.2)


def test_scal_wrong_coefficients(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        np.random.seed(len(v) + 99)
        for alpha in ([np.array([]), np.eye(v.len_ind(ind)), np.random.random(v.len_ind(ind) + 1)]
                      if v.len_ind(ind) > 0 else
                      [np.random.random(1)]):
            with pytest.raises(Exception):
                v[ind].scal(alpha)


def test_axpy_wrong_ind(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        if v2.len_ind(ind2) == 1:
            continue
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(0., c2[ind2])
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(1., c2[ind2])
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(-1., c2[ind2])
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(1.456, c2[ind2])


def test_axpy_wrong_coefficients(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        np.random.seed(len(v1) + 99)
        for alpha in ([np.array([]), np.eye(v1.len_ind(ind1)), np.random.random(v1.len_ind(ind1) + 1)]
                      if v1.len_ind(ind1) > 0 else
                      [np.random.random(1)]):
            with pytest.raises(Exception):
                v1[ind1].axpy(alpha, v2[ind2])


def test_pairwise_dot_wrong_ind(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].pairwise_dot(c2[ind2])


def test_pickle(picklable_vector_array):
    assert_picklable_without_dumps_function(picklable_vector_array)
