# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from numpy.random import uniform

from pymor.algorithms.randrangefinder import rrf, adaptive_rrf, approximate_operatornorm, randomized_svd
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import VectorArrayOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


np.random.seed(0)
A = uniform(low=-1.0, high=1.0, size=(100, 100))
A = A.dot(A.T)
range_product = NumpyMatrixOperator(A)

np.random.seed(1)
A = uniform(low=-1.0, high=1.0, size=(10, 10))
A = A.dot(A.T)
source_product = NumpyMatrixOperator(A)

B = range_product.range.random(10, seed=10)
op = VectorArrayOperator(B)

C = range_product.range.random(10, seed=11)+1j*range_product.range.random(10, seed=12)
op_complex = VectorArrayOperator(C)


def test_rrf():
    Q = rrf(op, source_product, range_product)
    assert Q in op.range
    assert len(Q) == 8

    Q = rrf(op_complex, iscomplex=True)
    assert np.iscomplexobj(Q.data)
    assert Q in op.range
    assert len(Q) == 8


def test_adaptive_rrf():
    B = adaptive_rrf(op, source_product, range_product)
    assert B in op.range

    B = adaptive_rrf(op_complex, iscomplex=True)
    assert np.iscomplexobj(B.data)
    assert B in op.range

    tol = 1e-1
    B = adaptive_rrf(op, source_product, range_product, tol=tol)
    T = op.as_range_array()-B.lincomb(B.inner(op.as_range_array(), range_product).T)
    snorm = approximate_operatornorm(VectorArrayOperator(T), source_product, range_product)
    assert snorm < tol


def test_rsvd():
    np.random.seed(3)
    A = uniform(low=-1.0, high=1.0, size=(100, 10))
    A = A.dot(uniform(low=-1.0, high=1.0, size=(10, 100)))
    A = NumpyVectorSpace.make_array(A)
    U, s, Va = randomized_svd(A, l=10, k=10)
    T = A-U.lincomb(np.diag(s)).lincomb(Va.data.T)
    assert approximate_operatornorm(VectorArrayOperator(T)) < 1e-10
