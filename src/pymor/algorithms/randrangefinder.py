# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.special import erfinv
from scipy.linalg import svd

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.constructions import VectorArrayOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.interfaces import VectorArrayInterface


@defaults('tol', 'failure_tolerance', 'num_testvecs')
def adaptive_rrf(A, source_product=None, range_product=None, tol=1e-4,
                 failure_tolerance=1e-15, num_testvecs=20, iscomplex=False):
    r"""Adaptive randomized range approximation of `A`.

    This is an implementation of Algorithm 1 in [BS18]_.

    Given the |Operator| `A`, the return value of this method is the |VectorArray|
    `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the inner product of the range
    of `A` is given by `range_product` and the inner product of the source of `A`
    is given by `source_product`.

    Parameters
    ----------
    A
        The |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    tol
        Error tolerance for the algorithm.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    iscomplex
        If `True`, the random vectors are chosen complex.

    Returns
    -------
    B
        |VectorArray| which contains the basis, whose span approximates the range of A.
    """

    assert source_product is None or isinstance(source_product, OperatorInterface)
    assert range_product is None or isinstance(range_product, OperatorInterface)
    assert isinstance(A, OperatorInterface)

    B = A.range.empty()

    R = A.source.random(num_testvecs, distribution='normal')
    if iscomplex:
        R += 1j*A.source.random(num_testvecs, distribution='normal')

    if source_product is None:
        lambda_min = 1
    else:
        def mv(v):
            return source_product.apply(NumpyVectorSpace.make_array(v)).data
        L = LinearOperator((source_product.source.dim, source_product.range.dim), matvec=mv)
        lambda_min = eigsh(L, which="SM", return_eigenvectors=False, k=1)[0]

    testfail = failure_tolerance / min(A.source.dim, A.range.dim)
    testlimit = np.sqrt(2. * lambda_min) * erfinv(testfail**(1. / num_testvecs)) * tol
    maxnorm = np.inf
    M = A.apply(R)

    while(maxnorm > testlimit):
        basis_length = len(B)
        v = A.source.random(distribution='normal')
        if iscomplex:
            v += 1j*A.source.random(distribution='normal')
        B.append(A.apply(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))

    return B


@defaults('q', 'l')
def rrf(A, source_product=None, range_product=None, q=2, l=8, iscomplex=False):
    """Randomized range approximation of `A`.

    This is an implementation of Algorithm 4.4 in [HMT11]_.

    Given the |Operator| `A`, the return value of this method is the |VectorArray|
    `Q` whose vectors form an orthonomal basis for the range of `A`.

    Parameters
    ----------
    A
        The |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    q
        The number of power iterations.
    l
        The block size of the normalized power iterations.
    iscomplex
        If `True`, the random vectors are chosen complex.

    Returns
    -------
    Q
        |VectorArray| which contains the basis, whose span approximates the range of A.
    """

    assert source_product is None or isinstance(source_product, OperatorInterface)
    assert range_product is None or isinstance(range_product, OperatorInterface)
    assert isinstance(A, OperatorInterface)

    R = A.source.random(l, distribution='normal')
    if iscomplex:
        R += 1j*A.source.random(l, distribution='normal')
    Q = A.apply(R)
    gram_schmidt(Q, range_product, atol=0, rtol=0, copy=False)

    for i in range(q):
        Q = A.apply_adjoint(Q)
        gram_schmidt(Q, source_product, atol=0, rtol=0, copy=False)
        Q = A.apply(Q)
        gram_schmidt(Q, range_product, atol=0, rtol=0, copy=False)

    return Q


@defaults('q')
def approximate_operatornorm(A, source_product=None, range_product=None, q=20, iscomplex=False):
    """Randomized approximation of the operator norm of `A`.

    Given the |Operator| `A`, the return value of this method is an estimate of the operator norm of `A`.

    Parameters
    ----------
    A
        The |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    q
        The number of power iterations.
    iscomplex
        If `True`, the random vectors are chosen complex.

    Returns
    -------
    snorm
        An estimate of the operator norm of A.
    """

    assert source_product is None or isinstance(source_product, OperatorInterface)
    assert range_product is None or isinstance(range_product, OperatorInterface)
    assert isinstance(A, OperatorInterface)

    x = A.source.random(distribution='normal')
    if iscomplex:
        x += 1j*A.source.random(distribution='normal')

    x.scal(1/x.norm())

    for i in range(q):
        y = A.apply(x)
        if range_product is not None:
            y = range_product.apply(y)
        x = A.apply_adjoint(y)
        if source_product is not None:
            x = source_product.apply_inverse(x)
        snorm = x.norm()
        if snorm == 0:
            return 0
        x.scal(1/snorm)
    snorm = np.sqrt(snorm)

    return snorm


@defaults('k', 'q', 'l')
def randomized_svd(A, k=6, q=2, l=8, iscomplex=False):
    """Randomized SVD approximation of `A`.

    This is an implementation of Algorithm 5.1 in [HMT11]_.

    The return value of this method is a rank-k approximation of the singular value
    decomposition of the |VectorArray| `A`.

    Parameters
    ----------
    A
        The |VectorArray| A.
    k
        The rank of the approximation.
    q
        The number of power iterations.
    l
        The block size of the normalized power iterations.
    iscomplex
        If `True`, the random vectors are chosen complex.

    Returns
    -------
    U
        |VectorArray| of the approximated left singular vectors of A.
    s
        Sequence of approximated singular values.
    Va
        |VectorArray| of the approximated adjoint right singular vectors of A.
    """

    assert isinstance(A, VectorArrayInterface)
    assert k <= l

    Q = rrf(VectorArrayOperator(A), q=q, l=l, iscomplex=iscomplex)
    B = Q.inner(A)
    U, s, Va = svd(B)
    U = Q.lincomb(U.T)
    Va = NumpyVectorSpace.make_array(Va)

    return U[:k], s[:k], Va[:k]
