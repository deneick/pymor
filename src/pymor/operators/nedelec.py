# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Andreas Buhr <andreas@andreasbuhr.de>

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from pymor.functions.interfaces import FunctionInterface
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.grids.referenceelements import triangle
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.parameters.base import Parametric
from pymor.vectorarrays.numpy import NumpyVectorSpace


class RotRotOperator(NumpyMatrixBasedOperator):

    sparse = True

    def __init__(self, grid, boundary_info, coefficient=1.,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element is triangle
        assert isinstance(coefficient, Number) \
            or (isinstance(coefficient, FunctionInterface) and
                coefficient.dim_domain == 2 and
                coefficient.shape_range == tuple())
        self.source = self.range = NumpyVectorSpace(grid.size(1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.coefficient = coefficient
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.solver_options = solver_options
        self.name = name
        if isinstance(coefficient, Parametric):
            self.build_parameter_type(inherits=(coefficient,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        self.logger.info('Calulate local matrices ...')
        M = np.empty((g.size(0), 3, 3))

        edge_lengths = g.volumes(1)[g.subentities(0, 1)]
        vol_inverse = 1. / g.volumes(0)

        M[:, 0, 0] = edge_lengths[:, 0] * edge_lengths[:, 0] * vol_inverse
        M[:, 1, 1] = edge_lengths[:, 1] * edge_lengths[:, 1] * vol_inverse
        M[:, 2, 2] = edge_lengths[:, 2] * edge_lengths[:, 2] * vol_inverse

        M[:, 2, 1] = vol_inverse * edge_lengths[:, 2] * edge_lengths[:, 1]
        M[:, 2, 0] = vol_inverse * edge_lengths[:, 2] * edge_lengths[:, 0]
        M[:, 1, 0] = vol_inverse * edge_lengths[:, 1] * edge_lengths[:, 0]

        M[:, 1, 2] = M[:, 2, 1]
        M[:, 0, 2] = M[:, 2, 0]
        M[:, 0, 1] = M[:, 1, 0]

        del vol_inverse, edge_lengths

        if isinstance(self.coefficient, FunctionInterface):
            M *= self.coefficient.evaluate(g.centers(0), mu=mu)[:, np.newaxis, np.newaxis]
        else:
            M *= self.coefficient

        SIGNS = np.empty((g.size(0), 3))
        SE = g.subentities(0, 2)
        SIGNS[:, 0] = np.sign(SE[:, 2] - SE[:, 1])
        SIGNS[:, 1] = np.sign(SE[:, 0] - SE[:, 2])
        SIGNS[:, 2] = np.sign(SE[:, 1] - SE[:, 0])
        M *= SIGNS[:, :, np.newaxis] * SIGNS[:, np.newaxis, :]
        del SE, SIGNS

        self.logger.info('Determine global dofs ...')
        M = M.ravel()
        SF_I0 = np.repeat(g.subentities(0, 1), 3, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, 1), [1, 3]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            M = np.where(bi.dirichlet_mask(1)[SF_I0], 0, M)
            if self.dirichlet_clear_columns:
                M = np.where(bi.dirichlet_mask(1)[SF_I1], 0, M)

            if not self.dirichlet_clear_diag:
                M = np.hstack((M, np.ones(bi.dirichlet_boundaries(1).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(1)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(1)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((M, (SF_I0, SF_I1)), shape=(g.size(1), g.size(1)))
        del M, SF_I0, SF_I1
        A = csc_matrix(A)
        A.eliminate_zeros()
        A = A.copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class L2ProductOperator(NumpyMatrixBasedOperator):

    sparse = True

    def __init__(self, grid, boundary_info, coefficient=1.,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element is triangle
        assert isinstance(coefficient, Number) \
            or (isinstance(coefficient, FunctionInterface) and
                coefficient.dim_domain == 2 and
                coefficient.shape_range == tuple())
        self.source = self.range = NumpyVectorSpace(grid.size(1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.coefficient = coefficient
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.solver_options = solver_options
        self.name = name
        if isinstance(coefficient, Parametric):
            self.build_parameter_type(inherits=(coefficient,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        self.logger.info('Calulate local matrices ...')
        M = np.empty((g.size(0), 3, 3))

        edge_lengths = g.volumes(1)[g.subentities(0, 1)]
        vol_inverse = 1. / g.volumes(0)
        CO = g.centers(2)[g.subentities(0, 2)]

        M[:, 0, 0] = (1. / 24. * edge_lengths[:, 0] * edge_lengths[:, 0] * vol_inverse *
                      (edge_lengths[:, 0] * edge_lengths[:, 0]
                       + 3. * ((CO[:, 2, 0] - CO[:, 0, 0]) * (CO[:, 1, 0] - CO[:, 0, 0]) +
                               (CO[:, 2, 1] - CO[:, 0, 1]) * (CO[:, 1, 1] - CO[:, 0, 1]))))

        M[:, 1, 1] = (1. / 24. * edge_lengths[:, 1] * edge_lengths[:, 1] * vol_inverse *
                      (edge_lengths[:, 1] * edge_lengths[:, 1]
                       + 3. * ((CO[:, 1, 0] - CO[:, 2, 0]) * (CO[:, 1, 0] - CO[:, 0, 0]) +
                               (CO[:, 1, 1] - CO[:, 2, 1]) * (CO[:, 1, 1] - CO[:, 0, 1]))))

        M[:, 2, 2] = (1. / 24. * edge_lengths[:, 2] * edge_lengths[:, 2] * vol_inverse *
                      (edge_lengths[:, 2] * edge_lengths[:, 2]
                       + 3. * ((CO[:, 1, 0] - CO[:, 2, 0]) * (CO[:, 0, 0] - CO[:, 2, 0]) +
                               (CO[:, 1, 1] - CO[:, 2, 1]) * (CO[:, 0, 1] - CO[:, 2, 1]))))

        M[:, 2, 1] = (-1. / 24. * vol_inverse * edge_lengths[:, 2] * edge_lengths[:, 1] *
                      (edge_lengths[:, 0] * edge_lengths[:, 0]
                       - ((CO[:, 1, 0] - CO[:, 0, 0]) * (CO[:, 2, 0] - CO[:, 0, 0]) +
                          (CO[:, 1, 1] - CO[:, 0, 1]) * (CO[:, 2, 1] - CO[:, 0, 1]))))

        M[:, 2, 0] = (-1. / 24. * vol_inverse * edge_lengths[:, 2] * edge_lengths[:, 0] *
                      (edge_lengths[:, 1] * edge_lengths[:, 1]
                       + ((CO[:, 1, 0] - CO[:, 0, 0]) * (CO[:, 2, 0] - CO[:, 1, 0]) +
                          (CO[:, 1, 1] - CO[:, 0, 1]) * (CO[:, 2, 1] - CO[:, 1, 1]))))

        M[:, 1, 0] = (-1. / 24. * vol_inverse * edge_lengths[:, 1] * edge_lengths[:, 0] *
                      (edge_lengths[:, 2] * edge_lengths[:, 2]
                       - ((CO[:, 2, 0] - CO[:, 0, 0]) * (CO[:, 2, 0] - CO[:, 1, 0]) +
                          (CO[:, 2, 1] - CO[:, 0, 1]) * (CO[:, 2, 1] - CO[:, 1, 1]))))

        M[:, 1, 2] = M[:, 2, 1]
        M[:, 0, 2] = M[:, 2, 0]
        M[:, 0, 1] = M[:, 1, 0]

        del CO, vol_inverse, edge_lengths

        if isinstance(self.coefficient, FunctionInterface):
            M *= self.coefficient.evaluate(g.centers(0), mu=mu)[:, np.newaxis, np.newaxis]
        else:
            M *= self.coefficient

        SIGNS = np.empty((g.size(0), 3))
        SE = g.subentities(0, 2)
        SIGNS[:, 0] = np.sign(SE[:, 2] - SE[:, 1])
        SIGNS[:, 1] = np.sign(SE[:, 0] - SE[:, 2])
        SIGNS[:, 2] = np.sign(SE[:, 1] - SE[:, 0])
        M *= SIGNS[:, :, np.newaxis] * SIGNS[:, np.newaxis, :]
        del SE, SIGNS

        self.logger.info('Determine global dofs ...')
        M = M.ravel()
        SF_I0 = np.repeat(g.subentities(0, 1), 3, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, 1), [1, 3]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            M = np.where(bi.dirichlet_mask(1)[SF_I0], 0, M)
            if self.dirichlet_clear_columns:
                M = np.where(bi.dirichlet_mask(1)[SF_I1], 0, M)

            if not self.dirichlet_clear_diag:
                M = np.hstack((M, np.ones(bi.dirichlet_boundaries(1).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(1)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(1)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((M, (SF_I0, SF_I1)), shape=(g.size(1), g.size(1)))
        del M, SF_I0, SF_I1
        A = csc_matrix(A)
        A.eliminate_zeros()
        A = A.copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class L2ProductFunctional(NumpyMatrixBasedOperator):

    sparse = False
    range = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, name=None):
        assert grid.reference_element is triangle
        assert function.dim_domain == 2 and function.shape_range == (2,)
        assert dirichlet_data is None \
            or (dirichlet_data.dim_domain == 2 and dirichlet_data.shape_range == (2,))
        self.source = NumpyVectorSpace(grid.size(1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.name = name
        self.build_parameter_type(inherits=(function, dirichlet_data))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        I = InterpolationOperator(g, self.function).assemble(mu)._matrix.ravel()
        M = L2ProductOperator(self.grid, EmptyBoundaryInfo(self.grid)).assemble()._matrix
        I = M.dot(I)
        del M

        if bi is not None and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(1)
            if self.dirichlet_data is not None:
                SE = g.subentities(1, 2)[DI]
                CO = g.centers(2)[SE]
                TANGENTS = (CO[:, 0] - CO[:, 1]) * np.sign(SE[:, 0] - SE[:, 1])[:, np.newaxis] / g.volumes(1)[DI, np.newaxis]
                I[DI] = np.sum(self.dirichlet_data(g.centers(1)[DI], mu=mu) * TANGENTS, axis=-1)
            else:
                I[DI] = 0

        return I.reshape((1, -1))


class InterpolationOperator(NumpyMatrixBasedOperator):

    source = NumpyVectorSpace(1)
    linear = True

    def __init__(self, grid, function):
        assert grid.reference_element is triangle
        assert function.dim_domain == 2
        assert function.shape_range == (2,)
        self.grid = grid
        self.function = function
        self.range = NumpyVectorSpace(grid.size(1))
        self.build_parameter_type(inherits=(function,))

    def _assemble(self, mu=None):
        g = self.grid
        SE = g.subentities(1, 2)
        CO = g.centers(2)[SE]
        TANGENTS = (CO[:, 0] - CO[:, 1]) * np.sign(SE[:, 0] - SE[:, 1])[:, np.newaxis] / g.volumes(1)[:, np.newaxis]
        return np.sum(self.function(g.centers(1), mu=mu) * TANGENTS, axis=-1).reshape((-1, 1))


class CenterEvaluation(NumpyMatrixBasedOperator):

    linear = True

    def __init__(self, grid):
        assert grid.reference_element is triangle
        self.grid = grid
        self.source = NumpyVectorSpace(grid.size(1))
        self.range = NumpyVectorSpace(grid.size(0) * 2)

    def _assemble(self, mu=None):
        g = self.grid
        M = np.empty((g.size(0), 2, 3))
        edge_lengths = g.volumes(1)[g.subentities(0, 1)]
        CO = g.centers(2)[g.subentities(0, 2)]

        M[:, 0, 2] = ((np.sqrt(np.power(CO[:, 0, 0] - CO[:, 1, 0],2) + np.power(CO[:, 0, 1] - CO[:, 1, 1],2))*(CO[:, 0, 1] + CO[:, 1, 1] - 2*CO[:, 2, 1]))/
            (3.*(CO[:, 0, 1]*(CO[:, 1, 0] - CO[:, 2, 0]) + CO[:, 1, 1]*CO[:, 2, 0] - CO[:, 1, 0]*CO[:, 2, 1] + CO[:, 0,
                0]*(-CO[:, 1, 1] + CO[:, 2, 1]))))

        M[:, 1, 2] =  ( -(np.sqrt(np.power(CO[:, 0, 0] - CO[:, 1, 0],2) + np.power(CO[:, 0, 1] - CO[:, 1, 1],2))*(CO[:, 0, 0] + CO[:, 1, 0] - 2*CO[:, 2, 0]))/
            (3.*(CO[:, 0, 1]*(CO[:, 1, 0] - CO[:, 2, 0]) + CO[:, 1, 1]*CO[:, 2, 0] - CO[:, 1, 0]*CO[:, 2, 1] + CO[:, 0,
                0]*(-CO[:, 1, 1] + CO[:, 2, 1]))))


        M[:, 0, 1] = - ((np.sqrt(np.power(CO[:, 0, 0] - CO[:, 2, 0],2) + np.power(CO[:, 0, 1] - CO[:, 2, 1],2))*(CO[:, 0, 1] - 2*CO[:, 1, 1] + CO[:, 2, 1]))/
            (3.*(-(CO[:, 1, 1]*CO[:, 2, 0]) + CO[:, 0, 1]*(-CO[:, 1, 0] + CO[:, 2, 0]) + CO[:, 0, 0]*(CO[:, 1, 1] - CO[:, 2, 1])
                + CO[:, 1, 0]*CO[:, 2, 1])))

        M[:, 1, 1] = - (((CO[:, 0, 0] - 2*CO[:, 1, 0] + CO[:, 2, 0])*np.sqrt(np.power(CO[:, 0, 0] - CO[:, 2, 0],2) + np.power(CO[:, 0, 1] - CO[:, 2, 1],2)))/
            (3.*(CO[:, 0, 1]*(CO[:, 1, 0] - CO[:, 2, 0]) + CO[:, 1, 1]*CO[:, 2, 0] - CO[:, 1, 0]*CO[:, 2, 1] + CO[:, 0,
                0]*(-CO[:, 1, 1] + CO[:, 2, 1]))))


        M[:, 0, 0] = ((np.sqrt(np.power(CO[:, 1, 0] - CO[:, 2, 0],2) + np.power(CO[:, 1, 1] - CO[:, 2, 1],2))*(-2*CO[:, 0, 1] + CO[:, 1, 1] + CO[:, 2, 1]))/
            (3.*(CO[:, 0, 1]*(CO[:, 1, 0] - CO[:, 2, 0]) + CO[:, 1, 1]*CO[:, 2, 0] - CO[:, 1, 0]*CO[:, 2, 1] + CO[:, 0,
                0]*(-CO[:, 1, 1] + CO[:, 2, 1]))))

        M[:, 1, 0] = (((-2*CO[:, 0, 0] + CO[:, 1, 0] + CO[:, 2, 0])*np.sqrt(np.power(CO[:, 1, 0] - CO[:, 2, 0],2) + np.power(CO[:, 1, 1] - CO[:, 2, 1],2)))/
        (3.*(-(CO[:, 1, 1]*CO[:, 2, 0]) + CO[:, 0, 1]*(-CO[:, 1, 0] + CO[:, 2, 0]) + CO[:, 0, 0]*(CO[:, 1, 1] - CO[:, 2, 1]) +
            CO[:, 1, 0]*CO[:, 2, 1])))

        SIGNS = np.empty((g.size(0), 3))
        SE = g.subentities(0, 2)
        SIGNS[:, 0] = np.sign(SE[:, 2] - SE[:, 1])
        SIGNS[:, 1] = np.sign(SE[:, 0] - SE[:, 2])
        SIGNS[:, 2] = np.sign(SE[:, 1] - SE[:, 0])

        M *= SIGNS[:, np.newaxis, :]

        self.logger.info('Determine global dofs ...')
        M = M.ravel()
        SF_I0 = np.repeat(np.arange(g.size(0) * 2), 3)
        SF_I1 = np.tile(g.subentities(0, 1), [1, 2]).ravel()

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((M, (SF_I0, SF_I1)), shape=(g.size(0) * 2, g.size(1)))
        del M, SF_I0, SF_I1
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class SimpleOpenBoundaryOperator(NumpyMatrixBasedOperator):
    """Simple linear diagonal operator for linear
    relationship between voltage and current. 
    Equivalend of 1/R
    
    A factor of '- i omega' has to be added later
    
    Parameters
    ----------
    grid
        The |Grid| over which to assemble the operator
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions
    robin_data
        A |Function| that represent the Robin parameter.
        If `None`, the resulting operator is zero.
    name
        Name of the operator
    """

    sparse = True

    def __init__(self, grid, boundary_info, robin_data=None,  name=None):
        assert robin_data is None or isinstance(robin_data, FunctionInterface) and robin_data.dim_domain == grid.dim_outer and robin_data.shape_range == tuple()
        self.source = self.range = NumpyVectorSpace(grid.size(grid.dim-1))
        self.grid = grid
        self.boundary_info = boundary_info
        self.robin_data = robin_data
        self.name = name
        if robin_data is not None:
            self.build_parameter_type(inherits=(robin_data,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        if not g.dim == 2:
            raise NotImplementedError

        if bi is None or not bi.has_robin or self.robin_data is None:
            return coo_matrix((g.size(1), g.size(1))).tocsc()

        OI = bi.robin_boundaries(1)
        xref = g.centers(1)[OI]
        robin_c = self.robin_data(xref, mu=mu)
        I = coo_matrix((robin_c, (OI, OI)), shape=(g.size(1), g.size(1)))
        return csc_matrix(I).copy()
