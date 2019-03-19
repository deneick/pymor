from __future__ import absolute_import, division, print_function

from itertools import product

import numpy as np
from scipy.sparse import issparse

from pymor.core.interfaces import ImmutableInterface
from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace


class NumpyLocalizer(ImmutableInterface):

    def __init__(self, space, subspaces):
        #assert space.type == NumpyVectorArray
        assert all(isinstance(s, np.ndarray) for s in subspaces)
        assert all(s.ndim == 1 for s in subspaces)
        self.space = space
        self.subspaces = subspaces
        self.subspace_dims = np.array([len(s) for s in subspaces]) #map(len, subspaces))
        self.subspace_map = np.zeros(self.space.dim, dtype=np.int32)
        for i, dofs in enumerate(subspaces):
            self.subspace_map[dofs] = i
        self._non_zeros = {}
        self._resultcache = {}

        def check_subspaces():
            if not all(len(np.unique(s)) == len(s) for s in subspaces):
                i, _ = next(i for i, s in enumerate if not len(np.unique(s)) == len(s))
                raise AssertionError('{}-th subspace contains repeated DOFs'.format(i))
            space = np.concatenate(tuple(subspaces))
            unique_space = np.unique(space)
            assert len(space) == len(unique_space)
            #assert unique_space[0] == 0
            #assert unique_space[-1] == len(space) - 1 == self.space.subtype - 1
            return True

        assert check_subspaces()

    def join_spaces(self, subspace_ind):
        r = []
        for i in subspace_ind:
            r.extend(self.subspaces[i])
        return r

    def localize_vector_array(self, va, ind):
        assert isinstance(va, NumpyVectorArray)
        subspace = self.join_spaces(ind)
        a = va._array[:, subspace]
        return NumpyVectorSpace.make_array(a)

    def to_space2(self, va, ind, target_ind):
        dims = self.subspace_dims[list(ind)]
        target_dims = self.subspace_dims[list(target_ind)]
        offsets = np.cumsum(np.insert(dims, [0], [0]))
        target_offsets = np.cumsum(np.insert(target_dims, [0], [0]))
        num_vectors = len(va.data)
        resultdata = np.zeros((num_vectors, target_offsets[-1]))
        for i, space_id in enumerate(ind):
            if space_id in target_ind:
                target_i = target_ind.index(space_id)
                resultdata[:, target_offsets[target_i]:target_offsets[target_i+1]] = va.data[:, offsets[i]:offsets[i+1]]
        return NumpyVectorSpace.make_array(resultdata)

    def to_space(self, va, ind, target_ind):
        return self.to_space2(va.real, ind, target_ind) +self.to_space2(va.imag, ind, target_ind)*1j

    def globalize_vector_array_add_inplace(self, r, va, ind):
        assert isinstance(va, NumpyVectorArray)
        subspace = self.join_spaces(ind)
        r.data[:, subspace] += va.data

    def globalize_vector_array2(self, va, ind, sub_ind=None):
        assert isinstance(va, NumpyVectorArray)
        r = self.space.zeros(len(va))
        if sub_ind is None:
            subspace = self.join_spaces(ind)
            r.data[:, subspace] = va.data
        else:
            dims = self.subspace_dims[list(ind)]
            for space_id in sub_ind:
                i = ind.index(space_id)
                offset = np.sum(dims[:i]) if i else 0
                r.data[:, self.subspaces[ind[i]]] = va.data[:, offset:offset + dims[i]]

        return NumpyVectorSpace.make_array(r, copy=False)

    def globalize_vector_array(self, va, ind, sub_ind=None):
        return self.globalize_vector_array2(va.real, ind, sub_ind) +self.globalize_vector_array2(va.imag, ind, sub_ind)*1j

    def localize_operator(self, op, range_ind, source_ind):
        if isinstance(op, VectorArrayOperator):
            assert range_ind is None
            source_ind = tuple(source_ind)
            source_subspace = self.join_spaces(source_ind)
            a = NumpyVectorSpace.make_array(op._array.data[:, source_subspace])
            result = VectorArrayOperator(a, transposed=op.transposed,
                                         name='{}-{}-{}'.format(op.name, source_ind, range_ind))
            return result

        if isinstance(op, NumpyMatrixOperator):
            # special case for functionals
            if range_ind is None:
                source_ind = tuple(source_ind)
                source_subspace = self.join_spaces(source_ind)
                m = op.matrix[source_subspace]
                result = NumpyMatrixOperator(m, name='{}-{}-{}'.format(op.name, source_ind, range_ind))
                return result

            source_ind = tuple(source_ind)
            range_ind = tuple(range_ind)
            op_id = getattr(op, 'sid', op.uid)
            identification_key = (op_id, source_ind, range_ind)
            if identification_key in self._resultcache:
                return self._resultcache[identification_key]

            if op_id not in self._non_zeros:
                # FIXME also handle dense case
                M = op.matrix.tocoo()
                incidences = np.empty(len(M.col), dtype=[('row', np.int32), ('col', np.int32)])
                incidences['row'] = self.subspace_map[M.row]
                incidences['col'] = self.subspace_map[M.col]
                incidences = np.unique(incidences)
                self._non_zeros[op_id] = set(incidences.tolist())

            non_zeros = self._non_zeros[op_id]
            if all((ri, si) not in non_zeros for si, ri in product(source_ind, range_ind)):
                return None

            source_subspace = self.join_spaces(source_ind)
            range_subspace = self.join_spaces(range_ind)
            if issparse(op.matrix):
                m = op.matrix.tocsc()[:, source_subspace][range_subspace, :]
            else:
                m = op.matrix[:, source_subspace][range_subspace, :]

            result = NumpyMatrixOperator(m, name='{}-{}-{}'.format(op.name, source_ind, range_ind))
            self._resultcache[identification_key] = result
            return result

        elif isinstance(op, LincombOperator):
            ops = [self.localize_operator(o, range_ind, source_ind) for o in op.operators]
            return LincombOperator(ops, op.coefficients)
        else:
            print("op is ", op)
            raise NotImplementedError

    def localize_localizer(self, ind):
        return NumpyLocalizer(self.space, [self.subspaces[i] for i in ind])
