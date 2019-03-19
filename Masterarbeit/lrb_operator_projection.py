import numpy as np

from pymor.operators.constructions import LincombOperator, VectorFunctional
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray

import scipy.sparse
from mybmat import mybmat
scipy.sparse.bmat = mybmat


class LRBOperatorProjection:
    def __init__(self, operator, rhs, localizer, range_spaces, range_bases, source_spaces, source_bases):
        self.range_spaces = [self._consolidate_space(s) for s in range_spaces]
        self.range_bases = [range_bases[s] for s in self.range_spaces]
        self.source_spaces = [self._consolidate_space(s) for s in source_spaces]
        self.source_bases = [source_bases[s] for s in self.source_spaces]
        self.l = localizer
        self.operator = operator
        self.rhs = rhs

        self.localized_operators = {}
        self.reduced_localized_operators = {}

        def generate_localized_operators(op):
            if isinstance(op, LincombOperator):
                for iop in op.operators:
                    generate_localized_operators(iop)
                return

            assert isinstance(op, NumpyMatrixOperator)
            for range_space in self.range_spaces:
                for source_space in self.source_spaces:
                    lop = self.l.localize_operator(op, range_space, source_space)
                    if lop:
                        identification_key = (op.uid, source_space, range_space)
                        self.localized_operators[identification_key] = lop

        generate_localized_operators(self.operator)

        self.reduction_todo_list = set(self.localized_operators.keys())
        self._generate_reduced_localized_operators()

        self.dirty_range_spaces = []
        self.dirty_source_spaces = []

    def set_range_basis(self, space, basis):
        space = self._consolidate_space(space)
        assert space in self.range_spaces
        self.range_bases[self.range_spaces.index(space)] = basis
        self.dirty_range_spaces.append(space)

    def set_source_basis(self, space, basis):
        space = self._consolidate_space(space)
        assert space in self.source_spaces
        self.source_bases[self.source_spaces.index(space)] = basis
        self.dirty_source_spaces.append(space)

    def get_reduced_operator(self):
        self._generate_reduction_todo_list()
        self._generate_reduced_localized_operators()
        return self._localizedly_project_operator(self.operator)

    def get_reduced_rhs(self):
        return self._localizedly_project_functional(self.rhs)

    def reconstruct_source(self, u):
        assert len(u) == 1
        v = u.data.ravel()
        offset = 0
        r = self.operator.source.zeros()

        for space, basis in zip(self.source_spaces, self.source_bases):
            if basis is None:
                #assert len(space) == 0 das macht doch kein Sinn
                space_dim = len(self.l.join_spaces(space))
                r += self.l.globalize_vector_array(NumpyVectorArray(v[offset:offset + space_dim]), space)
            else:
                space_dim = len(basis)
                r += self.l.globalize_vector_array(basis.lincomb(v[offset:offset + space_dim]), space)
            offset += space_dim

        return r

    def reconstruct_source_part(self, u, target_space):
        assert len(u) == 1
        v = u.data.ravel()
        offset = 0

        for space, basis in zip(self.source_spaces, self.source_bases):
            if basis is None:
                space_dim = len(self.l.join_spaces(space))
                if space == target_space:
                    return NumpyVectorArray(v[offset:offset + space_dim])
            else:
                space_dim = len(basis)
                if space == target_space:
                    return basis.lincomb(v[offset:offset + space_dim])

            offset += space_dim

        raise "das geht so nicht"



    def reconstruct_range(self, u):
        assert len(u) == 1
        v = u.data.ravel()
        offset = 0
        r = self.operator.range.zeros()

        for space, basis in zip(self.range_spaces, self.range_bases):
            if basis is None:
                assert len(space) == 0
                space_dim = len(self.l.join_spaces(space))
                self.l.globalize_vector_array_add_inplace(r, NumpyVectorArray(v[offset:offset + space_dim]), space)
            else:
                space_dim = len(basis)
                self.l.globalize_vector_array_add_inplace(r, basis.lincomb(v[offset:offset + space_dim]), space)
            offset += space_dim

        return r

    def _consolidate_space(self, space):
        if isinstance(space, (int, np.int32)):
            return (space,)
        else:
            return tuple(space)

    def _generate_reduced_localized_operators(self):
        for key in self.reduction_todo_list:
            self.reduced_localized_operators[key] \
                = self.localized_operators[key].projected(self.range_bases[self.range_spaces.index(key[2])],
                                                          self.source_bases[self.source_spaces.index(key[1])])

        self.reduction_todo_list = set()

    def _generate_reduction_todo_list(self):
        for key in self.localized_operators:
            if key[1] in self.dirty_source_spaces or key[2] in self.dirty_range_spaces:
                self.reduction_todo_list.add(key)

    def _localizedly_project_operator(self, op):
        if isinstance(op, LincombOperator):
            ops = [self._localizedly_project_operator(foo) for foo in op.operators]
            return LincombOperator(ops, coefficients=op.coefficients)
        assert op.linear and not op.parametric

        mats = [[None if (op.uid, source_space, range_space) not in self.reduced_localized_operators else
                 self.reduced_localized_operators[(op.uid, source_space, range_space)]._matrix
                 for source_space in self.source_spaces]
                for range_space in self.range_spaces]
        result =  NumpyMatrixOperator(scipy.sparse.bmat(mats).tocsc())
        return result

    def _localizedly_project_functional(self, op):
        if isinstance(op, LincombOperator):
            ops = [self._localizedly_project_functional(foo) for foo in op.operators]
            return LincombOperator(ops, coefficients=op.coefficients)

        assert op.linear and not op.parametric
        v = op.as_vector()

        def project_block(ids, basis):
            o = self.l.localize_vector_array(v, ids)
            if basis is None:
                return o.data
            else:
                return basis.dot(o).T

        mats = [project_block(ids, basis)
                for ids, basis in zip(self.range_spaces, self.range_bases)]

        return VectorFunctional(NumpyVectorArray(np.concatenate(mats, axis=1), copy=False))
