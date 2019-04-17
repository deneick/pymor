from pymor.models.basic import StationaryModel
from pymor.algorithms.randrangefinder import adaptive_rrf, rrf
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.numpy import NumpyMatrixOperator
from lmor.lrb_operator_projection import LRBOperatorProjection
from lmor.constants import (calculate_continuity_constant, calculate_inf_sup_constant2,
                            calculate_csis, calculate_Psi_norm)
import numpy as np
import scipy


def create_bases(gq, lq, num_testvecs, transfer='robin', testlimit=None,
                 target_accuracy=1e-3, max_failure_probability=1e-15, silent=True, calC=True):
    # adaptive Basiserstellung (Algorithmus 3)
    # Berechnung der Konstanten:
    if calC:
        if not silent:
            print("calculating constants")
        # calculate_lambda_min(gq, lq)
        calculate_Psi_norm(gq, lq)
        calculate_continuity_constant(gq, lq)
        calculate_inf_sup_constant2(gq, lq)
        calculate_csis(gq, lq)
    if not silent:
        print("creating bases")
    # Basisgenerierung:
    bases = {}
    for space in gq["spaces"]:
        ldict = lq[space]
        # Basis mit Shift-Loesung initialisieren:
        if transfer == 'dirichlet':
            lsol = ldict["local_solution_dirichlet"]
        else:
            lsol = ldict["local_solution_robin"]
        product = ldict["range_product"]
        if transfer == 'dirichlet':
            transop = ldict["dirichlet_transfer"]
        else:
            transop = ldict["robin_transfer"]

        tol_i = target_accuracy*gq["inf_sup_constant"] / \
            (2*4 * gq["continuity_constant"]) / (ldict["csi"]*ldict["Psi_norm"])
        local_failure_tolerance = max_failure_probability / ((gq["coarse_grid_resolution"] - 1)**2.)
        basis = adaptive_rrf(transop, ldict["source_product"], product, tol_i,
                             local_failure_tolerance, num_testvecs, True)
        basis.append(lsol)
        gram_schmidt(basis, product, copy=False)
        bases[space] = basis
    return bases


def create_bases2(gq, lq, basis_size, transfer='robin', silent=True):
    # nicht-adaptive Basiserstellung (Algorithmus 4)
    if not silent:
        print("creating bases")
    bases = {}
    for space in gq["spaces"]:
        ldict = lq[space]
        # Basis mit Shift-Loesung initialisieren:
        if transfer == 'dirichlet':
            lsol = ldict["local_solution_dirichlet"]
        else:
            lsol = ldict["local_solution_robin"]
        product = ldict["range_product"]
        if transfer == 'dirichlet':
            transop = ldict["dirichlet_transfer"]
        else:
            transop = ldict["robin_transfer"]
        basis = rrf(transop, ldict["source_product"], product, 0, basis_size, True)
        basis.append(lsol)
        gram_schmidt(basis, product, copy=False)
        bases[space] = basis
    return bases


def create_bases3(gq, lq, basis_size, q, transfer='robin', silent=True):
    # nicht-adaptive Basiserstellung mit power-iteration
    if not silent:
        print("creating bases")
    bases = {}
    for space in gq["spaces"]:
        ldict = lq[space]
        # Basis mit Shift-Loesung initialisieren:
        if transfer == 'dirichlet':
            lsol = ldict["local_solution_dirichlet"]
        else:
            lsol = ldict["local_solution_robin"]
        product = ldict["range_product"]
        if transfer == 'dirichlet':
            transop = NumpyMatrixOperator(ldict["transfer_matrix_dirichlet"])
        else:
            transop = NumpyMatrixOperator(ldict["transfer_matrix_robin"])
        basis = rrf(transop, ldict["source_product"], product, q, basis_size, True)
        basis.append(lsol)
        gram_schmidt(basis, product, copy=False)
        bases[space] = basis
    return bases


def reconstruct_solution(gq, lq, bases, silent=True):
    # Berechne reduzierte Loesung anhand gegebener Basis
    if not silent:
        print("reconstructing solution")
    op = gq["op"]
    rhs = gq["rhs"]
    localizer = gq["localizer"]
    spaces = gq["spaces"]
    operator_reductor = LRBOperatorProjection(op, rhs, localizer, spaces, bases, spaces, bases)
    rop = operator_reductor.get_reduced_operator()
    rrhs = operator_reductor.get_reduced_rhs()
    rd = StationaryModel(rop, rrhs, cache_region=None)
    ru = operator_reductor.reconstruct_source(rd.solve())
    return ru


def operator_svd(Top, source_inner, range_inner):
    sfac = scipy.sparse.linalg.factorized(source_inner)
    Tadj = sfac(Top.conj().T.dot(range_inner.todense()))
    blockmat = [[None, Tadj], [Top, None]]
    fullblockmat = scipy.sparse.bmat(blockmat).tocsc()
    w, v = np.linalg.eig(fullblockmat.todense())
    return np.abs(w[::2]), v[:source_inner.shape[0], ::2], v[source_inner.shape[0]:, ::2]


def operator_svd2(Top, source_inner, range_inner):
    mat_left = Top.conj().T.dot(range_inner.dot(Top))
    eigvals = scipy.linalg.eigvals(mat_left, source_inner.todense())
    eigvals = np.sqrt(np.abs(eigvals))
    eigvals[::-1].sort()
    return eigvals, None, None
