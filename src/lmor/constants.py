from pymor.operators.constructions import induced_norm
from pymor.vectorarrays.numpy import NumpyVectorSpace
from lmor.lrb_operator_projection import LRBOperatorProjection
import numpy as np
import scipy.sparse.linalg as sp
from scipy.sparse.linalg import LinearOperator


# Berechne lambda_min:
def calculate_lambda_min(gq, lq):
    spaces = gq["spaces"]
    for space in spaces:
        ldict = lq[space]
        mat = ldict["source_product"].matrix
        val = sp.eigsh(mat, return_eigenvectors=False, k=1, which="SM", tol=1e-3)[0]
        ldict["lambda_min"] = val
        # print "calculated lambda_min: ", val
    print("calculated all lambdas")


# Berechne inf-sup Konstante tilde_beta bezueglich des reduzierten Systems (vgl. S.9):
def calculate_inf_sup_constant(gq, lq, bases):
    op = gq["op_fixed"]
    rhs = gq["rhs"]
    spaces = gq["spaces"]
    localizer = gq["localizer"]
    operator_reductor = LRBOperatorProjection(op, rhs, localizer, spaces, bases, spaces, bases)
    A = operator_reductor.get_reduced_operator().matrix
    H1 = gq["k_product"]
    operator_reductor = LRBOperatorProjection(H1, rhs, localizer, spaces, bases, spaces, bases)
    X = operator_reductor.get_reduced_operator().matrix
    Y = operator_reductor.get_reduced_operator().matrix

    Yinv = sp.factorized(Y.astype(complex))

    def mv(v):
        return A.H.dot(Yinv(A.dot(v)))
    M1 = LinearOperator(A.shape, matvec=mv)

    eigvals = sp.eigs(M1, M=X, which='SM', tol=1e-4)[0]
    eigvals = np.sqrt(np.abs(eigvals))
    eigvals.sort()
    result = eigvals[0]
    gq["inf_sup_constant"] = result
    print("calculated_inf_sup_constant: ", result)
    return result


# Berechne inf-sup Konstante beta_h bezueglich des Finite-Elemente-Raumes (vgl. S.9):
def calculate_inf_sup_constant2(gq, lq):
    op = gq["op"]
    A = op.matrix
    H1 = gq["k_product"].matrix
    Y = H1
    X = H1

    try:
        a = gq["data"]['boundary_info'].dirichlet_boundaries(2)
        b = np.arange(A.shape[0])
        c = np.delete(b, a)
        A = A[:, c][c, :]
        X = X[:, c][c, :]
        Y = Y[:, c][c, :]
    except KeyError:
        pass

    Yinv = sp.factorized(Y.astype(complex))

    def mv(v):
        return A.H.dot(Yinv(A.dot(v)))
    M1 = LinearOperator(A.shape, matvec=mv)
    eigvals = sp.eigs(M1, M=X, which='SM', tol=1e-2, k=1)[0]
    eigvals = np.sqrt(np.abs(eigvals))
    eigvals.sort()
    result = eigvals[0]
    gq["inf_sup_constant"] = result
    print("calculated_inf_sup_constant: ", result)
    return result


# Berechne Stetigkeitskonstante (vgl. S.9):
def calculate_continuity_constant(gq, lq):
    A = gq["op"].matrix
    H1 = gq["k_product"].matrix
    Y = H1
    X = H1

    try:
        a = gq["data"]['boundary_info'].dirichlet_boundaries(2)
        b = np.arange(A.shape[0])
        c = np.delete(b, a)
        A = A[:, c][c, :]
        X = X[:, c][c, :]
        Y = Y[:, c][c, :]
    except KeyError:
        pass

    Yinv = sp.factorized(Y.astype(complex))

    def mv(v):
        return A.H.dot(Yinv(A.dot(v)))
    M1 = LinearOperator(A.shape, matvec=mv)
    eigvals = sp.eigs(M1, M=X, k=1, tol=1e-4)[0]
    eigvals = np.sqrt(np.abs(eigvals))
    eigvals[::-1].sort()
    result = eigvals[0]
    gq["continuity_constant"] = result
    print("calculated_continuity_constant: ", result)
    return result


def calculate_csis(gq, lq):
    spaces = gq["spaces"]
    for space in spaces:
        ldict = lq[space]
        T = ldict["solution_matrix_robin"]
        u_s = ldict["local_sol2"]
        product = ldict["omega_star_product"]
        norm = induced_norm(product)
        if norm(u_s).real < 1e-14:
            result = 1
        else:
            x = np.linalg.lstsq(T, u_s.data.T, rcond=None)[0]
            y = T.dot(x)
            y_p = NumpyVectorSpace.make_array(y.T)
            y_p = y_p.lincomb(1/norm(y_p).real)
            result = np.sqrt(norm(u_s).real[0]**2/(norm(u_s).real[0]**2 - product.apply2(u_s, y_p).real[0][0]**2))
        ldict["csi"] = result
    print("calculated csis")


def calculate_Psi_norm(gq, lq):
    spaces = gq["spaces"]
    for space in spaces:
        ldict = lq[space]
        H1om = ldict["omega_star_product"].matrix
        MS = ldict["source_product"].matrix

        Q = ldict["solution_matrix_robin"]
        M = Q.T.conj().dot(H1om.dot(Q))
        eigval = sp.eigs(MS, M=M, tol=1e-2, k=1)[0][0].real

        ldict["Psi_norm"] = eigval
        # print("calculated Psi_norm: ", eigval)
    print("calculated all Psi_norms")
