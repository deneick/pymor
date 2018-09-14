from problems import *
from pymor.discretizers.elliptic import discretize_elliptic_cg
from localizer import NumpyLocalizer
from partitioner import build_subspaces, partition_any_grid
from pymor.discretizations.basic import StationaryDiscretization
from pou import *
from discrete_pou import *
from lrb_operator_projection import LRBOperatorProjection
from pymor.algorithms.basisextension import *
from pymor.operators.numpy import NumpyGenericOperator
from pymor.operators.constructions import induced_norm
import scipy
import numpy as np
from itertools import izip
import copy
import scipy.sparse.linalg as sp

def calculate_lambda_min(gq, lq):
	coarse_grid_resolution = gq["coarse_grid_resolution"]
	spaces = gq["spaces"]
	for space in spaces:
		ldict = lq[space]
		mat = ldict["source_product"]._matrix
		#val = scipy.sparse.linalg.eigsh(mat, return_eigenvectors = False, k=1, which="SM", tol = 1e-3)[0]
		val = np.abs(np.sort(np.linalg.eig(mat)[0])[0])
		ldict["lambda_min"] = val
		print "calculated lambda_min: ", val
	print "calculated all lambdas"


def calculate_Psi_norm_d(gq,lq):
	spaces = gq["spaces"]
	l = gq["localizer"]
	for space in spaces:
		ldict = lq[space]
		mat1 = ldict["omega_star_energy_product"]
		mat1 = mat1._matrix
		mat2 = ldict["source_product"]
	
		omega_star_space = ldict["omega_star_space"]
		omega_star_size = len(l.join_spaces(omega_star_space))

		source_space = ldict["source_space"]
		import ipdb; ipdb.set_trace()
		foo1 = NumpyVectorArray(np.identity(omega_star_size))
		foo2 = l.to_space(foo1, omega_star_space, source_space)
		foo3 = mat2.apply(foo2)
		foo4 = l.to_space(foo3, source_space, omega_star_space)
		mat2 = scipy.sparse.csr_matrix(foo4.data)
		
		eigval = scipy.sparse.linalg.eigsh(mat2, M=mat1, k=1)
		maxval = np.sqrt(eigval[0][0])
		ldict["Psi_norm"] = maxval
		print "calculated Psi_norm: ", maxval
	print "calculated all Psi_norms"

def calculate_Psi_norm_r(gq,lq):
	spaces = gq["spaces"]
	l = gq["localizer"]
	for space in spaces:
		ldict = lq[space]
		mat1 = ldict["omega_star_energy_product"]
		mat1 = mat1._matrix
		mat2 = ldict["source_product"]._matrix

		omega_star_space = ldict["omega_star_space"]
		omega_star_size = len(l.join_spaces(omega_star_space))
		Q = ldict["solution_matrix_robin"]
		source_space = ldict["source_space"]
		psi = np.linalg.pinv(Q)		
		foo = psi.T.dot(mat2.dot(psi))
		foo = scipy.sparse.csr_matrix(foo)
		
		eigval = scipy.sparse.linalg.eigsh(foo, M=mat1, k=1)
		maxval = np.sqrt(eigval[0][0])
		ldict["Psi_norm"] = maxval
		print "calculated Psi_norm: ", maxval
	print "calculated all Psi_norms"

def calculate_inf_sup_constant(gq,lq, bases, k):
	#op = gq["op"]
	mus = {'k': k, 'c_glob': 6., 'c_loc': 6.}
	op = gq["d"].operator.assemble(mus)
	rhs = gq["rhs"]
	spaces = gq["spaces"]
	localizer = gq["localizer"]
	operator_reductor = LRBOperatorProjection(op, rhs, localizer, spaces, bases, spaces, bases)
	A = operator_reductor.get_reduced_operator()._matrix

	H10 = gq["h1_0_prod"]
	operator_reductor0 = LRBOperatorProjection(H10, rhs, localizer, spaces, bases, spaces, bases)
	Y = operator_reductor0.get_reduced_operator()._matrix
	H1 = gq["h1_prod"]
	operator_reductor = LRBOperatorProjection(H1, rhs, localizer, spaces, bases, spaces, bases)
	X = operator_reductor.get_reduced_operator()._matrix

	M = sp.inv(X).dot(A.H).dot(sp.inv(Y)).dot(A).todense()
	eigvals = np.linalg.eigvals(M)
	eigvals = np.sqrt(np.abs(eigvals))
	eigvals.sort()
	result = eigvals[0]
	gq["inf_sup_constant"] =  result
	return result

def calculate_inf_sup_constant2(gq,lq):
	A = gq["op"]._matrix
	H1 = gq["h1_prod"]._matrix
	H1_0 = gq["h1_0_prod"]._matrix
	Y = H1_0
	X = H1

	M = A.H.dot(sp.inv(Y)).dot(A)
	eigvals = sp.eigs(M, M=X, which = 'SM', tol = 1e-2)[0]
	eigvals = np.sqrt(np.abs(eigvals))
	eigvals.sort()
	result = eigvals[0]
	gq["inf_sup_constant"] =  result
	return result

def calculate_continuity_constant(gq, lq):
	A = gq["op"]._matrix
	H1 = gq["h1_prod"]._matrix
	H1_0 = gq["h1_0_prod"]._matrix
	Y = H1_0
	X = H1

	Yinv = sp.inv(Y)
	M = A.H.dot(Yinv).dot(A)
	eigvals = sp.eigs(M, M=X, k=1)[0]
	eigvals = np.sqrt(np.abs(eigvals))
	eigvals[::-1].sort()
	result = eigvals[0]
	gq["continuity_constant"] =  result
	return result

def testlimit(failure_tolerance, dim_S, dim_R, num_testvecs, target_error, lambda_min):
	"""
	failure_tolerance:  maximum probability for failure of algorithm
	dim_S: dimension of source space
	dim_R: dimension of range space
	num_testvecs: number of test vectors used
	target_error: desired maximal norm of tested operator
	lambda_min: smallest eigenvalue of matrix of inner product in source space
	"""
	return scipy.special.erfinv( (failure_tolerance / min(dim_S, dim_R))**(1./num_testvecs)) * target_error * np.sqrt(2. * lambda_min)

def calculate_testlimit(gq, lq, space, num_testvecs, target_accuracy, max_failure_probability = 1e-15):
	ldict = lq[space]
	coarse_grid_resolution = gq["coarse_grid_resolution"]
	tol_i = target_accuracy/( (coarse_grid_resolution -1)**2 *4) 
	local_failure_tolerance = max_failure_probability / ( (coarse_grid_resolution -1)*4. )
	testlimit_zeta = testlimit(
                failure_tolerance=local_failure_tolerance,
                dim_S=ldict["robin_transfer"].source.dim,
                dim_R=ldict["robin_transfer"].range.dim,
                num_testvecs=num_testvecs,
                target_error=tol_i,
                lambda_min=ldict["lambda_min"]
                )
	return testlimit_zeta
