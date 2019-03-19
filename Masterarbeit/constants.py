from problems import *
#from pymor.discretizers.elliptic import discretize_elliptic_cg
from localizer import NumpyLocalizer
from partitioner import build_subspaces, partition_any_grid
#from pymor.discretizations.basic import StationaryDiscretization
from pou import *
from discrete_pou import *
from lrb_operator_projection import LRBOperatorProjection
#from pymor.algorithms.basisextension import *
from pymor.operators.numpy import NumpyGenericOperator
from pymor.operators.constructions import induced_norm
import scipy
import numpy as np
#from itertools import izip
import copy
import scipy.sparse.linalg as sp
from scipy.sparse.linalg import LinearOperator

#Berechne lambda_min:
def calculate_lambda_min(gq, lq):
	coarse_grid_resolution = gq["coarse_grid_resolution"]
	spaces = gq["spaces"]
	for space in spaces:
		ldict = lq[space]
		mat = ldict["source_product"].matrix
		val = scipy.sparse.linalg.eigsh(mat, return_eigenvectors = False, k=1, which="SM", tol = 1e-3)[0]
		#val = np.abs(np.sort(np.linalg.eig(mat)[0])[0])
		ldict["lambda_min"] = val
		#print "calculated lambda_min: ", val
	print("calculated all lambdas")

#Berechne inf-sup Konstante tilde_beta bezueglich des reduzierten Systems (vgl. S.9):
def calculate_inf_sup_constant(gq,lq, bases):#, mus):
	op = gq["op_fixed"]
	#op = gq["op_fixed_not_assembled"].assemble(mus)
	rhs = gq["rhs"]
	spaces = gq["spaces"]
	localizer = gq["localizer"]
	operator_reductor = LRBOperatorProjection(op, rhs, localizer, spaces, bases, spaces, bases)
	A = operator_reductor.get_reduced_operator().matrix

	#H10 = gq["h1_0_prod"]
	#operator_reductor0 = LRBOperatorProjection(H10, rhs, localizer, spaces, bases, spaces, bases)
	#Y = operator_reductor0.get_reduced_operator().matrix
	H1 = gq["k_product"]
	operator_reductor = LRBOperatorProjection(H1, rhs, localizer, spaces, bases, spaces, bases)
	X = operator_reductor.get_reduced_operator().matrix
	Y = operator_reductor.get_reduced_operator().matrix

	Yinv = sp.factorized(Y.astype(complex))
	def mv(v):
		return A.H.dot(Yinv(A.dot(v)))
	M1 = LinearOperator(A.shape, matvec = mv)

	eigvals = sp.eigs(M1, M=X, which = 'SM', tol = 1e-1)[0]
	eigvals = np.sqrt(np.abs(eigvals))
	eigvals.sort()
	result = eigvals[0]
	gq["inf_sup_constant"] =  result
	print("calculated_inf_sup_constant: ", result)
	return result

#Berechne inf-sup Konstante beta_h bezueglich des Finite-Elemente-Raumes (vgl. S.9):
def calculate_inf_sup_constant2(gq,lq):	
	op = gq["op"]
	A = op.matrix
	H1 = gq["k_product"].matrix
	Y = H1
	X = H1

	try:
		a = gq["data"]['boundary_info'].dirichlet_boundaries(2)
		b = np.arange(A.shape[0])
		c = np.delete(b,a)
		A = A[:,c][c,:]
		X = X[:,c][c,:]
		Y = Y[:,c][c,:]
	except KeyError:
		pass

	Yinv = sp.factorized(Y.astype(complex))
	def mv(v):
		return A.H.dot(Yinv(A.dot(v)))
	M1 = LinearOperator(A.shape, matvec = mv)

	#t = time.time()
	eigvals = sp.eigs(M1, M=X, which = 'SM', tol = 1e-1, k=100)[0]
	#print time.time()-t
	eigvals = np.sqrt(np.abs(eigvals))
	eigvals.sort()
	result = eigvals[0]
	gq["inf_sup_constant"] =  result
	print("calculated_inf_sup_constant: ", result)
	return result

#Berechne Stetigkeitskonstante (vgl. S.9):
def calculate_continuity_constant(gq, lq):#, mus):
	A = gq["op"].matrix
	#A = gq["d"].operator.assemble(mus).matrix	
	H1 = gq["k_product"].matrix
	Y = H1
	X = H1
	
	try:
		a = gq["data"]['boundary_info'].dirichlet_boundaries(2)
		b = np.arange(A.shape[0])
		c = np.delete(b,a)
		A = A[:,c][c,:]
		X = X[:,c][c,:]
		Y = Y[:,c][c,:]
	except KeyError:
		pass

	Yinv = sp.factorized(Y.astype(complex))
	def mv(v):
		return A.H.dot(Yinv(A.dot(v)))
	M1 = LinearOperator(A.shape, matvec = mv)
	eigvals = sp.eigs(M1, M=X, k=1, tol = 1e-4)[0]
	eigvals = np.sqrt(np.abs(eigvals))
	eigvals[::-1].sort()
	result = eigvals[0]
	gq["continuity_constant"] =  result
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
			x = np.linalg.lstsq(T, u_s.data.T)[0]
			y = T.dot(x)
			y_p = NumpyVectorSpace.make_array(y.T)
			y_p = y_p.lincomb(1/norm(y_p).real)
			result = np.sqrt(norm(u_s).real[0]**2/(norm(u_s).real[0]**2 - product.apply2(u_s, y_p).real[0][0]**2))
		ldict["csi"] = result
	print("calculated csis")
			

#Berechne Testlimit (vgl. Alg 2):
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

#Berechne globales Testlimit (vgl. Alg. 2):
def calculate_testlimit(gq, lq, space, num_testvecs, target_accuracy, max_failure_probability = 1e-15):
	ldict = lq[space]
	coarse_grid_resolution = gq["coarse_grid_resolution"]
	tol_i = target_accuracy*gq["inf_sup_constant"]/( (coarse_grid_resolution -1) *4 * gq["continuity_constant"]) / (ldict["csi"]*ldict["Psi_norm"])
	#print "tol_i: ", tol_i
	local_failure_tolerance = max_failure_probability / ( (coarse_grid_resolution -1)*4. )  #**2??
	testlimit_zeta = testlimit(
                failure_tolerance=local_failure_tolerance,
                dim_S=ldict["robin_transfer"].source.dim,
                dim_R=ldict["robin_transfer"].range.dim,
                num_testvecs=num_testvecs,
                target_error=tol_i,
                lambda_min=ldict["lambda_min"]
                )
	return testlimit_zeta

def calculate_Psi_norm(gq,lq):
	spaces = gq["spaces"]
	l = gq["localizer"]
	for space in spaces:
		ldict = lq[space]
		H1om = ldict["omega_star_product"].matrix
		MS = ldict["source_product"].matrix

		Q = ldict["solution_matrix_robin"]
		M = Q.T.conj().dot(H1om.dot(Q))
		eigval = sp.eigs(MS, M=M)[0][0].real

		ldict["Psi_norm"] = eigval
		#print "calculated Psi_norm: ", eigval
	print("calculated all Psi_norms")

"""
def calculate_Psi_norm_d(gq,lq):
	spaces = gq["spaces"]
	l = gq["localizer"]
	for space in spaces:
		ldict = lq[space]
		mat1 = ldict["omega_star_energy_product"]
		mat1 = mat1.matrix
		mat2 = ldict["source_product"]
	
		omega_star_space = ldict["omega_star_space"]
		omega_star_size = len(l.join_spaces(omega_star_space))

		source_space = ldict["source_space"]
		import ipdb; ipdb.set_trace()
		foo1 = NumpyVectorSpace.make_array(np.identity(omega_star_size))
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
		mat1 = mat1.matrix
		mat2 = ldict["source_product"].matrix

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

def calculate_continuity_constant2(gq, lq, bases):#, mus):
	op = gq["op"]
	A = op.matrix
	#A = gq["d"].operator.assemble(mus).matrix	
	H1 = gq["h1_prod"].matrix
	H1_0 = gq["h1_0_prod"].matrix
	Y = H1
	X = H1

	try:
		a = gq["data"]['boundary_info'].dirichlet_boundaries(2)
		b = np.arange(A.shape[0])
		c = np.delete(b,a)
		A = A[:,c][c,:]
		X = X[:,c][c,:]
		Y = Y[:,c][c,:]
	except KeyError:
		pass

	rhs = gq["rhs"]
	spaces = gq["spaces"]
	localizer = gq["localizer"]
	operator_reductor = LRBOperatorProjection(op, rhs, localizer, spaces, bases, spaces, bases)
	A = operator_reductor.get_reduced_operator().matrix

	H10 = gq["h1_0_prod"]
	operator_reductor0 = LRBOperatorProjection(H10, rhs, localizer, spaces, bases, spaces, bases)
	Y = operator_reductor0.get_reduced_operator().matrix
	H1 = gq["h1_prod"]
	operator_reductor = LRBOperatorProjection(H1, rhs, localizer, spaces, bases, spaces, bases)
	X = operator_reductor.get_reduced_operator().matrix

	Yinv = sp.factorized(Y.astype(complex))
	def mv(v):
		return A.H.dot(Yinv(A.dot(v)))
	M1 = LinearOperator(A.shape, matvec = mv)
	eigvals = sp.eigs(M1, M=X, k=1, tol = 1e-4)[0]
	eigvals = np.sqrt(np.abs(eigvals))
	eigvals[::-1].sort()
	result = eigvals[0]
	gq["continuity_constant"] =  result
	print "calculated_continuity_constant: ", result
	return result
"""
