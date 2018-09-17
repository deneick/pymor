from pymor.discretizations.basic import StationaryDiscretization
from lrb_operator_projection import LRBOperatorProjection
from pymor.algorithms.basisextension import *
from pymor.operators.constructions import induced_norm
from pymor.vectorarrays.numpy import NumpyVectorArray
import numpy as np
import scipy
from constants import *

def create_bases(gq, lq, num_testvecs, transfer = 'dirichlet', testlimit = None, target_accuracy = 1e-3, silent = True):
	#adaptive Basiserstellung
	if testlimit is None:
		if not silent:
			print "calculating constants"
		#calculate_lambda_min(gq, lq)
		#calculate_Psi_norm(gq,lq)
		#calculate_continuity_constant(gq, lq)
		#calculate_inf_sup_constant2(gq, lq)
	if not silent:
		print "creating bases"
	bases = {}
	for space in gq["spaces"]:
		ldict = lq[space]
		if transfer == 'dirichlet':
			lsol = ldict["local_solution_dirichlet"]
		else: 
			lsol = ldict["local_solution_robin"]
		basis = lsol.copy()
		product = ldict["range_product"]
		norm = induced_norm(product)
		gram_schmidt(basis, copy=False, product = product)
		if transfer == 'dirichlet':
			transop = ldict["dirichlet_transfer"]
		else: 
			transop = ldict["robin_transfer"]

		if testlimit is None:		
			testlimit = calculate_testlimit(gq, lq, space, num_testvecs, target_accuracy)	
	
		testvecs = transop.apply(NumpyVectorArray(np.random.normal(size=transop.source.dim)))
		for i in range(num_testvecs-1):
			testvecs.append(transop.apply(NumpyVectorArray(np.random.normal(size=(1, transop.source.dim)))))
		maxnorm = float('inf')
		#print "Testlimit: ", testlimit
		progress = 1.
		while (maxnorm > testlimit and progress > 1e-16):
			if (progress <0):
				raise Exception
			vec = np.random.normal(size=(1,transop.source.dim))+1j*np.random.normal(size=(1,transop.source.dim))
			u_t = transop.apply(NumpyVectorArray(vec))
			basis_length = len(basis)
			basis.append(u_t)
			gram_schmidt(basis, offset=basis_length, product = product, copy = False)
			B = basis._array.T
			testvecs._array -= B.dot(B.conj().T.dot(product._matrix.dot(testvecs._array.T))).T
        		maxnorm_new = np.max(np.abs(norm(testvecs)))
			progress = maxnorm-maxnorm_new
			maxnorm = maxnorm_new
			#print "Maxnorm: ", maxnorm
		#print "Basisgroesse: ", len(basis)
		bases[space] = basis
	return bases

def create_bases2(gq, lq, basis_size, transfer = 'dirichlet', silent = True):
	#Basiserstellung mit Basisgroesse
	if not silent:
		print "creating bases"
	bases = {}
	for space in gq["spaces"]:
		ldict = lq[space]
		if transfer == 'dirichlet':
			lsol = ldict["local_solution_dirichlet"]
		else: 
			lsol = ldict["local_solution_robin"]
		product = ldict["range_product"]
		basis = lsol.copy()
		gram_schmidt(basis, copy=False, product = product)
		if transfer == 'dirichlet':
			transop = ldict["dirichlet_transfer"]
		else: 
			transop = ldict["robin_transfer"]
		for i in range(basis_size):
			vec = np.random.normal(size=(1,transop.source.dim))+1j*np.random.normal(size=(1,transop.source.dim))
			u_t = transop.apply(NumpyVectorArray(vec))
			basis_length = len(basis)
			basis.append(u_t)
			gram_schmidt(basis, offset=basis_length, product = product, copy = False)
		bases[space] = basis
	return bases

def reconstruct_solution(gq, lq, bases, silent = True):
	if not silent:
		print "reconstructing solution"	
	op = gq["op"]
	rhs = gq["rhs"]
	localizer = gq["localizer"]
	spaces = gq["spaces"]
	operator_reductor = LRBOperatorProjection(op, rhs, localizer, spaces, bases, spaces, bases)
	rop = operator_reductor.get_reduced_operator()
	rrhs = operator_reductor.get_reduced_rhs()
	rd = StationaryDiscretization(rop,rrhs,cache_region=None)
	rdu = rd.solve()
	ru = operator_reductor.reconstruct_source(rd.solve())
	return ru

def operator_svd(Top, source_inner, range_inner):
    sfac = scipy.sparse.linalg.factorized(source_inner)
    Tadj = sfac(Top.H.dot(range_inner.todense()))
    blockmat = [[None, Tadj], [Top, None]]
    fullblockmat = scipy.sparse.bmat(blockmat).tocsc()
    w,v = np.linalg.eig(fullblockmat.todense())
    return np.abs(w[::2]), v[:source_inner.shape[0],::2], v[source_inner.shape[0]:, ::2]

def operator_svd2(Top, source_inner, range_inner):
    mat_left = Top.H.dot(range_inner.dot(Top))
    mat_right = source_inner
    eigvals = scipy.linalg.eigvals(mat_left, mat_right)
    eigvals = np.sqrt(np.abs(eigvals))
    eigvals[::-1].sort()
    return eigvals, None, None
