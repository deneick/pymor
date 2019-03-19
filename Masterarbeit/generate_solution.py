from pymor.discretizations.basic import StationaryDiscretization
from lrb_operator_projection import LRBOperatorProjection
from pymor.algorithms.gram_schmidt import *
from pymor.operators.constructions import induced_norm
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace
import numpy as np
import scipy
from constants import *
import multiprocessing as mp

#adaptive Basiserstellung (Algorithmus 3)
def create_bases(gq, lq, num_testvecs, transfer = 'dirichlet', testlimit = None, target_accuracy = 1e-3, max_failure_probability = 1e-15, silent = True, calC = True):
	#Berechnung der Konstanten:	
	if calC:
		if not silent:
			print("calculating constants")
		calculate_lambda_min(gq, lq)
		calculate_Psi_norm(gq,lq)
		calculate_continuity_constant(gq, lq)
		calculate_inf_sup_constant2(gq, lq)
		calculate_csis(gq,lq)
	if not silent:
		print("creating bases")
	#Basisgenerierung:
	bases = {}
	for space in gq["spaces"]:
		ldict = lq[space]
		#Basis mit Shift-Loesung initialisieren:
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

		#testlimit berechnen:
		testlimit = calculate_testlimit(gq, lq, space, num_testvecs, target_accuracy, max_failure_probability)	
	
		#Generierung der Testvektoren:
		testvecs = transop.apply(NumpyVectorSpace.make_array(np.random.normal(size=transop.source.dim)))
		for i in range(num_testvecs-1):
			testvecs.append(transop.apply(NumpyVectorSpace.make_array(np.random.normal(size=(1, transop.source.dim)))))
		maxnorm = float('inf')
		#print "Testlimit: ", testlimit
		progress = 1.
		while (maxnorm > testlimit and progress > 1e-16):
			if (progress <0):
				raise Exception
			#normalverteilter Zufallsvektor r:
			vec = np.random.normal(size=(1,transop.source.dim))+1j*np.random.normal(size=(1,transop.source.dim))
			#u_t = T(r)
			u_t = transop.apply(NumpyVectorSpace.make_array(vec))
			basis_length = len(basis)
			#B <- B+T(r)
			basis.append(u_t)
			#orthonormalisieren
			gram_schmidt(basis, offset=basis_length, product = product, copy = False)
			B = basis._array.T
			#M <- t- P_span(B) t /t in M
			testvecs._array -= B.dot(B.conj().T.dot(product.matrix.dot(testvecs._array.T))).T
			maxnorm_new = np.max(np.abs(norm(testvecs)))
			progress = maxnorm-maxnorm_new
			maxnorm = maxnorm_new
			#print "Maxnorm: ", maxnorm
		#print "Basisgroesse: ", len(basis)
		bases[space] = basis
	return bases
"""
	#Fuer den Fall beta_tilde /= beta_h (vgl. S. 45):
	print "inf-sup Konstante wird neu berechnet"
	inf_sup_old = gq["inf_sup_constant"].copy()
	inf_sup_new = calculate_inf_sup_constant(gq,lq,bases)
	if np.abs((inf_sup_old-inf_sup_new)/inf_sup_old) > 1e-2:	
		testlimit = calculate_testlimit(gq, lq, space, num_testvecs, target_accuracy, max_failure_probability)
		for space in gq["spaces"]:
			ldict = lq[space]
			basis = bases[space]
			product = ldict["range_product"]
			norm = induced_norm(product)
			if transfer == 'dirichlet':
				transop = ldict["dirichlet_transfer"]
			else: 
				transop = ldict["robin_transfer"]
			testvecs = transop.apply(NumpyVectorSpace.make_array(np.random.normal(size=transop.source.dim)))
			for i in range(num_testvecs-1):
				testvecs.append(transop.apply(NumpyVectorSpace.make_array(np.random.normal(size=(1, transop.source.dim)))))
			B = basis._array.T
			testvecs._array -= B.dot(B.conj().T.dot(product.matrix.dot(testvecs._array.T))).T
			maxnorm = np.max(np.abs(norm(testvecs)))
			progress = 1.
			while (maxnorm > testlimit and progress > 1e-16):
				print "Basis nochmal erweitern"
				#import ipdb
				#ipdb.set_trace()
				if (progress <0):
					raise Exception
				vec = np.random.normal(size=(1,transop.source.dim))+1j*np.random.normal(size=(1,transop.source.dim))
				u_t = transop.apply(NumpyVectorSpace.make_array(vec))
				basis_length = len(basis)
				basis.append(u_t)
				gram_schmidt(basis, offset=basis_length, product = product, copy = False)
				B = basis._array.T
				testvecs._array -= B.dot(B.conj().T.dot(product.matrix.dot(testvecs._array.T))).T
				maxnorm_new = np.max(np.abs(norm(testvecs)))
				progress = maxnorm-maxnorm_new
				maxnorm = maxnorm_new
				#print "Maxnorm: ", maxnorm
			#print "Basisgroesse: ", len(basis)
			bases[space] = basis
	return bases
"""

#nicht-adaptive Basiserstellung (Algorithmus 4):
def create_bases3(gq, lq, basis_size, transfer = 'dirichlet', silent = True):
	#Basiserstellung mit Basisgroesse
	if not silent:
		print("creating bases")
	bases = {}
	for space in gq["spaces"]:
		ldict = lq[space]
		#Basis mit Shift-Loesung initialisieren:
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
		global cube
		def cube(i):
			vec = np.random.normal(size=(1,transop.source.dim))+1j*np.random.normal(size=(1,transop.source.dim))	
			u_t = transop.apply(NumpyVectorSpace.make_array(vec))
			return u_t
		pool = mp.Pool(processes=3)
		basislist = pool.map(cube,  range(basis_size))
		for i in range(basis_size): basis.append(basislist[i])
		gram_schmidt(basis, product = product, copy = False)
		bases[space] = basis
	return bases

#nicht-adaptive Basiserstellung (Algorithmus 4):
def create_bases2(gq, lq, basis_size, transfer = 'dirichlet', silent = True):
	#Basiserstellung mit Basisgroesse
	if not silent:
		print("creating bases")
	bases = {}
	for space in gq["spaces"]:
		ldict = lq[space]
		#Basis mit Shift-Loesung initialisieren:
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
			#normalverteilter Zufallsvektor r:
			vec = np.random.normal(size=(1,transop.source.dim))+1j*np.random.normal(size=(1,transop.source.dim))	
			u_t = transop.apply(NumpyVectorSpace.make_array(vec))
			basis_length = len(basis)
			#B <- B + T(r)
			basis.append(u_t)
			#orthonormalisieren
			gram_schmidt(basis, offset=basis_length, product = product, copy = False)
		bases[space] = basis
	return bases

#Berechne reduzierte Loesung anhand gegebener Basis:
def reconstruct_solution(gq, lq, bases, silent = True):
	if not silent:
		print("reconstructing solution")
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
    Tadj = sfac(Top.conj().T.dot(range_inner.todense()))
    blockmat = [[None, Tadj], [Top, None]]
    fullblockmat = scipy.sparse.bmat(blockmat).tocsc()
    w,v = np.linalg.eig(fullblockmat.todense())
    return np.abs(w[::2]), v[:source_inner.shape[0],::2], v[source_inner.shape[0]:, ::2]

def operator_svd2(Top, source_inner, range_inner):
    mat_left = Top.conj().T.dot(range_inner.dot(Top))
    mat_right = source_inner
    eigvals = scipy.sparse.linalg.eigs(mat_left, M = mat_right, k=1)[0]
    eigvals = np.sqrt(np.abs(eigvals))
    eigvals[::-1].sort()
    return eigvals, None, None
