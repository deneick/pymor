from problems import *
import numpy as np
from pymor.basic import *
from localize_problem import *
from constants import *
from generate_solution import *

resolution = 50
coarse_grid_resolution = 10

def evaluation(it, lim, k, boundary, save, plot = False):
	p = helmholtz(boundary = boundary)
	mus = (k, -1j*k, -1j*k)
	mu_glob = {'c': mus[1], 'k': mus[0]}
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
	d = gq["d"]
	u = d.solve(mu_glob)
	h1_dirichlet = []
	h1_robin = []
	for i in range(lim):
		print "i: ", i
		h1d = []
		h1r = []
		for j in range(it):
			print "j: ", j
			basis_dirichlet = create_bases2(gq,lq,i)
			ru_dirichlet = reconstruct_solution(gq,lq,basis_dirichlet)
			del basis_dirichlet
			basis_robin = create_bases2(gq,lq,i,transfer = 'robin')
			ru_robin = reconstruct_solution(gq,lq,basis_robin)
			del basis_robin
			dif_dirichlet = u -ru_dirichlet
			dif_robin = u-ru_robin
			h1d.append(d.h1_norm(dif_dirichlet)[0]/d.h1_norm(u)[0])
			h1r.append(d.h1_norm(dif_robin)[0]/d.h1_norm(u)[0])
		h1_dirichlet.append(h1d)
		h1_robin.append(h1r)
	limits = [0, 25, 50, 75, 100]
	means_dirichlet_h1 = np.mean(h1_dirichlet, axis = 1)
	means_robin_h1 = np.mean(h1_robin, axis = 1)
	percentiles_dirichlet_h1 = np.array(np.percentile(h1_dirichlet, limits, axis=1))
	percentiles_robin_h1 = np.array(np.percentile(h1_robin, limits, axis=1))
	#import ipdb; ipdb.set_trace()
	data = np.vstack([means_dirichlet_h1, percentiles_dirichlet_h1, means_robin_h1, percentiles_robin_h1]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(range(lim), means_dirichlet_h1, label = "Dirichlet h1")
		plt.semilogy(range(lim), means_robin_h1, label = "Robin h1")
		plt.legend(loc='upper right')
		plt.xlabel('Basis size')
		plt.show()

def evaluation_neumann(it, lim, k, boundary, save, plot = False):
	p = helmholtz(boundary = boundary)
	mus = (k, -1j*k, 0)
	mu_glob = {'c': mus[1], 'k': mus[0]}
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
	d = gq["d"]
	u = d.solve(mu_glob)
	h1_n = []
	for i in range(lim):
		print "i: ", i
		h1n = []
		for j in range(it):
			print "j: ", j
			basis_n = create_bases2(gq,lq,i, transfer = 'robin')
			ru_n = reconstruct_solution(gq,lq,basis_n)
			del basis_n
			dif_n = u -ru_n
			h1n.append(d.h1_norm(dif_n)[0]/d.h1_norm(u)[0])
		h1_n.append(h1n)
	limits = [0, 25, 50, 75, 100]
	means_n = np.mean(h1_n, axis = 1)
	percentiles_n = np.array(np.percentile(h1_n, limits, axis=1))
	data = np.vstack([means_n, percentiles_n]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(range(lim), means_n, label = "Neumann")
		plt.legend(loc='upper right')
		plt.xlabel('Basis size')
		plt.show()

def ungleichung(it, lim, k, boundary, save, plot=False):
	p = helmholtz(boundary = boundary)
	mus = (k, -1j*k, -1j*k)
	mu_glob = {'c': mus[1], 'k': mus[0]}
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus, calT = True, calQ = True)
	d = gq["d"]
	u = d.solve(mu_glob)	
	LS = []
	RS = []
	RS2 = []
	for i in range(lim):
		print i
		for j in range(it):
			ls = []
			rs = []
			rs2 = []
			bases = create_bases2(gq,lq,i,transfer = 'robin')
			rssum = 0
			rssum2 = 0
			for space in gq["spaces"]:
				ldict = lq[space]
				basis = bases[space]
				M = ldict["range_product"]._matrix
				S = ldict["source_product"]._matrix
				M_sparse = scipy.sparse.csr_matrix(M)
				T = ldict["transfer_matrix_robin"]
				B = basis._array.T
				T1 = T - B.dot(B.conj().T).dot(M_sparse.dot(T))
				maxval = operator_svd2(T1, S, M_sparse)[0][0]
				rssum += maxval
				rssum2 += maxval**2
			ru = reconstruct_solution(gq,lq,bases)
			ls.append(d.h1_norm(u-ru)[0]/d.h1_norm(u)[0])
			rs.append(rssum)
			rs2.append(4*np.sqrt(rssum2))
		LS.append(ls)
		RS.append(rs)
		RS2.append(rs2)
	means_ls = np.mean(LS, axis = 1)
	means_rs = np.mean(RS, axis = 1)
	means_rs2 = np.mean(RS2, axis = 1)
	data = np.vstack([means_ls, means_rs, means_rs2]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	"""limits = [0, 25, 50, 75, 100]
	percentiles_ls = np.array(np.percentile(LS, limits, axis=1))
	percentiles_rs = np.array(np.percentile(RS, limits, axis=1))
	open("dats/percentiles_ls.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in np.vstack((percentiles_ls.T,))])
	open("dats/percentiles_rs.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in np.vstack((percentiles_rs.T,))])	"""
	if plot:	
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(range(lim), means_ls, label = "ls")
		plt.semilogy(range(lim), means_rs, label = "rs")
		plt.semilogy(range(lim), means_rs2, label = "rs2")
		plt.legend(loc='upper right')
		plt.xlabel('Basis size')
		plt.show()

def accuracy(it, num_testvecs, k, boundary, save, plot = False):
	#tol/err
	p = helmholtz(boundary = boundary)
	mus = (k, -1j*k, -1j*k)
	mu_glob = {'c': mus[1], 'k': mus[0]}
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus, calQ = True)
	logspace = np.logspace(5, -10, num = 20)
	d = gq["d"]
	u = d.solve(mu_glob)
	ERR = []
	calculate_lambda_min(gq, lq)
	for target_accuracy in logspace:
		print target_accuracy
		for i in range(it):
			print i
			err = []
			#import ipdb; ipdb.set_trace()
			bases = create_bases(gq, lq, num_testvecs, transfer = 'robin', target_accuracy = target_accuracy)
			ru = reconstruct_solution(gq, lq, bases)
			err.append(d.h1_norm(u-ru)[0]/d.h1_norm(u)[0])
		ERR.append(err)
	means = np.mean(ERR, axis = 1)
	data1 = np.vstack([logspace, means]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data1])
	"""limits = [0, 25, 50, 75, 100]
	percentiles = np.array(np.percentile(ERR, limits, axis=1))
	data2 = np.vstack([logspace, percentiles]).T
	open("dats/percentiles_accuracy.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in data2])"""
	if plot:
		from matplotlib import pyplot as plt
		plt.figure()
		plt.loglog(logspace, means, label = "err")
		plt.legend(loc='upper right')
		plt.xlabel('target_accuracy')
		plt.gca().invert_xaxis()
		plt.show()

def test(transfer = 'robin', c=6):
	k=6	
	#k = np.pi*np.sqrt(2)*10/4
	mus = (k, -1j*k, -1j*k)
	mu_glob = {'c': mus[1], 'k': mus[0]}
	p = helmholtz(boundary = 'robin')
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus)
	basis = create_bases2(gq,lq,15,transfer = transfer)
	ru = reconstruct_solution(gq,lq,basis)
	d = gq["d"]
	u = d.solve(mu_glob)
	dif = u-ru
	d.visualize((dif.real, dif.imag, u.real, u.imag, ru.real, ru.imag), legend = ('dif.real', 'dif.imag', 'u.real', 'u.imag', 'ru.real', 'ru.imag'), separate_colorbars = True)

def kerr(it, n, boundary, save, plot = False):
	#k/err
	err_d =[]
	err_r = []
	rang = np.arange(0.,20.,.5)
	p = helmholtz(boundary = boundary)
	for k in rang:
		print k
		mus = (k, -1j*k, -1j*k)
		mu_glob = {'c': mus[1], 'k': mus[0]}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
		e_d = []
		e_r = []
		for i in range(it):
			bases = create_bases2(gq,lq,n,transfer = 'robin')
			ru_r = reconstruct_solution(gq, lq, bases)
			del bases
			bases = create_bases2(gq,lq,n,transfer = 'dirichlet')
			ru_d = reconstruct_solution(gq, lq, bases)
			del bases
			d = gq["d"]
			u = d.solve(mu_glob)
			dif_d = u-ru_d
			dif_r = u-ru_r
			e_d.append(d.h1_norm(dif_d)[0]/d.h1_norm(u)[0])
			e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
		err_d.append(e_d)
		err_r.append(e_r)
	means_d = np.mean(err_d, axis = 1)
	means_r = np.mean(err_r, axis = 1)
	data = np.vstack([rang, means_d, means_r]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:	
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(rang, means_r, label = "robin")
		plt.semilogy(rang, means_d, label = "dirichlet")
		plt.xlabel('k')
		plt.legend(loc='upper right')
		plt.show()

def kerr_neumann(it, n, boundary, save, plot = False):
	#k/err
	err_n =[]
	rang = np.arange(0.,20.,.1)
	p = helmholtz(boundary = boundary)
	for k in rang:
		print k
		mus = (k, -1j*k, 0)
		mu_glob = {'c': mus[1], 'k': mus[0]}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
		e_n = []
		for i in range(it):
			bases = create_bases2(gq,lq,n,transfer = 'robin')
			ru_n = reconstruct_solution(gq, lq, bases)
			del bases
			d = gq["d"]
			u = d.solve(mu_glob)
			dif_n = u-ru_n
			e_n.append(d.h1_norm(dif_n)[0]/d.h1_norm(u)[0])
		err_n.append(e_n)
	means_n = np.mean(err_n, axis = 1)
	data = np.vstack([rang, means_n]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:	
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(rang, means_n, label = "neumann")
		plt.xlabel('k')
		plt.legend(loc='upper right')
		plt.show()

def cerr(it, n, comp, k, c, boundary, save, plot = False):
	#c/err
	err_r = []
	if comp == True:
		im = 1j
	else:
		im = 1
	rang = np.arange(-100,100,1.)
	p = helmholtz(boundary = boundary)
	for c_loc in rang:
		print c_loc
		mus = (k,-1j*c,-1*im*c_loc)
		mu_glob = {'c': mus[1], 'k': mus[0]}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
		e_r = []
		for i in range(it):
			bases = create_bases2(gq,lq,n,transfer = 'robin')
			ru_r = reconstruct_solution(gq, lq, bases)
			del bases
			d = gq["d"]
			u = d.solve(mu_glob)
			dif_r = u-ru_r
			e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
		err_r.append(e_r)
	means_r = np.mean(err_r, axis = 1)
	#data_r = np.vstack([rang, means_r]).T
	limits = [0, 25, 50, 75, 100]
	percentiles_r = np.array(np.percentile(err_r, limits, axis=1))
	data = np.vstack([rang, means_r, percentiles_r]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])	
	if plot:	
		from matplotlib import pyplot as plt
		plt.figure()
		plt.plot(rang, means_r, label = "robin")
		plt.xlabel('c')
		plt.legend(loc='upper right')
		plt.show()
