"""
test berechnet und visualisiert die approximierte Loesung bzgl fester Basisgroesse

evaluation/evaluation_neumann berechnet den relativen Fehler abhaengig von der Basisgroesse

ungleichung berechnet die Genauigkeit der apriori Abschaetzung

accuracy berechnet den relativen Fehler abhaengig von der gewuenschten Genauigkeit gemaess des adaptiven Algos

kerr/kerr_neumann berechnet den relativen Fehler abhaengig von k

cerr berechnet den relativen Fehler abhaengig von c(dem lokalen Robin-Parameter)


zum plotten, setze plot=True
it = Anzahl der Wiederholungen fuer die Statistik
lim = maximale Basisgroesse
"""

from problems import *
import numpy as np
from pymor.basic import *
from localize_problem import *
from constants import *
from generate_solution import *

def evaluation(it, lim, k, boundary, save, c= None, plot = False, resolution = 200, coarse_grid_resolution = 10):
	import time
	p = helmholtz(boundary = boundary)
	if c is None:
		c=-1j*k
	mus = (k, -1j*k, c)
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
			print "i: ", i, "j: ", j
			print time.localtime(time.time()).tm_hour , " : ", time.localtime(time.time()).tm_min , " : ", time.localtime(time.time()).tm_sec
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

def ungleichung(it, lim, k, boundary, save, plot=False, resolution = 200, coarse_grid_resolution = 10):
	import time
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
			print i, j
			print time.localtime(time.time()).tm_hour , " : ", time.localtime(time.time()).tm_min , " : ", time.localtime(time.time()).tm_sec
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
				rssum2 += maxval**2
			ru = reconstruct_solution(gq,lq,bases)
			ls.append(d.h1_norm(u-ru)[0]/d.h1_norm(u)[0])
			rs2.append(4*np.sqrt(rssum2))
		LS.append(ls)
		RS2.append(rs2)
	means_ls = np.mean(LS, axis = 1)
	means_rs2 = np.mean(RS2, axis = 1)
	data = np.vstack([means_ls, means_rs2]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:	
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(range(lim), means_ls, label = "ls")
		plt.semilogy(range(lim), means_rs2, label = "rs2")
		plt.legend(loc='upper right')
		plt.xlabel('Basis size')
		plt.show()

def accuracy(it, num_testvecs, k, boundary, save, plot = False, resolution = 200, coarse_grid_resolution = 10):
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

def test(transfer = 'robin',boundary = 'dirichlet', n=15,k=6., c=6., title = 'test', resolution = 200, coarse_grid_resolution = 10):
	mus = (k, -1j*k, -1j*c)
	mu_glob = {'c': mus[1], 'k': mus[0]}
	p = helmholtz(boundary = boundary)
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus)
	basis = create_bases2(gq,lq,n,transfer = transfer)
	ru = reconstruct_solution(gq,lq,basis)
	d = gq["d"]
	u = d.solve(mu_glob)
	dif = u-ru
	d.visualize((dif.real, dif.imag, u.real, u.imag, ru.real, ru.imag), legend = ('dif.real', 'dif.imag', 'u.real', 'u.imag', 'ru.real', 'ru.imag'), separate_colorbars = True, title = title)

def kerr(it, n, boundary, save, c= None, rang = np.arange(0.2,50.,.2), plot = False, resolution = 200, coarse_grid_resolution = 10):
	#k/err
	err_d =[]
	err_r = []
	p = helmholtz(boundary = boundary)
	usec = not(c is None)
	for k in rang:
		print k
		if usec:
			mus = (k, -1j*k, c)
		else:
			mus = (k, -1j*k, -1j*k)
		mu_glob = {'c': mus[1], 'k': mus[0]}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
		d = gq["d"]
		u = d.solve(mu_glob)
		e_d = []
		e_r = []
		for i in range(it):
			print k, i
			bases = create_bases2(gq,lq,n,transfer = 'robin')
			ru_r = reconstruct_solution(gq, lq, bases)
			del bases
			bases = create_bases2(gq,lq,n,transfer = 'dirichlet')
			ru_d = reconstruct_solution(gq, lq, bases)
			del bases
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

def cerr(it, n, comp, k, c, boundary, save, rang = np.arange(-10.,10.,.1),plot = False, resolution = 200, coarse_grid_resolution = 10):
	#c/err
	err_r = []
	if comp == True:
		im = 1j
	else:
		im = 1
	p = helmholtz(boundary = boundary)
	for c_loc in rang:
		print c_loc
		mus = (k,-1j*c,-1*im*c_loc)
		mu_glob = {'c': mus[1], 'k': mus[0]}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
		d = gq["d"]
		u = d.solve(mu_glob)
		e_r = []
		for i in range(it):
			print c_loc, i
			bases = create_bases2(gq,lq,n,transfer = 'robin')
			ru_r = reconstruct_solution(gq, lq, bases)
			del bases
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

def cerr2D(it, n, k, boundary, save, rang = np.arange(-10.,10.,.5), plot = False, cglob = None, resolution = 200, coarse_grid_resolution = 10):
	#c/err
	err_r = np.zeros((len(rang),len(rang)))
	p = helmholtz(boundary = boundary)
	usec = not(cglob is None)
	xi = 0
	for x in rang:
		yi = 0
		for y in rang:
			c = x+1j*y
			print c
			if usec: 
				mus = (k,cglob, c)
			else:
				mus = (k,-1j*k,c)
			mu_glob = {'c': mus[1], 'k': mus[0]}
			gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
			d = gq["d"]
			u = d.solve(mu_glob)
			e_r = []
			for i in range(it):
				print c, i
				bases = create_bases2(gq,lq,n,transfer = 'robin')
				ru_r = reconstruct_solution(gq, lq, bases)
				del bases
				dif_r = u-ru_r
				e_r.append(np.minimum(1.,d.h1_norm(dif_r)[0]/d.h1_norm(u)[0]))
			err_r[xi][yi]=np.median(e_r)
			yi+=1
		xi+=1
	X,Y = np.meshgrid(rang, rang)
	data = np.vstack([Y.reshape(len(rang)**2,),X.reshape(len(rang)**2,),err_r.reshape(len(rang)**2,)]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:
		from mpl_toolkits.mplot3d import Axes3D
		import matplotlib.pyplot as plt
		from matplotlib import cm
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(X,Y,err_r,  cstride = 1, rstride =1, cmap = cm.coolwarm, linewidth=0, antialiased=False)
		fig.colorbar(surf, shrink =0.5, aspect=5)
		plt.show()

def cerr2Dp(it, n, k, boundary, save, rang = np.arange(-10.,10.,.5), plot = False, resolution = 200, coarse_grid_resolution = 10):
	import multiprocessing
	#c/err
	err_r = np.zeros((len(rang),len(rang)))
	prob = helmholtz(boundary = boundary)
	progress = 0
	def doit():
		bases = create_bases2(gq,lq,n,transfer = 'robin')
		ru_r = reconstruct_solution(gq, lq, bases)
		del bases
		dif_r = u-ru_r
		e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
	jobs = []
	xi = 0
	for x in rang:
		yi = 0
		for y in rang:
			c = x+1j*y
			print c
			mus = (k,-1j*k,c)
			mu_glob = {'c': mus[1], 'k': mus[0]}
			gq, lq = localize_problem(prob, coarse_grid_resolution, resolution, mus = mus)
			d = gq["d"]
			u = d.solve(mu_glob)
			e_r = []
			for i in range(it):
				print c, i
				p = multiprocessing.Process(target = doit)
				jobs.append(p)
				p.start()
			err_r[xi][yi]=np.mean(e_r)
			yi+=1
		xi+=1
	X,Y = np.meshgrid(rang, rang)
	data = np.vstack([Y.reshape(len(rang)**2,),X.reshape(len(rang)**2,),err_r.reshape(len(rang)**2,)]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	
