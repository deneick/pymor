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
import sys
from pymor.basic import *
from localize_problem import *
from constants import *
from generate_solution import *
import time

if not os.path.exists("dats"):
	os.makedirs("dats")
set_log_levels(levels={'pymor': 'WARN'})

def evaluation(it, lim, k, boundary, save, cglob = 0, cloc = 0, plot = False, resolution = 200, coarse_grid_resolution = 10):
	#import time
	p = helmholtz(boundary = boundary)
	mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
	d = gq["d"]
	u = d.solve(mus)
	h1_dirichlet = []
	h1_robin = []
	nrang = np.arange(0,lim,2)
	for i in nrang:
		print "i: ", i
		h1d = []
		h1r = []
		for j in range(it):
			print j,
			sys.stdout.flush()
			#print time.localtime(time.time()).tm_hour , " : ", time.localtime(time.time()).tm_min , " : ", time.localtime(time.time()).tm_sec
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
	#percentiles_dirichlet_h1 = np.array(np.percentile(h1_dirichlet, limits, axis=1))
	#percentiles_robin_h1 = np.array(np.percentile(h1_robin, limits, axis=1))
	#import ipdb; ipdb.set_trace()
	#data = np.vstack([means_dirichlet_h1, percentiles_dirichlet_h1, means_robin_h1, percentiles_robin_h1]).T
	data = np.vstack([nrang, means_dirichlet_h1, means_robin_h1]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(nrang, means_dirichlet_h1, label = "Dirichlet h1")
		plt.semilogy(nrang, means_robin_h1, label = "Robin h1")
		plt.legend(loc='upper right')
		plt.xlabel('Basis size')
		plt.show()

def ungleichung(it, k, boundary, save, nrang  = np.arange(0,100,5), cglob = 0, cloc = 0, plot=False, resolution = 200, coarse_grid_resolution = 10):
	#import time
	p = helmholtz(boundary = boundary)
	mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus, calT = True, calQ = True)
	calculate_continuity_constant(gq, lq)
	calculate_inf_sup_constant2(gq, lq)
	d = gq["d"]
	u = d.solve(mus)	
	LS = []
	RS2 = []
	for n in nrang:
		print n
		for j in range(min(20-n/5,it)):
			print j,
			sys.stdout.flush()
			#print time.localtime(time.time()).tm_hour , " : ", time.localtime(time.time()).tm_min , " : ", time.localtime(time.time()).tm_sec
			ls = []
			rs2 = []
			bases = create_bases2(gq,lq,n,transfer = 'robin')
			rssum2 = 0
			for space in gq["spaces"]:
				ldict = lq[space]
				basis = bases[space]
				M = ldict["range_product"]._matrix
				S = ldict["source_product"]._matrix
				M_sparse = scipy.sparse.csr_matrix(M)
				T = ldict["transfer_matrix_robin"]
				B = basis._array.T
				T1 = T - B.dot(B.H).dot(M_sparse.dot(T))
				maxval = operator_svd2(T1, S, M_sparse)[0][0]
				rssum2 += maxval**2
			ru = reconstruct_solution(gq,lq,bases)
			ls.append(d.h1_norm(u-ru)[0]/d.h1_norm(u)[0])
			rs2.append((gq["continuity_constant"]/gq["inf_sup_constant"])*4*np.sqrt(rssum2))
		LS.append(ls)
		RS2.append(rs2)
	means_ls = np.mean(LS, axis = 1)
	means_rs2 = np.mean(RS2, axis = 1)
	data = np.vstack([nrang, means_ls, means_rs2]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:	
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(nrang, means_ls, label = "ls")
		plt.semilogy(nrang, means_rs2, label = "rs2")
		plt.legend(loc='upper right')
		plt.xlabel('Basis size')
		plt.show()

def ungleichungk(it, n, boundary, save, krang  = np.arange(0.1,50.1,0.2), cloc0 = 0, cloc1 = 1, cloc2 = 1, plot=False, resolution = 100, coarse_grid_resolution = 10):
	#import time
	p = helmholtz(boundary = boundary)	
	LS = []
	RS2 = []
	for k in krang:
		print k
		cglob = -1j*k
		cloc = cloc0+ cloc1*k+cloc2*k**2
		mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus, calT = True, calQ = True)
		calculate_continuity_constant(gq, lq)
		calculate_inf_sup_constant2(gq, lq)		
		d = gq["d"]
		u = d.solve(mus)
		for j in range(min(20-n/5,it)):
			print j,
			sys.stdout.flush()
			#print time.localtime(time.time()).tm_hour , " : ", time.localtime(time.time()).tm_min , " : ", time.localtime(time.time()).tm_sec
			ls = []
			rs2 = []
			bases = create_bases2(gq,lq,n,transfer = 'robin')
			rssum2 = 0
			for space in gq["spaces"]:
				ldict = lq[space]
				basis = bases[space]
				M = ldict["range_product"]._matrix
				S = ldict["source_product"]._matrix
				M_sparse = scipy.sparse.csr_matrix(M)
				T = ldict["transfer_matrix_robin"]
				B = basis._array.T
				T1 = T - B.dot(B.H).dot(M_sparse.dot(T))
				maxval = operator_svd2(T1, S, M_sparse)[0][0]
				rssum2 += maxval**2
			ru = reconstruct_solution(gq,lq,bases)
			ls.append(d.h1_norm(u-ru)[0]/d.h1_norm(u)[0])
			rs2.append((gq["continuity_constant"]/gq["inf_sup_constant"])*4*np.sqrt(rssum2))
		LS.append(ls)
		RS2.append(rs2)
	means_ls = np.mean(LS, axis = 1)
	means_rs2 = np.mean(RS2, axis = 1)
	data = np.vstack([nrang, means_ls, means_rs2]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:	
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(nrang, means_ls, label = "ls")
		plt.semilogy(nrang, means_rs2, label = "rs2")
		plt.legend(loc='upper right')
		plt.xlabel('Basis size')
		plt.show()

def accuracy(it, num_testvecs, k, boundary, save, cglob = 0, cloc = 0, plot = False, resolution = 100, coarse_grid_resolution = 10):
	#tol/err
	p = helmholtz(boundary = boundary)
	mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus, calQ = True)
	logspace = np.logspace(5, -10, num = 20)
	d = gq["d"]
	u = d.solve(mus)
	ERR = []
	calculate_lambda_min(gq, lq)
	calculate_continuity_constant(gq, lq)
	calculate_inf_sup_constant2(gq, lq)
	for target_accuracy in logspace:
		print target_accuracy
		for i in range(it):
			print i,
			sys.stdout.flush()
			err = []
			#import ipdb; ipdb.set_trace()
			bases = create_bases(gq, lq, num_testvecs, transfer = 'robin', target_accuracy = target_accuracy, calC = False)
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

def accuracyk(it, num_testvecs, acc, boundary, save, cloc0 = 0, cloc1 = 1, cloc2 = 1, plot = False, resolution = 100, coarse_grid_resolution = 10):
	#tol/err
	p = helmholtz(boundary = boundary)
	kspace = np.arange(0.1,10,.1)
	ERR = []
	for k in kspace:
		print k
		cloc = cloc0+ cloc1*k+cloc2*k**2
		mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus, calQ = True)
		d = gq["d"]
		u = d.solve(mus)
		calculate_lambda_min(gq, lq)
		calculate_continuity_constant(gq, lq)
		calculate_inf_sup_constant2(gq, lq)
		for i in range(it):
			print i,
			sys.stdout.flush()
			err = []
			bases = create_bases(gq, lq, num_testvecs, transfer = 'robin', target_accuracy = acc, calC = False)
			ru = reconstruct_solution(gq, lq, bases)
			err.append(d.h1_norm(u-ru)[0]/d.h1_norm(u)[0])
		ERR.append(err)
	means = np.mean(ERR, axis = 1)
	data1 = np.vstack([kspace, means]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data1])
	"""limits = [0, 25, 50, 75, 100]
	percentiles = np.array(np.percentile(ERR, limits, axis=1))
	data2 = np.vstack([logspace, percentiles]).T
	open("dats/percentiles_accuracy.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in data2])"""
	if plot:
		from matplotlib import pyplot as plt
		plt.figure()
		plt.semilogy(kspace, means, label = "err")
		plt.legend(loc='upper right')
		plt.xlabel('k')
		plt.show()

def plotconstants(boundary, save, cloc0 = 0, cloc1 = 1, cloc2 = 1, resolution = 50, coarse_grid_resolution = 10, plot = False):
	p = helmholtz(boundary = boundary)
	kspace = np.arange(0.1,10,.1)
	#A = []
	B = []
	C = []
	#D = []
	for k in kspace:
		print k
		cloc = cloc0+ cloc1*k+cloc2*k**2
		mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus)
		#bases = create_bases2(gq, lq, 15, silent = False)
		#t = time.time()
		#a = calculate_inf_sup_constant(gq, lq, bases)
		#print time.time()-t, a
		t = time.time()
		b = calculate_inf_sup_constant2(gq, lq)
		print time.time()-t, b
		t = time.time()
		c = calculate_continuity_constant(gq, lq)
		print time.time()-t, c
		#t = time.time()
		#d = calculate_continuity_constant2(gq, lq, bases)
		#print time.time()-t, d
		#A.append(a)
		B.append(b)
		C.append(c)
		#D.append(d)
	data = np.vstack([kspace, B, C]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
	if plot:	
		plt.figure()
		plt.plot(K, B, label = "inf-sup-constant2")
		plt.xlabel('k')
		plt.legend(loc='upper right')
		plt.show()

		plt.figure()
		plt.plot(K, C, label = "continuity-constant")
		plt.xlabel('k')
		plt.legend(loc='upper right')
		plt.show()

def test(transfer = 'robin',boundary = 'dirichlet', n=15,k=6.,cglob= 6, cloc=6., title = 'test', resolution = 200, coarse_grid_resolution = 10):
	mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
	p = helmholtz(boundary = boundary)
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus)
	basis = create_bases2(gq,lq,n,transfer = transfer, silent = False)
	ru = reconstruct_solution(gq,lq,basis, silent = False)
	d = gq["d"]
	u = d.solve(mus)
	dif = u-ru
	print d.h1_norm(dif)[0]/d.h1_norm(u)[0]
	d.visualize((dif.real, dif.imag, u.real, u.imag, ru.real, ru.imag), legend = ('dif.real', 'dif.imag', 'u.real', 'u.imag', 'ru.real', 'ru.imag'), separate_colorbars = True, title = title)

def kerr(it, n, boundary, save, cglob = None, cloc0 = 0, cloc1 = 1, cloc2 = 1, rang = np.arange(0.1,50.1,.2), plot = False, resolution = 100, coarse_grid_resolution = 10):
	#k/err
	err_d =[]
	err_r = []
	p = helmholtz(boundary = boundary)
	usecglob = (cglob is None)
	for k in rang:
		if usecglob:
			cglob = -1j*k 			######wichtig!!
		cloc = cloc0+ cloc1*k+cloc2*k**2
		print "k: ", k, "cloc: ", cloc
		mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
		d = gq["d"]
		u = d.solve(mus)
		e_d = []
		e_r = []
		for i in range(it):
			print i,
			sys.stdout.flush()
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

def cerr2D(it, n, k, boundary, save, cglob = 0, rang = np.arange(-10.,10.,1.), yrang = None, plot = False, resolution = 200, coarse_grid_resolution = 10):
	#c/err
	if yrang is None:
		yrang = rang
	err_r = np.zeros((len(rang),len(yrang)))
	p = helmholtz(boundary = boundary)
	xi = 0
	for x in rang:
		yi = 0
		for y in yrang:
			c = x+1j*y
			print c
			mus = {'k': k, 'c_glob': cglob, 'c_loc': c}
			gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
			d = gq["d"]
			u = d.solve(mus)
			e_r = []
			for i in range(it):
				print i,
				sys.stdout.flush()
				bases = create_bases2(gq,lq,n,transfer = 'robin')
				ru_r = reconstruct_solution(gq, lq, bases)
				del bases
				dif_r = u-ru_r
				e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
			err_r[xi][yi]=np.mean(e_r)
			yi+=1
		xi+=1
	X,Y = np.meshgrid(rang, yrang)
	data = np.vstack([X.T.ravel(),Y.T.ravel(),err_r.ravel()]).T
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

def knerr2D(it, boundary, save, cglob = None, cloc0 = 0, cloc1 = 1, cloc2 = 1, krang = np.arange(0.1,200.1,10.), nrang  = np.arange(0,100,5), plot = False, resolution = 100, coarse_grid_resolution = 10):
	#c/err
	err_r = np.zeros((len(krang),len(nrang)))
	p = helmholtz(boundary = boundary)
	usecglob = (cglob is None)
	xi = 0
	for k in krang:
		yi = 0
		if usecglob:
			cglob = -1j*k
		cloc = cloc0+ cloc1*k+cloc2*k**2
		mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
		gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
		d = gq["d"]
		u = d.solve(mus)
		for n in nrang:
			print k, n
			e_r = []
			for i in range(min(20-n/5,it)):
				print i,
				sys.stdout.flush()
				bases = create_bases2(gq,lq,n,transfer = 'robin')
				ru_r = reconstruct_solution(gq, lq, bases)
				del bases
				dif_r = u-ru_r
				e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
			err_r[xi][yi]=np.mean(e_r)
			yi+=1
		xi+=1
	X,Y = np.meshgrid(krang, nrang)
	data = np.vstack([X.T.ravel(),Y.T.ravel(),err_r.ravel()]).T
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

def findcloc(it=50, k=20, n=15, s = 4., smin = 0.5, cloc = 0., boundary = 'robin'):
	p = helmholtz(boundary = boundary)
	dic = {}
	while(True):
		if cloc in dic:
			err0 = dic[cloc]
		else: 
			mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc}
			gq, lq = localize_problem(p, coarse_grid_resolution=10, fine_grid_resolution=50, mus = mus)
			d = gq["d"]
			u = d.solve(mus)
			e_r = []
			print "cloc: ", cloc, "k: ", k
			for i in range(it):
				print i,
				sys.stdout.flush()
				bases = create_bases2(gq,lq,n,transfer = 'robin')
				ru_r = reconstruct_solution(gq, lq, bases)
				del bases
				dif_r = u-ru_r
				e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
			err0 = np.mean(e_r)
			dic[cloc]=err0
		clocs = [cloc-s, cloc +s, cloc -s*1j, cloc +s*1j]
		errs = []
		for cloc1 in clocs:
			if cloc1 in dic:
				errs.append(dic[cloc1])
			else:
				mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc1}
				gq, lq = localize_problem(p, coarse_grid_resolution=10, fine_grid_resolution=50, mus = mus)
				d = gq["d"]
				u = d.solve(mus)
				e_r = []
				print "cloc: ", cloc, "cloc1: ", cloc1, "s: ", s, "k: ", k
				for i in range(it):
					print i, #"cloc: ", cloc, "cloc1: ", cloc1, "s: ", s, "i: " ,i
					sys.stdout.flush()
					bases = create_bases2(gq,lq,n,transfer = 'robin')
					ru_r = reconstruct_solution(gq, lq, bases)
					del bases
					dif_r = u-ru_r
					e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
				err = np.mean(e_r)
				errs.append(err)
				dic[cloc1]=err
		print errs
		print err0
		if np.min(errs)< err0:
			cloc = clocs[np.argmin(errs)]
			print "new cloc: ", cloc
		else: 
			s = s/2.
			print "new s: ", s
			if s < smin:
				print "found cloc!: ", cloc
				return cloc

def ckerr2D(it, n, boundary, save, cglob = None, krang = np.arange(0.,20.,2.), crang  = np.arange(0.,10.,1.), resolution = 50, coarse_grid_resolution = 10):
	#c/err
	err_r = np.zeros((len(krang),len(crang)))
	p = helmholtz(boundary = boundary)
	xi = 0
	for k in krang:
		yi = 0
		if cglob is None:
			cglob = -1j*k
		for cloc in crang:
			print k, cloc
			mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
			gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus = mus)
			d = gq["d"]
			u = d.solve(mus)
			e_r = []
			for i in range(it):
				print i,
				sys.stdout.flush()
				bases = create_bases2(gq,lq,n,transfer = 'robin')
				ru_r = reconstruct_solution(gq, lq, bases)
				del bases
				dif_r = u-ru_r
				e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
			err_r[xi][yi]=np.mean(e_r)
			yi+=1
		xi+=1
	X,Y = np.meshgrid(krang, crang)
	data = np.vstack([X.T.ravel(),Y.T.ravel(),err_r.ravel()]).T
	open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])