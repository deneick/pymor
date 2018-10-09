from problems import *
import numpy as np
from pymor.basic import *
from localize_problem import *
from constants import *
from generate_solution import *

set_log_levels(levels={'pymor': 'WARN'})
it = 10

n=15
k=0
cglob = 0
resolution = 50
coarse_grid_resolution = 10
rang = np.arange(-2,2,0.2)
err_r = []
for c in rang:
	print c
	p = poisson_problem(c=c)	
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution)
	d = gq["d"]
	u = d.solve()
	e_r = []
	for i in range(it):
		#print i,
		#sys.stdout.flush()
		bases = create_bases2(gq,lq,n,transfer = 'robin')
		ru_r = reconstruct_solution(gq, lq, bases)
		del bases
		dif_r = u-ru_r
		e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
	err_r.append(np.mean(e_r))
es = []
for i in range(it):
	bases = create_bases2(gq,lq,n,transfer = 'dirichlet')
	ru_r = reconstruct_solution(gq, lq, bases)
	dif_r = u-ru_r
	e = d.h1_norm(dif_r)[0]/d.h1_norm(u)[0]
	print e
	es.append(e)
em = np.mean(es)

from matplotlib import pyplot as plt
plt.figure()
plt.semilogy(rang, err_r, label = "err")
plt.legend(loc='upper right')
plt.xlabel('c')
plt.hlines(em,-2,2)
plt.show()


data = np.vstack([rang, err_r, np.full(len(rang),em)]).T
open("dats/cerrp1d.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in data])



rang = np.concatenate((-np.logspace(10,1,10),np.logspace(0,10,11)))
err_r = []
for c in rang:
	print c	
	p = h_problem(c=c)
	gq, lq = localize_problem(p, coarse_grid_resolution, resolution)
	d = gq["d"]
	u = d.solve()
	e_r = []
	for i in range(it):
		#print i,
		#sys.stdout.flush()
		bases = create_bases2(gq,lq,n,transfer = 'robin')
		ru_r = reconstruct_solution(gq, lq, bases)
		del bases
		dif_r = u-ru_r
		e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
	err_r.append(np.mean(e_r))
es = []
for i in range(it):
	bases = create_bases2(gq,lq,n,transfer = 'dirichlet')
	ru_r = reconstruct_solution(gq, lq, bases)
	dif_r = u-ru_r
	e = d.h1_norm(dif_r)[0]/d.h1_norm(u)[0]
	print e
	es.append(e)
em = np.mean(es)
"""
from matplotlib import pyplot as plt
plt.figure()
plt.loglog(rang, err_r, label = "err")
plt.legend(loc='upper right')
plt.xlabel('c')
plt.xscale('symlog')
plt.hlines(em,-1e10,1e10)
plt.show()
"""

rang2 = np.arange(-10,11)
data = np.vstack([rang2, err_r, np.full(len(rang2),em)]).T
open("dats/cerrh1d.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
