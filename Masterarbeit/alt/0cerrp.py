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
rang = np.arange(-2,2,0.1)
err_r = np.zeros((len(rang),len(rang)))
xi = 0
for x in rang:
	yi = 0
	for y in rang:
		c = x+1j*y
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
		err_r[xi][yi]=np.mean(e_r)
		yi+=1
	xi+=1
X,Y = np.meshgrid(rang, rang)
data = np.vstack([X.T.ravel(),Y.T.ravel(),err_r.ravel()]).T
open("dats/cerrp2d.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
