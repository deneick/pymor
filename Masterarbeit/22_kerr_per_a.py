from evaluations import *

it = 100
n = 15
boundary = 'robin'
cloc0 = 0
cloc1 = 0.02*(5-1j)
cloc2 = 0.0016*(8-1j)
rang = np.arange(0.2,10.2,.2)
resolution = 100
coarse_grid_resolution = 10

err_d =[]
err_r = []
p = helmholtz(boundary = boundary)

for k in rang:
	cglob = -1j*k 			
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
limits = [0, 25, 50, 75, 100]

percentiles_dirichlet_h1 = np.array(np.percentile(err_d, limits, axis=1))
percentiles_robin_h1 = np.array(np.percentile(err_r, limits, axis=1))

data = np.vstack([rang, percentiles_dirichlet_h1, percentiles_robin_h1]).T
open("dats/kerrper1.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in data])

from matplotlib import pyplot as plt
plt.figure()
plt.semilogy(rang, means_r, label = "robin")
plt.semilogy(rang, means_d, label = "dirichlet")
plt.xlabel('k')
plt.legend(loc='upper right')
plt.show()
