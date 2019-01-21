from evaluations import *
p = maxwell_problem()
d, data = discretize_maxwell(p, diameter = 1./100)
mus = {'c_loc' : 1. / 376.730313 *1./200, 'k': 1e9*2*np.pi, 'mu': 4*np.pi*1e-7, 'eps': 8.854187817e-12}
u = d.solve(mus)
rang = np.logspace(-10,10,21)
it = 20

errs = []
for cloc in rang:
	print cloc
	mus = {'c_loc' : cloc, 'k': 1e9*2*np.pi, 'mu': 4*np.pi*1e-7, 'eps': 8.854187817e-12}
	gq, lq = localize_problem(p, 10, 100, mus, dof_codim = 1, discretizer = discretize_maxwell)
	E = []
	for i in range(it):
		basis = create_bases2(gq,lq,15,transfer = 'robin', silent = False)
		ru = reconstruct_solution(gq,lq,basis, silent = False)
		dif = u-ru
		err = d.h1_norm(dif)[0]/d.h1_norm(u)[0]
		E.append(err)
	errs.append(E)

means = np.mean(errs, axis = 1)
data = np.vstack([rang, means]).T
save = "dats/cerr_maxwell.dat"
open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])

E = []
for i in range(it):
	basis = create_bases2(gq,lq,15,transfer = 'dirichlet', silent = False)
	ru = reconstruct_solution(gq,lq,basis, silent = False)
	dif = u-ru
	err = d.h1_norm(dif)[0]/d.h1_norm(u)[0]
	E.append(err)
data = np.mean(E)
save = "dats/cerr_maxwell_d.dat"
open(save, "w").writelines([" ".join(str(data))])

