from evaluations import *
p = maxwell_problem()
d, data = discretize_maxwell(p, diameter = 1./100)
mus = {'c_loc' : 0, 'k': 1e9*2*np.pi, 'mu': 4*np.pi*1e-7, 'eps': 8.854187817e-12}
u = d.solve(mus)
it = 20
n=15
save = "dats/cerr2d_maxwell.dat"

rang = np.logspace(-5,3,25)
yrang = np.logspace(-5,3,25)
err_r = np.zeros((len(rang),len(yrang)))
xi = 0
for x in rang:
	yi = 0
	for y in yrang:
		c = x+1j*y
		print c
		mus = {'c_loc' : c, 'k': 1e9*2*np.pi, 'mu': 4*np.pi*1e-7, 'eps': 8.854187817e-12}
		gq, lq = localize_problem(p, 10, 100, mus, dof_codim = 1, discretizer = discretize_maxwell)
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
