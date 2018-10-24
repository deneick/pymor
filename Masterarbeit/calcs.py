from evaluations import *
p = helmholtz(boundary = 'robin')
k=7.88
cglob = -1j*k
cloc = -1j
mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
gq, lq = localize_problem(p, 10, 100, mus)

d = gq["d"]
u = d.solve(mus)
localizer = gq["localizer"]

e = []
for space in gq["spaces"]:
	#space = gq["spaces"][20]
	#print space
	ldict = lq[space]
	omstar = ldict["omega_star_space"]

	u_l = localizer.localize_vector_array(u, omstar)

	u_g = localizer.globalize_vector_array(u_l, omstar)

	u_f = ldict["local_sol2"]
	u_fg = localizer.globalize_vector_array(u_f, omstar)

	dif = u_g-u_fg

	#print np.abs(d.h1_norm(u_fg))
	print np.abs(d.h1_norm(dif)/d.h1_norm(u_g))
	e.append(np.abs(d.h1_norm(dif)/d.h1_norm(u_g)))
print np.max(e), np.mean(e)



from evaluations import *
p = helmholtz(boundary = 'robin')
k=7.88
cglob = -1j*k
cloc = -1j*k
mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
gq, lq = localize_problem(p, 10, 100, mus)

d = gq["d"]
u = d.solve(mus)
localizer = gq["localizer"]

space = gq["spaces"][20]
ldict = lq[space]
omstar = ldict["omega_star_space"]

u_l = localizer.localize_vector_array(u, omstar)

u_g = localizer.globalize_vector_array(u_l, omstar)

u_f = ldict["local_sol2"]
u_fg = localizer.globalize_vector_array(u_f, omstar)

dif = u_g-u_fg

#print d.h1_norm(u_fg)
print d.h1_norm(dif)/d.h1_norm(u_g)


d.visualize((dif, u_g, u_fg), separate_colorbars = True, legend = ('dif', 'u_g', 'u_fg'))




from evaluations import *
p = helmholtz(boundary = 'dirichlet')
E = []
rang = np.arange(.1,10.,1.)
for k in rang:
	cglob = -1j*k
	cloc = 0
	mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
	print k
	gq, lq = localize_problem(p, 10, 100, mus)

	d = gq["d"]
	u = d.solve(mus)
	localizer = gq["localizer"]

	e = []
	for space in gq["spaces"]:
	#space = gq["spaces"][40]
		ldict = lq[space]
		omstar = ldict["omega_star_space"]

		u_l = localizer.localize_vector_array(u, omstar)

		u_g = localizer.globalize_vector_array(u_l, omstar)

		u_f = ldict["local_sol2"]
		u_fg = localizer.globalize_vector_array(u_f, omstar)

		dif = u_g-u_fg

		#print np.abs(d.h1_norm(u_fg))
		#print np.abs(d.h1_norm(dif)/d.h1_norm(u_g))
		e.append(np.abs(d.h1_norm(dif)/d.h1_norm(u_g)))
	print np.max(e), np.mean(e)
	E.append(np.max(e))

from matplotlib import pyplot as plt
plt.figure()
plt.semilogy(rang, E, label = "c_s")
plt.xlabel('k')
plt.legend(loc='upper right')
plt.show()

rang = np.arange(.1,2.1,.2)






from evaluations import *
p = helmholtz(boundary = 'dirichlet')
rang = np.arange(.1,10.,1.)
K = []
for k in rang:
	print k
	cglob = -1j*k
	cloc = 0
	mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
	gq, lq = localize_problem(p, 10, 100, mus, calQ = True, calT = True)

	d = gq["d"]
	u = d.solve(mus)
	localizer = gq["localizer"]

	#space = gq["spaces"][20]
	C = []
	for space in gq["spaces"]:
		ldict = lq[space]


		"""
		T = ldict["transfer_matrix_robin"]
		f = ldict["local_solution_robin"].data.T
		x = np.linalg.lstsq(T, f)[0]
		y = T.dot(x)
		y_g = localizer.globalize_vector_array(NumpyVectorArray(y.T), ldict["range_space"])
		u_f = localizer.globalize_vector_array(ldict["local_solution_robin"], ldict["range_space"])
		d.visualize((y_g, u_f, y_g-u_f), separate_colorbars = True, legend = ('y_g', 'u_f', 'dif'))
		"""

		T = ldict["solution_matrix_robin"]
		u_s = ldict["local_sol2"]
		x = np.linalg.lstsq(T, u_s.data.T)[0]
		y = T.dot(x)
		y_p = NumpyVectorArray(y.T)
		#y_g = localizer.globalize_vector_array(y_p, ldict["omega_star_space"])
		#u_f = localizer.globalize_vector_array(u_fl, ldict["omega_star_space"])
		#d.visualize((y_g, u_f, y_g-u_f), separate_colorbars = True, legend = ('y_g', 'u_f', 'dif'))
		product = ldict["omega_star_h1_product"]
		norm = induced_norm(product)
		y_p = y_p.lincomb(1/norm(y_p).real)
		c = np.sqrt(norm(u_s).real[0]**2/(norm(u_s).real[0]**2 - product.apply2(u_s, y_p).real[0][0]**2))
		if norm(u_s).real < 1e-14: c = 1
		#alpha = norm(u_s).real**2/product.apply2(y_p,u_s).real
		#xmax = y_p.lincomb(alpha)
		#c2 = norm(xmax).real/norm(xmax - u_s).real
		#print c
		C.append(c)
	print np.max(C)
	K.append(np.max(C))

from matplotlib import pyplot as plt
plt.figure()
plt.semilogy(rang, K, label = "c_s")
plt.xlabel('k')
plt.legend(loc='upper right')
plt.show()
