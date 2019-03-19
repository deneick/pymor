from evaluations import *

err = []
rang = range(20,200,10)
for i in rang:
	p = helmholtz(boundary = 'robin', g= -5j, f = False)
	d,data = discretize_elliptic_cg(p, diameter = 1./i)
	k = 10
	cglob = -1j*k
	cloc = -1j*k
	mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
	u = d.solve(mus)
	g = data["grid"]
	bi = data["boundary_info"]
	robin_edges = bi.robin_boundaries(1)
	robin_elements = g.superentities(1, 0)[robin_edges, 0]
	robin_indices = g.superentity_indices(1, 0)[robin_edges, 0]
	normals = g.unit_outer_normals()[robin_elements, robin_indices]
	SF_GRAD = np.array(([-1., -1.], [1., 0.], [0., 1.]))
	SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)
	grads = SF_GRADS[robin_elements]
	coeff = u.data[0].T
	robin_elements_viavertex = g.subentities(0, 2)[robin_elements]
	lhs = - np.einsum('ij, ijl, il -> i', coeff[robin_elements_viavertex], grads, normals)
	robin_c = p.robin_data[0](g.centers(1), mu=mus)[robin_edges]
	robin_g = p.robin_data[1](g.centers(1), mu=mus)[robin_edges]
	robin_edges_viavertex = g.subentities(1,2)[robin_edges]
	coeff_center = np.mean(coeff[robin_edges_viavertex],axis = 1)
	rhs = robin_c*(coeff_center-robin_g)
	dif = lhs-rhs
	err.append(np.linalg.norm(dif)/np.linalg.norm(lhs))

from matplotlib import pyplot as plt
plt.figure()
plt.semilogy(rang, err)
plt.xlabel('1/h')
plt.legend(loc='upper right')
plt.show()
