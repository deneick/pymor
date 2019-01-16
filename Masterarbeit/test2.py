from evaluations import *
p = maxwell_problem()
mus = {'c_loc' : 1. / 376.730313 *1./200, 'k': 1e9*2*np.pi, 'mu': 4*np.pi*1e-7, 'eps': 8.854187817e-12}
resolution = 200
coarse_grid_resolution = 10
gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus, dof_codim = 1, discretizer = discretize_maxwell)
