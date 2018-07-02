from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})

transfer = 'robin'
boundary = 'dirichlet'
n=15
k=6.
title = 'test'
resolution = 50
coarse_grid_resolution = 10
mus = {'k': k, 'c_glob': 6, 'c_loc': -1j*k}
p = helmholtz(boundary = boundary)
gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus)
basis = create_bases2(gq,lq,n,transfer = transfer)
ru = reconstruct_solution(gq,lq,basis)
d = gq["d"]
u = d.solve(mus)
dif = u-ru
print d.h1_norm(dif)[0]/d.h1_norm(u)[0]
d.visualize((dif.real, dif.imag, u.real, u.imag, ru.real, ru.imag), legend = ('dif.real', 'dif.imag', 'u.real', 'u.imag', 'ru.real', 'ru.imag'), separate_colorbars = True, title = title)
