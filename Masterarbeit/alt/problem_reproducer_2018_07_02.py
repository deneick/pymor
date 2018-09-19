

from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")


transfer = 'robin'
boundary = 'robin'
n=15
title = 'test'
resolution = 50
coarse_grid_resolution = 10
mus = (6, 6, -10)
mu_glob = {'c': mus[1], 'k': mus[0]}
p = helmholtz(boundary = boundary)
gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus)
basis = create_bases2(gq,lq,n,transfer = transfer)
ru = reconstruct_solution(gq,lq,basis)
d = gq["d"]
u = d.solve(mu_glob)
dif = u-ru
d.visualize((dif.real, dif.imag, u.real, u.imag, ru.real, ru.imag), legend = ('dif.real', 'dif.imag', 'u.real', 'u.imag', 'ru.real', 'ru.imag'), separate_colorbars = True, title = title)
print d.h1_norm(dif)[0]/d.h1_norm(u)[0]

