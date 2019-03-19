from pymor.basic import *
from problems import *

resolution = 50
cloc = 0
p = helmholtz(boundary = 'neumann')
d, data = discretize_elliptic_cg(p, diameter=1./resolution)
k=6
cglob = -1j*k
mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
u = d.solve(mus)
d.visualize(u)

k=30
cglob = -1j*k
mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
u = d.solve(mus)
d.visualize(u)
