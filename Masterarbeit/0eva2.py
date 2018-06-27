from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

evaluation(it = 10, lim = 80, k=6, boundary = 'dirichlet', c=2, save = "dats/eva_bsp2.dat")
