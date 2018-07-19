from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

evaluation(it = 2, lim = 80, k=6, boundary = 'dirichlet', save = "dats/eva_bsp2.dat", cloc=1.5)
