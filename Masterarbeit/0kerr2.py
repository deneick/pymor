from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

kerr(it = 10, n = 15, boundary = 'dirichlet', save = "dats/k_err_bsp2.dat", c=2)
