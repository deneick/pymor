from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

kerr(it = 10, n = 15, boundary = 'neumann', save = "dats/k_err_bsp3.dat", cloc = 1.5)
