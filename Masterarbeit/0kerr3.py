from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

kerr(it = 10, n = 30, boundary = 'neumann', save = "dats/k_err_bsp3.dat", cloc0 =0, cloc1 = 0.2, cloc2 = 0.01)
