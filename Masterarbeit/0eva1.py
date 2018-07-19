from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

evaluation(it = 2, lim = 80, k=6, boundary = 'robin', save = "dats/eva_bsp1.dat", cglob = -1j*6, cloc = 1.-0.5j)
