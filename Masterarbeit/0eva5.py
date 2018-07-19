from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

evaluation(it = 10, lim = 80, k=6, boundary = 'robin', save = "dats/eva_bsp5.dat", cglob = -6, cloc = -2.5)
