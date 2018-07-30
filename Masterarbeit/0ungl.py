from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

ungleichung(it = 10, k=6, boundary = 'robin', save = "dats/ungleichung_bsp1.dat", cglob = -1j*6, cloc = 1-0.5j)
