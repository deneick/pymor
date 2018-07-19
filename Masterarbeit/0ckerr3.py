from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

ckerr2D(it=50, n=15, boundary='neumann', save="dats/ckerr_bsp3.dat", cglob = None, krang = np.arange(1.,21.,1.), crang  = np.arange(0.,10.,.5), resolution = 50, coarse_grid_resolution = 10)
