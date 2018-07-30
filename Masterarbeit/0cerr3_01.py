from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(100,15,0.1, 'neumann', "dats/cerr2D_bsp3_01_small.dat", rang = np.arange(-5.,5.,.5), cglob = -1j*20, resolution = 50)
