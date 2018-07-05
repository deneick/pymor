from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(30,15,6, 'robin', "dats/cerr2D_bsp1_small.dat", rang = np.arange(-3.,3.,.2), cglob = -1j*6, resolution = 50)
