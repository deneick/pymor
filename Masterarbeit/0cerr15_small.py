from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(50,15,6, 'robin', "dats/cerr2D_bsp7_small.dat", rang = np.arange(-5.,5.,.5), cglob = 6+6j, resolution = 50)
