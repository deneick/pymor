from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(50,15,20, 'robin', "dats/cerr2D_bsp5_20_small.dat", rang = np.arange(-5.,5.,.5), cglob = 6, resolution = 50)
