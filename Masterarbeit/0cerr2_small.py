from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(100,15,6, 'dirichlet', "dats/cerr2D_bsp2_small.dat", rang = np.arange(-5.,5.,.5), resolution = 50)
