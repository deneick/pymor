from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(30,15,6, 'neumann', "dats/cerr2D_bsp3_small.dat", rang = np.arange(-3.,3.,.2), resolution = 50)
