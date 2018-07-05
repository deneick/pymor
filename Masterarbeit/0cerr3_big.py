from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(10,15,6, 'neumann', "dats/cerr2D_bsp3_big.dat", rang = np.arange(-50.,50.,2.), resolution = 50)
