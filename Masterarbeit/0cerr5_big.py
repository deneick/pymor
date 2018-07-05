from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(10,15,6, 'robin', "dats/cerr2D_bsp5_big.dat", rang = np.arange(-50.,50.,2.), cglob = -6, resolution = 50)
