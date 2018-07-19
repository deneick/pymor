from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(10,30,50, 'robin', "dats/cerr2D_bsp1_50.dat", rang = np.arange(-50.,50.,5.), cglob = -1j*50, resolution = 100)
