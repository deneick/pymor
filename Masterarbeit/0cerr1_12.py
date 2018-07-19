from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(10,50,12, 'robin', "dats/cerr2D_bsp1_12_2.dat", rang = np.arange(-5.,5.,1.), cglob = -1j*12, resolution = 100)
