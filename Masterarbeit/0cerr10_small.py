from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(50,15,0.1, 'robin', "dats/cerr2D_bsp1_01_small.dat", rang = np.arange(-5.,5.,.5), cglob = -1j*0.1, resolution = 50)
