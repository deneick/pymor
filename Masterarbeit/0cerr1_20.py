from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(100,15,20, 'robin', "dats/cerr2D_bsp1_20.dat", rang = np.arange(3.,13.,.5), yrang = np.arange(-5.,5.,.5), cglob = -1j*20, resolution = 50)
