from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(30,15,20, 'robin', "dats/cerr2D_bsp1_20_big.dat", rang = np.arange(-20.,20.,1.), cglob = -1j*20, resolution = 50)
