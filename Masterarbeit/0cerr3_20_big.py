from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(30,15,20, 'neumann', "dats/cerr2D_bsp3_20_big.dat", rang = np.arange(-20.,20.,1.), cglob = -1j*6, resolution = 50)
