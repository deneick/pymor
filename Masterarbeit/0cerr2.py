from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(1,15,6, 'dirichlet', "dats/cerr2D_bsp2_test.dat",rang = np.arange(-50.,50.,10.), plot = True, resolution = 50)
