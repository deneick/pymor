from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

cerr2D(30,15,12, 'dirichlet', "dats/cerr2D_bsp2_12.dat", rang = np.arange(-15.,15.,1.), cglob = -1j*12, resolution = 50)
