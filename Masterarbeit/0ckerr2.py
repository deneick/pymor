from evaluations import *

ckerr2D(it=50, n=15, boundary='dirichlet', save="dats/ckerr_bsp2.dat", cglob = None, krang = np.arange(1.,21.,1.), crang  = np.arange(0.,10.,.5), resolution = 50, coarse_grid_resolution = 10)
