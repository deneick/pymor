from evaluations import *

ungleichungk(it= 1, acc=1e-2, boundary = 'robin', save = "dats/ungleichungk_bsp1.dat", cloc0 = 0, cloc1 = 0.02*(5-1j), cloc2 = 0.0016*(8-1j), krang = np.arange(0.1,10.1,1.0))
