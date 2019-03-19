from evaluations import *

ungleichungkfull(it= 10, acc=1e-2, boundary = 'robin', save = "dats/ungleichungk_bsp1_full.dat", cloc0 = 0, cloc1 = 0.02*(5-1j), cloc2 = 0.0016*(8-1j))
ungleichungkfull(it= 10, acc=1e-2, boundary= 'dirichlet', save = "dats/ungleichungk_bsp2_full.dat", cloc0 = 1, cloc1 = -0.03, cloc2 = 0.014)
ungleichungkfull(it= 10, acc=1e-2, boundary= 'neumann', save = "dats/ungleichungk_bsp3_full.dat", cloc0 =0, cloc1 = 0.2, cloc2 = 0.01)
