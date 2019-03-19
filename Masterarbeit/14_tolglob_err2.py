from evaluations import *

ungleichung(it = 10, k=6, boundary = 'robin', save = "dats/ungleichung_bsp1.dat", cglob = -1j*6, cloc = 1-0.5j)
ungleichung(it = 10, k=6, boundary = 'dirichlet', save = "dats/ungleichung_bsp2.dat", cglob = -1j*6, cloc = 1.5)
ungleichung(it = 10, k=6, boundary = 'neumann', save = "dats/ungleichung_bsp3.dat", cglob = -1j*6, cloc = 1.5)
