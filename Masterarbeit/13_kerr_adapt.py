from evaluations import *

c1 = ungleichungk2(it= 10, acc=1e-2, boundary = 'robin', save = "dats/ungleichungk_bsp1.dat", cloc0 = 0, cloc1 = 0.02*(5-1j), cloc2 = 0.0016*(8-1j))
c2 = ungleichungk2(it= 10, acc=1e-2, boundary= 'dirichlet', save = "dats/ungleichungk_bsp2.dat", cloc0 = 1, cloc1 = -0.03, cloc2 = 0.014)
c3 = ungleichungk2(it= 10, acc=1e-2, boundary= 'neumann', save = "dats/ungleichungk_bsp3.dat", cloc0 =0, cloc1 = 0.2, cloc2 = 0.01)

"""
from matplotlib import pyplot as plt
plt.figure()
plt.semilogy(c1[0], c1[1], label = "Beispiel 1")
plt.semilogy(c2[0], c2[1], label = "Beispiel 2")
plt.semilogy(c3[0], c3[1], label = "Beispiel 3")
plt.semilogy(c1[0], c1[2], label = "a priori")
plt.semilogy(c3[0], np.ones(len(c3[0]))*1e-2, label = "tol_glob")
plt.legend(loc='upper right')
plt.xlabel('k')
plt.show()
"""
