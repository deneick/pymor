from evaluations import *

c1 = ungleichung(it = 10, k=6, boundary = 'robin', save = "dats/ungleichung_bsp1.dat", cglob = -1j*6, cloc = 1-0.5j, returnvals = True)
c2 = ungleichung(it = 10, k=6, boundary = 'dirichlet', save = "dats/ungleichung_bsp2.dat", cglob = -1j*6, cloc = 1.5, returnvals = True)
c3 = ungleichung(it = 10, k=6, boundary = 'neumann', save = "dats/ungleichung_bsp3.dat", cglob = -1j*6, cloc = 1.5, returnvals = True)

from matplotlib import pyplot as plt
plt.figure()
plt.loglog(c1[0], c1[1], label = "Beispiel 1")
plt.loglog(c2[0], c2[1], label = "Beispiel 2")
plt.loglog(c3[0], c3[1], label = "Beispiel 3")
plt.loglog(c1[0], c1[2], label = "a priori")
plt.loglog(c1[0], c1[0], label = "y=x")
plt.legend(loc='upper right')
plt.xlabel('tol_glob')
plt.gca().invert_xaxis()
plt.show()
