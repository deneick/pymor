from evaluations import *

c1 = plotconstants(boundary = 'robin', save = "dats/constants_bsp1.dat", cloc0 = 0, cloc1 = 0.02*(5-1j), cloc2 = 0.0016*(8-1j), returnvals = True)
c2 = plotconstants(boundary = 'dirichlet', save = "dats/constants_bsp2.dat", cloc0 = 1, cloc1 = -0.03, cloc2 = 0.014, returnvals = True)
c3 = plotconstants(boundary = 'neumann', save = "dats/constants_bsp3.dat", cloc0 =0, cloc1 = 0.2, cloc2 = 0.01, returnvals = True)

from matplotlib import pyplot as plt
plt.figure()
plt.plot(c1[0], c1[1], label = "Beispiel 1")
plt.plot(c2[0], c2[1], label = "Beispiel 2")
plt.plot(c3[0], c3[1], label = "Beispiel 3")
plt.legend(loc='upper right')
plt.xlabel('k')
plt.title('inf-sup Konstante')
plt.show()

plt.figure()
plt.plot(c1[0], c1[2], label = "Beispiel 1")
plt.plot(c2[0], c2[2], label = "Beispiel 2")
plt.plot(c3[0], c3[2], label = "Beispiel 3")
plt.legend(loc='upper right')
plt.xlabel('k')
plt.title('Stetigkeitskonstante')
plt.show()
