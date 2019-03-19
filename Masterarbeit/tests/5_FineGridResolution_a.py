from evaluations import *

r10 = resolution(50, 6, 10, 'robin', save = "dats/res6.10.dat", cloc = 1.-0.5j)
r20 = resolution(50, 6, 20, 'robin', save = "dats/res6.20.dat", cloc = 1.-0.5j)
r30 = resolution(50, 6, 30, 'robin', save = "dats/res6.30.dat", cloc = 1.-0.5j)
r40 = resolution(50, 6, 40, 'robin', save = "dats/res6.40.dat", cloc = 1.-0.5j)
r50 = resolution(50, 6, 50, 'robin', save = "dats/res6.50.dat", cloc = 1.-0.5j)

from matplotlib import pyplot as plt
plt.figure()
plt.plot(r10[0], r10[1], label = "n=10")
plt.plot(r20[0], r20[1], label = "n=20")
plt.plot(r30[0], r30[1], label = "n=30")
plt.plot(r40[0], r40[1], label = "n=40")
plt.plot(r50[0], r50[1], label = "n=50")
plt.legend(loc='upper right')
plt.xlabel('1/h')
plt.title('k=6')
plt.show()
