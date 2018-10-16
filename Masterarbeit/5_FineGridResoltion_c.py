from evaluations import *

r0 = resolution(50, 0.1, 30, 'robin', save = "dats/res0.30.dat", cloc = 0)
r25 = resolution(50, 25, 30, 'robin', save = "dats/res25.30.dat", cloc = 0)
r50 = resolution(50, 50, 30, 'robin', save = "dats/res50.30.dat", cloc = 0)
r100 = resolution(50, 100, 30, 'robin', save = "dats/res100.30.dat", cloc = 0)
r200 = resolution(50, 200, 30, 'robin', save = "dats/res200.30.dat", cloc = 0)

from matplotlib import pyplot as plt
plt.figure()
plt.plot(r0[0], r0[1], label = "k=0.1")
plt.plot(r25[0], r25[1], label = "k=25")
plt.plot(r50[0], r50[1], label = "k=50")
plt.plot(r100[0], r100[1], label = "k=100")
plt.plot(r200[0], r200[1], label = "k=200")
plt.legend(loc='upper right')
plt.xlabel('1/h')
plt.title('n=30')
plt.show()
