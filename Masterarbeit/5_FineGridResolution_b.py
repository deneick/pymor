from evaluations import *

r40 = resolution(50, 200, 40, 'robin', save = "dats/res200.40.dat", cloc = 0)
r50 = resolution(50, 200, 50, 'robin', save = "dats/res200.50.dat", cloc = 0)
r60 = resolution(50, 200, 60, 'robin', save = "dats/res200.60.dat", cloc = 0)
r70 = resolution(50, 200, 70, 'robin', save = "dats/res200.70.dat", cloc = 0)
r80 = resolution(50, 200, 80, 'robin', save = "dats/res200.80.dat", cloc = 0)

from matplotlib import pyplot as plt
plt.figure()
plt.plot(r40[0], r40[1], label = "n=40")
plt.plot(r50[0], r50[1], label = "n=50")
plt.plot(r60[0], r60[1], label = "n=60")
plt.plot(r70[0], r70[1], label = "n=70")
plt.plot(r80[0], r80[1], label = "n=80")
plt.legend(loc='upper right')
plt.xlabel('1/h')
plt.title('k=200')
plt.show()
