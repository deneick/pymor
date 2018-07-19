from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

knerr2D(it=1,boundary='robin', save="dats/knerr1.dat", krang = np.arange(10.,210.,10.), nrang = np.arange(0,105,5), resolution = 100, cloc = 1.-0.5j)
