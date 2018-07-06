from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

knerr2D(it=1,boundary='robin', save="dats/knerr_test3.dat", krang = np.arange(10.,210.,10.), nrang = np.arange(0,52,2), resolution = 50)
