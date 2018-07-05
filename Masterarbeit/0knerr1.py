from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

knerr2D(it=5, lim =50,boundary='robin', save="dats/knerr_test.dat", krang = np.arange(10.,210.,10.))
