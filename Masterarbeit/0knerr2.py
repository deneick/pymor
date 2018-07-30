from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

knerr2D(it=10,boundary='dirichlet', save="dats/knerr2.dat", cloc0 = 1, cloc1 = -0.03, cloc2 = 0.014)
