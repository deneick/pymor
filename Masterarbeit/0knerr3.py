from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")

knerr2D(it=10,boundary='neumann', save="dats/knerr3.dat", cloc0 =0, cloc1 = 0.2, cloc2 = 0.01)
