from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})

#test()
#cerr(it = 1, n=15, comp = True, k = 6, c = 6, boundary = 'mixed', save = "dats/cerr_mixed2.dat")
#ungleichung(it = 1, lim = 10, k=6, boundary = 'robin', save = "dats/test.dat", plot = True)

#kerr(it = 1, n = 15, boundary = 'robin', save = "dats/k_err_bsp1_test.dat", plot = True)
evaluation(it = 1, lim = 20, k=6, boundary = 'dirichlet', save = "~/dats/test.dat", plot = True)

#evaluation_neumann(it = 100, lim = 30, k=6, boundary = 'dirichlet', save = "dats/eva_bsp2_6_neumann.dat")
#kerr(it = 10, n = 15, boundary = 'robin', save = "dats/k_err_bsp1.dat")
#kerr(it = 10, n = 15, boundary = 'dirichlet', save = "dats/k_err_bsp2.dat")
##kerr_neumann(it = 10, n = 15, boundary = 'dirichlet', save = "dats/k_err_bsp2_n.dat")
##cerr(it = 10, n = 15, comp = True, k=6, c=6, boundary = 'dirichlet', save = "dats/c_err_r_bsp2_comp.dat")
##cerr(it = 10, n = 15, comp = False, k=6, c=6, boundary = 'dirichlet', save = "dats/c_err_r_bsp2_notcomp.dat")
#cerr(it = 1, n = 15, comp = True, k=6, c=6, boundary = 'robin', save = "dats/c_err_r_bsp1_comp.dat", plot = True)
##cerr(it = 10, n = 15, comp = False, k=6, c=6, boundary = 'robin', save = "dats/c_err_r_bsp1_notcomp.dat")

##accuracy(it = 10, num_testvecs = 20, k=6, boundary = 'robin', save = "dats/accuracy_bsp1.dat")
##accuracy(it = 10, num_testvecs = 20, k=6, boundary = 'dirichlet', save = "dats/accuracy_bsp2.dat")

#ungleichung(it = 10, lim = 50, k=6, boundary = 'robin', save = "dats/ungleichung_bsp1.dat")
#ungleichung(it = 10, lim = 50, k=6, boundary = 'dirichlet', save = "dats/ungleichung_bsp2.dat")

##evaluation(it = 100, lim = 30, k=6, boundary = 'dirichlet', save = "dats/eva_bsp2_6.dat")
##evaluation(it = 100, lim = 30, k=6, boundary = 'robin', save = "dats/eva_bsp1_6.dat")
