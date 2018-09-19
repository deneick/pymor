"""
test berechnet und visualisiert die approximierte Loesung bzgl fester Basisgroesse

evaluation/evaluation_neumann berechnet den relativen Fehler abhaengig von der Basisgroesse

ungleichung berechnet die Genauigkeit der apriori Abschaetzung

accuracy berechnet den relativen Fehler abhaengig von der gewuenschten Genauigkeit gemaess des adaptiven Algos

kerr/kerr_neumann berechnet den relativen Fehler abhaengig von k

cerr berechnet den relativen Fehler abhaengig von c(dem lokalen Robin-Parameter)


zum plotten, setze plot=True
it = Anzahl der Wiederholungen fuer die Statistik
lim = maximale Basisgroesse
"""


from evaluations import *

set_log_levels(levels={'pymor': 'WARN'})
if not os.path.exists("dats"):
	os.makedirs("dats")


# evaluation(it = 10, lim = 80, k=6, boundary = 'dirichlet', c=2, save = "dats/eva_bsp2_c2.dat")
#evaluation(it = 10, lim = 80, k=6, boundary = 'robin', save = "dats/eva_bsp1_6.dat")
#evaluation_neumann(it = 10, lim = 80, k=6, boundary = 'dirichlet', save = "dats/eva_bsp2_6_neumann.dat")
# evaluation(it = 10, lim = 80, k=6, boundary = 'neumann', save = "dats/eva_bsp3.dat", c= 2)


# kerr(it = 10, n = 15, boundary = 'robin', save = "dats/k_err_bsp1_200.dat")
#kerr(it = 10, n = 15, boundary = 'dirichlet', save = "dats/k_err_bsp2_200.dat")
# kerr2(it = 10, n = 15, boundary = 'dirichlet', save = "dats/k_err_bsp2_c2.dat")

# kerr(it = 10, n = 15, c=2,boundary = 'neumann', save = "dats/k_err_bsp3.dat")

#cerr(it = 20, n = 15, comp = True, k=6, c=6, boundary = 'dirichlet', save = "dats/c_err_r_bsp2_plus.dat")
 #cerr2D(20,15,6, 'dirichlet', "dats/cerr2D_bsp2_2.dat")
#cerr2D(10,15,6, 'dirichlet', "dats/cerr2D_bsp2_big.dat")
# cerr2D(10,15,6, 'robin', "dats/cerr2D_bsp1_big.dat")
# cerr2D(10,15,40, 'robin', "dats/cerr2D_bsp1_40.dat")
# cerr2D(10,15,40, 'robin', "dats/cerr2D_bsp1_40.dat", rang = np.arange(-50,50, 2.))
#cerr2D(10,15,40, 'robin', "dats/cerr2D_bsp4.dat", cglob = 6)
#cerr2D(10,15,40, 'robin', "dats/cerr2D_bsp5.dat", cglob = -6)
# cerr2D(10,15,6, 'neumann', "dats/cerr2D_bsp3.dat")
#cerr2Dp(10,15,6, 'robin', "dats/test.dat", rang = np.arange(-5,5,1.))

#cerr(it = 20, n = 15, comp = True, k=6, c=6, boundary = 'dirichlet', save = "dats/c_err_r_bsp2_comp.dat")
#cerr(it = 20, n = 15, comp = False, k=6, c=6, boundary = 'dirichlet', save = "dats/c_err_r_bsp2_notcomp.dat")



#test(title = 'robin')
#test(c=0., title = 'neumann')

#cerr(it = 10, n = 15, comp = True, k=6, c=6, boundary = 'dirichlet', save = "dats/c_err_r_bsp2_comp.dat", plot = True)


#cerr(it = 1, n=15, comp = True, k = 6, c = 6, boundary = 'mixed', save = "dats/cerr_mixed2.dat")
#ungleichung(it = 1, lim = 10, k=6, boundary = 'robin', save = "dats/test.dat", plot = True)

#kerr(it = 1, n = 15, boundary = 'robin', save = "dats/k_err_bsp1_test.dat", plot = True)


#evaluation_neumann(it = 100, lim = 30, k=6, boundary = 'dirichlet', save = "dats/eva_bsp2_6_neumann.dat")
#kerr(it = 10, n = 15, boundary = 'robin', save = "dats/k_err_bsp1.dat")
#kerr(it = 10, n = 15, boundary = 'dirichlet', save = "dats/k_err_bsp2.dat")
##kerr_neumann(it = 10, n = 15, boundary = 'dirichlet', save = "dats/k_err_bsp2_n.dat")
###cerr(it = 10, n = 15, comp = True, k=6, c=6, boundary = 'dirichlet', save = "dats/c_err_r_bsp2_comp.dat")
#cerr(it = 10, n = 15, comp = False, k=6, c=6, boundary = 'dirichlet', save = "dats/c_err_r_bsp2_notcomp.dat")
#cerr(it = 1, n = 15, comp = True, k=6, c=6, boundary = 'robin', save = "dats/c_err_r_bsp1_comp.dat", plot = True)
##cerr(it = 10, n = 15, comp = False, k=6, c=6, boundary = 'robin', save = "dats/c_err_r_bsp1_notcomp.dat")

##accuracy(it = 10, num_testvecs = 20, k=6, boundary = 'robin', save = "dats/accuracy_bsp1.dat")
##accuracy(it = 10, num_testvecs = 20, k=6, boundary = 'dirichlet', save = "dats/accuracy_bsp2.dat")

# ungleichung(it = 10, lim = 80, k=6, boundary = 'robin', save = "dats/ungleichung_bsp1.dat")
#ungleichung(it = 10, lim = 80, k=6, boundary = 'dirichlet', save = "dats/ungleichung_bsp2.dat")

##evaluation(it = 100, lim = 30, k=6, boundary = 'dirichlet', save = "dats/eva_bsp2_6.dat")
##evaluation(it = 100, lim = 30, k=6, boundary = 'robin', save = "dats/eva_bsp1_6.dat")
