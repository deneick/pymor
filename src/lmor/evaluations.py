"""
test1 berechnet und visualisiert eine reduzierte Loesung bzgl fester Basisgroesse

test2 berechnet und visualisiert eine reduzierte Loesung mit dem adaptiven Algorithmus (Alg. 3)

evaluation berechnet den relativen Fehler abhaengig von der Basisgroesse

ungleichung berechnet den relativen Fehler abhaengig von der gegebenen Toleranz bezueglich
des adaptiven Algorithmus und die a priori Grenze

ungleichungk berechnet den relativen Fehler abhaengig von der Wellenzahl k bezueglich des
adaptiven Algorithmus und die a priori Grenze

plotconstants berechnet die inf-sup Konstante und die Stetigkeitskonstante
in Abhaengigkeit von der Wellenzahl k

kerr berechnet den relativen Fehler abhaengig von der Wellenzahl k mit dem
nicht-adaptiven Algorithmus (Alg. 4)

resolution berechnet den relativen Fehler in Abhaengigkeit von der feinen
Gitteraufloesung mit Alg. 4

cerr2D berechnet den relativen Fehler abhaengig von c(dem lokalen Robin-Parameter) mit Alg. 4

knerr2D berechnet den relativen Fehler abhaengig von der Wellenzahl k und der
Basisgroesse n mit Alg. 4


it=Anzahl der Wiederholungen fuer die Statistik
lim=maximale Basisgroesse
k=Wellenzahl
boundary=globale Randbedingung
save=Speicherort fuer Daten
cglob=globaler Robin-Parameter
cloc=lokaler Robin-Parameter (fuer Transferoperator)
resolution=feine Gitteraufloesung
coarse_grid_resolution=grobe Gitteraufloesung
"""

from pymor.core.logger import set_log_levels
from lmor.problems import helmholtz
from lmor.localize_problem import localize_problem
from lmor.generate_solution import create_bases, create_bases2, create_bases3, reconstruct_solution, operator_svd2
from lmor.constants import (calculate_continuity_constant, calculate_inf_sup_constant2,
                            calculate_csis, calculate_Psi_norm)
import multiprocessing as mp
import time
import os
import numpy as np
import sys

if not os.path.exists("dats"):
    os.makedirs("dats")
set_log_levels(levels={'pymor': 'WARN'})

process_count = 10


def evaluation(it, lim, k, boundary, save, cglob=0, cloc=0, plot=False, resolution=200, coarse_grid_resolution=10):
    p = helmholtz(boundary=boundary)
    mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
    gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus=mus)
    d = gq["d"]
    u = d.solve(mus)
    h1_dirichlet = []
    h1_robin = []
    nrang = np.arange(0, lim, 5)
    for n in nrang:
        print("n: ", n)
        h1d = []
        h1r = []
        for j in range(it):
            print(j, )
            sys.stdout.flush()
            basis_dirichlet = create_bases2(gq, lq, n)
            ru_dirichlet = reconstruct_solution(gq, lq, basis_dirichlet)
            del basis_dirichlet
            basis_robin = create_bases2(gq, lq, n, transfer='robin')
            ru_robin = reconstruct_solution(gq, lq, basis_robin)
            del basis_robin
            dif_dirichlet = u - ru_dirichlet
            dif_robin = u-ru_robin
            h1d.append(gq["full_norm"](dif_dirichlet)[0]/gq["full_norm"](u)[0])
            h1r.append(gq["full_norm"](dif_robin)[0]/gq["full_norm"](u)[0])
        h1_dirichlet.append(h1d)
        h1_robin.append(h1r)
    means_dirichlet_h1 = np.mean(h1_dirichlet, axis=1)
    means_robin_h1 = np.mean(h1_robin, axis=1)
    data = np.vstack([nrang, means_dirichlet_h1, means_robin_h1]).T
    open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
    if plot:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.semilogy(nrang, means_dirichlet_h1, label="Dirichlet h1")
        plt.semilogy(nrang, means_robin_h1, label="Robin h1")
        plt.legend(loc='upper right')
        plt.xlabel('Basis size')
        plt.show()


def evaluationq(it, lim, k, boundary, save, cloc=0, resolution=100, coarse_grid_resolution=10):
    p = helmholtz(boundary=boundary)
    cglob = -1j*k
    mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
    gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus=mus, calT=True)
    d = gq["d"]
    u = d.solve(mus)
    nrang = np.arange(0, lim, 5)
    global cube

    def cube(n):
        print("n: ", n)
        Q0 = []
        Q1 = []
        Q2 = []
        for j in range(it):
            print(j, )
            sys.stdout.flush()
            basis_robin = create_bases3(gq, lq, n, 0)
            ru_robin = reconstruct_solution(gq, lq, basis_robin)
            del basis_robin
            dif_robin = u-ru_robin
            Q0.append(gq["full_norm"](dif_robin)[0]/gq["full_norm"](u)[0])

            basis_robin = create_bases3(gq, lq, n, 1)
            ru_robin = reconstruct_solution(gq, lq, basis_robin)
            del basis_robin
            dif_robin = u-ru_robin
            Q1.append(gq["full_norm"](dif_robin)[0]/gq["full_norm"](u)[0])

            basis_robin = create_bases3(gq, lq, n, 2)
            ru_robin = reconstruct_solution(gq, lq, basis_robin)
            del basis_robin
            dif_robin = u-ru_robin
            Q2.append(gq["full_norm"](dif_robin)[0]/gq["full_norm"](u)[0])
        return [np.mean(Q0), np.mean(Q1), np.mean(Q2)]
    pool = mp.Pool()
    results = pool.map(cube,  nrang)
    data = np.vstack([nrang, np.array(results).T]).T
    open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])


def ungleichung(it, acc, boundary, save, krang=np.arange(0.1, 10.1, 0.1),
                cloc0=0, cloc1=1, cloc2=1, returnvals=False, resolution=100, coarse_grid_resolution=10):
    p = helmholtz(boundary=boundary)
    global cube

    def cube(k):
        print(k)
        cglob = -1j*k
        cloc = cloc0 + cloc1 * k + cloc2 * k**2
        mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
        resolution  = int(np.ceil(float(k*1.5+50)/coarse_grid_resolution)*coarse_grid_resolution)
        gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus=mus, calT=True, calQ=True)
        calculate_continuity_constant(gq, lq)
        calculate_inf_sup_constant2(gq, lq)
        # calculate_lambda_min(gq, lq)
        calculate_csis(gq, lq)
        calculate_Psi_norm(gq, lq)
        d = gq["d"]
        u = d.solve(mus)
        norm = gq["full_norm"]
        localizer = gq["localizer"]
        pou = gq["pou"]
        dats = []
        for j in range(it):
            print(j, )
            sys.stdout.flush()
            bases = create_bases(gq, lq, num_testvecs=20, transfer='robin', target_accuracy=acc, calC=False)
            ru = reconstruct_solution(gq, lq, bases)
            ls = norm(u-ru)[0]/norm(u)[0]
            sum1 = u*0
            sum2 = 0
            sum3 = 0
            sum4 = 0
            sum51 = []
            sum52 = 0
            for space in gq["spaces"]:
                ldict = lq[space]
                B = bases[space]
                range_product = ldict["range_product"]
                source_product = ldict["source_product"]
                T = ldict["robin_transfer"].as_range_array()
                range_space = ldict["range_space"]
                omega_star_space = ldict["omega_star_space"]
                omega_star_product = ldict["omega_star_product"]
                u_s = ldict["local_solution_robin"]
                u_s2 = ldict["local_sol2"]

                u_loc = pou[range_space](localizer.localize_vector_array(u, range_space))
                u_loc2 = localizer.localize_vector_array(u, omega_star_space)
                u_dif = u_loc-u_s
                u_dif2 = u_loc2-u_s2
                term = u_dif-B.lincomb(B.inner(u_dif, range_product).T)

                T1 = T - B.lincomb(B.inner(T, range_product).T)
                maxval = operator_svd2(T1.data.T, source_product.matrix, range_product.matrix)[0][0]

                sum1 += localizer.globalize_vector_array(term, range_space)
                sum2 += term.norm(range_product)[0]**2
                sum3 += maxval**2*ldict["Psi_norm"]**2 * u_dif2.norm(omega_star_product)[0]**2
                sum4 += maxval**2*ldict["Psi_norm"]**2 * u_loc2.norm(omega_star_product)[0]**2 * ldict["csi"]**2
                sum51.append(maxval*ldict["Psi_norm"] * ldict["csi"])
                sum52 += u_loc2.norm(omega_star_product)[0]**2
            rs1 = gq["continuity_constant"]/gq["inf_sup_constant"] * norm(sum1)[0]/norm(u)[0]
            rs2 = gq["continuity_constant"]/gq["inf_sup_constant"] * 2 * np.sqrt(sum2)/norm(u)[0]
            rs3 = gq["continuity_constant"]/gq["inf_sup_constant"] * 2 * np.sqrt(sum3)/norm(u)[0]
            rs4 = gq["continuity_constant"]/gq["inf_sup_constant"] * 2 * np.sqrt(sum4)/norm(u)[0]
            rs5 = gq["continuity_constant"]/gq["inf_sup_constant"] * 2 * max(sum51) * np.sqrt(sum52)/norm(u)[0]
            rs6 = gq["continuity_constant"]/gq["inf_sup_constant"] * max(sum51) * 8
            ccs = gq["continuity_constant"]/gq["inf_sup_constant"]
            dats.append([ls, rs1, rs2, rs3, rs4, rs5, rs6, ccs])
        return dats
    pool = mp.Pool()
    results = pool.map(cube,  krang)
    means = np.mean(results, axis=1)
    data = np.vstack([krang, means.T]).T
    open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
    if returnvals:
        return means


def plotconstants(boundary, save, cloc0=0, cloc1=1, cloc2=1, resolution=50,
                  coarse_grid_resolution=10, returnvals=False):
    p = helmholtz(boundary=boundary)
    kspace = np.arange(0.1, 10.1, 1.)
    B = []
    C = []
    for k in kspace:
        print(k)
        cloc = cloc0 + cloc1 * k + cloc2 * k**2
        mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc}
        gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus)
        t = time.time()
        b = calculate_inf_sup_constant2(gq, lq)
        print(time.time()-t, b)
        t = time.time()
        c = calculate_continuity_constant(gq, lq)
        print(time.time()-t, c)
        B.append(b)
        C.append(c)
    data = np.vstack([kspace, B, C]).T
    open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
    if returnvals:
        return [kspace, B, C]


def test1(transfer='robin', boundary='dirichlet', n=15, k=6., cloc=6., title='test',
          resolution=100, coarse_grid_resolution=10, m=1):
    cglob = -1j*k
    mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
    p = helmholtz(boundary=boundary)
    gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus, m=m)
    basis = create_bases2(gq, lq, n, transfer=transfer, silent=False)
    ru = reconstruct_solution(gq, lq, basis, silent=False)
    d = gq["d"]
    u = d.solve(mus)
    dif = u-ru
    print(gq["full_norm"](dif)[0]/gq["full_norm"](u)[0])
    d.visualize((dif.real, dif.imag, u.real, u.imag, ru.real, ru.imag),
                legend=('dif.real', 'dif.imag', 'u.real', 'u.imag', 'ru.real', 'ru.imag'),
                separate_colorbars=True, title=title)


def test2(transfer='robin', boundary='dirichlet', acc=1e-2, k=6., cloc=6.,
          title='test', resolution=100, coarse_grid_resolution=10):
    cglob = -1j*k
    mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
    p = helmholtz(boundary=boundary)
    gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus, calQ=True)
    basis = create_bases(gq, lq, 20, transfer=transfer, target_accuracy=acc, silent=False)
    ru = reconstruct_solution(gq, lq, basis, silent=False)
    d = gq["d"]
    u = d.solve(mus)
    dif = u-ru
    print(gq["full_norm"](dif)[0]/gq["full_norm"](u)[0])
    d.visualize((dif.real, dif.imag, u.real, u.imag, ru.real, ru.imag),
                legend=('dif.real', 'dif.imag', 'u.real', 'u.imag', 'ru.real', 'ru.imag'),
                separate_colorbars=True, title=title)


def kerr(it, boundary, save, cloc0=0, cloc1=1, cloc2=1, rang=np.arange(0.5, 100.5, 0.5),
         plot=False, coarse_grid_resolution=10):
    p = helmholtz(boundary=boundary)
    global cube

    def cube(k):
        cglob = -1j * k
        cloc = cloc0 + cloc1 * k + cloc2 * k**2
        print("k: ", k, "cloc: ", cloc)
        mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
        resolution  = int(np.ceil(float(k*1.5+50)/coarse_grid_resolution)*coarse_grid_resolution)
        n = int(k/5+30)
        gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus=mus)
        d = gq["d"]
        u = d.solve(mus)
        e_r = []
        e_d = []
        for i in range(it):
            print(i, )
            sys.stdout.flush()
            bases = create_bases2(gq, lq, n, transfer='robin')
            ru_r = reconstruct_solution(gq, lq, bases)
            del bases
            dif_r = u-ru_r
            e_r.append(gq["full_norm"](dif_r)[0]/gq["full_norm"](u)[0])
            bases = create_bases2(gq, lq, n, transfer='dirichlet')
            ru_d = reconstruct_solution(gq, lq, bases)
            del bases
            dif_d = u-ru_d
            e_d.append(gq["full_norm"](dif_d)[0]/gq["full_norm"](u)[0])
        return np.mean(e_d), np.mean(e_r)
    pool = mp.Pool()
    results = pool.map(cube,  rang)
    means_d = np.array(results).T[0].tolist()
    means_r = np.array(results).T[1].tolist()

    data = np.vstack([rang, means_d, means_r]).T
    open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
    if plot:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.semilogy(rang, means_r, label="robin")
        plt.semilogy(rang, means_d, label="dirichlet")
        plt.xlabel('k')
        plt.legend(loc='upper right')
        plt.show()


def resolution(it, k, n, boundary, save, cloc=0, returnvals=False, coarse_grid_resolution=10):
    cglob = -1j*k
    mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
    p = helmholtz(boundary=boundary)
    space = np.arange(20, 160, 10)
    err = []
    for resolution in space:
        print(resolution)
        gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus=mus)
        d = gq["d"]
        u = d.solve(mus)
        E = []
        for i in range(it):
            bases = create_bases2(gq, lq, n, transfer='robin')
            ru = reconstruct_solution(gq, lq, bases)
            E.append(gq["full_norm"](u-ru)[0]/gq["full_norm"](u)[0])
        err.append(E)
    errs = np.mean(err, axis=1)
    data = np.vstack([space, errs]).T
    open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
    if returnvals:
        return [space, errs]


def cerr2D(it, n, k, boundary, save, cglob=0, rang=np.arange(-10., 10., 1.),
           yrang=None, plot=False, resolution=200, coarse_grid_resolution=10):
    if yrang is None:
        yrang = rang
    err_r = np.zeros((len(rang), len(yrang)))
    p = helmholtz(boundary=boundary)
    pool = mp.Pool(processes=process_count)
    xi = 0
    for x in rang:
        yi = 0
        for y in yrang:
            c = x+1j*y
            print(c)
            mus = {'k': k, 'c_glob': cglob, 'c_loc': c}
            gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus=mus)
            d = gq["d"]
            u = d.solve(mus)

            def cube():
                bases = create_bases2(gq, lq, n, transfer='robin')
                ru_r = reconstruct_solution(gq, lq, bases)
                del bases
                dif_r = u-ru_r
                return gq["full_norm"](dif_r)[0]/gq["full_norm"](u)[0]
            e_r = pool.map(cube, range(it))
            err_r[xi][yi] = np.mean(e_r)
            yi += 1
        xi += 1
    X, Y = np.meshgrid(rang, yrang)
    data = np.vstack([X.T.ravel(), Y.T.ravel(), err_r.ravel()]).T
    open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, err_r,  cstride=1, rstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


def knerr2D(it, boundary, save, cglob=None, cloc0=0, cloc1=1, cloc2=1,
            krang=np.arange(0.1, 200.1, 10.), nrang=np.arange(0, 100, 5),
            plot=False, resolution=100, coarse_grid_resolution=10):
    err_r = np.zeros((len(krang), len(nrang)))
    p = helmholtz(boundary=boundary)
    usecglob = (cglob is None)
    xi = 0
    for k in krang:
        yi = 0
        if usecglob:
            cglob = -1j*k
        cloc = cloc0 + cloc1 * k + cloc2 * k**2
        mus = {'k': k, 'c_glob': cglob, 'c_loc': cloc}
        gq, lq = localize_problem(p, coarse_grid_resolution, resolution, mus=mus)
        d = gq["d"]
        u = d.solve(mus)
        for n in nrang:
            print(k, n)
            e_r = []
            for i in range(min(20-n/5, it)):
                print(i, )
                sys.stdout.flush()
                bases = create_bases2(gq, lq, n, transfer='robin')
                ru_r = reconstruct_solution(gq, lq, bases)
                del bases
                dif_r = u-ru_r
                e_r.append(gq["full_norm"](dif_r)[0]/gq["full_norm"](u)[0])
            err_r[xi][yi] = np.mean(e_r)
            yi += 1
        xi += 1
    X, Y = np.meshgrid(krang, nrang)
    data = np.vstack([X.T.ravel(), Y.T.ravel(), err_r.ravel()]).T
    open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, err_r,  cstride=1, rstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
