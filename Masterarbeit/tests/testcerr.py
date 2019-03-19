from evaluations import *
import multiprocessing as mp

p = helmholtz(boundary = 'robin')
k=100
resolution = int(np.ceil(float(k*1.5+30)/10)*10)
n = k/4+20

def cube(cloc): 
    mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc}
    gq, lq = localize_problem(p, 10, resolution, mus = mus)
    d = gq["d"]
    u = d.solve(mus)
    bases = create_bases2(gq,lq,n,transfer = 'robin', silent = False)
    ru_r = reconstruct_solution(gq, lq, bases, silent = False)
    dif_r = u-ru_r
    return gq["full_norm"](dif_r)[0]/gq["full_norm"](u)[0]

pool = mp.Pool(processes=1)
results = pool.map(cube,  range(0,300,10))
print(results)

save = "dats/cerrtest.dat"
data = np.vstack([range(0,300,10), results]).T
open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
