from evaluations import *
import multiprocessing as mp


def cube(k): 
    return 1,2

pool = mp.Pool(processes=2)
a = pool.map(cube,  range(10))
print(a)
