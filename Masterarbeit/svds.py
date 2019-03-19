from evaluations import *
set_log_levels(levels={'pymor': 'WARN'})
k=6
resolution = 100
cloc = 1-0.5j
mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc}
p = helmholtz(boundary = 'robin')
gq, lq = localize_problem(p, coarse_grid_resolution=10, fine_grid_resolution=resolution, mus = mus, calQ = True, calT = True) 
print "done localizing"
svd = []
for space in gq["spaces"]:
	ldict = lq[space]
	Top = ldict["transfer_matrix_robin"]
	S = ldict["source_product"]._matrix
	R = ldict["range_product"]._matrix
	svd.append(operator_svd2(Top, S, R)[0])

id5 = [0,8,72,80]
id4 = [1,7,9,17,63,71,73,79]
id3 = [10,16,64,70]
id2 = [2,3,4,5,6,18,26,27,35,36,44,45,53,54,62,74,75,76,77,78]
id1 = [11,12,13,14,15,19,25,28,34,37,43,46,52,55,61,65,66,67,68,69]
id0 = [e for e in range(81) if e not in id5+id4+id3+id2+id1]
svd0 = list(np.mean([svd[i] for i in id0], axis =0))
svd1 = list(np.mean([svd[i] for i in id1], axis =0))
svd2 = list(np.mean([svd[i] for i in id2], axis =0))
svd3 = list(np.mean([svd[i] for i in id3], axis =0))
svd4 = list(np.mean([svd[i] for i in id4], axis =0))
svd5 = list(np.mean([svd[i] for i in id5], axis =0))

k=200
cloc = 532-68j
mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc}
p = helmholtz(boundary = 'robin')
gq, lq = localize_problem(p, coarse_grid_resolution=10, fine_grid_resolution=resolution, mus = mus, calQ = True, calT = True) 
print "done localizing"
svdd = []
for space in gq["spaces"]:
	ldict = lq[space]
	Top = ldict["transfer_matrix_robin"]
	S = ldict["source_product"]._matrix
	R = ldict["range_product"]._matrix
	svdd.append(operator_svd2(Top, S, R)[0])
svd00 = list(np.mean([svdd[i] for i in id0], axis =0))
svd10 = list(np.mean([svdd[i] for i in id1], axis =0))
svd20 = list(np.mean([svdd[i] for i in id2], axis =0))
svd30 = list(np.mean([svdd[i] for i in id3], axis =0))
svd40 = list(np.mean([svdd[i] for i in id4], axis =0))
svd50 = list(np.mean([svdd[i] for i in id5], axis =0))

while(len(svd1)<len(svd0)):
	svd1.append(svd1[len(svd1)-1])
while(len(svd2)<len(svd0)):
	svd2.append(svd2[len(svd2)-1])
while(len(svd3)<len(svd0)):
	svd3.append(svd3[len(svd3)-1])
while(len(svd4)<len(svd0)):
	svd4.append(svd4[len(svd4)-1])
while(len(svd5)<len(svd0)):
	svd5.append(svd5[len(svd5)-1])

while(len(svd10)<len(svd00)):
	svd10.append(svd10[len(svd10)-1])
while(len(svd20)<len(svd00)):
	svd20.append(svd20[len(svd20)-1])
while(len(svd30)<len(svd00)):
	svd30.append(svd30[len(svd30)-1])
while(len(svd40)<len(svd00)):
	svd40.append(svd40[len(svd40)-1])
while(len(svd50)<len(svd00)):
	svd50.append(svd50[len(svd50)-1])

data = np.vstack([svd0, svd1, svd2, svd3, svd4, svd5]).T
data1 = np.vstack([svd00, svd10, svd20, svd30, svd40, svd50]).T
open("dats/svd6.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
open("dats/svd200.dat", "w").writelines([" ".join(map(str, v)) + "\n" for v in data1])

