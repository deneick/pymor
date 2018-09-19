from evaluations import *
set_log_levels(levels={'pymor': 'WARN'})


save = "dats/findc_bsp1.dat"
csr = []
csi = []
kk=20
krang = list(np.union1d([.1],range(4,24,4)))
for k in krang:
	c= findcloc(it=200, k=k, n=15, s = 1, smin = 0.5, cloc = 0., boundary = 'robin')
	csr.append(c.real)
	csi.append(c.imag)
	print  "k: ", k, "cs: ", csr, csi

data = np.vstack([krang, csr, csi]).T
open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])
