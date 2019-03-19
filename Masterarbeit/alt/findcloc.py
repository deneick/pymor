from evaluations import *
set_log_levels(levels={'pymor': 'WARN'})


save = "dats/findc_bsp1.dat"
cs = []
kk=20
for k in range(1,kk):
	c= findcloc(it=100, k=k, n=15, s = 4, smin = 0.5, cloc = 0., boundary = 'robin')
	cs.append(c)

data = np.vstack([range(1,kk), cs]).T
open(save, "w").writelines([" ".join(map(str, v)) + "\n" for v in data])



def findcloc(it=50, k=20, n=15, s = 4., smin = 0.5, cloc = 0., boundary = 'robin'):
	p = helmholtz(boundary = boundary)
	dic = {}
	while(True):
		if cloc in dic:
			err0 = dic[cloc]
		else: 
			mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc}
			gq, lq = localize_problem(p, coarse_grid_resolution=10, fine_grid_resolution=50, mus = mus)
			d = gq["d"]
			u = d.solve(mus)
			e_r = []
			for i in range(it):
				print cloc, i
				bases = create_bases2(gq,lq,n,transfer = 'robin')
				ru_r = reconstruct_solution(gq, lq, bases)
				del bases
				dif_r = u-ru_r
				e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
			err0 = np.mean(e_r)
			dic[cloc]=err0
		clocs = [cloc-s, cloc +s, cloc -s*1j, cloc +s*1j]
		errs = []
		for cloc1 in clocs:
			if cloc1 in dic:
				errs.append(dic[cloc1])
			else:
				mus = {'k': k, 'c_glob': -1j*k, 'c_loc': cloc1}
				gq, lq = localize_problem(p, coarse_grid_resolution=10, fine_grid_resolution=50, mus = mus)
				d = gq["d"]
				u = d.solve(mus)
				e_r = []
				for i in range(it):
					print "cloc: ", cloc, "cloc1: ", cloc1, "s: ", s, "i: " ,i
					bases = create_bases2(gq,lq,n,transfer = 'robin')
					ru_r = reconstruct_solution(gq, lq, bases)
					del bases
					dif_r = u-ru_r
					e_r.append(d.h1_norm(dif_r)[0]/d.h1_norm(u)[0])
				err = np.mean(e_r)
				errs.append(err)
				dic[cloc1]=err
		print errs
		print err0
		if np.min(errs)< err0:
			cloc = clocs[np.argmin(errs)]
			print "new cloc: ", cloc
		else: 
			s = s/2.
			print "new s: ", s
			if s < smin:
				print "found cloc!: ", cloc
				break
