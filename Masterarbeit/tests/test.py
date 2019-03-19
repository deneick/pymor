from evaluations import *
from partitioner import build_subspaces, partition_any_grid
from discrete_pou import *

p = maxwell_problem()
diameter = 1./200
d, data = discretize_maxwell(p, diameter = diameter)
mus = {'c_loc' : 0., 'k': 1e9*2*np.pi, 'mu': 4*np.pi*1e-7, 'eps': 8.854187817e-12}
u = d.solve(mus)
d.visualize(u)

coarse_grid_resolution = 10
dof_codim = 1 # codim of grid entities for dofs
localization_codim = 2
grid = data["grid"]
dmask = data['boundary_info'].dirichlet_mask(2)
subspaces, subspaces_per_codim = build_subspaces(*partition_any_grid(grid, num_intervals=(coarse_grid_resolution, coarse_grid_resolution), dmask = dmask, codim = dof_codim))
spaces = [subspaces[s_id]["env"] for s_id in subspaces_per_codim[localization_codim]]
localizer = NumpyLocalizer(d.solution_space, subspaces['dofs'])
pou = localized_pou(subspaces, subspaces_per_codim, localizer, coarse_grid_resolution, grid, localization_codim, dof_codim)

space = spaces[25]
lsol = localizer.localize_vector_array(u, space)
gsol = localizer.globalize_vector_array(lsol, space)

pousol = pou[space](lsol)
pougsol = localizer.globalize_vector_array(pousol, space)
d.visualize((u, gsol, pougsol))


U = u*0
for i in range(len(spaces)):
	space = spaces[i]
	lsol = localizer.localize_vector_array(u, space)
	gsol = localizer.globalize_vector_array(lsol, space)
	#d.visualize(gsol)

	pousol = pou[space](lsol)
	pougsol = localizer.globalize_vector_array(pousol, space)
	#d.visualize((u, gsol, pougsol))
	U += pougsol

d.visualize(U)


source_space = subspaces[s_id]["cxenv"]
training_space = subspaces[s_id]["xenv"]
range_space = subspaces[s_id]["env"]
omega_space = tuple(sorted(set(subspaces[s_id]['env']) | set(subspaces[s_id]['cenv'])))
omega_star_space = tuple(sorted(set(training_space) | set(source_space)))

xpos = 5
ypos = 5

xmin = max(0,xpos - 1)
xsize = min(xpos + 3, coarse_grid_resolution - 2 + 3) - xmin
ymin = max(0,ypos - 1)
ysize = min(ypos + 3, coarse_grid_resolution - 2 + 3) - ymin
mysubgrid = getsubgrid(grid, xmin, ymin, coarse_grid_resolution, xsize=xsize, ysize=ysize)
mysubbi = SubGridBoundaryInfo(mysubgrid, grid, data['boundary_info'], BoundaryType('robin'))
ld, ldata = discretize_maxwell(p, grid=mysubgrid, boundary_info=mysubbi)
lop = ld.operator.assemble(mus)

ndofsext = len(ldata['grid'].parent_indices(2))
global_dofnrsext = -100000000* np.ones(shape=(d.solution_space.dim,))
global_dofnrsext[ldata['grid'].parent_indices(2)] = np.array(range(ndofsext))
lvecext = localizer.localize_vector_array(NumpyVectorArray(global_dofnrsext), omega_star_space).data[0]

bilifo = NumpyMatrixOperator(lop._matrix[:,lvecext][lvecext,:])
