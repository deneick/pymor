from localizer import NumpyLocalizer
from partitioner import build_subspaces, partition_any_grid
from pou import *
from discrete_pou import *
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.discretizers.maxwell import *
from pymor.discretizations.basic import StationaryDiscretization
from pymor.operators.numpy import NumpyGenericOperator
#from pymor.operators.constructions import induced_norm, LincombOperator
from pymor.grids.subgrid import SubGrid
from pymor.grids.boundaryinfos import SubGridBoundaryInfo
from pymor.domaindescriptions.boundarytypes import BoundaryType
import numpy as np
from itertools import izip
import copy
import scipy
from pymor.operators.constructions import VectorFunctional
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.operators.numpy import NumpyMatrixOperator
#from pymor.operators.cg import L2ProductP1, DiffusionOperatorP1

#Dirichlet-Transferoperator
def create_dirichlet_transfer(localizer, local_op, rhsop, source_space, training_space, range_space, pou):
	def transfer(va):
		range_solution = localizer.to_space(local_op.apply_inverse(-rhsop.apply(NumpyVectorArray(va))), training_space, range_space)
		return pou[range_space](range_solution)
	cavesize = len(localizer.join_spaces(source_space))
	rangesize = len(localizer.join_spaces(range_space))
	return NumpyGenericOperator(transfer, cavesize, rangesize, linear=True)

#Dirichlet-Loesungsoperator
def create_dirichlet_solop(localizer, local_op, rhsop, source_space, training_space):
	def solve(va):
		solution = local_op.apply_inverse(-rhsop.apply(NumpyVectorArray(va)))
		return solution
	cavesize = len(localizer.join_spaces(source_space))
	rangesize = len(localizer.join_spaces(training_space))
	return NumpyGenericOperator(solve, cavesize, rangesize, linear=True)

#Robin-Transferoperator
def create_robin_transfer(localizer, bilifo, source_space, omega_star_space, range_space, pou):
	def transfer(va):
		g = localizer.to_space(NumpyVectorArray(va), source_space, omega_star_space)			
		solution = StationaryDiscretization(bilifo, VectorFunctional(g), cache_region=None).solve()
		range_solution = localizer.to_space(solution, omega_star_space, range_space)
		return pou[range_space](range_solution)
	cavesize = len(localizer.join_spaces(source_space))
	rangesize = len(localizer.join_spaces(range_space))
	return NumpyGenericOperator(transfer, cavesize, rangesize, linear=True)

#Robin-Loesungsoperator
def create_robin_solop(localizer, bilifo, source_space, omega_star_space):
	def solve(va):
		g = localizer.to_space(NumpyVectorArray(va), source_space, omega_star_space)			
		solution = StationaryDiscretization(bilifo, VectorFunctional(g), cache_region=None).solve()
		return solution
	cavesize = len(localizer.join_spaces(source_space))
	rangesize = len(localizer.join_spaces(omega_star_space))
	return NumpyGenericOperator(solve, cavesize, rangesize, linear=True)

def localize_problem(p, coarse_grid_resolution, fine_grid_resolution, mus = None, calT = False, calQ = False, dof_codim = 2, localization_codim = 2, discretizer = None):

	print "localizing problem"
	global_quantities = {}
	global_quantities["coarse_grid_resolution"] = coarse_grid_resolution
	local_quantities = {}

	#Diskretisierung auf dem feinen Gitter:
	diameter = 1./fine_grid_resolution
	global_quantities["diameter"] = diameter
	if discretizer is None:
		discretizer = discretize_elliptic_cg
	d, data = discretizer(p, diameter=diameter)
	grid = data["grid"]

	global_quantities["d"] = d
	global_quantities["data"] = data
	
	global_operator = d.operator.assemble(mus)
	global_quantities["op"] = global_operator
	global_rhs = d.rhs.assemble(mus)
	global_quantities["rhs"] = global_rhs

	global_quantities["p"] = p
	
	try: 
		dmask = data['boundary_info'].dirichlet_mask(dof_codim)
	except KeyError:
		dmask = None

	#Konstruktion der Teilraeume:
	subspaces, subspaces_per_codim = build_subspaces(*partition_any_grid(grid, num_intervals=(coarse_grid_resolution, coarse_grid_resolution), dmask = dmask, codim = dof_codim))
	global_quantities["subspaces"] = subspaces
	global_quantities["subspaces_per_codim"] = subspaces_per_codim
	localizer = NumpyLocalizer(d.solution_space, subspaces['dofs'])
	global_quantities["localizer"] = localizer
	pou = localized_pou(subspaces, subspaces_per_codim, localizer, coarse_grid_resolution, grid, localization_codim, dof_codim)
	global_quantities["pou"] = pou
	spaces = [subspaces[s_id]["env"] for s_id in subspaces_per_codim[localization_codim]]
	global_quantities["spaces"] = spaces

	full_l2_product = d.products["l2"].assemble()
	full_h1_product = d.products["h1"].assemble()
	global_quantities["h1_prod"] = full_h1_product
	#full_h1_0_product = d.products["h1_0"].assemble()
	#global_quantities["h1_0_prod"] = full_h1_0_product
	for xpos in range(coarse_grid_resolution-1):
		for ypos in range(coarse_grid_resolution-1):
			#print "localizing..."
			s_id = subspaces_per_codim[localization_codim][ypos + xpos*(coarse_grid_resolution-1)]
			space = subspaces[s_id]["env"]
			ldict = {}
			local_quantities[space] = ldict
		
			#Konstruktion der lokalen Raeume:
			source_space = subspaces[s_id]["cxenv"]
			ldict["source_space"] = source_space
			training_space = subspaces[s_id]["xenv"]
			ldict["training_space"] = training_space
			range_space = subspaces[s_id]["env"]
			ldict["range_space"] = range_space
			omega_space = tuple(sorted(set(subspaces[s_id]['env']) | set(subspaces[s_id]['cenv'])))
			ldict["omega_space"] = omega_space
			omega_star_space = tuple(sorted(set(training_space) | set(source_space)))
			ldict["omega_star_space"] = omega_star_space
				
			#subgrid
			xmin = max(0,xpos - 1)
			xsize = min(xpos + 3, coarse_grid_resolution - 2 + 3) - xmin
			ymin = max(0,ypos - 1)
			ysize = min(ypos + 3, coarse_grid_resolution - 2 + 3) - ymin
			mysubgrid = getsubgrid(grid, xmin, ymin, coarse_grid_resolution, xsize=xsize, ysize=ysize)
			mysubbi = SubGridBoundaryInfo(mysubgrid, grid, data['boundary_info'], BoundaryType('robin'))
			ld, ldata = discretizer(p, grid=mysubgrid, boundary_info=mysubbi)
			lop = ld.operator.assemble(mus)

			#index conversion		
			ndofsext = len(ldata['grid'].parent_indices(dof_codim))
			global_dofnrsext = -100000000* np.ones(shape=(d.solution_space.dim,))
			global_dofnrsext[ldata['grid'].parent_indices(dof_codim)] = np.array(range(ndofsext))
			lvecext = localizer.localize_vector_array(NumpyVectorArray(global_dofnrsext), omega_star_space).data[0]

			#Robin Transferoperator:
			bilifo = NumpyMatrixOperator(lop._matrix[:,lvecext][lvecext,:])
			transop_robin = create_robin_transfer(localizer, bilifo, source_space, omega_star_space, range_space, pou)
			ldict["robin_transfer"] = transop_robin
			
			#lokale Shift-Loesung mit f(Robin)
			lrhs = ld.rhs.assemble(mus)
			llrhs = NumpyMatrixOperator(lrhs._matrix[:,lvecext.astype(int)])
			lrhs_f = llrhs._matrix
			local_solution = StationaryDiscretization(bilifo, llrhs, cache_region=None).solve()
			ldict["local_sol2"] = local_solution
			local_solution = localizer.to_space(local_solution, omega_star_space, range_space)
			local_solution_pou = pou[range_space](local_solution)
			ldict["local_solution_robin"] = local_solution_pou 
			
			#Konstruktion der Produkte:
			range_h1 = localizer.localize_operator(full_h1_product, range_space, range_space)

			ldict["range_product"] = range_h1

	return global_quantities, local_quantities

def getsubgrid(grid, xpos, ypos, coarse_grid_resolution, xsize=2, ysize=2):
    assert 0 <= xpos <= coarse_grid_resolution - 2
    assert 0 <= ypos <= coarse_grid_resolution - 2

    xstep = float(grid.domain[1][0] - grid.domain[0][0])/coarse_grid_resolution
    ystep = float(grid.domain[1][1] - grid.domain[0][1])/coarse_grid_resolution

    xmin = grid.domain[0][0] + xpos*xstep
    xmax = xmin + xsize*xstep

    ymin = grid.domain[0][1] + ypos * ystep
    ymax = ymin + ysize*ystep

    def filter(elem):
        return (xmin <= elem[0] <= xmax) and (ymin <= elem[1] <= ymax)

    mask = map(filter, grid.centers(0))
    indices = np.nonzero(mask)[0]
    return SubGrid(grid, indices)	

