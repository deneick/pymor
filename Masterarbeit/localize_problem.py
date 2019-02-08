from localizer import NumpyLocalizer
from partitioner import build_subspaces, partition_any_grid
from pou import *
from discrete_pou import *
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.discretizers.maxwell import *
from pymor.discretizations.basic import StationaryDiscretization
from pymor.operators.numpy import NumpyGenericOperator
from pymor.operators.constructions import induced_norm, LincombOperator
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
	global_quantities["op_not_assembled"] = d.operator
	global_rhs = d.rhs.assemble(mus)
	global_quantities["rhs"] = global_rhs

	op_fixed = copy.deepcopy(d.operator) 

	#Skalierung der Dirichlet-Freiheitsgrade:
	try:
		dirichlet_dofs = data['boundary_info'].dirichlet_boundaries(dof_codim)
		for op in op_fixed.operators:
			op.assemble(mus)._matrix[dirichlet_dofs, dirichlet_dofs] *= 1e5
		#d.rhs.assemble(mus)._matrix[:, dirichlet_dofs] *= 1e5
	except KeyError:
		pass

	global_operator_fixed = op_fixed.assemble(mus)
	global_quantities["op_fixed"] = global_operator_fixed
	global_quantities["op_fixed_not_assembled"] = op_fixed

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
	spaces = [subspaces[s_id]["env"] for s_id in subspaces_per_codim[2]]
	global_quantities["spaces"] = spaces

	full_l2_product = d.products["l2"].assemble()
	full_h1_semi_product = d.products["h1_semi"].assemble()
	k_product = LincombOperator((full_h1_semi_product,full_l2_product),(1,mus["k"]**2)).assemble()
	global_quantities["full_norm"] = induced_norm(k_product)
	global_quantities["k_product"] = k_product
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
		
			#lokale Shift-Loesung mit f(Dirichlet)
			local_op = localizer.localize_operator(global_operator, training_space, training_space)
			local_rhs = localizer.localize_operator(global_rhs, None, training_space)
			local_d = StationaryDiscretization(local_op, local_rhs, cache_region=None)
			local_solution = local_d.solve()
			local_solution = localizer.to_space(local_solution, training_space, range_space)
			local_solution = pou[range_space](local_solution)
			ldict["local_solution_dirichlet"] = local_solution

			#Dirichlet Transferoperator:
			rhsop = localizer.localize_operator(global_operator, training_space, source_space)
			transop_dirichlet = create_dirichlet_transfer(localizer, local_op, rhsop, source_space, training_space, range_space, pou)
			ldict["dirichlet_transfer"] = transop_dirichlet
		
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

			ldict["psi"] = localizer.to_space(NumpyVectorArray(bilifo._matrix.T), omega_star_space, source_space).data.T
			
			#lokale Shift-Loesung mit f(Robin)
			lrhs = ld.rhs.assemble(mus)
			llrhs = NumpyMatrixOperator(lrhs._matrix[:,lvecext.astype(int)])
			lrhs_f = llrhs._matrix
			local_solution = StationaryDiscretization(bilifo, llrhs, cache_region=None).solve()
			ldict["local_sol2"] = local_solution
			local_solution = localizer.to_space(local_solution, omega_star_space, range_space)
			local_solution_pou = pou[range_space](local_solution)
			ldict["local_solution_robin"] = local_solution_pou 

			if calT:
				#Transfer-Matrix:
				T_source_size = transop_robin.source.dim
				T_range_size = transop_robin.range.dim
				T_robin = np.zeros((T_range_size, T_source_size), dtype = complex)
				for i in range(T_source_size):
					ei = NumpyVectorArray(np.eye(1,T_source_size,i))
					ti = transop_robin.apply(ei)
					T_robin.T[i] = ti._array
				ldict["transfer_matrix_robin"] = T_robin

			#Konstruktion der Produkte:
			range_k = localizer.localize_operator(k_product, range_space, range_space)
			omstar_k = LincombOperator((NumpyMatrixOperator(ld.products["h1_semi"].assemble()._matrix[:,lvecext][lvecext,:]),NumpyMatrixOperator(ld.products["l2"].assemble()._matrix[:,lvecext][lvecext,:])),(1,mus["k"]**2)).assemble()


			ldict["omega_star_product"] = omstar_k
			ldict["range_product"] = range_k

			if calQ:
				#Loesungs-Matrix:
				solution_op_robin = create_robin_solop(localizer, bilifo, source_space, omega_star_space)		
				Q_r_source_size = solution_op_robin.source.dim
				Q_r_range_size = solution_op_robin.range.dim
				Q_r = np.zeros((Q_r_range_size, Q_r_source_size), dtype = complex)
				for i in range(Q_r_source_size):
					ei = NumpyVectorArray(np.eye(1,Q_r_source_size,i))
					qi = solution_op_robin.apply(ei)
					Q_r.T[i] = qi._array
				ldict["solution_matrix_robin"] = Q_r
				source_Q_r_product = NumpyMatrixOperator(Q_r.T.conj().dot(omstar_k._matrix.dot(Q_r)))
				ldict["source_product"] = source_Q_r_product

			lproduct = localizer.localize_operator(full_l2_product, source_space, source_space)
			lmat = lproduct._matrix.tocoo()
			lmat.data = np.array([4./6.*diameter if (row == col) else diameter/6. for row, col in izip(lmat.row, lmat.col)])
			ldict["source_product"] = NumpyMatrixOperator(lmat.tocsc())
			#ldict["source_product"] = NumpyMatrixOperator(scipy.sparse.csr_matrix(np.identity(len(localizer.join_spaces(source_space)))))

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

