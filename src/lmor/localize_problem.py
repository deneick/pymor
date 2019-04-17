from pymor.discretizers.cg import discretize_stationary_cg
from pymor.operators.numpy import NumpyGenericOperator, NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator, induced_norm
from pymor.grids.subgrid import SubGrid
from pymor.grids.boundaryinfos import SubGridBoundaryInfo
from pymor.vectorarrays.numpy import NumpyVectorSpace
from lmor.localizer import NumpyLocalizer
from lmor.partitioner import build_subspaces, partition_any_grid
from lmor.discrete_pou import localized_pou
import numpy as np
import copy


def create_dirichlet_transfer(localizer, local_op, rhsop, source_space, training_space, range_space, pou):
    # Dirichlet-Transferoperator
    def transfer(va):
        range_solution = localizer.to_space(
            local_op.apply_inverse(-rhsop.apply(NumpyVectorSpace.make_array(va))), training_space, range_space)
        return pou[range_space](range_solution).data
    cavesize = len(localizer.join_spaces(source_space))
    rangesize = len(localizer.join_spaces(range_space))
    return NumpyGenericOperator(transfer, dim_source=cavesize, dim_range=rangesize, linear=True)


def create_dirichlet_solop(localizer, local_op, rhsop, source_space, training_space):
    # Dirichlet-Loesungsoperator
    def solve(va):
        solution = local_op.apply_inverse(-rhsop.apply(NumpyVectorSpace.make_array(va)))
        return solution.data
    cavesize = len(localizer.join_spaces(source_space))
    rangesize = len(localizer.join_spaces(training_space))
    return NumpyGenericOperator(solve, dim_source=cavesize, dim_range=rangesize, linear=True)


def create_robin_transfer(localizer, bilifo, source_space, omega_star_space, range_space, pou):
    # Robin-Transferoperator
    def transfer(va):
        g = localizer.to_space(NumpyVectorSpace.make_array(va), source_space, omega_star_space)
        solution = bilifo.apply_inverse(g)
        range_solution = localizer.to_space(solution, omega_star_space, range_space)
        return pou[range_space](range_solution).data
    cavesize = len(localizer.join_spaces(source_space))
    rangesize = len(localizer.join_spaces(range_space))
    return NumpyGenericOperator(transfer, dim_source=cavesize, dim_range=rangesize, linear=True)


def create_robin_solop(localizer, bilifo, source_space, omega_star_space):
    # Robin-Loesungsoperator
    def solve(va):
        g = localizer.to_space(NumpyVectorSpace.make_array(va), source_space, omega_star_space)
        solution = bilifo.apply_inverse(g)
        return solution.data
    cavesize = len(localizer.join_spaces(source_space))
    rangesize = len(localizer.join_spaces(omega_star_space))
    return NumpyGenericOperator(solve, dim_source=cavesize, dim_range=rangesize, linear=True)


def localize_problem(p, coarse_grid_resolution, fine_grid_resolution, mus=None, calT=False,
                     calTd=False, calQ=False, dof_codim=2, localization_codim=2, discretizer=None, m=1):

    assert coarse_grid_resolution > (m+1)*2
    assert fine_grid_resolution % coarse_grid_resolution == 0

    print("localizing problem")
    global_quantities = {}
    global_quantities["coarse_grid_resolution"] = coarse_grid_resolution
    local_quantities = {}

    # Diskretisierung auf dem feinen Gitter:
    diameter = 1./fine_grid_resolution
    global_quantities["diameter"] = diameter
    if discretizer is None:
        discretizer = discretize_stationary_cg
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

    # Skalierung der Dirichlet-Freiheitsgrade:
    try:
        dirichlet_dofs = data['boundary_info'].dirichlet_boundaries(dof_codim)
        for op in op_fixed.operators:
            op.assemble(mus).matrix[dirichlet_dofs, dirichlet_dofs] *= 1e5
        # d.rhs.assemble(mus)._matrix[:, dirichlet_dofs] *= 1e5
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

    # Konstruktion der Teilraeume:
    subspaces, subspaces_per_codim = build_subspaces(
        *partition_any_grid(grid, num_intervals=(coarse_grid_resolution, coarse_grid_resolution),
                            dmask=dmask, codim=dof_codim))
    global_quantities["subspaces"] = subspaces
    global_quantities["subspaces_per_codim"] = subspaces_per_codim
    localizer = NumpyLocalizer(d.solution_space, subspaces['dofs'])
    global_quantities["localizer"] = localizer

    def create_m_patch(xspace, m):
        for i in range(m):
            space = xspace
            cspace = tuple(set(i for k in space if subspaces[k]['codim']
                               == 0 for i in subspaces[k]['cpatch'] if i not in space))
            xspace = tuple(set(i for k in cspace if subspaces[k]['codim']
                               == 1 for i in subspaces[k]['env']) | set(space))
        cxspace = tuple(set(i for k in xspace if subspaces[k]['codim']
                            == 0 for i in subspaces[k]['cpatch'] if i not in xspace))
        return xspace, cxspace

    pou = localized_pou(subspaces, subspaces_per_codim, localizer,
                        coarse_grid_resolution, grid, localization_codim, dof_codim)
    global_quantities["pou"] = pou
    spaces = [subspaces[s_id]["env"] for s_id in subspaces_per_codim[2]]
    global_quantities["spaces"] = spaces

    full_l2_product = d.products["l2"].assemble()
    full_h1_semi_product = d.products["h1_semi"].assemble()
    k_product = LincombOperator((full_h1_semi_product, full_l2_product), (1, mus["k"]**2)).assemble()
    global_quantities["full_norm"] = induced_norm(k_product)
    global_quantities["k_product"] = k_product
    for xpos in range(coarse_grid_resolution-1):
        for ypos in range(coarse_grid_resolution-1):
            # print "localizing..."
            s_id = subspaces_per_codim[localization_codim][ypos + xpos*(coarse_grid_resolution-1)]
            space = subspaces[s_id]["env"]
            ldict = {}
            local_quantities[space] = ldict
            ldict["pos"] = (xpos, ypos)

            # Konstruktion der lokalen Raeume:
            range_space = subspaces[s_id]["env"]
            ldict["range_space"] = range_space
            omega_space = tuple(sorted(set(subspaces[s_id]['env']) | set(subspaces[s_id]['cenv'])))
            ldict["omega_space"] = omega_space
            training_space, source_space = create_m_patch(range_space, m)
            # source_space = subspaces[s_id]["cxenv"]
            ldict["source_space"] = source_space
            # training_space = subspaces[s_id]["xenv"]
            ldict["training_space"] = training_space
            omega_star_space = tuple(sorted(set(training_space) | set(source_space)))
            ldict["omega_star_space"] = omega_star_space

            # lokale Shift-Loesung mit f(Dirichlet)
            local_op = localizer.localize_operator(global_operator, training_space, training_space)
            local_rhs = localizer.localize_operator(global_rhs, None, training_space)
            local_solution = local_op.apply_inverse(local_rhs.as_range_array())
            local_solution = localizer.to_space(local_solution, training_space, range_space)
            local_solution = pou[range_space](local_solution)
            ldict["local_solution_dirichlet"] = local_solution

            # Dirichlet Transferoperator:
            rhsop = localizer.localize_operator(global_operator, training_space, source_space)
            transop_dirichlet = create_dirichlet_transfer(
                localizer, local_op, rhsop, source_space, training_space, range_space, pou)
            ldict["dirichlet_transfer"] = transop_dirichlet

            # subgrid
            xmin = max(0, xpos - m)
            xsize = min(xpos + 2+m, coarse_grid_resolution) - xmin
            ymin = max(0, ypos - m)
            ysize = min(ypos + 2+m, coarse_grid_resolution) - ymin
            ldict["posext"] = [(xmin/coarse_grid_resolution, ymin/coarse_grid_resolution),
                               ((xmin+xsize)/coarse_grid_resolution, (ymin+ysize)/coarse_grid_resolution)]
            mysubgrid = getsubgrid(grid, xmin, ymin, coarse_grid_resolution, xsize=xsize, ysize=ysize)
            mysubbi = SubGridBoundaryInfo(mysubgrid, grid, data['boundary_info'], 'robin')
            ld, ldata = discretizer(p, grid=mysubgrid, boundary_info=mysubbi)
            lop = ld.operator.assemble(mus)

            # index conversion
            ndofsext = len(ldata['grid'].parent_indices(dof_codim))
            global_dofnrsext = -100000000 * np.ones(shape=(d.solution_space.dim,))
            global_dofnrsext[ldata['grid'].parent_indices(dof_codim)] = np.array(range(ndofsext))
            lvecext = localizer.localize_vector_array(
                NumpyVectorSpace.make_array(global_dofnrsext), omega_star_space).data[0]

            # Robin Transferoperator:
            bilifo = NumpyMatrixOperator(lop.matrix[:, lvecext][lvecext, :])
            transop_robin = create_robin_transfer(localizer, bilifo, source_space, omega_star_space, range_space, pou)
            ldict["robin_transfer"] = transop_robin

            # lokale Shift-Loesung mit f(Robin)
            lrhs = ld.rhs.assemble(mus)
            llrhs = NumpyMatrixOperator(lrhs.matrix[lvecext.astype(int)])
            local_solution = bilifo.apply_inverse(llrhs.as_range_array())
            ldict["local_sol2"] = local_solution
            local_solution = localizer.to_space(local_solution, omega_star_space, range_space)
            local_solution_pou = pou[range_space](local_solution)
            ldict["local_solution_robin"] = local_solution_pou

            if calT:
                # Transfer-Matrix:
                ldict["transfer_matrix_robin"] = transop_robin.as_range_array().data.T

            if calTd:
                # Transfer-Matrix:
                ldict["transfer_matrix_dirichlet"] = transop_dirichlet.as_range_array().data.T

            # Konstruktion der Produkte:
            range_k = localizer.localize_operator(k_product, range_space, range_space)
            omstar_k = LincombOperator((
                NumpyMatrixOperator(ld.products["h1_semi"].assemble().matrix[:, lvecext][lvecext, :]),
                NumpyMatrixOperator(ld.products["l2"].assemble().matrix[:, lvecext][lvecext, :])),
                (1, mus["k"]**2)).assemble()

            ldict["omega_star_product"] = omstar_k
            ldict["range_product"] = range_k

            if calQ:
                # Loesungs-Matrix:
                solution_op_robin = create_robin_solop(localizer, bilifo, source_space, omega_star_space)
                Q_r = solution_op_robin.as_range_array()
                ldict["solution_matrix_robin"] = Q_r.data.T
                source_Q_r_product = NumpyMatrixOperator(omstar_k.apply(Q_r).data.T)
                ldict["source_product"] = source_Q_r_product

            lproduct = localizer.localize_operator(full_l2_product, source_space, source_space)
            lmat = lproduct.matrix.tocoo()
            lmat.data = np.array([4./6. * diameter if (row == col) else diameter / 6.
                                  for row, col in zip(lmat.row, lmat.col)])
            ldict["source_product"] = NumpyMatrixOperator(lmat.tocsc().astype(np.cfloat))

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

    mask = [filter(e) for e in grid.centers(0)]
    indices = np.nonzero(mask)[0]
    return SubGrid(grid, indices)
