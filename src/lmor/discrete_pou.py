from pymor.vectorarrays.numpy import NumpyVectorSpace
from lmor.pou import partition_of_unity
import numpy as np


def discrete_multiply(va, function):
    assert len(function.shape) == 1
    return NumpyVectorSpace.make_array(va.data * function)


def subspace_multiplication(grid, function, subspace, codim):
    multiplicationvector = function(grid.centers(codim)[subspace])

    def fun(x):
        return discrete_multiply(x, multiplicationvector)

    return fun


def localized_pou(subspaces, subspaces_per_codim, localizer,
                  coarse_grid_resolution, grid, localization_codim, dof_codim):
    boundaries_x = np.linspace(grid.domain[0][0], grid.domain[1][0], coarse_grid_resolution+1)
    boundaries_y = np.linspace(grid.domain[0][1], grid.domain[1][1], coarse_grid_resolution+1)
    pou = partition_of_unity((boundaries_x, boundaries_y), True)

    result = {}
    for space_id, fun in zip(subspaces_per_codim[localization_codim], pou.flatten()):
        space = subspaces[space_id]['env']
        space_dofs = localizer.join_spaces(space)
        myfunction = subspace_multiplication(grid, fun, space_dofs, dof_codim)
        result[space] = myfunction

    return result


if __name__ == "__main__":
    from pymor.analyticalproblems.elliptic import EllipticProblem
    from pymor.domaindescriptions.basic import RectDomain
    from pymor.functions.basic import ConstantFunction
    from my_discretize_elliptic_cg import discretize_elliptic_cg
    from lmor.localizer import NumpyLocalizer
    from lmor.partitioner import build_subspaces, partition_any_grid

    coarse_grid_resolution = 10

    p = EllipticProblem(
        domain=RectDomain([[0, 0], [1, 1]]),
        diffusion_functions=(ConstantFunction(1., dim_domain=2),),
        diffusion_functionals=(1.,),
        rhs=ConstantFunction(1., dim_domain=2))
    d, data = discretize_elliptic_cg(p, diameter=0.01)
    grid = data["grid"]

    subspaces, subspaces_per_codim = build_subspaces(
        *partition_any_grid(grid, num_intervals=(coarse_grid_resolution, coarse_grid_resolution)))

    localizer = NumpyLocalizer(d.solution_space, subspaces['dofs'])

    images = d.solution_space.empty()
    fdict = localized_pou(subspaces, subspaces_per_codim, localizer, coarse_grid_resolution, grid)
    for space in sorted(fdict):
        lvec = localizer.localize_vector_array(NumpyVectorSpace.make_array(np.ones(d.solution_space.dim)), space)
        lvec = fdict[space](lvec)
        gvec = localizer.globalize_vector_array(lvec, space)
        images.append(gvec)

    sum = d.solution_space.zeros()
    for i in range(len(images)):
        sum += images.copy(ind=i)

    d.visualize(images)
    assert np.abs(np.max(sum.data)-1) < 1e-12
    assert np.abs(np.min(sum.data)-1) < 1e-12
