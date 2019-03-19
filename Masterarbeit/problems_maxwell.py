from __future__ import absolute_import, division, print_function

from functools import partial

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.analyticalproblems.maxwell import MaxwellProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.domaindescriptions.basic import RectDomain
from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.functions.basic import FunctionBase, ConstantFunction
from pymor.operators.constructions import VectorFunctional, LincombOperator, VectorOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.functions.bitmap import BitmapFunction
from pymor.grids.interfaces import BoundaryInfoInterface
from pymor.functions.basic import GenericFunction, ExpressionFunction
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.tria import TriaGrid
from pymor.discretizers.maxwell import discretize_maxwell

import numpy as np
import scipy

def deaffinize_discretization(d, shift=None, mu_shift=None):
    assert (shift is None) != (mu_shift is None)
    if isinstance(shift, OperatorInterface):
        raise NotImplementedError
    if mu_shift is not None:
        mu_shift = d.parse_parameter(mu_shift)
        shift = d.solve(mu=mu_shift)
    vecs = [-op.apply(shift) for op in d.operator.operators]
    shift_functional = LincombOperator([VectorFunctional(v) for v in vecs],
                                       d.operator.coefficients)
    vectors = {'shift': VectorOperator(shift)}
    vectors.update(d.vector_operators)
    return d.with_(rhs=d.rhs + shift_functional,
                   vector_operators=vectors)


class AddDirichletBoundaryInfo(BoundaryInfoInterface):
    def __init__(self, bi, grid, function):
        self.bi = bi
        self.grid = grid
        self.function = function
        
        self.boundary_types = bi.boundary_types
        if BoundaryType('dirichlet') not in self.boundary_types:
            self.boundary_types.append(BoundaryType('dirichlet'))
        

    def mask(self, boundary_type, codim):
        assert 1 <= codim <= self.grid.dim
        target_codim = self.grid.dim - 1
        if codim == target_codim and boundary_type == BoundaryType('dirichlet'):
            additional_mask = np.array(self.function(self.grid.centers(target_codim)), dtype='bool')
            if boundary_type in self.bi.boundary_types:
                return np.logical_or(self.bi.mask(boundary_type, codim), additional_mask)
            else:
                return additional_mask
            
        return self.bi.mask(boundary_type, codim)
        
        
def h_problem(diameter=1/200, closed_connections=(3, 4), use_solution_for_shift=True):
    for i in closed_connections:
        assert i in (1, 2, 3, 4, 5, 6)
    backgroundfunction = ConstantFunction(1., dim_domain=2)
    coefficientfunction = BitmapFunction('h_allopen.png', range=(1., 0.))
    for i in closed_connections:
        coefficientfunction = coefficientfunction + BitmapFunction('h_close'+str(i)+'.png', range=(1., 0))

    p = EllipticProblem(domain=RectDomain([[0, 0], [1, 1]],
                                          top=BoundaryType("neumann"), bottom=BoundaryType("neumann")),
                        diffusion_functions=(backgroundfunction,
                                             coefficientfunction),
                        diffusion_functionals=(1.,
                                               ExpressionParameterFunctional('10**channels', {'channels': tuple()})),
                        rhs=ConstantFunction(0., dim_domain=2),
                        dirichlet_data=BitmapFunction('h_boundary.png', range=(-1., 2. * 255./100. - 1.)),
                        parameter_space=CubicParameterSpace({'channels': tuple()}, 0., 5.))
    d, data = discretize_elliptic_cg(p, diameter=diameter)

    # hack the system matrices for improved numerical stablility
    dirichlet_dofs = data['boundary_info'].dirichlet_boundaries(2)
    for op in d.operator.operators:
        op.assemble()._matrix[dirichlet_dofs, dirichlet_dofs] *= 1e5
    d.rhs.assemble()._matrix[:, dirichlet_dofs] *= 1e5

    if use_solution_for_shift:
        da = deaffinize_discretization(d, mu_shift=0.)
    else:
        g, bi = data['grid'], data['boundary_info']
        dirichlet_dofs = bi.dirichlet_boundaries(2)
        U_shift = d.solution_space.zeros()
        U_shift.data[:, dirichlet_dofs] = p.dirichlet_data.evaluate(g.centers(2)[dirichlet_dofs])
        da = deaffinize_discretization(d, shift=U_shift)
    return d, da, data

def maxwell_dirichlet_functionv(x, seqno):
    return np.array([maxwell_dirichlet_function(d, seqno) for d in x])

def maxwell_dirichlet_function(x, seqno):
    assert seqno in [0,1]
    class rangecondition:
        def __init__(self, xmin, xmax, ymin, ymax):
            self.xmin = xmin
            self.xmax = xmax
            self.ymin = ymin
            self.ymax = ymax

        def __call__(self,x):
            return (self.xmin <= x[0] <= self.xmax) and (self.ymin <= x[1] <= self.ymax)

    class circlecondition:
        def __init__(self, radius, centerx, centery):
            self.radius = radius
            self.centerx = centerx
            self.centery = centery

        def __call__(self, x):
            return np.sqrt((x[0] - self.centerx)**2 + (x[1] - self.centery)**2) <= self.radius

    class composedcondition:
        def __init__(self, positiveconditions, negativeconditions):
            self.pc = positiveconditions
            self.nc = negativeconditions

        def __call__(self, x):
            for c in self.pc:
                if not c(x):
                    return False

            for c in self.nc:
                if c(x):
                    return False

            return True

    windows = []
    #windows.append(rangecondition(0.60, 0.80, 0.60, 0.81))
    #windows.append(rangecondition(0.60, 0.80, 0.20, 0.40))
    #windows.append(rangecondition(0.01, 0.20, 0.20, 0.40))
    if seqno == 1:
        windows.append(rangecondition(0.01, 0.20, 0.58, 0.80))

    conditions = []    
    #xmin, xmax, ymin, ymax tuples:
    conditions.append(rangecondition(0., 0.60, 0.39, 0.40))
    conditions.append(rangecondition(0., 0.60, 0.59, 0.60))
    conditions.append(rangecondition(0.20, 1.00, 0.20, 0.21))
    conditions.append(rangecondition(0.22, 1.00, 0.80, 0.81))
    conditions.append(rangecondition(0., 0.01, 0., 0.39))
    conditions.append(rangecondition(0., 0.01, 0.60, 1.00))
    conditions.append(rangecondition(0.80, 1., 0.21, 0.80))

    conditions.append(composedcondition(
            [rangecondition(0., 0.20, 0., 0.40)],
            [circlecondition(0.20,0.20,0.20)]
            ))

    conditions.append(composedcondition(
            [rangecondition(0.60, 0.80, 0.20, 0.80)],
            [circlecondition(0.20,0.60,0.40), circlecondition(0.20, 0.60, 0.60)]
            ))

    conditions.append(composedcondition(
            [rangecondition(0., 0.20, 0.60, 1.)],
            [circlecondition(0.20,0.20,0.80)]
            ))

    for w in windows:
        if w(x):
            return 0

    for c in conditions:
        if c(x):
            return 1

    return 0

def maxwell_problem(diameter=1./200, sequence_number=0):
    domain = RectDomain(
        domain=([0, 0], [1, 1]),
        left=BoundaryType('robin'),
        right=BoundaryType('robin'),
        top=BoundaryType('dirichlet'),
        bottom=BoundaryType('dirichlet'))
    
    excitation = GenericFunction(
        lambda x: (
            0j + np.outer(np.exp(-np.linalg.norm(x - np.array([0.1,0.5]), axis=1)**2 * 800.) , np.array([0., 1.]))
            )
            ,
        dim_domain=2,
        shape_range=(2,)
    )

    robinvalue = 1. / 376.730313 * diameter
    mu = 4*np.pi*1e-7
    eps = 8.854187817e-12
    def func(x, mu):
	return ((x[...,0]>1-1e-13)+(x[...,0]<1e-13)+(x[...,1]>1-1e-13)+(x[...,1]<1e-13)>0)*robinvalue +((x[...,0]>1-1e-13)+(x[...,0]<1e-13)+(x[...,1]>1-1e-13)+(x[...,1]<1e-13)==0)* mu["c_loc"]
	#return ((x[...,0]==1)+(x[...,0]==0)+(x[...,1]==1)+(x[...,1]==0)>0)*robinvalue +((x[...,0]==1)+(x[...,0]==0)+(x[...,1]==1)+(x[...,1]==0)==0)* mu["c_loc"]
    robin_data = GenericFunction(func, dim_domain =2, parameter_type={'c_loc' : (), 'k': (), 'mu': (), 'eps': ()})
    #robin_data = ConstantFunction(robinvalue, dim_domain = 2)
    parameter_range=(0., 100.)
    problem = MaxwellProblem(domain=domain, robin_data=robin_data, excitation=excitation,
		 curl_functionals = [ExpressionParameterFunctional('1./mu', parameter_type={'c_loc' : (), 'k': (), 'mu': (), 'eps': ()}, )], 
		reaction_functionals = [ExpressionParameterFunctional('-k**2*eps', parameter_type={'c_loc' : (), 'k': (), 'mu': (), 'eps': ()})], 
		robin_functionals = [ExpressionParameterFunctional('1j*k', parameter_type={'c_loc' : (), 'k': (), 'mu': (), 'eps': ()})],
		rhs_functionals = [ExpressionParameterFunctional('-1*k', parameter_type={'c_loc' : (), 'k': (), 'mu': (), 'eps': ()})]
)
    #grid, bi = discretize_domain_default(problem.domain, diameter = diameter, grid_type=TriaGrid)



    #fun = GenericFunction(partial(maxwell_dirichlet_functionv,seqno=sequence_number), dim_domain=2)
    #bi = AddDirichletBoundaryInfo(bi, grid, fun)
    

    #U_shift = d.solution_space.zeros()
    #da = deaffinize_discretization(d, shift=U_shift)

    return problem
