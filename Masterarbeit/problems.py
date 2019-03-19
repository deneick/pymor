from __future__ import absolute_import, division, print_function
import os

from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.domaindescriptions.basic import RectDomain
from pymor.operators.constructions import VectorFunctional, LincombOperator, VectorOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.functions.bitmap import BitmapFunction
from pymor.functions.basic import *

import numpy as np

def h_problem(closed_connections=(3, 4), contrast=1e5, c = 1e5, g = 0, boundary = 'dirichlet'):
    for i in closed_connections:
        assert i in (1, 2, 3, 4, 5, 6)

    scaling = 1
    domain=RectDomain(bottom = boundary, top = boundary, left = boundary, right = boundary)
    filepath = os.path.dirname(os.path.realpath(__file__))
    rhsfunction = BitmapFunction(os.path.join(filepath,"rhs.png"), range=(-128./127.*scaling, 127./127.*scaling))
    backgroundfunction = ConstantFunction(1., dim_domain=2)
    robin_data = (ConstantFunction(c, dim_domain=2), ConstantFunction(g, dim_domain=2))
    coefficientfunction = BitmapFunction(os.path.join(filepath,'h_allopen.png'), range=(1., 0.))
    for i in closed_connections:
        coefficientfunction = coefficientfunction + BitmapFunction(os.path.join(filepath,'h_close'+str(i)+'.png'), range=(1., 0))
    p = StationaryProblem(domain=domain,
                        diffusion_functions=(backgroundfunction,
                                             coefficientfunction),
                        diffusion_functionals=(1.,
                                               contrast),
                        rhs=rhsfunction,
			robin_data = robin_data,
                        name="H")
    return p

def poisson_problem(c = 1, g = 0, boundary = 'dirichlet'):
    backgroundfunction = ConstantFunction(1., dim_domain=2)
    robin_data = (ConstantFunction(c, dim_domain=2), ConstantFunction(g, dim_domain=2))
    domain=RectDomain(bottom = boundary, top = boundary, left = boundary, right = boundary)

    p = StationaryProblem(domain = domain,
                        diffusion_functions=(backgroundfunction,),
                        diffusion_functionals=(1.,),
                        rhs=ConstantFunction(1., dim_domain=2),
			robin_data = robin_data,
                        name="poisson")

    return p

def helmholtz(boundary = 'robin', g=0., f=True):
	domain=RectDomain(bottom = boundary, top = boundary, left = boundary, right = boundary)

	if f:
		rhs = ExpressionFunction('(( (x[...,0]-0.15)**2 + (x[...,1]-0.15)**2) <= 0.01) * 1.', 2, ())
	else:
		rhs=ConstantFunction(0., dim_domain=domain.dim)

	parameter_range=(0., 100.)
	parameter_type = {'k': (), 'c_glob': (), 'c_loc': ()}
	parameter_space = CubicParameterSpace(parameter_type, *parameter_range)

	eps = 1e-15

	def func(x, mu):
		return ((x[...,0]>1-eps)+(x[...,0]<eps)+(x[...,1]>1-eps)+(x[...,1]<eps)>0)*mu["c_glob"] +((x[...,0]>1-eps)+(x[...,0]<eps)+(x[...,1]>1-eps)+(x[...,1]<eps)==0)* mu["c_loc"]
	cfunc = GenericFunction(func, dim_domain =2, parameter_type=parameter_type)
	cfunc = ExpressionFunction('((x[...,0]>1-eps)+(x[...,0]<eps)+(x[...,1]>1-eps)+(x[...,1]<eps)>0)*c_glob +((x[...,0]>1-eps)+(x[...,0]<eps)+(x[...,1]>1-eps)+(x[...,1]<eps)==0)* c_loc', 2, (), parameter_type = parameter_type, values = {'eps': eps})

	p = StationaryProblem(
		diffusion= ConstantFunction(1., dim_domain=domain.dim),
		reaction = ExpressionFunction('-k**2+x[...,0]*0',2,(), parameter_type = parameter_type),
		domain=domain,
		rhs = rhs,
		#parameter_space=parameter_space,
		robin_data = (cfunc, ConstantFunction(g, dim_domain=2)),
		neumann_data = ConstantFunction(0, dim_domain = 2)
	)
	return p
