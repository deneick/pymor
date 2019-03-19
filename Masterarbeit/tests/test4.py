from evaluations import *


boundary = 'robin'
domain=RectDomain(bottom = boundary, top = boundary, left = boundary, right = boundary)

rhs = ExpressionFunction('(( (x[...,0]-0.15)**2 + (x[...,1]-0.15)**2) <= 0.01) * 1.', 2, ())

parameter_range=(0., 100.)
parameter_type = {'k': (), 'c_glob': (), 'c_loc': ()}
parameter_space = CubicParameterSpace(parameter_type, *parameter_range)

eps = 1e-15

cfunc = ExpressionFunction('((x[...,0]>1-eps)+(x[...,0]<eps)+(x[...,1]>1-eps)+(x[...,1]<eps)>0)*c_glob +((x[...,0]>1-eps)+(x[...,0]<eps)+(x[...,1]>1-eps)+(x[...,1]<eps)==0)* c_loc', 2, (), parameter_type = parameter_type, values = {'eps': eps})

p = StationaryProblem(
	diffusion= ConstantFunction(1., dim_domain=domain.dim),
	reaction = ExpressionFunction('-k**2+0*x[...,0]',2,(), parameter_type = parameter_type),
	domain=domain,
	rhs = rhs,
	#parameter_space=parameter_space,
	robin_data = (cfunc, ConstantFunction(0, dim_domain=2)),
	neumann_data = ConstantFunction(0, dim_domain = 2)
)

d,data = discretize_stationary_cg(p)
mus = {'k': 6, 'c_glob': 1. , 'c_loc': 1.}
u = d.solve(mus)
d.visualize(u)

