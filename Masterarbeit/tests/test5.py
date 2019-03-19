from evaluations import *



domain = RectDomain(top = 'robin', left = 'robin', bottom='robin')
neumann_data = ExpressionFunction('-cos(pi*x[...,0])**2*neum', 2, (), parameter_type= {'neum': (), 'diffu':()})

diffusion = ExpressionFunction(
    '1. - (sqrt( (np.mod(x[...,0],1./K)-0.5/K)**2 + (np.mod(x[...,1],1./K)-0.5/K)**2) <= 0.3/K) * (1 - diffu)',
    2, (),
    values={'K': 10},
    parameter_type= {'diffu': (), 'neum':()}
)
problem = StationaryProblem(
    domain=domain,
    diffusion=diffusion,
    robin_data=(neumann_data, ConstantFunction(1.,dim_domain= 2))
)

d, data = discretize_stationary_cg(problem, diameter=1/100)
mus = {'diffu': 5., 'neum': 6}
u= d.solve(mus)
#op = d.operator.assemble(mus)
d.visualize(u)

