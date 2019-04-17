from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.bitmap import BitmapFunction
from pymor.functions.basic import ConstantFunction, ExpressionFunction
import os


def h_problem(closed_connections=(3, 4), contrast=1e5, c=1e5, g=0, boundary='dirichlet'):
    for i in closed_connections:
        assert i in (1, 2, 3, 4, 5, 6)

    scaling = 1
    domain = RectDomain(bottom=boundary, top=boundary, left=boundary, right=boundary)
    filepath = os.path.dirname(os.path.realpath(__file__))
    rhsfunction = BitmapFunction(os.path.join(filepath, "rhs.png"), range=(-128./127.*scaling, 127./127.*scaling))
    backgroundfunction = ConstantFunction(1., dim_domain=2)
    robin_data = (ConstantFunction(c, dim_domain=2), ConstantFunction(g, dim_domain=2))
    coefficientfunction = BitmapFunction(os.path.join(filepath, 'h_allopen.png'), range=(1., 0.))
    for i in closed_connections:
        coefficientfunction = coefficientfunction + \
            BitmapFunction(os.path.join(filepath, 'h_close'+str(i)+'.png'), range=(1., 0))
    p = StationaryProblem(domain=domain,
                          diffusion_functions=(backgroundfunction, coefficientfunction),
                          diffusion_functionals=(1., contrast),
                          rhs=rhsfunction,
                          robin_data=robin_data,
                          name="H")
    return p


def poisson_problem(c=1, g=0, boundary='dirichlet'):
    backgroundfunction = ConstantFunction(1., dim_domain=2)
    robin_data = (ConstantFunction(c, dim_domain=2), ConstantFunction(g, dim_domain=2))
    domain = RectDomain(bottom=boundary, top=boundary, left=boundary, right=boundary)

    p = StationaryProblem(domain=domain,
                          diffusion_functions=(backgroundfunction,),
                          diffusion_functionals=(1.,),
                          rhs=ConstantFunction(1., dim_domain=2),
                          robin_data=robin_data,
                          name="poisson")

    return p


def helmholtz(boundary='robin', g=0., f=True):
    domain = RectDomain(bottom=boundary, top=boundary, left=boundary, right=boundary)

    if f:
        rhs = ExpressionFunction('(( (x[...,0]-0.15)**2 + (x[...,1]-0.15)**2) <= 0.01) * 1.', 2, ())
    else:
        rhs = ConstantFunction(0., dim_domain=domain.dim)

    parameter_type = {'k': (), 'c_glob': (), 'c_loc': ()}
    eps = 1e-15

    cfunc = ExpressionFunction('((x[...,0]>1-eps)+(x[...,0]<eps)+(x[...,1]>1-eps)+(x[...,1]<eps)>0)*\
                               c_glob +((x[...,0]>1-eps)+(x[...,0]<eps)+(x[...,1]>1-eps)+(x[...,1]<eps)==0)* c_loc',
                               2, (), parameter_type=parameter_type, values={'eps': eps})

    p = StationaryProblem(
        diffusion=ConstantFunction(1., dim_domain=domain.dim),
        reaction=ExpressionFunction('-k**2+x[...,0]*0', 2, (), parameter_type=parameter_type),
        domain=domain,
        rhs=rhs,
        robin_data=(cfunc, ConstantFunction(g, dim_domain=2)),
        neumann_data=ConstantFunction(0, dim_domain=2)
    )
    return p


def helmholtz_contrast(K=8, a=2, b=3):
    domain = RectDomain(top='robin', left='robin', bottom='robin', right='robin')
    parameter_type = {'eps': (), 'k': ()}
    diffusion = ExpressionFunction(
        '1. - (abs(np.mod(x[...,0],1./K)-0.5/K)<=0.25/K)*(abs(np.mod(x[...,1],1./K)-0.5/K)\
        <=0.25/K)*(x[...,0]>= 2./K)*(x[...,0]<=(1.-2./K))\
        *((x[...,0]<=x1/K)+(x[...,0]>=(x1+1)/K)+(x[...,1]<=x2/K)+(x[...,1]>=(x2+1)/K)>=1)* (1 - eps**2)',
        2, (),
        values={'K': K, 'x1': a, 'x2': b},
        parameter_type=parameter_type
    )
    problem = StationaryProblem(
        domain=domain,
        diffusion=diffusion,
        reaction=ExpressionFunction('-k**2+x[...,0]*0', 2, (), parameter_type=parameter_type),
        rhs=ExpressionFunction('(( (x[...,0]-0.25)**2 + (x[...,1]-0.5)**2) <= 0.001) * 1e4', 2, ()),
        robin_data=(ExpressionFunction('-k*1j+x[...,0]*0', 2, (),
                                       parameter_type=parameter_type), ConstantFunction(0., dim_domain=2))
    )
    return problem


def helmholtz_contrast2(K=8):
    domain = RectDomain(top='robin', left='robin', bottom='robin', right='robin')
    parameter_type = {'eps': (), 'k': (), 'x1': (), 'x2': ()}
    diffusion = ExpressionFunction(
        '1. - (abs(np.mod(x[...,0],1./K)-0.5/K)<=0.25/K)*(abs(np.mod(x[...,1],1./K)-0.5/K)\
        <=0.25/K)*(x[...,0]>= 2./K)*(x[...,0]<=(1.-2./K))\
        *((x[...,0]<=x1/K)+(x[...,0]>=(x1+1)/K)+(x[...,1]<=x2/K)+(x[...,1]>=(x2+1)/K)>=1)* (1 - eps**2)',
        2, (),
        values={'K': K},
        parameter_type=parameter_type
    )
    problem = StationaryProblem(
        domain=domain,
        diffusion=diffusion,
        reaction=ExpressionFunction('-k**2+x[...,0]*0', 2, (), parameter_type=parameter_type),
        rhs=ExpressionFunction('(( (x[...,0]-0.25)**2 + (x[...,1]-0.5)**2) <= 0.001) * 1e4', 2, ()),
        robin_data=(ExpressionFunction('-k*1j+x[...,0]*0', 2, (),
                                       parameter_type=parameter_type), ConstantFunction(0., dim_domain=2))
    )
    return problem
