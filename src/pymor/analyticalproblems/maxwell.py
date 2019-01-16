# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction


class MaxwellProblem(ImmutableInterface):

    def __init__(self, domain=RectDomain(), excitation=ConstantFunction(np.array([0., 0.]), dim_domain=2),
                 dirichlet_data=None, robin_data=None, parameter_space=None, name=None, curl_functionals = None, reaction_functionals = None, robin_functionals = None, rhs_functionals = None):
        assert excitation.dim_domain == 2 and excitation.shape_range == (2,)
        assert dirichlet_data is None or (dirichlet_data.dim_domain == 2 and dirichlet_data.shape_range == (2,))
        self.domain = domain
        self.excitation = excitation
        self.dirichlet_data = dirichlet_data
        self.robin_data = robin_data
        self.parameter_space = parameter_space
        self.name = name
	self.curl_functionals = curl_functionals
	self.reaction_functionals = reaction_functionals
	self.robin_functionals = robin_functionals
	self.rhs_functionals = rhs_functionals
