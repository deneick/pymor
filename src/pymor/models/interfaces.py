# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.cache import CacheableInterface
from pymor.core.interfaces import abstractmethod
from pymor.parameters.base import Parametric


class ModelInterface(CacheableInterface, Parametric):
    """Interface for model objects.

    A model object defines a discrete problem
    via its `class` and the |Operators| it contains.
    Furthermore, models can be
    :meth:`solved <ModelInterface.solve>` for a given
    |Parameter| resulting in a solution |VectorArray|.

    Attributes
    ----------
    solution_space
        |VectorSpace| of the |VectorArrays| returned by :meth:`solve`.
    linear
        `True` if the model describes a linear problem.
    products
        Dict of inner product operators associated with the model.
    """

    solution_space = None
    linear = False
    products = dict()

    @abstractmethod
    def _solve(self, mu=None):
        """Perform the actual solving."""
        pass

    def solve(self, mu=None, **kwargs):
        """Solve the discrete problem for the |Parameter| `mu`.

        The result will be :mod:`cached <pymor.core.cache>`
        in case caching has been activated for the given model.

        Parameters
        ----------
        mu
            |Parameter| for which to solve.

        Returns
        -------
        The solution given as a |VectorArray|.
        """
        mu = self.parse_parameter(mu)
        return self.cached_method_call(self._solve, mu=mu, **kwargs)

    def estimate(self, U, mu=None):
        """Estimate the model error for a given solution.

        The model error could be the error w.r.t. the analytical
        solution of the given problem or the model reduction error w.r.t.
        a corresponding high-dimensional |Model|.

        Parameters
        ----------
        U
            The solution obtained by :meth:`~solve`.
        mu
            |Parameter| for which `U` has been obtained.

        Returns
        -------
        The estimated error.
        """
        raise NotImplementedError

    def visualize(self, U, **kwargs):
        """Visualize a solution |VectorArray| U.

        Parameters
        ----------
        U
            The |VectorArray| from :attr:`~ModelInterface.solution_space`
            that shall be visualized.
        """
        raise NotImplementedError
