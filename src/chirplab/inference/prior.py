"""Module for prior distributions."""

from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chirplab.inference import distribution


class Prior:
    """
    Prior distribution.

    Parameters
    ----------
    distributions
        Collection of probability distributions.
    """

    def __init__(self, distributions: Sequence[distribution.Distribution]) -> None:
        self.distributions = distributions

    def transform(self, q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
        """
        Transform unit-hypercube samples to samples from the prior distribution.

        Parameters
        ----------
        q
            Unit-hypercube samples.

        Returns
        -------
        x
            Samples from the prior distribution.

        Notes
        -----
        Currently, it is assumed that the distributions are independent.
        """
        x = q.copy()
        for i, distribution in enumerate(self.distributions):
            x[i] = distribution.calculate_ppf(q[i])
        return x

    @property
    def n_dim(self) -> int:
        """Number of parameters in the prior distribution."""
        return len(self.distributions)

    @property
    def periodic_indices(self) -> None | list[int]:
        """Indices of parameters with periodic boundary conditions."""
        return [i for i, distribution in enumerate(self.distributions) if distribution.is_periodic] or None

    @property
    def reflective_indices(self) -> None | list[int]:
        """Indices of parameters with reflective boundary conditions."""
        return [i for i, distribution in enumerate(self.distributions) if distribution.is_reflective] or None

    def sample(self, rng: None | numpy.random.Generator = None) -> numpy.typing.NDArray[numpy.floating]:
        """
        Sample from the prior distribution.

        Parameters
        ----------
        rng
            Random number generator for the sampling.

        Returns
        -------
        x
            Samples from the prior distribution.
        """
        return numpy.array([distribution.sample(rng) for distribution in self.distributions])
