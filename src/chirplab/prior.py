"""Module for prior distributions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy

from . import constants

type BOUNDARY_TYPES = None | Literal["periodic", "reflective"]

# TODO: add uniform in comoving volume prior
# TODO: add constraint prior
# TODO: add Gaussian prior
# TODO: write sample method


class Prior(ABC):
    """
    Prior base class.

    Parameters
    ----------
    boundary
        Boundary condition for the prior.
    """

    @abstractmethod
    def __init__(self, boundary: BOUNDARY_TYPES = None) -> None:
        is_periodic = is_reflective = False
        if boundary == "periodic":
            is_periodic = True
        elif boundary == "reflective":
            is_reflective = True
        self.is_periodic = is_periodic
        self.is_reflective = is_reflective

    @abstractmethod
    def transform(self, u: float) -> float:
        """
        Transform sample between 0 and 1 to prior space.

        Parameters
        ----------
        u
            Random sample between 0 and 1.

        Returns
        -------
        x
            Transformed sample in prior space.
        """
        pass


class Uniform(Prior):
    """
    Uniform prior.

    Parameters
    ----------
    x_min
        Minimum value of the prior.
    x_max
        Maximum value of the prior.
    boundary
        Boundary condition for the prior.
    """

    def __init__(self, x_min: float, x_max: float, boundary: BOUNDARY_TYPES = None) -> None:
        super().__init__(boundary)
        self.x_min = x_min
        self.x_max = x_max

    def transform(self, u: float) -> float:
        """
        Transform sample between 0 and 1 to prior space.

        Parameters
        ----------
        u
            Random sample between 0 and 1.

        Returns
        -------
        x
            Transformed sample in prior space.
        """
        return self.x_min + (self.x_max - self.x_min) * u


class Cosine(Prior):
    """
    Cosine prior.

    Parameters
    ----------
    x_min
        Minimum value of the prior.
    x_max
        Maximum value of the prior.
    boundary
        Boundary condition for the prior.
    """

    def __init__(
        self, x_min: float = -constants.PI / 2, x_max: float = constants.PI / 2, boundary: BOUNDARY_TYPES = None
    ) -> None:
        super().__init__(boundary)
        self.x_min = x_min
        self.x_max = x_max

    def transform(self, u: float) -> numpy.float64:
        """
        Transform sample between 0 and 1 to prior space.

        Parameters
        ----------
        u
            Random sample between 0 and 1.

        Returns
        -------
        x
            Transformed sample in prior space.
        """
        sin_min = numpy.sin(self.x_min)
        sin_max = numpy.sin(self.x_max)
        x: numpy.float64 = numpy.arcsin(sin_min + (sin_max - sin_min) * u)
        return x


class Sine(Prior):
    """
    Sine prior.

    Parameters
    ----------
    x_min
        Minimum value of the prior.
    x_max
        Maximum value of the prior.
    boundary
        Boundary condition for the prior.
    """

    def __init__(self, x_min: float = 0, x_max: float = constants.PI, boundary: BOUNDARY_TYPES = None) -> None:
        super().__init__(boundary)
        self.x_min = x_min
        self.x_max = x_max

    def transform(self, u: float) -> numpy.float64:
        """
        Transform sample between 0 and 1 to prior space.

        Parameters
        ----------
        u
            Random sample between 0 and 1.

        Returns
        -------
        x
            Transformed sample in prior space.
        """
        cos_min = numpy.cos(self.x_min)
        cos_max = numpy.cos(self.x_max)
        x: numpy.float64 = numpy.arccos(cos_min + (cos_max - cos_min) * u)
        return x


@dataclass
class Priors:
    """Prior distributions on gravitational-wave signal parameters."""

    m_1: Prior | float
    m_2: Prior | float
    r: Prior | float
    iota: Prior | float
    t_c: Prior | float
    phi_c: Prior | float
    theta: Prior | float
    phi: Prior | float
    psi: Prior | float

    def transform(self, u: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
        """
        Transform unit hypercube samples to prior space.

        Parameters
        ----------
        u
            Random samples from the unit hypercube.

        Returns
        -------
        x
            Transformed samples in prior space.
        """
        x = u.copy()
        for i, name in enumerate(self.theta_name_sample):
            prior = getattr(self, name)
            assert isinstance(prior, Prior)
            x[i] = prior.transform(u[i])
        return x

    @property
    def n(self) -> int:
        """Number of sampled parameters."""
        return sum(1 for prior in self.__dict__.values() if isinstance(prior, Prior))

    @property
    def theta_name_sample(self) -> list[str]:
        """Names of the sampled parameters."""
        return [name for name, prior in self.__dict__.items() if isinstance(prior, Prior)]

    @property
    def theta_fixed(self) -> dict[str, float]:
        """Fixed parameters."""
        return {name: prior for name, prior in self.__dict__.items() if isinstance(prior, int | float)}

    @property
    def periodic_indices(self) -> None | list[int]:
        """Indices of the periodic parameters."""
        indices: list[int] = []
        for i, name in enumerate(self.theta_name_sample):
            prior = getattr(self, name)
            assert isinstance(prior, Prior)
            if prior.is_periodic:
                indices.append(i)
        return indices or None

    @property
    def reflective_indices(self) -> None | list[int]:
        """Indices of the reflective parameters."""
        indices: list[int] = []
        for i, name in enumerate(self.theta_name_sample):
            prior = getattr(self, name)
            assert isinstance(prior, Prior)
            if prior.is_reflective:
                indices.append(i)
        return indices or None
