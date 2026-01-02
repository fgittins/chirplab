"""Module for prior distributions."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, override

import numpy

from chirplab import constants
from chirplab.simulation import interferometer

type BOUNDARY_TYPES = None | Literal["periodic", "reflective"]

# TODO: add uniform in comoving volume prior
# TODO: add constraint prior
# TODO: add Gaussian prior


class Prior(ABC):
    """
    Prior base class.

    Parameters
    ----------
    boundary
        Boundary condition for the prior.
    """

    def __init__(self, boundary: BOUNDARY_TYPES = None) -> None:
        is_periodic = is_reflective = False
        if boundary == "periodic":
            is_periodic = True
        elif boundary == "reflective":
            is_reflective = True
        self.is_periodic = is_periodic
        self.is_reflective = is_reflective

    @abstractmethod
    def calculate_ppf(self, q: float) -> float:
        """
        Calculate the percent-point function (inverse cumulative distribution function).

        Parameters
        ----------
        q
            Lower-tail probability.

        Returns
        -------
        x
            Quantile corresponding to the given probability.
        """
        pass

    def sample(self, rng: None | numpy.random.Generator = None) -> float:
        """
        Sample from the prior distribution.

        Parameters
        ----------
        rng
            Random number generator for the sampling.

        Returns
        -------
        x
            Sample drawn from the prior distribution.
        """
        if rng is None:
            rng = numpy.random.default_rng()

        u = rng.uniform(0, 1)
        return self.calculate_ppf(u)


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

    def calculate_ppf(self, q: float) -> float:
        """
        Calculate the percent-point function (inverse cumulative distribution function).

        Parameters
        ----------
        q
            Lower-tail probability.

        Returns
        -------
        x
            Quantile corresponding to the given probability.
        """
        return self.x_min + (self.x_max - self.x_min) * q


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

    def calculate_ppf(self, q: float) -> numpy.float64:
        """
        Calculate the percent-point function (inverse cumulative distribution function).

        Parameters
        ----------
        q
            Lower-tail probability.

        Returns
        -------
        x
            Quantile corresponding to the given probability.
        """
        sin_x_min = numpy.sin(self.x_min)
        sin_x_max = numpy.sin(self.x_max)
        x: numpy.float64 = numpy.arcsin(sin_x_min + (sin_x_max - sin_x_min) * q)
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

    def calculate_ppf(self, q: float) -> numpy.float64:
        """
        Calculate the percent-point function (inverse cumulative distribution function).

        Parameters
        ----------
        q
            Lower-tail probability.

        Returns
        -------
        x
            Quantile corresponding to the given probability.
        """
        cos_x_min = numpy.cos(self.x_min)
        cos_x_max = numpy.cos(self.x_max)
        x: numpy.float64 = numpy.arccos(cos_x_min + (cos_x_max - cos_x_min) * q)
        return x


class Priors:
    """
    Joint prior distribution on the gravitational-wave signal parameters.

    Parameters
    ----------
    m_1
        Prior on the mass of the first component in the binary (kg).
    m_2
        Prior on the mass of the second component in the binary (kg).
    r
        Prior on the luminosity distance to the binary (m).
    iota
        Prior on the inclination angle of the binary (rad).
    t_c
        Prior on the coalescence time (s).
    phi_c
        Prior on the coalescence phase (rad).
    theta
        Prior on the polar angle of the binary in the detector frame (rad).
    phi
        Prior on the azimuthal angle of the binary in the detector frame (rad).
    psi
        Prior on the polarisation angle of the binary in the detector frame (rad).
    """

    def __init__(
        self,
        m_1: Prior | float,
        m_2: Prior | float,
        r: Prior | float,
        iota: Prior | float,
        t_c: Prior | float,
        phi_c: Prior | float,
        theta: Prior | float,
        phi: Prior | float,
        psi: Prior | float,
    ) -> None:
        self.m_1 = m_1
        self.m_2 = m_2
        self.r = r
        self.iota = iota
        self.t_c = t_c
        self.phi_c = phi_c
        self.theta = theta
        self.phi = phi
        self.psi = psi

        self.theta_name_sample = [name for name, prior in self.__dict__.items() if isinstance(prior, Prior)]

    def calculate_ppf(self, q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
        """
        Calculate the percent-point function (inverse cumulative distribution function) of each prior distribution.

        Parameters
        ----------
        q
            Lower-tail probabilities.

        Returns
        -------
        x
            Quantiles corresponding to the given probabilities.
        """
        x = q.copy()
        for i, name in enumerate(self.theta_name_sample):
            prior = getattr(self, name)
            assert isinstance(prior, Prior)
            x[i] = prior.calculate_ppf(q[i])
        return x

    def sample(self, rng: None | numpy.random.Generator = None) -> interferometer.SignalParameters:
        """
        Sample from the joint prior distribution.

        Parameters
        ----------
        rng
            Random number generator for the sampling.

        Returns
        -------
        theta
            Sampled signal parameters.
        """
        theta_dict: dict[str, float] = {}
        for name, prior in self.__dict__.items():
            if isinstance(prior, Prior):
                theta_dict[name] = prior.sample(rng)
            elif isinstance(prior, int | float):
                theta_dict[name] = prior
        return interferometer.SignalParameters(**theta_dict)

    @property
    def n(self) -> int:
        """Number of sampled parameters."""
        return sum(1 for prior in self.__dict__.values() if isinstance(prior, Prior))

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
