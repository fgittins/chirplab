"""Module for prior distributions."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy
from astropy import cosmology, units
from scipy import integrate, interpolate, special

from chirplab import constants
from chirplab.simulation import interferometer

type BOUNDARY_TYPES = None | Literal["periodic", "reflective"]


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

        q = rng.uniform(0, 1)
        return self.calculate_ppf(q)


class DeltaFunction(Prior):
    """
    Delta-function prior.

    Parameters
    ----------
    x_peak
        Value of the delta function.
    """

    def __init__(self, x_peak: float) -> None:
        super().__init__(None)
        self.x_peak = x_peak

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
        return self.x_peak


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


class Gaussian(Prior):
    """
    Gaussian prior.

    Parameters
    ----------
    mu
        Mean of the prior.
    sigma
        Standard deviation of the prior.
    boundary
        Boundary condition for the prior.
    """

    def __init__(self, mu: float = 0, sigma: float = 1, boundary: BOUNDARY_TYPES = None) -> None:
        super().__init__(boundary)
        self.mu = mu
        self.sigma = sigma

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
        x: numpy.float64 = self.mu + self.sigma * special.erfinv(2 * q - 1) * 2 ** (1 / 2)
        return x


class UniformComovingVolume(Prior):
    """
    Uniform in comoving volume prior for the luminosity distance.

    Parameters
    ----------
    r_min
        Minimum luminosity distance (m).
    r_max
        Maximum luminosity distance (m).
    boundary
        Boundary condition for the prior.

    Notes
    -----
    This prior assumes a flat Lambda-CDM cosmology with Planck 2018 parameters [1].

    References
    ----------
    [1]  <https://docs.astropy.org/en/stable/cosmology/realizations.html>.
    """

    def __init__(self, r_min: float, r_max: float, boundary: BOUNDARY_TYPES = None) -> None:
        super().__init__(boundary)
        self.r_min = r_min
        self.r_max = r_max

        cosmo = cosmology.Planck18
        z_min = cosmology.z_at_value(cosmo.luminosity_distance, r_min * units.m).value
        z_max = cosmology.z_at_value(cosmo.luminosity_distance, r_max * units.m).value

        z_array = numpy.linspace(z_min, z_max, 1_000)
        pdf_array = cosmo.differential_comoving_volume(z_array).value

        r_array = cosmo.luminosity_distance(z_array).value * 1e6 * constants.PC
        dr_dz_array = numpy.gradient(r_array, z_array)
        pdf_array /= dr_dz_array

        pdf_array /= numpy.trapezoid(pdf_array, r_array)
        cdf_array = integrate.cumulative_trapezoid(pdf_array, r_array, initial=0)

        self.ppf_function = interpolate.CubicSpline(cdf_array, r_array)

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
        return numpy.float64(self.ppf_function(q))


class Priors:
    """
    Joint prior distribution.

    Parameters
    ----------
    priors
        Dictionary of parameter names to their prior distributions.
    """

    def __init__(self, priors: dict[str, Prior]) -> None:
        self.names = tuple(priors.keys())
        self.priors = tuple(priors.values())

        i: int = 0

        sample_names: list[str] = []
        sample_priors: list[Prior] = []

        theta_fixed: dict[str, float] = {}

        periodic_indices: list[int] = []
        reflective_indices: list[int] = []

        for name, prior in zip(self.names, self.priors, strict=False):
            if isinstance(prior, DeltaFunction):
                theta_fixed[name] = prior.x_peak
            else:
                sample_names.append(name)
                sample_priors.append(prior)
                if prior.is_periodic:
                    periodic_indices.append(i)
                if prior.is_reflective:
                    reflective_indices.append(i)
                i += 1

        self.n_dim = i

        self.sample_names = tuple(sample_names)
        self.sample_priors = tuple(sample_priors)

        self.theta_fixed = theta_fixed

        self.periodic_indices = periodic_indices if periodic_indices else None
        self.reflective_indices = reflective_indices if reflective_indices else None

    def prior_transform(self, q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
        """
        Transform unit-hypercube samples to samples from the joint prior distribution.

        Parameters
        ----------
        q
            Unit-hypercube samples.

        Returns
        -------
        x
            Samples from the joint prior distribution.

        Notes
        -----
        For independent parameters, the joint prior transform is given by the individual percent-point functions.

        Fixed parameters are not included in the transformation.
        """
        x = q.copy()
        for i, prior in enumerate(self.sample_priors):
            x[i] = prior.calculate_ppf(q[i])
        return x

    def sample(self, rng: None | numpy.random.Generator = None) -> dict[str, float]:
        """
        Sample from the joint prior distribution.

        Parameters
        ----------
        rng
            Random number generator for the sampling.

        Returns
        -------
        theta
            Sampled parameters.
        """
        return {name: prior.sample(rng) for name, prior in zip(self.names, self.priors, strict=False)}
