"""Module for signal sampling grid."""

from dataclasses import dataclass

import numpy


@dataclass(frozen=True, slots=True)
class Grid:
    """
    Grid for signal sampling.

    Parameters
    ----------
    t_d
        Duration of the signal (s).
    f_s
        Sampling frequency (Hz).
    """

    t_d: float
    f_s: float

    def __post_init__(self) -> None:
        """Validate parameters after initialisation."""
        if not (self.t_d * self.f_s).is_integer():
            msg = "The product of t_d and f_s must be an integer."
            raise ValueError(msg)

        if self.n % 2 != 0:
            msg = "The product of t_d and f_s must be even."
            raise ValueError(msg)

    @property
    def f_max(self) -> float:
        """Maximum (Nyquist) frequency (Hz)."""
        return self.f_s / 2

    @property
    def n(self) -> int:
        """Number of time samples."""
        return round(self.t_d * self.f_s)

    @property
    def m(self) -> int:
        """Number of frequency samples."""
        return self.n // 2

    @property
    def delta_t(self) -> float:
        """Time resolution (s)."""
        return 1 / self.f_s

    @property
    def delta_f(self) -> float:
        """Frequency resolution (Hz)."""
        return 1 / self.t_d

    @property
    def t(self) -> numpy.typing.NDArray[numpy.float64]:
        """Time array (s)."""
        return numpy.arange(self.n + 1, dtype=numpy.float64) * self.delta_t

    @property
    def f(self) -> numpy.typing.NDArray[numpy.float64]:
        """Frequency array (Hz)."""
        return numpy.arange(self.m + 1, dtype=numpy.float64) * self.delta_f

    def generate_gaussian_noise(self, rng: numpy.random.Generator) -> numpy.typing.NDArray[numpy.complex128]:
        """
        Generate frequency-domain Gaussian noise per unit amplitude spectral density.

        Parameters
        ----------
        rng
            Random number generator for the noise generation.

        Returns
        -------
        m_tilde
            Frequency-domain Gaussian noise per unit amplitude spectral density (Hz^-1/2).
        """
        x = rng.standard_normal((2, self.m + 1))

        sigma = (1 / 4 * self.t_d) ** (1 / 2)

        m_tilde: numpy.typing.NDArray[numpy.complex128] = (x[0] + 1j * x[1]) * sigma
        m_tilde[0] = 0
        m_tilde[-1] = 0
        return m_tilde
