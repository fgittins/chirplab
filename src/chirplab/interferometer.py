"""Module for interferometer response to gravitational-wave signals."""

from dataclasses import dataclass
from pathlib import Path

import numpy

from . import waveform

PWD = Path(__file__).parent


@dataclass
class SignalParameters(waveform.WaveformParameters):
    """
    Parameters of the gravitational-wave signal as measured by the detector.

    Parameters
    ----------
    m_1
        Mass of the first component in the binary (kg).
    m_2
        Mass of the second component in the binary (kg).
    r
        Luminosity distance to the binary (m).
    iota
        Inclination angle of the binary (rad).
    t_c
        Coalescence time (s).
    Phi_c
        Coalescence phase (rad).
    theta
        Polar angle of the binary in the detector frame (rad).
    phi
        Azimuthal angle of the binary in the detector frame (rad).
    psi
        Polarisation angle of the binary in the detector frame (rad).
    """

    theta: float
    phi: float
    psi: float


@dataclass
class Grid:
    """
    Grid for signal sampling.

    Parameters
    ----------
    T
        Duration of the signal (s).
    f_s
        Sampling frequency (Hz).
    """

    T: float
    f_s: float

    def __post_init__(self) -> None:
        """Validate parameters after initialisation."""
        if not (self.T * self.f_s).is_integer():
            msg = "The product of T and f_s must be an integer."
            raise ValueError(msg)

        if self.N % 2 != 0:
            msg = "The number of time samples N must be even."
            raise ValueError(msg)

    @property
    def f_max(self) -> float:
        """Maximum (Nyquist) frequency (Hz)."""
        return self.f_s / 2

    @property
    def N(self) -> int:
        """Number of time samples."""
        return round(self.T * self.f_s)

    @property
    def M(self) -> int:
        """Number of frequency samples."""
        return self.N // 2

    @property
    def Delta_t(self) -> float:
        """Time resolution (s)."""
        return 1 / self.f_s

    @property
    def Delta_f(self) -> float:
        """Frequency resolution (Hz)."""
        return 1 / self.T

    @property
    def t(self) -> numpy.typing.NDArray[numpy.float64]:
        """Time array (s)."""
        return numpy.arange(self.N + 1, dtype=numpy.float64) * self.Delta_t

    @property
    def f(self) -> numpy.typing.NDArray[numpy.float64]:
        """Frequency array (Hz)."""
        return numpy.arange(self.M + 1, dtype=numpy.float64) * self.Delta_f

    # TODO: check this method
    def generate_gaussian_noise(self, rng: numpy.random.Generator) -> numpy.typing.NDArray[numpy.complex128]:
        """
        Generate frequency-domain Gaussian noise per unit power spectral density.

        Parameters
        ----------
        rng
            Random number generator.

        Returns
        -------
        m_tilde
            Frequency-domain Gaussian noise per unit power spectral density (Hz^-1/2).
        """
        x = rng.standard_normal((2, self.M + 1))

        sigma = (1 / 4 * self.T) ** (1 / 2)

        m_tilde: numpy.typing.NDArray[numpy.complex128] = (x[0] + 1j * x[1]) * sigma
        m_tilde[0] = 0
        m_tilde[-1] = 0
        return m_tilde


class Interferometer:
    """
    Gravitational-wave interferometer.

    Parameters
    ----------
    grid
        Grid for signal sampling.
    amplitude_spectral_density_file
        File containing the amplitude spectral density data.
    band
        Frequency band over which the interferometer is sensitive (Hz).
    rng
        Random number generator for the noise realisation.
    is_zero_noise
        Whether to use zero noise realisation.
    """

    def __init__(
        self,
        grid: Grid,
        amplitude_spectral_density_file: Path,
        band: tuple[float, float] = (0, numpy.inf),
        rng: None | numpy.random.Generator = None,
        is_zero_noise: bool = False,
    ) -> None:
        self.grid = grid
        f_min_sens, f_max_sens = band

        self.in_bounds_mask = (f_min_sens <= grid.f) & (grid.f <= f_max_sens)
        self.f = grid.f[self.in_bounds_mask]

        f_data, amplitude_spectral_density_data = numpy.loadtxt(
            amplitude_spectral_density_file, numpy.float64, unpack=True
        )
        S_n_data = amplitude_spectral_density_data**2
        self.S_n = numpy.interp(self.f, f_data, S_n_data)

        self.set_data(rng, is_zero_noise)

    def set_data(self, rng: None | numpy.random.Generator = None, is_zero_noise: bool = False) -> None:
        """
        Set the interferometer data with a noise realisation.

        Parameters
        ----------
        rng
            Random number generator.
        is_zero_noise
            Whether to use zero noise realisation.
        """
        # TODO: adapt this method to load real data
        if is_zero_noise:
            n_tilde = numpy.zeros_like(self.f, numpy.complex128)
        else:
            if rng is None:
                rng = numpy.random.default_rng()

            m_tilde = self.grid.generate_gaussian_noise(rng)
            n_tilde = self.S_n ** (1 / 2) * m_tilde[self.in_bounds_mask]

        self.d_tilde = n_tilde.copy()

    @staticmethod
    def calculate_pattern_functions(theta: float, phi: float, psi: float) -> tuple[numpy.float64, numpy.float64]:
        """
        Calculate the interferometer pattern functions.

        Parameters
        ----------
        theta
            Polar angle of the source in the detector frame (rad).
        phi
            Azimuthal angle of the source in the detector frame (rad).
        psi
            Polarization angle of the source in the detector frame (rad).

        Returns
        -------
        F_plus
            Plus pattern function.
        F_cross
            Cross pattern function.
        """
        F_plus_0 = 1 / 2 * (1 + numpy.cos(theta) ** 2) * numpy.cos(2 * phi)
        F_cross_0 = numpy.cos(theta) * numpy.sin(2 * phi)

        F_plus = F_plus_0 * numpy.cos(2 * psi) - F_cross_0 * numpy.sin(2 * psi)
        F_cross = F_plus_0 * numpy.sin(2 * psi) + F_cross_0 * numpy.cos(2 * psi)

        return F_plus, F_cross

    def calculate_strain(
        self, model: waveform.WaveformModel, Theta: SignalParameters
    ) -> numpy.typing.NDArray[numpy.complex128]:
        """
        Calculate the frequency-domain strain response in the interferometer.

        Parameters
        ----------
        model
            Gravitational-waveform model.
        Theta
            Parameters of the gravitational-wave signal.

        Returns
        -------
        h_tilde
            Frequency-domain strain (Hz^-1).
        """
        h_tilde_plus, h_tilde_cross = model.calculate_strain_polarisations(self.f, Theta)
        F_plus, F_cross = self.calculate_pattern_functions(Theta.theta, Theta.phi, Theta.psi)
        return h_tilde_plus * F_plus + h_tilde_cross * F_cross

    def inject_signal(
        self, model: waveform.WaveformModel, Theta: SignalParameters
    ) -> tuple[numpy.typing.NDArray[numpy.complex128], numpy.float64, numpy.complex128]:
        """
        Inject gravitational-wave signal into the interferometer.

        Parameters
        ----------
        model
            Gravitational-waveform model.
        Theta
            Parameters of the gravitational-wave signal.

        Returns
        -------
        h_tilde
            Frequency-domain strain (Hz^-1).
        rho
            Optimal signal-to-noise ratio.
        rho_MF
            Matched-filter signal-to-noise ratio.
        """
        h_tilde = self.calculate_strain(model, Theta)

        self.d_tilde += h_tilde

        rho = self.calculate_inner_product(h_tilde, h_tilde).real ** (1 / 2)
        rho_MF = self.calculate_inner_product(h_tilde, self.d_tilde) / rho

        return h_tilde, rho, rho_MF

    def calculate_inner_product(
        self, a_tilde: numpy.typing.NDArray[numpy.complexfloating], b_tilde: numpy.typing.NDArray[numpy.complexfloating]
    ) -> numpy.complex128:
        """
        Calculate the (complex) noise-weighted inner product of two frequency-domain functions.

        Parameters
        ----------
        a_tilde
            First frequency-domain function (Hz^-1).
        b_tilde
            Second frequency-domain function (Hz^-1).

        Returns
        -------
        inner_product
            Inner product.
        """
        return calculate_inner_product(a_tilde, b_tilde, self.S_n, self.grid.Delta_f)


class LIGO(Interferometer):
    """
    LIGO gravitational-wave interferometer.

    Parameters
    ----------
    grid
        Grid for signal sampling.
    rng
        Random number generator for the noise realisation.
    is_zero_noise
        Whether to use zero noise realisation.
    """

    def __init__(self, grid: Grid, rng: None | numpy.random.Generator = None, is_zero_noise: bool = False) -> None:
        band = (20, 2048)
        super().__init__(grid, PWD / "data/aligo_O4high.txt", band, rng, is_zero_noise)


def calculate_inner_product(
    a_tilde: numpy.typing.NDArray[numpy.complexfloating],
    b_tilde: numpy.typing.NDArray[numpy.complexfloating],
    S_n: numpy.typing.NDArray[numpy.floating],
    Delta_f: float,
) -> numpy.complex128:
    """
    Calculate the (complex) noise-weighted inner product of two frequency-domain functions.

    Parameters
    ----------
    a_tilde
        First frequency-domain function (Hz^-1).
    b_tilde
        Second frequency-domain function (Hz^-1).
    S_n
        Noise power spectral density (Hz^-1).
    Delta_f
        Frequency resolution (Hz).

    Returns
    -------
    inner_product
        Inner product.
    """
    assert a_tilde.size == b_tilde.size == S_n.size, "Input arrays must have the same size."
    integrand = a_tilde.conj() * b_tilde / S_n
    integral = numpy.sum(integrand, dtype=numpy.complex128) * Delta_f
    return 4 * integral
