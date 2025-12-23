"""Module for interferometer response to gravitational-wave signals."""

from dataclasses import dataclass
from pathlib import Path

import numpy

from . import waveform

PWD = Path(__file__).parent


@dataclass
class Grid:
    """
    Frequency grid for the data.

    Parameters
    ----------
    f_min
        Minimum frequency (Hz).
    f_max
        Maximum frequency (Hz).
    Delta_f
        Frequency resolution (Hz).
    """

    f_min: float
    f_max: float
    Delta_f: float


class Interferometer:
    """
    Gravitational-wave interferometer.

    Parameters
    ----------
    amplitude_spectral_density_file
        File containing the amplitude spectral density data.
    grid
        Frequency grid for the data.
    rng
        Random number generator for the noise realisation.
    is_zero_noise
        Whether to use zero noise realisation.
    """

    def __init__(
        self,
        amplitude_spectral_density_file: Path,
        grid: Grid,
        rng: None | numpy.random.Generator = None,
        is_zero_noise: bool = False,
    ) -> None:
        self.amplitude_spectral_density_file = amplitude_spectral_density_file
        self.f_min = grid.f_min
        self.f_max = grid.f_max
        self.Delta_f = grid.Delta_f
        self.rng = rng
        self.is_zero_noise = is_zero_noise

        self.f = numpy.arange(self.f_min, self.f_max + self.Delta_f, self.Delta_f)

        f_data, amplitude_spectral_density_data = numpy.loadtxt(
            amplitude_spectral_density_file, numpy.float64, unpack=True
        )
        in_bounds_mask = (self.f_min <= f_data) & (f_data <= self.f_max)
        f_data = f_data[in_bounds_mask]
        S_n_data = amplitude_spectral_density_data[in_bounds_mask] ** 2

        self.S_n = numpy.interp(self.f, f_data, S_n_data)

        if is_zero_noise:
            self.n_tilde = numpy.zeros_like(self.f, numpy.complex128)
        else:
            if rng is None:
                rng = numpy.random.default_rng()

            f_noise, white_noise = generate_white_noise(self.f_max, self.Delta_f, rng)
            in_bounds_mask = self.f_min <= f_noise
            assert numpy.all(self.f == f_noise[in_bounds_mask]), "Frequency arrays do not match."

            self.n_tilde = self.S_n ** (1 / 2) * white_noise[in_bounds_mask]

        self.d_tilde = self.n_tilde.copy()

    @staticmethod
    def calculate_pattern_functions(theta: float, phi: float, psi: float) -> tuple[numpy.floating, numpy.floating]:
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

    def inject(self, model: waveform.Waveform, theta: waveform.SignalParameters) -> None:
        """
        Inject gravitational-wave signal into the interferometer.

        Parameters
        ----------
        model
            Gravitational-waveform model
        theta
            Parameters of the gravitational-wave signal
        """
        h_tilde_plus, h_tilde_cross = model.calculate_strain_polarisations(self.f, theta)
        F_plus, F_cross = self.calculate_pattern_functions(theta.theta, theta.phi, theta.psi)
        self.h_tilde = F_plus * h_tilde_plus + F_cross * h_tilde_cross

        self.d_tilde += self.h_tilde

        self.rho = calculate_inner_product(self.h_tilde, self.h_tilde, self.S_n, self.Delta_f).real ** (1 / 2)
        self.rho_MF = calculate_inner_product(self.h_tilde, self.d_tilde, self.S_n, self.Delta_f) / self.rho


class LIGO(Interferometer):
    """
    LIGO gravitational-wave interferometer.

    Parameters
    ----------
    Delta_f
        Frequency resolution (Hz).
    rng
        Random number generator for the noise realisation.
    is_zero_noise
        Whether to use zero noise realisation.
    """

    def __init__(self, Delta_f: float, rng: None | numpy.random.Generator = None, is_zero_noise: bool = False) -> None:
        grid = Grid(f_min=20, f_max=2048, Delta_f=Delta_f)
        super().__init__(PWD / "data/aligo_O4high.txt", grid, rng, is_zero_noise)


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


def generate_white_noise(
    f_max: float, Delta_f: float, rng: numpy.random.Generator
) -> tuple[numpy.typing.NDArray[numpy.floating], numpy.typing.NDArray[numpy.complexfloating]]:
    """
    Generate white noise.

    Parameters
    ----------
    f_max
        Maximum frequency (Hz).
    Delta_f
        Frequency resolution (Hz).
    rng
        Random number generator.

    Returns
    -------
    f
        Frequency array (Hz).
    white_noise
        Frequency-domain white noise (Hz^-1).
    """
    N = round(2 * f_max / Delta_f)
    f = numpy.arange(0, f_max + Delta_f, Delta_f)

    sigma = 1 / 2 * (1 / Delta_f) ** (1 / 2)
    re, im = rng.normal(0, sigma, (2, f.size))

    white_noise = re + 1j * im
    white_noise[0] = 0
    if N % 2 == 0:  # NOTE: No Nyquist frequency component for odd N
        white_noise[-1] = 0

    return f, white_noise
