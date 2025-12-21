"""Module for interferometer response to gravitational-wave signals."""

from pathlib import Path

import numpy

from . import waveform

PWD = Path(__file__).parent


class Interferometer:
    """
    Gravitational-wave interferometer.

    :param amplitude_spectral_density_file: File containing the amplitude spectral density data
    :param f_min: Minimum frequency (Hz)
    :param f_max: Maximum frequency (Hz)
    """

    def __init__(self, amplitude_spectral_density_file: Path, f_min: float, f_max: float) -> None:
        self.amplitude_spectral_density_file = amplitude_spectral_density_file
        self.f_min = f_min
        self.f_max = f_max

        f: numpy.typing.NDArray[numpy.float64]
        amplitude_spectral_density: numpy.typing.NDArray[numpy.float64]
        f, amplitude_spectral_density = numpy.loadtxt(amplitude_spectral_density_file, numpy.float64, unpack=True)

        mask = (f_min <= f) & (f <= f_max)
        self.f = f[mask]
        self.S_n = amplitude_spectral_density[mask] ** 2

    def interpolate_power_spectral_density(
        self, f: numpy.typing.NDArray[numpy.floating]
    ) -> numpy.typing.NDArray[numpy.floating]:
        """
        Interpolate the noise power spectral density at specified frequencies.

        :param f: Frequency array (Hz)
        :return S_n: Noise power spectral density (Hz^-1)
        """
        S_n: numpy.typing.NDArray[numpy.floating] = numpy.interp(f, self.f, self.S_n)
        return S_n

    @staticmethod
    def calculate_pattern_functions(theta: float, phi: float, psi: float) -> tuple[numpy.floating, numpy.floating]:
        """
        Calculate the interferometer pattern functions.

        :param theta: Polar angle of the source in the detector frame (rad)
        :param phi: Azimuthal angle of the source in the detector frame (rad)
        :param psi: Polarization angle of the source in the detector frame (rad)
        :return F_plus: Plus pattern function
        :return F_cross: Cross pattern function
        """
        F_plus_0 = 1 / 2 * (1 + numpy.cos(theta) ** 2) * numpy.cos(2 * phi)
        F_cross_0 = numpy.cos(theta) * numpy.sin(2 * phi)

        F_plus = F_plus_0 * numpy.cos(2 * psi) - F_cross_0 * numpy.sin(2 * psi)
        F_cross = F_plus_0 * numpy.sin(2 * psi) + F_cross_0 * numpy.cos(2 * psi)

        return F_plus, F_cross

    def inject(
        self, signal: waveform.Waveform, is_zero_noise: bool = False, rng: numpy.random.Generator | None = None
    ) -> None:
        """
        Inject gravitational-wave signal into the interferometer.

        :param signal: Gravitational-wave signal
        :param is_zero_noise: Whether to use zero noise
        :param rng: Random number generator for the noise realisation
        """
        F_plus, F_cross = self.calculate_pattern_functions(
            signal.parameters.theta, signal.parameters.phi, signal.parameters.psi
        )
        self.h_tilde = F_plus * signal.h_tilde_plus + F_cross * signal.h_tilde_cross

        S_n = self.interpolate_power_spectral_density(signal.f)

        if is_zero_noise:
            self.n_tilde = numpy.zeros_like(self.h_tilde)
        else:
            if rng is None:
                rng = numpy.random.default_rng()

            self.n_tilde = S_n ** (1 / 2) * generate_white_noise(signal.f, rng)

        self.d_tilde = self.h_tilde + self.n_tilde

        self.rho = calculate_inner_product(self.h_tilde, self.h_tilde, S_n, signal.Delta_f).real ** (1 / 2)
        self.rho_MF = calculate_inner_product(self.d_tilde, self.h_tilde, S_n, signal.Delta_f) / self.rho


def calculate_inner_product(
    a_tilde: numpy.typing.NDArray[numpy.complexfloating],
    b_tilde: numpy.typing.NDArray[numpy.complexfloating],
    S_n: numpy.typing.NDArray[numpy.floating],
    Delta_f: float,
) -> numpy.complex128:
    """
    Calculate the (complex) noise-weighted inner product of two frequency-domain functions.

    :param a_tilde: First frequency-domain function (Hz^-1)
    :param b_tilde: Second frequency-domain function (Hz^-1)
    :param S_n: Noise power spectral density (Hz^-1)
    :param Delta_f: Frequency resolution (Hz)
    :return inner_product: Inner product
    """
    assert a_tilde.shape == b_tilde.shape == S_n.shape, "Input arrays must have the same shape."
    integrand = (a_tilde.conj() * b_tilde) / S_n
    integral = numpy.sum(integrand, dtype=numpy.complex128) * Delta_f
    return 4 * integral


def generate_white_noise(
    f: numpy.typing.NDArray[numpy.floating], rng: numpy.random.Generator
) -> numpy.typing.NDArray[numpy.complexfloating]:
    """
    Generate white noise.

    :param f: Frequency array (Hz)
    :param rng: Random number generator
    :return n_tilde: Frequency-domain white noise (Hz^-1)
    """
    diffs = numpy.diff(f)
    Delta_f: numpy.floating = diffs[0]
    assert numpy.all(diffs == Delta_f), "Frequency array must have uniform spacing."

    sigma = 1 / 2 * (1 / Delta_f) ** (1 / 2)

    re: numpy.typing.NDArray[numpy.floating]
    im: numpy.typing.NDArray[numpy.floating]
    re, im = rng.normal(0, sigma, (2, f.size))

    n_tilde = re + 1j * im
    n_tilde[0] = 0
    return n_tilde


class LIGO(Interferometer):
    """LIGO gravitational-wave interferometer."""

    def __init__(self) -> None:
        super().__init__(PWD / "data/aligo_O4high.txt", 20, 2048)
