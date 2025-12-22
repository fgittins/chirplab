"""Module for interferometer response to gravitational-wave signals."""

from pathlib import Path

import numpy

from . import waveform

PWD = Path(__file__).parent


class Interferometer:
    """
    Gravitational-wave interferometer.

    :param amplitude_spectral_density_file: File containing the amplitude spectral density data
    :param f_min_sens: Minimum sensitivity frequency (Hz)
    :param f_max_sens: Maximum sensitivity frequency (Hz)
    """

    def __init__(self, amplitude_spectral_density_file: Path, f_min_sens: float, f_max_sens: float) -> None:
        self.amplitude_spectral_density_file = amplitude_spectral_density_file
        self.f_min_sens = f_min_sens
        self.f_max_sens = f_max_sens

        f, amplitude_spectral_density = numpy.loadtxt(amplitude_spectral_density_file, numpy.float64, unpack=True)

        in_bounds_mask = (f_min_sens <= f) & (f <= f_max_sens)
        self.f_data = f[in_bounds_mask]
        self.S_n_data = amplitude_spectral_density[in_bounds_mask] ** 2

    def interpolate_power_spectral_density(
        self, f: numpy.typing.NDArray[numpy.floating]
    ) -> numpy.typing.NDArray[numpy.floating]:
        """
        Interpolate the noise power spectral density at specified frequencies.

        :param f: Frequency array (Hz)
        :return S_n: Noise power spectral density (Hz^-1)
        """
        S_n: numpy.typing.NDArray[numpy.floating] = numpy.interp(f, self.f_data, self.S_n_data)
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
        self,
        model: waveform.Waveform,
        parameters: waveform.Parameters,
        is_zero_noise: bool = False,
        rng: numpy.random.Generator | None = None,
    ) -> None:
        """
        Inject gravitational-wave signal into the interferometer.

        :param model: Gravitational-waveform model
        :param parameters: Parameters of the gravitational-wave signal
        :param is_zero_noise: Whether to use zero noise
        :param rng: Random number generator for the noise realisation
        """
        self.f_min = model.f_min
        self.f_max = model.f_max
        self.Delta_f = model.Delta_f
        self.f = numpy.arange(self.f_min, self.f_max + self.Delta_f, self.Delta_f)

        # NOTE: zero response outside the sensitivity band
        out_of_bounds_mask = (self.f < self.f_min_sens) | (self.f_max_sens < self.f)

        h_tilde_plus, h_tilde_cross = model.calculate_strain_polarisations(self.f, parameters)
        F_plus, F_cross = self.calculate_pattern_functions(parameters.theta, parameters.phi, parameters.psi)
        self.h_tilde = F_plus * h_tilde_plus + F_cross * h_tilde_cross
        self.h_tilde[out_of_bounds_mask] = 0

        self.S_n = self.interpolate_power_spectral_density(self.f)
        self.S_n[out_of_bounds_mask] = numpy.inf

        if is_zero_noise:
            self.n_tilde = numpy.zeros_like(self.h_tilde)
        else:
            if rng is None:
                rng = numpy.random.default_rng()

            f, white_noise = generate_white_noise(self.f_max, self.Delta_f, rng)
            in_bounds_mask = (self.f_min <= f) & (f <= self.f_max)
            self.n_tilde = self.S_n ** (1 / 2) * white_noise[in_bounds_mask]
            self.n_tilde[out_of_bounds_mask] = 0

        self.d_tilde = self.h_tilde + self.n_tilde

        self.rho = calculate_inner_product(self.h_tilde, self.h_tilde, self.S_n, self.Delta_f).real ** (1 / 2)
        self.rho_MF = calculate_inner_product(self.d_tilde, self.h_tilde, self.S_n, self.Delta_f) / self.rho


class LIGO(Interferometer):
    """LIGO gravitational-wave interferometer."""

    def __init__(self) -> None:
        super().__init__(PWD / "data/aligo_O4high.txt", 20, 2048)


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
    f_max: float, Delta_f: float, rng: numpy.random.Generator
) -> tuple[numpy.typing.NDArray[numpy.floating], numpy.typing.NDArray[numpy.complexfloating]]:
    """
    Generate white noise.

    :param f_max: Maximum frequency (Hz)
    :param Delta_f: Frequency resolution (Hz)
    :param rng: Random number generator
    :return f: Frequency array (Hz)
    :return white_noise: Frequency-domain white noise (Hz^-1)
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
