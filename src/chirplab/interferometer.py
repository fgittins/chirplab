"""Module for interferometer response to gravitational-wave signals."""

from pathlib import Path

import numpy

from . import waveform

PWD = Path(__file__).parent


class Interferometer:
    """Gravitational-wave interferometer class."""

    def __init__(
        self,
        amplitude_spectral_density_file: Path = PWD / "data/aligo_O4high.txt",
        f_min: float = 20,
        f_max: float = 2048,
    ) -> None:
        self.amplitude_spectral_density_file = amplitude_spectral_density_file
        self.f_min = f_min
        self.f_max = f_max

        f: numpy.typing.NDArray[numpy.float64]
        amplitude_spectral_density: numpy.typing.NDArray[numpy.float64]
        f, amplitude_spectral_density = numpy.loadtxt(self.amplitude_spectral_density_file, numpy.float64, unpack=True)

        mask = (self.f_min <= f) & (f <= self.f_max)
        self.f = f[mask]
        self.S_n = amplitude_spectral_density[mask] ** 2

    def interpolate_power_spectral_density(
        self,
        f: numpy.typing.NDArray[numpy.floating],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """
        Interpolate the noise power-spectral density at specified frequencies.

        :param f: Frequencies at which to evaluate (Hz)
        :return S_n: Interpolated values for the noise power-spectral density (Hz^-1)
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

    @staticmethod
    def calculate_inner_product(
        a_tilde: numpy.typing.NDArray[numpy.complexfloating],
        b_tilde: numpy.typing.NDArray[numpy.complexfloating],
        S_n: numpy.typing.NDArray[numpy.floating],
        delta_f: float,
    ) -> numpy.float64:
        """
        Calculate the noise-weighted inner product of two frequency-domain functions.

        :param a_tilde: First frequency-domain function (Hz^-1)
        :param b_tilde: Second frequency-domain function (Hz^-1)
        :param S_n: Noise power-spectral density (Hz^-1)
        :param delta_f: Frequency resolution (Hz)
        :return inner_product: Inner product
        """
        assert a_tilde.shape == b_tilde.shape == S_n.shape, "Input arrays must have the same shape."
        integrand = (a_tilde.conj() * b_tilde) / S_n
        integral = numpy.sum(integrand, dtype=numpy.float64) * delta_f
        return 4 * integral.real

    def inject_signal(self, waveform: waveform.Waveform, is_zero_noise: bool = True) -> None:
        """
        Inject gravitational-wave signal into the interferometer.

        :param waveform: Gravitational waveform
        :param is_zero_noise: Whether to use zero noise
        """
        F_plus, F_cross = self.calculate_pattern_functions(
            waveform.parameters.theta, waveform.parameters.phi, waveform.parameters.psi
        )
        self.h_tilde = F_plus * waveform.h_tilde_plus + F_cross * waveform.h_tilde_cross

        # TODO: add realistic noise realisation
        if is_zero_noise:
            self.n_tilde = numpy.zeros_like(self.h_tilde)
        else:
            msg = "Realistic noise generation is not yet implemented."
            raise NotImplementedError(msg)

        self.d_tilde = self.h_tilde + self.n_tilde

        S_n = self.interpolate_power_spectral_density(waveform.f)

        self.rho = self.calculate_inner_product(self.h_tilde, self.h_tilde, S_n, waveform.delta_f) ** (1 / 2)
        print(f"Optimal signal-to-noise ratio: {self.rho}")
