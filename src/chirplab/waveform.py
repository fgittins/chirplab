"""Module for gravitational-wave waveform generation."""

from dataclasses import dataclass

import numpy

c = 299792458.0
G = 6.6743e-11


@dataclass
class Parameters:
    """
    Parameters of the gravitational-wave signal.

    :param M_chirp: Chirp mass of the binary (kg)
    :param r: Luminosity distance to the source (m)
    :param iota: Inclination angle of the binary (rad)
    :param t_c: Coalescence time (s)
    :param Phi_c: Coalescence phase (rad)
    :param theta: Polar angle of the source in the detector frame (rad)
    :param phi: Azimuthal angle of the source in the detector frame (rad)
    :param psi: Polarization angle of the source in the detector frame (rad)
    """

    M_chirp: float
    r: float
    iota: float
    t_c: float
    Phi_c: float
    theta: float
    phi: float
    psi: float


class Waveform:
    """
    Gravitational waveform class.

    :param f_min: Minimum frequency (Hz)
    :param f_max: Maximum frequency (Hz)
    :param delta_f: Frequency resolution (Hz)
    :param parameters: Parameters of the gravitational-wave signal
    """

    def __init__(self, f_min: float, f_max: float, delta_f: float, parameters: Parameters) -> None:
        self.f_min = f_min
        self.f_max = f_max
        self.delta_f = delta_f
        self.parameters = parameters

        self.f = numpy.arange(f_min, f_max, delta_f, numpy.float64)
        self.h_tilde_plus, self.h_tilde_cross = self.calculate_strain_polarisations(self.f, parameters)

    @staticmethod
    def calculate_strain_polarisations(
        f: numpy.typing.NDArray[numpy.floating], parameters: Parameters
    ) -> tuple[numpy.typing.NDArray[numpy.complexfloating], numpy.typing.NDArray[numpy.complexfloating]]:
        """
        Calculate the frequency-domain strain polarisations.

        :param f: Frequency array (Hz)
        :param parameters: Parameters of the gravitational-wave signal
        :return h_tilde_plus: Frequency-domain plus-polarisation strain (Hz^-1)
        :return h_tilde_cross: Frequency-domain cross-polarisation strain (Hz^-1)
        """
        A = (
            (5 / 24) ** (1 / 2)
            * (1 / numpy.pi ** (2 / 3))
            * (c / parameters.r)
            * (G * parameters.M_chirp / c**3) ** (5 / 6)
            * (1 / f ** (7 / 6))
        )
        Psi = (
            2 * numpy.pi * f * parameters.t_c
            - parameters.Phi_c
            - numpy.pi / 4
            + 3 / 4 * (G * parameters.M_chirp / c**3 * 8 * numpy.pi * f) ** (-5 / 3)
        )

        h_tilde_plus = A * numpy.exp(1j * Psi) * (1 + numpy.cos(parameters.iota) ** 2) / 2
        h_tilde_cross = A * numpy.exp(1j * (Psi + numpy.pi / 2)) * numpy.cos(parameters.iota)

        return h_tilde_plus, h_tilde_cross
