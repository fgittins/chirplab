"""Module for gravitational-wave waveform generation."""

from dataclasses import dataclass

import numpy

c = 299792458.0
G = 6.6743e-11


@dataclass
class Parameters:
    """
    Parameters of the gravitational-wave signal.

    :param m_1: Mass of the first component in the binary (kg)
    :param m_2: Mass of the second component in the binary (kg)
    :param r: Luminosity distance to the binary (m)
    :param iota: Inclination angle of the binary (rad)
    :param t_c: Coalescence time (s)
    :param Phi_c: Coalescence phase (rad)
    :param theta: Polar angle of the binary in the detector frame (rad)
    :param phi: Azimuthal angle of the binary in the detector frame (rad)
    :param psi: Polarisation angle of the binary in the detector frame (rad)
    """

    m_1: float
    m_2: float
    r: float
    iota: float
    t_c: float
    Phi_c: float

    # NOTE: pattern function angles
    theta: float
    phi: float
    psi: float

    @property
    def M(self) -> float:
        """Total mass of the binary (kg)."""
        return self.m_1 + self.m_2

    @property
    def M_chirp(self) -> float:
        """Chirp mass of the binary (kg)."""
        M_chirp: float = (self.m_1 * self.m_2) ** (3 / 5) / (self.m_1 + self.m_2) ** (1 / 5)
        return M_chirp


class Waveform:
    """
    Gravitational waveform.

    :param f_min: Minimum frequency (Hz)
    :param f_max: Maximum frequency (Hz)
    :param Delta_f: Frequency resolution (Hz)
    :param parameters: Parameters of the gravitational-wave signal
    """

    def __init__(self, f_min: float, f_max: float, Delta_f: float, parameters: Parameters) -> None:
        self.f_min = f_min
        self.Delta_f = Delta_f
        self.parameters = parameters

        f_max = min(f_max, calculate_innermost_stable_circular_orbit_frequency(parameters.M))
        self.f_max = f_max

        self.f = numpy.arange(f_min, f_max, Delta_f, numpy.float64)
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


def calculate_innermost_stable_circular_orbit_frequency(M: float) -> float:
    """
    Calculate the innermost stable circular orbit frequency.

    :param M: Total mass of the binary (kg)
    :return f_ISCO: innermost stable circular orbit frequency (Hz)
    """
    return 1 / (6 ** (3 / 2) * numpy.pi) * c**3 / (G * M)
