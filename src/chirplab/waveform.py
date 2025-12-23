"""Module for gravitational-wave waveform generation."""

from dataclasses import dataclass

import numpy

c = 299792458.0
G = 6.6743e-11


@dataclass
class SignalParameters:
    """
    Parameters of the gravitational-wave signal.

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
    """Gravitational-waveform base class."""

    def calculate_strain_polarisations(
        self, f: numpy.typing.NDArray[numpy.floating], Theta: SignalParameters
    ) -> tuple[numpy.typing.NDArray[numpy.complex128], numpy.typing.NDArray[numpy.complex128]]:
        """
        Calculate the frequency-domain strain polarisations.

        Parameters
        ----------
        f
            Frequency array (Hz).
        Theta
            Parameters of the gravitational-wave signal.

        Returns
        -------
        h_tilde_plus
            Frequency-domain plus-polarisation strain (Hz^-1).
        h_tilde_cross
            Frequency-domain cross-polarisation strain (Hz^-1).
        """
        msg = "Waveform models must implement this method."
        raise NotImplementedError(msg)


class NewtonianWaveform(Waveform):
    """Gravitational waveform using the leading-order Newtonian approximation."""

    @staticmethod
    def calculate_strain_polarisations(
        f: numpy.typing.NDArray[numpy.floating], Theta: SignalParameters
    ) -> tuple[numpy.typing.NDArray[numpy.complex128], numpy.typing.NDArray[numpy.complex128]]:
        """
        Calculate the frequency-domain strain polarisations.

        Parameters
        ----------
        f
            Frequency array (Hz).
        Theta
            Parameters of the gravitational-wave signal.

        Returns
        -------
        h_tilde_plus
            Frequency-domain plus-polarisation strain (Hz^-1).
        h_tilde_cross
            Frequency-domain cross-polarisation strain (Hz^-1).

        Notes
        -----
        This implementation uses Eqs. (4.34)-(4.37) of Ref. [1].

        References
        ----------
        [1]  M. Maggiore, Gravitational Waves. Volume 1: Theory and Experiments, (Oxford University Press, 2008).
        """
        h_tilde_plus = numpy.zeros_like(f, numpy.complex128)
        h_tilde_cross = numpy.zeros_like(f, numpy.complex128)

        # NOTE: model does not apply above the innermost stable circular orbit frequency
        f_ISCO = calculate_innermost_stable_circular_orbit_frequency(Theta.M)
        valid_mask = f <= f_ISCO
        f_valid = f[valid_mask]

        A = (
            (5 / 24) ** (1 / 2)
            * (1 / numpy.pi ** (2 / 3))
            * (c / Theta.r)
            * (G * Theta.M_chirp / c**3) ** (5 / 6)
            * (1 / f_valid ** (7 / 6))
        )
        Psi = (
            2 * numpy.pi * f_valid * Theta.t_c
            - Theta.Phi_c
            - numpy.pi / 4
            + 3 / 4 * (G * Theta.M_chirp / c**3 * 8 * numpy.pi * f_valid) ** (-5 / 3)
        )

        h_tilde_plus[valid_mask] = A * numpy.exp(1j * Psi) * (1 + numpy.cos(Theta.iota) ** 2) / 2
        h_tilde_cross[valid_mask] = 1j * A * numpy.exp(1j * Psi) * numpy.cos(Theta.iota)

        return h_tilde_plus, h_tilde_cross


def calculate_innermost_stable_circular_orbit_frequency(M: float) -> float:
    """
    Calculate the gravitational-wave frequency of the innermost stable circular orbit.

    Parameters
    ----------
    M
        Total mass of the binary (kg).

    Returns
    -------
    f_ISCO
        Innermost stable circular orbit frequency (Hz).
    """
    return 1 / (6 ** (3 / 2) * numpy.pi) * c**3 / (G * M)
