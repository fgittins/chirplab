"""Module for gravitational-waveform models."""

from dataclasses import dataclass
from typing import Final

import numpy

C: Final[float] = 299792458.0
G: Final[float] = 6.6743e-11
M_SUN: Final[float] = 1.988409870698051e30
PC: Final[float] = 3.085677581491367e16


@dataclass
class WaveformParameters:
    """
    Parameters of the gravitational waveform.

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
    phi_c
        Coalescence phase (rad).
    """

    m_1: float
    m_2: float
    r: float
    iota: float
    t_c: float
    phi_c: float

    @property
    def m_total(self) -> float:
        """Total mass of the binary (kg)."""
        return self.m_1 + self.m_2

    @property
    def m_chirp(self) -> float:
        """Chirp mass of the binary (kg)."""
        m_chirp: float = (self.m_1 * self.m_2) ** (3 / 5) / (self.m_1 + self.m_2) ** (1 / 5)
        return m_chirp


class WaveformModel:
    """
    Gravitational-waveform model base class.

    Parameters
    ----------
    f_max
        Maximum frequency for the waveform (Hz).
    """

    def __init__(self, f_max: float = numpy.inf) -> None:
        self.f_max = f_max

    def calculate_strain_polarisations(
        self, f: numpy.typing.NDArray[numpy.floating], theta: WaveformParameters
    ) -> tuple[numpy.typing.NDArray[numpy.complex128], numpy.typing.NDArray[numpy.complex128]]:
        """
        Calculate the frequency-domain strain polarisations.

        Parameters
        ----------
        f
            Frequency array (Hz).
        theta
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


class NewtonianWaveformModel(WaveformModel):
    """
    Gravitational-waveform model using the leading-order Newtonian approximation.

    Parameters
    ----------
    f_max
        Maximum frequency for the waveform (Hz).
    is_isco_cutoff
        Whether to apply a cutoff at the innermost stable circular orbit frequency.
    """

    def __init__(self, f_max: float = numpy.inf, is_isco_cutoff: bool = True) -> None:
        super().__init__(f_max)
        self.is_isco_cutoff = is_isco_cutoff

    def calculate_strain_polarisations(
        self, f: numpy.typing.NDArray[numpy.floating], theta: WaveformParameters
    ) -> tuple[numpy.typing.NDArray[numpy.complex128], numpy.typing.NDArray[numpy.complex128]]:
        """
        Calculate the frequency-domain strain polarisations.

        Parameters
        ----------
        f
            Frequency array (Hz).
        theta
            Parameters of the gravitational-wave signal.

        Returns
        -------
        h_tilde_plus
            Frequency-domain plus-polarisation strain (Hz^-1).
        h_tilde_cross
            Frequency-domain cross-polarisation strain (Hz^-1).

        Notes
        -----
        This implementation follows Eqs. (4.34)-(4.37) of Ref. [1].

        References
        ----------
        [1]  M. Maggiore, Gravitational Waves. Volume 1: Theory and Experiments (Oxford University Press, 2008).
        """
        h_tilde_plus = numpy.zeros_like(f, numpy.complex128)
        h_tilde_cross = numpy.zeros_like(f, numpy.complex128)

        if self.is_isco_cutoff:
            f_isco = calculate_isco_frequency(theta.m_total)
            valid_mask = (f > 0) & (f <= f_isco) & (f <= self.f_max)
            f_valid = f[valid_mask]
        else:
            valid_mask = (f > 0) & (f <= self.f_max)
            f_valid = f[valid_mask]

        a = (
            (5 / 24) ** (1 / 2)
            * (1 / numpy.pi ** (2 / 3))
            * (C / theta.r)
            * (G * theta.m_chirp / C**3) ** (5 / 6)
            * (1 / f_valid ** (7 / 6))
        )
        psi = (
            2 * numpy.pi * f_valid * theta.t_c
            - theta.phi_c
            - numpy.pi / 4
            + 3 / 4 * (G * theta.m_chirp / C**3 * 8 * numpy.pi * f_valid) ** (-5 / 3)
        )

        h_tilde_plus[valid_mask] = a * numpy.exp(1j * psi) * (1 + numpy.cos(theta.iota) ** 2) / 2
        h_tilde_cross[valid_mask] = 1j * a * numpy.exp(1j * psi) * numpy.cos(theta.iota)

        return h_tilde_plus, h_tilde_cross


def calculate_isco_frequency(m_total: float) -> float:
    """
    Calculate the gravitational-wave frequency of the innermost stable circular orbit.

    Parameters
    ----------
    m_total
        Total mass of the binary (kg).

    Returns
    -------
    f_isco
        Innermost stable circular orbit frequency (Hz).
    """
    return 1 / (6 ** (3 / 2) * numpy.pi) * C**3 / (G * m_total)
