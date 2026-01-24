"""Module for gravitational-wave parameters."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astropy import time

from chirplab import constants

if TYPE_CHECKING:
    import numpy


@dataclass(frozen=True, slots=True)
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
    def m_chirp(self) -> float:
        """Chirp mass of the binary (kg)."""
        return float((self.m_1 * self.m_2) ** (3 / 5) / (self.m_1 + self.m_2) ** (1 / 5))

    @property
    def m_total(self) -> float:
        """Total mass of the binary (kg)."""
        return self.m_1 + self.m_2


# TODO: add functionality to initialise with celestial coordinates


@dataclass(frozen=True, slots=True)
class SignalParameters(WaveformParameters):
    """
    Parameters of the gravitational-wave signal in the geocentric frame.

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
    theta
        Polar angle of the binary in the geocentric frame (rad).
    phi
        Azimuthal angle of the binary in the geocentric frame (rad).
    psi
        Polarisation angle of the binary in the geocentric frame (rad).
    """

    theta: float
    phi: float
    psi: float

    @property
    def alpha(self) -> float:
        """Right ascension of the binary in the geocentric frame (rad)."""
        return self.phi + self.gmst

    @property
    def delta(self) -> float:
        """Declination of the binary in the geocentric frame (rad)."""
        return constants.PI / 2 - self.theta

    @property
    def gmst(self) -> float:
        """Greenwich mean sidereal time at coalescence (rad)."""
        t = time.Time(self.t_c, format="gps", scale="utc")
        gmst: numpy.float64 = t.sidereal_time("mean", "greenwich").to_value(unit="rad")
        return gmst
