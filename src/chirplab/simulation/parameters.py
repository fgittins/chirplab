"""Module for gravitational-wave parameters."""

from dataclasses import dataclass

# TODO: introduce celestial coordinates


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
        msg = "Right ascension not implemented."
        raise NotImplementedError(msg)

    @property
    def delta(self) -> float:
        """Declination of the binary in the geocentric frame (rad)."""
        msg = "Declination not implemented."
        raise NotImplementedError(msg)

    @property
    def gmst(self) -> float:
        """Greenwich mean sidereal time at coalescence (rad)."""
        msg = "Greenwich mean sidereal time not implemented."
        raise NotImplementedError(msg)
