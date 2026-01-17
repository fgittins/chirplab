"""Module for gravitational-wave signal parameters."""

from dataclasses import dataclass
from typing import Self

from chirplab import constants

# TODO: set gmst properly


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
        m_chirp: float = (self.m_1 * self.m_2) ** (3 / 5) / (self.m_1 + self.m_2) ** (1 / 5)
        return m_chirp

    @property
    def m_total(self) -> float:
        """Total mass of the binary (kg)."""
        return self.m_1 + self.m_2


@dataclass(frozen=True, slots=True)
class DetectorAngles:
    """
    Angles defining the source location and orientation in the geocentric frame.

    Parameters
    ----------
    alpha
        Right ascension of the binary in the geocentric frame (rad).
    delta
        Declination of the binary in the geocentric frame (rad).
    psi
        Polarisation angle of the binary in the geocentric frame (rad).
    gmst
        Greenwich mean sidereal time (rad).
    """

    alpha: float
    delta: float
    psi: float
    gmst: float = 0

    @property
    def theta(self) -> float:
        """Polar angle of the binary in the detector frame (rad)."""
        return constants.PI / 2 - self.delta

    @property
    def phi(self) -> float:
        """Azimuthal angle of the binary in the detector frame (rad)."""
        return self.alpha - self.gmst


@dataclass(frozen=True, slots=True)
class SignalParameters:
    """
    Parameters of the gravitational-wave signal as measured by the detector.

    Parameters
    ----------
    waveform_parameters
        Parameters of the gravitational waveform.
    detector_angles
        Angles defining the source location and orientation in the geocentric frame.
    """

    waveform_parameters: WaveformParameters
    detector_angles: DetectorAngles

    @classmethod
    def from_dict(cls, theta_dict: dict[str, float]) -> Self:
        """
        Create SignalParameters from a dictionary.

        Parameters
        ----------
        theta_dict
            Dictionary containing the signal parameters.

        Returns
        -------
        theta
            Parameters of the gravitational-wave signal as measured by the detector.
        """
        waveform_parameters = WaveformParameters(
            theta_dict["m_1"],
            theta_dict["m_2"],
            theta_dict["r"],
            theta_dict["iota"],
            theta_dict["t_c"],
            theta_dict["phi_c"],
        )
        detector_angles = DetectorAngles(theta_dict["alpha"], theta_dict["delta"], theta_dict["psi"])
        return cls(waveform_parameters, detector_angles)
