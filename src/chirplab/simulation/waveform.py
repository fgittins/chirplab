"""Module for gravitational-waveform models."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy

from chirplab import constants

if TYPE_CHECKING:
    from chirplab.simulation import parameters

logger = logging.getLogger(__name__)


class WaveformModel(ABC):
    """
    Gravitational-waveform model.

    Parameters
    ----------
    f_max
        Maximum frequency for the waveform (Hz).
    """

    def __init__(self, f_max: float = constants.INF) -> None:
        self.f_max = f_max

    @abstractmethod
    def calculate_strain_polarisations(
        self, f: numpy.typing.NDArray[numpy.floating], theta: parameters.WaveformParameters
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
        ...


class NewtonianWaveformModel(WaveformModel):
    """
    Gravitational-waveform model using the leading-order Newtonian approximation.

    Parameters
    ----------
    f_max
        Maximum frequency for the waveform (Hz).
    is_isco_cutoff
        Whether to apply a cutoff at the innermost stable circular orbit frequency.

    Notes
    -----
    This implementation follows Eqs. (4.34)-(4.37) of Ref. [1].

    References
    ----------
    [1]  M. Maggiore, Gravitational Waves. Volume 1: Theory and Experiments (Oxford University Press, 2008).
    """

    def __init__(self, f_max: float = constants.INF, is_isco_cutoff: bool = True) -> None:
        super().__init__(f_max)
        self.is_isco_cutoff = is_isco_cutoff
        logger.debug(
            "Initialised NewtonianWaveformModel: f_max=%.1f Hz, is_isco_cutoff=%s",
            f_max,
            is_isco_cutoff,
        )

    def calculate_strain_polarisations(
        self, f: numpy.typing.NDArray[numpy.floating], theta: parameters.WaveformParameters
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
        h_tilde_plus = numpy.zeros_like(f, numpy.complex128)
        h_tilde_cross = numpy.zeros_like(f, numpy.complex128)

        if self.is_isco_cutoff:
            f_isco = calculate_isco_frequency(theta.m_total)
            valid_mask = (f > 0) & (f <= f_isco) & (f <= self.f_max)
            f_valid = f[valid_mask]
        else:
            valid_mask = (f > 0) & (f <= self.f_max)
            f_valid = f[valid_mask]

        t_chirp = constants.G * theta.m_chirp / constants.C**3

        a = (
            (5 / 24) ** (1 / 2)
            * (1 / constants.PI ** (2 / 3))
            * (constants.C / theta.r)
            * t_chirp ** (5 / 6)
            * (1 / f_valid ** (7 / 6))
        )
        psi = (
            2 * constants.PI * f_valid * theta.t_c
            - theta.phi_c
            - constants.PI / 4
            + 3 / 4 * (t_chirp * 8 * constants.PI * f_valid) ** (-5 / 3)
        )

        b = a * numpy.exp(1j * psi)
        cos_iota = numpy.cos(theta.iota)

        h_tilde_plus[valid_mask] = b * (1 + cos_iota**2) / 2
        h_tilde_cross[valid_mask] = 1j * b * cos_iota

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
    return 1 / (6 ** (3 / 2) * constants.PI) * constants.C**3 / (constants.G * m_total)
