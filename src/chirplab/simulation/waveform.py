"""Module for gravitational-waveform models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import overload

import numpy

from chirplab import constants


@dataclass(init=False)
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

    Notes
    -----
    Instead of `m_1` and `m_2`, the chirp mass `m_chirp` (kg) and mass ratio `q` can be provided.
    """

    m_1: float
    m_2: float
    r: float
    iota: float
    t_c: float
    phi_c: float

    @overload
    def __init__(self, *, m_1: float, m_2: float, r: float, iota: float, t_c: float, phi_c: float) -> None: ...
    @overload
    def __init__(self, *, m_chirp: float, q: float, r: float, iota: float, t_c: float, phi_c: float) -> None: ...
    def __init__(
        self,
        *,
        m_1: None | float = None,
        m_2: None | float = None,
        m_chirp: None | float = None,
        q: None | float = None,
        r: float,
        iota: float,
        t_c: float,
        phi_c: float,
    ) -> None:
        if m_1 is not None and m_2 is not None:
            m_chirp = (m_1 * m_2) ** (3 / 5) / (m_1 + m_2) ** (1 / 5)
            q = m_2 / m_1
        elif m_chirp is not None and q is not None:
            m_1 = m_chirp * (1 + q) ** (1 / 5) / q ** (3 / 5)
            m_2 = q * m_1
        else:
            msg = "Either (m_1 and m_2) or (m_chirp and q) must be provided."
            raise ValueError(msg)

        self.m_1 = m_1
        self.m_2 = m_2
        self.r = r
        self.iota = iota
        self.t_c = t_c
        self.phi_c = phi_c

        self.m_chirp = m_chirp
        self.q = q
        self.m_total = self.m_1 + self.m_2


class WaveformModel(ABC):
    """
    Gravitational-waveform model base class.

    Parameters
    ----------
    f_max
        Maximum frequency for the waveform (Hz).
    """

    def __init__(self, f_max: float = constants.INF) -> None:
        self.f_max = f_max

    @abstractmethod
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
        pass


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

    def __init__(self, f_max: float = constants.INF, is_isco_cutoff: bool = True) -> None:
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
            * (1 / constants.PI ** (2 / 3))
            * (constants.C / theta.r)
            * (constants.G * theta.m_chirp / constants.C**3) ** (5 / 6)
            * (1 / f_valid ** (7 / 6))
        )
        psi = (
            2 * constants.PI * f_valid * theta.t_c
            - theta.phi_c
            - constants.PI / 4
            + 3 / 4 * (constants.G * theta.m_chirp / constants.C**3 * 8 * constants.PI * f_valid) ** (-5 / 3)
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
