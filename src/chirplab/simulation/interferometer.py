"""Module for interferometer response to gravitational-wave signals."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy

from chirplab import constants
from chirplab.simulation import waveform

# TODO: add multiple interferometers
# TODO: revisit calculation of polarisation tensor and conventions
# TODO: set gmst properly

GEOCENTRE: Final[numpy.typing.NDArray[numpy.float64]] = numpy.array([0, 0, 0], dtype=numpy.float64)
D: Final[numpy.typing.NDArray[numpy.float64]] = (
    1 / 2 * numpy.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=numpy.float64)
)


@dataclass(frozen=True, slots=True)
class DetectorAngles:
    """
    Angles defining the source location and orientation in the detector frame.

    Parameters
    ----------
    alpha
        Right ascension of the binary in the detector frame (rad).
    delta
        Declination of the binary in the detector frame (rad).
    psi
        Polarisation angle of the binary in the detector frame (rad).
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
        Angles defining the source location and orientation in the detector frame.
    """

    waveform_parameters: waveform.WaveformParameters
    detector_angles: DetectorAngles

    @classmethod
    def from_dict(cls, theta_dict: dict[str, float]) -> SignalParameters:
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
        waveform_parameters = waveform.WaveformParameters(
            theta_dict["m_1"],
            theta_dict["m_2"],
            theta_dict["r"],
            theta_dict["iota"],
            theta_dict["t_c"],
            theta_dict["phi_c"],
        )
        detector_angles = DetectorAngles(theta_dict["alpha"], theta_dict["delta"], theta_dict["psi"])
        return cls(waveform_parameters, detector_angles)


@dataclass
class Grid:
    """
    Grid for signal sampling.

    Parameters
    ----------
    t_d
        Duration of the signal (s).
    f_s
        Sampling frequency (Hz).
    """

    t_d: float
    f_s: float

    def __post_init__(self) -> None:
        """Validate parameters after initialisation."""
        if not (self.t_d * self.f_s).is_integer():
            msg = "The product of t_d and f_s must be an integer."
            raise ValueError(msg)

        if self.n % 2 != 0:
            msg = "The product of t_d and f_s must be even."
            raise ValueError(msg)

    @property
    def f_max(self) -> float:
        """Maximum (Nyquist) frequency (Hz)."""
        return self.f_s / 2

    @property
    def n(self) -> int:
        """Number of time samples."""
        return round(self.t_d * self.f_s)

    @property
    def m(self) -> int:
        """Number of frequency samples."""
        return self.n // 2

    @property
    def delta_t(self) -> float:
        """Time resolution (s)."""
        return 1 / self.f_s

    @property
    def delta_f(self) -> float:
        """Frequency resolution (Hz)."""
        return 1 / self.t_d

    @property
    def t(self) -> numpy.typing.NDArray[numpy.float64]:
        """Time array (s)."""
        return numpy.arange(self.n + 1, dtype=numpy.float64) * self.delta_t

    @property
    def f(self) -> numpy.typing.NDArray[numpy.float64]:
        """Frequency array (Hz)."""
        return numpy.arange(self.m + 1, dtype=numpy.float64) * self.delta_f

    def generate_gaussian_noise(self, rng: numpy.random.Generator) -> numpy.typing.NDArray[numpy.complex128]:
        """
        Generate frequency-domain Gaussian noise per unit amplitude spectral density.

        Parameters
        ----------
        rng
            Random number generator for the noise generation.

        Returns
        -------
        m_tilde
            Frequency-domain Gaussian noise per unit amplitude spectral density (Hz^-1/2).
        """
        x = rng.standard_normal((2, self.m + 1), dtype=numpy.float64)

        sigma = (1 / 4 * self.t_d) ** (1 / 2)

        m_tilde: numpy.typing.NDArray[numpy.complex128] = (x[0] + 1j * x[1]) * sigma
        m_tilde[0] = 0
        m_tilde[-1] = 0
        return m_tilde


class Interferometer:
    """
    Gravitational-wave interferometer.

    Parameters
    ----------
    grid
        Grid for signal sampling.
    amplitude_spectral_density_file
        File containing the amplitude spectral density data.
    f_min
        Minimum sensitivity frequency (Hz).
    f_max
        Maximum sensitivity frequency (Hz).
    d
        Response tensor in the geocentric frame.
    x
        Position vector in the geocentric frame (m).
    rng
        Random number generator for the noise realisation.
    is_zero_noise
        Whether to use zero noise realisation.
    """

    def __init__(
        self,
        grid: Grid,
        amplitude_spectral_density_file: Path,
        f_min: float = 0,
        f_max: float = constants.INF,
        x: numpy.typing.NDArray[numpy.floating] = GEOCENTRE,
        d: numpy.typing.NDArray[numpy.floating] = D,
        rng: None | numpy.random.Generator = None,
        is_zero_noise: bool = False,
    ) -> None:
        self.grid = grid
        self.x = x
        self.d = d

        self.in_bounds_mask = (f_min <= grid.f) & (grid.f <= f_max)
        self.f = grid.f[self.in_bounds_mask]

        f, amplitude_spectral_density = numpy.loadtxt(amplitude_spectral_density_file, numpy.float64, unpack=True)
        self.s_n: numpy.typing.NDArray[numpy.float64] = numpy.interp(self.f, f, amplitude_spectral_density**2)

        self.set_data(rng, is_zero_noise)

    def set_data(self, rng: None | numpy.random.Generator = None, is_zero_noise: bool = False) -> None:
        """
        Set the interferometer data with a noise realisation.

        Parameters
        ----------
        rng
            Random number generator for the noise realisation.
        is_zero_noise
            Whether to use zero noise realisation.
        """
        # TODO: adapt this method to load real data
        if is_zero_noise:
            n_tilde = numpy.zeros_like(self.f, numpy.complex128)
        else:
            if rng is None:
                rng = numpy.random.default_rng()

            m_tilde = self.grid.generate_gaussian_noise(rng)
            n_tilde = self.s_n ** (1 / 2) * m_tilde[self.in_bounds_mask]

        self.d_tilde = n_tilde.copy()

    def calculate_pattern_functions(
        self, alpha: float, delta: float, psi: float, gmst: float
    ) -> tuple[numpy.float64, numpy.float64]:
        """
        Calculate the interferometer pattern functions.

        Parameters
        ----------
        alpha
            Right ascension of the source in the detector frame (rad).
        delta
            Declination of the source in the detector frame (rad).
        psi
            Polarization angle of the source in the detector frame (rad).
        gmst
            Greenwich mean sidereal time (rad).

        Returns
        -------
        f_plus
            Plus pattern function.
        f_cross
            Cross pattern function.
        """
        e_plus, e_cross = calculate_polarisation_tensors(alpha, delta, psi, gmst)

        f_plus = numpy.float64(numpy.tensordot(self.d, e_plus))
        f_cross = numpy.float64(numpy.tensordot(self.d, e_cross))

        return f_plus, f_cross

    def calculate_strain(
        self, model: waveform.WaveformModel, theta: SignalParameters
    ) -> numpy.typing.NDArray[numpy.complex128]:
        """
        Calculate the frequency-domain strain response in the interferometer.

        Parameters
        ----------
        model
            Gravitational-waveform model.
        theta
            Parameters of the gravitational-wave signal.

        Returns
        -------
        h_tilde
            Frequency-domain strain (Hz^-1).
        """
        h_tilde_plus, h_tilde_cross = model.calculate_strain_polarisations(self.f, theta.waveform_parameters)
        f_plus, f_cross = self.calculate_pattern_functions(
            theta.detector_angles.alpha,
            theta.detector_angles.delta,
            theta.detector_angles.psi,
            theta.detector_angles.gmst,
        )
        delta_t = calculate_time_delay(
            self.x, GEOCENTRE, theta.detector_angles.alpha, theta.detector_angles.delta, theta.detector_angles.gmst
        )
        h_tilde: numpy.typing.NDArray[numpy.complex128] = (h_tilde_plus * f_plus + h_tilde_cross * f_cross) * numpy.exp(
            -2j * constants.PI * self.f * delta_t
        )
        return h_tilde

    def calculate_inner_product(
        self, a_tilde: numpy.typing.NDArray[numpy.complexfloating], b_tilde: numpy.typing.NDArray[numpy.complexfloating]
    ) -> numpy.complex128:
        """
        Calculate the (complex) noise-weighted inner product of two frequency-domain functions.

        Parameters
        ----------
        a_tilde
            First frequency-domain function (Hz^-1).
        b_tilde
            Second frequency-domain function (Hz^-1).

        Returns
        -------
        inner_product
            Inner product.
        """
        return calculate_inner_product(a_tilde, b_tilde, self.s_n, self.grid.delta_f)

    def calculate_optimal_signal_to_noise_ratio(
        self, h_tilde: numpy.typing.NDArray[numpy.complexfloating]
    ) -> numpy.float64:
        """
        Calculate the optimal signal-to-noise ratio.

        Parameters
        ----------
        h_tilde
            Frequency-domain strain (Hz^-1).

        Returns
        -------
        rho_opt
            Optimal signal-to-noise ratio.
        """
        return self.calculate_inner_product(h_tilde, h_tilde).real ** (1 / 2)

    def calculate_matched_filter_signal_to_noise_ratio(
        self, h_tilde: numpy.typing.NDArray[numpy.complexfloating]
    ) -> numpy.complex128:
        """
        Calculate the matched-filter signal-to-noise ratio.

        Parameters
        ----------
        h_tilde
            Frequency-domain strain (Hz^-1).

        Returns
        -------
        rho_mf
            Matched-filter signal-to-noise ratio.
        """
        rho_opt = self.calculate_optimal_signal_to_noise_ratio(h_tilde)
        return self.calculate_inner_product(h_tilde, self.d_tilde) / rho_opt

    def inject_signal(
        self, model: waveform.WaveformModel, theta: SignalParameters
    ) -> tuple[numpy.typing.NDArray[numpy.complex128], numpy.float64, numpy.complex128]:
        """
        Inject gravitational-wave signal into the interferometer.

        Parameters
        ----------
        model
            Gravitational-waveform model.
        theta
            Parameters of the gravitational-wave signal.

        Returns
        -------
        h_tilde
            Frequency-domain strain (Hz^-1).
        rho_opt
            Optimal signal-to-noise ratio.
        rho_mf
            Matched-filter signal-to-noise ratio.
        """
        h_tilde = self.calculate_strain(model, theta)

        self.d_tilde += h_tilde

        rho_opt = self.calculate_optimal_signal_to_noise_ratio(h_tilde)
        rho_mf = self.calculate_matched_filter_signal_to_noise_ratio(h_tilde)

        return h_tilde, rho_opt, rho_mf


class LHO(Interferometer):
    """
    LIGO Hanford gravitational-wave interferometer.

    Parameters
    ----------
    grid
        Grid for signal sampling.
    rng
        Random number generator for the noise realisation.
    is_zero_noise
        Whether to use zero noise realisation.

    Notes
    -----
    The location and orientation data are taken from LALSuite [1].

    References
    ----------
    [1]  <https://lscsoft.docs.ligo.org/lalsuite/lal/group___create_detector__c.html>.
    """

    def __init__(self, grid: Grid, rng: None | numpy.random.Generator = None, is_zero_noise: bool = False) -> None:
        pwd = Path(__file__).parent
        x = numpy.array([-2.1614149e6, -3.8346952e6, 4.6003502e6])
        d = numpy.array(
            [
                [-0.3926141, -0.0776130, -0.2473886],
                [-0.0776130, 0.3195244, 0.2279981],
                [-0.2473886, 0.2279981, 0.0730903],
            ]
        )
        super().__init__(grid, pwd / "data/aligo_O4high.txt", 20, 2048, x, d, rng, is_zero_noise)


class LLO(Interferometer):
    """
    LIGO Livingston gravitational-wave interferometer.

    Parameters
    ----------
    grid
        Grid for signal sampling.
    rng
        Random number generator for the noise realisation.
    is_zero_noise
        Whether to use zero noise realisation.

    Notes
    -----
    The location and orientation data are taken from LALSuite [1].

    References
    ----------
    [1]  <https://lscsoft.docs.ligo.org/lalsuite/lal/group___create_detector__c.html>.
    """

    def __init__(self, grid: Grid, rng: None | numpy.random.Generator = None, is_zero_noise: bool = False) -> None:
        pwd = Path(__file__).parent
        x = numpy.array([74276.044, -5.496283721e6, 3.224257018e6])
        d = numpy.array(
            [
                [0.4112809, 0.1402097, 0.2472943],
                [0.1402097, -0.1090056, -0.1816157],
                [0.2472943, -0.1816157, -0.3022755],
            ]
        )
        super().__init__(grid, pwd / "data/aligo_O4high.txt", 20, 2048, x, d, rng, is_zero_noise)


def calculate_inner_product(
    a_tilde: numpy.typing.NDArray[numpy.complexfloating],
    b_tilde: numpy.typing.NDArray[numpy.complexfloating],
    s_n: numpy.typing.NDArray[numpy.floating],
    delta_f: float,
) -> numpy.complex128:
    """
    Calculate the (complex) noise-weighted inner product of two frequency-domain functions.

    Parameters
    ----------
    a_tilde
        First frequency-domain function (Hz^-1).
    b_tilde
        Second frequency-domain function (Hz^-1).
    s_n
        Noise power spectral density (Hz^-1).
    delta_f
        Frequency resolution (Hz).

    Returns
    -------
    inner_product
        Inner product.
    """
    assert a_tilde.size == b_tilde.size == s_n.size, "Input arrays must have the same size."
    integrand = a_tilde.conj() * b_tilde / s_n
    integral = numpy.sum(integrand, dtype=numpy.complex128) * delta_f
    return 4 * integral


def calculate_polarisation_tensors(
    alpha: float, delta: float, psi: float, gmst: float
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    """
    Calculate the plus and cross polarisation tensors of a gravitational wave.

    Parameters
    ----------
    alpha
        Right ascension of the source (rad).
    delta
        Declination of the source (rad).
    psi
        Polarisation angle of the source (rad).
    gmst
        Greenwich mean sidereal time (rad).

    Returns
    -------
    e_plus
        Plus polarisation tensor.
    e_cross
        Cross polarisation tensor.

    Notes
    -----
    This follows the convention laid out in Ref. [1].

    References
    ----------
    [1]  <https://arxiv.org/abs/0903.0528>.
    """
    phi = alpha - gmst
    theta = constants.PI / 2 - delta

    u = numpy.array(
        [
            numpy.sin(phi) * numpy.cos(psi) - numpy.cos(theta) * numpy.cos(phi) * numpy.sin(psi),
            -(numpy.cos(phi) * numpy.cos(psi) + numpy.cos(theta) * numpy.sin(phi) * numpy.sin(psi)),
            numpy.sin(theta) * numpy.sin(psi),
        ]
    )
    v = numpy.array(
        [
            -numpy.sin(phi) * numpy.sin(psi) - numpy.cos(theta) * numpy.cos(phi) * numpy.cos(psi),
            numpy.cos(phi) * numpy.sin(psi) - numpy.cos(theta) * numpy.sin(phi) * numpy.cos(psi),
            numpy.sin(theta) * numpy.cos(psi),
        ]
    )

    e_plus = numpy.outer(u, u) - numpy.outer(v, v)
    e_cross = numpy.outer(u, v) + numpy.outer(v, u)

    return e_plus, e_cross


def calculate_time_delay(
    x_1: numpy.typing.NDArray[numpy.floating],
    x_2: numpy.typing.NDArray[numpy.floating],
    alpha: float,
    delta: float,
    gmst: float,
) -> numpy.float64:
    """
    Calculate the time delay between positions in geocentric coordinates.

    Parameters
    ----------
    x_1
        Cartesian coordinate vector for the first position in the geocentric frame (m).
    x_2
        Cartesian coordinate vector for the second position in the geocentric frame (m).
    alpha
        Right ascension of the source (rad).
    delta
        Declination of the source (rad).
    gmst
        Greenwich mean sidereal time (rad).

    Returns
    -------
    delta_t
        Time delay between the positions in the geocentric frame (s).

    Notes
    -----
    The time delay is calculated following the implementation from LALSuite [1].

    References
    ----------
    [1]  <https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_delay__h.html>.
    """
    phi = alpha - gmst
    theta = constants.PI / 2 - delta

    delta_t: numpy.float64 = (
        (x_2[0] - x_1[0]) * numpy.sin(theta) * numpy.cos(phi)
        + (x_2[1] - x_1[1]) * numpy.sin(theta) * numpy.sin(phi)
        + (x_2[2] - x_1[2]) * numpy.cos(theta)
    ) / constants.C
    return delta_t
