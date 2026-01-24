"""Module for gravitational-wave interferometers."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy

from chirplab import constants

if TYPE_CHECKING:
    from chirplab.simulation import grid, parameters, waveform

logger = logging.getLogger(__name__)

GEOCENTRE: Final[numpy.typing.NDArray[numpy.float64]] = numpy.array([0, 0, 0], dtype=numpy.float64)
"""Geocentre position vector (m)."""
D: Final[numpy.typing.NDArray[numpy.float64]] = 1 / 2 * numpy.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
"""Default response tensor in the geocentric frame."""
PWD = Path(__file__).parent


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
        grid: grid.Grid,
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

        self.f = grid.f
        self.t = grid.t
        self.in_bounds_mask = (f_min <= self.f) & (self.f <= f_max)

        f, amplitude_spectral_density = numpy.loadtxt(amplitude_spectral_density_file, numpy.float64, unpack=True)
        self.s_n = numpy.interp(self.f, f, amplitude_spectral_density**2)
        self.s_n[~self.in_bounds_mask] = constants.INF

        logger.debug(
            "Initialised interferometer: amplitude_spectral_density_file=%s, f_min=%.1f Hz, f_max=%.1f Hz",
            amplitude_spectral_density_file.name,
            f_min,
            f_max,
        )

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
        logger.debug("Set interferometer data: rng=%s, is_zero_noise=%s", rng, is_zero_noise)

        # TODO: adapt this method to load real data
        n_tilde = numpy.zeros_like(self.f, numpy.complex128)

        if not is_zero_noise:
            if rng is None:
                rng = numpy.random.default_rng()

            m_tilde = self.grid.generate_gaussian_noise(rng)
            n_tilde[self.in_bounds_mask] = self.s_n[self.in_bounds_mask] ** (1 / 2) * m_tilde[self.in_bounds_mask]

        self.s_tilde = n_tilde.copy()
        self.s = self.grid.calculate_inverse_fourier_transform(n_tilde.copy())

    def calculate_pattern_functions(self, theta: float, phi: float, psi: float) -> tuple[numpy.float64, numpy.float64]:
        """
        Calculate the interferometer pattern functions.

        Parameters
        ----------
        theta
            Polar angle of the source in the geocentric frame (rad).
        phi
            Azimuthal angle of the source in the geocentric frame (rad).
        psi
            Polarization angle of the source in the geocentric frame (rad).

        Returns
        -------
        f_plus
            Plus pattern function.
        f_cross
            Cross pattern function.
        """
        e_plus, e_cross = calculate_polarisation_tensors(theta, phi, psi)

        f_plus = numpy.float64(numpy.tensordot(self.d, e_plus))
        f_cross = numpy.float64(numpy.tensordot(self.d, e_cross))

        return f_plus, f_cross

    def calculate_strain(
        self, model: waveform.WaveformModel, theta: parameters.SignalParameters
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
        h_tilde_plus = numpy.zeros_like(self.f, numpy.complex128)
        h_tilde_cross = numpy.zeros_like(self.f, numpy.complex128)
        h_tilde_plus[self.in_bounds_mask], h_tilde_cross[self.in_bounds_mask] = model.calculate_strain_polarisations(
            self.f[self.in_bounds_mask], theta
        )

        f_plus, f_cross = self.calculate_pattern_functions(theta.theta, theta.phi, theta.psi)

        delta_t = calculate_time_delay(self.x, GEOCENTRE, theta.theta, theta.phi)

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
        a_inner_b
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
        return self.calculate_inner_product(h_tilde, self.s_tilde) / rho_opt

    def inject_signal(
        self, model: waveform.WaveformModel, theta: parameters.SignalParameters
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
        logger.info("Injecting signal: model=%s, theta=%s", model, theta)

        h_tilde = self.calculate_strain(model, theta)

        self.s_tilde += h_tilde
        self.s += self.grid.calculate_inverse_fourier_transform(h_tilde)

        rho_opt = self.calculate_optimal_signal_to_noise_ratio(h_tilde)
        rho_mf = self.calculate_matched_filter_signal_to_noise_ratio(h_tilde)

        logger.info("Injected signal: rho_opt=%.2f, rho_mf=%.2f%+.2fj", rho_opt, rho_mf.real, rho_mf.imag)

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

    def __init__(self, grid: grid.Grid, rng: None | numpy.random.Generator = None, is_zero_noise: bool = False) -> None:
        x = numpy.array([-2.1614149e6, -3.8346952e6, 4.6003502e6])
        d = numpy.array(
            [
                [-0.3926141, -0.0776130, -0.2473886],
                [-0.0776130, 0.3195244, 0.2279981],
                [-0.2473886, 0.2279981, 0.0730903],
            ]
        )
        super().__init__(grid, PWD / "data/aligo_O4high.txt", 20, 2048, x, d, rng, is_zero_noise)

        logger.info("Initialised LIGO Hanford interferometer")


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

    def __init__(self, grid: grid.Grid, rng: None | numpy.random.Generator = None, is_zero_noise: bool = False) -> None:
        x = numpy.array([74276.044, -5.496283721e6, 3.224257018e6])
        d = numpy.array(
            [
                [0.4112809, 0.1402097, 0.2472943],
                [0.1402097, -0.1090056, -0.1816157],
                [0.2472943, -0.1816157, -0.3022755],
            ]
        )
        super().__init__(grid, PWD / "data/aligo_O4high.txt", 20, 2048, x, d, rng, is_zero_noise)

        logger.info("Initialised LIGO Livingston interferometer")


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
    a_inner_b
        Inner product.
    """
    assert a_tilde.size == b_tilde.size == s_n.size, "Input arrays must have the same size."
    integrand = a_tilde.conj() * b_tilde / s_n
    integral = numpy.sum(integrand, dtype=numpy.complex128) * delta_f
    return 4 * integral


def calculate_polarisation_tensors(
    theta: float, phi: float, psi: float
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    """
    Calculate the plus and cross polarisation tensors of a gravitational wave.

    Parameters
    ----------
    theta
        Polar angle of the source with respect in the geocentric frame (rad).
    phi
        Azimuthal angle of the source with respect in the geocentric frame (rad).
    psi
        Polarisation angle of the source with respect in the geocentric frame (rad).

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
    sin_theta = numpy.sin(theta)
    cos_theta = numpy.cos(theta)
    sin_phi = numpy.sin(phi)
    cos_phi = numpy.cos(phi)
    sin_psi = numpy.sin(psi)
    cos_psi = numpy.cos(psi)

    theta_hat = numpy.array([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta])
    phi_hat = numpy.array([-sin_phi, cos_phi, 0])

    theta_hat_prime = cos_psi * theta_hat + sin_psi * phi_hat
    phi_hat_prime = -sin_psi * theta_hat + cos_psi * phi_hat

    e_plus = numpy.outer(theta_hat_prime, theta_hat_prime) - numpy.outer(phi_hat_prime, phi_hat_prime)
    e_cross = numpy.outer(theta_hat_prime, phi_hat_prime) + numpy.outer(phi_hat_prime, theta_hat_prime)

    return e_plus, e_cross


def calculate_time_delay(
    x_1: numpy.typing.NDArray[numpy.floating], x_2: numpy.typing.NDArray[numpy.floating], theta: float, phi: float
) -> numpy.float64:
    """
    Calculate the time delay between positions in the geocentric frame.

    Parameters
    ----------
    x_1
        Cartesian coordinate vector for the first position in the geocentric frame (m).
    x_2
        Cartesian coordinate vector for the second position in the geocentric frame (m).
    theta
        Polar angle of the source in the geocentric frame (rad).
    phi
        Azimuthal angle of the source in the geocentric frame (rad).

    Returns
    -------
    delta_t
        Time delay between the positions in the geocentric frame (s).

    Notes
    -----
    The time delay is calculated following the implementation in LALSuite [1].

    References
    ----------
    [1]  <https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_delay__h.html>.
    """
    sin_theta = numpy.sin(theta)
    cos_theta = numpy.cos(theta)
    sin_phi = numpy.sin(phi)
    cos_phi = numpy.cos(phi)

    r_hat = numpy.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
    delta_x = x_2 - x_1

    return numpy.float64(numpy.dot(delta_x, r_hat) / constants.C)
