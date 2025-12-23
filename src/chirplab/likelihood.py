"""Module for likelihood calculations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy

from . import interferometer, waveform


class Likelihood:
    """
    Likelihood function for gravitational-wave signals.

    Parameters
    ----------
    interferometer
        Gravitational-wave interferometer.
    model
        Gravitational-waveform model.

    Notes
    -----
    The likelihood is taken to be a Gaussian probability distribution. This follows from the hypothesis that a signal is
    present in the data and buried within stationary, Gaussian noise.
    """

    def __init__(self, interferometer: interferometer.Interferometer, model: waveform.Waveform) -> None:
        self.interferometer = interferometer
        self.model = model

        self.ln_L_noise = self.calculate_noise_log_likelihood()

    def calculate_log_likelihood(self, Theta: waveform.SignalParameters) -> numpy.float64:
        """
        Calculate the log-likelihood as a function of the gravitational-wave signal parameters.

        Parameters
        ----------
        Theta
            Parameters of the gravitational-wave signal.

        Returns
        -------
        ln_L
            Log-likelihood value.

        Notes
        -----
        Irrelevant additive constants have been omitted.
        """
        h_tilde_plus, h_tilde_cross = self.model.calculate_strain_polarisations(self.interferometer.f, Theta)
        F_plus, F_cross = self.interferometer.calculate_pattern_functions(Theta.theta, Theta.phi, Theta.psi)
        h_tilde = h_tilde_plus * F_plus + h_tilde_cross * F_cross

        h_inner_d = interferometer.calculate_inner_product(
            h_tilde, self.interferometer.d_tilde, self.interferometer.S_n, self.interferometer.Delta_f
        ).real
        rho_squared = interferometer.calculate_inner_product(
            h_tilde, h_tilde, self.interferometer.S_n, self.interferometer.Delta_f
        ).real

        return h_inner_d - 1 / 2 * rho_squared

    def calculate_noise_log_likelihood(self) -> numpy.float64:
        """
        Calculate the log-likelihood of the noise-only hypothesis.

        Returns
        -------
        ln_L
            Log-likelihood value.
        """
        n_inner_n = interferometer.calculate_inner_product(
            self.interferometer.n_tilde,
            self.interferometer.n_tilde,
            self.interferometer.S_n,
            self.interferometer.Delta_f,
        ).real
        return -1 / 2 * n_inner_n
