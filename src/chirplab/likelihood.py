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

    def __init__(self, interferometer: interferometer.Interferometer, model: waveform.WaveformModel) -> None:
        self.interferometer = interferometer
        self.model = model

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
        h_tilde = self.interferometer.calculate_strain(self.model, Theta)
        n_tilde = self.interferometer.d_tilde - h_tilde
        n_inner_n = self.interferometer.calculate_inner_product(n_tilde, n_tilde).real
        return -1 / 2 * n_inner_n

    @property
    def ln_L_noise(self) -> numpy.float64:
        """
        Log-likelihood of the noise-only hypothesis.

        Notes
        -----
        Irrelevant additive constants have been omitted.
        """
        n_tilde = self.interferometer.d_tilde
        n_inner_n = self.interferometer.calculate_inner_product(n_tilde, n_tilde).real
        return -1 / 2 * n_inner_n
