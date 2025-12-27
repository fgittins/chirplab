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
    present in the data and buried within Gaussian noise.
    """

    def __init__(self, interferometer: interferometer.Interferometer, model: waveform.WaveformModel) -> None:
        self.interferometer = interferometer
        self.model = model

    def calculate_log_likelihood(self, theta: interferometer.SignalParameters) -> numpy.float64:
        """
        Calculate the log-likelihood as a function of the gravitational-wave signal parameters.

        Parameters
        ----------
        theta
            Parameters of the gravitational-wave signal.

        Returns
        -------
        ln_l
            Log-likelihood value.

        Notes
        -----
        Irrelevant additive constants have been omitted.

        Under the gravitational-wave signal hypothesis, the collected data is given by the sum of a gravitational-wave
        signal and noise, which is assumed to be Gaussian.
        """
        h_tilde = self.interferometer.calculate_strain(self.model, theta)
        n_tilde = self.interferometer.d_tilde - h_tilde
        n_inner_n = self.interferometer.calculate_inner_product(n_tilde, n_tilde).real
        return -1 / 2 * n_inner_n

    @property
    def ln_l_noise(self) -> numpy.float64:
        """
        Log-likelihood of the noise hypothesis.

        Notes
        -----
        Irrelevant additive constants have been omitted.

        Under the noise hypothesis, the collected data is explained by noise alone, which is assumed to be Gaussian.
        """
        n_tilde = self.interferometer.d_tilde
        n_inner_n = self.interferometer.calculate_inner_product(n_tilde, n_tilde).real
        return -1 / 2 * n_inner_n
