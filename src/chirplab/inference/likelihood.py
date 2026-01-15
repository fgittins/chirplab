"""Module for likelihood functions."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy

    from chirplab.simulation import interferometer, waveform


class Likelihood(ABC):
    """Likelihood function."""

    @abstractmethod
    def calculate_log_pdf(self, x: numpy.typing.NDArray[numpy.floating]) -> float:
        """
        Calculate the log of the probability density function.

        Parameters
        ----------
        x
            Vector of parameters.

        Returns
        -------
        ln_p
            Log of the probability density function.
        """
        pass


class GravitationalWaveLikelihood(Likelihood):
    """
    Likelihood function for gravitational-wave signals.

    Parameters
    ----------
    interferometer
        Gravitational-wave interferometer.
    model
        Gravitational-waveform model.
    vector_to_parameters
        Function to convert vector to gravitational-wave signal parameters.

    Notes
    -----
    Irrelevant additive constants have been omitted.
    """

    def __init__(
        self,
        interferometer: interferometer.Interferometer,
        model: waveform.WaveformModel,
        vector_to_parameters: Callable[[numpy.typing.NDArray[numpy.floating]], interferometer.SignalParameters],
    ) -> None:
        self.interferometer = interferometer
        self.model = model
        self.vector_to_parameters = vector_to_parameters

    def calculate_log_pdf(self, x: numpy.typing.NDArray[numpy.floating]) -> numpy.float64:
        """
        Calculate the log of the probability density function.

        Parameters
        ----------
        x
            Vector of parameters.

        Returns
        -------
        ln_p
            Log of the probability density function.

        Notes
        -----
        Under the gravitational-wave signal hypothesis, the collected data is given by the sum of a gravitational-wave
        signal and noise, which is assumed to be Gaussian.
        """
        theta = self.vector_to_parameters(x)
        h_tilde = self.interferometer.calculate_strain(self.model, theta)
        n_tilde = self.interferometer.d_tilde - h_tilde
        n_inner_n = self.interferometer.calculate_inner_product(n_tilde, n_tilde).real
        return -1 / 2 * n_inner_n

    @property
    def ln_l_noise(self) -> numpy.float64:
        """
        Log of the probability density function under the noise hypothesis.

        Notes
        -----
        Under the noise hypothesis, the collected data is explained by noise alone, which is assumed to be Gaussian.
        """
        n_tilde = self.interferometer.d_tilde
        n_inner_n = self.interferometer.calculate_inner_product(n_tilde, n_tilde).real
        return -1 / 2 * n_inner_n
