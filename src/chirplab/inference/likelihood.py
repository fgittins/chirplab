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

        self.d_inner_d = self.interferometer.calculate_inner_product(
            self.interferometer.d_tilde, self.interferometer.d_tilde
        ).real

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
        Under the gravitational-wave signal hypothesis, the collected data are given by the sum of a gravitational-wave
        signal and noise, which is assumed to be Gaussian.
        """
        theta = self.vector_to_parameters(x)
        h_tilde = self.interferometer.calculate_strain(self.model, theta)
        inner_product = self.interferometer.calculate_inner_product(
            -2 * self.interferometer.d_tilde + h_tilde, h_tilde
        ).real
        return self.ln_l_noise - 1 / 2 * inner_product

    @property
    def ln_l_noise(self) -> numpy.float64:
        """
        Log of the probability density function under the noise hypothesis.

        Notes
        -----
        Under the noise hypothesis, the collected data are explained by noise alone, which is assumed to be Gaussian.
        """
        return -1 / 2 * self.d_inner_d
