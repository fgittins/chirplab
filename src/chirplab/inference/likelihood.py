"""Module for likelihood functions."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy

from chirplab import constants

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from chirplab.simulation import interferometer, parameters, waveform


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
        ln_l
            Log of the probability density function.
        """
        ...


class GravitationalWaveLikelihood(Likelihood):
    """
    Likelihood function for gravitational-wave signals.

    Parameters
    ----------
    interferometers
        Gravitational-wave interferometers.
    model
        Gravitational-waveform model.
    vector_to_parameters
        Function to convert vector to gravitational-wave signal parameters.
    is_normalised
        Whether the likelihood function is normalised.
    """

    def __init__(
        self,
        interferometers: Iterable[interferometer.Interferometer],
        model: waveform.WaveformModel,
        vector_to_parameters: Callable[[numpy.typing.NDArray[numpy.floating]], parameters.SignalParameters],
        is_normalised: bool = False,
    ) -> None:
        self.interferometers = interferometers
        self.model = model
        self.vector_to_parameters = vector_to_parameters

        self.ln_n = numpy.float64(0)
        self.s_inner_s: list[numpy.float64] = []
        for interferometer in interferometers:
            if is_normalised:
                self.ln_n += numpy.sum(
                    numpy.log(
                        2
                        * interferometer.grid.delta_f
                        / (constants.PI * interferometer.s_n[interferometer.in_bounds_mask])
                    ),
                    dtype=numpy.float64,
                )
            self.s_inner_s.append(
                interferometer.calculate_inner_product(interferometer.s_tilde, interferometer.s_tilde).real
            )

    def calculate_log_pdf(self, x: numpy.typing.NDArray[numpy.floating]) -> numpy.float64:
        """
        Calculate the log of the probability density function.

        Parameters
        ----------
        x
            Vector of parameters.

        Returns
        -------
        ln_l
            Log of the probability density function.

        Notes
        -----
        Under the gravitational-wave signal hypothesis, the collected data are given by the sum of a gravitational-wave
        signal and noise, which is assumed to be Gaussian.
        """
        theta = self.vector_to_parameters(x)
        ln_l = self.ln_l_noise
        for interferometer in self.interferometers:
            h_tilde = interferometer.calculate_strain(self.model, theta)
            inner = interferometer.calculate_inner_product(-2 * interferometer.s_tilde + h_tilde, h_tilde).real
            ln_l -= 1 / 2 * inner
        return ln_l

    @property
    def ln_l_noise(self) -> numpy.float64:
        """
        Log of the probability density function under the noise hypothesis.

        Notes
        -----
        Under the noise hypothesis, the collected data are explained by noise alone, which is assumed to be Gaussian.
        """
        return self.ln_n - 1 / 2 * numpy.sum(self.s_inner_s, dtype=numpy.float64)
