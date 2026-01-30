"""Fixtures for testing."""

from typing import TYPE_CHECKING

import numpy
import pytest

from chirplab import constants
from chirplab.inference import distribution, likelihood, prior
from chirplab.simulation import interferometer, parameters

if TYPE_CHECKING:
    from chirplab.simulation import grid, waveform


def vector_to_parameters(x: numpy.typing.NDArray[numpy.floating]) -> parameters.SignalParameters:
    """Convert a vector to signal parameters."""
    return parameters.SignalParameters(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8])


@pytest.fixture(scope="session")
def injected_interferometer_default(
    grid_default: grid.Grid,
    model_default: waveform.WaveformModel,
    theta_default: parameters.SignalParameters,
) -> interferometer.Interferometer:
    """Return default injected interferometer for testing."""
    rng = numpy.random.default_rng(42)
    ifo = interferometer.LLO(grid_default, rng=rng)
    ifo.inject_signal(model_default, theta_default)
    return ifo


@pytest.fixture(scope="session")
def likelihood_default(
    injected_interferometer_default: interferometer.Interferometer, model_default: waveform.WaveformModel
) -> likelihood.GravitationalWaveLikelihood:
    """Return default likelihood object for testing."""
    return likelihood.GravitationalWaveLikelihood(
        (injected_interferometer_default,), model_default, vector_to_parameters
    )


@pytest.fixture(scope="class")
def prior_default() -> prior.Prior:
    """Return default prior for testing."""
    return prior.Prior(
        (
            distribution.Uniform(20 * constants.M_SUN, 40 * constants.M_SUN),
            distribution.Uniform(20 * constants.M_SUN, 40 * constants.M_SUN),
            distribution.Uniform(400e6 * constants.PC, 600e6 * constants.PC),
            distribution.Sine(),
            distribution.Uniform(99, 101),
            distribution.Uniform(0, 2 * constants.PI, boundary="periodic"),
            distribution.Sine(),
            distribution.Uniform(0, 2 * constants.PI, boundary="periodic"),
            distribution.Uniform(0, constants.PI),
        )
    )
