"""Fixtures for testing."""

from typing import TYPE_CHECKING

import numpy
import pytest

from chirplab.inference import likelihood
from chirplab.simulation import interferometer, parameters

if TYPE_CHECKING:
    from chirplab.simulation import grid, waveform


def vector_to_parameters(x: numpy.typing.NDArray[numpy.floating]) -> parameters.SignalParameters:
    """Convert a vector to signal parameters."""
    return parameters.SignalParameters.from_dict(
        {
            "m_1": x[0],
            "m_2": x[1],
            "r": x[2],
            "iota": x[3],
            "t_c": x[4],
            "phi_c": x[5],
            "alpha": x[6],
            "delta": x[7],
            "psi": x[8],
        }
    )


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
    return likelihood.GravitationalWaveLikelihood(injected_interferometer_default, model_default, vector_to_parameters)
