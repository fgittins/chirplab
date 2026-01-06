"""Fixtures for testing."""

from typing import TYPE_CHECKING

import numpy
import pytest

from chirplab.inference import likelihood
from chirplab.simulation import interferometer

if TYPE_CHECKING:
    from chirplab.simulation import waveform


@pytest.fixture(scope="session")
def injected_interferometer_default(
    grid_default: interferometer.Grid,
    model_default: waveform.WaveformModel,
    theta_default: interferometer.SignalParameters,
) -> interferometer.Interferometer:
    """Return default injected interferometer for testing."""
    rng = numpy.random.default_rng(42)
    ifo = interferometer.LIGO(grid_default, rng=rng)
    ifo.inject_signal(model_default, theta_default)
    return ifo


@pytest.fixture(scope="session")
def likelihood_default(
    injected_interferometer_default: interferometer.Interferometer, model_default: waveform.WaveformModel
) -> likelihood.Likelihood:
    """Return default likelihood object for testing."""
    return likelihood.Likelihood(injected_interferometer_default, model_default)
