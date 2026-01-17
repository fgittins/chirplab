"""Fixtures for testing."""

import numpy
import pytest

from chirplab import constants
from chirplab.simulation import grid, parameters, waveform


@pytest.fixture(scope="session")
def theta_default() -> parameters.SignalParameters:
    """Return default set of signal parameters for testing."""
    return parameters.SignalParameters.from_dict(
        {
            "m_1": 30 * constants.M_SUN,
            "m_2": 30 * constants.M_SUN,
            "r": 500e6 * constants.PC,
            "iota": constants.PI / 3,
            "t_c": 100,
            "phi_c": 1.5,
            "alpha": constants.PI / 4,
            "delta": constants.PI / 2,
            "psi": 0.5,
        }
    )


@pytest.fixture(scope="session")
def grid_default() -> grid.Grid:
    """Return default grid for testing."""
    return grid.Grid(4, 4096)


@pytest.fixture(scope="session")
def model_default() -> waveform.WaveformModel:
    """Return default waveform model for testing."""
    return waveform.NewtonianWaveformModel()


@pytest.fixture
def rng_default() -> numpy.random.Generator:
    """Return default random number generator for testing."""
    return numpy.random.default_rng(42)
