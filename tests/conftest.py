"""Fixtures for testing."""

import numpy
import pytest

from chirplab import constants, interferometer, waveform


@pytest.fixture(scope="session")
def theta_default() -> interferometer.SignalParameters:
    """Return default set of signal parameters for testing."""
    return interferometer.SignalParameters(
        m_1=30 * constants.M_SUN,
        m_2=30 * constants.M_SUN,
        r=500e6 * constants.PC,
        iota=constants.PI / 3,
        t_c=100,
        phi_c=1.5,
        theta=0,
        phi=constants.PI / 4,
        psi=0.5,
    )


@pytest.fixture(scope="session")
def grid_default() -> interferometer.Grid:
    """Return default grid for testing."""
    return interferometer.Grid(t_d=4, f_s=4096)


@pytest.fixture(scope="session")
def model_default() -> waveform.WaveformModel:
    """Return default waveform model for testing."""
    return waveform.NewtonianWaveformModel()


@pytest.fixture(scope="function")
def rng_default() -> numpy.random.Generator:
    """Return default random number generator for testing."""
    return numpy.random.default_rng(42)
