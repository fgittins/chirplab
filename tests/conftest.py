"""Fixtures for testing."""

import numpy
import pytest

from chirplab import interferometer, waveform


@pytest.fixture
def Theta_default() -> interferometer.SignalParameters:
    """Return default set of signal parameters for testing."""
    return interferometer.SignalParameters(
        m_1=30 * waveform.M_sun,
        m_2=30 * waveform.M_sun,
        r=500e6 * waveform.pc,
        iota=numpy.pi / 3,
        t_c=100,
        Phi_c=1.5,
        theta=0,
        phi=numpy.pi / 4,
        psi=0.5,
    )


@pytest.fixture
def grid_default() -> interferometer.Grid:
    """Return default grid for testing."""
    return interferometer.Grid(T=4, f_s=4096)


@pytest.fixture
def model_default() -> waveform.WaveformModel:
    """Return default waveform model for testing."""
    return waveform.NewtonianWaveformModel()
