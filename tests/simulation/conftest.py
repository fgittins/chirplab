"""Fixtures for testing."""

import pytest

from chirplab import constants
from chirplab.simulation import parameters


@pytest.fixture(scope="session")
def theta_waveform_default() -> parameters.WaveformParameters:
    """Return default set of signal parameters for testing."""
    return parameters.WaveformParameters(
        30 * constants.M_SUN,
        30 * constants.M_SUN,
        500e6 * constants.PC,
        constants.PI / 3,
        100,
        1.5,
    )
