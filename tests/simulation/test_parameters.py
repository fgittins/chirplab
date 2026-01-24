"""Tests for the parameters module."""

import numpy

from chirplab import constants
from chirplab.simulation import parameters


class TestWaveformParameters:
    """Tests for the WaveformParameters dataclass."""

    def test_initialisation(self) -> None:
        """Test that WaveformParameters can be initialised with all required fields."""
        theta = parameters.WaveformParameters(
            30 * constants.M_SUN,
            30 * constants.M_SUN,
            500e6 * constants.PC,
            constants.PI / 3,
            100,
            1.5,
        )

        assert theta.m_1 == 30 * constants.M_SUN
        assert theta.m_2 == 30 * constants.M_SUN
        assert theta.r == 500e6 * constants.PC
        assert theta.iota == constants.PI / 3
        assert theta.t_c == 100
        assert theta.phi_c == 1.5

    def test_derived_properties(self, theta_waveform_default: parameters.WaveformParameters) -> None:
        """Test that derived properties m_total and m_chirp are computed correctly."""
        assert theta_waveform_default.m_total == 60 * constants.M_SUN
        assert theta_waveform_default.m_chirp == (theta_waveform_default.m_1 * theta_waveform_default.m_2) ** (
            3 / 5
        ) / (theta_waveform_default.m_1 + theta_waveform_default.m_2) ** (1 / 5)


class TestSignalParameters:
    """Tests for the SignalParameters dataclass."""

    def test_initialisation(self) -> None:
        """Test that SignalParameters can be initialised with all required fields."""
        theta = parameters.SignalParameters(
            30 * constants.M_SUN,
            30 * constants.M_SUN,
            500e6 * constants.PC,
            constants.PI / 3,
            100,
            1.5,
            0,
            constants.PI / 4,
            0.5,
        )

        assert theta.m_1 == 30 * constants.M_SUN
        assert theta.m_2 == 30 * constants.M_SUN
        assert theta.r == 500e6 * constants.PC
        assert theta.iota == constants.PI / 3
        assert theta.t_c == 100
        assert theta.phi_c == 1.5
        assert theta.theta == 0
        assert theta.phi == constants.PI / 4
        assert theta.psi == 0.5

    def test_derived_properties(self, theta_default: parameters.SignalParameters) -> None:
        """Test that derived properties m_total and m_chirp are computed correctly."""
        assert theta_default.m_total == 60 * constants.M_SUN
        assert theta_default.m_chirp == (theta_default.m_1 * theta_default.m_2) ** (3 / 5) / (
            theta_default.m_1 + theta_default.m_2
        ) ** (1 / 5)
        assert numpy.isclose(theta_default.alpha, 2.620830242272142)
        assert theta_default.delta == constants.PI / 2 - theta_default.theta
        assert numpy.isclose(theta_default.gmst, 1.8354320788746938)
