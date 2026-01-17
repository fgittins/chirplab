"""Tests for the parameters module."""

import pytest

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

    def test_implementation_errors(self, theta_default: parameters.SignalParameters) -> None:
        """Test that NotImplementedError is raised for alpha and delta properties."""
        with pytest.raises(NotImplementedError, match="Right ascension not implemented."):
            _ = theta_default.alpha

        with pytest.raises(NotImplementedError, match="Declination not implemented."):
            _ = theta_default.delta

        with pytest.raises(NotImplementedError, match="Greenwich mean sidereal time not implemented."):
            _ = theta_default.gmst
