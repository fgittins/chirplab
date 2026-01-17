"""Tests for the parameters module."""

from chirplab import constants
from chirplab.simulation import parameters


class TestWaveformParameters:
    """Tests for the WaveformParameters dataclass."""

    def test_initialisation(self) -> None:
        """Test that WaveformParameters can be initialised with all required fields."""
        theta = parameters.WaveformParameters(
            30 * constants.M_SUN, 30 * constants.M_SUN, 500e6 * constants.PC, constants.PI / 3, 100, 1.5
        )

        assert theta.m_1 == 30 * constants.M_SUN
        assert theta.m_2 == 30 * constants.M_SUN
        assert theta.r == 500e6 * constants.PC
        assert theta.iota == constants.PI / 3
        assert theta.t_c == 100
        assert theta.phi_c == 1.5
        assert theta.m_chirp == (theta.m_1 * theta.m_2) ** (3 / 5) / (theta.m_1 + theta.m_2) ** (1 / 5)
        assert theta.m_total == 60 * constants.M_SUN


class TestSignalParameters:
    """Tests for the SignalParameters dataclass."""

    def test_initialisation(self) -> None:
        """Test that SignalParameters can be initialised with all required fields."""
        theta = parameters.SignalParameters.from_dict(
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

        assert theta.waveform_parameters.m_1 == 30 * constants.M_SUN
        assert theta.waveform_parameters.m_2 == 30 * constants.M_SUN
        assert theta.waveform_parameters.r == 500e6 * constants.PC
        assert theta.waveform_parameters.iota == constants.PI / 3
        assert theta.waveform_parameters.t_c == 100
        assert theta.waveform_parameters.phi_c == 1.5
        assert theta.detector_angles.theta == 0
        assert theta.detector_angles.phi == constants.PI / 4
        assert theta.detector_angles.psi == 0.5
        assert theta.waveform_parameters.m_total == 60 * constants.M_SUN
        assert theta.waveform_parameters.m_chirp == (theta.waveform_parameters.m_1 * theta.waveform_parameters.m_2) ** (
            3 / 5
        ) / (theta.waveform_parameters.m_1 + theta.waveform_parameters.m_2) ** (1 / 5)
