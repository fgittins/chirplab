"""Unit tests for the likelihood module."""

from dataclasses import replace

import numpy
import pytest

from chirplab import interferometer, likelihood, waveform


@pytest.fixture
def injected_interferometer_default(
    grid_default: interferometer.Grid,
    model_default: waveform.WaveformModel,
    Theta_default: interferometer.SignalParameters,
) -> interferometer.Interferometer:
    """Return default injected interferometer for testing."""
    rng = numpy.random.default_rng(42)
    ifo = interferometer.LIGO(grid_default, rng=rng)
    ifo.inject(model_default, Theta_default)
    return ifo


@pytest.fixture
def likelihood_default(
    injected_interferometer_default: interferometer.Interferometer, model_default: waveform.WaveformModel
) -> likelihood.Likelihood:
    """Return default likelihood object for testing."""
    return likelihood.Likelihood(injected_interferometer_default, model_default)


class TestLikelihood:
    """Tests for the `Likelihood` class."""

    def test_initialisation(
        self,
        likelihood_default: likelihood.Likelihood,
        model_default: waveform.WaveformModel,
        injected_interferometer_default: interferometer.Interferometer,
    ) -> None:
        """Test that `Likelihood` can be initialised with an interferometer and waveform model."""
        assert likelihood_default.interferometer == injected_interferometer_default
        assert likelihood_default.model == model_default

    def test_log_likelihood_noise_is_real(self, likelihood_default: likelihood.Likelihood) -> None:
        """Test that the noise log-likelihood is a real number."""
        ln_L_noise = likelihood_default.ln_L_noise

        assert isinstance(ln_L_noise, (float, numpy.floating))

    def test_log_likelihood_noise_is_negative(self, likelihood_default: likelihood.Likelihood) -> None:
        """Test that the noise log-likelihood is negative."""
        ln_L_noise = likelihood_default.ln_L_noise

        assert ln_L_noise < 0

    def test_log_likelihood_noise_consistency(self, likelihood_default: likelihood.Likelihood) -> None:
        """Test that repeated calls to `ln_L_noise` property return the same value."""
        ln_L_noise_1 = likelihood_default.ln_L_noise
        ln_L_noise_2 = likelihood_default.ln_L_noise

        assert ln_L_noise_1 == ln_L_noise_2

    def test_calculate_log_likelihood_returns_real(
        self, likelihood_default: likelihood.Likelihood, Theta_default: interferometer.SignalParameters
    ) -> None:
        """Test that `calculate_log_likelihood` returns a real number."""
        ln_L = likelihood_default.calculate_log_likelihood(Theta_default)

        assert isinstance(ln_L, (float, numpy.floating))

    def test_calculate_log_likelihood_is_negative(
        self, likelihood_default: likelihood.Likelihood, Theta_default: interferometer.SignalParameters
    ) -> None:
        """Test that log-likelihood is negative."""
        ln_L = likelihood_default.calculate_log_likelihood(Theta_default)

        assert ln_L < 0

    def test_calculate_log_likelihood_zero_noise(
        self,
        grid_default: interferometer.Grid,
        model_default: waveform.WaveformModel,
        Theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test log-likelihood calculation with zero noise."""
        ifo = interferometer.LIGO(grid_default, is_zero_noise=True)
        ifo.inject(model_default, Theta_default)
        like = likelihood.Likelihood(ifo, model_default)
        ln_L = like.calculate_log_likelihood(Theta_default)

        assert not numpy.isnan(ln_L)
        assert not numpy.isinf(ln_L)

    def test_calculate_log_likelihood_with_noise(
        self, likelihood_default: likelihood.Likelihood, Theta_default: interferometer.SignalParameters
    ) -> None:
        """Test log-likelihood calculation with noise."""
        ln_L = likelihood_default.calculate_log_likelihood(Theta_default)

        assert not numpy.isnan(ln_L)
        assert not numpy.isinf(ln_L)

    def test_calculate_log_likelihood_signal_injection(
        self,
        grid_default: interferometer.Grid,
        model_default: waveform.WaveformModel,
        Theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test log-likelihood improves when true signal parameters are used."""
        ifo = interferometer.LIGO(grid_default, is_zero_noise=True)
        ifo.inject(model_default, Theta_default)
        like = likelihood.Likelihood(ifo, model_default)
        ln_L_true = like.calculate_log_likelihood(Theta_default)

        assert ln_L_true > like.ln_L_noise

    def test_calculate_log_likelihood_wrong_parameters(
        self,
        grid_default: interferometer.Grid,
        model_default: waveform.WaveformModel,
        Theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test log-likelihood with incorrect signal parameters."""
        ifo = interferometer.LIGO(grid_default, is_zero_noise=True)
        ifo.inject(model_default, Theta_default)
        like = likelihood.Likelihood(ifo, model_default)
        Theta_wrong = replace(Theta_default, m_1=20 * waveform.M_sun, m_2=20 * waveform.M_sun, r=2 * Theta_default.r)
        ln_L_true = like.calculate_log_likelihood(Theta_default)
        ln_L_wrong = like.calculate_log_likelihood(Theta_wrong)

        assert ln_L_true > ln_L_wrong
