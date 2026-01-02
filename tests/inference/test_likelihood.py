"""Tests for the likelihood module."""

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy
import pytest

from chirplab import constants
from chirplab.inference import likelihood
from chirplab.simulation import interferometer

if TYPE_CHECKING:
    from chirplab.simulation import waveform


@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
def injected_interferometer_zero_noise_default(
    grid_default: interferometer.Grid,
    model_default: waveform.WaveformModel,
    theta_default: interferometer.SignalParameters,
) -> interferometer.Interferometer:
    """Return default injected interferometer with zero noise for testing."""
    ifo = interferometer.LIGO(grid_default, is_zero_noise=True)
    ifo.inject_signal(model_default, theta_default)
    return ifo


@pytest.fixture(scope="class")
def likelihood_default(
    injected_interferometer_default: interferometer.Interferometer, model_default: waveform.WaveformModel
) -> likelihood.Likelihood:
    """Return default likelihood object for testing."""
    return likelihood.Likelihood(injected_interferometer_default, model_default)


class TestLikelihood:
    """Tests for the Likelihood class."""

    def test_initialisation(
        self, model_default: waveform.WaveformModel, injected_interferometer_default: interferometer.Interferometer
    ) -> None:
        """Test that Likelihood can be initialised with an interferometer and waveform model."""
        like = likelihood.Likelihood(injected_interferometer_default, model_default)

        assert like.interferometer == injected_interferometer_default
        assert like.model == model_default

    def test_log_likelihood_noise_is_real(self, likelihood_default: likelihood.Likelihood) -> None:
        """Test that the noise log-likelihood is a real number."""
        ln_l_noise = likelihood_default.ln_l_noise

        assert isinstance(ln_l_noise, (float, numpy.floating))

    def test_log_likelihood_noise_consistency(self, likelihood_default: likelihood.Likelihood) -> None:
        """Test that repeated calls to ln_l_noise property return the same value."""
        ln_l_noise_1 = likelihood_default.ln_l_noise
        ln_l_noise_2 = likelihood_default.ln_l_noise

        assert ln_l_noise_1 == ln_l_noise_2

    def test_calculate_log_likelihood_returns_real(
        self, likelihood_default: likelihood.Likelihood, theta_default: interferometer.SignalParameters
    ) -> None:
        """Test that calculate_log_likelihood returns a real number."""
        ln_l = likelihood_default.calculate_log_likelihood(theta_default)

        assert isinstance(ln_l, (float, numpy.floating))

    def test_calculate_log_likelihood_zero_noise(
        self,
        injected_interferometer_zero_noise_default: interferometer.Interferometer,
        model_default: waveform.WaveformModel,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test log-likelihood calculation with zero noise."""
        like = likelihood.Likelihood(injected_interferometer_zero_noise_default, model_default)
        ln_l = like.calculate_log_likelihood(theta_default)

        assert not numpy.isnan(ln_l)
        assert not numpy.isinf(ln_l)

    def test_calculate_log_likelihood_with_noise(
        self, likelihood_default: likelihood.Likelihood, theta_default: interferometer.SignalParameters
    ) -> None:
        """Test log-likelihood calculation with noise."""
        ln_l = likelihood_default.calculate_log_likelihood(theta_default)

        assert not numpy.isnan(ln_l)
        assert not numpy.isinf(ln_l)

    def test_calculate_log_likelihood_wrong_parameters(
        self,
        injected_interferometer_zero_noise_default: interferometer.Interferometer,
        model_default: waveform.WaveformModel,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test log-likelihood with incorrect signal parameters."""
        like = likelihood.Likelihood(injected_interferometer_zero_noise_default, model_default)
        theta_wrong = replace(theta_default, m_1=20 * constants.M_SUN, m_2=20 * constants.M_SUN, r=2 * theta_default.r)
        ln_l_true = like.calculate_log_likelihood(theta_default)
        ln_l_wrong = like.calculate_log_likelihood(theta_wrong)

        assert ln_l_true > ln_l_wrong
