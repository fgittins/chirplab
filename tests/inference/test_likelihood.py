"""Tests for the likelihood module."""

from typing import TYPE_CHECKING

import numpy
import pytest

from chirplab import constants
from chirplab.inference import likelihood
from chirplab.simulation import interferometer
from tests.inference.conftest import vector_to_parameters

if TYPE_CHECKING:
    from chirplab.simulation import grid, parameters, waveform


@pytest.fixture(scope="class")
def x_default() -> numpy.typing.NDArray[numpy.floating]:
    """Return default parameter vector for testing."""
    return numpy.array(
        [
            30 * constants.M_SUN,
            30 * constants.M_SUN,
            500e6 * constants.PC,
            constants.PI / 3,
            100,
            1.5,
            0,
            constants.PI / 4,
            0.5,
        ]
    )


@pytest.fixture(scope="class")
def injected_interferometer_zero_noise_default(
    grid_default: grid.Grid,
    model_default: waveform.WaveformModel,
    theta_default: parameters.SignalParameters,
) -> interferometer.Interferometer:
    """Return default injected interferometer with zero noise for testing."""
    ifo = interferometer.LLO(grid_default, is_zero_noise=True)
    ifo.inject_signal(model_default, theta_default)
    return ifo


class TestGravitationalWaveLikelihood:
    """Tests for the GravitationalWaveLikelihood class."""

    def test_initialisation(
        self, model_default: waveform.WaveformModel, injected_interferometer_default: interferometer.Interferometer
    ) -> None:
        """Test that GravitationalWaveLikelihood can be initialised with an interferometer and waveform model."""
        like = likelihood.GravitationalWaveLikelihood(
            (injected_interferometer_default,), model_default, vector_to_parameters
        )

        assert like.interferometers == (injected_interferometer_default,)
        assert like.model == model_default
        assert like.vector_to_parameters == vector_to_parameters
        assert isinstance(like.ln_n, numpy.floating)
        assert isinstance(like.s_inner_s, list)

    def test_initialisation_normalised(
        self, model_default: waveform.WaveformModel, injected_interferometer_default: interferometer.Interferometer
    ) -> None:
        """Test that GravitationalWaveLikelihood can be initialised with normalised likelihood."""
        like = likelihood.GravitationalWaveLikelihood(
            (injected_interferometer_default,), model_default, vector_to_parameters, is_normalised=True
        )

        assert like.ln_n != 0.0

    def test_initialisation_multiple_interferometers(
        self,
        grid_default: grid.Grid,
        rng_default: numpy.random.Generator,
        model_default: waveform.WaveformModel,
        theta_default: parameters.SignalParameters,
    ) -> None:
        """Test that GravitationalWaveLikelihood can be initialised with multiple interferometers."""
        ifo_1 = interferometer.LHO(grid_default, rng_default)
        ifo_2 = interferometer.LLO(grid_default, rng_default)
        ifo_1.inject_signal(model_default, theta_default)
        ifo_2.inject_signal(model_default, theta_default)
        like = likelihood.GravitationalWaveLikelihood((ifo_1, ifo_2), model_default, vector_to_parameters)

        assert like.interferometers == (ifo_1, ifo_2)

    def test_log_likelihood_noise_is_real(self, likelihood_default: likelihood.GravitationalWaveLikelihood) -> None:
        """Test that the noise log-likelihood is a real number."""
        ln_l_noise = likelihood_default.ln_l_noise

        assert isinstance(ln_l_noise, (float, numpy.floating))

    def test_log_likelihood_noise_consistency(self, likelihood_default: likelihood.GravitationalWaveLikelihood) -> None:
        """Test that repeated calls to ln_l_noise property return the same value."""
        ln_l_noise_1 = likelihood_default.ln_l_noise
        ln_l_noise_2 = likelihood_default.ln_l_noise

        assert ln_l_noise_1 == ln_l_noise_2

    def test_calculate_log_pdf_returns_real(
        self,
        likelihood_default: likelihood.GravitationalWaveLikelihood,
        x_default: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        """Test that calculate_log_pdf returns a real number."""
        ln_l = likelihood_default.calculate_log_pdf(x_default)

        assert isinstance(ln_l, (float, numpy.floating))

    def test_calculate_log_pdf_zero_noise(
        self,
        injected_interferometer_zero_noise_default: interferometer.Interferometer,
        model_default: waveform.WaveformModel,
        x_default: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        """Test log-likelihood calculation with zero noise."""
        like = likelihood.GravitationalWaveLikelihood(
            (injected_interferometer_zero_noise_default,), model_default, vector_to_parameters
        )
        ln_l = like.calculate_log_pdf(x_default)

        assert not numpy.isnan(ln_l)
        assert not numpy.isinf(ln_l)

    def test_calculate_log_pdf_with_noise(
        self,
        likelihood_default: likelihood.GravitationalWaveLikelihood,
        x_default: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        """Test log-likelihood calculation with noise."""
        ln_l = likelihood_default.calculate_log_pdf(x_default)

        assert not numpy.isnan(ln_l)
        assert not numpy.isinf(ln_l)

    def test_calculate_log_pdf_wrong_parameters(
        self,
        injected_interferometer_zero_noise_default: interferometer.Interferometer,
        model_default: waveform.WaveformModel,
        x_default: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        """Test log-likelihood with incorrect signal parameters."""
        like = likelihood.GravitationalWaveLikelihood(
            (injected_interferometer_zero_noise_default,), model_default, vector_to_parameters
        )
        x_wrong = x_default.copy()
        x_wrong[0] = 20 * constants.M_SUN
        x_wrong[1] = 20 * constants.M_SUN
        x_wrong[2] = 2 * x_default[2]
        ln_l_true = like.calculate_log_pdf(x_default)
        ln_l_wrong = like.calculate_log_pdf(x_wrong)

        assert ln_l_true > ln_l_wrong
