"""Tests for the sampler module."""

from typing import TYPE_CHECKING

import numpy
import pytest

from chirplab import constants
from chirplab.inference import likelihood, prior, sampler

if TYPE_CHECKING:
    from pathlib import Path

    from chirplab.simulation import interferometer, waveform


@pytest.fixture(scope="class")
def priors_default() -> prior.Priors:
    """Return default priors for testing."""
    return prior.Priors(
        {
            "m_1": prior.Uniform(20 * constants.M_SUN, 40 * constants.M_SUN),
            "m_2": prior.Uniform(20 * constants.M_SUN, 40 * constants.M_SUN),
            "r": prior.Uniform(400e6 * constants.PC, 600e6 * constants.PC),
            "iota": prior.Sine(),
            "t_c": prior.Uniform(99, 101),
            "phi_c": prior.Uniform(0, 2 * constants.PI, boundary="periodic"),
            "theta": prior.Sine(),
            "phi": prior.Uniform(0, 2 * constants.PI, boundary="periodic"),
            "psi": prior.Uniform(0, constants.PI),
        }
    )


class TestNestedSampler:
    """Tests for the NestedSampler class."""

    def test_initialisation(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that NestedSampler can be initialised with likelihood, priors and rng."""
        samp = sampler.NestedSampler(likelihood_default, priors_default, rng_default)

        assert isinstance(samp.t_eval, float)
        assert samp.is_restored is False
        assert samp.sampler is not None

    def test_initialisation_with_custom_nlive(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that NestedSampler can be initialised with custom nlive parameter."""
        nlive = 100
        samp = sampler.NestedSampler(likelihood_default, priors_default, rng_default, nlive)

        assert samp.sampler.nlive == nlive

    def test_initialisation_stores_benchmark_time(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that benchmark time is stored during initialisation."""
        samp = sampler.NestedSampler(likelihood_default, priors_default, rng_default)

        assert isinstance(samp.t_eval, float)
        assert samp.t_eval > 0

    def test_calculate_log_likelihood_returns_float(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test that calculate_log_likelihood returns a float."""
        x = numpy.array(
            [
                theta_default.waveform_parameters.m_1,
                theta_default.waveform_parameters.m_2,
                theta_default.waveform_parameters.r,
            ]
        )
        theta_name_sample = ["m_1", "m_2", "r"]
        theta_fixed = {
            "iota": theta_default.waveform_parameters.iota,
            "t_c": theta_default.waveform_parameters.t_c,
            "phi_c": theta_default.waveform_parameters.phi_c,
            "theta": theta_default.detector_angles.theta,
            "phi": theta_default.detector_angles.phi,
            "psi": theta_default.detector_angles.psi,
        }
        ln_l = sampler.NestedSampler.calculate_log_likelihood(x, likelihood_default, theta_name_sample, theta_fixed)

        assert isinstance(ln_l, (float, numpy.floating))

    def test_calculate_log_likelihood_not_nan(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test that calculate_log_likelihood returns finite values."""
        x = numpy.array(
            [
                theta_default.waveform_parameters.m_1,
                theta_default.waveform_parameters.m_2,
                theta_default.waveform_parameters.r,
            ]
        )
        theta_name_sample = ["m_1", "m_2", "r"]
        theta_fixed = {
            "iota": theta_default.waveform_parameters.iota,
            "t_c": theta_default.waveform_parameters.t_c,
            "phi_c": theta_default.waveform_parameters.phi_c,
            "theta": theta_default.detector_angles.theta,
            "phi": theta_default.detector_angles.phi,
            "psi": theta_default.detector_angles.psi,
        }
        ln_l = sampler.NestedSampler.calculate_log_likelihood(x, likelihood_default, theta_name_sample, theta_fixed)

        assert not numpy.isnan(ln_l)
        assert not numpy.isinf(ln_l)

    def test_calculate_log_likelihood_assembles_parameters_correctly(
        self,
        likelihood_default: likelihood.Likelihood,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test that calculate_log_likelihood correctly assembles sampled and fixed parameters."""
        x = numpy.array([theta_default.waveform_parameters.m_1, theta_default.waveform_parameters.m_2])
        theta_name_sample = ["m_1", "m_2"]
        theta_fixed = {
            "r": theta_default.waveform_parameters.r,
            "iota": theta_default.waveform_parameters.iota,
            "t_c": theta_default.waveform_parameters.t_c,
            "phi_c": theta_default.waveform_parameters.phi_c,
            "theta": theta_default.detector_angles.theta,
            "phi": theta_default.detector_angles.phi,
            "psi": theta_default.detector_angles.psi,
        }
        ln_l_static = sampler.NestedSampler.calculate_log_likelihood(
            x, likelihood_default, theta_name_sample, theta_fixed
        )
        ln_l_direct = likelihood_default.calculate_log_pdf(theta_default)

        assert numpy.isclose(ln_l_static, ln_l_direct)

    def test_run_nested(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that run_nested executes without errors."""
        samp = sampler.NestedSampler(likelihood_default, priors_default, rng_default)
        samp.run_nested(delta_ln_z=600)

        assert samp.results is not None

    def test_restore(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        rng_default: numpy.random.Generator,
        tmp_path: Path,
    ) -> None:
        """Test that NestedSampler can be restored from a checkpoint file."""
        samp = sampler.NestedSampler(likelihood_default, priors_default, rng_default)
        checkpoint_file = tmp_path / "checkpoint.save"
        samp.run_nested(delta_ln_z=600, add_live=False, checkpoint_file=str(checkpoint_file))
        samp_restored = sampler.NestedSampler.restore(str(checkpoint_file))

        assert samp_restored.sampler is not None
        assert samp_restored.is_restored is True


class TestBenchmark:
    """Tests for the benchmark function."""

    def test_benchmark_returns_positive_float(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that benchmark returns a positive float."""
        t_eval = sampler.benchmark(likelihood_default, priors_default, n=10, rng=rng_default)

        assert isinstance(t_eval, float)
        assert t_eval > 0

    def test_benchmark_no_rng(self, likelihood_default: likelihood.Likelihood, priors_default: prior.Priors) -> None:
        """Test that benchmark works without providing rng."""
        t_eval = sampler.benchmark(likelihood_default, priors_default, n=10)

        assert isinstance(t_eval, float)
        assert t_eval > 0

    def test_benchmark_with_custom_n(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that benchmark works with custom number of evaluations."""
        t_eval = sampler.benchmark(likelihood_default, priors_default, n=5, rng=rng_default)

        assert isinstance(t_eval, float)
        assert t_eval > 0
