"""Tests for the sampler module."""

from typing import TYPE_CHECKING

import numpy
import pytest

from chirplab import constants
from chirplab.inference import likelihood, prior, sampler

if TYPE_CHECKING:
    from chirplab.simulation import interferometer, waveform


@pytest.fixture(scope="class")
def priors_default() -> prior.Priors:
    """Return default priors for testing."""
    return prior.Priors(
        m_1=prior.Uniform(20 * constants.M_SUN, 40 * constants.M_SUN),
        m_2=prior.Uniform(20 * constants.M_SUN, 40 * constants.M_SUN),
        r=prior.Uniform(400e6 * constants.PC, 600e6 * constants.PC),
        iota=prior.Sine(),
        t_c=prior.Uniform(99, 101),
        phi_c=prior.Uniform(0, 2 * constants.PI, boundary="periodic"),
        theta=prior.Cosine(),
        phi=prior.Uniform(0, 2 * constants.PI, boundary="periodic"),
        psi=prior.Uniform(0, constants.PI),
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
        samp = sampler.NestedSampler(likelihood_default, priors_default, rng=rng_default)

        assert samp.sampler is not None
        assert samp.rng == rng_default
        assert isinstance(samp.t_eval, float)

    def test_initialisation_no_rng(
        self, likelihood_default: likelihood.Likelihood, priors_default: prior.Priors
    ) -> None:
        """Test that NestedSampler can be initialised without providing rng."""
        samp = sampler.NestedSampler(likelihood_default, priors_default)

        assert samp.rng is not None
        assert isinstance(samp.rng, numpy.random.Generator)

    def test_initialisation_with_custom_nlive(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that NestedSampler can be initialised with custom nlive parameter."""
        nlive = 100
        samp = sampler.NestedSampler(likelihood_default, priors_default, rng=rng_default, nlive=nlive)

        assert samp.sampler.nlive == nlive

    def test_initialisation_stores_benchmark_time(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that benchmark time is stored during initialisation."""
        samp = sampler.NestedSampler(likelihood_default, priors_default, rng=rng_default)

        assert samp.t_eval > 0

    def test_calculate_log_likelihood_returns_float(
        self,
        likelihood_default: likelihood.Likelihood,
        priors_default: prior.Priors,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test that calculate_log_likelihood returns a float."""
        x = numpy.array([theta_default.m_1, theta_default.m_2, theta_default.r])
        theta_name_sample = ["m_1", "m_2", "r"]
        theta_fixed = {
            "iota": theta_default.iota,
            "t_c": theta_default.t_c,
            "phi_c": theta_default.phi_c,
            "theta": theta_default.theta,
            "phi": theta_default.phi,
            "psi": theta_default.psi,
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
        x = numpy.array([theta_default.m_1, theta_default.m_2, theta_default.r])
        theta_name_sample = ["m_1", "m_2", "r"]
        theta_fixed = {
            "iota": theta_default.iota,
            "t_c": theta_default.t_c,
            "phi_c": theta_default.phi_c,
            "theta": theta_default.theta,
            "phi": theta_default.phi,
            "psi": theta_default.psi,
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
        x = numpy.array([theta_default.m_1, theta_default.m_2])
        theta_name_sample = ["m_1", "m_2"]
        theta_fixed = {
            "r": theta_default.r,
            "iota": theta_default.iota,
            "t_c": theta_default.t_c,
            "phi_c": theta_default.phi_c,
            "theta": theta_default.theta,
            "phi": theta_default.phi,
            "psi": theta_default.psi,
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
        samp = sampler.NestedSampler(likelihood_default, priors_default, rng=rng_default)
        samp.run_nested(delta_ln_z=10_000)

        assert samp.results is not None


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
