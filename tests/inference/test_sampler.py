"""Tests for the sampler module."""

from typing import TYPE_CHECKING

import pytest

from chirplab import constants
from chirplab.inference import distribution, likelihood, prior, sampler

if TYPE_CHECKING:
    from pathlib import Path

    import numpy


@pytest.fixture(scope="class")
def prior_default() -> prior.Prior:
    """Return default prior for testing."""
    return prior.Prior(
        (
            distribution.Uniform(20 * constants.M_SUN, 40 * constants.M_SUN),
            distribution.Uniform(20 * constants.M_SUN, 40 * constants.M_SUN),
            distribution.Uniform(400e6 * constants.PC, 600e6 * constants.PC),
            distribution.Sine(),
            distribution.Uniform(99, 101),
            distribution.Uniform(0, 2 * constants.PI, boundary="periodic"),
            distribution.Sine(),
            distribution.Uniform(0, 2 * constants.PI, boundary="periodic"),
            distribution.Uniform(0, constants.PI),
        )
    )


class TestNestedSampler:
    """Tests for the NestedSampler class."""

    def test_initialisation(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that NestedSampler can be initialised with likelihood, prior and rng."""
        samp = sampler.NestedSampler(likelihood_default, prior_default, rng_default)
        assert isinstance(samp.t_eval, float)
        assert samp.is_restored is False
        assert samp.sampler is not None

    def test_initialisation_with_custom_nlive(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that NestedSampler can be initialised with custom nlive parameter."""
        nlive = 100
        samp = sampler.NestedSampler(likelihood_default, prior_default, rng_default, nlive)

        assert samp.sampler.nlive == nlive

    def test_initialisation_stores_benchmark_time(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that benchmark time is stored during initialisation."""
        samp = sampler.NestedSampler(likelihood_default, prior_default, rng_default)

        assert isinstance(samp.t_eval, float)
        assert samp.t_eval > 0

    def test_run_nested(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that run_nested executes without errors."""
        samp = sampler.NestedSampler(likelihood_default, prior_default, rng_default)
        samp.run_nested(dlogz=600)

        assert samp.results is not None

    def test_restore(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: Path,
    ) -> None:
        """Test that NestedSampler can be restored from a checkpoint file."""
        samp = sampler.NestedSampler(likelihood_default, prior_default, rng_default)
        checkpoint_file = tmp_path / "checkpoint.save"
        samp.run_nested(dlogz=600, add_live=False, checkpoint_file=str(checkpoint_file))
        samp_restored = sampler.NestedSampler.restore(str(checkpoint_file))

        assert samp_restored.sampler is not None
        assert samp_restored.is_restored is True


class TestBenchmark:
    """Tests for the benchmark function."""

    def test_benchmark_returns_positive_float(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that benchmark returns a positive float."""
        t_eval = sampler.benchmark(likelihood_default, prior_default, n=10, rng=rng_default)

        assert isinstance(t_eval, float)
        assert t_eval > 0

    def test_benchmark_no_rng(self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior) -> None:
        """Test that benchmark works without providing rng."""
        t_eval = sampler.benchmark(likelihood_default, prior_default, n=10)

        assert isinstance(t_eval, float)
        assert t_eval > 0

    def test_benchmark_with_custom_n(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that benchmark works with custom number of evaluations."""
        t_eval = sampler.benchmark(likelihood_default, prior_default, n=5, rng=rng_default)

        assert isinstance(t_eval, float)
        assert t_eval > 0
