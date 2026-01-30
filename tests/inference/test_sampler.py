"""Tests for the sampler module."""

from typing import TYPE_CHECKING

import dynesty

from chirplab.inference import likelihood, prior, sampler

if TYPE_CHECKING:
    from pathlib import Path

    import numpy


class TestRun:
    """Tests for the run function."""

    def test_run_returns_results(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that run returns a results instance."""
        results = sampler.run(likelihood_default, prior_default, rng=rng_default, maxiter=100)

        assert isinstance(results, dynesty.results.Results)

    def test_run_with_multiprocessing(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that run works with multiple jobs."""
        results = sampler.run(likelihood_default, prior_default, rng=rng_default, njobs=2, maxiter=100)

        assert isinstance(results, dynesty.results.Results)

    def test_resume(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: Path,
    ) -> None:
        """Test that resuming from a checkpoint works."""
        checkpoint_file = tmp_path / "checkpoint.save"
        results_1 = sampler.run(
            likelihood_default,
            prior_default,
            rng=rng_default,
            maxiter=50,
            add_live=False,
            checkpoint_file=str(checkpoint_file),
        )

        assert checkpoint_file.exists()

        results_2 = sampler.run(
            likelihood_default, prior_default, rng=rng_default, maxiter=100, checkpoint_file=str(checkpoint_file)
        )

        assert results_2.niter > results_1.niter


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
