"""Tests for the sampler module."""

from typing import TYPE_CHECKING

import dynesty
import pytest

from chirplab import constants
from chirplab.inference import distribution, likelihood, prior, sampler

if TYPE_CHECKING:
    import pathlib

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
        tmp_path: pathlib.Path,
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

    def test_save_results(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that results are saved to a file."""
        results_file = tmp_path / "results.hdf5"
        sampler.run(likelihood_default, prior_default, rng=rng_default, maxiter=100, results_filename=str(results_file))

        assert results_file.exists()


class TestReadResults:
    """Tests for the load_results function."""

    def test_load_results_returns_results(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that load_results returns a results instance."""
        results_file = tmp_path / "results.hdf5"
        sampler.run(likelihood_default, prior_default, rng=rng_default, maxiter=100, results_filename=str(results_file))
        results = sampler.load_results(str(results_file))

        assert isinstance(results, dynesty.results.Results)


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
