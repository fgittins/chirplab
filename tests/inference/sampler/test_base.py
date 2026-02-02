"""Tests for the base module."""

from typing import TYPE_CHECKING

import numpy
import pytest

from chirplab.inference.sampler import base

if TYPE_CHECKING:
    from pathlib import Path

    from chirplab.inference import likelihood, prior


@pytest.fixture
def result_default(rng_default: numpy.random.Generator) -> base.Result:
    """Return a default Result instance for testing."""
    n_samples = 100
    n_dims = 9

    return base.Result(
        rng_default.standard_normal(n_samples),
        numpy.arange(n_samples),
        numpy.arange(n_samples),
        rng_default.random((n_samples, n_dims)),
        rng_default.random((n_samples, n_dims)),
        n_samples,
        numpy.arange(n_samples) * 10,
        rng_default.standard_normal(n_samples),
        numpy.abs(rng_default.standard_normal(n_samples)),
        rng_default.standard_normal(n_samples),
        0.25,
        500,
        rng_default.standard_normal(n_samples),
        rng_default.standard_normal(n_samples),
        numpy.zeros(n_samples, dtype=numpy.int64),
        numpy.zeros(n_samples, dtype=numpy.int64),
        numpy.ones(n_samples),
    )


class TestResult:
    """Tests for the Result dataclass."""

    def test_result_creation(self, result_default: base.Result) -> None:
        """Test that a Result can be created with all required fields."""
        assert isinstance(result_default, base.Result)
        assert result_default.niter == 100
        assert result_default.nlive == 500
        assert result_default.eff == 0.25

    def test_result_save(self, result_default: base.Result, tmp_path: Path) -> None:
        """Test that a Result can be saved to an HDF5 file."""
        filepath = tmp_path / "test_result.h5"
        result_default.save(str(filepath))

        assert filepath.exists()

    def test_result_load(self, result_default: base.Result, tmp_path: Path) -> None:
        """Test that a Result can be loaded from an HDF5 file."""
        filepath = tmp_path / "test_result.h5"
        result_default.save(str(filepath))
        result = base.Result.load(str(filepath))

        assert isinstance(result, base.Result)

    def test_result_save_load_roundtrip(self, result_default: base.Result, tmp_path: Path) -> None:
        """Test that saving and loading a Result preserves all data."""
        filepath = tmp_path / "test_result.h5"
        result_default.save(str(filepath))
        result = base.Result.load(str(filepath))

        assert result.niter == result_default.niter
        assert result.nlive == result_default.nlive
        assert result.eff == result_default.eff

        assert numpy.array_equal(result.logl, result_default.logl)
        assert numpy.array_equal(result.samples_it, result_default.samples_it)
        assert numpy.array_equal(result.samples_id, result_default.samples_id)
        assert numpy.array_equal(result.samples_u, result_default.samples_u)
        assert numpy.array_equal(result.samples, result_default.samples)
        assert numpy.array_equal(result.ncall, result_default.ncall)
        assert numpy.array_equal(result.logz, result_default.logz)
        assert numpy.array_equal(result.logzerr, result_default.logzerr)
        assert numpy.array_equal(result.logwt, result_default.logwt)
        assert numpy.array_equal(result.logvol, result_default.logvol)
        assert numpy.array_equal(result.information, result_default.information)
        assert numpy.array_equal(result.bound_iter, result_default.bound_iter)
        assert numpy.array_equal(result.samples_bound, result_default.samples_bound)
        assert numpy.array_equal(result.scale, result_default.scale)

    def test_result_arrays_have_correct_shape(self, result_default: base.Result) -> None:
        """Test that Result arrays have the expected shapes."""
        n_samples = 100
        n_dims = 9

        assert result_default.logl.shape == (n_samples,)
        assert result_default.samples_it.shape == (n_samples,)
        assert result_default.samples_id.shape == (n_samples,)
        assert result_default.samples_u.shape == (n_samples, n_dims)
        assert result_default.samples.shape == (n_samples, n_dims)
        assert result_default.ncall.shape == (n_samples,)
        assert result_default.logz.shape == (n_samples,)
        assert result_default.logzerr.shape == (n_samples,)
        assert result_default.logwt.shape == (n_samples,)
        assert result_default.logvol.shape == (n_samples,)
        assert result_default.information.shape == (n_samples,)
        assert result_default.bound_iter.shape == (n_samples,)
        assert result_default.samples_bound.shape == (n_samples,)
        assert result_default.scale.shape == (n_samples,)


class TestBenchmark:
    """Tests for the benchmark function."""

    def test_benchmark_returns_positive_float(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that benchmark returns a positive float."""
        t_eval = base.benchmark(likelihood_default, prior_default, n=10, rng=rng_default)

        assert isinstance(t_eval, float)
        assert t_eval > 0

    def test_benchmark_no_rng(self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior) -> None:
        """Test that benchmark works without providing rng."""
        t_eval = base.benchmark(likelihood_default, prior_default, n=10)

        assert isinstance(t_eval, float)
        assert t_eval > 0

    def test_benchmark_with_custom_n(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that benchmark works with custom number of evaluations."""
        t_eval = base.benchmark(likelihood_default, prior_default, n=5, rng=rng_default)

        assert isinstance(t_eval, float)
        assert t_eval > 0
