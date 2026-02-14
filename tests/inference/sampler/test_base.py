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
        rng_default.standard_normal((n_samples, n_dims)),
        rng_default.standard_normal(n_samples),
        rng_default.standard_normal(n_samples),
        rng_default.standard_normal(),
        rng_default.standard_normal(),
    )


class TestResult:
    """Tests for the Result dataclass."""

    def test_result_creation(self, result_default: base.Result) -> None:
        """Test that a Result can be created."""
        assert isinstance(result_default, base.Result)

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

        assert result.ln_l is not None
        assert result_default.ln_l is not None
        assert result.ln_z is not None
        assert result_default.ln_z is not None
        assert result.delta_ln_z is not None
        assert result_default.delta_ln_z is not None

        assert numpy.array_equal(result.x, result_default.x)
        assert numpy.array_equal(result.w, result_default.w)
        assert numpy.array_equal(result.ln_l, result_default.ln_l)
        assert numpy.array_equal(result.ln_z, result_default.ln_z)
        assert numpy.array_equal(result.delta_ln_z, result_default.delta_ln_z)

    def test_result_arrays_have_correct_shape(self, result_default: base.Result) -> None:
        """Test that Result arrays have the expected shapes."""
        n_samples = 100
        n_dims = 9

        assert result_default.x.shape == (n_samples, n_dims)
        assert result_default.w.shape == (n_samples,)
        assert result_default.ln_l is not None
        assert result_default.ln_l.shape == (n_samples,)
        assert result_default.ln_z is not None
        assert isinstance(result_default.ln_z, float)
        assert result_default.delta_ln_z is not None
        assert isinstance(result_default.delta_ln_z, float)


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
