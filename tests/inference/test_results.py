"""Tests for the results module."""

from typing import TYPE_CHECKING

import dynesty

from chirplab.inference import likelihood, prior, results, sampler

if TYPE_CHECKING:
    from pathlib import Path

    import numpy


class TestSaveResults:
    """Tests for the _save function."""

    def test_save_creates_file(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: Path,
    ) -> None:
        """Test that _save creates an HDF5 file."""
        run_results = sampler.run(likelihood_default, prior_default, rng=rng_default, maxiter=100)
        results_file = tmp_path / "results.hdf5"
        results._save(run_results, str(results_file))

        assert results_file.exists()


class TestReadResults:
    """Tests for the load function."""

    def test_load_returns_results(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: Path,
    ) -> None:
        """Test that load returns a results instance."""
        results_file = tmp_path / "results.hdf5"
        sampler.run(likelihood_default, prior_default, rng=rng_default, maxiter=100, results_filename=str(results_file))
        run_results = results.load(str(results_file))

        assert isinstance(run_results, dynesty.results.Results)
