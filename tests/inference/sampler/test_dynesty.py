"""Tests for the dynesty module."""

from pathlib import Path
from typing import TYPE_CHECKING

from chirplab.inference.sampler import base, dynesty

if TYPE_CHECKING:
    import numpy

    from chirplab.inference import likelihood, prior


class TestDynesty:
    """Tests for the Dynesty sampler class."""

    def test_init(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that Dynesty sampler can be initialised."""
        sampler = dynesty.Dynesty(likelihood_default, prior_default, rng=rng_default)

        assert sampler.sampler is not None
        assert sampler.result is None

    def test_init_with_bound_options(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test Dynesty sampler with different bound options."""
        for bound in ["none", "single", "multi", "balls", "cubes"]:
            sampler = dynesty.Dynesty(likelihood_default, prior_default, bound=bound, rng=rng_default)  # type: ignore[arg-type]

            assert sampler.sampler is not None

    def test_init_with_sample_options(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test Dynesty sampler with different sample options."""
        for sample in ["unif", "rwalk", "slice", "rslice"]:
            sampler = dynesty.Dynesty(likelihood_default, prior_default, sample=sample, rng=rng_default)  # type: ignore[arg-type]

            assert sampler.sampler is not None

    def test_run(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that Dynesty sampler can run with minimal iterations."""
        sampler = dynesty.Dynesty(likelihood_default, prior_default, rng=rng_default)
        sampler.run(maxiter=10, print_progress=False)

        assert sampler.result is not None
        assert isinstance(sampler.result, base.Result)

    def test_run_with_dlogz(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test Dynesty sampler run with dlogz stopping criterion."""
        sampler = dynesty.Dynesty(likelihood_default, prior_default, rng=rng_default)
        sampler.run(maxiter=100, print_progress=False)

        assert sampler.result is not None

    def test_run_with_checkpoint(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: Path,
    ) -> None:
        """Test Dynesty sampler run with checkpointing."""
        checkpoint_file = str(tmp_path / "checkpoint.save")
        sampler = dynesty.Dynesty(likelihood_default, prior_default, rng=rng_default)
        sampler.run(maxiter=100, print_progress=False, checkpoint_file=checkpoint_file, checkpoint_every=1)

        assert sampler.result is not None

    def test_restore(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: Path,
    ) -> None:
        """Test that Dynesty sampler can be restored from a checkpoint."""
        checkpoint_file = str(tmp_path / "checkpoint.save")
        sampler = dynesty.Dynesty(likelihood_default, prior_default, rng=rng_default)
        sampler.run(maxiter=100, print_progress=False, checkpoint_file=checkpoint_file, checkpoint_every=1)
        restored_sampler = dynesty.Dynesty.restore(checkpoint_file)

        assert restored_sampler.sampler is not None


class TestRunSampler:
    """Tests for the run_sampler function."""

    def test_run_sampler_basic(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test that run_sampler returns a Result."""
        result = dynesty.run_sampler(
            likelihood_default, prior_default, nlive=50, rng=rng_default, maxiter=10, print_progress=False
        )

        assert isinstance(result, base.Result)

    def test_run_sampler_with_checkpoint_resume(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: Path,
    ) -> None:
        """Test run_sampler with checkpoint and resume functionality."""
        checkpoint_file = str(tmp_path / "checkpoint.save")
        result = dynesty.run_sampler(
            likelihood_default,
            prior_default,
            rng=rng_default,
            maxiter=100,
            print_progress=False,
            checkpoint_file=checkpoint_file,
            checkpoint_every=1,
        )

        assert isinstance(result, base.Result)
        assert Path(checkpoint_file).exists()

    def test_run_sampler_resume(
        self,
        likelihood_default: likelihood.Likelihood,
        prior_default: prior.Prior,
        rng_default: numpy.random.Generator,
        tmp_path: Path,
    ) -> None:
        """Test run_sampler resume from checkpoint."""
        checkpoint_file = str(tmp_path / "checkpoint.save")
        dynesty.run_sampler(
            likelihood_default,
            prior_default,
            rng=rng_default,
            maxiter=100,
            print_progress=False,
            checkpoint_file=checkpoint_file,
            checkpoint_every=1,
        )
        result = dynesty.run_sampler(
            likelihood_default,
            prior_default,
            rng=rng_default,
            maxiter=200,
            print_progress=False,
            checkpoint_file=checkpoint_file,
        )

        assert isinstance(result, base.Result)

    def test_run_sampler_njobs(
        self, likelihood_default: likelihood.Likelihood, prior_default: prior.Prior, rng_default: numpy.random.Generator
    ) -> None:
        """Test run_sampler with multiple jobs."""
        result = dynesty.run_sampler(
            likelihood_default, prior_default, rng=rng_default, njobs=2, maxiter=100, print_progress=False
        )

        assert isinstance(result, base.Result)
