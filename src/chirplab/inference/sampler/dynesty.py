"""Module for dynesty sampler."""

import logging
import multiprocessing
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TypedDict

import dynesty
import numpy

from chirplab.inference.sampler import base

if TYPE_CHECKING:
    from dynesty import internal_samplers

    from chirplab.inference import likelihood, prior

logger = logging.getLogger(__name__)

type BoundaryType = Literal["none", "single", "multi", "balls", "cubes"] | dynesty.bounding.Bound
type SampleType = Literal["auto", "unif", "rwalk", "slice", "rslice"] | internal_samplers.InternalSampler


class FirstUpdateDict(TypedDict, total=False):
    """Dictionary for first_update parameters."""

    min_ncall: int
    min_eff: float


class UsePoolDict(TypedDict, total=False):
    """Dictionary for use_pool parameters."""

    prior_transform: bool
    loglikelihood: bool
    propose_point: bool
    update_bound: bool


class Dynesty(base.Sampler):
    """
    Dynesty sampler.

    Parameters
    ----------
    likelihood
        Likelihood function.
    prior
        Prior distribution.
    nlive
        Number of live points.
    bound
        Method used to approximately bound the prior using the current set of live points.
    sample
        Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds.
    update_interval
        If an integer is passed, only update the proposal distribution every `update_interval`-th likelihood call. If a
        float is passed, update the proposal after every `round(update_interval * nlive)`-th likelihood call.
    first_update
        A dictionary containing parameters governing when the sampler should first update the bounding distribution from
        the unit cube (`"none"`) to the one specified by `sample`.
    rng
        Random number generator for the sampling.
    njobs
        Number of parallel workers.
    pool
        Use this pool of workers to execute operations in parallel.
    use_pool
        A dictionary containing flags indicating where a pool should be used to execute operations in parallel.
    enlarge
        Enlarge the volumes of the specified bounding object(s) by this fraction.
    bootstrap
        Compute this many bootstrapped realisations of the bounding objects.
    walks
        For the `"rwalk"` sampling option, the minimum number of steps before proposing a new live point.
    facc
        The target acceptance fraction for the `"rwalk"` sampling option.
    slices
        For the `"slice"`, `"rslice"` sampling options, the number of times to execute a "slice update" before proposing
        a new live point.
    ncdim
        The number of clustering dimensions.
    """

    def __init__(
        self,
        likelihood: likelihood.Likelihood,
        prior: prior.Prior,
        nlive: int = 500,
        bound: BoundaryType = "multi",
        sample: SampleType = "auto",
        update_interval: None | int | float = None,
        first_update: None | FirstUpdateDict = None,
        rng: None | numpy.random.Generator = None,
        njobs: int = 1,
        pool: None | multiprocessing.pool.Pool = None,
        use_pool: None | UsePoolDict = None,
        enlarge: None | float = None,
        bootstrap: None | int = None,
        walks: None | int = None,
        facc: float = 0.5,
        slices: None | int = None,
        ncdim: None | int = None,
    ) -> None:
        super().__init__(likelihood, prior, rng)

        logger.info(
            "Initialising dynesty sampler with nlive=%d, bound='%s', sample='%s', update_interval=%s, "
            "first_update=%s, rng=%s, pool=%s, use_pool=%s, enlarge=%s, bootstrap=%s, walks=%s, facc=%s, slices=%s, ncdim=%s",
            nlive,
            bound,
            sample,
            update_interval,
            first_update,
            rng,
            pool,
            use_pool,
            enlarge,
            bootstrap,
            walks,
            facc,
            slices,
            ncdim,
        )

        self.sampler = dynesty.NestedSampler(
            likelihood.calculate_log_pdf,
            prior.transform,
            prior.n_dim,
            nlive=nlive,
            bound=bound,
            sample=sample,
            periodic=prior.periodic_indices,
            reflective=prior.reflective_indices,
            update_interval=update_interval,
            first_update=first_update,
            rstate=rng,
            pool=pool,
            use_pool=use_pool,
            queue_size=njobs,
            enlarge=enlarge,
            bootstrap=bootstrap,
            walks=walks,
            facc=facc,
            slices=slices,
            ncdim=ncdim,
        )

    @classmethod
    def restore(cls, checkpoint_file: str, pool: None | multiprocessing.pool.Pool = None) -> Self:
        """
        Restore the dynesty sampler from a checkpoint file.

        Parameters
        ----------
        checkpoint_file
            Checkpoint file.
        """
        obj = cls.__new__(cls)
        obj.sampler = dynesty.NestedSampler.restore(checkpoint_file, pool=pool)

        logger.info("Resumed nested sampling run from checkpoint file '%s'", checkpoint_file)

        return obj

    def run(
        self,
        maxiter: None | int = None,
        maxcall: None | int = None,
        dlogz: None | float = None,
        logl_max: float = numpy.inf,
        add_live: bool = True,
        print_progress: bool = True,
        save_bounds: bool = True,
        checkpoint_file: None | str = None,
        checkpoint_every: int = 60,
        resume: bool = False,
    ) -> None:
        """
        Run the dynesty sampler.

        Parameters
        ----------
        maxiter
            Maximum number of iterations.
        maxcall
            Maximum number of likelihood evaluations.
        dlogz
            Iteration will stop when the estimated contribution of the remaining prior volume to the total evidence
            falls below this threshold.
        logl_max
            Iteration will stop when the sampled ln(likelihood) exceeds the threshold set by `logl_max`.
        add_live
            Whether or not to add the remaining set of live points to the list of samples at the end of each run.
        print_progress
            Whether or not to output a simple summary of the current run that updates with each iteration.
        save_bounds
            Whether or not to save past bounding distributions used to bound the live points internally.
        checkpoint_file
            The state of the sampler will be saved into this file every `checkpoint_every` seconds.
        checkpoint_every
            The number of seconds between checkpoints that will save the internal state of the sampler.
        resume
            Whether to resume the sampler from a previous checkpoint file.

        Returns
        -------
        result
            Sampling result.
        """
        t_1 = time.time()

        self.sampler.run_nested(
            maxiter=maxiter,
            maxcall=maxcall,
            dlogz=dlogz,
            logl_max=logl_max,
            add_live=add_live,
            print_progress=print_progress,
            save_bounds=save_bounds,
            checkpoint_file=checkpoint_file,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )

        t_2 = time.time()

        results = self.sampler.results

        logger.info(
            "Nested sampling completed in %.2f s: niter=%d, ncall=%d, eff(%%)=%6.3f, logz=%6.3f +/- %6.3f",
            t_2 - t_1,
            results.niter,
            numpy.sum(results.ncall),
            results.eff,
            results.logz[-1],
            results.logzerr[-1],
        )

        self.result = base.Result(
            logl=results.logl,
            samples_it=results.samples_it,
            samples_id=results.samples_id,
            samples_u=results.samples_u,
            samples=results.samples,
            niter=results.niter,
            ncall=results.ncall,
            logz=results.logz,
            logzerr=results.logzerr,
            logwt=results.logwt,
            eff=results.eff,
            nlive=results.nlive,
            logvol=results.logvol,
            information=results.information,
            bound_iter=results.bound_iter,
            samples_bound=results.samples_bound,
            scale=results.scale,
        )


def run_sampler(
    likelihood: likelihood.Likelihood,
    prior: prior.Prior,
    nlive: int = 500,
    bound: BoundaryType = "multi",
    sample: SampleType = "auto",
    update_interval: None | int | float = None,
    first_update: None | FirstUpdateDict = None,
    rng: None | numpy.random.Generator = None,
    njobs: int = 1,
    use_pool: None | UsePoolDict = None,
    enlarge: None | float = None,
    bootstrap: None | int = None,
    walks: None | int = None,
    facc: float = 0.5,
    slices: None | int = None,
    ncdim: None | int = None,
    maxiter: None | int = None,
    maxcall: None | int = None,
    dlogz: None | float = None,
    logl_max: float = numpy.inf,
    add_live: bool = True,
    print_progress: bool = True,
    save_bounds: bool = True,
    checkpoint_file: None | str = None,
    checkpoint_every: int = 60,
) -> base.Result:
    """
    Run the dynesty sampler.

    Parameters
    ----------
    likelihood
        Likelihood function.
    prior
        Prior distribution.
    nlive
        Number of live points.
    bound
        Method used to approximately bound the prior using the current set of live points.
    sample
        Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds.
    update_interval
        If an integer is passed, only update the proposal distribution every `update_interval`-th likelihood call. If a
        float is passed, update the proposal after every `round(update_interval * nlive)`-th likelihood call.
    first_update
        A dictionary containing parameters governing when the sampler should first update the bounding distribution from
        the unit cube (`"none"`) to the one specified by `sample`.
    rng
        Random number generator for the sampling.
    njobs
        Number of parallel workers.
    use_pool
        A dictionary containing flags indicating where a pool should be used to execute operations in parallel.
    enlarge
        Enlarge the volumes of the specified bounding object(s) by this fraction.
    bootstrap
        Compute this many bootstrapped realisations of the bounding objects.
    walks
        For the `"rwalk"` sampling option, the minimum number of steps before proposing a new live point.
    facc
        The target acceptance fraction for the `"rwalk"` sampling option.
    slices
        For the `"slice"`, `"rslice"` sampling options, the number of times to execute a "slice update" before proposing
        a new live point.
    ncdim
        The number of clustering dimensions.
    maxiter
        Maximum number of iterations.
    maxcall
        Maximum number of likelihood evaluations.
    dlogz
        Iteration will stop when the estimated contribution of the remaining prior volume to the total evidence falls
        below this threshold.
    logl_max
        Iteration will stop when the sampled ln(likelihood) exceeds the threshold set by `logl_max`.
    add_live
        Whether or not to add the remaining set of live points to the list of samples at the end of each run.
    print_progress
        Whether or not to output a simple summary of the current run that updates with each iteration.
    save_bounds
        Whether or not to save past bounding distributions used to bound the live points internally.
    checkpoint_file
        The state of the sampler will be saved into this file every `checkpoint_every` seconds.
    checkpoint_every
        The number of seconds between checkpoints that will save the internal state of the sampler.

    Returns
    -------
    result
        Sampling result.
    """
    is_multiprocessed = njobs > 1
    is_resumed = checkpoint_file is not None and Path(checkpoint_file).exists()

    pool = multiprocessing.Pool(njobs) if is_multiprocessed else None

    if is_resumed:
        assert checkpoint_file is not None
        sampler = Dynesty.restore(checkpoint_file, pool=pool)
    else:
        sampler = Dynesty(
            likelihood,
            prior,
            nlive,
            bound,
            sample,
            update_interval,
            first_update,
            rng,
            njobs,
            pool,
            use_pool,
            enlarge,
            bootstrap,
            walks,
            facc,
            slices,
            ncdim,
        )

    sampler.run(
        maxiter,
        maxcall,
        dlogz,
        logl_max,
        add_live,
        print_progress,
        save_bounds,
        checkpoint_file,
        checkpoint_every,
        is_resumed,
    )

    if is_multiprocessed:
        assert pool is not None
        pool.close()
        pool.join()

    assert sampler.result is not None
    return sampler.result
