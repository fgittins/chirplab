"""Module for sampling algorithms."""

import logging
import multiprocessing
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import dynesty
import numpy

from chirplab.inference import pool, results

if TYPE_CHECKING:
    from dynesty import internal_samplers

    from chirplab.inference import likelihood, prior

type BoundaryType = Literal["none", "single", "multi", "balls", "cubes"] | dynesty.bounding.Bound
type SampleType = Literal["auto", "unif", "rwalk", "slice", "rslice"] | internal_samplers.InternalSampler

logger = logging.getLogger(__name__)


class FirstUpdateDict(TypedDict, total=False):
    """Dictionary for first update parameters."""

    min_ncall: int
    min_eff: float


# TODO: adjust printing frequency
def run(
    likelihood: likelihood.Likelihood,
    prior: prior.Prior,
    nlive: int = 500,
    bound: BoundaryType = "multi",
    sample: SampleType = "auto",
    update_interval: None | int | float = None,
    first_update: None | FirstUpdateDict = None,
    rng: None | numpy.random.Generator = None,
    njobs: int = 1,
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
    resume: bool = True,
    results_filename: None | str = None,
) -> dynesty.results.Results:
    """
    Run the nested sampler.

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
        The number of multiprocessing jobs.
    enlarge
        Enlarge the volumes of the specified bounding object(s) by this fraction.
    bootstrap
        Compute this many bootstrapped realizations of the bounding objects.
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
    resume
        Whether to resume the sampler from a previous checkpoint file.
    results_filename
        HDF5 file where the results will be saved.

    Returns
    -------
    results
        Sampling results.
    """
    sampler_kwargs: dict[str, Any] = {
        "nlive": nlive,
        "bound": bound,
        "sample": sample,
        "periodic": prior.periodic_indices,
        "reflective": prior.reflective_indices,
        "update_interval": update_interval,
        "first_update": first_update,
        "rstate": rng,
        "enlarge": enlarge,
        "bootstrap": bootstrap,
        "walks": walks,
        "facc": facc,
        "slices": slices,
        "ncdim": ncdim,
    }

    run_kwargs: dict[str, Any] = {
        "maxiter": maxiter,
        "maxcall": maxcall,
        "dlogz": dlogz,
        "logl_max": logl_max,
        "add_live": add_live,
        "print_progress": print_progress,
        "save_bounds": save_bounds,
        "checkpoint_file": checkpoint_file,
        "checkpoint_every": checkpoint_every,
        "resume": resume,
    }

    _log_run_parameters(
        likelihood=likelihood, prior=prior, **sampler_kwargs, **run_kwargs, results_filename=results_filename
    )

    t = benchmark(likelihood, prior)

    logger.debug("Likelihood benchmark: average log-likelihood evaluation time = %.3e s", t)

    is_multiprocessed = njobs > 1
    is_resumed = resume and checkpoint_file is not None and Path(checkpoint_file).is_file()

    if is_multiprocessed:
        logger.info("Initialising multiprocessing pool with %d jobs", njobs)

        loglikelihood = pool._calculate_log_likelihood_wrapper
        prior_transform = pool._transform_prior_wrapper
        process_pool = multiprocessing.Pool(njobs, pool._initialiser, (likelihood.calculate_log_pdf, prior.transform))
        queue_size = njobs
    else:
        loglikelihood = likelihood.calculate_log_pdf
        prior_transform = prior.transform
        process_pool = None
        queue_size = None

    t_1 = time.time()

    if is_resumed:
        assert checkpoint_file is not None
        sampler = dynesty.NestedSampler.restore(checkpoint_file, pool=process_pool)

        logger.info("Resumed nested sampling run from checkpoint file '%s'", checkpoint_file)
    else:
        sampler = dynesty.NestedSampler(
            loglikelihood, prior_transform, prior.n_dim, pool=process_pool, queue_size=queue_size, **sampler_kwargs
        )

    logger.info("Starting nested sampling run (%s mode)", "multiprocessing" if is_multiprocessed else "single-process")

    sampler.run_nested(**run_kwargs)

    t_2 = time.time()

    if is_multiprocessed:
        logger.info("Closing multiprocessing pool with %d jobs", njobs)

        assert process_pool is not None
        process_pool.close()
        process_pool.join()

    if print_progress:
        print()

    run_results = sampler.results
    _log_results_summary(run_results, t_2 - t_1)

    if results_filename is not None:
        results._save(run_results, results_filename)

    return run_results


def _log_run_parameters(likelihood: likelihood.Likelihood, prior: prior.Prior, **kwargs: object) -> None:
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())

    logger.info("Starting nested sampling with arguments: %s", args)
    logger.info("Likelihood function: %s", likelihood)
    logger.info("Prior distribution: %s", prior)


def _log_results_summary(results: dynesty.results.Results, t_elapsed: float) -> None:
    logger.info(
        "Nested sampling completed in %.2f s: niter=%d, ncall=%d, eff(%%)=%6.3f, logz=%6.3f +/- %6.3f",
        t_elapsed,
        results.niter,
        numpy.sum(results.ncall),
        results.eff,
        results.logz[-1],
        results.logzerr[-1],
    )


def benchmark(
    likelihood: likelihood.Likelihood, prior: prior.Prior, n: int = 1_000, rng: numpy.random.Generator | None = None
) -> float:
    """
    Benchmark the log of the likelihood function evaluation time.

    Parameters
    ----------
    likelihood
        Likelihood function.
    prior
        Prior distribution.
    n
        Number of evaluations to average over.
    rng
        Random number generator for the sampling.

    Returns
    -------
    t
        Average time per log-likelihood evaluation (s).
    """
    x_list = [prior.sample(rng) for _ in range(n)]

    t_1 = time.time()
    for x in x_list:
        likelihood.calculate_log_pdf(x)
    t_2 = time.time()

    return (t_2 - t_1) / n
