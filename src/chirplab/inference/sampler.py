"""Module for sampling algorithms."""

import logging
import multiprocessing
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Final, Literal, TypedDict

import dynesty
import h5py  # type: ignore[import-untyped]
import numpy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dynesty import internal_samplers

    from chirplab.inference import likelihood, prior

type BoundaryType = Literal["none", "single", "multi", "balls", "cubes"] | dynesty.bounding.Bound
type SampleType = Literal["auto", "unif", "rwalk", "slice", "rslice"] | internal_samplers.InternalSampler

logger = logging.getLogger(__name__)

_RESULTS_DATASETS: Final = (
    "logl",
    "samples_it",
    "samples_id",
    "samples_u",
    "samples",
    "niter",
    "ncall",
    "logz",
    "logzerr",
    "logwt",
    "eff",
    "nlive",
    "logvol",
    "information",
    "bound_iter",
    "samples_bound",
    "scale",
)


class FirstUpdateDict(TypedDict, total=False):
    """Dictionary for first update parameters."""

    min_ncall: int
    min_eff: float


class _Cache:
    calculate_log_likelihood: Callable[Concatenate[numpy.typing.NDArray[numpy.floating], ...], float]
    transform_prior: Callable[
        Concatenate[numpy.typing.NDArray[numpy.floating], ...], numpy.typing.NDArray[numpy.floating]
    ]
    calculate_log_likelihood_args: Iterable[Any]
    transform_prior_args: Iterable[Any]
    calculate_log_likelihood_kwargs: dict[str, Any]
    transform_prior_kwargs: dict[str, Any]


def _initialiser(
    calculate_log_likelihood: Callable[..., float],
    transform_prior: Callable[..., numpy.typing.NDArray[numpy.floating]],
    calculate_log_likelihood_args: None | Iterable[Any] = None,
    transform_prior_args: None | Iterable[Any] = None,
    calculate_log_likelihood_kwargs: None | dict[str, Any] = None,
    transform_prior_kwargs: None | dict[str, Any] = None,
) -> None:
    _Cache.calculate_log_likelihood = calculate_log_likelihood
    _Cache.transform_prior = transform_prior
    _Cache.calculate_log_likelihood_args = calculate_log_likelihood_args or ()
    _Cache.transform_prior_args = transform_prior_args or ()
    _Cache.calculate_log_likelihood_kwargs = calculate_log_likelihood_kwargs or {}
    _Cache.transform_prior_kwargs = transform_prior_kwargs or {}


def _cached_calculate_log_likelihood(x: numpy.typing.NDArray[numpy.floating]) -> float:
    return _Cache.calculate_log_likelihood(
        x, *_Cache.calculate_log_likelihood_args, **_Cache.calculate_log_likelihood_kwargs
    )


def _cached_transform_prior(u: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
    return _Cache.transform_prior(u, *_Cache.transform_prior_args, **_Cache.transform_prior_kwargs)


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

    t_1 = time.time()

    if is_multiprocessed:
        with multiprocessing.Pool(njobs, _initialiser, (likelihood.calculate_log_pdf, prior.transform)) as pool:
            if is_resumed:
                assert checkpoint_file is not None
                sampler = dynesty.NestedSampler.restore(checkpoint_file, pool=pool)

                logger.info("Resumed nested sampling run from checkpoint file '%s'", checkpoint_file)
            else:
                sampler = dynesty.NestedSampler(
                    _cached_calculate_log_likelihood,
                    _cached_transform_prior,
                    prior.n_dim,
                    pool=pool,
                    queue_size=njobs,
                    **sampler_kwargs,
                )

            logger.info("Starting nested sampling run (multiprocessing mode)")

            sampler.run_nested(**run_kwargs)
    else:
        if is_resumed:
            assert checkpoint_file is not None
            sampler = dynesty.NestedSampler.restore(checkpoint_file)

            logger.info("Resumed nested sampling run from checkpoint file '%s'", checkpoint_file)
        else:
            sampler = dynesty.NestedSampler(
                likelihood.calculate_log_pdf, prior.transform, prior.n_dim, **sampler_kwargs
            )

        logger.info("Starting nested sampling run (single-process mode)")

        sampler.run_nested(**run_kwargs)

    t_2 = time.time()

    if print_progress:
        print()

    results = sampler.results
    _log_results_summary(results, t_2 - t_1)

    if results_filename is not None:
        _save_results(results, results_filename)

    return results


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


def _save_results(results: dynesty.results.Results, results_filename: str) -> None:
    with h5py.File(results_filename, "w") as f:
        for name in _RESULTS_DATASETS:
            f.create_dataset(name, data=getattr(results, name))

    logger.info("Saved sampling results to '%s'", results_filename)


def load_results(results_filename: str) -> dynesty.results.Results:
    """
    Load sampling results from an HDF5 file.

    Parameters
    ----------
    results_filename
        HDF5 file containing the results.

    Returns
    -------
    results
        Sampling results.
    """
    results = {}
    with h5py.File(results_filename, "r") as f:
        for key in f:
            results[key] = f[key][()]

    return dynesty.utils.Results(results)


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
