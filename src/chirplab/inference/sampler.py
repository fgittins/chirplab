"""Module for sampling algorithms."""

import time
from typing import TYPE_CHECKING, Literal, TypedDict

import dynesty
import numpy

if TYPE_CHECKING:
    from dynesty import internal_samplers

    from chirplab.inference import likelihood, prior

type BOUNDARY_TYPES = Literal["none", "single", "multi", "balls", "cubes"] | dynesty.bounding.Bound
type SAMPLE_TYPES = Literal["auto", "unif", "rwalk", "slice", "rslice"] | internal_samplers.InternalSampler


class FirstUpdateDict(TypedDict, total=False):
    """Dictionary for first update parameters."""

    min_ncall: int
    min_eff: float


# TODO: add checkpointing
def run(
    likelihood: likelihood.Likelihood,
    prior: prior.Prior,
    nlive: int = 500,
    bound: BOUNDARY_TYPES = "multi",
    sample: SAMPLE_TYPES = "auto",
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
    history_filename: None | str = None,
    maxiter: None | int = None,
    maxcall: None | int = None,
    dlogz: None | float = None,
    logl_max: float = numpy.inf,
    add_live: bool = True,
    print_progress: bool = True,
    save_bounds: bool = True,
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
    history_filename
        The filename where the history will go.
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
    """
    if njobs > 1:
        with dynesty.pool.Pool(njobs, likelihood.calculate_log_pdf, prior.transform) as pool:
            sampler = dynesty.NestedSampler(
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
                queue_size=None,
                pool=pool,
                use_pool=None,
                live_points=None,
                enlarge=enlarge,
                bootstrap=bootstrap,
                walks=walks,
                facc=facc,
                slices=slices,
                ncdim=ncdim,
                blob=False,
                save_evaluation_history=history_filename is not None,
                history_filename=history_filename,
            )

            sampler.run_nested(
                maxiter=maxiter,
                maxcall=maxcall,
                dlogz=dlogz,
                logl_max=logl_max,
                add_live=add_live,
                print_progress=print_progress,
                save_bounds=save_bounds,
                checkpoint_file=None,
                checkpoint_every=60,
                resume=False,
            )
    else:
        sampler = dynesty.NestedSampler(
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
            queue_size=None,
            pool=None,
            use_pool=None,
            live_points=None,
            enlarge=enlarge,
            bootstrap=bootstrap,
            walks=walks,
            facc=facc,
            slices=slices,
            ncdim=ncdim,
            blob=False,
            save_evaluation_history=history_filename is not None,
            history_filename=history_filename,
        )

        sampler.run_nested(
            maxiter=maxiter,
            maxcall=maxcall,
            dlogz=dlogz,
            logl_max=logl_max,
            add_live=add_live,
            print_progress=print_progress,
            save_bounds=save_bounds,
            checkpoint_file=None,
            checkpoint_every=60,
            resume=False,
        )

    if print_progress:
        print()

    return sampler.results


def benchmark(
    likelihood: likelihood.Likelihood, prior: prior.Prior, n: int = 1_000, rng: None | numpy.random.Generator = None
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

    t_i = time.time()
    for x in x_list:
        likelihood.calculate_log_pdf(x)
    t_f = time.time()

    return (t_f - t_i) / n
