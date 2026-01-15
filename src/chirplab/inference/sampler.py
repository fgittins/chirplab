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


# TODO: add pool option for multiprocessing


class NestedSampler:
    """
    Nested sampler.

    Parameters
    ----------
    likelihood
        Likelihood function.
    prior
        Prior distribution.
    rng
        Random number generator for the sampling.
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
    """

    def __init__(
        self,
        likelihood: likelihood.Likelihood,
        prior: prior.Prior,
        rng: None | numpy.random.Generator = None,
        nlive: int = 500,
        bound: BOUNDARY_TYPES = "multi",
        sample: SAMPLE_TYPES = "auto",
        update_interval: None | int | float = None,
        first_update: None | FirstUpdateDict = None,
        enlarge: None | float = None,
        bootstrap: None | int = None,
        walks: None | int = None,
        facc: float = 0.5,
        slices: None | int = None,
        ncdim: None | int = None,
        history_filename: None | str = None,
    ) -> None:
        self.t_eval: None | float = benchmark(likelihood, prior, rng=rng)

        self.is_restored = False

        self.sampler: dynesty.sampler.Sampler = dynesty.NestedSampler(
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

    def run_nested(
        self,
        maxiter: None | int = None,
        maxcall: None | int = None,
        dlogz: None | float = None,
        logl_max: float = numpy.inf,
        add_live: bool = True,
        print_progress: bool = True,
        save_bounds: bool = True,
        checkpoint_file: None | str = None,
        checkpoint_every: float = 60,
    ) -> None:
        """
        Run the nested sampler.

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
            Iteration will stop when the sampled ln(likelihood) exceeds the threshold set by logl_max.
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
        """
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
            resume=self.is_restored,
        )
        if print_progress:
            print()

    @classmethod
    def restore(cls, filename: str) -> NestedSampler:
        """
        Restore the nested sampler from a checkpoint file.

        Parameters
        ----------
        filename
            Checkpoint filename.

        Returns
        -------
        sampler
            Restored nested sampler instance.
        """
        obj = cls.__new__(cls)
        obj.t_eval = None
        obj.is_restored = True
        obj.sampler = dynesty.NestedSampler.restore(filename, pool=None)
        return obj

    @property
    def results(self) -> dynesty.results.Results:
        """Results of the nested sampling run."""
        return self.sampler.results


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
