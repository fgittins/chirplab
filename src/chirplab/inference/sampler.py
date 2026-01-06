"""Module for sampling algorithms."""

import time
from typing import TYPE_CHECKING, Literal

import dynesty
import numpy

from chirplab.simulation import interferometer

if TYPE_CHECKING:
    from chirplab.inference import likelihood, prior

type BOUNDARY_TYPES = Literal["none", "single", "multi", "balls", "cubes"]
type SAMPLE_TYPES = Literal["auto", "unif", "rwalk", "slice", "rslice"]


class NestedSampler:
    """
    Nested sampler of gravitational-wave signals.

    Parameters
    ----------
    likelihood
        Likelihood function for gravitational-wave signals.
    priors
        Joint prior distribution on the gravitational-wave signal parameters.
    rng
        Random number generator for the sampling.
    """

    def __init__(
        self,
        likelihood: likelihood.Likelihood,
        priors: prior.Priors,
        rng: None | numpy.random.Generator = None,
        nlive: int = 500,
        bound: BOUNDARY_TYPES = "multi",
        sample: SAMPLE_TYPES = "auto",
        enlarge: None | float = None,
        bootstrap: None | int = None,
        walks: None | int = None,
        facc: float = 0.5,
        slices: None | int = None,
        ncdim: None | int = None,
    ) -> None:
        if rng is None:
            self.rng = numpy.random.default_rng()
        else:
            self.rng = rng

        self.t_eval = benchmark(likelihood, priors, rng=self.rng)

        self.sampler = dynesty.NestedSampler(
            self.calculate_log_likelihood,
            priors.calculate_ppf,
            priors.n,
            nlive=nlive,
            bound=bound,
            sample=sample,
            periodic=priors.periodic_indices,
            reflective=priors.reflective_indices,
            rstate=self.rng,
            logl_args=(likelihood, priors.theta_name_sample, priors.theta_fixed),
            enlarge=enlarge,
            bootstrap=bootstrap,
            walks=walks,
            facc=facc,
            slices=slices,
            ncdim=ncdim,
        )

    def run_nested(
        self,
        maxiter: None | int = None,
        maxcall: None | int = None,
        delta_ln_z: float = 0.1,
        ln_l_max: float = numpy.inf,
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
        delta_ln_z
            Stopping criterion in log-evidence.
        """
        self.sampler.run_nested(
            maxiter=maxiter,
            maxcall=maxcall,
            dlogz=delta_ln_z,
            logl_max=ln_l_max,
            add_live=add_live,
            print_progress=print_progress,
            save_bounds=save_bounds,
            checkpoint_file=checkpoint_file,
            checkpoint_every=checkpoint_every,
        )

        self.results = self.sampler.results

    @staticmethod
    def calculate_log_likelihood(
        x: numpy.typing.NDArray[numpy.floating],
        likelihood: likelihood.Likelihood,
        theta_name_sample: list[str],
        theta_fixed: dict[str, float],
    ) -> numpy.float64:
        """
        Calculate log of the Likelihood function.

        Parameters
        ----------
        x
            Sampled parameters.
        likelihood
            Likelihood function for gravitational-wave signals.
        theta_name_sample
            Names of sampled parameters.
        theta_fixed
            Fixed parameters.

        Returns
        -------
        log_likelihood
            Log of the Likelihood function.
        """
        theta_sampled = dict(zip(theta_name_sample, x, strict=False))
        theta = interferometer.SignalParameters(**theta_sampled, **theta_fixed)
        return likelihood.calculate_log_pdf(theta)


def benchmark(
    likelihood: likelihood.Likelihood, priors: prior.Priors, n: int = 1_000, rng: None | numpy.random.Generator = None
) -> float:
    """
    Benchmark the log of the likelihood function evaluation time.

    Parameters
    ----------
    likelihood
        Likelihood function for gravitational-wave signals.
    priors
        Joint prior distribution on the gravitational-wave signal parameters.
    n
        Number of evaluations to average over.
    rng
        Random number generator for the sampling.

    Returns
    -------
    t
        Average time per log-likelihood evaluation (s).
    """
    theta_list = [priors.sample(rng) for _ in range(n)]

    t_i = time.time()
    for theta in theta_list:
        likelihood.calculate_log_pdf(theta)
    t_f = time.time()

    return (t_f - t_i) / n
