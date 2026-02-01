"""Base module for sampling algorithms."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import h5py  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import numpy

    from chirplab.inference import likelihood, prior

logger = logging.getLogger(__name__)


@dataclass
class Result:
    """
    Sampling result.

    Parameters
    ----------
    logl
        Log-likelihood.
    samples_it
        The sampling iteration when the sample was proposed.
    samples_id
        The unique ID of the sample.
    samples_u
        The coordinates of live points in the unit cube coordinate system.
    samples
        The location (in original coordinates).
    niter
        Number of iterations.
    ncall
        Total number of likelihood calls.
    logz
        Array of cumulative log(Z) integrals.
    logzerr
        Array of uncertainty of log(Z).
    logwt
        Array of log-posterior weights.
    eff
        Sampling efficiency.
    nlive
        Number of live points for a static run.
    logvol
        Log-volumes of dead points.
    information
        Information integral.
    bound_iter
        Index of the bound being used for an iteration that generated the point.
    samples_bound
        The index of the bound that the corresponding sample was drawn from.
    scale
        Scalar scale applied for proposals.
    """

    logl: numpy.typing.NDArray[numpy.floating]
    samples_it: numpy.typing.NDArray[numpy.integer]
    samples_id: numpy.typing.NDArray[numpy.integer]
    samples_u: numpy.typing.NDArray[numpy.floating]
    samples: numpy.typing.NDArray[numpy.floating]
    niter: int
    ncall: numpy.typing.NDArray[numpy.integer]
    logz: numpy.typing.NDArray[numpy.floating]
    logzerr: numpy.typing.NDArray[numpy.floating]
    logwt: numpy.typing.NDArray[numpy.floating]
    eff: float
    nlive: int
    logvol: numpy.typing.NDArray[numpy.floating]
    information: numpy.typing.NDArray[numpy.floating]
    bound_iter: numpy.typing.NDArray[numpy.integer]
    samples_bound: numpy.typing.NDArray[numpy.integer]
    scale: numpy.typing.NDArray[numpy.floating]

    def save(self, results_filename: str) -> None:
        """
        Save sampling result to an HDF5 file.

        Parameters
        ----------
        results_filename
            HDF5 file to save the result to.
        """
        with h5py.File(results_filename, "w") as f:
            for name, value in self.__dict__.items():
                f.create_dataset(name, data=value)

        logger.info("Saved sampling results to '%s'", results_filename)

    @classmethod
    def load(cls, results_filename: str) -> Self:
        """
        Load sampling result from an HDF5 file.

        Parameters
        ----------
        results_filename
            HDF5 file containing the result.

        Returns
        -------
        result
            Sampling result.
        """
        result_dict = {}
        with h5py.File(results_filename, "r") as f:
            for key in f:
                result_dict[key] = f[key][()]

        return cls(**result_dict)


class Sampler(ABC):
    """
    Sampler.

    Parameters
    ----------
    likelihood
        Likelihood function.
    prior
        Prior distribution.
    rng
        Random number generator for the sampling.
    """

    @abstractmethod
    def __init__(
        self, likelihood: likelihood.Likelihood, prior: prior.Prior, rng: None | numpy.random.Generator = None
    ) -> None:
        t = benchmark(likelihood, prior, rng=rng)

        logger.debug("Likelihood benchmark: average log-likelihood evaluation time = %.3e s", t)
        logger.info("Likelihood function: %s", likelihood)
        logger.info("Prior distribution: %s", prior)

        self.result: None | Result = None

    @abstractmethod
    def run(self) -> None:
        """
        Run the sampler.

        Returns
        -------
        result
            Sampling result.
        """
        ...


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
