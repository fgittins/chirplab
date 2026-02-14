"""Base module for sampling algorithms."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import h5py  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import numpy

    from chirplab.inference import likelihood, prior

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Result:
    """
    Sampling result.

    Parameters
    ----------
    x
        Samples.
    w
        Sample weights.
    ln_l
        Log-likelihoods of the samples.
    ln_z
        Log-evidence.
    delta_ln_z
        Uncertainty of the log-evidence.
    """

    x: numpy.typing.NDArray[numpy.floating]
    w: numpy.typing.NDArray[numpy.floating]
    ln_l: numpy.typing.NDArray[numpy.floating]
    ln_z: float
    delta_ln_z: float

    def save(self, results_filename: str) -> None:
        """
        Save sampling result to an HDF5 file.

        Parameters
        ----------
        results_filename
            HDF5 file to save the result to.
        """
        with h5py.File(results_filename, "w") as f:
            for name in self.__slots__:
                value = getattr(self, name)
                if value is not None:
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
    def run(self, *args: Any, **kwargs: Any) -> None:
        """Run the sampler."""
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
