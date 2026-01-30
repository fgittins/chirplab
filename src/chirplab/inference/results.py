"""Module for sampling results."""

import logging
from typing import Final

import dynesty
import h5py  # type: ignore[import-untyped]

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


def _save(results: dynesty.results.Results, results_filename: str) -> None:
    with h5py.File(results_filename, "w") as f:
        for name in _RESULTS_DATASETS:
            f.create_dataset(name, data=getattr(results, name))

    logger.info("Saved sampling results to '%s'", results_filename)


def load(results_filename: str) -> dynesty.results.Results:
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
