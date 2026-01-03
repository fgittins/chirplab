from typing import Any, NamedTuple, TypedDict

import numpy
from dynesty import bounding

class ProposalStatsDict(TypedDict, total=False):
    n_proposals: int
    n_accept: int
    n_reject: int
    n_expand: int
    n_contract: int

class IteratorResult(NamedTuple):
    worst: int
    ustar: float
    vstar: float
    loglstar: float
    logvol: float
    logwt: float
    logz: float
    logzvar: float
    h: float
    nc: int
    worst_it: int
    boundidx: int
    bounditer: int
    eff: float
    delta_logz: float
    blob: Any
    proposal_stats: None | ProposalStatsDict

class Results:
    logl: numpy.typing.NDArray[numpy.floating]
    samples_it: numpy.typing.NDArray[numpy.int_]
    samples_id: numpy.typing.NDArray[numpy.int_]
    samples_n: numpy.typing.NDArray[numpy.int_]
    samples_u: numpy.typing.NDArray[numpy.floating]
    samples: numpy.typing.NDArray[numpy.floating]
    niter: int
    ncall: numpy.typing.NDArray[numpy.int_]
    logz: numpy.typing.NDArray[numpy.floating]
    logzerr: numpy.typing.NDArray[numpy.floating]
    logwt: numpy.typing.NDArray[numpy.floating]
    eff: float
    nlive: int
    logvol: numpy.typing.NDArray[numpy.floating]
    information: numpy.typing.NDArray[numpy.floating]
    bound: list[bounding.Bound]
    bound_iter: numpy.typing.NDArray[numpy.int_]
    samples_bound: numpy.typing.NDArray[numpy.int_]
    scale: numpy.typing.NDArray[numpy.floating]
    blob: numpy.typing.NDArray[Any]
    proposal_stats: numpy.typing.NDArray[Any]
    def summary(self) -> None: ...
