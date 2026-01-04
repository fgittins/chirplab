from collections.abc import Generator
from typing import Any, Literal, NamedTuple, TypedDict, overload

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
    ustar: numpy.typing.NDArray[numpy.floating]
    vstar: numpy.typing.NDArray[numpy.floating]
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

def mean_and_cov(
    samples: numpy.typing.NDArray[numpy.floating], weights: numpy.typing.NDArray[numpy.floating]
) -> tuple[numpy.typing.NDArray[numpy.floating], numpy.typing.NDArray[numpy.floating]]: ...
def resample_equal(
    samples: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    rstate: None | numpy.random.Generator = None,
) -> numpy.typing.NDArray[numpy.floating]: ...
def quantile(
    x: numpy.typing.NDArray[numpy.floating],
    q: numpy.typing.NDArray[numpy.floating],
    weights: None | numpy.typing.NDArray[numpy.floating] = None,
) -> numpy.typing.NDArray[numpy.floating]: ...
def jitter_run(res: Results, rstate: None | numpy.random.Generator = None, approx: bool = False) -> Results: ...
@overload
def resample_run(
    res: Results, rstate: None | numpy.random.Generator = None, return_idx: Literal[False] = False
) -> Results: ...
@overload
def resample_run(
    res: Results, rstate: None | numpy.random.Generator = None, return_idx: Literal[True] = True
) -> tuple[Results, numpy.typing.NDArray[numpy.int_]]: ...
def reweight_run(
    res: Results,
    logp_new: numpy.typing.NDArray[numpy.floating],
    logp_old: None | numpy.typing.NDArray[numpy.floating] = None,
) -> Results: ...
def unravel_run(res: Results, print_progress: bool = True) -> list[Results]: ...
def merge_runs(res_list: list[Results], print_progress: bool = True) -> Results: ...
@overload
def kld_error(
    res: Results,
    error: Literal["jitter", "resample"] = "jitter",
    rstate: None | numpy.random.Generator = None,
    return_new: Literal[False] = False,
    approx: bool = False,
) -> numpy.typing.NDArray[numpy.floating]: ...
@overload
def kld_error(
    res: Results,
    error: Literal["jitter", "resample"] = "jitter",
    rstate: None | numpy.random.Generator = None,
    return_new: Literal[True] = True,
    approx: bool = False,
) -> tuple[numpy.typing.NDArray[numpy.floating], Results]: ...

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
    def copy(self) -> Results: ...
    def asdict(self) -> dict[str, Any]: ...
    def keys(self) -> list[str]: ...
    def items(self) -> Generator[Any]: ...
    def isdynamic(self) -> bool: ...
    def importance_weights(self) -> numpy.typing.NDArray[numpy.floating]: ...
    def samples_equal(self) -> numpy.typing.NDArray[numpy.floating]: ...
    def summary(self) -> None: ...
