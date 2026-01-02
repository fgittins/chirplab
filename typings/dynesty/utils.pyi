from typing import Any, NamedTuple, TypedDict

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
    def summary(self) -> None: ...
