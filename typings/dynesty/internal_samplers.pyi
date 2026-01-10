from collections.abc import Callable
from typing import Any, NamedTuple

import numpy
from dynesty import sampler, utils

class SamplerArgument(NamedTuple):
    u: numpy.typing.NDArray[numpy.floating]
    loglstar: float
    axes: numpy.typing.NDArray[numpy.floating]
    scale: float
    prior_transform: Callable[[numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]]
    loglikelihood: Callable[[numpy.typing.NDArray[numpy.floating]], float]
    rseed: int
    kwargs: dict[str, Any]

class SamplerReturn(NamedTuple):
    u: numpy.typing.NDArray[numpy.floating]
    v: numpy.typing.NDArray[numpy.floating]
    logl: float
    ncalls: int
    evaluation_history: list[utils.SamplerHistoryItem]
    tuning_info: None | dict[str, Any]
    proposal_stats: utils.ProposalStatsDict

class InternalSampler:
    def __init__(self, **kwargs: Any) -> None: ...
    @property
    def update_bound_interval_ratio(self) -> int: ...
    def prepare_sampler(
        self,
        loglstar: None | float = None,
        points: None | numpy.typing.NDArray[numpy.floating] = None,
        axes: None | numpy.typing.NDArray[numpy.floating] = None,
        seeds: None | numpy.typing.NDArray[numpy.integer] = None,
        prior_transform: None
        | Callable[[numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]] = None,
        loglikelihood: None | Callable[[numpy.typing.NDArray[numpy.floating]], float] = None,
        nested_sampler: None | sampler.Sampler = None,
    ) -> list[SamplerArgument]: ...
    @staticmethod
    def sample(args: SamplerArgument) -> SamplerReturn: ...
    def tune(self, tuning_info: dict[str, Any], update: bool = False) -> None: ...

class UniformBoundSampler(InternalSampler): ...
class UnitCubeSampler(InternalSampler): ...
class RWalkSampler(InternalSampler): ...
class SliceSampler(InternalSampler): ...
class RSliceSampler(InternalSampler): ...
