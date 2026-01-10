from collections.abc import Callable, Iterable
from typing import Any, Concatenate, Literal

import numpy
from dynesty import internal_samplers, sampler

type SAMPLING_TYPES = Literal["auto", "unif", "rwalk", "slice", "rslice"] | internal_samplers.InternalSampler

class NestedSampler(sampler.Sampler):
    def __new__(
        cls,
        loglikelihood: Callable[Concatenate[numpy.typing.NDArray[numpy.floating], ...], float],
        prior_transform: Callable[
            Concatenate[numpy.typing.NDArray[numpy.floating], ...], numpy.typing.NDArray[numpy.floating]
        ],
        ndim: int,
        nlive: int = 500,
        bound: sampler.BOUND_TYPES = "multi",
        sample: SAMPLING_TYPES = "auto",
        periodic: None | Iterable[int] = None,
        reflective: None | Iterable[int] = None,
        update_interval: None | int | float = None,
        first_update: None | sampler.FirstUpdateDict = None,
        rstate: None | numpy.random.Generator = None,
        queue_size: None | int = None,
        pool: None | Any = None,
        use_pool: None | sampler.UsePoolDict = None,
        live_points: None | list[numpy.typing.NDArray[numpy.floating]] = None,
        logl_args: None | Iterable[Any] = None,
        logl_kwargs: None | dict[str, Any] = None,
        ptform_args: None | Iterable[Any] = None,
        ptform_kwargs: None | dict[str, Any] = None,
        enlarge: None | float = None,
        bootstrap: None | int = None,
        walks: None | int = None,
        facc: float = 0.5,
        slices: None | int = None,
        ncdim: None | int = None,
        blob: bool = False,
        save_evaluation_history: bool = False,
        history_filename: None | str = None,
    ) -> NestedSampler: ...
