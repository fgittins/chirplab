from collections.abc import Callable, Generator
from typing import Any, Literal, TypedDict, overload

import numpy
from dynesty import internal_samplers, results, utils
from dynesty.bounding import Bound

type BOUND_TYPES = Literal["none", "single", "multi", "balls", "cubes"] | Bound
type PRINT_FUNC_TYPES = Callable[[utils.IteratorResult, int, int, int, float, float], None]

class FirstUpdateDict(TypedDict, total=False):
    min_ncall: int
    min_eff: float

class UsePoolDict(TypedDict, total=False):
    prior_transform: bool
    loglikelihood: bool
    propose_point: bool
    update_bound: bool

class Sampler:
    loglikelihood: Callable[[numpy.typing.NDArray[numpy.floating]], float]
    prior_transform: Callable[[numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]]
    ndim: int
    ncdim: int
    blob: bool
    live_u: numpy.typing.NDArray[numpy.floating]
    live_v: numpy.typing.NDArray[numpy.floating]
    live_logl: numpy.typing.NDArray[numpy.floating]
    live_blobs: None | numpy.typing.NDArray[numpy.floating]
    nlive: int
    live_bound: numpy.typing.NDArray[numpy.integer]
    live_it: numpy.typing.NDArray[numpy.integer]
    rstate: numpy.random.Generator
    sampling: internal_samplers.InternalSampler
    internal_sampler_next: internal_samplers.InternalSampler
    internal_sampler: internal_samplers.InternalSampler
    pool: Any
    use_pool: UsePoolDict
    use_pool_ptform: bool
    use_pool_logl: bool
    use_pool_evolve: bool
    use_pool_update: bool
    queue_size: int
    queue: list[numpy.typing.NDArray[numpy.floating]]
    nqueue: int
    it: int
    ncall: int
    dlv: float
    added_live: bool
    eff: float
    save_bounds: bool
    bound_update_interval: int
    first_bound_update: FirstUpdateDict
    first_bound_update_ncall: int
    first_bound_update_eff: float
    logl_first_update: None | float
    ncall_at_last_update: int
    unit_cube_sampling: bool
    bound: Bound
    bound_list: list[Bound]
    nbound: int
    logvol_init: float
    plateau_mode: bool
    plateau_counter: None | int
    plateau_logdvol: None | float
    saved_run: utils.RunRecord
    bound_bootstrap: None | int
    bound_enlarge: None | float
    bounding: BOUND_TYPES
    bound_next: BOUND_TYPES
    @overload
    def __init__(
        self,
        loglikelihood: Callable[[numpy.typing.NDArray[numpy.floating]], float],
        prior_transform: Callable[[numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]],
        ndim: int,
        live_points: tuple[
            numpy.typing.NDArray[numpy.floating],
            numpy.typing.NDArray[numpy.floating],
            numpy.typing.NDArray[numpy.floating],
        ],
        sampling: internal_samplers.InternalSampler,
        bounding: BOUND_TYPES,
        ncdim: None | int = None,
        rstate: None | numpy.random.Generator = None,
        pool: Any = None,
        use_pool: None | UsePoolDict = None,
        queue_size: None | int = None,
        bound_update_interval: None | int = None,
        first_bound_update: None | FirstUpdateDict = None,
        bound_bootstrap: None | int = None,
        bound_enlarge: None | float = None,
        blob: Literal[False] = False,
        cite: None | str = None,
        logvol_init: float = 0,
    ) -> None: ...
    @overload
    def __init__(
        self,
        loglikelihood: Callable[[numpy.typing.NDArray[numpy.floating]], float],
        prior_transform: Callable[[numpy.typing.NDArray[numpy.floating]], numpy.typing.NDArray[numpy.floating]],
        ndim: int,
        live_points: tuple[
            numpy.typing.NDArray[numpy.floating],
            numpy.typing.NDArray[numpy.floating],
            numpy.typing.NDArray[numpy.floating],
            numpy.typing.NDArray[numpy.floating],
        ],
        sampling: internal_samplers.InternalSampler,
        bounding: BOUND_TYPES,
        ncdim: None | int = None,
        rstate: None | numpy.random.Generator = None,
        pool: Any = None,
        use_pool: None | UsePoolDict = None,
        queue_size: None | int = None,
        bound_update_interval: None | int = None,
        first_bound_update: None | FirstUpdateDict = None,
        bound_bootstrap: None | int = None,
        bound_enlarge: None | float = None,
        blob: Literal[True] = True,
        cite: None | str = None,
        logvol_init: float = 0,
    ) -> None: ...
    def save(self, fname: str) -> None: ...
    @staticmethod
    def restore(fname: str, pool: Any = None) -> Sampler: ...
    def propose_live(
        self, *args: Any
    ) -> tuple[numpy.typing.NDArray[numpy.floating], numpy.typing.NDArray[numpy.floating]]: ...
    def update_bound(self, subset: slice = slice(None)) -> Bound: ...
    def reset(self) -> None: ...
    @property
    def results(self) -> results.Results: ...
    @property
    def n_effective(self) -> int: ...
    def update_bound_if_needed(self, loglstar: float, ncall: None | int = None, force: bool = False) -> None: ...
    def add_live_points(self) -> Generator[utils.IteratorResult]: ...
    def sample(
        self,
        maxiter: None | int = None,
        maxcall: None | int = None,
        dlogz: float = 0.01,
        logl_max: float = numpy.inf,
        add_live: bool = True,
        save_bounds: bool = True,
        resume: bool = False,
    ) -> Generator[utils.IteratorResult]: ...
    def run_nested(
        self,
        maxiter: None | int = None,
        maxcall: None | int = None,
        dlogz: None | float = None,
        logl_max: None | float = numpy.inf,
        add_live: bool = True,
        print_progress: bool = True,
        print_func: None | PRINT_FUNC_TYPES = None,
        save_bounds: bool = True,
        checkpoint_file: None | str = None,
        checkpoint_every: float = 60,
        resume: bool = False,
    ) -> None: ...
