from collections.abc import Callable, Generator
from typing import Any

import numpy
from dynesty import bounding, results, utils

type PRINT_FUNC_TYPES = Callable[[utils.IteratorResult, int, int, int, float, float], None]

class Sampler:
    def save(self, filename: str) -> None: ...
    @staticmethod
    def restore(filename: str, pool: None | Any = None) -> Sampler: ...
    def propose_live(
        self, *args: Any
    ) -> tuple[numpy.typing.NDArray[numpy.floating], numpy.typing.NDArray[numpy.floating]]: ...
    def update_bound(self, subset: slice = slice(None)) -> bounding.Bound: ...
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
        loglmax: float = numpy.inf,
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
