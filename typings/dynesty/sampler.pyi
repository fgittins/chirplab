from collections.abc import Callable

import numpy
from dynesty import results, utils

type PRINT_FUNC_TYPES = Callable[[utils.IteratorResult, int, int, int, float, float], None]

class Sampler:
    results: results.Results
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
