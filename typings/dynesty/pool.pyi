from collections.abc import Callable, Iterable
from typing import Any, Concatenate, Self

import numpy

class Pool:
    logl_args: None | Iterable[Any]
    logl_kwargs: None | dict[str, Any]
    ptform_args: None | Iterable[Any]
    ptform_kwargs: None | dict[str, Any]
    njobs: int
    loglike_0: Callable[Concatenate[numpy.typing.NDArray[numpy.floating], ...], float]
    prior_transform_0: Callable[
        Concatenate[numpy.typing.NDArray[numpy.floating], ...], numpy.typing.NDArray[numpy.floating]
    ]
    loglike: Callable[Concatenate[numpy.typing.NDArray[numpy.floating], ...], float]
    prior_transform: Callable[
        Concatenate[numpy.typing.NDArray[numpy.floating], ...], numpy.typing.NDArray[numpy.floating]
    ]
    pool: Any
    def __init__(
        self,
        njobs: int,
        loglike: Callable[Concatenate[numpy.typing.NDArray[numpy.floating], ...], float],
        prior_transform: Callable[
            Concatenate[numpy.typing.NDArray[numpy.floating], ...], numpy.typing.NDArray[numpy.floating]
        ],
        logl_args: None | Iterable[Any] = None,
        logl_kwargs: None | dict[str, Any] = None,
        ptform_args: None | Iterable[Any] = None,
        ptform_kwargs: None | dict[str, Any] = None,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def map(self, F: Callable[[Iterable[float]], Any], x: Iterable[float]) -> Any: ...  # noqa: N803
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    @property
    def size(self) -> int: ...
    def close(self) -> None: ...
    def join(self) -> None: ...
