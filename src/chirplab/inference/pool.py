"""Module for multiprocessing pool."""

from typing import TYPE_CHECKING, Any, Concatenate

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy


# TODO: add tests
class _Cache:
    calculate_log_likelihood: Callable[Concatenate[numpy.typing.NDArray[numpy.floating], ...], float]
    transform_prior: Callable[
        Concatenate[numpy.typing.NDArray[numpy.floating], ...], numpy.typing.NDArray[numpy.floating]
    ]
    calculate_log_likelihood_args: Iterable[Any]
    transform_prior_args: Iterable[Any]
    calculate_log_likelihood_kwargs: dict[str, Any]
    transform_prior_kwargs: dict[str, Any]


def _initialiser(
    calculate_log_likelihood: Callable[Concatenate[numpy.typing.NDArray[numpy.floating], ...], float],
    transform_prior: Callable[
        Concatenate[numpy.typing.NDArray[numpy.floating], ...], numpy.typing.NDArray[numpy.floating]
    ],
    calculate_log_likelihood_args: None | Iterable[Any] = None,
    transform_prior_args: None | Iterable[Any] = None,
    calculate_log_likelihood_kwargs: None | dict[str, Any] = None,
    transform_prior_kwargs: None | dict[str, Any] = None,
) -> None:
    _Cache.calculate_log_likelihood = calculate_log_likelihood
    _Cache.transform_prior = transform_prior
    _Cache.calculate_log_likelihood_args = calculate_log_likelihood_args or ()
    _Cache.transform_prior_args = transform_prior_args or ()
    _Cache.calculate_log_likelihood_kwargs = calculate_log_likelihood_kwargs or {}
    _Cache.transform_prior_kwargs = transform_prior_kwargs or {}


def _calculate_log_likelihood_wrapper(x: numpy.typing.NDArray[numpy.floating]) -> float:
    return _Cache.calculate_log_likelihood(
        x, *_Cache.calculate_log_likelihood_args, **_Cache.calculate_log_likelihood_kwargs
    )


def _transform_prior_wrapper(q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
    return _Cache.transform_prior(q, *_Cache.transform_prior_args, **_Cache.transform_prior_kwargs)
