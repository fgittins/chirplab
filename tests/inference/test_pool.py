"""Tests for the pool module."""

import numpy

from chirplab.inference import pool


class TestInitialiser:
    """Tests for the _initialiser function."""

    def test_initialiser_sets_functions(self) -> None:
        """Test that _initialiser sets the likelihood and prior functions."""

        def calculate_log_likelihood(x: numpy.typing.NDArray[numpy.floating]) -> float:
            return float(-numpy.sum(x**2))

        def transform_prior(q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
            return q * 2

        pool._initialiser(calculate_log_likelihood, transform_prior)

        assert pool._Cache.calculate_log_likelihood is calculate_log_likelihood
        assert pool._Cache.transform_prior is transform_prior

    def test_initialiser_sets_default_args_kwargs(self) -> None:
        """Test that _initialiser sets empty args and kwargs by default."""

        def calculate_log_likelihood(x: numpy.typing.NDArray[numpy.floating]) -> float:
            return 0.0

        def transform_prior(q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
            return q

        pool._initialiser(calculate_log_likelihood, transform_prior)

        assert pool._Cache.calculate_log_likelihood_args == ()
        assert pool._Cache.transform_prior_args == ()
        assert pool._Cache.calculate_log_likelihood_kwargs == {}
        assert pool._Cache.transform_prior_kwargs == {}

    def test_initialiser_sets_custom_args_kwargs(self) -> None:
        """Test that _initialiser sets custom args and kwargs."""

        def calculate_log_likelihood(x: numpy.typing.NDArray[numpy.floating]) -> float:
            return 0

        def transform_prior(q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
            return q

        calculate_log_likelihood_args = (1, 2)
        transform_prior_args = ("a", "b")
        calculate_log_likelihood_kwargs = {"key1": "value1"}
        transform_prior_kwargs = {"key2": "value2"}
        pool._initialiser(
            calculate_log_likelihood,
            transform_prior,
            calculate_log_likelihood_args=calculate_log_likelihood_args,
            transform_prior_args=transform_prior_args,
            calculate_log_likelihood_kwargs=calculate_log_likelihood_kwargs,
            transform_prior_kwargs=transform_prior_kwargs,
        )

        assert pool._Cache.calculate_log_likelihood_args == calculate_log_likelihood_args
        assert pool._Cache.transform_prior_args == transform_prior_args
        assert pool._Cache.calculate_log_likelihood_kwargs == calculate_log_likelihood_kwargs
        assert pool._Cache.transform_prior_kwargs == transform_prior_kwargs


class TestCalculateLogLikelihoodWrapper:
    """Tests for the _calculate_log_likelihood_wrapper function."""

    def test_wrapper_calls_cached_function(self) -> None:
        """Test that the wrapper calls the cached likelihood function."""

        def calculate_log_likelihood(x: numpy.typing.NDArray[numpy.floating]) -> float:
            return float(-numpy.sum(x**2))

        def transform_prior(q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
            return q

        pool._initialiser(calculate_log_likelihood, transform_prior)
        x = numpy.array([1, 2, 3])
        result = pool._calculate_log_likelihood_wrapper(x)

        assert result == -14

    def test_wrapper_passes_args_and_kwargs(self) -> None:
        """Test that the wrapper passes args and kwargs to the function."""

        def calculate_log_likelihood(x: numpy.typing.NDArray[numpy.floating], scale: float, *, offset: float) -> float:
            return float(-numpy.sum(x**2) * scale + offset)

        def transform_prior(q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
            return q

        pool._initialiser(
            calculate_log_likelihood,
            transform_prior,
            calculate_log_likelihood_args=(2,),
            calculate_log_likelihood_kwargs={"offset": 10},
        )
        x = numpy.array([1, 2])
        result = pool._calculate_log_likelihood_wrapper(x)

        assert result == -10 + 10


class TestTransformPriorWrapper:
    """Tests for the _transform_prior_wrapper function."""

    def test_wrapper_calls_cached_function(self) -> None:
        """Test that the wrapper calls the cached prior transform function."""

        def calculate_log_likelihood(x: numpy.typing.NDArray[numpy.floating]) -> float:
            return 0

        def transform_prior(q: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
            return 2 * q

        pool._initialiser(calculate_log_likelihood, transform_prior)
        q = numpy.array([0.25, 0.5, 0.75])
        result = pool._transform_prior_wrapper(q)

        numpy.array_equal(result, numpy.array([0.5, 1, 1.5]))

    def test_wrapper_passes_args_and_kwargs(self) -> None:
        """Test that the wrapper passes args and kwargs to the function."""

        def calculate_log_likelihood(x: numpy.typing.NDArray[numpy.floating]) -> float:
            return 0

        def transform_prior(
            q: numpy.typing.NDArray[numpy.floating], scale: float, *, offset: float
        ) -> numpy.typing.NDArray[numpy.floating]:
            return scale * q + offset

        pool._initialiser(
            calculate_log_likelihood, transform_prior, transform_prior_args=(3,), transform_prior_kwargs={"offset": 1}
        )
        q = numpy.array([0, 1])
        result = pool._transform_prior_wrapper(q)

        numpy.array_equal(result, numpy.array([1, 4]))
