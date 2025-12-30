"""Unit tests for the prior module."""

import numpy
import pytest

from chirplab import constants, prior


class TestPrior:
    """Tests for the Prior base class."""

    def test_initialisation_no_boundary(self) -> None:
        """Test Prior initialisation with no boundary condition."""
        p = prior.Prior()

        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_initialisation_periodic_boundary(self) -> None:
        """Test Prior initialisation with periodic boundary condition."""
        p = prior.Prior("periodic")

        assert p.is_periodic is True
        assert p.is_reflective is False

    def test_initialisation_reflective_boundary(self) -> None:
        """Test Prior initialisation with reflective boundary condition."""
        p = prior.Prior("reflective")

        assert p.is_periodic is False
        assert p.is_reflective is True

    def test_transform_not_implemented(self) -> None:
        """Test that base Prior.transform raises NotImplementedError."""
        p = prior.Prior()

        with pytest.raises(NotImplementedError, match="Priors must implement this method"):
            p.transform(0.5)


class TestDeltaFunction:
    """Tests for the DeltaFunction prior."""

    def test_initialisation(self) -> None:
        """Test DeltaFunction initialisation."""
        x_peak = 3.5
        p = prior.DeltaFunction(x_peak)

        assert p.x_peak == 3.5
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_transform_returns_peak(self) -> None:
        """Test that transform always returns the peak value."""
        x_peak = 2.5
        p = prior.DeltaFunction(x_peak)

        assert p.transform(0) == x_peak
        assert p.transform(0.5) == x_peak
        assert p.transform(1) == x_peak

    def test_transform_independent_of_input(self) -> None:
        """Test that transform is independent of input value."""
        x_peak = 7.3
        p = prior.DeltaFunction(x_peak)
        u = [0, 0.25, 0.5, 0.75, 1]
        x = [p.transform(y) for y in u]

        assert all(y == x_peak for y in x)

    def test_negative_peak(self) -> None:
        """Test DeltaFunction with negative peak value."""
        x_peak = -5.2
        p = prior.DeltaFunction(x_peak)

        assert p.transform(0.5) == x_peak


class TestUniform:
    """Tests for the Uniform prior."""

    def test_initialisation(self) -> None:
        """Test Uniform prior initialisation."""
        x_min, x_max = 0, 1
        p = prior.Uniform(x_min, x_max)

        assert p.x_min == x_min
        assert p.x_max == x_max
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_initialisation_with_boundary(self) -> None:
        """Test Uniform prior initialisation with boundary condition."""
        p = prior.Uniform(0, 1, "periodic")

        assert p.is_periodic is True

    def test_transform_boundaries(self) -> None:
        """Test that transform returns correct boundary values."""
        x_min, x_max = 2, 5
        p = prior.Uniform(x_min, x_max)

        assert p.transform(0) == x_min
        assert p.transform(1) == x_max

    def test_transform_midpoint(self) -> None:
        """Test that transform returns correct midpoint value."""
        x_min, x_max = 0, 10
        p = prior.Uniform(x_min, x_max)

        assert p.transform(0.5) == (x_min + x_max) / 2

    def test_transform_linear(self) -> None:
        """Test that transform is linear in u."""
        x_min, x_max = -5, 15
        p = prior.Uniform(x_min, x_max)
        u = numpy.array([0, 0.25, 0.5, 0.75, 1])
        x = numpy.array([p.transform(y) for y in u])

        assert all(x == x_min + (x_max - x_min) * u)

    def test_negative_range(self) -> None:
        """Test Uniform prior with negative range."""
        x_min, x_max = -10, -2
        p = prior.Uniform(x_min, x_max)

        assert p.transform(0) == x_min
        assert p.transform(1) == x_max

    def test_range_crossing_zero(self) -> None:
        """Test Uniform prior with range crossing zero."""
        x_min, x_max = -3, 7
        p = prior.Uniform(x_min, x_max)

        assert p.transform(0) == x_min
        assert p.transform(1) == x_max
        assert p.transform(0.3) == x_min + (x_max - x_min) * 0.3


class TestCosine:
    """Tests for the Cosine prior."""

    def test_initialisation_defaults(self) -> None:
        """Test Cosine prior initialisation with default parameters."""
        p = prior.Cosine()

        assert p.x_min == -constants.PI / 2
        assert p.x_max == constants.PI / 2
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_initialisation_custom(self) -> None:
        """Test Cosine prior initialisation with custom parameters."""
        x_min, x_max = 0, constants.PI
        p = prior.Cosine(x_min, x_max)

        assert p.x_min == x_min
        assert p.x_max == x_max

    def test_initialisation_with_boundary(self) -> None:
        """Test Cosine prior initialisation with boundary condition."""
        p = prior.Cosine(boundary="reflective")

        assert p.is_reflective is True

    def test_transform_boundaries(self) -> None:
        """Test that transform returns correct boundary values."""
        x_min, x_max = -constants.PI / 4, constants.PI / 4
        p = prior.Cosine(x_min, x_max)

        assert numpy.isclose(p.transform(0), x_min)
        assert numpy.isclose(p.transform(1), x_max)

    def test_transform_monotonic(self) -> None:
        """Test that transform is monotonically increasing."""
        p = prior.Cosine()
        u = numpy.linspace(0, 1, 100)
        x = [p.transform(y) for y in u]
        diffs = numpy.diff(x)

        assert numpy.all(diffs >= 0)

    def test_transform_returns_float64(self) -> None:
        """Test that transform returns numpy.float64."""
        p = prior.Cosine()
        x = p.transform(0.5)

        assert isinstance(x, numpy.float64)

    def test_transform_symmetric_range(self) -> None:
        """Test Cosine prior with symmetric range around zero."""
        p = prior.Cosine(-constants.PI / 3, constants.PI / 3)
        x = p.transform(0.5)

        assert numpy.isclose(x, 0)


class TestSine:
    """Tests for the Sine prior."""

    def test_initialisation_defaults(self) -> None:
        """Test Sine prior initialisation with default parameters."""
        p = prior.Sine()

        assert p.x_min == 0
        assert p.x_max == constants.PI
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_initialisation_custom(self) -> None:
        """Test Sine prior initialisation with custom parameters."""
        x_min, x_max = constants.PI / 4, 3 * constants.PI / 4
        p = prior.Sine(x_min, x_max)

        assert p.x_min == x_min
        assert p.x_max == x_max

    def test_initialisation_with_boundary(self) -> None:
        """Test Sine prior initialisation with boundary condition."""
        p = prior.Sine(boundary="periodic")

        assert p.is_periodic is True

    def test_transform_boundaries(self) -> None:
        """Test that transform returns correct boundary values."""
        x_min, x_max = constants.PI / 4, 3 * constants.PI / 4
        p = prior.Sine(x_min, x_max)

        assert p.transform(0) == x_min
        assert p.transform(1) == x_max

    def test_transform_monotonic(self) -> None:
        """Test that transform is monotonically increasing."""
        p = prior.Sine()
        u = numpy.linspace(0, 1, 100)
        x = [p.transform(y) for y in u]
        diffs = numpy.diff(x)

        assert numpy.all(diffs >= 0)

    def test_transform_returns_float64(self) -> None:
        """Test that transform returns numpy.float64."""
        p = prior.Sine()
        result = p.transform(0.5)

        assert isinstance(result, numpy.float64)

    def test_transform_default_range(self) -> None:
        """Test Sine prior with default range [0, pi]."""
        p = prior.Sine()
        x = p.transform(0.5)

        assert x == constants.PI / 2


class TestPriors:
    """Tests for the Priors collection class."""

    def test_initialisation_empty(self) -> None:
        """Test Priors initialisation with empty list."""
        priors = prior.Priors([])

        assert len(priors) == 0

    def test_initialisation_single(self) -> None:
        """Test Priors initialisation with single prior."""
        p = prior.Uniform(0, 1)
        priors = prior.Priors([p])

        assert len(priors) == 1
        assert priors[0] == p

    def test_initialisation_multiple(self) -> None:
        """Test Priors initialisation with multiple priors."""
        priors_list = [prior.Uniform(0, 1), prior.DeltaFunction(5), prior.Cosine()]
        priors = prior.Priors(priors_list)

        assert len(priors_list) == 3
        for i, p in enumerate(priors_list):
            assert priors[i] == p

    def test_transform_single_prior(self) -> None:
        """Test transform with single prior."""
        p = prior.Uniform(0, 10)
        priors = prior.Priors([p])
        u = numpy.array([0.5])
        x = priors.transform(u)

        assert x[0] == 5

    def test_transform_multiple_priors(self) -> None:
        """Test transform with multiple priors."""
        priors_list = [prior.Uniform(0, 2), prior.Uniform(0, 10), prior.DeltaFunction(7)]
        priors = prior.Priors(priors_list)
        u = numpy.array([0.5, 0.3, 0.999])
        x = priors.transform(u)

        assert x[0] == 1
        assert x[1] == 3
        assert x[2] == 7

    def test_transform_output_shape(self) -> None:
        """Test that transform output has correct shape."""
        priors_list: list[prior.Prior] = [prior.Uniform(0, 1), prior.Uniform(0, 1), prior.Uniform(0, 1)]
        priors = prior.Priors(priors_list)
        u = numpy.array([0.1, 0.2, 0.3])
        x = priors.transform(u)

        assert x.shape == u.shape

    def test_transform_preserves_input_shape(self) -> None:
        """Test that transform preserves input array shape."""
        priors_list: list[prior.Prior] = [prior.Uniform(0, 1), prior.Uniform(0, 1)]
        priors = prior.Priors(priors_list)
        u = numpy.array([0.2, 0.8])
        x = priors.transform(u)

        assert x.dtype == u.dtype

    def test_transform_does_not_modify_input(self) -> None:
        """Test that transform does not modify input array."""
        priors_list: list[prior.Prior] = [prior.Uniform(0, 1), prior.Uniform(0, 1)]
        priors = prior.Priors(priors_list)
        u = numpy.array([0.3, 0.7])
        u_copy = u.copy()
        priors.transform(u)

        assert numpy.array_equal(u, u_copy)

    def test_periodic_indices_none(self) -> None:
        """Test periodic_indices property when no periodic priors."""
        priors_list = [prior.Uniform(0, 1), prior.DeltaFunction(5)]
        priors = prior.Priors(priors_list)

        assert priors.periodic_indices is None

    def test_periodic_indices_single(self) -> None:
        """Test periodic_indices property with single periodic prior."""
        priors_list = [
            prior.Uniform(0, 1),
            prior.Uniform(0, 2, boundary="periodic"),
            prior.DeltaFunction(5),
        ]
        priors = prior.Priors(priors_list)

        assert priors.periodic_indices == [1]

    def test_periodic_indices_multiple(self) -> None:
        """Test periodic_indices property with multiple periodic priors."""
        priors_list: list[prior.Prior] = [
            prior.Uniform(0, 1, boundary="periodic"),
            prior.Uniform(0, 2),
            prior.Uniform(0, 3, boundary="periodic"),
        ]
        priors = prior.Priors(priors_list)

        assert priors.periodic_indices == [0, 2]

    def test_reflective_indices_none(self) -> None:
        """Test reflective_indices property when no reflective priors."""
        priors_list = [prior.Uniform(0, 1), prior.DeltaFunction(5)]
        priors = prior.Priors(priors_list)

        assert priors.reflective_indices is None

    def test_reflective_indices_single(self) -> None:
        """Test reflective_indices property with single reflective prior."""
        priors_list = [prior.Uniform(0, 1), prior.Uniform(0, 2, boundary="reflective"), prior.DeltaFunction(5)]
        priors = prior.Priors(priors_list)

        assert priors.reflective_indices == [1]

    def test_reflective_indices_multiple(self) -> None:
        """Test reflective_indices property with multiple reflective priors."""
        priors_list = [
            prior.Uniform(0, 1, boundary="reflective"),
            prior.Uniform(0, 2),
            prior.Sine(boundary="reflective"),
        ]
        priors = prior.Priors(priors_list)

        assert priors.reflective_indices == [0, 2]

    def test_mixed_boundary_conditions(self) -> None:
        """Test Priors with mixed boundary conditions."""
        priors_list: list[prior.Prior] = [
            prior.Uniform(0, 1, boundary="periodic"),
            prior.Uniform(0, 2, boundary="reflective"),
            prior.Uniform(0, 3),
        ]
        priors = prior.Priors(priors_list)

        assert priors.periodic_indices == [0]
        assert priors.reflective_indices == [1]

    def test_list_behavior(self) -> None:
        """Test that Priors behaves as a list."""
        p_1 = prior.Uniform(0, 1)
        p_2 = prior.DeltaFunction(5)
        priors = prior.Priors([p_1])
        priors.append(p_2)

        assert len(priors) == 2
        assert priors[1] == p_2

    def test_iteration(self) -> None:
        """Test that Priors can be iterated over."""
        priors_list = [prior.Uniform(0, 1), prior.DeltaFunction(5), prior.Cosine()]
        priors = prior.Priors(priors_list)

        for i, p in enumerate(priors):
            assert p == priors_list[i]
