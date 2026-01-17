"""Tests for the distribution module."""

import numpy

from chirplab import constants
from chirplab.inference import distribution


class TestDeltaFunction:
    """Tests for the DeltaFunction distribution."""

    def test_initialisation(self) -> None:
        """Test DeltaFunction distribution initialisation."""
        x_peak = 2.5
        p = distribution.DeltaFunction(x_peak)

        assert p.x_peak == x_peak
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_calculate_ppf(self) -> None:
        """Test that calculate_ppf always returns the fixed value."""
        x_peak = -1.0
        p = distribution.DeltaFunction(x_peak)
        q = numpy.array([0, 0.25, 0.5, 0.75, 1])
        x = numpy.array([p.calculate_ppf(y) for y in q])

        assert numpy.all(x == x_peak)

    def test_sample(self) -> None:
        """Test that sampling gives the fixed value."""
        x_peak = 3.14
        p = distribution.DeltaFunction(x_peak)
        x_1 = p.sample()
        x_2 = p.sample()

        assert x_1 == x_peak
        assert x_2 == x_peak


class TestUniform:
    """Tests for the Uniform distribution."""

    def test_initialisation(self) -> None:
        """Test Uniform distribution initialisation."""
        x_min, x_max = 0, 1
        p = distribution.Uniform(x_min, x_max)

        assert p.x_min == x_min
        assert p.x_max == x_max
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_initialisation_with_boundary(self) -> None:
        """Test Uniform distribution initialisation with boundary condition."""
        p = distribution.Uniform(0, 1, "periodic")

        assert p.is_periodic is True

    def test_calculate_ppf_boundaries(self) -> None:
        """Test that calculate_ppf returns correct boundary values."""
        x_min, x_max = 2, 5
        p = distribution.Uniform(x_min, x_max)

        assert p.calculate_ppf(0) == x_min
        assert p.calculate_ppf(1) == x_max

    def test_calculate_ppf_midpoint(self) -> None:
        """Test that calculate_ppf returns correct midpoint value."""
        x_min, x_max = 0, 10
        p = distribution.Uniform(x_min, x_max)

        assert p.calculate_ppf(0.5) == (x_min + x_max) / 2

    def test_calculate_ppf_linear(self) -> None:
        """Test that calculate_ppf is linear."""
        x_min, x_max = -5, 15
        p = distribution.Uniform(x_min, x_max)
        q = numpy.array([0, 0.25, 0.5, 0.75, 1])
        x = numpy.array([p.calculate_ppf(y) for y in q])

        assert all(x == x_min + (x_max - x_min) * q)

    def test_sample_no_rng(self) -> None:
        """Test that sampling without a provided random number generator gives different results."""
        p = distribution.Uniform(2, 5)
        x_1 = p.sample()
        x_2 = p.sample()

        assert x_1 != x_2

    def test_sample_uses_rng(self) -> None:
        """Test that sampling uses the provided random number generator."""
        rng_1 = numpy.random.default_rng(1234)
        p = distribution.Uniform(2, 5)
        x_1 = p.sample(rng_1)
        rng_2 = numpy.random.default_rng(1234)
        q = rng_2.uniform(0, 1)
        x_2 = p.calculate_ppf(q)

        assert x_1 == x_2

    def test_negative_range(self) -> None:
        """Test Uniform distribution with negative range."""
        x_min, x_max = -10, -2
        p = distribution.Uniform(x_min, x_max)

        assert p.calculate_ppf(0) == x_min
        assert p.calculate_ppf(1) == x_max

    def test_range_crossing_zero(self) -> None:
        """Test Uniform distribution with range crossing zero."""
        x_min, x_max = -3, 7
        p = distribution.Uniform(x_min, x_max)

        assert p.calculate_ppf(0) == x_min
        assert p.calculate_ppf(1) == x_max
        assert p.calculate_ppf(0.3) == x_min + (x_max - x_min) * 0.3


class TestCosine:
    """Tests for the Cosine distribution."""

    def test_initialisation_defaults(self) -> None:
        """Test Cosine distribution initialisation with default parameters."""
        p = distribution.Cosine()

        assert p.x_min == -constants.PI / 2
        assert p.x_max == constants.PI / 2
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_initialisation_custom(self) -> None:
        """Test Cosine distribution initialisation with custom parameters."""
        x_min, x_max = 0, constants.PI
        p = distribution.Cosine(x_min, x_max)

        assert p.x_min == x_min
        assert p.x_max == x_max

    def test_initialisation_with_boundary(self) -> None:
        """Test Cosine distribution initialisation with boundary condition."""
        p = distribution.Cosine(boundary="reflective")

        assert p.is_reflective is True

    def test_calculate_ppf_boundaries(self) -> None:
        """Test that calculate_ppf returns correct boundary values."""
        x_min, x_max = -constants.PI / 4, constants.PI / 4
        p = distribution.Cosine(x_min, x_max)

        assert numpy.isclose(p.calculate_ppf(0), x_min)
        assert numpy.isclose(p.calculate_ppf(1), x_max)

    def test_calculate_ppf_monotonic(self) -> None:
        """Test that calculate_ppf is monotonically increasing."""
        p = distribution.Cosine()
        q = numpy.linspace(0, 1, 100)
        x = [p.calculate_ppf(y) for y in q]
        diffs = numpy.diff(x)

        assert numpy.all(diffs >= 0)

    def test_calculate_ppf_returns_float64(self) -> None:
        """Test that calculate_ppf returns numpy.float64."""
        p = distribution.Cosine()
        x = p.calculate_ppf(0.5)

        assert isinstance(x, numpy.float64)

    def test_calculate_ppf_symmetric_range(self) -> None:
        """Test Cosine distribution with symmetric range around zero."""
        p = distribution.Cosine(-constants.PI / 3, constants.PI / 3)
        x = p.calculate_ppf(0.5)

        assert numpy.isclose(x, 0)


class TestSine:
    """Tests for the Sine distribution."""

    def test_initialisation_defaults(self) -> None:
        """Test Sine distribution initialisation with default parameters."""
        p = distribution.Sine()

        assert p.x_min == 0
        assert p.x_max == constants.PI
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_initialisation_custom(self) -> None:
        """Test Sine distribution initialisation with custom parameters."""
        x_min, x_max = constants.PI / 4, 3 * constants.PI / 4
        p = distribution.Sine(x_min, x_max)

        assert p.x_min == x_min
        assert p.x_max == x_max

    def test_initialisation_with_boundary(self) -> None:
        """Test Sine distribution initialisation with boundary condition."""
        p = distribution.Sine(boundary="periodic")

        assert p.is_periodic is True

    def test_calculate_ppf_boundaries(self) -> None:
        """Test that calculate_ppf returns correct boundary values."""
        x_min, x_max = constants.PI / 4, 3 * constants.PI / 4
        p = distribution.Sine(x_min, x_max)

        assert p.calculate_ppf(0) == x_min
        assert p.calculate_ppf(1) == x_max

    def test_calculate_ppf_monotonic(self) -> None:
        """Test that calculate_ppf is monotonically increasing."""
        p = distribution.Sine()
        q = numpy.linspace(0, 1, 100)
        x = [p.calculate_ppf(y) for y in q]
        diffs = numpy.diff(x)

        assert numpy.all(diffs >= 0)

    def test_calculate_ppf_returns_float64(self) -> None:
        """Test that calculate_ppf returns numpy.float64."""
        p = distribution.Sine()
        result = p.calculate_ppf(0.5)

        assert isinstance(result, numpy.float64)

    def test_calculate_ppf_default_range(self) -> None:
        """Test Sine distribution with default range [0, pi]."""
        p = distribution.Sine()
        x = p.calculate_ppf(0.5)

        assert x == constants.PI / 2


class TestGaussian:
    """Tests for the Gaussian distribution."""

    def test_initialisation(self) -> None:
        """Test Gaussian distribution initialisation."""
        mu, sigma = 1, 2
        p = distribution.Gaussian(mu, sigma, "periodic")

        assert p.mu == mu
        assert p.sigma == sigma
        assert p.is_periodic is True
        assert p.is_reflective is False

    def test_calculate_ppf_at_mean(self) -> None:
        """Test that calculate_ppf at 0.5 returns the mean."""
        mu, sigma = 2, 0.5
        p = distribution.Gaussian(mu, sigma)

        assert p.calculate_ppf(0.5) == mu


class TestUniformComovingVolume:
    """Tests for the UniformComovingVolume distribution."""

    def test_initialisation(self) -> None:
        """Test UniformComovingVolume distribution initialisation."""
        r_min, r_max = 1e6 * constants.PC, 1e9 * constants.PC
        p = distribution.UniformComovingVolume(r_min, r_max)

        assert p.r_min == r_min
        assert p.r_max == r_max
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_calculate_ppf_boundaries(self) -> None:
        """Test that calculate_ppf returns correct boundary values."""
        r_min, r_max = 1e6 * constants.PC, 1e9 * constants.PC
        p = distribution.UniformComovingVolume(r_min, r_max)

        assert numpy.isclose(p.calculate_ppf(0), r_min)
        assert numpy.isclose(p.calculate_ppf(1), r_max)
