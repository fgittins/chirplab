"""Tests for the prior module."""

import numpy
import pytest

from chirplab import constants
from chirplab.inference import prior
from chirplab.simulation import interferometer


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

    def test_sample_no_rng(self) -> None:
        """Test that sampling without a provided random number generator gives different results."""
        p = prior.Uniform(2, 5)
        x_1 = p.sample()
        x_2 = p.sample()

        assert x_1 != x_2

    def test_sample_uses_rng(self) -> None:
        """Test that sampling uses the provided random number generator."""
        rng_1 = numpy.random.default_rng(1234)
        p = prior.Uniform(2, 5)
        x_1 = p.sample(rng_1)
        rng_2 = numpy.random.default_rng(1234)
        x_2 = p.transform(rng_2.uniform(0, 1))

        assert x_1 == x_2

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
    """Tests for the Priors dataclass."""

    def test_counts_and_names(self) -> None:
        """Test that sampled parameters are tracked in field order."""
        priors = prior.Priors(
            m_1=prior.Uniform(0, 1),
            m_2=prior.Uniform(1, 2),
            r=prior.Uniform(10, 20),
            iota=0.1,
            t_c=0.2,
            phi_c=prior.Sine(),
            theta=0.3,
            phi=0.4,
            psi=0.5,
        )

        assert priors.n == 4
        assert priors.theta_name_sample == ["m_1", "m_2", "r", "phi_c"]
        assert priors.theta_fixed == {"iota": 0.1, "t_c": 0.2, "theta": 0.3, "phi": 0.4, "psi": 0.5}

    def test_transform_applies_priors(self) -> None:
        """Test that transform maps unit samples through each Prior in order."""
        priors = prior.Priors(
            m_1=prior.Uniform(0, 10),
            m_2=prior.Uniform(5, 15),
            r=prior.Uniform(100, 200),
            iota=0.1,
            t_c=1,
            phi_c=2,
            theta=0.3,
            phi=0.4,
            psi=0.5,
        )
        u = numpy.array([0, 0.5, 1])
        u_copy = u.copy()
        x = priors.transform(u)

        assert x.shape == u.shape
        assert numpy.array_equal(u, u_copy)
        assert numpy.allclose(x, numpy.array([0, 10, 200]))

    def test_boundary_indices(self) -> None:
        """Test that periodic and reflective indices match Priors order."""
        priors = prior.Priors(
            m_1=prior.Uniform(0, 1, boundary="periodic"),
            m_2=prior.Uniform(0, 1),
            r=prior.Uniform(0, 1, boundary="reflective"),
            iota=prior.Sine(boundary="reflective"),
            t_c=1,
            phi_c=prior.Cosine(boundary="periodic"),
            theta=0.3,
            phi=0.4,
            psi=0.5,
        )

        assert priors.periodic_indices == [0, 4]
        assert priors.reflective_indices == [2, 3]

    def test_sample_returns_signal_parameters(self, rng_default: numpy.random.Generator) -> None:
        """Test that sampling produces a SignalParameters instance with set random number generation."""
        priors = prior.Priors(
            m_1=prior.Uniform(0, 1),
            m_2=2,
            r=prior.Uniform(10, 20),
            iota=prior.Sine(),
            t_c=1.5,
            phi_c=prior.Cosine(),
            theta=0.3,
            phi=0.4,
            psi=0.5,
        )
        rng_1 = numpy.random.default_rng(2024)
        theta_1 = priors.sample(rng_1)
        rng_2 = numpy.random.default_rng(2024)
        m_1 = priors.m_1.transform(rng_2.uniform(0, 1)) if isinstance(priors.m_1, prior.Prior) else priors.m_1
        r = priors.r.transform(rng_2.uniform(0, 1)) if isinstance(priors.r, prior.Prior) else priors.r
        iota = priors.iota.transform(rng_2.uniform(0, 1)) if isinstance(priors.iota, prior.Prior) else priors.iota
        phi_c = priors.phi_c.transform(rng_2.uniform(0, 1)) if isinstance(priors.phi_c, prior.Prior) else priors.phi_c
        theta_2 = interferometer.SignalParameters(
            m_1=m_1,
            m_2=2,
            r=r,
            iota=iota,
            t_c=1.5,
            phi_c=phi_c,
            theta=0.3,
            phi=0.4,
            psi=0.5,
        )

        assert isinstance(theta_1, interferometer.SignalParameters)
        assert theta_1 == theta_2
