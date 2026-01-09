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

    def test_calculate_ppf_boundaries(self) -> None:
        """Test that calculate_ppf returns correct boundary values."""
        x_min, x_max = 2, 5
        p = prior.Uniform(x_min, x_max)

        assert p.calculate_ppf(0) == x_min
        assert p.calculate_ppf(1) == x_max

    def test_calculate_ppf_midpoint(self) -> None:
        """Test that calculate_ppf returns correct midpoint value."""
        x_min, x_max = 0, 10
        p = prior.Uniform(x_min, x_max)

        assert p.calculate_ppf(0.5) == (x_min + x_max) / 2

    def test_calculate_ppf_linear(self) -> None:
        """Test that calculate_ppf is linear."""
        x_min, x_max = -5, 15
        p = prior.Uniform(x_min, x_max)
        q = numpy.array([0, 0.25, 0.5, 0.75, 1])
        x = numpy.array([p.calculate_ppf(y) for y in q])

        assert all(x == x_min + (x_max - x_min) * q)

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
        q = rng_2.uniform(0, 1)
        x_2 = p.calculate_ppf(q)

        assert x_1 == x_2

    def test_negative_range(self) -> None:
        """Test Uniform prior with negative range."""
        x_min, x_max = -10, -2
        p = prior.Uniform(x_min, x_max)

        assert p.calculate_ppf(0) == x_min
        assert p.calculate_ppf(1) == x_max

    def test_range_crossing_zero(self) -> None:
        """Test Uniform prior with range crossing zero."""
        x_min, x_max = -3, 7
        p = prior.Uniform(x_min, x_max)

        assert p.calculate_ppf(0) == x_min
        assert p.calculate_ppf(1) == x_max
        assert p.calculate_ppf(0.3) == x_min + (x_max - x_min) * 0.3


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

    def test_calculate_ppf_boundaries(self) -> None:
        """Test that calculate_ppf returns correct boundary values."""
        x_min, x_max = -constants.PI / 4, constants.PI / 4
        p = prior.Cosine(x_min, x_max)

        assert numpy.isclose(p.calculate_ppf(0), x_min)
        assert numpy.isclose(p.calculate_ppf(1), x_max)

    def test_calculate_ppf_monotonic(self) -> None:
        """Test that calculate_ppf is monotonically increasing."""
        p = prior.Cosine()
        q = numpy.linspace(0, 1, 100)
        x = [p.calculate_ppf(y) for y in q]
        diffs = numpy.diff(x)

        assert numpy.all(diffs >= 0)

    def test_calculate_ppf_returns_float64(self) -> None:
        """Test that calculate_ppf returns numpy.float64."""
        p = prior.Cosine()
        x = p.calculate_ppf(0.5)

        assert isinstance(x, numpy.float64)

    def test_calculate_ppf_symmetric_range(self) -> None:
        """Test Cosine prior with symmetric range around zero."""
        p = prior.Cosine(-constants.PI / 3, constants.PI / 3)
        x = p.calculate_ppf(0.5)

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

    def test_calculate_ppf_boundaries(self) -> None:
        """Test that calculate_ppf returns correct boundary values."""
        x_min, x_max = constants.PI / 4, 3 * constants.PI / 4
        p = prior.Sine(x_min, x_max)

        assert p.calculate_ppf(0) == x_min
        assert p.calculate_ppf(1) == x_max

    def test_calculate_ppf_monotonic(self) -> None:
        """Test that calculate_ppf is monotonically increasing."""
        p = prior.Sine()
        q = numpy.linspace(0, 1, 100)
        x = [p.calculate_ppf(y) for y in q]
        diffs = numpy.diff(x)

        assert numpy.all(diffs >= 0)

    def test_calculate_ppf_returns_float64(self) -> None:
        """Test that calculate_ppf returns numpy.float64."""
        p = prior.Sine()
        result = p.calculate_ppf(0.5)

        assert isinstance(result, numpy.float64)

    def test_calculate_ppf_default_range(self) -> None:
        """Test Sine prior with default range [0, pi]."""
        p = prior.Sine()
        x = p.calculate_ppf(0.5)

        assert x == constants.PI / 2


class TestGaussian:
    """Tests for the Gaussian prior."""

    def test_initialisation(self) -> None:
        """Test Gaussian prior initialisation."""
        mu, sigma = 1, 2
        p = prior.Gaussian(mu, sigma, "periodic")

        assert p.mu == mu
        assert p.sigma == sigma
        assert p.is_periodic is True
        assert p.is_reflective is False

    def test_calculate_ppf_at_mean(self) -> None:
        """Test that calculate_ppf at 0.5 returns the mean."""
        mu, sigma = 2, 0.5
        p = prior.Gaussian(mu, sigma)

        assert p.calculate_ppf(0.5) == mu


class TestUniformComovingVolume:
    """Tests for the UniformComovingVolume prior."""

    def test_initialisation(self) -> None:
        """Test UniformComovingVolume prior initialisation."""
        r_min, r_max = 1e6 * constants.PC, 1e9 * constants.PC
        p = prior.UniformComovingVolume(r_min, r_max)

        assert p.r_min == r_min
        assert p.r_max == r_max
        assert p.is_periodic is False
        assert p.is_reflective is False

    def test_calculate_ppf_boundaries(self) -> None:
        """Test that calculate_ppf returns correct boundary values."""
        r_min, r_max = 1e6 * constants.PC, 1e9 * constants.PC
        p = prior.UniformComovingVolume(r_min, r_max)

        assert numpy.isclose(p.calculate_ppf(0), r_min)
        assert numpy.isclose(p.calculate_ppf(1), r_max)


class TestPriors:
    """Tests for the Priors dataclass."""

    def test_initialisation(self) -> None:
        """Test Priors dataclass initialisation."""
        priors = prior.Priors(
            m_1=prior.Uniform(0, 1),
            m_2=prior.Uniform(1, 2),
            r=prior.Uniform(10, 20),
            iota=prior.Sine(),
            t_c=0.5,
            phi_c=prior.Cosine(),
            theta=0.1,
            phi=0.2,
            psi=0.3,
        )

        assert isinstance(priors.m_1, prior.Prior)
        assert isinstance(priors.m_2, prior.Prior)
        assert isinstance(priors.r, prior.Prior)
        assert isinstance(priors.iota, prior.Prior)
        assert isinstance(priors.t_c, float)
        assert isinstance(priors.phi_c, prior.Prior)
        assert isinstance(priors.theta, float)
        assert isinstance(priors.phi, float)
        assert isinstance(priors.psi, float)

    def test_initialisation_missing_parameters(self) -> None:
        """Test Priors initialisation raises ValueError when required parameters are missing."""
        with pytest.raises(ValueError, match="Either \\(m_1 and m_2\\) or \\(m_chirp and q\\) must be provided."):
            prior.Priors(  # type: ignore[call-overload]
                r=prior.Uniform(10, 20),
                iota=prior.Sine(),
                t_c=0.5,
                phi_c=prior.Cosine(),
                theta=0.1,
                phi=0.2,
                psi=0.3,
            )

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

    def test_calculate_ppf_applies_priors(self) -> None:
        """Test that calculate_ppf maps unit samples through each Prior in order."""
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
        q = numpy.array([0, 0.5, 1])
        q_copy = q.copy()
        x = priors.calculate_ppf(q)

        assert x.shape == q.shape
        assert numpy.array_equal(q, q_copy)
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
        assert isinstance(priors.m_1, prior.Prior)
        m_1 = priors.m_1.calculate_ppf(rng_2.uniform(0, 1))
        assert isinstance(priors.r, prior.Prior)
        r = priors.r.calculate_ppf(rng_2.uniform(0, 1))
        assert isinstance(priors.iota, prior.Prior)
        iota = priors.iota.calculate_ppf(rng_2.uniform(0, 1))
        assert isinstance(priors.phi_c, prior.Prior)
        phi_c = priors.phi_c.calculate_ppf(rng_2.uniform(0, 1))
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
