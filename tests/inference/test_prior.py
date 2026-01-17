"""Tests for the prior module."""

import numpy

from chirplab.inference import distribution, prior


class TestPrior:
    """Tests for the Prior distribution."""

    def test_initialisation(self) -> None:
        """Test Prior initialisation."""
        priors = prior.Prior(
            (
                distribution.Uniform(0, 1),
                distribution.Uniform(1, 2),
                distribution.Uniform(10, 20),
                distribution.Sine(),
                distribution.Cosine(),
            )
        )

        assert priors.n_dim == 5
        assert priors.periodic_indices is None
        assert priors.reflective_indices is None

    def test_transform_applies_priors(self) -> None:
        """Test that transform maps unit samples through each Prior in order."""
        priors = prior.Prior((distribution.Uniform(0, 10), distribution.Uniform(5, 15), distribution.Uniform(100, 200)))
        q = numpy.array([0, 0.5, 1])
        q_copy = q.copy()
        x = priors.transform(q)

        assert x.shape == q.shape
        assert numpy.array_equal(q, q_copy)
        assert numpy.allclose(x, numpy.array([0, 10, 200]))

    def test_boundary_indices(self) -> None:
        """Test that periodic and reflective indices match Priors order."""
        priors = prior.Prior(
            (
                distribution.Uniform(0, 1, boundary="periodic"),
                distribution.Uniform(0, 1),
                distribution.Uniform(0, 1, boundary="reflective"),
                distribution.Sine(boundary="reflective"),
                distribution.Cosine(boundary="periodic"),
            )
        )

        assert priors.periodic_indices == [0, 4]
        assert priors.reflective_indices == [2, 3]

    def test_sample_returns_signal_parameters(self) -> None:
        """Test that sampling produces a SignalParameters instance with set random number generation."""
        distributions = (
            distribution.Uniform(0, 1),
            distribution.Uniform(1, 2),
            distribution.Uniform(10, 20),
            distribution.Sine(),
            distribution.DeltaFunction(1.5),
            distribution.Cosine(),
            distribution.DeltaFunction(0.3),
            distribution.DeltaFunction(0.4),
            distribution.DeltaFunction(0.5),
        )
        p = prior.Prior(distributions)
        rng_1 = numpy.random.default_rng(2024)
        x_1 = p.sample(rng_1)
        rng_2 = numpy.random.default_rng(2024)
        x_2 = numpy.array([d.sample(rng_2) for d in distributions])

        assert numpy.array_equal(x_1, x_2)
