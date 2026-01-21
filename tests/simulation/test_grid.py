"""Tests for the grid module."""

import numpy
import pytest

from chirplab.simulation import grid


class TestGrid:
    """Tests for the Grid dataclass."""

    def test_initialisation(self) -> None:
        """Test that Grid can be initialised with valid parameters."""
        g = grid.Grid(4, 4096)

        assert g.t_d == 4
        assert g.f_s == 4096

    def test_invalid_product_t_d_f_s(self) -> None:
        """Test that Grid raises ValueError when t_d * f_s is not an integer."""
        with pytest.raises(ValueError, match="The product of t_d and f_s must be an integer"):
            grid.Grid(4.1, 4096)

    def test_invalid_odd_n(self) -> None:
        """Test that Grid raises ValueError when n is odd."""
        with pytest.raises(ValueError, match="The product of t_d and f_s must be even"):
            grid.Grid(1, 4095)

    def test_f_max_property(self, grid_default: grid.Grid) -> None:
        """Test that the f_max property returns the Nyquist frequency."""
        assert grid_default.f_max == 2048

    def test_n_property(self, grid_default: grid.Grid) -> None:
        """Test that the n property returns the correct number of time samples."""
        assert grid_default.n == 16384

    def test_m_property(self, grid_default: grid.Grid) -> None:
        """Test that the m property returns the correct number of frequency samples."""
        assert grid_default.m == 8193

    def test_delta_t_property(self, grid_default: grid.Grid) -> None:
        """Test that the delta_t property returns the correct time resolution."""
        assert grid_default.delta_t == 1 / 4096

    def test_delta_f_property(self, grid_default: grid.Grid) -> None:
        """Test that the delta_f property returns the correct frequency resolution."""
        assert grid_default.delta_f == 1 / 4

    def test_t_property(self, grid_default: grid.Grid) -> None:
        """Test that the t property returns the correct time array."""
        assert grid_default.t.size == grid_default.n
        assert grid_default.t[0] == 0
        assert grid_default.t[-1] == grid_default.t_d - grid_default.delta_t
        assert numpy.all(numpy.diff(grid_default.t) == grid_default.delta_t)

    def test_f_property(self, grid_default: grid.Grid) -> None:
        """Test that the f property returns the correct frequency array."""
        assert grid_default.f.size == grid_default.m
        assert grid_default.f[0] == 0
        assert grid_default.f[-1] == grid_default.f_max
        assert numpy.all(numpy.diff(grid_default.f) == grid_default.delta_f)

    def test_generate_gaussian_noise(self, grid_default: grid.Grid, rng_default: numpy.random.Generator) -> None:
        """Test that generate_gaussian_noise returns noise of correct shape."""
        n_tilde = grid_default.generate_gaussian_noise(rng_default)

        assert n_tilde.shape == (grid_default.m,)
        assert n_tilde.dtype == numpy.complex128

    def test_generate_gaussian_noise_reproducible(self, grid_default: grid.Grid) -> None:
        """Test that generate_gaussian_noise is reproducible with same seed."""
        rng_1 = numpy.random.default_rng(42)
        rng_2 = numpy.random.default_rng(42)

        assert numpy.array_equal(
            grid_default.generate_gaussian_noise(rng_1), grid_default.generate_gaussian_noise(rng_2)
        )

    def test_generate_gaussian_noise_different_seeds(self, grid_default: grid.Grid) -> None:
        """Test that generate_gaussian_noise produces different noise with different seeds."""
        rng_1 = numpy.random.default_rng(42)
        rng_2 = numpy.random.default_rng(43)

        assert not numpy.array_equal(
            grid_default.generate_gaussian_noise(rng_1), grid_default.generate_gaussian_noise(rng_2)
        )

    def test_generate_gaussian_noise_endpoints_zero(
        self, grid_default: grid.Grid, rng_default: numpy.random.Generator
    ) -> None:
        """Test that generate_gaussian_noise returns zero noise at direct-current and Nyquist frequencies."""
        n_tilde = grid_default.generate_gaussian_noise(rng_default)

        assert n_tilde[0] == 0
        assert n_tilde[-1] == 0

    def test_generate_gaussian_noise_expectation_values(
        self, grid_default: grid.Grid, rng_default: numpy.random.Generator
    ) -> None:
        """Test that generate_gaussian_noise returns noise with correct expectation values."""
        n_tilde = grid_default.generate_gaussian_noise(rng_default)
        rtol = 0
        atol = 0.02

        assert numpy.isclose(n_tilde[1:-1].mean(), 0, rtol, atol)
        assert numpy.isclose((n_tilde[1:-1].conj() * n_tilde[1:-1]).mean(), 1 / 2 * grid_default.t_d, rtol, atol)
