"""Tests for the interferometer module."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy
import pytest

from chirplab import constants
from chirplab.simulation import interferometer

if TYPE_CHECKING:
    from chirplab.simulation import grid, parameters, waveform

RTOL = 0
ATOL = 1e-25


@pytest.fixture(scope="class")
def s_n_default() -> numpy.typing.NDArray[numpy.float64]:
    """Return default power spectral density for testing."""
    return numpy.ones(1_000, dtype=numpy.float64)


@pytest.fixture(scope="class")
def amplitude_spectral_density_file_default() -> Path:
    """Return path to default amplitude spectral density file for testing."""
    return Path(__file__).parent.parent.parent / "src/chirplab/simulation/data/aligo_O4high.txt"


@pytest.fixture(scope="class")
def interferometer_default(
    grid_default: grid.Grid, amplitude_spectral_density_file_default: Path
) -> interferometer.Interferometer:
    """Return default interferometer for testing."""
    return interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default)


class TestInterferometer:
    """Tests for the Interferometer class."""

    def test_initialisation(self, grid_default: grid.Grid, amplitude_spectral_density_file_default: Path) -> None:
        """Test that Interferometer can be initialised."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default)

        assert ifo.grid == grid_default

    def test_frequency_band(self, grid_default: grid.Grid, amplitude_spectral_density_file_default: Path) -> None:
        """Test that frequency band restriction works correctly."""
        f_min = 50
        f_max = 500
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, f_min, f_max)

        assert numpy.all(f_min <= ifo.f)
        assert numpy.all(ifo.f <= f_max)

    def test_noise_generation_zero_noise(
        self, grid_default: grid.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that zero noise realisation is correctly generated."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, is_zero_noise=True)

        assert numpy.allclose(ifo.d_tilde, 0, RTOL, ATOL)

    def test_noise_generation_with_rng(
        self,
        grid_default: grid.Grid,
        amplitude_spectral_density_file_default: Path,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that noise realisation is generated with random number generator."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_default)

        assert not numpy.allclose(ifo.d_tilde, 0, RTOL, ATOL)

    def test_noise_generation_reproducible(
        self, grid_default: grid.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that noise realisation is reproducible with same seed."""
        rng_1 = numpy.random.default_rng(42)
        ifo_1 = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_1)
        rng_2 = numpy.random.default_rng(42)
        ifo_2 = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_2)

        assert numpy.array_equal(ifo_1.d_tilde, ifo_2.d_tilde)

    def test_regenerate_noise(
        self,
        grid_default: grid.Grid,
        amplitude_spectral_density_file_default: Path,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that noise can be regenerated."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, is_zero_noise=True)
        ifo.set_data(rng=rng_default)

        assert not numpy.allclose(ifo.d_tilde, 0, RTOL, ATOL)

    @pytest.mark.parametrize(
        ("theta", "phi", "psi", "f_plus_expected", "f_cross_expected"),
        [
            (0, 0, 0, 1, 0),
            (constants.PI / 2, 0, 0, 1 / 2, 0),
            (0, constants.PI / 2, 0, -1, 0),
            (0, 0, constants.PI / 4, 0, -1),
            (constants.PI / 2, 0, constants.PI / 4, 0, -1 / 2),
            (0, constants.PI / 2, constants.PI / 4, 0, 1),
        ],
    )
    def test_pattern_functions(
        self,
        theta: float,
        phi: float,
        psi: float,
        f_plus_expected: float,
        f_cross_expected: float,
        interferometer_default: interferometer.Interferometer,
    ) -> None:
        """Test pattern function calculations for various angles."""
        f_plus, f_cross = interferometer_default.calculate_pattern_functions(theta, phi, psi)

        assert numpy.isclose(f_plus_expected, f_plus)
        assert numpy.isclose(f_cross_expected, f_cross)

    def test_pattern_functions_polarisation(self, interferometer_default: interferometer.Interferometer) -> None:
        """Test that pattern functions rotate correctly with polarisation angle."""
        theta, phi = constants.PI / 4, constants.PI / 6
        f_plus_0, f_cross_0 = interferometer_default.calculate_pattern_functions(theta, phi, 0)
        f_plus_45, f_cross_45 = interferometer_default.calculate_pattern_functions(theta, phi, constants.PI / 4)

        assert numpy.isclose(f_plus_0, -f_cross_45)
        assert numpy.isclose(f_cross_0, f_plus_45)

    def test_inner_product_orthogonality(self, interferometer_default: interferometer.Interferometer) -> None:
        """Test that inner product of orthogonal signals is zero."""
        a_tilde = numpy.ones_like(interferometer_default.f, dtype=numpy.complex128)
        b_tilde = 1j * numpy.ones_like(interferometer_default.f, dtype=numpy.complex128)
        a_inner_b = interferometer_default.calculate_inner_product(a_tilde, b_tilde)

        assert numpy.isclose(a_inner_b.real, 0)

    def test_inner_product_self_positive(self, interferometer_default: interferometer.Interferometer) -> None:
        """Test that inner product of signal with itself is positive."""
        a_tilde = numpy.ones_like(interferometer_default.f, dtype=numpy.complex128)
        a_inner_a = interferometer_default.calculate_inner_product(a_tilde, a_tilde)

        assert a_inner_a.real > 0
        assert numpy.isclose(a_inner_a.imag, 0)

    def test_inject_signal(
        self,
        grid_default: grid.Grid,
        amplitude_spectral_density_file_default: Path,
        model_default: waveform.WaveformModel,
        theta_default: parameters.SignalParameters,
    ) -> None:
        """Test signal injection into interferometer."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default)
        h_tilde, rho_opt, rho_mf = ifo.inject_signal(model_default, theta_default)

        assert not numpy.allclose(h_tilde, 0, RTOL, ATOL)
        assert rho_opt > 0
        assert rho_mf > 0

    def test_inject_signal_updates_data(
        self,
        grid_default: grid.Grid,
        amplitude_spectral_density_file_default: Path,
        model_default: waveform.WaveformModel,
        theta_default: parameters.SignalParameters,
    ) -> None:
        """Test that signal injection updates data stream."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default)
        d_tilde_before = ifo.d_tilde.copy()
        h_tilde, rho_opt, rho_mf = ifo.inject_signal(model_default, theta_default)

        assert not numpy.array_equal(ifo.d_tilde, d_tilde_before)
        assert numpy.array_equal(ifo.d_tilde, d_tilde_before + h_tilde)

    def test_inject_signal_matched_filter_signal_to_noise_ratio(
        self,
        grid_default: grid.Grid,
        amplitude_spectral_density_file_default: Path,
        model_default: waveform.WaveformModel,
        theta_default: parameters.SignalParameters,
    ) -> None:
        """Test matched filter signal-to-noise ratio equals optimal signal-to-noise ratio for zero noise."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, is_zero_noise=True)
        h_tilde, rho_opt, rho_mf = ifo.inject_signal(model_default, theta_default)

        assert numpy.isclose(rho_opt, abs(rho_mf))


class TestLLO:
    """Tests for the LLO class."""

    def test_initialisation(self, grid_default: grid.Grid) -> None:
        """Test that LLO can be initialised."""
        llo = interferometer.LLO(grid_default)

        assert llo.grid == grid_default
        assert 20 <= llo.f.min()
        assert llo.f.max() <= 2048


class TestLHO:
    """Tests for the LHO class."""

    def test_initialisation(self, grid_default: grid.Grid) -> None:
        """Test that LHO can be initialised."""
        lho = interferometer.LHO(grid_default)

        assert lho.grid == grid_default
        assert 20 <= lho.f.min()
        assert lho.f.max() <= 2048


class TestCalculateInnerProduct:
    """Tests for the calculate_inner_product function."""

    def test_inner_product_shape(self, s_n_default: numpy.typing.NDArray[numpy.float64]) -> None:
        """Test that inner product returns a scalar."""
        n = s_n_default.size
        a_tilde = numpy.ones(n, dtype=numpy.float64) * (1 + 1j)
        b_tilde = numpy.ones(n, dtype=numpy.float64) * (1 + 1j)
        delta_f = 1 / 4
        a_inner_b = interferometer.calculate_inner_product(a_tilde, b_tilde, s_n_default, delta_f)

        assert a_inner_b.dtype == numpy.complex128

    def test_inner_product_self_is_real(
        self, s_n_default: numpy.typing.NDArray[numpy.float64], rng_default: numpy.random.Generator
    ) -> None:
        """Test that inner product of signal with itself is real."""
        n = s_n_default.size
        a_tilde = rng_default.standard_normal(n) + 1j * rng_default.standard_normal(n)
        delta_f = 1 / 4
        a_inner_a = interferometer.calculate_inner_product(a_tilde, a_tilde, s_n_default, delta_f)

        assert numpy.isclose(a_inner_a.imag, 0)

    def test_inner_product_conjugate_symmetry(
        self, s_n_default: numpy.typing.NDArray[numpy.float64], rng_default: numpy.random.Generator
    ) -> None:
        """Test that inner product satisfies conjugate symmetry."""
        n = s_n_default.size
        a_tilde = rng_default.standard_normal(n) + 1j * rng_default.standard_normal(n)
        b_tilde = rng_default.standard_normal(n) + 1j * rng_default.standard_normal(n)
        delta_f = 1 / 4
        a_inner_b = interferometer.calculate_inner_product(a_tilde, b_tilde, s_n_default, delta_f)
        b_inner_a = interferometer.calculate_inner_product(b_tilde, a_tilde, s_n_default, delta_f)

        assert numpy.isclose(a_inner_b, b_inner_a.conj())

    def test_inner_product_scales_with_delta_f(self, s_n_default: numpy.typing.NDArray[numpy.float64]) -> None:
        """Test that inner product scales linearly with frequency resolution."""
        n = s_n_default.size
        a_tilde = numpy.ones(n, dtype=numpy.float64) * (1 + 1j)
        b_tilde = numpy.ones(n, dtype=numpy.float64) * (1 + 1j)
        delta_f_1, delta_f_2 = 1 / 4, 1 / 2
        a_inner_b_1 = interferometer.calculate_inner_product(a_tilde, b_tilde, s_n_default, delta_f_1)
        a_inner_b_2 = interferometer.calculate_inner_product(a_tilde, b_tilde, s_n_default, delta_f_2)

        assert a_inner_b_2 / a_inner_b_1 == delta_f_2 / delta_f_1

    def test_inner_product_size_mismatch_raises(self, s_n_default: numpy.typing.NDArray[numpy.float64]) -> None:
        """Test that inner product raises assertion error for mismatched sizes."""
        n = s_n_default.size
        a_tilde = numpy.ones(n, dtype=numpy.float64) * (1 + 1j)
        b_tilde = numpy.ones(n // 2, dtype=numpy.float64) * (1 + 1j)
        delta_f = 1 / 4

        with pytest.raises(AssertionError):
            interferometer.calculate_inner_product(a_tilde, b_tilde, s_n_default, delta_f)
