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
        assert isinstance(ifo.x, numpy.ndarray)
        assert isinstance(ifo.d, numpy.ndarray)
        assert isinstance(ifo.in_bounds_mask, numpy.ndarray)
        assert isinstance(ifo.f, numpy.ndarray)
        assert isinstance(ifo.s_n, numpy.ndarray)

    def test_noise_generation_zero_noise(
        self, grid_default: grid.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that zero noise realisation is correctly generated."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, is_zero_noise=True)

        assert numpy.allclose(ifo.s_tilde, 0, RTOL, ATOL)

    def test_noise_generation_with_rng(
        self,
        grid_default: grid.Grid,
        amplitude_spectral_density_file_default: Path,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that noise realisation is generated with random number generator."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_default)

        assert not numpy.allclose(ifo.s_tilde, 0, RTOL, ATOL)

    def test_noise_generation_reproducible(
        self, grid_default: grid.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that noise realisation is reproducible with same seed."""
        rng_1 = numpy.random.default_rng(42)
        ifo_1 = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_1)
        rng_2 = numpy.random.default_rng(42)
        ifo_2 = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_2)

        assert numpy.array_equal(ifo_1.s_tilde, ifo_2.s_tilde)

    def test_regenerate_noise(
        self,
        grid_default: grid.Grid,
        amplitude_spectral_density_file_default: Path,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that noise can be regenerated."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, is_zero_noise=True)
        ifo.set_data(rng=rng_default)

        assert not numpy.allclose(ifo.s_tilde, 0, RTOL, ATOL)

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
        d_tilde_before = ifo.s_tilde.copy()
        h_tilde, rho_opt, rho_mf = ifo.inject_signal(model_default, theta_default)

        assert not numpy.array_equal(ifo.s_tilde, d_tilde_before)
        assert numpy.array_equal(ifo.s_tilde, d_tilde_before + h_tilde)

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
        assert isinstance(llo.x, numpy.ndarray)
        assert isinstance(llo.d, numpy.ndarray)
        assert isinstance(llo.in_bounds_mask, numpy.ndarray)
        assert isinstance(llo.f, numpy.ndarray)
        assert isinstance(llo.s_n, numpy.ndarray)


class TestLHO:
    """Tests for the LHO class."""

    def test_initialisation(self, grid_default: grid.Grid) -> None:
        """Test that LHO can be initialised."""
        lho = interferometer.LHO(grid_default)

        assert lho.grid == grid_default
        assert isinstance(lho.x, numpy.ndarray)
        assert isinstance(lho.d, numpy.ndarray)
        assert isinstance(lho.in_bounds_mask, numpy.ndarray)
        assert isinstance(lho.f, numpy.ndarray)
        assert isinstance(lho.s_n, numpy.ndarray)


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


class TestCalculatePolarisationTensors:
    """Tests for the calculate_polarisation_tensors function."""

    def test_polarisation_tensors_orthogonality(self) -> None:
        """Test that polarisation tensors are orthogonal."""
        theta_array = numpy.linspace(0, constants.PI, 10)
        phi_array = numpy.linspace(0, 2 * constants.PI, 10)
        psi_array = numpy.linspace(0, constants.PI, 10)
        for theta, phi, psi in zip(theta_array, phi_array, psi_array, strict=False):
            e_plus, e_cross = interferometer.calculate_polarisation_tensors(theta, phi, psi)
            dot_product = numpy.tensordot(e_plus, e_cross)

            assert numpy.isclose(dot_product, 0)

    def test_polarisation_tensors_self_product(self) -> None:
        """Test that self-product of polarisation tensors equals 2."""
        theta_array = numpy.linspace(0, constants.PI, 10)
        phi_array = numpy.linspace(0, 2 * constants.PI, 10)
        psi_array = numpy.linspace(0, constants.PI, 10)
        for theta, phi, psi in zip(theta_array, phi_array, psi_array, strict=False):
            e_plus, e_cross = interferometer.calculate_polarisation_tensors(theta, phi, psi)
            dot_product_plus = numpy.tensordot(e_plus, e_plus)
            dot_product_cross = numpy.tensordot(e_cross, e_cross)

            assert numpy.isclose(dot_product_plus, 2)
            assert numpy.isclose(dot_product_cross, 2)

    def test_case(self) -> None:
        """Test polarisation tensors for a specific case."""
        theta = phi = psi = 0
        e_plus, e_cross = interferometer.calculate_polarisation_tensors(theta, phi, psi)
        e_plus_expected = numpy.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        e_cross_expected = numpy.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        assert numpy.array_equal(e_plus, e_plus_expected)
        assert numpy.array_equal(e_cross, e_cross_expected)

    def test_polarisation(self) -> None:
        """Test that polarisation tensors rotate correctly with polarisation angle."""
        theta = phi = 0
        e_plus_0, e_cross_0 = interferometer.calculate_polarisation_tensors(theta, phi, 0)
        psi_array = numpy.linspace(0, constants.PI, 10)
        for psi in psi_array:
            e_plus, e_cross = interferometer.calculate_polarisation_tensors(theta, phi, psi)

            assert numpy.allclose(e_plus, numpy.cos(2 * psi) * e_plus_0 + numpy.sin(2 * psi) * e_cross_0)
            assert numpy.allclose(e_cross, -numpy.sin(2 * psi) * e_plus_0 + numpy.cos(2 * psi) * e_cross_0)


class TestCalculateTimeDelay:
    """Tests for the calculate_time_delay function."""

    def test_time_delay_zero_direction(self) -> None:
        """Test that time delay is zero for source at zenith."""
        theta = phi = 0
        x_1 = numpy.array([0, 0, 0])
        x_2 = numpy.array([1000, 1000, 0])
        delta_t = interferometer.calculate_time_delay(x_1, x_2, theta, phi)

        assert numpy.isclose(delta_t, 0)

    def test_time_delay_opposite_directions(self) -> None:
        """Test that time delays for opposite directions have opposite signs."""
        theta, phi = constants.PI / 4, constants.PI / 3
        x_1 = numpy.array([0, 0, 0])
        x_2 = numpy.array([1000, 1000, 0])
        delta_t_1 = interferometer.calculate_time_delay(x_1, x_2, theta, phi)
        delta_t_2 = interferometer.calculate_time_delay(x_1, x_2, constants.PI - theta, phi + constants.PI)

        assert numpy.isclose(delta_t_1, -delta_t_2)
