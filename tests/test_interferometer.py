"""Unit tests for the interferometer module."""

from dataclasses import replace
from pathlib import Path

import numpy
import pytest

from chirplab import interferometer, waveform

RTOL = 0
ATOL = 1e-27


@pytest.fixture(scope="class")
def s_n_default() -> numpy.typing.NDArray[numpy.float64]:
    """Return default power spectral density for testing."""
    return numpy.ones(1_000, dtype=numpy.float64)


@pytest.fixture(scope="class")
def amplitude_spectral_density_file_default() -> Path:
    """Return path to default amplitude spectral density file for testing."""
    return Path(__file__).parent.parent / "src/chirplab/data/aligo_O4high.txt"


@pytest.fixture(scope="class")
def interferometer_default(
    grid_default: interferometer.Grid, amplitude_spectral_density_file_default: Path
) -> interferometer.Interferometer:
    """Return default interferometer for testing."""
    return interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default)


class TestGrid:
    """Tests for the Grid dataclass."""

    def test_initialisation(self, grid_default: interferometer.Grid) -> None:
        """Test that Grid can be initialised with valid parameters."""
        assert grid_default.t_d == 4
        assert grid_default.f_s == 4096

    def test_invalid_product_t_d_f_s(self) -> None:
        """Test that Grid raises ValueError when t_d * f_s is not an integer."""
        with pytest.raises(ValueError, match="The product of t_d and f_s must be an integer"):
            interferometer.Grid(t_d=4.1, f_s=4096)

    def test_invalid_odd_n(self) -> None:
        """Test that Grid raises ValueError when n is odd."""
        with pytest.raises(ValueError, match="The number of time samples n must be even"):
            interferometer.Grid(t_d=1, f_s=4095)

    def test_f_max_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the f_max property returns the Nyquist frequency."""
        assert grid_default.f_max == 2048

    def test_n_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the n property returns the correct number of time samples."""
        assert grid_default.n == 16384

    def test_m_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the m property returns the correct number of frequency samples."""
        assert grid_default.m == 8192

    def test_delta_t_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the delta_t property returns the correct time resolution."""
        assert grid_default.delta_t == 1 / 4096

    def test_delta_f_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the delta_f property returns the correct frequency resolution."""
        assert grid_default.delta_f == 1 / 4

    def test_t_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the t property returns the correct time array."""
        assert grid_default.t.size == grid_default.n + 1
        assert grid_default.t[0] == 0
        assert grid_default.t[-1] == grid_default.t_d
        assert numpy.all(numpy.diff(grid_default.t) == grid_default.delta_t)

    def test_f_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the f property returns the correct frequency array."""
        assert grid_default.f.size == grid_default.m + 1
        assert grid_default.f[0] == 0
        assert grid_default.f[-1] == grid_default.f_max
        assert numpy.all(numpy.diff(grid_default.f) == grid_default.delta_f)

    def test_generate_gaussian_noise(
        self, grid_default: interferometer.Grid, rng_default: numpy.random.Generator
    ) -> None:
        """Test that generate_gaussian_noise returns noise of correct shape."""
        n_tilde = grid_default.generate_gaussian_noise(rng_default)

        assert n_tilde.shape == (grid_default.m + 1,)

    def test_generate_gaussian_noise_reproducible(self, grid_default: interferometer.Grid) -> None:
        """Test that generate_gaussian_noise is reproducible with same seed."""
        rng_1 = numpy.random.default_rng(42)
        rng_2 = numpy.random.default_rng(42)

        assert numpy.array_equal(
            grid_default.generate_gaussian_noise(rng_1), grid_default.generate_gaussian_noise(rng_2)
        )

    def test_generate_gaussian_noise_different_seeds(self, grid_default: interferometer.Grid) -> None:
        """Test that generate_gaussian_noise produces different noise with different seeds."""
        rng_1 = numpy.random.default_rng(42)
        rng_2 = numpy.random.default_rng(43)

        assert not numpy.array_equal(
            grid_default.generate_gaussian_noise(rng_1), grid_default.generate_gaussian_noise(rng_2)
        )

    def test_generate_gaussian_noise_endpoints_zero(
        self, grid_default: interferometer.Grid, rng_default: numpy.random.Generator
    ) -> None:
        """Test that generate_gaussian_noise returns noise that is zero at direct-current and Nyquist frequencies."""
        n_tilde = grid_default.generate_gaussian_noise(rng_default)

        assert n_tilde[0] == 0
        assert n_tilde[-1] == 0

    def test_generate_gaussian_noise_dtype(
        self, grid_default: interferometer.Grid, rng_default: numpy.random.Generator
    ) -> None:
        """Test that generate_gaussian_noise returns noise of correct dtype."""
        n_tilde = grid_default.generate_gaussian_noise(rng_default)

        assert n_tilde.dtype == numpy.complex128

    def test_generate_gaussian_noise_expectation_values(
        self, grid_default: interferometer.Grid, rng_default: numpy.random.Generator
    ) -> None:
        """Test that generate_gaussian_noise returns noise with correct expectation values."""
        n_tilde = grid_default.generate_gaussian_noise(rng_default)
        atol = 0.02

        assert numpy.isclose(n_tilde[1:-2].mean(), 0, RTOL, atol)
        assert numpy.isclose((numpy.abs(n_tilde[1:-2]) ** 2).mean(), 1 / 2 * grid_default.t_d, RTOL, atol)


class TestInterferometer:
    """Tests for the Interferometer class."""

    def test_initialisation(
        self, grid_default: interferometer.Grid, interferometer_default: interferometer.Interferometer
    ) -> None:
        """Test that Interferometer can be initialised."""
        assert interferometer_default.grid == grid_default

    def test_frequency_band(
        self, grid_default: interferometer.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that frequency band restriction works correctly."""
        f_min_sens = 50
        f_max_sens = 500
        band = (f_min_sens, f_max_sens)
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, band=band)

        assert numpy.all(f_min_sens <= ifo.f)
        assert numpy.all(ifo.f <= f_max_sens)

    def test_noise_generation_zero_noise(
        self, grid_default: interferometer.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that zero noise realisation is correctly generated."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, is_zero_noise=True)

        assert numpy.allclose(ifo.d_tilde, 0, RTOL, ATOL)

    def test_noise_generation_with_rng(
        self,
        grid_default: interferometer.Grid,
        amplitude_spectral_density_file_default: Path,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that noise realisation is generated with random number generator."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_default)

        assert not numpy.allclose(ifo.d_tilde, 0, RTOL, ATOL)

    def test_noise_generation_reproducible(
        self, grid_default: interferometer.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that noise realisation is reproducible with same seed."""
        rng_1 = numpy.random.default_rng(42)
        ifo_1 = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_1)
        rng_2 = numpy.random.default_rng(42)
        ifo_2 = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_2)

        assert numpy.array_equal(ifo_1.d_tilde, ifo_2.d_tilde)

    def test_regenerate_noise(
        self,
        grid_default: interferometer.Grid,
        amplitude_spectral_density_file_default: Path,
        rng_default: numpy.random.Generator,
    ) -> None:
        """Test that noise can be regenerated."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, is_zero_noise=True)
        ifo.set_data(rng=rng_default)

        assert not numpy.allclose(ifo.d_tilde, 0, RTOL, ATOL)

    @pytest.mark.parametrize(
        "theta, phi, psi, f_plus_expected, f_cross_expected",
        [
            (0, 0, 0, 1, 0),
            (numpy.pi / 2, 0, 0, 1 / 2, 0),
            (0, numpy.pi / 2, 0, -1, 0),
            (0, 0, numpy.pi / 4, 0, 1),
            (numpy.pi / 2, 0, numpy.pi / 4, 0, 1 / 2),
            (0, numpy.pi / 2, numpy.pi / 4, 0, -1),
        ],
    )
    def test_pattern_functions(
        self,
        theta: float,
        phi: float,
        psi: float,
        f_plus_expected: float,
        f_cross_expected: float,
    ) -> None:
        """Test pattern function calculations for various angles."""
        f_plus, f_cross = interferometer.Interferometer.calculate_pattern_functions(theta, phi, psi)

        assert numpy.isclose(f_plus_expected, f_plus)
        assert numpy.isclose(f_cross_expected, f_cross)

    def test_pattern_functions_polarisation(self) -> None:
        """Test that pattern functions rotate correctly with polarisation angle."""
        theta, phi = numpy.pi / 4, numpy.pi / 6
        f_plus_0, f_cross_0 = interferometer.Interferometer.calculate_pattern_functions(theta, phi, 0)
        f_plus_45, f_cross_45 = interferometer.Interferometer.calculate_pattern_functions(theta, phi, numpy.pi / 4)

        assert numpy.isclose(f_plus_0, f_cross_45)
        assert numpy.isclose(f_cross_0, -f_plus_45)

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
        grid_default: interferometer.Grid,
        amplitude_spectral_density_file_default: Path,
        model_default: waveform.WaveformModel,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test signal injection into interferometer."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, (1, numpy.inf))
        h_tilde, rho, rho_mf = ifo.inject_signal(model_default, theta_default)

        assert not numpy.allclose(h_tilde, 0, RTOL, ATOL)
        assert rho > 0
        assert rho_mf > 0

    def test_inject_signal_updates_data(
        self,
        grid_default: interferometer.Grid,
        amplitude_spectral_density_file_default: Path,
        model_default: waveform.WaveformModel,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test that signal injection updates data stream."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, (1, numpy.inf))
        d_tilde_before = ifo.d_tilde.copy()
        h_tilde, rho, rho_mf = ifo.inject_signal(model_default, theta_default)

        assert not numpy.array_equal(ifo.d_tilde, d_tilde_before)
        assert numpy.array_equal(ifo.d_tilde, d_tilde_before + h_tilde)

    def test_inject_signal_matched_filter_signal_to_noise_ratio(
        self,
        grid_default: interferometer.Grid,
        amplitude_spectral_density_file_default: Path,
        model_default: waveform.WaveformModel,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test matched filter signal-to-noise ratio equals optimal signal-to-noise ratio for zero noise."""
        ifo = interferometer.Interferometer(
            grid_default, amplitude_spectral_density_file_default, (1, numpy.inf), is_zero_noise=True
        )
        h_tilde, rho, rho_mf = ifo.inject_signal(model_default, theta_default)

        assert numpy.isclose(rho, abs(rho_mf))


class TestLIGO:
    """Tests for the LIGO class."""

    def test_initialisation(self, grid_default: interferometer.Grid) -> None:
        """Test that LIGO can be initialised."""
        ligo = interferometer.LIGO(grid_default)

        assert ligo.grid == grid_default
        assert 20 <= ligo.f.min()
        assert ligo.f.max() <= 2048

    def test_frequency_band(self, grid_default: interferometer.Grid) -> None:
        """Test that LIGO has correct frequency band."""
        ligo = interferometer.LIGO(grid_default)

        assert numpy.all(20 <= ligo.f)
        assert numpy.all(ligo.f <= 2048)

    def test_noise_generation(self, grid_default: interferometer.Grid, rng_default: numpy.random.Generator) -> None:
        """Test that LIGO generates noise correctly."""
        ligo = interferometer.LIGO(grid_default, rng=rng_default)

        assert not numpy.allclose(ligo.d_tilde, 0, RTOL, ATOL)

    def test_inject_signal(
        self,
        grid_default: interferometer.Grid,
        model_default: waveform.WaveformModel,
        theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test signal injection into LIGO."""
        ligo = interferometer.LIGO(grid_default)
        h_tilde, rho, rho_mf = ligo.inject_signal(model_default, theta_default)

        assert not numpy.allclose(h_tilde, 0, RTOL, ATOL)
        assert rho > 0
        assert rho_mf > 0


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
