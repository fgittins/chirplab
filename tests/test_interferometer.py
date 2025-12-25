"""Unit tests for the interferometer module."""

from dataclasses import replace
from pathlib import Path

import numpy
import pytest

from chirplab import interferometer, waveform

RTOL = 0
ATOL = 1e-27


class TestGrid:
    """Tests for the `Grid` dataclass."""

    def test_initialisation(self, grid_default: interferometer.Grid) -> None:
        """Test that `Grid` can be initialised with valid parameters."""
        assert grid_default.T == 4
        assert grid_default.f_s == 4096

    def test_invalid_product_T_fs(self) -> None:
        """Test that `Grid` raises ValueError when T * f_s is not an integer."""
        with pytest.raises(ValueError, match="The product of T and f_s must be an integer"):
            interferometer.Grid(T=4.1, f_s=4096)

    def test_invalid_odd_N(self) -> None:
        """Test that `Grid` raises ValueError when N is odd."""
        with pytest.raises(ValueError, match="The number of time samples N must be even"):
            interferometer.Grid(T=1, f_s=4095)

    def test_f_max_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the `f_max` property returns the Nyquist frequency."""
        assert grid_default.f_max == 2048

    def test_N_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the `N` property returns the correct number of time samples."""
        assert grid_default.N == 16384

    def test_M_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the `M` property returns the correct number of frequency samples."""
        assert grid_default.M == 8192

    def test_Delta_t_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the `Delta_t` property returns the correct time resolution."""
        assert grid_default.Delta_t == 1 / 4096

    def test_Delta_f_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the `Delta_f` property returns the correct frequency resolution."""
        assert grid_default.Delta_f == 1 / 4

    def test_t_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the `t` property returns the correct time array."""
        assert grid_default.t.size == grid_default.N + 1
        assert grid_default.t[0] == 0
        assert grid_default.t[-1] == grid_default.T
        assert numpy.all(numpy.diff(grid_default.t) == grid_default.Delta_t)

    def test_f_property(self, grid_default: interferometer.Grid) -> None:
        """Test that the `f` property returns the correct frequency array."""
        assert grid_default.f.size == grid_default.M + 1
        assert grid_default.f[0] == 0
        assert grid_default.f[-1] == grid_default.f_max
        assert numpy.all(numpy.diff(grid_default.f) == grid_default.Delta_f)


class TestInterferometer:
    """Tests for the `Interferometer` class."""

    @pytest.fixture
    def amplitude_spectral_density_file_default(self) -> Path:
        """Return path to default amplitude spectral density file for testing."""
        return Path(__file__).parent.parent / "src/chirplab/data/aligo_O4high.txt"

    @pytest.fixture
    def interferometer_default(
        self, grid_default: interferometer.Grid, amplitude_spectral_density_file_default: Path
    ) -> interferometer.Interferometer:
        """Return default interferometer for testing."""
        return interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default)

    def test_initialisation(
        self, grid_default: interferometer.Grid, interferometer_default: interferometer.Interferometer
    ) -> None:
        """Test that `Interferometer` can be initialised."""
        assert interferometer_default.grid == grid_default
        assert numpy.array_equal(
            interferometer_default.h_tilde, numpy.zeros_like(interferometer_default.f, numpy.complex128)
        )
        assert interferometer_default.rho == 0

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

        assert numpy.allclose(ifo.n_tilde, 0, RTOL, ATOL)
        assert numpy.allclose(ifo.d_tilde, 0, RTOL, ATOL)

    def test_noise_generation_with_rng(
        self, grid_default: interferometer.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that noise realisation is generated with random number generator."""
        rng = numpy.random.default_rng(42)
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng)

        assert not numpy.allclose(ifo.n_tilde, 0, RTOL, ATOL)
        assert numpy.array_equal(ifo.n_tilde, ifo.d_tilde)

    def test_noise_generation_reproducible(
        self, grid_default: interferometer.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that noise realisation is reproducible with same seed."""
        rng_1 = numpy.random.default_rng(42)
        ifo_1 = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_1)
        rng_2 = numpy.random.default_rng(42)
        ifo_2 = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, rng=rng_2)

        assert numpy.array_equal(ifo_1.n_tilde, ifo_2.n_tilde)

    def test_regenerate_noise(
        self, grid_default: interferometer.Grid, amplitude_spectral_density_file_default: Path
    ) -> None:
        """Test that noise can be regenerated."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, is_zero_noise=True)
        rng = numpy.random.default_rng(42)
        ifo.generate_noise(rng=rng)

        assert not numpy.allclose(ifo.n_tilde, 0, RTOL, ATOL)

    @pytest.mark.parametrize(
        "theta, phi, psi, F_plus, F_cross",
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
        F_plus: float,
        F_cross: float,
    ) -> None:
        """Test pattern function calculations for various angles."""
        F_plus_calculated, F_cross_calculated = interferometer.Interferometer.calculate_pattern_functions(
            theta, phi, psi
        )

        assert numpy.isclose(F_plus, F_plus_calculated)
        assert numpy.isclose(F_cross, F_cross_calculated)

    def test_pattern_functions_polarisation(self) -> None:
        """Test that pattern functions rotate correctly with polarisation angle."""
        theta, phi = numpy.pi / 4, numpy.pi / 6
        F_plus_0, F_cross_0 = interferometer.Interferometer.calculate_pattern_functions(theta, phi, 0)
        F_plus_45, F_cross_45 = interferometer.Interferometer.calculate_pattern_functions(theta, phi, numpy.pi / 4)

        assert numpy.isclose(F_plus_0, F_cross_45)
        assert numpy.isclose(F_cross_0, -F_plus_45)

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
        Theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test signal injection into interferometer."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, (1, numpy.inf))
        ifo.inject(model_default, Theta_default)

        assert ifo.h_tilde is not None
        assert ifo.rho is not None
        assert ifo.rho_MF is not None
        assert not numpy.allclose(ifo.h_tilde, 0, RTOL, ATOL)

    def test_inject_signal_updates_data(
        self,
        grid_default: interferometer.Grid,
        amplitude_spectral_density_file_default: Path,
        model_default: waveform.WaveformModel,
        Theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test that signal injection updates data stream."""
        ifo = interferometer.Interferometer(grid_default, amplitude_spectral_density_file_default, (1, numpy.inf))
        d_tilde_before = ifo.d_tilde.copy()
        ifo.inject(model_default, Theta_default)

        assert not numpy.array_equal(ifo.d_tilde, d_tilde_before)
        assert ifo.h_tilde is not None
        assert numpy.array_equal(ifo.d_tilde, d_tilde_before + ifo.h_tilde)

    def test_inject_signal_matched_filter_snr(
        self,
        grid_default: interferometer.Grid,
        amplitude_spectral_density_file_default: Path,
        model_default: waveform.WaveformModel,
        Theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test matched filter signal-to-noise ratio equals optimal signal-to-noise ratio for zero noise."""
        ifo = interferometer.Interferometer(
            grid_default, amplitude_spectral_density_file_default, (1, numpy.inf), is_zero_noise=True
        )
        ifo.inject(model_default, Theta_default)

        assert ifo.rho is not None
        assert ifo.rho_MF is not None
        assert numpy.isclose(abs(ifo.rho_MF), ifo.rho)


class TestLIGO:
    """Tests for the `LIGO` class."""

    def test_initialisation(self, grid_default: interferometer.Grid) -> None:
        """Test that `LIGO` can be initialised."""
        ligo = interferometer.LIGO(grid_default)

        assert ligo.grid == grid_default
        assert 20 <= ligo.f.min()
        assert ligo.f.max() <= 2048

    def test_frequency_band(self, grid_default: interferometer.Grid) -> None:
        """Test that LIGO has correct frequency band."""
        ligo = interferometer.LIGO(grid_default)

        assert numpy.all(20 <= ligo.f)
        assert numpy.all(ligo.f <= 2048)

    def test_noise_generation(self, grid_default: interferometer.Grid) -> None:
        """Test that LIGO generates noise correctly."""
        rng = numpy.random.default_rng(42)
        ligo = interferometer.LIGO(grid_default, rng=rng)

        assert not numpy.allclose(ligo.n_tilde, 0, RTOL, ATOL)

    def test_inject_signal(
        self,
        grid_default: interferometer.Grid,
        model_default: waveform.WaveformModel,
        Theta_default: interferometer.SignalParameters,
    ) -> None:
        """Test signal injection into LIGO."""
        ligo = interferometer.LIGO(grid_default)
        ligo.inject(model_default, Theta_default)

        assert ligo.h_tilde is not None
        assert ligo.rho is not None
        assert ligo.rho > 0


class TestCalculateInnerProduct:
    """Tests for the `calculate_inner_product` function."""

    def test_inner_product_shape(self) -> None:
        """Test that inner product returns a scalar."""
        N = 100
        a_tilde = numpy.ones(N, dtype=numpy.complex128)
        b_tilde = numpy.ones(N, dtype=numpy.complex128)
        S_n = numpy.ones(N, dtype=numpy.float64)
        Delta_f = 1 / 4
        a_inner_b = interferometer.calculate_inner_product(a_tilde, b_tilde, S_n, Delta_f)

        assert a_inner_b.dtype == numpy.complex128

    def test_inner_product_self_is_real(self) -> None:
        """Test that inner product of signal with itself is real."""
        N = 100
        a_tilde = numpy.random.randn(N) + 1j * numpy.random.randn(N)
        S_n = numpy.ones(N, dtype=numpy.float64)
        Delta_f = 1 / 4
        a_inner_a = interferometer.calculate_inner_product(a_tilde, a_tilde, S_n, Delta_f)

        assert numpy.isclose(a_inner_a.imag, 0)

    def test_inner_product_conjugate_symmetry(self) -> None:
        """Test that inner product satisfies conjugate symmetry."""
        N = 100
        a_tilde = numpy.random.randn(N) + 1j * numpy.random.randn(N)
        b_tilde = numpy.random.randn(N) + 1j * numpy.random.randn(N)
        S_n = numpy.ones(N, dtype=numpy.float64)
        Delta_f = 1 / 4
        a_inner_b = interferometer.calculate_inner_product(a_tilde, b_tilde, S_n, Delta_f)
        b_inner_a = interferometer.calculate_inner_product(b_tilde, a_tilde, S_n, Delta_f)

        assert numpy.isclose(a_inner_b, b_inner_a.conj())

    def test_inner_product_scales_with_delta_f(self) -> None:
        """Test that inner product scales linearly with frequency resolution."""
        N = 100
        a_tilde = numpy.ones(N, dtype=numpy.complex128)
        b_tilde = numpy.ones(N, dtype=numpy.complex128)
        S_n = numpy.ones(N, dtype=numpy.float64)
        Delta_f_1, Delta_f_2 = 1 / 4, 1 / 2
        a_inner_b_1 = interferometer.calculate_inner_product(a_tilde, b_tilde, S_n, Delta_f_1)
        a_inner_b_2 = interferometer.calculate_inner_product(a_tilde, b_tilde, S_n, Delta_f_2)

        assert numpy.isclose(a_inner_b_2 / a_inner_b_1, Delta_f_2 / Delta_f_1)

    def test_inner_product_size_mismatch_raises(self) -> None:
        """Test that inner product raises assertion error for mismatched sizes."""
        N = 100
        a_tilde = numpy.ones(N, dtype=numpy.complex128)
        b_tilde = numpy.ones(N // 2, dtype=numpy.complex128)
        S_n = numpy.ones(N, dtype=numpy.float64)
        Delta_f = 1 / 4

        with pytest.raises(AssertionError):
            interferometer.calculate_inner_product(a_tilde, b_tilde, S_n, Delta_f)


class TestGenerateGaussianNoise:
    """Tests for the `generate_gaussian_noise` function."""

    def test_noise_shape(self) -> None:
        """Test that generated noise has correct shape."""
        S_n = numpy.ones(100, dtype=numpy.float64)
        T = 4
        rng = numpy.random.default_rng(42)
        n_tilde = interferometer.generate_gaussian_noise(S_n, T, rng)

        assert n_tilde.shape == S_n.shape

    def test_noise_dtype(self) -> None:
        """Test that generated noise has correct dtype."""
        S_n = numpy.ones(100, dtype=numpy.float64)
        T = 4
        rng = numpy.random.default_rng(42)
        n_tilde = interferometer.generate_gaussian_noise(S_n, T, rng)

        assert n_tilde.dtype == numpy.complex128

    def test_noise_endpoints_zero(self) -> None:
        """Test that noise is zero at direct-current and Nyquist frequencies."""
        S_n = numpy.ones(100, dtype=numpy.float64)
        T = 4
        rng = numpy.random.default_rng(42)
        n_tilde = interferometer.generate_gaussian_noise(S_n, T, rng)

        assert n_tilde[0] == 0
        assert n_tilde[-1] == 0

    def test_noise_reproducible(self) -> None:
        """Test that noise is reproducible with same seed."""
        S_n = numpy.ones(100, dtype=numpy.float64)
        T = 4
        rng_1 = numpy.random.default_rng(42)
        n_tilde_1 = interferometer.generate_gaussian_noise(S_n, T, rng_1)
        rng_2 = numpy.random.default_rng(42)
        n_tilde_2 = interferometer.generate_gaussian_noise(S_n, T, rng_2)

        assert numpy.array_equal(n_tilde_1, n_tilde_2)

    def test_noise_different_seeds(self) -> None:
        """Test that noise is different with different seeds."""
        S_n = numpy.ones(100, dtype=numpy.float64)
        T = 4
        rng_1 = numpy.random.default_rng(42)
        n_tilde_1 = interferometer.generate_gaussian_noise(S_n, T, rng_1)
        rng_2 = numpy.random.default_rng(43)
        n_tilde_2 = interferometer.generate_gaussian_noise(S_n, T, rng_2)

        assert not numpy.array_equal(n_tilde_1, n_tilde_2)

    def test_noise_scales_with_psd(self) -> None:
        """Test that noise amplitude scales with power spectral density."""
        ratio = 4
        T = 4
        rng_1 = numpy.random.default_rng(42)
        S_n_1 = numpy.ones(100, dtype=numpy.float64)
        n_tilde_1 = interferometer.generate_gaussian_noise(S_n_1, T, rng_1)
        rng_2 = numpy.random.default_rng(42)
        S_n_2 = ratio * S_n_1
        n_tilde_2 = interferometer.generate_gaussian_noise(S_n_2, T, rng_2)
        ratio_calculated = numpy.abs(n_tilde_2[1:-2]) / numpy.abs(n_tilde_1[1:-2])

        assert numpy.allclose(ratio_calculated, ratio ** (1 / 2))

    def test_noise_scales_with_duration(self) -> None:
        """Test that noise amplitude scales with duration."""
        S_n = numpy.ones(100, dtype=numpy.float64)
        T_1, T_2 = 4, 16
        rng_1 = numpy.random.default_rng(42)
        n_tilde_1 = interferometer.generate_gaussian_noise(S_n, T_1, rng_1)
        rng_2 = numpy.random.default_rng(42)
        n_tilde_2 = interferometer.generate_gaussian_noise(S_n, T_2, rng_2)
        ratio_calculated = numpy.abs(n_tilde_2[1:-2]) / numpy.abs(n_tilde_1[1:-2])

        assert numpy.allclose(ratio_calculated, (T_2 / T_1) ** (1 / 2))
