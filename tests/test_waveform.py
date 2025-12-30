"""Unit tests for the waveform module."""

from dataclasses import replace

import numpy
import pytest

from chirplab import constants, waveform

RTOL = 0
ATOL = 1e-24


@pytest.fixture(scope="module")
def theta_default() -> waveform.WaveformParameters:
    """Return default set of signal parameters for testing."""
    return waveform.WaveformParameters(
        m_1=30 * constants.M_SUN,
        m_2=30 * constants.M_SUN,
        r=500e6 * constants.PC,
        iota=constants.PI / 3,
        t_c=100,
        phi_c=1.5,
    )


@pytest.fixture(scope="module")
def f_default() -> numpy.typing.NDArray[numpy.float64]:
    """Return default frequency array for testing."""
    return numpy.linspace(20, 70, 20, dtype=numpy.float64)


class TestWaveformParameters:
    """Tests for the WaveformParameters dataclass."""

    def test_initialisation(self) -> None:
        """Test that WaveformParameters can be initialised with all required fields."""
        theta = waveform.WaveformParameters(
            m_1=30 * constants.M_SUN,
            m_2=30 * constants.M_SUN,
            r=500e6 * constants.PC,
            iota=constants.PI / 3,
            t_c=100,
            phi_c=1.5,
        )

        assert theta.m_1 == 30 * constants.M_SUN
        assert theta.m_2 == 30 * constants.M_SUN
        assert theta.r == 500e6 * constants.PC
        assert theta.iota == constants.PI / 3
        assert theta.t_c == 100
        assert theta.phi_c == 1.5

    def test_total_mass_property(self, theta_default: waveform.WaveformParameters) -> None:
        """Test that the m_total property correctly calculates total mass."""
        assert theta_default.m_total == 60 * constants.M_SUN

    @pytest.mark.parametrize(
        "m_1, m_2",
        [
            (30 * constants.M_SUN, 30 * constants.M_SUN),
            (30 * constants.M_SUN, 15 * constants.M_SUN),
        ],
    )
    def test_chirp_mass_property(self, m_1: float, m_2: float, theta_default: waveform.WaveformParameters) -> None:
        """Test that the m_chirp property correctly calculates chirp mass."""
        theta = replace(theta_default, m_1=m_1, m_2=m_2)

        assert theta.m_chirp == (theta.m_1 * theta.m_2) ** (3 / 5) / (theta.m_1 + theta.m_2) ** (1 / 5)


class TestWaveformModel:
    """Tests for the WaveformModel base class."""

    def test_initialisation(self) -> None:
        """Test that WaveformModel can be initialised with default and custom f_max."""
        model_default = waveform.WaveformModel()

        assert model_default.f_max == constants.INF

        custom_f_max = 2048
        model_custom = waveform.WaveformModel(custom_f_max)

        assert model_custom.f_max == custom_f_max

    def test_base_class_not_implemented(
        self, f_default: numpy.typing.NDArray[numpy.float64], theta_default: waveform.WaveformParameters
    ) -> None:
        """Test that the base WaveformModel class raises NotImplementedError."""
        model = waveform.WaveformModel()

        with pytest.raises(NotImplementedError):
            model.calculate_strain_polarisations(f_default, theta_default)


class TestNewtonianWaveformModel:
    """Tests for the NewtonianWaveformModel class."""

    def test_initialisation(self) -> None:
        """Test that NewtonianWaveformModel can be initialised."""
        model_default = waveform.NewtonianWaveformModel()

        assert model_default.f_max == constants.INF
        assert model_default.is_isco_cutoff is True

        f_max = 2048
        model_custom = waveform.NewtonianWaveformModel(f_max, False)

        assert model_custom.f_max == f_max
        assert model_custom.is_isco_cutoff is False

    def test_output_shape(
        self,
        model_default: waveform.NewtonianWaveformModel,
        f_default: numpy.typing.NDArray[numpy.float64],
        theta_default: waveform.WaveformParameters,
    ) -> None:
        """Test that output arrays have the correct shape."""
        h_tilde_plus, h_tilde_cross = model_default.calculate_strain_polarisations(f_default, theta_default)

        assert h_tilde_plus.shape == f_default.shape
        assert h_tilde_cross.shape == f_default.shape

    def test_output_type(
        self,
        model_default: waveform.NewtonianWaveformModel,
        f_default: numpy.typing.NDArray[numpy.float64],
        theta_default: waveform.WaveformParameters,
    ) -> None:
        """Test that output arrays have the correct type."""
        h_tilde_plus, h_tilde_cross = model_default.calculate_strain_polarisations(f_default, theta_default)

        assert h_tilde_plus.dtype == numpy.complex128
        assert h_tilde_cross.dtype == numpy.complex128

    def test_zero_above_isco(
        self, model_default: waveform.NewtonianWaveformModel, theta_default: waveform.WaveformParameters
    ) -> None:
        """Test that the waveform is zero above the innermost stable circular orbit frequency."""
        f_isco = waveform.calculate_isco_frequency(theta_default.m_total)
        f = numpy.linspace(f_isco + 1, f_isco + 100, 20)
        h_tilde_plus, h_tilde_cross = model_default.calculate_strain_polarisations(f, theta_default)

        assert numpy.allclose(h_tilde_plus, 0, RTOL, ATOL)
        assert numpy.allclose(h_tilde_cross, 0, RTOL, ATOL)

    def test_non_zero_up_to_isco(
        self, model_default: waveform.NewtonianWaveformModel, theta_default: waveform.WaveformParameters
    ) -> None:
        """Test that the waveform is non-zero up to the innermost stable circular orbit frequency."""
        f_isco = waveform.calculate_isco_frequency(theta_default.m_total)
        f = numpy.linspace(20, f_isco, 20)
        h_tilde_plus, h_tilde_cross = model_default.calculate_strain_polarisations(f, theta_default)

        assert not numpy.allclose(h_tilde_plus, 0, RTOL, ATOL)
        assert not numpy.allclose(h_tilde_cross, 0, RTOL, ATOL)

    def test_non_zero_up_to_f_max(self, theta_default: waveform.WaveformParameters) -> None:
        """Test that the waveform is non-zero up to f_max when is_isco_cutoff is False."""
        f_max = 1000
        model = waveform.NewtonianWaveformModel(f_max, False)
        f = numpy.linspace(20, f_max, 20)
        h_tilde_plus, h_tilde_cross = model.calculate_strain_polarisations(f, theta_default)

        assert not numpy.allclose(h_tilde_plus, 0, RTOL, ATOL)
        assert not numpy.allclose(h_tilde_cross, 0, RTOL, ATOL)

    def test_edge_on_inclination(
        self,
        model_default: waveform.NewtonianWaveformModel,
        f_default: numpy.typing.NDArray[numpy.float64],
        theta_default: waveform.WaveformParameters,
    ) -> None:
        """Test waveform when the orbit is edge on."""
        theta = replace(theta_default, iota=constants.PI / 2)
        h_tilde_plus, h_tilde_cross = model_default.calculate_strain_polarisations(f_default, theta)

        assert not numpy.allclose(h_tilde_plus, 0, RTOL, ATOL)
        assert numpy.allclose(h_tilde_cross, 0, RTOL, ATOL)

    def test_face_on_inclination(
        self,
        model_default: waveform.NewtonianWaveformModel,
        f_default: numpy.typing.NDArray[numpy.float64],
        theta_default: waveform.WaveformParameters,
    ) -> None:
        """Test waveform when the orbit is face on."""
        theta = replace(theta_default, iota=0)
        h_tilde_plus, h_tilde_cross = model_default.calculate_strain_polarisations(f_default, theta)

        assert numpy.array_equal(abs(h_tilde_plus), abs(h_tilde_cross))

    def test_amplitude_decreases_with_frequency(
        self,
        model_default: waveform.NewtonianWaveformModel,
        f_default: numpy.typing.NDArray[numpy.float64],
        theta_default: waveform.WaveformParameters,
    ) -> None:
        """Test that amplitude decreases with increasing frequency."""
        h_tilde_plus, _ = model_default.calculate_strain_polarisations(f_default, theta_default)
        a = abs(h_tilde_plus)
        delta_a = numpy.diff(a)

        assert numpy.all(delta_a < 0)

    def test_amplitude_scales_with_distance(
        self,
        model_default: waveform.NewtonianWaveformModel,
        f_default: numpy.typing.NDArray[numpy.float64],
        theta_default: waveform.WaveformParameters,
    ) -> None:
        """Test that amplitude scales inversely with distance."""
        ratio = 2
        theta_near = theta_default
        theta_far = replace(theta_default, r=ratio * theta_near.r)
        h_tilde_plus_near, _ = model_default.calculate_strain_polarisations(f_default, theta_near)
        h_tilde_plus_far, _ = model_default.calculate_strain_polarisations(f_default, theta_far)

        assert numpy.all(ratio == abs(h_tilde_plus_near) / abs(h_tilde_plus_far))

    def test_single_frequency(
        self, model_default: waveform.NewtonianWaveformModel, theta_default: waveform.WaveformParameters
    ) -> None:
        """Test with a single frequency value."""
        f = numpy.array([40], dtype=numpy.float64)
        h_tilde_plus, h_tilde_cross = model_default.calculate_strain_polarisations(f, theta_default)

        assert h_tilde_plus.shape == (1,)
        assert h_tilde_cross.shape == (1,)
        assert h_tilde_plus.dtype == numpy.complex128
        assert h_tilde_cross.dtype == numpy.complex128


class TestCalculateInnermostStableCircularOrbitFrequency:
    """Tests for the calculate_isco_frequency function."""

    def test_returns_positive_frequency(self) -> None:
        """Test that the function returns a positive frequency."""
        assert waveform.calculate_isco_frequency(15 * constants.M_SUN) > 0

    def test_frequency_scales_inversely_with_mass(self) -> None:
        """Test that ISCO frequency scales inversely with mass."""
        ratio = 2
        m_small = 10 * constants.M_SUN
        m_large = 2 * m_small
        f_isco_small = waveform.calculate_isco_frequency(m_small)
        f_isco_large = waveform.calculate_isco_frequency(m_large)

        assert ratio == f_isco_small / f_isco_large
