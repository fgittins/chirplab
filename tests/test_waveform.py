"""Unit tests for the waveform module."""

from dataclasses import replace

import numpy
import pytest

from chirplab import waveform

RTOL = 0
ATOL = 1e-28


@pytest.fixture
def Theta_default() -> waveform.SignalParameters:
    """Return default set of signal parameters for testing."""
    return waveform.SignalParameters(
        m_1=30 * waveform.M_sun,
        m_2=30 * waveform.M_sun,
        r=500e6 * waveform.pc,
        iota=numpy.pi / 3,
        t_c=100,
        Phi_c=1.5,
        theta=0,
        phi=numpy.pi / 4,
        psi=0.5,
    )


@pytest.fixture
def f_default() -> numpy.typing.NDArray[numpy.float64]:
    """Return default frequency array for testing."""
    return numpy.linspace(20, 70, 20, dtype=numpy.float64)


class TestSignalParameters:
    """Tests for the `SignalParameters` dataclass."""

    def test_initialisation(self, Theta_default: waveform.SignalParameters) -> None:
        """Test that `SignalParameters` can be initialised with all required fields."""
        assert Theta_default.m_1 == 30 * waveform.M_sun
        assert Theta_default.m_2 == 30 * waveform.M_sun
        assert Theta_default.r == 500e6 * waveform.pc
        assert Theta_default.iota == numpy.pi / 3
        assert Theta_default.t_c == 100
        assert Theta_default.Phi_c == 1.5
        assert Theta_default.theta == 0
        assert Theta_default.phi == numpy.pi / 4
        assert Theta_default.psi == 0.5

    def test_total_mass_property(self, Theta_default: waveform.SignalParameters) -> None:
        """Test that the `M` property correctly calculates total mass."""
        assert Theta_default.M == 60 * waveform.M_sun

    @pytest.mark.parametrize(
        "m_1, m_2",
        [
            (30 * waveform.M_sun, 30 * waveform.M_sun),
            (30 * waveform.M_sun, 15 * waveform.M_sun),
        ],
    )
    def test_chirp_mass_property(self, m_1: float, m_2: float, Theta_default: waveform.SignalParameters) -> None:
        """Test that the `M_chirp` property correctly calculates chirp mass."""
        Theta = replace(Theta_default, m_1=m_1, m_2=m_2)
        M_chirp = (Theta.m_1 * Theta.m_2) ** (3 / 5) / (Theta.m_1 + Theta.m_2) ** (1 / 5)

        assert Theta.M_chirp == M_chirp


class TestWaveform:
    """Tests for the `Waveform` base class."""

    def test_base_class_not_implemented(
        self, f_default: numpy.typing.NDArray[numpy.float64], Theta_default: waveform.SignalParameters
    ) -> None:
        """Test that the base `Waveform` class raises `NotImplementedError`."""
        wform = waveform.Waveform()

        with pytest.raises(NotImplementedError):
            wform.calculate_strain_polarisations(f_default, Theta_default)


class TestNewtonianWaveform:
    """Tests for the `NewtonianWaveform` class."""

    wform = waveform.NewtonianWaveform()

    def test_output_shape(
        self, f_default: numpy.typing.NDArray[numpy.float64], Theta_default: waveform.SignalParameters
    ) -> None:
        """Test that output arrays have the correct shape."""
        h_tilde_plus, h_tilde_cross = self.wform.calculate_strain_polarisations(f_default, Theta_default)

        assert h_tilde_plus.shape == f_default.shape
        assert h_tilde_cross.shape == f_default.shape

    def test_output_type(
        self, f_default: numpy.typing.NDArray[numpy.float64], Theta_default: waveform.SignalParameters
    ) -> None:
        """Test that output arrays have the correct type."""
        h_tilde_plus, h_tilde_cross = self.wform.calculate_strain_polarisations(f_default, Theta_default)

        assert h_tilde_plus.dtype == numpy.complex128
        assert h_tilde_cross.dtype == numpy.complex128

    def test_zero_above_ISCO(self, Theta_default: waveform.SignalParameters) -> None:
        """Test that the waveform is zero above the innermost stable circular orbit frequency."""
        f_ISCO = waveform.calculate_innermost_stable_circular_orbit_frequency(Theta_default.M)
        f = numpy.linspace(f_ISCO + 1, f_ISCO + 100, 20)
        h_tilde_plus, h_tilde_cross = self.wform.calculate_strain_polarisations(f, Theta_default)

        assert numpy.allclose(h_tilde_plus, 0, RTOL, ATOL)
        assert numpy.allclose(h_tilde_cross, 0, RTOL, ATOL)

    def test_non_zero_up_to_ISCO(self, Theta_default: waveform.SignalParameters) -> None:
        """Test that the waveform is non-zero up to the innermost stable circular orbit frequency."""
        f_ISCO = waveform.calculate_innermost_stable_circular_orbit_frequency(Theta_default.M)
        f = numpy.linspace(20, f_ISCO, 20)
        h_tilde_plus, h_tilde_cross = self.wform.calculate_strain_polarisations(f, Theta_default)

        assert not numpy.allclose(h_tilde_plus, 0, RTOL, ATOL)
        assert not numpy.allclose(h_tilde_cross, 0, RTOL, ATOL)

    def test_edge_on_inclination(
        self, f_default: numpy.typing.NDArray[numpy.float64], Theta_default: waveform.SignalParameters
    ) -> None:
        """Test waveform when the orbit is edge on."""
        Theta = replace(Theta_default, iota=numpy.pi / 2)
        h_tilde_plus, h_tilde_cross = self.wform.calculate_strain_polarisations(f_default, Theta)

        assert not numpy.allclose(h_tilde_plus, 0, RTOL, ATOL)
        assert numpy.allclose(h_tilde_cross, 0, RTOL, ATOL)

    def test_face_on_inclination(
        self, f_default: numpy.typing.NDArray[numpy.float64], Theta_default: waveform.SignalParameters
    ) -> None:
        """Test waveform when the orbit is face on."""
        Theta = replace(Theta_default, iota=0)
        h_tilde_plus, h_tilde_cross = self.wform.calculate_strain_polarisations(f_default, Theta)

        assert numpy.allclose(abs(h_tilde_plus), abs(h_tilde_cross), RTOL, ATOL)

    def test_amplitude_decreases_with_frequency(
        self, f_default: numpy.typing.NDArray[numpy.float64], Theta_default: waveform.SignalParameters
    ) -> None:
        """Test that amplitude decreases with increasing frequency."""
        h_tilde_plus, _ = self.wform.calculate_strain_polarisations(f_default, Theta_default)
        A = abs(h_tilde_plus)

        assert A[0] > A[-1]

    def test_amplitude_scales_with_distance(
        self, f_default: numpy.typing.NDArray[numpy.float64], Theta_default: waveform.SignalParameters
    ) -> None:
        """Test that amplitude scales inversely with distance."""
        ratio = 2
        r = 500e6 * waveform.pc
        Theta_near = replace(Theta_default, r=r)
        Theta_far = replace(Theta_default, r=ratio * r)
        h_tilde_plus_near, _ = self.wform.calculate_strain_polarisations(f_default, Theta_near)
        h_tilde_plus_far, _ = self.wform.calculate_strain_polarisations(f_default, Theta_far)
        ratio_calculated = abs(h_tilde_plus_near) / abs(h_tilde_plus_far)

        assert numpy.all(ratio == ratio_calculated)

    def test_single_frequency(self, Theta_default: waveform.SignalParameters) -> None:
        """Test with a single frequency value."""
        f = numpy.array([40], dtype=numpy.float64)
        h_tilde_plus, h_tilde_cross = self.wform.calculate_strain_polarisations(f, Theta_default)

        assert h_tilde_plus.shape == (1,)
        assert h_tilde_cross.shape == (1,)
        assert h_tilde_plus.dtype == numpy.complex128
        assert h_tilde_cross.dtype == numpy.complex128


class TestCalculateInnermostStableCircularOrbitFrequency:
    """Tests for the `calculate_innermost_stable_circular_orbit_frequency` function."""

    def test_returns_positive_frequency(self) -> None:
        """Test that the function returns a positive frequency."""
        M = 15 * waveform.M_sun
        f_ISCO = waveform.calculate_innermost_stable_circular_orbit_frequency(M)

        assert f_ISCO > 0

    def test_frequency_scales_inversely_with_mass(self) -> None:
        """Test that ISCO frequency scales inversely with mass."""
        ratio = 2
        M_small = 10 * waveform.M_sun
        M_large = 2 * M_small
        f_ISCO_small = waveform.calculate_innermost_stable_circular_orbit_frequency(M_small)
        f_ISCO_large = waveform.calculate_innermost_stable_circular_orbit_frequency(M_large)
        ratio_calculated = f_ISCO_small / f_ISCO_large

        assert ratio == ratio_calculated
