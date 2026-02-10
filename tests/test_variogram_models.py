"""
Comprehensive tests for variogram models and model registry.

Tests cover:
- Individual model functions (spherical, exponential, gaussian, matern, etc.)
- Analytical values and mathematical properties
- Boundary behavior and edge cases
- Model registry and validation
- Parameter specifications and bounds
"""

import pytest
import numpy as np
from scipy.special import gamma as gamma_func, kv as bessel_kv

# Import directly from the module to avoid loading the full package
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topochange.variogram_models import (
    spherical,
    exponential,
    gaussian,
    matern,
    damped_hole_effect,
    power,
    linear,
    nugget,
    VariogramModelRegistry,
    VariogramModelSpec,
    MODEL_REGISTRY,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def registry():
    """Provide a fresh VariogramModelRegistry instance."""
    return VariogramModelRegistry()


@pytest.fixture
def lags():
    """Sample lag distances for testing."""
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])


@pytest.fixture
def sample_variogram():
    """Sample variogram values for parameter estimation."""
    lags = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    return lags, np.array([0.0, 0.2, 0.4, 0.55, 0.65, 0.8, 0.9])


# =============================================================================
# Test Spherical Model
# =============================================================================

class TestSphericalModel:
    """Tests for the spherical variogram model."""

    def test_spherical_at_origin(self):
        """Spherical model should equal 0 at h=0."""
        result = spherical(np.array([0.0]), sill=1.0, range_=1.0)
        np.testing.assert_allclose(result, [0.0], rtol=1e-10)

    def test_spherical_at_range(self):
        """Spherical model should equal sill exactly at h=range."""
        sill = 2.5
        range_ = 3.0
        result = spherical(np.array([range_]), sill=sill, range_=range_)
        np.testing.assert_allclose(result, [sill], rtol=1e-10)

    def test_spherical_beyond_range(self):
        """Spherical model should equal sill for h > range."""
        sill = 2.5
        range_ = 1.0
        h_vals = np.array([2.0, 5.0, 10.0, 100.0])
        result = spherical(h_vals, sill=sill, range_=range_)
        np.testing.assert_allclose(result, np.full_like(h_vals, sill), rtol=1e-10)

    def test_spherical_at_half_range(self):
        """Test spherical at h = range/2.

        γ(range/2) = C * [1.5*(0.5) - 0.5*(0.5)³]
                   = C * [0.75 - 0.0625]
                   = C * 0.6875
        """
        sill = 2.0
        range_ = 2.0
        h = range_ / 2
        expected = sill * (1.5 * 0.5 - 0.5 * 0.5**3)
        result = spherical(np.array([h]), sill=sill, range_=range_)
        np.testing.assert_allclose(result, [expected], rtol=1e-10)

    def test_spherical_monotonicity(self):
        """Spherical should be monotonically increasing with h."""
        sill = 1.0
        range_ = 2.0
        h_vals = np.linspace(0, 5, 50)
        result = spherical(h_vals, sill=sill, range_=range_)
        # Check that differences are non-negative (monotone increasing)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10), "Spherical model should be monotone increasing"

    def test_spherical_array_input(self):
        """Spherical should handle array inputs correctly."""
        sill = 1.5
        range_ = 2.0
        h_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        result = spherical(h_vals, sill=sill, range_=range_)
        assert result.shape == h_vals.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_spherical_list_input(self):
        """Spherical should handle list inputs."""
        sill = 1.0
        range_ = 1.0
        h_vals = [0.0, 0.5, 1.0, 2.0]
        result = spherical(h_vals, sill=sill, range_=range_)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4

    def test_spherical_zero_sill(self):
        """Spherical with zero sill should return all zeros."""
        h_vals = np.array([0.0, 1.0, 2.0, 3.0])
        result = spherical(h_vals, sill=0.0, range_=1.0)
        np.testing.assert_allclose(result, np.zeros_like(h_vals), atol=1e-14)

    def test_spherical_very_large_h(self):
        """Spherical should handle very large h values."""
        sill = 1.0
        range_ = 1.0
        h_vals = np.array([1e6, 1e10])
        result = spherical(h_vals, sill=sill, range_=range_)
        np.testing.assert_allclose(result, np.full_like(h_vals, sill), rtol=1e-10)


# =============================================================================
# Test Exponential Model
# =============================================================================

class TestExponentialModel:
    """Tests for the exponential variogram model."""

    def test_exponential_at_origin(self):
        """Exponential should equal 0 at h=0."""
        result = exponential(np.array([0.0]), sill=1.0, range_=1.0)
        np.testing.assert_allclose(result, [0.0], rtol=1e-10)

    def test_exponential_at_range(self):
        """Test exponential at h=range.

        γ(range) = C * [1 - exp(-1)] ≈ C * 0.6321
        """
        sill = 2.0
        range_ = 1.5
        h = range_
        expected = sill * (1 - np.exp(-1))
        result = exponential(np.array([h]), sill=sill, range_=range_)
        np.testing.assert_allclose(result, [expected], rtol=1e-10)

    def test_exponential_asymptotic(self):
        """Exponential should approach sill at large h."""
        sill = 1.0
        range_ = 1.0
        h_large = np.array([50.0, 100.0, 1000.0])
        result = exponential(h_large, sill=sill, range_=range_)
        # Should be very close to sill
        np.testing.assert_allclose(result, np.full_like(h_large, sill),
                                   rtol=1e-5)

    def test_exponential_monotonicity(self):
        """Exponential should be monotonically increasing."""
        sill = 1.0
        range_ = 1.0
        h_vals = np.linspace(0, 20, 100)
        result = exponential(h_vals, sill=sill, range_=range_)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10), "Exponential should be monotone increasing"

    def test_exponential_array_input(self):
        """Exponential should handle array inputs."""
        sill = 2.5
        range_ = 1.5
        h_vals = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        result = exponential(h_vals, sill=sill, range_=range_)
        assert result.shape == h_vals.shape
        assert np.all(np.isfinite(result))

    def test_exponential_bounds(self):
        """Exponential should be bounded by [0, sill]."""
        sill = 3.0
        range_ = 2.0
        h_vals = np.linspace(0, 100, 100)
        result = exponential(h_vals, sill=sill, range_=range_)
        assert np.all(result >= 0)
        assert np.all(result <= sill + 1e-10)


# =============================================================================
# Test Gaussian Model
# =============================================================================

class TestGaussianModel:
    """Tests for the Gaussian variogram model."""

    def test_gaussian_at_origin(self):
        """Gaussian should equal 0 at h=0."""
        result = gaussian(np.array([0.0]), sill=1.0, range_=1.0)
        np.testing.assert_allclose(result, [0.0], rtol=1e-10)

    def test_gaussian_at_range(self):
        """Test Gaussian at h=range.

        γ(range) = C * [1 - exp(-1)] ≈ C * 0.6321
        """
        sill = 2.0
        range_ = 1.5
        h = range_
        expected = sill * (1 - np.exp(-1))
        result = gaussian(np.array([h]), sill=sill, range_=range_)
        np.testing.assert_allclose(result, [expected], rtol=1e-10)

    def test_gaussian_asymptotic(self):
        """Gaussian should approach sill at large h."""
        sill = 1.0
        range_ = 1.0
        h_large = np.array([50.0, 100.0])
        result = gaussian(h_large, sill=sill, range_=range_)
        np.testing.assert_allclose(result, np.full_like(h_large, sill),
                                   rtol=1e-5)

    def test_gaussian_monotonicity(self):
        """Gaussian should be monotonically increasing."""
        sill = 1.0
        range_ = 1.0
        h_vals = np.linspace(0, 20, 100)
        result = gaussian(h_vals, sill=sill, range_=range_)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10), "Gaussian should be monotone increasing"

    def test_gaussian_parabolic_near_origin(self):
        """Gaussian should be parabolic near origin: γ(h) ≈ (C/a²)h²."""
        sill = 1.0
        range_ = 1.0
        h_small = np.array([0.01, 0.02])
        result = gaussian(h_small, sill=sill, range_=range_)
        expected = (sill / range_**2) * h_small**2
        # Expect good agreement near origin
        np.testing.assert_allclose(result, expected, rtol=0.01)

    def test_gaussian_array_input(self):
        """Gaussian should handle array inputs."""
        sill = 2.0
        range_ = 2.0
        h_vals = np.array([0.0, 1.0, 2.0, 5.0])
        result = gaussian(h_vals, sill=sill, range_=range_)
        assert result.shape == h_vals.shape
        assert np.all(np.isfinite(result))


# =============================================================================
# Test Matérn Model
# =============================================================================

class TestMaternModel:
    """Tests for the Matérn variogram model."""

    def test_matern_at_origin(self):
        """Matérn should equal 0 at h=0."""
        result = matern(np.array([0.0]), sill=1.0, range_=1.0, nu=1.5)
        np.testing.assert_allclose(result, [0.0], rtol=1e-10)

    def test_matern_nu_0_5_equals_exponential(self):
        """Matérn with nu=0.5 should match exponential model.

        The exponential variogram is a special case of Matérn with nu=0.5.
        """
        sill = 1.5
        range_ = 2.0
        h_vals = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])

        matern_result = matern(h_vals, sill=sill, range_=range_, nu=0.5)
        exponential_result = exponential(h_vals, sill=sill, range_=range_)

        # Should be very close (allowing for numerical precision)
        np.testing.assert_allclose(matern_result, exponential_result, rtol=0.02)

    def test_matern_monotonicity(self):
        """Matérn should be monotonically increasing."""
        sill = 1.0
        range_ = 1.5
        nu = 1.5
        h_vals = np.linspace(0, 20, 100)
        result = matern(h_vals, sill=sill, range_=range_, nu=nu)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10), "Matérn should be monotone increasing"

    def test_matern_bounded_by_sill(self):
        """Matérn should be bounded by sill."""
        sill = 2.0
        range_ = 1.5
        nu = 2.5
        h_vals = np.linspace(0, 100, 100)
        result = matern(h_vals, sill=sill, range_=range_, nu=nu)
        assert np.all(result >= -1e-10)
        assert np.all(result <= sill + 1e-10)

    def test_matern_various_nu(self):
        """Test Matérn with various nu values."""
        sill = 1.0
        range_ = 1.0
        h = np.array([0.5, 1.0, 2.0])

        for nu in [0.5, 1.0, 1.5, 2.5, 5.0]:
            result = matern(h, sill=sill, range_=range_, nu=nu)
            assert result.shape == h.shape
            assert np.all(np.isfinite(result))
            assert np.all(result >= 0)

    def test_matern_array_input(self):
        """Matérn should handle array inputs."""
        sill = 1.5
        range_ = 1.2
        nu = 1.5
        h_vals = np.array([0.0, 0.5, 1.0, 2.0])
        result = matern(h_vals, sill=sill, range_=range_, nu=nu)
        assert result.shape == h_vals.shape
        assert np.all(np.isfinite(result))


# =============================================================================
# Test Damped Hole-Effect Model
# =============================================================================

class TestDampedHoleEffectModel:
    """Tests for the damped hole-effect variogram model."""

    def test_damped_hole_at_origin(self):
        """Damped hole-effect should equal 0 at h=0."""
        result = damped_hole_effect(np.array([0.0]), sill=1.0, range_=1.0,
                                     wavelength=2.0)
        np.testing.assert_allclose(result, [0.0], rtol=1e-10)

    def test_damped_hole_oscillatory_behavior(self):
        """Damped hole-effect should show oscillatory behavior."""
        sill = 1.0
        range_ = 1.0
        wavelength = 2.0
        h_vals = np.linspace(0, 10, 100)
        result = damped_hole_effect(h_vals, sill=sill, range_=range_,
                                     wavelength=wavelength)

        # Check that we have both positive and potentially negative values
        # (though bounded by [0, sill*(1-exp(-h/r))*2])
        assert np.all(np.isfinite(result))
        assert np.min(result) >= -1e-10  # Slight numerical tolerance

    def test_damped_hole_bounded_asymptotically(self):
        """Damped hole-effect should approach sill at large h."""
        sill = 1.0
        range_ = 1.0
        wavelength = 2.0
        h_large = np.array([100.0])
        result = damped_hole_effect(h_large, sill=sill, range_=range_,
                                     wavelength=wavelength)
        # At large h, exp(-h/r) ≈ 0, so γ ≈ C*(1 - 0) = C
        np.testing.assert_allclose(result, [sill], rtol=0.01)

    def test_damped_hole_array_input(self):
        """Damped hole-effect should handle array inputs."""
        sill = 1.5
        range_ = 1.0
        wavelength = 2.0
        h_vals = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        result = damped_hole_effect(h_vals, sill=sill, range_=range_,
                                     wavelength=wavelength)
        assert result.shape == h_vals.shape
        assert np.all(np.isfinite(result))

    def test_damped_hole_various_parameters(self):
        """Test with various parameter combinations."""
        sill_vals = [0.5, 1.0, 2.0]
        range_vals = [0.5, 1.0, 2.0]
        wavelength_vals = [1.0, 2.0, 3.0]
        h = np.array([0.0, 1.0, 2.0])

        for sill in sill_vals:
            for range_ in range_vals:
                for wavelength in wavelength_vals:
                    result = damped_hole_effect(h, sill=sill, range_=range_,
                                                 wavelength=wavelength)
                    assert result.shape == h.shape
                    assert np.all(np.isfinite(result))


# =============================================================================
# Test Power Model
# =============================================================================

class TestPowerModel:
    """Tests for the power (unbounded) variogram model."""

    def test_power_at_h_one(self):
        """Power at h=1 should equal scale.

        γ(1) = α * 1^ω = α
        """
        scale = 2.5
        exponent = 1.5
        result = power(np.array([1.0]), scale=scale, exponent=exponent)
        np.testing.assert_allclose(result, [scale], rtol=1e-10)

    def test_power_scaling(self):
        """Test power law scaling: γ(2h) = 2^ω * γ(h)."""
        scale = 1.0
        exponent = 1.5
        h = 2.0

        gamma_h = power(np.array([h]), scale=scale, exponent=exponent)[0]
        gamma_2h = power(np.array([2*h]), scale=scale, exponent=exponent)[0]

        expected_ratio = 2.0**exponent
        actual_ratio = gamma_2h / gamma_h

        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-10)

    def test_power_monotonicity(self):
        """Power should be monotonically increasing (for 0 < ω < 2)."""
        scale = 1.0
        exponent = 1.5
        h_vals = np.linspace(0, 10, 100)
        result = power(h_vals, scale=scale, exponent=exponent)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10), "Power should be monotone increasing"

    def test_power_unbounded(self):
        """Power model should grow without bound."""
        scale = 1.0
        exponent = 1.5
        h_vals = np.array([1.0, 10.0, 100.0, 1000.0])
        result = power(h_vals, scale=scale, exponent=exponent)

        # Check that variance increases with distance
        assert result[1] > result[0]
        assert result[2] > result[1]
        assert result[3] > result[2]

    def test_power_array_input(self):
        """Power should handle array inputs."""
        scale = 0.5
        exponent = 1.2
        h_vals = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        result = power(h_vals, scale=scale, exponent=exponent)
        assert result.shape == h_vals.shape
        assert np.all(np.isfinite(result))

    def test_power_at_origin(self):
        """Power should equal 0 at h=0."""
        result = power(np.array([0.0]), scale=1.0, exponent=1.5)
        np.testing.assert_allclose(result, [0.0], rtol=1e-10)


# =============================================================================
# Test Linear Model
# =============================================================================

class TestLinearModel:
    """Tests for the linear (unbounded) variogram model."""

    def test_linear_proportional_to_h(self):
        """Linear should be proportional to h: γ(h) = β*h."""
        slope = 2.5
        h_vals = np.array([0.0, 1.0, 2.0, 3.0])
        result = linear(h_vals, slope=slope)
        expected = slope * h_vals
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_linear_at_origin(self):
        """Linear should equal 0 at h=0."""
        result = linear(np.array([0.0]), slope=2.0)
        np.testing.assert_allclose(result, [0.0], rtol=1e-10)

    def test_linear_unbounded(self):
        """Linear should grow without bound."""
        slope = 0.5
        h_vals = np.array([10.0, 100.0, 1000.0])
        result = linear(h_vals, slope=slope)
        assert result[1] > result[0]
        assert result[2] > result[1]

    def test_linear_array_input(self):
        """Linear should handle array inputs."""
        slope = 1.5
        h_vals = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        result = linear(h_vals, slope=slope)
        assert result.shape == h_vals.shape
        np.testing.assert_allclose(result, slope * h_vals)

    def test_linear_zero_slope(self):
        """Linear with zero slope should return all zeros."""
        h_vals = np.array([0.0, 1.0, 2.0, 3.0])
        result = linear(h_vals, slope=0.0)
        np.testing.assert_allclose(result, np.zeros_like(h_vals), atol=1e-14)


# =============================================================================
# Test Nugget Model
# =============================================================================

class TestNuggetModel:
    """Tests for the pure nugget effect model."""

    def test_nugget_at_origin(self):
        """Nugget should equal 0 at h=0."""
        result = nugget(np.array([0.0]), c0=1.0)
        np.testing.assert_allclose(result, [0.0], rtol=1e-10)

    def test_nugget_away_from_origin(self):
        """Nugget should equal c0 for h > 0."""
        c0 = 2.5
        h_vals = np.array([0.1, 0.5, 1.0, 2.0, 100.0])
        result = nugget(h_vals, c0=c0)
        np.testing.assert_allclose(result, np.full_like(h_vals, c0),
                                   rtol=1e-10)

    def test_nugget_discontinuity(self):
        """Nugget should have a discontinuity at h=0."""
        c0 = 1.5
        h_vals = np.array([0.0, 0.001])
        result = nugget(h_vals, c0=c0)
        assert result[0] == 0.0
        assert np.isclose(result[1], c0)

    def test_nugget_array_input(self):
        """Nugget should handle array inputs."""
        c0 = 0.5
        h_vals = np.array([0.0, 0.1, 0.5, 1.0, 5.0])
        result = nugget(h_vals, c0=c0)
        assert result.shape == h_vals.shape
        expected = np.array([0.0, c0, c0, c0, c0])
        np.testing.assert_allclose(result, expected)

    def test_nugget_zero_c0(self):
        """Nugget with c0=0 should return all zeros."""
        h_vals = np.array([0.0, 0.5, 1.0])
        result = nugget(h_vals, c0=0.0)
        np.testing.assert_allclose(result, np.zeros_like(h_vals), atol=1e-14)


# =============================================================================
# Test Model Registry
# =============================================================================

class TestVariogramModelRegistry:
    """Tests for the VariogramModelRegistry class."""

    def test_registry_get_model(self, registry):
        """Test retrieving models from registry."""
        spec = registry.get_model('spherical')
        assert isinstance(spec, VariogramModelSpec)
        assert spec.name == 'spherical'
        assert spec.func == spherical

    def test_registry_get_unknown_model_raises(self, registry):
        """Test that getting unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            registry.get_model('unknown_model')

    def test_registry_list_models(self, registry):
        """Test listing all available models."""
        models = registry.list_models()
        assert isinstance(models, list)
        assert 'spherical' in models
        assert 'exponential' in models
        assert 'gaussian' in models
        assert 'matern' in models
        assert 'damped_hole_effect' in models
        assert 'power' in models
        assert 'linear' in models

    def test_registry_list_bounded_models(self, registry):
        """Test listing bounded (stationary) models."""
        bounded = registry.list_bounded_models()
        assert isinstance(bounded, list)
        assert 'spherical' in bounded
        assert 'exponential' in bounded
        assert 'gaussian' in bounded
        assert 'matern' in bounded
        assert 'damped_hole_effect' in bounded
        # Unbounded models should not be included
        assert 'power' not in bounded
        assert 'linear' not in bounded

    def test_registry_list_unbounded_models(self, registry):
        """Test listing unbounded (non-stationary) models."""
        unbounded = registry.list_unbounded_models()
        assert isinstance(unbounded, list)
        assert 'power' in unbounded
        assert 'linear' in unbounded
        # Bounded models should not be included
        assert 'spherical' not in unbounded
        assert 'exponential' not in unbounded

    def test_registry_is_bounded(self, registry):
        """Test is_bounded method."""
        assert registry.is_bounded('spherical')
        assert registry.is_bounded('exponential')
        assert not registry.is_bounded('power')
        assert not registry.is_bounded('linear')

    def test_registry_validate_combination_single_bounded(self, registry):
        """Test validation of single bounded model."""
        valid, msg = registry.validate_combination(['spherical'])
        assert valid is True

    def test_registry_validate_combination_multiple_bounded(self, registry):
        """Test validation of multiple bounded models."""
        valid, msg = registry.validate_combination(['spherical', 'exponential'])
        assert valid is True
        assert "Valid" in msg or "valid" in msg.lower()

    def test_registry_validate_combination_single_unbounded(self, registry):
        """Test validation of single unbounded model."""
        valid, msg = registry.validate_combination(['power'])
        assert valid is True
        assert "NON-STATIONARY" in msg or "non-stationary" in msg

    def test_registry_validate_combination_empty(self, registry):
        """Test validation of empty combination."""
        valid, msg = registry.validate_combination([])
        assert valid is False
        assert "At least one" in msg or "required" in msg.lower()

    def test_registry_validate_combination_unknown_model(self, registry):
        """Test validation with unknown model."""
        valid, msg = registry.validate_combination(['unknown'])
        assert valid is False
        assert "Unknown" in msg

    def test_registry_validate_combination_multiple_unbounded(self, registry):
        """Test that multiple unbounded models are invalid."""
        valid, msg = registry.validate_combination(['power', 'linear'])
        assert valid is False
        assert "Cannot combine multiple unbounded" in msg or "unbounded" in msg.lower()

    def test_registry_validate_combination_gaussian_without_nugget(self, registry):
        """Test warning for Gaussian without nugget."""
        valid, msg = registry.validate_combination(['gaussian'], include_nugget=False)
        assert valid is True
        assert "Gaussian" in msg and "nugget" in msg.lower()

    def test_registry_validate_combination_gaussian_with_nugget(self, registry):
        """Test no warning for Gaussian with nugget."""
        valid, msg = registry.validate_combination(['gaussian'], include_nugget=True)
        assert valid is True
        # Should not have warning about Gaussian
        assert "without nugget" not in msg.lower()

    def test_registry_bounded_model_has_sill(self, registry):
        """Test that bounded models have sill."""
        for name in registry.list_bounded_models():
            spec = registry.get_model(name)
            assert spec.has_sill is True

    def test_registry_unbounded_model_no_sill(self, registry):
        """Test that unbounded models don't have sill."""
        for name in registry.list_unbounded_models():
            spec = registry.get_model(name)
            assert spec.has_sill is False


# =============================================================================
# Test VariogramModelSpec
# =============================================================================

class TestVariogramModelSpec:
    """Tests for the VariogramModelSpec dataclass."""

    def test_spec_default_guess(self, registry, sample_variogram):
        """Test default_guess method."""
        spec = registry.get_model('spherical')
        lags, variogram = sample_variogram
        guess = spec.default_guess(lags, variogram)

        assert isinstance(guess, list)
        assert len(guess) == 2  # sill and range
        assert all(isinstance(p, (int, float)) for p in guess)
        assert guess[0] > 0  # sill should be positive
        assert guess[1] > 0  # range should be positive

    def test_spec_bounds(self, registry, sample_variogram):
        """Test bounds method."""
        spec = registry.get_model('spherical')
        lags, variogram = sample_variogram
        lower, upper = spec.bounds(lags, variogram)

        assert isinstance(lower, list)
        assert isinstance(upper, list)
        assert len(lower) == 2
        assert len(upper) == 2
        assert all(l < u for l, u in zip(lower, upper))

    def test_spec_default_guess_matern(self, registry, sample_variogram):
        """Test default_guess for Matérn (3 parameters)."""
        spec = registry.get_model('matern')
        lags, variogram = sample_variogram
        guess = spec.default_guess(lags, variogram)

        assert len(guess) == 3  # sill, range, nu
        assert guess[2] > 0  # nu should be positive

    def test_spec_bounds_matern(self, registry, sample_variogram):
        """Test bounds for Matérn."""
        spec = registry.get_model('matern')
        lags, variogram = sample_variogram
        lower, upper = spec.bounds(lags, variogram)

        assert len(lower) == 3
        assert len(upper) == 3

    def test_spec_validate_damped_hole_valid(self, registry):
        """Test validation for valid damped hole effect parameters."""
        spec = registry.get_model('damped_hole_effect')
        # Valid: 2πr/λ = 2π*1/2 ≈ 3.14 > 1
        params = [1.0, 1.0, 2.0]  # sill=1, range=1, wavelength=2
        # Should not raise
        spec.validate(params)

    def test_spec_validate_damped_hole_invalid(self, registry):
        """Test validation for invalid damped hole effect parameters."""
        spec = registry.get_model('damped_hole_effect')
        # Invalid: 2πr/λ = 2π*0.1/2 ≈ 0.31 < 1
        params = [1.0, 0.1, 2.0]  # sill=1, range=0.1, wavelength=2
        with pytest.raises(ValueError, match="positive definiteness"):
            spec.validate(params)

    def test_spec_description(self, registry):
        """Test that specs have descriptions."""
        for name in registry.list_models():
            spec = registry.get_model(name)
            assert spec.description
            assert isinstance(spec.description, str)
            assert len(spec.description) > 0

    def test_spec_param_names(self, registry):
        """Test that param_names match func signature."""
        spec = registry.get_model('spherical')
        assert spec.param_names == ['sill', 'range']

        spec = registry.get_model('matern')
        assert spec.param_names == ['sill', 'range', 'nu']

        spec = registry.get_model('power')
        assert spec.param_names == ['scale', 'exponent']

        spec = registry.get_model('linear')
        assert spec.param_names == ['slope']

        # Note: nugget is a function but not registered as a model in registry
        # It's used as a separate component for compositing


# =============================================================================
# Test Global Registry
# =============================================================================

class TestGlobalRegistry:
    """Tests for the global MODEL_REGISTRY instance."""

    def test_global_registry_exists(self):
        """Test that global registry is available."""
        assert MODEL_REGISTRY is not None
        assert isinstance(MODEL_REGISTRY, VariogramModelRegistry)

    def test_global_registry_has_models(self):
        """Test that global registry has all models."""
        models = MODEL_REGISTRY.list_models()
        assert len(models) > 0
        assert 'spherical' in models
        assert 'exponential' in models

    def test_global_registry_get_model(self):
        """Test getting model from global registry."""
        spec = MODEL_REGISTRY.get_model('spherical')
        assert isinstance(spec, VariogramModelSpec)


# =============================================================================
# Test Edge Cases and Integration
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_all_models_handle_scalar(self):
        """All models should handle scalar input (converted to numpy scalar)."""
        models_and_params = [
            ('spherical', {'sill': 1.0, 'range_': 1.0}),
            ('exponential', {'sill': 1.0, 'range_': 1.0}),
            ('gaussian', {'sill': 1.0, 'range_': 1.0}),
            ('matern', {'sill': 1.0, 'range_': 1.0, 'nu': 1.5}),
            ('damped_hole_effect', {'sill': 1.0, 'range_': 1.0, 'wavelength': 2.0}),
            ('power', {'scale': 1.0, 'exponent': 1.5}),
            ('linear', {'slope': 1.0}),
        ]

        for name, kwargs in models_and_params:
            spec = MODEL_REGISTRY.get_model(name)
            result = spec.func(0.5, **kwargs)
            # Result should be a numpy scalar or array
            assert isinstance(result, (np.ndarray, np.generic))

    def test_all_models_handle_empty_array(self):
        """All models should handle empty arrays gracefully."""
        models_and_params = [
            ('spherical', {'sill': 1.0, 'range_': 1.0}),
            ('exponential', {'sill': 1.0, 'range_': 1.0}),
            ('gaussian', {'sill': 1.0, 'range_': 1.0}),
            ('matern', {'sill': 1.0, 'range_': 1.0, 'nu': 1.5}),
            ('damped_hole_effect', {'sill': 1.0, 'range_': 1.0, 'wavelength': 2.0}),
            ('power', {'scale': 1.0, 'exponent': 1.5}),
            ('linear', {'slope': 1.0}),
        ]

        for name, kwargs in models_and_params:
            spec = MODEL_REGISTRY.get_model(name)
            result = spec.func(np.array([]), **kwargs)
            assert isinstance(result, np.ndarray)
            assert result.shape == (0,)

    def test_all_models_return_nonnegative(self):
        """All models should return non-negative semivariances."""
        h_vals = np.linspace(0, 10, 20)

        models_and_params = [
            ('spherical', {'sill': 1.0, 'range_': 1.0}),
            ('exponential', {'sill': 1.0, 'range_': 1.0}),
            ('gaussian', {'sill': 1.0, 'range_': 1.0}),
            ('matern', {'sill': 1.0, 'range_': 1.0, 'nu': 1.5}),
            ('damped_hole_effect', {'sill': 1.0, 'range_': 1.0, 'wavelength': 2.0}),
            ('power', {'scale': 1.0, 'exponent': 1.5}),
            ('linear', {'slope': 1.0}),
        ]

        for name, kwargs in models_and_params:
            spec = MODEL_REGISTRY.get_model(name)
            result = spec.func(h_vals, **kwargs)
            assert np.all(result >= -1e-10), f"{name} returned negative values"

    def test_very_small_range(self):
        """Test models with very small range parameter."""
        h_vals = np.array([0.0, 0.1, 1.0])
        small_range = 1e-6

        result = spherical(h_vals, sill=1.0, range_=small_range)
        # With very small range, everything should be at sill except h=0
        assert result[0] == 0.0
        assert np.isclose(result[1], 1.0)
        assert np.isclose(result[2], 1.0)

    def test_very_large_sill(self):
        """Test models with very large sill."""
        h = np.array([1.0])
        large_sill = 1e10

        result = exponential(h, sill=large_sill, range_=1.0)
        assert np.isfinite(result[0])
        assert result[0] > 0

    def test_model_combination_bounded_with_nugget_flag(self, registry):
        """Test validation with include_nugget flag."""
        # When include_nugget=True, Gaussian should not produce warning
        valid, msg = registry.validate_combination(
            ['spherical'],
            include_nugget=True
        )
        assert valid is True

    def test_nugget_function_exists(self):
        """Test that nugget function is available for use."""
        h_vals = np.array([0.0, 0.5, 1.0])
        result = nugget(h_vals, c0=1.0)
        assert isinstance(result, np.ndarray)
        # Nugget should be 0 at h=0 and c0 elsewhere
        assert result[0] == 0.0
        assert np.isclose(result[1], 1.0)
        assert np.isclose(result[2], 1.0)


# =============================================================================
# Test Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of models."""

    def test_exponential_large_h_stability(self):
        """Exponential should be numerically stable at large h."""
        sill = 1.0
        range_ = 1.0
        h_large = np.array([1000.0, 1e6])
        result = exponential(h_large, sill=sill, range_=range_)
        assert np.all(np.isfinite(result))
        assert np.all(result <= sill + 1e-10)

    def test_gaussian_large_h_stability(self):
        """Gaussian should be numerically stable at large h."""
        sill = 1.0
        range_ = 1.0
        h_large = np.array([1000.0, 1e6])
        result = gaussian(h_large, sill=sill, range_=range_)
        assert np.all(np.isfinite(result))
        assert np.all(result <= sill + 1e-10)

    def test_matern_small_h_stability(self):
        """Matérn should be stable near h=0."""
        sill = 1.0
        range_ = 1.0
        nu = 1.5
        h_small = np.array([0.0, 1e-10, 1e-6])
        result = matern(h_small, sill=sill, range_=range_, nu=nu)
        assert np.all(np.isfinite(result))

    def test_power_large_exponent(self):
        """Power should handle large exponent values."""
        h = np.array([0.0, 0.5, 1.0, 2.0])
        result = power(h, scale=1.0, exponent=1.99)
        assert np.all(np.isfinite(result))

    def test_all_models_no_nan_at_regular_points(self):
        """All models should not return NaN at regular test points."""
        h_vals = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])

        models_and_params = [
            ('spherical', {'sill': 1.0, 'range_': 1.0}),
            ('exponential', {'sill': 1.0, 'range_': 1.0}),
            ('gaussian', {'sill': 1.0, 'range_': 1.0}),
            ('matern', {'sill': 1.0, 'range_': 1.0, 'nu': 1.5}),
            ('damped_hole_effect', {'sill': 1.0, 'range_': 1.0, 'wavelength': 2.0}),
            ('power', {'scale': 1.0, 'exponent': 1.5}),
            ('linear', {'slope': 1.0}),
        ]

        for name, kwargs in models_and_params:
            spec = MODEL_REGISTRY.get_model(name)
            result = spec.func(h_vals, **kwargs)
            assert not np.any(np.isnan(result)), f"{name} returned NaN"
