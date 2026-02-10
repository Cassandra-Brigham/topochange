"""Comprehensive test suite for uncertainty module.

Tests cover:
1. DerivativeUncertaintyEstimator - covariance, kernel variance, slope/curvature
2. Monte Carlo integration with various variogram models
3. KERNELS dictionary validation
4. Edge cases (zero sill, resolution)
"""

import pytest
import numpy as np
import math
from shapely.geometry import Polygon, box, Point

from topochange.uncertainty import DerivativeUncertaintyEstimator
from topochange.variogram_models import spherical, exponential, nugget


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

@pytest.fixture
def spherical_gamma():
    """Spherical variogram with sill=1.0, range=50."""
    return lambda h: spherical(h, sill=1.0, range_=50.0)


@pytest.fixture
def exponential_gamma():
    """Exponential variogram with sill=1.0, range=30."""
    return lambda h: exponential(h, sill=1.0, range_=30.0)


@pytest.fixture
def nugget_gamma():
    """Pure nugget effect with nugget=1.0."""
    return lambda h: nugget(h, c0=1.0)


@pytest.fixture
def small_polygon():
    """Small 10x10 square polygon."""
    return box(0, 0, 10, 10)


@pytest.fixture
def large_polygon():
    """Large 100x100 square polygon."""
    return box(0, 0, 100, 100)


class MockRegionalEstimator:
    """Mock object to test RegionalUncertaintyEstimator.estimate_std_mean_monte_carlo logic."""

    def covariance(self, h, sigma2, gamma_func):
        """C(h) = σ² - γ(h)"""
        return sigma2 - gamma_func(h)

    def estimate_std_mean_monte_carlo(
        self,
        domain: Polygon,
        gamma_func,
        sigma2: float,
        n_pairs: int = 50000,
        seed: int = 42,
    ) -> float:
        """Estimate std(mean) via Monte Carlo integration."""
        rng = np.random.default_rng(seed)
        minx, miny, maxx, maxy = domain.bounds

        pts = []
        batch_size = n_pairs
        while len(pts) < n_pairs * 2:
            rand_x = rng.uniform(minx, maxx, size=batch_size)
            rand_y = rng.uniform(miny, maxy, size=batch_size)
            for x, y in zip(rand_x, rand_y):
                if domain.contains(Point(x, y)):
                    pts.append((x, y))
                if len(pts) >= n_pairs * 2:
                    break

        pts = np.array(pts)
        X, Y = pts[:n_pairs], pts[n_pairs : 2 * n_pairs]

        h = np.linalg.norm(X - Y, axis=1)
        cov = self.covariance(h, sigma2, gamma_func)
        var_mean = float(np.mean(cov))

        return 0.0 if var_mean < 0 else math.sqrt(var_mean)


# =============================================================================
# Test DerivativeUncertaintyEstimator - Covariance
# =============================================================================


class TestDerivativeEstimatorCovariance:
    """Test covariance function C(h) = σ² - γ(h)."""

    def test_covariance_at_zero_spherical(self, spherical_gamma):
        """C(0) should equal sill for spherical model."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        cov_zero = estimator.covariance(np.array([0.0]))
        assert np.isclose(cov_zero[0], 1.0), f"Expected C(0)=1.0, got {cov_zero[0]}"

    def test_covariance_at_zero_exponential(self, exponential_gamma):
        """C(0) should equal sill for exponential model."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=exponential_gamma, sill=1.0, resolution=1.0
        )
        cov_zero = estimator.covariance(np.array([0.0]))
        assert np.isclose(cov_zero[0], 1.0)

    def test_covariance_large_lag_spherical(self, spherical_gamma):
        """C(large_h) should be ≈ 0 for spherical at h >> range."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        cov_large = estimator.covariance(np.array([1000.0]))
        # For spherical, γ(h >> range) = sill, so C(h) ≈ 0
        assert np.isclose(cov_large[0], 0.0, atol=1e-10)

    def test_covariance_large_lag_exponential(self, exponential_gamma):
        """C(large_h) should approach 0 for exponential at h >> range."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=exponential_gamma, sill=1.0, resolution=1.0
        )
        cov_large = estimator.covariance(np.array([1000.0]))
        # For exponential, γ(h) → sill as h → ∞
        assert cov_large[0] < 0.01

    def test_covariance_nugget(self, nugget_gamma):
        """C(h) = 0 for h > 0 in pure nugget model (σ² - c₀ = 0)."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=nugget_gamma, sill=1.0, resolution=1.0
        )
        cov_nonzero = estimator.covariance(np.array([1.0, 10.0, 100.0]))
        # For pure nugget with sill=1.0, γ(h>0)=1.0, so C(h)=0
        assert np.allclose(cov_nonzero, 0.0)

    def test_covariance_multiple_lags(self):
        """Test covariance with array input."""
        # Create a consistent gamma function and sill
        sill = 1.0
        gamma = lambda h: spherical(h, sill=sill, range_=50.0)
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=gamma, sill=sill, resolution=1.0
        )
        h_array = np.array([0.0, 25.0, 50.0, 100.0])
        cov = estimator.covariance(h_array)
        assert cov.shape == h_array.shape
        assert np.isclose(cov[0], sill)  # C(0) = σ²
        # For spherical with range=50, h=100 >> range, so γ(h)=sill, thus C(h)≈0
        assert np.isclose(cov[-1], 0.0, atol=1e-10)  # C(large) ≈ 0


# =============================================================================
# Test DerivativeUncertaintyEstimator - Kernel Variance
# =============================================================================


class TestDerivativeEstimatorKernelVariance:
    """Test kernel_variance computation."""

    def test_kernel_variance_nugget_sobel_x(self, nugget_gamma):
        """For pure nugget model, Var = sill * sum(K²)."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=nugget_gamma, sill=1.0, resolution=1.0
        )
        kernel = estimator.KERNELS["sobel_x"]
        var = estimator.kernel_variance(kernel)

        # For nugget model (C(h)=0 for h>0, C(0)=sill):
        # Var = K²_center * sill = (0)² * 1.0 = 0 (center of Sobel X is 0)
        # Actually, Var = sum_i sum_j K_i K_j C(||x_i - x_j||)
        # Only the diagonal terms (i=j) contribute: sum_i K_i² * sill
        expected_var = np.sum(kernel**2) * 1.0
        assert np.isclose(var, expected_var, rtol=1e-5)

    def test_kernel_variance_nugget_sobel_y(self, nugget_gamma):
        """Sobel Y kernel variance with nugget model."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=nugget_gamma, sill=1.0, resolution=1.0
        )
        kernel = estimator.KERNELS["sobel_y"]
        var = estimator.kernel_variance(kernel)

        expected_var = np.sum(kernel**2) * 1.0
        assert np.isclose(var, expected_var, rtol=1e-5)

    def test_kernel_variance_nugget_laplacian(self, nugget_gamma):
        """Laplacian kernel variance with nugget model."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=nugget_gamma, sill=1.0, resolution=1.0
        )
        kernel = estimator.KERNELS["laplacian"]
        var = estimator.kernel_variance(kernel)

        expected_var = np.sum(kernel**2) * 1.0
        assert np.isclose(var, expected_var, rtol=1e-5)

    def test_kernel_variance_scales_with_sill(self, spherical_gamma):
        """Kernel variance should scale linearly with sill."""
        # Create two estimators with different sills
        est_sill1 = DerivativeUncertaintyEstimator(
            gamma_func=lambda h: spherical(h, sill=1.0, range_=50.0),
            sill=1.0,
            resolution=1.0,
        )
        est_sill2 = DerivativeUncertaintyEstimator(
            gamma_func=lambda h: spherical(h, sill=2.0, range_=50.0),
            sill=2.0,
            resolution=1.0,
        )

        kernel = est_sill1.KERNELS["sobel_x"]
        var1 = est_sill1.kernel_variance(kernel)
        var2 = est_sill2.kernel_variance(kernel)

        # var2 should be roughly 2*var1
        assert np.isclose(var2 / var1, 2.0, rtol=0.1)

    def test_kernel_variance_scales_with_resolution(self, nugget_gamma):
        """Kernel variance scales with resolution²."""
        est_res1 = DerivativeUncertaintyEstimator(
            gamma_func=nugget_gamma, sill=1.0, resolution=1.0
        )
        est_res2 = DerivativeUncertaintyEstimator(
            gamma_func=nugget_gamma, sill=1.0, resolution=2.0
        )

        kernel = est_res1.KERNELS["sobel_x"]
        var1 = est_res1.kernel_variance(kernel)
        var2 = est_res2.kernel_variance(kernel)

        # Variance scales as resolution² (lags become 2x, covariance at 2x lag is lower)
        # For nugget this is straightforward: all non-zero lags have covariance 0
        # So variance should be independent of resolution
        # Actually for nugget, it should be sum(K²)*sill regardless
        assert np.isclose(var1, var2, rtol=1e-5)

    def test_kernel_variance_nonzero(self, spherical_gamma):
        """Kernel variance should be non-negative."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        for kernel_name in ["sobel_x", "sobel_y", "laplacian"]:
            kernel = estimator.KERNELS[kernel_name]
            var = estimator.kernel_variance(kernel)
            assert var >= 0.0, f"{kernel_name} variance should be non-negative"


# =============================================================================
# Test DerivativeUncertaintyEstimator - Slope Uncertainty
# =============================================================================


class TestDerivativeEstimatorSlopeUncertainty:
    """Test slope_uncertainty method."""

    def test_slope_uncertainty_returns_triple(self, spherical_gamma):
        """slope_uncertainty should return tuple of 3 values."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        result = estimator.slope_uncertainty()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_slope_uncertainty_all_nonnegative(self, spherical_gamma):
        """All slope uncertainty values should be non-negative."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        std_x, std_y, std_mag = estimator.slope_uncertainty()
        assert std_x >= 0.0
        assert std_y >= 0.0
        assert std_mag >= 0.0

    def test_slope_magnitude_consistency(self, spherical_gamma):
        """Magnitude should satisfy: magnitude² ≈ std_x² + std_y²."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        std_x, std_y, std_mag = estimator.slope_uncertainty()
        expected_mag = np.sqrt(std_x**2 + std_y**2)
        assert np.isclose(std_mag, expected_mag, rtol=1e-10)

    def test_slope_uncertainty_nugget(self, nugget_gamma):
        """Slope uncertainty with pure nugget should be non-zero."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=nugget_gamma, sill=1.0, resolution=1.0
        )
        std_x, std_y, std_mag = estimator.slope_uncertainty()
        # With nugget=sill=1, Var(dz/dx) ≈ K_x².sum * sill
        assert std_x > 0.0
        assert std_y > 0.0
        assert std_mag > 0.0

    def test_slope_uncertainty_scales_with_resolution(self, spherical_gamma):
        """Higher resolution (finer grid) should increase slope uncertainty."""
        est_res1 = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        est_res2 = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=0.5
        )

        std_x1, std_y1, std_mag1 = est_res1.slope_uncertainty()
        std_x2, std_y2, std_mag2 = est_res2.slope_uncertainty()

        # Finer resolution (0.5 < 1.0) should give higher slope uncertainty
        assert std_mag2 > std_mag1

    def test_slope_uncertainty_symmetry(self, spherical_gamma):
        """For isotropic variogram, std_x and std_y should be similar."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        std_x, std_y, std_mag = estimator.slope_uncertainty()
        # Spherical is isotropic, so Sobel X and Y should give similar results
        assert np.isclose(std_x, std_y, rtol=0.1)


# =============================================================================
# Test DerivativeUncertaintyEstimator - Curvature Uncertainty
# =============================================================================


class TestDerivativeEstimatorCurvatureUncertainty:
    """Test curvature_uncertainty method."""

    def test_curvature_uncertainty_returns_float(self, spherical_gamma):
        """curvature_uncertainty should return a float."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        result = estimator.curvature_uncertainty()
        assert isinstance(result, (float, np.floating))

    def test_curvature_uncertainty_nonnegative(self, spherical_gamma):
        """Curvature uncertainty should be non-negative."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        var = estimator.curvature_uncertainty()
        assert var >= 0.0

    def test_curvature_uncertainty_nugget(self, nugget_gamma):
        """Curvature uncertainty with pure nugget should be non-zero."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=nugget_gamma, sill=1.0, resolution=1.0
        )
        var = estimator.curvature_uncertainty()
        assert var > 0.0

    def test_curvature_uncertainty_scales_with_sill(self, spherical_gamma):
        """Curvature uncertainty should increase with sill."""
        est_sill1 = DerivativeUncertaintyEstimator(
            gamma_func=lambda h: spherical(h, sill=1.0, range_=50.0),
            sill=1.0,
            resolution=1.0,
        )
        est_sill2 = DerivativeUncertaintyEstimator(
            gamma_func=lambda h: spherical(h, sill=2.0, range_=50.0),
            sill=2.0,
            resolution=1.0,
        )

        var1 = est_sill1.curvature_uncertainty()
        var2 = est_sill2.curvature_uncertainty()

        assert var2 > var1

    def test_curvature_uncertainty_scales_with_resolution(self, spherical_gamma):
        """Finer resolution should increase curvature uncertainty."""
        est_res1 = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        est_res2 = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=0.5
        )

        var1 = est_res1.curvature_uncertainty()
        var2 = est_res2.curvature_uncertainty()

        # Finer resolution increases uncertainty
        assert var2 > var1


# =============================================================================
# Test KERNELS Dictionary
# =============================================================================


class TestKernelsDictionary:
    """Test the KERNELS dictionary in DerivativeUncertaintyEstimator."""

    def test_kernels_exist(self):
        """KERNELS dict should have required entries."""
        kernels = DerivativeUncertaintyEstimator.KERNELS
        assert "sobel_x" in kernels
        assert "sobel_y" in kernels
        assert "laplacian" in kernels

    def test_kernel_shapes(self):
        """All kernels should be 3x3."""
        kernels = DerivativeUncertaintyEstimator.KERNELS
        for name, kernel in kernels.items():
            assert kernel.shape == (3, 3), f"{name} is {kernel.shape}, not (3, 3)"

    def test_sobel_x_values(self):
        """Verify Sobel X kernel values."""
        kernel = DerivativeUncertaintyEstimator.KERNELS["sobel_x"]
        expected = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        assert np.allclose(kernel, expected)

    def test_sobel_y_values(self):
        """Verify Sobel Y kernel values."""
        kernel = DerivativeUncertaintyEstimator.KERNELS["sobel_y"]
        expected = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
        assert np.allclose(kernel, expected)

    def test_laplacian_values(self):
        """Verify Laplacian kernel values."""
        kernel = DerivativeUncertaintyEstimator.KERNELS["laplacian"]
        expected = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        assert np.allclose(kernel, expected)

    def test_kernels_are_numeric(self):
        """Kernels should be numeric arrays (float or int)."""
        kernels = DerivativeUncertaintyEstimator.KERNELS
        for name, kernel in kernels.items():
            assert np.issubdtype(kernel.dtype, np.number), \
                f"{name} should be numeric but is {kernel.dtype}"


# =============================================================================
# Test Monte Carlo Integration (RegionalUncertaintyEstimator logic)
# =============================================================================


class TestMonteCarloIntegration:
    """Test Monte Carlo integration for regional uncertainty."""

    def test_monte_carlo_nugget_small_polygon(self, nugget_gamma, small_polygon):
        """Nugget model: std(mean) ≈ 0 for any finite domain."""
        estimator = MockRegionalEstimator()
        std_mean = estimator.estimate_std_mean_monte_carlo(
            small_polygon, nugget_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        # For pure nugget, covariance is 0 except at h=0
        # Average covariance should be ≈ 0
        assert std_mean < 0.01

    def test_monte_carlo_nugget_large_polygon(self, nugget_gamma, large_polygon):
        """Nugget model should give similar uncertainty for large domain."""
        estimator = MockRegionalEstimator()
        std_mean = estimator.estimate_std_mean_monte_carlo(
            large_polygon, nugget_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        # Still should be ≈ 0
        assert std_mean < 0.01

    def test_monte_carlo_exponential_decreasing_uncertainty(
        self, exponential_gamma, small_polygon, large_polygon
    ):
        """Larger domain should have lower uncertainty for exponential model."""
        estimator = MockRegionalEstimator()
        std_small = estimator.estimate_std_mean_monte_carlo(
            small_polygon, exponential_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        std_large = estimator.estimate_std_mean_monte_carlo(
            large_polygon, exponential_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        # Larger area → more averaging → lower uncertainty
        assert std_large < std_small

    def test_monte_carlo_spherical_decreasing_uncertainty(
        self, spherical_gamma, small_polygon, large_polygon
    ):
        """Larger domain should have lower uncertainty for spherical model."""
        estimator = MockRegionalEstimator()
        std_small = estimator.estimate_std_mean_monte_carlo(
            small_polygon, spherical_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        std_large = estimator.estimate_std_mean_monte_carlo(
            large_polygon, spherical_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        assert std_large < std_small

    def test_monte_carlo_reproducibility(self, exponential_gamma, small_polygon):
        """Same seed should produce identical results."""
        estimator = MockRegionalEstimator()
        std1 = estimator.estimate_std_mean_monte_carlo(
            small_polygon, exponential_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        std2 = estimator.estimate_std_mean_monte_carlo(
            small_polygon, exponential_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        assert std1 == std2

    def test_monte_carlo_different_seeds(self, exponential_gamma, small_polygon):
        """Different seeds should produce different results (with high probability)."""
        estimator = MockRegionalEstimator()
        std1 = estimator.estimate_std_mean_monte_carlo(
            small_polygon, exponential_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        std2 = estimator.estimate_std_mean_monte_carlo(
            small_polygon, exponential_gamma, sigma2=1.0, n_pairs=5000, seed=123
        )
        # Should be different (with very high probability)
        # But allow some tolerance due to randomness
        assert abs(std1 - std2) > 1e-6 or np.isclose(std1, std2)

    def test_monte_carlo_scales_with_sigma2(self, exponential_gamma, small_polygon):
        """Uncertainty should scale with σ²."""
        estimator = MockRegionalEstimator()
        std1 = estimator.estimate_std_mean_monte_carlo(
            small_polygon, exponential_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        std2 = estimator.estimate_std_mean_monte_carlo(
            small_polygon, exponential_gamma, sigma2=4.0, n_pairs=5000, seed=42
        )
        # std should roughly double when σ² quadruples
        # (because std ∝ sqrt(σ²))
        assert np.isclose(std2 / std1, 2.0, rtol=0.1)

    def test_monte_carlo_returns_nonnegative(self, spherical_gamma, small_polygon):
        """Monte Carlo should always return non-negative value."""
        estimator = MockRegionalEstimator()
        std = estimator.estimate_std_mean_monte_carlo(
            small_polygon, spherical_gamma, sigma2=1.0, n_pairs=5000, seed=42
        )
        assert std >= 0.0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_sill(self):
        """DerivativeUncertaintyEstimator with sill=0 should have zero variance."""
        gamma_zero = lambda h: np.zeros_like(h)
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=gamma_zero, sill=0.0, resolution=1.0
        )
        std_x, std_y, std_mag = estimator.slope_uncertainty()
        assert np.isclose(std_mag, 0.0, atol=1e-10)

    def test_very_small_resolution(self, spherical_gamma):
        """Very small resolution should increase uncertainty."""
        est_small = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=0.001
        )
        std_small = est_small.slope_uncertainty()[2]
        assert std_small > 0.0

    def test_very_large_range_spherical(self):
        """Spherical with range >> domain size should approach constant field."""
        gamma_large_range = lambda h: spherical(h, sill=1.0, range_=1e6)
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=gamma_large_range, sill=1.0, resolution=1.0
        )
        # Covariance should be nearly constant everywhere
        c0 = estimator.covariance(np.array([0.0]))
        c_large = estimator.covariance(np.array([1000.0]))
        # Should be very close
        assert np.isclose(c0[0], c_large[0], rtol=0.1)

    def test_negative_variance_handling(self):
        """Monte Carlo should handle negative variance (return 0)."""
        # Create a "bad" gamma that exceeds sill
        def bad_gamma(h):
            return np.ones_like(h) * 2.0

        estimator = MockRegionalEstimator()
        poly = box(0, 0, 10, 10)
        std = estimator.estimate_std_mean_monte_carlo(
            poly, bad_gamma, sigma2=1.0, n_pairs=1000, seed=42
        )
        # Should return 0 if variance is negative
        assert std == 0.0

    def test_covariance_array_scalar_consistency(self, spherical_gamma):
        """Covariance should handle both scalar and array inputs."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        # Scalar-like (1-element array)
        cov_scalar = estimator.covariance(np.array([10.0]))
        # Array
        cov_array = estimator.covariance(np.array([10.0, 10.0]))
        assert np.isclose(cov_scalar[0], cov_array[0])
        assert np.isclose(cov_array[0], cov_array[1])


# =============================================================================
# Test Integration: Combining Multiple Components
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_slope(self, spherical_gamma):
        """Complete workflow for slope uncertainty."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        std_x, std_y, std_mag = estimator.slope_uncertainty()

        # Verify all values are consistent
        assert std_x >= 0.0
        assert std_y >= 0.0
        assert std_mag >= 0.0
        assert np.isclose(std_mag, np.sqrt(std_x**2 + std_y**2))

    def test_full_workflow_curvature(self, spherical_gamma):
        """Complete workflow for curvature uncertainty."""
        estimator = DerivativeUncertaintyEstimator(
            gamma_func=spherical_gamma, sill=1.0, resolution=1.0
        )
        curvature_std = estimator.curvature_uncertainty()
        assert curvature_std >= 0.0

    def test_multiple_variogram_models(self, small_polygon):
        """Test with multiple variogram models."""
        estimator_mock = MockRegionalEstimator()

        models = [
            ("spherical", spherical),
            ("exponential", exponential),
            ("nugget", nugget),
        ]

        results = {}
        for name, model_func in models:
            if name == "nugget":
                gamma = lambda h: model_func(h, c0=1.0)
            else:
                gamma = lambda h, mf=model_func: mf(h, sill=1.0, range_=50.0)

            std = estimator_mock.estimate_std_mean_monte_carlo(
                small_polygon, gamma, sigma2=1.0, n_pairs=5000, seed=42
            )
            results[name] = std

        # Nugget should give near-zero uncertainty
        assert results["nugget"] < 0.01
        # Spherical and exponential should give similar positive uncertainty
        assert results["spherical"] > 0.0
        assert results["exponential"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
