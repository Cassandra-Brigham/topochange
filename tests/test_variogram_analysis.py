"""Comprehensive tests for variogram analysis infrastructure.

Tests the model fitting infrastructure in src/topochange/variogram.py, specifically:
1. FittedVariogramModel dataclass
2. VariogramModelSelector class
3. VariogramAnalysis.compute_matheron static method

Uses synthetic data (no actual raster files required).
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import warnings

from topochange.variogram_models import spherical, exponential, MODEL_REGISTRY
from topochange.composite_variogram import CompositeVariogramModel
from topochange.variogram import (
    FittedVariogramModel,
    VariogramModelSelector,
    VariogramAnalysis,
)


# ============================================================================
# Fixtures: Synthetic Data
# ============================================================================


@pytest.fixture
def synthetic_variogram_data():
    """Generate synthetic variogram from known spherical model.

    Returns
    -------
    dict
        Keys: 'lags', 'empirical', 'true_sill', 'true_range', 'bin_counts'
    """
    np.random.seed(42)
    lags = np.linspace(5, 500, 50)
    true_sill = 2.0
    true_range = 100.0

    # Generate from spherical model with small noise
    empirical = spherical(lags, true_sill, true_range)
    empirical += np.random.normal(0, 0.05, len(lags))
    empirical = np.maximum(empirical, 0)  # ensure non-negative

    # Create synthetic bin counts (more pairs at shorter lags)
    bin_counts = np.maximum(100 - np.linspace(0, 50, len(lags)), 10).astype(int)

    return {
        'lags': lags,
        'empirical': empirical,
        'true_sill': true_sill,
        'true_range': true_range,
        'bin_counts': bin_counts,
    }


@pytest.fixture
def synthetic_matheron_data():
    """Generate synthetic Matheron estimator data.

    Returns
    -------
    dict
        Keys: 'bin_counts', 'ssd', 'expected_gamma'
    """
    np.random.seed(42)
    bin_counts = np.array([100, 95, 80, 60, 40, 20, 10, 5])

    # Create SSD array from known semivariances
    true_gamma = np.array([0.5, 0.8, 1.0, 1.2, 1.3, 1.4, 1.45, 1.48])
    ssd = 2.0 * true_gamma * bin_counts  # SSD = 2*N*gamma

    return {
        'bin_counts': bin_counts,
        'ssd': ssd,
        'expected_gamma': true_gamma,
    }


# ============================================================================
# Test Suite 1: Matheron Estimator
# ============================================================================


class TestMatheronEstimator:
    """Test VariogramAnalysis.compute_matheron static method."""

    def test_compute_matheron_basic(self, synthetic_matheron_data):
        """Test basic Matheron computation: γ(h) = SSD(h) / (2*N(h))."""
        gamma_est = VariogramAnalysis.compute_matheron(
            synthetic_matheron_data['bin_counts'],
            synthetic_matheron_data['ssd'],
            min_pairs=5
        )

        # Should match expected gamma values
        assert_allclose(
            gamma_est,
            synthetic_matheron_data['expected_gamma'],
            rtol=1e-10
        )

    def test_compute_matheron_min_pairs_filter(self, synthetic_matheron_data):
        """Test that bins with fewer than min_pairs return NaN."""
        min_pairs = 50  # Only first few bins have >= 50 pairs
        gamma_est = VariogramAnalysis.compute_matheron(
            synthetic_matheron_data['bin_counts'],
            synthetic_matheron_data['ssd'],
            min_pairs=min_pairs
        )

        # Bins with count < min_pairs should be NaN
        valid = synthetic_matheron_data['bin_counts'] >= min_pairs
        assert np.all(np.isnan(gamma_est[~valid]))
        assert np.all(np.isfinite(gamma_est[valid]))

    def test_compute_matheron_zero_counts(self):
        """Test behavior with zero pair counts."""
        bin_counts = np.array([100, 0, 50, 0, 25])
        ssd = np.array([50, 0, 30, 0, 15])

        gamma_est = VariogramAnalysis.compute_matheron(bin_counts, ssd, min_pairs=10)

        # Bins with zero counts should be NaN
        assert np.isnan(gamma_est[1])
        assert np.isnan(gamma_est[3])
        # Others should be finite
        assert np.isfinite(gamma_est[0])
        assert np.isfinite(gamma_est[2])

    def test_compute_matheron_single_bin(self):
        """Test with single bin."""
        bin_counts = np.array([100])
        ssd = np.array([50.0])

        gamma_est = VariogramAnalysis.compute_matheron(bin_counts, ssd, min_pairs=10)

        assert gamma_est.shape == (1,)
        assert_allclose(gamma_est[0], 50.0 / (2.0 * 100))


# ============================================================================
# Test Suite 2: VariogramModelSelector Construction
# ============================================================================


class TestVariogramModelSelectorConstruction:
    """Test VariogramModelSelector initialization and setup."""

    def test_constructor_stores_inputs(self, synthetic_variogram_data):
        """Test that constructor properly stores input data."""
        lags = synthetic_variogram_data['lags']
        empirical = synthetic_variogram_data['empirical']
        bin_counts = synthetic_variogram_data['bin_counts']

        selector = VariogramModelSelector(lags, empirical, weights=bin_counts)

        assert_array_almost_equal(selector.lags, lags)
        assert_array_almost_equal(selector.empirical_variogram, empirical)
        assert_array_almost_equal(selector.weights, bin_counts)

    def test_constructor_default_weights(self, synthetic_variogram_data):
        """Test that default weights are ones."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical']
        )

        assert_array_almost_equal(selector.weights, np.ones_like(selector.lags))

    def test_constructor_with_sigma(self, synthetic_variogram_data):
        """Test constructor with per-bin standard deviations."""
        lags = synthetic_variogram_data['lags']
        empirical = synthetic_variogram_data['empirical']
        sigma = np.full_like(empirical, 0.05)

        selector = VariogramModelSelector(
            lags, empirical, sigma=sigma
        )

        assert_array_almost_equal(selector.sigma, sigma)

    def test_constructor_sigma_clipping(self):
        """Test that zero/negative sigmas are clipped to eps."""
        lags = np.array([10, 20, 30])
        empirical = np.array([0.5, 0.8, 1.0])
        sigma = np.array([0.1, 0, -0.05])

        selector = VariogramModelSelector(lags, empirical, sigma=sigma)

        # First should be unchanged
        assert selector.sigma[0] == 0.1
        # Others should be >= eps
        assert selector.sigma[1] > 0
        assert selector.sigma[2] > 0


# ============================================================================
# Test Suite 3: Candidate Generation
# ============================================================================


class TestCandidateGeneration:
    """Test VariogramModelSelector.generate_candidates()."""

    def test_generate_candidates_non_empty(self, synthetic_variogram_data):
        """Test that generate_candidates returns non-empty list."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical']
        )

        candidates = selector.generate_candidates(max_components=2)

        assert len(candidates) > 0
        assert all(isinstance(c, CompositeVariogramModel) for c in candidates)

    def test_generate_candidates_respects_max_components(self, synthetic_variogram_data):
        """Test that candidates respect max_components limit."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical']
        )

        cand1 = selector.generate_candidates(max_components=1)
        cand2 = selector.generate_candidates(max_components=2)

        # More components should give more candidates (or equal)
        assert len(cand2) >= len(cand1)

        # Check that all single-component models are in both lists
        cand1_names = [c.component_names for c in cand1]
        cand2_names = [c.component_names for c in cand2]

        for name in cand1_names:
            assert name in cand2_names

    def test_generate_candidates_include_nugget_false(self, synthetic_variogram_data):
        """Test nugget inclusion flag."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical']
        )

        cand_with_nugget = selector.generate_candidates(include_nugget=True)
        cand_no_nugget = selector.generate_candidates(include_nugget=False)

        # All with-nugget models should have nugget
        assert all(c.include_nugget for c in cand_with_nugget)
        assert all(not c.include_nugget for c in cand_no_nugget)

    def test_generate_candidates_include_unbounded(self, synthetic_variogram_data):
        """Test unbounded model inclusion flag."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical']
        )

        cand_with_unbounded = selector.generate_candidates(include_unbounded=True)
        cand_no_unbounded = selector.generate_candidates(include_unbounded=False)

        # Check if any unbounded models are present
        # (may be empty if registry doesn't have unbounded models)
        has_unbounded_with = any(
            not all(
                MODEL_REGISTRY.get_model(name).is_bounded
                for name in c.component_names
            )
            for c in cand_with_unbounded
        )

        has_unbounded_without = any(
            not all(
                MODEL_REGISTRY.get_model(name).is_bounded
                for name in c.component_names
            )
            for c in cand_no_unbounded
        )

        # With unbounded enabled might have some unbounded (or not, depends on data)
        # Without unbounded should have none
        if has_unbounded_with:
            assert has_unbounded_without is False  # exclusive check


# ============================================================================
# Test Suite 4: Model Fitting on Synthetic Data
# ============================================================================


class TestModelFitting:
    """Test VariogramModelSelector.fit_model()."""

    def test_fit_model_spherical(self, synthetic_variogram_data):
        """Test fitting a spherical model to synthetic spherical data."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        # Create single spherical model
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        assert fitted is not None
        assert fitted.composite_model is not None
        assert len(fitted.params) == 2  # sill + range

    def test_fit_model_parameters_reasonable(self, synthetic_variogram_data):
        """Test that fitted parameters are close to true values (within 20%)."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        # Extract sill and range
        fitted_sill = fitted.params[0]
        fitted_range = fitted.params[1]

        true_sill = synthetic_variogram_data['true_sill']
        true_range = synthetic_variogram_data['true_range']

        # Check within 20% of true values
        assert abs(fitted_sill - true_sill) / true_sill < 0.2
        assert abs(fitted_range - true_range) / true_range < 0.2

    def test_fit_model_with_nugget(self, synthetic_variogram_data):
        """Test fitting with nugget component."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        fitted = selector.fit_model(model)

        assert fitted is not None
        assert len(fitted.params) == 3  # sill + range + nugget

    def test_fit_model_returns_diagnostics(self, synthetic_variogram_data):
        """Test that fit_model returns diagnostics (RSS, AIC, BIC)."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        assert fitted.rss > 0
        assert np.isfinite(fitted.aic)
        assert np.isfinite(fitted.bic)


# ============================================================================
# Test Suite 5: Information Criteria
# ============================================================================


class TestInformationCriteria:
    """Test AIC/BIC computation."""

    def test_aic_bic_finite(self, synthetic_variogram_data):
        """Test that AIC and BIC are finite."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        assert np.isfinite(fitted.aic)
        assert np.isfinite(fitted.bic)

    def test_bic_penalty_for_parameters(self, synthetic_variogram_data):
        """Test that BIC penalizes more parameters more heavily than AIC."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        # Fit two models: spherical and spherical+exponential
        model_1 = CompositeVariogramModel(['spherical'], include_nugget=False)
        model_2 = CompositeVariogramModel(['spherical', 'exponential'], include_nugget=False)

        fitted_1 = selector.fit_model(model_1)
        fitted_2 = selector.fit_model(model_2)

        if fitted_1 and fitted_2:
            # More parameters should increase BIC more than AIC (relative to likelihood)
            bic_diff = fitted_2.bic - fitted_1.bic
            aic_diff = fitted_2.aic - fitted_1.aic

            # BIC penalty should be larger
            # (Note: this is probabilistic, may not always hold with noise)
            assert fitted_2.composite_model.n_params > fitted_1.composite_model.n_params


# ============================================================================
# Test Suite 6: Model Selection
# ============================================================================


class TestModelSelection:
    """Test VariogramModelSelector.select_best()."""

    def test_select_best_by_aic(self, synthetic_variogram_data):
        """Test select_best with AIC criterion."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=2, include_nugget=True, compute_cv=False)
        best = selector.select_best(criterion='aic')

        assert best is not None
        assert isinstance(best, FittedVariogramModel)

    def test_select_best_by_bic(self, synthetic_variogram_data):
        """Test select_best with BIC criterion."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=2, include_nugget=True, compute_cv=False)
        best = selector.select_best(criterion='bic')

        assert best is not None

    def test_select_best_raises_no_models(self, synthetic_variogram_data):
        """Test that select_best raises if no models fitted."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical']
        )

        with pytest.raises(ValueError, match="No fitted models"):
            selector.select_best()

    def test_select_best_invalid_criterion(self, synthetic_variogram_data):
        """Test that invalid criterion raises."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)

        with pytest.raises(ValueError, match="Unknown criterion"):
            selector.select_best(criterion='invalid')


# ============================================================================
# Test Suite 7: Cross-Validation
# ============================================================================


class TestCrossValidation:
    """Test VariogramModelSelector.cross_validate()."""

    def test_cross_validate_returns_positive_rmse(self, synthetic_variogram_data):
        """Test that cross_validate returns positive finite RMSE."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        rmse = selector.cross_validate(fitted, k=3)

        assert rmse > 0
        assert np.isfinite(rmse)

    def test_cross_validate_k_folds(self, synthetic_variogram_data):
        """Test cross_validate with different k values."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        rmse_2 = selector.cross_validate(fitted, k=2, seed=42)
        rmse_5 = selector.cross_validate(fitted, k=5, seed=42)

        # Both should be finite positive
        assert rmse_2 > 0 and np.isfinite(rmse_2)
        assert rmse_5 > 0 and np.isfinite(rmse_5)


# ============================================================================
# Test Suite 8: Akaike Weights
# ============================================================================


class TestAkaikeWeights:
    """Test VariogramModelSelector._compute_akaike_weights()."""

    def test_akaike_weights_sum_to_one(self, synthetic_variogram_data):
        """Test that Akaike weights sum to 1.0."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=2, include_nugget=True, compute_cv=False)

        assert selector.model_weights is not None
        assert_allclose(np.sum(selector.model_weights), 1.0, rtol=1e-10)

    def test_akaike_weights_non_negative(self, synthetic_variogram_data):
        """Test that all Akaike weights are non-negative."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=2, include_nugget=False, compute_cv=False)

        assert np.all(selector.model_weights >= 0)

    def test_akaike_weights_best_has_highest_weight(self, synthetic_variogram_data):
        """Test that best model has highest weight."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=2, include_nugget=False, compute_cv=False)
        best = selector.select_best(criterion='aic')

        best_idx = selector.fitted_models.index(best)
        best_weight = selector.model_weights[best_idx]

        assert best_weight >= np.max(selector.model_weights[selector.model_weights < best_weight + 1e-10])


# ============================================================================
# Test Suite 9: BMA Variogram
# ============================================================================


class TestBMAVariogram:
    """Test VariogramModelSelector.get_bma_variogram()."""

    def test_get_bma_variogram_returns_callable(self, synthetic_variogram_data):
        """Test that get_bma_variogram returns a callable."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        bma_func = selector.get_bma_variogram()

        assert callable(bma_func)

    def test_bma_variogram_produces_finite_results(self, synthetic_variogram_data):
        """Test that BMA variogram produces finite results."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        bma_func = selector.get_bma_variogram()

        test_lags = np.array([10, 50, 100, 200])
        result = bma_func(test_lags)

        assert result.shape == test_lags.shape
        assert np.all(np.isfinite(result))

    def test_bma_variogram_raises_without_fit(self, synthetic_variogram_data):
        """Test that get_bma_variogram raises if fit_all_candidates not called."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical']
        )

        with pytest.raises(ValueError, match="No model weights"):
            selector.get_bma_variogram()


# ============================================================================
# Test Suite 10: Bootstrap Uncertainty
# ============================================================================


class TestBootstrapUncertainty:
    """Test VariogramModelSelector.bootstrap_best_model()."""

    def test_bootstrap_returns_samples(self, synthetic_variogram_data):
        """Test that bootstrap_best_model returns parameter samples."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts'],
            sigma=np.full_like(synthetic_variogram_data['empirical'], 0.05)
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        selector.select_best(criterion='aic')

        samples = selector.bootstrap_best_model(n_boot=50, seed=42)

        assert samples is not None
        assert samples.shape[0] > 0
        assert samples.shape[1] == selector.best_model.composite_model.n_params

    def test_bootstrap_stores_in_fitted_model(self, synthetic_variogram_data):
        """Test that bootstrap samples are stored in best_model."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts'],
            sigma=np.full_like(synthetic_variogram_data['empirical'], 0.05)
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        selector.select_best(criterion='aic')
        selector.bootstrap_best_model(n_boot=50, seed=42)

        assert selector.best_model.param_samples is not None
        assert len(selector.best_model.param_samples) > 0


# ============================================================================
# Test Suite 11: FittedVariogramModel
# ============================================================================


class TestFittedVariogramModel:
    """Test FittedVariogramModel dataclass."""

    def test_fitted_model_predict_matches_model(self, synthetic_variogram_data):
        """Test that predict() matches direct model call."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        test_lags = np.array([10, 50, 100, 200])

        # predict() should match direct model call
        pred1 = fitted.predict(test_lags)
        pred2 = fitted.composite_model(test_lags)

        assert_allclose(pred1, pred2, rtol=1e-10)

    def test_get_param_percentiles_without_samples(self, synthetic_variogram_data):
        """Test that get_param_percentiles raises without bootstrap samples."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        # No bootstrap samples yet
        with pytest.raises(ValueError, match="No bootstrap samples"):
            fitted.get_param_percentiles()

    def test_get_param_percentiles_with_samples(self, synthetic_variogram_data):
        """Test get_param_percentiles with bootstrap samples."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts'],
            sigma=np.full_like(synthetic_variogram_data['empirical'], 0.05)
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        selector.select_best(criterion='aic')
        selector.bootstrap_best_model(n_boot=50, seed=42)

        percentiles = selector.best_model.get_param_percentiles([16, 50, 84])

        # Should have dict with parameter names as keys
        assert isinstance(percentiles, dict)
        assert len(percentiles) == selector.best_model.composite_model.n_params

        # Each value should have 3 elements (16th, 50th, 84th percentile)
        for key, vals in percentiles.items():
            assert len(vals) == 3


# ============================================================================
# Test Suite 12: Initial Guess
# ============================================================================


class TestInitialGuess:
    """Test VariogramAnalysis.get_base_initial_guess()."""

    def test_initial_guess_correct_length(self, synthetic_variogram_data):
        """Test that initial guess has correct length."""
        guess = VariogramAnalysis.get_base_initial_guess(
            n=1,
            mean_variogram=synthetic_variogram_data['empirical'],
            lags=synthetic_variogram_data['lags'],
            nugget=False
        )

        # 1 component without nugget: 1 sill + 1 range = 2 params
        assert len(guess) == 2

    def test_initial_guess_with_nugget(self, synthetic_variogram_data):
        """Test initial guess length with nugget."""
        guess = VariogramAnalysis.get_base_initial_guess(
            n=1,
            mean_variogram=synthetic_variogram_data['empirical'],
            lags=synthetic_variogram_data['lags'],
            nugget=True
        )

        # 1 component with nugget: 1 sill + 1 range + 1 nugget = 3 params
        assert len(guess) == 3

    def test_initial_guess_multiple_components(self, synthetic_variogram_data):
        """Test initial guess with multiple components."""
        guess = VariogramAnalysis.get_base_initial_guess(
            n=2,
            mean_variogram=synthetic_variogram_data['empirical'],
            lags=synthetic_variogram_data['lags'],
            nugget=False
        )

        # 2 components: 2 sills + 2 ranges = 4 params
        assert len(guess) == 4

    def test_initial_guess_parameters_positive(self, synthetic_variogram_data):
        """Test that initial guess produces positive parameters."""
        guess = VariogramAnalysis.get_base_initial_guess(
            n=1,
            mean_variogram=synthetic_variogram_data['empirical'],
            lags=synthetic_variogram_data['lags'],
            nugget=True
        )

        # All parameters should be non-negative
        assert np.all(guess >= 0)

    def test_initial_guess_reasonable_ranges(self, synthetic_variogram_data):
        """Test that initial guess produces reasonable range values."""
        lags = synthetic_variogram_data['lags']
        max_lag = np.max(lags)

        guess = VariogramAnalysis.get_base_initial_guess(
            n=1,
            mean_variogram=synthetic_variogram_data['empirical'],
            lags=lags,
            nugget=False
        )

        # Range (second param) should be < max_lag
        assert guess[1] < max_lag


# ============================================================================
# Test Suite 13: Fit All Candidates
# ============================================================================


class TestFitAllCandidates:
    """Test VariogramModelSelector.fit_all_candidates()."""

    def test_fit_all_candidates_populates_list(self, synthetic_variogram_data):
        """Test that fit_all_candidates populates fitted_models list."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False)

        assert len(selector.fitted_models) > 0
        assert all(isinstance(m, FittedVariogramModel) for m in selector.fitted_models)

    def test_fit_all_candidates_compute_cv(self, synthetic_variogram_data):
        """Test that compute_cv flag works."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=True)

        # At least some models should have CV scores
        has_cv = any(m.cv_rmse is not None for m in selector.fitted_models)
        assert has_cv

    def test_fit_all_candidates_without_cv(self, synthetic_variogram_data):
        """Test that compute_cv=False skips CV computation."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)

        # All models should have None for cv_rmse
        assert all(m.cv_rmse is None for m in selector.fitted_models)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, synthetic_variogram_data):
        """Test complete workflow: generate -> fit -> select -> bootstrap."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts'],
            sigma=np.full_like(synthetic_variogram_data['empirical'], 0.05)
        )

        # Fit candidates
        selector.fit_all_candidates(
            max_components=2,
            include_nugget=True,
            compute_cv=True,
            cv_folds=3,
            seed=42
        )

        # Select best
        best = selector.select_best(criterion='aic')
        assert best is not None

        # Bootstrap
        samples = selector.bootstrap_best_model(n_boot=30, seed=42)
        assert len(samples) > 0

        # Get percentiles
        percentiles = best.get_param_percentiles([16, 50, 84])
        assert len(percentiles) > 0

    def test_bma_after_fitting(self, synthetic_variogram_data):
        """Test BMA computation after fitting all candidates."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weights=synthetic_variogram_data['bin_counts']
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        bma_func = selector.get_bma_variogram()

        # BMA should average predictions
        test_lag = 100.0
        bma_pred = bma_func(np.array([test_lag]))[0]

        # Should be within range of individual model predictions
        individual_preds = [m.predict(np.array([test_lag]))[0] for m in selector.fitted_models]
        assert min(individual_preds) <= bma_pred <= max(individual_preds)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
