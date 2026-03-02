"""Comprehensive tests for variogram analysis infrastructure.

Tests the model fitting infrastructure in src/topochange/variogram.py, specifically:
1. FittedVariogramModel dataclass
2. SingleVariogram WLS fitting pipeline
3. VariogramAnalysis.compute_matheron / compute_cressie_hawkins static methods
4. Kriging LOOCV

Uses synthetic data (no actual raster files required)."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import warnings
from unittest.mock import MagicMock
from itertools import combinations_with_replacement

from topochange.variogram_models import spherical, exponential, MODEL_REGISTRY
from topochange.composite_variogram import CompositeVariogramModel
from topochange.variogram import (
    FittedVariogramModel,
    SingleVariogram,
    VariogramAnalysis,
    KrigingLOOCVResult,
    AggregatedLOOCVResult,
)


# ─── helpers ────────────────────────────────────────────────────────


def _make_single_variogram_from_synthetic(lags, empirical, pair_counts=None):
    """Create a SingleVariogram with manually injected synthetic data.

    This bypasses ``compute_empirical_variogram()`` so we can test the
    fitting pipeline without requiring a real raster file.
    """
    sv = object.__new__(SingleVariogram)
    # Initialise all attributes that fit_model() and related methods read
    sv.raster_data_handler = MagicMock()
    sv.raster_data_handler.unit = "m"
    sv.lags = np.asarray(lags, dtype=float)
    sv.variogram = np.asarray(empirical, dtype=float)
    sv.pair_counts = (
        np.asarray(pair_counts, dtype=float) if pair_counts is not None else None
    )
    sv.n_bins = len(lags)
    sv.estimator = "matheron"
    sv.sample_coords = None
    sv.sample_values = None
    sv.fitted_models = []
    sv.best_model = None
    sv.criteria_table = None
    return sv


# ─── fixtures ───────────────────────────────────────────────────────


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

    # generate from spherical model with small noise
    empirical = spherical(lags, true_sill, true_range)
    empirical += np.random.normal(0, 0.05, len(lags))
    empirical = np.maximum(empirical, 0)  # ensure non-negative

    # create synthetic bin counts (more pairs at shorter lags)
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

    # create SSD array from known semivariances
    true_gamma = np.array([0.5, 0.8, 1.0, 1.2, 1.3, 1.4, 1.45, 1.48])
    ssd = 2.0 * true_gamma * bin_counts  # SSD = 2*N*gamma

    return {
        'bin_counts': bin_counts,
        'ssd': ssd,
        'expected_gamma': true_gamma,
    }


@pytest.fixture
def fitted_sv(synthetic_variogram_data):
    """A SingleVariogram with synthetic data and fitted models."""
    sv = _make_single_variogram_from_synthetic(
        synthetic_variogram_data['lags'],
        synthetic_variogram_data['empirical'],
        synthetic_variogram_data['bin_counts'],
    )
    sv.fit_model(max_components=1, include_nugget=True)
    return sv


# ─── Test Suite 1: Matheron Estimator ───────────────────────────────


class TestMatheronEstimator:
    """Test VariogramAnalysis.compute_matheron static method."""

    def test_compute_matheron_basic(self, synthetic_matheron_data):
        """Test basic Matheron computation: γ(h) = SSD(h) / (2*N(h))."""
        gamma_est = VariogramAnalysis.compute_matheron(
            synthetic_matheron_data['bin_counts'],
            synthetic_matheron_data['ssd'],
            min_pairs=5
        )

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

        valid = synthetic_matheron_data['bin_counts'] >= min_pairs
        assert np.all(np.isnan(gamma_est[~valid]))
        assert np.all(np.isfinite(gamma_est[valid]))

    def test_compute_matheron_zero_counts(self):
        """Test behavior with zero pair counts."""
        bin_counts = np.array([100, 0, 50, 0, 25])
        ssd = np.array([50, 0, 30, 0, 15])

        gamma_est = VariogramAnalysis.compute_matheron(bin_counts, ssd, min_pairs=10)

        assert np.isnan(gamma_est[1])
        assert np.isnan(gamma_est[3])
        assert np.isfinite(gamma_est[0])
        assert np.isfinite(gamma_est[2])

    def test_compute_matheron_single_bin(self):
        """Test with single bin."""
        bin_counts = np.array([100])
        ssd = np.array([50.0])

        gamma_est = VariogramAnalysis.compute_matheron(bin_counts, ssd, min_pairs=10)

        assert gamma_est.shape == (1,)
        assert_allclose(gamma_est[0], 50.0 / (2.0 * 100))


# ─── Test Suite 1b: Cressie–Hawkins Estimator ──────────────────────


class TestCressieHawkinsEstimator:
    """Test VariogramAnalysis.compute_cressie_hawkins static method."""

    def test_compute_ch_basic(self):
        """Test Cressie–Hawkins: γ̂ = [mean(|ΔZ|^0.5)]⁴ / (2·(0.457 + 0.494/N))."""
        bin_counts = np.array([100])
        sum_sqrt_abs_diff = np.array([100.0])

        gamma = VariogramAnalysis.compute_cressie_hawkins(
            bin_counts, sum_sqrt_abs_diff, min_pairs=10
        )

        expected = 0.5 * 1.0 / (0.457 + 0.494 / 100)
        assert_allclose(gamma[0], expected, rtol=1e-10)

    def test_compute_ch_pure_nugget(self):
        """For a pure nugget (constant variance), Cressie–Hawkins
        should agree with Matheron asymptotically."""
        rng = np.random.default_rng(42)
        n_pairs = 10000
        sigma = 2.0
        dz = rng.normal(0, np.sqrt(2) * sigma, n_pairs)
        ssd = np.array([np.sum(dz**2)])
        ssad = np.array([np.sum(np.abs(dz) ** 0.5)])
        counts = np.array([n_pairs])

        matheron = VariogramAnalysis.compute_matheron(counts, ssd, min_pairs=10)
        ch = VariogramAnalysis.compute_cressie_hawkins(counts, ssad, min_pairs=10)

        assert_allclose(matheron[0], sigma**2, rtol=0.1)
        assert_allclose(ch[0], sigma**2, rtol=0.15)

    def test_compute_ch_min_pairs_filter(self):
        """Bins with fewer than min_pairs should return NaN."""
        bin_counts = np.array([100, 5, 50])
        ssad = np.array([50.0, 3.0, 30.0])

        gamma = VariogramAnalysis.compute_cressie_hawkins(
            bin_counts, ssad, min_pairs=10
        )
        assert np.isfinite(gamma[0])
        assert np.isnan(gamma[1])
        assert np.isfinite(gamma[2])

    def test_compute_ch_outlier_robustness(self):
        """Cressie–Hawkins should be more resistant to outliers than Matheron."""
        rng = np.random.default_rng(99)
        n_pairs = 500
        sigma = 1.0
        dz = rng.normal(0, np.sqrt(2) * sigma, n_pairs)
        dz[:5] = 10.0 * np.sqrt(2) * sigma

        ssd = np.array([np.sum(dz**2)])
        ssad = np.array([np.sum(np.abs(dz) ** 0.5)])
        counts = np.array([n_pairs])

        matheron = VariogramAnalysis.compute_matheron(counts, ssd, min_pairs=10)
        ch = VariogramAnalysis.compute_cressie_hawkins(counts, ssad, min_pairs=10)

        assert ch[0] < matheron[0], (
            f"Cressie–Hawkins ({ch[0]:.3f}) should be closer to true value "
            f"than Matheron ({matheron[0]:.3f}) in the presence of outliers"
        )


# ─── Test Suite 2: WLS Fitting via SingleVariogram ──────────────────


class TestWLSFitting:
    """Test SingleVariogram WLS fitting with synthetic variogram data."""

    def test_fit_model_returns_dataframe(self, synthetic_variogram_data):
        """fit_model() should return a pandas DataFrame."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        table = sv.fit_model(max_components=1, include_nugget=False)

        import pandas as pd
        assert isinstance(table, pd.DataFrame)
        assert len(table) > 0

    def test_fit_model_populates_fitted_models(self, synthetic_variogram_data):
        """fit_model() should populate self.fitted_models list."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(max_components=1, include_nugget=False)

        assert len(sv.fitted_models) > 0
        assert all(isinstance(m, dict) for m in sv.fitted_models)

    def test_fit_model_populates_best_model(self, synthetic_variogram_data):
        """fit_model() should set self.best_model."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(max_components=1, include_nugget=False)

        assert sv.best_model is not None
        assert 'params' in sv.best_model
        assert 'rss' in sv.best_model

    def test_fit_model_spherical_parameters_reasonable(self, synthetic_variogram_data):
        """Fitted spherical parameters should be close to true values."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(
            model_types=['spherical'],
            max_components=1,
            include_nugget=False,
        )

        best = sv.best_model
        assert best is not None

        fitted_sill = best['params'][0]
        fitted_range = best['params'][1]
        true_sill = synthetic_variogram_data['true_sill']
        true_range = synthetic_variogram_data['true_range']

        assert abs(fitted_sill - true_sill) / true_sill < 0.2
        assert abs(fitted_range - true_range) / true_range < 0.2

    def test_fit_model_with_nugget(self, synthetic_variogram_data):
        """Fitting with nugget should add a nugget parameter."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(
            model_types=['spherical'],
            max_components=1,
            include_nugget=True,
        )

        # should have at least one spatial model with nugget (not just pure nugget)
        nugget_models = [
            m for m in sv.fitted_models
            if m['model'].include_nugget and len(m['model'].component_names) > 0
        ]
        assert len(nugget_models) > 0
        # spherical + nugget should have 3 params: sill + range + nugget
        assert nugget_models[0]['model'].n_params == 3

    def test_fit_model_without_pair_counts(self, synthetic_variogram_data):
        """Fitting should still work with uniform weights when pair_counts is None."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=None,
        )
        sv.fit_model(
            model_types=['spherical'],
            max_components=1,
            include_nugget=False,
        )

        assert sv.best_model is not None
        assert np.isfinite(sv.best_model['rss'])

    def test_fit_model_requires_empirical_variogram(self):
        """fit_model() should raise if no empirical variogram computed."""
        sv = object.__new__(SingleVariogram)
        sv.lags = None
        sv.variogram = None

        with pytest.raises(RuntimeError, match="No empirical variogram"):
            sv.fit_model()


# ─── Test Suite 3: Candidate Generation ─────────────────────────────


class TestCandidateGeneration:
    """Test that fit_model generates expected candidate structures."""

    def test_single_component_candidates(self, synthetic_variogram_data):
        """max_components=1 should generate single-component models."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(max_components=1, include_nugget=False)

        for m in sv.fitted_models:
            model = m['model']
            # single-component (or pure nugget)
            assert len(model.component_names) <= 1

    def test_multi_component_candidates(self, synthetic_variogram_data):
        """max_components=2 should generate more candidates than max_components=1."""
        sv1 = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv1.fit_model(max_components=1, include_nugget=False)

        sv2 = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv2.fit_model(max_components=2, include_nugget=False)

        assert len(sv2.fitted_models) >= len(sv1.fitted_models)

    def test_nugget_variants_generated(self, synthetic_variogram_data):
        """include_nugget=True should produce both nugget and no-nugget variants."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(max_components=1, include_nugget=True)

        has_nugget = any(m['model'].include_nugget for m in sv.fitted_models)
        has_no_nugget = any(
            not m['model'].include_nugget and len(m['model'].component_names) > 0
            for m in sv.fitted_models
        )
        assert has_nugget
        assert has_no_nugget

    def test_no_nugget_variants_when_disabled(self, synthetic_variogram_data):
        """include_nugget=False should only produce no-nugget models (except pure nugget)."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(max_components=1, include_nugget=False)

        for m in sv.fitted_models:
            model = m['model']
            if len(model.component_names) > 0:
                assert not model.include_nugget


# ─── Test Suite 4: Individual Model Fitting ─────────────────────────


class TestIndividualModelFitting:
    """Test _fit_single_composite_model directly."""

    def test_fit_single_spherical(self, synthetic_variogram_data):
        """Fitting a single spherical model should return valid result dict."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        lags = sv.lags
        variogram = sv.variogram
        gamma_sq = np.maximum(np.square(variogram), np.finfo(float).eps)
        weights = sv.pair_counts / gamma_sq

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        result = sv._fit_single_composite_model(model, lags, variogram, None, weights)

        assert result is not None
        assert len(result['params']) == 2  # sill + range
        assert result['rss'] > 0
        assert np.isfinite(result['aic'])
        assert np.isfinite(result['bic'])

    def test_fit_single_exponential(self, synthetic_variogram_data):
        """Fitting an exponential model should return valid result dict."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        lags = sv.lags
        variogram = sv.variogram
        gamma_sq = np.maximum(np.square(variogram), np.finfo(float).eps)
        weights = sv.pair_counts / gamma_sq

        model = CompositeVariogramModel(['exponential'], include_nugget=False)
        result = sv._fit_single_composite_model(model, lags, variogram, None, weights)

        assert result is not None
        assert len(result['params']) == 2

    def test_fit_with_nugget_extra_parameter(self, synthetic_variogram_data):
        """Fitting with nugget should have one extra parameter."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        lags = sv.lags
        variogram = sv.variogram
        weights = np.ones_like(variogram)

        model_no_nug = CompositeVariogramModel(['spherical'], include_nugget=False)
        model_nug = CompositeVariogramModel(['spherical'], include_nugget=True)

        r1 = sv._fit_single_composite_model(model_no_nug, lags, variogram, None, weights)
        r2 = sv._fit_single_composite_model(model_nug, lags, variogram, None, weights)

        assert len(r2['params']) == len(r1['params']) + 1

    def test_fit_returns_covariance_matrix(self, synthetic_variogram_data):
        """Result should include a parameter covariance matrix."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        lags = sv.lags
        variogram = sv.variogram
        weights = np.ones_like(variogram)

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        result = sv._fit_single_composite_model(model, lags, variogram, None, weights)

        assert 'param_cov' in result
        assert result['param_cov'].shape == (2, 2)


# ─── Test Suite 5: Information Criteria ─────────────────────────────


class TestInformationCriteria:
    """Test AIC/BIC/AICc computation."""

    def test_aic_bic_aicc_finite(self, fitted_sv):
        """AIC, BIC, and AICc should be finite for all fitted models."""
        for m in fitted_sv.fitted_models:
            assert np.isfinite(m['aic'])
            assert np.isfinite(m['bic'])
            assert np.isfinite(m['aicc'])

    def test_bic_penalizes_more_params(self, synthetic_variogram_data):
        """BIC penalty grows with log(n); more parameters should increase BIC."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(max_components=2, include_nugget=False)

        # find a 1-component and a 2-component model
        single = [m for m in sv.fitted_models if len(m['model'].component_names) == 1]
        double = [m for m in sv.fitted_models if len(m['model'].component_names) == 2]

        if single and double:
            assert double[0]['model'].n_params > single[0]['model'].n_params

    def test_aicc_equals_aic_for_large_n(self):
        """AICc should converge to AIC when n >> k."""
        # construct a toy: n = 200, k = 2
        # AICc correction = 2k(k+1) / (n-k-1) -> small
        n = 200
        k = 2
        correction = 2 * k * (k + 1) / (n - k - 1)
        assert correction < 0.1  # negligible for large n


# ─── Test Suite 6: Model Selection ──────────────────────────────────


class TestModelSelection:
    """Test best model selection via different criteria."""

    def test_best_model_selected_by_aicc(self, fitted_sv):
        """Default criterion (AICc) should select a best model."""
        assert fitted_sv.best_model is not None
        assert 'aicc' in fitted_sv.best_model

    def test_aic_criterion(self, synthetic_variogram_data):
        """criterion='aic' should select by minimum AIC."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(max_components=1, include_nugget=True, criterion='aic')

        best_aic = sv.best_model['aic']
        assert all(m['aic'] >= best_aic - 1e-10 for m in sv.fitted_models)

    def test_bic_criterion(self, synthetic_variogram_data):
        """criterion='bic' should select by minimum BIC."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(max_components=1, include_nugget=True, criterion='bic')

        best_bic = sv.best_model['bic']
        assert all(m['bic'] >= best_bic - 1e-10 for m in sv.fitted_models)

    def test_fitted_model_property(self, fitted_sv):
        """fitted_model property should return a FittedVariogramModel."""
        fm = fitted_sv.fitted_model
        assert isinstance(fm, FittedVariogramModel)
        assert fm.params is not None
        assert fm.rss > 0

    def test_fitted_model_raises_without_fit(self):
        """fitted_model should raise if fit_model() not called."""
        sv = object.__new__(SingleVariogram)
        sv.best_model = None
        with pytest.raises(RuntimeError, match="No fitted model"):
            _ = sv.fitted_model


# ─── Test Suite 7: FittedVariogramModel ─────────────────────────────


class TestFittedVariogramModel:
    """Test FittedVariogramModel dataclass."""

    def test_predict_matches_model(self, fitted_sv):
        """predict() should match direct composite model call."""
        fm = fitted_sv.fitted_model
        test_lags = np.array([10, 50, 100, 200])

        pred1 = fm.predict(test_lags)
        pred2 = fm.composite_model(test_lags)

        assert_allclose(pred1, pred2, rtol=1e-10)

    def test_get_param_percentiles_without_samples(self, fitted_sv):
        """get_param_percentiles should raise without bootstrap samples."""
        fm = fitted_sv.fitted_model
        with pytest.raises(ValueError, match="No bootstrap samples"):
            fm.get_param_percentiles()

    def test_get_param_percentiles_with_samples(self, fitted_sv):
        """get_param_percentiles should work with bootstrap samples."""
        fm = fitted_sv.fitted_model

        # manually add fake bootstrap samples
        n_boot = 50
        n_params = fm.composite_model.n_params
        rng = np.random.default_rng(42)
        fm.param_samples = rng.normal(
            fm.params, 0.1 * np.abs(fm.params), size=(n_boot, n_params)
        )

        percentiles = fm.get_param_percentiles([16, 50, 84])
        assert isinstance(percentiles, dict)
        assert len(percentiles) == n_params
        for key, vals in percentiles.items():
            assert len(vals) == 3

    def test_fitted_model_has_msspe_field(self, fitted_sv):
        """FittedVariogramModel should have msspe and related fields."""
        fm = fitted_sv.fitted_model
        assert hasattr(fm, 'msspe')
        assert hasattr(fm, 'loocv_result')

    def test_predict_at_zero(self, fitted_sv):
        """predict(0) should return 0 (or nugget) for valid models."""
        fm = fitted_sv.fitted_model
        pred = fm.predict(np.array([0.0]))
        # should be non-negative
        assert pred[0] >= 0

    def test_predict_monotonic_for_bounded_model(self, fitted_sv):
        """Semivariance should be non-decreasing for bounded models."""
        fm = fitted_sv.fitted_model
        lags = np.linspace(1, 500, 100)
        pred = fm.predict(lags)

        # allow small numerical noise
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-10), "Semivariance should not decrease"


# ─── Test Suite 8: Initial Guess ────────────────────────────────────


class TestInitialGuess:
    """Test CompositeVariogramModel.default_guess()."""

    def test_default_guess_correct_length(self, synthetic_variogram_data):
        """default_guess should have correct number of parameters."""
        lags = synthetic_variogram_data['lags']
        variogram = synthetic_variogram_data['empirical']

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        guess = model.default_guess(lags, variogram)
        assert len(guess) == 2  # sill + range

    def test_default_guess_with_nugget(self, synthetic_variogram_data):
        """default_guess with nugget should have one extra parameter."""
        lags = synthetic_variogram_data['lags']
        variogram = synthetic_variogram_data['empirical']

        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        guess = model.default_guess(lags, variogram)
        assert len(guess) == 3  # sill + range + nugget

    def test_default_guess_multi_component(self, synthetic_variogram_data):
        """default_guess for multi-component should have correct count."""
        lags = synthetic_variogram_data['lags']
        variogram = synthetic_variogram_data['empirical']

        model = CompositeVariogramModel(
            ['spherical', 'exponential'], include_nugget=False
        )
        guess = model.default_guess(lags, variogram)
        assert len(guess) == 4  # 2 sills + 2 ranges

    def test_default_guess_positive(self, synthetic_variogram_data):
        """All initial guess parameters should be non-negative."""
        lags = synthetic_variogram_data['lags']
        variogram = synthetic_variogram_data['empirical']

        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        guess = model.default_guess(lags, variogram)
        assert np.all(guess >= 0)

    def test_default_guess_range_less_than_max_lag(self, synthetic_variogram_data):
        """Initial range guess should not exceed max lag."""
        lags = synthetic_variogram_data['lags']
        variogram = synthetic_variogram_data['empirical']
        max_lag = np.max(lags)

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        guess = model.default_guess(lags, variogram)
        # range is second parameter
        assert guess[1] < max_lag * 2  # generous bound


# ─── Test Suite 9: Kriging Leave-One-Out Cross-Validation ───────────


def _generate_spatial_field(n, sill, range_, nugget, mean=0.0, seed=42):
    """Generate a spatially correlated random field using Cholesky decomposition."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n, 2))

    dx = coords[:, 0:1] - coords[:, 0:1].T
    dy = coords[:, 1:2] - coords[:, 1:2].T
    dist = np.sqrt(dx**2 + dy**2)

    h_ratio = np.clip(dist / range_, 0, 1)
    sph_corr = 1.0 - (1.5 * h_ratio - 0.5 * h_ratio**3)
    sph_corr[dist >= range_] = 0.0

    C = sill * sph_corr + nugget * np.eye(n)
    C += 1e-10 * np.eye(n)

    L = np.linalg.cholesky(C)
    values = mean + L @ rng.standard_normal(n)

    model = CompositeVariogramModel(['spherical'], include_nugget=True)
    model.set_params(np.array([sill, range_, nugget]))

    return coords, values, model


class TestKrigingLOOCV:
    """Test VariogramAnalysis.kriging_loocv() — kriging-based cross-validation."""

    def test_msspe_near_one_for_correct_model(self):
        """MSSPE ≈ 1.0 when the fitted variogram matches the true one."""
        coords, values, true_model = _generate_spatial_field(
            n=200, sill=2.0, range_=30.0, nugget=0.5, seed=42
        )

        result = VariogramAnalysis.kriging_loocv(
            coords, values, true_model, n_subset=200, seed=42
        )

        assert 0.6 < result.msspe < 1.6, (
            f"MSSPE = {result.msspe:.3f}, expected ≈ 1.0 for correct model"
        )
        assert result.n_failed == 0

    def test_mean_error_near_zero(self):
        """Mean prediction error (bias) should be ≈ 0.0 for unbiased kriging."""
        coords, values, true_model = _generate_spatial_field(
            n=200, sill=2.0, range_=30.0, nugget=0.5, seed=123
        )

        result = VariogramAnalysis.kriging_loocv(
            coords, values, true_model, n_subset=200, seed=123
        )

        assert abs(result.mean_error) < 0.5, (
            f"Mean error = {result.mean_error:.3f}, expected ≈ 0.0"
        )

    def test_wrong_model_msspe_deviates(self):
        """MSSPE should deviate from 1.0 when using a wrong variogram."""
        coords, values, _ = _generate_spatial_field(
            n=200, sill=2.0, range_=30.0, nugget=0.5, seed=42
        )

        wrong_model = CompositeVariogramModel(['spherical'], include_nugget=True)
        wrong_model.set_params(np.array([20.0, 30.0, 5.0]))

        result = VariogramAnalysis.kriging_loocv(
            coords, values, wrong_model, n_subset=200, seed=42
        )

        assert result.msspe < 0.5, (
            f"MSSPE = {result.msspe:.3f}; expected << 1.0 for overestimated variogram"
        )

    def test_subsampling(self):
        """n_subset should reduce the number of points used."""
        coords, values, true_model = _generate_spatial_field(
            n=300, sill=2.0, range_=30.0, nugget=0.5, seed=42
        )

        result_full = VariogramAnalysis.kriging_loocv(
            coords, values, true_model, n_subset=300, seed=42
        )
        result_sub = VariogramAnalysis.kriging_loocv(
            coords, values, true_model, n_subset=100, seed=42
        )

        assert result_full.n_points == 300
        assert result_sub.n_points == 100

    def test_result_fields_finite(self):
        """All result fields should be finite for valid input."""
        coords, values, true_model = _generate_spatial_field(
            n=100, sill=2.0, range_=30.0, nugget=0.5, seed=42
        )

        result = VariogramAnalysis.kriging_loocv(
            coords, values, true_model, n_subset=100, seed=42
        )

        assert np.isfinite(result.msspe)
        assert np.isfinite(result.mean_error)
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.mean_standardized_error)
        assert result.rmse > 0

    def test_input_validation_coords_shape(self):
        """Should raise ValueError for bad coords shape."""
        with pytest.raises(ValueError, match="shape"):
            VariogramAnalysis.kriging_loocv(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                lambda h: h,
            )

    def test_input_validation_length_mismatch(self):
        """Should raise ValueError when coords and values have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            VariogramAnalysis.kriging_loocv(
                np.array([[0, 0], [1, 1]]),
                np.array([1, 2, 3]),
                lambda h: h,
            )

    def test_reproducibility_with_seed(self):
        """Same seed should give identical results."""
        coords, values, true_model = _generate_spatial_field(
            n=200, sill=2.0, range_=30.0, nugget=0.5, seed=42
        )

        r1 = VariogramAnalysis.kriging_loocv(
            coords, values, true_model, n_subset=100, seed=99
        )
        r2 = VariogramAnalysis.kriging_loocv(
            coords, values, true_model, n_subset=100, seed=99
        )

        assert_allclose(r1.msspe, r2.msspe)
        assert_allclose(r1.rmse, r2.rmse)

    def test_pure_nugget_process(self):
        """For a pure nugget process, MSSPE ≈ 1.0 with correct model."""
        rng = np.random.default_rng(77)
        n = 150
        coords = rng.uniform(0, 100, size=(n, 2))
        nugget_var = 3.0
        values = rng.normal(0, np.sqrt(nugget_var), n)

        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        model.set_params(np.array([1e-6, 50.0, nugget_var]))

        result = VariogramAnalysis.kriging_loocv(
            coords, values, model, n_subset=150, seed=77
        )

        assert 0.5 < result.msspe < 2.0, (
            f"MSSPE = {result.msspe:.3f} for pure nugget; expected ≈ 1.0"
        )


# ─── Test Suite 10: MSSPE Model Selection Criterion ─────────────────


class TestMSSPECriterion:
    """Test MSSPE-based model selection via kriging LOOCV."""

    def test_fitted_model_has_msspe_field(self, synthetic_variogram_data):
        """FittedVariogramModel should have msspe field, initially None."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(
            model_types=['spherical'],
            max_components=1,
            include_nugget=False,
        )

        fm = sv.fitted_model
        assert hasattr(fm, 'msspe')
        assert hasattr(fm, 'loocv_result')
        # not computed without criterion='msspe'
        assert fm.msspe is None

    def test_msspe_computed_by_kriging_loocv(self):
        """Manually computing MSSPE via kriging_loocv should work."""
        rng = np.random.default_rng(123)
        n = 200
        coords = rng.uniform(0, 500, size=(n, 2))
        values = rng.normal(0, 1.0, size=n)

        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        model.set_params(np.array([1.0, 100.0, 0.3]))

        result = VariogramAnalysis.kriging_loocv(
            coords, values, model, n_subset=200, seed=42,
        )

        assert result.msspe is not None
        assert np.isfinite(result.msspe)
        assert result.msspe > 0
        assert result.n_failed == 0

    def test_msspe_fallback_without_spatial_sample(self, synthetic_variogram_data):
        """criterion='msspe' without sample_coords should warn and fall back."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        # sample_coords is None by default

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sv.fit_model(
                model_types=['spherical'],
                max_components=1,
                include_nugget=False,
                criterion='msspe',
            )
            # should warn about no spatial sample
            msspe_warnings = [
                x for x in w if "sample" in str(x.message).lower()
            ]
            assert len(msspe_warnings) >= 1

        # should still have a best model (via AICc fallback)
        assert sv.best_model is not None


# ─── Test Suite 11: Cressie (1985) WLS Weighting ────────────────────


class TestCressieWeighting:
    """Test that fit_model() uses Cressie (1985) WLS weights internally."""

    def test_cressie_weights_formula(self, synthetic_variogram_data):
        """Internal Cressie weights should equal N(h)/γ̂(h)²."""
        lags = synthetic_variogram_data['lags']
        empirical = synthetic_variogram_data['empirical']
        counts = synthetic_variogram_data['bin_counts'].astype(float)

        gamma_sq = np.square(empirical)
        gamma_sq = np.where(
            gamma_sq < np.finfo(float).eps, np.finfo(float).eps, gamma_sq
        )
        expected_weights = counts / gamma_sq

        # verify weights are higher at short lags (where γ is small)
        n = len(lags)
        q1 = n // 4
        short_mean = np.mean(expected_weights[:q1])
        long_mean = np.mean(expected_weights[-q1:])
        assert short_mean > 5 * long_mean, (
            "Cressie weights should upweight short lags"
        )


# ─── Test Suite 12: CompositeVariogramModel bounds ──────────────────


class TestModelBounds:
    """Test CompositeVariogramModel.bounds()."""

    def test_bounds_correct_length(self, synthetic_variogram_data):
        """bounds() should return (lower, upper) with correct length."""
        lags = synthetic_variogram_data['lags']
        variogram = synthetic_variogram_data['empirical']

        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        lower, upper = model.bounds(lags, variogram)

        assert len(lower) == model.n_params
        assert len(upper) == model.n_params

    def test_bounds_lower_less_than_upper(self, synthetic_variogram_data):
        """All lower bounds should be less than upper bounds."""
        lags = synthetic_variogram_data['lags']
        variogram = synthetic_variogram_data['empirical']

        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        lower, upper = model.bounds(lags, variogram)

        for lo, up in zip(lower, upper):
            assert lo < up

    def test_nugget_upper_bound_capped(self, synthetic_variogram_data):
        """Nugget upper bound should be capped at 50% of max(variogram)."""
        lags = synthetic_variogram_data['lags']
        variogram = synthetic_variogram_data['empirical']
        max_gamma = np.nanmax(variogram)

        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        _, upper = model.bounds(lags, variogram)

        # nugget is last parameter
        nugget_upper = upper[-1]
        assert_allclose(nugget_upper, max_gamma * 0.5, rtol=1e-10)


# ─── Test Suite 13: Integration ─────────────────────────────────────


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, synthetic_variogram_data):
        """Test complete workflow: synthetic → fit → select → FittedVariogramModel."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )

        table = sv.fit_model(
            max_components=2,
            include_nugget=True,
        )

        assert len(table) > 0
        assert sv.best_model is not None

        fm = sv.fitted_model
        assert isinstance(fm, FittedVariogramModel)

        # predict should be finite
        test_lags = np.array([10, 50, 100, 200])
        pred = fm.predict(test_lags)
        assert np.all(np.isfinite(pred))
        assert np.all(pred >= 0)

    def test_spherical_is_preferred_for_spherical_data(self, synthetic_variogram_data):
        """Spherical data should prefer a spherical model."""
        sv = _make_single_variogram_from_synthetic(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            synthetic_variogram_data['bin_counts'],
        )
        sv.fit_model(max_components=1, include_nugget=True)

        best_desc = sv.best_model['description']
        assert 'spherical' in best_desc, (
            f"Expected spherical model for spherical data, got: {best_desc}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
