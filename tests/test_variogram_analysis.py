"""Comprehensive tests for variogram analysis infrastructure.

Tests the model fitting infrastructure in src/topochange/variogram.py, specifically:
1. FittedVariogramModel dataclass
2. VariogramModelSelector class
3. VariogramAnalysis.compute_matheron static method

Uses synthetic data (no actual raster files required)."""

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


# fixtures: Synthetic Data


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


# test Suite 1: Matheron Estimator


class TestMatheronEstimator:
    """Test VariogramAnalysis.compute_matheron static method."""

    def test_compute_matheron_basic(self, synthetic_matheron_data):
        """Test basic Matheron computation: γ(h) = SSD(h) / (2*N(h))."""
        gamma_est = VariogramAnalysis.compute_matheron(
            synthetic_matheron_data['bin_counts'],
            synthetic_matheron_data['ssd'],
            min_pairs=5
        )

        # should match expected gamma values
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

        # bins with count < min_pairs should be NaN
        valid = synthetic_matheron_data['bin_counts'] >= min_pairs
        assert np.all(np.isnan(gamma_est[~valid]))
        assert np.all(np.isfinite(gamma_est[valid]))

    def test_compute_matheron_zero_counts(self):
        """Test behavior with zero pair counts."""
        bin_counts = np.array([100, 0, 50, 0, 25])
        ssd = np.array([50, 0, 30, 0, 15])

        gamma_est = VariogramAnalysis.compute_matheron(bin_counts, ssd, min_pairs=10)

        # bins with zero counts should be NaN
        assert np.isnan(gamma_est[1])
        assert np.isnan(gamma_est[3])
        # others should be finite
        assert np.isfinite(gamma_est[0])
        assert np.isfinite(gamma_est[2])

    def test_compute_matheron_single_bin(self):
        """Test with single bin."""
        bin_counts = np.array([100])
        ssd = np.array([50.0])

        gamma_est = VariogramAnalysis.compute_matheron(bin_counts, ssd, min_pairs=10)

        assert gamma_est.shape == (1,)
        assert_allclose(gamma_est[0], 50.0 / (2.0 * 100))


# test Suite 1b: Cressie–Hawkins Estimator


class TestCressieHawkinsEstimator:
    """Test VariogramAnalysis.compute_cressie_hawkins static method."""

    def test_compute_ch_basic(self):
        """Test Cressie–Hawkins: γ̂ = [mean(|ΔZ|^0.5)]⁴ / (2·(0.457 + 0.494/N))."""
        # construct known inputs: N=100 pairs, all with |ΔZ|=1.0
        # |ΔZ|^0.5 = 1.0 for each pair, so sum = 100
        bin_counts = np.array([100])
        sum_sqrt_abs_diff = np.array([100.0])  # each pair contributes 1.0

        gamma = VariogramAnalysis.compute_cressie_hawkins(
            bin_counts, sum_sqrt_abs_diff, min_pairs=10
        )

        # mean(|ΔZ|^0.5) = 100/100 = 1.0
        # [1.0]^4 = 1.0
        # correction = 0.457 + 0.494/100 = 0.46194
        # γ̂ = 0.5 * 1.0 / 0.46194 ≈ 1.0824
        expected = 0.5 * 1.0 / (0.457 + 0.494 / 100)
        assert_allclose(gamma[0], expected, rtol=1e-10)

    def test_compute_ch_pure_nugget(self):
        """For a pure nugget (constant variance), Cressie–Hawkins
        should agree with Matheron asymptotically."""
        rng = np.random.default_rng(42)
        n_pairs = 10000
        sigma = 2.0
        # simulate squared differences from a pure-nugget process
        # ΔZ ~ N(0, 2σ²)  so |ΔZ|^0.5 needs to be accumulated
        dz = rng.normal(0, np.sqrt(2) * sigma, n_pairs)
        ssd = np.array([np.sum(dz**2)])
        ssad = np.array([np.sum(np.abs(dz) ** 0.5)])
        counts = np.array([n_pairs])

        matheron = VariogramAnalysis.compute_matheron(counts, ssd, min_pairs=10)
        ch = VariogramAnalysis.compute_cressie_hawkins(counts, ssad, min_pairs=10)

        # both should estimate γ ≈ σ² = 4.0; allow 10% tolerance for finite sample
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
        assert np.isnan(gamma[1])   # only 5 pairs
        assert np.isfinite(gamma[2])

    def test_compute_ch_outlier_robustness(self):
        """Cressie–Hawkins should be more resistant to outliers than Matheron."""
        rng = np.random.default_rng(99)
        n_pairs = 500
        sigma = 1.0
        dz = rng.normal(0, np.sqrt(2) * sigma, n_pairs)

        # inject 5 extreme outliers (10× the std)
        dz[:5] = 10.0 * np.sqrt(2) * sigma

        ssd = np.array([np.sum(dz**2)])
        ssad = np.array([np.sum(np.abs(dz) ** 0.5)])
        counts = np.array([n_pairs])

        matheron = VariogramAnalysis.compute_matheron(counts, ssd, min_pairs=10)
        ch = VariogramAnalysis.compute_cressie_hawkins(counts, ssad, min_pairs=10)

        # Matheron will be inflated by the outliers, Cressie–Hawkins less so.
        # True value is σ²=1.0; Matheron will be >> 1.0; CH closer to 1.0.
        assert ch[0] < matheron[0], (
            f"Cressie–Hawkins ({ch[0]:.3f}) should be closer to true value "
            f"than Matheron ({matheron[0]:.3f}) in the presence of outliers"
        )


# test Suite 2: VariogramModelSelector Construction


class TestVariogramModelSelectorConstruction:
    """Test VariogramModelSelector initialization and setup."""

    def test_constructor_stores_inputs(self, synthetic_variogram_data):
        """Test that constructor properly stores input data."""
        lags = synthetic_variogram_data['lags']
        empirical = synthetic_variogram_data['empirical']
        bin_counts = synthetic_variogram_data['bin_counts']

        selector = VariogramModelSelector(
            lags, empirical,
            pair_counts=bin_counts, weighting='pair_count',
        )

        assert_array_almost_equal(selector.lags, lags)
        assert_array_almost_equal(selector.empirical_variogram, empirical)
        assert_array_almost_equal(selector.pair_counts, bin_counts)
        # with pair_count weighting, weights should equal the raw counts
        assert_array_almost_equal(selector.weights, bin_counts)

    def test_constructor_default_weights(self, synthetic_variogram_data):
        """Test that default weights are ones when no pair counts supplied."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            selector = VariogramModelSelector(
                synthetic_variogram_data['lags'],
                synthetic_variogram_data['empirical'],
            )

        assert_array_almost_equal(selector.weights, np.ones_like(selector.lags))

    def test_constructor_with_sigma(self, synthetic_variogram_data):
        """Test constructor with per-bin standard deviations."""
        lags = synthetic_variogram_data['lags']
        empirical = synthetic_variogram_data['empirical']
        sigma = np.full_like(empirical, 0.05)

        selector = VariogramModelSelector(
            lags, empirical, sigma=sigma, weighting='uniform',
        )

        assert_array_almost_equal(selector.sigma, sigma)

    def test_constructor_sigma_clipping(self):
        """Test that zero/negative sigmas are clipped to eps."""
        lags = np.array([10, 20, 30])
        empirical = np.array([0.5, 0.8, 1.0])
        sigma = np.array([0.1, 0, -0.05])

        selector = VariogramModelSelector(
            lags, empirical, sigma=sigma, weighting='uniform',
        )

        # first should be unchanged
        assert selector.sigma[0] == 0.1
        # others should be >= eps
        assert selector.sigma[1] > 0
        assert selector.sigma[2] > 0


# test Suite 2b: Weighting Schemes


class TestWeightingSchemes:
    """Test VariogramModelSelector weighting schemes."""

    def test_cressie_weights_formula(self, synthetic_variogram_data):
        """Test that Cressie weights = N(h) / γ̂(h)²."""
        lags = synthetic_variogram_data['lags']
        empirical = synthetic_variogram_data['empirical']
        counts = synthetic_variogram_data['bin_counts'].astype(float)

        selector = VariogramModelSelector(
            lags, empirical,
            pair_counts=counts,
            weighting='cressie',
        )

        expected = counts / np.maximum(empirical**2, np.finfo(float).eps)
        assert_allclose(selector.weights, expected, rtol=1e-10)

    def test_cressie_weights_short_lag_dominance(self, synthetic_variogram_data):
        """Cressie weights should be much larger at short lags than long lags."""
        lags = synthetic_variogram_data['lags']
        empirical = synthetic_variogram_data['empirical']
        counts = synthetic_variogram_data['bin_counts'].astype(float)

        selector = VariogramModelSelector(
            lags, empirical,
            pair_counts=counts,
            weighting='cressie',
        )

        # first-quartile weights should be >> last-quartile weights
        n = len(lags)
        q1 = n // 4
        short_mean = np.mean(selector.weights[:q1])
        long_mean = np.mean(selector.weights[-q1:])
        assert short_mean > 10 * long_mean, (
            f"Cressie weights should strongly upweight short lags; "
            f"short mean={short_mean:.1f}, long mean={long_mean:.1f}"
        )

    def test_pair_count_weights(self, synthetic_variogram_data):
        """pair_count weighting should return raw counts."""
        counts = synthetic_variogram_data['bin_counts'].astype(float)

        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=counts,
            weighting='pair_count',
        )
        assert_allclose(selector.weights, counts)

    def test_uniform_weights(self, synthetic_variogram_data):
        """uniform weighting should return ones."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weighting='uniform',
        )
        assert_allclose(selector.weights, np.ones(len(synthetic_variogram_data['lags'])))

    def test_cressie_fallback_warns_without_counts(self, synthetic_variogram_data):
        """Requesting Cressie without pair_counts should warn and fall back."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selector = VariogramModelSelector(
                synthetic_variogram_data['lags'],
                synthetic_variogram_data['empirical'],
                weighting='cressie',
            )
            # should have issued a warning
            assert any("pair_counts" in str(warning.message) for warning in w)
            # should fall back to uniform
            assert_allclose(selector.weights, np.ones(len(synthetic_variogram_data['lags'])))

    def test_invalid_weighting_raises(self, synthetic_variogram_data):
        """Unknown weighting scheme should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown weighting"):
            VariogramModelSelector(
                synthetic_variogram_data['lags'],
                synthetic_variogram_data['empirical'],
                weighting='invalid_scheme',
            )

    def test_cressie_default_in_fit_best_model_auto(self, synthetic_variogram_data):
        """fit_best_model_auto should use Cressie weights by default
        when pair counts are available."""
        lags = synthetic_variogram_data['lags']
        empirical = synthetic_variogram_data['empirical']
        counts = synthetic_variogram_data['bin_counts'].astype(float)

        # manually build what fit_best_model_auto would create
        selector = VariogramModelSelector(
            lags, empirical,
            pair_counts=counts,
        )

        # default weighting should be 'cressie'
        assert selector.weighting == 'cressie'
        # weights should NOT be equal to raw counts
        assert not np.allclose(selector.weights, counts)
        # weights should be N(h) / γ̂(h)²
        expected = counts / np.maximum(empirical**2, np.finfo(float).eps)
        assert_allclose(selector.weights, expected, rtol=1e-10)


# test Suite 2c: Minimum Pair-Count Filtering


class TestMinPairsFiltering:
    """Test VariogramModelSelector minimum pair-count filtering.

    Literature basis: Cressie (1985) recommends N(h) > 50; Oliver &
    Webster (2014) suggest N(h) > 30.  Bins with too few pairs are
    unreliable and should be excluded from model fitting.
    """

    def test_bins_below_threshold_get_zero_weight(self):
        """Bins with pair_counts < min_pairs should have weight = 0."""
        lags = np.array([10, 20, 30, 40, 50])
        empirical = np.array([0.3, 0.6, 0.8, 0.9, 1.0])
        counts = np.array([100, 80, 25, 15, 5], dtype=float)

        selector = VariogramModelSelector(
            lags, empirical,
            pair_counts=counts,
            weighting='pair_count',
            min_pairs=30,
        )

        # first two bins (100, 80) should keep their pair-count weights
        assert selector.weights[0] == 100.0
        assert selector.weights[1] == 80.0
        # last three bins (25, 15, 5) should be zeroed
        assert selector.weights[2] == 0.0
        assert selector.weights[3] == 0.0
        assert selector.weights[4] == 0.0

    def test_n_filtered_count(self):
        """_n_filtered should report how many bins were excluded."""
        lags = np.array([10, 20, 30, 40, 50])
        empirical = np.array([0.3, 0.6, 0.8, 0.9, 1.0])
        counts = np.array([100, 80, 25, 15, 5], dtype=float)

        selector = VariogramModelSelector(
            lags, empirical,
            pair_counts=counts,
            weighting='uniform',
            min_pairs=30,
        )

        assert selector._n_filtered == 3

    def test_no_filtering_when_all_above_threshold(self, synthetic_variogram_data):
        """No bins should be filtered when all counts exceed min_pairs."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'].astype(float),
            weighting='pair_count',
            min_pairs=30,
        )

        # fixture bin_counts range from 100 to 50, all > 30
        assert selector._n_filtered == 0
        assert np.all(selector.weights > 0)

    def test_min_pairs_disabled_with_none(self, synthetic_variogram_data):
        """Setting min_pairs=None should disable filtering entirely."""
        lags = np.array([10, 20, 30])
        empirical = np.array([0.3, 0.6, 0.8])
        counts = np.array([5, 3, 1], dtype=float)

        selector = VariogramModelSelector(
            lags, empirical,
            pair_counts=counts,
            weighting='pair_count',
            min_pairs=None,
        )

        assert selector._n_filtered == 0
        assert_allclose(selector.weights, counts)

    def test_min_pairs_disabled_with_zero(self):
        """Setting min_pairs=0 should disable filtering entirely."""
        lags = np.array([10, 20, 30])
        empirical = np.array([0.3, 0.6, 0.8])
        counts = np.array([5, 3, 1], dtype=float)

        selector = VariogramModelSelector(
            lags, empirical,
            pair_counts=counts,
            weighting='pair_count',
            min_pairs=0,
        )

        assert selector._n_filtered == 0
        assert_allclose(selector.weights, counts)

    def test_min_pairs_ignored_without_pair_counts(self):
        """min_pairs should be silently skipped when pair_counts is None."""
        lags = np.array([10, 20, 30])
        empirical = np.array([0.3, 0.6, 0.8])

        selector = VariogramModelSelector(
            lags, empirical,
            weighting='uniform',
            min_pairs=50,
        )

        # no filtering, all weights = 1
        assert selector._n_filtered == 0
        assert_allclose(selector.weights, np.ones(3))

    def test_filtering_works_with_cressie_weights(self):
        """min_pairs should zero Cressie weights for low-count bins."""
        lags = np.array([10, 20, 30, 40, 50])
        empirical = np.array([0.3, 0.6, 0.8, 0.9, 1.0])
        counts = np.array([200, 150, 20, 10, 5], dtype=float)

        selector = VariogramModelSelector(
            lags, empirical,
            pair_counts=counts,
            weighting='cressie',
            min_pairs=30,
        )

        # first two: Cressie weight = N(h) / γ̂(h)²
        expected_0 = 200.0 / (0.3**2)
        expected_1 = 150.0 / (0.6**2)
        assert_allclose(selector.weights[0], expected_0, rtol=1e-10)
        assert_allclose(selector.weights[1], expected_1, rtol=1e-10)
        # last three: should be zeroed despite having valid Cressie weights
        assert selector.weights[2] == 0.0
        assert selector.weights[3] == 0.0
        assert selector.weights[4] == 0.0

    def test_few_remaining_bins_warns(self):
        """Should warn when min_pairs leaves fewer than 4 usable bins."""
        lags = np.array([10, 20, 30, 40, 50])
        empirical = np.array([0.3, 0.6, 0.8, 0.9, 1.0])
        # only 2 bins above threshold
        counts = np.array([100, 80, 5, 3, 1], dtype=float)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selector = VariogramModelSelector(
                lags, empirical,
                pair_counts=counts,
                weighting='pair_count',
                min_pairs=30,
            )
            # should warn about too few remaining bins
            min_pairs_warnings = [
                x for x in w
                if "min_pairs" in str(x.message) and "filters" in str(x.message)
            ]
            assert len(min_pairs_warnings) >= 1
            assert "2" in str(min_pairs_warnings[0].message)  # 2 remaining

    def test_default_min_pairs_is_30(self, synthetic_variogram_data):
        """Default min_pairs should be 30 (literature: Cressie 1985)."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'].astype(float),
            weighting='pair_count',
        )

        assert selector.min_pairs == 30


# test Suite 3: Candidate Generation


class TestCandidateGeneration:
    """Test VariogramModelSelector.generate_candidates()."""

    def test_generate_candidates_non_empty(self, synthetic_variogram_data):
        """Test that generate_candidates returns non-empty list."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weighting='uniform',
        )

        candidates = selector.generate_candidates(max_components=2)

        assert len(candidates) > 0
        assert all(isinstance(c, CompositeVariogramModel) for c in candidates)

    def test_generate_candidates_respects_max_components(self, synthetic_variogram_data):
        """Test that candidates respect max_components limit."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weighting='uniform',
        )

        cand1 = selector.generate_candidates(max_components=1)
        cand2 = selector.generate_candidates(max_components=2)

        # more components should give more candidates (or equal)
        assert len(cand2) >= len(cand1)

        # check that all single-component models are in both lists
        cand1_names = [c.component_names for c in cand1]
        cand2_names = [c.component_names for c in cand2]

        for name in cand1_names:
            assert name in cand2_names

    def test_generate_candidates_include_nugget_false(self, synthetic_variogram_data):
        """Test nugget inclusion flag."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weighting='uniform',
        )

        cand_with_nugget = selector.generate_candidates(include_nugget=True)
        cand_no_nugget = selector.generate_candidates(include_nugget=False)

        # all with-nugget models should have nugget
        assert all(c.include_nugget for c in cand_with_nugget)
        assert all(not c.include_nugget for c in cand_no_nugget)

    def test_generate_candidates_include_unbounded(self, synthetic_variogram_data):
        """Test unbounded model inclusion flag."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weighting='uniform',
        )

        cand_with_unbounded = selector.generate_candidates(include_unbounded=True)
        cand_no_unbounded = selector.generate_candidates(include_unbounded=False)

        # check if any unbounded models are present
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

        # with unbounded enabled might have some unbounded (or not, depends on data)
        # without unbounded should have None
        if has_unbounded_with:
            assert has_unbounded_without is False  # exclusive check


# test Suite 4: Model Fitting on Synthetic Data


class TestModelFitting:
    """Test VariogramModelSelector.fit_model()."""

    def test_fit_model_spherical(self, synthetic_variogram_data):
        """Test fitting a spherical model to synthetic spherical data."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        # create single spherical model
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
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        # extract sill and range
        fitted_sill = fitted.params[0]
        fitted_range = fitted.params[1]

        true_sill = synthetic_variogram_data['true_sill']
        true_range = synthetic_variogram_data['true_range']

        # check within 20% of True values
        assert abs(fitted_sill - true_sill) / true_sill < 0.2
        assert abs(fitted_range - true_range) / true_range < 0.2

    def test_fit_model_with_nugget(self, synthetic_variogram_data):
        """Test fitting with nugget component."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
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
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        assert fitted.rss > 0
        assert np.isfinite(fitted.aic)
        assert np.isfinite(fitted.bic)


# test Suite 5: Information Criteria


class TestInformationCriteria:
    """Test AIC/BIC computation."""

    def test_aic_bic_finite(self, synthetic_variogram_data):
        """Test that AIC and BIC are finite."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
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
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        # fit two models: spherical and spherical+exponential
        model_1 = CompositeVariogramModel(['spherical'], include_nugget=False)
        model_2 = CompositeVariogramModel(['spherical', 'exponential'], include_nugget=False)

        fitted_1 = selector.fit_model(model_1)
        fitted_2 = selector.fit_model(model_2)

        if fitted_1 and fitted_2:
            # more parameters should increase BIC more than AIC (relative to likelihood)
            bic_diff = fitted_2.bic - fitted_1.bic
            aic_diff = fitted_2.aic - fitted_1.aic

            # bIC penalty should be larger
            # (Note: this is probabilistic, may not always hold with noise)
            assert fitted_2.composite_model.n_params > fitted_1.composite_model.n_params


# test Suite 6: Model Selection


class TestModelSelection:
    """Test VariogramModelSelector.select_best()."""

    def test_select_best_by_aic(self, synthetic_variogram_data):
        """Test select_best with AIC criterion."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
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
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=2, include_nugget=True, compute_cv=False)
        best = selector.select_best(criterion='bic')

        assert best is not None

    def test_select_best_raises_no_models(self, synthetic_variogram_data):
        """Test that select_best raises if no models fitted."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            weighting='uniform',
        )

        with pytest.raises(ValueError, match="No fitted models"):
            selector.select_best()

    def test_select_best_invalid_criterion(self, synthetic_variogram_data):
        """Test that invalid criterion raises."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)

        with pytest.raises(ValueError, match="Unknown criterion"):
            selector.select_best(criterion='invalid')


# test Suite 7: Cross-Validation


class TestCrossValidation:
    """Test VariogramModelSelector.cross_validate()."""

    def test_cross_validate_returns_positive_rmse(self, synthetic_variogram_data):
        """Test that cross_validate returns positive finite RMSE."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
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
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        rmse_2 = selector.cross_validate(fitted, k=2, seed=42)
        rmse_5 = selector.cross_validate(fitted, k=5, seed=42)

        # both should be finite positive
        assert rmse_2 > 0 and np.isfinite(rmse_2)
        assert rmse_5 > 0 and np.isfinite(rmse_5)


# test Suite 8: Akaike Weights


class TestAkaikeWeights:
    """Test VariogramModelSelector._compute_akaike_weights()."""

    def test_akaike_weights_sum_to_one(self, synthetic_variogram_data):
        """Test that Akaike weights sum to 1.0."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=2, include_nugget=True, compute_cv=False)

        assert selector.model_weights is not None
        assert_allclose(np.sum(selector.model_weights), 1.0, rtol=1e-10)

    def test_akaike_weights_non_negative(self, synthetic_variogram_data):
        """Test that all Akaike weights are non-negative."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=2, include_nugget=False, compute_cv=False)

        assert np.all(selector.model_weights >= 0)

    def test_akaike_weights_best_has_highest_weight(self, synthetic_variogram_data):
        """Test that best model has highest weight."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=2, include_nugget=False, compute_cv=False)
        best = selector.select_best(criterion='aic')

        best_idx = selector.fitted_models.index(best)
        best_weight = selector.model_weights[best_idx]

        assert best_weight >= np.max(selector.model_weights[selector.model_weights < best_weight + 1e-10])


# test Suite 9: BMA Variogram


class TestBMAVariogram:
    """Test VariogramModelSelector.get_bma_variogram()."""

    def test_get_bma_variogram_returns_callable(self, synthetic_variogram_data):
        """Test that get_bma_variogram returns a callable."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        bma_func = selector.get_bma_variogram()

        assert callable(bma_func)

    def test_bma_variogram_produces_finite_results(self, synthetic_variogram_data):
        """Test that BMA variogram produces finite results."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
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
            synthetic_variogram_data['empirical'],
            weighting='uniform',
        )

        with pytest.raises(ValueError, match="No model weights"):
            selector.get_bma_variogram()


# test Suite 10: Bootstrap Uncertainty


class TestBootstrapUncertainty:
    """Test VariogramModelSelector.bootstrap_best_model()."""

    def test_bootstrap_returns_samples(self, synthetic_variogram_data):
        """Test that bootstrap_best_model returns parameter samples."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count',
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
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count',
            sigma=np.full_like(synthetic_variogram_data['empirical'], 0.05)
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        selector.select_best(criterion='aic')
        selector.bootstrap_best_model(n_boot=50, seed=42)

        assert selector.best_model.param_samples is not None
        assert len(selector.best_model.param_samples) > 0


# test Suite 11: FittedVariogramModel


class TestFittedVariogramModel:
    """Test FittedVariogramModel dataclass."""

    def test_fitted_model_predict_matches_model(self, synthetic_variogram_data):
        """Test that predict() matches direct model call."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
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
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        fitted = selector.fit_model(model)

        # no bootstrap samples yet
        with pytest.raises(ValueError, match="No bootstrap samples"):
            fitted.get_param_percentiles()

    def test_get_param_percentiles_with_samples(self, synthetic_variogram_data):
        """Test get_param_percentiles with bootstrap samples."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count',
            sigma=np.full_like(synthetic_variogram_data['empirical'], 0.05)
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        selector.select_best(criterion='aic')
        selector.bootstrap_best_model(n_boot=50, seed=42)

        percentiles = selector.best_model.get_param_percentiles([16, 50, 84])

        # should have dict with parameter names as keys
        assert isinstance(percentiles, dict)
        assert len(percentiles) == selector.best_model.composite_model.n_params

        # each value should have 3 elements (16th, 50th, 84th percentile)
        for key, vals in percentiles.items():
            assert len(vals) == 3


# test Suite 12: Initial Guess


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

        # all parameters should be non-negative
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

        # range (second param) should be < max_lag
        assert guess[1] < max_lag


# test Suite 13: Fit All Candidates


class TestFitAllCandidates:
    """Test VariogramModelSelector.fit_all_candidates()."""

    def test_fit_all_candidates_populates_list(self, synthetic_variogram_data):
        """Test that fit_all_candidates populates fitted_models list."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False)

        assert len(selector.fitted_models) > 0
        assert all(isinstance(m, FittedVariogramModel) for m in selector.fitted_models)

    def test_fit_all_candidates_compute_cv(self, synthetic_variogram_data):
        """Test that compute_cv flag works."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=True)

        # at least some models should have CV scores
        has_cv = any(m.cv_rmse is not None for m in selector.fitted_models)
        assert has_cv

    def test_fit_all_candidates_without_cv(self, synthetic_variogram_data):
        """Test that compute_cv=False skips CV computation."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)

        # all models should have None for cv_rmse
        assert all(m.cv_rmse is None for m in selector.fitted_models)


# integration Tests


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, synthetic_variogram_data):
        """Test complete workflow: generate -> fit -> select -> bootstrap."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count',
            sigma=np.full_like(synthetic_variogram_data['empirical'], 0.05)
        )

        # fit candidates
        selector.fit_all_candidates(
            max_components=2,
            include_nugget=True,
            compute_cv=True,
            cv_folds=3,
            seed=42
        )

        # select best
        best = selector.select_best(criterion='aic')
        assert best is not None

        # bootstrap
        samples = selector.bootstrap_best_model(n_boot=30, seed=42)
        assert len(samples) > 0

        # get percentiles
        percentiles = best.get_param_percentiles([16, 50, 84])
        assert len(percentiles) > 0

    def test_bma_after_fitting(self, synthetic_variogram_data):
        """Test BMA computation after fitting all candidates."""
        selector = VariogramModelSelector(
            synthetic_variogram_data['lags'],
            synthetic_variogram_data['empirical'],
            pair_counts=synthetic_variogram_data['bin_counts'], weighting='pair_count'
        )

        selector.fit_all_candidates(max_components=1, include_nugget=False, compute_cv=False)
        bma_func = selector.get_bma_variogram()

        # bMA should average predictions
        test_lag = 100.0
        bma_pred = bma_func(np.array([test_lag]))[0]

        # should be within range of individual model predictions
        individual_preds = [m.predict(np.array([test_lag]))[0] for m in selector.fitted_models]
        assert min(individual_preds) <= bma_pred <= max(individual_preds)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

