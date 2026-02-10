"""
Comprehensive test suite for CompositeVariogramModel class.

Tests the composite variogram model builder including:
- Single and multi-component model construction
- Parameter setting and validation
- Component evaluation
- Sill and stationarity calculations
- Variance decomposition
- Covariance functions
- Default parameter guesses and bounds
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from topochange.composite_variogram import CompositeVariogramModel
from topochange.variogram_models import (
    MODEL_REGISTRY,
    spherical,
    exponential,
    gaussian,
    matern,
    damped_hole_effect,
    power,
    linear,
    nugget as nugget_func,
)


class TestCompositeVariogramConstruction:
    """Test model construction with various component combinations."""

    def test_single_bounded_component_without_nugget(self):
        """Test creating a single spherical model without nugget."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)

        assert model.component_names == ['spherical']
        assert not model.include_nugget
        assert model.n_params == 2  # sill, range
        assert model.param_names == ['spherical_sill', 'spherical_range']
        assert model.is_stationary
        assert model.bounded_components == ['spherical']
        assert model.unbounded_components == []

    def test_single_bounded_component_with_nugget(self):
        """Test creating a single spherical model with nugget."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)

        assert model.n_params == 3  # sill, range, nugget
        assert model.param_names == ['spherical_sill', 'spherical_range', 'nugget']
        assert model.include_nugget
        assert model.is_stationary

    def test_multi_bounded_components_with_nugget(self):
        """Test creating a multi-component model (spherical + exponential + nugget)."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )

        assert model.n_params == 5  # sill1, range1, sill2, range2, nugget
        assert model.param_names == [
            'spherical_sill', 'spherical_range',
            'exponential_sill', 'exponential_range',
            'nugget'
        ]
        assert model.is_stationary
        assert len(model.bounded_components) == 2
        assert len(model.unbounded_components) == 0

    def test_duplicate_component_names(self):
        """Test creating model with duplicate component names."""
        model = CompositeVariogramModel(
            ['spherical', 'spherical'],
            include_nugget=True
        )

        # Duplicate names should be disambiguated with indices
        assert model.param_names == [
            'spherical_0_sill', 'spherical_0_range',
            'spherical_1_sill', 'spherical_1_range',
            'nugget'
        ]
        assert model.n_params == 5

    def test_mixed_bounded_unbounded(self):
        """Test creating model with both bounded and unbounded components."""
        model = CompositeVariogramModel(
            ['spherical', 'linear'],
            include_nugget=True
        )

        assert not model.is_stationary
        assert model.bounded_components == ['spherical']
        assert model.unbounded_components == ['linear']
        assert model.n_params == 4  # sill, range, slope, nugget

    def test_single_unbounded_component(self):
        """Test creating model with single unbounded component."""
        model = CompositeVariogramModel(
            ['power'],
            include_nugget=False
        )

        assert not model.is_stationary
        assert model.bounded_components == []
        assert model.unbounded_components == ['power']
        assert model.n_params == 2  # scale, exponent

    def test_matern_component(self):
        """Test creating model with Matérn component."""
        model = CompositeVariogramModel(['matern'], include_nugget=True)

        assert model.n_params == 4  # sill, range, nu, nugget
        assert 'matern_nu' in model.param_names
        assert model.is_stationary

    def test_damped_hole_effect_component(self):
        """Test creating model with damped hole-effect component."""
        model = CompositeVariogramModel(['damped_hole_effect'], include_nugget=True)

        assert model.n_params == 4  # sill, range, wavelength, nugget
        assert 'damped_hole_effect_wavelength' in model.param_names
        assert model.is_stationary

    def test_invalid_combination_multiple_unbounded(self):
        """Test that combining multiple unbounded models raises ValueError."""
        with pytest.raises(ValueError, match="Cannot combine multiple unbounded"):
            CompositeVariogramModel(['power', 'linear'], include_nugget=False)

    def test_invalid_model_name(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            CompositeVariogramModel(['invalid_model'], include_nugget=False)


class TestParameterSetting:
    """Test parameter setting and validation."""

    def test_set_params_correct_length(self):
        """Test setting parameters with correct number of values."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        params = np.array([0.5, 100.0, 0.1])

        model.set_params(params)

        assert_array_almost_equal(model.params, params)

    def test_set_params_wrong_length_raises_error(self):
        """Test that wrong number of parameters raises ValueError."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)

        with pytest.raises(ValueError, match="Expected 3 parameters"):
            model.set_params([0.5, 100.0])  # Missing nugget

        with pytest.raises(ValueError, match="Expected 3 parameters"):
            model.set_params([0.5, 100.0, 0.1, 0.2])  # Too many

    def test_set_params_as_list(self):
        """Test setting parameters from a list."""
        model = CompositeVariogramModel(['exponential'], include_nugget=False)
        params = [0.3, 50.0]

        model.set_params(params)

        assert_array_almost_equal(model.params, params)

    def test_get_component_params_single(self):
        """Test getting parameters for a specific component."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )
        params = [0.5, 100.0, 0.3, 300.0, 0.1]
        model.set_params(params)

        sph_params = model.get_component_params(0)
        exp_params = model.get_component_params(1)

        assert_array_almost_equal(sph_params, [0.5, 100.0])
        assert_array_almost_equal(exp_params, [0.3, 300.0])

    def test_get_component_params_duplicate_names(self):
        """Test getting component params with duplicate component names."""
        model = CompositeVariogramModel(
            ['spherical', 'spherical'],
            include_nugget=False
        )
        params = [0.5, 100.0, 0.3, 300.0]
        model.set_params(params)

        params0 = model.get_component_params(0)
        params1 = model.get_component_params(1)

        assert_array_almost_equal(params0, [0.5, 100.0])
        assert_array_almost_equal(params1, [0.3, 300.0])

    def test_get_component_params_before_set_raises_error(self):
        """Test that getting component params before set_params raises error."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)

        with pytest.raises(ValueError, match="Parameters not set"):
            model.get_component_params(0)

    def test_get_nugget_with_nugget(self):
        """Test getting nugget value when nugget is included."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        model.set_params([0.5, 100.0, 0.15])

        nugget_val = model.get_nugget()
        assert_allclose(nugget_val, 0.15)

    def test_get_nugget_without_nugget(self):
        """Test getting nugget value when nugget is not included."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        model.set_params([0.5, 100.0])

        nugget_val = model.get_nugget()
        assert_allclose(nugget_val, 0.0)

    def test_get_nugget_before_set_raises_error(self):
        """Test that getting nugget before set_params (with nugget) raises error."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)

        with pytest.raises(ValueError, match="Parameters not set"):
            model.get_nugget()


class TestEvaluation:
    """Test variogram evaluation."""

    def test_call_spherical_only(self):
        """Test evaluating spherical model at lag distances."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        sill, range_ = 0.5, 100.0
        model.set_params([sill, range_])

        h = np.array([0.0, 50.0, 100.0, 200.0])
        gamma = model(h)
        expected = spherical(h, sill, range_)

        assert_array_almost_equal(gamma, expected)

    def test_call_spherical_with_nugget(self):
        """Test evaluating spherical model with nugget."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        sill, range_, nugget = 0.5, 100.0, 0.1
        model.set_params([sill, range_, nugget])

        h = np.array([0.0, 50.0, 100.0, 200.0])
        gamma = model(h)
        expected = spherical(h, sill, range_) + nugget_func(h, nugget)

        assert_array_almost_equal(gamma, expected)

    def test_call_multi_component(self):
        """Test evaluating multi-component model (sum of components)."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )
        sill1, range1 = 0.5, 100.0
        sill2, range2 = 0.3, 300.0
        nugget = 0.1
        model.set_params([sill1, range1, sill2, range2, nugget])

        h = np.array([0.0, 50.0, 100.0, 200.0, 500.0])
        gamma = model(h)
        expected = (
            spherical(h, sill1, range1) +
            exponential(h, sill2, range2) +
            nugget_func(h, nugget)
        )

        assert_array_almost_equal(gamma, expected)

    def test_call_before_set_params_raises_error(self):
        """Test that calling model before set_params raises error."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        h = np.array([0.0, 100.0])

        with pytest.raises(ValueError, match="Parameters not set"):
            model(h)

    def test_call_with_scalar_input(self):
        """Test that scalar input is converted to array."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        model.set_params([0.5, 100.0])

        h_scalar = 50.0
        gamma = model(h_scalar)

        # Should return array
        assert isinstance(gamma, np.ndarray)
        expected = spherical(np.array([50.0]), 0.5, 100.0)
        assert_array_almost_equal(gamma, expected)

    def test_evaluate_component_single(self):
        """Test evaluating individual component."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=False
        )
        sill1, range1 = 0.5, 100.0
        sill2, range2 = 0.3, 300.0
        model.set_params([sill1, range1, sill2, range2])

        h = np.array([0.0, 50.0, 100.0, 200.0])

        gamma0 = model.evaluate_component(0, h)
        gamma1 = model.evaluate_component(1, h)

        expected0 = spherical(h, sill1, range1)
        expected1 = exponential(h, sill2, range2)

        assert_array_almost_equal(gamma0, expected0)
        assert_array_almost_equal(gamma1, expected1)

    def test_evaluate_component_before_set_params_raises_error(self):
        """Test that evaluate_component before set_params raises error."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        h = np.array([0.0, 100.0])

        with pytest.raises(ValueError, match="Parameters not set"):
            model.evaluate_component(0, h)

    def test_evaluate_component_matern(self):
        """Test evaluating Matérn component."""
        model = CompositeVariogramModel(['matern'], include_nugget=False)
        sill, range_, nu = 0.6, 150.0, 1.5
        model.set_params([sill, range_, nu])

        h = np.array([0.0, 50.0, 100.0, 200.0])
        gamma = model.evaluate_component(0, h)
        expected = matern(h, sill, range_, nu)

        assert_array_almost_equal(gamma, expected)


class TestStationarity:
    """Test stationarity and sill calculations."""

    def test_is_stationary_all_bounded(self):
        """Test is_stationary returns True for all bounded components."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential', 'gaussian'],
            include_nugget=True
        )
        assert model.is_stationary

    def test_is_stationary_with_unbounded(self):
        """Test is_stationary returns False with unbounded component."""
        model = CompositeVariogramModel(
            ['spherical', 'power'],
            include_nugget=False
        )
        assert not model.is_stationary

    def test_bounded_components_list(self):
        """Test bounded_components property."""
        model = CompositeVariogramModel(
            ['spherical', 'power', 'exponential'],
            include_nugget=False
        )
        assert model.bounded_components == ['spherical', 'exponential']

    def test_unbounded_components_list(self):
        """Test unbounded_components property."""
        # Can't combine two unbounded models, so test with one bounded + one unbounded
        model_power = CompositeVariogramModel(['spherical', 'power'], include_nugget=False)
        assert model_power.unbounded_components == ['power']
        assert model_power.bounded_components == ['spherical']

    def test_get_total_sill_stationary(self):
        """Test get_total_sill for stationary model."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )
        sill1, range1 = 0.5, 100.0
        sill2, range2 = 0.3, 300.0
        nugget = 0.1
        model.set_params([sill1, range1, sill2, range2, nugget])

        total_sill = model.get_total_sill()
        expected = sill1 + sill2 + nugget

        assert_allclose(total_sill, expected)

    def test_get_total_sill_non_stationary_returns_none(self):
        """Test get_total_sill returns None for non-stationary model."""
        model = CompositeVariogramModel(['spherical', 'power'], include_nugget=False)
        model.set_params([0.5, 100.0, 0.1, 1.5])  # sill, range, scale, exponent

        total_sill = model.get_total_sill()
        assert total_sill is None

    def test_get_total_sill_before_set_params_raises_error(self):
        """Test that get_total_sill before set_params raises error."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)

        with pytest.raises(ValueError, match="Parameters not set"):
            model.get_total_sill()

    def test_get_stationary_sill_all_bounded(self):
        """Test get_stationary_sill with all bounded components."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )
        sill1, range1 = 0.5, 100.0
        sill2, range2 = 0.3, 300.0
        nugget = 0.1
        model.set_params([sill1, range1, sill2, range2, nugget])

        stationary_sill = model.get_stationary_sill()
        expected = sill1 + sill2 + nugget

        assert_allclose(stationary_sill, expected)

    def test_get_stationary_sill_mixed(self):
        """Test get_stationary_sill with mixed bounded/unbounded."""
        model = CompositeVariogramModel(
            ['spherical', 'power'],
            include_nugget=True
        )
        sill, range_ = 0.5, 100.0
        scale, exponent = 0.1, 1.5
        nugget = 0.05
        model.set_params([sill, range_, scale, exponent, nugget])

        stationary_sill = model.get_stationary_sill()
        expected = sill + nugget  # Only spherical contributes

        assert_allclose(stationary_sill, expected)

    def test_get_stationary_sill_before_set_params_raises_error(self):
        """Test that get_stationary_sill before set_params raises error."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)

        with pytest.raises(ValueError, match="Parameters not set"):
            model.get_stationary_sill()


class TestVarianceDecomposition:
    """Test variance decomposition by component."""

    def test_decompose_variance_stationary(self):
        """Test decompose_variance for stationary model."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )
        sill1, range1 = 0.5, 100.0
        sill2, range2 = 0.3, 300.0
        nugget = 0.1
        model.set_params([sill1, range1, sill2, range2, nugget])

        decomp = model.decompose_variance()

        assert 'nugget' in decomp
        assert 'spherical' in decomp
        assert 'exponential' in decomp
        assert 'total_stationary' in decomp
        assert 'total_at_reference' in decomp
        assert 'reference_lag' in decomp

        assert_allclose(decomp['nugget'], nugget)
        assert_allclose(decomp['spherical'], sill1)
        assert_allclose(decomp['exponential'], sill2)
        assert_allclose(decomp['total_stationary'], sill1 + sill2 + nugget)

    def test_decompose_variance_non_stationary(self):
        """Test decompose_variance for non-stationary model."""
        model = CompositeVariogramModel(
            ['spherical', 'power'],
            include_nugget=True
        )
        sill, range_ = 0.5, 100.0
        scale, exponent = 0.1, 1.5
        nugget = 0.05
        model.set_params([sill, range_, scale, exponent, nugget])

        decomp = model.decompose_variance()

        assert 'nugget' in decomp
        assert 'spherical' in decomp
        assert 'power' in decomp
        assert 'power_is_nonstationary' in decomp
        assert 'total_stationary' in decomp
        assert_allclose(decomp['total_stationary'], sill + nugget)

    def test_decompose_variance_with_custom_reference_lag(self):
        """Test decompose_variance with custom reference lag."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        model.set_params([0.5, 100.0])

        ref_lag = 500.0
        decomp = model.decompose_variance(reference_lag=ref_lag)

        assert_allclose(decomp['reference_lag'], ref_lag)

    def test_decompose_variance_before_set_params_raises_error(self):
        """Test that decompose_variance before set_params raises error."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)

        with pytest.raises(ValueError, match="Parameters not set"):
            model.decompose_variance()

    def test_decompose_variance_duplicate_component_names(self):
        """Test decompose_variance with duplicate component names."""
        model = CompositeVariogramModel(
            ['spherical', 'spherical'],
            include_nugget=False
        )
        model.set_params([0.5, 100.0, 0.3, 300.0])

        decomp = model.decompose_variance()

        # Should have keys with indices to disambiguate
        assert 'spherical_0' in decomp
        assert 'spherical_1' in decomp


class TestCovarianceFunction:
    """Test covariance function C(h) = sill - γ(h)."""

    def test_get_covariance_function_stationary(self):
        """Test getting covariance function for stationary model."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        sill, range_, nugget = 0.5, 100.0, 0.1
        model.set_params([sill, range_, nugget])

        cov_func = model.get_covariance_function()
        assert cov_func is not None

        # C(h) = sill - γ(h)
        h = np.array([0.0, 50.0, 100.0, 200.0])
        cov = cov_func(h)
        expected_sill = sill + nugget
        expected = expected_sill - model(h)

        assert_array_almost_equal(cov, expected)

    def test_covariance_at_origin(self):
        """Test that C(0) = sill for stationary model."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )
        sill1, range1 = 0.5, 100.0
        sill2, range2 = 0.3, 300.0
        nugget = 0.1
        model.set_params([sill1, range1, sill2, range2, nugget])

        cov_func = model.get_covariance_function()
        total_sill = model.get_total_sill()

        # At h=0, variogram is 0 (with nugget), so C(0) = sill
        c_at_zero = cov_func(np.array([0.0]))[0]
        expected = total_sill

        # Note: Due to nugget at h>0, C(0) = total_sill
        assert_allclose(c_at_zero, expected)

    def test_covariance_at_large_lag(self):
        """Test covariance at large lag approaches 0 for bounded models."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        sill = 0.5
        range_ = 100.0
        model.set_params([sill, range_])

        cov_func = model.get_covariance_function()

        # At large h > range, γ = sill, so C(h) = sill - sill = 0
        c_large = cov_func(np.array([1000.0]))[0]
        assert_allclose(c_large, 0.0, atol=1e-10)

    def test_get_covariance_function_non_stationary_returns_none(self):
        """Test get_covariance_function returns None for non-stationary."""
        model = CompositeVariogramModel(['power'], include_nugget=False)
        model.set_params([0.1, 1.5])

        cov_func = model.get_covariance_function()
        assert cov_func is None

    def test_get_covariance_function_before_set_params_raises_error(self):
        """Test that get_covariance_function before set_params raises error."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)

        # Note: This will raise when trying to compute sill before params set
        # The error will come from get_total_sill called inside
        with pytest.raises(ValueError, match="Parameters not set"):
            model.get_covariance_function()


class TestDefaultGuessAndBounds:
    """Test default parameter guess and bounds generation."""

    def test_default_guess_spherical(self):
        """Test default guess for spherical model."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)

        lags = np.array([10.0, 50.0, 100.0, 200.0, 500.0])
        variogram = np.array([0.1, 0.3, 0.4, 0.45, 0.5])

        guess = model.default_guess(lags, variogram)

        assert len(guess) == 2  # sill, range
        assert guess[0] > 0  # sill should be positive
        assert guess[1] > 0  # range should be positive

    def test_default_guess_with_nugget(self):
        """Test default guess includes nugget."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)

        lags = np.array([10.0, 50.0, 100.0, 200.0, 500.0])
        variogram = np.array([0.1, 0.3, 0.4, 0.45, 0.5])

        guess = model.default_guess(lags, variogram)

        assert len(guess) == 3  # sill, range, nugget
        assert guess[2] > 0  # nugget should be positive

    def test_default_guess_multi_component(self):
        """Test default guess for multi-component model."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )

        lags = np.array([10.0, 50.0, 100.0, 200.0, 500.0])
        variogram = np.array([0.1, 0.3, 0.4, 0.45, 0.5])

        guess = model.default_guess(lags, variogram)

        assert len(guess) == 5  # sill1, range1, sill2, range2, nugget
        # Ranges should be spread differently for multi-component
        assert guess[1] < guess[3]  # range1 < range2

    def test_bounds_spherical(self):
        """Test parameter bounds for spherical model."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)

        lags = np.array([10.0, 50.0, 100.0, 200.0, 500.0])
        variogram = np.array([0.1, 0.3, 0.4, 0.45, 0.5])

        lower, upper = model.bounds(lags, variogram)

        assert len(lower) == 2
        assert len(upper) == 2
        assert lower[0] <= upper[0]  # sill bounds
        assert lower[1] <= upper[1]  # range bounds

    def test_bounds_with_nugget(self):
        """Test parameter bounds include nugget."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)

        lags = np.array([10.0, 50.0, 100.0, 200.0, 500.0])
        variogram = np.array([0.1, 0.3, 0.4, 0.45, 0.5])

        lower, upper = model.bounds(lags, variogram)

        assert len(lower) == 3
        assert len(upper) == 3
        assert lower[2] == 0  # nugget lower bound
        assert upper[2] > 0  # nugget upper bound

    def test_bounds_multi_component(self):
        """Test parameter bounds for multi-component model."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )

        lags = np.array([10.0, 50.0, 100.0, 200.0, 500.0])
        variogram = np.array([0.1, 0.3, 0.4, 0.45, 0.5])

        lower, upper = model.bounds(lags, variogram)

        assert len(lower) == 5
        assert len(upper) == 5


class TestDescription:
    """Test human-readable model description."""

    def test_description_spherical(self):
        """Test description for spherical model."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        model.set_params([0.5, 100.0, 0.1])

        desc = model.description()

        assert isinstance(desc, str)
        assert len(desc) > 0
        assert 'Nugget' in desc
        assert 'Spherical' in desc
        assert 'sill' in desc
        assert 'range' in desc

    def test_description_multi_component(self):
        """Test description for multi-component model."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )
        model.set_params([0.5, 100.0, 0.3, 300.0, 0.1])

        desc = model.description()

        assert 'Nugget' in desc
        assert 'Spherical' in desc
        assert 'Exponential' in desc
        assert 'Total sill' in desc

    def test_description_non_stationary(self):
        """Test description for non-stationary model."""
        model = CompositeVariogramModel(
            ['spherical', 'power'],
            include_nugget=False
        )
        model.set_params([0.5, 100.0, 0.1, 1.5])

        desc = model.description()

        assert 'NON-STATIONARY' in desc
        assert 'Stationary sill' in desc
        assert 'Total at reference' in desc

    def test_description_before_set_params_raises_error(self):
        """Test that description before set_params raises error."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)

        # description() will call get_nugget() which requires params set
        with pytest.raises(ValueError, match="Parameters not set"):
            model.description()


class TestPropertyAccess:
    """Test property access and queries."""

    def test_params_property_before_set(self):
        """Test params property returns None before set_params."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        assert model.params is None

    def test_params_property_after_set(self):
        """Test params property returns set values."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        params = np.array([0.5, 100.0])
        model.set_params(params)

        assert_array_almost_equal(model.params, params)

    def test_n_params_property(self):
        """Test n_params property."""
        model1 = CompositeVariogramModel(['spherical'], include_nugget=False)
        assert model1.n_params == 2

        model2 = CompositeVariogramModel(['spherical'], include_nugget=True)
        assert model2.n_params == 3

        model3 = CompositeVariogramModel(['matern'], include_nugget=True)
        assert model3.n_params == 4

    def test_param_names_property(self):
        """Test param_names property returns copy."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        names1 = model.param_names
        names2 = model.param_names

        # Should be equal but different objects
        assert names1 == names2
        assert names1 is not names2  # Different object instances


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    def test_spherical_exponential_nugget_workflow(self):
        """Test complete workflow with spherical + exponential + nugget model."""
        # Create model
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )

        # Generate synthetic data
        h = np.array([0.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0])
        true_params = [0.4, 80.0, 0.35, 250.0, 0.05]
        model.set_params(true_params)

        # Evaluate model
        gamma = model(h)

        # Verify properties
        assert model.is_stationary
        assert_allclose(model.get_total_sill(), 0.8, atol=1e-6)
        assert_allclose(model.get_nugget(), 0.05)

        # Check variance decomposition
        decomp = model.decompose_variance()
        assert_allclose(decomp['total_stationary'], 0.8, atol=1e-6)

        # Check covariance function
        cov_func = model.get_covariance_function()
        assert cov_func is not None
        c_at_zero = cov_func(np.array([0.0]))[0]
        assert_allclose(c_at_zero, 0.8, atol=1e-6)

    def test_matern_with_nugget_workflow(self):
        """Test workflow with Matérn model and nugget."""
        model = CompositeVariogramModel(['matern'], include_nugget=True)

        # Set realistic parameters
        sill, range_, nu, nugget = 0.6, 150.0, 1.5, 0.1
        model.set_params([sill, range_, nu, nugget])

        # Evaluate
        h = np.array([0.0, 30.0, 100.0, 300.0])
        gamma = model(h)

        # Verify
        assert_allclose(gamma[0], nugget_func(h[0], nugget))  # At h=0, only nugget
        assert gamma[1] < gamma[-1]  # Should increase with distance

        # Component evaluation
        gamma_matern = model.evaluate_component(0, h)
        expected_matern = matern(h, sill, range_, nu)
        assert_array_almost_equal(gamma_matern, expected_matern)

    def test_non_stationary_workflow(self):
        """Test workflow with non-stationary model."""
        model = CompositeVariogramModel(
            ['spherical', 'power'],
            include_nugget=True
        )

        # Set parameters
        sill, range_ = 0.3, 100.0
        scale, exponent = 0.05, 1.5
        nugget = 0.05
        model.set_params([sill, range_, scale, exponent, nugget])

        # Verify non-stationarity
        assert not model.is_stationary
        assert model.get_total_sill() is None
        assert_allclose(model.get_stationary_sill(), 0.35)  # sill + nugget

        # Decomposition
        decomp = model.decompose_variance()
        assert 'power_is_nonstationary' in decomp
        assert_allclose(decomp['total_stationary'], 0.35)

    def test_parameter_validation_with_damped_hole_effect(self):
        """Test parameter validation for damped hole effect."""
        model = CompositeVariogramModel(
            ['damped_hole_effect'],
            include_nugget=False
        )

        # Valid parameters: 2πr/λ > 1
        valid_params = [0.5, 100.0, 500.0]  # 2π*100/500 ≈ 1.26 > 1
        model.set_params(valid_params)  # Should not raise

        # Invalid parameters: 2πr/λ < 1
        invalid_params = [0.5, 10.0, 100.0]  # 2π*10/100 ≈ 0.63 < 1
        with pytest.raises(ValueError, match="positive definiteness"):
            model.set_params(invalid_params)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_sill(self):
        """Test model with very small sill values."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        model.set_params([1e-6, 100.0])

        h = np.array([0.0, 50.0, 100.0])
        gamma = model(h)

        assert np.all(gamma >= 0)
        assert np.all(gamma <= 1e-5)

    def test_very_large_range(self):
        """Test model with very large range parameter."""
        model = CompositeVariogramModel(['exponential'], include_nugget=False)
        model.set_params([0.5, 1e6])

        h = np.array([0.0, 100.0, 1000.0])
        gamma = model(h)

        # Values should be small relative to lag
        assert gamma[0] < gamma[1] < gamma[2]

    def test_zero_nugget(self):
        """Test model with zero nugget."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        model.set_params([0.5, 100.0, 0.0])

        assert_allclose(model.get_nugget(), 0.0)

        h = np.array([0.0])
        gamma = model(h)
        assert_allclose(gamma[0], 0.0)

    def test_very_large_nugget(self):
        """Test model with large nugget relative to sill."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        model.set_params([0.1, 100.0, 0.9])

        total_sill = model.get_total_sill()
        assert_allclose(total_sill, 1.0)
        assert_allclose(model.get_nugget(), 0.9)

    def test_single_lag_distance(self):
        """Test model evaluation with single lag distance."""
        model = CompositeVariogramModel(['spherical'], include_nugget=True)
        model.set_params([0.5, 100.0, 0.1])

        gamma = model(np.array([50.0]))

        assert gamma.shape == (1,)
        assert gamma[0] > 0

    def test_many_components(self):
        """Test model with many components."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential', 'gaussian'],
            include_nugget=True
        )

        assert model.n_params == 7  # 2+2+2 params + nugget

        params = [0.3, 80.0, 0.2, 150.0, 0.2, 200.0, 0.1]
        model.set_params(params)

        h = np.array([0.0, 100.0, 300.0])
        gamma = model(h)

        # Verify composition
        comp0 = model.evaluate_component(0, h)
        comp1 = model.evaluate_component(1, h)
        comp2 = model.evaluate_component(2, h)
        expected = comp0 + comp1 + comp2 + nugget_func(h, 0.1)

        assert_array_almost_equal(gamma, expected)


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_consistency_across_multiple_evaluations(self):
        """Test that repeated evaluations give consistent results."""
        model = CompositeVariogramModel(
            ['spherical', 'exponential'],
            include_nugget=True
        )
        model.set_params([0.5, 100.0, 0.3, 300.0, 0.1])

        h = np.array([0.0, 50.0, 100.0, 200.0])

        gamma1 = model(h)
        gamma2 = model(h)
        gamma3 = model(h)

        assert_array_almost_equal(gamma1, gamma2)
        assert_array_almost_equal(gamma2, gamma3)

    def test_vectorization_vs_scalar_evaluation(self):
        """Test that vectorized and scalar evaluations agree."""
        model = CompositeVariogramModel(['gaussian'], include_nugget=False)
        model.set_params([0.5, 100.0])

        h_values = [0.0, 25.0, 50.0, 100.0, 200.0]

        # Vectorized
        gamma_vec = model(np.array(h_values))

        # Scalar
        gamma_scalar = np.array([model(np.array([h]))[0] for h in h_values])

        assert_array_almost_equal(gamma_vec, gamma_scalar)

    def test_large_lag_distances(self):
        """Test model with very large lag distances."""
        model = CompositeVariogramModel(['exponential'], include_nugget=False)
        model.set_params([0.5, 100.0])

        h = np.array([0.0, 1e3, 1e4, 1e5])
        gamma = model(h)

        # For exponential, should approach sill
        assert_allclose(gamma[-1], 0.5, rtol=1e-4)

    def test_very_small_lag_distances(self):
        """Test model with very small lag distances."""
        model = CompositeVariogramModel(['spherical'], include_nugget=False)
        model.set_params([0.5, 100.0])

        h = np.array([0.0, 1e-6, 1e-3, 1.0])
        gamma = model(h)

        # Should be monotonically increasing for spherical
        assert np.all(np.diff(gamma) >= -1e-10)  # Allow for numerical errors
