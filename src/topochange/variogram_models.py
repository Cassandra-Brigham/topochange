"""variogram model definitions and registry.

references:
- Chilès & Delfiner (2012). Geostatistics: Modeling Spatial Uncertainty.
- Webster & Oliver (2007). Geostatistics for Environmental Scientists.
- Stein (1999). Interpolation of Spatial Data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.special import gamma as gamma_func, kv as bessel_kv


# model Specifications

@dataclass
class VariogramModelSpec:
    """Specification for a single variogram model type.
    
    Attributes
    ----------
    name : str
        Model identifier (e.g., 'spherical', 'exponential').
    func : Callable
        Function γ(h, *params) returning semivariance at lag h.
    param_names : List[str]
        Names of parameters in order expected by func.
    is_bounded : bool
        True if model has a finite sill (stationary process).
    has_sill : bool
        True if one of the parameters represents a sill.
    practical_range_factor : float or None
        Multiplier to get practical range (95% of sill) from range parameter.
        None for unbounded models.
    description : str
        Human-readable description of the model.
    """
    name: str
    func: Callable
    param_names: List[str]
    is_bounded: bool
    has_sill: bool
    practical_range_factor: Optional[float]
    description: str
    
    # callables for default guesses, bounds, and validation (set after creation)
    _default_guess: Optional[Callable] = field(default=None, repr=False)
    _bounds: Optional[Callable] = field(default=None, repr=False)
    _validate: Optional[Callable] = field(default=None, repr=False)
    
    def default_guess(self, lags: np.ndarray, variogram: np.ndarray) -> List[float]:
        """Generate default initial parameter guess."""
        if self._default_guess is None:
            raise NotImplementedError(f"No default guess for {self.name}")
        return self._default_guess(lags, variogram)
    
    def bounds(self, lags: np.ndarray, variogram: np.ndarray) -> Tuple[List[float], List[float]]:
        """Generate parameter bounds."""
        if self._bounds is None:
            raise NotImplementedError(f"No bounds for {self.name}")
        return self._bounds(lags, variogram)

    def validate(self, params: List[float]) -> None:
        """Validate parameters (raises ValueError if invalid)."""
        if self._validate is not None:
            self._validate(params)


# individual Model Functions

def spherical(h: np.ndarray, sill: float, range_: float) -> np.ndarray:
    """Spherical variogram model.
    
    γ(h) = C * [1.5(h/a) - 0.5(h/a)³]  for h ≤ a
         = C                            for h > a
    
    Parameters
    ----------
    h : array-like
        Lag distances.
    sill : float
        Sill (total variance contribution), C ≥ 0.
    range_ : float
        Range parameter, a > 0.
    
    Returns
    -------
    gamma : ndarray
        Semivariance values.
    
    Notes
    -----
    The spherical model reaches its sill exactly at h = a.
    Practical range equals the range parameter.
    
    Physical interpretation: Hard cutoff in correlation, common in
    sedimentary deposits and soil horizons.
    """
    h = np.asarray(h, dtype=float)
    gamma = np.zeros_like(h)
    
    mask = h <= range_
    ratio = h[mask] / range_
    gamma[mask] = sill * (1.5 * ratio - 0.5 * ratio**3)
    gamma[~mask] = sill
    
    return gamma


def exponential(h: np.ndarray, sill: float, range_: float) -> np.ndarray:
    """Exponential variogram model.
    
    γ(h) = C * [1 - exp(-h/a)]
    
    Parameters
    ----------
    h : array-like
        Lag distances.
    sill : float
        Sill (asymptotic variance), C ≥ 0.
    range_ : float
        Range parameter, a > 0.
    
    Returns
    -------
    gamma : ndarray
        Semivariance values.
    
    Notes
    -----
    Practical range (95% of sill) = 3a.
    Near origin: γ(h) ≈ (C/a)h (linear).
    
    Physical interpretation: Exponential decay of correlation,
    common in hydrogeology and atmospheric sciences.
    """
    h = np.asarray(h, dtype=float)
    return sill * (1 - np.exp(-h / range_))


def gaussian(h: np.ndarray, sill: float, range_: float) -> np.ndarray:
    """Gaussian variogram model.
    
    γ(h) = C * [1 - exp(-h²/a²)]
    
    Parameters
    ----------
    h : array-like
        Lag distances.
    sill : float
        Sill (asymptotic variance), C ≥ 0.
    range_ : float
        Range parameter, a > 0.
    
    Returns
    -------
    gamma : ndarray
        Semivariance values.
    
    Notes
    -----
    Practical range (95% of sill) ≈ 1.73a.
    Near origin: γ(h) ≈ (C/a²)h² (parabolic).
    
    WARNING: The Gaussian model implies infinite differentiability,
    which can cause numerical instability in kriging. It should typically
    be combined with a nugget effect.
    
    Physical interpretation: Very smooth spatial variation.
    """
    h = np.asarray(h, dtype=float)
    return sill * (1 - np.exp(-(h / range_)**2))


def matern(h: np.ndarray, sill: float, range_: float, nu: float) -> np.ndarray:
    """Matérn variogram model.
    
    γ(h) = C * [1 - (2^(1-ν)/Γ(ν)) * (h/a)^ν * K_ν(h/a)]
    
    Parameters
    ----------
    h : array-like
        Lag distances.
    sill : float
        Sill (asymptotic variance), C ≥ 0.
    range_ : float
        Range parameter, a > 0.
    nu : float
        Smoothness parameter, ν > 0.
    
    Returns
    -------
    gamma : ndarray
        Semivariance values.
    
    Notes
    -----
    Special cases:
        ν = 0.5 → Exponential
        ν = 1.5 → Matérn-3/2 (once differentiable)
        ν = 2.5 → Matérn-5/2 (twice differentiable)
        ν → ∞  → Gaussian
    
    The smoothness ν controls differentiability of the underlying field.
    
    References
    ----------
    Stein, M.L. (1999). Interpolation of Spatial Data. Springer.
    """
    h = np.asarray(h, dtype=float)
    gamma = np.zeros_like(h)
    
    # handle h = 0 separately
    nonzero = h > 0
    if not np.any(nonzero):
        return gamma
    
    h_nz = h[nonzero]
    scaled = h_nz / range_
    
    # compute Matérn correlation
    coef = (2**(1 - nu)) / gamma_func(nu)
    bessel_term = bessel_kv(nu, scaled)
    
    # handle numerical issues with Bessel function
    bessel_term = np.where(np.isfinite(bessel_term), bessel_term, 0)
    
    correlation = coef * (scaled**nu) * bessel_term
    correlation = np.clip(correlation, 0, 1)  # Ensure valid range
    
    gamma[nonzero] = sill * (1 - correlation)
    
    return gamma


def damped_hole_effect(h: np.ndarray, sill: float, range_: float,
                       wavelength: float) -> np.ndarray:
    """Damped hole-effect variogram model.

    γ(h) = C × [1 - exp(-h/r) × cos(2πh/λ)]

    Parameters
    ----------
    h : array-like
        Lag distances.
    sill : float
        Sill (asymptotic variance), C ≥ 0.
    range_ : float
        Damping range, r > 0. Controls exponential decay of oscillations.
    wavelength : float
        Wavelength of oscillation, λ > 0.

    Returns
    -------
    gamma : ndarray
        Semivariance values.

    Notes
    -----
    VALIDITY CONSTRAINT: For positive definiteness in 3D, requires
    approximately 2πr/λ > 1. Parameters violating this will raise
    an error during model fitting.

    The variogram can temporarily exceed the sill when the cosine
    term is negative:this represents anti-correlation (the "hole")
    and is physically meaningful for quasi-periodic phenomena.

    Physical interpretation: Layered structures (sedimentary sequences,
    weathering horizons) where correlation alternates but weakens with
    distance.

    References
    ----------
    Chilès, J.P. & Delfiner, P. (2012). Geostatistics, Section 2.6.3.
    Ma, Y.Z. & Jones, T.A. (2001). Modeling hole-effect variograms.
        Mathematical Geology, 33(5), 631-648.
        https://doi.org/10.1023/A:1011001029880
    """
    h = np.asarray(h, dtype=float)
    damping = np.exp(-h / range_)
    oscillation = np.cos(2 * np.pi * h / wavelength)
    return sill * (1 - damping * oscillation)


def power(h: np.ndarray, scale: float, exponent: float) -> np.ndarray:
    """Power variogram model (unbounded/non-stationary).
    
    γ(h) = α * h^ω,  where 0 < ω < 2
    
    Parameters
    ----------
    h : array-like
        Lag distances.
    scale : float
        Scale parameter, α > 0.
    exponent : float
        Power exponent, ω ∈ (0, 2).
    
    Returns
    -------
    gamma : ndarray
        Semivariance values.
    
    Notes
    -----
    WARNING: NON-STATIONARY: This model has no finite sill. The variance
    increases indefinitely with distance, implying:
        - Trend in the data that wasn't removed
        - Process operates at scales larger than your study area
        - True fractal behavior (rare in practice)
    
    The exponent relates to fractal dimension: D = 2 - ω/2 (for 1D).
    
    Constraint: ω ∈ (0, 2) is required for conditional negative definiteness.
    
    References
    ----------
    Chilès, J.P. & Delfiner, P. (2012). Geostatistics, Section 2.5.
    """
    h = np.asarray(h, dtype=float)
    return scale * np.power(h, exponent)


def linear(h: np.ndarray, slope: float) -> np.ndarray:
    """Linear variogram model (unbounded/non-stationary).
    
    γ(h) = β * h
    
    This is a special case of the power model with exponent = 1.
    
    Parameters
    ----------
    h : array-like
        Lag distances.
    slope : float
        Slope parameter, β > 0.
    
    Returns
    -------
    gamma : ndarray
        Semivariance values.
    
    Notes
    -----
    WARNING: NON-STATIONARY: Indicates unresolved trend or drift in the mean.
    Consider detrending the data before variogram analysis.
    """
    h = np.asarray(h, dtype=float)
    return slope * h


def nugget(h: np.ndarray, c0: float) -> np.ndarray:
    """Pure nugget effect model.
    
    γ(h) = 0   for h = 0
         = C₀  for h > 0
    
    Parameters
    ----------
    h : array-like
        Lag distances.
    c0 : float
        Nugget variance, C₀ ≥ 0.
    
    Returns
    -------
    gamma : ndarray
        Semivariance values.
    
    Notes
    -----
    The nugget represents:
        1. Measurement error (instrument precision)
        2. Microscale variation (below sampling resolution)
        3. Georeferencing error
    
    A large nugget (>50% of total sill) suggests noisy data or
    strong micro-topographic variation. A small nugget (<10%)
    indicates smooth variation. Zero nugget is rare in practice.
    """
    h = np.asarray(h, dtype=float)
    return np.where(h > 0, c0, 0.0)


# model Registry

class VariogramModelRegistry:
    """Registry of available variogram models.
    
    This class provides a centralized catalog of valid variogram models
    with their specifications, default parameters, and validation rules.
    
    Examples
    --------
    >>> registry = VariogramModelRegistry()
    >>> spec = registry.get_model('spherical')
    >>> print(spec.description)
    
    >>> # Check if combination is valid
    >>> valid, msg = registry.validate_combination(['spherical', 'exponential'])
    """
    
    def __init__(self):
        self._models: Dict[str, VariogramModelSpec] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register all standard models."""
        
        # helper for default guesses and bounds
        # ── range bound helpers ──────────────────────────────────────
        # We bound the **effective (practical) range** — the distance at
        # which γ(h) essentially reaches the sill — to half the observed
        # lag domain.  The empirical variogram is unreliable beyond this
        # point (Journel & Huijbregts, 1978, Mining Geostatistics).
        #
        # Because each model's raw "range" parameter relates to the
        # effective range by a factor (practical_range_factor), we set
        #
        #     raw_range_upper = max_lag / (2 × practical_range_factor)
        #
        # so that effective_range_upper = max_lag / 2 for every model.

        def _make_bounded_guess(prf):
            """Create initial-guess function for a model with the given
            practical_range_factor."""
            def _guess(lags, variogram):
                max_gamma = np.nanmax(variogram)
                max_lag = np.nanmax(lags)
                # initial raw-range guess: ~1/3 of effective-range ceiling
                return [max_gamma * 0.9, max_lag / (3 * prf)]
            return _guess

        def _make_bounded_bounds(prf):
            """Create bounds function for a model with the given
            practical_range_factor."""
            def _bounds(lags, variogram):
                max_gamma = np.nanmax(variogram) * 3
                max_lag = np.nanmax(lags)
                # range lower bound = bin width (minimum resolvable scale)
                bin_width = float(np.min(np.diff(lags))) if len(lags) > 1 else 1e-6
                return ([0, bin_width], [max_gamma, max_lag])
            return _bounds

        # spherical  (practical_range_factor = 1.0)
        spherical_spec = VariogramModelSpec(
            name='spherical',
            func=spherical,
            param_names=['sill', 'range'],
            is_bounded=True,
            has_sill=True,
            practical_range_factor=1.0,
            description="Spherical model: reaches sill exactly at range. "
                       "Common for sedimentary deposits."
        )
        spherical_spec._default_guess = _make_bounded_guess(1.0)
        spherical_spec._bounds = _make_bounded_bounds(1.0)
        self._models['spherical'] = spherical_spec

        # exponential  (practical_range_factor = 3.0)
        exp_spec = VariogramModelSpec(
            name='exponential',
            func=exponential,
            param_names=['sill', 'range'],
            is_bounded=True,
            has_sill=True,
            practical_range_factor=3.0,
            description="Exponential model: asymptotic approach to sill. "
                       "Practical range = 3 × range parameter."
        )
        exp_spec._default_guess = _make_bounded_guess(3.0)
        exp_spec._bounds = _make_bounded_bounds(3.0)
        self._models['exponential'] = exp_spec

        # Gaussian  (practical_range_factor = 1.73)
        gauss_spec = VariogramModelSpec(
            name='gaussian',
            func=gaussian,
            param_names=['sill', 'range'],
            is_bounded=True,
            has_sill=True,
            practical_range_factor=1.73,
            description="Gaussian model: very smooth variation. "
                       "WARNING: May cause numerical instability without nugget."
        )
        gauss_spec._default_guess = _make_bounded_guess(1.73)
        gauss_spec._bounds = _make_bounded_bounds(1.73)
        self._models['gaussian'] = gauss_spec

        # Matérn  (practical_range_factor depends on ν)
        # Since ν is optimised jointly with range, we use a conservative
        # factor of 3.0 (same as exponential, which is Matérn with
        # ν = 0.5) to prevent the effective range from exceeding the
        # half-lag ceiling even at low ν.
        _matern_prf_conservative = 3.0

        def _matern_guess(lags, variogram):
            max_gamma = np.nanmax(variogram)
            max_lag = np.nanmax(lags)
            return [max_gamma * 0.9,
                    max_lag / (3 * _matern_prf_conservative),
                    1.5]  # default nu=1.5

        def _matern_bounds(lags, variogram):
            max_gamma = np.nanmax(variogram) * 3
            max_lag = np.nanmax(lags)
            bin_width = float(np.min(np.diff(lags))) if len(lags) > 1 else 1e-6
            return ([0, bin_width, 0.1],
                    [max_gamma, max_lag, 5.0])
        
        matern_spec = VariogramModelSpec(
            name='matern',
            func=matern,
            param_names=['sill', 'range', 'nu'],
            is_bounded=True,
            has_sill=True,
            practical_range_factor=None,  # Depends on nu
            description="Matérn model: flexible smoothness via nu parameter. "
                       "nu=0.5 → exponential, nu→∞ → Gaussian."
        )
        matern_spec._default_guess = _matern_guess
        matern_spec._bounds = _matern_bounds
        self._models['matern'] = matern_spec
        
        # damped hole-effect
        def _damped_hole_guess(lags, variogram):
            max_gamma = np.nanmax(variogram)
            max_lag = np.nanmax(lags)
            # estimate: sill from max variance, range = max_lag/4, wavelength = max_lag/3
            # this default ensures 2πr/λ ≈ 2.36 > 1 (valid)
            return [max_gamma * 0.9, max_lag / 4, max_lag / 3]

        def _damped_hole_bounds(lags, variogram):
            max_gamma = np.nanmax(variogram) * 3
            max_lag = np.nanmax(lags)
            bin_width = float(np.min(np.diff(lags))) if len(lags) > 1 else 1e-6
            return ([0, bin_width, bin_width], [max_gamma, max_lag, max_lag])

        def _damped_hole_validate(params):
            """Validate positive definiteness constraint for damped hole effect.

            Raises ValueError if 2πr/λ < 1, which violates positive definiteness
            in 3D and causes unreliable uncertainty propagation.
            """
            sill, range_, wavelength = params
            ratio = 2 * np.pi * range_ / wavelength
            if ratio < 1.0:
                raise ValueError(
                    f"Damped hole effect parameters violate positive definiteness: "
                    f"2πr/λ = {ratio:.2f} < 1. Increase 'range' or decrease 'wavelength' "
                    f"so the damping is sufficient (2πr/λ > 1)."
                )

        damped_hole_spec = VariogramModelSpec(
            name='damped_hole_effect',
            func=damped_hole_effect,
            param_names=['sill', 'range', 'wavelength'],
            is_bounded=True,
            has_sill=True,
            practical_range_factor=3.0,  # Similar to exponential
            description="Damped hole-effect: quasi-periodic variation with "
                       "exponential decay. Valid in 3D if 2πr/λ > 1."
        )
        damped_hole_spec._default_guess = _damped_hole_guess
        damped_hole_spec._bounds = _damped_hole_bounds
        damped_hole_spec._validate = _damped_hole_validate
        self._models['damped_hole_effect'] = damped_hole_spec
        
        # power (unbounded)
        def _power_guess(lags, variogram):
            # estimate from log-log regression
            valid = (lags > 0) & (variogram > 0)
            if np.sum(valid) < 2:
                return [0.1, 1.0]
            log_h = np.log(lags[valid])
            log_g = np.log(variogram[valid])
            slope = np.polyfit(log_h, log_g, 1)[0]
            slope = np.clip(slope, 0.1, 1.9)
            scale = np.exp(np.mean(log_g - slope * log_h))
            return [scale, slope]
        
        def _power_bounds(lags, variogram):
            return ([1e-10, 0.01], [100, 1.99])
        
        power_spec = VariogramModelSpec(
            name='power',
            func=power,
            param_names=['scale', 'exponent'],
            is_bounded=False,
            has_sill=False,
            practical_range_factor=None,
            description="Power model: γ(h) = α·h^ω. "
                       "WARNING: NON-STATIONARY: No finite sill. "
                       "Implies trend or fractal behavior."
        )
        power_spec._default_guess = _power_guess
        power_spec._bounds = _power_bounds
        self._models['power'] = power_spec
        
        # linear (unbounded)
        def _linear_guess(lags, variogram):
            valid = lags > 0
            if np.sum(valid) < 2:
                return [0.1]
            slope = np.polyfit(lags[valid], variogram[valid], 1)[0]
            return [max(slope, 1e-6)]
        
        def _linear_bounds(lags, variogram):
            return ([1e-10], [100])
        
        linear_spec = VariogramModelSpec(
            name='linear',
            func=linear,
            param_names=['slope'],
            is_bounded=False,
            has_sill=False,
            practical_range_factor=None,
            description="Linear model: γ(h) = β·h. "
                       "WARNING: NON-STATIONARY: Indicates trend/drift."
        )
        linear_spec._default_guess = _linear_guess
        linear_spec._bounds = _linear_bounds
        self._models['linear'] = linear_spec
    
    def get_model(self, name: str) -> VariogramModelSpec:
        """Get model specification by name."""
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}. "
                           f"Available: {list(self._models.keys())}")
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self._models.keys())
    
    def list_bounded_models(self) -> List[str]:
        """List only bounded (stationary) models."""
        return [name for name, spec in self._models.items() if spec.is_bounded]
    
    def list_unbounded_models(self) -> List[str]:
        """List only unbounded (non-stationary) models."""
        return [name for name, spec in self._models.items() if not spec.is_bounded]
    
    def is_bounded(self, name: str) -> bool:
        """Check if a model is bounded (stationary)."""
        return self.get_model(name).is_bounded
    
    def validate_combination(
        self, 
        model_names: List[str],
        include_nugget: bool = False
    ) -> Tuple[bool, str]:
        """Validate if a combination of models is allowed.
        
        Parameters
        ----------
        model_names : List[str]
            Names of models to combine.
        include_nugget : bool
            Whether a nugget will be added.
        
        Returns
        -------
        valid : bool
            True if combination is valid.
        message : str
            Description of validation result or warning.
        """
        if not model_names:
            if include_nugget:
                return True, "Pure nugget model (no spatial structure)."
            return False, "At least one model required."
        
        # check all models exist
        for name in model_names:
            if name not in self._models:
                return False, f"Unknown model: {name}"
        
        # count bounded vs unbounded
        bounded = [n for n in model_names if self._models[n].is_bounded]
        unbounded = [n for n in model_names if not self._models[n].is_bounded]
        
        # rule: Cannot combine multiple unbounded models
        if len(unbounded) > 1:
            return False, (
                f"Cannot combine multiple unbounded models: {unbounded}. "
                "This would violate positive definiteness constraints."
            )
        
        # rule: power + linear specifically forbidden
        if 'power' in unbounded and 'linear' in unbounded:
            return False, "Cannot combine power and linear models."
        
        # warning for unbounded models
        if unbounded:
            msg = (
                f"WARNING: NON-STATIONARY combination: includes {unbounded}. "
                "The process has no finite variance. Results will be scale-dependent. "
                "Consider detrending your data if this is unexpected."
            )
            if bounded:
                msg += f" Stationary components ({bounded}) will be reported separately."
            return True, msg
        
        # warning for Gaussian without nugget
        if 'gaussian' in bounded and not include_nugget:
            return True, (
                "WARNING: Gaussian model without nugget may cause numerical instability. "
                "Consider adding a nugget effect."
            )
        
        return True, "Valid combination."


# global registry instance
MODEL_REGISTRY = VariogramModelRegistry()
