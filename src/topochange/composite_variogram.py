"""Composite variogram model builder.

variogram models from individual components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .variogram_models import (
    VariogramModelSpec, 
    VariogramModelRegistry, 
    MODEL_REGISTRY,
    nugget as nugget_func
)


@dataclass
class CompositeVariogramModel:
    """A composite variogram model built from multiple components.
    
    This class represents a nested variogram structure:
        γ(h) = C₀·1{h>0} + Σᵢ γᵢ(h; θᵢ)
    
    where C₀ is the optional nugget and γᵢ are individual model components.
    
    Attributes
    ----------
    component_names : List[str]
        Names of component models (e.g., ['spherical', 'exponential']).
    include_nugget : bool
        Whether a nugget effect is included.
    params : ndarray or None
        Fitted parameter values (set after fitting).
    is_stationary : bool
        True if all components are bounded.
    
    Examples
    --------
    >>> model = CompositeVariogramModel(['spherical', 'exponential'], nugget=True)
    >>> model.set_params([0.5, 100, 0.3, 300, 0.1])  # sill1, range1, sill2, range2, nugget
    >>> gamma_values = model(lags)
    """
    
    component_names: List[str]
    include_nugget: bool = True
    registry: VariogramModelRegistry = field(default_factory=lambda: MODEL_REGISTRY, repr=False)
    
    # set after initialization
    _components: List[VariogramModelSpec] = field(default_factory=list, repr=False)
    _params: Optional[np.ndarray] = field(default=None, repr=False)
    _param_names: List[str] = field(default_factory=list, repr=False)
    _param_slices: Dict[str, slice] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Initialize components and parameter structure."""
        # validate combination
        valid, msg = self.registry.validate_combination(
            self.component_names, self.include_nugget
        )
        if not valid:
            raise ValueError(msg)
        self._validation_message = msg
        
        # get component specs
        self._components = [
            self.registry.get_model(name) for name in self.component_names
        ]
        
        # build parameter name list and slices
        self._param_names = []
        self._param_slices = {}
        idx = 0
        
        for i, (name, spec) in enumerate(zip(self.component_names, self._components)):
            n_params = len(spec.param_names)
            component_key = f"{name}_{i}" if self.component_names.count(name) > 1 else name
            
            for pname in spec.param_names:
                self._param_names.append(f"{component_key}_{pname}")
            
            self._param_slices[component_key] = slice(idx, idx + n_params)
            idx += n_params
        
        if self.include_nugget:
            self._param_names.append('nugget')
            self._param_slices['nugget'] = slice(idx, idx + 1)
    
    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        return len(self._param_names)
    
    @property
    def param_names(self) -> List[str]:
        """List of all parameter names."""
        return self._param_names.copy()
    
    @property
    def params(self) -> Optional[np.ndarray]:
        """Current parameter values."""
        return self._params
    
    @property
    def is_stationary(self) -> bool:
        """Check if model is stationary (all bounded components)."""
        return all(spec.is_bounded for spec in self._components)
    
    @property
    def bounded_components(self) -> List[str]:
        """List of bounded component names."""
        return [name for name, spec in zip(self.component_names, self._components) 
                if spec.is_bounded]
    
    @property
    def unbounded_components(self) -> List[str]:
        """List of unbounded component names."""
        return [name for name, spec in zip(self.component_names, self._components) 
                if not spec.is_bounded]
    
    def set_params(self, params: np.ndarray) -> None:
        """Set parameter values.

        Parameters
        ----------
        params : array-like
            Parameter values in order: [comp1_params..., comp2_params..., nugget]

        Raises
        ------
        ValueError
            If wrong number of parameters or parameters fail validation.
        """
        params = np.asarray(params, dtype=float)
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")

        # validate parameters for each component before accepting
        idx = 0
        for i, spec in enumerate(self._components):
            n_params = len(spec.param_names)
            component_params = params[idx:idx + n_params]
            spec.validate(list(component_params))  # Raises if invalid
            idx += n_params

        self._params = params
    
    def get_component_params(self, component_idx: int) -> np.ndarray:
        """Get parameters for a specific component."""
        if self._params is None:
            raise ValueError("Parameters not set. Call set_params() first.")
        
        name = self.component_names[component_idx]
        # handle duplicate names
        count = self.component_names[:component_idx + 1].count(name)
        key = f"{name}_{component_idx}" if self.component_names.count(name) > 1 else name
        
        return self._params[self._param_slices[key]]
    
    def get_nugget(self) -> float:
        """Get nugget value (0 if no nugget)."""
        if not self.include_nugget:
            return 0.0
        if self._params is None:
            raise ValueError("Parameters not set.")
        return float(self._params[self._param_slices['nugget']])
    
    def __call__(self, h: np.ndarray) -> np.ndarray:
        """Evaluate variogram at lag distances.
        
        Parameters
        ----------
        h : array-like
            Lag distances.
        
        Returns
        -------
        gamma : ndarray
            Semivariance values.
        """
        if self._params is None:
            raise ValueError("Parameters not set. Call set_params() first.")
        
        h = np.asarray(h, dtype=float)
        gamma = np.zeros_like(h)
        
        # add each component
        for i, spec in enumerate(self._components):
            comp_params = self.get_component_params(i)
            gamma += spec.func(h, *comp_params)
        
        # add nugget
        if self.include_nugget:
            gamma += nugget_func(h, self.get_nugget())
        
        return gamma
    
    def evaluate_component(self, component_idx: int, h: np.ndarray) -> np.ndarray:
        """Evaluate a single component at lag distances."""
        if self._params is None:
            raise ValueError("Parameters not set.")
        
        h = np.asarray(h, dtype=float)
        spec = self._components[component_idx]
        comp_params = self.get_component_params(component_idx)
        return spec.func(h, *comp_params)
    
    def default_guess(self, lags: np.ndarray, variogram: np.ndarray) -> np.ndarray:
        """Generate default initial parameter guess.

        For multi-component models, spreads initial *practical* ranges across
        the lag span, then converts back to the model's native range parameter
        using ``practical_range_factor``.  For example, if the target practical
        range is 250 and the model is exponential (factor = 3), the range
        parameter guess becomes 250 / 3 ≈ 83.
        """
        guess = []
        n_bounded = len(self.bounded_components)
        max_gamma = np.nanmax(variogram)
        max_lag = np.nanmax(lags)

        for i, spec in enumerate(self._components):
            base_guess = spec.default_guess(lags, variogram)

            # adjust sill to share variance among components
            if spec.has_sill and n_bounded > 0:
                base_guess[0] = max_gamma * 0.8 / max(n_bounded, 1)

            # spread ranges for multi-component bounded models
            if spec.is_bounded and 'range' in spec.param_names:
                range_idx = spec.param_names.index('range')
                # target practical range distributed across the lag span
                range_factor = (i + 1) / (len(self._components) + 1)
                practical_range_target = max_lag * range_factor
                # convert from practical range to native range parameter
                prf = spec.practical_range_factor
                if prf is not None and prf > 0:
                    base_guess[range_idx] = practical_range_target / prf
                else:
                    base_guess[range_idx] = practical_range_target

            guess.extend(base_guess)

        if self.include_nugget:
            # estimate nugget from short-lag extrapolation rather than
            # a fixed fraction of the sill (more physically meaningful)
            nugget_guess = self._estimate_nugget_from_lags(lags, variogram)
            guess.append(nugget_guess)

        return np.array(guess)

    @staticmethod
    def _estimate_nugget_from_lags(
        lags: np.ndarray, variogram: np.ndarray
    ) -> float:
        """Estimate nugget from short-lag linear extrapolation to h=0.

        Uses the first few lag bins to fit γ(h) ≈ C₀ + bh and extrapolates
        the y-intercept as a nugget estimate.  Falls back to 10% of the max
        semivariance if there are too few points.
        """
        max_gamma = np.nanmax(variogram)
        n_extrap = min(5, max(2, len(lags) // 4))
        if n_extrap >= 2:
            short_lags = lags[:n_extrap]
            short_gamma = variogram[:n_extrap]
            try:
                _slope, intercept = np.polyfit(short_lags, short_gamma, 1)
                nugget_est = float(np.clip(intercept, 0, max_gamma * 0.5))
            except (np.linalg.LinAlgError, ValueError):
                nugget_est = max_gamma * 0.1
        else:
            nugget_est = max_gamma * 0.1
        return nugget_est
    
    def bounds(self, lags: np.ndarray, variogram: np.ndarray) -> Tuple[List[float], List[float]]:
        """Generate parameter bounds.

        Nugget is capped at 50% of the maximum observed semivariance.
        A nugget exceeding half the total sill is unusual in practice and
        often indicates fitting artefacts rather than genuine micro-scale
        variation (Cressie, 1993, §2.4).
        """
        lower, upper = [], []

        for spec in self._components:
            lb, ub = spec.bounds(lags, variogram)
            lower.extend(lb)
            upper.extend(ub)

        if self.include_nugget:
            max_gamma = np.nanmax(variogram)
            # cap nugget at 50% of observed max semivariance
            lower.append(0)
            upper.append(max_gamma * 0.5)

        return (lower, upper)
    
    def get_total_sill(self) -> Optional[float]:
        """Get total sill (sum of bounded component sills + nugget).
        
        Returns None if model contains unbounded components.
        """
        if not self.is_stationary:
            return None
        
        if self._params is None:
            raise ValueError("Parameters not set.")
        
        total = self.get_nugget()
        for i, spec in enumerate(self._components):
            if spec.has_sill:
                params = self.get_component_params(i)
                total += params[0]  # Sill is always first parameter
        
        return total
    
    def get_stationary_sill(self) -> float:
        """Get sill from bounded components only (+ nugget)."""
        if self._params is None:
            raise ValueError("Parameters not set.")
        
        total = self.get_nugget()
        for i, spec in enumerate(self._components):
            if spec.is_bounded and spec.has_sill:
                params = self.get_component_params(i)
                total += params[0]
        
        return total
    
    def decompose_variance(self, reference_lag: Optional[float] = None) -> Dict[str, float]:
        """Decompose variance by component.
        
        Parameters
        ----------
        reference_lag : float, optional
            For unbounded components, compute variance contribution at this lag.
            If None, uses the maximum fitted parameter range or 1000.
        
        Returns
        -------
        decomposition : dict
            Dictionary with keys:
            - 'nugget': Nugget variance
            - '{component_name}': Variance from each component
            - 'total_stationary': Sum of bounded components + nugget
            - 'total_at_reference': Total variance at reference lag (if unbounded)
            - 'reference_lag': The reference lag used
        """
        if self._params is None:
            raise ValueError("Parameters not set.")
        
        result = {'nugget': self.get_nugget()}
        
        # determine reference lag for unbounded
        if reference_lag is None:
            reference_lag = 1000  # Default
            for i, spec in enumerate(self._components):
                if spec.is_bounded and 'range' in spec.param_names:
                    params = self.get_component_params(i)
                    range_idx = spec.param_names.index('range')
                    reference_lag = max(reference_lag, params[range_idx] * 3)
        
        result['reference_lag'] = reference_lag
        
        stationary_total = result['nugget']
        
        for i, spec in enumerate(self._components):
            name = self.component_names[i]
            key = f"{name}_{i}" if self.component_names.count(name) > 1 else name
            
            if spec.is_bounded and spec.has_sill:
                params = self.get_component_params(i)
                variance = params[0]  # Sill
                result[key] = variance
                stationary_total += variance
            elif not spec.is_bounded:
                # compute variance at reference lag
                h_ref = np.array([reference_lag])
                variance_at_ref = spec.func(h_ref, *self.get_component_params(i))[0]
                result[key] = variance_at_ref
                result[f'{key}_is_nonstationary'] = True
        
        result['total_stationary'] = stationary_total
        
        # total at reference lag
        h_ref = np.array([reference_lag])
        result['total_at_reference'] = float(self(h_ref)[0])
        
        return result
    
    def get_covariance_function(self) -> Optional[Callable]:
        """Get covariance function C(h) = sill - γ(h).
        
        Only valid for stationary models.
        
        Returns
        -------
        cov_func : Callable or None
            Covariance function, or None if non-stationary.
        """
        if not self.is_stationary:
            return None
        
        sill = self.get_total_sill()
        
        def cov_func(h):
            return sill - self(h)
        
        return cov_func
    
    def structural_description(self) -> str:
        """Model structure without parameter values, for grouping.

        Returns a string like ``'spherical + exponential + nugget'``
        that identifies the model *family* regardless of fitted
        parameter values.
        """
        parts = [spec.name for spec in self._components]
        if self.include_nugget:
            parts.append('nugget')
        return ' + '.join(parts)

    def description(self) -> str:
        """Generate human-readable model description."""
        parts = []
        
        if self.include_nugget:
            parts.append(f"Nugget: C₀ = {self.get_nugget():.4f}")
        
        for i, spec in enumerate(self._components):
            params = self.get_component_params(i)
            param_str = ", ".join(
                f"{name}={val:.4f}" 
                for name, val in zip(spec.param_names, params)
            )
            bounded_str = "" if spec.is_bounded else " [NON-STATIONARY]"
            parts.append(f"{spec.name.capitalize()}{bounded_str}: {param_str}")
        
        if self.is_stationary:
            parts.append(f"Total sill: {self.get_total_sill():.4f}")
        else:
            decomp = self.decompose_variance()
            parts.append(f"Stationary sill: {decomp['total_stationary']:.4f}")
            parts.append(f"Total at reference ({decomp['reference_lag']:.0f}): "
                        f"{decomp['total_at_reference']:.4f}")
        
        return "\n".join(parts)
