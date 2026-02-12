"""variogram-based regional uncertainty propagation via Krige's relation.

references:
- Chilès & Delfiner (2012). Geostatistics: Modeling Spatial Uncertainty.
- Hugonnet et al. (2022). IEEE JSTARS, 15, 6456-6472.
"""

from __future__ import annotations

import math
import warnings
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box, Point
from shapely.ops import unary_union
from pathlib import Path
import geopandas as gpd
from typing import Optional, Callable, Tuple, List, Dict, Any

from rasterio.features import geometry_mask

from .variogram import (
    RasterDataHandler,
    VariogramAnalysis,
    FittedVariogramModel,
    VariogramModelSelector,
)


class RegionalUncertaintyEstimator:
    """
    Estimate regional uncertainty σ_A over a polygon.

    Supports both legacy spherical models and the new model-agnostic
    FittedVariogramModel/CompositeVariogramModel approach.
    
    Computes central, min, and max uncertainty estimates based on
    bootstrap parameter percentiles (16th, 50th, 84th).
    """

    @staticmethod
    def _as_multipolygon(geom):
        if isinstance(geom, MultiPolygon):
            return geom
        elif isinstance(geom, Polygon):
            return MultiPolygon([geom])
        raise TypeError(f"Geometry must be Polygon or MultiPolygon, not {type(geom).__name__}.")

    def __init__(
        self,
        raster_data_handler: RasterDataHandler,
        variogram_analysis: VariogramAnalysis,
        area_of_interest,
        stable_geoms=None,
        unstable_geoms=None,
        derive_stable_from_unstable: bool = True,
        fitted_model: Optional[FittedVariogramModel] = None,
        use_bma: bool = False,
    ):
        self.raster_data_handler = raster_data_handler
        self.variogram_analysis = variogram_analysis

        # --- Setup gamma functions (central, min, max) ---
        self._setup_gamma_functions(
            variogram_analysis, fitted_model, use_bma
        )

        # --- Resolve Polygon of interest ---
        if isinstance(area_of_interest, (str, Path)):
            gdf = gpd.read_file(area_of_interest)
            if gdf.empty:
                raise ValueError(f"No geometries in file: {area_of_interest}")
            polygon = gdf.unary_union
        elif isinstance(area_of_interest, (Polygon, MultiPolygon)):
            polygon = area_of_interest
        else:
            raise TypeError("area_of_interest must be file path or Polygon/MultiPolygon.")

        if isinstance(polygon, MultiPolygon):
            polygon = unary_union(polygon)
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            raise ValueError("Area of interest must be a valid non-empty Polygon.")

        self.polygon = polygon
        self.area = float(polygon.area)

        # --- Stable / unstable geometries ---
        self.stable_geom = None
        self.unstable_geom = None

        if unstable_geoms is not None:
            if isinstance(unstable_geoms, (Polygon, MultiPolygon)):
                self.unstable_geom = unstable_geoms
            else:
                self.unstable_geom = unary_union(list(unstable_geoms))

        if stable_geoms is not None:
            if isinstance(stable_geoms, (Polygon, MultiPolygon)):
                self.stable_geom = self._as_multipolygon(stable_geoms)
            else:
                self.stable_geom = self._as_multipolygon(unary_union(list(stable_geoms)))

        if self.stable_geom is None and self.unstable_geom is not None and derive_stable_from_unstable:
            raster_data_handler.get_detailed_area()
            footprint = raster_data_handler.merged_geom or box(*raster_data_handler.bounds)
            stable = footprint.difference(self.unstable_geom)
            if isinstance(stable, (Polygon, MultiPolygon)):
                self.stable_geom = self._as_multipolygon(stable)

        # --- Results storage ---
        self._init_result_storage()

    def _setup_gamma_functions(
        self,
        va: VariogramAnalysis,
        fitted_model: Optional[FittedVariogramModel],
        use_bma: bool,
    ) -> None:
        """Setup gamma functions for central, min, max parameter estimates."""
        
        # initialize all gamma functions to None
        self.gamma_func = None
        self.gamma_func_min = None
        self.gamma_func_max = None
        
        # component-wise gamma functions
        self.gamma_funcs_components: List[Optional[Callable]] = [None, None, None]
        self.gamma_funcs_components_min: List[Optional[Callable]] = [None, None, None]
        self.gamma_funcs_components_max: List[Optional[Callable]] = [None, None, None]
        
        # total variance (sill + nugget)
        self.sigma2 = None
        self.sigma2_min = None
        self.sigma2_max = None

        if fitted_model is not None:
            # validate model is suitable for uncertainty propagation
            if not fitted_model.composite_model.is_stationary:
                unbounded = fitted_model.composite_model.unbounded_components
                raise ValueError(
                    f"Non-stationary variogram models ({unbounded}) cannot be used "
                    f"for uncertainty propagation because they have infinite variance. "
                    f"Consider detrending your data or using a bounded model "
                    f"(spherical, exponential, gaussian, matern, damped_hole_effect)."
                )

            # warn about Gaussian without nugget (numerical instability risk)
            cm = fitted_model.composite_model
            has_gaussian = 'gaussian' in cm.component_names
            has_nugget = cm.include_nugget and cm.get_nugget() > 0
            if has_gaussian and not has_nugget:
                warnings.warn(
                    "Gaussian variogram model without nugget may cause numerical "
                    "instability. The model implies infinite differentiability at h=0, "
                    "which can lead to ill-conditioned covariance matrices. Consider "
                    "adding a small nugget effect.",
                    UserWarning,
                    stacklevel=2
                )

            # new approach: use FittedVariogramModel
            self.gamma_func = fitted_model.predict
            self.sigma2 = fitted_model.composite_model.get_total_sill()
            
            # min/max from bootstrap if available
            if fitted_model.param_samples is not None and len(fitted_model.param_samples) > 0:
                self._setup_minmax_from_bootstrap(fitted_model)
            else:
                self.gamma_func_min = self.gamma_func
                self.gamma_func_max = self.gamma_func
                self.sigma2_min = self.sigma2
                self.sigma2_max = self.sigma2
                
        elif use_bma and hasattr(va, 'model_selector') and va.model_selector is not None:
            # validate BMA models are all stationary
            selector = va.model_selector
            for m in selector.fitted_models:
                if not m.composite_model.is_stationary:
                    unbounded = m.composite_model.unbounded_components
                    raise ValueError(
                        f"BMA ensemble contains non-stationary model ({unbounded}). "
                        f"Non-stationary models cannot be used for uncertainty propagation. "
                        f"Exclude non-stationary models from the ensemble."
                    )

            # bayesian Model Averaging
            self.gamma_func = va.get_bma_variogram_function()
            self.sigma2 = sum(
                w * m.composite_model.get_total_sill()
                for m, w in zip(selector.fitted_models, selector.model_weights)
            )
            # bMA doesn't have simple min/max - use same for all
            self.gamma_func_min = self.gamma_func
            self.gamma_func_max = self.gamma_func
            self.sigma2_min = self.sigma2
            self.sigma2_max = self.sigma2
            
        elif va.best_model_func is not None:
            # legacy spherical model
            self._setup_legacy_gamma(va)
        else:
            raise ValueError("No variogram model available. Fit a model first.")

    def _setup_minmax_from_bootstrap(self, fitted_model: FittedVariogramModel) -> None:
        """Store bootstrap samples for sample-based uncertainty propagation.

        Rather than constructing min/max gamma functions from marginal
        parameter percentiles (which ignores parameter correlations and
        can produce physically impossible models where σ² < γ(h)),
        we store the full bootstrap ensemble.  At propagation time,
        ``_propagate_bootstrap_uncertainty`` evaluates the Monte Carlo
        integral for each joint parameter sample and takes percentiles
        of the *output* distribution — the correct approach when
        parameters are correlated (Diggle & Ribeiro, 2007, §6.4).

        For the min/max gamma functions used by component-wise
        decomposition and direct calls, we set them equal to the central
        estimate.  The actual min/max uncertainty bounds are computed
        during propagation via the bootstrap ensemble.

        References
        ----------
        Diggle, P.J. & Ribeiro, P.J. (2007). Model-based Geostatistics.
            Springer.  Section 6.4 — prediction under parameter uncertainty.

        Christensen, R. (2011). Plane Answers to Complex Questions.
            Springer.  Section 15.2 — propagation of parameter uncertainty.
        """
        samples = fitted_model.param_samples
        model = fitted_model.composite_model
        central_params = fitted_model.params

        # Store bootstrap ensemble for propagation-time evaluation
        self._bootstrap_samples = samples
        self._bootstrap_model = model
        self._bootstrap_central_params = central_params.copy()

        # For min/max gamma functions (used in component-wise calls and
        # as fallbacks), use central estimate — the real bounds come
        # from _propagate_bootstrap_uncertainty at propagation time.
        self.gamma_func_min = self.gamma_func
        self.gamma_func_max = self.gamma_func

        # Compute sigma2 min/max from the bootstrap ensemble properly.
        # For each sample, compute total sill = sum(component sills) + nugget.
        sigma2_samples = self._compute_sigma2_from_samples(model, samples)
        self.sigma2_min = float(np.percentile(sigma2_samples, 16))
        self.sigma2_max = float(np.percentile(sigma2_samples, 84))

    @staticmethod
    def _compute_sigma2_from_samples(
        model: 'CompositeVariogramModel',
        samples: np.ndarray,
    ) -> np.ndarray:
        """Compute total sill for each bootstrap parameter sample.

        Parses the composite model parameter structure to sum
        component sills + nugget for each row in ``samples``.

        Returns
        -------
        sigma2_samples : ndarray, shape (n_samples,)
        """
        n = len(samples)
        sigma2 = np.zeros(n)

        # Sum sills from bounded components
        idx = 0
        for spec in model._components:
            n_p = len(spec.param_names)
            if spec.has_sill:
                # sill is always the first parameter
                sigma2 += samples[:, idx]
            idx += n_p

        # Add nugget (always last parameter when present)
        if model.include_nugget:
            sigma2 += samples[:, -1]

        return sigma2

    def _propagate_bootstrap_uncertainty(
        self,
        domain,
        n_pairs: int = 200_000,
        seed: Optional[int] = None,
        n_boot_eval: int = 100,
    ) -> Tuple[float, float]:
        """Propagate parameter uncertainty through Monte Carlo integration.

        For each of ``n_boot_eval`` bootstrap parameter samples, evaluates
        the full Krige's-relation Monte Carlo integral:

            σ_A(θ_k) = sqrt( E[C_k(||X−Y||)] )

        where C_k(h) = σ²(θ_k) − γ(h; θ_k) is the covariance function
        under parameter vector θ_k.  Returns the 16th and 84th percentiles
        of the resulting σ_A distribution.

        This correctly handles parameter correlations because each θ_k is
        a joint sample from the bootstrap — no independent-marginal
        assumption is needed.

        Parameters
        ----------
        domain : Polygon
            Spatial domain for Monte Carlo integration.
        n_pairs : int
            Number of point pairs for MC integration.
        seed : int, optional
            Random seed for reproducible point sampling.
        n_boot_eval : int
            Number of bootstrap samples to evaluate (subsampled from
            the full ensemble for speed).

        Returns
        -------
        sigma_a_p16 : float
            16th percentile of σ_A distribution (lower bound).
        sigma_a_p84 : float
            84th percentile of σ_A distribution (upper bound).

        References
        ----------
        Diggle, P.J. & Ribeiro, P.J. (2007). Model-based Geostatistics.
            Springer.  Section 6.4.

        Marchetti, Y. et al. (2012). Methods for summarizing and
            communicating uncertainty in spatial prediction.
            doi:10.1002/env.2149
        """
        if not hasattr(self, '_bootstrap_samples') or self._bootstrap_samples is None:
            raise ValueError("No bootstrap samples. Fit model with bootstrap first.")

        samples = self._bootstrap_samples
        model = self._bootstrap_model
        central_params = self._bootstrap_central_params

        # Subsample bootstrap ensemble if large
        rng = np.random.default_rng(seed)
        n_total = len(samples)
        if n_total > n_boot_eval:
            boot_idx = rng.choice(n_total, n_boot_eval, replace=False)
        else:
            boot_idx = np.arange(n_total)
            n_boot_eval = n_total

        # Pre-generate point pairs (shared across all bootstrap evaluations)
        # This dramatically reduces cost vs. regenerating per sample.
        pair_seed = rng.integers(0, 2**31)
        h_pairs = self._sample_pair_distances(domain, n_pairs, pair_seed)

        # Evaluate MC integral for each bootstrap sample
        sigma_a_values = np.empty(n_boot_eval)
        for k, bi in enumerate(boot_idx):
            params_k = samples[bi]
            model.set_params(params_k)

            # Compute sigma2 for this sample
            sigma2_k = model.get_total_sill()
            if sigma2_k is None or sigma2_k <= 0:
                sigma_a_values[k] = np.nan
                continue

            # Evaluate gamma at pair distances
            gamma_k = model(h_pairs)

            # C(h) = sigma2 - gamma(h)
            cov_k = sigma2_k - gamma_k
            var_mean_k = float(np.mean(cov_k))

            sigma_a_values[k] = math.sqrt(max(var_mean_k, 0.0))

        # Restore central params
        model.set_params(central_params)

        # Remove failed evaluations
        valid = np.isfinite(sigma_a_values)
        if not np.any(valid):
            warnings.warn(
                "All bootstrap propagation evaluations failed. "
                "Falling back to central estimate for bounds.",
                UserWarning,
                stacklevel=2,
            )
            return 0.0, 0.0

        sigma_a_valid = sigma_a_values[valid]
        return (
            float(np.percentile(sigma_a_valid, 16)),
            float(np.percentile(sigma_a_valid, 84)),
        )

    def _sample_pair_distances(
        self,
        domain,
        n_pairs: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample random point-pair distances within a domain.

        Re-used across bootstrap evaluations so the MC noise is
        identical for every parameter sample (variance reduction).

        Returns
        -------
        h : ndarray, shape (n_pairs,)
            Euclidean distances between randomly paired points.
        """
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
        X, Y = pts[:n_pairs], pts[n_pairs:2 * n_pairs]
        return np.linalg.norm(X - Y, axis=1)

    def _setup_legacy_gamma(self, va: VariogramAnalysis) -> None:
        """Setup gamma functions from legacy spherical model."""
        # central estimates
        self.sills = np.array(va.sills, dtype=float)
        self.ranges = np.array(va.ranges, dtype=float)
        self.nugget = getattr(va, 'best_nugget', 0.0) or 0.0
        self.sigma2 = self.nugget + float(np.sum(self.sills))

        # min/max estimates
        self.sills_min = np.array(getattr(va, 'sills_min', self.sills), dtype=float)
        self.sills_max = np.array(getattr(va, 'sills_max', self.sills), dtype=float)
        self.ranges_min = np.array(getattr(va, 'ranges_min', self.ranges), dtype=float)
        self.ranges_max = np.array(getattr(va, 'ranges_max', self.ranges), dtype=float)
        self.nugget_min = getattr(va, 'min_nugget', self.nugget) or 0.0
        self.nugget_max = getattr(va, 'max_nugget', self.nugget) or 0.0

        self.sigma2_min = self.nugget_min + float(np.sum(self.sills_min))
        self.sigma2_max = self.nugget_max + float(np.sum(self.sills_max))

        has_nugget = va.best_model_config.get('nugget', False)
        model_func = va.best_model_func

        # central gamma function
        params = list(self.sills) + list(self.ranges)
        if has_nugget:
            params.append(self.nugget)
        self.gamma_func = lambda h, p=params: model_func(np.asarray(h, dtype=float), *p)

        # min gamma function
        params_min = list(self.sills_min) + list(self.ranges_min)
        if has_nugget:
            params_min.append(self.nugget_min)
        self.gamma_func_min = lambda h, p=params_min: model_func(np.asarray(h, dtype=float), *p)

        # max gamma function
        params_max = list(self.sills_max) + list(self.ranges_max)
        if has_nugget:
            params_max.append(self.nugget_max)
        self.gamma_func_max = lambda h, p=params_max: model_func(np.asarray(h, dtype=float), *p)

        # component-wise gamma functions (for multi-component models)
        n_components = len(self.sills)
        for i in range(min(n_components, 3)):
            # central
            p = [self.sills[i], self.ranges[i]]
            if has_nugget:
                p.append(self.nugget)
            self.gamma_funcs_components[i] = lambda h, p=p: model_func(np.asarray(h, dtype=float), *p)
            
            # min
            p_min = [self.sills_min[i], self.ranges_min[i]]
            if has_nugget:
                p_min.append(self.nugget_min)
            self.gamma_funcs_components_min[i] = lambda h, p=p_min: model_func(np.asarray(h, dtype=float), *p)
            
            # max
            p_max = [self.sills_max[i], self.ranges_max[i]]
            if has_nugget:
                p_max.append(self.nugget_max)
            self.gamma_funcs_components_max[i] = lambda h, p=p_max: model_func(np.asarray(h, dtype=float), *p)

    def _init_result_storage(self) -> None:
        """Initialize all result storage attributes."""
        # uncorrelated
        self.sigma0_uncorrelated = None
        self.mean_uncorrelated_polygon = None
        self.mean_uncorrelated_raster = None

        # correlated - Polygon (central, min, max)
        self.mean_correlated_polygon = None
        self.mean_correlated_polygon_min = None
        self.mean_correlated_polygon_max = None

        # correlated - Raster (central, min, max)
        self.mean_correlated_raster = None
        self.mean_correlated_raster_min = None
        self.mean_correlated_raster_max = None

        # component-wise - Polygon
        self.mean_correlated_components_polygon: List[Optional[float]] = [None, None, None]
        self.mean_correlated_components_polygon_min: List[Optional[float]] = [None, None, None]
        self.mean_correlated_components_polygon_max: List[Optional[float]] = [None, None, None]

        # component-wise - Raster
        self.mean_correlated_components_raster: List[Optional[float]] = [None, None, None]
        self.mean_correlated_components_raster_min: List[Optional[float]] = [None, None, None]
        self.mean_correlated_components_raster_max: List[Optional[float]] = [None, None, None]

        # total uncertainty (central, min, max)
        self.total_uncertainty_polygon = None
        self.total_uncertainty_polygon_min = None
        self.total_uncertainty_polygon_max = None

        self.total_uncertainty_raster = None
        self.total_uncertainty_raster_min = None
        self.total_uncertainty_raster_max = None

    def covariance(self, h: np.ndarray, sigma2: float, gamma_func: Callable) -> np.ndarray:
        """C(h) = σ² - γ(h)"""
        return sigma2 - gamma_func(h)

    def calc_mean_uncorrelated(self, use_stable_areas: bool = True) -> None:
        """Compute uncorrelated noise contribution to mean uncertainty."""
        da = self.raster_data_handler.rioxarray_obj
        if da is None:
            raise RuntimeError("Call load_raster() first.")

        arr = da.values
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        arr = np.asarray(arr, dtype=float)

        valid_mask = np.isfinite(arr)
        mask_for_sigma = valid_mask.copy()

        if use_stable_areas and self.stable_geom is not None:
            transform = da.rio.transform()
            stable_outside = geometry_mask(
                [self.stable_geom],
                out_shape=arr.shape,
                transform=transform,
                invert=False,
            )
            mask_for_sigma = valid_mask & ~stable_outside
            if not np.any(mask_for_sigma):
                mask_for_sigma = valid_mask

        values = arr[mask_for_sigma]
        if values.size == 0:
            raise RuntimeError("No valid values for uncorrelated sigma estimation.")

        sigma0 = float(np.sqrt(np.mean(values ** 2)))
        self.sigma0_uncorrelated = sigma0

        res = float(self.raster_data_handler.resolution)
        cell_area = res ** 2

        # Polygon mean
        N_poly = max(self.area / cell_area, 1.0)
        self.mean_uncorrelated_polygon = sigma0 / math.sqrt(N_poly)

        N_raster = int(valid_mask.sum())
        if N_raster > 0:
            self.mean_uncorrelated_raster = sigma0 / math.sqrt(N_raster)

    def estimate_std_mean_monte_carlo(
        self,
        domain: Polygon,
        gamma_func: Callable,
        sigma2: float,
        n_pairs: int = 200_000,
        seed: Optional[int] = None,
    ) -> float:
        """
        Estimate std(mean) via Monte Carlo integration of covariance.

        Returns sqrt(Var(mean)) = sqrt(E[C(||X-Y||)])
        """
        rng = np.random.default_rng(seed)
        minx, miny, maxx, maxy = domain.bounds

        # sample points inside domain
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
        X, Y = pts[:n_pairs], pts[n_pairs:2*n_pairs]

        h = np.linalg.norm(X - Y, axis=1)
        cov = self.covariance(h, sigma2, gamma_func)
        var_mean = float(np.mean(cov))

        return 0.0 if var_mean < 0 else math.sqrt(var_mean)

    def calc_mean_correlated_polygon(
        self,
        n_pairs: int = 200_000,
        seed: Optional[int] = None,
        n_boot_eval: int = 100,
    ) -> None:
        """Compute correlated uncertainty for polygon mean (central, min, max).

        When bootstrap samples are available, min/max are computed by
        propagating each joint parameter sample through the full MC
        integral (correct treatment of parameter correlations).
        Otherwise falls back to independent min/max gamma functions.

        Parameters
        ----------
        n_pairs : int
            Number of Monte Carlo point pairs.
        seed : int, optional
            Random seed for reproducibility.
        n_boot_eval : int
            Number of bootstrap samples to evaluate for bounds.
        """
        # central
        if self.gamma_func is not None:
            self.mean_correlated_polygon = self.estimate_std_mean_monte_carlo(
                self.polygon, self.gamma_func, self.sigma2, n_pairs, seed
            )

        # min/max via bootstrap propagation (preferred) or legacy fallback
        has_bootstrap = (
            hasattr(self, '_bootstrap_samples')
            and self._bootstrap_samples is not None
        )
        if has_bootstrap:
            p16, p84 = self._propagate_bootstrap_uncertainty(
                self.polygon, n_pairs=n_pairs, seed=seed,
                n_boot_eval=n_boot_eval,
            )
            self.mean_correlated_polygon_min = p16
            self.mean_correlated_polygon_max = p84
        else:
            # Legacy path: independent min/max gamma functions
            if self.gamma_func_min is not None:
                self.mean_correlated_polygon_min = self.estimate_std_mean_monte_carlo(
                    self.polygon, self.gamma_func_min, self.sigma2_min, n_pairs, seed
                )
            if self.gamma_func_max is not None:
                self.mean_correlated_polygon_max = self.estimate_std_mean_monte_carlo(
                    self.polygon, self.gamma_func_max, self.sigma2_max, n_pairs, seed
                )

        # component-wise (central only; bootstrap handles min/max above)
        for i in range(3):
            if self.gamma_funcs_components[i] is not None:
                self.mean_correlated_components_polygon[i] = self.estimate_std_mean_monte_carlo(
                    self.polygon, self.gamma_funcs_components[i], self.sigma2, n_pairs, seed
                )
            if self.gamma_funcs_components_min[i] is not None:
                self.mean_correlated_components_polygon_min[i] = self.estimate_std_mean_monte_carlo(
                    self.polygon, self.gamma_funcs_components_min[i], self.sigma2_min, n_pairs, seed
                )
            if self.gamma_funcs_components_max[i] is not None:
                self.mean_correlated_components_polygon_max[i] = self.estimate_std_mean_monte_carlo(
                    self.polygon, self.gamma_funcs_components_max[i], self.sigma2_max, n_pairs, seed
                )

    def calc_mean_correlated_raster(
        self,
        n_pairs: int = 200_000,
        seed: Optional[int] = None,
        n_boot_eval: int = 100,
    ) -> None:
        """Compute correlated uncertainty for raster mean (central, min, max).

        Parameters
        ----------
        n_pairs : int
            Number of Monte Carlo point pairs.
        seed : int, optional
            Random seed for reproducibility.
        n_boot_eval : int
            Number of bootstrap samples to evaluate for bounds.
        """
        self.raster_data_handler.get_detailed_area()
        raster_geom = self.raster_data_handler.merged_geom or box(*self.raster_data_handler.bounds)

        # central
        if self.gamma_func is not None:
            self.mean_correlated_raster = self.estimate_std_mean_monte_carlo(
                raster_geom, self.gamma_func, self.sigma2, n_pairs, seed
            )

        # min/max via bootstrap propagation (preferred) or legacy fallback
        has_bootstrap = (
            hasattr(self, '_bootstrap_samples')
            and self._bootstrap_samples is not None
        )
        if has_bootstrap:
            p16, p84 = self._propagate_bootstrap_uncertainty(
                raster_geom, n_pairs=n_pairs, seed=seed,
                n_boot_eval=n_boot_eval,
            )
            self.mean_correlated_raster_min = p16
            self.mean_correlated_raster_max = p84
        else:
            # Legacy path
            if self.gamma_func_min is not None:
                self.mean_correlated_raster_min = self.estimate_std_mean_monte_carlo(
                    raster_geom, self.gamma_func_min, self.sigma2_min, n_pairs, seed
                )
            if self.gamma_func_max is not None:
                self.mean_correlated_raster_max = self.estimate_std_mean_monte_carlo(
                    raster_geom, self.gamma_func_max, self.sigma2_max, n_pairs, seed
                )

        # component-wise
        for i in range(3):
            if self.gamma_funcs_components[i] is not None:
                self.mean_correlated_components_raster[i] = self.estimate_std_mean_monte_carlo(
                    raster_geom, self.gamma_funcs_components[i], self.sigma2, n_pairs, seed
                )
            if self.gamma_funcs_components_min[i] is not None:
                self.mean_correlated_components_raster_min[i] = self.estimate_std_mean_monte_carlo(
                    raster_geom, self.gamma_funcs_components_min[i], self.sigma2_min, n_pairs, seed
                )
            if self.gamma_funcs_components_max[i] is not None:
                self.mean_correlated_components_raster_max[i] = self.estimate_std_mean_monte_carlo(
                    raster_geom, self.gamma_funcs_components_max[i], self.sigma2_max, n_pairs, seed
                )

    def calc_total_uncertainty(
        self,
        n_pairs: int = 200_000,
        seed: Optional[int] = None,
        use_stable_areas: bool = True,
        n_boot_eval: int = 100,
    ) -> None:
        """Compute full uncertainty budget (uncorrelated + correlated).

        Parameters
        ----------
        n_pairs : int
            Number of Monte Carlo point pairs.
        seed : int, optional
            Random seed for reproducibility.
        use_stable_areas : bool
            Whether to use stable areas for uncorrelated noise estimation.
        n_boot_eval : int
            Number of bootstrap samples to evaluate for uncertainty bounds.
        """
        self.calc_mean_uncorrelated(use_stable_areas=use_stable_areas)
        self.calc_mean_correlated_polygon(
            n_pairs=n_pairs, seed=seed, n_boot_eval=n_boot_eval
        )
        self.calc_mean_correlated_raster(
            n_pairs=n_pairs, seed=seed, n_boot_eval=n_boot_eval
        )

        def quadrature(uncorr, corr):
            if uncorr is not None and corr is not None:
                return math.sqrt(uncorr**2 + corr**2)
            return None

        # Polygon totals
        self.total_uncertainty_polygon = quadrature(
            self.mean_uncorrelated_polygon, self.mean_correlated_polygon
        )
        self.total_uncertainty_polygon_min = quadrature(
            self.mean_uncorrelated_polygon, self.mean_correlated_polygon_min
        )
        self.total_uncertainty_polygon_max = quadrature(
            self.mean_uncorrelated_polygon, self.mean_correlated_polygon_max
        )

        self.total_uncertainty_raster = quadrature(
            self.mean_uncorrelated_raster, self.mean_correlated_raster
        )
        self.total_uncertainty_raster_min = quadrature(
            self.mean_uncorrelated_raster, self.mean_correlated_raster_min
        )
        self.total_uncertainty_raster_max = quadrature(
            self.mean_uncorrelated_raster, self.mean_correlated_raster_max
        )

    def summary(self) -> str:
        """Return formatted summary of results."""
        def fmt_triple(name: str, val: float, val_min: float, val_max: float) -> str:
            parts = []
            if val is not None:
                parts.append(f"{val:.6f}")
            if val_min is not None:
                parts.append(f"min: {val_min:.6f}")
            if val_max is not None:
                parts.append(f"max: {val_max:.6f}")
            return f"{name}: {'; '.join(parts)}" if parts else ""

        lines = [
            "=" * 70,
            "REGIONAL UNCERTAINTY SUMMARY",
            "=" * 70,
            f"Polygon area: {self.area:.2f} m²",
            fmt_triple("Total variance (σ²)", self.sigma2, self.sigma2_min, self.sigma2_max),
            "",
        ]

        if self.sigma0_uncorrelated:
            lines.append(f"Uncorrelated σ₀: {self.sigma0_uncorrelated:.6f}")
        if self.mean_uncorrelated_polygon:
            lines.append(f"Uncorrelated (polygon mean): {self.mean_uncorrelated_polygon:.6f}")

        lines.append("")
        lines.append("POLYGON CORRELATED UNCERTAINTY:")
        lines.append(fmt_triple("  Total", self.mean_correlated_polygon,
                                self.mean_correlated_polygon_min, self.mean_correlated_polygon_max))
        for i in range(3):
            if self.mean_correlated_components_polygon[i] is not None:
                lines.append(fmt_triple(f"  Component {i+1}",
                                        self.mean_correlated_components_polygon[i],
                                        self.mean_correlated_components_polygon_min[i],
                                        self.mean_correlated_components_polygon_max[i]))

        lines.append("")
        lines.append("POLYGON TOTAL UNCERTAINTY:")
        lines.append(fmt_triple("  Total", self.total_uncertainty_polygon,
                                self.total_uncertainty_polygon_min, self.total_uncertainty_polygon_max))

        lines.append("")
        lines.append("-" * 70)
        lines.append("RASTER CORRELATED UNCERTAINTY:")
        if self.mean_uncorrelated_raster:
            lines.append(f"  Uncorrelated (raster mean): {self.mean_uncorrelated_raster:.6f}")
        lines.append(fmt_triple("  Total", self.mean_correlated_raster,
                                self.mean_correlated_raster_min, self.mean_correlated_raster_max))

        lines.append("")
        lines.append("RASTER TOTAL UNCERTAINTY:")
        lines.append(fmt_triple("  Total", self.total_uncertainty_raster,
                                self.total_uncertainty_raster_min, self.total_uncertainty_raster_max))

        lines.append("=" * 70)
        return "\n".join(lines)


class DerivativeUncertaintyEstimator:
    """
    Estimate uncertainty for spatial derivatives (slope, curvature).

    For a linear filter with kernel K:
        Var(O) = Σᵢ Σⱼ Kᵢ Kⱼ C(||xᵢ - xⱼ||)

    References
    ----------
    Heuvelink, G.B.M. (1998). Error Propagation in Environmental Modelling
    with GIS. Taylor & Francis, Chapter 7.
    """

    KERNELS = {
        'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0,
        'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0,
        'laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    }

    def __init__(
        self,
        gamma_func: Callable[[np.ndarray], np.ndarray],
        sill: float,
        resolution: float,
    ):
        self.gamma_func = gamma_func
        self.sill = sill
        self.resolution = resolution

    def covariance(self, h: np.ndarray) -> np.ndarray:
        """C(h) = σ² - γ(h)"""
        return self.sill - self.gamma_func(np.asarray(h, dtype=float))

    def kernel_variance(self, kernel: np.ndarray) -> float:
        """Var(O) = Σᵢ Σⱼ Kᵢ Kⱼ C(||xᵢ - xⱼ||)"""
        rows, cols = kernel.shape
        cy, cx = rows // 2, cols // 2

        positions = [(j - cx, i - cy) for i in range(rows) for j in range(cols)]
        weights = kernel.flatten()
        n = len(weights)

        total = 0.0
        for i in range(n):
            for j in range(n):
                dx = (positions[i][0] - positions[j][0]) * self.resolution
                dy = (positions[i][1] - positions[j][1]) * self.resolution
                h = np.sqrt(dx**2 + dy**2)
                total += weights[i] * weights[j] * self.covariance(np.array([h]))[0]

        return max(0.0, total)

    def slope_uncertainty(self) -> Tuple[float, float, float]:
        """Returns (std_dzdx, std_dzdy, std_slope_magnitude)."""
        var_x = self.kernel_variance(self.KERNELS['sobel_x']) / self.resolution**2
        var_y = self.kernel_variance(self.KERNELS['sobel_y']) / self.resolution**2
        return np.sqrt(var_x), np.sqrt(var_y), np.sqrt(var_x + var_y)

    def curvature_uncertainty(self) -> float:
        """Returns std(∇²z)."""
        var = self.kernel_variance(self.KERNELS['laplacian']) / self.resolution**4
        return np.sqrt(var)
