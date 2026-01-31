"""Propagate variogram-based uncertainty to specific regions.

This module implements model-agnostic uncertainty propagation based on Krige's
Relation, which guarantees that Monte Carlo integration works for ANY valid
variogram function (spherical, exponential, Gaussian, Matérn, etc.).

Mathematical Foundation
-----------------------
For a second-order stationary random field Z(x) with covariance C(h) = σ² - γ(h),
the variance of the spatial average over domain A is:

    Var(Z̄_A) = (1/|A|²) ∬_{A×A} C(x-y) dx dy

This integral depends ONLY on the geometry of A and the functional form of γ(h).

References
----------
Chilès, J.P. & Delfiner, P. (2012). Geostatistics: Modeling Spatial Uncertainty.
Hugonnet, R., et al. (2022). IEEE JSTARS, 15, 6456-6472.
"""

from __future__ import annotations

import math
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

        # --- Resolve polygon of interest ---
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
        
        # Initialize all gamma functions to None
        self.gamma_func = None
        self.gamma_func_min = None
        self.gamma_func_max = None
        
        # Component-wise gamma functions
        self.gamma_funcs_components: List[Optional[Callable]] = [None, None, None]
        self.gamma_funcs_components_min: List[Optional[Callable]] = [None, None, None]
        self.gamma_funcs_components_max: List[Optional[Callable]] = [None, None, None]
        
        # Total variance (sill + nugget)
        self.sigma2 = None
        self.sigma2_min = None
        self.sigma2_max = None

        if fitted_model is not None:
            # New approach: use FittedVariogramModel
            self.gamma_func = fitted_model.predict
            self.sigma2 = fitted_model.composite_model.total_sill
            
            # Min/max from bootstrap if available
            if fitted_model.param_samples is not None and len(fitted_model.param_samples) > 0:
                self._setup_minmax_from_bootstrap(fitted_model)
            else:
                self.gamma_func_min = self.gamma_func
                self.gamma_func_max = self.gamma_func
                self.sigma2_min = self.sigma2
                self.sigma2_max = self.sigma2
                
        elif use_bma and hasattr(va, 'model_selector') and va.model_selector is not None:
            # Bayesian Model Averaging
            self.gamma_func = va.get_bma_variogram_function()
            selector = va.model_selector
            self.sigma2 = sum(
                w * m.composite_model.total_sill
                for m, w in zip(selector.fitted_models, selector.model_weights)
            )
            # BMA doesn't have simple min/max - use same for all
            self.gamma_func_min = self.gamma_func
            self.gamma_func_max = self.gamma_func
            self.sigma2_min = self.sigma2
            self.sigma2_max = self.sigma2
            
        elif va.best_model_func is not None:
            # Legacy spherical model
            self._setup_legacy_gamma(va)
        else:
            raise ValueError("No variogram model available. Fit a model first.")

    def _setup_minmax_from_bootstrap(self, fitted_model: FittedVariogramModel) -> None:
        """Extract min/max gamma functions from bootstrap parameter samples."""
        samples = fitted_model.param_samples
        model = fitted_model.composite_model
        
        # Get parameter percentiles
        params_16 = np.percentile(samples, 16, axis=0)
        params_84 = np.percentile(samples, 84, axis=0)
        
        # Create min/max model functions
        def make_gamma(params):
            def gamma(h):
                model.set_params(params)
                return model(np.asarray(h, dtype=float))
            return gamma
        
        self.gamma_func_min = make_gamma(params_16)
        self.gamma_func_max = make_gamma(params_84)
        
        # Reset to central params
        model.set_params(fitted_model.params)
        
        # Estimate sigma2 min/max (sum of sills + nugget at percentiles)
        # This is approximate - proper way would parse param structure
        self.sigma2_min = self.sigma2 * 0.8  # rough approximation
        self.sigma2_max = self.sigma2 * 1.2

    def _setup_legacy_gamma(self, va: VariogramAnalysis) -> None:
        """Setup gamma functions from legacy spherical model."""
        # Central estimates
        self.sills = np.array(va.sills, dtype=float)
        self.ranges = np.array(va.ranges, dtype=float)
        self.nugget = getattr(va, 'best_nugget', 0.0) or 0.0
        self.sigma2 = self.nugget + float(np.sum(self.sills))

        # Min/max estimates
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

        # Central gamma function
        params = list(self.sills) + list(self.ranges)
        if has_nugget:
            params.append(self.nugget)
        self.gamma_func = lambda h, p=params: model_func(np.asarray(h, dtype=float), *p)

        # Min gamma function
        params_min = list(self.sills_min) + list(self.ranges_min)
        if has_nugget:
            params_min.append(self.nugget_min)
        self.gamma_func_min = lambda h, p=params_min: model_func(np.asarray(h, dtype=float), *p)

        # Max gamma function
        params_max = list(self.sills_max) + list(self.ranges_max)
        if has_nugget:
            params_max.append(self.nugget_max)
        self.gamma_func_max = lambda h, p=params_max: model_func(np.asarray(h, dtype=float), *p)

        # Component-wise gamma functions (for multi-component models)
        n_components = len(self.sills)
        for i in range(min(n_components, 3)):
            # Central
            p = [self.sills[i], self.ranges[i]]
            if has_nugget:
                p.append(self.nugget)
            self.gamma_funcs_components[i] = lambda h, p=p: model_func(np.asarray(h, dtype=float), *p)
            
            # Min
            p_min = [self.sills_min[i], self.ranges_min[i]]
            if has_nugget:
                p_min.append(self.nugget_min)
            self.gamma_funcs_components_min[i] = lambda h, p=p_min: model_func(np.asarray(h, dtype=float), *p)
            
            # Max
            p_max = [self.sills_max[i], self.ranges_max[i]]
            if has_nugget:
                p_max.append(self.nugget_max)
            self.gamma_funcs_components_max[i] = lambda h, p=p_max: model_func(np.asarray(h, dtype=float), *p)

    def _init_result_storage(self) -> None:
        """Initialize all result storage attributes."""
        # Uncorrelated
        self.sigma0_uncorrelated = None
        self.mean_uncorrelated_polygon = None
        self.mean_uncorrelated_raster = None

        # Correlated - polygon (central, min, max)
        self.mean_correlated_polygon = None
        self.mean_correlated_polygon_min = None
        self.mean_correlated_polygon_max = None

        # Correlated - raster (central, min, max)
        self.mean_correlated_raster = None
        self.mean_correlated_raster_min = None
        self.mean_correlated_raster_max = None

        # Component-wise - polygon
        self.mean_correlated_components_polygon: List[Optional[float]] = [None, None, None]
        self.mean_correlated_components_polygon_min: List[Optional[float]] = [None, None, None]
        self.mean_correlated_components_polygon_max: List[Optional[float]] = [None, None, None]

        # Component-wise - raster
        self.mean_correlated_components_raster: List[Optional[float]] = [None, None, None]
        self.mean_correlated_components_raster_min: List[Optional[float]] = [None, None, None]
        self.mean_correlated_components_raster_max: List[Optional[float]] = [None, None, None]

        # Total uncertainty (central, min, max)
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

        # Raster mean
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

        # Sample points inside domain
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
    ) -> None:
        """Compute correlated uncertainty for polygon mean (central, min, max)."""
        # Central
        if self.gamma_func is not None:
            self.mean_correlated_polygon = self.estimate_std_mean_monte_carlo(
                self.polygon, self.gamma_func, self.sigma2, n_pairs, seed
            )

        # Min
        if self.gamma_func_min is not None:
            self.mean_correlated_polygon_min = self.estimate_std_mean_monte_carlo(
                self.polygon, self.gamma_func_min, self.sigma2_min, n_pairs, seed
            )

        # Max
        if self.gamma_func_max is not None:
            self.mean_correlated_polygon_max = self.estimate_std_mean_monte_carlo(
                self.polygon, self.gamma_func_max, self.sigma2_max, n_pairs, seed
            )

        # Component-wise
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
    ) -> None:
        """Compute correlated uncertainty for raster mean (central, min, max)."""
        self.raster_data_handler.get_detailed_area()
        raster_geom = self.raster_data_handler.merged_geom or box(*self.raster_data_handler.bounds)

        # Central
        if self.gamma_func is not None:
            self.mean_correlated_raster = self.estimate_std_mean_monte_carlo(
                raster_geom, self.gamma_func, self.sigma2, n_pairs, seed
            )

        # Min
        if self.gamma_func_min is not None:
            self.mean_correlated_raster_min = self.estimate_std_mean_monte_carlo(
                raster_geom, self.gamma_func_min, self.sigma2_min, n_pairs, seed
            )

        # Max
        if self.gamma_func_max is not None:
            self.mean_correlated_raster_max = self.estimate_std_mean_monte_carlo(
                raster_geom, self.gamma_func_max, self.sigma2_max, n_pairs, seed
            )

        # Component-wise
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
    ) -> None:
        """Compute full uncertainty budget (uncorrelated + correlated)."""
        self.calc_mean_uncorrelated(use_stable_areas=use_stable_areas)
        self.calc_mean_correlated_polygon(n_pairs=n_pairs, seed=seed)
        self.calc_mean_correlated_raster(n_pairs=n_pairs, seed=seed)

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

        # Raster totals
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