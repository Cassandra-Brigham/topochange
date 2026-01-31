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
from typing import Optional, Callable, Tuple

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

        # --- Setup gamma function (model-agnostic) ---
        if fitted_model is not None:
            # New approach: use FittedVariogramModel directly
            self.gamma_func = fitted_model.predict
            self.sigma2 = fitted_model.composite_model.total_sill
        elif use_bma and hasattr(variogram_analysis, 'model_selector'):
            # Bayesian Model Averaging
            self.gamma_func = variogram_analysis.get_bma_variogram_function()
            # Weighted average of sills
            selector = variogram_analysis.model_selector
            self.sigma2 = sum(
                w * m.composite_model.total_sill
                for m, w in zip(selector.fitted_models, selector.model_weights)
            )
        elif variogram_analysis.best_model_func is not None:
            # Legacy: use spherical model from VariogramAnalysis
            self._setup_legacy_gamma(variogram_analysis)
        else:
            raise ValueError("No variogram model available. Fit a model first.")

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
        self.sigma0_uncorrelated = None
        self.mean_uncorrelated_polygon = None
        self.mean_uncorrelated_raster = None
        self.mean_correlated_polygon = None
        self.mean_correlated_raster = None
        self.total_uncertainty_polygon = None
        self.total_uncertainty_raster = None

    def _setup_legacy_gamma(self, va: VariogramAnalysis) -> None:
        """Setup gamma function from legacy spherical model."""
        self.sills = np.array(va.sills, dtype=float)
        self.ranges = np.array(va.ranges, dtype=float)
        self.nugget = getattr(va, 'best_nugget', 0.0) or 0.0
        self.sigma2 = self.nugget + float(np.sum(self.sills))

        params = list(self.sills) + list(self.ranges)
        if va.best_model_config.get('nugget'):
            params.append(self.nugget)

        self.gamma_func = lambda h: va.best_model_func(np.asarray(h, dtype=float), *params)

    def covariance(self, h: np.ndarray) -> np.ndarray:
        """C(h) = σ² - γ(h)"""
        return self.sigma2 - self.gamma_func(h)

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
        cov = self.covariance(h)
        var_mean = float(np.mean(cov))

        return 0.0 if var_mean < 0 else math.sqrt(var_mean)

    def calc_mean_correlated_polygon(self, n_pairs: int = 200_000, seed: Optional[int] = None) -> None:
        """Compute correlated uncertainty for polygon mean."""
        self.mean_correlated_polygon = self.estimate_std_mean_monte_carlo(
            self.polygon, n_pairs=n_pairs, seed=seed
        )

    def calc_mean_correlated_raster(self, n_pairs: int = 200_000, seed: Optional[int] = None) -> None:
        """Compute correlated uncertainty for raster mean."""
        self.raster_data_handler.get_detailed_area()
        raster_geom = self.raster_data_handler.merged_geom or box(*self.raster_data_handler.bounds)

        self.mean_correlated_raster = self.estimate_std_mean_monte_carlo(
            raster_geom, n_pairs=n_pairs, seed=seed
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

        # Quadrature sum
        if self.mean_uncorrelated_polygon and self.mean_correlated_polygon:
            self.total_uncertainty_polygon = math.sqrt(
                self.mean_uncorrelated_polygon**2 + self.mean_correlated_polygon**2
            )

        if self.mean_uncorrelated_raster and self.mean_correlated_raster:
            self.total_uncertainty_raster = math.sqrt(
                self.mean_uncorrelated_raster**2 + self.mean_correlated_raster**2
            )

    def summary(self) -> str:
        """Return formatted summary of results."""
        lines = [
            "=" * 60,
            "REGIONAL UNCERTAINTY SUMMARY",
            "=" * 60,
            f"Polygon area: {self.area:.2f} m²",
            f"Total variance (σ²): {self.sigma2:.6f}",
            "",
        ]

        if self.sigma0_uncorrelated:
            lines.append(f"Uncorrelated σ₀: {self.sigma0_uncorrelated:.6f}")
        if self.mean_uncorrelated_polygon:
            lines.append(f"Uncorrelated (polygon mean): {self.mean_uncorrelated_polygon:.6f}")
        if self.mean_correlated_polygon:
            lines.append(f"Correlated (polygon mean): {self.mean_correlated_polygon:.6f}")
        if self.total_uncertainty_polygon:
            lines.append(f"Total uncertainty (polygon): {self.total_uncertainty_polygon:.6f}")

        lines.append("")
        if self.mean_uncorrelated_raster:
            lines.append(f"Uncorrelated (raster mean): {self.mean_uncorrelated_raster:.6f}")
        if self.mean_correlated_raster:
            lines.append(f"Correlated (raster mean): {self.mean_correlated_raster:.6f}")
        if self.total_uncertainty_raster:
            lines.append(f"Total uncertainty (raster): {self.total_uncertainty_raster:.6f}")

        lines.append("=" * 60)
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