from __future__ import annotations

import copy
import warnings
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, norm as sp_norm
from numba import njit, prange, get_num_threads, get_thread_id
import rasterio
import rioxarray as rio
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape
from itertools import combinations_with_replacement

from .variogram_models import MODEL_REGISTRY, VariogramModelRegistry
from .composite_variogram import CompositeVariogramModel


# kriging LOOCV result

@dataclass
class KrigingLOOCVResult:
    """Results from a single leave-one-out kriging cross-validation run.

    The primary diagnostic is ``msspe`` (Mean Standardized Squared
    Prediction Error), which should be approximately 1.0 for a well-calibrated
    variogram model.

    References
    ----------
    Cressie, N. (1993). *Statistics for Spatial Data*, rev. ed., Wiley.
        Section 5.6.
    Webster, R. & Oliver, M.A. (2007). *Geostatistics for Environmental
        Scientists*, 2nd ed., Wiley. Section 8.3.
    """
    msspe: float
    mean_error: float
    rmse: float
    mean_standardized_error: float
    n_points: int
    n_failed: int


# aggregated LOOCV across multiple runs 

@dataclass
class AggregatedLOOCVResult:
    """Aggregated diagnostics from repeated kriging LOOCV runs."""
    n_runs: int
    total_points: int
    msspe_mean: float
    msspe_median: float
    msspe_std: float
    msspe_nmad: float
    mean_error_mean: float
    mean_error_median: float
    mean_error_std: float
    mean_error_nmad: float
    rmse_mean: float
    rmse_median: float
    rmse_std: float
    rmse_nmad: float
    mse_mean: float
    mse_median: float
    mse_std: float
    mse_nmad: float
    n_failed_total: int
    per_run_results: List[KrigingLOOCVResult] = field(default_factory=list)

    @staticmethod
    def from_results(results: List[KrigingLOOCVResult]) -> AggregatedLOOCVResult:
        """Compute aggregate statistics from a list of per-run results."""
        def _nmad(arr: np.ndarray) -> float:
            return float(1.4826 * np.median(np.abs(arr - np.median(arr))))

        def _stats(arr: np.ndarray) -> tuple:
            return (
                float(np.mean(arr)),
                float(np.median(arr)),
                float(np.std(arr)),
                _nmad(arr),
            )

        msspes = np.array([r.msspe for r in results])
        mean_errors = np.array([r.mean_error for r in results])
        rmses = np.array([r.rmse for r in results])
        mses = np.array([r.mean_standardized_error for r in results])

        return AggregatedLOOCVResult(
            n_runs=len(results),
            total_points=sum(r.n_points for r in results),
            msspe_mean=_stats(msspes)[0],
            msspe_median=_stats(msspes)[1],
            msspe_std=_stats(msspes)[2],
            msspe_nmad=_stats(msspes)[3],
            mean_error_mean=_stats(mean_errors)[0],
            mean_error_median=_stats(mean_errors)[1],
            mean_error_std=_stats(mean_errors)[2],
            mean_error_nmad=_stats(mean_errors)[3],
            rmse_mean=_stats(rmses)[0],
            rmse_median=_stats(rmses)[1],
            rmse_std=_stats(rmses)[2],
            rmse_nmad=_stats(rmses)[3],
            mse_mean=_stats(mses)[0],
            mse_median=_stats(mses)[1],
            mse_std=_stats(mses)[2],
            mse_nmad=_stats(mses)[3],
            n_failed_total=sum(r.n_failed for r in results),
            per_run_results=list(results),
        )


# RasterDataHandler 
class RasterDataHandler:
    """Load raster data, optionally subtract systematic error, and sample.

    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    unit : str
        Measurement unit label (for plots).
    resolution : float
        Nominal raster resolution (linear units).
    """

    def __init__(self, raster_path: str, unit: str, resolution: float):
        self.raster_path = raster_path
        self.unit = unit
        self.resolution = resolution
        self.rioxarray_obj = None
        self.data_array = None
        self.samples = None
        self.coords = None
        self.shapely_geoms = None
        self.merged_geom = None
        self.detailed_area = None

        with rasterio.open(self.raster_path) as src:
            bounds = src.bounds
            self.bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            self.bbox = box(*self.bounds)

    def get_detailed_area(self) -> None:
        """Compute precise area covered by valid raster data."""
        with rasterio.open(self.raster_path) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            valid = (
                (~np.isnan(data))
                if nodata is None
                else ((data != nodata) & ~np.isnan(data))
            )
            geoms = shapes(
                valid.astype(np.uint8), mask=valid, transform=src.transform
            )
        self.shapely_geoms = [
            shapely_shape(geom) for geom, val in geoms if val == 1
        ]
        self.merged_geom = unary_union(self.shapely_geoms)
        self.detailed_area = self.merged_geom.area

    def load_raster(self, masked: bool = True) -> None:
        """Load raster data; store finite values in ``self.data_array``."""
        da = rio.open_rasterio(self.raster_path, masked=masked)
        if "band" in da.dims and da.sizes.get("band", 1) == 1:
            da = da.squeeze("band", drop=True)
        arr = da.values
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        nodata = da.rio.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata
        self.rioxarray_obj = da
        self.data_array = np.asarray(arr[valid], dtype=float).ravel()

    def subtract_value_from_raster(
        self, output_raster_path: str, value_to_subtract: float
    ) -> None:
        """Subtract a value from every valid pixel and write a new raster."""
        with rasterio.open(self.raster_path) as src:
            data = src.read()
            nodata = src.nodata
            mask = (
                (data != nodata)
                if nodata is not None
                else np.ones(data.shape, dtype=bool)
            )
            data = data.astype(float)
            data[mask] -= value_to_subtract
            out_meta = src.meta.copy()
            out_meta.update({"dtype": "float32", "nodata": nodata})
            with rasterio.open(output_raster_path, "w", **out_meta) as dst:
                dst.write(data)

    def sample_raster(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        *,
        seed: Optional[int] = None,
    ) -> None:
        """Randomly sample valid pixels; stores values and coords."""
        with rasterio.open(self.raster_path) as src:
            rng = np.random.default_rng(seed)
            data = src.read(1).astype(float)
            nodata = src.nodata
            valid = np.isfinite(data)
            if nodata is not None:
                valid &= data != nodata

            cell_area_m2 = abs(src.res[0] * src.res[1])
            valid_rows, valid_cols = np.where(valid)
            valid_count = valid_rows.size
            cell_area_ref = cell_area_m2 / (area_side ** 2)
            total_samples = min(
                int(cell_area_ref * samples_per_area * valid_count),
                max_samples,
            )
            if total_samples < 1:
                raise ValueError(
                    "Computed total_samples < 1. "
                    "Increase samples_per_area or max_samples."
                )
            if total_samples > valid_count:
                raise ValueError(
                    "Requested samples exceed valid pixel count. "
                    "Reduce samples_per_area."
                )

            chosen = rng.choice(valid_count, size=total_samples, replace=False)
            rows = valid_rows[chosen]
            cols = valid_cols[chosen]
            samples = data[rows, cols]
            x_coords, y_coords = src.xy(rows, cols)
            coords = np.vstack([x_coords, y_coords]).T

            mask = np.isfinite(samples)
            self.samples = samples[mask]
            self.coords = coords[mask]


#  Numba-accelerated pairwise binning

@njit(parallel=True)
def _bin_distances_and_squared_differences(
    coords, values, bin_width, max_lag
):
    """Bin pairwise distances and accumulate squared-diff / sqrt-abs-diff.

    Uses per-thread accumulators to avoid race conditions when
    multiple threads write to the same lag bin.  After the parallel
    loop, the thread-local arrays are reduced into a single result.

    Returns
    -------
    n_bins, bin_counts, binned_sum_sq_diff, binned_sum_sqrt_abs_diff,
    max_distance
    """
    n_bins = int(np.ceil(max_lag / bin_width)) + 1
    M = coords.shape[0]
    n_threads = get_num_threads()

    # Per-thread accumulators — each thread writes only to its own
    # row, so no two threads ever touch the same memory location.
    local_counts = np.zeros((n_threads, n_bins), dtype=np.int64)
    local_sum_sq = np.zeros((n_threads, n_bins), dtype=np.float64)
    local_sum_sqrt = np.zeros((n_threads, n_bins), dtype=np.float64)
    local_sum_dist = np.zeros((n_threads, n_bins), dtype=np.float64)
    local_max_dist = np.zeros(n_threads, dtype=np.float64)

    for i in prange(M):
        tid = get_thread_id()
        for j in range(i + 1, M):
            d = 0.0
            for k in range(coords.shape[1]):
                tmp = coords[i, k] - coords[j, k]
                d += tmp * tmp
            dist = np.sqrt(d)
            if dist > local_max_dist[tid]:
                local_max_dist[tid] = dist

            diff = values[i] - values[j]
            bin_idx = int(dist / bin_width)
            if 0 <= bin_idx < n_bins:
                local_counts[tid, bin_idx] += 1
                local_sum_sq[tid, bin_idx] += diff ** 2
                local_sum_sqrt[tid, bin_idx] += np.sqrt(np.abs(diff))
                local_sum_dist[tid, bin_idx] += dist

    # Reduce per-thread accumulators into final arrays.
    # This is single-threaded but very fast (n_threads × n_bins).
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    binned_sum_sq = np.zeros(n_bins, dtype=np.float64)
    binned_sum_sqrt = np.zeros(n_bins, dtype=np.float64)
    binned_sum_dist = np.zeros(n_bins, dtype=np.float64)
    max_distance = 0.0

    for t in range(n_threads):
        if local_max_dist[t] > max_distance:
            max_distance = local_max_dist[t]
        for b in range(n_bins):
            bin_counts[b] += local_counts[t, b]
            binned_sum_sq[b] += local_sum_sq[t, b]
            binned_sum_sqrt[b] += local_sum_sqrt[t, b]
            binned_sum_dist[b] += local_sum_dist[t, b]

    return n_bins, bin_counts, binned_sum_sq, binned_sum_sqrt, binned_sum_dist, max_distance


#  SingleVariogram 

class SingleVariogram:
    """Compute, fit, and plot a single empirical variogram.

    Workflow
    --------
    1. Instantiate with a :class:`RasterDataHandler`.
    2. Call :meth:`compute_empirical_variogram` to sample the raster
       and compute binned semivariance across multiple realisations.
    3. (Optional) Call :meth:`fit_model` to fit candidate variogram
       models (spherical, exponential, matern) and select the best
       via MSSPE or AIC/BIC.
    4. Call :meth:`plot_single_variogram` to visualise.

    Parameters
    ----------
    raster_data_handler : RasterDataHandler
        Provides raster data and sampling capability.
    """

    # supported estimators
    ESTIMATORS = ("matheron", "cressie_hawkins")
    # bounded model families to try
    BOUNDED_MODELS = ["spherical", "exponential", "matern", "damped_hole_effect"]
    # minimum pair count threshold
    MIN_PAIRS = 10

    def __init__(self, raster_data_handler: RasterDataHandler):
        self.rdh = raster_data_handler

        # empirical variogram attributes (populated by compute_empirical_variogram)
        self.lags: Optional[np.ndarray] = None
        self.variogram: Optional[np.ndarray] = None
        self.pair_counts: Optional[np.ndarray] = None
        self.n_bins: int = 0
        self.estimator: Optional[str] = None
        self.sample_coords: Optional[np.ndarray] = None
        self.sample_values: Optional[np.ndarray] = None

        # detrending (P0-2): optional low-order polynomial drift removal
        self.trend_info: Optional[Dict[str, Any]] = None
        self._detrend_order: int = 0

        # model fitting attributes (populated by fit_model)
        self.fitted_models: List[Dict[str, Any]] = []
        self.best_model: Optional[Dict[str, Any]] = None
        self.criteria_table: Optional[pd.DataFrame] = None

    # empirical variogram computation 

    @staticmethod
    def _compute_matheron(bin_counts, ssd, min_pairs=10):
        """Matheron semivariance: γ(h) = SSD(h) / (2N(h))."""
        gamma = np.full(len(bin_counts), np.nan, dtype=float)
        for i in range(len(bin_counts)):
            if bin_counts[i] >= min_pairs:
                gamma[i] = ssd[i] / (2.0 * bin_counts[i])
        return gamma

    @staticmethod
    def _compute_cressie_hawkins(bin_counts, sum_sqrt_abs, min_pairs=10):
        """Cressie-Hawkins robust semivariance estimator.

        References
        ----------
        Cressie, N. & Hawkins, D.M. (1980). Robust estimation of the
        variogram: I. *J. Int. Assoc. Math. Geol.*, 12(2), 115-125.
        """
        gamma = np.full(len(bin_counts), np.nan, dtype=float)
        for i in range(len(bin_counts)):
            if bin_counts[i] >= min_pairs:
                mean_fourth = (sum_sqrt_abs[i] / bin_counts[i]) ** 4
                correction = 0.457 + 0.494 / bin_counts[i] + 0.045 / bin_counts[i]**2
                gamma[i] = 0.5 * mean_fourth / correction
        return gamma

    # ── detrending (P0-2) ───────────────────────────────────────────

    @staticmethod
    def _trend_design_matrix(coords, order, center, scale):
        """Polynomial design matrix in centered/scaled coordinates.

        order 0 -> [1]; order 1 -> [1, x, y];
        order 2 -> [1, x, y, x^2, x*y, y^2].  Centering and scaling keep
        X^T X well conditioned when coordinates are large (e.g. UTM).
        """
        coords = np.asarray(coords, dtype=float)
        x = (coords[:, 0] - center[0]) / scale[0]
        y = (coords[:, 1] - center[1]) / scale[1]
        cols = [np.ones_like(x)]
        if order >= 1:
            cols += [x, y]
        if order >= 2:
            cols += [x * x, x * y, y * y]
        return np.column_stack(cols)

    @classmethod
    def _fit_polynomial_trend(cls, coords, values, order):
        """Fit and remove a low-order polynomial drift by ordinary least squares.

        Removing a deterministic ramp/offset before variogram estimation
        prevents long-wavelength drift from being aliased into spurious
        long-range correlation, which otherwise inflates the area-averaged
        uncertainty (Chiles & Delfiner, 2012; Erten et al., 2020;
        Hugonnet et al., 2022).

        Returns
        -------
        residuals : ndarray
            ``values`` minus the fitted trend (same units as ``values``).
        trend_info : dict
            Keys: order, beta, beta_cov, center, scale, var_explained.
            ``beta_cov`` is the OLS coefficient covariance (an independence
            approximation; a GLS/REML covariance would be tighter, but the
            drift term is small over large stable areas).
        """
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)
        center = coords.mean(axis=0)
        scale = coords.std(axis=0)
        scale[scale == 0.0] = 1.0
        X = cls._trend_design_matrix(coords, order, center, scale)
        beta, _res, _rank, _sv = np.linalg.lstsq(X, values, rcond=None)
        residuals = values - X @ beta
        n, p = X.shape
        dof = max(n - p, 1)
        s2 = float(residuals @ residuals) / dof
        beta_cov = s2 * np.linalg.pinv(X.T @ X)
        total_var = float(np.var(values))
        var_explained = (
            0.0 if total_var <= 0.0 else 1.0 - float(np.var(residuals)) / total_var
        )
        trend_info = {
            "order": int(order),
            "beta": beta,
            "beta_cov": beta_cov,
            "center": center,
            "scale": scale,
            "var_explained": var_explained,
        }
        return residuals, trend_info

    def _single_variogram_run(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        bin_width: float,
        max_lag: float,
        estimator: str,
        seed: Optional[int] = None,
        detrend_order: Optional[int] = None,
        store_trend: bool = False,
    ):
        """Sample the raster once and compute one empirical variogram.

        Returns (bin_counts, gamma_estimate, n_bins, coords, values, mean_lags).

        When ``detrend_order`` (or the stored ``self._detrend_order``) is
        > 0, a low-order polynomial drift is removed from the sampled values
        before binning (P0-2), so the variogram describes the residual,
        stationary error rather than an aliased trend.  Passing
        ``detrend_order=None`` inherits the stored order, keeping bootstrap
        resampling consistent with the main computation.
        """
        self.rdh.sample_raster(
            area_side, samples_per_area, max_samples, seed=seed
        )
        coords = self.rdh.coords.copy()
        values = self.rdh.samples.copy()

        # optional polynomial detrending (P0-2) before binning
        order = self._detrend_order if detrend_order is None else int(detrend_order)
        if order and order > 0 and values.size > 3 * (order + 1):
            values, trend_info = self._fit_polynomial_trend(coords, values, order)
            if store_trend:
                self.trend_info = trend_info

        (n_bins, bin_counts, bssd, bssad, binned_sum_dist, _max_dist) = (
            _bin_distances_and_squared_differences(
                coords, values, bin_width, max_lag
            )
        )

        # actual mean distance per bin
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_lags = np.where(
                bin_counts > 0,
                binned_sum_dist / bin_counts,
                np.nan,
            )

        if estimator == "cressie_hawkins":
            gamma = self._compute_cressie_hawkins(
                bin_counts, bssad, self.MIN_PAIRS
            )
        else:
            gamma = self._compute_matheron(bin_counts, bssd, self.MIN_PAIRS)

        return bin_counts, gamma, n_bins, coords, values, mean_lags

    def compute_empirical_variogram(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        bin_width: float,
        max_lag_multiplier: float = 1 / 3,
        *,
        seed: Optional[int] = None,
        estimator: str = "matheron",
        return_sample: bool = False,
        detrend_order: int = 0,
    ) -> None:
        """Compute a single empirical variogram from one spatial sample.

        Draws one random sample from the raster, computes pairwise
        lag-binned semivariance, and stores the result.  For ensemble
        averaging across multiple samples, use :class:`GridVariogram`.

        Parameters
        ----------
        area_side : float
            Reference side length for sampling density conversion.
        samples_per_area : float
            Sampling density (points per area_side²).
        max_samples : int
            Hard cap on sample points.
        bin_width : float
            Lag bin width (same units as coordinates).
        max_lag_multiplier : float
            Maximum lag as a fraction of the spatial extent diagonal.
        seed : int, optional
            RNG seed for reproducibility.
        estimator : {'matheron', 'cressie_hawkins'}
            Semivariance estimator.
        return_sample : bool
            If True, retain the sampled coordinates and values
            (needed for downstream MSSPE computation in ``fit_model``).
        detrend_order : int, default 0
            If > 0, remove a polynomial drift of this order (1 = planar,
            2 = quadratic) from the sampled values before binning (P0-2).
            This prevents a ramp/offset from being aliased into spurious
            long-range structure that inflates the area-averaged
            uncertainty.  The fitted trend is stored on ``self.trend_info``
            and is inherited by the spatial-resampling bootstrap.
        """
        if estimator not in self.ESTIMATORS:
            raise ValueError(
                f"Unknown estimator '{estimator}'. "
                f"Choose from {self.ESTIMATORS}."
            )

        # record detrend setting so bootstrap resampling inherits it (P0-2)
        self._detrend_order = int(detrend_order)
        self.trend_info = None

        # compute max_lag from raster extent
        xs = self.rdh.rioxarray_obj.x.values
        ys = self.rdh.rioxarray_obj.y.values
        x_extent = float(np.max(xs) - np.min(xs))
        y_extent = float(np.max(ys) - np.min(ys))
        diag = np.sqrt(x_extent ** 2 + y_extent ** 2)
        max_lag = float(diag * max_lag_multiplier)

        # sample and compute
        bin_counts, gamma, n_bins_run, coords, values, mean_lags = (
            self._single_variogram_run(
                area_side, samples_per_area, max_samples,
                bin_width, max_lag, estimator, seed=seed,
                detrend_order=detrend_order, store_trend=True,
            )
        )

        # keep only valid (non-NaN) bins
        valid = ~np.isnan(gamma)
        n_kept = int(np.sum(valid))

        self.variogram = gamma[valid]
        self.pair_counts = bin_counts[valid].astype(float)
        self.lags = mean_lags[valid]
        self.n_bins = n_kept
        self.estimator = estimator

        if return_sample:
            self.sample_coords = coords
            self.sample_values = values

        # store for bootstrap_parameters() reuse
        self._stored_area_side = area_side
        self._stored_samples_per_area = samples_per_area
        self._stored_max_samples = max_samples
        self._stored_bin_width = bin_width
        self._stored_max_lag_multiplier = max_lag_multiplier
        self._stored_estimator = estimator

    # ── model fitting ───────────────────────────────────────────────

    @staticmethod
    def _estimate_nugget_from_short_lags(lags, variogram, n_lags=5):
        """Pre-estimate nugget by extrapolating short-lag bins to h=0."""
        max_gamma = float(np.nanmax(variogram))
        n_fit = min(n_lags, max(2, len(lags) // 4))
        if n_fit < 2:
            return max_gamma * 0.1
        try:
            _slope, intercept = np.polyfit(
                lags[:n_fit], variogram[:n_fit], 1
            )
            return float(np.clip(intercept, 0.0, max_gamma * 0.5))
        except (np.linalg.LinAlgError, ValueError):
            return max_gamma * 0.1

    @staticmethod
    def _generate_multistart_guesses(p0_base, bounds, n_restarts, rng):
        """Generate diverse starting points for optimisation."""
        lb = np.asarray(bounds[0], dtype=float)
        ub = np.asarray(bounds[1], dtype=float)
        guesses = [p0_base.copy()]

        if n_restarts >= 2:
            guesses.append(np.clip(p0_base * 0.5, lb, ub))
        if n_restarts >= 3:
            guesses.append(np.clip(p0_base * 2.0, lb, ub))

        for _ in range(max(0, n_restarts - len(guesses))):
            sample = np.empty_like(p0_base)
            rand = rng.random(len(p0_base))
            for j in range(len(p0_base)):
                lo, hi = max(lb[j], 1e-12), ub[j]
                if hi / lo > 50:
                    sample[j] = np.exp(
                        np.log(lo) + rand[j] * (np.log(hi) - np.log(lo))
                    )
                else:
                    sample[j] = lo + rand[j] * (hi - lo)
            guesses.append(np.clip(sample, lb, ub))

        return guesses[:n_restarts]

    def _fit_single_composite_model(
        self,
        model: CompositeVariogramModel,
        lags: np.ndarray,
        variogram: np.ndarray,
        sigma: Optional[np.ndarray],
        weights: np.ndarray,
        n_restarts: int = 8,
        maxfev: int = 10_000,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fit one composite model via multi-start WLS.

        Uses Cressie (1985) weights w_i = N(h_i) / gamma_hat(h_i)^2
        passed through ``weights``.  These are converted to ``sigma``
        values for ``scipy.optimize.curve_fit`` so that the optimiser
        minimises the weighted residual sum of squares directly.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Source of randomness for the multi-start initial guesses.
            Pass a seeded generator for reproducible fits (e.g. inside a
            seeded bootstrap loop). If ``None`` (default), a fresh
            entropy-seeded generator is used, so results vary run to run.

        Returns a dict with keys: 'model', 'params', 'rss', 'aic', 'bic',
        'warnings', or None if fitting fails entirely.
        """
        p0_base = model.default_guess(lags, variogram)
        bounds = model.bounds(lags, variogram)

        # nugget pre-estimation
        if model.include_nugget:
            nugget_pre = self._estimate_nugget_from_short_lags(lags, variogram)
            nugget_idx = model.n_params - 1
            p0_base[nugget_idx] = np.clip(
                nugget_pre, bounds[0][nugget_idx], bounds[1][nugget_idx]
            )

        def model_func(h, *params):
            model.set_params(np.array(params))
            return model(h)

        #  Convert Cressie weights to sigma for curve_fit 

        if sigma is None and weights is not None:
            safe_w = np.where(
                np.isfinite(weights) & (weights > 0),
                weights,
                np.finfo(float).eps,
            )
            sigma_from_weights = 1.0 / np.sqrt(safe_w)
        else:
            sigma_from_weights = sigma

        if rng is None:
            rng = np.random.default_rng()
        guesses = self._generate_multistart_guesses(
            p0_base, bounds, n_restarts, rng
        )

        best_result = None
        best_rss = np.inf

        for p0 in guesses:
            try:
                popt, pcov = curve_fit(
                    model_func, lags, variogram,
                    p0=p0, sigma=sigma_from_weights,
                    absolute_sigma=True,
                    bounds=bounds, maxfev=maxfev,
                )
                model.set_params(popt)
                residuals = variogram - model(lags)
                rss = float(np.sum(weights * residuals ** 2))

                if rss < best_rss:
                    best_rss = rss
                    best_result = (popt.copy(), pcov.copy(), rss)
            except (RuntimeError, ValueError):
                continue

        if best_result is None:
            return None

        popt, pcov, rss = best_result
        model.set_params(popt)
        k = model.n_params
        n_eff = len(lags)

        # ── Pseudo-AIC/BIC for WLS ──
        #
        # Following Cressie (1985) and Webster & Oliver (2007, §5.3):
        # use the weighted RSS as a pseudo-deviance for model ranking.
        # These are NOT proper log-likelihoods, so AIC/BIC values are
        # only meaningful for *relative* comparison among candidates
        # fitted with the same weights on the same empirical variogram.
        #
        # NOTE: n_eff is the number of lag bins, NOT the number of
        # data points.  With typical bin counts of 15–30, the AICc
        # correction can be large.  The AICc formula below is the
        # standard Burnham & Anderson (2002) second-order correction,
        # but its bias-correction properties are derived for MLE
        # under Gaussian assumptions — not for WLS pseudo-deviance.
        # We apply it as a conservative heuristic that penalises
        # complex models more heavily when n_eff/k is small, which
        # is appropriate for our use case.  For more rigorous model
        # selection, prefer LOOCV-based MSSPE (criterion='msspe').
        aic = n_eff * np.log(rss / max(n_eff, 1)) + 2 * k
        bic = n_eff * np.log(rss / max(n_eff, 1)) + k * np.log(max(n_eff, 1))

        # AICc (small-sample correction; Burnham & Anderson, 2002)
        # Applied as heuristic to WLS pseudo-deviance — see note above.
        denom = max(n_eff - k - 1, 1)
        aicc = float(aic) + 2 * k * (k + 1) / denom

        return {
            "model": model,
            "params": popt,
            "param_cov": pcov,
            "rss": rss,
            "aic": float(aic),
            "aicc": float(aicc),
            "bic": float(bic),
            "msspe": None,
            "msspe_std": None,
            "msspe_n_runs": 0,
            "description": model.structural_description(),
            "warnings": [],
            "fitting_method": "wls",
        }

    @staticmethod
    def _kriging_loocv(
        coords: np.ndarray,
        values: np.ndarray,
        model: CompositeVariogramModel,
        n_subset: int = 500,
        dist_matrix: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> KrigingLOOCVResult:
        """Leave-one-out kriging cross-validation on a point set.

        Parameters
        ----------
        coords : (n, 2) array
        values : (n,) array
        model : CompositeVariogramModel  (params must be set)
        n_subset : int
            Subsample size for computational tractability.
        dist_matrix : (n, n) array, optional
            Pre-computed distance matrix.

        Returns
        -------
        KrigingLOOCVResult
        """
        n = len(values)
        if n > n_subset:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, n_subset, replace=False)
            coords = coords[idx]
            values = values[idx]
            n = n_subset
            dist_matrix = None  # invalidated by subsample

        if dist_matrix is None:
            dx = coords[:, 0:1] - coords[:, 0:1].T
            dy = coords[:, 1:2] - coords[:, 1:2].T
            dist_matrix = np.sqrt(dx ** 2 + dy ** 2)

        # build covariance matrix from variogram
        total_sill = model.get_total_sill()
        if total_sill is None or total_sill <= 0:
            raise ValueError("Model must be stationary with positive sill.")

        gamma_mat = model(dist_matrix)
        C = total_sill - gamma_mat
        np.fill_diagonal(C, total_sill)

        # ── Optimised LOO via single matrix inversion ──
        # Instead of solving n separate (n-1)×(n-1) systems [O(n⁴)],
        # invert C once [O(n³)] and use the identity:
        #   LOO error_i     = (C⁻¹ z)_i  /  (C⁻¹)_ii
        #   LOO variance_i  = 1 / (C⁻¹)_ii
        #
        # References:
        #   Ripley, B.D. (1981). Spatial Statistics. Wiley. §4.4.
        #   Cressie, N. (1993). Statistics for Spatial Data. §5.6.
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return KrigingLOOCVResult(
                msspe=np.nan, mean_error=np.nan, rmse=np.nan,
                mean_standardized_error=np.nan,
                n_points=0, n_failed=n,
            )

        C_inv_z = C_inv @ values
        diag_C_inv = np.diag(C_inv)

        # guard against zero/negative diagonal entries
        valid = diag_C_inv > 1e-12
        n_failed = int(np.sum(~valid))

        if not np.any(valid):
            return KrigingLOOCVResult(
                msspe=np.nan, mean_error=np.nan, rmse=np.nan,
                mean_standardized_error=np.nan,
                n_points=0, n_failed=n_failed,
            )

        errors = C_inv_z[valid] / diag_C_inv[valid]
        sigma2_k = 1.0 / diag_C_inv[valid]
        std_errors = errors / np.sqrt(sigma2_k)

        return KrigingLOOCVResult(
            msspe=float(np.mean(std_errors ** 2)),
            mean_error=float(np.mean(errors)),
            rmse=float(np.sqrt(np.mean(errors ** 2))),
            mean_standardized_error=float(np.mean(std_errors)),
            n_points=int(np.sum(valid)),
            n_failed=n_failed,
        )

    @staticmethod
    def kriging_loocv(
        coords: np.ndarray,
        values: np.ndarray,
        model,
        n_subset: int = 500,
        *,
        seed: Optional[int] = None,
    ) -> KrigingLOOCVResult:
        """Public kriging leave-one-out cross-validation with input validation.

        Parameters
        ----------
        coords : (n, 2) array
            Spatial coordinates.
        values : (n,) array
            Observed values.
        model : CompositeVariogramModel or callable
            Fitted variogram model (params must be set).
        n_subset : int
            Subsample size for computational tractability.
        seed : int, optional
            RNG seed for reproducible subsampling.

        Returns
        -------
        KrigingLOOCVResult
        """
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"coords must have shape (n, 2), got shape {coords.shape}"
            )
        if len(coords) != len(values):
            raise ValueError(
                f"coords and values must have same length, "
                f"got {len(coords)} and {len(values)}"
            )

        n = len(values)
        if n > n_subset:
            rng = np.random.default_rng(seed)  # fixed seed for reproducibility
            idx = rng.choice(n, n_subset, replace=False)
            coords = coords[idx]
            values = values[idx]
            n = n_subset
            dist_matrix = None  # invalidated by subsample

        return SingleVariogram._kriging_loocv(
            coords, values, model, n_subset=len(values),
        )

    def fit_model(
        self,
        model_types: Optional[List[str]] = None,
        include_nugget: bool = True,
        max_components: int = 3,
        criterion: str = "aicc",
        msspe_n_subset: int = 500,
        msspe_n_runs: int = 10,
        msspe_prefilter: int = 0,
        *,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fit candidate variogram models and select the best.

        Fits all combinations of ``model_types`` (up to ``max_components``
        nested structures) via Weighted Least Squares on the binned
        empirical variogram using Cressie (1985) weights:

            w_i = N(h_i) / gamma_hat(h_i)^2

        Parameters
        ----------
        model_types : list of str, optional
            Model families to try. Default: ``['spherical', 'exponential',
            'matern']``.
        include_nugget : bool
            Whether to generate with-nugget and without-nugget variants.
        max_components : int
            Maximum nested structures per candidate (1–3).
        criterion : {'aicc', 'aic', 'bic', 'msspe'}
            Model selection criterion.

            - ``'aicc'``: select by minimum corrected Akaike Information
              Criterion (recommended, especially for small n/k ratios).
            - ``'aic'``: select by minimum Akaike Information Criterion.
            - ``'bic'``: select by minimum Bayesian Information Criterion.
            - ``'msspe'``: select the model whose kriging LOOCV MSSPE is
              closest to 1.0.  Requires spatial sample data
              (``return_sample=True`` in
              ``compute_empirical_variogram``).
        msspe_n_subset : int
            Points per LOOCV evaluation (only used when
            ``criterion='msspe'``).
        msspe_n_runs : int
            Independent LOOCV repetitions for stable MSSPE.
        msspe_prefilter : int
            If > 0, evaluate MSSPE only on the top-N models by AIC.
        seed : int, optional
            Seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Criteria table sorted by the chosen criterion.

        References
        ----------
        Cressie, N. (1985). Fitting variogram models by weighted least
        squares. *Journal of the International Association for
        Mathematical Geology*, 17(5), 563–586.
        """
        if self.lags is None or self.variogram is None:
            raise RuntimeError(
                "No empirical variogram. "
                "Call compute_empirical_variogram() first."
            )

        if model_types is None:
            model_types = list(self.BOUNDED_MODELS)

        lags = self.lags
        variogram = self.variogram
        pair_counts = self.pair_counts

        # ── prepare Cressie (1985) WLS weights ──
        # w_i = N(h_i) / γ̂(h_i)²
        gamma_sq = np.square(variogram)
        gamma_sq = np.where(
            gamma_sq < np.finfo(float).eps, np.finfo(float).eps, gamma_sq
        )
        weights = (
            pair_counts / gamma_sq
            if pair_counts is not None
            else np.ones_like(variogram)
        )

        # ── generate candidates ──
        bounded = [m for m in model_types if MODEL_REGISTRY.is_bounded(m)]
        candidates: List[CompositeVariogramModel] = []

        # pure nugget (no spatial structure)
        try:
            candidates.append(
                CompositeVariogramModel([], include_nugget=True)
            )
        except ValueError:
            pass

        for n in range(1, min(max_components, 3) + 1):
            for combo in combinations_with_replacement(bounded, n):
                combo_list = list(combo)
                if combo_list.count("matern") > 1:
                    continue
                nugget_options = [True, False] if include_nugget else [False]
                for use_nugget in nugget_options:
                    try:
                        model = CompositeVariogramModel(
                            combo_list, include_nugget=use_nugget
                        )
                        candidates.append(model)
                    except ValueError:
                        continue

        # ── fit each candidate via multi-start WLS ──
        # Seed the multi-start guesses so model selection is reproducible
        # when a seed is supplied (e.g. from run()); falls back to fresh
        # entropy when seed is None.
        self.fitted_models = []
        fit_rng = np.random.default_rng(seed)
        for cand in candidates:
            result = self._fit_single_composite_model(
                cand, lags, variogram, None, weights, rng=fit_rng,
            )
            if result is not None:
                result["trend_order"] = 0
                result["trend_coefficients"] = np.array([np.nan])
                result["n_trend_params"] = 1
                self.fitted_models.append(result)

        if not self.fitted_models:
            raise RuntimeError(
                "No models successfully fitted. Check input data."
            )

        # ── compute MSSPE (only when criterion requires it) ──
        compute_msspe = criterion == "msspe"
        if compute_msspe and (self.sample_coords is None or self.sample_values is None):
            warnings.warn(
                "No spatial sample available for MSSPE computation. "
                "Re-run compute_empirical_variogram with return_sample=True. "
                "Falling back to AICc selection.",
                UserWarning,
            )
            compute_msspe = False

        if compute_msspe:
            coords = self.sample_coords
            values = self.sample_values

            # determine which models to evaluate
            if msspe_prefilter > 0 and len(self.fitted_models) > msspe_prefilter:
                ranked = np.argsort([m["aicc"] for m in self.fitted_models])
                eval_idx = set(ranked[:msspe_prefilter].tolist())
            else:
                eval_idx = set(range(len(self.fitted_models)))

            # pre-draw shared subsample indices for fair comparison
            run_rng = np.random.default_rng(seed)
            run_seeds = run_rng.integers(0, 2 ** 31, size=msspe_n_runs)
            n_pts = len(values)

            subsample_indices = []
            for rs in run_seeds:
                sub_rng = np.random.default_rng(int(rs))
                if n_pts > msspe_n_subset:
                    idx = sub_rng.choice(n_pts, msspe_n_subset, replace=False)
                else:
                    idx = np.arange(n_pts)
                subsample_indices.append(idx)

            # precompute distance matrices
            precomputed_runs = []
            for idx in subsample_indices:
                sc = coords[idx]
                sv = values[idx]
                dx = sc[:, 0:1] - sc[:, 0:1].T
                dy = sc[:, 1:2] - sc[:, 1:2].T
                sd = np.sqrt(dx ** 2 + dy ** 2)
                precomputed_runs.append((sc, sv, sd))

            for i, fitted in enumerate(self.fitted_models):
                if i not in eval_idx:
                    continue
                model = fitted["model"]
                if not model.is_stationary:
                    continue
                model.set_params(fitted["params"])

                run_results: List[KrigingLOOCVResult] = []
                for sc, sv, sd in precomputed_runs:
                    try:
                        result = self._kriging_loocv(
                            sc, sv, model,
                            n_subset=len(sv), dist_matrix=sd,
                        )
                        if np.isfinite(result.msspe):
                            run_results.append(result)
                    except Exception:
                        continue

                if run_results:
                    agg = AggregatedLOOCVResult.from_results(run_results)
                    fitted["msspe"] = agg.msspe_mean
                    fitted["msspe_std"] = agg.msspe_std
                    fitted["msspe_n_runs"] = agg.n_runs

        # ── select best model ──
        has_msspe = any(
            m["msspe"] is not None for m in self.fitted_models
        )

        if criterion == "msspe" and has_msspe:
            scores = [
                abs(m["msspe"] - 1.0) if m["msspe"] is not None else np.inf
                for m in self.fitted_models
            ]
        elif criterion == "bic":
            scores = [m["bic"] for m in self.fitted_models]
        elif criterion == "aicc":
            scores = [m["aicc"] for m in self.fitted_models]
        else:
            scores = [m["aic"] for m in self.fitted_models]

        best_idx = int(np.argmin(scores))
        self.best_model = self.fitted_models[best_idx]

        # ── build criteria table ──
        rows = []
        for m in self.fitted_models:
            rows.append({
                "model": m["description"],
                "trend_order": m.get("trend_order", 0),
                "AIC": m["aic"],
                "AICc": m["aicc"],
                "BIC": m["bic"],
                "MSSPE": m["msspe"],
                "MSSPE_std": m["msspe_std"],
                "params": m["params"].tolist(),
            })
        df = pd.DataFrame(rows)

        # sort by the chosen criterion
        if criterion == "msspe" and has_msspe:
            df["abs_MSSPE_minus_1"] = df["MSSPE"].apply(
                lambda x: abs(x - 1.0) if x is not None and np.isfinite(x) else np.inf
            )
            df = df.sort_values("abs_MSSPE_minus_1").drop(
                columns="abs_MSSPE_minus_1"
            )
        elif criterion == "bic":
            df = df.sort_values("BIC")
        elif criterion == "aicc":
            df = df.sort_values("AICc")
        else:
            df = df.sort_values("AIC")
        df = df.reset_index(drop=True)
        self.criteria_table = df

        return df

    # parameter bootstrap 

    def bootstrap_parameters(
        self,
        n_realizations: int = 100,
        area_side: float = None,
        samples_per_area: float = None,
        max_samples: int = None,
        bin_width: float = None,
        max_lag_multiplier: float = None,
        estimator: str = None,
        *,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Estimate parameter uncertainty via spatial resampling bootstrap.

        For each realisation, draws a new spatial sample, computes a
        fresh empirical variogram, and fits *only* the winning model
        structure (from :meth:`fit_model`).  The result is a
        ``(n_succeeded, n_params)`` array of fitted parameters whose
        spread quantifies parameter uncertainty conditional on the
        selected model family.

        Parameters
        ----------
        n_realizations : int
            Number of bootstrap realisations.
        area_side, samples_per_area, max_samples, bin_width,
        max_lag_multiplier, estimator
            Sampling / variogram parameters.  If ``None``, reuses the
            values from the most recent
            :meth:`compute_empirical_variogram` call.
        seed : int, optional
            Base seed for reproducibility (child seeds spawned via
            ``SeedSequence``).
        verbose : bool
            Print per-realisation progress.

        Returns
        -------
        param_samples : ndarray, shape (n_succeeded, n_params)
            Fitted parameter values for each successful realisation.
            Column order matches ``self.best_model["model"].param_names``.

        Raises
        ------
        RuntimeError
            If :meth:`fit_model` has not been called yet.

        Notes
        -----
        This method separates *parameter uncertainty* from *model
        selection uncertainty*.  Model selection is handled by the
        :class:`GridVariogram` ensemble; this method quantifies how
        much the parameters of the *chosen* model vary across
        independent spatial samples.

        """
        if self.best_model is None:
            raise RuntimeError(
                "No fitted model. Call fit_model() first."
            )

        # resolve defaults from stored parameters
        def _resolve(val, attr_name):
            if val is not None:
                return val
            stored = getattr(self, attr_name, None)
            if stored is None:
                raise ValueError(
                    f"'{attr_name.replace('_stored_', '')}' not provided "
                    f"and no stored value from compute_empirical_variogram()."
                )
            return stored

        area_side = _resolve(area_side, "_stored_area_side")
        samples_per_area = _resolve(samples_per_area, "_stored_samples_per_area")
        max_samples = _resolve(max_samples, "_stored_max_samples")
        bin_width = _resolve(bin_width, "_stored_bin_width")
        max_lag_multiplier = _resolve(max_lag_multiplier, "_stored_max_lag_multiplier")
        estimator = _resolve(estimator, "_stored_estimator")

        # compute max_lag (same logic as compute_empirical_variogram)
        xs = self.rdh.rioxarray_obj.x.values
        ys = self.rdh.rioxarray_obj.y.values
        x_extent = float(np.max(xs) - np.min(xs))
        y_extent = float(np.max(ys) - np.min(ys))
        diag = np.sqrt(x_extent ** 2 + y_extent ** 2)
        max_lag = float(diag * max_lag_multiplier)

        # winning model structure
        model_template = self.best_model["model"]

        # reproducible child seeds
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_realizations)

        collected = []
        n_failed = 0

        for i in range(n_realizations):
            run_seed = int(child_seeds[i].generate_state(1)[0])
            try:
                # draw new sample and compute empirical variogram
                bin_counts, gamma, n_bins, _coords, _values, mean_lags = (
                    self._single_variogram_run(
                        area_side, samples_per_area, max_samples,
                        bin_width, max_lag, estimator, seed=run_seed,
                    )
                )

                # keep only valid bins
                valid = ~np.isnan(gamma)
                n_kept = int(np.sum(valid))
                if n_kept < 3:
                    n_failed += 1
                    if verbose:
                        print(f"  [{i + 1}/{n_realizations}] SKIP: too few bins ({n_kept})")
                    continue

                boot_gamma = gamma[valid]
                boot_counts = bin_counts[valid].astype(float)
                boot_lags = mean_lags[valid]

                # Cressie (1985) weights
                gamma_sq = np.square(boot_gamma)
                gamma_sq = np.where(
                    gamma_sq < np.finfo(float).eps,
                    np.finfo(float).eps,
                    gamma_sq,
                )
                weights = boot_counts / gamma_sq

                # deep-copy winning model for independent fit
                model = copy.deepcopy(model_template)

                result = self._fit_single_composite_model(
                    model, boot_lags, boot_gamma, None, weights,
                    rng=np.random.default_rng(run_seed),
                )

                if result is not None:
                    collected.append(result["params"])
                    if verbose:
                        print(f"  [{i + 1}/{n_realizations}] OK")
                else:
                    n_failed += 1
                    if verbose:
                        print(f"  [{i + 1}/{n_realizations}] FIT FAILED")

            except Exception as e:
                n_failed += 1
                if verbose:
                    print(f"  [{i + 1}/{n_realizations}] ERROR: {e}")
                continue

        if not collected:
            raise RuntimeError(
                f"All {n_realizations} bootstrap realisations failed."
            )

        param_samples = np.array(collected)  # (n_succeeded, n_params)

        # compute percentile summary per parameter
        pct_keys = [5, 16, 50, 84, 95]
        param_names = list(model_template.param_names)
        param_percentiles = {}
        for j, name in enumerate(param_names):
            vals = param_samples[:, j]
            param_percentiles[name] = {
                f"p{p}": float(np.percentile(vals, p)) for p in pct_keys
            }

        # store for downstream use
        self.bootstrap_param_samples = param_samples
        self.bootstrap_param_percentiles = param_percentiles
        self.best_model["param_samples"] = param_samples
        self.best_model["param_percentiles"] = param_percentiles

        if verbose or n_failed > 0:
            print(
                f"Bootstrap complete: {len(collected)}/{n_realizations} "
                f"succeeded, {n_failed} failed."
            )

        return param_samples
    
    from scipy.spatial.distance import cdist
    from scipy.stats import rankdata, norm as sp_norm
    
    """
    Two additional bootstrap methods for variogram parameter uncertainty.

    INSERT BOTH METHODS into class SingleVariogram, immediately after
    bootstrap_parameters() and before the `# FittedVariogramModel` comment
    (i.e. after line 1272 in variogram-668c3dcd.py).

    Also add this import at the top of the file (line 11, after the
    existing scipy import):

        from scipy.spatial.distance import cdist
        from scipy.stats import rankdata, norm as sp_norm

    Both methods share the same storage convention as bootstrap_parameters():
    they populate self.bootstrap_param_samples, self.bootstrap_param_percentiles,
    and self.best_model["param_samples"] / ["param_percentiles"], so the
    existing plot_single_variogram() bootstrap envelope code works unchanged.

    References
    ----------
    [1] Olea, R.A. & Pardo-Igúzquiza, E. (2011). Generalized Bootstrap Method
        for Assessment of Uncertainty in Semivariogram Inference. Mathematical
        Geosciences, 43, 203–228.
        https://link.springer.com/article/10.1007/s11004-010-9269-6

    [2] Pardo-Igúzquiza, E. & Olea, R.A. (2012). VARBOOT: A Spatial Bootstrap
        Program for Semivariogram Uncertainty Assessment. Computers &
        Geosciences, 41, 188–198.
        https://doi.org/10.1016/j.cageo.2011.09.002

    [3] Diggle, P.J. & Ribeiro Jr, P.J. (2007). Model-based Geostatistics.
        Springer. Section 5.4.

    [4] Sjöstedt-de Luna, S. & Young, A. (2003). The Bootstrap and Kriging
        Prediction Intervals. Scandinavian Journal of Statistics, 30, 175–192.
    """

    # ─────────────────────────────────────────────────────────────────────
    # OPTION 1 — Generalized (spatial) bootstrap
    # ─────────────────────────────────────────────────────────────────────

    def generalized_bootstrap_parameters(
        self,
        n_realizations: int = 1000,
        *,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Estimate parameter uncertainty via the generalized spatial bootstrap.

        Keeps the original sample locations fixed and resamples *values*
        through a decorrelation–resample–recorrelation cycle that preserves
        spatial correlation structure.  This isolates estimation noise from
        sampling-geometry variability, producing tighter intervals than the
        nonparametric :meth:`bootstrap_parameters`.

        Procedure (per realisation)
        ---------------------------
        1. Normal-score transform the sample values.
        2. Decorrelate via Cholesky: w = L⁻¹ z_ns.
        3. Bootstrap (with replacement) the independent residuals w.
        4. Re-correlate: z_boot = L w*.
        5. Back-transform to original marginal distribution.
        6. Compute empirical variogram on (original coords, z_boot).
        7. Fit the winning model structure via WLS.

        Parameters
        ----------
        n_realizations : int
            Number of bootstrap realisations (≥ 1000 recommended;
            Pardo-Igúzquiza & Olea 2012).
        seed : int, optional
            Base seed for reproducibility.
        verbose : bool
            Print per-realisation progress.

        Returns
        -------
        param_samples : ndarray, shape (n_succeeded, n_params)

        Raises
        ------
        RuntimeError
            If :meth:`fit_model` has not been called, or if
            ``compute_empirical_variogram`` was called without
            ``return_sample=True``.

        References
        ----------
        Olea, R.A. & Pardo-Igúzquiza, E. (2011). Mathematical Geosciences,
        43, 203–228.
        """
        # ── guards ──────────────────────────────────────────────────────
        if self.best_model is None:
            raise RuntimeError("No fitted model. Call fit_model() first.")
        if self.sample_coords is None or self.sample_values is None:
            raise RuntimeError(
                "No spatial sample stored. Re-run "
                "compute_empirical_variogram(return_sample=True)."
            )

        coords = self.sample_coords          # (n, 2)
        values = self.sample_values           # (n,)
        n = len(values)
        model_template = self.best_model["model"]
        model_template.set_params(self.best_model["params"])

        bin_width = self._stored_bin_width
        estimator = self._stored_estimator

        # max_lag from stored multiplier
        xs = self.rdh.rioxarray_obj.x.values
        ys = self.rdh.rioxarray_obj.y.values
        x_ext = float(np.max(xs) - np.min(xs))
        y_ext = float(np.max(ys) - np.min(ys))
        max_lag = float(
            np.sqrt(x_ext ** 2 + y_ext ** 2) * self._stored_max_lag_multiplier
        )

        # ── 1. build covariance matrix from fitted model ────────────────
        D = cdist(coords, coords)
        total_sill = model_template.get_total_sill()
        if total_sill is None or total_sill <= 0:
            raise ValueError(
                "Model must be stationary with positive sill for "
                "generalized bootstrap."
            )

        gamma_mat = model_template(D)
        C = total_sill - gamma_mat
        np.fill_diagonal(C, total_sill)

        # small jitter for numerical PD
        jitter = 1e-10 * total_sill
        C += np.eye(n) * jitter

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # fall back to eigenvalue repair
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, jitter)
            C_repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(C_repaired)
            warnings.warn(
                "Covariance matrix was not positive-definite; eigenvalues "
                "were clipped. Bootstrap intervals may be approximate.",
                UserWarning,
            )

        L_inv = np.linalg.solve(L, np.eye(n))

        # ── 2. normal-score transform ───────────────────────────────────
        # rank → uniform → Gaussian
        ranks = rankdata(values, method="average")
        u = ranks / (n + 1)               # avoid 0 and 1
        z_ns = sp_norm.ppf(u)

        # store sorted original values for back-transform
        sorted_values = np.sort(values)

        # ── 3. decorrelate ──────────────────────────────────────────────
        w = L_inv @ z_ns                   # independent standard-ish residuals

        # ── 4. bootstrap loop ───────────────────────────────────────────
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_realizations)

        collected = []
        n_failed = 0

        for i in range(n_realizations):
            rng = np.random.default_rng(child_seeds[i])
            try:
                # resample decorrelated residuals with replacement
                w_star = rng.choice(w, size=n, replace=True)

                # re-correlate
                z_boot_ns = L @ w_star

                # back-transform: normal-score → original marginal
                # map quantile positions of z_boot_ns into sorted_values
                boot_u = sp_norm.cdf(z_boot_ns)
                indices = np.clip(
                    (boot_u * n).astype(int), 0, n - 1
                )
                z_boot = sorted_values[indices]

                # compute empirical variogram on original coords
                (n_bins, bin_counts, bssd, bssad,
                binned_sum_dist, _max_dist) = (
                    _bin_distances_and_squared_differences(
                        coords, z_boot, bin_width, max_lag,
                    )
                )

                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_lags = np.where(
                        bin_counts > 0,
                        binned_sum_dist / bin_counts,
                        np.nan,
                    )

                if estimator == "cressie_hawkins":
                    gamma = self._compute_cressie_hawkins(
                        bin_counts, bssad, self.MIN_PAIRS
                    )
                else:
                    gamma = self._compute_matheron(
                        bin_counts, bssd, self.MIN_PAIRS
                    )

                valid = ~np.isnan(gamma)
                n_kept = int(np.sum(valid))
                if n_kept < 3:
                    n_failed += 1
                    if verbose:
                        print(
                            f"  [{i+1}/{n_realizations}] SKIP: "
                            f"too few bins ({n_kept})"
                        )
                    continue

                boot_lags = mean_lags[valid]
                boot_gamma = gamma[valid]
                boot_counts = bin_counts[valid].astype(float)

                # Cressie (1985) WLS weights
                gamma_sq = np.maximum(
                    np.square(boot_gamma), np.finfo(float).eps
                )
                weights = boot_counts / gamma_sq

                model = copy.deepcopy(model_template)
                result = self._fit_single_composite_model(
                    model, boot_lags, boot_gamma, None, weights, rng=rng,
                )

                if result is not None:
                    collected.append(result["params"])
                    if verbose:
                        print(f"  [{i+1}/{n_realizations}] OK")
                else:
                    n_failed += 1
                    if verbose:
                        print(f"  [{i+1}/{n_realizations}] FIT FAILED")

            except Exception as e:
                n_failed += 1
                if verbose:
                    print(f"  [{i+1}/{n_realizations}] ERROR: {e}")
                continue

        if not collected:
            raise RuntimeError(
                f"All {n_realizations} generalized bootstrap "
                f"realisations failed."
            )

        param_samples = np.array(collected)

        # ── percentile summary (same convention as bootstrap_parameters) ─
        pct_keys = [5, 16, 50, 84, 95]
        param_names = list(model_template.param_names)
        param_percentiles = {}
        for j, name in enumerate(param_names):
            vals = param_samples[:, j]
            param_percentiles[name] = {
                f"p{p}": float(np.percentile(vals, p)) for p in pct_keys
            }

        self.bootstrap_param_samples = param_samples
        self.bootstrap_param_percentiles = param_percentiles
        self.best_model["param_samples"] = param_samples
        self.best_model["param_percentiles"] = param_percentiles

        if verbose or n_failed > 0:
            print(
                f"Generalized bootstrap complete: "
                f"{len(collected)}/{n_realizations} succeeded, "
                f"{n_failed} failed."
            )

        return param_samples


    # ─────────────────────────────────────────────────────────────────────
    # OPTION 2 — Parametric (model-based) bootstrap
    # ─────────────────────────────────────────────────────────────────────

    def parametric_bootstrap_parameters(
        self,
        n_realizations: int = 1000,
        *,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Estimate parameter uncertainty via parametric (model-based) bootstrap.

        Simulates Gaussian random fields from the fitted variogram model
        at the original sample locations, computes the empirical variogram
        of each simulated field, and refits the model structure.  This
        gives the tightest intervals because all simulated data obey the
        fitted model exactly — the only variability is finite-sample
        estimation noise.

        Procedure (per realisation)
        ---------------------------
        1. Generate z_sim = L ε,  where L = chol(C),  ε ~ N(0, I).
        2. Compute empirical variogram of (original coords, z_sim).
        3. Fit the winning model structure via WLS.
        4. Collect fitted parameters.

        Parameters
        ----------
        n_realizations : int
            Number of bootstrap realisations (≥ 1000 recommended).
        seed : int, optional
            Base seed for reproducibility.
        verbose : bool
            Print per-realisation progress.

        Returns
        -------
        param_samples : ndarray, shape (n_succeeded, n_params)

        Raises
        ------
        RuntimeError
            If :meth:`fit_model` has not been called, or if
            ``compute_empirical_variogram`` was called without
            ``return_sample=True``.

        Notes
        -----
        Intervals from this method are a *lower bound* on true parameter
        uncertainty because they assume perfect model specification.  If
        the true spatial covariance departs from the fitted model, the
        intervals will be anti-conservative.  Compare with
        :meth:`bootstrap_parameters` (upper bound) or
        :meth:`generalized_bootstrap_parameters` (intermediate) to gauge
        model-specification sensitivity.

        References
        ----------
        Diggle, P.J. & Ribeiro Jr, P.J. (2007). Model-based Geostatistics.
        Springer. §5.4.

        Sjöstedt-de Luna, S. & Young, A. (2003). Scandinavian Journal of
        Statistics, 30, 175–192.
        """
        # ── guards ──────────────────────────────────────────────────────
        if self.best_model is None:
            raise RuntimeError("No fitted model. Call fit_model() first.")
        if self.sample_coords is None or self.sample_values is None:
            raise RuntimeError(
                "No spatial sample stored. Re-run "
                "compute_empirical_variogram(return_sample=True)."
            )

        coords = self.sample_coords
        n = len(self.sample_values)
        model_template = self.best_model["model"]
        model_template.set_params(self.best_model["params"])

        bin_width = self._stored_bin_width
        estimator = self._stored_estimator

        xs = self.rdh.rioxarray_obj.x.values
        ys = self.rdh.rioxarray_obj.y.values
        x_ext = float(np.max(xs) - np.min(xs))
        y_ext = float(np.max(ys) - np.min(ys))
        max_lag = float(
            np.sqrt(x_ext ** 2 + y_ext ** 2) * self._stored_max_lag_multiplier
        )

        # ── 1. build covariance matrix from fitted model ────────────────
        D = cdist(coords, coords)
        total_sill = model_template.get_total_sill()
        if total_sill is None or total_sill <= 0:
            raise ValueError(
                "Model must be stationary with positive sill for "
                "parametric bootstrap."
            )

        gamma_mat = model_template(D)
        C = total_sill - gamma_mat
        np.fill_diagonal(C, total_sill)

        # small jitter for numerical PD
        jitter = 1e-10 * total_sill
        C += np.eye(n) * jitter

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, jitter)
            C_repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(C_repaired)
            warnings.warn(
                "Covariance matrix was not positive-definite; eigenvalues "
                "were clipped. Bootstrap intervals may be approximate.",
                UserWarning,
            )

        # ── 2. simulation loop ──────────────────────────────────────────
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_realizations)

        collected = []
        n_failed = 0

        for i in range(n_realizations):
            rng = np.random.default_rng(child_seeds[i])
            try:
                # simulate Gaussian field: z = L ε, ε ~ N(0, I)
                epsilon = rng.standard_normal(n)
                z_sim = L @ epsilon

                # compute empirical variogram on original coords
                (n_bins, bin_counts, bssd, bssad,
                binned_sum_dist, _max_dist) = (
                    _bin_distances_and_squared_differences(
                        coords, z_sim, bin_width, max_lag,
                    )
                )

                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_lags = np.where(
                        bin_counts > 0,
                        binned_sum_dist / bin_counts,
                        np.nan,
                    )

                if estimator == "cressie_hawkins":
                    gamma = self._compute_cressie_hawkins(
                        bin_counts, bssad, self.MIN_PAIRS
                    )
                else:
                    gamma = self._compute_matheron(
                        bin_counts, bssd, self.MIN_PAIRS
                    )

                valid = ~np.isnan(gamma)
                n_kept = int(np.sum(valid))
                if n_kept < 3:
                    n_failed += 1
                    if verbose:
                        print(
                            f"  [{i+1}/{n_realizations}] SKIP: "
                            f"too few bins ({n_kept})"
                        )
                    continue

                boot_lags = mean_lags[valid]
                boot_gamma = gamma[valid]
                boot_counts = bin_counts[valid].astype(float)

                # Cressie (1985) WLS weights
                gamma_sq = np.maximum(
                    np.square(boot_gamma), np.finfo(float).eps
                )
                weights = boot_counts / gamma_sq

                model = copy.deepcopy(model_template)
                result = self._fit_single_composite_model(
                    model, boot_lags, boot_gamma, None, weights, rng=rng,
                )

                if result is not None:
                    collected.append(result["params"])
                    if verbose:
                        print(f"  [{i+1}/{n_realizations}] OK")
                else:
                    n_failed += 1
                    if verbose:
                        print(f"  [{i+1}/{n_realizations}] FIT FAILED")

            except Exception as e:
                n_failed += 1
                if verbose:
                    print(f"  [{i+1}/{n_realizations}] ERROR: {e}")
                continue

        if not collected:
            raise RuntimeError(
                f"All {n_realizations} parametric bootstrap "
                f"realisations failed."
            )

        param_samples = np.array(collected)

        # ── percentile summary (same convention as bootstrap_parameters) ─
        pct_keys = [5, 16, 50, 84, 95]
        param_names = list(model_template.param_names)
        param_percentiles = {}
        for j, name in enumerate(param_names):
            vals = param_samples[:, j]
            param_percentiles[name] = {
                f"p{p}": float(np.percentile(vals, p)) for p in pct_keys
            }

        self.bootstrap_param_samples = param_samples
        self.bootstrap_param_percentiles = param_percentiles
        self.best_model["param_samples"] = param_samples
        self.best_model["param_percentiles"] = param_percentiles

        if verbose or n_failed > 0:
            print(
                f"Parametric bootstrap complete: "
                f"{len(collected)}/{n_realizations} succeeded, "
                f"{n_failed} failed."
            )

        return param_samples

    def akaike_weights(self, *, criterion: str = "aicc") -> np.ndarray:
        """Akaike weights over ``self.fitted_models`` (Burnham & Anderson, 2002).

        w_k = exp(-0.5 ΔIC_k) / Σ exp(-0.5 ΔIC_j), where ΔIC is the
        information-criterion difference from the best model.  Used to weight
        candidate structures when building the calibrated ensemble.
        """
        if not getattr(self, "fitted_models", None):
            raise RuntimeError("Call fit_model() first.")
        key = criterion if criterion in ("aic", "aicc", "bic") else "aicc"
        ic = np.array([m[key] for m in self.fitted_models], float)
        d = ic - np.nanmin(ic)
        w = np.exp(-0.5 * d)
        s = np.nansum(w)
        return w / s if s > 0 else np.ones_like(w) / len(w)

    def calibrated_parameter_ensemble(
        self, n_realizations: int = 300, *, criterion: str = "aicc",
        max_models: int = 3, weight_floor: float = 0.05,
        seed=None, store: bool = True, verbose: bool = False,
    ):
        """Build a realization + model-selection aware parameter ensemble.

        For the top Akaike-weighted model structures, runs the parametric
        (model-based) bootstrap to capture finite-sample realization variance,
        and tags each structure with its Akaike weight so that between-model
        (structure-selection) variance is also propagated to sigma_A
        (Buckland et al., 1997; Burnham & Anderson, 2002).  This replaces the
        anti-conservative single-model, within-fit envelope.

        Populates ``best_model['model_ensemble']`` with a list of
        ``(composite_model, weight, param_samples)`` and
        ``best_model['param_samples']`` with the winning model's draws.

        Requires ``compute_empirical_variogram(return_sample=True)`` and
        ``fit_model()``.  Returns the ensemble list.
        """
        if self.best_model is None:
            raise RuntimeError("Call fit_model() first.")
        if self.sample_coords is None or self.sample_values is None:
            raise RuntimeError(
                "No spatial sample stored. Re-run "
                "compute_empirical_variogram(return_sample=True)."
            )
        weights = self.akaike_weights(criterion=criterion)
        order = np.argsort(weights)[::-1]
        chosen = [int(i) for i in order if weights[i] >= weight_floor][:max_models]
        if not chosen:
            chosen = [int(order[0])]
        wsum = float(np.sum([weights[i] for i in chosen]))

        # parametric_bootstrap_parameters mutates best_model / bootstrap_param_samples;
        # preserve and restore caller state.
        saved_best = self.best_model
        saved_boot = getattr(self, "bootstrap_param_samples", None)
        ensemble = []
        try:
            for i in chosen:
                md = self.fitted_models[i]
                w = float(weights[i] / wsum) if wsum > 0 else 1.0 / len(chosen)
                self.best_model = md
                try:
                    samp = self.parametric_bootstrap_parameters(
                        n_realizations=max(50, int(round(n_realizations * w))),
                        seed=seed, verbose=verbose,
                    )
                except Exception as e:
                    if verbose:
                        print(f"  model {i} ensemble failed: {e}")
                    continue
                if samp is None or len(samp) == 0:
                    continue
                mm = copy.deepcopy(md["model"]); mm.set_params(md["params"])
                ensemble.append((mm, w, np.asarray(samp, float)))
        finally:
            self.best_model = saved_best
            self.bootstrap_param_samples = saved_boot

        if not ensemble:
            wm = copy.deepcopy(self.best_model["model"])
            wm.set_params(self.best_model["params"])
            ensemble = [(wm, 1.0,
                         np.atleast_2d(np.asarray(self.best_model["params"], float)))]
        tw = float(sum(w for _, w, _ in ensemble))
        ensemble = [(m, (w / tw if tw > 0 else 1.0 / len(ensemble)), s)
                    for m, w, s in ensemble]

        # winner's own samples for the single-model path / percentiles
        win_desc = self.best_model["description"]
        win_samples = None
        for m, _w, s in ensemble:
            if m.structural_description() == win_desc:
                win_samples = s
                break
        if win_samples is None:
            win_samples = ensemble[0][2]
        if store:
            self.best_model["model_ensemble"] = ensemble
            self.best_model["param_samples"] = win_samples
            self.bootstrap_param_samples = win_samples
        return ensemble

    # FittedVariogramModel

    @property
    def fitted_model(self) -> "FittedVariogramModel":
        """

        Returns
        -------
        FittedVariogramModel
            Populated with the best-fit model, parameters, covariance,
            and bootstrap samples (if :meth:`bootstrap_parameters` has
            been called).

        Raises
        ------
        RuntimeError
            If :meth:`fit_model` has not been called.
        """
        if self.best_model is None:
            raise RuntimeError(
                "No fitted model. Call fit_model() first."
            )
        bm = self.best_model
        model = bm["model"]
        model.set_params(bm["params"])

        return FittedVariogramModel(
            composite_model=model,
            params=bm["params"],
            param_cov=bm["param_cov"],
            rss=bm["rss"],
            aic=bm["aic"],
            bic=bm["bic"],
            param_samples=bm.get("param_samples"),
            warnings=bm.get("warnings", []),
            msspe=bm.get("msspe"),
            msspe_std=bm.get("msspe_std"),
            msspe_n_runs=bm.get("msspe_n_runs", 0),
            model_ensemble=bm.get("model_ensemble"),
        )

    #  plotting

    def plot_single_variogram(
        self,
        include_model: bool = False,
        include_bootstrap: bool = False,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """Plot the empirical variogram with optional fitted model overlay.

        Parameters
        ----------
        include_model : bool
            If True and :meth:`fit_model` has been called, overlay the
            best-fit model curve and annotate with model name, parameters,
            and selection criteria (MSSPE, AIC, BIC).
        include_bootstrap : bool
            If True and :meth:`bootstrap_parameters` has been called,
            overlay the bootstrap median model (dashed) and shaded
            1-sigma (16th–84th pctl) and 2-sigma (5th–95th pctl)
            envelopes.
        figsize : tuple
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.lags is None or self.variogram is None:
            raise RuntimeError(
                "No variogram data. "
                "Call compute_empirical_variogram() first."
            )

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        fig, axs = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 3]},
            figsize=figsize, sharex=True,
        )

        lags = self.lags
        bar_width = (
            (lags[1] - lags[0]) * 0.9
            if len(lags) > 1
            else (lags[0] if len(lags) else 1.0) * 0.9
        )

        # ── top panel: pair counts ──
        valid_c = (~np.isnan(self.pair_counts)) & (self.pair_counts > 0)
        axs[0].bar(
            lags[valid_c], self.pair_counts[valid_c],
            width=bar_width, color="orange", alpha=0.5,
        )
        axs[0].set_ylabel("Pair Count")
        axs[0].tick_params(labelbottom=False)

        # ── bottom panel: variogram ──
        ax = axs[1]

        # If a median variogram was replaced, show it as faded reference
        if hasattr(self, "_median_variogram") and self._median_variogram is not None:
            ax.plot(
                self._median_lags, self._median_variogram,
                "s-", color="gray", markersize=3, alpha=0.4, zorder=3,
                label="Median variogram (multi-sample)",
            )

        # empirical variogram points
        ax.plot(
            lags, self.variogram,
            "o-", color="blue", markersize=4, zorder=5,
            label="Empirical variogram",
        )

        # ── overlay fitted model ──
        lag_fine = None
        if include_model and self.best_model is not None:
            model = self.best_model["model"]
            model.set_params(self.best_model["params"])

            lag_fine = np.linspace(0, float(lags[-1]) * 1.05, 300)
            gamma_fine = model(lag_fine)
            ax.plot(
                lag_fine, gamma_fine,
                color="darkred", linewidth=2.2, zorder=6,
                label=f'Fitted: {self.best_model["description"]}',
            )

            # annotate with criteria
            parts = [self.best_model["description"]]
            if self.best_model["msspe"] is not None:
                parts.append(f'MSSPE={self.best_model["msspe"]:.3f}')
            parts.append(f'AICc={self.best_model["aicc"]:.1f}')
            parts.append(f'BIC={self.best_model["bic"]:.1f}')

            # show total sill and range (more useful than component sills)
            total_sill = model.get_total_sill()
            if total_sill is not None:
                parts.append(f"sill={total_sill:.4g}")
            # show practical range (largest range among components)
            for i, spec in enumerate(model._components):
                comp_params = model.get_component_params(i)
                if 'range' in spec.param_names:
                    ridx = spec.param_names.index('range')
                    parts.append(f"range={comp_params[ridx]:.4g}")
            if model.include_nugget:
                parts.append(f"nugget={model.get_nugget():.4g}")

            ax.set_title("  |  ".join(parts[:6]), fontsize=9)

        # ── overlay bootstrap envelopes ──
        has_bootstrap = (
            include_bootstrap
            and hasattr(self, "bootstrap_param_samples")
            and self.bootstrap_param_samples is not None
            and self.best_model is not None
        )
        if has_bootstrap:
            model_template = self.best_model["model"]
            samples = self.bootstrap_param_samples

            if lag_fine is None:
                lag_fine = np.linspace(0, float(lags[-1]) * 1.05, 300)

            # evaluate model at every bootstrap parameter set
            gamma_ensemble = np.empty((len(samples), len(lag_fine)))
            for k, params_k in enumerate(samples):
                m = copy.deepcopy(model_template)
                m.set_params(params_k)
                gamma_ensemble[k, :] = m(lag_fine)

            # lag-wise percentiles
            p5  = np.percentile(gamma_ensemble, 5, axis=0)
            p16 = np.percentile(gamma_ensemble, 16, axis=0)
            p50 = np.percentile(gamma_ensemble, 50, axis=0)
            p84 = np.percentile(gamma_ensemble, 84, axis=0)
            p95 = np.percentile(gamma_ensemble, 95, axis=0)

            # 2-sigma band (5th–95th)
            ax.fill_between(
                lag_fine, p5, p95,
                color="darkred", alpha=0.15, zorder=4,
                label="Bootstrap 90% (5th–95th)",
            )
            # 1-sigma band (16th–84th)
            ax.fill_between(
                lag_fine, p16, p84,
                color="darkred", alpha=0.25, zorder=4,
                label="Bootstrap 68% (16th–84th)",
            )
            # median curve
            ax.plot(
                lag_fine, p50,
                color="darkred", linewidth=1.5, linestyle="--", zorder=7,
                label="Bootstrap median",
            )

            # range / nugget annotations from bootstrap percentiles
            boot_pctl = self.bootstrap_param_percentiles
            range_colors = ["red", "green", "lightblue"]
            r_idx = 0
            for key, stats in boot_pctl.items():
                median_val = stats["p50"]
                lo_1sig = stats["p16"]
                hi_1sig = stats["p84"]
                lo_2sig = stats["p5"]
                hi_2sig = stats["p95"]

                if "range" in key:
                    c = range_colors[r_idx % len(range_colors)]
                    ax.axvline(
                        median_val, color=c, linewidth=1.8,
                        linestyle="-", zorder=5,
                        label=f'{key}: {median_val:.0f}',
                    )
                    ax.axvspan(
                        lo_1sig, hi_1sig,
                        color=c, alpha=0.20, zorder=1,
                    )
                    ax.axvspan(
                        lo_2sig, hi_2sig,
                        color=c, alpha=0.10, zorder=1,
                    )
                    r_idx += 1
                elif key == "nugget":
                    ax.axhline(
                        median_val, color="orange", linewidth=1.8,
                        linestyle="-", zorder=5,
                        label=f'nugget: {median_val:.4g}',
                    )
                    ax.axhspan(
                        lo_1sig, hi_1sig,
                        color="orange", alpha=0.20, zorder=1,
                    )
                    ax.axhspan(
                        lo_2sig, hi_2sig,
                        color="orange", alpha=0.10, zorder=1,
                    )

        ax.set_xlabel("Lag Distance (m)")
        ax.set_ylabel("Semivariance")

        # legend
        handles = []
        if hasattr(self, "_median_variogram") and self._median_variogram is not None:
            handles.append(
                Line2D([0], [0], marker="s", color="gray", alpha=0.4,
                       linestyle="-", markersize=3,
                       label="Median variogram (multi-sample)"),
            )
        handles.append(
            Line2D([0], [0], marker="o", color="blue",
                   linestyle="-", markersize=4, label="Empirical variogram"),
        )
        if include_model and self.best_model is not None:
            handles.append(
                Line2D([0], [0], color="darkred", linewidth=2.2,
                       label=f'Fitted: {self.best_model["description"]}')
            )
        if has_bootstrap:
            handles.append(
                Line2D([0], [0], color="darkred", linewidth=1.5,
                       linestyle="--", label="Bootstrap median"),
            )
            handles.append(
                Patch(facecolor="darkred", alpha=0.25,
                      label="Bootstrap 68% (1σ)"),
            )
            handles.append(
                Patch(facecolor="darkred", alpha=0.15,
                      label="Bootstrap 90% (2σ)"),
            )
        ax.legend(handles=handles, loc="lower right", fontsize=8)

        if not include_model and not has_bootstrap:
            ax.set_title(
                f"Empirical variogram ({self.estimator})",
            )

        plt.tight_layout()
        #return fig


#  GridVariogram 

class GridVariogram:
    """Run multiple independent SingleVariogram realisations and aggregate.

    Each realisation draws a fresh spatial sample, computes an empirical
    variogram, and (optionally) fits the full model selection pipeline.
    The ensemble captures both model selection uncertainty and parameter uncertainty.

    Parameters
    ----------
    raster_data_handler : RasterDataHandler
        Provides raster data and sampling.
    n_realizations : int
        Number of independent SingleVariogram realisations to run.

    """

    def __init__(
        self,
        raster_data_handler: RasterDataHandler,
        n_realizations: int = 50,
    ):
        self.rdh = raster_data_handler
        self.n_realizations = n_realizations

        # per-realisation SingleVariogram objects
        self.variograms: List[SingleVariogram] = []

        # aggregated results (populated by run())
        self.model_counts: Optional[Dict[str, int]] = None
        self.model_fractions: Optional[Dict[str, float]] = None
        self.central_model_name: Optional[str] = None
        self.central_trend_order: int = 0
        self.central_params: Optional[Dict[str, Dict[str, float]]] = None
        self.all_criteria: Optional[pd.DataFrame] = None
        self.n_failed: int = 0

    def run(
        self,
        # empirical variogram parameters
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        bin_width: float,
        max_lag_multiplier: float = 1 / 3,
        estimator: str = "matheron",
        n_samples: int = 1,
        # model fitting parameters
        fit_model: bool = True,
        model_types: Optional[List[str]] = None,
        include_nugget: bool = True,
        max_components: int = 3,
        criterion: str = "aicc",
        msspe_n_subset: int = 500,
        msspe_n_runs: int = 10,
        msspe_prefilter: int = 0,
        *,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Run the ensemble of SingleVariogram realisations.

        Each realisation draws ``n_samples`` independent spatial samples,
        computes one empirical variogram per sample, takes the **median**
        variogram across those samples, and (optionally) fits models to
        that smoothed curve via Weighted Least Squares.

        Parameters
        ----------
        area_side, samples_per_area, max_samples, bin_width,
        max_lag_multiplier, estimator
            Passed through to
            :meth:`SingleVariogram.compute_empirical_variogram` (or the
            internal sampling helper when ``n_samples > 1``).
        n_samples : int
            Number of independent spatial samples drawn *within* each
            realisation.  When ``n_samples > 1``, the median empirical
            variogram is used as the fitting target, producing more
            stable fits.  Default is 1 (single sample per realisation,
            same as before).
        fit_model : bool
            If True, run :meth:`SingleVariogram.fit_model` on each
            realisation and aggregate model selection results.
        criterion : {'aicc', 'aic', 'bic', 'msspe'}
            Model selection criterion passed to each
            :meth:`SingleVariogram.fit_model`.
        model_types, include_nugget, max_components, msspe_n_subset,
        msspe_n_runs, msspe_prefilter
            Passed through to :meth:`SingleVariogram.fit_model`.
        seed : int, optional
            Base seed; each realisation gets a child seed.
        verbose : bool
            Print progress.
        """
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(self.n_realizations)

        self.variograms = []
        self.n_failed = 0
        self.n_samples = n_samples

        # store for bootstrap_parameters() reuse
        self._stored_area_side = area_side
        self._stored_samples_per_area = samples_per_area
        self._stored_max_samples = max_samples
        self._stored_bin_width = bin_width
        self._stored_max_lag_multiplier = max_lag_multiplier
        self._stored_estimator = estimator
        self._stored_n_samples = n_samples

        # MSSPE needs the spatial sample retained
        needs_sample = fit_model and criterion == "msspe"

        for i in range(self.n_realizations):
            run_seed = int(child_seeds[i].generate_state(1)[0])
            sv = SingleVariogram(self.rdh)
            try:
                if n_samples <= 1:
                    # single sample per realisation (original behaviour) 
                    sv.compute_empirical_variogram(
                        area_side=area_side,
                        samples_per_area=samples_per_area,
                        max_samples=max_samples,
                        bin_width=bin_width,
                        max_lag_multiplier=max_lag_multiplier,
                        seed=run_seed,
                        estimator=estimator,
                        return_sample=needs_sample,
                    )
                else:
                    #  n_samples: draw multiple, fit on median 
                    self._compute_median_variogram(
                        sv,
                        n_samples=n_samples,
                        area_side=area_side,
                        samples_per_area=samples_per_area,
                        max_samples=max_samples,
                        bin_width=bin_width,
                        max_lag_multiplier=max_lag_multiplier,
                        estimator=estimator,
                        return_sample=needs_sample,
                        seed=run_seed,
                    )

                if fit_model:
                    sv.fit_model(
                        model_types=model_types,
                        include_nugget=include_nugget,
                        max_components=max_components,
                        criterion=criterion,
                        msspe_n_subset=msspe_n_subset,
                        msspe_n_runs=msspe_n_runs,
                        msspe_prefilter=msspe_prefilter,
                        seed=run_seed,
                    )

                self.variograms.append(sv)

                if verbose:
                    status = "OK"
                    if sv.best_model is not None:
                        desc = sv.best_model["description"]
                        parts = [desc]
                        aicc = sv.best_model.get("aicc")
                        if aicc is not None:
                            parts.append(f"AICc={aicc:.1f}")
                        msspe = sv.best_model.get("msspe")
                        if msspe is not None:
                            parts.append(f"MSSPE={msspe:.3f}")
                        status = "  ".join(parts)
                    print(f"  [{i + 1}/{self.n_realizations}] {status}")

            except Exception as e:
                self.n_failed += 1
                if verbose:
                    print(f"  [{i + 1}/{self.n_realizations}] FAILED: {e}")
                continue

        if not self.variograms:
            raise RuntimeError(
                f"All {self.n_realizations} realisations failed."
            )

        # aggregate results
        if fit_model:
            self._aggregate_model_results()

    @staticmethod
    def _compute_median_variogram(
        sv: "SingleVariogram",
        n_samples: int,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        bin_width: float,
        max_lag_multiplier: float,
        estimator: str,
        return_sample: bool,
        seed: int,
    ) -> None:
        """Draw *n_samples* independent variograms and assign the median to *sv*.

        The median is taken bin-by-bin across the ``n_samples`` empirical
        variograms.  Pair counts are averaged (mean) since they inform
        Cressie weights.  If ``return_sample`` is True, the coordinates
        and values from one of the samples are retained for downstream
        MSSPE / LOOCV computation.

        Parameters
        ----------
        sv : SingleVariogram
            Target object whose attributes will be populated.
        n_samples : int
            Number of independent spatial samples to draw.
        Other parameters
            Same semantics as
            :meth:`SingleVariogram.compute_empirical_variogram`.
        """
        # compute max_lag (same logic as SingleVariogram.compute_empirical_variogram)
        xs = sv.rdh.rioxarray_obj.x.values
        ys = sv.rdh.rioxarray_obj.y.values
        x_extent = float(np.max(xs) - np.min(xs))
        y_extent = float(np.max(ys) - np.min(ys))
        diag = np.sqrt(x_extent ** 2 + y_extent ** 2)
        max_lag = float(diag * max_lag_multiplier)

        # generate child seeds for the inner samples
        inner_ss = np.random.SeedSequence(seed)
        inner_seeds = inner_ss.spawn(n_samples)

        all_gamma = []
        all_counts = []
        all_mean_lags = [] 
        kept_coords = None
        kept_values = None

        for j in range(n_samples):
            s = int(inner_seeds[j].generate_state(1)[0])
            bin_counts, gamma, n_bins_run, coords, values, mean_lags = (
            sv._single_variogram_run(
                area_side, samples_per_area, max_samples,
                bin_width, max_lag, estimator, seed=s,
                )
            )
            all_gamma.append(gamma)
            all_counts.append(bin_counts.astype(float))
            all_mean_lags.append(mean_lags)
            

            # keep the first valid sample for MSSPE
            if return_sample and kept_coords is None:
                kept_coords = coords
                kept_values = values

        # stack and compute median variogram, mean pair counts
        gamma_arr = np.array(all_gamma)        # (n_samples, n_bins)
        count_arr = np.array(all_counts)

        with np.errstate(all="ignore"):
            median_gamma = np.nanmedian(gamma_arr, axis=0)
            mean_counts = np.nanmean(count_arr, axis=0)
            mean_lags_arr = np.array(all_mean_lags)
            median_lags = np.nanmedian(mean_lags_arr, axis=0)

        # keep only valid (non-NaN) bins
        valid = ~np.isnan(median_gamma)
        n_kept = int(np.sum(valid))

        sv.variogram = median_gamma[valid]
        sv.pair_counts = mean_counts[valid]
        sv.lags = median_lags[valid]
        sv.n_bins = n_kept
        sv.estimator = estimator

        if return_sample and kept_coords is not None:
            sv.sample_coords = kept_coords
            sv.sample_values = kept_values

    def _aggregate_model_results(self) -> None:
        """Aggregate candidate results across all realisations.

        Attributes set
        --------------
        model_counts : dict
            {description: n_times_selected_as_best}
        model_fractions : dict
            {description: fraction_of_realisations_selected_as_best}
        central_model_name : str
            The modal best-model (highest count, MSSPE tiebreaker).
        central_params : dict
            Parameter statistics for the central model (from *all*
            realisations that fitted it, not just wins).
        all_criteria : pd.DataFrame
            One row per model structure with best-count, average
            AIC/BIC/MSSPE, and spread.
        """
        from collections import Counter, defaultdict

        n_ok = len(self.variograms)

        # ── 1. tally best-model selections ─────────────────────────
        best_descriptions: List[str] = []
        # also collect the best-model dicts keyed by description
        best_by_description: Dict[str, list] = defaultdict(list)

        for sv in self.variograms:
            if sv.best_model is not None:
                desc = sv.best_model["description"]
                best_descriptions.append(desc)
                best_by_description[desc].append(sv.best_model)

        if not best_descriptions:
            return

        best_counts = Counter(best_descriptions)

        # ── 2. collect ALL candidates from ALL realisations ────────
        # keyed by description → list of fitted-model dicts
        by_description: Dict[str, list] = defaultdict(list)

        for sv in self.variograms:
            if not hasattr(sv, "fitted_models") or sv.fitted_models is None:
                continue
            for fm in sv.fitted_models:
                by_description[fm["description"]].append(fm)

        # ── helper: extract param dict from a fitted-model record ──
        def _extract_params(fm_dict) -> Dict[str, float]:
            model = fm_dict["model"]
            model.set_params(fm_dict["params"])
            rec: Dict[str, float] = {}
            for i, spec in enumerate(model._components):
                comp_params = model.get_component_params(i)
                for j, pname in enumerate(spec.param_names):
                    key = f"{model.component_names[i]}_{pname}"
                    rec[key] = float(comp_params[j])
            if model.include_nugget:
                rec["nugget"] = float(model.get_nugget())
            return rec

        def _summarise_params(
            param_records: List[Dict[str, float]],
        ) -> Dict[str, Dict[str, float]]:
            all_keys: set = set()
            for rec in param_records:
                all_keys.update(rec.keys())
            stats: Dict[str, Dict[str, float]] = {}
            for key in sorted(all_keys):
                vals = np.array([
                    rec.get(key, np.nan) for rec in param_records
                ])
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    stats[key] = {
                        "median": float(np.median(vals)),
                        "std": float(np.std(vals)),
                        "p16": float(np.percentile(vals, 16)),
                        "p84": float(np.percentile(vals, 84)),
                        "p2_5": float(np.percentile(vals, 2.5)),
                        "p97_5": float(np.percentile(vals, 97.5)),
                        "count": len(vals),
                    }
            return stats

        # ── 3. build summary table (one row per model structure) ───
        summary_rows = []
        for desc, records in by_description.items():
            aics = np.array([r["aic"] for r in records])
            aiccs = np.array([
                r["aicc"] if r.get("aicc") is not None else np.nan
                for r in records
            ])
            bics = np.array([r["bic"] for r in records])
            msspes = np.array([
                r["msspe"] if r.get("msspe") is not None else np.nan
                for r in records
            ])

            n_fitted = len(records)
            n_best = best_counts.get(desc, 0)

            # extract trend_order from the first record (same for all
            # records with the same description)
            trend_order = records[0].get("trend_order", 0)

            row: Dict[str, Any] = {
                "model": desc,
                "trend_order": trend_order,
                "best_count": n_best,
                "best_fraction": n_best / n_ok,
                "fitted_count": n_fitted,
                "AIC_mean": float(np.mean(aics)),
                "AIC_std": float(np.std(aics)),
            }

            valid_aicc = aiccs[np.isfinite(aiccs)]
            if len(valid_aicc) > 0:
                row["AICc_mean"] = float(np.mean(valid_aicc))
                row["AICc_std"] = float(np.std(valid_aicc))
            else:
                row["AICc_mean"] = None
                row["AICc_std"] = None

            row["BIC_mean"] = float(np.mean(bics))
            row["BIC_std"] = float(np.std(bics))

            valid_msspe = msspes[np.isfinite(msspes)]
            if len(valid_msspe) > 0:
                row["MSSPE_mean"] = float(np.mean(valid_msspe))
                row["MSSPE_std"] = float(np.std(valid_msspe))
            else:
                row["MSSPE_mean"] = None
                row["MSSPE_std"] = None

            # param stats from ALL fitted instances (for table display)
            row["param_stats"] = _summarise_params(
                [_extract_params(r) for r in records]
            )

            # param stats from only BEST-selected instances (for central model)
            best_records = best_by_description.get(desc, [])
            if best_records:
                row["best_param_stats"] = _summarise_params(
                    [_extract_params(r) for r in best_records]
                )
            else:
                row["best_param_stats"] = {}

            summary_rows.append(row)

        # ── 4. sort by best_count desc, then AICc asc ──────────────
        def _sort_key(row):
            aicc = row.get("AICc_mean")
            aicc_val = aicc if aicc is not None and np.isfinite(aicc) else np.inf
            return (-row["best_count"], aicc_val)

        summary_rows.sort(key=_sort_key)

        # ── 5. store aggregated attributes ─────────────────────────
        self.model_counts = {
            r["model"]: r["best_count"] for r in summary_rows
        }
        self.model_fractions = {
            r["model"]: r["best_fraction"] for r in summary_rows
        }

        # central model = highest best_count, AICc tiebreaker
        self.central_model_name = summary_rows[0]["model"]
        self.central_trend_order = summary_rows[0].get("trend_order", 0)
        # use parameters only from realisations where it won
        self.central_params = summary_rows[0]["best_param_stats"]

        # DataFrame (without param_stats column)
        df_rows = []
        for r in summary_rows:
            df_rows.append({
                "model": r["model"],
                "trend_order": r.get("trend_order", 0),
                "best_count": r["best_count"],
                "best_fraction": r["best_fraction"],
                "fitted_count": r["fitted_count"],
                "AIC_mean": r["AIC_mean"],
                "AIC_std": r["AIC_std"],
                "AICc_mean": r.get("AICc_mean"),
                "AICc_std": r.get("AICc_std"),
                "BIC_mean": r["BIC_mean"],
                "BIC_std": r["BIC_std"],
                "MSSPE_mean": r["MSSPE_mean"],
                "MSSPE_std": r["MSSPE_std"],
            })
        self.all_criteria = pd.DataFrame(df_rows)

        # full summary rows (with param_stats) for programmatic access
        self._summary_rows = summary_rows

        # Raw central-model parameter vectors across realisations. These act
        # as fallback ``param_samples`` for :attr:`fitted_model`, so the
        # RegionalUncertaintyEstimator obtains min/max bounds from the
        # realization-to-realization spread even when no explicit bootstrap
        # has been run. All records under one description share the same
        # parameter layout, so stacking is safe.
        central_records = by_description.get(self.central_model_name, [])
        try:
            self.ensemble_param_samples = (
                np.array([r["params"] for r in central_records])
                if central_records else None
            )
        except Exception:
            self.ensemble_param_samples = None

    # parameter bootstrap

    def bootstrap_parameters(
        self,
        n_realizations: int = 100,
        area_side: float = None,
        samples_per_area: float = None,
        max_samples: int = None,
        bin_width: float = None,
        max_lag_multiplier: float = None,
        n_samples: int = None,
        estimator: str = None,
        *,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Estimate parameter uncertainty for the central model via bootstrap.

        For each realisation, draws new spatial sample(s), computes a
        (median-stabilised) empirical variogram, and fits the
        central model structure selected by :meth:`run`.  The result
        is a ``(n_succeeded, n_params)`` array whose spread quantifies
        parameter uncertainty conditional on the chosen model family.

        Parameters
        ----------
        n_realizations : int
            Number of bootstrap realisations.
        area_side, samples_per_area, max_samples, bin_width,
        max_lag_multiplier, estimator
            Sampling / variogram parameters.  If ``None``, reuses the
            values from the most recent :meth:`run` call.
        n_samples : int, optional
            Number of inner spatial samples per realisation (median
            variogram stabilisation).  If ``None``, reuses the value
            from the most recent :meth:`run` call.
        seed : int, optional
            Base seed for reproducibility (child seeds spawned via
            ``SeedSequence``).
        verbose : bool
            Print per-realisation progress.

        Returns
        -------
        param_samples : ndarray, shape (n_succeeded, n_params)
            Fitted parameter values for each successful realisation.
            Column order matches ``model_template.param_names``.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called yet.

        Notes
        -----
        When ``n_samples > 1``, each bootstrap realisation computes
        the median variogram across multiple independent spatial
        samples before fitting — the same stabilisation used in
        :meth:`run`.
        """
        if self.central_model_name is None:
            raise RuntimeError(
                "No central model. Call run() first."
            )

        # resolve defaults from stored parameters
        def _resolve(val, attr_name):
            if val is not None:
                return val
            stored = getattr(self, attr_name, None)
            if stored is None:
                raise ValueError(
                    f"'{attr_name.replace('_stored_', '')}' not provided "
                    f"and no stored value from run()."
                )
            return stored

        area_side = _resolve(area_side, "_stored_area_side")
        samples_per_area = _resolve(samples_per_area, "_stored_samples_per_area")
        max_samples = _resolve(max_samples, "_stored_max_samples")
        bin_width = _resolve(bin_width, "_stored_bin_width")
        max_lag_multiplier = _resolve(max_lag_multiplier, "_stored_max_lag_multiplier")
        estimator = _resolve(estimator, "_stored_estimator")
        n_samples = _resolve(n_samples, "_stored_n_samples")

        # find the winning model template from fitted realisations
        model_template = None
        for sv in self.variograms:
            if (
                sv.best_model is not None
                and sv.best_model["description"] == self.central_model_name
            ):
                model_template = sv.best_model["model"]
                break

        if model_template is None:
            raise RuntimeError(
                f"Could not find a fitted model matching the central "
                f"model '{self.central_model_name}' in any realisation."
            )

        # reproducible child seeds
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_realizations)

        collected = []
        n_failed = 0

        for i in range(n_realizations):
            run_seed = int(child_seeds[i].generate_state(1)[0])
            sv = SingleVariogram(self.rdh)
            try:
                if n_samples <= 1:
                    # single sample per realisation
                    sv.compute_empirical_variogram(
                        area_side=area_side,
                        samples_per_area=samples_per_area,
                        max_samples=max_samples,
                        bin_width=bin_width,
                        max_lag_multiplier=max_lag_multiplier,
                        seed=run_seed,
                        estimator=estimator,
                        return_sample=False,
                    )
                else:
                    # median-stabilised variogram
                    self._compute_median_variogram(
                        sv,
                        n_samples=n_samples,
                        area_side=area_side,
                        samples_per_area=samples_per_area,
                        max_samples=max_samples,
                        bin_width=bin_width,
                        max_lag_multiplier=max_lag_multiplier,
                        estimator=estimator,
                        return_sample=False,
                        seed=run_seed,
                    )

                if sv.lags is None or sv.variogram is None or len(sv.lags) < 3:
                    n_failed += 1
                    if verbose:
                        print(f"  [{i + 1}/{n_realizations}] SKIP: too few bins")
                    continue

                # Cressie (1985) weights
                gamma_sq = np.square(sv.variogram)
                gamma_sq = np.where(
                    gamma_sq < np.finfo(float).eps,
                    np.finfo(float).eps,
                    gamma_sq,
                )
                weights = sv.pair_counts / gamma_sq

                # deep-copy winning model for independent fit
                model = copy.deepcopy(model_template)

                result = sv._fit_single_composite_model(
                    model, sv.lags, sv.variogram, None, weights,
                    rng=np.random.default_rng(run_seed),
                )

                if result is not None:
                    collected.append(result["params"])
                    if verbose:
                        print(f"  [{i + 1}/{n_realizations}] OK")
                else:
                    n_failed += 1
                    if verbose:
                        print(f"  [{i + 1}/{n_realizations}] FIT FAILED")

            except Exception as e:
                n_failed += 1
                if verbose:
                    print(f"  [{i + 1}/{n_realizations}] ERROR: {e}")
                continue

        if not collected:
            raise RuntimeError(
                f"All {n_realizations} bootstrap realisations failed."
            )

        param_samples = np.array(collected)  # (n_succeeded, n_params)

        # compute percentile summary per parameter
        pct_keys = [5, 16, 50, 84, 95]
        param_names = list(model_template.param_names)
        param_percentiles = {}
        for j, name in enumerate(param_names):
            vals = param_samples[:, j]
            param_percentiles[name] = {
                f"p{p}": float(np.percentile(vals, p)) for p in pct_keys
            }

        # store for downstream use
        self.bootstrap_param_samples = param_samples
        self.bootstrap_param_percentiles = param_percentiles

        if verbose or n_failed > 0:
            print(
                f"Bootstrap complete: {len(collected)}/{n_realizations} "
                f"succeeded, {n_failed} failed."
            )

        return param_samples

    def calibrated_parameter_ensemble(self, seed=None, min_realizations: int = 2):
        """Build the realization-based model ensemble used for uncertainty bounds.

        One entry per successful realisation: a deep copy of that
        realisation's AICc-winning model at its fitted parameters, weighted
        equally, with that realisation's parameter vector (plus its
        parametric-bootstrap samples, when present) as the sample set.  The
        :class:`RegionalUncertaintyEstimator` propagates sigma_A per entry and
        takes percentiles of the pooled *output* distribution
        (``_propagate_model_averaged``), so the reported bounds contain
        realization-to-realization sampling variance and — whenever model
        selection flips between realisations — structure-selection variance.

        Design note: unlike the Akaike-weighted candidate averaging of
        :meth:`SingleVariogram.calibrated_parameter_ensemble`, no *losing*
        candidates are averaged in.  On unambiguous data every entry shares
        one structure and the envelope reduces to the pure realization
        spread, so this estimator cannot dilute the clean case — the failure
        mode of candidate averaging identified in the calibration
        experiments (see ``validation/check_gridvariogram_bounds.py``).

        Parameters
        ----------
        seed : int, optional
            Unused; accepted for interface compatibility with the
            SingleVariogram method (the ensemble is deterministic given
            ``run()``).
        min_realizations : int
            Minimum number of successful realisations required; below this
            no ensemble is built and ``None`` is returned with a warning.

        Returns
        -------
        list of (composite_model, weight, param_samples) or None
            Also stored for the :attr:`fitted_model` bridge.
        """
        if not self.variograms:
            raise RuntimeError("No realisations. Call run() first.")
        entries = []
        for sv in self.variograms:
            bm = sv.best_model
            if bm is None:
                continue
            model = copy.deepcopy(bm["model"])
            params = np.asarray(bm["params"], dtype=float)
            samples = bm.get("param_samples")
            if samples is not None and len(samples) > 0:
                samples = np.atleast_2d(np.asarray(samples, dtype=float))
            else:
                samples = params.reshape(1, -1)
            model.set_params(params.copy())
            entries.append((model, samples))
        if len(entries) < min_realizations:
            warnings.warn(
                f"Only {len(entries)} realisation(s) carry a fitted model; "
                f"need >= {min_realizations} for a realization ensemble. "
                f"Increase n_realizations or call bootstrap_parameters().",
                UserWarning, stacklevel=2,
            )
            self._realization_model_ensemble = None
            return None
        w = 1.0 / len(entries)
        ensemble = [(m, w, smp) for m, smp in entries]
        self._realization_model_ensemble = ensemble
        return ensemble



    # ─────────────────────────────────────────────────────────────────────
    # OPTION 1 — Generalized (spatial) bootstrap for GridVariogram
    # ─────────────────────────────────────────────────────────────────────

    def generalized_bootstrap_parameters(
        self,
        n_realizations: int = 1000,
        *,
        max_cov_points: Optional[int] = 3000,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Generalized spatial bootstrap for the ensemble's central model.

        Draws one reference spatial sample from the raster, builds the
        covariance matrix from the central model, then runs the
        decorrelation–resample–recorrelation cycle from Olea &
        Pardo-Igúzquiza (2011) to produce bootstrap parameter samples.

        Parameters
        ----------
        n_realizations : int
            Number of bootstrap realisations (≥ 1000 recommended).
        max_cov_points : int or None, optional
            Cap on the number of reference points used to build the
            dense covariance matrix (default 3000). This method holds
            several ``n × n`` matrices at once (``C``, ``L``, ``L_inv``),
            so the full sampling density used by :meth:`run` would
            exhaust RAM. When the reference sample exceeds this cap it is
            randomly subsampled (reproducibly, from ``seed``). Set to
            ``None`` to disable the cap (only safe for small rasters).
        seed : int, optional
            Base seed for reproducibility.  The first child seed is used
            to draw the reference sample; remaining seeds drive the
            bootstrap iterations.
        verbose : bool
            Print per-realisation progress.

        Returns
        -------
        param_samples : ndarray, shape (n_succeeded, n_params)

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called yet.
        """
        if self.central_model_name is None:
            raise RuntimeError("No central model. Call run() first.")

        # ── resolve stored parameters ───────────────────────────────────
        def _resolve(val, attr_name):
            if val is not None:
                return val
            stored = getattr(self, attr_name, None)
            if stored is None:
                raise ValueError(
                    f"'{attr_name.replace('_stored_', '')}' not provided "
                    f"and no stored value from run()."
                )
            return stored

        area_side = _resolve(None, "_stored_area_side")
        samples_per_area = _resolve(None, "_stored_samples_per_area")
        max_samples = _resolve(None, "_stored_max_samples")
        bin_width = _resolve(None, "_stored_bin_width")
        max_lag_multiplier = _resolve(None, "_stored_max_lag_multiplier")
        estimator = _resolve(None, "_stored_estimator")

        # ── find central model template ─────────────────────────────────
        model_template = None
        for sv_obj in self.variograms:
            if (
                sv_obj.best_model is not None
                and sv_obj.best_model["description"] == self.central_model_name
            ):
                model_template = sv_obj.best_model["model"]
                break

        if model_template is None:
            raise RuntimeError(
                f"Could not find a fitted model matching the central "
                f"model '{self.central_model_name}' in any realisation."
            )

        # ── draw one reference spatial sample ───────────────────────────
        ss = np.random.SeedSequence(seed)
        # child 0 → reference sample, children 1..N → bootstrap iterations
        ref_seed, *boot_child_seeds_raw = ss.spawn(n_realizations + 1)
        ref_seed_int = int(ref_seed.generate_state(1)[0])

        ref_sv = SingleVariogram(self.rdh)
        ref_sv.compute_empirical_variogram(
            area_side=area_side,
            samples_per_area=samples_per_area,
            max_samples=max_samples,
            bin_width=bin_width,
            max_lag_multiplier=max_lag_multiplier,
            seed=ref_seed_int,
            estimator=estimator,
            return_sample=True,
        )

        coords = ref_sv.sample_coords       # (n, 2)
        values = ref_sv.sample_values        # (n,)
        n = len(values)

        # ── cap covariance dimension to bound memory / Cholesky cost ────
        # See parametric_bootstrap_parameters: the dense n×n covariance
        # (plus L and L_inv here) would OOM the kernel at run()'s full
        # sampling density, so subsample reference points past the cap.
        # coords and values share the same indices to stay aligned.
        if max_cov_points is not None and n > max_cov_points:
            sub_rng = np.random.default_rng(ref_seed_int)
            sub_idx = sub_rng.choice(n, size=max_cov_points, replace=False)
            coords = coords[sub_idx]
            values = values[sub_idx]
            if verbose:
                print(
                    f"  Subsampled reference points {n} -> {max_cov_points} "
                    f"for the {max_cov_points}x{max_cov_points} covariance "
                    f"matrix (set max_cov_points to adjust)."
                )
            n = max_cov_points

        xs = self.rdh.rioxarray_obj.x.values
        ys = self.rdh.rioxarray_obj.y.values
        x_ext = float(np.max(xs) - np.min(xs))
        y_ext = float(np.max(ys) - np.min(ys))
        max_lag = float(
            np.sqrt(x_ext ** 2 + y_ext ** 2) * max_lag_multiplier
        )

        # ── build covariance matrix from central model ──────────────────
        total_sill = model_template.get_total_sill()
        if total_sill is None or total_sill <= 0:
            raise ValueError(
                "Central model must be stationary with positive sill."
            )

        # Build C in place and free intermediates promptly to keep peak
        # memory near a single n×n array rather than several.
        D = cdist(coords, coords)
        gamma_mat = model_template(D)
        del D
        C = total_sill - gamma_mat
        del gamma_mat
        # Diagonal = covariance at lag 0 (total_sill) plus a tiny jitter
        # for numerical positive-definiteness; avoids allocating np.eye(n).
        jitter = 1e-10 * total_sill
        np.fill_diagonal(C, total_sill + jitter)

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, jitter)
            C_repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(C_repaired)
            warnings.warn(
                "Covariance matrix was not positive-definite; eigenvalues "
                "were clipped. Bootstrap intervals may be approximate.",
                UserWarning,
            )
        del C

        # Triangular solve for L⁻¹ (L is lower-triangular) — cheaper and
        # lighter than a general solve against a full identity matrix.
        from scipy.linalg import solve_triangular
        L_inv = solve_triangular(L, np.eye(n), lower=True)

        # ── normal-score transform ──────────────────────────────────────
        ranks = rankdata(values, method="average")
        u = ranks / (n + 1)
        z_ns = sp_norm.ppf(u)
        sorted_values = np.sort(values)

        # ── decorrelate ─────────────────────────────────────────────────
        w = L_inv @ z_ns

        # ── bootstrap loop ──────────────────────────────────────────────
        collected = []
        n_failed = 0

        for i in range(n_realizations):
            rng = np.random.default_rng(boot_child_seeds_raw[i])
            try:
                w_star = rng.choice(w, size=n, replace=True)
                z_boot_ns = L @ w_star

                boot_u = sp_norm.cdf(z_boot_ns)
                indices = np.clip((boot_u * n).astype(int), 0, n - 1)
                z_boot = sorted_values[indices]

                (n_bins, bin_counts, bssd, bssad,
                binned_sum_dist, _max_dist) = (
                    _bin_distances_and_squared_differences(
                        coords, z_boot, bin_width, max_lag,
                    )
                )

                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_lags = np.where(
                        bin_counts > 0,
                        binned_sum_dist / bin_counts,
                        np.nan,
                    )

                if estimator == "cressie_hawkins":
                    gamma = SingleVariogram._compute_cressie_hawkins(
                        bin_counts, bssad, SingleVariogram.MIN_PAIRS
                    )
                else:
                    gamma = SingleVariogram._compute_matheron(
                        bin_counts, bssd, SingleVariogram.MIN_PAIRS
                    )

                valid = ~np.isnan(gamma)
                n_kept = int(np.sum(valid))
                if n_kept < 3:
                    n_failed += 1
                    if verbose:
                        print(
                            f"  [{i+1}/{n_realizations}] SKIP: "
                            f"too few bins ({n_kept})"
                        )
                    continue

                boot_lags = mean_lags[valid]
                boot_gamma = gamma[valid]
                boot_counts = bin_counts[valid].astype(float)

                gamma_sq = np.maximum(
                    np.square(boot_gamma), np.finfo(float).eps
                )
                weights = boot_counts / gamma_sq

                model = copy.deepcopy(model_template)
                result = ref_sv._fit_single_composite_model(
                    model, boot_lags, boot_gamma, None, weights, rng=rng,
                )

                if result is not None:
                    collected.append(result["params"])
                    if verbose:
                        print(f"  [{i+1}/{n_realizations}] OK")
                else:
                    n_failed += 1
                    if verbose:
                        print(f"  [{i+1}/{n_realizations}] FIT FAILED")

            except Exception as e:
                n_failed += 1
                if verbose:
                    print(f"  [{i+1}/{n_realizations}] ERROR: {e}")
                continue

        if not collected:
            raise RuntimeError(
                f"All {n_realizations} generalized bootstrap "
                f"realisations failed."
            )

        param_samples = np.array(collected)

        pct_keys = [5, 16, 50, 84, 95]
        param_names = list(model_template.param_names)
        param_percentiles = {}
        for j, name in enumerate(param_names):
            vals = param_samples[:, j]
            param_percentiles[name] = {
                f"p{p}": float(np.percentile(vals, p)) for p in pct_keys
            }

        self.bootstrap_param_samples = param_samples
        self.bootstrap_param_percentiles = param_percentiles

        if verbose or n_failed > 0:
            print(
                f"Generalized bootstrap complete: "
                f"{len(collected)}/{n_realizations} succeeded, "
                f"{n_failed} failed."
            )

        return param_samples


    # ─────────────────────────────────────────────────────────────────────
    # OPTION 2 — Parametric (model-based) bootstrap for GridVariogram
    # ─────────────────────────────────────────────────────────────────────

    def parametric_bootstrap_parameters(
        self,
        n_realizations: int = 1000,
        *,
        max_cov_points: Optional[int] = 3000,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Parametric bootstrap for the ensemble's central model.

        Draws one reference spatial sample from the raster for the
        coordinate geometry, builds the covariance matrix from the
        central model, then simulates Gaussian random fields and refits.

        Parameters
        ----------
        n_realizations : int
            Number of bootstrap realisations (≥ 1000 recommended).
        max_cov_points : int or None, optional
            Cap on the number of reference points used to build the
            dense covariance matrix (default 3000). The matrix is
            ``n × n`` in memory and Cholesky-factored at ``O(n³)`` cost,
            so the full sampling density used by :meth:`run` (often tens
            of thousands of points) would exhaust RAM — e.g. ``n=20000``
            needs ~3.2 GB *per* matrix and several are held at once.
            When the reference sample exceeds this cap it is randomly
            subsampled (reproducibly, from ``seed``). Set to ``None`` to
            disable the cap and use every reference point (only safe for
            small rasters). Raising it tightens the simulated-field
            geometry at a memory/time cost that grows as ``n²``–``n³``.
        seed : int, optional
            Base seed for reproducibility. Governs the reference sample,
            the simulated fields, *and* the multi-start optimiser guesses,
            so repeated calls return identical results — exact when run
            single-threaded; differences are at the level of parallel
            floating-point reduction noise otherwise (most visible in
            weakly-identified parameters such as a Matérn ``nu`` at its
            bound).
        verbose : bool
            Print per-realisation progress.

        Returns
        -------
        param_samples : ndarray, shape (n_succeeded, n_params)

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called yet.

        Notes
        -----
        Produces the tightest intervals (lower bound on uncertainty)
        because simulated fields perfectly obey the central model.
        Compare with :meth:`bootstrap_parameters` (upper bound) or
        :meth:`generalized_bootstrap_parameters` (intermediate).
        """
        if self.central_model_name is None:
            raise RuntimeError("No central model. Call run() first.")

        # ── resolve stored parameters ───────────────────────────────────
        def _resolve(val, attr_name):
            if val is not None:
                return val
            stored = getattr(self, attr_name, None)
            if stored is None:
                raise ValueError(
                    f"'{attr_name.replace('_stored_', '')}' not provided "
                    f"and no stored value from run()."
                )
            return stored

        area_side = _resolve(None, "_stored_area_side")
        samples_per_area = _resolve(None, "_stored_samples_per_area")
        max_samples = _resolve(None, "_stored_max_samples")
        bin_width = _resolve(None, "_stored_bin_width")
        max_lag_multiplier = _resolve(None, "_stored_max_lag_multiplier")
        estimator = _resolve(None, "_stored_estimator")

        # ── find central model template ─────────────────────────────────
        model_template = None
        for sv_obj in self.variograms:
            if (
                sv_obj.best_model is not None
                and sv_obj.best_model["description"] == self.central_model_name
            ):
                model_template = sv_obj.best_model["model"]
                break

        if model_template is None:
            raise RuntimeError(
                f"Could not find a fitted model matching the central "
                f"model '{self.central_model_name}' in any realisation."
            )

        # ── draw one reference spatial sample ───────────────────────────
        ss = np.random.SeedSequence(seed)
        ref_seed, *boot_child_seeds_raw = ss.spawn(n_realizations + 1)
        ref_seed_int = int(ref_seed.generate_state(1)[0])

        ref_sv = SingleVariogram(self.rdh)
        ref_sv.compute_empirical_variogram(
            area_side=area_side,
            samples_per_area=samples_per_area,
            max_samples=max_samples,
            bin_width=bin_width,
            max_lag_multiplier=max_lag_multiplier,
            seed=ref_seed_int,
            estimator=estimator,
            return_sample=True,
        )

        coords = ref_sv.sample_coords
        n = len(ref_sv.sample_values)

        # ── cap covariance dimension to bound memory / Cholesky cost ────
        # The dense covariance below is n×n in RAM and factored at O(n³).
        # run()'s empirical variograms tolerate huge n (numba binning is
        # O(n_bins) in memory), but here the full sample would OOM the
        # kernel, so subsample the reference points when they exceed the
        # cap. Reproducible from the reference seed.
        if max_cov_points is not None and n > max_cov_points:
            sub_rng = np.random.default_rng(ref_seed_int)
            sub_idx = sub_rng.choice(n, size=max_cov_points, replace=False)
            coords = coords[sub_idx]
            if verbose:
                print(
                    f"  Subsampled reference points {n} -> {max_cov_points} "
                    f"for the {max_cov_points}x{max_cov_points} covariance "
                    f"matrix (set max_cov_points to adjust)."
                )
            n = max_cov_points

        xs = self.rdh.rioxarray_obj.x.values
        ys = self.rdh.rioxarray_obj.y.values
        x_ext = float(np.max(xs) - np.min(xs))
        y_ext = float(np.max(ys) - np.min(ys))
        max_lag = float(
            np.sqrt(x_ext ** 2 + y_ext ** 2) * max_lag_multiplier
        )

        # ── build covariance matrix from central model ──────────────────
        total_sill = model_template.get_total_sill()
        if total_sill is None or total_sill <= 0:
            raise ValueError(
                "Central model must be stationary with positive sill."
            )

        # Build C in place and free intermediates promptly to keep peak
        # memory near a single n×n array rather than several.
        D = cdist(coords, coords)
        gamma_mat = model_template(D)
        del D
        C = total_sill - gamma_mat
        del gamma_mat
        # Diagonal = covariance at lag 0 (total_sill) plus a tiny jitter
        # for numerical positive-definiteness; avoids allocating np.eye(n).
        jitter = 1e-10 * total_sill
        np.fill_diagonal(C, total_sill + jitter)

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, jitter)
            C_repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(C_repaired)
            warnings.warn(
                "Covariance matrix was not positive-definite; eigenvalues "
                "were clipped. Bootstrap intervals may be approximate.",
                UserWarning,
            )
        del C

        # ── simulation loop ─────────────────────────────────────────────
        collected = []
        n_failed = 0

        for i in range(n_realizations):
            rng = np.random.default_rng(boot_child_seeds_raw[i])
            try:
                epsilon = rng.standard_normal(n)
                z_sim = L @ epsilon

                (n_bins, bin_counts, bssd, bssad,
                binned_sum_dist, _max_dist) = (
                    _bin_distances_and_squared_differences(
                        coords, z_sim, bin_width, max_lag,
                    )
                )

                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_lags = np.where(
                        bin_counts > 0,
                        binned_sum_dist / bin_counts,
                        np.nan,
                    )

                if estimator == "cressie_hawkins":
                    gamma = SingleVariogram._compute_cressie_hawkins(
                        bin_counts, bssad, SingleVariogram.MIN_PAIRS
                    )
                else:
                    gamma = SingleVariogram._compute_matheron(
                        bin_counts, bssd, SingleVariogram.MIN_PAIRS
                    )

                valid = ~np.isnan(gamma)
                n_kept = int(np.sum(valid))
                if n_kept < 3:
                    n_failed += 1
                    if verbose:
                        print(
                            f"  [{i+1}/{n_realizations}] SKIP: "
                            f"too few bins ({n_kept})"
                        )
                    continue

                boot_lags = mean_lags[valid]
                boot_gamma = gamma[valid]
                boot_counts = bin_counts[valid].astype(float)

                gamma_sq = np.maximum(
                    np.square(boot_gamma), np.finfo(float).eps
                )
                weights = boot_counts / gamma_sq

                model = copy.deepcopy(model_template)
                result = ref_sv._fit_single_composite_model(
                    model, boot_lags, boot_gamma, None, weights, rng=rng,
                )

                if result is not None:
                    collected.append(result["params"])
                    if verbose:
                        print(f"  [{i+1}/{n_realizations}] OK")
                else:
                    n_failed += 1
                    if verbose:
                        print(f"  [{i+1}/{n_realizations}] FIT FAILED")

            except Exception as e:
                n_failed += 1
                if verbose:
                    print(f"  [{i+1}/{n_realizations}] ERROR: {e}")
                continue

        if not collected:
            raise RuntimeError(
                f"All {n_realizations} parametric bootstrap "
                f"realisations failed."
            )

        param_samples = np.array(collected)

        pct_keys = [5, 16, 50, 84, 95]
        param_names = list(model_template.param_names)
        param_percentiles = {}
        for j, name in enumerate(param_names):
            vals = param_samples[:, j]
            param_percentiles[name] = {
                f"p{p}": float(np.percentile(vals, p)) for p in pct_keys
            }

        self.bootstrap_param_samples = param_samples
        self.bootstrap_param_percentiles = param_percentiles

        if verbose or n_failed > 0:
            print(
                f"Parametric bootstrap complete: "
                f"{len(collected)}/{n_realizations} succeeded, "
                f"{n_failed} failed."
            )

        return param_samples
    
    # FittedVariogramModel bridge 

    @property
    def fitted_model(self) -> "FittedVariogramModel":
        """Build a :class:`FittedVariogramModel` from the central model.

        This bridges the ``GridVariogram`` ensemble workflow to the
        :class:`RegionalUncertaintyEstimator`, which expects a
        ``FittedVariogramModel`` with ``param_samples`` for bootstrap
        uncertainty propagation.

        The model is constructed from the central (modal) model
        evaluated at its median parameters.  If
        :meth:`bootstrap_parameters` has been called, the bootstrap
        parameter samples are attached.

        Returns
        -------
        FittedVariogramModel

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called with ``fit_model=True``.
        """
        if self.central_model_name is None:
            raise RuntimeError(
                "No central model. Call run() with fit_model=True first."
            )

        # find a reference fitted-model dict matching the central name
        ref_fm = None
        for sv in self.variograms:
            if sv.best_model is not None and sv.best_model["description"] == self.central_model_name:
                ref_fm = sv.best_model
                break

        if ref_fm is None:
            raise RuntimeError(
                f"Could not find a fitted model matching "
                f"'{self.central_model_name}'."
            )

        model = copy.deepcopy(ref_fm["model"])

        # set median parameters from central_params
        median_params = ref_fm["params"].copy()
        if self.central_params is not None:
            param_idx = 0
            for i, spec in enumerate(model._components):
                for j, pname in enumerate(spec.param_names):
                    key = f"{model.component_names[i]}_{pname}"
                    if key in self.central_params:
                        median_params[param_idx] = self.central_params[key]["median"]
                    param_idx += 1
            if model.include_nugget and "nugget" in self.central_params:
                median_params[-1] = self.central_params["nugget"]["median"]

        model.set_params(median_params)

        # attach bootstrap samples if available; otherwise fall back to the
        # per-realisation central-model fits collected by run(), so the
        # ensemble's realization spread propagates into min/max bounds
        param_samples = getattr(self, "bootstrap_param_samples", None)
        if param_samples is None or len(param_samples) == 0:
            param_samples = getattr(self, "ensemble_param_samples", None)

        return FittedVariogramModel(
            composite_model=model,
            params=median_params,
            param_cov=ref_fm["param_cov"],
            rss=ref_fm["rss"],
            aic=ref_fm["aic"],
            bic=ref_fm["bic"],
            param_samples=param_samples,
            model_ensemble=getattr(self, "_realization_model_ensemble", None),
            warnings=ref_fm.get("warnings", []),
            msspe=ref_fm.get("msspe"),
            msspe_std=ref_fm.get("msspe_std"),
            msspe_n_runs=ref_fm.get("msspe_n_runs", 0),
        )

    def summary(self) -> str:
        """Human-readable summary of ensemble results.

        Displays the full aggregated table of all candidate model
        structures across all realisations, followed by the central
        model and its parameter statistics.
        """
        lines = ["=" * 72, "GRID VARIOGRAM ENSEMBLE RESULTS", "=" * 72]
        n_ok = len(self.variograms)
        lines.append(
            f"Realisations: {self.n_realizations} total, "
            f"{n_ok} successful, {self.n_failed} failed"
        )
        lines.append("")

        # ── aggregated model table ──
        if self.all_criteria is not None and len(self.all_criteria) > 0:
            lines.append("MODEL STRUCTURE SUMMARY (across all realisations)")
            lines.append("-" * 110)
            header = (
                f"  {'Model':<40s}  {'Best':>4s}  {'Frac':>6s}  "
                f"{'Fitted':>6s}  "
                f"{'AICc_mean':>10s}  {'AICc_std':>8s}  "
                f"{'BIC_mean':>10s}  "
                f"{'MSSPE_mean':>10s}"
            )
            lines.append(header)
            lines.append("-" * 110)
            for _, row in self.all_criteria.iterrows():
                aicc_str = (
                    f"{row['AICc_mean']:10.1f}"
                    if row.get("AICc_mean") is not None
                    and np.isfinite(row["AICc_mean"])
                    else "       N/A"
                )
                aicc_std_str = (
                    f"{row['AICc_std']:8.1f}"
                    if row.get("AICc_std") is not None
                    and np.isfinite(row["AICc_std"])
                    else "     N/A"
                )
                bic_str = (
                    f"{row['BIC_mean']:10.1f}"
                    if np.isfinite(row["BIC_mean"])
                    else "       N/A"
                )
                msspe_str = (
                    f"{row['MSSPE_mean']:10.4f}"
                    if row["MSSPE_mean"] is not None and np.isfinite(row["MSSPE_mean"])
                    else "       N/A"
                )
                lines.append(
                    f"  {row['model']:<40s}  {row['best_count']:4d}  "
                    f"{row['best_fraction']:5.1%}  "
                    f"{row['fitted_count']:6d}  "
                    f"{aicc_str}  {aicc_std_str}  "
                    f"{bic_str}  "
                    f"{msspe_str}"
                )
            lines.append("")

        # ── central model + parameter stats ──
        if self.central_model_name is not None:
            cnt = self.model_counts.get(self.central_model_name, 0)
            lines.append(
                f"CENTRAL MODEL: {self.central_model_name}  "
                f"(selected best in {cnt}/{n_ok} realisations)"
            )

        if self.central_params:
            lines.append(
                "PARAMETER SUMMARY (median [16th, 84th])"
            )
            lines.append("-" * 60)
            for key, stats in self.central_params.items():
                lines.append(
                    f"  {key:<25s}  {stats['median']:10.4f}  "
                    f"[{stats['p16']:.4f}, {stats['p84']:.4f}]  "
                    f"(n={stats['count']})"
                )

        lines.append("=" * 72)
        return "\n".join(lines)

    #  plotting 

    def plot_variogram(
        self,
        include_central_model: bool = True,
        include_all_models: bool = False,
        include_bootstrap: bool = False,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """Plot ensemble variogram results.

        Parameters
        ----------
        include_central_model : bool
            Overlay the modal model evaluated at median parameters.
        include_all_models : bool
            Overlay every realisation's fitted model (thin lines).
        include_bootstrap : bool
            If True and :meth:`bootstrap_parameters` has been called,
            overlay the bootstrap median model (dashed) and shaded
            1-sigma (16th–84th pctl) and 2-sigma (5th–95th pctl)
            envelopes.
        figsize : tuple
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self.variograms:
            raise RuntimeError("No variograms. Call run() first.")

        if (include_central_model or include_all_models):
            has_fits = any(
                hasattr(sv, "fitted_models") and sv.fitted_models
                for sv in self.variograms
            )
            if not has_fits:
                raise RuntimeError(
                    "Model fitting required. "
                    "Re-run with fit_model=True."
                )

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        fig, axs = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 3]},
            figsize=figsize, sharex=True,
        )

        # stack empirical variograms across realisations
        ref_lags = self.variograms[0].lags
        n_lags = len(ref_lags)
        all_emp = np.full((len(self.variograms), n_lags), np.nan)
        all_counts_arr = np.full((len(self.variograms), n_lags), np.nan)

        for i, sv in enumerate(self.variograms):
            n = min(len(sv.variogram), n_lags)
            all_emp[i, :n] = sv.variogram[:n]
            all_counts_arr[i, :n] = sv.pair_counts[:n]

        with np.errstate(all="ignore"):
            median_emp = np.nanmedian(all_emp, axis=0)
            p16_emp = np.nanpercentile(all_emp, 16, axis=0)
            p84_emp = np.nanpercentile(all_emp, 84, axis=0)
            p2_5_emp = np.nanpercentile(all_emp, 2.5, axis=0)
            p97_5_emp = np.nanpercentile(all_emp, 97.5, axis=0)
            median_counts = np.nanmedian(all_counts_arr, axis=0)

        lags = ref_lags
        bar_width = (
            (lags[1] - lags[0]) * 0.9
            if len(lags) > 1
            else (lags[0] if len(lags) else 1.0) * 0.9
        )

        # ── top panel: pair counts ──
        valid_c = np.isfinite(median_counts) & (median_counts > 0)
        axs[0].bar(
            lags[valid_c], median_counts[valid_c],
            width=bar_width, color="orange", alpha=0.5,
        )
        axs[0].set_ylabel("Median pair count")
        axs[0].tick_params(labelbottom=False)

        # ── bottom panel ──
        ax = axs[1]

        # empirical variogram with error bars
        valid_emp = np.isfinite(median_emp)
        err_1sig_lo = median_emp - p16_emp
        err_1sig_hi = p84_emp - median_emp

        ax.errorbar(
            lags[valid_emp], median_emp[valid_emp],
            yerr=[err_1sig_lo[valid_emp], err_1sig_hi[valid_emp]],
            fmt="o", color="steelblue", ecolor="steelblue",
            elinewidth=1.8, capsize=4, capthick=1.8,
            markersize=5, markerfacecolor="steelblue",
            markeredgecolor="navy", markeredgewidth=0.5,
            zorder=4, label="Median empirical",
        )

        # 2sigma spread (lighter)
        ax.fill_between(
            lags, p2_5_emp, p97_5_emp,
            color="steelblue", alpha=0.1, zorder=1,
        )

        # ── all individual model curves (thin, transparent) ──
        if include_all_models:
            for sv in self.variograms:
                if sv.best_model is not None:
                    model = sv.best_model["model"]
                    model.set_params(sv.best_model["params"])
                    lag_fine = np.linspace(0, float(lags[-1]) * 1.05, 200)
                    try:
                        ax.plot(
                            lag_fine, model(lag_fine),
                            color="gray", alpha=0.2, linewidth=0.8, zorder=2,
                        )
                    except Exception:
                        pass

        # ── central (modal) model curve ──
        if include_central_model and self.central_model_name is not None:
            # find a reference fitted model dict that matches the central name
            ref_fm = None
            for sv in self.variograms:
                if not hasattr(sv, "fitted_models") or sv.fitted_models is None:
                    continue
                for fm in sv.fitted_models:
                    if fm["description"] == self.central_model_name:
                        ref_fm = fm
                        break
                if ref_fm is not None:
                    break

            if ref_fm is not None and self.central_params is not None:
                model = ref_fm["model"]
                # set median parameters
                median_params = ref_fm["params"].copy()
                # overwrite with central values where available
                param_idx = 0
                for i, spec in enumerate(model._components):
                    for j, pname in enumerate(spec.param_names):
                        key = f"{model.component_names[i]}_{pname}"
                        if key in self.central_params:
                            median_params[param_idx] = self.central_params[key]["median"]
                        param_idx += 1
                if model.include_nugget and "nugget" in self.central_params:
                    median_params[-1] = self.central_params["nugget"]["median"]

                model.set_params(median_params)
                lag_fine = np.linspace(0, float(lags[-1]) * 1.05, 300)
                ax.plot(
                    lag_fine, model(lag_fine),
                    color="darkred", linewidth=2.2, zorder=6,
                    label=f"Central model ({self.central_model_name})",
                )

                # range / nugget annotations
                # prefer bootstrap percentiles when available; fall back
                # to ensemble central_params
                _use_boot_pctl = (
                    include_bootstrap
                    and hasattr(self, "bootstrap_param_percentiles")
                    and self.bootstrap_param_percentiles is not None
                )
                annot_source = (
                    self.bootstrap_param_percentiles
                    if _use_boot_pctl
                    else self.central_params
                )
                src_label = "bootstrap" if _use_boot_pctl else "ensemble"

                range_colors = ["red", "green", "lightblue"]
                r_idx = 0
                for key, stats in annot_source.items():
                    # bootstrap_param_percentiles uses p5/p16/p50/p84/p95;
                    # central_params uses median/p16/p84/p2_5/p97_5
                    median_val = stats.get("p50", stats.get("median"))
                    lo_1sig = stats.get("p16")
                    hi_1sig = stats.get("p84")
                    lo_2sig = stats.get("p5", stats.get("p2_5"))
                    hi_2sig = stats.get("p95", stats.get("p97_5"))

                    if median_val is None:
                        continue

                    if "range" in key:
                        c = range_colors[r_idx % len(range_colors)]
                        ax.axvline(
                            median_val, color=c, linewidth=1.8,
                            linestyle="-", zorder=5,
                            label=f'{key}: {median_val:.0f}',
                        )
                        if lo_1sig is not None and hi_1sig is not None:
                            ax.axvspan(
                                lo_1sig, hi_1sig,
                                color=c, alpha=0.20, zorder=1,
                            )
                        if lo_2sig is not None and hi_2sig is not None:
                            ax.axvspan(
                                lo_2sig, hi_2sig,
                                color=c, alpha=0.10, zorder=1,
                            )
                        r_idx += 1
                    elif key == "nugget":
                        ax.axhline(
                            median_val, color="orange", linewidth=1.8,
                            linestyle="-", zorder=5,
                            label=f'nugget: {median_val:.4g}',
                        )
                        if lo_1sig is not None and hi_1sig is not None:
                            ax.axhspan(
                                lo_1sig, hi_1sig,
                                color="orange", alpha=0.20, zorder=1,
                            )
                        if lo_2sig is not None and hi_2sig is not None:
                            ax.axhspan(
                                lo_2sig, hi_2sig,
                                color="orange", alpha=0.10, zorder=1,
                            )

        # ── overlay bootstrap parameter envelopes ──
        has_bootstrap = (
            include_bootstrap
            and hasattr(self, "bootstrap_param_samples")
            and self.bootstrap_param_samples is not None
        )
        if has_bootstrap:
            # find a model template matching the central model
            boot_model_template = None
            for sv in self.variograms:
                if (
                    sv.best_model is not None
                    and sv.best_model["description"] == self.central_model_name
                ):
                    boot_model_template = sv.best_model["model"]
                    break

            if boot_model_template is not None:
                samples = self.bootstrap_param_samples
                lag_fine_boot = np.linspace(0, float(lags[-1]) * 1.05, 300)

                # evaluate model at every bootstrap parameter set
                gamma_ensemble = np.empty((len(samples), len(lag_fine_boot)))
                for k, params_k in enumerate(samples):
                    m = copy.deepcopy(boot_model_template)
                    m.set_params(params_k)
                    gamma_ensemble[k, :] = m(lag_fine_boot)

                # lag-wise percentiles
                bp5  = np.percentile(gamma_ensemble, 5, axis=0)
                bp16 = np.percentile(gamma_ensemble, 16, axis=0)
                bp50 = np.percentile(gamma_ensemble, 50, axis=0)
                bp84 = np.percentile(gamma_ensemble, 84, axis=0)
                bp95 = np.percentile(gamma_ensemble, 95, axis=0)

                # 2-sigma band (5th–95th)
                ax.fill_between(
                    lag_fine_boot, bp5, bp95,
                    color="darkred", alpha=0.15, zorder=3,
                    label="Bootstrap 90% (5th–95th)",
                )
                # 1-sigma band (16th–84th)
                ax.fill_between(
                    lag_fine_boot, bp16, bp84,
                    color="darkred", alpha=0.25, zorder=3,
                    label="Bootstrap 68% (16th–84th)",
                )
                # median curve
                ax.plot(
                    lag_fine_boot, bp50,
                    color="darkred", linewidth=1.5, linestyle="--", zorder=7,
                    label="Bootstrap median",
                )

        ax.set_xlabel("Lag Distance (m)")
        ax.set_ylabel("Semivariance")
        ax.set_xlim(0, float(lags[-1]) * 1.05)
        ax.set_ylim(bottom=0)

        # legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="lower right", fontsize=8,
                  framealpha=0.9, edgecolor="gray")

        # title
        n_ok = len(self.variograms)
        title_parts = [f"Grid variogram (n={n_ok} realisations)"]
        if self.central_model_name:
            cnt = self.model_counts.get(self.central_model_name, 0)
            title_parts.append(
                f"Modal: {self.central_model_name} ({cnt}/{n_ok})"
            )
        ax.set_title("  |  ".join(title_parts), fontsize=9)

        plt.tight_layout()
        #return fig





@dataclass
class FittedVariogramModel:
    """Minimal stub for backward compatibility with uncertainty.py."""
    composite_model: CompositeVariogramModel
    params: np.ndarray
    param_cov: np.ndarray
    rss: float
    aic: float
    bic: float
    param_samples: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)
    msspe: Optional[float] = None
    msspe_std: Optional[float] = None
    msspe_n_runs: int = 0
    loocv_result: Optional[AggregatedLOOCVResult] = None
    # calibrated model-averaged ensemble: list of (model, weight, param_samples)
    model_ensemble: Optional[list] = None

    def predict(self, h: np.ndarray) -> np.ndarray:
        return self.composite_model(h)

    def get_param_percentiles(
        self, percentiles: List[float] = [16, 50, 84]
    ) -> Dict[str, np.ndarray]:
        if self.param_samples is None:
            raise ValueError("No bootstrap samples available.")
        result = {}
        for i, name in enumerate(self.composite_model.param_names):
            result[name] = np.percentile(self.param_samples[:, i], percentiles)
        return result


class VariogramAnalysis(SingleVariogram):
    """Deprecated alias for SingleVariogram (backward compatibility)."""

    # expose private estimators as the legacy public names
    compute_matheron = SingleVariogram._compute_matheron
    compute_cressie_hawkins = SingleVariogram._compute_cressie_hawkins



@dataclass
class EmpiricalVariogram:
    """Minimal stub for backward compatibility."""
    lags: np.ndarray
    median_variogram: np.ndarray
    mean_variogram: np.ndarray
    pair_counts: np.ndarray
    sigma: np.ndarray


class StatisticalAnalysis:
    """Statistical utilities for exploratory plotting and bootstrap uncertainty.

    Backward-compatible re-implementation of the original class.
    """

    def __init__(self, raster_data_handler: RasterDataHandler):
        self.raster_data_handler = raster_data_handler

    def plot_data_stats(self, filtered: bool = True):
        """Plot histogram of raster values with basic statistics annotated.

        Parameters
        ----------
        filtered : bool
            If True, clip to 1st-99th percentiles for visualisation only.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from scipy import stats as sp_stats

        data = self.raster_data_handler.data_array
        if data is None or len(data) == 0:
            raise ValueError(
                "No data available to plot. Call load_raster() first."
            )

        mean = np.mean(data)
        median = np.median(data)
        mode_result = sp_stats.mode(data, nan_policy="omit", keepdims=False)
        mode_vals = np.atleast_1d(mode_result.mode).astype(float)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        p1 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)
        minimum = np.min(data)
        maximum = np.max(data)

        plot_data = data[(data >= p1) & (data <= p99)] if filtered else data

        fig, ax = plt.subplots()
        ax.hist(plot_data, bins=60, density=False, alpha=0.6, color="g")
        ax.axvline(mean, color="r", linestyle="dashed", linewidth=1, label="Mean")
        ax.axvline(median, color="b", linestyle="dashed", linewidth=1, label="Median")
        for i, m in enumerate(mode_vals):
            ax.axvline(
                m, color="purple", linestyle="dashed", linewidth=1,
                label="Mode" if i == 0 else "_nolegend_",
            )

        mode_str = ", ".join([f"{m:.3f}" for m in mode_vals])
        textstr = "\n".join((
            f"Mean: {mean:.3f}",
            f"Median: {median:.3f}",
            f"Mode(s): {mode_str}",
            f"Min: {minimum:.3f}  Max: {maximum:.3f}",
            f"Q1: {q1:.3f}  Q3: {q3:.3f}",
        ))
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props,
        )
        ax.set_xlabel(f"Vertical Difference ({self.raster_data_handler.unit})")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of differencing results with exploratory statistics")
        ax.legend()
        plt.tight_layout()
        #return fig

    def bootstrap_uncertainty_subsample(
        self, n_bootstrap: int = 1000, subsample_proportion: float = 0.1
    ) -> float:
        """Estimate uncertainty of the median via bootstrap on random subsamples.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap resamples.
        subsample_proportion : float
            Fraction of data per resample.

        Returns
        -------
        float
            Standard deviation of bootstrap medians.
        """
        data = self.raster_data_handler.data_array
        if data is None or len(data) == 0:
            raise ValueError(
                "No data available for bootstrap. Call load_raster() first."
            )

        subsample_size = max(1, int(round(subsample_proportion * len(data))))
        rng = np.random.default_rng()
        bootstrap_medians = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=subsample_size, replace=True)
            bootstrap_medians[i] = np.median(sample)
        return float(np.std(bootstrap_medians))

