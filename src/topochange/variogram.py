# """variogram analysis and model fitting for differencing uncertainty."""

# from __future__ import annotations

# import math
# import warnings
# from typing import Sequence, Optional, Dict, Any, Callable
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import stats
# from scipy.optimize import curve_fit
# import rasterio
# import rioxarray as rio
# from numba import njit, prange
# from pathlib import Path
# from shapely.geometry import Polygon, MultiPolygon, Point, box, shape
# from shapely.prepared import prep
# import geopandas as gpd
# from rasterio.features import shapes
# from shapely.ops import unary_union
# from .variogram_models import MODEL_REGISTRY, VariogramModelRegistry
# from .composite_variogram import CompositeVariogramModel
# from itertools import combinations_with_replacement
# from dataclasses import dataclass, field
# from typing import Callable, Dict, List, Optional, Tuple


# @dataclass
# class FittedVariogramModel:
#     """Container for a fitted variogram model with diagnostics.

#     Attributes
#     ----------
#     composite_model : CompositeVariogramModel
#         The fitted composite model.
#     params : ndarray
#         Optimal parameter values.
#     param_cov : ndarray
#         Parameter covariance matrix from curve_fit.
#     rss : float
#         Residual sum of squares.
#     aic : float
#         Akaike Information Criterion.
#     bic : float
#         Bayesian Information Criterion.
#     param_samples : ndarray or None
#         Bootstrap parameter samples (n_boot × n_params).
#     warnings : list of str
#         Diagnostic warnings (e.g. range exceeds half max lag).
#     msspe : float or None
#         Mean Standardized Squared Prediction Error from kriging
#         leave-one-out cross-validation, averaged over multiple
#         independent random subsamples.  Target value is 1.0;
#         values >> 1 indicate the model underestimates spatial
#         uncertainty; values << 1 indicate overestimation.
#     msspe_std : float or None
#         Standard deviation of MSSPE across repeated runs.
#         A large value relative to ``|msspe − 1|`` suggests the
#         ranking is sensitive to the particular subsample drawn.
#     msspe_n_runs : int
#         Number of MSSPE runs that succeeded.
#     loocv_result : AggregatedLOOCVResult or None
#         Aggregated kriging LOOCV diagnostics across all repeated
#         runs, including mean/median/std/nmad for MSSPE, bias,
#         RMSE, and standardized error, plus the individual per-run
#         results.
#     """
#     composite_model: CompositeVariogramModel
#     params: np.ndarray
#     param_cov: np.ndarray
#     rss: float
#     aic: float
#     bic: float
#     param_samples: Optional[np.ndarray] = None
#     warnings: List[str] = field(default_factory=list)
#     msspe: Optional[float] = None
#     msspe_std: Optional[float] = None
#     msspe_n_runs: int = 0
#     loocv_result: Optional['AggregatedLOOCVResult'] = None

#     def predict(self, h: np.ndarray) -> np.ndarray:
#         """Evaluate fitted model at lag distances."""
#         return self.composite_model(h)

#     def check_half_lag(self, max_lag: float) -> None:
#         """Check whether any fitted range exceeds half the maximum lag.

#         Appends a diagnostic warning if so.  The half-lag heuristic
#         (Journel & Huijbregts, 1978) holds that variogram parameters
#         estimated from lags beyond half the domain may be unreliable
#         because few point pairs constrain the fit there.
#         """
#         half_lag = max_lag / 2.0
#         model = self.composite_model
#         for i, spec in enumerate(model._components):
#             if 'range' in spec.param_names:
#                 range_idx = spec.param_names.index('range')
#                 comp_params = model.get_component_params(i)
#                 fitted_range = comp_params[range_idx]
#                 if fitted_range > half_lag:
#                     name = model.component_names[i]
#                     msg = (
#                         f"WARNING: {name} range ({fitted_range:.1f}) exceeds "
#                         f"half the maximum lag ({half_lag:.1f}).  The variogram "
#                         f"is poorly constrained beyond this distance — consider "
#                         f"increasing max_lag or treating this range estimate "
#                         f"with caution."
#                     )
#                     self.warnings.append(msg)

#     def get_param_percentiles(
#         self,
#         percentiles: List[float] = [16, 50, 84]
#     ) -> Dict[str, np.ndarray]:
#         """Get parameter percentiles from bootstrap samples.

#         Returns
#         -------
#         percentiles_dict : dict
#             Keys are parameter names, values are arrays of percentiles.
#         """
#         if self.param_samples is None:
#             raise ValueError("No bootstrap samples available.")

#         result = {}
#         for i, name in enumerate(self.composite_model.param_names):
#             result[name] = np.percentile(self.param_samples[:, i], percentiles)

#         return result


# @dataclass
# class EmpiricalVariogram:
#     """Container for an empirical variogram computed from spatial data.

#     Holds the lag-binned semivariance estimates, pair counts, and
#     spread across multiple independent sampling realizations.
#     Returned by :meth:`VariogramAnalysis.compute_empirical_variogram`.

#     Attributes
#     ----------
#     lags : ndarray
#         Lag-bin centres (same units as input coordinates).
#     median_variogram : ndarray
#         Median semivariance per lag bin across realizations.
#     mean_variogram : ndarray
#         Mean semivariance per lag bin (for WLS fitting input).
#     pair_counts : ndarray
#         Mean number of point pairs per lag bin.
#     sigma : ndarray
#         Per-bin standard deviation across realizations (useful as
#         WLS weight input).
#     p2_5 : ndarray
#         2.5th percentile per bin (lower 2σ bound).
#     p16 : ndarray
#         16th percentile per bin (lower 1σ bound).
#     p84 : ndarray
#         84th percentile per bin (upper 1σ bound).
#     p97_5 : ndarray
#         97.5th percentile per bin (upper 2σ bound).
#     n_runs : int
#         Number of independent realizations averaged.
#     n_bins : int
#         Effective number of lag bins.
#     estimator : str
#         Which estimator was used (``'matheron'`` or
#         ``'cressie_hawkins'``).
#     sample_coords : ndarray or None
#         Coordinates from the last sampling realization (shape (n, 2)).
#     sample_values : ndarray or None
#         Values from the last sampling realization (shape (n,)).
#     all_variograms : ndarray or None
#         Full matrix of per-run semivariance estimates (n_runs × n_lags).
#     all_counts : ndarray or None
#         Full matrix of per-run pair counts.
#     """

#     lags: np.ndarray
#     median_variogram: np.ndarray
#     mean_variogram: np.ndarray
#     pair_counts: np.ndarray
#     sigma: np.ndarray
#     p2_5: np.ndarray
#     p16: np.ndarray
#     p84: np.ndarray
#     p97_5: np.ndarray
#     n_runs: int
#     n_bins: int
#     estimator: str
#     sample_coords: Optional[np.ndarray] = None
#     sample_values: Optional[np.ndarray] = None
#     all_variograms: Optional[np.ndarray] = None
#     all_counts: Optional[np.ndarray] = None

#     def plot(self, ax=None, show_spread: bool = True, show_counts: bool = True):
#         """Plot the empirical variogram with two-level uncertainty shading.

#         Mirrors the style of
#         :meth:`VariogramAnalysis.plot_best_model`:

#         - **Top panel**: bar chart of mean pair counts per lag bin.
#         - **Bottom panel**: median binned semivariance (blue dots)
#           with 1σ (68 %, darker) and 2σ (95 %, lighter) spread
#           envelopes across realizations.

#         Parameters
#         ----------
#         ax : matplotlib.axes.Axes, optional
#             Axes to draw on.  If *None*, creates a new two-panel
#             figure.  When an external axis is supplied, only the
#             variogram panel is drawn (no pair-count panel).
#         show_spread : bool, default True
#             Shade the 1σ and 2σ envelopes across realizations.
#         show_counts : bool, default True
#             Show the pair-count bar chart above the variogram.

#         Returns
#         -------
#         matplotlib.figure.Figure
#         """
#         from matplotlib.patches import Patch
#         from matplotlib.lines import Line2D

#         if ax is not None:
#             fig = ax.get_figure()
#             axs = [None, ax]  # [count_ax, vario_ax]
#             show_counts = False
#         else:
#             fig, axs = plt.subplots(
#                 2, 1,
#                 gridspec_kw={'height_ratios': [1, 3]},
#                 figsize=(10, 8),
#                 sharex=True,
#             )

#         # ── bar width ──
#         if len(self.lags) > 1:
#             bar_width = (self.lags[1] - self.lags[0]) * 0.9
#         else:
#             bar_width = (self.lags[0] if len(self.lags) else 1.0) * 0.9

#         # ── top panel: pair counts ──
#         if show_counts:
#             valid_c = (~np.isnan(self.pair_counts)) & (self.pair_counts > 0)
#             axs[0].bar(
#                 self.lags[valid_c], self.pair_counts[valid_c],
#                 width=bar_width, color='orange', alpha=0.5,
#             )
#             axs[0].set_ylabel('Mean Count')
#             axs[0].tick_params(labelbottom=False)

#         # ── bottom panel: empirical variogram ──
#         ax_v = axs[1] if show_counts else (axs[1] if axs[0] is None else axs[0])

#         # median binned values
#         ax_v.plot(
#             self.lags, self.median_variogram,
#             'o-', color='blue', markersize=4, zorder=5,
#             label='Median variogram',
#         )

#         # spread envelopes
#         if show_spread and self.n_runs > 1:
#             # 2σ (95 %) — light shading
#             ax_v.fill_between(
#                 self.lags, self.p2_5, self.p97_5,
#                 color='blue', alpha=0.1,
#             )
#             # 1σ (68 %) — darker shading
#             ax_v.fill_between(
#                 self.lags, self.p16, self.p84,
#                 color='blue', alpha=0.3,
#             )

#         ax_v.set_xlabel('Lag Distance')
#         ax_v.set_ylabel('Semivariance')
#         ax_v.set_title(
#             f'Empirical variogram ({self.estimator}, '
#             f'{self.n_runs} run{"s" if self.n_runs > 1 else ""})'
#         )

#         # legend
#         legend_elements = [
#             Line2D([0], [0], marker='o', color='blue',
#                    label='Median variogram', linestyle='-', markersize=4),
#         ]
#         if show_spread and self.n_runs > 1:
#             legend_elements += [
#                 Patch(facecolor='blue', alpha=0.3, label='1σ spread (68 %)'),
#                 Patch(facecolor='blue', alpha=0.1, label='2σ spread (95 %)'),
#             ]
#         ax_v.legend(handles=legend_elements, loc='lower right')

#         if show_counts:
#             plt.setp(axs[0].get_xticklabels(), visible=False)
#         plt.tight_layout()
#         return fig


# @dataclass
# class KrigingLOOCVResult:
#     """Results from leave-one-out kriging cross-validation.

#     The primary diagnostic is ``msspe`` (Mean Standardized Squared
#     Prediction Error), which should be ≈ 1.0 for a well-calibrated
#     variogram model.  Think of it like a χ² / df ratio — values near
#     1.0 mean the variogram correctly captures the spatial uncertainty;
#     MSSPE >> 1 means the model underestimates uncertainty; MSSPE << 1
#     means it overestimates.

#     Attributes
#     ----------
#     msspe : float
#         Mean Standardized Squared Prediction Error:
#         (1/n) Σ (z_i − ẑ₋ᵢ)² / σ²₋ᵢ.   Target: ≈ 1.0.
#     mean_error : float
#         Mean prediction error (bias):  (1/n) Σ (z_i − ẑ₋ᵢ).
#         Target: ≈ 0.0.
#     rmse : float
#         Root mean squared prediction error.
#     mean_standardized_error : float
#         Mean of (z_i − ẑ₋ᵢ) / σ₋ᵢ.   Target: ≈ 0.0.
#     n_points : int
#         Number of points used (after subsampling and excluding failures).
#     n_failed : int
#         Number of LOO iterations that failed (singular matrix, etc.).

#     References
#     ----------
#     Cressie, N. (1993). Statistics for Spatial Data, rev. ed., Wiley.
#         Section 5.6 — kriging cross-validation.

#     Webster, R. & Oliver, M.A. (2007). Geostatistics for Environmental
#         Scientists, 2nd ed., Wiley.  Section 8.3.

#     Lark, R.M. (2000). A comparison of some robust estimators of the
#         variogram for use in soil survey.  Eur. J. Soil Sci., 51,
#         137–157.  Uses MSSPE ≈ 1.0 as acceptance criterion.
#     """

#     msspe: float
#     mean_error: float
#     rmse: float
#     mean_standardized_error: float
#     n_points: int
#     n_failed: int


# @dataclass
# class AggregatedLOOCVResult:
#     """Aggregated diagnostics from repeated kriging LOOCV runs.

#     Each field stores summary statistics (mean, median, std, nmad)
#     across ``n_runs`` independent random subsamples.  The spread
#     quantifies how sensitive each diagnostic is to the particular
#     subsample drawn — large spread relative to ``|msspe_mean − 1|``
#     suggests the ranking may not be stable.

#     Attributes
#     ----------
#     n_runs : int
#         Number of successful LOOCV runs.
#     total_points : int
#         Total number of LOOCV predictions across all runs.
#     msspe_mean, msspe_median, msspe_std, msspe_nmad : float
#         Summary statistics of MSSPE across runs.
#     mean_error_mean, mean_error_median, mean_error_std, mean_error_nmad : float
#         Summary statistics of mean prediction error (bias).
#     rmse_mean, rmse_median, rmse_std, rmse_nmad : float
#         Summary statistics of RMSE.
#     mse_mean, mse_median, mse_std, mse_nmad : float
#         Summary statistics of mean standardized error.
#     n_failed_total : int
#         Total LOO iterations that failed across all runs.
#     per_run_results : list of KrigingLOOCVResult
#         Individual run results for detailed inspection.
#     """

#     n_runs: int
#     total_points: int

#     msspe_mean: float
#     msspe_median: float
#     msspe_std: float
#     msspe_nmad: float

#     mean_error_mean: float
#     mean_error_median: float
#     mean_error_std: float
#     mean_error_nmad: float

#     rmse_mean: float
#     rmse_median: float
#     rmse_std: float
#     rmse_nmad: float

#     mse_mean: float
#     mse_median: float
#     mse_std: float
#     mse_nmad: float

#     n_failed_total: int

#     per_run_results: List[KrigingLOOCVResult] = field(default_factory=list)

#     @staticmethod
#     def from_results(results: List[KrigingLOOCVResult]) -> 'AggregatedLOOCVResult':
#         """Compute aggregate statistics from a list of per-run results.

#         Parameters
#         ----------
#         results : list of KrigingLOOCVResult
#             Individual LOOCV results, one per subsample run.

#         Returns
#         -------
#         AggregatedLOOCVResult
#         """
#         def _nmad(arr: np.ndarray) -> float:
#             """Normalized Median Absolute Deviation (robust σ estimate)."""
#             return float(1.4826 * np.median(np.abs(arr - np.median(arr))))

#         def _stats(arr: np.ndarray) -> tuple:
#             return (
#                 float(np.mean(arr)),
#                 float(np.median(arr)),
#                 float(np.std(arr)),
#                 _nmad(arr),
#             )

#         msspes = np.array([r.msspe for r in results])
#         mean_errors = np.array([r.mean_error for r in results])
#         rmses = np.array([r.rmse for r in results])
#         mses = np.array([r.mean_standardized_error for r in results])

#         return AggregatedLOOCVResult(
#             n_runs=len(results),
#             total_points=sum(r.n_points for r in results),
#             msspe_mean=_stats(msspes)[0],
#             msspe_median=_stats(msspes)[1],
#             msspe_std=_stats(msspes)[2],
#             msspe_nmad=_stats(msspes)[3],
#             mean_error_mean=_stats(mean_errors)[0],
#             mean_error_median=_stats(mean_errors)[1],
#             mean_error_std=_stats(mean_errors)[2],
#             mean_error_nmad=_stats(mean_errors)[3],
#             rmse_mean=_stats(rmses)[0],
#             rmse_median=_stats(rmses)[1],
#             rmse_std=_stats(rmses)[2],
#             rmse_nmad=_stats(rmses)[3],
#             mse_mean=_stats(mses)[0],
#             mse_median=_stats(mses)[1],
#             mse_std=_stats(mses)[2],
#             mse_nmad=_stats(mses)[3],
#             n_failed_total=sum(r.n_failed for r in results),
#             per_run_results=list(results),
#         )


# @dataclass
# class EnsembleVariogramResult:
#     """Results from ensemble variogram fitting across independent spatial samples.

#     Each realization draws an independent random sample from the raster,
#     computes an empirical variogram, runs the full model selection pipeline
#     (all candidate model types, WLS fitting, MSSPE-based selection), and
#     records the winning model's structure and parameters.  The ensemble
#     captures both *model selection uncertainty* (does the pipeline always
#     pick the same model family?) and *parameter uncertainty* (how stable
#     are the fitted sills, ranges, nuggets, and Matérn ν?).

#     This is conceptually similar to a parametric bootstrap, but instead
#     of perturbing a single empirical variogram, it re-samples the spatial
#     field — making it a Monte Carlo assessment of the entire pipeline
#     from sampling through model selection.

#     Attributes
#     ----------
#     n_realizations : int
#         Number of independent variogram realizations fitted.
#     n_failed : int
#         Realizations where no model could be fitted.
#     model_counts : dict
#         {model_description: count} — how often each model structure
#         was selected (e.g. ``{'spherical + nugget': 35, 'exponential + matern + nugget': 15}``).
#     model_fractions : dict
#         {model_description: fraction} — selection frequency as proportion.
#     sills : ndarray, shape (n_success, max_n_sills)
#         Per-realization sill values (NaN-padded where models have
#         fewer sill components).
#     ranges : ndarray, shape (n_success, max_n_ranges)
#         Per-realization range values (NaN-padded).
#     nuggets : ndarray, shape (n_success,)
#         Per-realization nugget values (NaN where model has no nugget).
#     nus : ndarray, shape (n_success,)
#         Per-realization Matérn ν values (NaN for non-Matérn models).
#     msspes : ndarray, shape (n_success,)
#         Per-realization MSSPE of the selected model.
#     lags : ndarray
#         Common lag vector (from the first successful realization).
#     variograms : ndarray, shape (n_success, n_lags)
#         Per-realization fitted variogram curves evaluated at ``lags``.
#     empirical_variograms : ndarray, shape (n_success, n_lags)
#         Per-realization empirical variograms (NaN-padded to common length).
#     pair_counts : ndarray, shape (n_success, n_lags)
#         Per-realization pair counts (NaN-padded to common length).
#     per_realization : list of dict
#         Detailed per-realization records including model description,
#         all parameters, component names, MSSPE, AIC, and fitted curve.

#     References
#     ----------
#     Lark, R.M. (2000). A comparison of some robust estimators of the
#     variogram for use in soil survey. *Eur. J. Soil Sci.*, 51, 137–157.

#     Marchetti, Y. et al. (2018). An assessment of model selection
#     uncertainty in spatial prediction. *Environmetrics*, 29(7–8),
#     e2530. doi:10.1002/env.2530
#     """

#     n_realizations: int
#     n_failed: int
#     model_counts: Dict[str, int]
#     model_fractions: Dict[str, float]
#     sills: np.ndarray
#     ranges: np.ndarray
#     nuggets: np.ndarray
#     nus: np.ndarray
#     msspes: np.ndarray
#     lags: np.ndarray
#     variograms: np.ndarray
#     empirical_variograms: np.ndarray
#     pair_counts: np.ndarray
#     per_realization: List[Dict[str, Any]]

#     def summary(self) -> str:
#         """Human-readable summary of ensemble results."""
#         lines = []
#         lines.append("=" * 72)
#         lines.append("ENSEMBLE VARIOGRAM FITTING RESULTS")
#         lines.append("=" * 72)
#         n_ok = self.n_realizations - self.n_failed
#         lines.append(
#             f"Realizations: {self.n_realizations} total, "
#             f"{n_ok} successful, {self.n_failed} failed"
#         )
#         lines.append("")

#         # model selection frequency
#         lines.append("MODEL SELECTION FREQUENCY")
#         lines.append("-" * 50)
#         for desc, frac in sorted(
#             self.model_fractions.items(), key=lambda x: -x[1]
#         ):
#             cnt = self.model_counts[desc]
#             lines.append(f"  {desc:<40s}  {cnt:3d} ({frac:5.1%})")
#         lines.append("")

#         # parameter summaries
#         lines.append("PARAMETER SUMMARY (median [16th, 84th percentile])")
#         lines.append("-" * 50)

#         def _summarize(arr, name):
#             valid = arr[np.isfinite(arr)]
#             if len(valid) == 0:
#                 return
#             med = np.median(valid)
#             p16 = np.percentile(valid, 16)
#             p84 = np.percentile(valid, 84)
#             lines.append(f"  {name:<20s}  {med:10.4f}  [{p16:.4f}, {p84:.4f}]")

#         # sills
#         for j in range(self.sills.shape[1] if self.sills.ndim > 1 else 0):
#             _summarize(self.sills[:, j], f"Sill {j + 1}")

#         # ranges
#         for j in range(self.ranges.shape[1] if self.ranges.ndim > 1 else 0):
#             _summarize(self.ranges[:, j], f"Range {j + 1}")

#         _summarize(self.nuggets, "Nugget")
#         _summarize(self.nus, "Matérn ν")
#         lines.append("")

#         # MSSPE
#         valid_msspe = self.msspes[np.isfinite(self.msspes)]
#         if len(valid_msspe) > 0:
#             lines.append("MSSPE SUMMARY")
#             lines.append("-" * 50)
#             lines.append(
#                 f"  Median: {np.median(valid_msspe):.4f}  "
#                 f"Mean: {np.mean(valid_msspe):.4f}  "
#                 f"Std: {np.std(valid_msspe):.4f}"
#             )
#             lines.append(
#                 f"  [16th, 84th]: [{np.percentile(valid_msspe, 16):.4f}, "
#                 f"{np.percentile(valid_msspe, 84):.4f}]"
#             )
#         lines.append("=" * 72)
#         return "\n".join(lines)

#     def plot(self, figsize=(10, 8), unit: str = ''):
#         """Plot ensemble variogram results.

#         Creates a 2-panel figure:

#         Upper panel
#             Median bin pair counts as a bar chart.

#         Lower panel
#             - **Empirical variogram**: median per bin as points with
#               1σ (solid blue) and 2σ (dotted blue) error bars.
#             - **Ranges**: median as vertical lines (red / green /
#               lightblue for range 1 / 2 / 3), with 1σ and 2σ
#               envelopes as translucent vertical rectangles.
#             - **Nugget**: median as horizontal orange line, with 1σ
#               and 2σ envelopes as translucent horizontal rectangles.
#             - **Fitted model**: dark-red curve evaluated at the
#               median parameters of the most frequently selected
#               model structure (modal model + median params).

#         Parameters
#         ----------
#         figsize : tuple
#             Figure size (width, height) in inches.
#         unit : str
#             Distance unit label for axes (e.g. 'm').

#         Returns
#         -------
#         fig : matplotlib.figure.Figure
#         """
#         from matplotlib.patches import Patch, Rectangle
#         from matplotlib.lines import Line2D

#         lags = self.lags
#         n_ok = self.n_realizations - self.n_failed

#         # ── compute percentiles across realizations ──
#         with np.errstate(all='ignore'):
#             median_emp = np.nanmedian(self.empirical_variograms, axis=0)
#             p16_emp = np.nanpercentile(self.empirical_variograms, 16, axis=0)
#             p84_emp = np.nanpercentile(self.empirical_variograms, 84, axis=0)
#             p2_5_emp = np.nanpercentile(self.empirical_variograms, 2.5, axis=0)
#             p97_5_emp = np.nanpercentile(self.empirical_variograms, 97.5, axis=0)
#             median_counts = np.nanmedian(self.pair_counts, axis=0)

#         valid_emp = np.isfinite(median_emp)

#         # ── range percentiles (per component) ──
#         n_range_cols = self.ranges.shape[1] if self.ranges.ndim > 1 else 0
#         range_stats = []  # list of (median, p16, p84, p2.5, p97.5) per col
#         for j in range(n_range_cols):
#             col = self.ranges[:, j]
#             v = col[np.isfinite(col)]
#             if len(v) > 0:
#                 range_stats.append({
#                     'median': float(np.median(v)),
#                     'p16': float(np.percentile(v, 16)),
#                     'p84': float(np.percentile(v, 84)),
#                     'p2_5': float(np.percentile(v, 2.5)),
#                     'p97_5': float(np.percentile(v, 97.5)),
#                 })

#         # ── nugget percentiles ──
#         nug_valid = self.nuggets[np.isfinite(self.nuggets)]
#         nug_stats = None
#         if len(nug_valid) > 0:
#             nug_stats = {
#                 'median': float(np.median(nug_valid)),
#                 'p16': float(np.percentile(nug_valid, 16)),
#                 'p84': float(np.percentile(nug_valid, 84)),
#                 'p2_5': float(np.percentile(nug_valid, 2.5)),
#                 'p97_5': float(np.percentile(nug_valid, 97.5)),
#             }

#         # ── build median model curve (modal structure + median params) ──
#         modal_desc = max(self.model_counts, key=self.model_counts.get)
#         modal_records = [
#             rec for rec in self.per_realization
#             if rec['model_description'] == modal_desc
#         ]

#         median_model_curve = None
#         if modal_records:
#             # extract component_names and include_nugget from the first
#             ref = modal_records[0]
#             comp_names = ref['component_names']
#             inc_nugget = ref['include_nugget']

#             # stack parameter arrays and take element-wise median
#             all_params = np.array([rec['params'] for rec in modal_records])
#             median_params = np.nanmedian(all_params, axis=0)

#             try:
#                 median_model = CompositeVariogramModel(
#                     comp_names, include_nugget=inc_nugget,
#                 )
#                 median_model.set_params(median_params)
#                 # evaluate at a dense set of lags for a smooth curve
#                 lag_fine = np.linspace(0, float(lags[-1]) * 1.05, 300)
#                 median_model_curve = (lag_fine, median_model(lag_fine))
#             except Exception:
#                 median_model_curve = None

#         # ── create figure ──
#         fig, axs = plt.subplots(
#             2, 1, figsize=figsize, sharex=True,
#             gridspec_kw={'height_ratios': [1, 3]},
#         )

#         # ── upper panel: pair counts ──
#         ax_top = axs[0]
#         valid_c = np.isfinite(median_counts) & (median_counts > 0)
#         if len(lags) > 1:
#             bar_w = (lags[1] - lags[0]) * 0.9
#         else:
#             bar_w = (lags[0] if len(lags) else 1.0) * 0.9
#         ax_top.bar(
#             lags[valid_c], median_counts[valid_c],
#             width=bar_w, color='orange', alpha=0.5,
#         )
#         ax_top.set_ylabel('Median pair count')
#         ax_top.tick_params(labelbottom=False)

#         # ── lower panel: main variogram plot ──
#         ax = axs[1]

#         # — empirical variogram: median points + error bars —
#         err_1sig_lo = median_emp - p16_emp
#         err_1sig_hi = p84_emp - median_emp
#         err_2sig_lo = median_emp - p2_5_emp
#         err_2sig_hi = p97_5_emp - median_emp

#         # 2σ error bars (dotted blue, plotted first so 1σ overlays)
#         ax.errorbar(
#             lags[valid_emp], median_emp[valid_emp],
#             yerr=[err_2sig_lo[valid_emp], err_2sig_hi[valid_emp]],
#             fmt='none', ecolor='steelblue', elinewidth=1.0,
#             capsize=3, capthick=1.0,
#             linestyle=':', zorder=3,
#         )
#         # draw dotted caps manually by setting the caplines linestyle
#         # (errorbar capstyle can't be dotted, so use thin alpha)

#         # 1σ error bars (solid blue)
#         _, caps_1, bars_1 = ax.errorbar(
#             lags[valid_emp], median_emp[valid_emp],
#             yerr=[err_1sig_lo[valid_emp], err_1sig_hi[valid_emp]],
#             fmt='o', color='steelblue', ecolor='steelblue',
#             elinewidth=1.8, capsize=4, capthick=1.8,
#             markersize=5, markerfacecolor='steelblue',
#             markeredgecolor='navy', markeredgewidth=0.5,
#             zorder=4, label='Median empirical',
#         )

#         # — range envelopes (vertical rectangles) —
#         range_colors = ['red', 'green', 'lightblue']

#         for i, rs in enumerate(range_stats):
#             c = range_colors[i % len(range_colors)]
#             # 2σ envelope (very translucent)
#             ax.axvspan(
#                 rs['p2_5'], rs['p97_5'],
#                 color=c, alpha=0.08, zorder=1,
#             )
#             # 1σ envelope (translucent)
#             ax.axvspan(
#                 rs['p16'], rs['p84'],
#                 color=c, alpha=0.20, zorder=1,
#             )
#             # median range (solid vertical line)
#             ax.axvline(
#                 rs['median'], color=c, linewidth=1.8,
#                 linestyle='-', zorder=5,
#                 label=f'Range {i + 1}: {rs["median"]:.0f}',
#             )

#         # — nugget envelope (horizontal rectangles) —
#         if nug_stats is not None:
#             # 2σ envelope
#             ax.axhspan(
#                 nug_stats['p2_5'], nug_stats['p97_5'],
#                 color='orange', alpha=0.08, zorder=1,
#             )
#             # 1σ envelope
#             ax.axhspan(
#                 nug_stats['p16'], nug_stats['p84'],
#                 color='orange', alpha=0.20, zorder=1,
#             )
#             # median nugget line
#             ax.axhline(
#                 nug_stats['median'], color='orange', linewidth=1.8,
#                 linestyle='-', zorder=5,
#                 label=f'Nugget: {nug_stats["median"]:.4g}',
#             )

#         # — median model curve (dark red) —
#         if median_model_curve is not None:
#             lag_fine, gamma_fine = median_model_curve
#             ax.plot(
#                 lag_fine, gamma_fine,
#                 color='darkred', linewidth=2.2, zorder=6,
#                 label=f'Median model ({modal_desc})',
#             )

#         # ── axis labels, legend, title ──
#         xlabel = 'Lag distance'
#         if unit:
#             xlabel += f' ({unit})'
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel('Semivariance')
#         ax.set_xlim(0, float(lags[-1]) * 1.05)
#         ax.set_ylim(bottom=0)

#         # build a compact legend
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(
#             handles, labels, loc='lower right', fontsize=8,
#             framealpha=0.9, edgecolor='gray',
#         )

#         # informative title
#         msspe_valid = self.msspes[np.isfinite(self.msspes)]
#         title_parts = [
#             f'Ensemble variogram  (n={n_ok} realizations)',
#         ]
#         if len(msspe_valid) > 0:
#             title_parts.append(
#                 f'MSSPE={np.median(msspe_valid):.3f}'
#             )
#         title_parts.append(
#             f'Modal: {modal_desc} '
#             f'({self.model_counts[modal_desc]}/{n_ok})'
#         )
#         ax.set_title('   |   '.join(title_parts), fontsize=9)

#         plt.tight_layout()
#         return fig


# class VariogramModelSelector:
#     """Select and fit optimal variogram model from candidates.

#     This class implements:
#     - Automatic generation of candidate nested models
#     - Fitting via weighted least squares
#     - Model selection via AIC, BIC, or MSSPE
#     - Bootstrap parameter uncertainty
    
#     Parameters
#     ----------
#     lags : ndarray
#         Lag distances from empirical variogram.
#     empirical_variogram : ndarray
#         Empirical semivariance values.
#     pair_counts : ndarray, optional
#         Number of point pairs per lag bin.  Required for ``'cressie'``
#         and ``'pair_count'`` weighting schemes, and for ``min_pairs``
#         filtering.
#     sigma : ndarray, optional
#         Per-bin standard deviations for likelihood-based AIC/BIC.
#     weighting : {'cressie', 'pair_count', 'uniform'}, default ``'cressie'``
#         WLS weighting scheme.  ``'cressie'`` computes w_i = N(h)/γ̂(h)²
#         (Cressie, 1985).  ``'pair_count'`` uses raw pair counts.
#         ``'uniform'`` sets all weights to 1.
#     min_pairs : int or None, default 30
#         Minimum number of point pairs required for a lag bin to be
#         included in model fitting.  Bins with fewer pairs receive
#         zero weight and are effectively excluded from the WLS fit.
#         Set to ``None`` or ``0`` to disable filtering.  Requires
#         ``pair_counts`` to be supplied; ignored otherwise.

#         The literature recommends thresholds of 30–50
#         (Cressie, 1985; Oliver & Webster, 2014).

#     Examples
#     --------
#     >>> selector = VariogramModelSelector(lags, gamma, pair_counts=counts)
#     >>> selector.fit_all_candidates(max_components=2, include_nugget=True)
#     >>> best = selector.select_best(criterion='aic')
#     >>> print(best.composite_model.description())
    
#     References
#     ----------
#     Cressie, N. (1985). Fitting variogram models by weighted least squares.
#     J. Int. Assoc. Math. Geol., 17(5), 563–586.

#     Oliver, M.A. & Webster, R. (2014). A tutorial guide to geostatistics.
#     Catena, 113, 56–69.

#     Webster, R. & McBratney, A.B. (1989). On the Akaike Information Criterion
#     for choosing models for variograms of soil properties. Eur. J. Soil Sci.
#     """
    
#     # models to include in candidate generation
#     BOUNDED_MODELS = ['spherical', 'exponential', 'matern']
#     UNBOUNDED_MODELS = ['power', 'linear']
    
#     # supported weighting schemes for WLS fitting
#     WEIGHTING_SCHEMES = ('cressie', 'pair_count', 'uniform')

#     def __init__(
#         self,
#         lags: np.ndarray,
#         empirical_variogram: np.ndarray,
#         pair_counts: Optional[np.ndarray] = None,
#         sigma: Optional[np.ndarray] = None,
#         weighting: str = 'cressie',
#         min_pairs: Optional[int] = 30,
#     ):
#         self.lags = np.asarray(lags, dtype=float)
#         self.empirical_variogram = np.asarray(empirical_variogram, dtype=float)

#         if pair_counts is not None:
#             self.pair_counts = np.asarray(pair_counts, dtype=float)
#         else:
#             self.pair_counts = None

#         if weighting not in self.WEIGHTING_SCHEMES:
#             raise ValueError(
#                 f"Unknown weighting scheme '{weighting}'. "
#                 f"Choose from {self.WEIGHTING_SCHEMES}."
#             )
#         self.weighting = weighting
#         self.min_pairs = min_pairs if min_pairs else 0

#         # compute WLS weights
#         self.weights = self._compute_weights(
#             self.empirical_variogram, self.pair_counts, weighting
#         )

#         # standard deviation per bin (for likelihood-based criteria)
        
#         if sigma is not None:
#             self.sigma = np.asarray(sigma, dtype=float)
            
#             positive_sigma = self.sigma[self.sigma > 0]
#             if len(positive_sigma) > 0:
#                 sigma_floor = float(np.median(positive_sigma) * 0.1)
#             else:
#                 sigma_floor = float(np.finfo(float).eps)
#             self.sigma = np.where(
#                 self.sigma <= 0, sigma_floor, self.sigma
#             )
#         else:
#             self.sigma = None

#         # apply minimum pair-count filter: zero-weight bins with
#         # too few pairs
#         self._n_filtered = 0
#         if self.min_pairs > 0 and self.pair_counts is not None:
#             low_count_mask = self.pair_counts < self.min_pairs
#             self._n_filtered = int(np.sum(low_count_mask))
#             if self._n_filtered > 0:
#                 self.weights[low_count_mask] = 0.0
                
#                 if self.sigma is not None:
#                     self.sigma[low_count_mask] = np.inf
#                 n_remaining = len(self.lags) - self._n_filtered
#                 if n_remaining < 4:
#                     import warnings as _w
#                     _w.warn(
#                         f"min_pairs={self.min_pairs} filters "
#                         f"{self._n_filtered} of {len(self.lags)} "
#                         f"lag bins, leaving only {n_remaining}. "
#                         f"Model fitting may fail or be unreliable. "
#                         f"Consider lowering min_pairs.",
#                         UserWarning,
#                         stacklevel=2,
#                     )

#         self.fitted_models: List[FittedVariogramModel] = []
#         self.best_model: Optional[FittedVariogramModel] = None

#     @staticmethod
#     def _compute_weights(
#         empirical_variogram: np.ndarray,
#         pair_counts: Optional[np.ndarray],
#         weighting: str,
#     ) -> np.ndarray:
#         """Compute WLS weights for variogram fitting.

#         Parameters
#         ----------
#         empirical_variogram : ndarray
#             Empirical semivariance values.
#         pair_counts : ndarray or None
#             Number of point pairs per lag bin.
#         weighting : str
#             ``'cressie'``  – N(h) / γ̂(h)²  (Cressie, 1985).
#             ``'pair_count'`` – N(h) only.
#             ``'uniform'``  – all weights equal to 1.

#         Returns
#         -------
#         weights : ndarray

#         References
#         ----------
#         Cressie, N. (1985). Fitting variogram models by weighted least
#         squares. J. Int. Assoc. Math. Geol., 17(5), 563–586.
#         """
#         n = len(empirical_variogram)

#         if weighting == 'uniform' or pair_counts is None:
#             if weighting == 'cressie' and pair_counts is None:
#                 import warnings as _w
#                 _w.warn(
#                     "Cressie weighting requested but no pair_counts supplied; "
#                     "falling back to uniform weights.",
#                     UserWarning,
#                     stacklevel=3,
#                 )
#             return np.ones(n, dtype=float)

#         counts = np.asarray(pair_counts, dtype=float)

#         if weighting == 'pair_count':
#             return counts

#         # --- Cressie weighting:  w_i = N(h_i) / γ̂(h_i)² ---
#         gamma_sq = np.square(empirical_variogram)
#         # guard against division by zero at lags where γ̂ ≈ 0
#         gamma_sq = np.where(gamma_sq < np.finfo(float).eps, np.finfo(float).eps, gamma_sq)
#         return counts / gamma_sq

#     def generate_candidates(
#         self,
#         max_components: int = 1,
#         include_nugget: bool = True,
#         include_unbounded: bool = True,
#         bounded_only_combinations: bool = True,
#     ) -> List[CompositeVariogramModel]:
#         """Generate candidate composite models.

#         When ``include_nugget=True``, every candidate is generated in both
#         a with-nugget and a without-nugget variant so that model selection
#         can decide whether an explicit nugget is needed.

#         The Matérn model's smoothness parameter ν already controls the
#         shape at short lags, making multiple Matérn components redundant
#         and poorly identifiable.  Combinations containing more than one
#         Matérn component are therefore excluded.

#         Parameters
#         ----------
#         max_components : int
#             Maximum number of component models to combine.
#         include_nugget : bool
#             Whether to include nugget variants.  When True, multi-component
#             models are generated both with and without nugget; single-
#             component models always include the nugget.
#         include_unbounded : bool
#             Whether to include unbounded (non-stationary) models.
#         bounded_only_combinations : bool
#             If True, only combine bounded models together.
#             Unbounded models are only tried as single components.

#         Returns
#         -------
#         candidates : List[CompositeVariogramModel]
#             List of candidate composite models.

#         References
#         ----------
#         Stein, M.L. (1999). *Interpolation of Spatial Data: Some Theory
#         for Kriging*. Springer.  Argues that the Matérn class subsumes
#         exponential (ν=0.5) and Gaussian (ν→∞), making nested Matérn
#         structures redundant for kriging.
#         """
#         candidates = []

#         # get model lists
#         bounded = self.BOUNDED_MODELS
#         unbounded = self.UNBOUNDED_MODELS if include_unbounded else []

#         # generate bounded-only combinations
#         for n in range(1, max_components + 1):
#             for combo in combinations_with_replacement(bounded, n):
#                 combo_list = list(combo)

#                 # ── block multiple Matérns (Stein, 1999) ──
#                 # A single Matérn's ν parameter already controls
#                 # smoothness; stacking Matérns creates redundant,
#                 # poorly identifiable parameters.  Allow at most one.
#                 if combo_list.count('matern') > 1:
#                     continue

#                 # ── nugget variants ──
#                 # When include_nugget is True, generate both with-
#                 # and without-nugget variants so the model selection
#                 # can decide whether an explicit nugget is needed or
#                 # whether a short-range component can absorb the
#                 # micro-scale variance.
#                 if include_nugget:
#                     nugget_options = [True, False]
#                 else:
#                     nugget_options = [False]

#                 for use_nugget in nugget_options:
#                     try:
#                         model = CompositeVariogramModel(
#                             combo_list,
#                             include_nugget=use_nugget,
#                         )
#                         candidates.append(model)
#                     except ValueError:
#                         continue

#         # add single unbounded models (not in combinations to preserve validity)
#         if include_unbounded:
#             for name in unbounded:
#                 try:
#                     model = CompositeVariogramModel(
#                         [name],
#                         include_nugget=True,  # always include nugget
#                     )
#                     candidates.append(model)
#                 except ValueError:
#                     continue

#             # optionally: bounded + unbounded combinations
#             if not bounded_only_combinations:
#                 for bounded_name in bounded:
#                     for unbounded_name in unbounded:
#                         nugget_options = [True, False] if include_nugget else [False]
#                         for use_nugget in nugget_options:
#                             try:
#                                 model = CompositeVariogramModel(
#                                     [bounded_name, unbounded_name],
#                                     include_nugget=use_nugget,
#                                 )
#                                 candidates.append(model)
#                             except ValueError:
#                                 continue

#         return candidates
    
#     # ── nugget pre-estimation (two-stage approach) ──────────────────

#     @staticmethod
#     def _estimate_nugget_from_short_lags(
#         lags: np.ndarray,
#         variogram: np.ndarray,
#         n_lags: int = 5,
#     ) -> float:
#         """Pre-estimate nugget by extrapolating short-lag bins to h=0.

#         Fits γ(h) ≈ C₀ + bh to the first few lags and returns the
#         y-intercept clamped to [0, 0.5 × max(γ)].  This is the "Stage 1"
#         of two-stage fitting and prevents the optimizer from trading sill
#         for nugget inappropriately.

#         References
#         ----------
#         Cressie, N. (1985). Fitting variogram models by weighted least
#         squares.  J. Int. Assoc. Math. Geol., 17(5), 563–586.
#         """
#         max_gamma = float(np.nanmax(variogram))
#         n_fit = min(n_lags, max(2, len(lags) // 4))
#         if n_fit < 2:
#             return max_gamma * 0.1

#         short_lags = lags[:n_fit]
#         short_gamma = variogram[:n_fit]

#         try:
#             _slope, intercept = np.polyfit(short_lags, short_gamma, 1)
#             return float(np.clip(intercept, 0.0, max_gamma * 0.5))
#         except (np.linalg.LinAlgError, ValueError):
#             return max_gamma * 0.1

#     # ── multi-start initial guesses ──────────────────────────────

#     @staticmethod
#     def _generate_multistart_guesses(
#         p0_base: np.ndarray,
#         bounds: tuple,
#         n_restarts: int,
#         rng: np.random.Generator,
#     ) -> list:
#         """Generate diverse starting points using Latin-Hypercube-like sampling.

#         Restart 0:  use the default guess unchanged.
#         Restart 1:  halve all range parameters (explore short-range basin).
#         Restart 2:  double all range parameters (explore long-range basin).
#         Remaining:  random uniform samples in [lower, upper] for each param.
#         """
#         lb = np.asarray(bounds[0], dtype=float)
#         ub = np.asarray(bounds[1], dtype=float)
#         guesses = [p0_base.copy()]

#         if n_restarts >= 2:
#             # short-range variant
#             short = p0_base.copy()
#             short = np.clip(short * 0.5, lb, ub)
#             guesses.append(short)

#         if n_restarts >= 3:
#             # long-range variant
#             long_ = p0_base.copy()
#             long_ = np.clip(long_ * 2.0, lb, ub)
#             guesses.append(long_)

#         # fill remaining restarts with random samples in the feasible region
#         for _ in range(max(0, n_restarts - len(guesses))):
#             # uniform random between lb and ub (log-scale for wide bounds)
#             rand = rng.random(len(p0_base))
#             # use log-uniform for strictly positive params with wide range
#             sample = np.empty_like(p0_base)
#             for j in range(len(p0_base)):
#                 lo, hi = max(lb[j], 1e-12), ub[j]
#                 if hi / lo > 50:
#                     # log-uniform sampling for wide-range params
#                     sample[j] = np.exp(
#                         np.log(lo) + rand[j] * (np.log(hi) - np.log(lo))
#                     )
#                 else:
#                     sample[j] = lo + rand[j] * (hi - lo)
#             guesses.append(np.clip(sample, lb, ub))

#         return guesses[:n_restarts]

#     # ── range separation enforcement ─────────────────────────────

#     #: Minimum ratio between successive practical ranges in a
#     #: multi-component model.  Two components whose practical ranges
#     #: are closer than this factor are nearly non-identifiable:
#     #: the optimizer can trade sill between them freely.  A 3×
#     #: separation ensures each component captures a distinct spatial
#     #: scale, consistent with the physical error-source hierarchy
#     #: in topographic differencing (meter-scale misclassification →
#     #: hundred-meter-scale flight-line striping → kilometre-scale
#     #: calibration bias).
#     #:
#     #: No standard numerical rule exists in the literature; the
#     #: constraint is a practical identifiability guard.  Gringarten
#     #: & Deutsch (2001) and Webster & Oliver (2007) recommend that
#     #: nested structures represent "distinct scales" without
#     #: specifying a ratio.  3× is conservative: with spherical
#     #: models, two components with ranges a and 3a have their
#     #: transition zones overlapping in only the first third of the
#     #: longer component's range — enough for the optimizer to
#     #: distinguish them.
#     MIN_RANGE_SEPARATION: float = 2.0

#     #: Minimum practical range as a fraction of the first lag.
#     #: Components with practical range below this threshold are
#     #: degenerate — they contribute semivariance entirely at sub-bin
#     #: scales, acting as nugget surrogates rather than genuine spatial
#     #: structure.  A floor of 1.0 × bin_width ensures every component
#     #: contributes spatially resolvable structure.
#     MIN_RANGE_FRACTION: float = 1.0

#     @staticmethod
#     def _get_practical_ranges(
#         model: CompositeVariogramModel,
#         params: np.ndarray,
#     ) -> List[float]:
#         """Extract practical (effective) ranges from fitted params.

#         The practical range is the distance at which the model reaches
#         ~95% of its sill.  For spherical models this equals the range
#         parameter; for exponential models it is 3× the range parameter.

#         Returns a list of practical ranges for bounded components that
#         have a 'range' parameter, in component order.
#         """
#         model.set_params(params)
#         practical = []
#         for i, spec in enumerate(model._components):
#             if spec.is_bounded and 'range' in spec.param_names:
#                 range_idx = spec.param_names.index('range')
#                 comp_params = model.get_component_params(i)
#                 raw_range = comp_params[range_idx]
#                 prf = spec.practical_range_factor
#                 if prf is not None and prf > 0:
#                     practical.append(raw_range * prf)
#                 else:
#                     practical.append(raw_range)
#         return practical

#     #: Maximum allowed total sill as a multiple of the observed
#     #: maximum semivariance.  With multiple freely-parameterised
#     #: components the optimizer can inflate one component's sill to
#     #: compensate for another, producing a total sill far exceeding
#     #: the data.  2× allows some headroom for noise while rejecting
#     #: physically unrealistic fits (Gringarten & Deutsch, 2001).
#     MAX_SILL_RATIO: float = 2.0

#     def validate_fit(
#         self,
#         model: CompositeVariogramModel,
#         params: np.ndarray,
#         max_lag: Optional[float] = None,
#     ) -> Tuple[bool, List[str]]:
#         """Consolidated post-fit validation.

#         Runs all validation checks on a fitted model and returns a single
#         pass/fail result with diagnostic warnings.

#         Checks performed:

#         1. **Range floor**: every practical range must exceed
#            ``MIN_RANGE_FRACTION × bin_width`` (rejects sub-bin
#            nugget surrogates).
#         2. **Range separation**: for multi-component models, sorted
#            practical ranges must satisfy ``r_{i+1}/r_i >=
#            MIN_RANGE_SEPARATION`` (rejects non-identifiable fits).
#         3. **Total sill**: total sill must not exceed
#            ``MAX_SILL_RATIO × max(γ̂)`` (rejects physically
#            unrealistic fits).
#         4. **Half-lag heuristic** (warning only): any range exceeding
#            ``max_lag / 2`` triggers a warning that the fit is poorly
#            constrained at long distances (Journel & Huijbregts, 1978).

#         Parameters
#         ----------
#         model : CompositeVariogramModel
#             The candidate model with params already set.
#         params : ndarray
#             Optimal parameter vector.
#         max_lag : float, optional
#             Maximum lag distance for the half-lag warning.
#             Defaults to ``max(self.lags)``.

#         Returns
#         -------
#         passes : bool
#             True if all hard checks (1–3) pass.
#         warnings : list of str
#             Diagnostic warnings (e.g. half-lag exceeded).

#         References
#         ----------
#         Gringarten, E. & Deutsch, C.V. (2001). Teacher's aide:
#         variogram interpretation and modeling. *Math. Geol.*,
#         33(4), 507–534.

#         Journel, A.G. & Huijbregts, Ch.J. (1978). *Mining
#         Geostatistics*. Academic Press.
#         """
#         warnings_list: List[str] = []

#         if max_lag is None:
#             max_lag = float(np.nanmax(self.lags))

#         practical = self._get_practical_ranges(model, params)

#         # ── 1. Range floor ──
#         if len(self.lags) >= 2:
#             bin_width = float(self.lags[1] - self.lags[0])
#         else:
#             bin_width = float(self.lags[0]) if len(self.lags) == 1 else 1.0
#         min_range = self.MIN_RANGE_FRACTION * bin_width
#         for pr in practical:
#             if pr < min_range:
#                 return False, warnings_list

#         # ── 2. Range separation ──
#         if len(practical) > 1:
#             ordered = sorted(practical)
#             for j in range(len(ordered) - 1):
#                 if ordered[j] < np.finfo(float).eps:
#                     return False, warnings_list
#                 if ordered[j + 1] / ordered[j] < self.MIN_RANGE_SEPARATION:
#                     return False, warnings_list

#         # ── 3. Total sill ──
#         model.set_params(params)
#         if model.is_stationary:
#             total_sill = model.get_total_sill()
#             if total_sill is not None:
#                 max_gamma = float(np.nanmax(self.empirical_variogram))
#                 if total_sill > self.MAX_SILL_RATIO * max_gamma:
#                     return False, warnings_list

#         # ── 4. Half-lag heuristic (warning only) ──
#         half_lag = max_lag / 2.0
#         for i, spec in enumerate(model._components):
#             if 'range' in spec.param_names:
#                 range_idx = spec.param_names.index('range')
#                 comp_params = model.get_component_params(i)
#                 fitted_range = comp_params[range_idx]
#                 if fitted_range > half_lag:
#                     name = model.component_names[i]
#                     warnings_list.append(
#                         f"WARNING: {name} range ({fitted_range:.1f}) exceeds "
#                         f"half the maximum lag ({half_lag:.1f}).  The variogram "
#                         f"is poorly constrained beyond this distance — consider "
#                         f"increasing max_lag or treating this range estimate "
#                         f"with caution."
#                     )

#         return True, warnings_list

#     # ── core model fitting ───────────────────────────────────────

#     def fit_model(
#         self,
#         model: CompositeVariogramModel,
#         maxfev: int = 10000,
#         n_restarts: int = 8,
#     ) -> Optional[FittedVariogramModel]:
#         """Fit a single composite model to the empirical variogram.

#         When the model includes a nugget, a two-stage approach is used:
#         the nugget is first pre-estimated from short-lag extrapolation,
#         then the full optimisation is run with the nugget constrained
#         in an asymmetric window around the pre-estimate (−50% / +80%,
#         with a floor of 15% of max semivariance).  This prevents the
#         common failure mode where the optimizer trades sill for nugget.

#         For multi-component models, fits that violate the minimum
#         practical-range separation (``MIN_RANGE_SEPARATION``, default
#         3×) are rejected.  This prevents the optimizer from collapsing
#         two components onto the same spatial scale, which causes
#         parameter non-identifiability.

#         Parameters
#         ----------
#         model : CompositeVariogramModel
#             Model to fit.
#         maxfev : int
#             Maximum function evaluations for optimizer.
#         n_restarts : int
#             Number of random restarts to avoid local minima.

#         Returns
#         -------
#         fitted : FittedVariogramModel or None
#             Fitted model, or None if fitting failed.
#         """
#         # get initial guess and bounds
#         p0_base = model.default_guess(self.lags, self.empirical_variogram)
#         bounds = model.bounds(self.lags, self.empirical_variogram)

#         # ── nugget bounds ──
#         # Use the full [0, 0.5 × max_γ] range from the composite model
#         # bounds rather than the previous tight asymmetric constraint
#         # (−50%/+80% around short-lag extrapolation with a 15% floor).
#         # The tight bounds locked the nugget too low when the first few
#         # lags still contained spatial structure, causing MSSPE >> 1
#         # (kriging variance underestimation).
#         #
#         # The initial guess is still informed by short-lag extrapolation
#         # for a reasonable starting point, but the optimizer is free to
#         # explore the full feasible range.
#         if model.include_nugget:
#             nugget_pre = self._estimate_nugget_from_short_lags(
#                 self.lags, self.empirical_variogram
#             )
#             nugget_idx = model.n_params - 1  # nugget is always last
#             # bounds already set to [0, 0.5 * max_gamma] by
#             # CompositeVariogramModel.bounds(); no further tightening
#             p0_base[nugget_idx] = np.clip(
#                 nugget_pre, bounds[0][nugget_idx], bounds[1][nugget_idx]
#             )

#         # prepare fitting function
#         def model_func(h, *params):
#             model.set_params(np.array(params))
#             return model(h)

#         rng = np.random.default_rng()
#         guesses = self._generate_multistart_guesses(
#             p0_base, bounds, n_restarts, rng
#         )

#         best_result = None
#         best_rss = np.inf

#         for p0 in guesses:
#             try:
#                 popt, pcov = curve_fit(
#                     model_func,
#                     self.lags,
#                     self.empirical_variogram,
#                     p0=p0,
#                     sigma=self.sigma,
#                     absolute_sigma=True if self.sigma is not None else False,
#                     bounds=bounds,
#                     maxfev=maxfev,
#                 )

#                 # ── consolidated validation ──
#                 passes, fit_warnings = self.validate_fit(model, popt)
#                 if not passes:
#                     continue

#                 # compute RSS using the same objective curve_fit minimized
                
#                 model.set_params(popt)
#                 residuals = self.empirical_variogram - model(self.lags)
#                 if self.sigma is not None:
#                     # Match curve_fit's sigma-weighted objective
#                     safe_sigma = np.where(
#                         np.isfinite(self.sigma) & (self.sigma > 0),
#                         self.sigma,
#                         np.inf,
#                     )
#                     rss = np.sum((residuals / safe_sigma) ** 2)
#                 else:
#                     rss = np.sum(self.weights * residuals**2)

#                 if rss < best_rss:
#                     best_rss = rss
#                     best_result = (popt, pcov, rss, fit_warnings)

#             except (RuntimeError, ValueError):
#                 continue

#         if best_result is None:
#             return None

#         popt, pcov, rss, best_warnings = best_result
#         model.set_params(popt)

#         # ── boundary detection ──
#         # Warn when fitted parameters land within 2% of their bounds.
#         # This almost always indicates the optimizer wanted to go
#         # further but was blocked — the fitted model is suspect.
#         lower_bounds, upper_bounds = bounds
#         param_names = model.param_names
#         for i, (val, lb, ub) in enumerate(
#             zip(popt, lower_bounds, upper_bounds)
#         ):
#             span = ub - lb
#             if span <= 0:
#                 continue
#             name = param_names[i] if i < len(param_names) else f"param[{i}]"
#             if (val - lb) / span < 0.02:
#                 msg = (
#                     f"WARNING: {model.structural_description()} "
#                     f"parameter '{name}' ({val:.4g}) is at its "
#                     f"lower bound ({lb:.4g}). The fit may be "
#                     f"poorly constrained."
#                 )
#                 warnings.warn(msg)
#                 best_warnings.append(msg)
#             elif (ub - val) / span < 0.02:
#                 msg = (
#                     f"WARNING: {model.structural_description()} "
#                     f"parameter '{name}' ({val:.4g}) is at its "
#                     f"upper bound ({ub:.4g}). The variogram may "
#                     f"not reach a sill within the observed lag "
#                     f"range — consider increasing max_lag or "
#                     f"treating this estimate with caution."
#                 )
#                 warnings.warn(msg)
#                 best_warnings.append(msg)

#         # compute information criteria
        
#         if self.sigma is not None:
#             finite_mask = np.isfinite(self.sigma) & (self.sigma > 0)
#             n_eff = int(np.sum(finite_mask))
#         else:
#             n_eff = len(self.lags)
#         k = model.n_params

#         if self.sigma is not None:
#             ll = self._log_likelihood(model, popt)
#             aic = 2 * k - 2 * ll
#             bic = k * np.log(max(n_eff, 1)) - 2 * ll
#         else:
#             n = n_eff
#             aic = n * np.log(rss / max(n, 1)) + 2 * k
#             bic = n * np.log(rss / max(n, 1)) + k * np.log(max(n, 1))

#         return FittedVariogramModel(
#             composite_model=model,
#             params=popt,
#             param_cov=pcov,
#             rss=rss,
#             aic=aic,
#             bic=bic,
#             warnings=best_warnings,
#         )
    
#     def _log_likelihood(
#         self,
#         model: CompositeVariogramModel,
#         params: np.ndarray
#     ) -> float:
#         """Compute log-likelihood assuming Gaussian errors.

#         Excludes bins with non-finite sigma (e.g. sigma=inf from min_pairs filtering).  

#         """
#         model.set_params(params)
#         predicted = model(self.lags)
#         residuals = self.empirical_variogram - predicted

#         # Only include bins with finite, positive sigma
#         finite_mask = np.isfinite(self.sigma) & (self.sigma > 0)
#         sig = self.sigma[finite_mask]
#         res = residuals[finite_mask]

#         # heteroscedastic Gaussian log-likelihood
#         ll = -0.5 * np.sum(
#             np.log(2 * np.pi * sig**2) + (res**2) / (sig**2)
#         )
#         return ll
    
#     def fit_all_candidates(
#         self,
#         max_components: int = 1,
#         include_nugget: bool = True,
#         include_unbounded: bool = True,
#         bounded_only_combinations: bool = True,
#     ) -> None:
#         """Fit all candidate models.

#         Parameters
#         ----------
#         max_components : int
#             Maximum number of nested components (default 1).
#         include_nugget : bool
#             Include nugget in all models.
#         include_unbounded : bool
#             Include non-stationary models.
#         bounded_only_combinations : bool
#             If True (default), unbounded models are only tried as single
#             components.  If False, bounded + unbounded combinations are
#             also generated (e.g. spherical + linear + nugget).
#         """
#         candidates = self.generate_candidates(
#             max_components=max_components,
#             include_nugget=include_nugget,
#             include_unbounded=include_unbounded,
#             bounded_only_combinations=bounded_only_combinations,
#         )

#         self.fitted_models = []
#         max_lag = float(np.nanmax(self.lags))

#         for model in candidates:
#             fitted = self.fit_model(model)
#             if fitted is not None:
#                 self.fitted_models.append(fitted)
    
#     def select_best(self, criterion: str = 'aic') -> FittedVariogramModel:
#         """Select best model by criterion.

#         Parameters
#         ----------
#         criterion : str
#             Selection criterion: ``'aic'``, ``'bic'``, or
#             ``'msspe'``.  For ``'msspe'``, the model whose kriging
#             LOOCV MSSPE is closest to 1.0 is selected (minimise
#             ``|MSSPE − 1|``).  Requires that MSSPE has been computed
#             for each candidate (see ``fit_best_model_auto`` with
#             ``criterion='msspe'`` or ``compute_msspe=True``).

#         Returns
#         -------
#         best : FittedVariogramModel
#             Best model according to criterion.

#         References
#         ----------
#         Lark, R.M. (2000).  A comparison of some robust estimators of
#         the variogram for use in soil survey.  *Eur. J. Soil Sci.*,
#         51, 137–157.  Uses MSSPE ≈ 1.0 as acceptance criterion.
#         """
#         if not self.fitted_models:
#             raise ValueError("No fitted models. Call fit_all_candidates() first.")

#         if criterion == 'aic':
#             scores = [m.aic for m in self.fitted_models]
#         elif criterion == 'bic':
#             scores = [m.bic for m in self.fitted_models]
#         elif criterion == 'msspe':
#             scores = [
#                 abs(m.msspe - 1.0) if m.msspe is not None else np.inf
#                 for m in self.fitted_models
#             ]
#         else:
#             raise ValueError(f"Unknown criterion: {criterion}")
        
#         best_idx = np.argmin(scores)
#         self.best_model = self.fitted_models[best_idx]

#         # emit any diagnostic warnings from the selected model
#         for w in self.best_model.warnings:
#             warnings.warn(w, UserWarning, stacklevel=2)

#         return self.best_model
    
#     def bootstrap_best_model(
#         self,
#         n_boot: int = 500,
#         seed: Optional[int] = None,
#     ) -> np.ndarray:
#         """Bootstrap parameter uncertainty for best model.

#         Generates synthetic variograms by adding noise to the empirical
#         variogram and re-fitting the best model.  The bootstrap refit
#         uses the same WLS weighting scheme as the original fit so that
#         the uncertainty samples faithfully reflect the fitted model.

#         Parameters
#         ----------
#         n_boot : int
#             Number of bootstrap samples.
#         seed : int, optional
#             Random seed.

#         Returns
#         -------
#         param_samples : ndarray
#             Bootstrap samples, shape (n_valid, n_params).  Rows where
#             fitting failed are removed.
#         """
#         if self.best_model is None:
#             raise ValueError("No best model selected. Call select_best() first.")

#         rng = np.random.default_rng(seed)
#         model = self.best_model.composite_model
#         p0 = self.best_model.params
#         bounds = model.bounds(self.lags, self.empirical_variogram)

#         # generate synthetic variograms
#         if self.sigma is not None:
#             noise = rng.normal(
#                 loc=self.empirical_variogram,
#                 scale=self.sigma,
#                 size=(n_boot, len(self.lags))
#             )
#         else:
#             # use residual-based bootstrap
#             fitted = self.best_model.predict(self.lags)
#             residuals = self.empirical_variogram - fitted
#             noise = fitted + rng.choice(residuals, size=(n_boot, len(self.lags)))

    
#         boot_sigma = self.sigma  # None when sigma wasn't provided
        
#         param_samples = []

#         def model_func(h, *params):
#             model.set_params(np.array(params))
#             return model(h)

#         for i in range(n_boot):
#             try:
#                 popt, _ = curve_fit(
#                     model_func,
#                     self.lags,
#                     noise[i],
#                     p0=p0,
#                     sigma=boot_sigma,
#                     absolute_sigma=False,
#                     bounds=bounds,
#                     maxfev=5000,
#                 )
#                 param_samples.append(popt)
#             except RuntimeError:
#                 param_samples.append([np.nan] * model.n_params)

#         param_samples = np.array(param_samples)

#         # remove failed fits
#         valid = ~np.isnan(param_samples).any(axis=1)
#         param_samples = param_samples[valid]

#         self.best_model.param_samples = param_samples
#         return param_samples
    
#     def summary(self) -> str:
#         """Generate summary of fitted models."""
#         if not self.fitted_models:
#             return "No models fitted yet."

#         has_msspe = any(m.msspe is not None for m in self.fitted_models)
#         has_msspe_std = any(
#             getattr(m, 'msspe_std', None) is not None
#             for m in self.fitted_models
#         )

#         lines = ["=" * 80]
#         lines.append("VARIOGRAM MODEL SELECTION SUMMARY")
#         lines.append("=" * 80)
#         header = f"{'Model':<40} {'AIC':>10} {'BIC':>10}"
#         if has_msspe:
#             header += f" {'MSSPE':>12}"
#         lines.append(header)
#         lines.append("-" * 80)

#         # sort by AIC
#         sorted_models = sorted(self.fitted_models, key=lambda m: m.aic)

#         for model in sorted_models:
#             name = "+".join(model.composite_model.component_names)
#             if model.composite_model.include_nugget:
#                 name += "+nugget"

#             row = f"{name:<40} {model.aic:>10.2f} {model.bic:>10.2f}"
#             if has_msspe:
#                 if model.msspe is not None:
#                     msspe_str = f"{model.msspe:.3f}"
#                     std = getattr(model, 'msspe_std', None)
#                     if std is not None:
#                         msspe_str += f"±{std:.3f}"
#                 else:
#                     msspe_str = "N/A"
#                 row += f" {msspe_str:>12}"
#             lines.append(row)

#         lines.append("=" * 80)

#         if self.best_model:
#             lines.append("\nBEST MODEL DETAILS:")
#             lines.append(self.best_model.composite_model.description())

#             if not self.best_model.composite_model.is_stationary:
#                 lines.append("\nWARNING: Selected model is NON-STATIONARY.")
#                 lines.append("The process has no finite variance.")
#                 lines.append("Results are scale-dependent. Consider detrending.")

#         return "\n".join(lines)


# class RasterDataHandler:
#     """
#     Load vertical differencing raster data, subtract a vertical systematic error
#     from the raster, and randomly sample raster data for further analysis.

#     Attributes
#     ----------
#     raster_path : str
#         File path to the raster data.
#     unit : str
#         Unit of measurement for the raster data (for plotting labels).
#     resolution : float
#         Nominal raster resolution (linear units).
#     rioxarray_obj : rioxarray.DataArray | None
#         The rioxarray object holding the raster data.
#     data_array : np.ndarray | None
#         Loaded raster values as a 1D array of finite pixels.
#     samples : np.ndarray | None
#         Sampled values from the raster.
#     coords : np.ndarray | None
#         Coordinates (x, y) of the sampled values.
#     bbox : shapely.geometry.Polygon
#         Bounding box of the raster.
#     """

#     def __init__(self, raster_path: str, unit: str, resolution: float):
#         self.raster_path = raster_path
#         self.unit = unit
#         self.resolution = resolution
#         self.rioxarray_obj = None
#         self.data_array = None
#         self.samples = None
#         self.coords = None
#         self.shapely_geoms = None
#         self.merged_geom = None
#         self.detailed_area = None

#         with rasterio.open(self.raster_path) as src:
#             bounds = src.bounds
#             self.bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
#             self.bbox = box(*self.bounds)

#     def get_detailed_area(self) -> None:
#         """
#         Compute the precise area covered by valid data in the raster by vectorizing
#         the finite/nodata mask into polygon shapes and dissolving them.
#         """
#         with rasterio.open(self.raster_path) as src:
#             data = src.read(1).astype(float)
#             nodata = src.nodata
#             valid = (~np.isnan(data)) if nodata is None else ((data != nodata) & ~np.isnan(data))
#             geoms = shapes(valid.astype(np.uint8), mask=valid, transform=src.transform)
#         self.shapely_geoms = [shape(geom) for geom, val in geoms if val == 1]
#         self.merged_geom = unary_union(self.shapely_geoms)
#         self.detailed_area = self.merged_geom.area

#     def load_raster(self, masked: bool = True) -> None:
#         """
#         Load raster data and store finite values in self.data_array.

#         Parameters
#         ----------
#         masked : bool
#             If True, open as masked and coerce mask to NaN.
#         """
        
#         da = rio.open_rasterio(self.raster_path, masked=masked)
#         if "band" in da.dims and da.sizes.get("band", 1) == 1:
#             da = da.squeeze("band", drop=True)
#         arr = da.values
#         if np.ma.isMaskedArray(arr):
#             arr = arr.filled(np.nan)
#         nodata = da.rio.nodata
#         valid = np.isfinite(arr)
#         if nodata is not None:
#             valid &= (arr != nodata)
#         self.rioxarray_obj = da
#         self.data_array = np.asarray(arr[valid], dtype=float).ravel()

#     def subtract_value_from_raster(self, output_raster_path: str, value_to_subtract: float) -> None:
#         """
#         Subtract a specified value from the raster and write a new file.

#         Parameters
#         ----------
#         output_raster_path : str
#             Path to the output raster.
#         value_to_subtract : float
#             Value to subtract from each valid pixel.
#         """
#         with rasterio.open(self.raster_path) as src:
#             data = src.read()
#             nodata = src.nodata
#             mask = (data != nodata) if nodata is not None else np.ones(data.shape, dtype=bool)
#             data = data.astype(float)
#             data[mask] -= value_to_subtract
#             out_meta = src.meta.copy()
#             out_meta.update({'dtype': 'float32', 'nodata': nodata})
#             with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
#                 dst.write(data)

#     def plot_raster(self, plot_title: str):
#         """
#         Plot the loaded rioxarray DataArray with a diverging colormap.

#         Raises
#         ------
#         RuntimeError
#             If raster has not been loaded yet.
#         """
        
#         if self.rioxarray_obj is None:
#             raise RuntimeError("Raster not loaded. Call load_raster() first.")
#         rio_data = self.rioxarray_obj
#         fig, ax = plt.subplots(figsize=(10, 6))
#         rio_data.plot(cmap="bwr_r", ax=ax, robust=True)
#         ax.set_title(plot_title, pad=30)
#         ax.set_xlabel('Easting')
#         ax.set_ylabel('Northing')
#         ax.ticklabel_format(style="plain")
#         ax.set_aspect('equal')
#         return fig

#     def sample_raster(
#         self,
#         area_side: float,
#         samples_per_area: float,
#         max_samples: int,
#         *,
#         seed: Optional[int] = None
#     ) -> None:
#         """
#         Randomly sample valid pixels from the raster, storing (values, coords).

#         Parameters
#         ----------
#         area_side : float
#             Reference side length used to convert pixel area to reference-area units
#             (e.g., 1000 for km² if coordinates are meters).
#         samples_per_area : float
#             Number of samples to draw per unit of reference area.
#         max_samples : int
#             Maximum total samples to draw.
#         seed : int | None
#             RNG seed for reproducibility.

#         Raises
#         ------
#         ValueError
#             If requested samples exceed valid pixels, or computed total is < 1.
#         """
#         with rasterio.open(self.raster_path) as src:
#             rng = np.random.default_rng(seed)

#             data = src.read(1).astype(float)
#             nodata = src.nodata
#             valid = np.isfinite(data)
#             if nodata is not None:
#                 valid &= (data != nodata)

#             cell_area_m2 = abs(src.res[0] * src.res[1])
#             valid_rows, valid_cols = np.where(valid)
#             valid_count = valid_rows.size
#             cell_area_in_reference = cell_area_m2 / (area_side ** 2)
#             total_samples = min(int(cell_area_in_reference * samples_per_area * valid_count), max_samples)

            
#             if total_samples < 1:
#                 raise ValueError("Computed total_samples < 1. Increase samples_per_area or max_samples.")

#             if total_samples > valid_count:
#                 raise ValueError("Requested samples exceed valid pixel count. Reduce samples_per_area.")

#             chosen = rng.choice(valid_count, size=total_samples, replace=False)
#             rows = valid_rows[chosen]
#             cols = valid_cols[chosen]
#             samples = data[rows, cols]
#             x_coords, y_coords = src.xy(rows, cols)
#             coords = np.vstack([x_coords, y_coords]).T

#             mask = np.isfinite(samples)
#             self.samples = samples[mask]
#             self.coords = coords[mask]


# class StatisticalAnalysis:
#     """
#     Statistical utilities for exploratory plotting and bootstrap uncertainty of the median.
#     """

#     def __init__(self, raster_data_handler: RasterDataHandler):
#         self.raster_data_handler = raster_data_handler

#     def plot_data_stats(self, filtered: bool = True):
#         """
#         Plot histogram of raster values with basic statistics annotated.

#         Parameters
#         ----------
#         filtered : bool
#             If True, clip to 1st–99th percentiles for visualization only.

#         Returns
#         -------
#         matplotlib.figure.Figure
#         """
#         data = self.raster_data_handler.data_array
#         if data is None or len(data) == 0:
#             raise ValueError("No data available to plot. Call load_raster() first.")

#         mean = np.mean(data)
#         median = np.median(data)
#         # mode on continuous data is often not meaningful; kept for completeness
#         mode_result = stats.mode(data, nan_policy="omit", keepdims=False)
#         mode_vals = np.atleast_1d(mode_result.mode).astype(float)
#         q1 = np.percentile(data, 25)
#         q3 = np.percentile(data, 75)
#         p1 = np.percentile(data, 1)
#         p99 = np.percentile(data, 99)
#         minimum = np.min(data)
#         maximum = np.max(data)

#         if filtered:
#             data = data[(data >= p1) & (data <= p99)]

#         fig, ax = plt.subplots()
#         ax.hist(data, bins=60, density=False, alpha=0.6, color='g')
#         ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
#         ax.axvline(median, color='b', linestyle='dashed', linewidth=1, label='Median')
#         for i, m in enumerate(mode_vals):
#             ax.axvline(m, color='purple', linestyle='dashed', linewidth=1,
#                        label='Mode' if i == 0 else "_nolegend_")

#         mode_str = ", ".join([f"{m:.3f}" for m in mode_vals])
#         textstr = "\n".join((
#             f"Mean: {mean:.3f}",
#             f"Median: {median:.3f}",
#             f"Mode(s): {mode_str}",
#             f"Min: {minimum:.3f}  Max: {maximum:.3f}",
#             f"Q1: {q1:.3f}  Q3: {q3:.3f}",
#         ))

#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
#                 verticalalignment='top', bbox=props)
#         ax.set_xlabel(f'Vertical Difference ({self.raster_data_handler.unit})')
#         ax.set_ylabel('Count')
#         ax.set_title('Histogram of differencing results with exploratory statistics')
#         ax.legend()
#         plt.tight_layout()
#         return fig

#     def bootstrap_uncertainty_subsample(self, n_bootstrap: int = 1000, subsample_proportion: float = 0.1) -> float:
#         """
#         Estimate uncertainty of the median via bootstrap on random subsamples.

#         Parameters
#         ----------
#         n_bootstrap : int
#             Number of bootstrap resamples.
#         subsample_proportion : float
#             Fraction of data per resample.

#         Returns
#         -------
#         float
#             Standard deviation of bootstrap medians.
#         """
#         data = self.raster_data_handler.data_array
#         if data is None or len(data) == 0:
#             raise ValueError("No data available for bootstrap. Call load_raster() first.")

        
#         subsample_size = max(1, int(round(subsample_proportion * len(data))))
#         rng = np.random.default_rng()
#         bootstrap_medians = np.zeros(n_bootstrap)
#         for i in range(n_bootstrap):
#             sample = rng.choice(data, size=subsample_size, replace=True)
#             bootstrap_medians[i] = np.median(sample)
#         return float(np.std(bootstrap_medians))


# class VariogramAnalysis:
#     """
#     Compute empirical variograms across multiple random samples, fit spherical
#     models (with optional nugget), and bootstrap parameter uncertainty.
#     """

#     MIN_PAIRS = 10

#     def __init__(self, raster_data_handler: RasterDataHandler):
#         self.raster_data_handler = raster_data_handler
#         self.mean_variogram = None
#         self.lags = None
#         self.mean_count = None
#         self.err_variogram = None
#         self.fitted_variogram = None
#         self.rmse = None
#         self.sills = None
#         self.ranges = None
#         # full range percentiles (2.5th to 97.5th)
#         self.ranges_min = None
#         self.ranges_max = None
#         self.ranges_median = None
#         # 1σ range percentiles (16th to 84th)
#         self.ranges_p16 = None
#         self.ranges_p84 = None
#         self.err_sills = None
#         self.err_ranges = None
#         # full range percentiles for sills
#         self.sills_min = None
#         self.sills_max = None
#         self.sills_median = None
#         # 1σ range percentiles for sills
#         self.sills_p16 = None
#         self.sills_p84 = None
#         # nugget parameters
#         self.best_nugget = None
#         # full range percentiles for nugget
#         self.min_nugget = None
#         self.max_nugget = None
#         self.median_nugget = None
#         # 1σ range percentiles for nugget
#         self.nugget_p16 = None
#         self.nugget_p84 = None
#         # model selection attributes
#         self.best_aic = None
#         self.best_bic = None
#         self.best_params = None
#         self.best_model_config = None
#         self.cv_mean_error_best_aic = None
#         self.fitted_model = None
#         self.param_samples = None
#         self.n_bins = None
#         self.sigma_variogram = None
#         self.best_model_func = None
#         self.all_variograms = None
#         self.all_counts = None
#         self.estimator = None

    
#     @staticmethod
#     @njit(parallel=True)
#     def bin_distances_and_squared_differences(coords, values, bin_width, max_lag_multiplier, x_extent, y_extent):
#         """
#         Compute and bin pairwise distances, squared differences, and
#         fourth-root absolute differences in a single pass.

#         The squared differences feed the Matheron estimator;
#         the fourth-root absolute differences feed the Cressie–Hawkins
#         robust estimator.  Both are accumulated with O(n_bins) memory.

#         Parameters
#         ----------
#         coords : np.ndarray
#             Array of coordinates of shape (M, 2).
#         values : np.ndarray
#             Array of values of shape (M,).
#         bin_width : float
#             Lag bin width.
#         max_lag_multiplier : float or str
#             Controls maximum lag distance.
#         x_extent, y_extent : float
#             Spatial extent of the domain.

#         Returns
#         -------
#         n_bins : int
#             Number of lag bins.
#         bin_counts : np.ndarray
#             Counts of pairs in each bin.
#         binned_sum_squared_diff : np.ndarray
#             Sum of squared differences per bin (for Matheron).
#         binned_sum_sqrt_abs_diff : np.ndarray
#             Sum of |ΔZ|^0.5 per bin (for Cressie–Hawkins).
#         max_distance : float
#             Maximum observed pairwise distance.
#         max_lag : float
#             Maximum lag used for binning.
#         """
#         approx_max_distance = np.sqrt(x_extent**2 + y_extent**2)

#         if max_lag_multiplier == "max":
#             max_lag = approx_max_distance
#         elif max_lag_multiplier == "median":
#             max_lag = 0.5 * approx_max_distance  # simple heuristic
#         else:
#             max_lag = float(approx_max_distance * max_lag_multiplier)

#         # determine bin edges using diagonal distance as maximum lag
#         n_bins = int(np.ceil(max_lag / bin_width)) + 1
#         bin_edges = np.arange(0, n_bins * bin_width, bin_width)

#         M = coords.shape[0]
#         max_distance = 0.0
#         bin_counts = np.zeros(n_bins, dtype=np.int64)
#         binned_sum_squared_diff = np.zeros(n_bins, dtype=np.float64)
#         binned_sum_sqrt_abs_diff = np.zeros(n_bins, dtype=np.float64)

#         for i in prange(M):
#             for j in range(i + 1, M):
#                 # compute the pairwise distance
#                 d = 0.0
#                 for k in range(coords.shape[1]):
#                     tmp = coords[i, k] - coords[j, k]
#                     d += tmp * tmp
#                 dist = np.sqrt(d)
#                 max_distance = max(max_distance, dist)

#                 # compute the difference
#                 diff = values[i] - values[j]

#                 # Matheron accumulator: squared difference
#                 diff_squared = diff ** 2

#                 # Cressie–Hawkins accumulator: |diff|^0.5
#                 sqrt_abs_diff = np.sqrt(np.abs(diff))

#                 # find the bin for this distance
#                 bin_idx = int(dist / bin_width)
#                 if 0 <= bin_idx < n_bins:
#                     bin_counts[bin_idx] += 1
#                     binned_sum_squared_diff[bin_idx] += diff_squared
#                     binned_sum_sqrt_abs_diff[bin_idx] += sqrt_abs_diff


#         bin_edges = bin_edges[:n_bins]
#         bin_counts = bin_counts[:n_bins]
#         binned_sum_squared_diff = binned_sum_squared_diff[:n_bins]
#         binned_sum_sqrt_abs_diff = binned_sum_sqrt_abs_diff[:n_bins]

#         return n_bins, bin_counts, binned_sum_squared_diff, binned_sum_sqrt_abs_diff, max_distance, max_lag

#     @staticmethod
#     def compute_matheron(bin_counts, ssd, min_pairs: int = 10) -> np.ndarray:
#         """
#         Compute Matheron semivariance γ(h) = SSD(h) / (2 N(h)) for bins with >= min_pairs.
#         """
#         gamma_est = np.full_like(bin_counts, np.nan, dtype=float)
#         for i, (cnt, sum_sq) in enumerate(zip(bin_counts, ssd)):
#             if cnt >= min_pairs:
#                 gamma_est[i] = sum_sq / (2.0 * cnt)
#         return gamma_est

#     @staticmethod
#     def compute_cressie_hawkins(
#         bin_counts, sum_sqrt_abs_diff, min_pairs: int = 10
#     ) -> np.ndarray:
#         """
#         Compute the Cressie–Hawkins robust semivariance estimator.

#         γ̂(h) = [mean(|ΔZ|^0.5)]⁴ / (2 · (0.457 + 0.494 / N(h)))

#         This estimator downweights large squared differences by operating
#         on fourth-root-transformed absolute differences, making it resistant
#         to outliers while remaining a consistent estimator of the variogram.

#         Parameters
#         ----------
#         bin_counts : np.ndarray
#             Number of pairs per lag bin.
#         sum_sqrt_abs_diff : np.ndarray
#             Sum of |Z(x+h) − Z(x)|^0.5 per lag bin.
#         min_pairs : int
#             Minimum pair count for a bin to be considered valid.

#         Returns
#         -------
#         gamma_est : np.ndarray
#             Robust semivariance estimates; NaN for bins with < min_pairs.

#         References
#         ----------
#         Cressie, N. (1985). Fitting variogram models by weighted least
#         squares. J. Int. Assoc. Math. Geol., 17(5), 563–586.

#         Cressie, N. & Hawkins, D.M. (1980). Robust estimation of the
#         variogram: I. J. Int. Assoc. Math. Geol., 12(2), 115–125.
#         """
#         gamma_est = np.full_like(bin_counts, np.nan, dtype=float)
#         for i, (cnt, s) in enumerate(zip(bin_counts, sum_sqrt_abs_diff)):
#             if cnt >= min_pairs:
#                 mean_fourth = (s / cnt) ** 4
#                 correction = 0.457 + 0.494 / cnt
#                 gamma_est[i] = 0.5 * mean_fourth / correction
#         return gamma_est

#     # valid estimator names ------------------------------------------------
#     ESTIMATORS = ('matheron', 'cressie_hawkins')

#     def numba_variogram(
#         self,
#         area_side: float,
#         samples_per_area: float,
#         max_samples: int,
#         bin_width: float,
#         max_lag_multiplier,
#         *,
#         seed: Optional[int] = None,
#         estimator: str = 'matheron',
#         return_sample: bool = False,
#     ):
#         """
#         Compute one empirical variogram by sampling the raster and binning
#         pairwise differences by distance.

#         Parameters
#         ----------
#         area_side, samples_per_area, max_samples : see ``sample_raster``
#         bin_width : float
#         max_lag_multiplier : {"max", "median"} or float
#         seed : int | None
#         estimator : {'matheron', 'cressie_hawkins'}
#             Which semivariance estimator to apply.
#         return_sample : bool, default False
#             If True, also return copies of the sampled coordinates
#             and values.  Useful for downstream MSSPE evaluation on
#             the same spatial sample that produced this variogram
#             (e.g. inside ``fit_variogram_ensemble``).

#         Returns
#         -------
#         bin_counts : np.ndarray
#         variogram : np.ndarray
#         n_bins : int
#         min_distance : float
#         max_distance : float
#         max_lag : float
#         sample_coords : np.ndarray, optional
#             Only returned when ``return_sample=True``.
#         sample_values : np.ndarray, optional
#             Only returned when ``return_sample=True``.
#         """
#         if estimator not in self.ESTIMATORS:
#             raise ValueError(
#                 f"Unknown estimator '{estimator}'. "
#                 f"Choose from {self.ESTIMATORS}."
#             )

#         self.raster_data_handler.sample_raster(area_side, samples_per_area, max_samples, seed=seed)

#         min_distance = 0.0  # retained for compatibility
#         xs = self.raster_data_handler.rioxarray_obj.x.values
#         ys = self.raster_data_handler.rioxarray_obj.y.values
#         x_extent = float(np.max(xs) - np.min(xs))
#         y_extent = float(np.max(ys) - np.min(ys))

#         (n_bins, bin_counts, bssd, bssad,
#          max_distance, max_lag) = self.bin_distances_and_squared_differences(
#             self.raster_data_handler.coords,
#             self.raster_data_handler.samples,
#             bin_width,
#             max_lag_multiplier,
#             x_extent,
#             y_extent,
#         )

#         if estimator == 'cressie_hawkins':
#             estimates = self.compute_cressie_hawkins(
#                 bin_counts, bssad, min_pairs=self.MIN_PAIRS
#             )
#         else:
#             estimates = self.compute_matheron(
#                 bin_counts, bssd, min_pairs=self.MIN_PAIRS
#             )

#         result = (bin_counts, estimates, n_bins, min_distance, max_distance, max_lag)
#         if return_sample:
#             result = result + (
#                 self.raster_data_handler.coords.copy(),
#                 self.raster_data_handler.samples.copy(),
#             )
#         return result

#     def calculate_mean_variogram_numba(
#         self,
#         area_side: float,
#         samples_per_area: float,
#         max_samples: int,
#         bin_width: float,
#         max_n_bins: int,
#         n_runs: int,
#         max_lag_multiplier=1 / 3,
#         *,
#         seed: Optional[int] = None,
#         estimator: str = 'matheron',
#     ) -> None:
#         """
#         Run multiple variogram realizations and compute the mean semivariogram
#         and spread across runs.

#         Parameters
#         ----------
#         area_side, samples_per_area, max_samples : see numba_variogram
#         bin_width : float
#         max_n_bins : int
#         n_runs : int
#         max_lag_multiplier : {"max","median"} or float
#         seed : int | None
#             Base seed; each run uses a child seed for reproducibility.
#         estimator : {'matheron', 'cressie_hawkins'}
#             Empirical variogram estimator to use.  ``'matheron'`` is the
#             classical method-of-moments estimator.  ``'cressie_hawkins'``
#             is a robust estimator that downweights outlying squared
#             differences (Cressie & Hawkins, 1980; Cressie, 1985).
#         """
#         # child seeds for each run to keep realizations independent but reproducible.
#         ss = np.random.SeedSequence(seed)
#         child_seeds = ss.spawn(n_runs)

#         all_variograms = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
#         counts = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
#         all_n_bins = np.zeros(n_runs, dtype=int)

#         for run in range(n_runs):
#             count, variogram, n_bins, _, _, _ = self.numba_variogram(
#                 area_side, samples_per_area, max_samples, bin_width, max_lag_multiplier,
#                 seed=int(child_seeds[run].generate_state(1)[0]),
#                 estimator=estimator,
#             )
#             all_variograms.loc[run, :variogram.size - 1] = variogram
#             counts.loc[run, :count.size - 1] = count
#             all_n_bins[run] = n_bins

#         vario_arr = all_variograms.values
#         count_arr = counts.values

#         with np.errstate(all='ignore'):
#             mean_variogram = np.nanmean(vario_arr, axis=0)
#             # use robust spread visualization width; stored as err_variogram
#             err_variogram = (np.nanpercentile(vario_arr, 97.5, axis=0) -
#                              np.nanpercentile(vario_arr, 2.5, axis=0)) / 2.0
#             mean_count = np.nanmean(count_arr, axis=0)
#             sigma_variogram = np.nanstd(vario_arr, axis=0)

        
#         sigma_filtered = sigma_variogram.copy()
#         positive_sigma = sigma_filtered[sigma_filtered > 0]
#         if len(positive_sigma) > 0:
#             sigma_floor = float(np.median(positive_sigma) * 0.1)
#         else:
#             sigma_floor = float(np.finfo(float).eps)
#         sigma_filtered[sigma_filtered <= 0] = sigma_floor

#         valid = ~np.isnan(mean_variogram)
#         self.mean_variogram = mean_variogram[valid]
#         self.err_variogram = err_variogram[valid]
#         self.mean_count = mean_count[valid]
#         self.sigma_variogram = sigma_filtered[valid]

#         n_kept = self.mean_variogram.size
#         self.lags = np.linspace(bin_width / 2, bin_width * n_kept - bin_width / 2, n_kept)

#         self.all_variograms = vario_arr
#         self.all_counts = count_arr
#         self.n_bins = int(np.nanmean(all_n_bins))
#         self.estimator = estimator

#     def compute_empirical_variogram(
#         self,
#         area_side: float,
#         samples_per_area: float,
#         max_samples: int,
#         bin_width: float,
#         max_n_bins: int,
#         n_runs: int = 10,
#         max_lag_multiplier: float = 1 / 3,
#         *,
#         seed: Optional[int] = None,
#         estimator: str = 'matheron',
#     ) -> 'EmpiricalVariogram':
#         """Compute an empirical variogram and return it as a dataclass.

#         This is the recommended entry point for computing an empirical
#         variogram.  Unlike ``calculate_mean_variogram_numba()``, it
#         returns an :class:`EmpiricalVariogram` object that bundles all
#         results and provides a ``plot()`` method.  No fitting is
#         performed — call :meth:`fit_best_model_auto` separately if
#         needed.

#         Internally this calls ``calculate_mean_variogram_numba()`` to
#         populate the legacy attributes, so downstream code that reads
#         ``self.mean_variogram`` etc. continues to work.

#         Parameters
#         ----------
#         area_side : float
#             Side length of the random sampling square (map units).
#         samples_per_area : float
#             Sampling density (points per area_side²).
#         max_samples : int
#             Hard cap on total sample points per realization.
#         bin_width : float
#             Lag bin width (same units as coordinates).
#         max_n_bins : int
#             Maximum number of lag bins.
#         n_runs : int, default 10
#             Number of independent variogram realizations to average.
#             More runs → smoother mean and tighter spread estimate.
#         max_lag_multiplier : float, default 1/3
#             Maximum lag as fraction of the sampling extent.
#         seed : int, optional
#             Base seed for reproducibility.
#         estimator : {'matheron', 'cressie_hawkins'}, default 'matheron'
#             ``'matheron'`` — classical method-of-moments:
#             γ̂(h) = (1/2N) Σ [Z(xᵢ) − Z(xⱼ)]².
#             ``'cressie_hawkins'`` — robust fourth-root estimator
#             (Cressie & Hawkins, 1980) that downweights outlier
#             pairs.

#         Returns
#         -------
#         EmpiricalVariogram
#             Dataclass with ``lags``, ``mean_variogram``,
#             ``pair_counts``, ``sigma``, ``err_low``, ``err_high``,
#             plus a ``plot()`` method.

#         Examples
#         --------
#         >>> va = VariogramAnalysis(rdh)
#         >>> emp = va.compute_empirical_variogram(
#         ...     area_side=1000, samples_per_area=1.0,
#         ...     max_samples=10000, bin_width=50, max_n_bins=40,
#         ...     n_runs=20, estimator='matheron',
#         ... )
#         >>> emp.plot()          # quick look — no fitting needed
#         >>> emp.mean_variogram  # access the raw array
#         """
#         # Delegate to legacy method (populates self.mean_variogram etc.)
#         self.calculate_mean_variogram_numba(
#             area_side=area_side,
#             samples_per_area=samples_per_area,
#             max_samples=max_samples,
#             bin_width=bin_width,
#             max_n_bins=max_n_bins,
#             n_runs=n_runs,
#             max_lag_multiplier=max_lag_multiplier,
#             seed=seed,
#             estimator=estimator,
#         )

#         # Compute percentile bounds for the EmpiricalVariogram
#         with np.errstate(all='ignore'):
#             valid = ~np.isnan(np.nanmean(self.all_variograms, axis=0))
#             p2_5 = np.nanpercentile(self.all_variograms, 2.5, axis=0)[valid]
#             p16 = np.nanpercentile(self.all_variograms, 16, axis=0)[valid]
#             median = np.nanmedian(self.all_variograms, axis=0)[valid]
#             p84 = np.nanpercentile(self.all_variograms, 84, axis=0)[valid]
#             p97_5 = np.nanpercentile(self.all_variograms, 97.5, axis=0)[valid]

#         rdh = self.raster_data_handler
#         return EmpiricalVariogram(
#             lags=self.lags.copy(),
#             median_variogram=median,
#             mean_variogram=self.mean_variogram.copy(),
#             pair_counts=self.mean_count.copy(),
#             sigma=self.sigma_variogram.copy(),
#             p2_5=p2_5,
#             p16=p16,
#             p84=p84,
#             p97_5=p97_5,
#             n_runs=n_runs,
#             n_bins=self.n_bins,
#             estimator=estimator,
#             sample_coords=rdh.coords.copy() if rdh.coords is not None else None,
#             sample_values=rdh.samples.copy() if rdh.samples is not None else None,
#             all_variograms=self.all_variograms.copy(),
#             all_counts=self.all_counts.copy(),
#         )

#     def fit(
#         self,
#         # ── sampling / empirical variogram parameters ──
#         area_side: float,
#         samples_per_area: float,
#         max_samples: int,
#         bin_width: float,
#         max_n_bins: int,
#         n_runs: int = 10,
#         max_lag_multiplier: float = 1 / 3,
#         estimator: str = 'matheron',
#         # ── model fitting parameters ──
#         model_types: Optional[List[str]] = None,
#         max_components: int = 1,
#         include_nugget: bool = True,
#         criterion: str = 'msspe',
#         n_bootstrap: int = 500,
#         min_pairs: Optional[int] = 30,
#         msspe_n_subset: int = 500,
#         msspe_n_runs: int = 10,
#         msspe_prefilter: int = 0,
#         *,
#         seed: Optional[int] = None,
#     ) -> Tuple['EmpiricalVariogram', 'FittedVariogramModel']:
#         """Compute empirical variogram, fit models, and select the best.

#         Single entry point that chains
#         :meth:`compute_empirical_variogram` →
#         :meth:`fit_best_model_auto`.  After calling, use
#         ``va.plot_best_model()`` to visualise the result.

#         Parameters
#         ----------
#         area_side : float
#             Side length of the random sampling square (map units).
#         samples_per_area : float
#             Sampling density (points per area_side²).
#         max_samples : int
#             Hard cap on total sample points per realization.
#         bin_width : float
#             Lag bin width (same units as coordinates).
#         max_n_bins : int
#             Maximum number of lag bins.
#         n_runs : int, default 10
#             Independent variogram realizations to average.
#         max_lag_multiplier : float, default 1/3
#             Maximum lag as fraction of sampling extent.
#         estimator : {'matheron', 'cressie_hawkins'}, default 'matheron'
#             Empirical variogram estimator.
#         model_types : list of str, optional
#             Model types to try (default: spherical, exponential,
#             matern).
#         max_components : int, default 1
#             Max nested structures per candidate.
#         include_nugget : bool, default True
#             Generate with/without nugget variants.
#         criterion : {'aic', 'bic', 'msspe'}, default 'msspe'
#             Model selection criterion.
#         n_bootstrap : int, default 500
#             Bootstrap resamples for parameter uncertainty.
#         min_pairs : int or None, default 30
#             Minimum pair count per lag bin.
#         msspe_n_subset : int, default 500
#             Points per LOOCV evaluation.
#         msspe_n_runs : int, default 10
#             Independent LOOCV repetitions.
#         msspe_prefilter : int, default 0
#             If > 0, evaluate MSSPE on top-N AIC models only.
#         seed : int, optional
#             Base seed for reproducibility.

#         Returns
#         -------
#         (EmpiricalVariogram, FittedVariogramModel)
#             The empirical variogram and the best fitted model.

#         Examples
#         --------
#         >>> va = VariogramAnalysis(rdh)
#         >>> emp, fitted = va.fit(
#         ...     area_side=1000, samples_per_area=1.0,
#         ...     max_samples=10000, bin_width=50, max_n_bins=40,
#         ...     n_runs=20, criterion='msspe',
#         ... )
#         >>> va.plot_best_model()       # combined plot
#         >>> emp.plot()                  # empirical only
#         >>> fitted.composite_model     # the winning model
#         """
#         emp = self.compute_empirical_variogram(
#             area_side=area_side,
#             samples_per_area=samples_per_area,
#             max_samples=max_samples,
#             bin_width=bin_width,
#             max_n_bins=max_n_bins,
#             n_runs=n_runs,
#             max_lag_multiplier=max_lag_multiplier,
#             seed=seed,
#             estimator=estimator,
#         )

#         fitted = self.fit_best_model_auto(
#             model_types=model_types,
#             max_components=max_components,
#             include_nugget=include_nugget,
#             criterion=criterion,
#             n_bootstrap=n_bootstrap,
#             seed=seed,
#             min_pairs=min_pairs,
#             compute_msspe=(criterion == 'msspe'),
#             msspe_n_subset=msspe_n_subset,
#             msspe_n_runs=msspe_n_runs,
#             msspe_prefilter=msspe_prefilter,
#         )

#         return emp, fitted

#     def fit_best_model_auto(
#         self,
#         model_types: Optional[List[str]] = None,
#         max_components: int = 1,
#         include_nugget: bool = True,
#         criterion: str = 'msspe',
#         n_bootstrap: int = 500,
#         seed: Optional[int] = None,
#         min_pairs: Optional[int] = 30,
#         compute_msspe: bool = False,
#         msspe_n_subset: int = 500,
#         msspe_n_runs: int = 10,
#         msspe_prefilter: int = 0,
#     ) -> 'FittedVariogramModel':
#         """
#         Fit multiple variogram model types and automatically select the best.

#         Parameters
#         ----------
#         model_types : list of str, optional
#             Model types to consider. Default: ['spherical', 'exponential',
#             'matern'].
#         max_components : int
#             Maximum number of nested components (default: 1, max: 3).
#         include_nugget : bool
#             Whether to include nugget effect in all candidate models.
#         criterion : {'aic', 'bic', 'msspe'}
#             Selection criterion.  Default is ``'msspe'``, which selects
#             the model whose kriging LOOCV MSSPE is closest to 1.0
#             (|MSSPE − 1| minimised).  This directly evaluates whether
#             the kriging variance matches actual prediction errors,
#             which is what matters for uncertainty propagation.

#             AIC/BIC treat variogram lag bins as independent observations,
#             which they are not — adjacent bins share point pairs and are
#             correlated.  This systematically favours complex models.
#             MSSPE avoids this by evaluating spatial prediction
#             calibration directly (Lark, 2000).

#             Requires spatial data from ``sample_raster()`` or
#             ``calculate_mean_variogram_numba()``.
#         n_bootstrap : int
#             Number of bootstrap resamples for parameter uncertainty.
#         seed : int, optional
#             Random seed.
#         min_pairs : int or None, default 30
#             Minimum pair count per lag bin.  Bins with fewer pairs are
#             excluded from fitting.  Set to ``None`` to disable.
#         compute_msspe : bool
#             Whether to compute kriging LOOCV MSSPE for each candidate
#             model.  Automatically set to True when ``criterion='msspe'``.
#         msspe_n_subset : int
#             Number of points per kriging LOOCV run (default 500).
#             Runtime is O(n²) per model per run, so ~1 s per model at
#             n=500.
#         msspe_n_runs : int
#             Number of independent random subsamples over which to
#             average the MSSPE (default 10).  A single subsample
#             introduces sampling variability; averaging over multiple
#             runs produces a more stable estimate — analogous to
#             repeated k-fold CV versus a single train/test split.
#             Total LOOCV effort is ``msspe_n_runs × msspe_n_subset``
#             points per candidate model.
#         msspe_prefilter : int
#             If > 0, compute MSSPE only for the top-N models ranked by
#             AIC, plus the AIC-best model from each complexity level
#             (number of components).  The stratified inclusion ensures
#             that simpler model families are always represented, even
#             when complex models dominate the AIC ranking.  Saves time
#             when many candidates are fitted.  If 0 (default), all
#             candidates are evaluated.

#         Returns
#         -------
#         FittedVariogramModel
#             Best fitted model with diagnostics.

#         Notes
#         -----
#         This method is model-agnostic for uncertainty propagation because
#         Monte Carlo integration works for ANY valid variogram function
#         (Krige's Relation — Chilès & Delfiner, 2012, Chapter 4).

#         The ``'msspe'`` criterion evaluates spatial prediction calibration
#         directly, rather than variogram curve-fitting quality.  MSSPE ≈ 1.0
#         indicates that the kriging variance correctly matches actual
#         prediction errors (Lark, 2000).  This is preferred when the goal
#         is uncertainty propagation, because AIC/BIC can favour complex
#         models that overfit the empirical variogram while producing
#         miscalibrated kriging variances.

#         References
#         ----------
#         Lark, R.M. (2000).  A comparison of some robust estimators of
#         the variogram for use in soil survey.  *Eur. J. Soil Sci.*, 51,
#         137–157.
#         """
#         if self.mean_variogram is None:
#             raise RuntimeError("No variogram data. Call calculate_mean_variogram_numba() first.")

#         # implicitly enable MSSPE computation when criterion requires it
#         if criterion == 'msspe':
#             compute_msspe = True

#         if model_types is None:
#             model_types = ['spherical', 'exponential', 'matern']

#         # validate model_types
#         available = MODEL_REGISTRY.list_models()
#         for mt in model_types:
#             if mt not in available:
#                 raise ValueError(f"Unknown model type '{mt}'. Available: {available}")

#         # validate spatial data availability for MSSPE
#         if compute_msspe:
#             rdh = self.raster_data_handler
#             if rdh.coords is None or rdh.samples is None:
#                 raise RuntimeError(
#                     "Kriging LOOCV MSSPE requires spatial data. "
#                     "Call sample_raster() or calculate_mean_variogram_numba() first."
#                 )

#         # create selector
#         selector = VariogramModelSelector(
#             lags=self.lags,
#             empirical_variogram=self.mean_variogram,
#             pair_counts=self.mean_count,
#             sigma=self.sigma_variogram,
#             min_pairs=min_pairs,
#         )

#         # override model lists with user selection
#         selector.BOUNDED_MODELS = [m for m in model_types if MODEL_REGISTRY.is_bounded(m)]
#         selector.UNBOUNDED_MODELS = [m for m in model_types if not MODEL_REGISTRY.is_bounded(m)]

#         # fit all candidates
#         selector.fit_all_candidates(
#             max_components=min(max_components, 3),
#             include_nugget=include_nugget,
#             include_unbounded=bool(selector.UNBOUNDED_MODELS),
#         )

#         if not selector.fitted_models:
#             raise RuntimeError("No models successfully fitted. Check input data.")

#         # ── compute kriging LOOCV MSSPE for candidate models ──
#         if compute_msspe:
#             rdh = self.raster_data_handler
#             coords = rdh.coords
#             values = rdh.samples

#             # determine which models to evaluate
#             if msspe_prefilter > 0 and len(selector.fitted_models) > msspe_prefilter:
#                 # Stratified prefilter: take top-N by AIC, but also
#                 # ensure the AIC-best model from each complexity level
#                 # (number of components) is included.  Without this,
#                 # prefiltering can exclude entire model families — e.g.
#                 # all 1-component models when 3-component models
#                 # dominate the AIC ranking — and miss the model with
#                 # the best MSSPE.
#                 ranked_idx = np.argsort([m.aic for m in selector.fitted_models])
#                 eval_idx = set(ranked_idx[:msspe_prefilter].tolist())

#                 # add best-AIC model per complexity level
#                 best_per_complexity: Dict[int, int] = {}
#                 for i in ranked_idx:
#                     m = selector.fitted_models[i]
#                     n_comp = len(m.composite_model.component_names)
#                     if n_comp not in best_per_complexity:
#                         best_per_complexity[n_comp] = i
#                 eval_idx.update(best_per_complexity.values())
#             else:
#                 eval_idx = set(range(len(selector.fitted_models)))

#             # ── precompute shared distance matrix + subsample ──
#             # The distance matrix is the same for all candidate models;
#             # only γ(dist) changes.  Precomputing once and passing it
#             # through avoids redundant O(n²) distance calculations.
#             run_rng = np.random.default_rng(seed)
#             run_seeds = run_rng.integers(0, 2**31, size=msspe_n_runs)

#             # Pre-draw subsample indices (shared across all models
#             # within each run for fair comparison).
#             n_pts = len(values)
#             subsample_indices = []
#             for rs in run_seeds:
#                 sub_rng = np.random.default_rng(int(rs))
#                 if n_pts > msspe_n_subset:
#                     idx = sub_rng.choice(n_pts, msspe_n_subset, replace=False)
#                 else:
#                     idx = np.arange(n_pts)
#                 subsample_indices.append(idx)

#             # Precompute distance matrices for each subsample
#             precomputed_runs = []
#             for idx in subsample_indices:
#                 sub_coords = coords[idx]
#                 sub_values = values[idx]
#                 dx = sub_coords[:, 0:1] - sub_coords[:, 0:1].T
#                 dy = sub_coords[:, 1:2] - sub_coords[:, 1:2].T
#                 sub_dist = np.sqrt(dx**2 + dy**2)
#                 precomputed_runs.append((sub_coords, sub_values, sub_dist))

#             for i, fitted in enumerate(selector.fitted_models):
#                 if i not in eval_idx:
#                     continue
#                 # only evaluate stationary models (non-stationary have
#                 # infinite variance → kriging is undefined)
#                 if not fitted.composite_model.is_stationary:
#                     continue

#                 run_results: list[KrigingLOOCVResult] = []
#                 for sub_coords, sub_values, sub_dist in precomputed_runs:
#                     try:
#                         result = self.kriging_loocv(
#                             sub_coords, sub_values,
#                             fitted.composite_model,
#                             n_subset=len(sub_values),
#                             dist_matrix=sub_dist,
#                         )
#                         if np.isfinite(result.msspe):
#                             run_results.append(result)
#                     except Exception as exc:
#                         model_name = "+".join(fitted.composite_model.component_names)
#                         warnings.warn(
#                             f"LOOCV failed for {model_name}: "
#                             f"{type(exc).__name__}: {exc}",
#                             stacklevel=2,
#                         )
#                         continue

#                 if run_results:
#                     agg = AggregatedLOOCVResult.from_results(run_results)
#                     fitted.msspe = agg.msspe_mean
#                     fitted.msspe_std = agg.msspe_std
#                     fitted.msspe_n_runs = agg.n_runs
#                     fitted.loocv_result = agg
#                 else:
#                     fitted.msspe = None
#                     fitted.msspe_std = None
#                     fitted.msspe_n_runs = 0
#                     fitted.loocv_result = None

#         # select best model
#         best = selector.select_best(criterion=criterion)

#         # bootstrap parameter uncertainty
#         if n_bootstrap > 0:
#             selector.bootstrap_best_model(n_boot=n_bootstrap, seed=seed)

#         # store results for compatibility
#         self._store_fitted_model_results(best, selector)
#         self.fitted_model = best
#         self.model_selector = selector

#         return best
    
#     def _store_fitted_model_results(
#         self,
#         fitted: 'FittedVariogramModel',
#         selector: 'VariogramModelSelector'
#     ) -> None:
#         """Transfer FittedVariogramModel results to VariogramAnalysis attributes.

#         Ensures backward compatibility with code expecting traditional attributes.
#         Stores both full range (2.5th/97.5th) and 1σ range (16th/84th) percentiles.
#         """
#         model = fitted.composite_model
#         params = fitted.params

#         # extract sills, ranges from composite model
#         # note: 'wavelength' (damped_hole_effect) is treated as a range-like parameter
#         sills = []
#         ranges = []
#         sill_indices = []
#         range_indices = []
#         range_labels = []  # Track whether each range is 'range' or 'wavelength'
#         param_offset = 0

#         for i, spec in enumerate(model._components):
#             comp_params = model.get_component_params(i)
#             if spec.has_sill:
#                 sills.append(comp_params[0])
#                 sill_indices.append(param_offset)
#             for range_key in ('range', 'wavelength'):
#                 if range_key in spec.param_names:
#                     range_idx = spec.param_names.index(range_key)
#                     ranges.append(comp_params[range_idx])
#                     range_indices.append(param_offset + range_idx)
#                     range_labels.append(range_key)
#                     break
#             param_offset += len(spec.param_names)

#         self.sills = np.array(sills) if sills else np.array([])
#         self.ranges = np.array(ranges) if ranges else np.array([])
#         self.range_labels = range_labels  # 'range' or 'wavelength' per entry
#         self.best_nugget = model.get_nugget() if model.include_nugget else None
#         self.best_params = params
#         self.best_aic = fitted.aic
#         self.best_bic = fitted.bic

#         # callable for the model function
#         self.best_model_func = lambda h, *p: model(np.asarray(h, dtype=float))
#         self.fitted_variogram = model(self.lags)

#         self.best_model_config = {
#             'components': len(model.component_names),
#             'nugget': model.include_nugget,
#             'model_types': model.component_names,
#         }

#         # Store bootstrap samples without appending the optimal point
#         # estimate — appending it biases percentile estimates.
#         if fitted.param_samples is not None and len(fitted.param_samples) > 0:
#             self.param_samples = fitted.param_samples
#         else:
#             self.param_samples = np.array([params])

#         if self.param_samples.size > 0:
#             samples = self.param_samples

#             # extract sill percentiles
#             if sill_indices:
#                 sill_samps = samples[:, sill_indices]
#                 # full range (2.5th to 97.5th)
#                 self.sills_min = np.percentile(sill_samps, 2.5, axis=0)
#                 self.sills_max = np.percentile(sill_samps, 97.5, axis=0)
#                 self.sills_median = np.percentile(sill_samps, 50, axis=0)
#                 # 1σ range (16th to 84th)
#                 self.sills_p16 = np.percentile(sill_samps, 16, axis=0)
#                 self.sills_p84 = np.percentile(sill_samps, 84, axis=0)
#             else:
#                 self.sills_min = self.sills_max = self.sills_median = np.array([])
#                 self.sills_p16 = self.sills_p84 = np.array([])

#             # extract range percentiles
#             if range_indices:
#                 range_samps = samples[:, range_indices]
#                 # full range (2.5th to 97.5th)
#                 self.ranges_min = np.percentile(range_samps, 2.5, axis=0)
#                 self.ranges_max = np.percentile(range_samps, 97.5, axis=0)
#                 self.ranges_median = np.percentile(range_samps, 50, axis=0)
#                 # 1σ range (16th to 84th)
#                 self.ranges_p16 = np.percentile(range_samps, 16, axis=0)
#                 self.ranges_p84 = np.percentile(range_samps, 84, axis=0)
#             else:
#                 self.ranges_min = self.ranges_max = self.ranges_median = np.array([])
#                 self.ranges_p16 = self.ranges_p84 = np.array([])

#             # extract nugget percentiles
#             if model.include_nugget:
#                 nugget_idx = model.n_params - 1  # Nugget is always last
#                 nug_samps = samples[:, nugget_idx]
#                 # full range
#                 self.min_nugget = float(np.percentile(nug_samps, 2.5))
#                 self.max_nugget = float(np.percentile(nug_samps, 97.5))
#                 self.median_nugget = float(np.percentile(nug_samps, 50))
#                 # 1σ range
#                 self.nugget_p16 = float(np.percentile(nug_samps, 16))
#                 self.nugget_p84 = float(np.percentile(nug_samps, 84))
#             else:
#                 self.min_nugget = self.max_nugget = self.median_nugget = None
#                 self.nugget_p16 = self.nugget_p84 = None
#         else:
#             # fallback to point estimates when no bootstrap samples
#             self.sills_min = self.sills_max = self.sills_median = self.sills
#             self.sills_p16 = self.sills_p84 = self.sills
#             self.ranges_min = self.ranges_max = self.ranges_median = self.ranges
#             self.ranges_p16 = self.ranges_p84 = self.ranges
#             if model.include_nugget:
#                 self.min_nugget = self.max_nugget = self.median_nugget = self.best_nugget
#                 self.nugget_p16 = self.nugget_p84 = self.best_nugget
#             else:
#                 self.min_nugget = self.max_nugget = self.median_nugget = None
#                 self.nugget_p16 = self.nugget_p84 = None

#         self.cv_mean_error_best_aic = None

#         # transfer MSSPE diagnostics
#         self.best_msspe = fitted.msspe
#         self.best_loocv_result = fitted.loocv_result

#     def get_model_comparison_summary(self) -> str:
#         """Get a summary of all fitted models for comparison.

#         Returns formatted table with AIC, BIC, and MSSPE.
#         MSSPE column is included when at least one candidate
#         has been evaluated with kriging LOOCV.
#         """
#         if not hasattr(self, 'model_selector') or self.model_selector is None:
#             raise RuntimeError("No model comparison available. Call fit_best_model_auto() first.")

#         selector = self.model_selector

#         # check if any model has MSSPE computed
#         has_msspe = any(
#             m.msspe is not None for m in selector.fitted_models
#         )

#         lines = ["=" * 80]
#         lines.append("VARIOGRAM MODEL SELECTION SUMMARY")
#         lines.append("=" * 80)
#         header = f"{'Model':<40} {'AIC':>10} {'BIC':>10}"
#         if has_msspe:
#             header += f" {'MSSPE':>12}"
#         lines.append(header)
#         lines.append("-" * 80)

#         sorted_models = sorted(selector.fitted_models, key=lambda x: x.aic)

#         for model in sorted_models:
#             name = "+".join(model.composite_model.component_names)
#             if model.composite_model.include_nugget:
#                 name += "+nugget"
#             marker = " *" if model is selector.best_model else ""
#             row = f"{name:<40} {model.aic:>10.2f} {model.bic:>10.2f}"
#             if has_msspe:
#                 if model.msspe is not None:
#                     msspe_str = f"{model.msspe:.3f}"
#                     std = getattr(model, 'msspe_std', None)
#                     if std is not None:
#                         msspe_str += f"±{std:.3f}"
#                 else:
#                     msspe_str = "N/A"
#                 row += f" {msspe_str:>12}"
#             row += marker
#             lines.append(row)

#         lines.append("=" * 80)
#         return "\n".join(lines)
    
# # ── kriging-based cross-validation ──────────────────────────────

#     @staticmethod
#     def kriging_loocv(
#         coords: np.ndarray,
#         values: np.ndarray,
#         variogram_func: Callable,
#         n_subset: int = 2000,
#         seed: Optional[int] = None,
#         dist_matrix: Optional[np.ndarray] = None,
#     ) -> 'KrigingLOOCVResult':
#         """Leave-one-out cross-validation via Dubrule (1983) shortcut.

#         Instead of solving *n* separate (n-1)-point kriging systems
#         [O(n**4)], inverts the full n-point augmented system **once**
#         and extracts all LOO predictions from the diagonal of the
#         inverse [O(n**3)].  This is the kriging analogue of the hat-
#         matrix identity in linear regression:

#             e_{-i} = -(Q[i,:n] . z) / Q[i,i]
#             sigma2_{-i} = 1 / Q[i,i]

#         where Q = A**-1 and A is the augmented semivariance matrix.

#         Parameters
#         ----------
#         coords : ndarray, shape (n, 2)
#             Spatial coordinates of observed locations.
#         values : ndarray, shape (n,)
#             Observed values (e.g. elevation differences).
#         variogram_func : callable
#             Fitted variogram model gamma(h).  Must accept an ndarray
#             of distances and return semivariances.  A
#             ``CompositeVariogramModel`` instance works directly.
#         n_subset : int, default 2000
#             Maximum number of points for the CV.  If ``len(values)``
#             exceeds this, a random subsample is drawn.  The Dubrule
#             shortcut makes n=2000 affordable (~3 s vs ~250 s for
#             brute-force LOO).
#         seed : int, optional
#             Random seed for reproducible subsampling.
#         dist_matrix : ndarray, shape (n, n), optional
#             Pre-computed pairwise distance matrix.  When supplied,
#             the distance computation is skipped -- useful when
#             evaluating multiple candidate models on the same point
#             set (e.g. inside ``fit_best_model_auto`` or
#             ``fit_variogram_ensemble``).  Must correspond to the
#             (possibly subsampled) ``coords``.

#         Returns
#         -------
#         KrigingLOOCVResult
#             Dataclass with ``msspe``, ``mean_error``, ``rmse``,
#             ``mean_standardized_error``, ``n_points``, ``n_failed``.

#         Notes
#         -----
#         **Dubrule shortcut** -- the ordinary kriging augmented
#         system::

#             A = | Gamma  1 |      (n+1) x (n+1)
#                 | 1'     0 |

#         Invert once to get Q = A**-1.  Then for every point *i*::

#             LOO error:     e_{-i} = -(Q[i,:n] . z) / Q[i,i]
#             LOO variance:  sigma2_{-i} = -1 / Q[i,i]   (Q[i,i] < 0)
#             SSPE_i:        e_{-i}**2 / sigma2_{-i}

#         Note: because A uses the semivariance matrix (not covariance),
#         Q[i,i] < 0 for valid variogram models.  The negative sign in
#         the variance formula ensures σ² > 0.

#         and MSSPE = mean(SSPE_i).

#         **Numerical stability** --

#         1. *Diagonal regularisation*: a small nugget-like term
#            ``eps * mean(diag(Gamma))`` is added to the diagonal of
#            Gamma before inversion.  This prevents singularity for
#            smooth variogram models (Gaussian, high-nu Matern) where
#            nearby points produce nearly identical semivariances
#            (Deutsch & Journel, 1998, p. 78).

#         2. *Pivoted LU factorisation*: ``np.linalg.solve(A, I)``
#            is used instead of ``np.linalg.inv(A)`` because LAPACK's
#            ``dgesv`` uses partial pivoting, which is more robust for
#            ill-conditioned systems.

#         3. *Condition number check*: if kappa(A) > 1e12 the matrix
#            is too ill-conditioned for reliable inversion and a
#            warning is issued.  Individual unstable points are
#            filtered by the diagonal check below.

#         4. *Diagonal filter*: LOO points where Q[i,i] >= 0 are
#            excluded (for the semivariance formulation, Q[i,i] < 0
#            is the valid condition; Q[i,i] >= 0 indicates local
#            numerical instability, typically < 1% of points).

#         References
#         ----------
#         Dubrule, O. (1983). Cross validation of kriging in a unique
#             neighborhood. *J. Int. Assoc. Math. Geol.*, 15(6),
#             687-699.

#         Deutsch, C.V. & Journel, A.G. (1998). *GSLIB*, 2nd ed.
#             Oxford University Press.  (diagonal regularisation,
#             p. 78)

#         Cressie, N. (1993). *Statistics for Spatial Data*, rev. ed.,
#             Wiley.  Section 5.6 -- kriging cross-validation.
#         """
#         coords = np.asarray(coords, dtype=float)
#         values = np.asarray(values, dtype=float)

#         if coords.ndim != 2 or coords.shape[1] != 2:
#             raise ValueError(
#                 f"coords must be shape (n, 2), got {coords.shape}"
#             )
#         if len(coords) != len(values):
#             raise ValueError(
#                 f"coords ({len(coords)}) and values ({len(values)}) "
#                 f"must have the same length."
#             )

#         n = len(values)
#         rng = np.random.default_rng(seed)

#         # -- subsample if too many points --
#         if n > n_subset:
#             idx = rng.choice(n, n_subset, replace=False)
#             coords = coords[idx]
#             values = values[idx]
#             if dist_matrix is not None:
#                 dist_matrix = dist_matrix[np.ix_(idx, idx)]
#             n = n_subset

#         _nan_result = KrigingLOOCVResult(
#             msspe=np.nan, mean_error=np.nan, rmse=np.nan,
#             mean_standardized_error=np.nan,
#             n_points=0, n_failed=n,
#         )

#         # -- pairwise distance matrix (n x n) --
#         if dist_matrix is None:
#             dx = coords[:, 0:1] - coords[:, 0:1].T
#             dy = coords[:, 1:2] - coords[:, 1:2].T
#             dist_matrix = np.sqrt(dx**2 + dy**2)

#         # -- semivariance matrix (n x n) --
#         gamma_matrix = variogram_func(dist_matrix)

#         # -- diagonal regularisation --
#         # A small nugget-like term prevents singularity for smooth
#         # variogram models (Gaussian, high-nu Matern) where nearby
#         # points produce nearly identical semivariances.
#         # Scale: 1e-10 * mean diagonal -- large enough to
#         # regularise, small enough not to bias LOO diagnostics.
#         diag_mean = np.mean(np.diag(gamma_matrix))
#         if diag_mean > 0:
#             eps_reg = 1e-10 * diag_mean
#         else:
#             # pure nugget or zero-distance diagonal -- use absolute
#             # fallback scaled to overall semivariance magnitude
#             eps_reg = 1e-10 * (
#                 np.mean(gamma_matrix) + np.finfo(float).eps
#             )
#         gamma_reg = gamma_matrix + eps_reg * np.eye(n)

#         # -- build augmented kriging matrix --
#         m = n + 1
#         A = np.empty((m, m))
#         A[:n, :n] = gamma_reg
#         A[:n, n] = 1.0
#         A[n, :n] = 1.0
#         A[n, n] = 0.0

#         # -- condition number check --
#         # If the matrix is too ill-conditioned, the LOO diagnostics
#         # from the inverse will be numerically meaningless.
#         try:
#             cond = np.linalg.cond(A)
#         except np.linalg.LinAlgError:
#             return _nan_result
#         if cond > 1e12:
#             warnings.warn(
#                 f"Kriging system condition number {cond:.2e} exceeds "
#                 f"1e12 -- LOO diagnostics may be unreliable. "
#                 f"Consider increasing the nugget or reducing "
#                 f"n_subset.",
#                 stacklevel=2,
#             )
#             # Still attempt the solve; the diagonal filter below
#             # will catch individual unstable points.

#         # -- Dubrule shortcut: single matrix "inversion" --
#         # Use solve(A, I) instead of inv(A) for better numerical
#         # stability -- LAPACK dgesv uses partial pivoting.
#         try:
#             Q = np.linalg.solve(A, np.eye(m))
#         except np.linalg.LinAlgError:
#             return _nan_result

#         # -- extract LOO diagnostics from Q --
#         # Q_diag[i] = Q[i,i] for the n data points (not the
#         #             Lagrange multiplier row).
#         # LOO error:     e_{-i} = -(Q[i,:n] . z) / Q[i,i]
#         # LOO variance:  sigma2_{-i} = -1 / Q[i,i]
#         #
#         # Sign convention: because A is built from the *semivariance*
#         # matrix Γ (not the covariance matrix C), the diagonal of
#         # Q = A⁻¹ is negative for valid variogram models.  The LOO
#         # kriging variance is σ²_{-i} = −1/Q[i,i], which is positive
#         # when Q[i,i] < 0.  The LOO error formula is unchanged
#         # because the sign cancels: e_{-i} = −(Q[i,:n]·z)/Q[i,i].
#         Q_diag = np.diag(Q)[:n]
#         Qz = Q[:n, :n] @ values          # shape (n,)

#         # Filter out numerically unstable points (Q[i,i] >= 0
#         # means the LOO variance would be non-positive).
#         valid = Q_diag < 0
#         n_failed = int(np.sum(~valid))

#         if not np.any(valid):
#             return _nan_result

#         Qz_v = Qz[valid]
#         Q_diag_v = Q_diag[valid]

#         errors = -Qz_v / Q_diag_v        # LOO prediction errors
#         variances = -1.0 / Q_diag_v      # LOO kriging variances
#         sigma = np.sqrt(variances)
#         n_valid = int(np.sum(valid))

#         return KrigingLOOCVResult(
#             msspe=float(np.mean(errors**2 / variances)),
#             mean_error=float(np.mean(errors)),
#             rmse=float(np.sqrt(np.mean(errors**2))),
#             mean_standardized_error=float(np.mean(errors / sigma)),
#             n_points=n_valid,
#             n_failed=n_failed,
#         )

#     def fit_ensemble(
#         self,
#         # ── sampling parameters ──
#         area_side: float,
#         samples_per_area: float,
#         max_samples: int,
#         bin_width: float,
#         max_lag_multiplier: float = 1 / 3,
#         estimator: str = 'matheron',
#         # ── ensemble parameters ──
#         n_realizations: int = 50,
#         model_types: Optional[List[str]] = None,
#         max_components: int = 1,
#         include_nugget: bool = True,
#         bounded_only_combinations: bool = True,
#         criterion: str = 'msspe',
#         msspe_n_subset: int = 2000,
#         min_pairs: Optional[int] = 30,
#         *,
#         seed: Optional[int] = None,
#         verbose: bool = True,
#     ) -> Tuple['EmpiricalVariogram', 'EnsembleVariogramResult']:
#         """Run ensemble fitting and derive the mean empirical variogram.

#         Each of the ``n_realizations`` independently samples the
#         raster, computes an empirical variogram, fits candidates,
#         and selects the best model.  The mean empirical variogram is
#         derived *from the ensemble's own realizations* — no separate
#         pre-computation step, no redundant ``n_variogram_runs``
#         parameter.

#         Parameters
#         ----------
#         area_side : float
#             Side length of the random sampling square (map units).
#         samples_per_area : float
#             Sampling density (points per area_side²).
#         max_samples : int
#             Hard cap on total sample points per realization.
#         bin_width : float
#             Lag bin width (same units as coordinates).
#         max_lag_multiplier : float, default 1/3
#             Maximum lag as fraction of sampling extent.
#         estimator : {'matheron', 'cressie_hawkins'}, default 'matheron'
#             Empirical variogram estimator.
#         n_realizations : int, default 50
#             Independent ensemble realizations.  Each one draws a
#             fresh sample, computes an empirical variogram, fits all
#             candidates, evaluates MSSPE, and selects the best model.
#         model_types : list of str, optional
#             Model types to try (default: spherical, exponential,
#             matern).
#         max_components : int, default 1
#             Max nested structures per candidate.
#         include_nugget : bool, default True
#             Generate with/without nugget variants.
#         bounded_only_combinations : bool, default True
#             Unbounded models only as single-component candidates.
#         criterion : {'aic', 'bic', 'msspe'}, default 'msspe'
#             Model selection criterion.
#         msspe_n_subset : int, default 500
#             Points per LOOCV evaluation.
#         min_pairs : int or None, default 30
#             Minimum pair count per lag bin.
#         seed : int, optional
#             Base seed for reproducibility.
#         verbose : bool, default True
#             Print progress.

#         Returns
#         -------
#         (EmpiricalVariogram, EnsembleVariogramResult)
#             The mean empirical variogram (derived from all
#             ensemble realizations) and the full ensemble result.

#         Examples
#         --------
#         >>> va = VariogramAnalysis(rdh)
#         >>> emp, ensemble = va.fit_ensemble(
#         ...     area_side=1000, samples_per_area=1.0,
#         ...     max_samples=10000, bin_width=50,
#         ...     n_realizations=50, criterion='msspe',
#         ... )
#         >>> emp.plot()              # empirical variogram only
#         >>> ensemble.plot()         # full ensemble results
#         >>> print(ensemble.summary())
#         """
#         ensemble = self.fit_variogram_ensemble(
#             n_realizations=n_realizations,
#             area_side=area_side,
#             samples_per_area=samples_per_area,
#             max_samples=max_samples,
#             bin_width=bin_width,
#             max_lag_multiplier=max_lag_multiplier,
#             model_types=model_types,
#             max_components=max_components,
#             include_nugget=include_nugget,
#             bounded_only_combinations=bounded_only_combinations,
#             criterion=criterion,
#             msspe_n_subset=msspe_n_subset,
#             estimator=estimator,
#             min_pairs=min_pairs,
#             seed=seed,
#             verbose=verbose,
#         )

#         # Derive the mean empirical variogram from the ensemble's
#         # own realizations — each realization already produced an
#         # independent empirical variogram, so n_realizations *is*
#         # the number of averaging runs.
#         emp_arr = ensemble.empirical_variograms  # (n_success, n_lags)
#         cnt_arr = ensemble.pair_counts            # (n_success, n_lags)
#         lags = ensemble.lags

#         with np.errstate(all='ignore'):
#             emp = EmpiricalVariogram(
#                 lags=lags.copy(),
#                 median_variogram=np.nanmedian(emp_arr, axis=0),
#                 mean_variogram=np.nanmean(emp_arr, axis=0),
#                 pair_counts=np.nanmean(cnt_arr, axis=0),
#                 sigma=np.nanstd(emp_arr, axis=0),
#                 p2_5=np.nanpercentile(emp_arr, 2.5, axis=0),
#                 p16=np.nanpercentile(emp_arr, 16, axis=0),
#                 p84=np.nanpercentile(emp_arr, 84, axis=0),
#                 p97_5=np.nanpercentile(emp_arr, 97.5, axis=0),
#                 n_runs=ensemble.n_realizations - ensemble.n_failed,
#                 n_bins=len(lags),
#                 estimator=estimator,
#                 all_variograms=emp_arr,
#                 all_counts=cnt_arr,
#             )

#         return emp, ensemble

#     def fit_variogram_ensemble(
#         self,
#         n_realizations: int = 50,
#         area_side: float = 1000,
#         samples_per_area: float = 1.0,
#         max_samples: int = 10000,
#         bin_width: float = 50.0,
#         max_lag_multiplier: float = 1 / 3,
#         *,
#         model_types: Optional[List[str]] = None,
#         max_components: int = 1,
#         include_nugget: bool = True,
#         bounded_only_combinations: bool = True,
#         criterion: str = 'msspe',
#         msspe_n_subset: int = 500,
#         estimator: str = 'matheron',
#         min_pairs: Optional[int] = 30,
#         seed: Optional[int] = None,
#         verbose: bool = True,
#     ) -> 'EnsembleVariogramResult':
#         """Fit variogram models to independent spatial samples — ensemble approach.

#         Generates ``n_realizations`` independent empirical variograms
#         (each from a fresh random sample of the raster), runs the full
#         model selection pipeline on each, and collects results.  This
#         captures both *model selection uncertainty* and *parameter
#         uncertainty* in a single Monte Carlo experiment.

#         The approach is analogous to a spatial bootstrap: instead of
#         perturbing a single empirical variogram, it re-samples the
#         underlying field.  Each realization goes through candidate
#         generation → WLS fitting → MSSPE-based selection independently,
#         so the ensemble naturally reflects how sensitive model choice
#         and parameter estimates are to the particular sample drawn.

#         Parameters
#         ----------
#         n_realizations : int
#             Number of independent variogram realizations (default 50).
#         area_side : float
#             Reference side length for sampling density (e.g. 1000 for
#             km² if coordinates are in meters).
#         samples_per_area : float
#             Sampling density per reference area unit.
#         max_samples : int
#             Maximum points per realization.
#         bin_width : float
#             Lag bin width (same units as coordinates).
#         max_lag_multiplier : float
#             Maximum lag as fraction of domain diagonal.
#         model_types : list of str, optional
#             Model types to try (default: spherical, exponential,
#             matern).
#         max_components : int
#             Maximum number of nested components (default 1, max 3).
#         include_nugget : bool
#             Whether to include nugget variants in candidates.
#         bounded_only_combinations : bool
#             If True (default), unbounded models (linear, power) are
#             only tried as single-component models.  Set to False to
#             allow bounded + unbounded combinations (e.g. spherical +
#             linear + nugget) for variograms that don't reach a sill.
#         criterion : str
#             Selection criterion (default 'msspe').
#         msspe_n_subset : int
#             Points per MSSPE LOOCV evaluation.  A single LOOCV run
#             is performed per candidate per realization (using the
#             same spatial sample that produced the empirical variogram).
#             The outer ensemble loop provides variance estimation
#             across independent spatial draws, making inner repetitions
#             redundant.
#         estimator : str
#             Empirical variogram estimator ('matheron' or 'cressie_hawkins').
#         min_pairs : int or None
#             Minimum pair count per lag bin for inclusion.
#         seed : int, optional
#             Base random seed for reproducibility.
#         verbose : bool
#             Print progress (default True).

#         Returns
#         -------
#         EnsembleVariogramResult
#             Dataclass with per-realization parameters, model selection
#             frequencies, and aggregate statistics.  Call ``.summary()``
#             for a text report or ``.plot()`` for a multi-panel figure.

#         Notes
#         -----
#         Each realization uses the *same* spatial sample for both the
#         empirical variogram and the MSSPE evaluation, ensuring
#         coherence.  The distance matrix for the LOOCV subsample is
#         computed once per realization and shared across all candidate
#         models; the virtual LOO identity (Dubrule, 1983) reduces
#         each evaluation to a single matrix inversion rather than *n*
#         separate linear solves.

#         With these optimisations, runtime is dominated by the WLS
#         fitting step rather than LOOCV.  Typical runtime: ~1–5 min
#         for 50 realizations with 6 candidates at n_subset=500.

#         References
#         ----------
#         Marchetti, Y., Paciorek, C.J., & Genton, M.G. (2018).
#         An assessment of model selection uncertainty in spatial
#         prediction. *Environmetrics*, 29(7–8), e2530.
#         doi:10.1002/env.2530

#         Lark, R.M. (2000). A comparison of some robust estimators
#         of the variogram for use in soil survey. *Eur. J. Soil Sci.*,
#         51, 137–157.
#         """
#         if model_types is None:
#             model_types = ['spherical', 'exponential', 'matern']

#         # reproducible child seeds
#         ss = np.random.SeedSequence(seed)
#         child_seeds = ss.spawn(n_realizations)

#         # storage
#         records: List[Dict[str, Any]] = []
#         all_empirical: List[np.ndarray] = []
#         all_counts_list: List[np.ndarray] = []
#         all_fitted_curves: List[np.ndarray] = []
#         common_lags: Optional[np.ndarray] = None
#         n_failed = 0

#         for r in range(n_realizations):
#             run_seed = int(child_seeds[r].generate_state(1)[0])

#             if verbose:
#                 print(f"  Realization {r + 1}/{n_realizations} "
#                       f"(seed={run_seed})", end=" ... ", flush=True)

#             try:
#                 # ── 1. Sample raster and compute empirical variogram ──
#                 # return_sample=True captures the coords/values that
#                 # produced this variogram for coherent MSSPE evaluation.
#                 (bin_counts, variogram_est, n_bins,
#                  min_dist, max_dist, max_lag,
#                  sample_coords, sample_values) = self.numba_variogram(
#                     area_side, samples_per_area, max_samples,
#                     bin_width, max_lag_multiplier,
#                     seed=run_seed,
#                     estimator=estimator,
#                     return_sample=True,
#                 )

#                 # build lags and filter valid bins
#                 n_lags = variogram_est.size
#                 lags = np.linspace(
#                     bin_width / 2,
#                     bin_width * n_lags - bin_width / 2,
#                     n_lags,
#                 )
#                 valid = np.isfinite(variogram_est)
#                 if valid.sum() < 5:
#                     raise ValueError("Fewer than 5 valid lag bins.")

#                 emp_vario = variogram_est[valid]
#                 emp_lags = lags[valid]
#                 emp_counts = bin_counts[valid] if bin_counts is not None else None

#                 # Store common lag vector from first success
#                 if common_lags is None:
#                     common_lags = emp_lags.copy()

#                 # Compute sigma (std across bins) — use simple approach
#                 # for single realizations: std ∝ γ(h) / √N(h)
#                 if emp_counts is not None:
#                     safe_counts = np.where(emp_counts > 0, emp_counts, 1)
#                     sigma = emp_vario / np.sqrt(safe_counts)
#                     sigma = np.where(sigma > 0, sigma, np.finfo(float).eps)
#                 else:
#                     sigma = None

#                 # ── 2. Fit all candidate models ──
#                 selector = VariogramModelSelector(
#                     lags=emp_lags,
#                     empirical_variogram=emp_vario,
#                     pair_counts=emp_counts,
#                     sigma=sigma,
#                     min_pairs=min_pairs,
#                 )

#                 # override model lists
#                 available = MODEL_REGISTRY.list_models()
#                 selector.BOUNDED_MODELS = [m for m in model_types
#                                            if m in available and MODEL_REGISTRY.is_bounded(m)]
#                 selector.UNBOUNDED_MODELS = [m for m in model_types
#                                              if m in available and not MODEL_REGISTRY.is_bounded(m)]

#                 selector.fit_all_candidates(
#                     max_components=min(max_components, 3),
#                     include_nugget=include_nugget,
#                     include_unbounded=bool(selector.UNBOUNDED_MODELS),
#                     bounded_only_combinations=bounded_only_combinations,
#                 )

#                 if not selector.fitted_models:
#                     raise ValueError("No models successfully fitted.")

#                 # ── 3. Compute MSSPE and select best ──
#                 # Use the SAME spatial sample that produced the empirical
#                 # variogram.  Precompute the LOOCV subsample and distance
#                 # matrix once, then share across all candidate models.
#                 if criterion == 'msspe':
#                     sub_rng = np.random.default_rng(run_seed)
#                     n_pts = len(sample_values)
#                     if n_pts > msspe_n_subset:
#                         sub_idx = sub_rng.choice(
#                             n_pts, msspe_n_subset, replace=False)
#                         sub_coords = sample_coords[sub_idx]
#                         sub_values = sample_values[sub_idx]
#                     else:
#                         sub_coords = sample_coords
#                         sub_values = sample_values

#                     if verbose:
#                         print(f"    MSSPE: {len(sub_coords)} points, "
#                               f"{len(selector.fitted_models)} candidates")

#                     # distance matrix: computed once, reused for every model
#                     dx = sub_coords[:, 0:1] - sub_coords[:, 0:1].T
#                     dy = sub_coords[:, 1:2] - sub_coords[:, 1:2].T
#                     sub_dist = np.sqrt(dx**2 + dy**2)

#                     for fitted in selector.fitted_models:
#                         model_name = "+".join(fitted.composite_model.component_names)
#                         if not fitted.composite_model.is_stationary:
#                             if verbose:
#                                 print(f"      {model_name}: skipped (non-stationary)")
#                             continue
#                         try:
#                             result = self.kriging_loocv(
#                                 sub_coords, sub_values,
#                                 fitted.composite_model,
#                                 n_subset=len(sub_values),
#                                 dist_matrix=sub_dist,
#                             )
#                             if verbose:
#                                 print(f"      {model_name}: "
#                                       f"MSSPE={result.msspe:.4f} "
#                                       f"({result.n_points} pts, "
#                                       f"{result.n_failed} failed)")
#                             if np.isfinite(result.msspe):
#                                 fitted.msspe = result.msspe
#                                 fitted.msspe_std = None
#                                 fitted.msspe_n_runs = 1
#                                 fitted.loocv_result = AggregatedLOOCVResult.from_results([result])
#                         except Exception as exc:
#                             warnings.warn(
#                                 f"MSSPE failed for {model_name}: "
#                                 f"{type(exc).__name__}: {exc}"
#                             )
#                             continue

#                 best = selector.select_best(criterion=criterion)

#                 # ── 4. Extract parameters ──
#                 model = best.composite_model
#                 params = best.params
#                 # structural description for grouping (no parameter values)
#                 struct_desc = model.structural_description()

#                 # extract sills, ranges, nugget, nu
#                 sills_r = []
#                 ranges_r = []
#                 nu_val = np.nan

#                 for i, spec in enumerate(model._components):
#                     comp_params = model.get_component_params(i)
#                     if spec.has_sill:
#                         sills_r.append(comp_params[0])
#                     for rkey in ('range', 'wavelength'):
#                         if rkey in spec.param_names:
#                             ridx = spec.param_names.index(rkey)
#                             raw_range = comp_params[ridx]
#                             # Compute effective (practical) range.
#                             # For Matérn, practical_range_factor is None
#                             # because it depends on ν: effective range
#                             # ≈ range × √(2ν) × 2.7 (distance to 95%
#                             # of sill).  For all others, use the fixed
#                             # factor.
#                             prf = spec.practical_range_factor
#                             if prf is None and 'nu' in spec.param_names:
#                                 nu_idx_local = spec.param_names.index('nu')
#                                 nu_local = comp_params[nu_idx_local]
#                                 # Matérn effective range (95% of sill):
#                                 # approx 2.7 * range * sqrt(2*nu) for
#                                 # typical nu.  Simplified: use the
#                                 # scipy kv-based factor, but a good
#                                 # approximation is 3*range for nu≤0.5,
#                                 # range*sqrt(8*nu) for nu>0.5.
#                                 if nu_local <= 0.5:
#                                     prf = 3.0
#                                 else:
#                                     prf = float(np.sqrt(8 * nu_local))
#                             elif prf is None:
#                                 prf = 1.0
#                             ranges_r.append(raw_range * prf)
#                             break
#                     if 'nu' in spec.param_names:
#                         nu_idx = spec.param_names.index('nu')
#                         nu_val = comp_params[nu_idx]

#                 nugget_val = model.get_nugget() if model.include_nugget else np.nan

#                 # fitted curve evaluated at common lags
#                 fitted_curve = model(common_lags) if common_lags is not None else np.array([])

#                 # empirical variogram and counts padded/trimmed to common lag length
#                 if common_lags is not None:
#                     emp_padded = np.full(len(common_lags), np.nan)
#                     n_copy = min(len(emp_vario), len(common_lags))
#                     emp_padded[:n_copy] = emp_vario[:n_copy]

#                     counts_padded = np.full(len(common_lags), np.nan)
#                     if emp_counts is not None:
#                         n_copy_c = min(len(emp_counts), len(common_lags))
#                         counts_padded[:n_copy_c] = emp_counts[:n_copy_c]
#                 else:
#                     emp_padded = emp_vario
#                     counts_padded = emp_counts if emp_counts is not None else np.zeros_like(emp_vario)

#                 record = {
#                     'model_description': struct_desc,
#                     'component_names': list(model.component_names),
#                     'include_nugget': model.include_nugget,
#                     'params': params.copy(),
#                     'param_names': model.param_names,
#                     'sills': sills_r,
#                     'ranges': ranges_r,
#                     'nugget': nugget_val,
#                     'nu': nu_val,
#                     'msspe': best.msspe,
#                     'msspe_std': best.msspe_std,
#                     'aic': best.aic,
#                     'rss': best.rss,
#                     'fitted_curve': fitted_curve,
#                     'empirical_variogram': emp_padded,
#                     'seed': run_seed,
#                 }
#                 records.append(record)
#                 all_fitted_curves.append(fitted_curve)
#                 all_empirical.append(emp_padded)
#                 all_counts_list.append(counts_padded)

#                 if verbose:
#                     msspe_str = (f"MSSPE={best.msspe:.3f}" if best.msspe is not None
#                                  else "MSSPE=N/A")
#                     print(f"{struct_desc}  {msspe_str}")

#             except Exception as e:
#                 n_failed += 1
#                 if verbose:
#                     print(f"FAILED ({e})")
#                 continue

#         # ── Assemble ensemble result ──
#         if not records:
#             raise RuntimeError(
#                 f"All {n_realizations} realizations failed. "
#                 "Check raster data and fitting parameters."
#             )

#         # model selection counts
#         from collections import Counter
#         desc_list = [rec['model_description'] for rec in records]
#         model_counts = dict(Counter(desc_list))
#         n_ok = len(records)
#         model_fractions = {k: v / n_ok for k, v in model_counts.items()}

#         # pad sills/ranges to common width
#         max_n_sills = max(len(rec['sills']) for rec in records)
#         max_n_ranges = max(len(rec['ranges']) for rec in records)

#         sills_arr = np.full((n_ok, max(max_n_sills, 1)), np.nan)
#         ranges_arr = np.full((n_ok, max(max_n_ranges, 1)), np.nan)
#         nuggets_arr = np.array([rec['nugget'] for rec in records])
#         nus_arr = np.array([rec['nu'] for rec in records])
#         msspes_arr = np.array([
#             rec['msspe'] if rec['msspe'] is not None else np.nan
#             for rec in records
#         ])

#         for i, rec in enumerate(records):
#             for j, s in enumerate(rec['sills']):
#                 sills_arr[i, j] = s
#             for j, rng in enumerate(rec['ranges']):
#                 ranges_arr[i, j] = rng

#         # stack fitted curves, empirical variograms, and pair counts
#         variograms_arr = np.array(all_fitted_curves)
#         empirical_arr = np.array(all_empirical)
#         counts_arr = np.array(all_counts_list) if all_counts_list else np.zeros_like(empirical_arr)

#         result = EnsembleVariogramResult(
#             n_realizations=n_realizations,
#             n_failed=n_failed,
#             model_counts=model_counts,
#             model_fractions=model_fractions,
#             sills=sills_arr,
#             ranges=ranges_arr,
#             nuggets=nuggets_arr,
#             nus=nus_arr,
#             msspes=msspes_arr,
#             lags=common_lags if common_lags is not None else np.array([]),
#             variograms=variograms_arr,
#             empirical_variograms=empirical_arr,
#             pair_counts=counts_arr,
#             per_realization=records,
#         )

#         if verbose:
#             print("\n" + result.summary())

#         return result

#     def plot_best_model(self):
#         """
#         Plot mean variogram ± spread and fitted model; also show bar plot of mean pair counts.

#         Uses two-level uncertainty shading:
#         - Full range (very light shading, α=0.1): 2.5th to 97.5th percentiles
#         - 1σ range (darker shading, α=0.3): 16th to 84th percentiles
#         - Optimal value: dashed line
#         """
#         from matplotlib.patches import Patch
#         from matplotlib.lines import Line2D

#         if any(attr is None for attr in (self.mean_variogram, self.err_variogram, self.mean_count, self.lags, self.fitted_variogram)):
#             raise RuntimeError("Missing variogram data. Call calculate_mean_variogram_numba() and fit method first.")

#         n = min(len(self.lags), len(self.mean_variogram), len(self.err_variogram), len(self.fitted_variogram))
#         lags = self.lags[:n]
#         gamma = self.mean_variogram[:n]
#         errs = self.err_variogram[:n]
#         model = self.fitted_variogram[:n]
#         counts = self.mean_count[:n]

#         valid_counts = (~np.isnan(counts)) & (counts > 0)
#         count_lags = lags[valid_counts]
#         count_vals = counts[valid_counts]

#         fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 8), sharex=True)

#         # guard single-bin bar width
#         if len(lags) > 1:
#             bar_width = (lags[1] - lags[0]) * 0.9
#         else:
#             bar_width = (lags[0] if len(lags) else 1.0) * 0.9
#         axs[0].bar(count_lags, count_vals, width=bar_width, color='orange', alpha=0.5)
#         axs[0].set_ylabel('Mean Count')
#         axs[0].tick_params(labelbottom=False)

#         # plot empirical variogram and fitted model
#         axs[1].errorbar(lags, gamma, yerr=errs, fmt='o-', color='blue', label='Mean Variogram ± spread')
#         axs[1].plot(lags, model, 'r-', label='Fitted Model')

#         # range uncertainty shading (vertical bands)
#         colors = ['red', 'green', 'blue']
#         if self.ranges is not None and self.ranges_min is not None and self.ranges_max is not None:
#             ylim = axs[1].get_ylim()
#             for i, (r, rmin, rmax) in enumerate(zip(self.ranges, self.ranges_min, self.ranges_max)):
#                 c = colors[i % len(colors)]
#                 # full range (very light shading)
#                 axs[1].fill_betweenx(ylim, rmin, rmax, color=c, alpha=0.1)
#                 # 1σ range (darker shading) if available
#                 if hasattr(self, 'ranges_p16') and self.ranges_p16 is not None:
#                     r_p16 = self.ranges_p16[i]
#                     r_p84 = self.ranges_p84[i]
#                     axs[1].fill_betweenx(ylim, r_p16, r_p84, color=c, alpha=0.3)
#                 # optimal value (dashed line)
#                 axs[1].axvline(r, color=c, linestyle='--', linewidth=1.5)

#         # nugget uncertainty shading (horizontal bands)
#         if self.best_nugget is not None and self.min_nugget is not None and self.max_nugget is not None:
#             # full range (very light shading)
#             axs[1].fill_between(lags, [self.min_nugget] * len(lags), [self.max_nugget] * len(lags), color='orange', alpha=0.1)
#             # 1σ range (darker shading) if available
#             if hasattr(self, 'nugget_p16') and self.nugget_p16 is not None:
#                 axs[1].fill_between(lags, [self.nugget_p16] * len(lags), [self.nugget_p84] * len(lags), color='orange', alpha=0.3)
#             # optimal value (dashed line)
#             axs[1].axhline(self.best_nugget, color='orange', linestyle='--', linewidth=1.5)

#         # build custom legend
#         legend_elements = [
#             Line2D([0], [0], marker='o', color='blue', label='Mean Variogram ± spread', linestyle='-'),
#             Line2D([0], [0], color='r', linestyle='-', label='Fitted Model'),
#             Patch(facecolor='gray', alpha=0.1, label='Full range (95%)'),
#             Patch(facecolor='gray', alpha=0.4, label='1σ range (68%)'),
#             Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Optimal'),
#         ]
#         # add color swatches for each range/wavelength
#         if self.ranges is not None:
#             for i in range(len(self.ranges)):
#                 c = colors[i % len(colors)]
#                 lbl = 'Wavelength' if (hasattr(self, 'range_labels') and
#                     i < len(self.range_labels) and
#                     self.range_labels[i] == 'wavelength') else 'Range'
#                 legend_elements.append(Patch(facecolor=c, alpha=0.5, label=f'{lbl} {i + 1}'))
#         # add nugget to legend if present
#         if self.best_nugget is not None:
#             legend_elements.append(Patch(facecolor='orange', alpha=0.5, label='Nugget'))

#         axs[1].set_xlabel('Lag Distance')
#         axs[1].set_ylabel('Semivariance')
#         axs[1].legend(handles=legend_elements, loc='upper right')

#         # add model info to title
#         title_str = ""
#         if hasattr(self, 'best_msspe') and self.best_msspe is not None:
#             title_str = f'MSSPE: {self.best_msspe:.3f}'
#         axs[1].set_title(title_str)
#         plt.setp(axs[0].get_xticklabels(), visible=False)
#         plt.tight_layout()
#         return fig


# =====================================================================
# NEW SIMPLIFIED IMPLEMENTATION
# =====================================================================

from __future__ import annotations

import warnings
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit, prange
import rasterio
import rioxarray as rio
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape
from itertools import combinations_with_replacement

from .variogram_models import MODEL_REGISTRY, VariogramModelRegistry
from .composite_variogram import CompositeVariogramModel


# ── helper dataclass: kriging LOOCV result ──────────────────────────

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


# ── helper: aggregated LOOCV across multiple runs ───────────────────

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


# ── RasterDataHandler (unchanged from original) ─────────────────────

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


# ── Numba-accelerated pairwise binning ───────────────────────────────

@njit(parallel=True)
def _bin_distances_and_squared_differences(
    coords, values, bin_width, max_lag
):
    """Bin pairwise distances and accumulate squared-diff / sqrt-abs-diff.

    Returns
    -------
    n_bins, bin_counts, binned_sum_sq_diff, binned_sum_sqrt_abs_diff,
    max_distance
    """
    n_bins = int(np.ceil(max_lag / bin_width)) + 1
    M = coords.shape[0]
    max_distance = 0.0
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    binned_sum_sq = np.zeros(n_bins, dtype=np.float64)
    binned_sum_sqrt = np.zeros(n_bins, dtype=np.float64)

    for i in prange(M):
        for j in range(i + 1, M):
            d = 0.0
            for k in range(coords.shape[1]):
                tmp = coords[i, k] - coords[j, k]
                d += tmp * tmp
            dist = np.sqrt(d)
            if dist > max_distance:
                max_distance = dist

            diff = values[i] - values[j]
            bin_idx = int(dist / bin_width)
            if 0 <= bin_idx < n_bins:
                bin_counts[bin_idx] += 1
                binned_sum_sq[bin_idx] += diff ** 2
                binned_sum_sqrt[bin_idx] += np.sqrt(np.abs(diff))

    return n_bins, bin_counts, binned_sum_sq, binned_sum_sqrt, max_distance


# ── SingleVariogram ──────────────────────────────────────────────────

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
    BOUNDED_MODELS = ["spherical", "exponential", "matern"]
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

        # model fitting attributes (populated by fit_model)
        self.fitted_models: List[Dict[str, Any]] = []
        self.best_model: Optional[Dict[str, Any]] = None
        self.criteria_table: Optional[pd.DataFrame] = None

    # ── empirical variogram computation ─────────────────────────────

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
                correction = 0.457 + 0.494 / bin_counts[i]
                gamma[i] = 0.5 * mean_fourth / correction
        return gamma

    def _single_variogram_run(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        bin_width: float,
        max_lag: float,
        estimator: str,
        seed: Optional[int] = None,
    ):
        """Sample the raster once and compute one empirical variogram.

        Returns (bin_counts, gamma_estimate, n_bins, coords, values).
        """
        self.rdh.sample_raster(
            area_side, samples_per_area, max_samples, seed=seed
        )
        coords = self.rdh.coords.copy()
        values = self.rdh.samples.copy()

        (n_bins, bin_counts, bssd, bssad, _max_dist) = (
            _bin_distances_and_squared_differences(
                coords, values, bin_width, max_lag
            )
        )

        if estimator == "cressie_hawkins":
            gamma = self._compute_cressie_hawkins(
                bin_counts, bssad, self.MIN_PAIRS
            )
        else:
            gamma = self._compute_matheron(bin_counts, bssd, self.MIN_PAIRS)

        return bin_counts, gamma, n_bins, coords, values

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
        """
        if estimator not in self.ESTIMATORS:
            raise ValueError(
                f"Unknown estimator '{estimator}'. "
                f"Choose from {self.ESTIMATORS}."
            )

        # compute max_lag from raster extent
        xs = self.rdh.rioxarray_obj.x.values
        ys = self.rdh.rioxarray_obj.y.values
        x_extent = float(np.max(xs) - np.min(xs))
        y_extent = float(np.max(ys) - np.min(ys))
        diag = np.sqrt(x_extent ** 2 + y_extent ** 2)
        max_lag = float(diag * max_lag_multiplier)

        # sample and compute
        bin_counts, gamma, n_bins_run, coords, values = (
            self._single_variogram_run(
                area_side, samples_per_area, max_samples,
                bin_width, max_lag, estimator, seed=seed,
            )
        )

        # keep only valid (non-NaN) bins
        valid = ~np.isnan(gamma)
        n_kept = int(np.sum(valid))

        self.variogram = gamma[valid]
        self.pair_counts = bin_counts[valid].astype(float)
        self.lags = np.linspace(
            bin_width / 2, bin_width * n_kept - bin_width / 2, n_kept
        )
        self.n_bins = n_kept
        self.estimator = estimator

        if return_sample:
            self.sample_coords = coords
            self.sample_values = values

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
    ) -> Optional[Dict[str, Any]]:
        """Fit one composite model via multi-start WLS.

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
                    p0=p0, sigma=sigma,
                    absolute_sigma=sigma is not None,
                    bounds=bounds, maxfev=maxfev,
                )
                model.set_params(popt)
                residuals = variogram - model(lags)
                if sigma is not None:
                    safe_sigma = np.where(
                        np.isfinite(sigma) & (sigma > 0), sigma, np.inf
                    )
                    rss = float(np.sum((residuals / safe_sigma) ** 2))
                else:
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

        # information criteria
        if sigma is not None:
            finite_mask = np.isfinite(sigma) & (sigma > 0)
            n_eff = int(np.sum(finite_mask))
            sig = sigma[finite_mask]
            res = (variogram - model(lags))[finite_mask]
            ll = -0.5 * np.sum(np.log(2 * np.pi * sig ** 2) + res ** 2 / sig ** 2)
            aic = 2 * k - 2 * ll
            bic = k * np.log(max(n_eff, 1)) - 2 * ll
        else:
            n_eff = len(lags)
            aic = n_eff * np.log(rss / max(n_eff, 1)) + 2 * k
            bic = n_eff * np.log(rss / max(n_eff, 1)) + k * np.log(max(n_eff, 1))

        # AICc (small-sample corrected)
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

    # ── REML fitting ─────────────────────────────────────────────

    @staticmethod
    def _build_trend_matrix(
        coords: np.ndarray,
        trend_order: int = 0,
    ) -> np.ndarray:
        """Build the trend design matrix F for given coordinates.

        Follows the convention used by geoR's ``likfit`` function
        (Ribeiro & Diggle, 2001), where ``trend`` = ``"cte"``
        corresponds to ``trend_order=0``, ``"1st"`` to
        ``trend_order=1``, and ``"2nd"`` to ``trend_order=2``.

        Parameters
        ----------
        coords : (n, 2) array
            Spatial coordinates.
        trend_order : {0, 1, 2}
            Polynomial order of the spatial trend.

            - 0: constant mean only, F = [1] (p=1).
            - 1: linear, F = [1, x, y] (p=3).
            - 2: quadratic, F = [1, x, y, x², xy, y²] (p=6).

        Returns
        -------
        F : (n, p) array
            Design matrix.

        References
        ----------
        Ribeiro, P.J. & Diggle, P.J. (2001). geoR: A package for
        geostatistical analysis. *R-NEWS*, 1(2), 14–18.
        """
        n = len(coords)
        x = coords[:, 0]
        y = coords[:, 1]

        if trend_order == 0:
            return np.ones((n, 1))
        elif trend_order == 1:
            return np.column_stack([np.ones(n), x, y])
        elif trend_order == 2:
            return np.column_stack([
                np.ones(n), x, y, x ** 2, x * y, y ** 2,
            ])
        else:
            raise ValueError(
                f"trend_order must be 0, 1, or 2, got {trend_order}"
            )

    @staticmethod
    def _reml_neg_log_likelihood(
        params: np.ndarray,
        model: CompositeVariogramModel,
        dist_matrix: np.ndarray,
        values: np.ndarray,
        F: np.ndarray,
    ) -> float:
        """Profiled REML negative log-likelihood.

        The total variance σ² is profiled out analytically,
        so the optimiser only searches over the correlation
        structure (ranges, nugget-to-sill ratio).  This exactly
        mirrors geoR's ``likGRF.R`` computation::

            V      <- correlation matrix (diagonal = 1)
            ivx    <- solve(V, xmat)
            xivx   <- crossprod(ivx, xmat)
            betahat <- solve(xivx, crossprod(ivx, z))
            res    <- z - xmat %*% betahat
            ssres  <- drop(crossprod(res, solve(V, res)))
            choldet <- sum(log(diag(chol(xivx))))
            negloglik <- ((n-p)/2)*log(ssres)
                         + log_det_V_half + choldet

        The profiled REML NLL (up to a constant) is::

            (n-p)/2 · log(r'V⁻¹r) + ½ log|V| + ½ log|F'V⁻¹F|

        where V = C / σ² is the normalised correlation matrix,
        and σ̂² = r'V⁻¹r / (n-p) is the profiled MLE of the
        total variance.

        Working with the unit-scale V rather than C eliminates
        scale-dependent conditioning problems that cause Cholesky
        failures when the sill is far from the data variance.

        Parameters
        ----------
        params : (k,) array
            Variogram model parameters (sill, range, ... nugget).
            Only the *ratios* of the sill components affect V;
            the absolute scale is recovered from the profile.
        model : CompositeVariogramModel
            Model whose ``__call__`` returns γ(h).
        dist_matrix : (n, n) array
            Pre-computed pairwise distances.
        values : (n,) array
            Observed values at sample locations.
        F : (n, p) array
            Trend design matrix.

        Returns
        -------
        float
            Negative profiled REML log-likelihood (to minimise).

        References
        ----------
        Patterson, H.D. & Thompson, R. (1971). Recovery of inter-block
        information when block sizes are unequal. *Biometrika*, 58,
        545–554.

        Ribeiro, P.J. & Diggle, P.J. (2001). geoR: A package for
        geostatistical analysis. *R-NEWS*, 1(2), 14–18.
        Source: ``github.com/rundel/geoR/blob/master/R/likGRF.R``

        Cressie, N. (1993). *Statistics for Spatial Data*, rev. ed.,
        Wiley. §2.6.
        """
        from scipy.linalg import cho_factor, cho_solve

        n = len(values)
        p = F.shape[1]
        model.set_params(params)

        # total sill (used only as a normalising constant)
        total_sill = model.get_total_sill()
        if total_sill is None or total_sill <= 0:
            return 1e20

        # ── normalised correlation matrix V = C / σ² ──
        # C(h) = total_sill - γ(h),  V = C / total_sill
        # V_ii = 1,  V_ij = 1 - γ(h_ij) / total_sill
        gamma_matrix = model(dist_matrix)
        V = 1.0 - gamma_matrix / total_sill

        # jitter on unit-scale diagonal for numerical stability
        np.fill_diagonal(V, 1.0 + 1e-6)

        # Cholesky of V (unit-scale → much better conditioned)
        try:
            cho = cho_factor(V, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e20

        # log|V|
        log_det_V = 2.0 * np.sum(np.log(np.diag(cho[0])))

        # ── batch-solve V⁻¹[F | z] ──
        Fz = np.empty((n, p + 1))
        Fz[:, :p] = F
        Fz[:, p] = values
        V_inv_Fz = cho_solve(cho, Fz, check_finite=False)

        V_inv_F = V_inv_Fz[:, :p]   # (n, p)
        V_inv_z = V_inv_Fz[:, p]     # (n,)

        # ── F'V⁻¹F  and its log-determinant ──
        FtVinvF = F.T @ V_inv_F
        try:
            cho_small = cho_factor(FtVinvF, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e20
        log_det_FtVinvF = 2.0 * np.sum(np.log(np.diag(cho_small[0])))

        # ── GLS trend: β̂ = (F'V⁻¹F)⁻¹ F'V⁻¹z ──
        FtVinvz = F.T @ V_inv_z
        beta_hat = cho_solve(cho_small, FtVinvz, check_finite=False)

        # ── r'V⁻¹r  (profiled sufficient statistic) ──
        V_inv_r = V_inv_z - V_inv_F @ beta_hat
        r = values - F @ beta_hat
        ssres = r @ V_inv_r           # r'V⁻¹r

        if ssres <= 0:
            return 1e20

        # ── profiled REML NLL (constant terms dropped) ──
        # σ̂² = ssres / (n-p)  →  (n-p) log(σ̂²) = (n-p) log(ssres/(n-p))
        # = (n-p) log(ssres) - (n-p) log(n-p)  [second term is constant]
        nll = 0.5 * (
            (n - p) * np.log(ssres)
            + log_det_V
            + log_det_FtVinvF
        )

        if not np.isfinite(nll):
            return 1e20

        return float(nll)

    _TREND_LABELS = {0: "cte", 1: "1st", 2: "2nd"}

    def _fit_single_composite_model_reml(
        self,
        model: CompositeVariogramModel,
        coords: np.ndarray,
        values: np.ndarray,
        dist_matrix: np.ndarray,
        lags: np.ndarray,
        variogram: np.ndarray,
        trend_order: int = 0,
        n_restarts: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Fit one composite model via profiled REML.

        Adapts the joint trend + covariance estimation approach
        from geoR's ``likfit`` (Ribeiro & Diggle, 2001).  Both
        the trend coefficients β *and* the total variance σ² are
        profiled out analytically — the optimiser only searches
        over correlation parameters (ranges, nugget-to-sill ratio).

        After optimisation, σ̂² = r'V⁻¹r / (n-p) is recovered
        from the profile and used to rescale the model parameters
        to their absolute values.

        Parameters
        ----------
        model : CompositeVariogramModel
            Candidate model to fit.
        coords : (n, 2) array
            Spatial coordinates of sample points.
        values : (n,) array
            Observed values at sample locations.
        dist_matrix : (n, n) array
            Pre-computed pairwise distance matrix.
        lags, variogram : arrays
            Empirical variogram (used only for generating initial
            guesses and bounds, not for fitting).
        trend_order : {0, 1, 2}
            Polynomial trend order.  0 = constant mean (p=1),
            1 = linear in x,y (p=3), 2 = quadratic (p=6).
            Matches geoR's ``"cte"`` / ``"1st"`` / ``"2nd"``.
        n_restarts : int
            Number of multi-start optimisations.

        Returns
        -------
        dict or None
            Fitted model dict with keys: model, params, param_cov,
            aic, aicc, bic, reml_nll, trend_order, trend_coefficients,
            description, etc.
            None if all optimisation attempts fail.

        References
        ----------
        Ribeiro, P.J. & Diggle, P.J. (2001). geoR: A package for
        geostatistical analysis. *R-NEWS*, 1(2), 14–18.
        Source: ``github.com/rundel/geoR/blob/master/R/likGRF.R``
        """
        from scipy.optimize import minimize
        from scipy.linalg import cho_factor, cho_solve

        # ── build trend design matrix ──
        F = self._build_trend_matrix(coords, trend_order)
        p = F.shape[1]  # number of trend parameters

        p0_base = model.default_guess(lags, variogram)
        bounds_pair = model.bounds(lags, variogram)
        lb = np.array(bounds_pair[0])
        ub = np.array(bounds_pair[1])

        # For REML, relax the nugget cap — the profiled likelihood
        # can properly estimate any nugget-to-sill ratio, including
        # data that is predominantly nugget (pure noise).
        if model.include_nugget:
            nugget_idx = model.n_params - 1
            max_gamma = np.nanmax(variogram)
            ub[nugget_idx] = max_gamma * 3.0    # allow nugget up to 3× max γ

            # nugget pre-estimation
            nugget_pre = self._estimate_nugget_from_short_lags(lags, variogram)
            p0_base[nugget_idx] = np.clip(
                nugget_pre, lb[nugget_idx], ub[nugget_idx]
            )

        # scipy bounds format
        sp_bounds = list(zip(lb, ub))

        # generate multi-start guesses
        rng = np.random.default_rng()
        guesses = self._generate_multistart_guesses(
            p0_base, (list(lb), list(ub)), n_restarts, rng
        )

        best_result = None
        best_nll = np.inf
        n = len(values)

        for p0 in guesses:
            try:
                res = minimize(
                    self._reml_neg_log_likelihood,
                    p0,
                    args=(model, dist_matrix, values, F),
                    method="L-BFGS-B",
                    bounds=sp_bounds,
                    options={"maxiter": 500, "ftol": 1e-10},
                )
                # Accept any result with a finite NLL better than
                # what we've seen — don't require res.success, as
                # L-BFGS-B may report non-convergence even when it
                # has found a good point (common for flat directions
                # in the profiled likelihood).
                if np.isfinite(res.fun) and res.fun < best_nll:
                    best_nll = res.fun
                    best_result = res
            except Exception:
                continue

        if best_result is None:
            return None

        # ── recover σ̂² from the profiled likelihood ──
        # The optimiser found the correlation shape; now we
        # compute the optimal total variance analytically.
        popt = best_result.x
        nll_profiled = best_result.fun
        model.set_params(popt)

        total_sill_opt = model.get_total_sill()
        if total_sill_opt is None or total_sill_opt <= 0:
            return None

        # Build V (normalised correlation matrix) at the optimum
        gamma_matrix = model(dist_matrix)
        V = 1.0 - gamma_matrix / total_sill_opt
        np.fill_diagonal(V, 1.0 + 1e-6)

        try:
            cho = cho_factor(V, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return None

        # Solve V⁻¹[F | z]
        Fz = np.empty((n, p + 1))
        Fz[:, :p] = F
        Fz[:, p] = values
        V_inv_Fz = cho_solve(cho, Fz, check_finite=False)
        V_inv_F = V_inv_Fz[:, :p]
        V_inv_z = V_inv_Fz[:, p]

        FtVinvF = F.T @ V_inv_F
        try:
            cho_small = cho_factor(FtVinvF, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return None

        FtVinvz = F.T @ V_inv_z
        beta_hat = cho_solve(cho_small, FtVinvz, check_finite=False)

        r = values - F @ beta_hat
        V_inv_r = V_inv_z - V_inv_F @ beta_hat
        ssres = float(r @ V_inv_r)

        n_reml = n - p
        sigma2_hat = ssres / n_reml       # profiled MLE of total variance

        # ── diagnostic: check σ̂² is consistent with data variance ──
        data_var = float(np.var(values))
        if data_var > 0:
            _sv_ratio = sigma2_hat / data_var
            if _sv_ratio < 0.33 or _sv_ratio > 3.0:
                warnings.warn(
                    f"REML σ̂²={sigma2_hat:.4g} differs substantially "
                    f"from sample variance={data_var:.4g} "
                    f"(ratio={_sv_ratio:.2f}). "
                    f"This may indicate a data/model mismatch. "
                    f"Model: {model.structural_description()}, "
                    f"trend_order={trend_order}, "
                    f"scale_factor={sigma2_hat/total_sill_opt:.4g}",
                    UserWarning,
                    stacklevel=2,
                )

        # ── rescale model parameters to true absolute values ──
        # The optimiser found the shape (ratios); σ̂² gives the scale.
        scale_factor = sigma2_hat / total_sill_opt
        rescaled_params = popt.copy()
        # Scale each sill parameter and the nugget by the same factor
        for i, spec in enumerate(model._components):
            if spec.has_sill:
                comp_slice = model._param_slices[
                    model.component_names[i]
                    if model.component_names.count(model.component_names[i]) == 1
                    else f"{model.component_names[i]}_{i}"
                ]
                rescaled_params[comp_slice.start] *= scale_factor
        if model.include_nugget:
            nug_idx = model._param_slices['nugget'].start
            rescaled_params[nug_idx] *= scale_factor

        model.set_params(rescaled_params)
        popt = rescaled_params

        k_cov = model.n_params       # covariance parameters
        k_total = k_cov + p           # total estimated parameters

        # ── full REML NLL at the optimum (for AIC/BIC) ──
        # profiled NLL was: (n-p)/2 log(ssres) + ½ log|V| + ½ log|F'V⁻¹F|
        # full NLL adds: (n-p)/2 (1 + log(2π) - log(n-p))
        log_det_V = 2.0 * np.sum(np.log(np.diag(cho[0])))
        log_det_FtVinvF = 2.0 * np.sum(np.log(np.diag(cho_small[0])))
        full_nll = 0.5 * (
            n_reml * (1.0 + np.log(2 * np.pi))
            + n_reml * np.log(sigma2_hat)
            + log_det_V
            + log_det_FtVinvF
        )
        reml_ll = -full_nll
        aic = 2 * k_total - 2 * reml_ll
        bic = k_total * np.log(n_reml) - 2 * reml_ll
        denom = max(n_reml - k_total - 1, 1)
        aicc = aic + 2 * k_total * (k_total + 1) / denom

        # parameter covariance from inverse Hessian
        # (correlation parameters only — approximate)
        param_cov = None
        if hasattr(best_result, "hess_inv"):
            hi = best_result.hess_inv
            if hasattr(hi, "todense"):
                param_cov = np.array(hi.todense())
            else:
                param_cov = np.array(hi)
        if param_cov is None:
            param_cov = np.full((k_cov, k_cov), np.nan)

        # ── build description including trend ──
        trend_label = self._TREND_LABELS.get(trend_order, f"poly{trend_order}")
        base_desc = model.structural_description()
        if trend_order > 0:
            description = f"{base_desc} | trend={trend_label}"
        else:
            description = base_desc

        return {
            "model": model,
            "params": popt,
            "param_cov": param_cov,
            "rss": np.nan,  # not applicable for REML
            "aic": float(aic),
            "aicc": float(aicc),
            "bic": float(bic),
            "reml_nll": float(full_nll),
            "msspe": None,
            "msspe_std": None,
            "msspe_n_runs": 0,
            "description": description,
            "trend_order": trend_order,
            "trend_coefficients": beta_hat,
            "n_trend_params": p,
            "sigma2_hat": float(sigma2_hat),
            "warnings": [],
            "fitting_method": "reml",
        }

    @staticmethod
    def _kriging_loocv(
        coords: np.ndarray,
        values: np.ndarray,
        model: CompositeVariogramModel,
        n_subset: int = 500,
        dist_matrix: Optional[np.ndarray] = None,
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
            rng = np.random.default_rng()
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

    def fit_model(
        self,
        model_types: Optional[List[str]] = None,
        include_nugget: bool = True,
        max_components: int = 1,
        criterion: str = "aicc",
        fitting_method: str = "reml",
        reml_n_subset: int = 500,
        trend_orders: Optional[List[int]] = None,
        msspe_n_subset: int = 500,
        msspe_n_runs: int = 10,
        msspe_prefilter: int = 0,
        *,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fit candidate variogram models and select the best.

        Fits all combinations of ``model_types`` (up to ``max_components``
        nested structures) using either Weighted Least Squares on the
        binned empirical variogram or Restricted Maximum Likelihood on
        the spatial sample.

        When ``fitting_method='reml'``, each covariance candidate is
        fitted under every requested ``trend_orders``, and AICc (or
        the chosen criterion) selects jointly across covariance
        structure *and* trend order.  This follows the workflow used
        by geoR's ``likfit`` (Ribeiro & Diggle, 2001): fit each
        (trend, covariance) combination, compare AIC/BIC.

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
        fitting_method : {'reml', 'wls'}
            How to fit models to the data.

            - ``'reml'``: Restricted Maximum Likelihood — fits the
              covariance model directly to the spatial sample points.
              Jointly estimates trend and covariance via profiled
              likelihood.  Gives theoretically valid AIC/BIC and
              parameter uncertainty from the Hessian.  Requires
              spatial sample data (``return_sample=True``).
            - ``'wls'``: Weighted Least Squares — fits the model to the
              binned empirical variogram using Cressie (1985) weights.
              Does not estimate trend (assumes stationarity).
        reml_n_subset : int
            Subsample size for REML fitting (only used when
            ``fitting_method='reml'``).  REML requires O(n³)
            computation per likelihood evaluation; 500 is a good
            trade-off between information and speed.
        trend_orders : list of int, optional
            Polynomial trend orders to try when
            ``fitting_method='reml'``.  Each covariance candidate is
            fitted under each trend order, and the information
            criterion selects jointly.

            - 0: constant mean (p=1).
            - 1: linear in x, y (p=3).
            - 2: quadratic in x, y (p=6).

            Default: ``[0, 1]`` (try constant and linear).
            Ignored when ``fitting_method='wls'``.
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
        Ribeiro, P.J. & Diggle, P.J. (2001). geoR: A package for
        geostatistical analysis. *R-NEWS*, 1(2), 14–18.
        """
        if self.lags is None or self.variogram is None:
            raise RuntimeError(
                "No empirical variogram. "
                "Call compute_empirical_variogram() first."
            )

        use_reml = fitting_method == "reml"

        # REML requires spatial sample
        if use_reml and (self.sample_coords is None or self.sample_values is None):
            warnings.warn(
                "No spatial sample available for REML fitting. "
                "Re-run compute_empirical_variogram with return_sample=True. "
                "Falling back to WLS.",
                UserWarning,
            )
            use_reml = False

        if model_types is None:
            model_types = list(self.BOUNDED_MODELS)

        if trend_orders is None:
            trend_orders = [0, 1] if use_reml else [0]

        lags = self.lags
        variogram = self.variogram
        pair_counts = self.pair_counts

        # ── prepare REML data (subsample + distance matrix) ──
        reml_coords = None
        reml_values = None
        reml_dist = None

        if use_reml:
            coords_full = self.sample_coords
            values_full = self.sample_values
            n_full = len(values_full)

            # subsample for computational tractability
            if n_full > reml_n_subset:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_full, reml_n_subset, replace=False)
                reml_coords = coords_full[idx]
                reml_values = values_full[idx]
            else:
                reml_coords = coords_full
                reml_values = values_full

            # precompute distance matrix
            dx = reml_coords[:, 0:1] - reml_coords[:, 0:1].T
            dy = reml_coords[:, 1:2] - reml_coords[:, 1:2].T
            reml_dist = np.sqrt(dx ** 2 + dy ** 2)

            # ── diagnostic: compare REML sample variance with
            #    empirical variogram to catch data mismatches early ──
            _reml_var = float(np.var(reml_values))
            _emp_max = float(np.nanmax(variogram))
            if _reml_var > 0 and _emp_max > 0:
                _ratio = _emp_max / _reml_var
                if _ratio > 3.0 or _ratio < 0.33:
                    warnings.warn(
                        f"REML sample variance ({_reml_var:.4g}) differs "
                        f"substantially from empirical variogram max "
                        f"({_emp_max:.4g}), ratio={_ratio:.2f}. "
                        f"The sample_values used for REML may not match "
                        f"the data used to compute the empirical variogram. "
                        f"n_reml={len(reml_values)}, "
                        f"n_full={n_full}.",
                        UserWarning,
                        stacklevel=2,
                    )

        # ── prepare WLS weights (always needed for initial guesses) ──
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

        # ── fit each candidate ──
        # For REML, each covariance candidate is fitted under every
        # requested trend order — joint selection across (trend, cov).
        # This mirrors the geoR workflow of calling likfit() with
        # different trend= settings and comparing AIC/BIC.
        self.fitted_models = []
        for cand in candidates:
            if use_reml:
                for t_order in trend_orders:
                    # deep-copy the model so each (cov, trend) combo
                    # gets its own independent model object
                    import copy
                    cand_copy = copy.deepcopy(cand)
                    result = self._fit_single_composite_model_reml(
                        cand_copy, reml_coords, reml_values, reml_dist,
                        lags, variogram, trend_order=t_order,
                    )
                    if result is not None:
                        self.fitted_models.append(result)
            else:
                result = self._fit_single_composite_model(
                    cand, lags, variogram, None, weights
                )
                if result is not None:
                    # WLS has no trend estimation
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
                "fitting_method": m.get("fitting_method", "wls"),
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

    # ── plotting ────────────────────────────────────────────────────

    def plot_single_variogram(
        self,
        include_model: bool = False,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """Plot the empirical variogram with optional fitted model overlay.

        Parameters
        ----------
        include_model : bool
            If True and :meth:`fit_model` has been called, overlay the
            best-fit model curve and annotate with model name, parameters,
            and selection criteria (MSSPE, AIC, BIC).
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

        # empirical variogram points
        ax.plot(
            lags, self.variogram,
            "o-", color="blue", markersize=4, zorder=5,
            label="Empirical variogram",
        )

        # ── overlay fitted model ──
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

        ax.set_xlabel("Lag Distance (m)")
        ax.set_ylabel("Semivariance")

        # legend
        handles = [
            Line2D([0], [0], marker="o", color="blue",
                   linestyle="-", markersize=4, label="Empirical variogram"),
        ]
        if include_model and self.best_model is not None:
            handles.append(
                Line2D([0], [0], color="darkred", linewidth=2.2,
                       label=f'Fitted: {self.best_model["description"]}')
            )
        ax.legend(handles=handles, loc="lower right", fontsize=8)

        if not include_model:
            ax.set_title(
                f"Empirical variogram ({self.estimator})",
            )

        plt.tight_layout()
        return fig


# ── GridVariogram ────────────────────────────────────────────────────

class GridVariogram:
    """Run multiple independent SingleVariogram realisations and aggregate.

    Each realisation draws a fresh spatial sample, computes an empirical
    variogram, and (optionally) fits the full model selection pipeline.
    The ensemble captures both *model selection uncertainty* (which family
    wins?) and *parameter uncertainty* (how stable are sill / range /
    nugget?).

    Parameters
    ----------
    raster_data_handler : RasterDataHandler
        Provides raster data and sampling.
    n_realizations : int
        Number of independent SingleVariogram realisations to run.

    References
    ----------
    Lark, R.M. (2000). A comparison of some robust estimators of the
    variogram for use in soil survey. *Eur. J. Soil Sci.*, 51, 137-157.

    Marchetti, Y. et al. (2018). An assessment of model selection
    uncertainty in spatial prediction. *Environmetrics*, 29(7-8), e2530.
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
        max_components: int = 1,
        criterion: str = "aicc",
        fitting_method: str = "reml",
        reml_n_subset: int = 500,
        trend_orders: Optional[List[int]] = None,
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
        that smoothed curve.

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
        fitting_method : {'reml', 'wls'}
            Fitting method passed to each
            :meth:`SingleVariogram.fit_model`.  ``'reml'`` fits the
            covariance model directly to the spatial sample via
            Restricted Maximum Likelihood; ``'wls'`` fits to the binned
            empirical variogram via Weighted Least Squares.
        reml_n_subset : int
            Subsample size for REML fitting (passed through to
            :meth:`SingleVariogram.fit_model`).
        trend_orders : list of int, optional
            Polynomial trend orders to try (passed through to
            :meth:`SingleVariogram.fit_model`).  Default: ``[0, 1]``
            for REML, ``[0]`` for WLS.
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

        # REML and MSSPE both need the spatial sample retained
        needs_sample = fit_model and (
            fitting_method == "reml" or criterion == "msspe"
        )

        for i in range(self.n_realizations):
            run_seed = int(child_seeds[i].generate_state(1)[0])
            sv = SingleVariogram(self.rdh)
            try:
                if n_samples <= 1:
                    # ── single sample per realisation (original behaviour) ──
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
                    # ── n_samples: draw multiple, fit on median ─────────
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
                        fitting_method=fitting_method,
                        reml_n_subset=reml_n_subset,
                        trend_orders=trend_orders,
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
                        fm = sv.best_model.get("fitting_method", "?")
                        parts.append(f"[{fm}]")
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

        # ── aggregate results ──
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
        MSSPE computation.

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
        kept_coords = None
        kept_values = None

        for j in range(n_samples):
            s = int(inner_seeds[j].generate_state(1)[0])
            bin_counts, gamma, n_bins_run, coords, values = (
                sv._single_variogram_run(
                    area_side, samples_per_area, max_samples,
                    bin_width, max_lag, estimator, seed=s,
                )
            )
            all_gamma.append(gamma)
            all_counts.append(bin_counts.astype(float))

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

        # keep only valid (non-NaN) bins
        valid = ~np.isnan(median_gamma)
        n_kept = int(np.sum(valid))

        sv.variogram = median_gamma[valid]
        sv.pair_counts = mean_counts[valid]
        sv.lags = np.linspace(
            bin_width / 2, bin_width * n_kept - bin_width / 2, n_kept
        )
        sv.n_bins = n_kept
        sv.estimator = estimator

        if return_sample and kept_coords is not None:
            sv.sample_coords = kept_coords
            sv.sample_values = kept_values

    def _aggregate_model_results(self) -> None:
        """Aggregate candidate results across all realisations.

        Two things are tracked separately:

        1. **Best-model counts** — how many realisations selected each
           model structure as the winner.  This drives central model
           selection (highest count, MSSPE as tiebreaker).
        2. **Criteria averages** — AIC, BIC, MSSPE, and parameters
           averaged across *all* realisations that fitted each structure
           (not just the ones where it won).  This gives a fuller picture
           of each structure's quality.

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

    # ── plotting ────────────────────────────────────────────────────

    def plot_variogram(
        self,
        include_central_model: bool = True,
        include_all_models: bool = False,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """Plot ensemble variogram results.

        Parameters
        ----------
        include_central_model : bool
            Overlay the modal model evaluated at median parameters.
        include_all_models : bool
            Overlay every realisation's fitted model (thin lines).
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
                range_colors = ["red", "green", "lightblue"]
                r_idx = 0
                for key, stats in self.central_params.items():
                    if "range" in key:
                        c = range_colors[r_idx % len(range_colors)]
                        ax.axvline(
                            stats["median"], color=c, linewidth=1.8,
                            linestyle="-", zorder=5,
                            label=f'{key}: {stats["median"]:.0f}',
                        )
                        ax.axvspan(
                            stats["p16"], stats["p84"],
                            color=c, alpha=0.20, zorder=1,
                        )
                        r_idx += 1
                    elif key == "nugget":
                        ax.axhline(
                            stats["median"], color="orange", linewidth=1.8,
                            linestyle="-", zorder=5,
                            label=f'nugget: {stats["median"]:.4g}',
                        )
                        ax.axhspan(
                            stats["p16"], stats["p84"],
                            color="orange", alpha=0.20, zorder=1,
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
        return fig


# ── backward-compatibility aliases for modules that still import ─────
# the old class names (e.g. uncertainty.py).  These are thin stubs so
# the package doesn't break at import time while those modules are
# being updated.

class VariogramAnalysis(SingleVariogram):
    """Deprecated alias for SingleVariogram (backward compatibility)."""
    pass


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


class VariogramModelSelector:
    """Minimal stub for backward compatibility with uncertainty.py."""
    pass


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
        return fig

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

