"""variogram analysis and model fitting for differencing uncertainty."""

from __future__ import annotations

import math
from typing import Sequence, Optional, Dict, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import rasterio
import rioxarray as rio
from numba import njit, prange
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, Point, box, shape
from shapely.prepared import prep
import geopandas as gpd
from rasterio.features import shapes
from shapely.ops import unary_union
from .variogram_models import MODEL_REGISTRY, VariogramModelRegistry
from .composite_variogram import CompositeVariogramModel
from itertools import combinations_with_replacement
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class FittedVariogramModel:
    """Container for a fitted variogram model with diagnostics.

    Attributes
    ----------
    composite_model : CompositeVariogramModel
        The fitted composite model.
    params : ndarray
        Optimal parameter values.
    param_cov : ndarray
        Parameter covariance matrix from curve_fit.
    rss : float
        Residual sum of squares.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    cv_rmse : float
        Cross-validation RMSE.
    param_samples : ndarray or None
        Bootstrap parameter samples (n_boot × n_params).
    warnings : list of str
        Diagnostic warnings (e.g. range exceeds half max lag).
    msspe : float or None
        Mean Standardized Squared Prediction Error from kriging
        leave-one-out cross-validation, averaged over multiple
        independent random subsamples.  Target value is 1.0;
        values >> 1 indicate the model underestimates spatial
        uncertainty; values << 1 indicate overestimation.
    msspe_std : float or None
        Standard deviation of MSSPE across repeated runs.
        A large value relative to ``|msspe − 1|`` suggests the
        ranking is sensitive to the particular subsample drawn.
    msspe_n_runs : int
        Number of MSSPE runs that succeeded.
    loocv_result : AggregatedLOOCVResult or None
        Aggregated kriging LOOCV diagnostics across all repeated
        runs, including mean/median/std/nmad for MSSPE, bias,
        RMSE, and standardized error, plus the individual per-run
        results.
    """
    composite_model: CompositeVariogramModel
    params: np.ndarray
    param_cov: np.ndarray
    rss: float
    aic: float
    bic: float
    cv_rmse: Optional[float] = None
    param_samples: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)
    msspe: Optional[float] = None
    msspe_std: Optional[float] = None
    msspe_n_runs: int = 0
    loocv_result: Optional['AggregatedLOOCVResult'] = None

    def predict(self, h: np.ndarray) -> np.ndarray:
        """Evaluate fitted model at lag distances."""
        return self.composite_model(h)

    def check_half_lag(self, max_lag: float) -> None:
        """Check whether any fitted range exceeds half the maximum lag.

        Appends a diagnostic warning if so.  The half-lag heuristic
        (Journel & Huijbregts, 1978) holds that variogram parameters
        estimated from lags beyond half the domain may be unreliable
        because few point pairs constrain the fit there.
        """
        half_lag = max_lag / 2.0
        model = self.composite_model
        for i, spec in enumerate(model._components):
            if 'range' in spec.param_names:
                range_idx = spec.param_names.index('range')
                comp_params = model.get_component_params(i)
                fitted_range = comp_params[range_idx]
                if fitted_range > half_lag:
                    name = model.component_names[i]
                    msg = (
                        f"WARNING: {name} range ({fitted_range:.1f}) exceeds "
                        f"half the maximum lag ({half_lag:.1f}).  The variogram "
                        f"is poorly constrained beyond this distance — consider "
                        f"increasing max_lag or treating this range estimate "
                        f"with caution."
                    )
                    self.warnings.append(msg)

    def get_param_percentiles(
        self,
        percentiles: List[float] = [16, 50, 84]
    ) -> Dict[str, np.ndarray]:
        """Get parameter percentiles from bootstrap samples.

        Returns
        -------
        percentiles_dict : dict
            Keys are parameter names, values are arrays of percentiles.
        """
        if self.param_samples is None:
            raise ValueError("No bootstrap samples available.")

        result = {}
        for i, name in enumerate(self.composite_model.param_names):
            result[name] = np.percentile(self.param_samples[:, i], percentiles)

        return result


@dataclass
class KrigingLOOCVResult:
    """Results from leave-one-out kriging cross-validation.

    The primary diagnostic is ``msspe`` (Mean Standardized Squared
    Prediction Error), which should be ≈ 1.0 for a well-calibrated
    variogram model.  Think of it like a χ² / df ratio — values near
    1.0 mean the variogram correctly captures the spatial uncertainty;
    MSSPE >> 1 means the model underestimates uncertainty; MSSPE << 1
    means it overestimates.

    Attributes
    ----------
    msspe : float
        Mean Standardized Squared Prediction Error:
        (1/n) Σ (z_i − ẑ₋ᵢ)² / σ²₋ᵢ.   Target: ≈ 1.0.
    mean_error : float
        Mean prediction error (bias):  (1/n) Σ (z_i − ẑ₋ᵢ).
        Target: ≈ 0.0.
    rmse : float
        Root mean squared prediction error.
    mean_standardized_error : float
        Mean of (z_i − ẑ₋ᵢ) / σ₋ᵢ.   Target: ≈ 0.0.
    n_points : int
        Number of points used (after subsampling and excluding failures).
    n_failed : int
        Number of LOO iterations that failed (singular matrix, etc.).

    References
    ----------
    Cressie, N. (1993). Statistics for Spatial Data, rev. ed., Wiley.
        Section 5.6 — kriging cross-validation.

    Webster, R. & Oliver, M.A. (2007). Geostatistics for Environmental
        Scientists, 2nd ed., Wiley.  Section 8.3.

    Lark, R.M. (2000). A comparison of some robust estimators of the
        variogram for use in soil survey.  Eur. J. Soil Sci., 51,
        137–157.  Uses MSSPE ≈ 1.0 as acceptance criterion.
    """

    msspe: float
    mean_error: float
    rmse: float
    mean_standardized_error: float
    n_points: int
    n_failed: int


@dataclass
class AggregatedLOOCVResult:
    """Aggregated diagnostics from repeated kriging LOOCV runs.

    Each field stores summary statistics (mean, median, std, nmad)
    across ``n_runs`` independent random subsamples.  The spread
    quantifies how sensitive each diagnostic is to the particular
    subsample drawn — large spread relative to ``|msspe_mean − 1|``
    suggests the ranking may not be stable.

    Attributes
    ----------
    n_runs : int
        Number of successful LOOCV runs.
    total_points : int
        Total number of LOOCV predictions across all runs.
    msspe_mean, msspe_median, msspe_std, msspe_nmad : float
        Summary statistics of MSSPE across runs.
    mean_error_mean, mean_error_median, mean_error_std, mean_error_nmad : float
        Summary statistics of mean prediction error (bias).
    rmse_mean, rmse_median, rmse_std, rmse_nmad : float
        Summary statistics of RMSE.
    mse_mean, mse_median, mse_std, mse_nmad : float
        Summary statistics of mean standardized error.
    n_failed_total : int
        Total LOO iterations that failed across all runs.
    per_run_results : list of KrigingLOOCVResult
        Individual run results for detailed inspection.
    """

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
    def from_results(results: List[KrigingLOOCVResult]) -> 'AggregatedLOOCVResult':
        """Compute aggregate statistics from a list of per-run results.

        Parameters
        ----------
        results : list of KrigingLOOCVResult
            Individual LOOCV results, one per subsample run.

        Returns
        -------
        AggregatedLOOCVResult
        """
        def _nmad(arr: np.ndarray) -> float:
            """Normalized Median Absolute Deviation (robust σ estimate)."""
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


@dataclass
class EnsembleVariogramResult:
    """Results from ensemble variogram fitting across independent spatial samples.

    Each realization draws an independent random sample from the raster,
    computes an empirical variogram, runs the full model selection pipeline
    (all candidate model types, WLS fitting, MSSPE-based selection), and
    records the winning model's structure and parameters.  The ensemble
    captures both *model selection uncertainty* (does the pipeline always
    pick the same model family?) and *parameter uncertainty* (how stable
    are the fitted sills, ranges, nuggets, and Matérn ν?).

    This is conceptually similar to a parametric bootstrap, but instead
    of perturbing a single empirical variogram, it re-samples the spatial
    field — making it a Monte Carlo assessment of the entire pipeline
    from sampling through model selection.

    Attributes
    ----------
    n_realizations : int
        Number of independent variogram realizations fitted.
    n_failed : int
        Realizations where no model could be fitted.
    model_counts : dict
        {model_description: count} — how often each model structure
        was selected (e.g. ``{'spherical + nugget': 35, 'exponential + matern + nugget': 15}``).
    model_fractions : dict
        {model_description: fraction} — selection frequency as proportion.
    sills : ndarray, shape (n_success, max_n_sills)
        Per-realization sill values (NaN-padded where models have
        fewer sill components).
    ranges : ndarray, shape (n_success, max_n_ranges)
        Per-realization range values (NaN-padded).
    nuggets : ndarray, shape (n_success,)
        Per-realization nugget values (NaN where model has no nugget).
    nus : ndarray, shape (n_success,)
        Per-realization Matérn ν values (NaN for non-Matérn models).
    msspes : ndarray, shape (n_success,)
        Per-realization MSSPE of the selected model.
    lags : ndarray
        Common lag vector (from the first successful realization).
    variograms : ndarray, shape (n_success, n_lags)
        Per-realization fitted variogram curves evaluated at ``lags``.
    empirical_variograms : ndarray, shape (n_success, n_lags)
        Per-realization empirical variograms (NaN-padded to common length).
    per_realization : list of dict
        Detailed per-realization records including model description,
        all parameters, component names, MSSPE, AIC, and fitted curve.

    References
    ----------
    Lark, R.M. (2000). A comparison of some robust estimators of the
    variogram for use in soil survey. *Eur. J. Soil Sci.*, 51, 137–157.

    Marchetti, Y. et al. (2018). An assessment of model selection
    uncertainty in spatial prediction. *Environmetrics*, 29(7–8),
    e2530. doi:10.1002/env.2530
    """

    n_realizations: int
    n_failed: int
    model_counts: Dict[str, int]
    model_fractions: Dict[str, float]
    sills: np.ndarray
    ranges: np.ndarray
    nuggets: np.ndarray
    nus: np.ndarray
    msspes: np.ndarray
    lags: np.ndarray
    variograms: np.ndarray
    empirical_variograms: np.ndarray
    per_realization: List[Dict[str, Any]]

    def summary(self) -> str:
        """Human-readable summary of ensemble results."""
        lines = []
        lines.append("=" * 72)
        lines.append("ENSEMBLE VARIOGRAM FITTING RESULTS")
        lines.append("=" * 72)
        n_ok = self.n_realizations - self.n_failed
        lines.append(
            f"Realizations: {self.n_realizations} total, "
            f"{n_ok} successful, {self.n_failed} failed"
        )
        lines.append("")

        # model selection frequency
        lines.append("MODEL SELECTION FREQUENCY")
        lines.append("-" * 50)
        for desc, frac in sorted(
            self.model_fractions.items(), key=lambda x: -x[1]
        ):
            cnt = self.model_counts[desc]
            lines.append(f"  {desc:<40s}  {cnt:3d} ({frac:5.1%})")
        lines.append("")

        # parameter summaries
        lines.append("PARAMETER SUMMARY (median [16th, 84th percentile])")
        lines.append("-" * 50)

        def _summarize(arr, name):
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return
            med = np.median(valid)
            p16 = np.percentile(valid, 16)
            p84 = np.percentile(valid, 84)
            lines.append(f"  {name:<20s}  {med:10.4f}  [{p16:.4f}, {p84:.4f}]")

        # sills
        for j in range(self.sills.shape[1] if self.sills.ndim > 1 else 0):
            _summarize(self.sills[:, j], f"Sill {j + 1}")

        # ranges
        for j in range(self.ranges.shape[1] if self.ranges.ndim > 1 else 0):
            _summarize(self.ranges[:, j], f"Range {j + 1}")

        _summarize(self.nuggets, "Nugget")
        _summarize(self.nus, "Matérn ν")
        lines.append("")

        # MSSPE
        valid_msspe = self.msspes[np.isfinite(self.msspes)]
        if len(valid_msspe) > 0:
            lines.append("MSSPE SUMMARY")
            lines.append("-" * 50)
            lines.append(
                f"  Median: {np.median(valid_msspe):.4f}  "
                f"Mean: {np.mean(valid_msspe):.4f}  "
                f"Std: {np.std(valid_msspe):.4f}"
            )
            lines.append(
                f"  [16th, 84th]: [{np.percentile(valid_msspe, 16):.4f}, "
                f"{np.percentile(valid_msspe, 84):.4f}]"
            )
        lines.append("=" * 72)
        return "\n".join(lines)

    def plot(self, figsize=(12, 10)):
        """Plot ensemble variogram results.

        Creates a 3-panel figure:
          - Top: model selection frequency bar chart
          - Middle: median fitted variogram ± 16th/84th percentile
            envelope, with individual empirical variograms in light gray
          - Bottom: parameter distributions (sills, ranges, nugget, ν)

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from collections import Counter

        n_ok = self.n_realizations - self.n_failed

        fig, axes = plt.subplots(3, 1, figsize=figsize,
                                 gridspec_kw={'height_ratios': [1, 2.5, 1.5]})

        # ── Panel 1: model selection frequency ──
        ax = axes[0]
        sorted_models = sorted(
            self.model_counts.items(), key=lambda x: -x[1]
        )
        names = [m[0] for m in sorted_models]
        counts = [m[1] for m in sorted_models]
        bars = ax.barh(range(len(names)), counts, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Times selected')
        ax.set_title(f'Model selection frequency (n={n_ok})')
        ax.invert_yaxis()

        # ── Panel 2: median variogram + envelope ──
        ax = axes[1]
        lags = self.lags

        # individual empirical variograms (light gray)
        for i in range(self.empirical_variograms.shape[0]):
            ev = self.empirical_variograms[i]
            valid = np.isfinite(ev)
            if np.any(valid):
                ax.plot(lags[valid], ev[valid], color='gray', alpha=0.1,
                        linewidth=0.5)

        # fitted variogram envelope
        vario_arr = self.variograms
        with np.errstate(all='ignore'):
            median_curve = np.nanmedian(vario_arr, axis=0)
            p16_curve = np.nanpercentile(vario_arr, 16, axis=0)
            p84_curve = np.nanpercentile(vario_arr, 84, axis=0)
            p025_curve = np.nanpercentile(vario_arr, 2.5, axis=0)
            p975_curve = np.nanpercentile(vario_arr, 97.5, axis=0)

        # also plot median empirical variogram
        with np.errstate(all='ignore'):
            median_emp = np.nanmedian(self.empirical_variograms, axis=0)
        valid_emp = np.isfinite(median_emp)
        ax.plot(lags[valid_emp], median_emp[valid_emp], 'ko', markersize=4,
                alpha=0.6, label='Median empirical')

        valid_fit = np.isfinite(median_curve)
        ax.fill_between(lags[valid_fit], p025_curve[valid_fit],
                        p975_curve[valid_fit], color='steelblue',
                        alpha=0.1, label='95% envelope')
        ax.fill_between(lags[valid_fit], p16_curve[valid_fit],
                        p84_curve[valid_fit], color='steelblue',
                        alpha=0.3, label='68% envelope')
        ax.plot(lags[valid_fit], median_curve[valid_fit], 'b-',
                linewidth=2, label='Median fitted')

        ax.set_xlabel('Lag distance')
        ax.set_ylabel('Semivariance')
        ax.set_title('Ensemble variogram')
        ax.legend(loc='lower right', fontsize=8)

        # ── Panel 3: parameter distributions ──
        ax = axes[2]
        # Collect all non-NaN parameter arrays for box plots
        box_data = []
        box_labels = []

        for j in range(self.sills.shape[1] if self.sills.ndim > 1 else 0):
            vals = self.sills[:, j]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                box_data.append(valid)
                box_labels.append(f'Sill {j + 1}')

        valid_nug = self.nuggets[np.isfinite(self.nuggets)]
        if len(valid_nug) > 0:
            box_data.append(valid_nug)
            box_labels.append('Nugget')

        valid_nu = self.nus[np.isfinite(self.nus)]
        if len(valid_nu) > 0:
            box_data.append(valid_nu)
            box_labels.append('Matérn ν')

        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, vert=True,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(markersize=3, alpha=0.4))
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.5)
            ax.set_ylabel('Parameter value')
            ax.set_title('Parameter distributions (sills, nugget, ν)')

            # add a secondary axis for ranges (different scale)
            range_data = []
            range_labels = []
            for j in range(self.ranges.shape[1] if self.ranges.ndim > 1 else 0):
                vals = self.ranges[:, j]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    range_data.append(valid)
                    range_labels.append(f'Range {j + 1}')

            if range_data:
                ax2 = ax.twinx()
                positions = list(range(
                    len(box_data) + 1,
                    len(box_data) + 1 + len(range_data),
                ))
                bp2 = ax2.boxplot(
                    range_data, positions=positions,
                    labels=range_labels, vert=True,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(markersize=3, alpha=0.4),
                )
                for patch in bp2['boxes']:
                    patch.set_facecolor('coral')
                    patch.set_alpha(0.5)
                ax2.set_ylabel('Range value', color='coral')
                ax.set_xlim(0.5, len(box_data) + len(range_data) + 0.5)
                all_labels = box_labels + range_labels
                all_positions = list(range(1, len(box_data) + 1)) + positions
                ax.set_xticks(all_positions)
                ax.set_xticklabels(all_labels, fontsize=8, rotation=15)

        plt.tight_layout()
        return fig


class VariogramModelSelector:
    """Select and fit optimal variogram model from candidates.

    This class implements:
    - Automatic generation of candidate nested models
    - Fitting via weighted least squares
    - Model selection via AIC, BIC, and cross-validation
    - Bootstrap parameter uncertainty
    - Bayesian Model Averaging (optional)
    
    Parameters
    ----------
    lags : ndarray
        Lag distances from empirical variogram.
    empirical_variogram : ndarray
        Empirical semivariance values.
    pair_counts : ndarray, optional
        Number of point pairs per lag bin.  Required for ``'cressie'``
        and ``'pair_count'`` weighting schemes, and for ``min_pairs``
        filtering.
    sigma : ndarray, optional
        Per-bin standard deviations for likelihood-based AIC/BIC.
    weighting : {'cressie', 'pair_count', 'uniform'}, default ``'cressie'``
        WLS weighting scheme.  ``'cressie'`` computes w_i = N(h)/γ̂(h)²
        (Cressie, 1985).  ``'pair_count'`` uses raw pair counts.
        ``'uniform'`` sets all weights to 1.
    min_pairs : int or None, default 30
        Minimum number of point pairs required for a lag bin to be
        included in model fitting.  Bins with fewer pairs receive
        zero weight and are effectively excluded from the WLS fit.
        Set to ``None`` or ``0`` to disable filtering.  Requires
        ``pair_counts`` to be supplied; ignored otherwise.

        The literature recommends thresholds of 30–50
        (Cressie, 1985; Oliver & Webster, 2014).

    Examples
    --------
    >>> selector = VariogramModelSelector(lags, gamma, pair_counts=counts)
    >>> selector.fit_all_candidates(max_components=2, include_nugget=True)
    >>> best = selector.select_best(criterion='aic')
    >>> print(best.composite_model.description())
    
    References
    ----------
    Cressie, N. (1985). Fitting variogram models by weighted least squares.
    J. Int. Assoc. Math. Geol., 17(5), 563–586.

    Oliver, M.A. & Webster, R. (2014). A tutorial guide to geostatistics.
    Catena, 113, 56–69.

    Webster, R. & McBratney, A.B. (1989). On the Akaike Information Criterion
    for choosing models for variograms of soil properties. Eur. J. Soil Sci.
    """
    
    # models to include in candidate generation
    BOUNDED_MODELS = ['spherical', 'exponential', 'gaussian', 'matern']
    UNBOUNDED_MODELS = ['power', 'linear']
    
    # supported weighting schemes for WLS fitting
    WEIGHTING_SCHEMES = ('cressie', 'pair_count', 'uniform')

    def __init__(
        self,
        lags: np.ndarray,
        empirical_variogram: np.ndarray,
        pair_counts: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        weighting: str = 'cressie',
        min_pairs: Optional[int] = 30,
    ):
        self.lags = np.asarray(lags, dtype=float)
        self.empirical_variogram = np.asarray(empirical_variogram, dtype=float)

        if pair_counts is not None:
            self.pair_counts = np.asarray(pair_counts, dtype=float)
        else:
            self.pair_counts = None

        if weighting not in self.WEIGHTING_SCHEMES:
            raise ValueError(
                f"Unknown weighting scheme '{weighting}'. "
                f"Choose from {self.WEIGHTING_SCHEMES}."
            )
        self.weighting = weighting
        self.min_pairs = min_pairs if min_pairs else 0

        # compute WLS weights
        self.weights = self._compute_weights(
            self.empirical_variogram, self.pair_counts, weighting
        )

        # apply minimum pair-count filter: zero-weight bins with too few pairs
        self._n_filtered = 0
        if self.min_pairs > 0 and self.pair_counts is not None:
            low_count_mask = self.pair_counts < self.min_pairs
            self._n_filtered = int(np.sum(low_count_mask))
            if self._n_filtered > 0:
                self.weights[low_count_mask] = 0.0
                n_remaining = len(self.lags) - self._n_filtered
                if n_remaining < 4:
                    import warnings as _w
                    _w.warn(
                        f"min_pairs={self.min_pairs} filters {self._n_filtered} "
                        f"of {len(self.lags)} lag bins, leaving only "
                        f"{n_remaining}. Model fitting may fail or be "
                        f"unreliable. Consider lowering min_pairs.",
                        UserWarning,
                        stacklevel=2,
                    )

        # standard deviation per bin (for likelihood-based criteria)
        if sigma is not None:
            self.sigma = np.asarray(sigma, dtype=float)
            self.sigma = np.where(self.sigma <= 0, np.finfo(float).eps, self.sigma)
        else:
            self.sigma = None

        self.fitted_models: List[FittedVariogramModel] = []
        self.best_model: Optional[FittedVariogramModel] = None
        self.model_weights: Optional[np.ndarray] = None  # Akaike weights

    @staticmethod
    def _compute_weights(
        empirical_variogram: np.ndarray,
        pair_counts: Optional[np.ndarray],
        weighting: str,
    ) -> np.ndarray:
        """Compute WLS weights for variogram fitting.

        Parameters
        ----------
        empirical_variogram : ndarray
            Empirical semivariance values.
        pair_counts : ndarray or None
            Number of point pairs per lag bin.
        weighting : str
            ``'cressie'``  – N(h) / γ̂(h)²  (Cressie, 1985).
            ``'pair_count'`` – N(h) only.
            ``'uniform'``  – all weights equal to 1.

        Returns
        -------
        weights : ndarray

        References
        ----------
        Cressie, N. (1985). Fitting variogram models by weighted least
        squares. J. Int. Assoc. Math. Geol., 17(5), 563–586.
        """
        n = len(empirical_variogram)

        if weighting == 'uniform' or pair_counts is None:
            if weighting == 'cressie' and pair_counts is None:
                import warnings as _w
                _w.warn(
                    "Cressie weighting requested but no pair_counts supplied; "
                    "falling back to uniform weights.",
                    UserWarning,
                    stacklevel=3,
                )
            return np.ones(n, dtype=float)

        counts = np.asarray(pair_counts, dtype=float)

        if weighting == 'pair_count':
            return counts

        # --- Cressie weighting:  w_i = N(h_i) / γ̂(h_i)² ---
        gamma_sq = np.square(empirical_variogram)
        # guard against division by zero at lags where γ̂ ≈ 0
        gamma_sq = np.where(gamma_sq < np.finfo(float).eps, np.finfo(float).eps, gamma_sq)
        return counts / gamma_sq

    def generate_candidates(
        self,
        max_components: int = 2,
        include_nugget: bool = True,
        include_unbounded: bool = True,
        bounded_only_combinations: bool = True,
    ) -> List[CompositeVariogramModel]:
        """Generate candidate composite models.

        When ``include_nugget=True``, every candidate is generated in both
        a with-nugget and a without-nugget variant so that model selection
        can decide whether an explicit nugget is needed.

        The Matérn model's smoothness parameter ν already controls the
        shape at short lags, making multiple Matérn components redundant
        and poorly identifiable.  Combinations containing more than one
        Matérn component are therefore excluded.

        Parameters
        ----------
        max_components : int
            Maximum number of component models to combine.
        include_nugget : bool
            Whether to include nugget variants.  When True, multi-component
            models are generated both with and without nugget; single-
            component models always include the nugget.
        include_unbounded : bool
            Whether to include unbounded (non-stationary) models.
        bounded_only_combinations : bool
            If True, only combine bounded models together.
            Unbounded models are only tried as single components.

        Returns
        -------
        candidates : List[CompositeVariogramModel]
            List of candidate composite models.

        References
        ----------
        Stein, M.L. (1999). *Interpolation of Spatial Data: Some Theory
        for Kriging*. Springer.  Argues that the Matérn class subsumes
        exponential (ν=0.5) and Gaussian (ν→∞), making nested Matérn
        structures redundant for kriging.
        """
        candidates = []

        # get model lists
        bounded = self.BOUNDED_MODELS
        unbounded = self.UNBOUNDED_MODELS if include_unbounded else []

        # generate bounded-only combinations
        for n in range(1, max_components + 1):
            for combo in combinations_with_replacement(bounded, n):
                combo_list = list(combo)

                # ── block multiple Matérns (Stein, 1999) ──
                # A single Matérn's ν parameter already controls
                # smoothness; stacking Matérns creates redundant,
                # poorly identifiable parameters.  Allow at most one.
                if combo_list.count('matern') > 1:
                    continue

                # ── nugget variants ──
                # When include_nugget is True, generate both with-
                # and without-nugget variants so the model selection
                # can decide whether an explicit nugget is needed or
                # whether a short-range component can absorb the
                # micro-scale variance.
                if include_nugget:
                    nugget_options = [True, False]
                else:
                    nugget_options = [False]

                for use_nugget in nugget_options:
                    try:
                        model = CompositeVariogramModel(
                            combo_list,
                            include_nugget=use_nugget,
                        )
                        candidates.append(model)
                    except ValueError:
                        continue

        # add single unbounded models (not in combinations to preserve validity)
        if include_unbounded:
            for name in unbounded:
                try:
                    model = CompositeVariogramModel(
                        [name],
                        include_nugget=True,  # always include nugget
                    )
                    candidates.append(model)
                except ValueError:
                    continue

            # optionally: bounded + unbounded combinations
            if not bounded_only_combinations:
                for bounded_name in bounded:
                    for unbounded_name in unbounded:
                        nugget_options = [True, False] if include_nugget else [False]
                        for use_nugget in nugget_options:
                            try:
                                model = CompositeVariogramModel(
                                    [bounded_name, unbounded_name],
                                    include_nugget=use_nugget,
                                )
                                candidates.append(model)
                            except ValueError:
                                continue

        return candidates
    
    # ── nugget pre-estimation (two-stage approach) ──────────────────

    @staticmethod
    def _estimate_nugget_from_short_lags(
        lags: np.ndarray,
        variogram: np.ndarray,
        n_lags: int = 5,
    ) -> float:
        """Pre-estimate nugget by extrapolating short-lag bins to h=0.

        Fits γ(h) ≈ C₀ + bh to the first few lags and returns the
        y-intercept clamped to [0, 0.5 × max(γ)].  This is the "Stage 1"
        of two-stage fitting and prevents the optimizer from trading sill
        for nugget inappropriately.

        References
        ----------
        Cressie, N. (1985). Fitting variogram models by weighted least
        squares.  J. Int. Assoc. Math. Geol., 17(5), 563–586.
        """
        max_gamma = float(np.nanmax(variogram))
        n_fit = min(n_lags, max(2, len(lags) // 4))
        if n_fit < 2:
            return max_gamma * 0.1

        short_lags = lags[:n_fit]
        short_gamma = variogram[:n_fit]

        try:
            _slope, intercept = np.polyfit(short_lags, short_gamma, 1)
            return float(np.clip(intercept, 0.0, max_gamma * 0.5))
        except (np.linalg.LinAlgError, ValueError):
            return max_gamma * 0.1

    # ── multi-start initial guesses ──────────────────────────────

    @staticmethod
    def _generate_multistart_guesses(
        p0_base: np.ndarray,
        bounds: tuple,
        n_restarts: int,
        rng: np.random.Generator,
    ) -> list:
        """Generate diverse starting points using Latin-Hypercube-like sampling.

        Restart 0:  use the default guess unchanged.
        Restart 1:  halve all range parameters (explore short-range basin).
        Restart 2:  double all range parameters (explore long-range basin).
        Remaining:  random uniform samples in [lower, upper] for each param.
        """
        lb = np.asarray(bounds[0], dtype=float)
        ub = np.asarray(bounds[1], dtype=float)
        guesses = [p0_base.copy()]

        if n_restarts >= 2:
            # short-range variant
            short = p0_base.copy()
            short = np.clip(short * 0.5, lb, ub)
            guesses.append(short)

        if n_restarts >= 3:
            # long-range variant
            long_ = p0_base.copy()
            long_ = np.clip(long_ * 2.0, lb, ub)
            guesses.append(long_)

        # fill remaining restarts with random samples in the feasible region
        for _ in range(max(0, n_restarts - len(guesses))):
            # uniform random between lb and ub (log-scale for wide bounds)
            rand = rng.random(len(p0_base))
            # use log-uniform for strictly positive params with wide range
            sample = np.empty_like(p0_base)
            for j in range(len(p0_base)):
                lo, hi = max(lb[j], 1e-12), ub[j]
                if hi / lo > 50:
                    # log-uniform sampling for wide-range params
                    sample[j] = np.exp(
                        np.log(lo) + rand[j] * (np.log(hi) - np.log(lo))
                    )
                else:
                    sample[j] = lo + rand[j] * (hi - lo)
            guesses.append(np.clip(sample, lb, ub))

        return guesses[:n_restarts]

    # ── range separation enforcement ─────────────────────────────

    #: Minimum ratio between successive practical ranges in a
    #: multi-component model.  Two components whose practical ranges
    #: are closer than this factor are nearly non-identifiable:
    #: the optimizer can trade sill between them freely.  A 3×
    #: separation ensures each component captures a distinct spatial
    #: scale, consistent with the physical error-source hierarchy
    #: in topographic differencing (meter-scale misclassification →
    #: hundred-meter-scale flight-line striping → kilometre-scale
    #: calibration bias).
    #:
    #: No standard numerical rule exists in the literature; the
    #: constraint is a practical identifiability guard.  Gringarten
    #: & Deutsch (2001) and Webster & Oliver (2007) recommend that
    #: nested structures represent "distinct scales" without
    #: specifying a ratio.  3× is conservative: with spherical
    #: models, two components with ranges a and 3a have their
    #: transition zones overlapping in only the first third of the
    #: longer component's range — enough for the optimizer to
    #: distinguish them.
    MIN_RANGE_SEPARATION: float = 3.0

    @staticmethod
    def _get_practical_ranges(
        model: CompositeVariogramModel,
        params: np.ndarray,
    ) -> List[float]:
        """Extract practical (effective) ranges from fitted params.

        The practical range is the distance at which the model reaches
        ~95% of its sill.  For spherical models this equals the range
        parameter; for exponential models it is 3× the range parameter.

        Returns a list of practical ranges for bounded components that
        have a 'range' parameter, in component order.
        """
        model.set_params(params)
        practical = []
        for i, spec in enumerate(model._components):
            if spec.is_bounded and 'range' in spec.param_names:
                range_idx = spec.param_names.index('range')
                comp_params = model.get_component_params(i)
                raw_range = comp_params[range_idx]
                prf = spec.practical_range_factor
                if prf is not None and prf > 0:
                    practical.append(raw_range * prf)
                else:
                    practical.append(raw_range)
        return practical

    def _passes_range_separation(
        self,
        model: CompositeVariogramModel,
        params: np.ndarray,
    ) -> bool:
        """Check whether fitted practical ranges satisfy separation.

        For single-component models this always returns True.
        For multi-component models, the sorted practical ranges must
        satisfy  r_{i+1} / r_i  >=  MIN_RANGE_SEPARATION.
        """
        practical = self._get_practical_ranges(model, params)
        if len(practical) <= 1:
            return True
        ordered = sorted(practical)
        for j in range(len(ordered) - 1):
            if ordered[j] < np.finfo(float).eps:
                return False  # degenerate zero range
            if ordered[j + 1] / ordered[j] < self.MIN_RANGE_SEPARATION:
                return False
        return True

    #: Maximum allowed total sill as a multiple of the observed
    #: maximum semivariance.  With multiple freely-parameterised
    #: components the optimizer can inflate one component's sill to
    #: compensate for another, producing a total sill far exceeding
    #: the data.  2× allows some headroom for noise while rejecting
    #: physically unrealistic fits (Gringarten & Deutsch, 2001).
    MAX_SILL_RATIO: float = 2.0

    def _passes_total_sill_check(
        self,
        model: CompositeVariogramModel,
        params: np.ndarray,
    ) -> bool:
        """Reject fits whose total sill exceeds MAX_SILL_RATIO × max(γ̂).

        The total sill is the sum of all component sills plus the
        nugget.  For stationary models this equals the process
        variance at infinite lag.  A total sill far exceeding the
        observed semivariance plateau is physically unrealistic.
        """
        model.set_params(params)
        if not model.is_stationary:
            return True  # unbounded models have no finite sill
        total_sill = model.get_total_sill()
        if total_sill is None:
            return True
        max_gamma = float(np.nanmax(self.empirical_variogram))
        return total_sill <= self.MAX_SILL_RATIO * max_gamma

    # ── core model fitting ───────────────────────────────────────

    def fit_model(
        self,
        model: CompositeVariogramModel,
        maxfev: int = 10000,
        n_restarts: int = 8,
    ) -> Optional[FittedVariogramModel]:
        """Fit a single composite model to the empirical variogram.

        When the model includes a nugget, a two-stage approach is used:
        the nugget is first pre-estimated from short-lag extrapolation,
        then the full optimisation is run with the nugget constrained
        in an asymmetric window around the pre-estimate (−50% / +80%,
        with a floor of 15% of max semivariance).  This prevents the
        common failure mode where the optimizer trades sill for nugget.

        For multi-component models, fits that violate the minimum
        practical-range separation (``MIN_RANGE_SEPARATION``, default
        3×) are rejected.  This prevents the optimizer from collapsing
        two components onto the same spatial scale, which causes
        parameter non-identifiability.

        Parameters
        ----------
        model : CompositeVariogramModel
            Model to fit.
        maxfev : int
            Maximum function evaluations for optimizer.
        n_restarts : int
            Number of random restarts to avoid local minima.

        Returns
        -------
        fitted : FittedVariogramModel or None
            Fitted model, or None if fitting failed.
        """
        # get initial guess and bounds
        p0_base = model.default_guess(self.lags, self.empirical_variogram)
        bounds = model.bounds(self.lags, self.empirical_variogram)

        # ── nugget bounds ──
        # Use the full [0, 0.5 × max_γ] range from the composite model
        # bounds rather than the previous tight asymmetric constraint
        # (−50%/+80% around short-lag extrapolation with a 15% floor).
        # The tight bounds locked the nugget too low when the first few
        # lags still contained spatial structure, causing MSSPE >> 1
        # (kriging variance underestimation).
        #
        # The initial guess is still informed by short-lag extrapolation
        # for a reasonable starting point, but the optimizer is free to
        # explore the full feasible range.
        if model.include_nugget:
            nugget_pre = self._estimate_nugget_from_short_lags(
                self.lags, self.empirical_variogram
            )
            nugget_idx = model.n_params - 1  # nugget is always last
            # bounds already set to [0, 0.5 * max_gamma] by
            # CompositeVariogramModel.bounds(); no further tightening
            p0_base[nugget_idx] = np.clip(
                nugget_pre, bounds[0][nugget_idx], bounds[1][nugget_idx]
            )

        # prepare fitting function
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
                    model_func,
                    self.lags,
                    self.empirical_variogram,
                    p0=p0,
                    sigma=self.sigma,
                    absolute_sigma=True if self.sigma is not None else False,
                    bounds=bounds,
                    maxfev=maxfev,
                )

                # ── range separation check ──
                # Reject fits where multi-component practical ranges
                # are not sufficiently separated (non-identifiable).
                if not self._passes_range_separation(model, popt):
                    continue

                # ── total sill check ──
                # Reject fits where the total sill (all component
                # sills + nugget) exceeds MAX_SILL_RATIO × max(γ̂).
                if not self._passes_total_sill_check(model, popt):
                    continue

                # compute RSS
                model.set_params(popt)
                residuals = self.empirical_variogram - model(self.lags)
                rss = np.sum(self.weights * residuals**2)

                if rss < best_rss:
                    best_rss = rss
                    best_result = (popt, pcov, rss)

            except (RuntimeError, ValueError):
                continue

        if best_result is None:
            return None

        popt, pcov, rss = best_result
        model.set_params(popt)

        # compute information criteria
        n = len(self.lags)
        k = model.n_params

        if self.sigma is not None:
            ll = self._log_likelihood(model, popt)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
        else:
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)

        return FittedVariogramModel(
            composite_model=model,
            params=popt,
            param_cov=pcov,
            rss=rss,
            aic=aic,
            bic=bic,
        )
    
    def _log_likelihood(
        self, 
        model: CompositeVariogramModel, 
        params: np.ndarray
    ) -> float:
        """Compute log-likelihood assuming Gaussian errors."""
        model.set_params(params)
        predicted = model(self.lags)
        residuals = self.empirical_variogram - predicted
        
        # heteroscedastic Gaussian log-likelihood
        ll = -0.5 * np.sum(
            np.log(2 * np.pi * self.sigma**2) + 
            (residuals**2) / (self.sigma**2)
        )
        return ll
    
    def cross_validate(
        self,
        fitted_model: FittedVariogramModel,
        k: int = 5,
        seed: Optional[int] = None,
    ) -> float:
        """Compute k-fold cross-validation RMSE.
        
        Parameters
        ----------
        fitted_model : FittedVariogramModel
            Model to evaluate.
        k : int
            Number of folds.
        seed : int, optional
            Random seed for fold assignment.
        
        Returns
        -------
        cv_rmse : float
            Cross-validation RMSE.
        """
        rng = np.random.default_rng(seed)
        n = len(self.lags)
        indices = rng.permutation(n)
        fold_size = max(1, n // k)
        
        squared_errors = []
        
        for i in range(k):
            # split indices
            val_idx = indices[i * fold_size: min((i + 1) * fold_size, n)]
            train_idx = np.setdiff1d(indices, val_idx)
            
            if len(train_idx) < fitted_model.composite_model.n_params:
                continue
            
            # fit on training set
            model_copy = CompositeVariogramModel(
                fitted_model.composite_model.component_names,
                fitted_model.composite_model.include_nugget,
            )
            
            def model_func(h, *params):
                model_copy.set_params(np.array(params))
                return model_copy(h)
            
            try:
                popt, _ = curve_fit(
                    model_func,
                    self.lags[train_idx],
                    self.empirical_variogram[train_idx],
                    p0=fitted_model.params,
                    bounds=model_copy.bounds(self.lags, self.empirical_variogram),
                    maxfev=5000,
                )
                
                # predict on validation set
                model_copy.set_params(popt)
                predictions = model_copy(self.lags[val_idx])
                errors = self.empirical_variogram[val_idx] - predictions
                squared_errors.extend(errors**2)
                
            except RuntimeError:
                continue
        
        if not squared_errors:
            return np.inf
        
        return np.sqrt(np.mean(squared_errors))
    
    def fit_all_candidates(
        self,
        max_components: int = 2,
        include_nugget: bool = True,
        include_unbounded: bool = True,
        compute_cv: bool = False,
        cv_folds: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        """Fit all candidate models.

        Parameters
        ----------
        max_components : int
            Maximum number of nested components.
        include_nugget : bool
            Include nugget in all models.
        include_unbounded : bool
            Include non-stationary models.
        compute_cv : bool
            Whether to compute variogram k-fold CV scores.
            Default False — variogram k-fold CV randomly shuffles
            lag bins, which are ordered and correlated, making the
            resulting RMSE unreliable for model selection.  Use
            kriging LOOCV (MSSPE) instead for spatial cross-
            validation.  Retained for backward compatibility.
        cv_folds : int
            Number of CV folds (only used if ``compute_cv=True``).
        seed : int, optional
            Random seed for CV.
        """
        candidates = self.generate_candidates(
            max_components=max_components,
            include_nugget=include_nugget,
            include_unbounded=include_unbounded,
        )

        self.fitted_models = []
        max_lag = float(np.nanmax(self.lags))

        for model in candidates:
            fitted = self.fit_model(model)
            if fitted is not None:
                if compute_cv:
                    fitted.cv_rmse = self.cross_validate(fitted, k=cv_folds, seed=seed)
                # check half-lag heuristic
                fitted.check_half_lag(max_lag)
                self.fitted_models.append(fitted)

        # compute Akaike weights
        if self.fitted_models:
            self._compute_akaike_weights()
    
    def _compute_akaike_weights(self) -> None:
        """Compute Akaike weights for model averaging."""
        aics = np.array([m.aic for m in self.fitted_models])
        delta_aic = aics - np.min(aics)
        
        # akaike weights
        exp_terms = np.exp(-0.5 * delta_aic)
        self.model_weights = exp_terms / np.sum(exp_terms)
    
    def select_best(self, criterion: str = 'aic') -> FittedVariogramModel:
        """Select best model by criterion.

        Parameters
        ----------
        criterion : str
            Selection criterion: ``'aic'``, ``'bic'``, ``'cv'``, or
            ``'msspe'``.  For ``'msspe'``, the model whose kriging
            LOOCV MSSPE is closest to 1.0 is selected (minimise
            ``|MSSPE − 1|``).  Requires that MSSPE has been computed
            for each candidate (see ``fit_best_model_auto`` with
            ``criterion='msspe'`` or ``compute_msspe=True``).

        Returns
        -------
        best : FittedVariogramModel
            Best model according to criterion.

        References
        ----------
        Lark, R.M. (2000).  A comparison of some robust estimators of
        the variogram for use in soil survey.  *Eur. J. Soil Sci.*,
        51, 137–157.  Uses MSSPE ≈ 1.0 as acceptance criterion.
        """
        if not self.fitted_models:
            raise ValueError("No fitted models. Call fit_all_candidates() first.")

        if criterion == 'aic':
            scores = [m.aic for m in self.fitted_models]
        elif criterion == 'bic':
            scores = [m.bic for m in self.fitted_models]
        elif criterion == 'cv':
            scores = [m.cv_rmse if m.cv_rmse is not None else np.inf
                     for m in self.fitted_models]
        elif criterion == 'msspe':
            scores = [
                abs(m.msspe - 1.0) if m.msspe is not None else np.inf
                for m in self.fitted_models
            ]
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        best_idx = np.argmin(scores)
        self.best_model = self.fitted_models[best_idx]

        # emit any diagnostic warnings from the selected model
        import warnings as _warnings
        for w in self.best_model.warnings:
            _warnings.warn(w, UserWarning, stacklevel=2)

        return self.best_model
    
    def bootstrap_best_model(
        self,
        n_boot: int = 500,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Bootstrap parameter uncertainty for best model.

        Generates synthetic variograms by adding noise to the empirical
        variogram and re-fitting the best model.  The bootstrap refit
        uses the same WLS weighting scheme as the original fit so that
        the uncertainty samples faithfully reflect the fitted model.

        Parameters
        ----------
        n_boot : int
            Number of bootstrap samples.
        seed : int, optional
            Random seed.

        Returns
        -------
        param_samples : ndarray
            Bootstrap samples, shape (n_valid, n_params).  Rows where
            fitting failed are removed.
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best() first.")

        rng = np.random.default_rng(seed)
        model = self.best_model.composite_model
        p0 = self.best_model.params
        bounds = model.bounds(self.lags, self.empirical_variogram)

        # generate synthetic variograms
        if self.sigma is not None:
            noise = rng.normal(
                loc=self.empirical_variogram,
                scale=self.sigma,
                size=(n_boot, len(self.lags))
            )
        else:
            # use residual-based bootstrap
            fitted = self.best_model.predict(self.lags)
            residuals = self.empirical_variogram - fitted
            noise = fitted + rng.choice(residuals, size=(n_boot, len(self.lags)))

        # Construct per-bin sigma for bootstrap refits.
        # Must match the weighting used by the original fit so bootstrap
        # samples come from the same optimization landscape.
        # For WLS with weights w_i, curve_fit sigma = 1/sqrt(w_i).
        boot_sigma = None
        if self.weights is not None:
            safe_weights = np.where(
                self.weights > 0, self.weights, np.finfo(float).eps
            )
            boot_sigma = 1.0 / np.sqrt(safe_weights)

        param_samples = []

        def model_func(h, *params):
            model.set_params(np.array(params))
            return model(h)

        for i in range(n_boot):
            try:
                popt, _ = curve_fit(
                    model_func,
                    self.lags,
                    noise[i],
                    p0=p0,
                    sigma=boot_sigma,
                    absolute_sigma=False,
                    bounds=bounds,
                    maxfev=5000,
                )
                param_samples.append(popt)
            except RuntimeError:
                param_samples.append([np.nan] * model.n_params)

        param_samples = np.array(param_samples)

        # remove failed fits
        valid = ~np.isnan(param_samples).any(axis=1)
        param_samples = param_samples[valid]

        self.best_model.param_samples = param_samples
        return param_samples
    
    def get_bma_variogram(self) -> Callable:
        """Get Bayesian Model Averaged variogram function.
        
        Returns
        -------
        bma_func : Callable
            Function γ_BMA(h) = Σ wₖ γₖ(h)
        """
        if self.model_weights is None:
            raise ValueError("No model weights. Call fit_all_candidates() first.")
        
        def bma_variogram(h):
            h = np.asarray(h)
            result = np.zeros_like(h, dtype=float)
            for model, weight in zip(self.fitted_models, self.model_weights):
                result += weight * model.predict(h)
            return result
        
        return bma_variogram
    
    def get_bma_variance_at_lag(self, h: float) -> Tuple[float, float, float]:
        """Get BMA mean and variance at a specific lag.
        
        Returns
        -------
        mean : float
            BMA weighted mean γ_BMA(h)
        within_var : float
            Within-model variance component
        between_var : float
            Between-model variance component
        """
        predictions = np.array([m.predict(np.array([h]))[0] for m in self.fitted_models])
        
        # bMA mean
        bma_mean = np.sum(self.model_weights * predictions)
        
        # between-model variance
        between_var = np.sum(self.model_weights * (predictions - bma_mean)**2)
        
        # within-model variance (would require bootstrap samples from each model)
        # for now, approximate as 0 or could be computed if needed
        within_var = 0.0
        
        return bma_mean, within_var, between_var

    def summary(self) -> str:
        """Generate summary of fitted models."""
        if not self.fitted_models:
            return "No models fitted yet."

        has_msspe = any(m.msspe is not None for m in self.fitted_models)
        has_msspe_std = any(
            getattr(m, 'msspe_std', None) is not None
            for m in self.fitted_models
        )

        lines = ["=" * 95]
        lines.append("VARIOGRAM MODEL SELECTION SUMMARY")
        lines.append("=" * 95)
        header = f"{'Model':<40} {'AIC':>10} {'BIC':>10} {'CV-RMSE':>10}"
        if has_msspe:
            header += f" {'MSSPE':>12}"
        header += f" {'Weight':>8}"
        lines.append(header)
        lines.append("-" * 95)

        # sort by AIC
        weights = (
            self.model_weights
            if self.model_weights is not None
            else [0] * len(self.fitted_models)
        )
        sorted_models = sorted(
            zip(self.fitted_models, weights),
            key=lambda x: x[0].aic
        )

        for model, weight in sorted_models:
            name = "+".join(model.composite_model.component_names)
            if model.composite_model.include_nugget:
                name += "+nugget"

            cv_str = f"{model.cv_rmse:.4f}" if model.cv_rmse else "N/A"
            row = (
                f"{name:<40} {model.aic:>10.2f} {model.bic:>10.2f} "
                f"{cv_str:>10}"
            )
            if has_msspe:
                if model.msspe is not None:
                    msspe_str = f"{model.msspe:.3f}"
                    std = getattr(model, 'msspe_std', None)
                    if std is not None:
                        msspe_str += f"±{std:.3f}"
                else:
                    msspe_str = "N/A"
                row += f" {msspe_str:>12}"
            row += f" {weight:>8.3f}"
            lines.append(row)

        lines.append("=" * 70)

        if self.best_model:
            lines.append("\nBEST MODEL DETAILS:")
            lines.append(self.best_model.composite_model.description())

            if not self.best_model.composite_model.is_stationary:
                lines.append("\nWARNING: Selected model is NON-STATIONARY.")
                lines.append("The process has no finite variance.")
                lines.append("Results are scale-dependent. Consider detrending.")

        return "\n".join(lines)


class RasterDataHandler:
    """
    Load vertical differencing raster data, subtract a vertical systematic error
    from the raster, and randomly sample raster data for further analysis.

    Attributes
    ----------
    raster_path : str
        File path to the raster data.
    unit : str
        Unit of measurement for the raster data (for plotting labels).
    resolution : float
        Nominal raster resolution (linear units).
    rioxarray_obj : rioxarray.DataArray | None
        The rioxarray object holding the raster data.
    data_array : np.ndarray | None
        Loaded raster values as a 1D array of finite pixels.
    samples : np.ndarray | None
        Sampled values from the raster.
    coords : np.ndarray | None
        Coordinates (x, y) of the sampled values.
    bbox : shapely.geometry.Polygon
        Bounding box of the raster.
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
        """
        Compute the precise area covered by valid data in the raster by vectorizing
        the finite/nodata mask into polygon shapes and dissolving them.
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            valid = (~np.isnan(data)) if nodata is None else ((data != nodata) & ~np.isnan(data))
            geoms = shapes(valid.astype(np.uint8), mask=valid, transform=src.transform)
        self.shapely_geoms = [shape(geom) for geom, val in geoms if val == 1]
        self.merged_geom = unary_union(self.shapely_geoms)
        self.detailed_area = self.merged_geom.area

    def load_raster(self, masked: bool = True) -> None:
        """
        Load raster data and store finite values in self.data_array.

        Parameters
        ----------
        masked : bool
            If True, open as masked and coerce mask to NaN.
        """
        
        da = rio.open_rasterio(self.raster_path, masked=masked)
        if "band" in da.dims and da.sizes.get("band", 1) == 1:
            da = da.squeeze("band", drop=True)
        arr = da.values
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        nodata = da.rio.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= (arr != nodata)
        self.rioxarray_obj = da
        self.data_array = np.asarray(arr[valid], dtype=float).ravel()

    def subtract_value_from_raster(self, output_raster_path: str, value_to_subtract: float) -> None:
        """
        Subtract a specified value from the raster and write a new file.

        Parameters
        ----------
        output_raster_path : str
            Path to the output raster.
        value_to_subtract : float
            Value to subtract from each valid pixel.
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read()
            nodata = src.nodata
            mask = (data != nodata) if nodata is not None else np.ones(data.shape, dtype=bool)
            data = data.astype(float)
            data[mask] -= value_to_subtract
            out_meta = src.meta.copy()
            out_meta.update({'dtype': 'float32', 'nodata': nodata})
            with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
                dst.write(data)

    def plot_raster(self, plot_title: str):
        """
        Plot the loaded rioxarray DataArray with a diverging colormap.

        Raises
        ------
        RuntimeError
            If raster has not been loaded yet.
        """
        
        if self.rioxarray_obj is None:
            raise RuntimeError("Raster not loaded. Call load_raster() first.")
        rio_data = self.rioxarray_obj
        fig, ax = plt.subplots(figsize=(10, 6))
        rio_data.plot(cmap="bwr_r", ax=ax, robust=True)
        ax.set_title(plot_title, pad=30)
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.ticklabel_format(style="plain")
        ax.set_aspect('equal')
        return fig

    def sample_raster(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        *,
        seed: Optional[int] = None
    ) -> None:
        """
        Randomly sample valid pixels from the raster, storing (values, coords).

        Parameters
        ----------
        area_side : float
            Reference side length used to convert pixel area to reference-area units
            (e.g., 1000 for km² if coordinates are meters).
        samples_per_area : float
            Number of samples to draw per unit of reference area.
        max_samples : int
            Maximum total samples to draw.
        seed : int | None
            RNG seed for reproducibility.

        Raises
        ------
        ValueError
            If requested samples exceed valid pixels, or computed total is < 1.
        """
        with rasterio.open(self.raster_path) as src:
            rng = np.random.default_rng(seed)

            data = src.read(1).astype(float)
            nodata = src.nodata
            valid = np.isfinite(data)
            if nodata is not None:
                valid &= (data != nodata)

            cell_area_m2 = abs(src.res[0] * src.res[1])
            valid_rows, valid_cols = np.where(valid)
            valid_count = valid_rows.size
            cell_area_in_reference = cell_area_m2 / (area_side ** 2)
            total_samples = min(int(cell_area_in_reference * samples_per_area * valid_count), max_samples)

            
            if total_samples < 1:
                raise ValueError("Computed total_samples < 1. Increase samples_per_area or max_samples.")

            if total_samples > valid_count:
                raise ValueError("Requested samples exceed valid pixel count. Reduce samples_per_area.")

            chosen = rng.choice(valid_count, size=total_samples, replace=False)
            rows = valid_rows[chosen]
            cols = valid_cols[chosen]
            samples = data[rows, cols]
            x_coords, y_coords = src.xy(rows, cols)
            coords = np.vstack([x_coords, y_coords]).T

            mask = np.isfinite(samples)
            self.samples = samples[mask]
            self.coords = coords[mask]


class StatisticalAnalysis:
    """
    Statistical utilities for exploratory plotting and bootstrap uncertainty of the median.
    """

    def __init__(self, raster_data_handler: RasterDataHandler):
        self.raster_data_handler = raster_data_handler

    def plot_data_stats(self, filtered: bool = True):
        """
        Plot histogram of raster values with basic statistics annotated.

        Parameters
        ----------
        filtered : bool
            If True, clip to 1st–99th percentiles for visualization only.

        Returns
        -------
        matplotlib.figure.Figure
        """
        data = self.raster_data_handler.data_array
        if data is None or len(data) == 0:
            raise ValueError("No data available to plot. Call load_raster() first.")

        mean = np.mean(data)
        median = np.median(data)
        # mode on continuous data is often not meaningful; kept for completeness
        mode_result = stats.mode(data, nan_policy="omit", keepdims=False)
        mode_vals = np.atleast_1d(mode_result.mode).astype(float)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        p1 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)
        minimum = np.min(data)
        maximum = np.max(data)

        if filtered:
            data = data[(data >= p1) & (data <= p99)]

        fig, ax = plt.subplots()
        ax.hist(data, bins=60, density=False, alpha=0.6, color='g')
        ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
        ax.axvline(median, color='b', linestyle='dashed', linewidth=1, label='Median')
        for i, m in enumerate(mode_vals):
            ax.axvline(m, color='purple', linestyle='dashed', linewidth=1,
                       label='Mode' if i == 0 else "_nolegend_")

        mode_str = ", ".join([f"{m:.3f}" for m in mode_vals])
        textstr = "\n".join((
            f"Mean: {mean:.3f}",
            f"Median: {median:.3f}",
            f"Mode(s): {mode_str}",
            f"Min: {minimum:.3f}  Max: {maximum:.3f}",
            f"Q1: {q1:.3f}  Q3: {q3:.3f}",
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        ax.set_xlabel(f'Vertical Difference ({self.raster_data_handler.unit})')
        ax.set_ylabel('Count')
        ax.set_title('Histogram of differencing results with exploratory statistics')
        ax.legend()
        plt.tight_layout()
        return fig

    def bootstrap_uncertainty_subsample(self, n_bootstrap: int = 1000, subsample_proportion: float = 0.1) -> float:
        """
        Estimate uncertainty of the median via bootstrap on random subsamples.

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
            raise ValueError("No data available for bootstrap. Call load_raster() first.")

        
        subsample_size = max(1, int(round(subsample_proportion * len(data))))
        rng = np.random.default_rng()
        bootstrap_medians = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=subsample_size, replace=True)
            bootstrap_medians[i] = np.median(sample)
        return float(np.std(bootstrap_medians))


class VariogramAnalysis:
    """
    Compute empirical variograms across multiple random samples, fit spherical
    models (with optional nugget), and bootstrap parameter uncertainty.
    """

    MIN_PAIRS = 10

    def __init__(self, raster_data_handler: RasterDataHandler):
        self.raster_data_handler = raster_data_handler
        self.mean_variogram = None
        self.lags = None
        self.mean_count = None
        self.err_variogram = None
        self.fitted_variogram = None
        self.rmse = None
        self.sills = None
        self.ranges = None
        # full range percentiles (2.5th to 97.5th)
        self.ranges_min = None
        self.ranges_max = None
        self.ranges_median = None
        # 1σ range percentiles (16th to 84th)
        self.ranges_p16 = None
        self.ranges_p84 = None
        self.err_sills = None
        self.err_ranges = None
        # full range percentiles for sills
        self.sills_min = None
        self.sills_max = None
        self.sills_median = None
        # 1σ range percentiles for sills
        self.sills_p16 = None
        self.sills_p84 = None
        # nugget parameters
        self.best_nugget = None
        # full range percentiles for nugget
        self.min_nugget = None
        self.max_nugget = None
        self.median_nugget = None
        # 1σ range percentiles for nugget
        self.nugget_p16 = None
        self.nugget_p84 = None
        # model selection attributes
        self.best_aic = None
        self.best_bic = None
        self.selection_criterion = None
        self.best_params = None
        self.best_model_config = None
        self.cv_mean_error_best_aic = None
        self.fitted_model = None
        self.param_samples = None
        self.n_bins = None
        self.sigma_variogram = None
        self.best_model_func = None
        self.best_guess = None
        self.best_bounds = None
        self.all_variograms = None
        self.all_counts = None
        self.estimator = None

    
    @staticmethod
    @njit(parallel=True)
    def bin_distances_and_squared_differences(coords, values, bin_width, max_lag_multiplier, x_extent, y_extent):
        """
        Compute and bin pairwise distances, squared differences, and
        fourth-root absolute differences in a single pass.

        The squared differences feed the Matheron estimator;
        the fourth-root absolute differences feed the Cressie–Hawkins
        robust estimator.  Both are accumulated with O(n_bins) memory.

        Parameters
        ----------
        coords : np.ndarray
            Array of coordinates of shape (M, 2).
        values : np.ndarray
            Array of values of shape (M,).
        bin_width : float
            Lag bin width.
        max_lag_multiplier : float or str
            Controls maximum lag distance.
        x_extent, y_extent : float
            Spatial extent of the domain.

        Returns
        -------
        n_bins : int
            Number of lag bins.
        bin_counts : np.ndarray
            Counts of pairs in each bin.
        binned_sum_squared_diff : np.ndarray
            Sum of squared differences per bin (for Matheron).
        binned_sum_sqrt_abs_diff : np.ndarray
            Sum of |ΔZ|^0.5 per bin (for Cressie–Hawkins).
        max_distance : float
            Maximum observed pairwise distance.
        max_lag : float
            Maximum lag used for binning.
        """
        approx_max_distance = np.sqrt(x_extent**2 + y_extent**2)

        if max_lag_multiplier == "max":
            max_lag = approx_max_distance
        elif max_lag_multiplier == "median":
            max_lag = 0.5 * approx_max_distance  # simple heuristic
        else:
            max_lag = float(approx_max_distance * max_lag_multiplier)

        # determine bin edges using diagonal distance as maximum lag
        n_bins = int(np.ceil(max_lag / bin_width)) + 1
        bin_edges = np.arange(0, n_bins * bin_width, bin_width)

        M = coords.shape[0]
        max_distance = 0.0
        bin_counts = np.zeros(n_bins, dtype=np.int64)
        binned_sum_squared_diff = np.zeros(n_bins, dtype=np.float64)
        binned_sum_sqrt_abs_diff = np.zeros(n_bins, dtype=np.float64)

        for i in prange(M):
            for j in range(i + 1, M):
                # compute the pairwise distance
                d = 0.0
                for k in range(coords.shape[1]):
                    tmp = coords[i, k] - coords[j, k]
                    d += tmp * tmp
                dist = np.sqrt(d)
                max_distance = max(max_distance, dist)

                # compute the difference
                diff = values[i] - values[j]

                # Matheron accumulator: squared difference
                diff_squared = diff ** 2

                # Cressie–Hawkins accumulator: |diff|^0.5
                sqrt_abs_diff = np.sqrt(np.abs(diff))

                # find the bin for this distance
                bin_idx = int(dist / bin_width)
                if 0 <= bin_idx < n_bins:
                    bin_counts[bin_idx] += 1
                    binned_sum_squared_diff[bin_idx] += diff_squared
                    binned_sum_sqrt_abs_diff[bin_idx] += sqrt_abs_diff


        bin_edges = bin_edges[:n_bins]
        bin_counts = bin_counts[:n_bins]
        binned_sum_squared_diff = binned_sum_squared_diff[:n_bins]
        binned_sum_sqrt_abs_diff = binned_sum_sqrt_abs_diff[:n_bins]

        return n_bins, bin_counts, binned_sum_squared_diff, binned_sum_sqrt_abs_diff, max_distance, max_lag

    @staticmethod
    def compute_matheron(bin_counts, ssd, min_pairs: int = 10) -> np.ndarray:
        """
        Compute Matheron semivariance γ(h) = SSD(h) / (2 N(h)) for bins with >= min_pairs.
        """
        gamma_est = np.full_like(bin_counts, np.nan, dtype=float)
        for i, (cnt, sum_sq) in enumerate(zip(bin_counts, ssd)):
            if cnt >= min_pairs:
                gamma_est[i] = sum_sq / (2.0 * cnt)
        return gamma_est

    @staticmethod
    def compute_cressie_hawkins(
        bin_counts, sum_sqrt_abs_diff, min_pairs: int = 10
    ) -> np.ndarray:
        """
        Compute the Cressie–Hawkins robust semivariance estimator.

        γ̂(h) = [mean(|ΔZ|^0.5)]⁴ / (2 · (0.457 + 0.494 / N(h)))

        This estimator downweights large squared differences by operating
        on fourth-root-transformed absolute differences, making it resistant
        to outliers while remaining a consistent estimator of the variogram.

        Parameters
        ----------
        bin_counts : np.ndarray
            Number of pairs per lag bin.
        sum_sqrt_abs_diff : np.ndarray
            Sum of |Z(x+h) − Z(x)|^0.5 per lag bin.
        min_pairs : int
            Minimum pair count for a bin to be considered valid.

        Returns
        -------
        gamma_est : np.ndarray
            Robust semivariance estimates; NaN for bins with < min_pairs.

        References
        ----------
        Cressie, N. (1985). Fitting variogram models by weighted least
        squares. J. Int. Assoc. Math. Geol., 17(5), 563–586.

        Cressie, N. & Hawkins, D.M. (1980). Robust estimation of the
        variogram: I. J. Int. Assoc. Math. Geol., 12(2), 115–125.
        """
        gamma_est = np.full_like(bin_counts, np.nan, dtype=float)
        for i, (cnt, s) in enumerate(zip(bin_counts, sum_sqrt_abs_diff)):
            if cnt >= min_pairs:
                mean_fourth = (s / cnt) ** 4
                correction = 0.457 + 0.494 / cnt
                gamma_est[i] = 0.5 * mean_fourth / correction
        return gamma_est

    # valid estimator names ------------------------------------------------
    ESTIMATORS = ('matheron', 'cressie_hawkins')

    def numba_variogram(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        bin_width: float,
        max_lag_multiplier,
        *,
        seed: Optional[int] = None,
        estimator: str = 'matheron',
    ):
        """
        Compute one empirical variogram by sampling the raster and binning
        pairwise differences by distance.

        Parameters
        ----------
        area_side, samples_per_area, max_samples : see ``sample_raster``
        bin_width : float
        max_lag_multiplier : {"max", "median"} or float
        seed : int | None
        estimator : {'matheron', 'cressie_hawkins'}
            Which semivariance estimator to apply.

        Returns
        -------
        bin_counts : np.ndarray
        variogram : np.ndarray
        n_bins : int
        min_distance : float
        max_distance : float
        max_lag : float
        """
        if estimator not in self.ESTIMATORS:
            raise ValueError(
                f"Unknown estimator '{estimator}'. "
                f"Choose from {self.ESTIMATORS}."
            )

        self.raster_data_handler.sample_raster(area_side, samples_per_area, max_samples, seed=seed)

        min_distance = 0.0  # retained for compatibility
        xs = self.raster_data_handler.rioxarray_obj.x.values
        ys = self.raster_data_handler.rioxarray_obj.y.values
        x_extent = float(np.max(xs) - np.min(xs))
        y_extent = float(np.max(ys) - np.min(ys))

        (n_bins, bin_counts, bssd, bssad,
         max_distance, max_lag) = self.bin_distances_and_squared_differences(
            self.raster_data_handler.coords,
            self.raster_data_handler.samples,
            bin_width,
            max_lag_multiplier,
            x_extent,
            y_extent,
        )

        if estimator == 'cressie_hawkins':
            estimates = self.compute_cressie_hawkins(
                bin_counts, bssad, min_pairs=self.MIN_PAIRS
            )
        else:
            estimates = self.compute_matheron(
                bin_counts, bssd, min_pairs=self.MIN_PAIRS
            )

        return bin_counts, estimates, n_bins, min_distance, max_distance, max_lag

    def calculate_mean_variogram_numba(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        bin_width: float,
        max_n_bins: int,
        n_runs: int,
        max_lag_multiplier=1 / 3,
        *,
        seed: Optional[int] = None,
        estimator: str = 'matheron',
    ) -> None:
        """
        Run multiple variogram realizations and compute the mean semivariogram
        and spread across runs.

        Parameters
        ----------
        area_side, samples_per_area, max_samples : see numba_variogram
        bin_width : float
        max_n_bins : int
        n_runs : int
        max_lag_multiplier : {"max","median"} or float
        seed : int | None
            Base seed; each run uses a child seed for reproducibility.
        estimator : {'matheron', 'cressie_hawkins'}
            Empirical variogram estimator to use.  ``'matheron'`` is the
            classical method-of-moments estimator.  ``'cressie_hawkins'``
            is a robust estimator that downweights outlying squared
            differences (Cressie & Hawkins, 1980; Cressie, 1985).
        """
        # child seeds for each run to keep realizations independent but reproducible.
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_runs)

        all_variograms = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
        counts = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
        all_n_bins = np.zeros(n_runs, dtype=int)

        for run in range(n_runs):
            count, variogram, n_bins, _, _, _ = self.numba_variogram(
                area_side, samples_per_area, max_samples, bin_width, max_lag_multiplier,
                seed=int(child_seeds[run].generate_state(1)[0]),
                estimator=estimator,
            )
            all_variograms.loc[run, :variogram.size - 1] = variogram
            counts.loc[run, :count.size - 1] = count
            all_n_bins[run] = n_bins

        vario_arr = all_variograms.values
        count_arr = counts.values

        with np.errstate(all='ignore'):
            mean_variogram = np.nanmean(vario_arr, axis=0)
            # use robust spread visualization width; stored as err_variogram
            err_variogram = (np.nanpercentile(vario_arr, 97.5, axis=0) -
                             np.nanpercentile(vario_arr, 2.5, axis=0)) / 2.0
            mean_count = np.nanmean(count_arr, axis=0)
            sigma_variogram = np.nanstd(vario_arr, axis=0)

        
        sigma_filtered = sigma_variogram.copy()
        sigma_filtered[sigma_filtered == 0] = np.finfo(float).eps

        valid = ~np.isnan(mean_variogram)
        self.mean_variogram = mean_variogram[valid]
        self.err_variogram = err_variogram[valid]
        self.mean_count = mean_count[valid]
        self.sigma_variogram = sigma_filtered[valid]

        n_kept = self.mean_variogram.size
        self.lags = np.linspace(bin_width / 2, bin_width * n_kept - bin_width / 2, n_kept)

        self.all_variograms = vario_arr
        self.all_counts = count_arr
        self.n_bins = int(np.nanmean(all_n_bins))
        self.estimator = estimator

    @staticmethod
    def get_base_initial_guess(n: int, mean_variogram, lags, nugget: bool = False) -> np.ndarray:
        """
        Initial guess for variogram parameters with improved nugget estimation.

        For sills: distributes the total sill evenly across components.
        For ranges: spreads ranges linearly up to half the max lag.
        For nugget: extrapolates to h=0 using a linear fit to the first few lags,
                    which provides a more physically meaningful estimate than
                    using an arbitrary fraction of the max semivariance.

        Parameters
        ----------
        n : int
            Number of variogram components.
        mean_variogram : array-like
            Empirical semivariance values.
        lags : array-like
            Lag distances.
        nugget : bool
            Whether to include a nugget parameter.

        Returns
        -------
        p0 : ndarray
            Initial parameter vector [C1, ..., Cn, a1, ..., an, (nugget)].
        """
        max_semivariance = np.max(mean_variogram)
        half_max_lag = np.max(lags) / 2

        # estimate nugget by extrapolating to h=0 from first few lags
        if nugget:
            # use first 3-5 valid lag points for linear extrapolation
            n_extrap = min(5, len(lags) // 3, len(lags))
            n_extrap = max(2, n_extrap)  # Need at least 2 points

            short_lags = lags[:n_extrap]
            short_gamma = mean_variogram[:n_extrap]

            # linear fit: γ(h) = nugget + slope * h
            # extrapolate to h=0 to get nugget estimate
            if len(short_lags) >= 2:
                slope, intercept = np.polyfit(short_lags, short_gamma, 1)
                nugget_estimate = max(0.0, intercept)  # Nugget can't be negative
                # cap nugget at 50% of max semivariance as sanity check
                nugget_estimate = min(nugget_estimate, max_semivariance * 0.5)
            else:
                # fallback: use smallest lag value as upper bound
                nugget_estimate = mean_variogram[0] * 0.5

            # partial sill is what remains after nugget
            partial_sill_total = max(max_semivariance - nugget_estimate, max_semivariance * 0.5)
        else:
            nugget_estimate = 0.0
            partial_sill_total = max_semivariance

        # distribute sill across components
        C = [partial_sill_total / n] * n

        # spread ranges linearly
        a = [((half_max_lag) / 3) * (i + 1) for i in range(n)]

        p0 = C + a + ([nugget_estimate] if nugget else [])
        return np.array(p0, dtype=float)

    @staticmethod
    def pure_nugget_model(h, nugget):
        """γ(h) for pure nugget: constant variance independent of h."""
        return np.full_like(h, nugget)

    @staticmethod
    def spherical_model(h, *params):
        """
        Multi-component spherical model without nugget.

        Parameters
        ----------
        h : array-like
        params : [C1..Cn, a1..an] (n sills followed by n ranges)
        """
        n = len(params) // 2
        C = params[:n]
        a = params[n:]
        model = np.zeros_like(h, dtype=float)
        for i in range(n):
            ai = a[i]
            Ci = C[i]
            mask = h <= ai
            ratio = h[mask] / ai
            model[mask] += Ci * (1.5 * ratio - 0.5 * ratio ** 3)
            model[~mask] += Ci
        return model

    def spherical_model_with_nugget(self, h, *params):
        """
        Spherical model with nugget at the END of the parameter vector.

        Parameters
        ----------
        params : [C1..Cn, a1..an, nugget]
        """
        nugget = params[-1]
        structural = params[:-1]
        return nugget + self.spherical_model(h, *structural)

    @staticmethod
    def bootstrap_fit_variogram(
        lags: np.ndarray,
        mean_vario: np.ndarray,
        sigma_vario: np.ndarray,
        model_func: Callable,
        p0: np.ndarray,
        bounds: Optional[tuple] = None,
        n_boot: int = 100,
        maxfev: int = 10000,
        *,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Parametric bootstrap for parameter uncertainty using per-lag standard deviations.

        Parameters
        ----------
        lags, mean_vario, sigma_vario : arrays
        model_func : callable
        p0 : initial params
        bounds : bounds tuple
        n_boot : int
        maxfev : int
        seed : int | None

        Returns
        -------
        np.ndarray
            Successful parameter vectors (rows).
        """
        rng = np.random.default_rng(seed)
        # draw synthetic variograms with Gaussian noise per bin using sigma_vario.
        noise_array = rng.normal(loc=mean_vario, scale=np.where(sigma_vario > 0, sigma_vario, 0.0), size=(n_boot, len(mean_vario)))

        param_samples = []
        n_params = len(p0)
        bnds = bounds if bounds is not None else (-np.inf, np.inf)

        for n in range(n_boot):
            synth = noise_array[n, :]
            try:
                popt_synth, _ = curve_fit(
                    model_func,
                    lags,
                    synth,
                    p0=p0,
                    sigma=None,        # Using unconditional fits to the synthetic draws
                    bounds=bnds,
                    maxfev=maxfev
                )
                param_samples.append(popt_synth)
            except RuntimeError:
                param_samples.append([np.nan] * n_params)

        param_samples = np.array(param_samples)
        valid = ~np.isnan(param_samples).any(axis=1)
        return param_samples[valid]

    @staticmethod
    def _weighted_loglike_gaussian(y, yhat, sigma):
        """
        Log-likelihood for heteroscedastic Gaussian errors: sum over bins of
        -0.5 * [log(2π σ_i^2) + ((y_i - ŷ_i)^2 / σ_i^2)]
        """
        s = np.asarray(sigma, dtype=float)
        s = np.where(s <= 0, np.finfo(float).eps, s)
        resid = y - yhat
        return -0.5 * np.sum(np.log(2 * np.pi * s ** 2) + (resid ** 2) / (s ** 2))

    def cross_validate_variogram(self, model_func, p0, bounds, k: int = 5, *, seed: Optional[int] = None):
        """
        k-fold cross-validation on (lags, mean_variogram) with provided model/bounds.

        Returns
        -------
        dict | None
            {'rmse','mae','me','mse'} averaged across folds, or None if all folds fail.
        """
        rng = np.random.default_rng(seed)
        lags = self.lags
        mean_variogram = self.mean_variogram
        sigma_filtered = self.sigma_variogram

        n_bins = len(lags)
        indices = rng.permutation(n_bins)
        fold_size = max(1, n_bins // k)
        rmses, maes, mes, mses = [], [], [], []

        for i in range(k):
            valid_idx = indices[i * fold_size: (i + 1) * fold_size]
            train_idx = np.setdiff1d(indices, valid_idx)

            lags_train = lags[train_idx]
            vario_train = mean_variogram[train_idx]
            sigma_train = sigma_filtered[train_idx]

            try:
                popt, _ = curve_fit(model_func, lags_train, vario_train, p0=p0, bounds=bounds, sigma=sigma_train, absolute_sigma=True, maxfev=10000)
            except RuntimeError:
                continue

            lags_valid = lags[valid_idx]
            vario_valid = mean_variogram[valid_idx]
            predictions = model_func(lags_valid, *popt)

            errors = vario_valid - predictions
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mae = float(np.mean(np.abs(errors)))
            me = float(np.mean(errors))
            mse = float(np.mean(errors ** 2))

            rmses.append(rmse)
            maes.append(mae)
            mes.append(me)
            mses.append(mse)

        if not rmses:
            return None

        return {'rmse': np.mean(rmses), 'mae': np.mean(maes), 'me': np.mean(mes), 'mse': np.mean(mses)}

    def fit_best_spherical_model(
        self,
        sigma_type: str = 'std',
        bounds: Optional[tuple] = None,
        method: str = 'trf',
        criterion: str = 'aic',
        *,
        seed: Optional[int] = None
    ) -> None:
        """
        Fit spherical variogram models (1–3 components, optional nugget) and select the best.

        Parameters
        ----------
        sigma_type : {'std','linear','exp','sqrt','sq'}
            Bin weighting scheme used in curve_fit as sigma.
        bounds : tuple | None
            Optional (lb, ub) for parameters; if None, internal bounds are used.
        method : str
            curve_fit method (default 'trf').
        criterion : {'aic', 'bic'}
            Information criterion for model selection.
            - 'aic': Akaike Information Criterion (default)
            - 'bic': Bayesian Information Criterion
        seed : int | None
            RNG seed for randomized initial guesses.

        Notes
        -----
        - Nugget parameter is ALWAYS last in the parameter vector.
        """
        rng = np.random.default_rng(seed)

        if self.all_variograms is None:
            raise RuntimeError("No variogram data. Call calculate_mean_variogram_numba() first.")

        # choose weights
        if sigma_type == 'std':
            sigma = self.sigma_variogram
        elif sigma_type == 'linear':
            sigma = 1.0 / (1.0 + self.lags)
        elif sigma_type == 'exp':
            sigma = np.exp(-self.lags)
        elif sigma_type == 'sqrt':
            sigma = 1.0 / np.sqrt(1.0 + self.lags)
        elif sigma_type == 'sq':
            sigma = 1.0 / (1.0 + self.lags ** 2)
        elif sigma_type == 'nugget_focus':
            # weight short lags heavily for better nugget estimation
            # uses exponential decay with characteristic length = 20% of max lag
            char_length = np.max(self.lags) * 0.2
            sigma = np.exp(-self.lags / char_length)
            # normalize so first bin has sigma=1
            sigma = sigma / sigma[0]
        elif sigma_type == 'cressie':
            # cressie's robust weighting: N_h / gamma(h)^2
            # downweights large semivariance values which are more variable
            sigma = self.mean_variogram**2 / np.maximum(self.mean_count, 1)
            sigma = np.sqrt(sigma)  # curve_fit expects std dev
        else:
            raise ValueError(f"Unknown sigma_type '{sigma_type}'. Use 'std', 'linear', 'exp', 'sqrt', 'sq', 'nugget_focus', or 'cressie'.")
        
        if criterion not in ('aic', 'bic'):
            raise ValueError(f"criterion must be 'aic' or 'bic', got '{criterion}'")

        best_params = None
        best_model = None
        best_func = None
        best_criterion_value = np.inf
        best_aic = np.inf
        best_bic = np.inf
        best_bounds = None
        best_guess = None
        n_obs = len(self.lags)

        lag_max = float(np.max(self.lags)) if self.lags is not None and len(self.lags) else 1.0
        for config in (
            {'components': 1, 'nugget': False},
            {'components': 1, 'nugget': True},
            {'components': 2, 'nugget': False},
            {'components': 2, 'nugget': True},
            {'components': 3, 'nugget': False},
            {'components': 3, 'nugget': True},
        ):
            n = config['components']
            nugget = config['nugget']
            if n == 0:
                model = self.pure_nugget_model
                auto_bounds = ([0.0], [np.inf])
                p0s = [np.array([np.max(self.mean_variogram)])]
            else:
                model = self.spherical_model_with_nugget if nugget else self.spherical_model

                # constrain nugget upper bound to prevent overestimation
                # nugget > 50% of total sill is unusual and often indicates
                # fitting problems rather than True micro-scale variation
                max_gamma = float(np.max(self.mean_variogram))
                nugget_upper = max_gamma * 0.5  # Cap nugget at 50% of observed max

                lb = [0.0] * n + [1e-6] * n + ([0.0] if nugget else [])
                ub = [np.inf] * n + [2.0 * lag_max] * n + ([nugget_upper] if nugget else [])
                auto_bounds = (lb, ub)

                base = self.get_base_initial_guess(n, self.mean_variogram, self.lags, nugget)
                p0s = []
                for _ in range(5):
                    perturb = (rng.random(len(base)) - 0.5) * 2.0 * 0.5  # +/-50%
                    guess = np.clip(base * (1 + perturb), 1e-6, None)
                    p0s.append(guess)

            bounds_tuple = bounds if bounds is not None else auto_bounds

            for p0 in p0s:
                try:
                    popt, _ = curve_fit(
                        model,
                        self.lags,
                        self.mean_variogram,
                        p0=p0,
                        sigma=sigma,
                        absolute_sigma=True,  
                        bounds=bounds_tuple,
                        method=method,
                        maxfev=20000
                    )
                except RuntimeError:
                    continue

                yhat = model(self.lags, *popt)
                ll = self._weighted_loglike_gaussian(self.mean_variogram, yhat, sigma)
                k = len(popt)
                aic = 2 * k - 2 * ll
                bic = k * np.log(n_obs) - 2 * ll

                current_criterion_value = aic if criterion == 'aic' else bic

                if current_criterion_value < best_criterion_value:
                    best_criterion_value = current_criterion_value
                    best_aic = aic
                    best_bic = bic
                    best_params = popt
                    best_model = config
                    best_func = model
                    best_bounds = bounds_tuple
                    best_guess = p0

        if best_params is None:
            raise RuntimeError("No valid model fit found. Check input data for NaN values.")

        self.best_params = best_params
        self.best_model_config = best_model
        self.best_model_func = best_func
        self.best_aic = best_aic
        self.best_bounds = best_bounds
        self.best_guess = best_guess
        self.best_bic = best_bic
        self.selection_criterion = criterion
        self.fitted_variogram = (
            self.spherical_model_with_nugget if self.best_model_config['nugget']
            else self.spherical_model
        )(self.lags, *self.best_params)

        # extract sill & range point estimates; nugget last if present
        n = self.best_model_config['components']
        if self.best_model_config['nugget']:
            self.sills = self.best_params[:n]
            self.ranges = self.best_params[n:2 * n]
            self.best_nugget = float(self.best_params[-1])
        else:
            self.sills = self.best_params[:n]
            self.ranges = self.best_params[n:2 * n]
            self.best_nugget = None

        # prepare bounds for bootstrap consistent with nugget-last convention
        max_gamma = float(np.max(self.mean_variogram))
        nugget_upper = max_gamma * 0.5  # Same constraint as fitting

        if n == 0:
            bounds_boot = ([0.0], [max_gamma * 1.5])
        else:
            lb = [0.0] * n + [1e-6] * n + ([0.0] if self.best_model_config['nugget'] else [])
            ub = [np.inf] * n + [2.0 * lag_max] * n + ([nugget_upper] if self.best_model_config['nugget'] else [])
            bounds_boot = (lb, ub)

        # parametric bootstrap using per-bin sigma (std across runs)
        samples = self.bootstrap_fit_variogram(
            self.lags,
            self.mean_variogram,
            self.sigma_variogram,  
            self.best_model_func,
            self.best_params,
            bounds=bounds_boot,
            n_boot=500,
            maxfev=20000,
            seed=seed,
        )
        # Store bootstrap samples without appending the optimal point
        # estimate — appending it biases percentile estimates toward
        # the MLE (the bias is small for n_boot=500 but unnecessary).
        if samples.size:
            self.param_samples = samples
        else:
            self.param_samples = np.array([self.best_params])

        # percentiles of parameters
        # full range: 2.5th to 97.5th percentiles (robust bounds for plotting)
        # 1σ range: 16th to 84th percentiles (darker shading)
        if self.param_samples.size:
            if self.best_model_config['nugget']:
                nug_samps = self.param_samples[:, -1]
                samp = self.param_samples[:, :-1]
            else:
                nug_samps = None
                samp = self.param_samples

            sill_samps = samp[:, :n]
            range_samps = samp[:, n:2 * n]

            # full range (2.5th to 97.5th percentiles)
            self.sills_min = np.percentile(sill_samps, 2.5, axis=0)
            self.sills_max = np.percentile(sill_samps, 97.5, axis=0)
            self.sills_median = np.percentile(sill_samps, 50, axis=0)

            # 1σ range (16th to 84th percentiles)
            self.sills_p16 = np.percentile(sill_samps, 16, axis=0)
            self.sills_p84 = np.percentile(sill_samps, 84, axis=0)

            # full range for ranges
            self.ranges_min = np.percentile(range_samps, 2.5, axis=0)
            self.ranges_max = np.percentile(range_samps, 97.5, axis=0)
            self.ranges_median = np.percentile(range_samps, 50, axis=0)

            # 1σ range for ranges
            self.ranges_p16 = np.percentile(range_samps, 16, axis=0)
            self.ranges_p84 = np.percentile(range_samps, 84, axis=0)

            if nug_samps is not None:
                # full range for nugget
                self.min_nugget = float(np.percentile(nug_samps, 2.5))
                self.max_nugget = float(np.percentile(nug_samps, 97.5))
                self.median_nugget = float(np.percentile(nug_samps, 50))
                # 1σ range for nugget
                self.nugget_p16 = float(np.percentile(nug_samps, 16))
                self.nugget_p84 = float(np.percentile(nug_samps, 84))
            else:
                self.min_nugget = self.max_nugget = self.median_nugget = None
                self.nugget_p16 = self.nugget_p84 = None
        else:
            # fallback to point estimates
            self.sills_min = self.sills_max = self.sills_median = np.array(self.sills)
            self.sills_p16 = self.sills_p84 = np.array(self.sills)
            self.ranges_min = self.ranges_max = self.ranges_median = np.array(self.ranges)
            self.ranges_p16 = self.ranges_p84 = np.array(self.ranges)
            if self.best_model_config['nugget']:
                self.min_nugget = self.max_nugget = self.median_nugget = self.best_nugget
                self.nugget_p16 = self.nugget_p84 = self.best_nugget
            else:
                self.min_nugget = self.max_nugget = self.median_nugget = None
                self.nugget_p16 = self.nugget_p84 = None

        # compute and store cross-validation metrics on best model
        self.cv_mean_error_best_aic = self.cross_validate_variogram(
            self.best_model_func, self.best_params, self.best_bounds, k=5, seed=seed
        )

    def estimate_nugget_from_short_lags(
        self,
        n_lags: int = 5,
        method: str = 'linear_extrap'
    ) -> float:
        """
        Estimate nugget independently from short-lag behavior.

        This two-stage approach first estimates the nugget from the y-intercept
        region, then can be used to constrain full model fitting. This prevents
        the optimizer from trading off sill for nugget inappropriately.

        Think of it like this: the nugget represents instantaneous variance
        (measurement error + micro-scale variation), which is only "visible"
        at the shortest lags. By extrapolating to h=0 from short lags, we get
        a more physically meaningful estimate.

        Parameters
        ----------
        n_lags : int
            Number of short-lag bins to use for estimation (default: 5).
        method : str
            Estimation method:
            - 'linear_extrap': Linear regression extrapolated to h=0
            - 'quadratic_extrap': Quadratic fit extrapolated to h=0
            - 'first_lag': Use first lag value directly (conservative upper bound)

        Returns
        -------
        float
            Estimated nugget value.

        Notes
        -----
        The linear extrapolation method assumes γ(h) ≈ C₀ + bh near the origin,
        which is valid for models with a nugget (discontinuity at origin).

        References
        ----------
        Cressie, N. (1985). Fitting variogram models by weighted least squares.
        Journal of the International Association for Mathematical Geology, 17(5).
        """
        if self.mean_variogram is None:
            raise RuntimeError("No variogram data. Call calculate_mean_variogram_numba() first.")

        n_fit = min(n_lags, len(self.lags))
        lags_short = self.lags[:n_fit]
        gamma_short = self.mean_variogram[:n_fit]

        if method == 'linear_extrap':
            # γ(h) ≈ C₀ + bh → intercept is nugget estimate
            slope, intercept = np.polyfit(lags_short, gamma_short, 1)
            nugget_est = max(intercept, 0.0)

        elif method == 'quadratic_extrap':
            # γ(h) ≈ C₀ + bh + ch² → allows for curvature near origin
            if n_fit >= 3:
                coeffs = np.polyfit(lags_short, gamma_short, 2)
                nugget_est = max(coeffs[2], 0.0)  # Constant term
            else:
                # fall back to linear if not enough points
                slope, intercept = np.polyfit(lags_short, gamma_short, 1)
                nugget_est = max(intercept, 0.0)

        elif method == 'first_lag':
            # conservative: first lag value is upper bound on nugget
            # (γ(h) at small h includes both nugget and some spatial correlation)
            nugget_est = gamma_short[0] * 0.5  # Assume ~half is nugget

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'linear_extrap', 'quadratic_extrap', or 'first_lag'.")

        # sanity check: nugget shouldn't exceed observed semivariance
        max_gamma = np.max(self.mean_variogram)
        if nugget_est > max_gamma * 0.5:
            import warnings
            warnings.warn(
                f"Estimated nugget ({nugget_est:.4f}) exceeds 50% of max semivariance "
                f"({max_gamma:.4f}). This may indicate measurement issues or that the "
                "variogram doesn't have a true nugget. Capping at 50%.",
                UserWarning
            )
            nugget_est = max_gamma * 0.5

        return nugget_est

    def fit_model_two_stage(
        self,
        n_components: int = 1,
        nugget_method: str = 'linear_extrap',
        nugget_tolerance: float = 0.2,
        sigma_type: str = 'nugget_focus',
        criterion: str = 'aic',
        seed: Optional[int] = None,
    ) -> None:
        """
        Option 4: Two-stage fitting with pre-estimated nugget.

        Stage 1: Estimate nugget from short-lag extrapolation
        Stage 2: Fit full model with nugget constrained near Stage 1 estimate

        This approach prevents the common problem of the optimizer
        overestimating the nugget by trading off with the sill.

        Parameters
        ----------
        n_components : int
            Number of spherical components (default: 1).
        nugget_method : str
            Method for Stage 1 nugget estimation (see estimate_nugget_from_short_lags).
        nugget_tolerance : float
            Fractional tolerance around Stage 1 estimate for Stage 2 bounds.
            E.g., 0.2 means nugget constrained to [0.8*est, 1.2*est].
        sigma_type : str
            Weighting scheme for fitting (default: 'nugget_focus' for better
            short-lag fitting).
        criterion : str
            Selection criterion ('aic' or 'bic').
        seed : int, optional
            Random seed.

        Notes
        -----
        This method stores results in the same attributes as fit_model() for
        compatibility with downstream analysis.
        """
        # stage 1: Estimate nugget
        nugget_est = self.estimate_nugget_from_short_lags(method=nugget_method)

        # stage 2: Fit with constrained nugget
        lag_max = float(np.max(self.lags))
        nugget_lb = max(0.0, nugget_est * (1 - nugget_tolerance))
        nugget_ub = nugget_est * (1 + nugget_tolerance)

        # build bounds with constrained nugget
        n = n_components
        lb = [0.0] * n + [1e-6] * n + [nugget_lb]
        ub = [np.inf] * n + [2.0 * lag_max] * n + [nugget_ub]
        constrained_bounds = (lb, ub)

        # now call fit_model with the constrained bounds
        self.fit_model(
            criterion=criterion,
            sigma_type=sigma_type,
            bounds=constrained_bounds,
            seed=seed,
        )

        # store the Stage 1 estimate for reference
        self.stage1_nugget_estimate = nugget_est

    def fit_best_model_auto(
        self,
        model_types: Optional[List[str]] = None,
        max_components: int = 2,
        include_nugget: bool = True,
        criterion: str = 'msspe',
        compute_cv: bool = False,
        cv_folds: int = 5,
        n_bootstrap: int = 500,
        seed: Optional[int] = None,
        min_pairs: Optional[int] = 30,
        compute_msspe: bool = False,
        msspe_n_subset: int = 500,
        msspe_n_runs: int = 10,
        msspe_prefilter: int = 0,
    ) -> 'FittedVariogramModel':
        """
        Fit multiple variogram model types and automatically select the best.

        Parameters
        ----------
        model_types : list of str, optional
            Model types to consider. Default: ['spherical', 'exponential',
            'gaussian', 'matern'].
        max_components : int
            Maximum number of nested components (default: 2, max: 3).
        include_nugget : bool
            Whether to include nugget effect in all candidate models.
        criterion : {'aic', 'bic', 'cv', 'msspe'}
            Selection criterion.  Default is ``'msspe'``, which selects
            the model whose kriging LOOCV MSSPE is closest to 1.0
            (|MSSPE − 1| minimised).  This directly evaluates whether
            the kriging variance matches actual prediction errors,
            which is what matters for uncertainty propagation.

            AIC/BIC treat variogram lag bins as independent observations,
            which they are not — adjacent bins share point pairs and are
            correlated.  This systematically favours complex models.
            MSSPE avoids this by evaluating spatial prediction
            calibration directly (Lark, 2000).

            Requires spatial data from ``sample_raster()`` or
            ``calculate_mean_variogram_numba()``.
        compute_cv : bool
            Whether to compute variogram k-fold CV scores.
        cv_folds : int
            Number of CV folds.
        n_bootstrap : int
            Number of bootstrap resamples for parameter uncertainty.
        seed : int, optional
            Random seed.
        min_pairs : int or None, default 30
            Minimum pair count per lag bin.  Bins with fewer pairs are
            excluded from fitting.  Set to ``None`` to disable.
        compute_msspe : bool
            Whether to compute kriging LOOCV MSSPE for each candidate
            model.  Automatically set to True when ``criterion='msspe'``.
        msspe_n_subset : int
            Number of points per kriging LOOCV run (default 500).
            Runtime is O(n²) per model per run, so ~1 s per model at
            n=500.
        msspe_n_runs : int
            Number of independent random subsamples over which to
            average the MSSPE (default 10).  A single subsample
            introduces sampling variability; averaging over multiple
            runs produces a more stable estimate — analogous to
            repeated k-fold CV versus a single train/test split.
            Total LOOCV effort is ``msspe_n_runs × msspe_n_subset``
            points per candidate model.
        msspe_prefilter : int
            If > 0, compute MSSPE only for the top-N models ranked by
            AIC, plus the AIC-best model from each complexity level
            (number of components).  The stratified inclusion ensures
            that simpler model families are always represented, even
            when complex models dominate the AIC ranking.  Saves time
            when many candidates are fitted.  If 0 (default), all
            candidates are evaluated.

        Returns
        -------
        FittedVariogramModel
            Best fitted model with diagnostics.

        Notes
        -----
        This method is model-agnostic for uncertainty propagation because
        Monte Carlo integration works for ANY valid variogram function
        (Krige's Relation — Chilès & Delfiner, 2012, Chapter 4).

        The ``'msspe'`` criterion evaluates spatial prediction calibration
        directly, rather than variogram curve-fitting quality.  MSSPE ≈ 1.0
        indicates that the kriging variance correctly matches actual
        prediction errors (Lark, 2000).  This is preferred when the goal
        is uncertainty propagation, because AIC/BIC can favour complex
        models that overfit the empirical variogram while producing
        miscalibrated kriging variances.

        References
        ----------
        Lark, R.M. (2000).  A comparison of some robust estimators of
        the variogram for use in soil survey.  *Eur. J. Soil Sci.*, 51,
        137–157.
        """
        if self.mean_variogram is None:
            raise RuntimeError("No variogram data. Call calculate_mean_variogram_numba() first.")

        # implicitly enable MSSPE computation when criterion requires it
        if criterion == 'msspe':
            compute_msspe = True

        if model_types is None:
            model_types = ['spherical', 'exponential', 'gaussian', 'matern']

        # validate model_types
        available = MODEL_REGISTRY.list_models()
        for mt in model_types:
            if mt not in available:
                raise ValueError(f"Unknown model type '{mt}'. Available: {available}")

        # validate spatial data availability for MSSPE
        if compute_msspe:
            rdh = self.raster_data_handler
            if rdh.coords is None or rdh.samples is None:
                raise RuntimeError(
                    "Kriging LOOCV MSSPE requires spatial data. "
                    "Call sample_raster() or calculate_mean_variogram_numba() first."
                )

        # create selector
        selector = VariogramModelSelector(
            lags=self.lags,
            empirical_variogram=self.mean_variogram,
            pair_counts=self.mean_count,
            sigma=self.sigma_variogram,
            min_pairs=min_pairs,
        )

        # override model lists with user selection
        selector.BOUNDED_MODELS = [m for m in model_types if MODEL_REGISTRY.is_bounded(m)]
        selector.UNBOUNDED_MODELS = [m for m in model_types if not MODEL_REGISTRY.is_bounded(m)]

        # fit all candidates
        selector.fit_all_candidates(
            max_components=min(max_components, 3),
            include_nugget=include_nugget,
            include_unbounded=bool(selector.UNBOUNDED_MODELS),
            compute_cv=compute_cv,
            cv_folds=cv_folds,
            seed=seed,
        )

        if not selector.fitted_models:
            raise RuntimeError("No models successfully fitted. Check input data.")

        # ── compute kriging LOOCV MSSPE for candidate models ──
        if compute_msspe:
            rdh = self.raster_data_handler
            coords = rdh.coords
            values = rdh.samples

            # determine which models to evaluate
            if msspe_prefilter > 0 and len(selector.fitted_models) > msspe_prefilter:
                # Stratified prefilter: take top-N by AIC, but also
                # ensure the AIC-best model from each complexity level
                # (number of components) is included.  Without this,
                # prefiltering can exclude entire model families — e.g.
                # all 1-component models when 3-component models
                # dominate the AIC ranking — and miss the model with
                # the best MSSPE.
                ranked_idx = np.argsort([m.aic for m in selector.fitted_models])
                eval_idx = set(ranked_idx[:msspe_prefilter].tolist())

                # add best-AIC model per complexity level
                best_per_complexity: Dict[int, int] = {}
                for i in ranked_idx:
                    m = selector.fitted_models[i]
                    n_comp = len(m.composite_model.component_names)
                    if n_comp not in best_per_complexity:
                        best_per_complexity[n_comp] = i
                eval_idx.update(best_per_complexity.values())
            else:
                eval_idx = set(range(len(selector.fitted_models)))

            # ── repeated MSSPE runs for stability ──
            # A single random subsample introduces sampling variability
            # in the MSSPE estimate.  Averaging over multiple independent
            # subsamples (each of size msspe_n_subset) produces a more
            # stable ranking — analogous to repeated k-fold CV.
            run_rng = np.random.default_rng(seed)
            run_seeds = run_rng.integers(0, 2**31, size=msspe_n_runs)

            for i, fitted in enumerate(selector.fitted_models):
                if i not in eval_idx:
                    continue
                # only evaluate stationary models (non-stationary have
                # infinite variance → kriging is undefined)
                if not fitted.composite_model.is_stationary:
                    continue

                run_results: list[KrigingLOOCVResult] = []
                for run_seed in run_seeds:
                    try:
                        result = self.kriging_loocv(
                            coords, values,
                            fitted.composite_model,
                            n_subset=msspe_n_subset,
                            seed=int(run_seed),
                        )
                        if np.isfinite(result.msspe):
                            run_results.append(result)
                    except Exception:
                        continue

                if run_results:
                    agg = AggregatedLOOCVResult.from_results(run_results)
                    fitted.msspe = agg.msspe_mean
                    fitted.msspe_std = agg.msspe_std
                    fitted.msspe_n_runs = agg.n_runs
                    fitted.loocv_result = agg
                else:
                    fitted.msspe = None
                    fitted.msspe_std = None
                    fitted.msspe_n_runs = 0
                    fitted.loocv_result = None

        # select best model
        best = selector.select_best(criterion=criterion)

        # bootstrap parameter uncertainty
        if n_bootstrap > 0:
            selector.bootstrap_best_model(n_boot=n_bootstrap, seed=seed)

        # store results for compatibility
        self._store_fitted_model_results(best, selector)
        self.fitted_model = best
        self.model_selector = selector

        return best
    
    def _store_fitted_model_results(
        self,
        fitted: 'FittedVariogramModel',
        selector: 'VariogramModelSelector'
    ) -> None:
        """Transfer FittedVariogramModel results to VariogramAnalysis attributes.

        Ensures backward compatibility with code expecting traditional attributes.
        Stores both full range (2.5th/97.5th) and 1σ range (16th/84th) percentiles.
        """
        model = fitted.composite_model
        params = fitted.params

        # extract sills, ranges from composite model
        # note: 'wavelength' (damped_hole_effect) is treated as a range-like parameter
        sills = []
        ranges = []
        sill_indices = []
        range_indices = []
        range_labels = []  # Track whether each range is 'range' or 'wavelength'
        param_offset = 0

        for i, spec in enumerate(model._components):
            comp_params = model.get_component_params(i)
            if spec.has_sill:
                sills.append(comp_params[0])
                sill_indices.append(param_offset)
            for range_key in ('range', 'wavelength'):
                if range_key in spec.param_names:
                    range_idx = spec.param_names.index(range_key)
                    ranges.append(comp_params[range_idx])
                    range_indices.append(param_offset + range_idx)
                    range_labels.append(range_key)
                    break
            param_offset += len(spec.param_names)

        self.sills = np.array(sills) if sills else np.array([])
        self.ranges = np.array(ranges) if ranges else np.array([])
        self.range_labels = range_labels  # 'range' or 'wavelength' per entry
        self.best_nugget = model.get_nugget() if model.include_nugget else None
        self.best_params = params
        self.best_aic = fitted.aic
        self.best_bic = fitted.bic

        # callable for the model function
        self.best_model_func = lambda h, *p: model(np.asarray(h, dtype=float))
        self.fitted_variogram = model(self.lags)

        self.best_model_config = {
            'components': len(model.component_names),
            'nugget': model.include_nugget,
            'model_types': model.component_names,
        }

        # Store bootstrap samples without appending the optimal point
        # estimate — appending it biases percentile estimates.
        if fitted.param_samples is not None and len(fitted.param_samples) > 0:
            self.param_samples = fitted.param_samples
        else:
            self.param_samples = np.array([params])

        if self.param_samples.size > 0:
            samples = self.param_samples

            # extract sill percentiles
            if sill_indices:
                sill_samps = samples[:, sill_indices]
                # full range (2.5th to 97.5th)
                self.sills_min = np.percentile(sill_samps, 2.5, axis=0)
                self.sills_max = np.percentile(sill_samps, 97.5, axis=0)
                self.sills_median = np.percentile(sill_samps, 50, axis=0)
                # 1σ range (16th to 84th)
                self.sills_p16 = np.percentile(sill_samps, 16, axis=0)
                self.sills_p84 = np.percentile(sill_samps, 84, axis=0)
            else:
                self.sills_min = self.sills_max = self.sills_median = np.array([])
                self.sills_p16 = self.sills_p84 = np.array([])

            # extract range percentiles
            if range_indices:
                range_samps = samples[:, range_indices]
                # full range (2.5th to 97.5th)
                self.ranges_min = np.percentile(range_samps, 2.5, axis=0)
                self.ranges_max = np.percentile(range_samps, 97.5, axis=0)
                self.ranges_median = np.percentile(range_samps, 50, axis=0)
                # 1σ range (16th to 84th)
                self.ranges_p16 = np.percentile(range_samps, 16, axis=0)
                self.ranges_p84 = np.percentile(range_samps, 84, axis=0)
            else:
                self.ranges_min = self.ranges_max = self.ranges_median = np.array([])
                self.ranges_p16 = self.ranges_p84 = np.array([])

            # extract nugget percentiles
            if model.include_nugget:
                nugget_idx = model.n_params - 1  # Nugget is always last
                nug_samps = samples[:, nugget_idx]
                # full range
                self.min_nugget = float(np.percentile(nug_samps, 2.5))
                self.max_nugget = float(np.percentile(nug_samps, 97.5))
                self.median_nugget = float(np.percentile(nug_samps, 50))
                # 1σ range
                self.nugget_p16 = float(np.percentile(nug_samps, 16))
                self.nugget_p84 = float(np.percentile(nug_samps, 84))
            else:
                self.min_nugget = self.max_nugget = self.median_nugget = None
                self.nugget_p16 = self.nugget_p84 = None
        else:
            # fallback to point estimates when no bootstrap samples
            self.sills_min = self.sills_max = self.sills_median = self.sills
            self.sills_p16 = self.sills_p84 = self.sills
            self.ranges_min = self.ranges_max = self.ranges_median = self.ranges
            self.ranges_p16 = self.ranges_p84 = self.ranges
            if model.include_nugget:
                self.min_nugget = self.max_nugget = self.median_nugget = self.best_nugget
                self.nugget_p16 = self.nugget_p84 = self.best_nugget
            else:
                self.min_nugget = self.max_nugget = self.median_nugget = None
                self.nugget_p16 = self.nugget_p84 = None

        self.cv_mean_error_best_aic = {'rmse': fitted.cv_rmse} if fitted.cv_rmse else None

        # transfer MSSPE diagnostics
        self.best_msspe = fitted.msspe
        self.best_loocv_result = fitted.loocv_result

    def get_model_comparison_summary(self) -> str:
        """Get a summary of all fitted models for comparison.

        Returns formatted table with AIC, BIC, CV-RMSE, MSSPE, and
        Akaike weights.  MSSPE column is included when at least one
        candidate has been evaluated with kriging LOOCV.
        """
        if not hasattr(self, 'model_selector') or self.model_selector is None:
            raise RuntimeError("No model comparison available. Call fit_best_model_auto() first.")

        selector = self.model_selector

        # check if any model has MSSPE computed
        has_msspe = any(
            m.msspe is not None for m in selector.fitted_models
        )

        lines = ["=" * 105]
        lines.append("VARIOGRAM MODEL SELECTION SUMMARY")
        lines.append("=" * 105)
        header = f"{'Model':<40} {'AIC':>10} {'BIC':>10} {'CV-RMSE':>10}"
        if has_msspe:
            header += f" {'MSSPE':>12}"
        header += f" {'Weight':>10}"
        lines.append(header)
        lines.append("-" * 105)

        weights = selector.model_weights if selector.model_weights is not None else [0] * len(selector.fitted_models)
        sorted_models = sorted(zip(selector.fitted_models, weights), key=lambda x: x[0].aic)

        for model, weight in sorted_models:
            name = "+".join(model.composite_model.component_names)
            if model.composite_model.include_nugget:
                name += "+nugget"
            cv_str = f"{model.cv_rmse:.4f}" if model.cv_rmse else "N/A"
            marker = " *" if model is selector.best_model else ""
            row = f"{name:<40} {model.aic:>10.2f} {model.bic:>10.2f} {cv_str:>10}"
            if has_msspe:
                if model.msspe is not None:
                    msspe_str = f"{model.msspe:.3f}"
                    std = getattr(model, 'msspe_std', None)
                    if std is not None:
                        msspe_str += f"±{std:.3f}"
                else:
                    msspe_str = "N/A"
                row += f" {msspe_str:>12}"
            row += f" {weight:>10.4f}{marker}"
            lines.append(row)

        lines.append("=" * 95)
        return "\n".join(lines)
    
    def get_bma_variogram_function(self) -> Callable:
        """Get Bayesian Model Averaged variogram function.

        Computes γ_BMA(h) = Σᵢ wᵢ · γᵢ(h) where wᵢ are Akaike weights.

        References
        ----------
        Hoeting, J.A., et al. (1999). Bayesian Model Averaging: A Tutorial.
        Statistical Science, 14(4), 382-417.
        """
        if not hasattr(self, 'model_selector') or self.model_selector is None:
            raise RuntimeError("No model selector available. Call fit_best_model_auto() first.")

        return self.model_selector.get_bma_variogram()
    
    # ── kriging-based cross-validation ──────────────────────────────

    @staticmethod
    def kriging_loocv(
        coords: np.ndarray,
        values: np.ndarray,
        variogram_func: Callable,
        n_subset: int = 500,
        seed: Optional[int] = None,
    ) -> 'KrigingLOOCVResult':
        """Leave-one-out cross-validation using ordinary kriging.

        For each point, removes it from the dataset, predicts its value
        via ordinary kriging using the remaining points and the supplied
        variogram model, then compares the prediction with the observed
        value.  The key diagnostic is the MSSPE — like a reduced-χ²
        statistic for spatial predictions.

        Parameters
        ----------
        coords : ndarray, shape (n, 2)
            Spatial coordinates of observed locations.
        values : ndarray, shape (n,)
            Observed values (e.g. elevation differences).
        variogram_func : callable
            Fitted variogram model γ(h).  Must accept an ndarray of
            distances and return semivariances.  A
            ``CompositeVariogramModel`` instance works directly.
        n_subset : int, default 500
            Maximum number of points for the CV.  If ``len(values)``
            exceeds this, a random subsample is drawn.  Keeps runtime
            manageable (≈ seconds rather than minutes).
        seed : int, optional
            Random seed for reproducible subsampling.

        Returns
        -------
        KrigingLOOCVResult
            Dataclass with ``msspe``, ``mean_error``, ``rmse``,
            ``mean_standardized_error``, ``n_points``, ``n_failed``.

        Notes
        -----
        The ordinary kriging system for *n* data points is the
        (n+1) × (n+1) augmented matrix problem::

            ⎡ Γ   𝟏 ⎤ ⎡ λ ⎤   ⎡ γ₀ ⎤
            ⎣ 𝟏ᵀ  0 ⎦ ⎣ μ ⎦ = ⎣  1 ⎦

        where Γ is the *n × n* semivariance matrix between data
        locations, γ₀ is the semivariance from each data location to
        the prediction point, λ are the kriging weights, and μ is the
        Lagrange multiplier enforcing unbiasedness.

        For each LOO iteration the prediction point is removed from the
        data, giving an (*n*−1+1) × (*n*−1+1) system.  At
        ``n_subset=500`` this takes ≈ 1 s on typical hardware.

        References
        ----------
        Cressie, N. (1993). *Statistics for Spatial Data*, rev. ed.,
        Wiley, Section 5.6.

        Webster, R. & Oliver, M.A. (2007). *Geostatistics for
        Environmental Scientists*, 2nd ed., Wiley, Section 8.3.

        Pang, Y. et al. (2023). Enhanced Kriging leave-one-out cross-
        validation in improving model estimation and optimization.
        *Comput. Methods Appl. Mech. Engrg.*, 414, 116194.
        doi:10.1016/j.cma.2023.116194
        — confirms that using global (pre-fitted) hyperparameters for
        all LOO iterations ("enhanced-LOOCV") is both more accurate
        and more efficient than re-optimising per iteration.
        """
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"coords must be shape (n, 2), got {coords.shape}"
            )
        if len(coords) != len(values):
            raise ValueError(
                f"coords ({len(coords)}) and values ({len(values)}) "
                f"must have the same length."
            )

        n = len(values)
        rng = np.random.default_rng(seed)

        # ── subsample if too many points ──
        if n > n_subset:
            idx = rng.choice(n, n_subset, replace=False)
            coords = coords[idx]
            values = values[idx]
            n = n_subset

        # ── pairwise distance matrix (n × n) ──
        dx = coords[:, 0:1] - coords[:, 0:1].T
        dy = coords[:, 1:2] - coords[:, 1:2].T
        dist_matrix = np.sqrt(dx**2 + dy**2)

        # ── semivariance matrix ──
        # The variogram model handles γ(0) = 0 correctly via the
        # nugget function (which returns 0 at h=0, c₀ at h>0).
        gamma_matrix = variogram_func(dist_matrix)

        # ── leave-one-out loop ──
        errors = np.empty(n)
        variances = np.empty(n)
        n_failed = 0

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            # semivariance sub-matrix for remaining n-1 points
            gamma_sub = gamma_matrix[np.ix_(mask, mask)]  # (n-1, n-1)
            n_sub = n - 1

            # build augmented ordinary-kriging system
            A = np.empty((n_sub + 1, n_sub + 1))
            A[:n_sub, :n_sub] = gamma_sub
            A[:n_sub, n_sub] = 1.0
            A[n_sub, :n_sub] = 1.0
            A[n_sub, n_sub] = 0.0

            # RHS: semivariances from remaining points to prediction point
            gamma_rhs = gamma_matrix[mask, i]
            b = np.empty(n_sub + 1)
            b[:n_sub] = gamma_rhs
            b[n_sub] = 1.0

            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                errors[i] = np.nan
                variances[i] = np.nan
                n_failed += 1
                continue

            # kriging prediction:  ẑ = λᵀ z_{-i}
            z_pred = np.dot(x[:n_sub], values[mask])

            # kriging variance:  σ² = λ̃ᵀ b  (includes Lagrange term)
            sigma2 = np.dot(x, b)

            errors[i] = values[i] - z_pred
            variances[i] = max(sigma2, np.finfo(float).eps)

        # ── aggregate diagnostics ──
        valid = np.isfinite(errors) & np.isfinite(variances) & (variances > 0)
        e = errors[valid]
        v = variances[valid]
        n_valid = int(np.sum(valid))

        if n_valid == 0:
            return KrigingLOOCVResult(
                msspe=np.nan,
                mean_error=np.nan,
                rmse=np.nan,
                mean_standardized_error=np.nan,
                n_points=0,
                n_failed=n_failed,
            )

        sigma = np.sqrt(v)
        return KrigingLOOCVResult(
            msspe=float(np.mean(e**2 / v)),
            mean_error=float(np.mean(e)),
            rmse=float(np.sqrt(np.mean(e**2))),
            mean_standardized_error=float(np.mean(e / sigma)),
            n_points=n_valid,
            n_failed=n_failed,
        )

    def fit_variogram_ensemble(
        self,
        n_realizations: int = 50,
        area_side: float = 1000,
        samples_per_area: float = 1.0,
        max_samples: int = 10000,
        bin_width: float = 50.0,
        max_lag_multiplier: float = 1 / 3,
        *,
        model_types: Optional[List[str]] = None,
        max_components: int = 2,
        include_nugget: bool = True,
        criterion: str = 'msspe',
        msspe_n_subset: int = 500,
        msspe_n_runs: int = 5,
        estimator: str = 'matheron',
        min_pairs: Optional[int] = 30,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> 'EnsembleVariogramResult':
        """Fit variogram models to independent spatial samples — ensemble approach.

        Generates ``n_realizations`` independent empirical variograms
        (each from a fresh random sample of the raster), runs the full
        model selection pipeline on each, and collects results.  This
        captures both *model selection uncertainty* and *parameter
        uncertainty* in a single Monte Carlo experiment.

        The approach is analogous to a spatial bootstrap: instead of
        perturbing a single empirical variogram, it re-samples the
        underlying field.  Each realization goes through candidate
        generation → WLS fitting → MSSPE-based selection independently,
        so the ensemble naturally reflects how sensitive model choice
        and parameter estimates are to the particular sample drawn.

        Parameters
        ----------
        n_realizations : int
            Number of independent variogram realizations (default 50).
        area_side : float
            Reference side length for sampling density (e.g. 1000 for
            km² if coordinates are in meters).
        samples_per_area : float
            Sampling density per reference area unit.
        max_samples : int
            Maximum points per realization.
        bin_width : float
            Lag bin width (same units as coordinates).
        max_lag_multiplier : float
            Maximum lag as fraction of domain diagonal.
        model_types : list of str, optional
            Model types to try (default: spherical, exponential,
            gaussian, matern).
        max_components : int
            Maximum number of nested components (default 2, max 3).
        include_nugget : bool
            Whether to include nugget variants in candidates.
        criterion : str
            Selection criterion (default 'msspe').
        msspe_n_subset : int
            Points per MSSPE LOOCV run.
        msspe_n_runs : int
            MSSPE subsample repeats per candidate per realization.
            Reduced from default 10 to 5 because the outer ensemble
            loop already provides variance estimation.
        estimator : str
            Empirical variogram estimator ('matheron' or 'cressie_hawkins').
        min_pairs : int or None
            Minimum pair count per lag bin for inclusion.
        seed : int, optional
            Base random seed for reproducibility.
        verbose : bool
            Print progress (default True).

        Returns
        -------
        EnsembleVariogramResult
            Dataclass with per-realization parameters, model selection
            frequencies, and aggregate statistics.  Call ``.summary()``
            for a text report or ``.plot()`` for a multi-panel figure.

        Notes
        -----
        Runtime scales as ``n_realizations × n_candidates × msspe_n_runs``.
        With 50 realizations, ~30 candidates, and 5 MSSPE runs each,
        expect ~7500 LOOCV evaluations — on the order of 1–2 hours for
        n_subset=500 on typical hardware.  Reduce ``n_realizations`` or
        ``msspe_n_runs`` for faster exploratory runs.

        References
        ----------
        Marchetti, Y., Paciorek, C.J., & Genton, M.G. (2018).
        An assessment of model selection uncertainty in spatial
        prediction. *Environmetrics*, 29(7–8), e2530.
        doi:10.1002/env.2530

        Lark, R.M. (2000). A comparison of some robust estimators
        of the variogram for use in soil survey. *Eur. J. Soil Sci.*,
        51, 137–157.
        """
        if model_types is None:
            model_types = ['spherical', 'exponential', 'gaussian', 'matern']

        # reproducible child seeds
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_realizations)

        # storage
        records: List[Dict[str, Any]] = []
        all_empirical: List[np.ndarray] = []
        all_fitted_curves: List[np.ndarray] = []
        common_lags: Optional[np.ndarray] = None
        n_failed = 0

        for r in range(n_realizations):
            run_seed = int(child_seeds[r].generate_state(1)[0])

            if verbose:
                print(f"  Realization {r + 1}/{n_realizations} "
                      f"(seed={run_seed})", end=" ... ", flush=True)

            try:
                # ── 1. Sample raster and compute empirical variogram ──
                (bin_counts, variogram_est, n_bins,
                 min_dist, max_dist, max_lag) = self.numba_variogram(
                    area_side, samples_per_area, max_samples,
                    bin_width, max_lag_multiplier,
                    seed=run_seed,
                    estimator=estimator,
                )

                # build lags and filter valid bins
                n_lags = variogram_est.size
                lags = np.linspace(
                    bin_width / 2,
                    bin_width * n_lags - bin_width / 2,
                    n_lags,
                )
                valid = np.isfinite(variogram_est)
                if valid.sum() < 5:
                    raise ValueError("Fewer than 5 valid lag bins.")

                emp_vario = variogram_est[valid]
                emp_lags = lags[valid]
                emp_counts = bin_counts[valid] if bin_counts is not None else None

                # Store common lag vector from first success
                if common_lags is None:
                    common_lags = emp_lags.copy()

                # Compute sigma (std across bins) — use simple approach
                # for single realizations: std ∝ γ(h) / √N(h)
                if emp_counts is not None:
                    safe_counts = np.where(emp_counts > 0, emp_counts, 1)
                    sigma = emp_vario / np.sqrt(safe_counts)
                    sigma = np.where(sigma > 0, sigma, np.finfo(float).eps)
                else:
                    sigma = None

                # ── 2. Fit all candidate models ──
                selector = VariogramModelSelector(
                    lags=emp_lags,
                    empirical_variogram=emp_vario,
                    pair_counts=emp_counts,
                    sigma=sigma,
                    min_pairs=min_pairs,
                )

                # override model lists
                available = MODEL_REGISTRY.list_models()
                selector.BOUNDED_MODELS = [m for m in model_types
                                           if m in available and MODEL_REGISTRY.is_bounded(m)]
                selector.UNBOUNDED_MODELS = [m for m in model_types
                                             if m in available and not MODEL_REGISTRY.is_bounded(m)]

                selector.fit_all_candidates(
                    max_components=min(max_components, 3),
                    include_nugget=include_nugget,
                    include_unbounded=bool(selector.UNBOUNDED_MODELS),
                    compute_cv=False,
                    seed=run_seed,
                )

                if not selector.fitted_models:
                    raise ValueError("No models successfully fitted.")

                # ── 3. Compute MSSPE and select best ──
                if criterion == 'msspe':
                    rdh = self.raster_data_handler
                    coords = rdh.coords
                    values = rdh.samples

                    run_rng = np.random.default_rng(run_seed)
                    msspe_seeds = run_rng.integers(0, 2**31, size=msspe_n_runs)

                    for fitted in selector.fitted_models:
                        if not fitted.composite_model.is_stationary:
                            continue
                        run_results: list = []
                        for ms in msspe_seeds:
                            try:
                                result = self.kriging_loocv(
                                    coords, values,
                                    fitted.composite_model,
                                    n_subset=msspe_n_subset,
                                    seed=int(ms),
                                )
                                if np.isfinite(result.msspe):
                                    run_results.append(result)
                            except Exception:
                                continue
                        if run_results:
                            agg = AggregatedLOOCVResult.from_results(run_results)
                            fitted.msspe = agg.msspe_mean
                            fitted.msspe_std = agg.msspe_std
                            fitted.msspe_n_runs = agg.n_runs
                            fitted.loocv_result = agg

                best = selector.select_best(criterion=criterion)

                # ── 4. Extract parameters ──
                model = best.composite_model
                params = best.params
                desc = model.description()

                # extract sills, ranges, nugget, nu
                sills_r = []
                ranges_r = []
                nu_val = np.nan

                for i, spec in enumerate(model._components):
                    comp_params = model.get_component_params(i)
                    if spec.has_sill:
                        sills_r.append(comp_params[0])
                    for rkey in ('range', 'wavelength'):
                        if rkey in spec.param_names:
                            ridx = spec.param_names.index(rkey)
                            ranges_r.append(
                                comp_params[ridx] * (spec.practical_range_factor or 1.0)
                            )
                            break
                    if 'nu' in spec.param_names:
                        nu_idx = spec.param_names.index('nu')
                        nu_val = comp_params[nu_idx]

                nugget_val = model.get_nugget() if model.include_nugget else np.nan

                # fitted curve evaluated at common lags
                fitted_curve = model(common_lags) if common_lags is not None else np.array([])

                # empirical variogram padded/trimmed to common lag length
                if common_lags is not None:
                    emp_padded = np.full(len(common_lags), np.nan)
                    n_copy = min(len(emp_vario), len(common_lags))
                    emp_padded[:n_copy] = emp_vario[:n_copy]
                else:
                    emp_padded = emp_vario

                record = {
                    'model_description': desc,
                    'component_names': list(model.component_names),
                    'include_nugget': model.include_nugget,
                    'params': params.copy(),
                    'param_names': model.param_names,
                    'sills': sills_r,
                    'ranges': ranges_r,
                    'nugget': nugget_val,
                    'nu': nu_val,
                    'msspe': best.msspe,
                    'msspe_std': best.msspe_std,
                    'aic': best.aic,
                    'rss': best.rss,
                    'fitted_curve': fitted_curve,
                    'empirical_variogram': emp_padded,
                    'seed': run_seed,
                }
                records.append(record)
                all_fitted_curves.append(fitted_curve)
                all_empirical.append(emp_padded)

                if verbose:
                    msspe_str = (f"MSSPE={best.msspe:.3f}" if best.msspe is not None
                                 else "MSSPE=N/A")
                    print(f"{desc}  {msspe_str}")

            except Exception as e:
                n_failed += 1
                if verbose:
                    print(f"FAILED ({e})")
                continue

        # ── Assemble ensemble result ──
        if not records:
            raise RuntimeError(
                f"All {n_realizations} realizations failed. "
                "Check raster data and fitting parameters."
            )

        # model selection counts
        from collections import Counter
        desc_list = [rec['model_description'] for rec in records]
        model_counts = dict(Counter(desc_list))
        n_ok = len(records)
        model_fractions = {k: v / n_ok for k, v in model_counts.items()}

        # pad sills/ranges to common width
        max_n_sills = max(len(rec['sills']) for rec in records)
        max_n_ranges = max(len(rec['ranges']) for rec in records)

        sills_arr = np.full((n_ok, max(max_n_sills, 1)), np.nan)
        ranges_arr = np.full((n_ok, max(max_n_ranges, 1)), np.nan)
        nuggets_arr = np.array([rec['nugget'] for rec in records])
        nus_arr = np.array([rec['nu'] for rec in records])
        msspes_arr = np.array([
            rec['msspe'] if rec['msspe'] is not None else np.nan
            for rec in records
        ])

        for i, rec in enumerate(records):
            for j, s in enumerate(rec['sills']):
                sills_arr[i, j] = s
            for j, rng in enumerate(rec['ranges']):
                ranges_arr[i, j] = rng

        # stack fitted curves and empirical variograms
        variograms_arr = np.array(all_fitted_curves)
        empirical_arr = np.array(all_empirical)

        result = EnsembleVariogramResult(
            n_realizations=n_realizations,
            n_failed=n_failed,
            model_counts=model_counts,
            model_fractions=model_fractions,
            sills=sills_arr,
            ranges=ranges_arr,
            nuggets=nuggets_arr,
            nus=nus_arr,
            msspes=msspes_arr,
            lags=common_lags if common_lags is not None else np.array([]),
            variograms=variograms_arr,
            empirical_variograms=empirical_arr,
            per_realization=records,
        )

        if verbose:
            print("\n" + result.summary())

        return result

    def plot_best_model(self):
        """
        Plot mean variogram ± spread and fitted model; also show bar plot of mean pair counts.

        Uses two-level uncertainty shading:
        - Full range (very light shading, α=0.1): 2.5th to 97.5th percentiles
        - 1σ range (darker shading, α=0.3): 16th to 84th percentiles
        - Optimal value: dashed line
        """
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        if any(attr is None for attr in (self.mean_variogram, self.err_variogram, self.mean_count, self.lags, self.fitted_variogram)):
            raise RuntimeError("Missing variogram data. Call calculate_mean_variogram_numba() and fit method first.")

        n = min(len(self.lags), len(self.mean_variogram), len(self.err_variogram), len(self.fitted_variogram))
        lags = self.lags[:n]
        gamma = self.mean_variogram[:n]
        errs = self.err_variogram[:n]
        model = self.fitted_variogram[:n]
        counts = self.mean_count[:n]

        valid_counts = (~np.isnan(counts)) & (counts > 0)
        count_lags = lags[valid_counts]
        count_vals = counts[valid_counts]

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 8), sharex=True)

        # guard single-bin bar width
        if len(lags) > 1:
            bar_width = (lags[1] - lags[0]) * 0.9
        else:
            bar_width = (lags[0] if len(lags) else 1.0) * 0.9
        axs[0].bar(count_lags, count_vals, width=bar_width, color='orange', alpha=0.5)
        axs[0].set_ylabel('Mean Count')
        axs[0].tick_params(labelbottom=False)

        # plot empirical variogram and fitted model
        axs[1].errorbar(lags, gamma, yerr=errs, fmt='o-', color='blue', label='Mean Variogram ± spread')
        axs[1].plot(lags, model, 'r-', label='Fitted Model')

        # range uncertainty shading (vertical bands)
        colors = ['red', 'green', 'blue']
        if self.ranges is not None and self.ranges_min is not None and self.ranges_max is not None:
            ylim = axs[1].get_ylim()
            for i, (r, rmin, rmax) in enumerate(zip(self.ranges, self.ranges_min, self.ranges_max)):
                c = colors[i % len(colors)]
                # full range (very light shading)
                axs[1].fill_betweenx(ylim, rmin, rmax, color=c, alpha=0.1)
                # 1σ range (darker shading) if available
                if hasattr(self, 'ranges_p16') and self.ranges_p16 is not None:
                    r_p16 = self.ranges_p16[i]
                    r_p84 = self.ranges_p84[i]
                    axs[1].fill_betweenx(ylim, r_p16, r_p84, color=c, alpha=0.3)
                # optimal value (dashed line)
                axs[1].axvline(r, color=c, linestyle='--', linewidth=1.5)

        # nugget uncertainty shading (horizontal bands)
        if self.best_nugget is not None and self.min_nugget is not None and self.max_nugget is not None:
            # full range (very light shading)
            axs[1].fill_between(lags, [self.min_nugget] * len(lags), [self.max_nugget] * len(lags), color='orange', alpha=0.1)
            # 1σ range (darker shading) if available
            if hasattr(self, 'nugget_p16') and self.nugget_p16 is not None:
                axs[1].fill_between(lags, [self.nugget_p16] * len(lags), [self.nugget_p84] * len(lags), color='orange', alpha=0.3)
            # optimal value (dashed line)
            axs[1].axhline(self.best_nugget, color='orange', linestyle='--', linewidth=1.5)

        # build custom legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='blue', label='Mean Variogram ± spread', linestyle='-'),
            Line2D([0], [0], color='r', linestyle='-', label='Fitted Model'),
            Patch(facecolor='gray', alpha=0.1, label='Full range (95%)'),
            Patch(facecolor='gray', alpha=0.4, label='1σ range (68%)'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Optimal'),
        ]
        # add color swatches for each range/wavelength
        if self.ranges is not None:
            for i in range(len(self.ranges)):
                c = colors[i % len(colors)]
                lbl = 'Wavelength' if (hasattr(self, 'range_labels') and
                    i < len(self.range_labels) and
                    self.range_labels[i] == 'wavelength') else 'Range'
                legend_elements.append(Patch(facecolor=c, alpha=0.5, label=f'{lbl} {i + 1}'))
        # add nugget to legend if present
        if self.best_nugget is not None:
            legend_elements.append(Patch(facecolor='orange', alpha=0.5, label='Nugget'))

        axs[1].set_xlabel('Lag Distance')
        axs[1].set_ylabel('Semivariance')
        axs[1].legend(handles=legend_elements, loc='upper right')

        # add model info to title
        rmse_str = ""
        if isinstance(self.cv_mean_error_best_aic, dict):
            rmse = self.cv_mean_error_best_aic.get('rmse', None)
            if rmse is not None:
                rmse_str = f'RMSE (CV): {rmse:.4f}'
        axs[1].set_title(rmse_str)
        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.tight_layout()
        return fig

    # alias for backward compatibility
    def plot_best_spherical_model(self):
        """Alias for plot_best_model() for backward compatibility."""
        return self.plot_best_model()


# class RegionalUncertaintyEstimator:

