"""Synthetic variogram fitting diagnostic suite.

Generates synthetic empirical variograms with KNOWN parameters across
different model types and noise levels, then tests the fit_best_model_auto
pathway (VariogramModelSelector) to evaluate:

1. Parameter recovery accuracy (sills, ranges, nuggets)
2. Nugget estimation bias (overestimation tendency)
3. Range constraint behavior (ranges beyond half max lag)
4. Model selection correctness (does AIC pick the right model?)
5. Effect of noise level on all of the above

Usage:
    python tests/test_synthetic_variogram_fitting.py
"""

from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

# ensure topochange is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topochange.variogram_models import (
    spherical, exponential, gaussian, matern, nugget as nugget_func,
    MODEL_REGISTRY,
)
from topochange.composite_variogram import CompositeVariogramModel
from topochange.variogram import VariogramModelSelector, FittedVariogramModel

# ── output directory ──────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent.parent / "synthetic_fitting_results"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# ── synthetic variogram generation ────────────────────────────────────

def generate_synthetic_variogram(
    model_func,
    params: dict,
    nugget_value: float = 0.0,
    n_lags: int = 40,
    max_lag: float = 500.0,
    noise_std_frac: float = 0.05,
    n_runs: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic empirical variogram with known parameters.

    Simulates multiple "runs" of empirical variogram estimation by adding
    heteroscedastic noise to the true model, mimicking the variance you'd
    see across spatially subsampled variogram estimates.

    Parameters
    ----------
    model_func : callable
        True variogram model function (e.g., spherical).
    params : dict
        Parameters for model_func (e.g., {'sill': 1.0, 'range_': 100.0}).
    nugget_value : float
        True nugget value to add.
    n_lags : int
        Number of lag bins.
    max_lag : float
        Maximum lag distance.
    noise_std_frac : float
        Noise standard deviation as fraction of local semivariance.
        Mimics sampling variability in empirical variograms.
    n_runs : int
        Number of synthetic runs (to compute mean and sigma).
    seed : int
        Random seed.

    Returns
    -------
    lags : ndarray
        Lag distances (bin centers).
    mean_variogram : ndarray
        Mean empirical semivariance across runs.
    sigma_variogram : ndarray
        Standard deviation across runs.
    """
    rng = np.random.default_rng(seed)
    bin_width = max_lag / n_lags
    lags = np.linspace(bin_width / 2, max_lag - bin_width / 2, n_lags)

    # true variogram
    true_gamma = model_func(lags, **params)
    if nugget_value > 0:
        true_gamma = true_gamma + nugget_func(lags, nugget_value)

    # generate noisy runs
    # noise scales with sqrt(gamma) + baseline, mimicking real variogram variance
    # (Cressie 1985: variance of Matheron estimator ~ gamma(h)^2 / N_h)
    all_runs = np.zeros((n_runs, n_lags))
    for r in range(n_runs):
        # heteroscedastic noise: larger at larger semivariances
        noise_std = noise_std_frac * (true_gamma + 0.01 * np.max(true_gamma))
        noise = rng.normal(0, noise_std)
        all_runs[r] = np.maximum(true_gamma + noise, 0)  # semivariance >= 0

    mean_variogram = np.mean(all_runs, axis=0)
    sigma_variogram = np.std(all_runs, axis=0)
    sigma_variogram[sigma_variogram == 0] = np.finfo(float).eps

    return lags, mean_variogram, sigma_variogram


# ── test scenarios ────────────────────────────────────────────────────

@dataclass
class TestScenario:
    """A synthetic test case with known ground truth."""
    name: str
    model_type: str              # e.g. 'spherical'
    true_params: dict            # params for model function
    true_nugget: float           # true nugget value
    max_lag: float = 500.0
    n_lags: int = 40
    description: str = ""


def build_scenarios() -> List[TestScenario]:
    """Build the full suite of synthetic test scenarios."""
    scenarios = []

    # ── 1. Single-component models with nugget sweep ──
    nugget_fractions = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.60]

    for nug_frac in nugget_fractions:
        total_sill = 1.0
        nug = nug_frac * total_sill
        partial_sill = total_sill - nug

        # Spherical
        scenarios.append(TestScenario(
            name=f"spherical_nug{int(nug_frac*100):02d}",
            model_type="spherical",
            true_params={"sill": partial_sill, "range_": 150.0},
            true_nugget=nug,
            description=f"Spherical, nugget={nug_frac:.0%} of total sill",
        ))

        # Exponential
        scenarios.append(TestScenario(
            name=f"exponential_nug{int(nug_frac*100):02d}",
            model_type="exponential",
            true_params={"sill": partial_sill, "range_": 50.0},
            true_nugget=nug,
            description=f"Exponential (practical range=150), nugget={nug_frac:.0%}",
        ))

        # Gaussian
        scenarios.append(TestScenario(
            name=f"gaussian_nug{int(nug_frac*100):02d}",
            model_type="gaussian",
            true_params={"sill": partial_sill, "range_": 87.0},
            true_nugget=nug,
            description=f"Gaussian (practical range~150), nugget={nug_frac:.0%}",
        ))

    # ── 2. Matern with varying smoothness ──
    for nu in [0.5, 1.5, 2.5]:
        scenarios.append(TestScenario(
            name=f"matern_nu{nu:.1f}_nug10",
            model_type="matern",
            true_params={"sill": 0.9, "range_": 80.0, "nu": nu},
            true_nugget=0.1,
            description=f"Matern nu={nu}, nugget=10%",
        ))

    # ── 3. Range near/beyond half lag ──
    # These test whether the fitter correctly handles ranges close to
    # and beyond the empirical half-lag boundary
    for range_frac in [0.15, 0.30, 0.50, 0.70, 0.90]:
        max_lag = 500.0
        true_range = range_frac * max_lag
        scenarios.append(TestScenario(
            name=f"spherical_range{int(range_frac*100):02d}pct",
            model_type="spherical",
            true_params={"sill": 0.9, "range_": true_range},
            true_nugget=0.1,
            max_lag=max_lag,
            description=f"Spherical range={range_frac:.0%} of max lag",
        ))

    # ── 4. Two-component nested models ──
    # Short + long range structure
    scenarios.append(TestScenario(
        name="nested_sph_sph",
        model_type="spherical+spherical",
        true_params={
            "comp0": {"sill": 0.4, "range_": 50.0},
            "comp1": {"sill": 0.5, "range_": 250.0},
        },
        true_nugget=0.1,
        description="Nested spherical: short (50) + long (250) range, nugget=10%",
    ))

    scenarios.append(TestScenario(
        name="nested_exp_sph",
        model_type="exponential+spherical",
        true_params={
            "comp0": {"sill": 0.3, "range_": 30.0},
            "comp1": {"sill": 0.6, "range_": 200.0},
        },
        true_nugget=0.1,
        description="Nested exponential(30) + spherical(200), nugget=10%",
    ))

    # ── 5. High-sill-spread scenario ──
    # Extreme ratio between component sills
    scenarios.append(TestScenario(
        name="nested_extreme_sill_ratio",
        model_type="spherical+spherical",
        true_params={
            "comp0": {"sill": 0.05, "range_": 30.0},
            "comp1": {"sill": 0.85, "range_": 300.0},
        },
        true_nugget=0.1,
        description="Nested spherical with extreme sill ratio (0.05 vs 0.85)",
    ))

    return scenarios


# ── synthetic variogram for nested models ─────────────────────────────

MODEL_FUNCS = {
    "spherical": spherical,
    "exponential": exponential,
    "gaussian": gaussian,
    "matern": matern,
}


def generate_nested_variogram(
    scenario: TestScenario,
    noise_std_frac: float = 0.05,
    n_runs: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic variogram for any scenario (single or nested).

    Returns lags, mean_variogram, sigma_variogram, true_variogram.
    """
    rng = np.random.default_rng(seed)
    bin_width = scenario.max_lag / scenario.n_lags
    lags = np.linspace(bin_width / 2, scenario.max_lag - bin_width / 2, scenario.n_lags)

    # compute true variogram
    if "+" in scenario.model_type:
        # nested model
        model_types = scenario.model_type.split("+")
        true_gamma = np.zeros_like(lags)
        for i, mt in enumerate(model_types):
            func = MODEL_FUNCS[mt]
            comp_params = scenario.true_params[f"comp{i}"]
            true_gamma += func(lags, **comp_params)
    else:
        func = MODEL_FUNCS[scenario.model_type]
        true_gamma = func(lags, **scenario.true_params)

    if scenario.true_nugget > 0:
        true_gamma += nugget_func(lags, scenario.true_nugget)

    # generate noisy runs
    all_runs = np.zeros((n_runs, scenario.n_lags))
    for r in range(n_runs):
        noise_std = noise_std_frac * (true_gamma + 0.01 * np.max(true_gamma))
        noise = rng.normal(0, noise_std)
        all_runs[r] = np.maximum(true_gamma + noise, 0)

    mean_variogram = np.mean(all_runs, axis=0)
    sigma_variogram = np.std(all_runs, axis=0)
    sigma_variogram[sigma_variogram == 0] = np.finfo(float).eps

    return lags, mean_variogram, sigma_variogram, true_gamma


# ── fitting wrapper ───────────────────────────────────────────────────

@dataclass
class FitResult:
    """Results of fitting a single scenario."""
    scenario_name: str
    true_model_type: str
    true_nugget: float
    true_total_sill: float
    true_params: dict

    # fitted
    selected_model_name: str
    correct_model_selected: bool
    fitted_nugget: Optional[float]
    fitted_sills: list
    fitted_ranges: list
    fitted_total_sill: float

    # errors
    nugget_error: Optional[float]        # fitted - true
    nugget_relative_error: Optional[float]
    total_sill_error: float
    total_sill_relative_error: float
    range_errors: list                   # per component

    # diagnostics
    aic: float
    bic: float
    cv_rmse: Optional[float]
    rss: float
    n_candidates_fitted: int

    # range constraint check
    max_fitted_range: float
    half_max_lag: float
    range_exceeds_half_lag: bool


def fit_scenario(
    scenario: TestScenario,
    noise_std_frac: float = 0.05,
    n_runs: int = 20,
    seed: int = 42,
    include_nugget: bool = True,
    max_components: int = 2,
) -> Tuple[FitResult, FittedVariogramModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a scenario and return detailed results."""

    lags, mean_vario, sigma_vario, true_vario = generate_nested_variogram(
        scenario, noise_std_frac=noise_std_frac, n_runs=n_runs, seed=seed
    )

    # create selector (the pathway under test)
    selector = VariogramModelSelector(
        lags=lags,
        empirical_variogram=mean_vario,
        weights=np.ones_like(lags),  # uniform weights
        sigma=sigma_vario,
    )

    # fit all candidates
    selector.fit_all_candidates(
        max_components=max_components,
        include_nugget=include_nugget,
        include_unbounded=False,  # only bounded for this test
        compute_cv=True,
        cv_folds=5,
        seed=seed,
    )

    best = selector.select_best(criterion='aic')

    # extract fitted parameters
    model = best.composite_model
    fitted_nugget = model.get_nugget() if model.include_nugget else None

    fitted_sills = []
    fitted_ranges = []
    for i, spec in enumerate(model._components):
        cp = model.get_component_params(i)
        if spec.has_sill:
            fitted_sills.append(float(cp[0]))
        if 'range' in spec.param_names:
            range_idx = spec.param_names.index('range')
            fitted_ranges.append(float(cp[range_idx]))

    fitted_total_sill = sum(fitted_sills) + (fitted_nugget or 0)

    # compute true total sill
    if "+" in scenario.model_type:
        true_total_sill = sum(
            scenario.true_params[f"comp{i}"]["sill"]
            for i in range(len(scenario.model_type.split("+")))
        ) + scenario.true_nugget
    else:
        true_total_sill = scenario.true_params.get("sill", 0) + scenario.true_nugget

    # check model selection correctness
    true_base = scenario.model_type.replace("+", " + ")
    selected_name = "+".join(model.component_names)
    # simple check: does the selected model match the true type?
    correct_model = selected_name == scenario.model_type.replace("+", "+")

    # nugget errors
    if fitted_nugget is not None and scenario.true_nugget > 0:
        nugget_error = fitted_nugget - scenario.true_nugget
        nugget_rel_error = nugget_error / scenario.true_nugget
    elif fitted_nugget is not None and scenario.true_nugget == 0:
        nugget_error = fitted_nugget
        nugget_rel_error = float('inf') if fitted_nugget > 1e-6 else 0.0
    else:
        nugget_error = None
        nugget_rel_error = None

    # total sill error
    total_sill_error = fitted_total_sill - true_total_sill
    total_sill_rel_error = total_sill_error / true_total_sill if true_total_sill > 0 else 0

    # range errors (match by ordering, best effort)
    if "+" in scenario.model_type:
        true_ranges = [
            scenario.true_params[f"comp{i}"].get("range_", 0)
            for i in range(len(scenario.model_type.split("+")))
        ]
    else:
        true_ranges = [scenario.true_params.get("range_", 0)]

    # sort both for comparison
    true_ranges_sorted = sorted(true_ranges)
    fitted_ranges_sorted = sorted(fitted_ranges)
    range_errors = []
    for tr, fr in zip(true_ranges_sorted, fitted_ranges_sorted):
        range_errors.append(fr - tr)

    half_max_lag = scenario.max_lag / 2
    max_fitted_range = max(fitted_ranges) if fitted_ranges else 0

    result = FitResult(
        scenario_name=scenario.name,
        true_model_type=scenario.model_type,
        true_nugget=scenario.true_nugget,
        true_total_sill=true_total_sill,
        true_params=scenario.true_params if not isinstance(scenario.true_params, dict) else scenario.true_params,
        selected_model_name=selected_name,
        correct_model_selected=correct_model,
        fitted_nugget=fitted_nugget,
        fitted_sills=fitted_sills,
        fitted_ranges=fitted_ranges,
        fitted_total_sill=fitted_total_sill,
        nugget_error=nugget_error,
        nugget_relative_error=nugget_rel_error,
        total_sill_error=total_sill_error,
        total_sill_relative_error=total_sill_rel_error,
        range_errors=range_errors,
        aic=best.aic,
        bic=best.bic,
        cv_rmse=best.cv_rmse,
        rss=best.rss,
        n_candidates_fitted=len(selector.fitted_models),
        max_fitted_range=max_fitted_range,
        half_max_lag=half_max_lag,
        range_exceeds_half_lag=max_fitted_range > half_max_lag,
    )

    return result, best, lags, mean_vario, sigma_vario, true_vario


# ── plotting ──────────────────────────────────────────────────────────

def plot_fit(
    scenario: TestScenario,
    result: FitResult,
    best: FittedVariogramModel,
    lags: np.ndarray,
    mean_vario: np.ndarray,
    sigma_vario: np.ndarray,
    true_vario: np.ndarray,
    save_path: Path,
):
    """Plot empirical variogram, true model, and fitted model."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # empirical data
    ax.errorbar(lags, mean_vario, yerr=sigma_vario, fmt='o', color='gray',
                alpha=0.6, markersize=4, label='Synthetic empirical', capsize=2)

    # true model
    h_fine = np.linspace(0, scenario.max_lag, 500)
    if "+" in scenario.model_type:
        true_fine = np.zeros_like(h_fine)
        for i, mt in enumerate(scenario.model_type.split("+")):
            func = MODEL_FUNCS[mt]
            true_fine += func(h_fine, **scenario.true_params[f"comp{i}"])
    else:
        func = MODEL_FUNCS[scenario.model_type]
        true_fine = func(h_fine, **scenario.true_params)
    if scenario.true_nugget > 0:
        true_fine += nugget_func(h_fine, scenario.true_nugget)

    ax.plot(h_fine, true_fine, 'k-', linewidth=2, label='True model')

    # fitted model
    fitted_fine = best.predict(h_fine)
    ax.plot(h_fine, fitted_fine, 'r--', linewidth=2,
            label=f'Fitted: {result.selected_model_name}')

    # mark half max lag
    ax.axvline(scenario.max_lag / 2, color='blue', linestyle=':', alpha=0.5,
               label=f'Half max lag ({scenario.max_lag/2:.0f})')

    # mark true nugget
    if scenario.true_nugget > 0:
        ax.axhline(scenario.true_nugget, color='green', linestyle=':', alpha=0.3,
                    label=f'True nugget ({scenario.true_nugget:.3f})')
    if result.fitted_nugget is not None and result.fitted_nugget > 1e-6:
        ax.axhline(result.fitted_nugget, color='red', linestyle=':', alpha=0.3,
                    label=f'Fitted nugget ({result.fitted_nugget:.3f})')

    ax.set_xlabel('Lag distance')
    ax.set_ylabel('Semivariance')
    ax.set_title(f'{scenario.name}\n{scenario.description}')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, scenario.max_lag)
    ax.set_ylim(bottom=0)

    # add text box with fit metrics
    textstr = (
        f"AIC: {result.aic:.1f}\n"
        f"Nugget err: {result.nugget_error:.4f}\n"
        f"Sill err: {result.total_sill_error:.4f}\n"
        f"Max range: {result.max_fitted_range:.1f}"
    ) if result.nugget_error is not None else (
        f"AIC: {result.aic:.1f}\n"
        f"Sill err: {result.total_sill_error:.4f}\n"
        f"Max range: {result.max_fitted_range:.1f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_nugget_bias_summary(results: List[FitResult], save_path: Path):
    """Plot nugget estimation bias across noise levels."""
    # filter to single-component spherical with varying nugget
    sph_results = [r for r in results if r.scenario_name.startswith("spherical_nug")]
    if not sph_results:
        return

    true_nugs = [r.true_nugget for r in sph_results]
    fitted_nugs = [r.fitted_nugget or 0 for r in sph_results]
    nug_errors = [r.nugget_error or 0 for r in sph_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: true vs fitted nugget
    ax = axes[0]
    ax.scatter(true_nugs, fitted_nugs, c='steelblue', s=60, zorder=3)
    max_val = max(max(true_nugs), max(fitted_nugs)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
    ax.set_xlabel('True nugget')
    ax.set_ylabel('Fitted nugget')
    ax.set_title('Nugget Recovery (Spherical)')
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Panel 2: nugget error vs true nugget fraction
    ax = axes[1]
    true_fracs = [r.true_nugget / r.true_total_sill if r.true_total_sill > 0 else 0
                  for r in sph_results]
    ax.bar(range(len(sph_results)), nug_errors, color='coral', alpha=0.7)
    ax.set_xticks(range(len(sph_results)))
    ax.set_xticklabels([f"{f:.0%}" for f in true_fracs], rotation=45)
    ax.set_xlabel('True nugget / total sill')
    ax.set_ylabel('Nugget error (fitted - true)')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_title('Nugget Estimation Error')

    # Panel 3: same for exponential and gaussian
    for model_prefix, color, label in [
        ("exponential_nug", "forestgreen", "Exponential"),
        ("gaussian_nug", "darkorange", "Gaussian"),
    ]:
        model_results = [r for r in results if r.scenario_name.startswith(model_prefix)]
        if model_results:
            true_n = [r.true_nugget for r in model_results]
            fitted_n = [r.fitted_nugget or 0 for r in model_results]
            axes[2].scatter(true_n, fitted_n, c=color, s=60, label=label, zorder=3)

    sph_true = [r.true_nugget for r in sph_results]
    sph_fitted = [r.fitted_nugget or 0 for r in sph_results]
    axes[2].scatter(sph_true, sph_fitted, c='steelblue', s=60, label='Spherical', zorder=3)
    max_val2 = max(max(sph_fitted), 0.7) * 1.1
    axes[2].plot([0, max_val2], [0, max_val2], 'k--', alpha=0.5)
    axes[2].set_xlabel('True nugget')
    axes[2].set_ylabel('Fitted nugget')
    axes[2].set_title('Nugget Recovery by Model Type')
    axes[2].legend(fontsize=8)
    axes[2].set_xlim(0, max_val2)
    axes[2].set_ylim(0, max_val2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_range_analysis(results: List[FitResult], save_path: Path):
    """Plot range recovery and half-lag exceedance analysis."""
    range_results = [r for r in results if r.scenario_name.startswith("spherical_range")]
    if not range_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: True vs fitted range
    ax = axes[0]
    true_ranges = []
    fitted_ranges = []
    for r in range_results:
        if "+" in r.true_model_type:
            continue
        tr = r.true_params.get("range_", 0)
        fr = r.fitted_ranges[0] if r.fitted_ranges else 0
        true_ranges.append(tr)
        fitted_ranges.append(fr)

    colors = ['red' if r.range_exceeds_half_lag else 'steelblue' for r in range_results]
    ax.scatter(true_ranges, fitted_ranges, c=colors, s=80, zorder=3, edgecolors='k', linewidth=0.5)
    max_val = max(max(true_ranges), max(fitted_ranges)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
    ax.axhline(250, color='blue', linestyle=':', alpha=0.5, label='Half max lag (250)')
    ax.axvline(250, color='blue', linestyle=':', alpha=0.5)
    ax.set_xlabel('True range')
    ax.set_ylabel('Fitted range')
    ax.set_title('Range Recovery\n(red = fitted range exceeds half max lag)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Panel 2: Range relative error vs range/max_lag fraction
    ax = axes[1]
    range_fracs = [tr / 500 for tr in true_ranges]
    range_rel_errors = [(fr - tr) / tr if tr > 0 else 0
                        for tr, fr in zip(true_ranges, fitted_ranges)]
    ax.bar(range(len(range_fracs)), range_rel_errors, color=colors, alpha=0.7, edgecolor='k', linewidth=0.5)
    ax.set_xticks(range(len(range_fracs)))
    ax.set_xticklabels([f"{f:.0%}" for f in range_fracs], rotation=45)
    ax.set_xlabel('True range / max lag')
    ax.set_ylabel('Range relative error')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_title('Range Estimation Error vs Range/MaxLag')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_parameter_spread(results: List[FitResult], save_path: Path):
    """Analyze parameter spread in multi-component fits."""
    nested = [r for r in results if "+" in r.true_model_type]
    if not nested:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for idx, r in enumerate(nested):
        # sill spread
        if len(r.fitted_sills) >= 2:
            axes[0].barh(idx, max(r.fitted_sills) - min(r.fitted_sills),
                        left=min(r.fitted_sills), color='steelblue', alpha=0.6,
                        height=0.6, edgecolor='k', linewidth=0.5)
            # true sill spread
            if "+" in r.true_model_type:
                n_comp = len(r.true_model_type.split("+"))
                true_sills = [r.true_params[f"comp{i}"]["sill"] for i in range(n_comp)]
                axes[0].barh(idx + 0.3, max(true_sills) - min(true_sills),
                            left=min(true_sills), color='green', alpha=0.4,
                            height=0.3, edgecolor='k', linewidth=0.5)

        # range spread
        if len(r.fitted_ranges) >= 2:
            axes[1].barh(idx, max(r.fitted_ranges) - min(r.fitted_ranges),
                        left=min(r.fitted_ranges), color='coral', alpha=0.6,
                        height=0.6, edgecolor='k', linewidth=0.5)
            if "+" in r.true_model_type:
                n_comp = len(r.true_model_type.split("+"))
                true_ranges = [r.true_params[f"comp{i}"].get("range_", 0) for i in range(n_comp)]
                axes[1].barh(idx + 0.3, max(true_ranges) - min(true_ranges),
                            left=min(true_ranges), color='green', alpha=0.4,
                            height=0.3, edgecolor='k', linewidth=0.5)

    names = [r.scenario_name for r in nested]
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel('Sill value')
    axes[0].set_title('Sill Spread (blue=fitted, green=true)')

    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel('Range value')
    axes[1].set_title('Range Spread (coral=fitted, green=true)')
    axes[1].axvline(250, color='blue', linestyle=':', alpha=0.5, label='Half max lag')
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_model_selection_accuracy(results: List[FitResult], save_path: Path):
    """Plot model selection accuracy summary."""
    # group by true model type
    from collections import defaultdict
    type_results = defaultdict(list)
    for r in results:
        base_type = r.true_model_type.split("_")[0] if "_" not in r.true_model_type else r.true_model_type
        type_results[r.true_model_type].append(r)

    fig, ax = plt.subplots(figsize=(12, 6))

    model_types = list(type_results.keys())
    correct_counts = []
    total_counts = []

    for mt in model_types:
        rs = type_results[mt]
        correct_counts.append(sum(1 for r in rs if r.correct_model_selected))
        total_counts.append(len(rs))

    # only show types with multiple results
    valid_idx = [i for i, tc in enumerate(total_counts) if tc > 1]
    if not valid_idx:
        # show all single entries
        valid_idx = list(range(len(model_types)))

    positions = range(len(valid_idx))
    bars_total = ax.bar(positions, [total_counts[i] for i in valid_idx],
                        color='lightgray', edgecolor='k', linewidth=0.5, label='Total')
    bars_correct = ax.bar(positions, [correct_counts[i] for i in valid_idx],
                          color='steelblue', edgecolor='k', linewidth=0.5, label='Correct selection')

    ax.set_xticks(positions)
    ax.set_xticklabels([model_types[i] for i in valid_idx], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Count')
    ax.set_title('Model Selection Accuracy by True Model Type')
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sill_recovery(results: List[FitResult], save_path: Path):
    """Plot total sill recovery across all scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    true_sills = [r.true_total_sill for r in results]
    fitted_sills = [r.fitted_total_sill for r in results]
    sill_errors = [r.total_sill_relative_error for r in results]

    # Panel 1: True vs fitted total sill
    ax = axes[0]
    ax.scatter(true_sills, fitted_sills, c='steelblue', s=40, alpha=0.7, zorder=3)
    max_val = max(max(true_sills), max(fitted_sills)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
    ax.set_xlabel('True total sill')
    ax.set_ylabel('Fitted total sill')
    ax.set_title('Total Sill Recovery (all scenarios)')
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Panel 2: relative error distribution
    ax = axes[1]
    ax.hist(sill_errors, bins=20, color='steelblue', alpha=0.7, edgecolor='k')
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(np.mean(sill_errors), color='red', linestyle='--', label=f'Mean: {np.mean(sill_errors):.3f}')
    ax.set_xlabel('Total sill relative error')
    ax.set_ylabel('Count')
    ax.set_title('Total Sill Relative Error Distribution')
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── bounds analysis ───────────────────────────────────────────────────

def analyze_bounds(scenarios: List[TestScenario]) -> dict:
    """Analyze default bounds relative to true parameters and lag distances."""
    results = {}
    for sc in scenarios:
        if "+" in sc.model_type:
            # use first component for single-model bounds analysis
            continue

        lags = np.linspace(sc.max_lag / sc.n_lags / 2, sc.max_lag, sc.n_lags)
        func = MODEL_FUNCS[sc.model_type]
        true_gamma = func(lags, **sc.true_params)
        if sc.true_nugget > 0:
            true_gamma += nugget_func(lags, sc.true_nugget)

        # get bounds from registry
        spec = MODEL_REGISTRY.get_model(sc.model_type)
        lb, ub = spec.bounds(lags, true_gamma)

        # check if true parameters fall within bounds
        true_vals = list(sc.true_params.values())
        within_bounds = all(l <= v <= u for l, v, u in zip(lb, true_vals, ub))

        # range bound relative to max lag
        if 'range' in spec.param_names:
            range_idx = spec.param_names.index('range')
            range_upper_bound = ub[range_idx]
            range_upper_vs_maxlag = range_upper_bound / sc.max_lag
        else:
            range_upper_bound = None
            range_upper_vs_maxlag = None

        # nugget bound from CompositeVariogramModel
        comp_model = CompositeVariogramModel([sc.model_type], include_nugget=True)
        comp_lb, comp_ub = comp_model.bounds(lags, true_gamma)
        nugget_upper = comp_ub[-1]
        nugget_upper_vs_sill = nugget_upper / np.max(true_gamma) if np.max(true_gamma) > 0 else float('inf')

        results[sc.name] = {
            'spec_bounds_lower': lb,
            'spec_bounds_upper': ub,
            'true_within_bounds': within_bounds,
            'range_upper_bound': range_upper_bound,
            'range_upper_vs_maxlag': range_upper_vs_maxlag,
            'nugget_upper_bound': nugget_upper,
            'nugget_upper_vs_max_semivariance': nugget_upper_vs_sill,
            'sill_upper_bound': ub[0] if ub else None,
            'sill_upper_vs_max_semivariance': ub[0] / np.max(true_gamma) if ub and np.max(true_gamma) > 0 else None,
        }

    return results


# ── initial guess analysis ────────────────────────────────────────────

def analyze_initial_guesses(scenarios: List[TestScenario]) -> dict:
    """Analyze how far initial guesses are from true parameters."""
    results = {}
    for sc in scenarios:
        if "+" in sc.model_type:
            continue

        lags = np.linspace(sc.max_lag / sc.n_lags / 2, sc.max_lag, sc.n_lags)
        func = MODEL_FUNCS[sc.model_type]
        true_gamma = func(lags, **sc.true_params)
        if sc.true_nugget > 0:
            true_gamma += nugget_func(lags, sc.true_nugget)

        comp_model = CompositeVariogramModel(
            [sc.model_type], include_nugget=(sc.true_nugget > 0)
        )
        guess = comp_model.default_guess(lags, true_gamma)
        true_vals = list(sc.true_params.values())
        if sc.true_nugget > 0:
            true_vals.append(sc.true_nugget)

        # compute relative errors of initial guesses
        guess_errors = {}
        param_names = comp_model.param_names
        for i, (name, g, t) in enumerate(zip(param_names, guess, true_vals)):
            if t > 0:
                guess_errors[name] = (g - t) / t
            else:
                guess_errors[name] = g  # absolute error when true is 0

        results[sc.name] = {
            'guess': list(guess),
            'true': true_vals,
            'param_names': param_names,
            'relative_errors': guess_errors,
        }

    return results


# ── main runner ───────────────────────────────────────────────────────

def run_all_tests():
    """Run full synthetic testing suite."""
    print("=" * 70)
    print("SYNTHETIC VARIOGRAM FITTING DIAGNOSTIC SUITE")
    print("=" * 70)

    scenarios = build_scenarios()
    print(f"\nBuilt {len(scenarios)} test scenarios")

    # ── run fits ──
    all_results: List[FitResult] = []
    all_fits = {}
    noise_frac = 0.05  # 5% noise for main tests

    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] Fitting: {scenario.name} ... ", end="", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result, best, lags, mean_v, sigma_v, true_v = fit_scenario(
                    scenario,
                    noise_std_frac=noise_frac,
                    include_nugget=True,
                    max_components=2,
                )
            all_results.append(result)
            all_fits[scenario.name] = (result, best, lags, mean_v, sigma_v, true_v, scenario)

            status = "OK"
            if result.nugget_error is not None and abs(result.nugget_relative_error) > 0.5:
                status += " [NUGGET BIAS]"
            if result.range_exceeds_half_lag:
                status += " [RANGE>HALF_LAG]"
            if not result.correct_model_selected:
                status += f" [SELECTED: {result.selected_model_name}]"
            print(status)

        except Exception as e:
            print(f"FAILED: {e}")

    # ── generate individual fit plots ──
    print("\n\nGenerating individual fit plots...")
    for name, (result, best, lags, mean_v, sigma_v, true_v, scenario) in all_fits.items():
        plot_fit(scenario, result, best, lags, mean_v, sigma_v, true_v,
                 PLOT_DIR / f"fit_{name}.png")

    # ── generate summary plots ──
    print("Generating summary plots...")
    plot_nugget_bias_summary(all_results, PLOT_DIR / "summary_nugget_bias.png")
    plot_range_analysis(all_results, PLOT_DIR / "summary_range_analysis.png")
    plot_parameter_spread(all_results, PLOT_DIR / "summary_parameter_spread.png")
    plot_model_selection_accuracy(all_results, PLOT_DIR / "summary_model_selection.png")
    plot_sill_recovery(all_results, PLOT_DIR / "summary_sill_recovery.png")

    # ── bounds analysis ──
    print("Analyzing bounds...")
    bounds_analysis = analyze_bounds(scenarios)

    # ── initial guess analysis ──
    print("Analyzing initial guesses...")
    guess_analysis = analyze_initial_guesses(scenarios)

    # ── noise sensitivity sweep ──
    print("\nRunning noise sensitivity sweep...")
    noise_sweep_results = {}
    sweep_scenario = TestScenario(
        name="sweep_spherical",
        model_type="spherical",
        true_params={"sill": 0.8, "range_": 150.0},
        true_nugget=0.2,
        description="Spherical for noise sweep",
    )
    noise_fracs = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    for nf in noise_fracs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _, _, _, _, _ = fit_scenario(
                    sweep_scenario, noise_std_frac=nf, include_nugget=True,
                )
            noise_sweep_results[nf] = r
            print(f"  noise={nf:.0%}: nugget_err={r.nugget_error:.4f}, "
                  f"sill_err={r.total_sill_error:.4f}, "
                  f"range_err={r.range_errors}")
        except Exception as e:
            print(f"  noise={nf:.0%}: FAILED ({e})")

    # plot noise sweep
    if noise_sweep_results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        nfs = sorted(noise_sweep_results.keys())
        nug_errs = [noise_sweep_results[nf].nugget_error for nf in nfs]
        sill_errs = [noise_sweep_results[nf].total_sill_error for nf in nfs]
        range_errs = [noise_sweep_results[nf].range_errors[0] if noise_sweep_results[nf].range_errors else 0 for nf in nfs]

        axes[0].plot(nfs, nug_errs, 'o-', color='coral')
        axes[0].axhline(0, color='k', alpha=0.3)
        axes[0].set_xlabel('Noise fraction')
        axes[0].set_ylabel('Nugget error')
        axes[0].set_title('Nugget Error vs Noise Level')

        axes[1].plot(nfs, sill_errs, 'o-', color='steelblue')
        axes[1].axhline(0, color='k', alpha=0.3)
        axes[1].set_xlabel('Noise fraction')
        axes[1].set_ylabel('Total sill error')
        axes[1].set_title('Total Sill Error vs Noise Level')

        axes[2].plot(nfs, range_errs, 'o-', color='forestgreen')
        axes[2].axhline(0, color='k', alpha=0.3)
        axes[2].set_xlabel('Noise fraction')
        axes[2].set_ylabel('Range error')
        axes[2].set_title('Range Error vs Noise Level')

        fig.tight_layout()
        fig.savefig(PLOT_DIR / "summary_noise_sweep.png", dpi=150)
        plt.close(fig)

    # ── save results as JSON ──
    print("\nSaving results...")
    summary_data = {
        "n_scenarios": len(scenarios),
        "n_fitted": len(all_results),
        "noise_frac": noise_frac,
        "results": [],
    }
    for r in all_results:
        summary_data["results"].append({
            "scenario": r.scenario_name,
            "true_model": r.true_model_type,
            "selected_model": r.selected_model_name,
            "correct_selection": r.correct_model_selected,
            "true_nugget": r.true_nugget,
            "fitted_nugget": r.fitted_nugget,
            "nugget_error": r.nugget_error,
            "nugget_relative_error": r.nugget_relative_error,
            "true_total_sill": r.true_total_sill,
            "fitted_total_sill": r.fitted_total_sill,
            "total_sill_relative_error": r.total_sill_relative_error,
            "fitted_ranges": r.fitted_ranges,
            "range_errors": r.range_errors,
            "max_fitted_range": r.max_fitted_range,
            "half_max_lag": r.half_max_lag,
            "range_exceeds_half_lag": r.range_exceeds_half_lag,
            "aic": r.aic,
            "rss": r.rss,
        })

    # handle inf/nan in JSON
    def clean_for_json(obj):
        if isinstance(obj, float):
            if np.isinf(obj) or np.isnan(obj):
                return str(obj)
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    with open(OUTPUT_DIR / "fitting_results.json", "w") as f:
        json.dump(clean_for_json(summary_data), f, indent=2)

    # save bounds analysis
    with open(OUTPUT_DIR / "bounds_analysis.json", "w") as f:
        json.dump(clean_for_json(bounds_analysis), f, indent=2, default=str)

    # save guess analysis
    with open(OUTPUT_DIR / "guess_analysis.json", "w") as f:
        json.dump(clean_for_json(guess_analysis), f, indent=2, default=str)

    # ── print summary table ──
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Scenario':<35} {'True->Selected':<30} {'NugErr':>8} {'SillErr':>8} {'RngExceed':>10}")
    print("-" * 100)
    for r in all_results:
        nug_str = f"{r.nugget_error:+.4f}" if r.nugget_error is not None else "N/A"
        model_str = f"{r.true_model_type}->{r.selected_model_name}"
        exceed_str = "YES" if r.range_exceeds_half_lag else "no"
        print(f"{r.scenario_name:<35} {model_str:<30} {nug_str:>8} {r.total_sill_error:+.4f} {exceed_str:>10}")

    # ── summary statistics ──
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    nug_errors = [r.nugget_error for r in all_results if r.nugget_error is not None]
    if nug_errors:
        print(f"Nugget error: mean={np.mean(nug_errors):.4f}, "
              f"std={np.std(nug_errors):.4f}, "
              f"median={np.median(nug_errors):.4f}")
        positive_bias = sum(1 for e in nug_errors if e > 0.01)
        print(f"  Positive bias (>0.01): {positive_bias}/{len(nug_errors)} "
              f"({positive_bias/len(nug_errors):.0%})")

    sill_errors = [r.total_sill_relative_error for r in all_results]
    print(f"Total sill relative error: mean={np.mean(sill_errors):.4f}, "
          f"std={np.std(sill_errors):.4f}")

    range_exceed = sum(1 for r in all_results if r.range_exceeds_half_lag)
    print(f"Range exceeds half lag: {range_exceed}/{len(all_results)} "
          f"({range_exceed/len(all_results):.0%})")

    correct_model = sum(1 for r in all_results if r.correct_model_selected)
    print(f"Correct model selected: {correct_model}/{len(all_results)} "
          f"({correct_model/len(all_results):.0%})")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")

    return all_results, bounds_analysis, guess_analysis, noise_sweep_results


if __name__ == "__main__":
    results, bounds, guesses, noise_sweep = run_all_tests()
