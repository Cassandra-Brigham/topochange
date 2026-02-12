"""Stress tests for variogram fitting with realistic noise patterns.

Real empirical variograms have:
- Fewer pairs at short lags → higher variance at short lags
- Heteroscedastic noise that scales with pair count
- Sometimes erratic behavior at the longest lags
- Finite number of runs (not 1000s)

This script mimics those patterns to test fitting under realistic conditions.
"""

from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topochange.variogram_models import (
    spherical, exponential, gaussian, matern, nugget as nugget_func,
    MODEL_REGISTRY,
)
from topochange.composite_variogram import CompositeVariogramModel
from topochange.variogram import VariogramModelSelector, FittedVariogramModel

OUTPUT_DIR = Path(__file__).parent.parent / "synthetic_fitting_results"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def realistic_noise_variogram(
    model_func,
    params: dict,
    nugget_value: float = 0.0,
    n_lags: int = 30,
    max_lag: float = 500.0,
    n_runs: int = 10,
    base_pairs: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic variogram with REALISTIC noise structure.

    Mimics real empirical variograms by:
    1. Pair count increases with lag (annular area grows)
    2. Noise ~ gamma(h)^2 / N(h) (Cressie 1985 variance formula)
    3. Short lags have far fewer pairs, much higher noise
    4. Small number of runs (typical real workflow)
    """
    rng = np.random.default_rng(seed)
    bin_width = max_lag / n_lags
    lags = np.linspace(bin_width / 2, max_lag - bin_width / 2, n_lags)

    true_gamma = model_func(lags, **params)
    if nugget_value > 0:
        true_gamma += nugget_func(lags, nugget_value)

    # realistic pair counts: N(h) ~ h for 2D random fields
    # with some randomness and lower counts at short lags
    pair_counts = base_pairs * (lags / max_lag) * (0.8 + 0.4 * rng.random(n_lags))
    pair_counts = np.maximum(pair_counts, 30)  # minimum pairs

    # Cressie (1985) variance: Var[gamma_hat(h)] ~ 2 * gamma(h)^2 / N(h)
    # for the Matheron estimator
    noise_std = np.sqrt(2) * np.abs(true_gamma + 0.01) / np.sqrt(pair_counts)

    all_runs = np.zeros((n_runs, n_lags))
    for r in range(n_runs):
        noise = rng.normal(0, noise_std)
        all_runs[r] = np.maximum(true_gamma + noise, 0)

    mean_variogram = np.mean(all_runs, axis=0)
    sigma_variogram = np.std(all_runs, axis=0)
    sigma_variogram[sigma_variogram == 0] = np.finfo(float).eps

    return lags, mean_variogram, sigma_variogram, true_gamma


def run_stress_test(
    name: str,
    model_func,
    params: dict,
    nugget_value: float,
    max_lag: float = 500.0,
    n_lags: int = 30,
    n_runs: int = 10,
    base_pairs: int = 5000,
    seed: int = 42,
    max_components: int = 2,
) -> dict:
    """Run a single stress test and return results."""

    lags, mean_v, sigma_v, true_v = realistic_noise_variogram(
        model_func, params, nugget_value,
        n_lags=n_lags, max_lag=max_lag, n_runs=n_runs,
        base_pairs=base_pairs, seed=seed,
    )

    selector = VariogramModelSelector(
        lags=lags,
        empirical_variogram=mean_v,
        sigma=sigma_v,
        weighting='uniform',
    )

    selector.fit_all_candidates(
        max_components=max_components,
        include_nugget=True,
        include_unbounded=False,
        compute_cv=True,
        cv_folds=5,
        seed=seed,
    )

    best = selector.select_best(criterion='aic')
    model = best.composite_model

    fitted_nugget = model.get_nugget() if model.include_nugget else 0
    fitted_sills = []
    fitted_ranges = []
    for i, spec in enumerate(model._components):
        cp = model.get_component_params(i)
        if spec.has_sill:
            fitted_sills.append(float(cp[0]))
        if 'range' in spec.param_names:
            fitted_ranges.append(float(cp[spec.param_names.index('range')]))

    fitted_total_sill = sum(fitted_sills) + fitted_nugget
    true_total_sill = params.get('sill', 0) + nugget_value

    return {
        'name': name,
        'true_nugget': nugget_value,
        'fitted_nugget': fitted_nugget,
        'nugget_error': fitted_nugget - nugget_value,
        'nugget_rel_error': (fitted_nugget - nugget_value) / nugget_value if nugget_value > 0 else (fitted_nugget if fitted_nugget > 1e-6 else 0),
        'true_total_sill': true_total_sill,
        'fitted_total_sill': fitted_total_sill,
        'sill_error': fitted_total_sill - true_total_sill,
        'true_range': params.get('range_', 0),
        'fitted_ranges': fitted_ranges,
        'max_fitted_range': max(fitted_ranges) if fitted_ranges else 0,
        'half_max_lag': max_lag / 2,
        'range_exceeds_half': max(fitted_ranges) > max_lag / 2 if fitted_ranges else False,
        'selected_model': '+'.join(model.component_names),
        'n_components': len(model.component_names),
        'aic': best.aic,
        'lags': lags,
        'mean_v': mean_v,
        'sigma_v': sigma_v,
        'true_v': true_v,
        'best': best,
        'selector': selector,
    }


def main():
    print("=" * 70)
    print("STRESS TESTS: REALISTIC NOISE PATTERNS")
    print("=" * 70)

    all_results = []

    # ── Test 1: Nugget bias across pair counts ──
    # Fewer pairs → more noise at short lags → potential nugget overestimation
    print("\n--- Test 1: Nugget sensitivity to pair count ---")
    for base_pairs in [200, 500, 1000, 3000, 10000]:
        for nug_frac in [0.0, 0.1, 0.3, 0.5]:
            true_sill = 0.8
            true_nug = nug_frac * (true_sill + nug_frac)  # fraction of total
            true_sill_adj = 1.0 - true_nug  # so total = 1.0

            r = run_stress_test(
                name=f"pairs{base_pairs}_nug{int(nug_frac*100)}",
                model_func=spherical,
                params={'sill': true_sill_adj, 'range_': 150.0},
                nugget_value=true_nug,
                base_pairs=base_pairs,
                n_runs=10,
            )
            all_results.append(r)
            print(f"  pairs={base_pairs:>5}, nug={nug_frac:.0%}: "
                  f"nugErr={r['nugget_error']:+.4f}, "
                  f"sillErr={r['sill_error']:+.4f}, "
                  f"selected={r['selected_model']}")

    # ── Test 2: Range beyond half lag with realistic noise ──
    print("\n--- Test 2: Range recovery with realistic noise ---")
    for true_range in [50, 100, 200, 300, 400]:
        r = run_stress_test(
            name=f"range{true_range}",
            model_func=spherical,
            params={'sill': 0.9, 'range_': float(true_range)},
            nugget_value=0.1,
            base_pairs=2000,
            n_runs=10,
        )
        all_results.append(r)
        range_err = r['fitted_ranges'][0] - true_range if r['fitted_ranges'] else float('nan')
        print(f"  true_range={true_range:>4}: "
              f"fitted_range={r['max_fitted_range']:.1f}, "
              f"err={range_err:+.1f}, "
              f"exceeds_half={r['range_exceeds_half']}")

    # ── Test 3: Exponential with various practical ranges ──
    # Exponential range param ≠ practical range (×3 factor)
    # Does the fitter handle this correctly or confuse range/practical_range?
    print("\n--- Test 3: Exponential practical range confusion ---")
    for exp_range in [20, 50, 100, 150]:
        pract_range = exp_range * 3
        r = run_stress_test(
            name=f"exp_range{exp_range}",
            model_func=exponential,
            params={'sill': 0.9, 'range_': float(exp_range)},
            nugget_value=0.1,
            base_pairs=3000,
            n_runs=10,
        )
        all_results.append(r)
        fitted_r = r['fitted_ranges'][0] if r['fitted_ranges'] else float('nan')
        print(f"  exp_range={exp_range:>4} (practical={pract_range}): "
              f"fitted={fitted_r:.1f}, "
              f"selected={r['selected_model']}")

    # ── Test 4: Very few runs (common in practice) ──
    print("\n--- Test 4: Few runs (n_runs=3,5,10,20) ---")
    for n_runs in [3, 5, 10, 20]:
        r = run_stress_test(
            name=f"runs{n_runs}",
            model_func=spherical,
            params={'sill': 0.8, 'range_': 150.0},
            nugget_value=0.2,
            base_pairs=2000,
            n_runs=n_runs,
        )
        all_results.append(r)
        print(f"  n_runs={n_runs:>2}: nugErr={r['nugget_error']:+.4f}, "
              f"sillErr={r['sill_error']:+.4f}, "
              f"selected={r['selected_model']}")

    # ── Test 5: Model selection with noise ──
    print("\n--- Test 5: Model selection under noise (base_pairs=1000) ---")
    models = [
        ("spherical", spherical, {'sill': 0.9, 'range_': 150.0}),
        ("exponential", exponential, {'sill': 0.9, 'range_': 50.0}),
        ("gaussian", gaussian, {'sill': 0.9, 'range_': 87.0}),
        ("matern", matern, {'sill': 0.9, 'range_': 80.0, 'nu': 1.5}),
    ]
    for mname, mfunc, mparams in models:
        r = run_stress_test(
            name=f"select_{mname}",
            model_func=mfunc,
            params=mparams,
            nugget_value=0.1,
            base_pairs=1000,
            n_runs=10,
        )
        all_results.append(r)
        match = "MATCH" if mname in r['selected_model'] else f"MISMATCH({r['selected_model']})"
        print(f"  true={mname}: selected={r['selected_model']} [{match}], "
              f"AIC={r['aic']:.1f}")

    # ── Test 6: Bounds analysis ──
    # What do the bounds actually look like for typical data?
    print("\n--- Test 6: Bounds & initial guess analysis ---")
    lags_test = np.linspace(25/2, 500 - 25/2, 30)
    gamma_test = spherical(lags_test, sill=0.9, range_=150.0) + nugget_func(lags_test, 0.1)

    for model_name in ['spherical', 'exponential', 'gaussian', 'matern']:
        comp = CompositeVariogramModel([model_name], include_nugget=True)
        lb, ub = comp.bounds(lags_test, gamma_test)
        guess = comp.default_guess(lags_test, gamma_test)
        print(f"\n  {model_name}:")
        print(f"    bounds_lower = {[f'{v:.4f}' for v in lb]}")
        print(f"    bounds_upper = {[f'{v:.4f}' for v in ub]}")
        print(f"    initial_guess = {[f'{v:.4f}' for v in guess]}")
        print(f"    param_names = {comp.param_names}")
        # key ratios
        range_idx = -1
        for i, pn in enumerate(comp.param_names):
            if 'range' in pn and 'nugget' not in pn:
                range_idx = i
                break
        if range_idx >= 0:
            print(f"    range_upper / max_lag = {ub[range_idx] / 500:.1f}x")
            print(f"    range_guess / max_lag = {guess[range_idx] / 500:.2f}")
        nug_idx = len(comp.param_names) - 1  # nugget is always last
        print(f"    nugget_upper / max_gamma = {ub[nug_idx] / np.max(gamma_test):.1f}x")
        print(f"    nugget_guess / true_nug = {guess[nug_idx] / 0.1:.1f}x")

    # ── Test 7: Multi-restart sensitivity ──
    print("\n--- Test 7: Multi-start sensitivity (different seeds) ---")
    results_by_seed = {}
    for seed in range(10):
        r = run_stress_test(
            name=f"seed{seed}",
            model_func=spherical,
            params={'sill': 0.8, 'range_': 150.0},
            nugget_value=0.2,
            base_pairs=1000,
            n_runs=10,
            seed=seed,
        )
        results_by_seed[seed] = r

    nug_errs = [r['nugget_error'] for r in results_by_seed.values()]
    sill_errs = [r['sill_error'] for r in results_by_seed.values()]
    range_errs = [r['fitted_ranges'][0] - 150 for r in results_by_seed.values() if r['fitted_ranges']]
    models_selected = [r['selected_model'] for r in results_by_seed.values()]

    print(f"  Nugget error: mean={np.mean(nug_errs):.4f}, std={np.std(nug_errs):.4f}, range=[{min(nug_errs):.4f}, {max(nug_errs):.4f}]")
    print(f"  Sill error: mean={np.mean(sill_errs):.4f}, std={np.std(sill_errs):.4f}")
    print(f"  Range error: mean={np.mean(range_errs):.1f}, std={np.std(range_errs):.1f}")
    print(f"  Models selected: {models_selected}")

    # ── Generate summary plots ──
    print("\n\nGenerating stress test plots...")

    # Plot 1: Nugget error vs pair count heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pair_counts = [200, 500, 1000, 3000, 10000]
    nug_fracs = [0.0, 0.1, 0.3, 0.5]

    nug_error_matrix = np.zeros((len(pair_counts), len(nug_fracs)))
    for r in all_results:
        if r['name'].startswith('pairs'):
            parts = r['name'].split('_')
            pc = int(parts[0].replace('pairs', ''))
            nf = int(parts[1].replace('nug', '')) / 100
            if pc in pair_counts and nf in nug_fracs:
                i = pair_counts.index(pc)
                j = nug_fracs.index(nf)
                nug_error_matrix[i, j] = r['nugget_error']

    im = axes[0].imshow(nug_error_matrix, aspect='auto', cmap='RdBu_r',
                         vmin=-0.05, vmax=0.05)
    axes[0].set_xticks(range(len(nug_fracs)))
    axes[0].set_xticklabels([f"{f:.0%}" for f in nug_fracs])
    axes[0].set_yticks(range(len(pair_counts)))
    axes[0].set_yticklabels(pair_counts)
    axes[0].set_xlabel('True nugget fraction')
    axes[0].set_ylabel('Base pair count')
    axes[0].set_title('Nugget Error (fitted - true)')
    plt.colorbar(im, ax=axes[0])
    # annotate
    for i in range(len(pair_counts)):
        for j in range(len(nug_fracs)):
            axes[0].text(j, i, f"{nug_error_matrix[i,j]:+.3f}",
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(nug_error_matrix[i,j]) > 0.03 else 'black')

    # Plot 2: Range recovery
    range_results = [r for r in all_results if r['name'].startswith('range')]
    if range_results:
        true_rs = [r['true_range'] for r in range_results]
        fitted_rs = [r['max_fitted_range'] for r in range_results]
        colors = ['red' if r['range_exceeds_half'] else 'steelblue' for r in range_results]
        axes[1].scatter(true_rs, fitted_rs, c=colors, s=80, zorder=3, edgecolor='k')
        max_val = max(max(true_rs), max(fitted_rs)) * 1.1
        axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
        axes[1].axhline(250, color='blue', linestyle=':', alpha=0.5, label='Half max lag')
        axes[1].set_xlabel('True range')
        axes[1].set_ylabel('Fitted range')
        axes[1].set_title('Range Recovery (realistic noise)\nred=exceeds half lag')
        axes[1].legend(fontsize=8)

    # Plot 3: Multi-seed stability
    axes[2].errorbar(
        ['Nugget', 'Sill', 'Range'],
        [np.mean(nug_errs), np.mean(sill_errs), np.mean(range_errs)/150],
        yerr=[np.std(nug_errs), np.std(sill_errs), np.std(range_errs)/150],
        fmt='o', capsize=5, color='steelblue', markersize=8,
    )
    axes[2].axhline(0, color='k', alpha=0.3)
    axes[2].set_ylabel('Error (normalized for range)')
    axes[2].set_title('Parameter Error Across 10 Seeds\n(spherical, nugget=0.2, 1000 pairs)')

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stress_test_summary.png", dpi=150)
    plt.close(fig)

    # Plot individual stress fits for worst cases
    worst_nug = max(all_results, key=lambda r: abs(r.get('nugget_error', 0)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(worst_nug['lags'], worst_nug['mean_v'], yerr=worst_nug['sigma_v'],
                fmt='o', color='gray', alpha=0.6, markersize=4, capsize=2, label='Empirical')
    h_fine = np.linspace(0, 500, 500)
    ax.plot(h_fine, worst_nug['best'].predict(h_fine), 'r--', linewidth=2,
            label=f"Fitted: {worst_nug['selected_model']}")
    ax.plot(worst_nug['lags'], worst_nug['true_v'], 'k-', linewidth=2, label='True')
    ax.axhline(worst_nug['true_nugget'], color='green', linestyle=':', alpha=0.5,
               label=f"True nugget ({worst_nug['true_nugget']:.3f})")
    if worst_nug['fitted_nugget']:
        ax.axhline(worst_nug['fitted_nugget'], color='red', linestyle=':', alpha=0.5,
                    label=f"Fitted nugget ({worst_nug['fitted_nugget']:.3f})")
    ax.set_title(f"Worst Nugget Error Case: {worst_nug['name']}\n"
                 f"NugErr={worst_nug['nugget_error']:+.4f}")
    ax.legend(fontsize=8)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Semivariance')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stress_worst_nugget.png", dpi=150)
    plt.close(fig)

    # ── Save results ──
    def clean(obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
            return str(obj)
        return obj

    summary = []
    for r in all_results:
        summary.append({k: clean(v) for k, v in r.items()
                       if k not in ('lags', 'mean_v', 'sigma_v', 'true_v', 'best', 'selector')})

    with open(OUTPUT_DIR / "stress_test_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
