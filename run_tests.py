#!/usr/bin/env python
"""
Test runner for the topochange package.

Usage
-----
    python run_tests.py              # run everything
    python run_tests.py --quick      # skip slow / integration tests
    python run_tests.py --coverage   # generate an HTML coverage report
    python run_tests.py --file variogram_models  # run one test file
    python run_tests.py --check-deps # just print dependency status
"""

import argparse
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency availability helpers
# ---------------------------------------------------------------------------

_DEP_CACHE: dict[str, bool] = {}


def _is_available(name: str) -> bool:
    """Return True if *name* can be imported."""
    if name not in _DEP_CACHE:
        try:
            importlib.import_module(name)
            _DEP_CACHE[name] = True
        except (ImportError, OSError):
            _DEP_CACHE[name] = False
    return _DEP_CACHE[name]


def _pdal_available() -> bool:
    """Check whether a *functional* PDAL installation exists.

    The topochange.pdal_wrapper module can be imported even when PDAL
    itself is missing (it just defines a fallback class), so we inspect
    the wrapper's internal availability flags.
    """
    if _is_available("pdal"):
        return True
    try:
        from topochange.pdal_wrapper import (
            _CONDA_PDAL_AVAILABLE,
            _NATIVE_PDAL_AVAILABLE,
        )
        return _NATIVE_PDAL_AVAILABLE or _CONDA_PDAL_AVAILABLE
    except ImportError:
        return False


def _check_dependencies() -> dict[str, bool]:
    """Return a dict of {name: available} for every relevant dependency."""
    deps = {
        # Core (required)
        "numpy": _is_available("numpy"),
        "scipy": _is_available("scipy"),
        "rasterio": _is_available("rasterio"),
        "pyproj": _is_available("pyproj"),
        # Testing
        "pytest": _is_available("pytest"),
        "pytest-cov": _is_available("pytest_cov"),
        # Optional – point-cloud pipeline
        "pdal": _pdal_available(),
        "small_gicp": _is_available("small_gicp"),
    }
    return deps


# ---------------------------------------------------------------------------
# Test groups
# ---------------------------------------------------------------------------

# Tests that need no optional C-library dependencies:
CORE_TESTS = [
    "test_variogram_models",
    "test_composite_variogram",
    "test_variogram_analysis",
    "test_uncertainty",
    "test_utils",
    "test_crs_utils",
    "test_raster_and_rasterpair",
]

# Tests that require PDAL (point cloud I/O):
PDAL_TESTS = [
    "test_pointcloud_metadata",
    "test_pointcloud_transformation",
]

# Tests that require PDAL + small_gicp (alignment / DEM):
ALIGNMENT_TESTS = [
    "test_alignment",
    "test_dem_creation",
    "test_option1_integration",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the topochange test suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python run_tests.py                     # full suite
              python run_tests.py --quick             # core tests only
              python run_tests.py --coverage          # with HTML report
              python run_tests.py --file variogram    # just *variogram* tests
              python run_tests.py --check-deps        # dependency report
        """),
    )
    p.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run only core tests (no PDAL / small_gicp needed).",
    )
    p.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable pytest-cov and produce an HTML coverage report.",
    )
    p.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help=(
            "Run a single test file (substring match). "
            "E.g. --file variogram_models"
        ),
    )
    p.add_argument(
        "--check-deps",
        action="store_true",
        help="Print dependency status and exit.",
    )
    p.add_argument(
        "--markers", "-m",
        type=str,
        default=None,
        help="Pytest marker expression, e.g. 'not slow'.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    tests_dir = root / "tests"

    # ---- dependency report ------------------------------------------------
    deps = _check_dependencies()

    if args.check_deps:
        print("\n  topochange – dependency status\n")
        for name, ok in deps.items():
            icon = "✓" if ok else "✗"
            print(f"    {icon}  {name}")
        print()

        runnable = list(CORE_TESTS)
        if deps["pdal"]:
            runnable += PDAL_TESTS
        if deps["pdal"] and deps["small_gicp"]:
            runnable += ALIGNMENT_TESTS

        skipped = [
            t for t in (CORE_TESTS + PDAL_TESTS + ALIGNMENT_TESTS)
            if t not in runnable
        ]
        print(f"  Runnable test files : {len(runnable)}")
        print(f"  Will-skip files    : {len(skipped)}")
        if skipped:
            for s in skipped:
                print(f"    – {s}")
        print()
        return 0

    # ---- make sure pytest is available ------------------------------------
    if not deps["pytest"]:
        print("ERROR: pytest is not installed.", file=sys.stderr)
        print("  pip install pytest", file=sys.stderr)
        return 1

    # ---- decide which files to run ----------------------------------------
    if args.file:
        # Substring match against test file names
        matches = sorted(tests_dir.glob(f"*{args.file}*.py"))
        if not matches:
            print(f"ERROR: no test file matching '{args.file}'", file=sys.stderr)
            return 1
        targets = [str(m) for m in matches]
    elif args.quick:
        targets = [str(tests_dir / f"{t}.py") for t in CORE_TESTS]
    else:
        targets = [str(tests_dir)]

    # ---- build the pytest command -----------------------------------------
    cmd: list[str] = [sys.executable, "-m", "pytest"]

    # verbosity
    if args.verbose >= 2:
        cmd.append("-vv")
    elif args.verbose >= 1 or not args.quick:
        cmd.append("-v")

    cmd.append("--tb=short")

    # markers
    if args.markers:
        cmd.extend(["-m", args.markers])

    # coverage
    if args.coverage:
        cmd.extend([
            "--cov=topochange",
            "--cov-report=term-missing",
            f"--cov-report=html:{root / 'htmlcov'}",
        ])

    cmd.extend(targets)

    # ---- print a friendly header ------------------------------------------
    has_pdal = deps["pdal"]
    has_gicp = deps["small_gicp"]
    mode = "quick (core only)" if args.quick else "full"
    print(f"\n  topochange test runner  •  mode: {mode}")
    print(f"  PDAL: {'available' if has_pdal else 'not found (point-cloud tests will skip)'}")
    print(f"  small_gicp: {'available' if has_gicp else 'not found (alignment tests will skip)'}")
    if args.coverage:
        print(f"  Coverage report → {root / 'htmlcov' / 'index.html'}")
    print(f"  Command: {' '.join(cmd)}\n")

    # ---- run --------------------------------------------------------------
    result = subprocess.run(cmd, cwd=str(root))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
