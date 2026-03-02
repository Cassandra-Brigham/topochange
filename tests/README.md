# Tests for topochange

This directory contains the test suite for the `topochange` package, covering the full workflow from point cloud and raster I/O through variogram analysis and uncertainty propagation.

## Test Suite Overview

The suite contains 18 test modules with approximately 760 test functions organized across roughly 160 test classes. Tests are written for pytest and use synthetic data generated in shared fixtures so that most tests run without external data files.

### Shared Infrastructure

- **conftest.py** — Session-scoped fixtures that generate synthetic LAZ point clouds (rolling-hills terrain, 50,000 points each, EPSG:32610) with a known +0.5 m vertical shift between the compare and reference clouds. Provides `compare_pc`, `reference_pc`, `pc_pair`, and file path fixtures.
- **skip_markers.py** — Conditional skip decorators (`@requires_pdal`, `@requires_small_gicp`, `@requires_gdal`) that detect optional dependencies at import time. Also defines `SYNTHETIC` ground-truth constants.

### Test Modules

#### Raster and DEM

| Module | Tests | Description |
|--------|-------|-------------|
| [test_raster_and_rasterpair.py](test_raster_and_rasterpair.py) | 47 | `Raster` and `RasterPair` classes: creation from synthetic GeoTIFFs, unit conversion (meters, feet, US survey feet), CRS/geoid/unit equivalence checks, data masking, DEM differencing, and provenance tracking |
| [test_dem_creation.py](test_dem_creation.py) | 4 | DEM generation from aligned point cloud pairs, resolution setting, and unit validation |

#### Point Clouds

| Module | Tests | Description |
|--------|-------|-------------|
| [test_pointcloud_metadata.py](test_pointcloud_metadata.py) | 16 | `PointCloud` metadata extraction and updating: compound/horizontal/vertical CRS, geoid model assignment, epoch handling (single dates and ranges), consistency checks |
| [test_pointcloud_transformation.py](test_pointcloud_transformation.py) | 7 | `PointCloudPair` CRS transformation workflow: horizontal and vertical CRS reconciliation, verbose output |

#### Alignment and Registration

| Module | Tests | Description |
|--------|-------|-------------|
| [test_alignment.py](test_alignment.py) | 35 | All ICP variants (ICP, Plane-ICP, GICP, VGICP), multi-resolution coarse-to-fine alignment, `RegistrationConfig` validation, auto-revert on already-aligned data, transformation matrix properties, RMSE/fitness/convergence checks, backward-compat shims |
| [test_performance_optimizations.py](test_performance_optimizations.py) | 30 | Multi-resolution convergence, sign-convention regression, CRS transformer LRU cache, noise/outlier resilience, download resume (range headers), PDAL pipeline composition, parallel crop equivalence |

#### Variogram Analysis

| Module | Tests | Description |
|--------|-------|-------------|
| [test_variogram_analysis.py](test_variogram_analysis.py) | 60 | Matheron and Cressie-Hawkins estimators, WLS fitting, candidate generation, individual model fitting, AIC/BIC information criteria, model selection, kriging LOOCV, MSSPE criterion, Cressie weighting, model bounds |
| [test_variogram_models.py](test_variogram_models.py) | 87 | All variogram model implementations (spherical, exponential, Gaussian, Matérn, damped hole-effect, power, linear, nugget), model registry, edge cases, numerical stability |
| [test_composite_variogram.py](test_composite_variogram.py) | 75 | `CompositeVariogramModel` construction, parameter setting, evaluation, stationarity, variance decomposition, covariance function, default guess/bounds, edge cases, numerical stability |
| [test_synthetic_variogram_fitting.py](test_synthetic_variogram_fitting.py) | — | Standalone diagnostic script for model recovery from synthetic data (not collected by pytest) |

#### Uncertainty Propagation

| Module | Tests | Description |
|--------|-------|-------------|
| [test_uncertainty.py](test_uncertainty.py) | 54 | `DerivativeUncertaintyEstimator` (covariance, kernel variance, slope/curvature uncertainty), Monte Carlo integration over polygons, bootstrap parameter propagation (sigma², sample storage, correctness), edge cases |

#### CRS, Datum, and Utilities

| Module | Tests | Description |
|--------|-------|-------------|
| [test_crs_utils.py](test_crs_utils.py) | 129 | CRS conversion (ensure, equals, WKT2, ProjectionJSON), orthometric/3D checks, compound CRS creation, component parsing, epoch extraction, transformer-with-epoch, horizontal/vertical unit scales, vertical datum transforms, dynamic transforms, `vertical_datum_to_crs`, `build_output_crs_wkt` |
| [test_utils.py](test_utils.py) | 136 | `UnitInfo` dataclass, unit lookup/parsing, CRS unit extraction, length conversion round-trips, PDAL/catalog unit parsing, formatting, datetime-to-decimal-year, epoch parsing, GPS epoch/leap-seconds, numpy integration, category separation |
| [test_metadata_propagation.py](test_metadata_propagation.py) | 36 | Metadata survival through horizontal reprojection and grid alignment, transformer error paths, geoid grid errors, epoch skip warnings, compound CRS setter edge cases, `add_metadata` edge cases, time-info provenance, CRS history completeness, pipeline integration, bug-scenario regressions (H1, M4, M8, M2, H3) |
| [test_audit_fixes.py](test_audit_fixes.py) | 15 | Regression tests for audit findings: vertical CRS preserved through 2D compound assignment (H1), transformer fallback on runtime/type errors (H2), time-info updated after epoch (M2), CRS history records vertical unit conversion (M4), compound CRS consistency after component updates (M8) |

#### Data Access

| Module | Tests | Description |
|--------|-------|-------------|
| [test_data_access_pipelines.py](test_data_access_pipelines.py) | 17 | PDAL pipeline stage construction with WKT2 CRS parameters: LAS writer `a_srs`, file/AWS pipeline building, GDAL writer `override_srs`, DEM pipeline CRS handling. Uses mocking (no GDAL/PDAL required). |

#### Integration and Stress Testing

| Module | Tests | Description |
|--------|-------|-------------|
| [test_option1_integration.py](test_option1_integration.py) | 8 | End-to-end Option 1 workflow: metadata → CRS transform → alignment → DEM creation, step-by-step validation, error handling, performance timing |
| [test_synthetic_stress.py](test_synthetic_stress.py) | — | Standalone diagnostic script for variogram fitting under noise (not collected by pytest) |

## Running the Tests

### Prerequisites

```bash
# Install dev dependencies
pip install -e ".[dev]"

# For point cloud and alignment tests
pip install -e ".[dev,pointcloud,alignment]"
```

### Run All Tests

```bash
pytest
```

### Run by Marker

```bash
pytest -m alignment        # Point cloud alignment tests
pytest -m metadata         # Metadata extraction tests
pytest -m transformation   # CRS transformation tests
pytest -m dem              # DEM creation tests
pytest -m integration      # End-to-end integration tests
pytest -m "not slow"       # Skip long-running tests
```

### Run Specific Modules

```bash
pytest tests/test_variogram_analysis.py -v
pytest tests/test_crs_utils.py -v
pytest tests/test_alignment.py -v
```

### Run with Coverage

```bash
pytest --cov=topochange --cov-report=html
```

### Run a Specific Test Class or Method

```bash
pytest tests/test_alignment.py::TestAlignFunction -v
pytest tests/test_alignment.py::TestAlignFunction::test_vgicp_method -v
```

## Test Data

Most tests use synthetic data generated by `conftest.py` fixtures (session-scoped, created once per test run). Tests that require real data from `test_data/` will skip automatically if the files are not present.

Synthetic data:

- **Point clouds:** Two 50,000-point rolling-hills surfaces (EPSG:32610) with a known 0.5 m vertical offset
- **Rasters:** GeoTIFF DEMs generated at test time with controlled CRS and units
- **Variograms:** Synthetic empirical variogram data from known model parameters

## Optional Dependency Handling

Tests gracefully skip when optional libraries are not installed:

| Decorator | Library | Affected Tests |
|-----------|---------|----------------|
| `@requires_pdal` | PDAL | Point cloud I/O, DEM creation, integration |
| `@requires_small_gicp` | small_gicp | All alignment and registration tests |
| `@requires_gdal` | GDAL (osgeo) | Download resume, data access pipelines |

Variogram, uncertainty, CRS utility, and unit utility tests have no optional dependencies and always run.

## CI/CD Integration

```bash
# Minimal install (runs ~600+ tests without optional deps)
pip install -e ".[dev]"
pytest --cov=topochange --cov-report=term --cov-report=xml

# Full install (runs all ~760 tests)
pip install -e ".[dev,pointcloud,alignment]"
conda install -c conda-forge pdal python-pdal gdal
pytest --cov=topochange --cov-report=term --cov-report=xml
```

## Troubleshooting

### Import Errors

If `small_gicp` or PDAL imports fail, install them:
```bash
pip install small_gicp
conda install -c conda-forge pdal python-pdal
```

### Slow Tests

Alignment and integration tests can take 30–300 seconds. For faster iteration:
```bash
pytest -m "not slow"
pytest --ignore=tests/test_option1_integration.py
```

### Missing CRS Grids

Some CRS transformation tests may skip if PROJ grid files (geoid models, datum shifts) are not available in the local PROJ data directory.
