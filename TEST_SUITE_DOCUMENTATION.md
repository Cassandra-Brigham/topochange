# topochange Test Suite Documentation

> **18 files · ~164 test classes · ~751+ test methods**

Comprehensive guide to all test files, classes, and coverage areas in the topochange test suite.

---

## Table of Contents

- [Overview](#overview)
- [File Summary](#file-summary)
- [Support Files](#support-files)
  - [conftest.py](#conftestpy)
  - [skip_markers.py](#skip_markerspy)
- [Test File Documentation](#test-file-documentation)
  - [test_alignment.py](#test_alignmentpy)
  - [test_performance_optimizations.py](#test_performance_optimizationspy)
  - [test_audit_fixes.py](#test_audit_fixespy)
  - [test_metadata_propagation.py](#test_metadata_propagationpy)
  - [test_crs_utils.py](#test_crs_utilspy)
  - [test_data_access_pipelines.py](#test_data_access_pipelinespy)
  - [test_dem_creation.py](#test_dem_creationpy)
  - [test_option1_integration.py](#test_option1_integrationpy)
  - [test_pointcloud_metadata.py](#test_pointcloud_metadatapy)
  - [test_pointcloud_transformation.py](#test_pointcloud_transformationpy)
  - [test_raster_and_rasterpair.py](#test_raster_and_rasterpairpy)
  - [test_uncertainty.py](#test_uncertaintypy)
  - [test_utils.py](#test_utilspy)
  - [test_variogram_models.py](#test_variogram_modelspy)
  - [test_variogram_analysis.py](#test_variogram_analysispy)
  - [test_composite_variogram.py](#test_composite_variogrampy)
- [Dependency Requirements](#dependency-requirements)

---

## Overview

The topochange test suite validates a geospatial Python library for topographic change detection using lidar point clouds and raster DEMs. Tests cover the full pipeline: point cloud I/O, CRS handling, ICP alignment, DEM creation, raster differencing, variogram modeling, and uncertainty estimation.

The suite is organized into 15 test files plus 2 support files (`conftest.py` and `skip_markers.py`). Optional dependencies (PDAL, small_gicp, GDAL) are detected at import time; tests that require unavailable packages are automatically skipped via pytest markers.

---

## File Summary

| File | Classes | Tests | Primary Coverage |
|------|:-------:|:-----:|------------------|
| `conftest.py` | 0 | 6 | Shared pytest fixtures for the entire test suite |
| `skip_markers.py` | 0 | 3 | Shared skip markers and constants used by all test files |
| `test_alignment.py` | 6 | 17 | Point cloud alignment (ICP registration) |
| `test_performance_optimizations.py` | 11 | 30 | Performance optimizations (R1–R10) and coarse-to-fine multi-resolution alignment (Rec #4) |
| `test_audit_fixes.py` | 6 | 16 | Regression tests for the seven ICP alignment audit findings |
| `test_metadata_propagation.py` | 12 | 28 | Metadata propagation through transformation chains |
| `test_crs_utils.py` | 21 | 130 | CRS object creation, equality, WKT2/PROJJSON conversion, compound CRS, epoch-aware transformers, unit scales, vertical datum transforms |
| `test_data_access_pipelines.py` | 6 | 15 | PDAL pipeline construction in the data\_access module |
| `test_dem_creation.py` | 2 | 4 | DEM creation from aligned point clouds (Option 1 workflow) |
| `test_option1_integration.py` | 4 | 14 | Complete Option 1 workflow integration |
| `test_pointcloud_metadata.py` | 3 | 18 | Point cloud metadata extraction and updating |
| `test_pointcloud_transformation.py` | 4 | 10 | Point cloud CRS transformation |
| `test_raster_and_rasterpair.py` | 16 | 60 | Raster and RasterPair classes |
| `test_uncertainty.py` | 8 | 50 | Derivative-based uncertainty estimation and Monte Carlo integration |
| `test_utils.py` | 26 | 100 | unit\_utils (unit conversion, lookup, parsing) and time\_utils (datetime, epoch, GPS time) |
| `test_variogram_models.py` | 13 | 100 | Individual variogram model functions and model registry |
| `test_variogram_analysis.py` | 14 | 50 | Variogram analysis infrastructure (fitting, selection, bootstrap) |
| `test_composite_variogram.py` | 12 | 100 | CompositeVariogramModel class |
| **TOTAL** | **~164** | **~751+** | **Full pipeline coverage** |

---

## Support Files

### `conftest.py`

The shared fixture module for the entire test suite. It generates two synthetic LAZ point-cloud files using PDAL: `compare.laz` and `reference.laz`. Both share the same deterministic sum-of-sinusoids terrain (rolling hills), but `reference.laz` has a known +0.5 m vertical shift, simulating real elevation change between two survey epochs.

**Fixtures provided:**

- **`synthetic_test_data_dir`** — Session-scoped directory containing `compare.laz` and `reference.laz` (EPSG:32610, 50K points each, 500×500 m extent).
- **`compare_laz_path` / `reference_laz_path`** — String paths to the individual LAZ files.
- **`compare_pc` / `reference_pc`** — Loaded `PointCloud` objects ready for testing.
- **`pc_pair`** — A `PointCloudPair` built from both synthetic clouds.

### `skip_markers.py`

Detects optional dependencies at import time and provides reusable pytest skip decorators. Also exports ground-truth constants for the synthetic test data.

**Markers & constants:**

- **`requires_pdal`** — Skips test if PDAL is not installed (checks both native `pdal` and the project wrapper).
- **`requires_small_gicp`** — Skips test if the `small_gicp` ICP registration library is missing.
- **`requires_gdal`** — Skips test if GDAL (`osgeo`) is not available.
- **`SYNTHETIC` dict** — Ground-truth values: EPSG 32610, 50K points, spatial origin (500000 E, 4000000 N), 500 m extent, z\_base 1000 m, z\_shift\_reference +0.5 m.

---

## Test File Documentation

### `test_alignment.py`

Tests for point cloud alignment (ICP registration). Covers the full `PointCloudPair` alignment path using synthetic LAZ files (requires PDAL + small\_gicp) and direct numpy-array alignment (requires only small\_gicp). Verifies convergence, transformation matrix properties, RMSE, fitness, and method consistency.

**Statistics:** 6 test classes, ~17 test methods

**Test classes:**

- **`TestAlignmentBasics`** — Basic VGICP alignment through `PointCloudPair`, including memory-constrained runs with larger voxel sizes.
- **`TestAlignmentMethods`** — Compares VGICP, GICP, and ICP methods through the `LandscapeAligner` interface.
- **`TestAlignmentResults`** — Validates transformation matrix structure (4×4, orthogonal rotation block, `[0,0,0,1]` bottom row), reasonable RMSE (<10 m), fitness (>0.3), and convergence flag.
- **`TestAlignmentConfiguration`** — Parametric tests for different downsample resolutions (0.5/1.0/2.0 m) and `max_points` (100K/500K).
- **`TestAlignmentWithTransformation`** — End-to-end alignment + transformation application workflow.
- **`TestSyntheticAlignment`** — Pure-numpy tests with known ground truth: recovery of pure translation, translation+rotation, VGICP vs GICP consistency, partial overlap convergence, and near-identity alignment.

---

### `test_performance_optimizations.py`

Tests for performance optimizations (R1–R10) and the coarse-to-fine multi-resolution alignment strategy (Rec #4). Covers multi-resolution convergence, sign convention regression, PLANE\_ICP code path, CRS Transformer LRU cache, noise/outlier resilience, HTTP Range-header download resume, deque BFS ordering, PDAL pipeline composition, and parallel crop equivalence.

**Statistics:** 11 test classes, ~30 test methods

**Test classes:**

- **`TestMultiResolutionConvergence`** — Verifies coarse-to-fine alignment converges for large misalignments (10 m translation, 5° rotation) and is not worse than single-resolution.
- **`TestMultiResolutionDisabled`** — Confirms single-resolution alignment (`multi_resolution=False`) still converges for small offsets.
- **`TestMultiResolutionCustomStages`** — Tests custom 2-stage and 5-stage resolution schedules.
- **`TestSignConventionRegression`** — Guards the `T_target_source` sign convention: positive source offset yields negative translation in the transform matrix, and vice versa.
- **`TestPlaneICPSynthetic`** — Tests the previously-untested `PLANE_ICP` code path (point-to-plane error metric) for pure translation, rotation, and multi-resolution.
- **`TestCRSTransformerCache`** — Verifies the R10 LRU-cached CRS transformer: cache hits on repeated calls, CRS object/string sharing, correct coordinate output, and fallback for non-authority CRS.
- **`TestNoiseAndOutlierResilience`** — Tests GICP robustness to 0.1 m Gaussian noise, 5% gross outliers (10 m jumps), and combined noise+outliers with multi-resolution.
- **`TestDownloadResume`** — Mock-based tests for HTTP Range-header resume logic: fresh download, 206 partial resume, 416 skip, and server-ignores-Range fallback.
- **`TestDequeBFSOrdering`** — Confirms `deque.popleft()` produces identical BFS traversal order to `list.pop(0)`, both statically and with dynamic child appends.
- **`TestRunChainComposition`** — Tests PDAL pipeline composition: classification+voxel filter chain, voxel-only size reduction, and `overwrite=False` skip behavior.
- **`TestParallelCropEquivalence`** — Verifies `ThreadPoolExecutor`-based parallel `crop_to_overlap` produces two valid cropped point cloud files.

---

### `test_audit_fixes.py`

Regression tests for the seven ICP alignment audit findings. Each test class targets a specific bug fix, ensuring the fix holds and the original bug does not reappear.

**Statistics:** 6 test classes, ~16 test methods

**Test classes:**

- **`TestH1_VerticalCRSPreserved`** — Verifies that setting `current_compound_crs` to a 2D CRS does not wipe the existing vertical CRS.
- **`TestH2_TransformerWithEpochExceptionHandling`** — Confirms `transformer_with_epoch()` catches `RuntimeError` and `TypeError` (not just `TypeError`) and falls back to non-epoch transform.
- **`TestM2_TimeInfoUpdatedAfterEpoch`** — Checks that `add_metadata(epoch=...)` updates `time_info` with the new epoch value and `epoch_source='add_metadata'` for single values, ranges, and date strings.
- **`TestM4_CRSHistoryRecordTransformation`** — Ensures `convert_vertical_units()` calls `record_transformation_entry()` (not the non-existent `add_entry()`).
- **`TestM8_CompoundCRSConsistency`** — Verifies `_update_current_compound_from_components()` stores horizontal CRS as compound when no vertical exists, never leaving compound as `None`.
- **`TestIntegration`** — Combined regression tests: H1+M8 together, and M2 with both epoch and compound CRS.

---

### `test_metadata_propagation.py`

Tests for metadata propagation through transformation chains. Targets five structural gaps: cross-function metadata survival, error-path handling, negative-case property tests, provenance/audit-trail completeness, and pipeline integration.

**Statistics:** 12 test classes, ~28 test methods

**Test classes:**

- **`TestMetadataSurvivesHorizontalReproject`** — Confirms vertical CRS, vertical units, and epoch survive horizontal-only `warp_raster()` calls.
- **`TestMetadataSurvivesGridAlignment`** — Verifies metadata survives alignment-only warps (same CRS, different grid).
- **`TestTransformerWithEpochErrorPaths`** — Tests `RuntimeError` and `TypeError` fallback paths in `transformer_with_epoch`, verifying warning messages and call counts.
- **`TestGeoidGridErrorPaths`** — Ensures `_apply_geoid_to_raster` raises for missing geoid grids instead of silently returning unchanged data.
- **`TestEpochSkipWarning`** — Verifies a warning fires when epoch mismatch exists but target epoch is `None`.
- **`TestCompoundCRSSetterEdgeCases`** — Edge cases for compound CRS property: setting `None`, 2D CRS, true compound, horizontal-only, vertical-only.
- **`TestAddMetadataEdgeCases`** — Tests that `add_metadata` with horizontal-only CRS preserves vertical CRS, and sequential epoch+CRS calls are additive.
- **`TestTimeInfoProvenance`** — Checks `time_info` `epoch_source` tracking, overwrite behavior, and preservation of non-epoch keys.
- **`TestCRSHistoryCompleteness`** — Verifies `convert_vertical_units` and `warp_raster` record CRS history transformation entries.
- **`TestPipelineMetadataIntegration`** — End-to-end multi-step pipeline tests: horizontal+alignment chain, additive `add_metadata` calls, 3-warp chain stability, and `time_info` through warp.
- **`TestParseCRSComponentsEdgeCases`** — Tests `parse_crs_components` with 2D projected, geographic, compound, and vertical-only CRS inputs.
- **`TestBugScenarioRegressions`** — Concrete reproductions of original bugs: H1 vertical CRS loss, H1 cascade false-match, M4 `add_entry` `AttributeError`, M8 compound inconsistency, M2 stale `time_info`, and H3 silent geoid failure.

---

### `test_crs_utils.py`

Comprehensive tests for the `crs_utils` module: CRS object creation, equality, WKT2/PROJJSON conversion, coordinate metadata wrapping, orthometric/ellipsoidal detection, compound CRS creation/parsing, epoch-aware transformers, unit scale extraction, vertical datum transforms, and dynamic 3D transforms.

**Statistics:** 21 test classes, ~130 test methods

**Test classes:**

- **`TestEnsureCrsObj`** — Tests `_ensure_crs_obj` parsing from EPSG strings, WKT, PROJJSON, and CRS objects.
- **`TestCrsEquals`** — Tests CRS equality comparisons across different input formats (EPSG codes, WKT, CRS objects).
- **`TestCrsToWkt2_2019`** — Tests WKT2:2019 format output with pretty printing.
- **`TestWrapCoordinateMetadataWkt`** — Tests `COORDINATEMETADATA` WKT wrapper with epoch values.
- **`TestCrsToProjectionJson` / `TestMakeCoordinateMetadataProjectionJson`** — Tests PROJJSON format conversion with and without epoch metadata.
- **`TestIsOrthometric` / `TestIs3dGeographicCrs`** — Tests height type detection (orthometric vs ellipsoidal, 2D vs 3D geographic).
- **`TestExtractEllipsoidalHeightAsVerticalCrs`** — Tests extracting vertical CRS from 3D geographic CRS.
- **`TestCreateCompoundCrs` / `TestParseCrsComponents`** — Tests compound CRS creation from horizontal+vertical components and parsing back into components.
- **`TestTransformerWithEpoch`** — Tests coordinate transformer creation with source/destination epochs.
- **`TestHorizontalUnitScale` / `TestVerticalUnitScale`** — Tests unit conversion factor extraction from CRS (metres, feet, US survey feet).
- **`TestApplyVerticalDatumTransform` / `TestApplyDynamicTransform`** — Tests vertical datum transformations and full 3D coordinate transforms.
- **`TestVerticalDatumToCrs`** — Tests vertical datum name → CRS mapping (NAVD88, NGVD29, EGM96, EGM2008).
- **`TestBuildOutputCrsWkt` / `TestExtractEpochFromWkt`** — Tests compound WKT2 builder and epoch extraction from `COORDINATEMETADATA` WKT.
- **`TestCrsUtilsIntegration` / `TestEdgeCasesAndErrorHandling`** — Integration tests combining multiple functions, plus edge cases (empty strings, extreme epochs, large arrays).

---

### `test_data_access_pipelines.py`

Tests for PDAL pipeline construction in the `data_access` module. Verifies that writer stages include proper `a_srs` (CRS) parameters for LAS writers and `override_srs` parameters for GDAL writers when `output_crs_wkt` is provided. Uses `unittest.mock` to patch heavy GDAL imports.

**Statistics:** 6 test classes, ~15 test methods

**Test classes:**

- **`TestWriterLasAsrs`** — Tests `_writer_las()` with/without `a_srs`, LAS vs LAZ compression, and invalid extension handling.
- **`TestBuildPipelineFromFileAsrs`** — Tests `build_pdal_pipeline_from_file` passes `a_srs` to writer stages, omits when not provided, and skips writer when `savePointCloud=False`.
- **`TestBuildAwsPipelineAsrs`** — Tests `build_aws_pdal_pipeline` passes `a_srs` to writer stages and omits by default.
- **`TestWriterGdalOverrideSrs`** — Tests `_writer_gdal()` with/without `override_srs`, and explicit `None` handling.
- **`TestMakeDemPipelineGdalSrs` / `TestMakeDemPipelineAwsGdalSrs`** — Tests `make_DEM_pipeline_from_file` and `make_DEM_pipeline_aws` use `output_crs_wkt` for `writers.gdal` `override_srs`, with fallback to `outCRS` string.

---

### `test_dem_creation.py`

Tests for DEM creation from aligned point clouds (Option 1 workflow). Requires PDAL and small\_gicp for the full alignment pipeline before creating DTMs.

**Statistics:** 2 test classes, ~4 test methods

**Test classes:**

- **`TestDEMCreation`** — Tests DTM pair creation from aligned point clouds: verifies DEM objects are non-null, have data, and files exist on disk.
- **`TestDEMProperties`** — Tests DEM units (horizontal/vertical) and the `set_units` method for post-creation metadata.

---

### `test_option1_integration.py`

Integration tests for the complete Option 1 workflow (the notebook-style pipeline): load point clouds, add metadata, transform CRS, align, and create DEMs.

**Statistics:** 4 test classes, ~14 test methods

**Test classes:**

- **`TestOption1CompleteWorkflow`** — Full end-to-end workflow test running all steps in sequence.
- **`TestOption1WorkflowStepByStep`** — Tests each pipeline step individually: metadata loading, pairing, CRS transformation, alignment, DEM creation.
- **`TestOption1ErrorHandling`** — Tests error handling for missing metadata and invalid CRS values.
- **`TestOption1Performance`** — Tests that the workflow completes in reasonable time without hanging.

---

### `test_pointcloud_metadata.py`

Tests for point cloud metadata extraction and updating. Verifies that `PointCloud` objects correctly read CRS, bounds, point count, and epoch from files, and that metadata can be updated programmatically.

**Statistics:** 3 test classes, ~18 test methods

**Test classes:**

- **`TestPointCloudMetadataExtraction`** — Tests `PointCloud` creation and metadata reading (CRS, bounds, point count, classification).
- **`TestPointCloudMetadataUpdate`** — Tests updating CRS, vertical datum, geoid model, and epoch on `PointCloud` objects.
- **`TestReferencePointCloudMetadata`** — Tests reference point cloud metadata handling and pretty-print output.

---

### `test_pointcloud_transformation.py`

Tests for point cloud CRS transformation. Verifies `PointCloudPair` creation, horizontal/vertical CRS transformation with `skip_epoch`, and verbose output during transforms.

**Statistics:** 4 test classes, ~10 test methods

**Test classes:**

- **`TestPointCloudPairCreation`** — Tests `PointCloudPair` object creation and comparison attributes.
- **`TestPointCloudTransformation`** — Tests CRS transformation with `skip_epoch=True` flag.
- **`TestCRSTransformationComponents`** — Tests horizontal and vertical CRS transformation components independently.
- **`TestTransformationWithVerboseOutput`** — Tests that verbose output is generated during transformation.

---

### `test_raster_and_rasterpair.py`

Comprehensive tests for `Raster` and `RasterPair` classes. Covers creation from file, data access, unit conversion, metadata, CRS properties and tracking, equivalence helpers, valid data masks, pair comparisons, DEM differencing, provenance tracking, and extent/overlap detection.

**Statistics:** 16 test classes, ~60 test methods

**Test classes:**

- **`TestRasterCreation`** — Tests `Raster.from_file()`, CRS detection, bounds, shape, and resolution.
- **`TestRasterDataAccess`** — Tests data property and value matching against the source GeoTIFF.
- **`TestRasterUnitConversion`** — Tests `set_units`, `convert_to_meters`, `convert_from_meters`, and roundtrip precision.
- **`TestRasterMetadata`** — Tests `add_metadata` for epoch and CRS updates.
- **`TestRasterCRSProperties`** — Tests original vs current CRS tracking during reprojection.
- **`TestCRSEquivalence` / `TestGeoidEquivalence` / `TestUnitsEquivalent`** — Tests helper functions for CRS, geoid, and unit equivalence comparisons.
- **`TestCreateValidDataMask`** — Tests valid data mask creation handling nodata, NaN, and Inf values.
- **`TestRasterPairCreation`** — Tests `RasterPair` object creation and attribute access.
- **`TestRasterPairComparisons`** — Tests CRS match checking: same CRS, different horizontal, different compound.
- **`TestRasterPairDifferencing`** — Tests DEM differencing with statistics (mean, std, count).
- **`TestRasterPairProvenance` / `TestRasterPairExtent`** — Tests transformation history, provenance tracking, and overlap polygon generation.
- **`TestRasterIntegration` / `TestRasterPairIntegration`** — End-to-end integration for load→transform→metadata and compare→transform→difference workflows.

---

### `test_uncertainty.py`

Tests for the uncertainty module: derivative-based uncertainty estimation (covariance functions, kernel variance, slope/curvature uncertainty) and Monte Carlo integration for regional uncertainty. Uses synthetic variogram models as input.

**Statistics:** 8 test classes, ~50 test methods

**Test classes:**

- **`TestDerivativeEstimatorCovariance`** — Tests covariance function C(h) = sill − γ(h) for various variogram models.
- **`TestDerivativeEstimatorKernelVariance`** — Tests kernel variance computation for Sobel and Laplacian operators.
- **`TestDerivativeEstimatorSlopeUncertainty`** — Tests slope uncertainty magnitude and directional component consistency.
- **`TestDerivativeEstimatorCurvatureUncertainty`** — Tests curvature uncertainty from the Laplacian kernel.
- **`TestKernelsDictionary`** — Tests the `KERNELS` dictionary entries (Sobel X, Sobel Y, Laplacian) for correct values.
- **`TestMonteCarloIntegration`** — Tests Monte Carlo regional uncertainty integration for nugget, exponential, and spherical models.
- **`TestEdgeCases`** — Tests edge cases: zero sill, very small resolution, large range, negative variance handling.
- **`TestIntegration`** — End-to-end tests combining derivative estimation and Monte Carlo.

---

### `test_utils.py`

Tests for `unit_utils` (unit conversion, lookup, parsing, CRS unit extraction) and `time_utils` (datetime/decimal year conversion, epoch parsing, GPS time). The largest test file by class count.

**Statistics:** 26 test classes, ~100 test methods

**Test classes:**

- **`TestUnitInfoDataclass`** — Tests `UnitInfo` creation, immutability, and string representation.
- **`TestUnitConversion`** — Tests conversion methods: meter↔foot, kilometer, degree↔radian.
- **`TestLookupUnit` / `TestLookupUnitStrict`** — Tests unit lookup by name/alias with case insensitivity and strict-mode error handling.
- **`TestParseUnitString` / `TestCRSUnitExtraction`** — Tests parsing units from parenthesized strings and extracting units from CRS objects.
- **`TestConvertLength` / `TestConvertToMeters` / `TestConvertFromMeters`** — Tests length conversion wrapper, to-meters, and from-meters conversions.
- **`TestConversionRoundtrip` / `TestGetConversionFactor`** — Tests roundtrip conversion precision and direct conversion factor computation.
- **`TestParsePdalUnits` / `TestParseCatalogVerticalUnits`** — Tests PDAL SRS metadata and catalog vertical unit string parsing.
- **`TestFormatValueWithUnit` / `TestDescribeUnit`** — Tests value formatting with units/precision and human-readable unit descriptions.
- **`TestDatetimeToDecimalYear` / `TestParseEpochString`** — Tests datetime→decimal year and epoch string parsing (ISO, US, ranges).
- **`TestGpsEpoch` / `TestGpsLeapSeconds` / `TestGpsSecondsToDecimalYear`** — Tests GPS epoch constant, leap second computation, and GPS seconds→decimal year.
- **`TestGuessInTimeFromStats` / `TestTimeUtilsIntegration`** — Tests time format heuristics and integration tests combining time utilities.
- **`TestEdgeCases` / `TestUnitConstantsExist` / `TestNumPyIntegration` / `TestCategorySeparation`** — Tests edge cases (NaN, Inf, very large/small), unit constant availability, numpy array handling, and linear vs angular category separation.

---

### `test_variogram_models.py`

Tests for individual variogram model functions and the model registry. Covers all eight model types: spherical, exponential, Gaussian, Matérn, damped hole-effect, power, linear, and nugget. Verifies mathematical properties (origin behavior, monotonicity, boundedness, sill convergence) and registry operations.

**Statistics:** 13 test classes, ~100 test methods

**Test classes:**

- **`TestSphericalModel`** — Tests spherical model: γ(0)=0, reaches sill at range, monotonic increase, bounded.
- **`TestExponentialModel`** — Tests exponential model: asymptotic approach to sill, never exceeds sill, monotonic.
- **`TestGaussianModel`** — Tests Gaussian model: parabolic behavior near origin, asymptotic sill approach.
- **`TestMaternModel`** — Tests Matérn model with various ν values, including special cases.
- **`TestDampedHoleEffectModel`** — Tests damped hole-effect: oscillatory behavior, asymptotic convergence to sill.
- **`TestPowerModel`** — Tests power model: unbounded growth, scaling laws.
- **`TestLinearModel`** — Tests linear model: proportional to h, unbounded.
- **`TestNuggetModel`** — Tests nugget model: discontinuity at h=0, constant for h>0.
- **`TestVariogramModelRegistry`** — Tests model registry: get, list, validate, bounded vs unbounded classification.
- **`TestVariogramModelSpec`** — Tests `VariogramModelSpec` dataclass: default guess, bounds, validation.
- **`TestGlobalRegistry`** — Tests the global `MODEL_REGISTRY` instance.
- **`TestEdgeCases` / `TestNumericalStability`** — Tests edge cases (scalar inputs, empty arrays, extreme parameters) and numerical stability.

---

### `test_variogram_analysis.py`

Tests for the variogram analysis infrastructure: the Matheron estimator, `VariogramModelSelector` for candidate generation and fitting, information criteria (AIC/BIC), model selection, cross-validation, Akaike weights, Bayesian Model Averaging, bootstrap uncertainty, and the `FittedVariogramModel` interface.

**Statistics:** 14 test classes, ~50 test methods

**Test classes:**

- **`TestMatheronEstimator`** : Tests Matheron estimator computation and minimum-pairs filtering.
- **`TestVariogramModelSelectorConstruction`** — Tests selector initialization with models, weights, and distance bins.
- **`TestCandidateGeneration`** — Tests candidate model generation with various configuration options.
- **`TestModelFitting`** — Tests model fitting on synthetic data with parameter estimation accuracy.
- **`TestInformationCriteria`** — Tests AIC and BIC computation with correct parameter penalties.
- **`TestModelSelection`** — Tests best model selection by AIC and BIC criteria.
- **`TestCrossValidation`** — Tests k-fold cross-validation for model evaluation.
- **`TestAkaikeWeights`** — Tests Akaike weight computation: sum to 1, best model gets highest weight.
- **`TestBMAVariogram`** — Tests Bayesian Model Averaging variogram: weighted combination of models.
- **`TestBootstrapUncertainty`** — Tests bootstrap uncertainty estimation for variogram parameters.
- **`TestFittedVariogramModel`** — Tests `FittedVariogramModel` prediction and percentile computation.
- **`TestInitialGuess` / `TestFitAllCandidates`** — Tests initial parameter guess generation and fitting all candidates with optional cross-validation.
- **`TestIntegration`** — End-to-end tests: fit→select→bootstrap→predict workflow.

---

### `test_composite_variogram.py`

Tests for the `CompositeVariogramModel` class, which combines multiple variogram component models (e.g., spherical + exponential + nugget) into a single model. Covers construction, parameter handling, evaluation, stationarity, variance decomposition, covariance functions, and numerical stability.

**Statistics:** 12 test classes, ~100 test methods

**Test classes:**

- **`TestCompositeVariogramConstruction`** — Tests model construction: single/multi-component, with/without nugget, bounded/unbounded.
- **`TestParameterSetting`** — Tests parameter validation, setting, and component parameter extraction.
- **`TestEvaluation`** — Tests variogram evaluation at different lag distances with scalar and array inputs.
- **`TestStationarity`** — Tests stationarity determination, sill calculations, and stationary sill extraction.
- **`TestVarianceDecomposition`** — Tests variance decomposition by component with custom reference lags.
- **`TestCovarianceFunction`** — Tests covariance function C(h) = sill − γ(h).
- **`TestDefaultGuessAndBounds`** — Tests default parameter guess and bounds generation for curve fitting.
- **`TestDescription`** — Tests human-readable model description output.
- **`TestPropertyAccess`** — Tests property queries: `params`, `n_params`, `param_names`.
- **`TestIntegrationScenarios`** — Realistic workflows: spherical+exponential, Matérn, non-stationary.
- **`TestEdgeCases` / `TestNumericalStability`** — Tests edge cases (very small sill, large range, many components) and evaluation consistency.

---

## Dependency Requirements

Tests are designed to run in environments with varying dependency availability. The `skip_markers` module detects each optional dependency once at import time, and tests that need them are decorated with the corresponding marker.

### Always Required

`pytest`, `numpy`, `pyproj`, `rasterio`, `shapely` : core test infrastructure and CRS/raster operations.

### Conditionally Required

- **PDAL** : Required for LAZ file I/O, PDAL pipeline tests, conftest fixture generation, DEM creation tests, and integration tests. Detected via native `pdal` import or the project wrapper.
- **small\_gicp** : Required for all ICP alignment tests (GICP, VGICP, PLANE\_ICP), multi-resolution convergence tests, and sign convention regression tests.
- **GDAL (osgeo)** : Required for the download resume tests in `test_performance_optimizations.py` (`GetDEMs` imports `osgeo.gdal` at module level).

### Running the Suite

```bash
cd tests && python -m pytest -v
```

Unavailable tests will be reported as `SKIPPED` with reason messages indicating which dependency is missing.
