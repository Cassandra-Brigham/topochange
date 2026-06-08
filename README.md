# topochange

**Geostatistical uncertainty estimation for airborne lidar topographic differencing**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

`topochange` is an open-source Python package for quantifying spatially correlated uncertainty in lidar-based topographic change detection. It decomposes vertical differencing error into bias, correlated, and uncorrelated components at multiple spatial scales using nested variogram models, and propagates that uncertainty over user-defined regions of interest.

The package is integrated into the [OpenTopography](https://opentopography.org) platform for on-demand, cloud-based analysis and is also available as a standalone tool with accompanying Jupyter notebooks for customizable workflows.

> **Manuscript:**
> Brigham, C., Scott, C., Arrowsmith, R., Phan, M., DeWitt, J., Palaseanu-Lovejoy, M., Nandigam, V., Stoker, J., Anderson, S. W., Gesch, D. B., Crosby, C. J., & Beckley, M. (2026). Geostatistical error analysis in airborne lidar topographic differencing: Workflow for multi-scale uncertainty estimation of common error sources. *Earth and Space Science*.

---

## Table of Contents

- [Scientific Background](#scientific-background)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Google Colab Setup](#google-colab-setup)
- [Quick Start](#quick-start)
- [Jupyter Notebook Workflows](#jupyter-notebook-workflows)
- [API Overview](#api-overview)
- [Running Tests](#running-tests)
- [Citation](#citation)
- [Funding and Acknowledgements](#funding-and-acknowledgements)
- [License](#license)

---

## Scientific Background

Vertical topographic differencing, the pixel-by-pixel subtraction of digital elevation models (DEMs) collected at different times, is fundamental to studying landscape change from wildfires, landslides, erosion, tectonic activity, and vegetation dynamics. As data quality improves and researchers attempt to resolve smaller-magnitude changes, accounting for spatially structured uncertainty becomes critical.

The measured elevation change at any location can be thought of as a combination of true surface change plus error introduced during data collection, processing, and alignment. Think of it like trying to measure how much sand has shifted on a beach by comparing two photographs taken from slightly different angles: even if the sand hasn't moved, the photos won't line up perfectly. The misalignment _is_ the error, and it has spatial patterns that depend on its source.

These errors in airborne lidar differencing arise at multiple spatial scales:

- **Short-range (meters to tens of meters):** Random sensor noise, point misclassification (e.g., vegetation returns labeled as ground), and geometric distortion on steep slopes.
- **Mid-range (hundreds of meters):** Flight-line alignment artifacts that appear as banded stripes, and topographically correlated errors from horizontal georeferencing offsets.
- **Long-range (kilometers):** Systematic calibration biases, incorrect vertical datum metadata, or mismatched geoid models that shift an entire dataset up or down.

`topochange` uses geostatistics, specifically **semivariogram analysis**, to characterize these multi-scale error structures without needing to model each source explicitly. The approach treats the net differencing error as a spatially correlated random field (Matheron, 1965) and fits **nested spherical variogram models** to decompose the total variance into a nugget (uncorrelated noise), one or more spatially correlated components (each with a characteristic sill and range), and a systematic bias estimated from the median of the difference raster.

The fitted variogram is then used to propagate uncertainty over arbitrary polygonal regions via Monte Carlo integration of the covariance function (following Rolstad et al., 2009; Hugonnet et al., 2022). This yields a **regionalized standard deviation** that correctly accounts for spatial correlation, analogous to how averaging correlated measurements reduces uncertainty more slowly than averaging independent ones.

The variogram computation is accelerated using [Numba](https://numba.pydata.org/) JIT compilation, achieving roughly 40x faster runtimes and four orders of magnitude lower peak memory usage compared to [scikit-gstat](https://github.com/mmaelicke/scikit-gstat) at 10,000 samples.

### Key references

- Matheron, G. (1965). *Les variables régionalisées et leur estimation: une application de la théorie de fonctions aléatoires aux sciences de la nature*. Masson et Cie.
- Rolstad, C., Haug, T., & Denby, B. (2009). Spatially integrated geodetic glacier mass balance and its uncertainty based on geostatistical analysis. *Journal of Glaciology*, 15, 6456-6472. https://doi.org/10.3189/002214309789470950 
- Hugonnet, R., et al. (2022). Uncertainty analysis of digital elevation model differencing. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 270, 112876. https://doi.org/10.1109/JSTARS.2022.3188922
- Oliver, M. A., & Webster, R. (2015). *Basic Steps in Geostatistics: The Variogram and Kriging*. Springer International Publishing.
- Anderson, S. W. (2019). Uncertainty in quantitative analyses of topographic change: error propagation and the role of thresholding. *Earth Surface Processes and Landforms *, 44(5), 1015-1033. https://doi.org/10.1002/esp.4551 

---

## Features

- **Raster DEM comparison** — Load, align, and difference GeoTIFF DEMs with automatic CRS and vertical datum reconciliation
- **Point cloud workflows** — Load LAS/LAZ files via PDAL, classify, filter, generate DEMs, and register point clouds using ICP variants (ICP, point-to-plane ICP, GICP, VGICP) via [small_gicp](https://github.com/koide3/small_gicp)
- **Variogram-based uncertainty analysis** — Compute empirical variograms with bootstrap confidence intervals, fit nested models (spherical, exponential, Gaussian, Matern, damped hole-effect) using weighted nonlinear least squares with AIC model selection
- **Regional uncertainty propagation** — Monte Carlo integration of the covariance function over arbitrary polygonal areas to obtain regionalized standard deviations
- **Interactive stable area identification** — Select stable (no-change) regions on interactive maps using [ipyleaflet](https://ipyleaflet.readthedocs.io/) for focused error characterization
- **CRS and datum transformations** — Automatic vertical CRS reconciliation, geoid model handling, and tectonic deformation corrections via PROJ deformation grids
- **OpenTopography integration** — Query the OpenTopography catalog API and download data from AWS-hosted Entwine Point Tile (EPT) archives
- **Performance** — Numba-accelerated variogram computation 

---

## Repository Structure

```
topochange/
├── src/topochange/                  # Package source code
│   ├── __init__.py                  # Public API exports
│   ├── raster.py                    # GeoTIFF loading, metadata, unit conversions
│   ├── rasterpair.py                # Raster pair operations, CRS transforms, differencing
│   ├── pointcloud.py                # LAS/LAZ loading via PDAL, metadata, CRS transforms
│   ├── pointcloudpair.py            # Point cloud pair comparison, registration
│   ├── variogram.py                 # Empirical variogram computation, bootstrap CIs
│   ├── variogram_models.py          # Spherical, exponential, Gaussian, Matérn, damped hole-effect models
│   ├── composite_variogram.py       # Composite/nested variogram builder
│   ├── uncertainty.py               # Regional & derivative uncertainty propagation
│   ├── stable_area_analysis.py      # Interactive stable area identification
│   ├── alignment.py                 # ICP registration via small_gicp
│   ├── alignment_utils.py           # Registration preprocessing and quality metrics
│   ├── data_access.py               # OpenTopography API integration and EPT downloads
│   ├── pipeline_builder.py          # Automated CRS/datum transformation pipelines
│   ├── velocity_model_registry.py   # Crustal deformation model management (20+ models)
│   ├── velocity_model_converters.py # Velocity model format conversion utilities
│   ├── deformation_utils.py         # Velocity model selection
│   ├── crs_history.py               # CRS transformation history tracking
│   ├── crs_utils.py                 # CRS conversion utilities
│   ├── geoid_utils.py               # Geoid grid discovery (EGM96, EGM2008, etc.)
│   ├── unit_utils.py                # Unit conversions (meters, feet, US survey feet)
│   ├── time_utils.py                # Epoch and time conversions
│   ├── pdal_wrapper.py              # Colab-compatible PDAL integration
│   └── data/                        # Bundled data files (velocity model registry YAML)
│
├── 1_DifferencingWorkflow_user_pointclouds.ipynb  # Point cloud workflow
├── 2_DifferencingWorkflow_user_dems.ipynb         # Raster DEM workflow
├── 3_DifferencingWorkflow_download_data.ipynb     # Data download + full pipeline
│
├── tests/                           # Test suite (pytest, 18 test files)
│   ├── conftest.py                  # Shared fixtures and configuration
│   ├── test_raster_and_rasterpair.py
│   ├── test_pointcloud_metadata.py
│   ├── test_pointcloud_transformation.py
│   ├── test_alignment.py
│   ├── test_dem_creation.py
│   ├── test_option1_integration.py
│   ├── test_variogram_analysis.py
│   ├── test_variogram_models.py
│   ├── test_synthetic_variogram_fitting.py
│   ├── test_composite_variogram.py
│   ├── test_uncertainty.py
│   ├── test_crs_utils.py
│   ├── test_metadata_propagation.py
│   ├── test_data_access_pipelines.py
│   ├── test_synthetic_stress.py
│   ├── test_performance_optimizations.py
│   ├── test_audit_fixes.py
│   └── test_utils.py
│
├── test_data/                       # Example datasets
│   ├── quick_test/                  # Small reference case for fast testing
│   ├── chalk_creek/                 # Chalk Creek case study (.laz, .tif)
│   ├── paper_examples/              # Manuscript case studies (AZ, CA, IN, CO, NZ, GA, WA)
│   ├── test_dems/                   # DEM-only test data
│   ├── polygons/                    # Stable area polygon definitions
│   └── ...                          # Additional datasets (slumgullion, yakima, etc.)
│
├── pyproject.toml                   # Package configuration and dependencies
├── requirements.txt                 # Dependency list with annotations
├── pytest.ini                       # Test configuration
├── CITATION.cff                     # Citation metadata
├── MANIFEST.in                      # Package data manifest
├── run_tests.py                     # Test runner script
└── README.md                        # This file
```

---

## Installation

### Local Setup

**Prerequisites:** Python 3.9 or higher. A conda-based environment (e.g., [Miniforge](https://github.com/conda-forge/miniforge)) is recommended for managing geospatial system libraries.

#### 1. Create a conda environment (recommended)

```bash
conda create -n topochange python=3.11
conda activate topochange
```

#### 2. Install system-level geospatial libraries

Some dependencies (PDAL, GDAL/rasterio, PROJ/pyproj) rely on C/C++ libraries that are easiest to install through conda-forge:

```bash
conda install -c conda-forge pdal python-pdal gdal rasterio pyproj geopandas
```

#### 3. Install topochange

**From PyPI:**

```bash
pip install topochange
```

**From source (for development or latest changes):**

```bash
git clone https://github.com/Cassandra-Brigham/topochange.git
cd topochange
pip install -e .
```

#### 4. Install optional dependencies

```bash
# Interactive Jupyter widgets (ipyleaflet, ipywidgets)
pip install -e ".[interactive]"

# Point cloud support (PDAL, laspy) — if not already installed via conda
pip install -e ".[pointcloud]"

# Point cloud alignment (small_gicp)
pip install -e ".[alignment]"

# OpenTopography data access (boto3; also requires GDAL via conda)
pip install -e ".[data_access]"

# Velocity model format converters (xarray)
pip install -e ".[converters]"

# All optional dependencies
pip install -e ".[all]"

# Development tools (pytest, pytest-cov, black, ruff)
pip install -e ".[dev]"
```

#### Summary of dependencies

| Category | Packages |
|----------|----------|
| **Core scientific** | numpy, pandas, scipy, matplotlib, colormaps |
| **Geospatial** | rasterio, pyproj, geopandas, shapely, rioxarray |
| **Performance** | numba |
| **Networking / config** | requests, pyyaml |
| **Optional: interactive** | ipyleaflet, ipywidgets |
| **Optional: point cloud** | pdal, laspy |
| **Optional: alignment** | small_gicp |
| **Optional: data access** | boto3 (+ GDAL via conda) |
| **Optional: converters** | xarray |
| **Optional: dev** | pytest, pytest-cov, black, ruff |

### Google Colab Setup

The Jupyter notebooks include built-in Colab setup cells that handle the entire installation automatically. To run in Colab:

1. Open any of the three workflow notebooks in Google Colab (File > Open notebook > GitHub tab > paste the repository URL).
2. Run the first setup cell. It will:
   - Install [condacolab](https://github.com/conda-incubator/condacolab) to enable conda/mamba within Colab
   - Install PDAL and python-pdal via mamba
   - Pin `numpy<2.2` for compatibility
   - Install `topochange` directly from the GitHub repository
   - Install additional packages (`small-gicp`, `colormaps`, `boto3`, `ipywidgets`, `ipyleaflet`)
   - Fix the pyproj PROJ data path for the Colab environment

**Note:** The condacolab installation triggers a kernel restart. After the restart, re-run the setup cell — it will detect that conda is already configured and skip the reinstall.

The complete Colab setup sequence looks like this:

```python
import os, sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive

    conda_ready = (
        os.path.exists("/usr/local/bin/conda") and
        os.getenv("LD_LIBRARY_PATH", "").find("/usr/local/lib") >= 0
    )

    if conda_ready:
        import condacolab
        condacolab.check()
    else:
        !pip install -q condacolab
        import condacolab
        condacolab.install()  # Triggers kernel restart
```

After the environment is ready:

```bash
!mamba install -y -c conda-forge pdal python-pdal
!{sys.executable} -m pip install -q "numpy<2.2"
!{sys.executable} -m pip install -q --no-cache-dir git+https://github.com/Cassandra-Brigham/topochange.git
!{sys.executable} -m pip install -q small-gicp colormaps boto3
%pip install -q comm ipywidgets
%pip install -q ipyleaflet
```

**For Notebook 3 (data download):** You will also need an [OpenTopography API key](https://opentopography.org/). The notebook supports storing it via Google Colab Secrets or as an environment variable.

---

## Quick Start

### Comparing two DEMs (rasters)

```python
from topochange import Raster, RasterPair, RasterDataHandler, SingleVariogram

# Load two DEMs
dem_old = Raster.from_file("dem_2019.tif")
dem_new = Raster.from_file("dem_2023.tif")

# Create pair and compute difference
pair = RasterPair(dem_old, dem_new)
diff = pair.compute_difference()

# Run variogram analysis on the difference raster
handler = RasterDataHandler(diff)
variogram = SingleVariogram(handler)
results = variogram.run()
```

### Comparing two point clouds

```python
from topochange import PointCloud, PointCloudPair

# Load point clouds
pc_old = PointCloud("survey_2019.laz")
pc_new = PointCloud("survey_2023.laz")

# Create pair, register, generate DEMs, and difference
pair = PointCloudPair(pc_old, pc_new)
```

---

## Jupyter Notebook Workflows

The repository includes three notebooks that demonstrate complete workflows from data loading through uncertainty quantification:

| Notebook | Description | Input Data |
|----------|-------------|------------|
| **`1_DifferencingWorkflow_user_pointclouds.ipynb`** | Full pipeline starting from user-provided LAS/LAZ point clouds. Covers metadata inspection, CRS transformation, ICP registration, DEM generation, differencing, variogram analysis, stable area identification, and uncertainty propagation. | LAS/LAZ files |
| **`2_DifferencingWorkflow_user_dems.ipynb`** | Streamlined workflow starting from user-provided GeoTIFF DEMs. Focuses on CRS/datum reconciliation, differencing, and geostatistical uncertainty analysis. | GeoTIFF DEMs |
| **`3_DifferencingWorkflow_download_data.ipynb`** | End-to-end pipeline that downloads point cloud data from OpenTopography, then runs the full differencing and uncertainty workflow. Requires an OpenTopography API key. | OpenTopography catalog |


---

## API Overview

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Raster` | `raster` | Load GeoTIFF files, inspect metadata, handle unit conversions |
| `RasterPair` | `rasterpair` | Compare two rasters: CRS/datum transforms, differencing |
| `PointCloud` | `pointcloud` | Load LAS/LAZ via PDAL, extract metadata, CRS transforms |
| `PointCloudPair` | `pointcloudpair` | Compare two point clouds: intersection, registration, DEM generation |

### Variogram Analysis and Uncertainty

| Class / Function | Module | Description |
|------------------|--------|-------------|
| `RasterDataHandler` | `variogram` | Sample raster data for variogram computation |
| `SingleVariogram` | `variogram` | Single-resolution empirical variogram with bootstrap CIs |
| `GridVariogram` | `variogram` | Multi-resolution grid-based variogram computation |
| `KrigingLOOCVResult` | `variogram` | Leave-one-out cross-validation diagnostics |
| `AggregatedLOOCVResult` | `variogram` | Aggregated LOOCV results across resolutions |
| `MODEL_REGISTRY` | `variogram_models` | Registry of available variogram model types |
| `VariogramModelRegistry` | `variogram_models` | Fit and select nested variogram models via AIC/BIC |
| `CompositeVariogramModel` | `composite_variogram` | Build composite variogram functions from fitted components |
| `RegionalUncertaintyEstimator` | `uncertainty` | Propagate uncertainty over polygonal regions (Krige's relation) |
| `DerivativeUncertaintyEstimator` | `uncertainty` | Uncertainty for derived products |

### Alignment and Registration

| Class / Function | Module | Description |
|------------------|--------|-------------|
| `LandscapeAligner` | `alignment` | ICP-based point cloud registration via small_gicp |
| `RegistrationConfig` | `alignment` | Configuration for registration parameters |
| `RegistrationResult` | `alignment` | Registration output with quality metrics |
| `RegistrationMethod` | `alignment` | Enum of available methods (ICP, Plane-ICP, GICP, VGICP) |
| `align_point_clouds()` | `alignment` | Convenience function for one-step alignment |
| `PointCloudPreprocessor` | `alignment_utils` | Point cloud preparation and filtering |
| `AlignmentQualityMetrics` | `alignment_utils` | Registration quality assessment |
| `load_points_from_las()` | `alignment_utils` | Load points from LAS/LAZ for alignment |
| `save_transformed_las()` | `alignment_utils` | Save transformed point cloud to LAS/LAZ |
| `compute_alignment_quality()` | `alignment_utils` | Compute RMSE, fitness, and convergence metrics |

### CRS and Datum Management

| Class / Function | Module | Description |
|------------------|--------|-------------|
| `CRSHistory` | `crs_history` | Track CRS transformations through workflow for audit trails |
| `CRSState` | `pipeline_builder` | CRS descriptor with epoch, vertical kind, and geoid info |
| `build_vertical_pipeline()` | `pipeline_builder` | Construct PROJ transformation pipelines |

### Data Access

| Class | Module | Description |
|-------|--------|-------------|
| `DataAccess` | `data_access` | Unified data access interface |
| `OpenTopographyQuery` | `data_access` | Query OpenTopography catalog API |
| `GetDEMs` | `data_access` | Download and process DEM data from AWS EPT archives |

### Interactive Tools

| Class | Module | Description |
|-------|--------|-------------|
| `TopoMapInteractor` | `stable_area_analysis` | Interactive map for selecting stable areas |
| `StableAreaRasterizer` | `stable_area_analysis` | Rasterize selected polygons for masking |
| `StableAreaAnalyzer` | `stable_area_analysis` | Analyze error statistics in stable regions |

### Backward Compatibility

The following aliases are available for code written against earlier versions: `VariogramAnalysis`, `FittedVariogramModel`, `EmpiricalVariogram`, `StatisticalAnalysis`.

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=topochange

# Run specific test categories
pytest -m alignment        # Point cloud alignment tests
pytest -m metadata         # Metadata extraction tests
pytest -m transformation   # CRS transformation tests
pytest -m dem              # DEM creation tests
pytest -m integration      # End-to-end integration tests
pytest -m "not slow"       # Skip long-running tests
```

The test suite includes 18 test modules covering raster and point cloud I/O, CRS transformations, variogram analysis and model fitting, composite variograms, uncertainty propagation, alignment, data access pipelines, metadata propagation, synthetic stress tests, performance benchmarks, and regression tests.

---

## Citation

If you use `topochange` in your research, please cite:

```bibtex
@software{brigham2026topochange,
  author       = {Brigham, Cassandra and Scott, Chelsea and Arrowsmith, Ramon and
                  Phan, Minh and DeWitt, Jessica and Palaseanu-Lovejoy, Monica and
                  Nandigam, Viswanath and Stoker, Jason and Anderson, Scott Wallace and
                  Gesch, Dean B and Crosby, Christopher J and Beckley, Matthew},
  title        = {topochange},
  version      = {0.1.0},
  year         = {2026},
  url          = {https://github.com/Cassandra-Brigham/topochange},
  license      = {MIT}
}
```

And the accompanying manuscript:

> Brigham, C., Scott, C., Arrowsmith, R., Phan, M., DeWitt, J., Palaseanu-Lovejoy, M., Nandigam, V., Stoker, J., Anderson, S. W., Gesch, D. B., Crosby, C. J., & Beckley, M. (2026). Geostatistical error analysis in airborne lidar topographic differencing: Workflow for multi-scale uncertainty estimation of common error sources. *Earth and Space Science*.

---

## Funding and Acknowledgements

This work was supported by:

- **U.S. Geological Survey Powell Center** (Grant #G23AC00336)
- **National Science Foundation** (Grants #2410800, #2410799, #2410801)

We acknowledge the members of the USGS Powell Center working group on Topographic Change: Pete Chirico, Kara Doran, Zhong Lu, Carrie Middleton, Aldo Plascencia, Giulia Sofia, and Joe Wheaton.

The on-demand implementation is accessible through the [OpenTopography](https://opentopography.org) platform (Crosby et al., 2020), with compute resources provided by the San Diego Supercomputer Center.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
