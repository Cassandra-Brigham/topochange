"""Shared test fixtures for topochange test suite.

Provides:
  - Synthetic LAZ file generation with known geometry, CRS, and GPS time
  - Reusable PointCloud and PointCloudPair fixtures

All synthetic data is written using PDAL pipelines (the same library the
project itself uses), so the test files are guaranteed to be readable by
PointCloud.from_file() without any extra dependencies."""

# Exclude standalone diagnostic scripts that are not pytest suites.
collect_ignore = [
    "test_synthetic_stress.py",
    "test_synthetic_variogram_fitting.py",
]

import json
import pytest
import numpy as np

from skip_markers import HAS_PDAL


# synthetic LAZ generation via PDAL

def _create_synthetic_laz(
    filepath: str,
    *,
    n_points: int = 50_000,
    x_offset: float = 500_000.0,
    y_offset: float = 4_000_000.0,
    x_extent: float = 500.0,
    y_extent: float = 500.0,
    z_base: float = 1000.0,
    z_shift: float = 0.0,
    epsg: int = 32610,
    gps_time_base: float = 7.9e8,
    seed: int = 42,
):
    """Create a synthetic LAZ file with terrain-like geometry using PDAL.

    The surface is a deterministic sum-of-sinusoids (rolling hills) so that
    two files created with identical seeds but different ``z_shift`` values
    represent the *same* landscape with a known vertical offset : analogous
    to measuring the same terrain at two epochs with some real elevation
    change between them.

    Parameters
    ----------
    filepath : str
        Output path (should end in .laz).
    n_points : int
        Number of points to generate.
    x_offset, y_offset : float
        UTM easting / northing origin.
    x_extent, y_extent : float
        Spatial extent of the point cloud in metres.
    z_base : float
        Base elevation in metres.
    z_shift : float
        Constant vertical shift applied to all Z values.
    epsg : int
        EPSG code written into the output CRS via ``a_srs``.
    gps_time_base : float
        Base GPS timestamp (adjusted GPS seconds).
    seed : int
        Random seed for reproducibility.
    """
    # import PDAL the same way the project does
    try:
        from topochange.pdal_wrapper import pdal
    except ImportError:
        import pdal

    rng = np.random.default_rng(seed)

    # --- coordinates ---
    x_raw = rng.uniform(0, x_extent, n_points)
    y_raw = rng.uniform(0, y_extent, n_points)

    # deterministic terrain surface (rolling hills)
    z_terrain = (
        5.0 * np.sin(x_raw / 40.0) * np.cos(y_raw / 40.0)
        + 2.0 * np.sin(x_raw / 15.0) * np.sin(y_raw / 20.0)
        + 0.5 * np.cos(x_raw / 8.0 + y_raw / 8.0)
    )
    z_noise = rng.normal(0, 0.05, n_points)

    x = x_raw + x_offset
    y = y_raw + y_offset
    z = z_base + z_terrain + z_noise + z_shift

    # --- GPS time (adjusted GPS seconds) ---
    gps_time = gps_time_base + rng.uniform(0, 86400.0 * 10, n_points)

    # --- classification: all ground (class 2) ---
    classification = np.full(n_points, 2, dtype=np.uint8)

    # --- Build structured numpy array ---
    # point format 1 fields: X, Y, Z, Intensity, ReturnNumber,
    # numberOfReturns, Classification, GpsTime
    dtype = np.dtype([
        ("X", "f8"),
        ("Y", "f8"),
        ("Z", "f8"),
        ("Intensity", "u2"),
        ("ReturnNumber", "u1"),
        ("NumberOfReturns", "u1"),
        ("Classification", "u1"),
        ("GpsTime", "f8"),
    ])
    arr = np.zeros(n_points, dtype=dtype)
    arr["X"] = x
    arr["Y"] = y
    arr["Z"] = z
    arr["Intensity"] = 0
    arr["ReturnNumber"] = 1
    arr["NumberOfReturns"] = 1
    arr["Classification"] = classification
    arr["GpsTime"] = gps_time

    # --- Write to LAZ via PDAL writers.las ---
    pipeline_spec = {
        "pipeline": [
            {
                "type": "writers.las",
                "filename": filepath,
                "a_srs": f"EPSG:{epsg}",
                "compression": "laszip",
                "minor_version": 4,
                "dataformat_id": 1,       # point format 1 (has GPS time)
                "offset_x": x_offset,
                "offset_y": y_offset,
                "offset_z": 0.0,
                "scale_x": 0.001,
                "scale_y": 0.001,
                "scale_z": 0.001,
            }
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_spec), arrays=[arr])
    pipeline.execute()


# session-scoped fixtures (one set of files for the whole test run)

@pytest.fixture(scope="session")
def synthetic_test_data_dir(tmp_path_factory):
    """Create a temporary directory with synthetic compare.laz and reference.laz.

    Known properties
    ----------------
    Both files:
      - CRS: EPSG:32610  (UTM Zone 10N, metres)
      - 50 000 points, all classified as ground (class 2)
      - Spatial extent: 500 x 500 m starting at (500000 E, 4000000 N)
      - Terrain: deterministic sum-of-sinusoids, same seed

    compare.laz:
      - z_base = 1000 m,  z_shift = 0
      - GPS time ~2005  (gps_time_base = 7.9e8 adjusted seconds)

    reference.laz:
      - z_base = 1000 m,  z_shift = +0.5 m  (known surface change)
      - GPS time ~2018  (gps_time_base = 1.2e9 adjusted seconds)
    """
    if not HAS_PDAL:
        pytest.skip("PDAL is not installed : cannot generate synthetic LAZ")

    d = tmp_path_factory.mktemp("synthetic_test_data")

    _create_synthetic_laz(
        str(d / "compare.laz"),
        n_points=50_000,
        z_shift=0.0,
        gps_time_base=7.9e8,
        seed=42,
    )

    _create_synthetic_laz(
        str(d / "reference.laz"),
        n_points=50_000,
        z_shift=0.5,
        gps_time_base=1.2e9,
        seed=42,
    )

    return d


@pytest.fixture(scope="session")
def compare_laz_path(synthetic_test_data_dir):
    """Path to the synthetic compare.laz file."""
    return str(synthetic_test_data_dir / "compare.laz")


@pytest.fixture(scope="session")
def reference_laz_path(synthetic_test_data_dir):
    """Path to the synthetic reference.laz file."""
    return str(synthetic_test_data_dir / "reference.laz")


# PointCloud fixtures (require PDAL)

@pytest.fixture
def compare_pc(compare_laz_path):
    """Loaded PointCloud for the synthetic compare cloud."""
    if not HAS_PDAL:
        pytest.skip("PDAL is not installed")
    from topochange import PointCloud
    pc = PointCloud(compare_laz_path)
    pc.from_file()
    return pc


@pytest.fixture
def reference_pc(reference_laz_path):
    """Loaded PointCloud for the synthetic reference cloud."""
    if not HAS_PDAL:
        pytest.skip("PDAL is not installed")
    from topochange import PointCloud
    pc = PointCloud(reference_laz_path)
    pc.from_file()
    return pc


@pytest.fixture
def pc_pair(compare_pc, reference_pc):
    """PointCloudPair built from the two synthetic clouds."""
    from topochange import PointCloudPair
    return PointCloudPair(compare_pc, reference_pc)

