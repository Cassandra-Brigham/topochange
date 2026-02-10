"""Tests for PDAL pipeline construction in data_access module.

Verifies that writer stages include proper ``a_srs`` CRS parameters
when ``output_crs_wkt`` is provided.

These tests use unittest.mock to patch heavy imports (osgeo/gdal)
so that the test can run without GDAL installed.
"""
import sys
import types
import pytest
from unittest.mock import MagicMock
from shapely.geometry import box

# Patch osgeo/gdal before importing data_access (they are top-level imports)
_osgeo_mock = types.ModuleType("osgeo")
_osgeo_mock.gdal = MagicMock()
sys.modules.setdefault("osgeo", _osgeo_mock)
sys.modules.setdefault("osgeo.gdal", _osgeo_mock.gdal)

from topochange.data_access import GetDEMs  # noqa: E402


# A representative WKT2 compound CRS string for testing
_SAMPLE_CRS_WKT = (
    'COMPOUNDCRS["WGS 84 / UTM zone 13N + NAVD88 height",'
    'PROJCRS["WGS 84 / UTM zone 13N",'
    'BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563]]],'
    'CONVERSION["UTM zone 13N",METHOD["Transverse Mercator"],'
    'PARAMETER["Latitude of natural origin",0],'
    'PARAMETER["Longitude of natural origin",-105],'
    'PARAMETER["Scale factor at natural origin",0.9996],'
    'PARAMETER["False easting",500000],'
    'PARAMETER["False northing",0]],'
    'CS[Cartesian,2],AXIS["easting",east],AXIS["northing",north],'
    'UNIT["metre",1]],'
    'VERTCRS["NAVD88 height",VDATUM["North American Vertical Datum 1988"],'
    'CS[vertical,1],AXIS["gravity-related height",up],'
    'UNIT["metre",1]]]'
)

# A simple bounding polygon in EPSG:3857
_EXTENT_3857 = box(-11700000, 4500000, -11690000, 4510000)


# ==============================================================================
# _writer_las tests
# ==============================================================================

class TestWriterLasAsrs:
    """Verify _writer_las() includes a_srs when provided."""

    def test_without_a_srs(self):
        """Default call has no a_srs key."""
        w = GetDEMs._writer_las("test", "laz")
        assert "a_srs" not in w
        assert w["type"] == "writers.las"
        assert w["compression"] == "laszip"

    def test_with_a_srs(self):
        """Passing a_srs includes it in the dict."""
        w = GetDEMs._writer_las("test", "laz", a_srs=_SAMPLE_CRS_WKT)
        assert w["a_srs"] == _SAMPLE_CRS_WKT

    def test_las_no_compression(self):
        """LAS (uncompressed) has no compression key."""
        w = GetDEMs._writer_las("test", "las")
        assert "compression" not in w

    def test_invalid_ext_raises(self):
        """Invalid extension raises ValueError."""
        with pytest.raises(ValueError):
            GetDEMs._writer_las("test", "xyz")


# ==============================================================================
# build_pdal_pipeline_from_file tests
# ==============================================================================

class TestBuildPipelineFromFileAsrs:
    """Verify build_pdal_pipeline_from_file passes a_srs to the writer stage."""

    def test_writer_has_a_srs_when_provided(self):
        """Writer stage includes a_srs when output_crs_wkt is set."""
        pipeline = GetDEMs.build_pdal_pipeline_from_file(
            filename="dummy.laz",
            extent=_EXTENT_3857,
            filterNoise=False,
            reclassify=False,
            savePointCloud=True,
            outCRS="EPSG:32613",
            pc_outName="test_out",
            pc_outType="laz",
            output_crs_wkt=_SAMPLE_CRS_WKT,
        )
        # pipeline is a list of stage dicts
        writer_stages = [s for s in pipeline if s.get("type") == "writers.las"]
        assert len(writer_stages) == 1
        assert writer_stages[0]["a_srs"] == _SAMPLE_CRS_WKT

    def test_writer_no_a_srs_when_not_provided(self):
        """Writer stage omits a_srs by default."""
        pipeline = GetDEMs.build_pdal_pipeline_from_file(
            filename="dummy.laz",
            extent=_EXTENT_3857,
            savePointCloud=True,
            outCRS="EPSG:32613",
        )
        writer_stages = [s for s in pipeline if s.get("type") == "writers.las"]
        assert len(writer_stages) == 1
        assert "a_srs" not in writer_stages[0]

    def test_no_writer_when_save_false(self):
        """No writer stage when savePointCloud=False."""
        pipeline = GetDEMs.build_pdal_pipeline_from_file(
            filename="dummy.laz",
            extent=_EXTENT_3857,
            savePointCloud=False,
            outCRS="EPSG:32613",
        )
        writer_stages = [s for s in pipeline if s.get("type") == "writers.las"]
        assert len(writer_stages) == 0


# ==============================================================================
# build_aws_pdal_pipeline tests
# ==============================================================================

class TestBuildAwsPipelineAsrs:
    """Verify build_aws_pdal_pipeline passes a_srs to the writer stage."""

    def test_writer_has_a_srs_when_provided(self):
        """Writer stage includes a_srs when output_crs_wkt is set."""
        pipeline = GetDEMs.build_aws_pdal_pipeline(
            extent_epsg3857=_EXTENT_3857,
            property_ids=["CO_SanLuisJuanMiguel_1_2020"],
            pc_resolution=10.0,
            data_source="usgs",
            savePointCloud=True,
            outCRS="EPSG:32613",
            pc_outType="laz",
            output_crs_wkt=_SAMPLE_CRS_WKT,
        )
        writer_stages = [
            s for s in pipeline["pipeline"] if s.get("type") == "writers.las"
        ]
        assert len(writer_stages) == 1
        assert writer_stages[0]["a_srs"] == _SAMPLE_CRS_WKT

    def test_writer_no_a_srs_by_default(self):
        """Writer stage omits a_srs when output_crs_wkt is None."""
        pipeline = GetDEMs.build_aws_pdal_pipeline(
            extent_epsg3857=_EXTENT_3857,
            property_ids=["CO_SanLuisJuanMiguel_1_2020"],
            pc_resolution=10.0,
            data_source="usgs",
            savePointCloud=True,
            outCRS="EPSG:32613",
            pc_outType="laz",
        )
        writer_stages = [
            s for s in pipeline["pipeline"] if s.get("type") == "writers.las"
        ]
        assert len(writer_stages) == 1
        assert "a_srs" not in writer_stages[0]
