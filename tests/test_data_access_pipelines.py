"""Tests for PDAL pipeline construction in data_access module.

Verifies that writer stages include proper ``a_srs`` CRS parameters
when ``output_crs_wkt`` is provided.

These tests use unittest.mock to patch heavy imports (osgeo/gdal)
so that the test can run without GDAL installed."""
import sys
import types
import pytest
from unittest.mock import MagicMock
from shapely.geometry import box

# patch osgeo/gdal before importing data_access (they are top-level imports)
_osgeo_mock = types.ModuleType("osgeo")
_osgeo_mock.gdal = MagicMock()
sys.modules.setdefault("osgeo", _osgeo_mock)
sys.modules.setdefault("osgeo.gdal", _osgeo_mock.gdal)

from topochange.data_access import GetDEMs  # noqa: E402


# a representative WKT2 compound CRS string for testing
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

# a simple bounding Polygon in EPSG:3857
_EXTENT_3857 = box(-11700000, 4500000, -11690000, 4510000)


# _writer_las tests

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


# build_pdal_pipeline_from_file tests

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
        # Pipeline is a list of stage dicts
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


# build_aws_pdal_pipeline tests

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


# _writer_gdal override_srs tests

class TestWriterGdalOverrideSrs:
    """Verify _writer_gdal() includes override_srs when provided."""

    def test_without_override_srs(self):
        """Default call has no override_srs key."""
        w = GetDEMs._writer_gdal("test.tif")
        assert "override_srs" not in w
        assert w["type"] == "writers.gdal"

    def test_with_override_srs(self):
        """Providing override_srs includes it in the dict."""
        w = GetDEMs._writer_gdal("test.tif", override_srs=_SAMPLE_CRS_WKT)
        assert w["override_srs"] == _SAMPLE_CRS_WKT

    def test_override_srs_none_omitted(self):
        """Explicitly passing None omits override_srs."""
        w = GetDEMs._writer_gdal("test.tif", override_srs=None)
        assert "override_srs" not in w


class TestMakeDemPipelineGdalSrs:
    """Verify make_DEM_pipeline_from_file writers.gdal uses WKT2 CRS."""

    @staticmethod
    def _make_instance():
        """Create a bare GetDEMs instance (no __init__ side effects)."""
        obj = object.__new__(GetDEMs)
        return obj

    def test_dsm_uses_output_crs_wkt(self):
        """DSM writers.gdal override_srs should use output_crs_wkt when provided."""
        inst = self._make_instance()
        pipeline = inst.make_DEM_pipeline_from_file(
            filename="test.laz",
            extent=_EXTENT_3857,
            dem_resolution=1.0,
            outCRS="EPSG:32613",
            demType="dsm",
            gridMethod="max",
            dem_outName="test_dsm",
            dem_outExt="tif",
            output_crs_wkt=_SAMPLE_CRS_WKT,
        )
        gdal_stages = [
            s for s in pipeline["pipeline"] if s.get("type") == "writers.gdal"
        ]
        assert len(gdal_stages) == 1
        assert gdal_stages[0]["override_srs"] == _SAMPLE_CRS_WKT

    def test_dtm_uses_output_crs_wkt(self):
        """DTM writers.gdal override_srs should use output_crs_wkt when provided."""
        inst = self._make_instance()
        pipeline = inst.make_DEM_pipeline_from_file(
            filename="test.laz",
            extent=_EXTENT_3857,
            dem_resolution=1.0,
            outCRS="EPSG:32613",
            demType="dtm",
            gridMethod="idw",
            dem_outName="test_dtm",
            dem_outExt="tif",
            output_crs_wkt=_SAMPLE_CRS_WKT,
        )
        gdal_stages = [
            s for s in pipeline["pipeline"] if s.get("type") == "writers.gdal"
        ]
        assert len(gdal_stages) == 1
        assert gdal_stages[0]["override_srs"] == _SAMPLE_CRS_WKT

    def test_falls_back_to_outcrs_when_no_wkt(self):
        """writers.gdal should fall back to outCRS if output_crs_wkt is None."""
        inst = self._make_instance()
        pipeline = inst.make_DEM_pipeline_from_file(
            filename="test.laz",
            extent=_EXTENT_3857,
            dem_resolution=1.0,
            outCRS="EPSG:32613",
            demType="dsm",
            gridMethod="max",
            dem_outName="test_dsm",
            dem_outExt="tif",
            output_crs_wkt=None,
        )
        gdal_stages = [
            s for s in pipeline["pipeline"] if s.get("type") == "writers.gdal"
        ]
        assert len(gdal_stages) == 1
        assert gdal_stages[0]["override_srs"] == "EPSG:32613"


class TestMakeDemPipelineAwsGdalSrs:
    """Verify make_DEM_pipeline_aws writers.gdal uses WKT2 CRS."""

    @staticmethod
    def _make_instance():
        """Create a bare GetDEMs instance (no __init__ side effects)."""
        obj = object.__new__(GetDEMs)
        return obj

    def test_dsm_uses_output_crs_wkt(self):
        """AWS DSM writers.gdal override_srs should use output_crs_wkt."""
        inst = self._make_instance()
        pipeline = inst.make_DEM_pipeline_aws(
            extent_epsg3857=_EXTENT_3857,
            property_ids=["CO_SanLuisJuanMiguel_1_2020"],
            pc_resolution=10.0,
            dem_resolution=1.0,
            data_source="usgs",
            outCRS="EPSG:32613",
            demType="dsm",
            gridMethod="max",
            dem_outName="test_dsm",
            dem_outExt="tif",
            output_crs_wkt=_SAMPLE_CRS_WKT,
        )
        gdal_stages = [
            s for s in pipeline["pipeline"] if s.get("type") == "writers.gdal"
        ]
        assert len(gdal_stages) == 1
        assert gdal_stages[0]["override_srs"] == _SAMPLE_CRS_WKT

    def test_falls_back_to_outcrs(self):
        """AWS DSM should fall back to outCRS when output_crs_wkt is None."""
        inst = self._make_instance()
        pipeline = inst.make_DEM_pipeline_aws(
            extent_epsg3857=_EXTENT_3857,
            property_ids=["CO_SanLuisJuanMiguel_1_2020"],
            pc_resolution=10.0,
            dem_resolution=1.0,
            data_source="usgs",
            outCRS="EPSG:32613",
            demType="dsm",
            gridMethod="max",
            dem_outName="test_dsm",
            dem_outExt="tif",
            output_crs_wkt=None,
        )
        gdal_stages = [
            s for s in pipeline["pipeline"] if s.get("type") == "writers.gdal"
        ]
        assert len(gdal_stages) == 1
        assert gdal_stages[0]["override_srs"] == "EPSG:32613"

