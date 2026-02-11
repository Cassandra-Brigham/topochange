"""Comprehensive test suite for Raster and RasterPair classes.

Tests `topochange.raster.Raster` and `topochange.rasterpair.RasterPair`
with synthetic GeoTIFF fixtures."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

import rasterio
from rasterio.transform import from_bounds
from pyproj import CRS

from topochange.raster import Raster
from topochange.rasterpair import (
    RasterPair,
    _crs_equivalent,
    _geoid_equivalent,
    _units_equivalent,
    _create_valid_data_mask,
)


# synthetic GeoTIFF Fixture Functions

def create_synthetic_raster(
    filepath,
    data,
    bounds,
    crs="EPSG:32610",
    nodata=-9999.0,
):
    """
    Create a synthetic GeoTIFF for testing.

    Parameters
    ----------
    filepath : str
        Output file path
    data : np.ndarray
        2D array of raster data
    bounds : tuple
        (left, bottom, right, top) geographic bounds
    crs : str
        CRS string (EPSG code or WKT)
    nodata : float
        Nodata value for the raster
    """
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


# pytest Fixtures

@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for test files."""
    return str(tmp_path)


@pytest.fixture
def synthetic_dem_path(tmp_dir):
    """Create a synthetic DEM in UTM zone 10N."""
    # 100x100 elevation surface with mean ~1000m
    data = np.random.RandomState(42).randn(100, 100) * 10 + 1000
    data = data.astype(np.float32)

    # add some nodata pixels in corners
    data[0:5, 0:5] = -9999.0
    data[-5:, -5:] = -9999.0

    # bounds in UTM zone 10N (meters)
    bounds = (500000, 4000000, 500500, 4000500)

    filepath = os.path.join(tmp_dir, "dem1.tif")
    create_synthetic_raster(
        filepath,
        data,
        bounds,
        crs="EPSG:32610",  # UTM zone 10N
        nodata=-9999.0,
    )

    return filepath


@pytest.fixture
def synthetic_dem_path_2(tmp_dir):
    """Create a second synthetic DEM for differencing (slightly different data)."""
    # same data structure as DEM1, but with small changes
    data = np.random.RandomState(42).randn(100, 100) * 10 + 1000
    data = data + np.random.RandomState(43).randn(100, 100) * 0.5  # Small changes
    data = data.astype(np.float32)

    # add some nodata pixels in corners
    data[0:5, 0:5] = -9999.0
    data[-5:, -5:] = -9999.0

    # same bounds as DEM1
    bounds = (500000, 4000000, 500500, 4000500)

    filepath = os.path.join(tmp_dir, "dem2.tif")
    create_synthetic_raster(
        filepath,
        data,
        bounds,
        crs="EPSG:32610",
        nodata=-9999.0,
    )

    return filepath


@pytest.fixture
def synthetic_dem_feet(tmp_dir):
    """Create a synthetic DEM in US survey feet units."""
    data = np.random.RandomState(44).randn(80, 80) * 100 + 3000  # elevation in feet
    data = data.astype(np.float32)

    # add nodata pixels
    data[0:4, 0:4] = -9999.0

    # bounds in US feet (approximate)
    bounds = (1640419, 13123360, 1641959, 13124900)

    filepath = os.path.join(tmp_dir, "dem_feet.tif")
    create_synthetic_raster(
        filepath,
        data,
        bounds,
        crs="EPSG:2230",  # CA State Plane Zone 10 (US Feet)
        nodata=-9999.0,
    )

    return filepath


@pytest.fixture
def synthetic_dem_geographic(tmp_dir):
    """Create a synthetic DEM in geographic coordinates (EPSG:4326)."""
    data = np.random.RandomState(45).randn(60, 60) * 50 + 1500
    data = data.astype(np.float32)

    # add nodata pixels
    data[0:3, 0:3] = -9999.0

    # bounds in geographic degrees (roughly California)
    bounds = (-120.5, 37.5, -120.0, 38.0)

    filepath = os.path.join(tmp_dir, "dem_geographic.tif")
    create_synthetic_raster(
        filepath,
        data,
        bounds,
        crs="EPSG:4326",
        nodata=-9999.0,
    )

    return filepath


# tests for Raster class

class TestRasterCreation:
    """Tests for Raster creation and loading."""

    def test_from_file_basic(self, synthetic_dem_path):
        """Test Raster.from_file returns Raster with correct filename."""
        raster = Raster.from_file(synthetic_dem_path)

        assert isinstance(raster, Raster)
        assert raster.filename == synthetic_dem_path

    def test_from_file_crs_detected(self, synthetic_dem_path):
        """Test that CRS is detected and set."""
        raster = Raster.from_file(synthetic_dem_path)

        assert raster.crs is not None
        assert raster.current_horizontal_crs is not None
        # UTM zone 10N should be EPSG:32610
        assert "32610" in str(raster.crs)

    def test_from_file_bounds(self, synthetic_dem_path):
        """Test that bounds are correctly read."""
        raster = Raster.from_file(synthetic_dem_path)

        assert raster.bounds is not None
        # check bounds are reasonable (should match our fixture bounds)
        assert raster.bounds.left == 500000
        assert raster.bounds.bottom == 4000000
        assert raster.bounds.right == 500500
        assert raster.bounds.top == 4000500

    def test_from_file_shape(self, synthetic_dem_path):
        """Test that shape matches synthetic data dimensions."""
        raster = Raster.from_file(synthetic_dem_path)

        assert raster.height == 100
        assert raster.width == 100

    def test_from_file_resolution(self, synthetic_dem_path):
        """Test that resolution is correctly calculated."""
        raster = Raster.from_file(synthetic_dem_path)

        # bounds: 500000 to 500500 (500 units) over 100 pixels = 5 units/pixel
        assert raster.resolution is not None
        assert abs(raster.resolution - 5.0) < 0.01


class TestRasterDataAccess:
    """Tests for accessing raster data."""

    def test_data_property_returns_array(self, synthetic_dem_path):
        """Test that .data property returns array-like object."""
        raster = Raster.from_file(synthetic_dem_path)

        # data property returns rioxarray DataArray
        data = raster.data
        assert hasattr(data, "values")
        assert hasattr(data, "shape")

    def test_data_values_match(self, synthetic_dem_path):
        """Test that loaded data matches synthetic input."""
        # create reference data
        ref_data = np.random.RandomState(42).randn(100, 100) * 10 + 1000
        ref_data[0:5, 0:5] = -9999.0
        ref_data[-5:, -5:] = -9999.0
        ref_data = ref_data.astype(np.float32)

        # load Raster
        raster = Raster.from_file(synthetic_dem_path)
        loaded_data = raster.data.values

        # check that data matches (allowing for float precision)
        assert loaded_data.shape == ref_data.shape
        # compare non-nodata regions
        valid = ref_data != -9999.0
        assert np.allclose(loaded_data[valid], ref_data[valid], atol=0.01)


class TestRasterUnitConversion:
    """Tests for unit conversion methods."""

    def test_set_units(self, synthetic_dem_path):
        """Test set_units method."""
        raster = Raster.from_file(synthetic_dem_path)

        # should not raise
        raster.set_units(horizontal_unit="meter", vertical_unit="meter")

    def test_convert_values_to_meters_with_meter_units(self, synthetic_dem_path):
        """Test that meter values remain unchanged when converting to meters."""
        raster = Raster.from_file(synthetic_dem_path)
        raster.set_units(vertical_unit="meter")

        values = np.array([100.0, 200.0, 300.0])
        converted = raster.convert_values_to_meters(values)

        assert np.allclose(converted, values)

    def test_convert_values_to_meters_with_foot_units(self, synthetic_dem_path):
        """Test that foot values are scaled correctly."""
        raster = Raster.from_file(synthetic_dem_path)
        raster.set_units(vertical_unit="foot")

        values = np.array([100.0, 200.0, 300.0])  # in feet
        converted = raster.convert_values_to_meters(values)

        # 1 foot = 0.3048 meters
        expected = values * 0.3048
        assert np.allclose(converted, expected, rtol=0.01)

    def test_convert_values_roundtrip(self, synthetic_dem_path):
        """Test roundtrip conversion: to_meters then from_meters."""
        raster = Raster.from_file(synthetic_dem_path)
        raster.set_units(vertical_unit="foot")

        original = np.array([100.0, 200.0, 300.0])

        # convert to meters
        in_meters = raster.convert_values_to_meters(original)

        # convert back from meters
        back_to_original = raster.convert_values_from_meters(in_meters)

        # should match original
        assert np.allclose(back_to_original, original, rtol=0.01)

    def test_are_units_metric(self, synthetic_dem_path):
        """Test are_units_metric method."""
        raster = Raster.from_file(synthetic_dem_path)

        # UTM zone 10N uses meters for horizontal, but vertical may not be defined
        is_metric_horiz, is_metric_vert = raster.are_units_metric()

        # horizontal should be metric (UTM uses meters)
        assert is_metric_horiz is True
        # vertical may not be metric or may be undefined
        assert isinstance(is_metric_vert, (bool, type(None)))


class TestRasterMetadata:
    """Tests for metadata methods."""

    def test_add_metadata_epoch(self, synthetic_dem_path):
        """Test add_metadata with epoch parameter."""
        raster = Raster.from_file(synthetic_dem_path)

        epoch_value = 2020.5
        raster.add_metadata(epoch=epoch_value)

        assert raster.epoch == epoch_value

    def test_add_metadata_crs(self, synthetic_dem_path):
        """Test add_metadata updates CRS."""
        raster = Raster.from_file(synthetic_dem_path)

        # update with a different horizontal CRS
        new_crs = CRS.from_epsg(32611)  # UTM zone 11N
        raster.add_metadata(horizontal_CRS=new_crs)

        assert raster.current_horizontal_crs is not None


class TestRasterCRSProperties:
    """Tests for CRS properties and tracking."""

    def test_original_crs_preserved(self, synthetic_dem_path):
        """Test that original CRS is preserved after loading."""
        raster = Raster.from_file(synthetic_dem_path)

        assert raster.original_horizontal_crs is not None
        assert "32610" in str(raster.crs)

    def test_current_crs_initially_matches_original(self, synthetic_dem_path):
        """Test that current CRS matches original CRS on load."""
        raster = Raster.from_file(synthetic_dem_path)

        assert raster.current_horizontal_crs == raster.original_horizontal_crs


# tests for utility functions

class TestCRSEquivalence:
    """Tests for _crs_equivalent function."""

    def test_crs_equivalent_same(self):
        """Test that identical CRS are equivalent."""
        crs = CRS.from_epsg(32610)

        assert _crs_equivalent(crs, crs) is True

    def test_crs_equivalent_different(self):
        """Test that different CRS are not equivalent."""
        crs1 = CRS.from_epsg(32610)
        crs2 = CRS.from_epsg(4326)

        assert _crs_equivalent(crs1, crs2) is False

    def test_crs_equivalent_none_both(self):
        """Test that both None CRS are equivalent."""
        assert _crs_equivalent(None, None) is True

    def test_crs_equivalent_none_one(self):
        """Test that None CRS is not equivalent to other CRS."""
        crs = CRS.from_epsg(32610)

        assert _crs_equivalent(None, crs) is False
        assert _crs_equivalent(crs, None) is False


class TestGeoidEquivalence:
    """Tests for _geoid_equivalent function."""

    def test_geoid_equivalent_same(self):
        """Test that identical geoids are equivalent."""
        assert _geoid_equivalent("geoid12B", "geoid12B") is True

    def test_geoid_equivalent_case_insensitive(self):
        """Test that geoid comparison is case-insensitive."""
        assert _geoid_equivalent("geoid12B", "GEOID12B") is True
        assert _geoid_equivalent("geoid12b", "geoid12B") is True

    def test_geoid_equivalent_different(self):
        """Test that different geoids are not equivalent."""
        assert _geoid_equivalent("geoid12B", "EGM96") is False

    def test_geoid_equivalent_with_prefix(self):
        """Test normalization of geoid with prefixes."""
        # both should normalize to "geoid12b"
        assert _geoid_equivalent("us_noaa_geoid12b", "geoid12B") is True
        assert _geoid_equivalent("noaa_geoid12b", "geoid12b") is True

    def test_geoid_equivalent_none(self):
        """Test None geoid equivalence."""
        assert _geoid_equivalent(None, None) is True
        assert _geoid_equivalent(None, "geoid12B") is False
        assert _geoid_equivalent("geoid12B", None) is False


class TestUnitsEquivalent:
    """Tests for _units_equivalent function."""

    def test_units_equivalent_same(self):
        """Test that identical units are equivalent."""
        are_equiv, factor = _units_equivalent("meter", "meter")

        assert are_equiv is True
        assert factor is None

    def test_units_equivalent_different(self):
        """Test that different units have conversion factor."""
        are_equiv, factor = _units_equivalent("meter", "foot")

        assert are_equiv is False
        assert factor is not None
        # 1 meter = 3.28084 feet, so meter -> foot factor is ~3.28
        assert abs(factor - 3.28084) < 0.01

    def test_units_equivalent_none_both(self):
        """Test that both None units are equivalent."""
        are_equiv, factor = _units_equivalent(None, None)

        assert are_equiv is True
        assert factor is None

    def test_units_equivalent_none_one(self):
        """Test that None unit is not equivalent to other unit."""
        are_equiv, factor = _units_equivalent(None, "meter")

        assert are_equiv is False


class TestCreateValidDataMask:
    """Tests for _create_valid_data_mask function."""

    def test_mask_basic_all_valid(self):
        """Test mask with all valid data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        nodata = -9999.0

        mask = _create_valid_data_mask(data, nodata)

        assert mask.all()  # All should be True

    def test_mask_nodata_values(self):
        """Test that nodata values are masked."""
        data = np.array([[1.0, -9999.0], [3.0, 4.0]])
        nodata = -9999.0

        mask = _create_valid_data_mask(data, nodata)

        assert mask[0, 0]
        assert not mask[0, 1]
        assert mask[1, 0]
        assert mask[1, 1]

    def test_mask_nan_values(self):
        """Test that NaN values are masked."""
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        nodata = None

        mask = _create_valid_data_mask(data, nodata)

        assert mask[0, 0]
        assert not mask[0, 1]
        assert mask[1, 0]
        assert mask[1, 1]

    def test_mask_inf_values(self):
        """Test that Inf values are masked."""
        data = np.array([[1.0, np.inf], [3.0, -np.inf]])
        nodata = None

        mask = _create_valid_data_mask(data, nodata)

        assert mask[0, 0]
        assert not mask[0, 1]
        assert mask[1, 0]
        assert not mask[1, 1]


# tests for RasterPair class

class TestRasterPairCreation:
    """Tests for RasterPair creation."""

    def test_create_rasterpair(self, synthetic_dem_path, synthetic_dem_path_2):
        """Test RasterPair creation."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)

        assert isinstance(pair, RasterPair)

    def test_rasterpair_attributes(self, synthetic_dem_path, synthetic_dem_path_2):
        """Test RasterPair has raster1 and raster2 attributes."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)

        assert hasattr(pair, "raster1")
        assert hasattr(pair, "raster2")
        assert pair.raster1 is raster1
        assert pair.raster2 is raster2


class TestRasterPairComparisons:
    """Tests for RasterPair comparison methods."""

    def test_check_horizontal_crs_match_same(self, synthetic_dem_path, synthetic_dem_path_2):
        """Test CRS match check for same CRS."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)
        result = pair.check_horizontal_crs_match()

        assert result["match"] is True

    def test_check_horizontal_crs_match_different(self, synthetic_dem_path, synthetic_dem_feet):
        """Test CRS match check for different CRS."""
        raster1 = Raster.from_file(synthetic_dem_path)  # UTM zone 10N
        raster2 = Raster.from_file(synthetic_dem_feet)  # CA State Plane (feet)

        pair = RasterPair(raster1, raster2)
        result = pair.check_horizontal_crs_match()

        assert result["match"] is False

    def test_check_compound_crs_match(self, synthetic_dem_path, synthetic_dem_path_2):
        """Test compound CRS match check."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)
        result = pair.check_compound_crs_match()

        assert "match" in result
        assert isinstance(result["match"], bool)


class TestRasterPairDifferencing:
    """Tests for DEM differencing functionality."""

    def test_compute_difference_returns_dict(self, synthetic_dem_path, synthetic_dem_path_2, tmp_dir):
        """Test that compute_difference returns dict."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)
        result = pair.compute_difference(
            transform_first=False,  # Skip transformation for simplicity
            verbose=False,
        )

        assert isinstance(result, dict)

    def test_compute_difference_has_required_keys(self, synthetic_dem_path, synthetic_dem_path_2, tmp_dir):
        """Test that compute_difference result has required keys."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)
        result = pair.compute_difference(
            transform_first=False,
            verbose=False,
        )

        assert "stats" in result
        assert "difference_raster" in result or "difference_raster_path" in result

    def test_compute_difference_stats_keys(self, synthetic_dem_path, synthetic_dem_path_2, tmp_dir):
        """Test that stats have expected keys."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)
        result = pair.compute_difference(
            transform_first=False,
            verbose=False,
        )

        stats = result.get("stats", {})

        # check for expected statistical keys
        expected_keys = {"mean", "std", "min", "max"}
        for key in expected_keys:
            assert key in stats, f"Expected key '{key}' in stats"

    def test_compute_difference_symmetry(self, synthetic_dem_path, synthetic_dem_path_2):
        """Test that mean difference is small (DEMs are close)."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)
        result = pair.compute_difference(
            transform_first=False,
            verbose=False,
        )

        stats = result.get("stats", {})
        mean_diff = stats.get("mean", 0)

        # difference should be small since DEMs are similar (within 0.5 * sigma)
        assert abs(mean_diff) < 1.0, f"Mean difference too large: {mean_diff}"


class TestRasterPairProvenance:
    """Tests for provenance tracking."""

    def test_get_transformation_history(self, synthetic_dem_path, synthetic_dem_path_2):
        """Test get_transformation_history returns list."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)
        history = pair.get_transformation_history()

        assert isinstance(history, list)

    def test_get_full_provenance_returns_dict(self, synthetic_dem_path, synthetic_dem_path_2):
        """Test get_full_provenance returns dict."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)
        provenance = pair.get_full_provenance()

        assert isinstance(provenance, dict)


class TestRasterPairExtent:
    """Tests for extent and overlap methods."""

    def test_get_overlap_polygon(self, synthetic_dem_path, synthetic_dem_path_2):
        """Test get_overlap_polygon returns dict with overlap info."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)
        result = pair.get_overlap_polygon()

        assert isinstance(result, dict)
        assert "has_overlap" in result
        assert result["has_overlap"] is True  # Same bounds, so should overlap


# integration tests

class TestRasterIntegration:
    """Integration tests combining multiple Raster methods."""

    def test_load_transform_metadata(self, synthetic_dem_path):
        """Test loading, transforming metadata, and querying."""
        # load Raster
        raster = Raster.from_file(synthetic_dem_path)

        # add metadata
        raster.add_metadata(epoch=2020.5)

        # check various properties
        assert raster.filename == synthetic_dem_path
        assert raster.epoch == 2020.5
        assert raster.width == 100
        assert raster.height == 100
        assert raster.bounds is not None


class TestRasterPairIntegration:
    """Integration tests for RasterPair workflows."""

    def test_compare_transform_difference(self, synthetic_dem_path, synthetic_dem_path_2):
        """Test complete workflow: compare, transform, difference."""
        raster1 = Raster.from_file(synthetic_dem_path)
        raster2 = Raster.from_file(synthetic_dem_path_2)

        pair = RasterPair(raster1, raster2)

        # step 1: Compare
        crs_check = pair.check_horizontal_crs_match()
        assert crs_check["match"] is True

        # step 2: Compute difference
        diff_result = pair.compute_difference(
            transform_first=False,
            verbose=False,
        )

        # step 3: Check results
        assert "stats" in diff_result
        assert diff_result["stats"]["count_valid"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

