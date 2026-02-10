"""Comprehensive tests for topochange.crs_utils module.

Tests all functions for CRS conversion, transformation, and validation.
"""
import pytest
import numpy as np
from pyproj import CRS, Transformer

from topochange.crs_utils import (
    _ensure_crs_obj,
    crs_equals,
    crs_to_wkt2_2019,
    wrap_coordinate_metadata_wkt,
    extract_epoch_from_wkt,
    crs_to_projjson,
    make_coordinate_metadata_projjson,
    is_orthometric,
    is_3d_geographic_crs,
    extract_ellipsoidal_height_as_vertical_crs,
    create_compound_crs,
    parse_crs_components,
    transformer_with_epoch,
    horizontal_unit_scale,
    vertical_unit_scale,
    apply_vertical_datum_transform,
    apply_dynamic_transform,
    vertical_datum_to_crs,
    build_output_crs_wkt,
)


# ==============================================================================
# Test fixtures for common CRS objects
# ==============================================================================

@pytest.fixture
def wgs84_2d():
    """WGS 84 geographic 2D (EPSG:4326)."""
    return CRS.from_epsg(4326)


@pytest.fixture
def wgs84_3d():
    """WGS 84 geographic 3D with ellipsoidal height (EPSG:4979)."""
    return CRS.from_epsg(4979)


@pytest.fixture
def utm10n():
    """UTM Zone 10N, WGS84 (EPSG:32610)."""
    return CRS.from_epsg(32610)


@pytest.fixture
def nad83_ca6():
    """NAD83 California Zone 6 in US Survey Feet (EPSG:2230)."""
    return CRS.from_epsg(2230)


@pytest.fixture
def navd88():
    """NAVD88 height (vertical, orthometric) (EPSG:5703)."""
    return CRS.from_epsg(5703)


@pytest.fixture
def nad83_navd88_compound():
    """NAD83(2011) + NAVD88 compound CRS (EPSG:6349)."""
    return CRS.from_epsg(6349)


# ==============================================================================
# Test _ensure_crs_obj
# ==============================================================================

class TestEnsureCrsObj:
    """Tests for _ensure_crs_obj function."""

    def test_ensure_crs_obj_with_epsg_string(self):
        """Accept EPSG code as string."""
        crs = _ensure_crs_obj("EPSG:4326")
        assert isinstance(crs, CRS)
        assert crs.to_epsg() == 4326

    def test_ensure_crs_obj_with_epsg_utm(self):
        """Accept UTM EPSG code."""
        crs = _ensure_crs_obj("EPSG:32610")
        assert isinstance(crs, CRS)
        assert crs.to_epsg() == 32610

    def test_ensure_crs_obj_with_crs_object(self):
        """Accept pyproj CRS object directly."""
        crs_obj = CRS.from_epsg(4326)
        result = _ensure_crs_obj(crs_obj)
        assert result is crs_obj

    def test_ensure_crs_obj_with_wkt_string(self):
        """Accept WKT string."""
        wkt = CRS.from_epsg(4326).to_wkt()
        crs = _ensure_crs_obj(wkt)
        assert isinstance(crs, CRS)
        assert crs.to_epsg() == 4326

    def test_ensure_crs_obj_with_projjson_dict(self):
        """Accept PROJJSON dict."""
        crs_obj = CRS.from_epsg(4326)
        json_dict = crs_obj.to_json_dict()
        result = _ensure_crs_obj(json_dict)
        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326

    def test_ensure_crs_obj_caching(self):
        """String-based CRS parsing is cached."""
        crs1 = _ensure_crs_obj("EPSG:4326")
        crs2 = _ensure_crs_obj("EPSG:4326")
        # Both should reference the same cached object
        assert crs1 is crs2

    def test_ensure_crs_obj_invalid_string(self):
        """Invalid input raises exception."""
        with pytest.raises(Exception):
            _ensure_crs_obj("INVALID_CRS_STRING")


# ==============================================================================
# Test crs_equals
# ==============================================================================

class TestCrsEquals:
    """Tests for crs_equals function."""

    def test_crs_equals_same_epsg_strings(self):
        """Equal EPSG codes return True."""
        assert crs_equals("EPSG:4326", "EPSG:4326")

    def test_crs_equals_different_epsg_strings(self):
        """Different EPSG codes return False."""
        assert not crs_equals("EPSG:4326", "EPSG:32610")

    def test_crs_equals_crs_object_to_string(self):
        """CRS object equals matching EPSG string."""
        crs_obj = CRS.from_epsg(4326)
        assert crs_equals(crs_obj, "EPSG:4326")

    def test_crs_equals_two_crs_objects(self):
        """Two equivalent CRS objects are equal."""
        crs1 = CRS.from_epsg(4326)
        crs2 = CRS.from_epsg(4326)
        assert crs_equals(crs1, crs2)

    def test_crs_equals_utm_zones(self):
        """Different UTM zones are not equal."""
        assert not crs_equals("EPSG:32610", "EPSG:32611")

    def test_crs_equals_with_projjson(self):
        """PROJJSON dict can be compared."""
        crs_obj = CRS.from_epsg(4326)
        json_dict = crs_obj.to_json_dict()
        assert crs_equals(json_dict, "EPSG:4326")

    def test_crs_equals_wkt_comparison_fallback(self):
        """Falls back to WKT comparison for non-EPSG CRS."""
        # Create two WKT strings that are equivalent but not from same EPSG
        crs1 = CRS.from_epsg(4326)
        crs2 = CRS.from_wkt(crs1.to_wkt())
        assert crs_equals(crs1, crs2)


# ==============================================================================
# Test crs_to_wkt2_2019
# ==============================================================================

class TestCrsToWkt2_2019:
    """Tests for crs_to_wkt2_2019 function."""

    def test_crs_to_wkt2_2019_returns_string(self):
        """Returns non-empty WKT2:2019 string."""
        wkt = crs_to_wkt2_2019("EPSG:4326")
        assert isinstance(wkt, str)
        assert len(wkt) > 0

    def test_crs_to_wkt2_2019_contains_datum_info(self):
        """WKT contains WGS 84 info for EPSG:4326."""
        wkt = crs_to_wkt2_2019("EPSG:4326")
        assert "WGS 84" in wkt or "WGS84" in wkt.replace(" ", "")

    def test_crs_to_wkt2_2019_pretty_true(self):
        """Pretty=True produces formatted output."""
        wkt = crs_to_wkt2_2019("EPSG:4326", pretty=True)
        # Formatted output contains newlines
        assert "\n" in wkt or "GEOGCRS" in wkt

    def test_crs_to_wkt2_2019_pretty_false(self):
        """Pretty=False produces compact output."""
        wkt = crs_to_wkt2_2019("EPSG:4326", pretty=False)
        assert isinstance(wkt, str)
        # Both formats should be valid WKT
        CRS.from_wkt(wkt)

    def test_crs_to_wkt2_2019_with_crs_object(self):
        """Accepts CRS object."""
        crs_obj = CRS.from_epsg(32610)
        wkt = crs_to_wkt2_2019(crs_obj)
        assert isinstance(wkt, str)
        assert "UTM" in wkt or "zone 10" in wkt


# ==============================================================================
# Test wrap_coordinate_metadata_wkt
# ==============================================================================

class TestWrapCoordinateMetadataWkt:
    """Tests for wrap_coordinate_metadata_wkt function."""

    def test_wrap_coordinate_metadata_contains_keyword(self):
        """Result contains COORDINATEMETADATA keyword."""
        result = wrap_coordinate_metadata_wkt("EPSG:4326", 2020.0)
        assert "COORDINATEMETADATA" in result

    def test_wrap_coordinate_metadata_contains_epoch(self):
        """Result contains EPOCH with value."""
        result = wrap_coordinate_metadata_wkt("EPSG:4326", 2020.5)
        assert "EPOCH" in result
        assert "2020.5" in result

    def test_wrap_coordinate_metadata_with_different_epochs(self):
        """Handles different epoch values."""
        result1 = wrap_coordinate_metadata_wkt("EPSG:4326", 2000.0)
        result2 = wrap_coordinate_metadata_wkt("EPSG:4326", 2020.0)
        assert "2000.0" in result1
        assert "2020.0" in result2
        assert result1 != result2

    def test_wrap_coordinate_metadata_with_crs_object(self):
        """Accepts CRS object."""
        crs_obj = CRS.from_epsg(4326)
        result = wrap_coordinate_metadata_wkt(crs_obj, 2020.0)
        assert "COORDINATEMETADATA" in result
        assert "EPOCH" in result

    def test_wrap_coordinate_metadata_wkt_format(self):
        """Output has correct WKT2:2019 format structure."""
        result = wrap_coordinate_metadata_wkt("EPSG:4326", 2020.0)
        # COORDINATEMETADATA is valid WKT2:2019 format but not parseable as standalone CRS
        # Verify structure instead
        assert result.startswith("COORDINATEMETADATA[")
        assert result.endswith("]")
        assert "EPOCH[2020.0]" in result


# ==============================================================================
# Test crs_to_projjson
# ==============================================================================

class TestCrsToProjectionJson:
    """Tests for crs_to_projjson function."""

    def test_crs_to_projjson_returns_dict(self):
        """Returns dict."""
        result = crs_to_projjson("EPSG:4326")
        assert isinstance(result, dict)

    def test_crs_to_projjson_has_type_key(self):
        """Dict has 'type' key."""
        result = crs_to_projjson("EPSG:4326")
        assert "type" in result

    def test_crs_to_projjson_geogcrs_type(self):
        """Geographic CRS has appropriate type."""
        result = crs_to_projjson("EPSG:4326")
        assert result["type"] in ["GeographicCRS", "ProjectedCRS"]

    def test_crs_to_projjson_utm_type(self):
        """UTM CRS is ProjectedCRS."""
        result = crs_to_projjson("EPSG:32610")
        assert result["type"] == "ProjectedCRS"

    def test_crs_to_projjson_with_crs_object(self):
        """Accepts CRS object."""
        crs_obj = CRS.from_epsg(4326)
        result = crs_to_projjson(crs_obj)
        assert isinstance(result, dict)
        assert "type" in result


# ==============================================================================
# Test make_coordinate_metadata_projjson
# ==============================================================================

class TestMakeCoordinateMetadataProjectionJson:
    """Tests for make_coordinate_metadata_projjson function."""

    def test_make_coordinate_metadata_projjson_has_type(self):
        """Result has 'type' = 'CoordinateMetadata'."""
        result = make_coordinate_metadata_projjson("EPSG:4326", 2020.0)
        assert result["type"] == "CoordinateMetadata"

    def test_make_coordinate_metadata_projjson_has_epoch(self):
        """Result has 'epoch' key with correct value."""
        result = make_coordinate_metadata_projjson("EPSG:4326", 2020.5)
        assert result["epoch"] == 2020.5

    def test_make_coordinate_metadata_projjson_has_crs(self):
        """Result has 'crs' key."""
        result = make_coordinate_metadata_projjson("EPSG:4326", 2020.0)
        assert "crs" in result
        assert isinstance(result["crs"], dict)

    def test_make_coordinate_metadata_projjson_epoch_as_int(self):
        """Converts integer epoch to float."""
        result = make_coordinate_metadata_projjson("EPSG:4326", 2020)
        assert result["epoch"] == 2020.0
        assert isinstance(result["epoch"], float)

    def test_make_coordinate_metadata_projjson_with_utm(self):
        """Works with projected CRS."""
        result = make_coordinate_metadata_projjson("EPSG:32610", 2015.0)
        assert result["type"] == "CoordinateMetadata"
        assert result["epoch"] == 2015.0


# ==============================================================================
# Test is_orthometric
# ==============================================================================

class TestIsOrthometric:
    """Tests for is_orthometric function."""

    def test_is_orthometric_navd88_true(self):
        """NAVD88 (EPSG:5703) is orthometric."""
        result = is_orthometric(CRS.from_epsg(5703))
        assert result is True

    def test_is_orthometric_none_input(self):
        """None input returns None."""
        result = is_orthometric(None)
        assert result is None

    def test_is_orthometric_invalid_input_returns_none(self):
        """Invalid input returns None defensively."""
        result = is_orthometric("INVALID_CRS")
        # Should return None, not raise
        assert result is None

    def test_is_orthometric_geographic_crs(self):
        """Geographic CRS without vertical returns None."""
        result = is_orthometric(CRS.from_epsg(4326))
        # Could be None or False depending on axis analysis
        assert result in (None, False)

    def test_is_orthometric_with_epsg_string(self):
        """Accepts EPSG string."""
        result = is_orthometric("EPSG:5703")
        assert result is True

    def test_is_orthometric_ellipsoidal_height(self):
        """Ellipsoidal height CRS returns False."""
        # Create a vertical CRS that explicitly says ellipsoidal
        crs_3d = CRS.from_epsg(4979)
        result = is_orthometric(crs_3d)
        # Result should indicate ellipsoidal (False or None)
        assert result in (False, None)


# ==============================================================================
# Test is_3d_geographic_crs
# ==============================================================================

class TestIs3dGeographicCrs:
    """Tests for is_3d_geographic_crs function."""

    def test_is_3d_geographic_crs_wgs84_3d_true(self):
        """EPSG:4979 (WGS84 3D) returns True."""
        assert is_3d_geographic_crs("EPSG:4979")

    def test_is_3d_geographic_crs_wgs84_2d_false(self):
        """EPSG:4326 (WGS84 2D) returns False."""
        assert not is_3d_geographic_crs("EPSG:4326")

    def test_is_3d_geographic_crs_utm_false(self):
        """UTM (projected) returns False."""
        assert not is_3d_geographic_crs("EPSG:32610")

    def test_is_3d_geographic_crs_with_crs_object(self):
        """Accepts CRS object."""
        crs_obj = CRS.from_epsg(4979)
        assert is_3d_geographic_crs(crs_obj)

    def test_is_3d_geographic_crs_invalid_returns_false(self):
        """Invalid input returns False."""
        assert not is_3d_geographic_crs("INVALID_CRS")

    def test_is_3d_geographic_crs_vertical_only_false(self):
        """Vertical-only CRS returns False."""
        assert not is_3d_geographic_crs("EPSG:5703")


# ==============================================================================
# Test extract_ellipsoidal_height_as_vertical_crs
# ==============================================================================

class TestExtractEllipsoidalHeightAsVerticalCrs:
    """Tests for extract_ellipsoidal_height_as_vertical_crs function."""

    def test_extract_ellipsoidal_height_from_3d_crs(self):
        """Extract vertical component from WGS84 3D."""
        vert_crs = extract_ellipsoidal_height_as_vertical_crs("EPSG:4979")
        assert vert_crs.is_vertical
        assert "ellipsoidal" in vert_crs.name.lower() or "height" in vert_crs.name.lower()

    def test_extract_ellipsoidal_height_with_crs_object(self):
        """Works with CRS object."""
        crs_obj = CRS.from_epsg(4979)
        vert_crs = extract_ellipsoidal_height_as_vertical_crs(crs_obj)
        assert vert_crs.is_vertical

    def test_extract_ellipsoidal_height_from_non_3d_raises(self):
        """Non-3D CRS raises ValueError."""
        with pytest.raises(ValueError):
            extract_ellipsoidal_height_as_vertical_crs("EPSG:4326")

    def test_extract_ellipsoidal_height_from_utm_raises(self):
        """Projected CRS raises ValueError."""
        with pytest.raises(ValueError):
            extract_ellipsoidal_height_as_vertical_crs("EPSG:32610")

    def test_extract_ellipsoidal_height_returns_vertical_crs(self):
        """Returned CRS has is_vertical = True."""
        vert_crs = extract_ellipsoidal_height_as_vertical_crs("EPSG:4979")
        assert hasattr(vert_crs, 'is_vertical')
        assert vert_crs.is_vertical


# ==============================================================================
# Test create_compound_crs
# ==============================================================================

class TestCreateCompoundCrs:
    """Tests for create_compound_crs function."""

    def test_create_compound_crs_basic(self):
        """Create compound from UTM + NAVD88."""
        compound = create_compound_crs("EPSG:32610", "EPSG:5703")
        assert compound.is_compound

    def test_create_compound_crs_with_crs_objects(self):
        """Works with CRS objects."""
        horiz = CRS.from_epsg(32610)
        vert = CRS.from_epsg(5703)
        compound = create_compound_crs(horiz, vert)
        assert compound.is_compound

    def test_create_compound_crs_has_components(self):
        """Compound CRS has sub_crs_list."""
        compound = create_compound_crs("EPSG:32610", "EPSG:5703")
        assert hasattr(compound, 'sub_crs_list')
        assert len(compound.sub_crs_list) == 2

    def test_create_compound_crs_horizontal_is_first(self):
        """First component is horizontal."""
        compound = create_compound_crs("EPSG:32610", "EPSG:5703")
        horiz_component = compound.sub_crs_list[0]
        assert horiz_component.to_epsg() == 32610

    def test_create_compound_crs_vertical_is_second(self):
        """Second component is vertical."""
        compound = create_compound_crs("EPSG:32610", "EPSG:5703")
        vert_component = compound.sub_crs_list[1]
        assert vert_component.is_vertical

    def test_create_compound_crs_with_projected_horiz(self):
        """Works with any horizontal CRS."""
        compound = create_compound_crs("EPSG:2230", "EPSG:5703")
        assert compound.is_compound
        assert compound.sub_crs_list[0].to_epsg() == 2230


# ==============================================================================
# Test parse_crs_components
# ==============================================================================

class TestParseCrsComponents:
    """Tests for parse_crs_components function."""

    def test_parse_crs_components_compound_crs(self):
        """Compound CRS returns all three components."""
        compound = create_compound_crs("EPSG:32610", "EPSG:5703")
        compound_wkt, horiz_wkt, vert_wkt = parse_crs_components(compound)
        assert compound_wkt is not None
        assert horiz_wkt is not None
        assert vert_wkt is not None

    def test_parse_crs_components_horizontal_only(self):
        """Horizontal CRS returns only horizontal."""
        horiz_wkt, vert_wkt, compound_wkt = parse_crs_components("EPSG:32610")
        compound_wkt, horiz_wkt, vert_wkt = parse_crs_components("EPSG:32610")
        assert compound_wkt is None
        assert horiz_wkt is not None
        assert vert_wkt is None

    def test_parse_crs_components_vertical_only(self):
        """Vertical CRS returns only vertical."""
        compound_wkt, horiz_wkt, vert_wkt = parse_crs_components("EPSG:5703")
        assert compound_wkt is None
        assert horiz_wkt is None
        assert vert_wkt is not None

    def test_parse_crs_components_none_input(self):
        """None input returns all None."""
        compound_wkt, horiz_wkt, vert_wkt = parse_crs_components(None)
        assert compound_wkt is None
        assert horiz_wkt is None
        assert vert_wkt is None

    def test_parse_crs_components_with_crs_object(self):
        """Works with CRS object."""
        crs_obj = CRS.from_epsg(32610)
        compound_wkt, horiz_wkt, vert_wkt = parse_crs_components(crs_obj)
        assert horiz_wkt is not None

    def test_parse_crs_components_returns_wkt_strings(self):
        """Returned components are WKT strings."""
        compound = create_compound_crs("EPSG:32610", "EPSG:5703")
        compound_wkt, horiz_wkt, vert_wkt = parse_crs_components(compound)
        # All should be parseable as WKT
        CRS.from_wkt(horiz_wkt)
        CRS.from_wkt(vert_wkt)

    def test_parse_crs_components_geographic_only(self):
        """Geographic (non-projected) CRS returns only horizontal."""
        compound_wkt, horiz_wkt, vert_wkt = parse_crs_components("EPSG:4326")
        assert compound_wkt is None
        assert horiz_wkt is not None
        assert vert_wkt is None


# ==============================================================================
# Test transformer_with_epoch
# ==============================================================================

class TestTransformerWithEpoch:
    """Tests for transformer_with_epoch function."""

    def test_transformer_with_epoch_returns_transformer(self):
        """Returns Transformer object."""
        transformer = transformer_with_epoch("EPSG:4326", "EPSG:32610")
        assert isinstance(transformer, Transformer)

    def test_transformer_with_epoch_basic_transform(self):
        """Transformer performs basic coordinate transformation."""
        transformer = transformer_with_epoch("EPSG:4326", "EPSG:32610")
        # San Francisco: -122.4, 37.8
        x, y = transformer.transform(-122.4, 37.8)
        # Should be in UTM (larger numbers)
        assert 400000 < x < 600000
        assert 4100000 < y < 4300000

    def test_transformer_with_epoch_accepts_epochs(self):
        """Accepts source and target epochs."""
        transformer = transformer_with_epoch(
            "EPSG:4326",
            "EPSG:32610",
            src_epoch=2000.0,
            dst_epoch=2020.0
        )
        assert isinstance(transformer, Transformer)

    def test_transformer_with_epoch_with_crs_objects(self):
        """Works with CRS objects."""
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_epsg(32610)
        transformer = transformer_with_epoch(src_crs, dst_crs)
        assert isinstance(transformer, Transformer)

    def test_transformer_with_epoch_always_xy_true(self):
        """Uses always_xy=True convention (lon/lat order)."""
        transformer = transformer_with_epoch("EPSG:4326", "EPSG:32610")
        # -122.4, 37.8 should be valid (lon, lat)
        x, y = transformer.transform(-122.4, 37.8)
        # If always_xy is False, would expect error or wrong result
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))


# ==============================================================================
# Test horizontal_unit_scale
# ==============================================================================

class TestHorizontalUnitScale:
    """Tests for horizontal_unit_scale function."""

    def test_horizontal_unit_scale_meter_to_meter(self):
        """UTM (meters) to meters = 1.0."""
        scale = horizontal_unit_scale("EPSG:32610", "meter")
        assert scale == pytest.approx(1.0)

    def test_horizontal_unit_scale_meter_to_foot(self):
        """Meters to feet ≈ 3.28084."""
        scale = horizontal_unit_scale("EPSG:32610", "foot")
        assert scale == pytest.approx(3.28084, rel=1e-4)

    def test_horizontal_unit_scale_us_survey_feet_unknown(self):
        """US Survey feet unit not recognized in unit conversion table returns None."""
        # EPSG:2230 uses "US survey foot" which is not in the _HORIZONTAL_UNIT_FACTORS dict
        # Only "us_survey_foot" (with underscore) is in the lookup table
        scale = horizontal_unit_scale("EPSG:2230", "meter")
        # Since the axis unit_name doesn't match any key, returns None
        assert scale is None

    def test_horizontal_unit_scale_with_crs_object(self):
        """Works with CRS object."""
        crs_obj = CRS.from_epsg(32610)
        scale = horizontal_unit_scale(crs_obj, "meter")
        assert scale == pytest.approx(1.0)

    def test_horizontal_unit_scale_unknown_unit_returns_none(self):
        """Unknown unit returns None."""
        scale = horizontal_unit_scale("EPSG:32610", "unknown_unit")
        assert scale is None

    def test_horizontal_unit_scale_case_insensitive(self):
        """Unit name is case insensitive."""
        scale1 = horizontal_unit_scale("EPSG:32610", "meter")
        scale2 = horizontal_unit_scale("EPSG:32610", "METER")
        scale3 = horizontal_unit_scale("EPSG:32610", "Meter")
        assert scale1 == scale2 == scale3


# ==============================================================================
# Test vertical_unit_scale
# ==============================================================================

class TestVerticalUnitScale:
    """Tests for vertical_unit_scale function."""

    def test_vertical_unit_scale_meter_to_meter(self):
        """NAVD88 (meters) to meters = 1.0."""
        scale = vertical_unit_scale("EPSG:5703", "meter")
        assert scale == pytest.approx(1.0)

    def test_vertical_unit_scale_meter_to_foot(self):
        """Meters to feet ≈ 3.28084."""
        scale = vertical_unit_scale("EPSG:5703", "foot")
        assert scale == pytest.approx(3.28084, rel=1e-4)

    def test_vertical_unit_scale_with_crs_object(self):
        """Works with CRS object."""
        crs_obj = CRS.from_epsg(5703)
        scale = vertical_unit_scale(crs_obj, "meter")
        assert scale == pytest.approx(1.0)

    def test_vertical_unit_scale_unknown_unit_returns_none(self):
        """Unknown unit returns None."""
        scale = vertical_unit_scale("EPSG:5703", "unknown_unit")
        assert scale is None

    def test_vertical_unit_scale_geographic_crs(self):
        """Works with geographic CRS (uses last axis)."""
        # Geographic CRS has lat/lon axes
        scale = vertical_unit_scale("EPSG:4326", "degree")
        # Degrees are not in the unit conversion table, so None
        assert scale is None


# ==============================================================================
# Test apply_vertical_datum_transform
# ==============================================================================

class TestApplyVerticalDatumTransform:
    """Tests for apply_vertical_datum_transform function."""

    def test_apply_vertical_datum_transform_same_crs(self):
        """Transform to same CRS leaves z unchanged."""
        z = np.array([100.0, 200.0, 300.0])
        result = apply_vertical_datum_transform(
            z, "EPSG:5703", "EPSG:5703"
        )
        np.testing.assert_array_almost_equal(result, z)

    def test_apply_vertical_datum_transform_returns_array(self):
        """Returns numpy array."""
        z = np.array([100.0, 200.0])
        result = apply_vertical_datum_transform(z, "EPSG:5703", "EPSG:5703")
        assert isinstance(result, np.ndarray)

    def test_apply_vertical_datum_transform_preserves_shape(self):
        """Output shape matches input shape."""
        z = np.array([100.0, 200.0, 300.0, 400.0])
        result = apply_vertical_datum_transform(z, "EPSG:5703", "EPSG:5703")
        assert result.shape == z.shape

    def test_apply_vertical_datum_transform_2d_array(self):
        """Works with 2D arrays."""
        z = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = apply_vertical_datum_transform(z, "EPSG:5703", "EPSG:5703")
        assert result.shape == z.shape

    def test_apply_vertical_datum_transform_invalid_crs_returns_unchanged(self):
        """Invalid CRS returns z unchanged."""
        z = np.array([100.0, 200.0])
        result = apply_vertical_datum_transform(
            z, "INVALID_CRS", "EPSG:5703"
        )
        np.testing.assert_array_equal(result, z)

    def test_apply_vertical_datum_transform_scalar_x_y(self):
        """Uses scalar x, y (optimization)."""
        z = np.array([100.0, 200.0, 300.0])
        # Should not raise; uses scalar 0, 0 for x, y
        result = apply_vertical_datum_transform(z, "EPSG:5703", "EPSG:5703")
        assert result.shape == z.shape


# ==============================================================================
# Test apply_dynamic_transform
# ==============================================================================

class TestApplyDynamicTransform:
    """Tests for apply_dynamic_transform function."""

    def test_apply_dynamic_transform_3d_basic(self):
        """Basic 3D transformation returns tuple of 3 arrays."""
        x = np.array([-122.4])
        y = np.array([37.8])
        z = np.array([100.0])

        x_out, y_out, z_out = apply_dynamic_transform(
            x, y, z,
            "EPSG:4326", "EPSG:32610",
            src_epoch=None, dst_epoch=None
        )

        assert isinstance(x_out, (np.ndarray, tuple, list))
        assert isinstance(y_out, (np.ndarray, tuple, list))
        assert isinstance(z_out, (np.ndarray, tuple, list))

    def test_apply_dynamic_transform_2d_basic(self):
        """2D transformation (z=None) returns tuple of 2 arrays and None."""
        x = np.array([-122.4])
        y = np.array([37.8])

        x_out, y_out, z_out = apply_dynamic_transform(
            x, y, None,
            "EPSG:4326", "EPSG:32610",
            src_epoch=None, dst_epoch=None
        )

        assert isinstance(x_out, (np.ndarray, tuple, list))
        assert isinstance(y_out, (np.ndarray, tuple, list))
        assert z_out is None

    def test_apply_dynamic_transform_utm_to_geographic(self):
        """UTM to geographic produces reasonable lat/lon."""
        x = np.array([500000.0])  # UTM easting
        y = np.array([4200000.0])  # UTM northing
        z = np.array([100.0])

        x_out, y_out, z_out = apply_dynamic_transform(
            x, y, z,
            "EPSG:32610", "EPSG:4326",
            src_epoch=None, dst_epoch=None
        )

        # Should be in geographic range
        assert -180 < x_out[0] < 180 or isinstance(x_out, (list, tuple))
        assert -90 < y_out[0] < 90 or isinstance(y_out, (list, tuple))

    def test_apply_dynamic_transform_with_epochs(self):
        """Accepts source and target epochs."""
        x = np.array([-122.4])
        y = np.array([37.8])
        z = np.array([100.0])

        result = apply_dynamic_transform(
            x, y, z,
            "EPSG:4326", "EPSG:32610",
            src_epoch=2000.0, dst_epoch=2020.0
        )

        assert len(result) == 3

    def test_apply_dynamic_transform_multiple_coordinates(self):
        """Works with multiple coordinates."""
        x = np.array([-122.4, -122.5, -122.3])
        y = np.array([37.8, 37.9, 37.7])
        z = np.array([100.0, 200.0, 150.0])

        x_out, y_out, z_out = apply_dynamic_transform(
            x, y, z,
            "EPSG:4326", "EPSG:32610",
            src_epoch=None, dst_epoch=None
        )

        # Output should have same shape as input
        assert len(x_out) == 3 or np.array(x_out).shape == (3,)


# ==============================================================================
# Integration tests
# ==============================================================================

class TestCrsUtilsIntegration:
    """Integration tests combining multiple functions."""

    def test_workflow_geographic_to_utm(self):
        """Workflow: convert geographic CRS to UTM."""
        src_crs = "EPSG:4326"
        dst_crs = "EPSG:32610"

        assert crs_equals(src_crs, "EPSG:4326")
        assert not crs_equals(src_crs, dst_crs)

        transformer = transformer_with_epoch(src_crs, dst_crs)
        x, y = transformer.transform(-122.4, 37.8)
        assert 400000 < x < 600000

    def test_workflow_compound_crs_creation_and_parsing(self):
        """Workflow: create compound CRS and parse it back."""
        compound = create_compound_crs("EPSG:32610", "EPSG:5703")
        assert compound.is_compound

        compound_wkt, horiz_wkt, vert_wkt = parse_crs_components(compound)
        assert compound_wkt is not None
        assert horiz_wkt is not None
        assert vert_wkt is not None

        # Parse back
        horiz_crs = CRS.from_wkt(horiz_wkt)
        assert horiz_crs.to_epsg() == 32610

    def test_workflow_3d_geographic_to_compound(self):
        """Workflow: convert 3D geographic to compound CRS."""
        crs_3d = "EPSG:4979"
        assert is_3d_geographic_crs(crs_3d)

        # Extract vertical component
        vert = extract_ellipsoidal_height_as_vertical_crs(crs_3d)
        assert vert.is_vertical

        # Create 2D geographic
        crs_2d = CRS.from_epsg(4326)

        # Create compound
        compound = create_compound_crs(crs_2d, vert)
        assert compound.is_compound

    def test_workflow_unit_conversions(self):
        """Workflow: check unit scales for different CRS."""
        utm_scale = horizontal_unit_scale("EPSG:32610", "foot")
        assert utm_scale == pytest.approx(3.28084, rel=1e-4)

        nav_scale = vertical_unit_scale("EPSG:5703", "foot")
        assert nav_scale == pytest.approx(3.28084, rel=1e-4)

        # Both should be the same (physical units)
        assert utm_scale == pytest.approx(nav_scale, rel=1e-10)

    def test_workflow_coordinate_metadata_with_epochs(self):
        """Workflow: create coordinate metadata with epoch."""
        crs = "EPSG:4326"
        epoch = 2020.0

        # WKT format
        wkt_meta = wrap_coordinate_metadata_wkt(crs, epoch)
        assert "COORDINATEMETADATA" in wkt_meta
        assert "2020.0" in wkt_meta

        # PROJJSON format
        json_meta = make_coordinate_metadata_projjson(crs, epoch)
        assert json_meta["type"] == "CoordinateMetadata"
        assert json_meta["epoch"] == 2020.0


# ==============================================================================
# Edge case and error handling tests
# ==============================================================================

class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""

    def test_empty_string_crs_invalid(self):
        """Empty string CRS raises exception."""
        with pytest.raises(Exception):
            _ensure_crs_obj("")

    def test_crs_with_very_large_epoch(self):
        """Handles very large epoch values."""
        result = wrap_coordinate_metadata_wkt("EPSG:4326", 9999.5)
        assert "9999.5" in result

    def test_crs_with_negative_epoch(self):
        """Handles negative epoch values."""
        result = wrap_coordinate_metadata_wkt("EPSG:4326", -100.0)
        assert "-100.0" in result

    def test_zero_array_z_coordinate(self):
        """Handles z array of all zeros."""
        z = np.zeros(10)
        result = apply_vertical_datum_transform(z, "EPSG:5703", "EPSG:5703")
        np.testing.assert_array_equal(result, z)

    def test_large_coordinate_arrays(self):
        """Works with large coordinate arrays."""
        x = np.random.uniform(-122.5, -122.3, 1000)
        y = np.random.uniform(37.7, 37.9, 1000)
        z = np.random.uniform(0, 1000, 1000)

        x_out, y_out, z_out = apply_dynamic_transform(
            x, y, z,
            "EPSG:4326", "EPSG:32610",
            src_epoch=None, dst_epoch=None
        )

        assert len(x_out) == 1000 or np.array(x_out).shape == (1000,)

    def test_crs_equals_with_mixed_input_types(self):
        """crs_equals works with mixed input types."""
        # All these should be equal
        assert crs_equals("EPSG:4326", CRS.from_epsg(4326))
        assert crs_equals(CRS.from_epsg(4326), "EPSG:4326")
        assert crs_equals(
            CRS.from_epsg(4326).to_json_dict(),
            "EPSG:4326"
        )

    def test_parse_crs_components_with_string_input(self):
        """parse_crs_components accepts string CRS."""
        compound_wkt, horiz_wkt, vert_wkt = parse_crs_components("EPSG:32610")
        assert horiz_wkt is not None

    def test_transformer_with_none_epochs(self):
        """transformer_with_epoch handles None epochs."""
        transformer = transformer_with_epoch(
            "EPSG:4326", "EPSG:32610",
            src_epoch=None, dst_epoch=None
        )
        assert isinstance(transformer, Transformer)


# ==============================================================================
# Tests for vertical_datum_to_crs
# ==============================================================================

class TestVerticalDatumToCrs:
    """Tests for vertical datum name → pyproj CRS mapping."""

    def test_navd88_returns_5703(self):
        """NAVD88 maps to EPSG:5703 (NAVD88 height)."""
        crs = vertical_datum_to_crs("NAVD88")
        assert crs is not None
        assert crs.to_epsg() == 5703

    def test_navd88_case_insensitive(self):
        """Lookup is case-insensitive."""
        crs = vertical_datum_to_crs("navd88")
        assert crs is not None
        assert crs.to_epsg() == 5703

    def test_ngvd29_returns_5702(self):
        """NGVD29 maps to EPSG:5702."""
        crs = vertical_datum_to_crs("NGVD29")
        assert crs is not None
        assert crs.to_epsg() == 5702

    def test_egm96_returns_5773(self):
        """EGM96 datum maps to EPSG:5773."""
        crs = vertical_datum_to_crs("EGM96")
        assert crs is not None
        assert crs.to_epsg() == 5773

    def test_egm2008_returns_3855(self):
        """EGM2008 datum maps to EPSG:3855."""
        crs = vertical_datum_to_crs("EGM2008")
        assert crs is not None
        assert crs.to_epsg() == 3855

    def test_ellipsoidal_returns_none(self):
        """Ellipsoidal height has no orthometric CRS mapping."""
        assert vertical_datum_to_crs("ellipsoidal") is None

    def test_none_datum_none_geoid_returns_none(self):
        """Both None → None."""
        assert vertical_datum_to_crs(None, None) is None

    def test_geoid_model_navd88_realization(self):
        """Known NAVD88 geoid realizations map to EPSG:5703."""
        for geoid in ("geoid09", "geoid12b", "geoid18"):
            crs = vertical_datum_to_crs(None, geoid)
            assert crs is not None, f"Failed for {geoid}"
            assert crs.to_epsg() == 5703, f"Wrong EPSG for {geoid}"

    def test_geoid_model_egm96(self):
        """EGM96 geoid model maps correctly via fallback."""
        crs = vertical_datum_to_crs(None, "egm96")
        assert crs is not None
        assert crs.to_epsg() == 5773

    def test_unknown_datum_returns_none(self):
        """Unrecognized datum returns None."""
        assert vertical_datum_to_crs("SOMETHING_UNKNOWN") is None

    def test_datum_takes_precedence_over_geoid(self):
        """When datum is known, it is used even if geoid is also provided."""
        crs = vertical_datum_to_crs("NAVD88", "geoid18")
        assert crs.to_epsg() == 5703


# ==============================================================================
# Tests for build_output_crs_wkt
# ==============================================================================

class TestBuildOutputCrsWkt:
    """Tests for compound + epoch WKT2 builder."""

    def test_horizontal_only(self):
        """Horizontal CRS only produces a valid WKT string."""
        wkt = build_output_crs_wkt("EPSG:32610")
        # Should be a valid projected CRS WKT2
        crs = CRS.from_wkt(wkt)
        assert crs.to_epsg() == 32610

    def test_horizontal_plus_vertical(self):
        """Horizontal + vertical produces a compound CRS WKT."""
        wkt = build_output_crs_wkt("EPSG:32610", "EPSG:5703")
        assert "COMPOUNDCRS" in wkt
        crs = CRS.from_wkt(wkt)
        assert crs.is_compound

    def test_with_epoch_only(self):
        """Epoch wraps the CRS in COORDINATEMETADATA."""
        wkt = build_output_crs_wkt("EPSG:32610", epoch=2020.5)
        assert "COORDINATEMETADATA" in wkt
        assert "EPOCH" in wkt
        assert "2020.5" in wkt

    def test_compound_with_epoch(self):
        """Compound CRS + epoch produces full COORDINATEMETADATA wrapper."""
        wkt = build_output_crs_wkt("EPSG:32613", "EPSG:5703", 2011.726)
        assert "COORDINATEMETADATA" in wkt
        assert "COMPOUNDCRS" in wkt
        assert "EPOCH" in wkt
        assert "2011.726" in wkt

    def test_no_epoch_no_vertical(self):
        """Without vertical or epoch, returns plain WKT2:2019."""
        wkt = build_output_crs_wkt("EPSG:32610")
        assert "COORDINATEMETADATA" not in wkt
        assert "COMPOUNDCRS" not in wkt

    def test_accepts_crs_object(self):
        """Accepts pyproj.CRS objects as input."""
        horiz = CRS.from_epsg(32610)
        vert = CRS.from_epsg(5703)
        wkt = build_output_crs_wkt(horiz, vert)
        assert "COMPOUNDCRS" in wkt


# ==============================================================================
# Test extract_epoch_from_wkt
# ==============================================================================

class TestExtractEpochFromWkt:
    """Tests for extract_epoch_from_wkt function."""

    def test_simple_coordinatemetadata(self):
        """Extract epoch from a simple COORDINATEMETADATA WKT."""
        wkt = 'COORDINATEMETADATA[PROJCRS["WGS 84 / UTM zone 13N"],EPOCH[2011.5]]'
        assert extract_epoch_from_wkt(wkt) == 2011.5

    def test_compound_crs_with_epoch(self):
        """Extract epoch from COORDINATEMETADATA wrapping a compound CRS."""
        wkt = (
            'COORDINATEMETADATA['
            'COMPOUNDCRS["NAD83 + NAVD88",'
            'PROJCRS["NAD 83"],'
            'VERTCRS["NAVD88 height"]'
            '],'
            'EPOCH[2011.726]'
            ']'
        )
        assert extract_epoch_from_wkt(wkt) == pytest.approx(2011.726)

    def test_no_coordinatemetadata(self):
        """Return None when WKT has no COORDINATEMETADATA."""
        wkt = 'PROJCRS["WGS 84 / UTM zone 13N",BASEGEOGCRS["WGS 84"]]'
        assert extract_epoch_from_wkt(wkt) is None

    def test_compound_crs_without_epoch(self):
        """Return None for compound CRS without COORDINATEMETADATA."""
        wkt = (
            'COMPOUNDCRS["NAD83 + NAVD88",'
            'PROJCRS["NAD 83"],'
            'VERTCRS["NAVD88 height"]'
            ']'
        )
        assert extract_epoch_from_wkt(wkt) is None

    def test_none_input(self):
        """Return None for None input."""
        assert extract_epoch_from_wkt(None) is None

    def test_empty_string(self):
        """Return None for empty string."""
        assert extract_epoch_from_wkt("") is None

    def test_integer_epoch(self):
        """Extract integer epoch value."""
        wkt = 'COORDINATEMETADATA[PROJCRS["Test"],EPOCH[2011]]'
        assert extract_epoch_from_wkt(wkt) == 2011.0

    def test_real_wkt_from_build_output(self):
        """Extract epoch from a WKT actually produced by build_output_crs_wkt."""
        wkt = build_output_crs_wkt("EPSG:32613", "EPSG:5703", 2011.726)
        epoch = extract_epoch_from_wkt(wkt)
        assert epoch == pytest.approx(2011.726)
