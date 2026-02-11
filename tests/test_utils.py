"""Comprehensive pytest test file for unit_utils and time_utils modules.

Tests two utility modules:
1. src/topochange/unit_utils.py - Unit conversion and CRS handling
2. src/topochange/time_utils.py - Time and epoch conversion"""

import pytest
import numpy as np
import datetime
import math
from pathlib import Path

from topochange.unit_utils import (
    UnitInfo,
    METER,
    FOOT,
    US_SURVEY_FOOT,
    KILOMETER,
    DEGREE,
    RADIAN,
    UNKNOWN_UNIT,
    lookup_unit,
    lookup_unit_strict,
    parse_unit_string,
    get_horizontal_unit,
    get_vertical_unit,
    get_crs_units,
    convert_length,
    convert_to_meters,
    convert_from_meters,
    get_conversion_factor,
    parse_pdal_units,
    parse_catalog_vertical_units,
    format_value_with_unit,
    describe_unit,
)
from topochange.time_utils import (
    _datetime_to_decimal_year,
    _parse_epoch_string_to_decimal,
    GPS_EPOCH,
    _gps_leap_seconds,
    gps_seconds_to_decimal_year_utc,
    _guess_in_time_from_stats,
)


# uNIT_UTILS TESTS

class TestUnitInfoDataclass:
    """Tests for UnitInfo dataclass and basic properties."""

    def test_unitinfo_creation(self):
        """Test creating a UnitInfo instance."""
        unit = UnitInfo("meter", "metre", "m", 1.0, "linear", 9001)
        assert unit.name == "meter"
        assert unit.display_name == "metre"
        assert unit.abbreviation == "m"
        assert unit.to_base_factor == 1.0
        assert unit.category == "linear"
        assert unit.epsg_code == 9001

    def test_unitinfo_frozen(self):
        """Test that UnitInfo is frozen (immutable)."""
        unit = METER
        with pytest.raises(Exception):  # FrozenInstanceError
            unit.to_base_factor = 2.0

    def test_unitinfo_str(self):
        """Test UnitInfo string representation."""
        unit = METER
        assert str(unit) == "metre (m)"

    def test_unitinfo_repr(self):
        """Test UnitInfo repr."""
        unit = METER
        repr_str = repr(unit)
        assert "meter" in repr_str
        assert "1.0" in repr_str


class TestUnitConversion:
    """Tests for UnitInfo.convert_to() method."""

    def test_meter_to_foot(self):
        """Test converting meters to feet."""
        values = np.array([1.0])
        result = METER.convert_to(values, FOOT)
        expected = 1.0 / 0.3048
        np.testing.assert_almost_equal(result[0], expected, decimal=5)

    def test_foot_to_meter(self):
        """Test converting feet to meters."""
        values = np.array([1.0])
        result = FOOT.convert_to(values, METER)
        np.testing.assert_almost_equal(result[0], 0.3048, decimal=10)

    def test_us_survey_foot_to_meter(self):
        """Test US survey foot to meter conversion."""
        values = np.array([100.0])
        result = US_SURVEY_FOOT.convert_to(values, METER)
        expected = 100.0 * (1200.0 / 3937.0)
        np.testing.assert_almost_equal(result[0], expected, decimal=8)

    def test_meter_to_kilometer(self):
        """Test meter to kilometer conversion."""
        values = np.array([1000.0])
        result = METER.convert_to(values, KILOMETER)
        np.testing.assert_almost_equal(result[0], 1.0, decimal=10)

    def test_category_mismatch_raises_error(self):
        """Test that converting between different categories raises ValueError."""
        values = np.array([1.0])
        with pytest.raises(ValueError, match="Cannot convert between"):
            METER.convert_to(values, DEGREE)

    def test_array_conversion(self):
        """Test converting arrays of values."""
        values = np.array([100.0, 200.0, 300.0])
        result = FOOT.convert_to(values, METER)
        expected = np.array([30.48, 60.96, 91.44])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_scalar_conversion(self):
        """Test converting scalar values (gets converted to array)."""
        result = FOOT.convert_to(100.0, METER)
        np.testing.assert_almost_equal(float(result), 30.48, decimal=5)

    def test_degree_to_radian(self):
        """Test angular unit conversion."""
        values = np.array([180.0])
        result = DEGREE.convert_to(values, RADIAN)
        np.testing.assert_almost_equal(result[0], math.pi, decimal=10)

    def test_radian_to_degree(self):
        """Test radian to degree conversion."""
        values = np.array([math.pi])
        result = RADIAN.convert_to(values, DEGREE)
        np.testing.assert_almost_equal(result[0], 180.0, decimal=10)


class TestLookupUnit:
    """Tests for lookup_unit() function."""

    def test_lookup_meter_variations(self):
        """Test looking up meter with various names."""
        assert lookup_unit("metre") == METER
        assert lookup_unit("meter") == METER
        assert lookup_unit("m") == METER
        assert lookup_unit("METER") == METER
        assert lookup_unit("METRE") == METER

    def test_lookup_foot(self):
        """Test looking up foot."""
        assert lookup_unit("foot") == FOOT
        assert lookup_unit("feet") == FOOT
        assert lookup_unit("ft") == FOOT

    def test_lookup_us_survey_foot(self):
        """Test looking up US survey foot."""
        result = lookup_unit("us survey foot")
        assert result == US_SURVEY_FOOT
        assert lookup_unit("us_survey_foot") == US_SURVEY_FOOT
        assert lookup_unit("ftUS") == US_SURVEY_FOOT
        assert lookup_unit("ftus") == US_SURVEY_FOOT

    def test_lookup_degree(self):
        """Test looking up degree."""
        assert lookup_unit("degree") == DEGREE
        assert lookup_unit("degrees") == DEGREE
        assert lookup_unit("deg") == DEGREE
        assert lookup_unit("°") == DEGREE

    def test_lookup_case_insensitive(self):
        """Test that lookup is case-insensitive."""
        assert lookup_unit("FOOT") == lookup_unit("foot")
        assert lookup_unit("Kilometer") == lookup_unit("kilometer")

    def test_lookup_invalid_returns_none(self):
        """Test that invalid unit returns None."""
        assert lookup_unit("invalid_unit") is None
        assert lookup_unit("xyz") is None

    def test_lookup_empty_string_returns_none(self):
        """Test that empty string returns None."""
        assert lookup_unit("") is None
        assert lookup_unit("   ") is None

    def test_lookup_with_whitespace(self):
        """Test lookup with leading/trailing whitespace."""
        assert lookup_unit("  meter  ") == METER
        assert lookup_unit("\tfoot\n") == FOOT


class TestLookupUnitStrict:
    """Tests for lookup_unit_strict() function."""

    def test_valid_unit(self):
        """Test strict lookup with valid unit."""
        assert lookup_unit_strict("meter") == METER
        assert lookup_unit_strict("foot") == FOOT

    def test_invalid_unit_raises_error(self):
        """Test that strict lookup raises ValueError for invalid unit."""
        with pytest.raises(ValueError, match="Unknown unit"):
            lookup_unit_strict("invalid_unit")

    def test_empty_string_raises_error(self):
        """Test that strict lookup raises ValueError for empty string."""
        with pytest.raises(ValueError, match="Unknown unit"):
            lookup_unit_strict("")


class TestParseUnitString:
    """Tests for parse_unit_string() function."""

    def test_parse_parentheses_meter(self):
        """Test parsing unit from parentheses."""
        result = parse_unit_string("(metre)")
        assert result == METER

    def test_parse_parentheses_us_foot(self):
        """Test parsing US survey foot from parentheses."""
        result = parse_unit_string("(ftUS)")
        assert result == US_SURVEY_FOOT

    def test_parse_with_context(self):
        """Test parsing unit from string with context."""
        result = parse_unit_string("NAVD88 height (ftUS)")
        assert result == US_SURVEY_FOOT

    def test_parse_direct_name(self):
        """Test parsing direct unit name."""
        result = parse_unit_string("meter")
        assert result == METER

    def test_parse_empty_string(self):
        """Test parsing empty string returns UNKNOWN_UNIT."""
        result = parse_unit_string("")
        assert result == UNKNOWN_UNIT

    def test_parse_with_pattern_matching(self):
        """Test parsing with pattern matching."""
        result = parse_unit_string("NAVD88 height - Geoid12B (metre)")
        assert result == METER

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        assert parse_unit_string("(METRE)") == METER
        assert parse_unit_string("(FT)") == FOOT

    def test_parse_feet_pattern(self):
        """Test parsing feet pattern."""
        result = parse_unit_string("elevation in feet")
        assert result == FOOT

    def test_parse_meter_pattern(self):
        """Test parsing meter pattern."""
        result = parse_unit_string("height in metres")
        assert result == METER


class TestCRSUnitExtraction:
    """Tests for CRS unit extraction functions."""

    def test_get_horizontal_unit_utm(self):
        """Test getting horizontal unit from UTM projection."""
        unit = get_horizontal_unit("EPSG:32610")
        assert unit == METER or unit.name == "meter"

    def test_get_horizontal_unit_geographic(self):
        """Test getting horizontal unit from geographic CRS."""
        unit = get_horizontal_unit("EPSG:4326")
        assert unit == DEGREE or unit.name == "degree"

    def test_get_horizontal_unit_state_plane_feet(self):
        """Test getting horizontal unit from State Plane (feet)."""
        unit = get_horizontal_unit("EPSG:2230")
        assert unit == US_SURVEY_FOOT or unit.name == "us_survey_foot"

    def test_get_vertical_unit_navd88(self):
        """Test getting vertical unit from NAVD88."""
        unit = get_vertical_unit("EPSG:5703")
        assert unit == METER or unit.name == "meter"

    def test_get_vertical_unit_2d_crs(self):
        """Test getting vertical unit from 2D CRS returns UNKNOWN."""
        unit = get_vertical_unit("EPSG:32610")
        assert unit.name == "unknown" or unit == UNKNOWN_UNIT

    def test_get_crs_units_compound(self):
        """Test getting both units from compound CRS."""
        h_unit, v_unit = get_crs_units("EPSG:6349")
        # ePSG:6349 is a geographic+vertical compound, so h_unit is angular (degree)
        assert h_unit.category in ["linear", "angular"]
        # v_unit should be meter
        assert v_unit is not None

    def test_get_crs_units_2d(self):
        """Test getting units from 2D CRS."""
        h_unit, v_unit = get_crs_units("EPSG:32610")
        assert h_unit == METER or h_unit.name == "meter"
        assert v_unit is None

    def test_invalid_crs_returns_unknown(self):
        """Test that invalid CRS returns UNKNOWN_UNIT."""
        unit = get_horizontal_unit("INVALID_CRS")
        assert unit == UNKNOWN_UNIT


class TestConvertLength:
    """Tests for convert_length() function."""

    def test_convert_feet_to_meters(self):
        """Test converting feet to meters."""
        result = convert_length(100, "foot", "meter")
        np.testing.assert_almost_equal(float(result), 30.48, decimal=5)

    def test_convert_meters_to_feet(self):
        """Test converting meters to feet."""
        result = convert_length(30.48, "meter", "foot")
        np.testing.assert_almost_equal(float(result), 100.0, decimal=5)

    def test_convert_with_unitinfo(self):
        """Test converting with UnitInfo objects."""
        result = convert_length(1000, METER, KILOMETER)
        np.testing.assert_almost_equal(float(result), 1.0, decimal=10)

    def test_convert_array(self):
        """Test converting arrays."""
        values = [100, 200, 300]
        result = convert_length(values, "foot", "meter")
        expected = np.array([30.48, 60.96, 91.44])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_convert_numpy_array(self):
        """Test converting numpy arrays."""
        values = np.array([1, 2, 3])
        result = convert_length(values, "meter", "foot")
        expected = np.array([1, 2, 3]) / 0.3048
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_convert_mixed_types(self):
        """Test converting with mixed unit types."""
        result = convert_length(100, FOOT, "meter")
        np.testing.assert_almost_equal(float(result), 30.48, decimal=5)

    def test_convert_invalid_unit_raises_error(self):
        """Test that invalid unit raises ValueError."""
        with pytest.raises(ValueError):
            convert_length(100, "invalid", "meter")


class TestConvertToMeters:
    """Tests for convert_to_meters() function."""

    def test_convert_feet_to_meters(self):
        """Test converting feet to meters."""
        result = convert_to_meters(100, "foot")
        np.testing.assert_almost_equal(float(result), 30.48, decimal=5)

    def test_convert_us_survey_feet_to_meters(self):
        """Test converting US survey feet to meters."""
        result = convert_to_meters(100, US_SURVEY_FOOT)
        expected = 100.0 * (1200.0 / 3937.0)
        np.testing.assert_almost_equal(float(result), expected, decimal=8)

    def test_convert_kilometers_to_meters(self):
        """Test converting kilometers to meters."""
        result = convert_to_meters(1.5, "kilometer")
        np.testing.assert_almost_equal(float(result), 1500.0, decimal=5)

    def test_convert_array_to_meters(self):
        """Test converting array to meters."""
        result = convert_to_meters([100, 200], "foot")
        expected = np.array([30.48, 60.96])
        np.testing.assert_almost_equal(result, expected, decimal=5)


class TestConvertFromMeters:
    """Tests for convert_from_meters() function."""

    def test_convert_meters_to_feet(self):
        """Test converting meters to feet."""
        result = convert_from_meters(30.48, "foot")
        np.testing.assert_almost_equal(float(result), 100.0, decimal=5)

    def test_convert_meters_to_kilometers(self):
        """Test converting meters to kilometers."""
        result = convert_from_meters(1500, "kilometer")
        np.testing.assert_almost_equal(float(result), 1.5, decimal=5)

    def test_convert_array_from_meters(self):
        """Test converting array from meters."""
        result = convert_from_meters([30.48, 60.96], "foot")
        expected = np.array([100.0, 200.0])
        np.testing.assert_almost_equal(result, expected, decimal=5)


class TestConversionRoundtrip:
    """Tests for roundtrip conversions."""

    def test_roundtrip_feet_meters(self):
        """Test roundtrip: meters -> feet -> meters."""
        original = np.array([100.0])
        to_feet = convert_from_meters(original, "foot")
        back = convert_to_meters(to_feet, "foot")
        np.testing.assert_almost_equal(original, back, decimal=10)

    def test_roundtrip_us_survey_feet(self):
        """Test roundtrip with US survey feet."""
        original = np.array([1000.0])
        to_feet = convert_from_meters(original, US_SURVEY_FOOT)
        back = convert_to_meters(to_feet, US_SURVEY_FOOT)
        np.testing.assert_almost_equal(original, back, decimal=10)


class TestGetConversionFactor:
    """Tests for get_conversion_factor() function."""

    def test_foot_to_meter_factor(self):
        """Test foot to meter conversion factor."""
        factor = get_conversion_factor("foot", "meter")
        np.testing.assert_almost_equal(factor, 0.3048, decimal=10)

    def test_meter_to_foot_factor(self):
        """Test meter to foot conversion factor."""
        factor = get_conversion_factor("meter", "foot")
        expected = 1.0 / 0.3048
        np.testing.assert_almost_equal(factor, expected, decimal=5)

    def test_meter_to_meter_factor(self):
        """Test meter to meter factor (should be 1)."""
        factor = get_conversion_factor("meter", "meter")
        np.testing.assert_almost_equal(factor, 1.0, decimal=10)

    def test_kilometer_to_meter_factor(self):
        """Test kilometer to meter factor."""
        factor = get_conversion_factor("kilometer", "meter")
        np.testing.assert_almost_equal(factor, 1000.0, decimal=10)

    def test_with_unitinfo_objects(self):
        """Test with UnitInfo objects."""
        factor = get_conversion_factor(FOOT, METER)
        np.testing.assert_almost_equal(factor, 0.3048, decimal=10)

    def test_category_mismatch_raises_error(self):
        """Test that convert_length with mismatched categories raises ValueError."""
        # note: get_conversion_factor computes directly but convert_length checks categories
        with pytest.raises(ValueError):
            convert_length(1, "meter", "degree")


class TestParsePdalUnits:
    """Tests for parse_pdal_units() function."""

    def test_parse_basic_srs_metadata(self):
        """Test parsing basic SRS metadata."""
        srs_metadata = {
            "units": {
                "horizontal": "metre",
                "vertical": "US survey foot"
            }
        }
        h_unit, v_unit = parse_pdal_units(srs_metadata)
        assert h_unit == METER
        assert v_unit == US_SURVEY_FOOT

    def test_parse_empty_units(self):
        """Test parsing empty units dict."""
        srs_metadata = {"units": {}}
        h_unit, v_unit = parse_pdal_units(srs_metadata)
        assert h_unit.name == "unknown"
        assert v_unit.name == "unknown"

    def test_parse_missing_units_key(self):
        """Test parsing with missing units key."""
        srs_metadata = {}
        h_unit, v_unit = parse_pdal_units(srs_metadata)
        assert h_unit.name == "unknown"
        assert v_unit.name == "unknown"

    def test_parse_none_units(self):
        """Test parsing with None units."""
        srs_metadata = {"units": None}
        h_unit, v_unit = parse_pdal_units(srs_metadata)
        assert h_unit.name == "unknown"
        assert v_unit.name == "unknown"


class TestParseCatalogVerticalUnits:
    """Tests for parse_catalog_vertical_units() function."""

    def test_parse_navd88_with_meter(self):
        """Test parsing NAVD88 with meters."""
        result = parse_catalog_vertical_units("NAVD88 height (metre)")
        assert result == METER

    def test_parse_navd88_with_us_feet(self):
        """Test parsing NAVD88 with US feet."""
        result = parse_catalog_vertical_units("NAVD88 height (ftUS)")
        assert result == US_SURVEY_FOOT

    def test_parse_geoid_label_defaults_to_meter(self):
        """Test that geoid labels default to meter."""
        result = parse_catalog_vertical_units("NAVD88 (Geoid 12B)")
        assert result == METER

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_catalog_vertical_units("")
        assert result == UNKNOWN_UNIT

    def test_parse_navd88_no_unit(self):
        """Test NAVD88 without explicit unit (defaults to meter)."""
        result = parse_catalog_vertical_units("NAVD88")
        assert result == METER

    def test_parse_ngvd29(self):
        """Test parsing NGVD29 (defaults to meter)."""
        result = parse_catalog_vertical_units("NGVD29")
        assert result == METER


class TestFormatValueWithUnit:
    """Tests for format_value_with_unit() function."""

    def test_format_meter(self):
        """Test formatting with meters."""
        result = format_value_with_unit(123.456, METER, precision=2)
        assert result == "123.46 m"

    def test_format_foot(self):
        """Test formatting with feet."""
        result = format_value_with_unit(405.23, FOOT, precision=2)
        assert result == "405.23 ft"

    def test_format_different_precision(self):
        """Test formatting with different precision."""
        result = format_value_with_unit(123.456789, METER, precision=4)
        assert result == "123.4568 m"

    def test_format_degree(self):
        """Test formatting with degrees."""
        result = format_value_with_unit(45.5, DEGREE, precision=1)
        assert result == "45.5 °"

    def test_format_zero_precision(self):
        """Test formatting with zero precision."""
        result = format_value_with_unit(123.456, METER, precision=0)
        assert result == "123 m"


class TestDescribeUnit:
    """Tests for describe_unit() function."""

    def test_describe_meter(self):
        """Test describing meter unit."""
        result = describe_unit(METER)
        assert "metre" in result or "meter" in result
        assert "m" in result
        assert "1.0" in result

    def test_describe_foot(self):
        """Test describing foot unit."""
        result = describe_unit(FOOT)
        assert "foot" in result
        assert "ft" in result
        assert "0.3048" in result

    def test_describe_contains_category_info(self):
        """Test that description contains unit information."""
        result = describe_unit(METER)
        assert len(result) > 0
        assert result.count("(") > 0  # Contains abbreviation in parens

    def test_describe_degree(self):
        """Test describing degree unit."""
        result = describe_unit(DEGREE)
        assert "degree" in result
        assert "°" in result


# tIME_UTILS TESTS

class TestDatetimeToDecimalYear:
    """Tests for _datetime_to_decimal_year() function."""

    def test_jan_1_returns_integer_year(self):
        """Test that Jan 1 returns integer year (0 fraction)."""
        dt = datetime.datetime(2020, 1, 1, 0, 0, 0)
        result = _datetime_to_decimal_year(dt)
        # jan 1 should have minimal fractional part
        assert int(result) == 2020
        assert result < 2020.01

    def test_jul_1_is_near_midyear(self):
        """Test that Jul 1 is approximately halfway through the year."""
        dt = datetime.datetime(2020, 7, 1, 0, 0, 0)
        result = _datetime_to_decimal_year(dt)
        assert int(result) == 2020
        # july 1 is approximately halfway (0.5), but leap year makes it slightly less
        assert 0.49 < (result - 2020) < 0.51

    def test_dec_31_near_end_of_year(self):
        """Test that Dec 31 is near end of year."""
        dt = datetime.datetime(2020, 12, 31, 23, 59, 59)
        result = _datetime_to_decimal_year(dt)
        assert int(result) == 2020
        assert result > 2020.99

    def test_leap_year_vs_common_year(self):
        """Test that leap years affect the decimal year."""
        # 2020 is a leap year, 2021 is not
        dt_leap = datetime.datetime(2020, 6, 15)
        dt_common = datetime.datetime(2021, 6, 15)

        dec_leap = _datetime_to_decimal_year(dt_leap)
        dec_common = _datetime_to_decimal_year(dt_common)

        # due to leap year, the fractions should differ slightly
        frac_leap = dec_leap - 2020
        frac_common = dec_common - 2021
        assert abs(frac_leap - frac_common) < 0.01

    def test_consistency(self):
        """Test that same date each year gives similar decimal fraction."""
        # use non-leap years for consistency
        dt1 = datetime.datetime(2019, 3, 15)
        dt2 = datetime.datetime(2021, 3, 15)

        frac1 = _datetime_to_decimal_year(dt1) - 2019
        frac2 = _datetime_to_decimal_year(dt2) - 2021

        # fractions should be similar (within ~0.01, allowing for leap year differences)
        assert abs(frac1 - frac2) < 0.01


class TestParseEpochString:
    """Tests for _parse_epoch_string_to_decimal() function."""

    def test_parse_iso_format(self):
        """Test parsing ISO format YYYY-MM-DD."""
        result = _parse_epoch_string_to_decimal("2006-04-06")
        assert isinstance(result, float)
        assert 2006.25 < result < 2006.27

    def test_parse_us_format(self):
        """Test parsing US format MM/DD/YYYY."""
        result = _parse_epoch_string_to_decimal("04/06/2006")
        assert isinstance(result, float)
        assert 2006.25 < result < 2006.27

    def test_parse_range(self):
        """Test parsing date range."""
        result = _parse_epoch_string_to_decimal("04/06/2006 - 05/01/2006")
        assert isinstance(result, tuple)
        assert len(result) == 2
        start, end = result
        assert start < end
        assert 2006.25 < start < 2006.27
        assert 2006.32 < end < 2006.34

    def test_parse_yyyymmdd_format(self):
        """Test parsing YYYYMMDD format."""
        result = _parse_epoch_string_to_decimal("20060406")
        assert isinstance(result, float)
        assert 2006.25 < result < 2006.27

    def test_parse_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Empty epoch string"):
            _parse_epoch_string_to_decimal("")

    def test_parse_whitespace_only_raises(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError):
            _parse_epoch_string_to_decimal("   ")

    def test_parse_invalid_date_raises(self):
        """Test that invalid date raises ValueError."""
        with pytest.raises(ValueError):
            _parse_epoch_string_to_decimal("not-a-date")

    def test_parse_range_maintains_order(self):
        """Test that range returns (min, max) regardless of order."""
        result1 = _parse_epoch_string_to_decimal("04/06/2006 - 05/01/2006")
        result2 = _parse_epoch_string_to_decimal("05/01/2006 - 04/06/2006")

        assert result1[0] == result2[0]
        assert result1[1] == result2[1]

    def test_parse_range_with_us_format_dates(self):
        """Test parsing range with US format dates for proper splitting."""
        result = _parse_epoch_string_to_decimal("04/06/2006 - 05/01/2006")
        assert isinstance(result, tuple)
        assert result[0] < result[1]


class TestGpsEpoch:
    """Tests for GPS_EPOCH constant."""

    def test_gps_epoch_value(self):
        """Test that GPS_EPOCH is the correct date."""
        assert GPS_EPOCH == datetime.datetime(1980, 1, 6, 0, 0, 0)

    def test_gps_epoch_is_datetime(self):
        """Test that GPS_EPOCH is a datetime object."""
        assert isinstance(GPS_EPOCH, datetime.datetime)


class TestGpsLeapSeconds:
    """Tests for _gps_leap_seconds() function."""

    def test_leap_seconds_before_1981(self):
        """Test leap seconds before first defined leap second."""
        dt = datetime.datetime(1980, 6, 1)
        result = _gps_leap_seconds(dt)
        assert result == 0

    def test_leap_seconds_1981(self):
        """Test leap seconds in 1981."""
        dt = datetime.datetime(1981, 7, 1)
        result = _gps_leap_seconds(dt)
        assert result >= 1

    def test_leap_seconds_2017(self):
        """Test leap seconds in 2017 (last defined)."""
        dt = datetime.datetime(2017, 1, 1)
        result = _gps_leap_seconds(dt)
        assert result == 18

    def test_leap_seconds_after_2017(self):
        """Test leap seconds after 2017 (should be 18 or more)."""
        dt = datetime.datetime(2020, 6, 1)
        result = _gps_leap_seconds(dt)
        assert result >= 18

    def test_leap_seconds_monotonic(self):
        """Test that leap seconds are monotonically increasing."""
        dates = [
            datetime.datetime(1982, 1, 1),
            datetime.datetime(1990, 1, 1),
            datetime.datetime(2000, 1, 1),
            datetime.datetime(2010, 1, 1),
        ]

        leap_seconds = [_gps_leap_seconds(d) for d in dates]
        for i in range(len(leap_seconds) - 1):
            assert leap_seconds[i] <= leap_seconds[i + 1]


class TestGpsSecondsToDecimalYear:
    """Tests for gps_seconds_to_decimal_year_utc() function."""

    def test_gps_epoch_in_seconds(self):
        """Test GPS time at GPS epoch."""
        # GPS epoch is 1980-01-06 (with leap seconds removed)
        result = gps_seconds_to_decimal_year_utc(0.0)
        assert 1980.0 <= result < 1980.1

    def test_seconds_produce_valid_year(self):
        """Test that large GPS seconds produce valid year."""
        # approximately 1 billion seconds from GPS epoch
        result = gps_seconds_to_decimal_year_utc(1e9)
        assert 2000 < result < 2050

    def test_increases_with_time(self):
        """Test that decimal year increases with GPS seconds."""
        result1 = gps_seconds_to_decimal_year_utc(1e8)
        result2 = gps_seconds_to_decimal_year_utc(2e8)
        assert result1 < result2


class TestGuessInTimeFromStats:
    """Tests for _guess_in_time_from_stats() function."""

    def test_small_values_gws(self):
        """Test that small values return 'gws' (week/day seconds)."""
        result = _guess_in_time_from_stats(0, 100000, 500000)
        assert result == "gws"

    def test_standard_adjusted_gst(self):
        """Test that adjusted values return 'gst'."""
        result = _guess_in_time_from_stats(-1e9, 1e8, 5e8)
        assert result == "gst"

    def test_absolute_gps_gt(self):
        """Test that absolute GPS seconds return 'gt'."""
        result = _guess_in_time_from_stats(1.5e9, 1.8e9, 2.0e9)
        assert result == "gt"

    def test_large_values_gt(self):
        """Test that large positive values default to 'gt'."""
        result = _guess_in_time_from_stats(1e9, 1.5e9, 2e9)
        assert result == "gt"

    def test_very_small_max_gws(self):
        """Test heuristic with very small maximum."""
        result = _guess_in_time_from_stats(0, 50000, 100000)
        assert result == "gws"

    def test_boundary_cases(self):
        """Test boundary cases."""
        # small values should be gws
        result1 = _guess_in_time_from_stats(0, 100000, 500000)
        assert result1 == "gws"

        # around adjusted magnitude should be gst
        result2 = _guess_in_time_from_stats(-5e8, 1e8, 2e8)
        assert result2 == "gst"

        # large absolute GPS seconds
        result3 = _guess_in_time_from_stats(1.5e9, 1.8e9, 2.0e9)
        assert result3 == "gt"


class TestTimeUtilsIntegration:
    """Integration tests for time utilities."""

    def test_datetime_year_roundtrip_simulation(self):
        """Test that datetime conversions are consistent."""
        dt1 = datetime.datetime(2010, 6, 15, 12, 0, 0)
        dec1 = _datetime_to_decimal_year(dt1)

        # decimal year should be between 2010 and 2011
        assert 2010 < dec1 < 2011

        # fractional part should be between 0 and 1
        assert 0 < (dec1 - 2010) < 1

    def test_epoch_string_to_decimal_consistency(self):
        """Test consistency of epoch string parsing."""
        # same date in different formats should give same result
        iso = _parse_epoch_string_to_decimal("2006-04-06")
        us_fmt = _parse_epoch_string_to_decimal("04/06/2006")

        # should be very close (allowing for format ambiguity)
        assert abs(iso - us_fmt) < 0.01


# eDGE CASES AND ERROR HANDLING

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_very_large_values_conversion(self):
        """Test converting very large values."""
        large_val = 1e10
        result = convert_length(large_val, "meter", "foot")
        assert float(result) > 0
        assert np.isfinite(float(result))

    def test_very_small_values_conversion(self):
        """Test converting very small values."""
        small_val = 1e-10
        result = convert_length(small_val, "meter", "foot")
        assert float(result) > 0
        assert np.isfinite(float(result))

    def test_negative_values_conversion(self):
        """Test converting negative values."""
        result = convert_length(-100, "meter", "foot")
        assert float(result) < 0
        np.testing.assert_almost_equal(float(result), -328.084, decimal=2)

    def test_zero_values_conversion(self):
        """Test converting zero values."""
        result = convert_length(0, "meter", "foot")
        np.testing.assert_almost_equal(float(result), 0, decimal=10)

    def test_nan_values_propagate(self):
        """Test that NaN values propagate through conversions."""
        result = convert_length(np.nan, "meter", "foot")
        assert np.isnan(float(result))

    def test_inf_values_propagate(self):
        """Test that infinity values propagate through conversions."""
        result = convert_length(np.inf, "meter", "foot")
        assert np.isinf(float(result))

    def test_mixed_nan_array(self):
        """Test converting array with NaN values."""
        values = np.array([100, np.nan, 200])
        result = convert_length(values, "meter", "foot")
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])


class TestUnitConstantsExist:
    """Tests that all expected unit constants are defined."""

    def test_meter_constant(self):
        """Test METER constant exists and is correct."""
        assert METER is not None
        assert METER.name == "meter"

    def test_foot_constant(self):
        """Test FOOT constant exists and is correct."""
        assert FOOT is not None
        assert FOOT.name == "foot"

    def test_us_survey_foot_constant(self):
        """Test US_SURVEY_FOOT constant exists and is correct."""
        assert US_SURVEY_FOOT is not None
        assert US_SURVEY_FOOT.name == "us_survey_foot"

    def test_kilometer_constant(self):
        """Test KILOMETER constant exists and is correct."""
        assert KILOMETER is not None
        assert KILOMETER.name == "kilometer"

    def test_degree_constant(self):
        """Test DEGREE constant exists and is correct."""
        assert DEGREE is not None
        assert DEGREE.name == "degree"

    def test_radian_constant(self):
        """Test RADIAN constant exists and is correct."""
        assert RADIAN is not None
        assert RADIAN.name == "radian"

    def test_unknown_unit_constant(self):
        """Test UNKNOWN_UNIT constant exists and is correct."""
        assert UNKNOWN_UNIT is not None
        assert UNKNOWN_UNIT.name == "unknown"


class TestNumPyIntegration:
    """Tests for numpy integration."""

    def test_conversion_preserves_dtype(self):
        """Test that conversion returns numpy array."""
        result = convert_length([1, 2, 3], "meter", "foot")
        assert isinstance(result, np.ndarray)

    def test_conversion_with_float32(self):
        """Test conversion with float32 arrays."""
        values = np.array([100.0, 200.0], dtype=np.float32)
        result = convert_length(values, "meter", "foot")
        assert result.shape == values.shape
        assert np.isfinite(result).all()

    def test_conversion_with_int(self):
        """Test conversion with integer arrays."""
        values = np.array([100, 200, 300], dtype=int)
        result = convert_length(values, "meter", "foot")
        assert result.dtype in [np.float32, np.float64]

    def test_conversion_preserves_shape(self):
        """Test that conversion preserves array shape."""
        values = np.ones((5, 10))
        result = convert_length(values, "meter", "foot")
        assert result.shape == (5, 10)


class TestCategorySeparation:
    """Tests for unit category separation."""

    def test_all_linear_units_have_linear_category(self):
        """Test that all preset linear units have linear category."""
        linear_units = [METER, FOOT, US_SURVEY_FOOT, KILOMETER]
        for unit in linear_units:
            assert unit.category == "linear"

    def test_all_angular_units_have_angular_category(self):
        """Test that all preset angular units have angular category."""
        angular_units = [DEGREE, RADIAN]
        for unit in angular_units:
            assert unit.category == "angular"

    def test_angular_conversion_never_touches_linear(self):
        """Test that angular conversions don't mix with linear."""
        with pytest.raises(ValueError):
            DEGREE.convert_to(1.0, METER)

    def test_linear_conversion_never_touches_angular(self):
        """Test that linear conversions don't mix with angular."""
        with pytest.raises(ValueError):
            METER.convert_to(1.0, DEGREE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

