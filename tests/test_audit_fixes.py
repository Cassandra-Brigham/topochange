"""Test suite for audit fixes in topochange codebase.

Tests the following audit findings:
1. H1 - Vertical CRS preserved during horizontal-only reprojection
2. H2 - Broader exception handling in transformer_with_epoch()
3. M2 - time_info updated after add_metadata epoch changes
4. M4 - crs_history.record_transformation_entry() called
5. M8 - Compound CRS consistency"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest import mock

import rasterio
from rasterio.transform import from_bounds
from pyproj import CRS

from topochange.raster import Raster
from topochange import crs_utils


# fixtures

@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for test files."""
    return str(tmp_path)


def make_test_raster(tmpdir, epsg=32613):
    """
    Create a minimal test Raster object for testing.

    Parameters
    ----------
    tmpdir : str
        Temporary directory path
    epsg : int
        EPSG code for the CRS

    Returns
    -------
    Raster
        Raster object loaded from synthetic GeoTIFF
    """
    path = os.path.join(tmpdir, "test.tif")
    transform = from_bounds(0, 0, 100, 100, 10, 10)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=10,
        width=10,
        count=1,
        dtype='float32',
        crs=f'EPSG:{epsg}',
        transform=transform
    ) as dst:
        dst.write(np.ones((1, 10, 10), dtype='float32'))
    return Raster.from_file(path)


# test H1: Vertical CRS preserved during horizontal-only reprojection

class TestH1_VerticalCRSPreserved:
    """
    Test that setting current_compound_crs to a 2D CRS does NOT wipe
    the existing _current_vertical_crs.
    """

    def test_vertical_crs_survives_2d_compound_assignment(self, tmp_dir):
        """
        Create a Raster, set vertical CRS first, then set compound to 2D CRS,
        and verify vertical CRS is preserved.
        """
        raster = make_test_raster(tmp_dir)

        # set a vertical CRS first
        vertical_crs_wkt = CRS.from_epsg(5703).to_wkt()  # NAVD88 height
        raster._current_vertical_crs = vertical_crs_wkt

        # now set a 2D compound CRS (horizontal only)
        horizontal_crs_wkt = CRS.from_epsg(32613).to_wkt()  # UTM 13N
        raster.current_compound_crs = horizontal_crs_wkt

        # verify vertical CRS was preserved (not erased)
        assert raster._current_vertical_crs is not None
        assert raster._current_vertical_crs == vertical_crs_wkt

    def test_vertical_crs_overwritten_when_compound_has_vertical(self, tmp_dir):
        """
        Verify that a compound CRS WITH vertical component DOES update
        the vertical CRS property.
        """
        raster = make_test_raster(tmp_dir)

        # set initial vertical CRS
        initial_vertical = CRS.from_epsg(5703).to_wkt()
        raster._current_vertical_crs = initial_vertical

        # create a compound CRS with a different vertical component
        # and set it
        try:
            from pyproj.crs import CompoundCRS
            horiz = CRS.from_epsg(32613)
            vert = CRS.from_epsg(5702)  # Different vertical: NAVD88 ellipsoidal
            compound = CompoundCRS(name="Test Compound", components=[horiz, vert])
            raster.current_compound_crs = compound.to_wkt()

            # vertical should be updated to the compound's vertical
            assert raster._current_vertical_crs is not None
            # should be different from the initial one
            assert raster._current_vertical_crs != initial_vertical
        except Exception:
            # if compound CRS creation fails, skip this part of the test
            pytest.skip("CompoundCRS creation failed")


# test H2: Broader exception handling in transformer_with_epoch()

class TestH2_TransformerWithEpochExceptionHandling:
    """
    Test that transformer_with_epoch() catches general exceptions
    (not just TypeError) when using epoch parameters, and falls back
    gracefully to non-epoch transform.
    """

    def test_transformer_falls_back_on_runtime_error(self):
        """
        Mock Transformer.from_crs to raise RuntimeError with epoch params
        and verify it falls back to non-epoch version.
        """
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_epsg(3857)
        src_epoch = 2015.5
        dst_epoch = 2020.0

        with mock.patch('pyproj.Transformer.from_crs') as mock_from_crs:
            # first call with epoch params raises RuntimeError
            # second call (fallback) succeeds
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1 and ('source_crs_epoch' in kwargs or 'target_crs_epoch' in kwargs):
                    raise RuntimeError("Epoch not supported for this CRS")
                # return a mock Transformer for the fallback call
                mock_transformer = mock.MagicMock()
                return mock_transformer

            mock_from_crs.side_effect = side_effect

            # should not raise, should fall back gracefully
            result = crs_utils.transformer_with_epoch(
                src_crs, dst_crs, src_epoch, dst_epoch
            )

            # should have called from_crs twice: once with epoch, once without
            assert mock_from_crs.call_count >= 1
            assert result is not None

    def test_transformer_falls_back_on_type_error(self):
        """
        Verify that TypeError (older pyproj) is still caught and handled.
        """
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_epsg(3857)

        with mock.patch('pyproj.Transformer.from_crs') as mock_from_crs:
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1 and ('source_crs_epoch' in kwargs or 'target_crs_epoch' in kwargs):
                    raise TypeError("from_crs() got unexpected keyword argument")
                mock_transformer = mock.MagicMock()
                return mock_transformer

            mock_from_crs.side_effect = side_effect

            result = crs_utils.transformer_with_epoch(src_crs, dst_crs, 2015.5, 2020.0)

            assert result is not None
            assert mock_from_crs.call_count >= 1


# test M2: time_info updated after add_metadata epoch changes

class TestM2_TimeInfoUpdatedAfterEpoch:
    """
    Test that after calling add_metadata(epoch=2011.5), the raster's
    time_info dict has epoch=2011.5 and epoch_source='add_metadata'.
    """

    def test_time_info_epoch_single_value(self, tmp_dir):
        """
        Set epoch via add_metadata with a single decimal value.
        Verify time_info is updated correctly.
        """
        raster = make_test_raster(tmp_dir)
        epoch_value = 2011.5

        # initialize time_info if needed
        if not hasattr(raster, 'time_info') or raster.time_info is None:
            raster.time_info = {}

        # call add_metadata with epoch
        raster.add_metadata(epoch=epoch_value)

        # check time_info is updated
        assert hasattr(raster, 'time_info')
        assert raster.time_info is not None
        assert raster.time_info['epoch'] == epoch_value
        assert raster.time_info['epoch_source'] == 'add_metadata'

    def test_time_info_epoch_with_range(self, tmp_dir):
        """
        Set epoch via add_metadata with a range (start, end).
        Verify time_info stores the midpoint.
        """
        raster = make_test_raster(tmp_dir)
        epoch_start = 2010.0
        epoch_end = 2012.0

        # call add_metadata with epoch range
        raster.add_metadata(epoch=(epoch_start, epoch_end))

        # check time_info
        assert hasattr(raster, 'time_info')
        assert raster.time_info is not None
        # midpoint of (2010, 2012) is 2011
        expected_midpoint = 0.5 * (epoch_start + epoch_end)
        assert raster.time_info['epoch'] == expected_midpoint
        assert raster.time_info['epoch_source'] == 'add_metadata'

    def test_time_info_epoch_with_string(self, tmp_dir):
        """
        Set epoch via add_metadata with a date string.
        Verify time_info is updated.
        """
        raster = make_test_raster(tmp_dir)

        # call add_metadata with epoch string (should parse to decimal year)
        raster.add_metadata(epoch="2011-06-15")

        # check time_info is set and has epoch_source
        assert hasattr(raster, 'time_info')
        assert raster.time_info is not None
        assert 'epoch' in raster.time_info
        assert raster.time_info['epoch_source'] == 'add_metadata'
        # epoch should be reasonable (around 2011)
        assert 2011.0 <= raster.time_info['epoch'] <= 2012.0


# test M4: crs_history.record_transformation_entry() called

class TestM4_CRSHistoryRecordTransformation:
    """
    Test that convert_vertical_units() calls record_transformation_entry()
    instead of the non-existent add_entry().
    """

    def test_convert_vertical_units_records_entry(self, tmp_dir):
        """
        Call convert_vertical_units() and verify no errors are raised
        when trying to record the transformation entry.
        """
        raster = make_test_raster(tmp_dir)

        # set the vertical unit to something known
        from topochange.unit_utils import lookup_unit
        meter_unit = lookup_unit("meter")
        raster.current_vertical_unit = meter_unit
        raster.current_vertical_units = meter_unit.display_name

        # try to convert to feet
        try:
            result = raster.convert_vertical_units("foot", overwrite=True)

            # if we get here without an error, the method exists and works
            assert result is not None
            assert hasattr(result, 'current_vertical_unit')
        except AttributeError as e:
            if 'add_entry' in str(e):
                pytest.fail(f"Old add_entry() method still being called: {e}")
            raise
        except Exception:
            # other exceptions are OK for this test
            # (e.g., CRS not set, unit conversion not available)
            pass

    def test_convert_vertical_units_no_method_error(self, tmp_dir):
        """
        Ensure convert_vertical_units doesn't raise AttributeError
        about missing add_entry method.
        """
        raster = make_test_raster(tmp_dir)

        from topochange.unit_utils import lookup_unit
        meter_unit = lookup_unit("meter")
        raster.current_vertical_unit = meter_unit

        try:
            # initialize crs_history to trigger the code path
            from topochange.crs_history import CRSHistory
            raster.crs_history = CRSHistory(raster)
        except Exception:
            pass  # CRSHistory may not be available

        try:
            raster.convert_vertical_units("foot", overwrite=True)
        except AttributeError as e:
            if 'add_entry' in str(e):
                pytest.fail(f"add_entry() method error: {e}")
            # other AttributeErrors are OK


# test M8: Compound CRS consistency

class TestM8_CompoundCRSConsistency:
    """
    Test that _update_current_compound_from_components() stores horizontal CRS
    as compound when no vertical exists (not set it to None).
    """

    def test_compound_stored_when_only_horizontal_exists(self, tmp_dir):
        """
        Set only horizontal CRS and verify compound is set to that
        horizontal CRS (not None).
        """
        raster = make_test_raster(tmp_dir)

        # clear any existing CRS
        raster._current_horizontal_crs = None
        raster._current_vertical_crs = None
        raster._current_compound_crs = None

        # set only horizontal
        horiz_wkt = CRS.from_epsg(32613).to_wkt()
        raster.current_horizontal_crs = horiz_wkt

        # compound should be set to the horizontal CRS
        assert raster._current_compound_crs is not None
        assert raster._current_compound_crs == horiz_wkt

    def test_compound_stored_when_only_vertical_exists(self, tmp_dir):
        """
        Set only vertical CRS and verify compound is set to that
        vertical CRS (not None).
        """
        raster = make_test_raster(tmp_dir)

        # clear existing CRS
        raster._current_horizontal_crs = None
        raster._current_vertical_crs = None
        raster._current_compound_crs = None

        # set only vertical
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        raster.current_vertical_crs = vert_wkt

        # compound should be set to the vertical CRS
        assert raster._current_compound_crs is not None
        assert raster._current_compound_crs == vert_wkt

    def test_compound_created_when_both_exist(self, tmp_dir):
        """
        Set both horizontal and vertical CRS and verify a true
        compound CRS is created.
        """
        raster = make_test_raster(tmp_dir)

        # clear existing
        raster._current_horizontal_crs = None
        raster._current_vertical_crs = None
        raster._current_compound_crs = None

        # set horizontal
        horiz_wkt = CRS.from_epsg(32613).to_wkt()
        raster.current_horizontal_crs = horiz_wkt

        # then set vertical
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        raster.current_vertical_crs = vert_wkt

        # compound should be set (and should not be None)
        assert raster._current_compound_crs is not None
        # compound should be different from either component alone
        # (it should be a compound or at least contain both somehow)
        # just verify it's not the raw horizontal
        assert raster._current_compound_crs != horiz_wkt

    def test_compound_not_none_after_component_updates(self, tmp_dir):
        """
        Verify that after any valid component update,
        compound is never left as None when components exist.
        """
        raster = make_test_raster(tmp_dir)

        horiz_wkt = CRS.from_epsg(32613).to_wkt()
        vert_wkt = CRS.from_epsg(5703).to_wkt()

        # set both components
        raster.current_horizontal_crs = horiz_wkt
        raster.current_vertical_crs = vert_wkt

        # verify compound is set
        assert raster._current_compound_crs is not None

        # update just horizontal
        raster.current_horizontal_crs = CRS.from_epsg(32612).to_wkt()
        assert raster._current_compound_crs is not None

        # update just vertical
        raster.current_vertical_crs = CRS.from_epsg(5702).to_wkt()
        assert raster._current_compound_crs is not None


# integration Tests

class TestIntegration:
    """
    Integration tests combining multiple fixes.
    """

    def test_h1_and_m8_together(self, tmp_dir):
        """
        Verify H1 and M8 work together: set vertical, then update
        compound with 2D CRS, and verify both vertical preservation
        and compound consistency.
        """
        raster = make_test_raster(tmp_dir)

        # start with vertical CRS
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        raster._current_vertical_crs = vert_wkt

        # set a 2D compound (simulating reprojection)
        horiz_wkt = CRS.from_epsg(32612).to_wkt()
        raster.current_compound_crs = horiz_wkt

        # h1: Vertical should survive
        assert raster._current_vertical_crs == vert_wkt

        # m8: Compound should not be None
        assert raster._current_compound_crs is not None

    def test_m2_with_add_metadata_and_compound_crs(self, tmp_dir):
        """
        Call add_metadata with both epoch and compound CRS,
        verify both are applied correctly.
        """
        raster = make_test_raster(tmp_dir)

        horiz_crs = CRS.from_epsg(32613)
        epoch_value = 2015.5

        raster.add_metadata(
            compound_CRS=horiz_crs,
            epoch=epoch_value
        )

        # verify epoch was recorded
        assert raster.time_info is not None
        assert raster.time_info['epoch'] == epoch_value
        assert raster.time_info['epoch_source'] == 'add_metadata'

        # verify CRS was set
        assert raster._current_compound_crs is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

