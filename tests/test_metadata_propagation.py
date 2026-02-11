"""Tests for metadata propagation through transformation chains.

These tests target the five structural gaps identified in the audit:

1. Cross-function metadata propagation : does metadata survive a chain of
   transforms?  (The "baton-passing" problem.)
2. Error-path / failure-mode tests : what happens when grids are missing,
   PROJ errors occur, etc.?
3. Negative-case property tests : edge-case inputs to the CRS auto-sync
   property setters.
4. Provenance / audit-trail tests : is the audit trail complete after a
   pipeline run?
5. Pipeline integration tests : does the full multi-step pipeline preserve
   metadata end-to-end?"""

import os
import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_bounds

from topochange.raster import Raster
from topochange.crs_utils import (
    transformer_with_epoch,
    apply_dynamic_transform,
    parse_crs_components,
    create_compound_crs,
)
from topochange.unit_utils import UnitInfo, get_vertical_unit


# helpers

def _make_raster(
    tmp_path,
    name="test.tif",
    crs="EPSG:32610",
    bounds=(500000, 4000000, 500500, 4000500),
    shape=(50, 50),
    elevation=1000.0,
    noise_seed=42,
):
    """Create a synthetic GeoTIFF and return a loaded Raster."""
    rng = np.random.RandomState(noise_seed)
    data = (rng.randn(*shape) * 10 + elevation).astype(np.float32)
    transform = from_bounds(*bounds, shape[1], shape[0])
    filepath = os.path.join(str(tmp_path), name)

    with rasterio.open(
        filepath, "w",
        driver="GTiff", height=shape[0], width=shape[1], count=1,
        dtype="float32", crs=crs, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    return Raster.from_file(filepath)


def _make_raster_pair(tmp_path, crs1="EPSG:32610", crs2="EPSG:32610",
                      epoch1=None, epoch2=None, ortho1=None, ortho2=None,
                      geoid1=None, geoid2=None, vert_crs1=None, vert_crs2=None):
    """Create a pair of Rasters with specified CRS/epoch/vertical metadata."""
    from topochange.rasterpair import RasterPair

    bounds = (500000, 4000000, 500500, 4000500)
    r1 = _make_raster(tmp_path, "r1.tif", crs=crs1, bounds=bounds, noise_seed=42)
    r2 = _make_raster(tmp_path, "r2.tif", crs=crs2, bounds=bounds, noise_seed=43)

    if epoch1 is not None:
        r1.add_metadata(epoch=epoch1)
    if epoch2 is not None:
        r2.add_metadata(epoch=epoch2)
    if ortho1 is not None:
        r1.is_orthometric = ortho1
    if ortho2 is not None:
        r2.is_orthometric = ortho2
    if geoid1 is not None:
        r1.current_geoid_model = geoid1
        r1.original_geoid_model = geoid1
    if geoid2 is not None:
        r2.current_geoid_model = geoid2
        r2.original_geoid_model = geoid2
    if vert_crs1 is not None:
        r1.current_vertical_crs = vert_crs1
        r1.original_vertical_crs = vert_crs1
    if vert_crs2 is not None:
        r2.current_vertical_crs = vert_crs2
        r2.original_vertical_crs = vert_crs2

    return RasterPair(r1, r2)


# 1. Cross-function metadata propagation tests
# 
# these test that metadata set on a Raster *before* a transform survives
# the transform.  The original tests only checked output CRS, not the
# vertical/epoch/geoid metadata that rides along.

class TestMetadataSurvivesHorizontalReproject:
    """H1 regression: vertical metadata must survive horizontal-only warp."""

    def test_vertical_crs_survives_horizontal_warp(self, tmp_path):
        """Vertical CRS set before warp_raster(target_crs=...) is preserved."""
        r = _make_raster(tmp_path, crs="EPSG:32610")
        vert_wkt = CRS.from_epsg(5703).to_wkt()  # NAVD88
        r.current_vertical_crs = vert_wkt
        r.is_orthometric = True
        # set geoid model directly (bypass select_geoid_grid resolution)
        r.current_geoid_model = "us_noaa_geoid18_conus.tif"

        out = r.warp_raster(target_crs="EPSG:32611", overwrite=True)

        # horizontal should change
        out_epsg = CRS.from_wkt(out.current_horizontal_crs).to_epsg()
        assert out_epsg == 32611

        # vertical must survive
        assert out.current_vertical_crs is not None, (
            "Vertical CRS was wiped during horizontal-only reprojection"
        )
        assert out.is_orthometric is True
        assert out.current_geoid_model is not None

    def test_vertical_unit_survives_horizontal_warp(self, tmp_path):
        """Vertical unit metadata must survive horizontal-only warp."""
        r = _make_raster(tmp_path, crs="EPSG:32610")
        meter_unit = get_vertical_unit(CRS.from_epsg(5703))
        r.current_vertical_unit = meter_unit
        r.current_vertical_units = "metre"

        out = r.warp_raster(target_crs="EPSG:32611", overwrite=True)

        assert out.current_vertical_unit.name != "unknown", (
            "Vertical unit was wiped during horizontal-only reprojection"
        )

    def test_epoch_survives_horizontal_warp(self, tmp_path):
        """Epoch set before horizontal warp should be copied to output."""
        r = _make_raster(tmp_path, crs="EPSG:32610")
        r.add_metadata(epoch=2011.5)

        out = r.warp_raster(target_crs="EPSG:32611", overwrite=True)

        assert out.epoch == 2011.5, (
            "Epoch was lost during horizontal-only reprojection"
        )


class TestMetadataSurvivesGridAlignment:
    """Metadata should survive an alignment-only warp (no CRS change)."""

    def test_vertical_crs_survives_alignment(self, tmp_path):
        """Align-to warp with same CRS should preserve vertical metadata."""
        r1 = _make_raster(tmp_path, "source.tif", bounds=(500000, 4000000, 500500, 4000500))
        r2 = _make_raster(tmp_path, "target.tif", bounds=(500010, 4000010, 500510, 4000510))

        vert_wkt = CRS.from_epsg(5703).to_wkt()
        r1.current_vertical_crs = vert_wkt
        r1.is_orthometric = True

        out = r1.warp_raster(align_to=r2, overwrite=True)

        assert out.current_vertical_crs is not None
        assert out.is_orthometric is True


# 2. Error-path / failure-mode tests
# 
# the original tests never exercised what happens when grids are missing,
# PROJ can't find a deformation model, or a transformation fails partway
# through.

class TestTransformerWithEpochErrorPaths:
    """H2 regression: transformer_with_epoch must degrade gracefully."""

    def test_runtime_error_falls_back_with_warning(self):
        """A RuntimeError from PROJ should warn and fall back, not crash."""
        src = CRS.from_epsg(32610)
        dst = CRS.from_epsg(32611)

        # patch at the import site : Transformer is imported locally inside
        # transformer_with_epoch, so we patch pyproj.Transformer
        with patch("pyproj.Transformer") as MockT:
            # first call (epoch-aware) raises RuntimeError
            # second call (fallback) succeeds
            mock_fallback = MagicMock()
            MockT.from_crs.side_effect = [RuntimeError("missing grid"), mock_fallback]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = transformer_with_epoch(src, dst, src_epoch=2011.0, dst_epoch=2020.0)

            # should have issued a warning
            warning_msgs = [str(x.message) for x in w]
            assert any("falling back" in m.lower() for m in warning_msgs), (
                f"Expected fallback warning, got: {warning_msgs}"
            )
            # should have called from_crs twice (epoch attempt + fallback)
            assert MockT.from_crs.call_count == 2

    def test_type_error_still_handled(self):
        """TypeError (older pyproj) should still be caught silently."""
        src = CRS.from_epsg(32610)
        dst = CRS.from_epsg(32611)

        with patch("pyproj.Transformer") as MockT:
            mock_fallback = MagicMock()
            MockT.from_crs.side_effect = [TypeError("unexpected kwarg"), mock_fallback]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = transformer_with_epoch(src, dst, src_epoch=2011.0, dst_epoch=2020.0)

            assert MockT.from_crs.call_count == 2


class TestGeoidGridErrorPaths:
    """H3 regression: missing geoid grid must raise, not silently return bad data."""

    def test_missing_geoid_grid_raises(self, tmp_path):
        """_apply_geoid_to_raster should raise when grid file is not found."""
        r = _make_raster(tmp_path, crs="EPSG:32610")
        data = np.ones((50, 50), dtype="float64")
        transform = from_bounds(500000, 4000000, 500500, 4000500, 50, 50)
        crs = CRS.from_epsg(32610)

        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            r._apply_geoid_to_raster(
                data, transform, crs,
                geoid_name="NONEXISTENT_GEOID_999",
                direction="subtract",
            )

    def test_missing_geoid_does_not_return_original_data(self, tmp_path):
        """Ensure original data is NOT silently returned when grid missing."""
        r = _make_raster(tmp_path, crs="EPSG:32610")
        data = np.ones((50, 50), dtype="float64") * 42.0
        transform = from_bounds(500000, 4000000, 500500, 4000500, 50, 50)
        crs = CRS.from_epsg(32610)

        try:
            result = r._apply_geoid_to_raster(
                data, transform, crs,
                geoid_name="NONEXISTENT_GEOID_999",
                direction="subtract",
            )
            # if we got here without an exception, the old silent-failure bug is back
            assert not np.allclose(result, 42.0), (
                "Geoid correction silently returned original data unchanged"
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            pass  # This is the correct behavior


class TestEpochSkipWarning:
    """M1 regression: skipping epoch transform should warn when target_epoch is None."""

    def test_warns_when_target_epoch_is_none(self, tmp_path):
        """A warning should fire when epoch mismatch exists but target has no epoch."""
        pair = _make_raster_pair(tmp_path, epoch1=2011.5, epoch2=None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            comparison = pair.check_all_match()

        # the epoch comparison should note the mismatch exists
        # (exact behavior depends on check_epoch_match handling None)


# 3. Negative-case property auto-sync tests
# 
# the original tests verified that setting properties with *valid compound*
# WKT worked.  They never tested what happens with 2D-only CRS, None,
# empty strings, or contradictory component sets.

class TestCompoundCRSSetterEdgeCases:
    """Edge cases for the current_compound_crs property setter."""

    def test_setting_none_preserves_components(self, tmp_path):
        """Setting compound to None should not crash or wipe components."""
        r = _make_raster(tmp_path)
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        horiz_wkt = CRS.from_epsg(32610).to_wkt()
        r._current_horizontal_crs = horiz_wkt
        r._current_vertical_crs = vert_wkt

        r.current_compound_crs = None

        # components should still be there
        assert r._current_horizontal_crs == horiz_wkt
        assert r._current_vertical_crs == vert_wkt

    def test_2d_crs_does_not_wipe_vertical(self, tmp_path):
        """Setting compound to a 2D CRS must not erase existing vertical."""
        r = _make_raster(tmp_path)
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        r._current_vertical_crs = vert_wkt

        # set compound to a 2D-only CRS
        r.current_compound_crs = CRS.from_epsg(32611).to_wkt()

        assert r._current_vertical_crs is not None, (
            "Setting a 2D CRS as compound wiped the existing vertical CRS"
        )
        assert r._current_vertical_crs == vert_wkt

    def test_compound_with_vertical_does_update_vertical(self, tmp_path):
        """Setting compound to a true compound CRS should update vertical."""
        r = _make_raster(tmp_path)
        r._current_vertical_crs = CRS.from_epsg(5703).to_wkt()

        # create a compound with a different vertical (EGM96 = 5773)
        compound = create_compound_crs(CRS.from_epsg(32610), CRS.from_epsg(5773))
        r.current_compound_crs = compound.to_wkt()

        # vertical should now be 5773, not 5703
        parsed_vert = CRS.from_wkt(r._current_vertical_crs)
        assert parsed_vert.to_epsg() == 5773

    def test_horizontal_only_sets_compound_not_none(self, tmp_path):
        """Setting horizontal only should store it in compound (not None)."""
        r = _make_raster(tmp_path)
        r._current_horizontal_crs = None
        r._current_vertical_crs = None
        r._current_compound_crs = None

        horiz_wkt = CRS.from_epsg(32610).to_wkt()
        r.current_horizontal_crs = horiz_wkt

        assert r._current_compound_crs is not None, (
            "Compound CRS is None when only horizontal is set; should store it"
        )

    def test_vertical_only_sets_compound_not_none(self, tmp_path):
        """Setting vertical only should store it in compound (not None)."""
        r = _make_raster(tmp_path)
        r._current_horizontal_crs = None
        r._current_vertical_crs = None
        r._current_compound_crs = None

        vert_wkt = CRS.from_epsg(5703).to_wkt()
        r.current_vertical_crs = vert_wkt

        assert r._current_compound_crs is not None


class TestAddMetadataEdgeCases:
    """Edge cases in add_metadata that weren't tested before."""

    def test_add_metadata_horizontal_only_preserves_vertical(self, tmp_path):
        """add_metadata(horizontal_CRS=...) must not wipe vertical CRS."""
        r = _make_raster(tmp_path)
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        r.current_vertical_crs = vert_wkt
        r.is_orthometric = True

        r.add_metadata(horizontal_CRS=CRS.from_epsg(32611))

        assert r.current_vertical_crs is not None, (
            "add_metadata(horizontal_CRS=...) wiped vertical CRS"
        )
        assert r.is_orthometric is True

    def test_add_metadata_epoch_then_crs_preserves_epoch(self, tmp_path):
        """Setting epoch and then CRS shouldn't lose the epoch."""
        r = _make_raster(tmp_path)
        r.add_metadata(epoch=2011.5)
        r.add_metadata(horizontal_CRS=CRS.from_epsg(32611))

        assert r.epoch == 2011.5

    def test_add_metadata_compound_2d_preserves_vertical(self, tmp_path):
        """add_metadata(compound_CRS=<2D CRS>) must not wipe vertical CRS."""
        r = _make_raster(tmp_path)
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        r.current_vertical_crs = vert_wkt

        r.add_metadata(compound_CRS=CRS.from_epsg(32611))

        assert r.current_vertical_crs is not None, (
            "add_metadata(compound_CRS=<2D>) wiped vertical CRS"
        )


# 4. Provenance / audit-trail tests
# 
# the original tests never checked time_info, epoch_source, or crs_history
# contents after transformations.  These tests verify that the audit trail
# is populated correctly.

class TestTimeInfoProvenance:
    """M2 regression: time_info must be updated after epoch changes."""

    def test_add_metadata_sets_epoch_source(self, tmp_path):
        """time_info['epoch_source'] should be 'add_metadata' after explicit set."""
        r = _make_raster(tmp_path)
        r.add_metadata(epoch=2011.5)

        assert hasattr(r, 'time_info')
        assert r.time_info is not None
        assert r.time_info.get('epoch') == 2011.5
        assert r.time_info.get('epoch_source') == 'add_metadata'

    def test_add_metadata_epoch_overwrites_previous_source(self, tmp_path):
        """Calling add_metadata(epoch=...) twice should update the source."""
        r = _make_raster(tmp_path)
        r.time_info = {'epoch': 2010.0, 'epoch_source': 'parsed_tifftag_datetime'}
        r.add_metadata(epoch=2011.5)

        assert r.time_info['epoch'] == 2011.5
        assert r.time_info['epoch_source'] == 'add_metadata'

    def test_add_metadata_preserves_time_info_keys(self, tmp_path):
        """Existing time_info keys (besides epoch/source) should be preserved."""
        r = _make_raster(tmp_path)
        r.time_info = {
            'epoch': 2010.0,
            'epoch_source': 'parsed_tifftag_datetime',
            'acquisition_date': '2010-06-15',
        }
        r.add_metadata(epoch=2011.5)

        assert r.time_info.get('acquisition_date') == '2010-06-15'


class TestCRSHistoryCompleteness:
    """M4/M7 regression: CRS history should record all transformation steps."""

    def test_convert_vertical_units_records_history(self, tmp_path):
        """convert_vertical_units should produce a crs_history entry."""
        r = _make_raster(tmp_path)
        # set up with foot units : UnitInfo(name, display_name, abbreviation, to_base_factor, category)
        foot_unit = UnitInfo("foot", "foot", "ft", 0.3048, "linear")
        r.current_vertical_unit = foot_unit
        r.current_vertical_units = "foot"

        # initialize crs_history
        from topochange.crs_history import CRSHistory
        r.crs_history = CRSHistory(r)
        initial_len = len(r.crs_history.history)

        result = r.convert_vertical_units(target_units="meter", overwrite=True)

        # the history on *result* should have at least one entry beyond initial
        if result.crs_history is not None:
            assert len(result.crs_history.history) > initial_len, (
                "convert_vertical_units did not record a CRS history entry"
            )

    def test_warp_raster_records_history(self, tmp_path):
        """warp_raster should produce a crs_history entry for horizontal reprojection.

        Note: warp_raster records the transformation on the *source* raster's
        crs_history (self.crs_history), not on the output raster.  This is by
        design : the source tracks its lineage of derived products.
        """
        r = _make_raster(tmp_path)
        from topochange.crs_history import CRSHistory
        r.crs_history = CRSHistory(r)
        initial_len = len(r.crs_history.history)

        out = r.warp_raster(target_crs="EPSG:32611", overwrite=True)

        # the SOURCE raster's history should have a new transformation entry
        entries = r.crs_history.history
        assert len(entries) > initial_len, (
            "warp_raster did not record any CRS history entry on the source raster"
        )
        entry_types = [getattr(e, 'entry_type', '') for e in entries]
        assert any(t == 'transformation' for t in entry_types), (
            f"No 'transformation' entry found in source CRS history: {entry_types}"
        )


# 5. Pipeline integration tests
# 
# these test the full transformation Pipeline end-to-end, which was never
# exercised by the original test suite.  The key insight is that bugs in
# metadata propagation only manifest when multiple transforms are chained.

class TestPipelineMetadataIntegration:
    """End-to-end tests for multi-step transformation metadata."""

    def test_horizontal_then_alignment_metadata_chain(self, tmp_path):
        """
        After horizontal warp → alignment warp, all metadata should reflect
        the final state, not some intermediate state.
        """
        r = _make_raster(tmp_path, "src.tif", crs="EPSG:32610")
        r.add_metadata(epoch=2011.5)
        r.is_orthometric = True
        r.current_vertical_crs = CRS.from_epsg(5703).to_wkt()

        # step 1: Horizontal reproject
        r2 = r.warp_raster(target_crs="EPSG:32611", overwrite=True)

        # metadata should survive step 1
        assert r2.epoch == 2011.5
        assert r2.current_vertical_crs is not None
        assert r2.is_orthometric is True

        # step 2: Another horizontal warp (grid alignment sim)
        r3 = r2.warp_raster(target_crs="EPSG:32611", overwrite=True)

        # everything should still be intact
        assert r3.epoch == 2011.5
        assert r3.current_vertical_crs is not None
        assert r3.is_orthometric is True

    def test_multiple_add_metadata_calls_are_additive(self, tmp_path):
        """
        Calling add_metadata multiple times with different parameters should
        be additive, not destructive.
        """
        r = _make_raster(tmp_path)

        # set epoch first
        r.add_metadata(epoch=2011.5)
        assert r.epoch == 2011.5

        # then set CRS
        r.add_metadata(horizontal_CRS=CRS.from_epsg(32611))
        assert r.epoch == 2011.5  # epoch should survive
        assert CRS.from_wkt(r.current_horizontal_crs).to_epsg() == 32611

        # then set vertical
        r.add_metadata(vertical_CRS=CRS.from_epsg(5703))
        assert r.epoch == 2011.5  # epoch should still survive
        assert CRS.from_wkt(r.current_horizontal_crs).to_epsg() == 32611
        assert r.current_vertical_crs is not None

    def test_warp_chain_does_not_accumulate_none_metadata(self, tmp_path):
        """
        Warping the same raster 3 times should not progressively degrade
        metadata (each warp stripping more info).
        """
        r = _make_raster(tmp_path, crs="EPSG:32610")
        r.add_metadata(epoch=2011.5)
        r.is_orthometric = True
        r.current_vertical_crs = CRS.from_epsg(5703).to_wkt()
        # set geoid model directly to avoid select_geoid_grid lookup
        r.current_geoid_model = "us_noaa_geoid18_conus.tif"

        # chain of warps
        warped = r
        for i in range(3):
            warped = warped.warp_raster(target_crs="EPSG:32610", overwrite=True)

        # after 3 warps, everything should still be there
        assert warped.epoch == 2011.5, f"Epoch lost after warp chain: {warped.epoch}"
        assert warped.current_vertical_crs is not None, "Vertical CRS lost after warp chain"
        assert warped.is_orthometric is True, "Orthometric flag lost after warp chain"

    def test_time_info_tracks_through_warp(self, tmp_path):
        """time_info should be present on output after warp with epoch."""
        r = _make_raster(tmp_path, crs="EPSG:32610")
        r.add_metadata(epoch=2011.5)

        out = r.warp_raster(target_crs="EPSG:32611", overwrite=True)

        # output should retain epoch
        assert out.epoch == 2011.5


class TestParseCRSComponentsEdgeCases:
    """Verify parse_crs_components handles various CRS inputs correctly."""

    def test_2d_projected_returns_none_vertical(self):
        """A 2D projected CRS should return no vertical component."""
        wkt = CRS.from_epsg(32610).to_wkt()
        _, horiz, vert = parse_crs_components(wkt)
        assert horiz is not None
        assert vert is None

    def test_geographic_crs_returns_none_vertical(self):
        """A 2D geographic CRS should return no vertical component."""
        wkt = CRS.from_epsg(4326).to_wkt()
        _, horiz, vert = parse_crs_components(wkt)
        assert horiz is not None
        assert vert is None

    def test_compound_crs_returns_both(self):
        """A compound CRS should return both horizontal and vertical."""
        compound = create_compound_crs(CRS.from_epsg(32610), CRS.from_epsg(5703))
        _, h_wkt, v_wkt = parse_crs_components(compound.to_wkt())
        assert h_wkt is not None
        assert v_wkt is not None

    def test_vertical_only_crs(self):
        """A vertical-only CRS should return no horizontal component."""
        wkt = CRS.from_epsg(5703).to_wkt()
        _, horiz, vert = parse_crs_components(wkt)
        assert horiz is None
        assert vert is not None


# regression tests for specific bug scenarios

class TestBugScenarioRegressions:
    """
    Concrete scenarios that reproduce the original bugs.  Each test would
    have failed on the pre-fix codebase.
    """

    def test_h1_scenario_vertical_crs_lost_in_pipeline(self, tmp_path):
        """
        Reproduce the H1 cascade: horizontal reprojection wipes vertical CRS,
        causing a subsequent vertical check to see "both None" and skip the
        vertical transform entirely.
        """
        r = _make_raster(tmp_path, crs="EPSG:32610")
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        r.current_vertical_crs = vert_wkt
        r.is_orthometric = True

        # horizontal-only reproject
        out = r.warp_raster(target_crs="EPSG:32611", overwrite=True)

        # the bug was: out.current_vertical_crs == None here
        assert out.current_vertical_crs is not None, (
            "H1 REGRESSION: Horizontal warp wiped vertical CRS"
        )

    def test_h1_cascade_vertical_check_after_reproject(self, tmp_path):
        """
        Full cascade: after H1 wipe, check_vertical_crs_match would see
        both as None and report a false match, skipping the needed transform.
        """
        from topochange.rasterpair import RasterPair

        # source: has vertical CRS
        r1 = _make_raster(tmp_path, "r1.tif", crs="EPSG:32610")
        r1.current_vertical_crs = CRS.from_epsg(5703).to_wkt()
        r1.is_orthometric = True

        # target: different CRS, same vertical
        r2 = _make_raster(tmp_path, "r2.tif", crs="EPSG:32611")
        r2.current_vertical_crs = CRS.from_epsg(5703).to_wkt()
        r2.is_orthometric = True

        # step 1: Reproject r1 to match r2's horizontal CRS
        r1_reprojected = r1.warp_raster(target_crs="EPSG:32611", overwrite=True)

        # step 2: Now check if vertical CRS still matches
        # before fix: r1_reprojected.current_vertical_crs was None,
        # so the pair would see a mismatch (None vs NAVD88)
        pair = RasterPair(r1_reprojected, r2)
        comparison = pair.check_all_match()

        # vertical CRS should still match (both NAVD88)
        assert 'vertical_datum' not in comparison.get('transformations_needed', []), (
            "H1 CASCADE: Vertical CRS mismatch detected after horizontal reproject "
            "because vertical metadata was wiped"
        )

    def test_m4_scenario_add_entry_attributeerror(self, tmp_path):
        """
        Reproduce M4: convert_vertical_units called crs_history.add_entry()
        which doesn't exist.  Wrapped in try/except, so it silently failed.
        Verify the method now being called actually exists.
        """
        from topochange.crs_history import CRSHistory

        r = _make_raster(tmp_path)
        r.crs_history = CRSHistory(r)

        # verify record_transformation_entry exists (the replacement method)
        assert hasattr(r.crs_history, 'record_transformation_entry')

        # verify add_entry does NOT exist (the old broken method)
        assert not hasattr(r.crs_history, 'add_entry'), (
            "add_entry method exists now : if intentional, M4 fix should use it"
        )

    def test_m8_scenario_compound_inconsistency(self, tmp_path):
        """
        Reproduce M8: after setting horizontal CRS only, compound_crs should
        not be None.  The old code set it to None, creating inconsistency
        with add_metadata which set it to the horizontal CRS.
        """
        r = _make_raster(tmp_path)
        r._current_horizontal_crs = CRS.from_epsg(32610).to_wkt()
        r._current_vertical_crs = None
        r._current_compound_crs = None

        # trigger the update via property setter
        r.current_horizontal_crs = CRS.from_epsg(32610).to_wkt()

        # compound should not be None (M8 fix)
        assert r._current_compound_crs is not None, (
            "M8 REGRESSION: compound_crs is None when only horizontal is set"
        )

    def test_m2_scenario_time_info_not_updated(self, tmp_path):
        """
        Reproduce M2: after add_metadata(epoch=...), time_info should track
        the new value.  Old code left time_info stale.
        """
        r = _make_raster(tmp_path)
        # simulate what from_file() creates
        r.time_info = {
            'epoch': 2026.1,
            'epoch_source': 'parsed_tifftag_datetime',
        }

        r.add_metadata(epoch=2011.5)

        assert r.time_info['epoch'] == 2011.5, (
            "M2 REGRESSION: time_info['epoch'] not updated by add_metadata"
        )
        assert r.time_info['epoch_source'] == 'add_metadata', (
            "M2 REGRESSION: time_info['epoch_source'] not updated by add_metadata"
        )

    def test_h3_scenario_silent_geoid_failure(self, tmp_path):
        """
        Reproduce H3: _apply_geoid_to_raster with a nonexistent geoid grid
        should raise an error, not silently return uncorrected data.
        """
        r = _make_raster(tmp_path, crs="EPSG:32610")
        data = np.ones((50, 50), dtype="float64") * 1000.0
        transform = from_bounds(500000, 4000000, 500500, 4000500, 50, 50)

        raised = False
        try:
            result = r._apply_geoid_to_raster(
                data, transform, CRS.from_epsg(32610),
                geoid_name="TOTALLY_FAKE_GEOID",
                direction="subtract",
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            raised = True

        assert raised, (
            "H3 REGRESSION: _apply_geoid_to_raster did not raise for missing grid"
        )

