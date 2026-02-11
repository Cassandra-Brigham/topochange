"""Tests for DEM creation from aligned point clouds (Option 1 workflow)
Uses synthetic LAZ data from conftest fixtures."""

import pytest
import os
import tempfile
import numpy as np
from topochange import PointCloud, PointCloudPair, Raster
from skip_markers import requires_pdal, requires_small_gicp


class TestDEMCreation:
    """Tests for creating DEMs from point cloud pairs"""

    @pytest.fixture
    def aligned_pc_pair(self, compare_pc, reference_pc):
        """Create an aligned PointCloudPair for testing"""
        pc_pair = PointCloudPair(compare_pc, reference_pc)

        # transform and align
        try:
            pc_pair.transform_compare_to_match_reference(skip_epoch=True, verbose=False)
            pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=1.0,
                max_points=500_000,
            )
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")

        return pc_pair

    @requires_pdal
    @requires_small_gicp
    def test_create_dtm_pair(self, aligned_pc_pair):
        """Test creating DTM pair from aligned point clouds"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dem1, dem2 = aligned_pc_pair.create_dtm_pair(
                resolution=1.0,
                output_dir=tmpdir
            )

            # check that DEMs were created
            assert dem1 is not None
            assert dem2 is not None

            # check that DEMs have data
            assert hasattr(dem1, 'data')
            assert hasattr(dem2, 'data')

            # check that DEM files were created
            assert dem1.filename is not None
            assert dem2.filename is not None
            assert os.path.exists(dem1.filename)
            assert os.path.exists(dem2.filename)

    @requires_pdal
    @requires_small_gicp
    def test_dem_resolution(self, aligned_pc_pair):
        """Test DEM creation with specified resolution"""
        with tempfile.TemporaryDirectory() as tmpdir:
            resolution = 2.0
            dem1, dem2 = aligned_pc_pair.create_dtm_pair(
                resolution=resolution,
                output_dir=tmpdir
            )

            # check that resolution is correct
            assert abs(dem1.resolution[0] - resolution) < 0.01
            assert abs(dem2.resolution[0] - resolution) < 0.01


class TestDEMProperties:
    """Tests for DEM properties and metadata"""

    @pytest.fixture
    def test_dems(self, compare_pc, reference_pc):
        """Create test DEMs"""
        pc_pair = PointCloudPair(compare_pc, reference_pc)

        try:
            pc_pair.transform_compare_to_match_reference(skip_epoch=True, verbose=False)
            pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=1.0,
                max_points=500_000,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                dem1, dem2 = pc_pair.create_dtm_pair(
                    resolution=1.0,
                    output_dir=tmpdir
                )

                return dem1, dem2

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")

    @requires_pdal
    @requires_small_gicp
    def test_dem_units(self, test_dems):
        """Test checking and setting DEM units"""
        dem1, dem2 = test_dems

        # check that units attributes exist
        assert hasattr(dem1, 'horizontal_unit')
        assert hasattr(dem1, 'vertical_unit')

        # test setting units
        dem1.set_units(vertical_unit="meter")
        dem2.set_units(vertical_unit="meter")

        # check that units were set
        assert dem1.vertical_unit == "meter"
        assert dem2.vertical_unit == "meter"

