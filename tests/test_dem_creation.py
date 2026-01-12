"""
Tests for DEM creation from aligned point clouds (Option 1 workflow)
"""

import pytest
import os
import tempfile
from pathlib import Path
import numpy as np
from topochange import PointCloud, PointCloudPair, Raster


class TestDEMCreation:
    """Tests for creating DEMs from point cloud pairs"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def aligned_pc_pair(self, test_data_dir):
        """Create an aligned PointCloudPair for testing"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        pc1 = PointCloud(str(compare_path))
        pc1.from_file()

        pc2 = PointCloud(str(reference_path))
        pc2.from_file()

        pc_pair = PointCloudPair(pc1, pc2)

        # Transform and align
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

    def test_create_dtm_pair(self, aligned_pc_pair):
        """Test creating DTM pair from aligned point clouds"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dem1, dem2 = aligned_pc_pair.create_dtm_pair(
                resolution=1.0,
                output_dir=tmpdir
            )

            # Check that DEMs were created
            assert dem1 is not None
            assert dem2 is not None

            # Check that DEMs have data
            assert hasattr(dem1, 'data')
            assert hasattr(dem2, 'data')

            # Check that DEM files were created
            assert dem1.filename is not None
            assert dem2.filename is not None
            assert os.path.exists(dem1.filename)
            assert os.path.exists(dem2.filename)

    def test_dem_resolution(self, aligned_pc_pair):
        """Test DEM creation with specified resolution"""
        with tempfile.TemporaryDirectory() as tmpdir:
            resolution = 2.0
            dem1, dem2 = aligned_pc_pair.create_dtm_pair(
                resolution=resolution,
                output_dir=tmpdir
            )

            # Check that resolution is correct
            assert abs(dem1.resolution[0] - resolution) < 0.01
            assert abs(dem2.resolution[0] - resolution) < 0.01


class TestDEMProperties:
    """Tests for DEM properties and metadata"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def test_dems(self, test_data_dir):
        """Create test DEMs"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        pc1 = PointCloud(str(compare_path))
        pc1.from_file()

        pc2 = PointCloud(str(reference_path))
        pc2.from_file()

        pc_pair = PointCloudPair(pc1, pc2)

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

    def test_dem_units(self, test_dems):
        """Test checking and setting DEM units"""
        dem1, dem2 = test_dems

        # Check that units attributes exist
        assert hasattr(dem1, 'horizontal_unit')
        assert hasattr(dem1, 'vertical_unit')

        # Test setting units
        dem1.set_units(vertical_unit="meter")
        dem2.set_units(vertical_unit="meter")

        # Check that units were set
        assert dem1.vertical_unit == "meter"
        assert dem2.vertical_unit == "meter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
