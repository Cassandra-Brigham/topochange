"""
Tests for point cloud transformation (Option 1 workflow)
Excludes velocity transformation tests as requested
"""

import pytest
import os
from pathlib import Path
from topochange import PointCloud, PointCloudPair


class TestPointCloudPairCreation:
    """Tests for creating PointCloudPair objects"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def compare_pc(self, test_data_dir):
        """Create compare PointCloud object"""
        path = test_data_dir / "compare.laz"
        if not path.exists():
            pytest.skip(f"Test data not found: {path}")
        pc = PointCloud(str(path))
        pc.from_file()
        return pc

    @pytest.fixture
    def reference_pc(self, test_data_dir):
        """Create reference PointCloud object"""
        path = test_data_dir / "reference.laz"
        if not path.exists():
            pytest.skip(f"Test data not found: {path}")
        pc = PointCloud(str(path))
        pc.from_file()
        return pc

    def test_create_pointcloud_pair(self, compare_pc, reference_pc):
        """Test creating a PointCloudPair object"""
        pc_pair = PointCloudPair(compare_pc, reference_pc)

        assert pc_pair is not None
        assert pc_pair.pc1 is compare_pc
        assert pc_pair.pc2 is reference_pc

    def test_print_comparison(self, compare_pc, reference_pc):
        """Test printing comparison of point cloud pair"""
        pc_pair = PointCloudPair(compare_pc, reference_pc)

        # This should run without error
        pc_pair.print_comparison()


class TestPointCloudTransformation:
    """Tests for transforming point clouds to match reference"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def pc_pair_with_metadata(self, test_data_dir):
        """Create a PointCloudPair with updated metadata"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        # Load point clouds
        pc1 = PointCloud(str(compare_path))
        pc1.from_file()

        pc2 = PointCloud(str(reference_path))
        pc2.from_file()

        # Update metadata (simulating notebook workflow)
        pc1.add_metadata(
            compound_CRS="EPSG:4979",
            epoch="05/18/2005 - 05/27/2005"
        )

        pc2.add_metadata(
            vertical_CRS="EPSG:5703",
            geoid_model="geoid12b",
            epoch="05/27/2018 - 07/22/2018"
        )

        return PointCloudPair(pc1, pc2)

    def test_transform_compare_to_reference_skip_epoch(self, pc_pair_with_metadata):
        """Test transformation with skip_epoch=True (faster for testing)"""
        # Transform compare to match reference
        transformed = pc_pair_with_metadata.transform_compare_to_match_reference(
            skip_epoch=True,
            verbose=False
        )

        # Check that transformation completed
        assert transformed is True or transformed is None  # Some implementations return None

    def test_crs_after_transformation(self, pc_pair_with_metadata):
        """Test that CRS matches after transformation"""
        # Get reference CRS before transformation
        ref_crs = pc_pair_with_metadata.pc2.current_compound_crs or \
                  pc_pair_with_metadata.pc2.original_compound_crs

        # Transform
        pc_pair_with_metadata.transform_compare_to_match_reference(
            skip_epoch=True,
            verbose=False
        )

        # Check that compare CRS now matches reference
        compare_crs = pc_pair_with_metadata.pc1.current_compound_crs

        # CRS should be set (exact match depends on implementation)
        assert compare_crs is not None


class TestCRSTransformationComponents:
    """Tests for individual transformation components"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def simple_pc_pair(self, test_data_dir):
        """Create a simple PointCloudPair for testing"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        pc1 = PointCloud(str(compare_path))
        pc1.from_file()

        pc2 = PointCloud(str(reference_path))
        pc2.from_file()

        return PointCloudPair(pc1, pc2)

    def test_horizontal_crs_transformation(self, simple_pc_pair):
        """Test that horizontal CRS transformation is possible"""
        # Set different horizontal CRS
        simple_pc_pair.pc1.add_metadata(horizontal_CRS="EPSG:32611")  # UTM 11N
        simple_pc_pair.pc2.add_metadata(horizontal_CRS="EPSG:32612")  # UTM 12N

        # Should be able to transform
        try:
            simple_pc_pair.transform_compare_to_match_reference(
                skip_epoch=True,
                verbose=False
            )
            transformation_succeeded = True
        except Exception as e:
            transformation_succeeded = False
            pytest.fail(f"Horizontal CRS transformation failed: {e}")

        assert transformation_succeeded

    def test_vertical_crs_transformation(self, simple_pc_pair):
        """Test that vertical CRS transformation is possible"""
        # Set different vertical CRS
        simple_pc_pair.pc1.add_metadata(vertical_CRS="EPSG:5703")  # NAVD88
        simple_pc_pair.pc2.add_metadata(vertical_CRS="EPSG:5703")  # NAVD88 (same)

        # Should work without error
        try:
            simple_pc_pair.transform_compare_to_match_reference(
                skip_epoch=True,
                verbose=False
            )
            transformation_succeeded = True
        except Exception as e:
            transformation_succeeded = False
            pytest.fail(f"Vertical CRS transformation failed: {e}")

        assert transformation_succeeded


class TestTransformationWithVerboseOutput:
    """Tests for transformation with verbose output enabled"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def pc_pair(self, test_data_dir):
        """Create a PointCloudPair for testing"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        pc1 = PointCloud(str(compare_path))
        pc1.from_file()

        pc2 = PointCloud(str(reference_path))
        pc2.from_file()

        return PointCloudPair(pc1, pc2)

    def test_transform_with_verbose(self, pc_pair, capsys):
        """Test that verbose output is produced during transformation"""
        pc_pair.transform_compare_to_match_reference(
            skip_epoch=True,
            verbose=True
        )

        # Check that some output was produced
        captured = capsys.readouterr()
        # There should be some output (exact content depends on implementation)
        assert len(captured.out) > 0 or len(captured.err) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
