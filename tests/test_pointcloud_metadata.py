"""
Tests for point cloud metadata extraction and updating (Option 1 workflow)
"""

import pytest
import os
from pathlib import Path
from topochange import PointCloud


class TestPointCloudMetadataExtraction:
    """Tests for extracting metadata from point cloud files"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def compare_pc_path(self, test_data_dir):
        """Return path to compare point cloud"""
        path = test_data_dir / "compare.laz"
        if not path.exists():
            pytest.skip(f"Test data not found: {path}")
        return str(path)

    @pytest.fixture
    def reference_pc_path(self, test_data_dir):
        """Return path to reference point cloud"""
        path = test_data_dir / "reference.laz"
        if not path.exists():
            pytest.skip(f"Test data not found: {path}")
        return str(path)

    def test_create_pointcloud_object(self, compare_pc_path):
        """Test creating a PointCloud object from file"""
        pc = PointCloud(compare_pc_path)
        assert pc is not None
        assert pc.filename == compare_pc_path

    def test_read_metadata_from_file(self, compare_pc_path):
        """Test reading metadata from point cloud file"""
        pc = PointCloud(compare_pc_path)
        pc.from_file()

        # Check that basic metadata is loaded
        assert pc.original_compound_crs is not None or pc.current_compound_crs is not None
        assert hasattr(pc, 'bounds')
        assert hasattr(pc, 'total_points')

    def test_metadata_attributes_exist(self, compare_pc_path):
        """Test that expected metadata attributes exist after loading"""
        pc = PointCloud(compare_pc_path)
        pc.from_file()

        # Check for expected attributes
        assert hasattr(pc, 'original_compound_crs')
        assert hasattr(pc, 'original_horizontal_crs')
        assert hasattr(pc, 'original_vertical_crs')
        assert hasattr(pc, 'current_compound_crs')
        assert hasattr(pc, 'current_horizontal_crs')
        assert hasattr(pc, 'current_vertical_crs')
        assert hasattr(pc, 'geoid_model')
        assert hasattr(pc, 'epoch')
        assert hasattr(pc, 'is_orthometric')


class TestPointCloudMetadataUpdate:
    """Tests for updating point cloud metadata"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def compare_pc(self, test_data_dir):
        """Create a PointCloud object for testing"""
        path = test_data_dir / "compare.laz"
        if not path.exists():
            pytest.skip(f"Test data not found: {path}")
        pc = PointCloud(str(path))
        pc.from_file()
        return pc

    def test_update_compound_crs(self, compare_pc):
        """Test updating compound CRS"""
        original_crs = compare_pc.current_compound_crs

        # Update with a known CRS
        test_crs = "EPSG:4979"
        compare_pc.add_metadata(compound_CRS=test_crs)

        # Check that CRS was updated
        assert compare_pc.current_compound_crs is not None
        # The CRS should have changed (or been set)
        assert compare_pc.current_compound_crs != original_crs or original_crs is None

    def test_update_horizontal_crs(self, compare_pc):
        """Test updating horizontal CRS"""
        test_crs = "EPSG:32611"  # UTM Zone 11N
        compare_pc.add_metadata(horizontal_CRS=test_crs)

        # Check that horizontal CRS was updated
        assert compare_pc.current_horizontal_crs is not None
        # Check that compound CRS was also updated
        assert compare_pc.current_compound_crs is not None

    def test_update_vertical_crs(self, compare_pc):
        """Test updating vertical CRS"""
        test_crs = "EPSG:5703"  # NAVD88
        compare_pc.add_metadata(vertical_CRS=test_crs)

        # Check that vertical CRS was updated
        assert compare_pc.current_vertical_crs is not None
        # Check that compound CRS was also updated
        assert compare_pc.current_compound_crs is not None

    def test_update_geoid_model(self, compare_pc):
        """Test updating geoid model"""
        from topochange.geoid_utils import select_geoid_grid

        geoid_name = "geoid12b"
        geoid_grid, _ = select_geoid_grid(geoid_name, verbose=False)

        compare_pc.add_metadata(geoid_model=geoid_grid)

        # Check that geoid model was updated
        assert compare_pc.geoid_model is not None
        assert geoid_name.lower() in str(compare_pc.geoid_model).lower()

    def test_update_epoch_single_date(self, compare_pc):
        """Test updating epoch with a single date"""
        epoch_str = "05/18/2005"
        compare_pc.add_metadata(epoch=epoch_str)

        # Check that epoch was updated
        assert compare_pc.epoch is not None
        # Epoch should be a decimal year
        assert isinstance(compare_pc.epoch, (int, float))
        assert 2005.0 <= compare_pc.epoch <= 2006.0

    def test_update_epoch_date_range(self, compare_pc):
        """Test updating epoch with a date range"""
        epoch_str = "05/18/2005 - 05/27/2005"
        compare_pc.add_metadata(epoch=epoch_str)

        # Check that epoch was updated (should be midpoint)
        assert compare_pc.epoch is not None
        assert isinstance(compare_pc.epoch, (int, float))
        assert 2005.0 <= compare_pc.epoch <= 2006.0

    def test_update_multiple_metadata_fields(self, compare_pc):
        """Test updating multiple metadata fields at once"""
        compare_pc.add_metadata(
            horizontal_CRS="EPSG:32611",
            vertical_CRS="EPSG:5703",
            epoch="05/18/2005"
        )

        # Check that all fields were updated
        assert compare_pc.current_horizontal_crs is not None
        assert compare_pc.current_vertical_crs is not None
        assert compare_pc.epoch is not None

    def test_metadata_consistency_after_updates(self, compare_pc):
        """Test that metadata remains consistent after multiple updates"""
        # Update horizontal and vertical separately
        compare_pc.add_metadata(horizontal_CRS="EPSG:32611")
        compare_pc.add_metadata(vertical_CRS="EPSG:5703")

        # Both should be reflected in the compound CRS
        assert compare_pc.current_compound_crs is not None
        assert compare_pc.current_horizontal_crs is not None
        assert compare_pc.current_vertical_crs is not None


class TestReferencePointCloudMetadata:
    """Tests for reference point cloud metadata handling"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def reference_pc(self, test_data_dir):
        """Create a reference PointCloud object for testing"""
        path = test_data_dir / "reference.laz"
        if not path.exists():
            pytest.skip(f"Test data not found: {path}")
        pc = PointCloud(str(path))
        pc.from_file()
        return pc

    def test_load_reference_pointcloud(self, reference_pc):
        """Test loading reference point cloud"""
        assert reference_pc is not None
        assert reference_pc.current_compound_crs is not None or reference_pc.original_compound_crs is not None

    def test_update_reference_metadata(self, reference_pc):
        """Test updating reference point cloud metadata"""
        reference_pc.add_metadata(
            horizontal_CRS="EPSG:32611",
            vertical_CRS="EPSG:5703",
            geoid_model="geoid12b",
            epoch="05/27/2018 - 07/22/2018"
        )

        # Verify all fields were updated
        assert reference_pc.current_horizontal_crs is not None
        assert reference_pc.current_vertical_crs is not None
        assert reference_pc.geoid_model is not None
        assert reference_pc.epoch is not None

    def test_print_metadata(self, reference_pc):
        """Test that print_metadata runs without error"""
        # This should not raise an exception
        reference_pc.print_metadata()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
