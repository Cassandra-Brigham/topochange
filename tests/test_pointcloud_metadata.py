"""Tests for point cloud metadata extraction and updating (Option 1 workflow)
Uses synthetic LAZ data from conftest fixtures."""

import pytest
from topochange import PointCloud
from skip_markers import HAS_PDAL, requires_pdal, SYNTHETIC


class TestPointCloudMetadataExtraction:
    """Tests for extracting metadata from point cloud files"""

    def test_create_pointcloud_object(self, compare_laz_path):
        """Test creating a PointCloud object from file"""
        pc = PointCloud(compare_laz_path)
        assert pc is not None
        assert pc.filename == compare_laz_path

    @requires_pdal
    def test_read_metadata_from_file(self, compare_pc):
        """Test reading metadata from point cloud file"""
        # check that basic metadata is loaded
        assert compare_pc.original_compound_crs is not None or compare_pc.current_compound_crs is not None
        assert hasattr(compare_pc, 'bounds')
        assert hasattr(compare_pc, 'total_points')

    @requires_pdal
    def test_metadata_attributes_exist(self, compare_pc):
        """Test that expected metadata attributes exist after loading"""
        # check for expected attributes
        assert hasattr(compare_pc, 'original_compound_crs')
        assert hasattr(compare_pc, 'original_horizontal_crs')
        assert hasattr(compare_pc, 'original_vertical_crs')
        assert hasattr(compare_pc, 'current_compound_crs')
        assert hasattr(compare_pc, 'current_horizontal_crs')
        assert hasattr(compare_pc, 'current_vertical_crs')
        assert hasattr(compare_pc, 'geoid_model')
        assert hasattr(compare_pc, 'epoch')
        assert hasattr(compare_pc, 'is_orthometric')

    @requires_pdal
    def test_synthetic_data_properties(self, compare_pc):
        """Test that synthetic data has expected properties"""
        # check known synthetic values
        assert compare_pc.total_points == SYNTHETIC['n_points']
        assert compare_pc.bounds is not None
        # check bounds contain the expected x_offset and y_offset
        x_min, x_max, y_min, y_max = compare_pc.bounds[:4]
        assert SYNTHETIC['x_offset'] <= x_min < SYNTHETIC['x_offset'] + SYNTHETIC['x_extent']
        assert SYNTHETIC['x_offset'] < x_max <= SYNTHETIC['x_offset'] + SYNTHETIC['x_extent']
        assert SYNTHETIC['y_offset'] <= y_min < SYNTHETIC['y_offset'] + SYNTHETIC['y_extent']
        assert SYNTHETIC['y_offset'] < y_max <= SYNTHETIC['y_offset'] + SYNTHETIC['y_extent']


class TestPointCloudMetadataUpdate:
    """Tests for updating point cloud metadata"""

    @requires_pdal
    def test_update_compound_crs(self, compare_pc):
        """Test updating compound CRS"""
        original_crs = compare_pc.current_compound_crs

        # update with a known CRS
        test_crs = "EPSG:4979"
        compare_pc.add_metadata(compound_CRS=test_crs)

        # check that CRS was updated
        assert compare_pc.current_compound_crs is not None
        # the CRS should have changed (or been set)
        assert compare_pc.current_compound_crs != original_crs or original_crs is None

    @requires_pdal
    def test_update_horizontal_crs(self, compare_pc):
        """Test updating horizontal CRS"""
        test_crs = "EPSG:32611"  # UTM Zone 11N
        compare_pc.add_metadata(horizontal_CRS=test_crs)

        # check that horizontal CRS was updated
        assert compare_pc.current_horizontal_crs is not None
        # check that compound CRS was also updated
        assert compare_pc.current_compound_crs is not None

    @requires_pdal
    def test_update_vertical_crs(self, compare_pc):
        """Test updating vertical CRS"""
        test_crs = "EPSG:5703"  # NAVD88
        compare_pc.add_metadata(vertical_CRS=test_crs)

        # check that vertical CRS was updated
        assert compare_pc.current_vertical_crs is not None
        # check that compound CRS was also updated
        assert compare_pc.current_compound_crs is not None

    @requires_pdal
    def test_update_geoid_model(self, compare_pc):
        """Test updating geoid model"""
        from topochange.geoid_utils import select_geoid_grid

        geoid_name = "geoid12b"
        geoid_grid, _ = select_geoid_grid(geoid_name, verbose=False)

        compare_pc.add_metadata(geoid_model=geoid_grid)

        # check that geoid model was updated
        assert compare_pc.geoid_model is not None
        assert geoid_name.lower() in str(compare_pc.geoid_model).lower()

    @requires_pdal
    def test_update_epoch_single_date(self, compare_pc):
        """Test updating epoch with a single date"""
        epoch_str = "05/18/2005"
        compare_pc.add_metadata(epoch=epoch_str)

        # check that epoch was updated
        assert compare_pc.epoch is not None
        # epoch should be a decimal year
        assert isinstance(compare_pc.epoch, (int, float))
        assert 2005.0 <= compare_pc.epoch <= 2006.0

    @requires_pdal
    def test_update_epoch_date_range(self, compare_pc):
        """Test updating epoch with a date range"""
        epoch_str = "05/18/2005 - 05/27/2005"
        compare_pc.add_metadata(epoch=epoch_str)

        # check that epoch was updated (should be midpoint)
        assert compare_pc.epoch is not None
        assert isinstance(compare_pc.epoch, (int, float))
        assert 2005.0 <= compare_pc.epoch <= 2006.0

    @requires_pdal
    def test_update_multiple_metadata_fields(self, compare_pc):
        """Test updating multiple metadata fields at once"""
        compare_pc.add_metadata(
            horizontal_CRS="EPSG:32611",
            vertical_CRS="EPSG:5703",
            epoch="05/18/2005"
        )

        # check that all fields were updated
        assert compare_pc.current_horizontal_crs is not None
        assert compare_pc.current_vertical_crs is not None
        assert compare_pc.epoch is not None

    @requires_pdal
    def test_metadata_consistency_after_updates(self, compare_pc):
        """Test that metadata remains consistent after multiple updates"""
        # update horizontal and vertical separately
        compare_pc.add_metadata(horizontal_CRS="EPSG:32611")
        compare_pc.add_metadata(vertical_CRS="EPSG:5703")

        # both should be reflected in the compound CRS
        assert compare_pc.current_compound_crs is not None
        assert compare_pc.current_horizontal_crs is not None
        assert compare_pc.current_vertical_crs is not None


class TestReferencePointCloudMetadata:
    """Tests for reference point cloud metadata handling"""

    @requires_pdal
    def test_load_reference_pointcloud(self, reference_pc):
        """Test loading reference point cloud"""
        assert reference_pc is not None
        assert reference_pc.current_compound_crs is not None or reference_pc.original_compound_crs is not None

    @requires_pdal
    def test_update_reference_metadata(self, reference_pc):
        """Test updating reference point cloud metadata"""
        reference_pc.add_metadata(
            horizontal_CRS="EPSG:32611",
            vertical_CRS="EPSG:5703",
            geoid_model="geoid12b",
            epoch="05/27/2018 - 07/22/2018"
        )

        # verify all fields were updated
        assert reference_pc.current_horizontal_crs is not None
        assert reference_pc.current_vertical_crs is not None
        assert reference_pc.geoid_model is not None
        assert reference_pc.epoch is not None

    @requires_pdal
    def test_print_metadata(self, reference_pc):
        """Test that print_metadata runs without error"""
        # this should not raise an exception
        reference_pc.print_metadata()

    @requires_pdal
    def test_reference_synthetic_data_properties(self, reference_pc):
        """Test that reference synthetic data has expected properties"""
        # check known synthetic values
        assert reference_pc.total_points == SYNTHETIC['n_points']
        assert reference_pc.bounds is not None
        # check bounds contain the expected x_offset and y_offset
        x_min, x_max, y_min, y_max = reference_pc.bounds[:4]
        assert SYNTHETIC['x_offset'] <= x_min < SYNTHETIC['x_offset'] + SYNTHETIC['x_extent']
        assert SYNTHETIC['x_offset'] < x_max <= SYNTHETIC['x_offset'] + SYNTHETIC['x_extent']
        assert SYNTHETIC['y_offset'] <= y_min < SYNTHETIC['y_offset'] + SYNTHETIC['y_extent']
        assert SYNTHETIC['y_offset'] < y_max <= SYNTHETIC['y_offset'] + SYNTHETIC['y_extent']


