"""Integration tests for complete Option 1 workflow from my_implementation.ipynb
Tests the full pipeline: metadata -> transformation -> alignment -> DEM creation
Uses synthetic LAZ data from conftest fixtures."""

import pytest
import os
import tempfile
import numpy as np
from topochange import PointCloud, PointCloudPair, RasterPair
from skip_markers import requires_pdal, requires_small_gicp


class TestOption1CompleteWorkflow:
    """Integration test for the complete Option 1 workflow"""

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @requires_pdal
    @requires_small_gicp
    def test_full_workflow_without_velocity(self, compare_pc, reference_pc, output_dir):
        """
        Test the complete workflow from Option 1:
        1. Load point clouds
        2. Update metadata
        3. Transform compare to match reference (skip epoch)
        4. Align point clouds
        5. Create DEMs
        6. Verify DEM properties
        """
        # step 1: Point clouds already loaded via fixtures
        pc1 = compare_pc
        pc2 = reference_pc

        # verify point clouds loaded
        assert pc1 is not None
        assert pc2 is not None

        # step 2: Update metadata
        pc1.add_metadata(
            compound_CRS="EPSG:4979",
            epoch="05/18/2005 - 05/27/2005"
        )

        pc2.add_metadata(
            vertical_CRS="EPSG:5703",
            geoid_model="geoid12b",
            epoch="05/27/2018 - 07/22/2018"
        )

        # verify metadata was updated
        assert pc1.current_compound_crs is not None
        assert pc1.epoch is not None
        assert pc2.current_vertical_crs is not None
        assert pc2.epoch is not None

        # step 3: Create pair and transform (skip epoch for speed)
        pc_pair = PointCloudPair(pc1, pc2)

        try:
            transformed = pc_pair.transform_compare_to_match_reference(
                skip_epoch=True,
                verbose=False
            )

            # verify transformation completed
            assert transformed is not False

            # step 4: Align point clouds
            align_result = pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=1.0,
                initial_voxel_size=2.0,
                max_points=1_000_000,
            )

            # verify alignment succeeded
            assert align_result is not None
            assert align_result.transformation.shape == (4, 4)
            assert align_result.converged is True

            # step 5: Create DEMs
            dem1, dem2 = pc_pair.create_dtm_pair(
                resolution=1.0,
                output_dir=output_dir
            )

            # verify DEMs were created
            assert dem1 is not None
            assert dem2 is not None
            assert os.path.exists(dem1.filename)
            assert os.path.exists(dem2.filename)

            # step 6: Verify DEM properties
            # check units
            assert hasattr(dem1, 'horizontal_unit')
            assert hasattr(dem1, 'vertical_unit')

            # set units if needed
            dem1.set_units(vertical_unit="meter")
            dem2.set_units(vertical_unit="meter")

            assert dem1.vertical_unit == "meter"
            assert dem2.vertical_unit == "meter"

            # verify DEMs have valid data
            assert hasattr(dem1, 'data')
            assert hasattr(dem2, 'data')
            assert np.any(np.isfinite(dem1.data))
            assert np.any(np.isfinite(dem2.data))

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")


class TestOption1WorkflowStepByStep:
    """Test each step of Option 1 workflow independently"""

    @requires_pdal
    def test_step1_load_and_metadata(self, compare_pc, reference_pc):
        """Test Step 1: Load point clouds and update metadata"""
        # load compare point cloud
        pc1 = compare_pc

        # update compare metadata
        pc1.add_metadata(
            compound_CRS="EPSG:4979",
            epoch="05/18/2005 - 05/27/2005"
        )

        # load reference point cloud
        pc2 = reference_pc

        # update reference metadata
        pc2.add_metadata(
            vertical_CRS="EPSG:5703",
            geoid_model="geoid12b",
            epoch="05/27/2018 - 07/22/2018"
        )

        # verify
        assert pc1.current_compound_crs is not None
        assert pc1.epoch is not None
        assert pc2.current_vertical_crs is not None
        assert pc2.geoid_model is not None
        assert pc2.epoch is not None

    @requires_pdal
    def test_step2_create_pair_and_transform(self, compare_pc, reference_pc):
        """Test Step 2: Create pair and transform"""
        pc1 = compare_pc
        pc1.add_metadata(compound_CRS="EPSG:4979")

        pc2 = reference_pc
        pc2.add_metadata(vertical_CRS="EPSG:5703")

        # create pair
        pc_pair = PointCloudPair(pc1, pc2)
        assert pc_pair is not None

        # transform (skip epoch)
        try:
            transformed = pc_pair.transform_compare_to_match_reference(
                skip_epoch=True,
                verbose=False
            )
            assert transformed is not False
        except Exception as e:
            pytest.fail(f"Transformation failed: {e}")

    @requires_pdal
    @requires_small_gicp
    def test_step3_alignment(self, compare_pc, reference_pc):
        """Test Step 3: Align point clouds"""
        pc1 = compare_pc
        pc2 = reference_pc

        pc_pair = PointCloudPair(pc1, pc2)

        try:
            # transform first
            pc_pair.transform_compare_to_match_reference(skip_epoch=True, verbose=False)

            # align
            align_result = pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=1.0,
                max_points=1_000_000,
            )

            assert align_result is not None
            assert align_result.transformation.shape == (4, 4)
            assert align_result.converged is True
            assert align_result.rmse > 0

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")

    @requires_pdal
    @requires_small_gicp
    def test_step4_dem_creation(self, compare_pc, reference_pc):
        """Test Step 4: Create DEMs from aligned clouds"""
        pc1 = compare_pc
        pc2 = reference_pc

        pc_pair = PointCloudPair(pc1, pc2)

        try:
            # full workflow up to DEM creation
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

                assert dem1 is not None
                assert dem2 is not None
                assert os.path.exists(dem1.filename)
                assert os.path.exists(dem2.filename)

                # set and verify units
                dem1.set_units(vertical_unit="meter")
                dem2.set_units(vertical_unit="meter")

                assert dem1.vertical_unit == "meter"
                assert dem2.vertical_unit == "meter"

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")


class TestOption1ErrorHandling:
    """Test error handling in Option 1 workflow"""

    @requires_pdal
    def test_missing_metadata_handling(self, compare_pc, reference_pc):
        """Test that workflow handles missing metadata gracefully"""
        pc1 = compare_pc
        pc2 = reference_pc

        # create pair without updating metadata
        pc_pair = PointCloudPair(pc1, pc2)

        # should still be able to create pair
        assert pc_pair is not None

    @requires_pdal
    def test_invalid_crs_handling(self, compare_pc):
        """Test handling of invalid CRS values"""
        pc = compare_pc

        # try to set invalid CRS - should handle gracefully
        try:
            pc.add_metadata(compound_CRS="INVALID:12345")
            # if it succeeds or raises specific error, that's okay
        except Exception as e:
            # should raise a meaningful error
            assert len(str(e)) > 0


class TestOption1Performance:
    """Performance tests for Option 1 workflow"""

    @requires_pdal
    @requires_small_gicp
    def test_workflow_completes_in_reasonable_time(self, compare_pc, reference_pc):
        """Test that workflow completes without hanging"""
        import time

        start_time = time.time()

        try:
            pc1 = compare_pc
            pc2 = reference_pc

            pc1.add_metadata(compound_CRS="EPSG:4979")
            pc2.add_metadata(vertical_CRS="EPSG:5703")

            pc_pair = PointCloudPair(pc1, pc2)
            pc_pair.transform_compare_to_match_reference(skip_epoch=True, verbose=False)

            # alignment with reduced points for speed
            pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=2.0,
                max_points=100_000,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                dem1, dem2 = pc_pair.create_dtm_pair(
                    resolution=2.0,
                    output_dir=tmpdir
                )

            elapsed_time = time.time() - start_time

            # workflow should complete in reasonable time (adjust as needed)
            # this is a generous timeout
            assert elapsed_time < 600, f"Workflow took {elapsed_time:.1f}s, which is too long"

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")

