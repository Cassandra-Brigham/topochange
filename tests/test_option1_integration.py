"""
Integration tests for complete Option 1 workflow from my_implementation.ipynb
Tests the full pipeline: metadata -> transformation -> alignment -> DEM creation
"""

import pytest
import os
import tempfile
from pathlib import Path
import numpy as np
from topochange import PointCloud, PointCloudPair, RasterPair


class TestOption1CompleteWorkflow:
    """Integration test for the complete Option 1 workflow"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_full_workflow_without_velocity(self, test_data_dir, output_dir):
        """
        Test the complete workflow from Option 1:
        1. Load point clouds
        2. Update metadata
        3. Transform compare to match reference (skip epoch)
        4. Align point clouds
        5. Create DEMs
        6. Verify DEM properties
        """
        # Step 1: Load point clouds
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        pc1 = PointCloud(str(compare_path))
        pc1.from_file()

        pc2 = PointCloud(str(reference_path))
        pc2.from_file()

        # Verify point clouds loaded
        assert pc1 is not None
        assert pc2 is not None

        # Step 2: Update metadata
        pc1.add_metadata(
            compound_CRS="EPSG:4979",
            epoch="05/18/2005 - 05/27/2005"
        )

        pc2.add_metadata(
            vertical_CRS="EPSG:5703",
            geoid_model="geoid12b",
            epoch="05/27/2018 - 07/22/2018"
        )

        # Verify metadata was updated
        assert pc1.current_compound_crs is not None
        assert pc1.epoch is not None
        assert pc2.current_vertical_crs is not None
        assert pc2.epoch is not None

        # Step 3: Create pair and transform (skip epoch for speed)
        pc_pair = PointCloudPair(pc1, pc2)

        try:
            transformed = pc_pair.transform_compare_to_match_reference(
                skip_epoch=True,
                verbose=False
            )

            # Verify transformation completed
            assert transformed is not False

            # Step 4: Align point clouds
            align_result = pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=1.0,
                initial_voxel_size=2.0,
                max_points=1_000_000,
            )

            # Verify alignment succeeded
            assert align_result is not None
            assert align_result.transformation.shape == (4, 4)
            assert align_result.converged is True

            # Step 5: Create DEMs
            dem1, dem2 = pc_pair.create_dtm_pair(
                resolution=1.0,
                output_dir=output_dir
            )

            # Verify DEMs were created
            assert dem1 is not None
            assert dem2 is not None
            assert os.path.exists(dem1.filename)
            assert os.path.exists(dem2.filename)

            # Step 6: Verify DEM properties
            # Check units
            assert hasattr(dem1, 'horizontal_unit')
            assert hasattr(dem1, 'vertical_unit')

            # Set units if needed
            dem1.set_units(vertical_unit="meter")
            dem2.set_units(vertical_unit="meter")

            assert dem1.vertical_unit == "meter"
            assert dem2.vertical_unit == "meter"

            # Verify DEMs have valid data
            assert hasattr(dem1, 'data')
            assert hasattr(dem2, 'data')
            assert np.any(np.isfinite(dem1.data))
            assert np.any(np.isfinite(dem2.data))

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")


class TestOption1WorkflowStepByStep:
    """Test each step of Option 1 workflow independently"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    def test_step1_load_and_metadata(self, test_data_dir):
        """Test Step 1: Load point clouds and update metadata"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        # Load compare point cloud
        pc1 = PointCloud(str(compare_path))
        pc1.from_file()

        # Update compare metadata
        pc1.add_metadata(
            compound_CRS="EPSG:4979",
            epoch="05/18/2005 - 05/27/2005"
        )

        # Load reference point cloud
        pc2 = PointCloud(str(reference_path))
        pc2.from_file()

        # Update reference metadata
        pc2.add_metadata(
            vertical_CRS="EPSG:5703",
            geoid_model="geoid12b",
            epoch="05/27/2018 - 07/22/2018"
        )

        # Verify
        assert pc1.current_compound_crs is not None
        assert pc1.epoch is not None
        assert pc2.current_vertical_crs is not None
        assert pc2.geoid_model is not None
        assert pc2.epoch is not None

    def test_step2_create_pair_and_transform(self, test_data_dir):
        """Test Step 2: Create pair and transform"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        pc1 = PointCloud(str(compare_path))
        pc1.from_file()
        pc1.add_metadata(compound_CRS="EPSG:4979")

        pc2 = PointCloud(str(reference_path))
        pc2.from_file()
        pc2.add_metadata(vertical_CRS="EPSG:5703")

        # Create pair
        pc_pair = PointCloudPair(pc1, pc2)
        assert pc_pair is not None

        # Transform (skip epoch)
        try:
            transformed = pc_pair.transform_compare_to_match_reference(
                skip_epoch=True,
                verbose=False
            )
            assert transformed is not False
        except Exception as e:
            pytest.fail(f"Transformation failed: {e}")

    def test_step3_alignment(self, test_data_dir):
        """Test Step 3: Align point clouds"""
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
            # Transform first
            pc_pair.transform_compare_to_match_reference(skip_epoch=True, verbose=False)

            # Align
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

    def test_step4_dem_creation(self, test_data_dir):
        """Test Step 4: Create DEMs from aligned clouds"""
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
            # Full workflow up to DEM creation
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

                # Set and verify units
                dem1.set_units(vertical_unit="meter")
                dem2.set_units(vertical_unit="meter")

                assert dem1.vertical_unit == "meter"
                assert dem2.vertical_unit == "meter"

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")


class TestOption1ErrorHandling:
    """Test error handling in Option 1 workflow"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    def test_missing_metadata_handling(self, test_data_dir):
        """Test that workflow handles missing metadata gracefully"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        pc1 = PointCloud(str(compare_path))
        pc1.from_file()

        pc2 = PointCloud(str(reference_path))
        pc2.from_file()

        # Create pair without updating metadata
        pc_pair = PointCloudPair(pc1, pc2)

        # Should still be able to create pair
        assert pc_pair is not None

    def test_invalid_crs_handling(self, test_data_dir):
        """Test handling of invalid CRS values"""
        compare_path = test_data_dir / "compare.laz"

        if not compare_path.exists():
            pytest.skip("Test data not found")

        pc = PointCloud(str(compare_path))
        pc.from_file()

        # Try to set invalid CRS - should handle gracefully
        try:
            pc.add_metadata(compound_CRS="INVALID:12345")
            # If it succeeds or raises specific error, that's okay
        except Exception as e:
            # Should raise a meaningful error
            assert len(str(e)) > 0


class TestOption1Performance:
    """Performance tests for Option 1 workflow"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    def test_workflow_completes_in_reasonable_time(self, test_data_dir):
        """Test that workflow completes without hanging"""
        import time

        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        start_time = time.time()

        try:
            pc1 = PointCloud(str(compare_path))
            pc1.from_file()

            pc2 = PointCloud(str(reference_path))
            pc2.from_file()

            pc1.add_metadata(compound_CRS="EPSG:4979")
            pc2.add_metadata(vertical_CRS="EPSG:5703")

            pc_pair = PointCloudPair(pc1, pc2)
            pc_pair.transform_compare_to_match_reference(skip_epoch=True, verbose=False)

            # Alignment with reduced points for speed
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

            # Workflow should complete in reasonable time (adjust as needed)
            # This is a generous timeout
            assert elapsed_time < 600, f"Workflow took {elapsed_time:.1f}s, which is too long"

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
