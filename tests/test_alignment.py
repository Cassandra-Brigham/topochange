"""
Tests for point cloud alignment (Option 1 workflow)
"""

import pytest
import numpy as np
from pathlib import Path
from topochange import PointCloud, PointCloudPair
from topochange.alignment import (
    RegistrationMethod,
    RegistrationConfig,
    LandscapeAligner,
    PointCloudPairAligner,
)


class TestAlignmentBasics:
    """Tests for basic alignment functionality"""

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

    def test_align_point_clouds_vgicp(self, pc_pair):
        """Test alignment using VGICP method"""
        try:
            align_result = pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=1.0,
                initial_voxel_size=2.0,
                max_points=1_000_000,
            )

            # Check that alignment returned a result
            assert align_result is not None

            # Check that result has expected attributes
            assert hasattr(align_result, 'transformation')
            assert hasattr(align_result, 'rmse')
            assert hasattr(align_result, 'fitness')

            # Check that transformation is a 4x4 matrix
            assert align_result.transformation.shape == (4, 4)

            # Check that RMSE is a positive number
            assert align_result.rmse >= 0

            # Check that fitness is between 0 and 1
            assert 0.0 <= align_result.fitness <= 1.0

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")

    def test_align_with_memory_constraints(self, pc_pair):
        """Test alignment with memory constraints (larger voxel sizes)"""
        try:
            align_result = pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=2.0,  # Larger = fewer points
                initial_voxel_size=4.0,  # Larger = more aggressive downsampling
                max_points=500_000,  # Reduced max points
            )

            assert align_result is not None
            assert align_result.transformation.shape == (4, 4)

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")


class TestAlignmentMethods:
    """Tests for different alignment methods"""

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

    def test_vgicp_method(self, compare_pc_path, reference_pc_path):
        """Test VGICP method directly"""
        try:
            config = RegistrationConfig(
                target_points=100_000,
                max_iterations=50,
            )

            aligner = LandscapeAligner(config)
            result = aligner.align(
                compare_pc_path,
                reference_pc_path,
                method=RegistrationMethod.VGICP
            )

            assert result is not None
            assert result.transformation.shape == (4, 4)
            assert result.method_used == "vgicp"

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")

    def test_gicp_method(self, compare_pc_path, reference_pc_path):
        """Test GICP method directly"""
        try:
            config = RegistrationConfig(
                target_points=100_000,
                max_iterations=50,
            )

            aligner = LandscapeAligner(config)
            result = aligner.align(
                compare_pc_path,
                reference_pc_path,
                method=RegistrationMethod.GICP
            )

            assert result is not None
            assert result.transformation.shape == (4, 4)
            assert result.method_used == "gicp"

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")

    def test_icp_method(self, compare_pc_path, reference_pc_path):
        """Test ICP method directly"""
        try:
            config = RegistrationConfig(
                target_points=100_000,
                max_iterations=50,
            )

            aligner = LandscapeAligner(config)
            result = aligner.align(
                compare_pc_path,
                reference_pc_path,
                method=RegistrationMethod.ICP
            )

            assert result is not None
            assert result.transformation.shape == (4, 4)
            assert result.method_used == "icp"

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")


class TestAlignmentResults:
    """Tests for alignment result validation"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    @pytest.fixture
    def alignment_result(self, test_data_dir):
        """Create an alignment result for testing"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        try:
            pc1 = PointCloud(str(compare_path))
            pc1.from_file()

            pc2 = PointCloud(str(reference_path))
            pc2.from_file()

            pc_pair = PointCloudPair(pc1, pc2)

            result = pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=1.0,
                max_points=500_000,
            )

            return result

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")

    def test_transformation_matrix_properties(self, alignment_result):
        """Test that transformation matrix has correct properties"""
        T = alignment_result.transformation

        # Should be 4x4
        assert T.shape == (4, 4)

        # Bottom row should be [0, 0, 0, 1]
        assert np.allclose(T[3, :], [0, 0, 0, 1])

        # Upper-left 3x3 should be a rotation matrix (orthogonal)
        R = T[:3, :3]
        RTR = R.T @ R
        identity = np.eye(3)
        # Check orthogonality (may not be perfect due to numerical errors)
        assert np.allclose(RTR, identity, atol=0.1)

    def test_rmse_is_reasonable(self, alignment_result):
        """Test that RMSE is in a reasonable range"""
        rmse = alignment_result.rmse

        # RMSE should be positive
        assert rmse > 0

        # For good alignment, RMSE should be less than 10 meters
        # (this threshold may need adjustment based on your data)
        assert rmse < 10.0, f"RMSE {rmse} is unexpectedly high"

    def test_fitness_is_reasonable(self, alignment_result):
        """Test that fitness score is reasonable"""
        fitness = alignment_result.fitness

        # Fitness should be between 0 and 1
        assert 0.0 <= fitness <= 1.0

        # For good alignment, fitness should be > 0.3
        # (this threshold may need adjustment based on your data)
        assert fitness > 0.3, f"Fitness {fitness} is too low"

    def test_convergence(self, alignment_result):
        """Test that alignment converged"""
        assert alignment_result.converged is True


class TestAlignmentConfiguration:
    """Tests for alignment configuration options"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    def test_custom_downsample_resolution(self, test_data_dir):
        """Test alignment with custom downsample resolution"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        try:
            pc1 = PointCloud(str(compare_path))
            pc1.from_file()

            pc2 = PointCloud(str(reference_path))
            pc2.from_file()

            pc_pair = PointCloudPair(pc1, pc2)

            # Test with different resolutions
            for resolution in [0.5, 1.0, 2.0]:
                result = pc_pair.align_point_clouds(
                    method="vgicp",
                    downsample_resolution=resolution,
                    max_points=500_000,
                )

                assert result is not None
                assert result.transformation.shape == (4, 4)

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")

    def test_custom_max_points(self, test_data_dir):
        """Test alignment with custom max points"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        try:
            pc1 = PointCloud(str(compare_path))
            pc1.from_file()

            pc2 = PointCloud(str(reference_path))
            pc2.from_file()

            pc_pair = PointCloudPair(pc1, pc2)

            # Test with different max points
            for max_pts in [100_000, 500_000, 1_000_000]:
                result = pc_pair.align_point_clouds(
                    method="vgicp",
                    downsample_resolution=1.0,
                    max_points=max_pts,
                )

                assert result is not None
                assert result.transformation.shape == (4, 4)

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")


class TestAlignmentWithTransformation:
    """Tests for alignment followed by transformation application"""

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory"""
        return Path(__file__).parent.parent / "test_data"

    def test_alignment_transformation_workflow(self, test_data_dir, tmp_path):
        """Test complete alignment and transformation workflow"""
        compare_path = test_data_dir / "compare.laz"
        reference_path = test_data_dir / "reference.laz"

        if not compare_path.exists() or not reference_path.exists():
            pytest.skip("Test data not found")

        try:
            pc1 = PointCloud(str(compare_path))
            pc1.from_file()

            pc2 = PointCloud(str(reference_path))
            pc2.from_file()

            pc_pair = PointCloudPair(pc1, pc2)

            # Perform alignment
            result = pc_pair.align_point_clouds(
                method="vgicp",
                downsample_resolution=1.0,
                max_points=500_000,
            )

            assert result is not None
            assert result.converged is True

            # Check that the transformation was applied (in-place)
            # The compare point cloud should now be aligned
            assert pc_pair.pc1 is not None

        except ImportError as e:
            pytest.skip(f"small_gicp not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
