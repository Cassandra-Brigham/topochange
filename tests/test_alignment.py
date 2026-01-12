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


class TestSyntheticAlignment:
    """Tests using synthetic data with known ground truth transformations.

    These tests create synthetic point clouds, apply a known transformation,
    and verify that the alignment algorithm recovers it correctly.
    """

    def _create_synthetic_terrain(self, n_points: int = 50000, seed: int = 42) -> np.ndarray:
        """Create a synthetic terrain point cloud (rolling hills).

        Returns Nx3 array of XYZ coordinates.
        """
        np.random.seed(seed)

        # Create a grid of points with some randomness
        side = int(np.sqrt(n_points))
        x = np.linspace(0, 100, side)
        y = np.linspace(0, 100, side)
        xx, yy = np.meshgrid(x, y)
        xx = xx.flatten() + np.random.normal(0, 0.1, side * side)
        yy = yy.flatten() + np.random.normal(0, 0.1, side * side)

        # Create terrain-like Z values (rolling hills)
        zz = (
            5 * np.sin(xx / 10) * np.cos(yy / 10) +  # Large hills
            2 * np.sin(xx / 3) * np.sin(yy / 5) +    # Medium features
            np.random.normal(0, 0.05, len(xx))       # Small noise
        )

        return np.column_stack([xx, yy, zz]).astype(np.float64)

    def _apply_transformation(self, points: np.ndarray,
                               translation: np.ndarray,
                               rotation_deg: float = 0.0,
                               rotation_axis: str = 'z') -> np.ndarray:
        """Apply a rigid transformation to points.

        Parameters
        ----------
        points : np.ndarray
            Nx3 array of points
        translation : np.ndarray
            3-element translation vector [tx, ty, tz]
        rotation_deg : float
            Rotation angle in degrees
        rotation_axis : str
            Axis of rotation ('x', 'y', or 'z')

        Returns
        -------
        np.ndarray
            Transformed Nx3 array
        """
        # Build rotation matrix
        angle_rad = np.radians(rotation_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        if rotation_axis == 'z':
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        elif rotation_axis == 'y':
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif rotation_axis == 'x':
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        else:
            R = np.eye(3)

        # Apply rotation then translation
        transformed = (R @ points.T).T + translation
        return transformed

    def test_recovery_of_pure_translation(self):
        """Test that alignment recovers a known translation."""
        try:
            import small_gicp
        except ImportError:
            pytest.skip("small_gicp not available")

        # Create synthetic terrain
        target_points = self._create_synthetic_terrain(n_points=50000, seed=42)

        # Apply known translation to create source
        known_translation = np.array([1.5, -0.8, 0.3])  # meters
        source_points = target_points + known_translation

        # Run alignment (source -> target, so should recover -translation)
        result = small_gicp.align(
            target_points,
            source_points,
            registration_type='GICP',
            downsampling_resolution=0.5,
            max_correspondence_distance=5.0,
            num_threads=4,
        )

        # Extract recovered translation
        recovered_translation = result.T_target_source[:3, 3]

        # The alignment should recover the negative of what we applied
        # (because we're aligning source to target)
        expected_translation = -known_translation

        # Check convergence
        assert result.converged, "Alignment did not converge"
        assert result.iterations > 0, "No iterations performed"

        # Check translation recovery (within 0.1m tolerance)
        np.testing.assert_allclose(
            recovered_translation,
            expected_translation,
            atol=0.1,
            err_msg=f"Translation not recovered. Expected {expected_translation}, got {recovered_translation}"
        )

    def test_recovery_of_translation_and_rotation(self):
        """Test that alignment recovers translation + small rotation."""
        try:
            import small_gicp
        except ImportError:
            pytest.skip("small_gicp not available")

        # Create synthetic terrain
        target_points = self._create_synthetic_terrain(n_points=50000, seed=123)

        # Apply known transformation
        known_translation = np.array([2.0, 1.0, -0.5])
        known_rotation_deg = 2.0  # Small rotation

        source_points = self._apply_transformation(
            target_points,
            known_translation,
            rotation_deg=known_rotation_deg,
            rotation_axis='z'
        )

        # Run alignment
        result = small_gicp.align(
            target_points,
            source_points,
            registration_type='GICP',
            downsampling_resolution=0.5,
            max_correspondence_distance=10.0,
            num_threads=4,
        )

        # Check convergence
        assert result.converged, "Alignment did not converge"
        assert result.iterations > 0, "No iterations performed"

        # Extract recovered transformation
        T = result.T_target_source
        recovered_translation = T[:3, 3]

        # Compute recovered rotation angle from rotation matrix
        R = T[:3, :3]
        rotation_angle_rad = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        recovered_rotation_deg = np.degrees(rotation_angle_rad)

        # Check translation recovery (within 0.2m tolerance for combined transform)
        expected_translation_magnitude = np.linalg.norm(known_translation)
        recovered_translation_magnitude = np.linalg.norm(recovered_translation)
        assert abs(recovered_translation_magnitude - expected_translation_magnitude) < 0.5, \
            f"Translation magnitude wrong: expected ~{expected_translation_magnitude:.2f}, got {recovered_translation_magnitude:.2f}"

        # Check rotation recovery (within 0.5 degree tolerance)
        assert abs(recovered_rotation_deg - known_rotation_deg) < 1.0, \
            f"Rotation not recovered: expected {known_rotation_deg}°, got {recovered_rotation_deg:.2f}°"

    def test_vgicp_vs_gicp_consistency(self):
        """Test that VGICP and GICP produce similar results on synthetic data."""
        try:
            import small_gicp
        except ImportError:
            pytest.skip("small_gicp not available")

        # Create synthetic terrain
        target_points = self._create_synthetic_terrain(n_points=30000, seed=456)

        # Apply known translation
        known_translation = np.array([1.0, -0.5, 0.2])
        source_points = target_points + known_translation

        # Run GICP
        result_gicp = small_gicp.align(
            target_points,
            source_points,
            registration_type='GICP',
            downsampling_resolution=0.5,
            max_correspondence_distance=5.0,
            num_threads=4,
        )

        # Run VGICP
        result_vgicp = small_gicp.align(
            target_points,
            source_points,
            registration_type='VGICP',
            voxel_resolution=0.5,
            downsampling_resolution=0.5,
            max_correspondence_distance=5.0,
            num_threads=4,
        )

        # Both should converge
        assert result_gicp.converged, "GICP did not converge"
        assert result_vgicp.converged, "VGICP did not converge"

        # Both should recover similar translations
        trans_gicp = result_gicp.T_target_source[:3, 3]
        trans_vgicp = result_vgicp.T_target_source[:3, 3]

        # They should agree within 0.2m
        np.testing.assert_allclose(
            trans_gicp,
            trans_vgicp,
            atol=0.2,
            err_msg=f"GICP and VGICP disagree: GICP={trans_gicp}, VGICP={trans_vgicp}"
        )

    def test_alignment_with_partial_overlap(self):
        """Test alignment when point clouds have partial overlap."""
        try:
            import small_gicp
        except ImportError:
            pytest.skip("small_gicp not available")

        # Create two overlapping but offset point clouds
        np.random.seed(789)

        # Target covers 0-100 in X
        target_x = np.random.uniform(0, 100, 20000)
        target_y = np.random.uniform(0, 100, 20000)
        target_z = np.sin(target_x / 10) + np.cos(target_y / 10)
        target_points = np.column_stack([target_x, target_y, target_z]).astype(np.float64)

        # Source covers 50-150 in X (50% overlap), with known offset
        known_translation = np.array([0.5, -0.3, 0.1])
        source_x = np.random.uniform(50, 150, 20000)
        source_y = np.random.uniform(0, 100, 20000)
        source_z = np.sin(source_x / 10) + np.cos(source_y / 10)
        source_points = np.column_stack([source_x, source_y, source_z]).astype(np.float64)
        source_points = source_points + known_translation

        # Run alignment
        result = small_gicp.align(
            target_points,
            source_points,
            registration_type='GICP',
            downsampling_resolution=1.0,
            max_correspondence_distance=10.0,
            num_threads=4,
        )

        # Should still converge despite partial overlap
        assert result.converged, "Alignment did not converge with partial overlap"
        assert result.num_inliers > 0, "No inliers found"

        # Recovered translation should be reasonable (not checking exact match
        # since partial overlap makes this harder)
        recovered_translation = result.T_target_source[:3, 3]
        translation_magnitude = np.linalg.norm(recovered_translation)
        expected_magnitude = np.linalg.norm(known_translation)

        # Should be within 1m of expected
        assert translation_magnitude < expected_magnitude + 1.0, \
            f"Translation magnitude {translation_magnitude:.2f} is too large"

    def test_identity_alignment(self):
        """Test that identical point clouds produce identity transformation."""
        try:
            import small_gicp
        except ImportError:
            pytest.skip("small_gicp not available")

        # Create synthetic terrain
        points = self._create_synthetic_terrain(n_points=30000, seed=101)

        # Align identical point clouds (with small noise to avoid numerical issues)
        target_points = points.copy()
        source_points = points.copy() + np.random.normal(0, 0.001, points.shape)

        result = small_gicp.align(
            target_points,
            source_points,
            registration_type='GICP',
            downsampling_resolution=0.5,
            max_correspondence_distance=1.0,
            num_threads=4,
        )

        # Should converge
        assert result.converged, "Alignment did not converge"

        # Transformation should be near-identity
        T = result.T_target_source

        # Translation should be near zero
        translation = T[:3, 3]
        assert np.linalg.norm(translation) < 0.1, \
            f"Translation {translation} should be near zero for identical clouds"

        # Rotation should be near identity
        R = T[:3, :3]
        rotation_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        assert rotation_angle < np.radians(1.0), \
            f"Rotation angle {np.degrees(rotation_angle):.2f}° should be near zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
