"""Tests for point cloud alignment.

Tests in TestAlignmentBasics–TestAlignmentWithTransformation require both PDAL
and small_gicp.  Tests in TestSyntheticAlignment and TestAlignFunction require
only small_gicp (they build numpy arrays directly).
"""

import pytest
import numpy as np
from topochange import PointCloud, PointCloudPair
from topochange.alignment import (
    RegistrationMethod,
    RegistrationConfig,
    RegistrationResult,
    LandscapeAligner,
    align_point_clouds,
)
from skip_markers import requires_pdal, requires_small_gicp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_terrain(n_points: int = 50000, seed: int = 42) -> np.ndarray:
    """Rolling-hills terrain as Nx3 float64 array."""
    np.random.seed(seed)
    side = int(np.sqrt(n_points))
    x = np.linspace(0, 100, side)
    y = np.linspace(0, 100, side)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten() + np.random.normal(0, 0.1, side * side)
    yy = yy.flatten() + np.random.normal(0, 0.1, side * side)
    zz = (
        5 * np.sin(xx / 10) * np.cos(yy / 10)
        + 2 * np.sin(xx / 3) * np.sin(yy / 5)
        + np.random.normal(0, 0.05, len(xx))
    )
    return np.column_stack([xx, yy, zz]).astype(np.float64)


def _apply_transformation(
    points: np.ndarray,
    translation: np.ndarray,
    rotation_deg: float = 0.0,
    rotation_axis: str = "z",
) -> np.ndarray:
    """Apply a rigid body transformation to points."""
    angle_rad = np.radians(rotation_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if rotation_axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif rotation_axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif rotation_axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        R = np.eye(3)
    return (R @ points.T).T + translation


# ---------------------------------------------------------------------------
# Tests for the standalone align_point_clouds() function
# ---------------------------------------------------------------------------

@requires_small_gicp
class TestAlignFunction:
    """Tests for the core align_point_clouds() function using synthetic data."""

    def test_recovery_of_pure_translation(self):
        """align_point_clouds should recover a known translation."""
        target = _create_synthetic_terrain(n_points=50000, seed=42)
        known_translation = np.array([1.5, -0.8, 0.3])
        source = target + known_translation

        config = RegistrationConfig(
            method="gicp",
            voxel_resolution=0.5,
            max_correspondence_distance=5.0,
            verbose=False,
        )
        result = align_point_clouds(source, target, config)

        assert result.converged, "Alignment did not converge"
        assert result.rmse < 0.5, f"RMSE too high: {result.rmse}"
        assert result.pre_rmse > 0
        # Translation should be approximately -known_translation (source→target)
        np.testing.assert_allclose(
            result.translation, -known_translation, atol=0.15,
        )

    def test_recovery_of_translation_and_rotation(self):
        """align_point_clouds should recover translation + small rotation."""
        target = _create_synthetic_terrain(n_points=50000, seed=123)
        known_translation = np.array([2.0, 1.0, -0.5])
        source = _apply_transformation(
            target, known_translation, rotation_deg=2.0, rotation_axis="z",
        )

        config = RegistrationConfig(
            method="gicp",
            voxel_resolution=0.5,
            max_correspondence_distance=10.0,
            verbose=False,
        )
        result = align_point_clouds(source, target, config)

        assert result.converged
        recovered_mag = np.linalg.norm(result.translation)
        expected_mag = np.linalg.norm(known_translation)
        assert abs(recovered_mag - expected_mag) < 0.5
        assert abs(result.rotation_angle_deg - 2.0) < 1.0

    def test_vgicp_method(self):
        """VGICP should converge and produce reasonable results."""
        target = _create_synthetic_terrain(n_points=30000, seed=456)
        source = target + np.array([1.0, -0.5, 0.2])

        config = RegistrationConfig(method="vgicp", voxel_resolution=0.5, verbose=False)
        result = align_point_clouds(source, target, config)

        assert result.converged
        assert result.method_used == "vgicp"
        assert result.rmse < 1.0

    def test_icp_method(self):
        """Plain ICP should converge."""
        target = _create_synthetic_terrain(n_points=30000, seed=789)
        source = target + np.array([0.5, -0.3, 0.1])

        config = RegistrationConfig(method="icp", voxel_resolution=0.5, verbose=False)
        result = align_point_clouds(source, target, config)

        assert result.converged
        assert result.method_used == "icp"

    def test_plane_icp_method(self):
        """Plane ICP should converge."""
        target = _create_synthetic_terrain(n_points=30000, seed=321)
        source = target + np.array([0.3, 0.2, -0.1])

        config = RegistrationConfig(method="plane_icp", voxel_resolution=0.5, verbose=False)
        result = align_point_clouds(source, target, config)

        assert result.converged
        assert result.method_used == "plane_icp"

    def test_multi_resolution_produces_different_point_counts(self, capsys):
        """Multi-resolution stages should produce different point counts."""
        target = _create_synthetic_terrain(n_points=50000, seed=42)
        source = target + np.array([1.0, 0.0, 0.0])

        config = RegistrationConfig(
            method="vgicp",
            voxel_resolution=0.5,
            multi_resolution=True,
            verbose=True,  # need verbose to see point counts
        )
        result = align_point_clouds(source, target, config)

        # Check stderr for varying point counts across stages
        import sys
        # The verbose output goes to stderr — captured by capsys
        captured = capsys.readouterr()
        # Should see at least 2 "Points:" lines with different numbers
        import re
        point_lines = re.findall(r"Points:\s+([\d,]+)\s+source", captured.err)
        assert len(point_lines) >= 2, (
            f"Expected multiple resolution stages, got {len(point_lines)} point reports"
        )
        # Coarser stage should have fewer points
        counts = [int(p.replace(",", "")) for p in point_lines]
        assert counts[0] < counts[-1], (
            f"Coarse stage ({counts[0]}) should have fewer points than fine stage ({counts[-1]})"
        )

    def test_single_resolution_when_disabled(self):
        """multi_resolution=False should use a single stage."""
        target = _create_synthetic_terrain(n_points=30000, seed=42)
        source = target + np.array([1.0, 0.0, 0.0])

        config = RegistrationConfig(
            method="vgicp",
            voxel_resolution=0.5,
            multi_resolution=False,
            verbose=False,
        )
        result = align_point_clouds(source, target, config)
        assert result.converged

    def test_auto_revert_on_already_aligned_data(self):
        """Auto-revert should fire when alignment worsens RMSE on well-aligned data."""
        target = _create_synthetic_terrain(n_points=30000, seed=42)
        # Source is nearly identical (tiny noise only)
        source = target + np.random.normal(0, 0.001, target.shape)

        config = RegistrationConfig(
            method="vgicp",
            voxel_resolution=0.5,
            auto_revert=True,
            verbose=False,
        )
        result = align_point_clouds(source, target, config)

        # Either alignment was good (near-identity), or it reverted
        if result.reverted:
            # Transform should be identity
            np.testing.assert_allclose(result.transformation, np.eye(4), atol=1e-10)
        else:
            # Translation should be very small
            assert np.linalg.norm(result.translation) < 0.1

    def test_auto_revert_disabled(self):
        """auto_revert=False should keep the alignment even if RMSE worsened."""
        target = _create_synthetic_terrain(n_points=30000, seed=42)
        source = target + np.random.normal(0, 0.001, target.shape)

        config = RegistrationConfig(
            method="vgicp",
            voxel_resolution=0.5,
            auto_revert=False,
            verbose=False,
        )
        result = align_point_clouds(source, target, config)
        assert not result.reverted

    def test_identity_alignment(self):
        """Identical clouds should produce near-identity transformation."""
        points = _create_synthetic_terrain(n_points=30000, seed=101)
        target = points.copy()
        source = points.copy() + np.random.normal(0, 0.001, points.shape)

        config = RegistrationConfig(
            method="gicp",
            voxel_resolution=0.5,
            max_correspondence_distance=1.0,
            verbose=False,
        )
        result = align_point_clouds(source, target, config)

        assert result.converged
        assert np.linalg.norm(result.translation) < 0.1
        assert result.rotation_angle_deg < 1.0

    def test_default_config(self):
        """align_point_clouds with config=None should use defaults."""
        target = _create_synthetic_terrain(n_points=20000, seed=42)
        source = target + np.array([0.5, 0.0, 0.0])

        result = align_point_clouds(source, target)
        assert result.converged
        assert result.method_used == "vgicp"  # default

    def test_custom_resolution_stages(self):
        """Custom resolution_stages should be respected."""
        target = _create_synthetic_terrain(n_points=30000, seed=42)
        source = target + np.array([1.0, 0.5, 0.0])

        config = RegistrationConfig(
            method="vgicp",
            resolution_stages=[4.0, 2.0, 1.0, 0.5],
            verbose=False,
        )
        result = align_point_clouds(source, target, config)
        assert result.converged

    def test_result_has_pre_and_post_rmse(self):
        """Result should contain both pre_rmse and rmse (post)."""
        target = _create_synthetic_terrain(n_points=30000, seed=42)
        source = target + np.array([1.0, 0.0, 0.0])

        config = RegistrationConfig(method="vgicp", verbose=False)
        result = align_point_clouds(source, target, config)

        assert hasattr(result, 'pre_rmse')
        assert hasattr(result, 'rmse')
        assert result.pre_rmse > 0
        assert result.rmse > 0
        # After alignment, RMSE should improve
        assert result.rmse < result.pre_rmse

    def test_centroid_stored_in_result(self):
        """Centroid should be stored in result for file-level transform application."""
        target = _create_synthetic_terrain(n_points=20000, seed=42)
        source = target + np.array([1.0, 0.0, 0.0])

        result = align_point_clouds(source, target)
        assert result.centroid is not None
        assert result.centroid.shape == (3,)

    def test_to_dict(self):
        """RegistrationResult.to_dict() should return all expected keys."""
        target = _create_synthetic_terrain(n_points=20000, seed=42)
        source = target + np.array([1.0, 0.0, 0.0])

        result = align_point_clouds(source, target, RegistrationConfig(verbose=False))
        d = result.to_dict()

        expected_keys = {
            'transformation', 'rmse', 'pre_rmse', 'fitness', 'num_inliers',
            'converged', 'iterations', 'method_used', 'reverted',
            'translation', 'rotation_angle_deg', 'centroid',
            'source_path', 'target_path', 'output_path',
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_empty_point_cloud_returns_invalid_result(self):
        """Empty arrays should return a result with converged=False."""
        import warnings
        empty = np.zeros((0, 3))
        nonempty = _create_synthetic_terrain(n_points=1000, seed=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = align_point_clouds(empty, nonempty)
        assert not result.converged

    def test_invalid_method_raises(self):
        """Invalid method name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            RegistrationConfig(method="invalid_method")


# ---------------------------------------------------------------------------
# Tests for RegistrationConfig validation
# ---------------------------------------------------------------------------

class TestRegistrationConfig:
    """Tests for RegistrationConfig dataclass."""

    def test_defaults(self):
        config = RegistrationConfig()
        assert config.method == "vgicp"
        assert config.max_correspondence_distance == 1.0
        assert config.voxel_resolution == 0.5
        assert config.multi_resolution is True
        assert config.auto_revert is True
        assert config.crop_to_overlap is True

    def test_method_normalization(self):
        config = RegistrationConfig(method="VGICP")
        assert config.method == "vgicp"

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            RegistrationConfig(method="invalid")

    def test_invalid_voxel_resolution(self):
        with pytest.raises(ValueError):
            RegistrationConfig(voxel_resolution=-1.0)

    def test_invalid_point_filter(self):
        with pytest.raises(ValueError):
            RegistrationConfig(point_filter="invalid_filter")

    def test_to_dict(self):
        config = RegistrationConfig()
        d = config.to_dict()
        assert d['method'] == 'vgicp'
        assert d['voxel_resolution'] == 0.5

    def test_list_point_filter(self):
        config = RegistrationConfig(point_filter=[2, 6])
        assert config.point_filter == [2, 6]


# ---------------------------------------------------------------------------
# Backward-compat shims
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Verify deprecated shims still work."""

    def test_registration_method_enum(self):
        assert RegistrationMethod.VGICP == "vgicp"
        assert RegistrationMethod.GICP == "gicp"
        assert RegistrationMethod.ICP == "icp"
        assert RegistrationMethod.PLANE_ICP == "plane_icp"

    @requires_small_gicp
    def test_landscape_aligner_shim(self):
        """LandscapeAligner shim should delegate to align_point_clouds."""
        import warnings
        target = _create_synthetic_terrain(n_points=20000, seed=42)
        source = target + np.array([1.0, 0.0, 0.0])

        config = RegistrationConfig(verbose=False)
        aligner = LandscapeAligner(config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = aligner.align(source, target)

        assert result.converged
        assert isinstance(result, RegistrationResult)


# ---------------------------------------------------------------------------
# Full-pipeline tests (require PDAL + small_gicp + synthetic LAZ fixtures)
# ---------------------------------------------------------------------------

@requires_pdal
@requires_small_gicp
class TestAlignmentBasics:
    """Tests for basic alignment through PointCloudPair."""

    def test_align_point_clouds_vgicp(self, pc_pair):
        """Test alignment using VGICP method."""
        align_result = pc_pair.align_point_clouds(method="vgicp")

        assert align_result is not None
        assert 'transformation' in align_result
        assert 'rmse' in align_result
        assert 'fitness' in align_result
        assert 'pre_rmse' in align_result
        assert 'reverted' in align_result
        assert align_result['transformation'].shape == (4, 4)
        assert align_result['rmse'] >= 0
        assert 0.0 <= align_result['fitness'] <= 1.0

    def test_align_with_config_object(self, pc_pair):
        """Test alignment by passing a RegistrationConfig directly."""
        config = RegistrationConfig(
            method="vgicp",
            voxel_resolution=1.0,
            multi_resolution=False,
            verbose=False,
        )
        align_result = pc_pair.align_point_clouds(alignment_config=config)

        assert align_result is not None
        assert align_result['transformation'].shape == (4, 4)
        # Verify the config was passed through (single resolution)
        assert align_result['method'] == 'vgicp'


@requires_pdal
@requires_small_gicp
class TestAlignmentResults:
    """Tests for alignment result validation."""

    @pytest.fixture
    def alignment_result(self, pc_pair):
        """Create an alignment result for testing."""
        return pc_pair.align_point_clouds(method="vgicp", verbose=False)

    def test_transformation_matrix_properties(self, alignment_result):
        T = alignment_result['transformation']
        assert T.shape == (4, 4)
        assert np.allclose(T[3, :], [0, 0, 0, 1])
        R = T[:3, :3]
        RTR = R.T @ R
        assert np.allclose(RTR, np.eye(3), atol=0.1)

    def test_rmse_is_reasonable(self, alignment_result):
        assert alignment_result['rmse'] > 0
        assert alignment_result['rmse'] < 10.0

    def test_fitness_is_reasonable(self, alignment_result):
        fitness = alignment_result['fitness']
        assert 0.0 <= fitness <= 1.0
        assert fitness > 0.3

    def test_convergence(self, alignment_result):
        assert alignment_result['converged'] is True


@requires_pdal
@requires_small_gicp
class TestAlignmentWithTransformation:
    """Tests for alignment followed by transformation application."""

    def test_alignment_transformation_workflow(self, compare_pc, reference_pc):
        """Test complete alignment and transformation workflow."""
        pc_pair = PointCloudPair(compare_pc, reference_pc)
        result = pc_pair.align_point_clouds(method="vgicp", verbose=False)

        assert result is not None
        assert result['converged'] is True
        assert pc_pair.pc1 is not None


# ---------------------------------------------------------------------------
# Legacy synthetic tests (raw small_gicp calls, no wrapper)
# ---------------------------------------------------------------------------

class TestSyntheticAlignment:
    """Legacy tests using raw small_gicp.align() calls for sanity checking."""

    @requires_small_gicp
    def test_vgicp_vs_gicp_consistency(self):
        """VGICP and GICP should produce similar translations."""
        import small_gicp

        target = _create_synthetic_terrain(n_points=30000, seed=456)
        source = target + np.array([1.0, -0.5, 0.2])

        result_gicp = small_gicp.align(
            target, source,
            registration_type="GICP",
            downsampling_resolution=0.5,
            max_correspondence_distance=5.0,
            num_threads=4,
        )
        result_vgicp = small_gicp.align(
            target, source,
            registration_type="VGICP",
            voxel_resolution=0.5,
            downsampling_resolution=0.5,
            max_correspondence_distance=5.0,
            num_threads=4,
        )

        assert result_gicp.converged
        assert result_vgicp.converged
        np.testing.assert_allclose(
            result_gicp.T_target_source[:3, 3],
            result_vgicp.T_target_source[:3, 3],
            atol=0.2,
        )

    @requires_small_gicp
    def test_alignment_with_partial_overlap(self):
        """Alignment should converge even with only partial overlap."""
        import small_gicp

        np.random.seed(789)
        tx = np.random.uniform(0, 100, 20000)
        ty = np.random.uniform(0, 100, 20000)
        tz = np.sin(tx / 10) + np.cos(ty / 10)
        target = np.column_stack([tx, ty, tz]).astype(np.float64)

        known_translation = np.array([0.5, -0.3, 0.1])
        sx = np.random.uniform(50, 150, 20000)
        sy = np.random.uniform(0, 100, 20000)
        sz = np.sin(sx / 10) + np.cos(sy / 10)
        source = np.column_stack([sx, sy, sz]).astype(np.float64) + known_translation

        result = small_gicp.align(
            target, source,
            registration_type="GICP",
            downsampling_resolution=1.0,
            max_correspondence_distance=10.0,
            num_threads=4,
        )

        assert result.converged
        assert result.num_inliers > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
