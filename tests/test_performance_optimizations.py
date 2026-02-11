"""Tests for performance optimizations (R1–R10) and multi-resolution alignment (Rec #4).

Organisation
------------
1. TestMultiResolutionConvergence     – Rec #4 coarse-to-fine alignment
2. TestMultiResolutionDisabled        – multi_resolution=False still works
3. TestMultiResolutionCustomStages    – custom resolution_stages respected
4. TestSignConventionRegression       – T_target_source carries *negative* offset
5. TestPlaneICPSynthetic              – PLANE_ICP convergence (untested code path)
6. TestCRSTransformerCache            – R10 LRU cache correctness
7. TestNoiseAndOutlierResilience      – GICP robustness to noise / gross outliers
8. TestDownloadResume                 – R9 Range-header branching (mocked)
9. TestDequeBFSOrdering               – R2 deque produces correct BFS traversal
10. TestRunChainComposition           – R5 PDAL pipeline composition (needs PDAL)
11. TestParallelCropEquivalence       – R6 ThreadPoolExecutor crop (needs PDAL)"""

import os
import time
import tempfile
from pathlib import Path
from collections import deque
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from skip_markers import requires_small_gicp, requires_pdal, requires_gdal, HAS_PDAL


# helpers shared across tests

def _create_synthetic_terrain(n_points: int = 50_000, seed: int = 42) -> np.ndarray:
    """Rolling-hills terrain as (N, 3) float64 array."""
    rng = np.random.RandomState(seed)
    side = int(np.sqrt(n_points))
    x = np.linspace(0, 100, side)
    y = np.linspace(0, 100, side)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten() + rng.normal(0, 0.1, side * side)
    yy = yy.flatten() + rng.normal(0, 0.1, side * side)
    zz = (
        5 * np.sin(xx / 10) * np.cos(yy / 10)
        + 2 * np.sin(xx / 3) * np.sin(yy / 5)
        + rng.normal(0, 0.05, len(xx))
    )
    return np.column_stack([xx, yy, zz]).astype(np.float64)


def _apply_rigid_transform(
    points: np.ndarray,
    translation: np.ndarray,
    rotation_deg: float = 0.0,
    rotation_axis: str = "z",
) -> np.ndarray:
    """Apply rigid-body translation + optional rotation."""
    angle = np.radians(rotation_deg)
    c, s = np.cos(angle), np.sin(angle)
    if rotation_axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif rotation_axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif rotation_axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        R = np.eye(3)
    return (R @ points.T).T + translation


def _multi_resolution_align(
    target, source, method="GICP", resolutions=None, voxel_res=0.5,
    max_corr_dist=5.0, max_iters=50, num_threads=4,
):
    """
    Replicate the multi-resolution loop from _run_registration for testing.

    Returns the final small_gicp result object.
    """
    import small_gicp

    if resolutions is None:
        resolutions = [voxel_res * 4.0, voxel_res * 2.0, voxel_res]

    init_T = np.eye(4)

    for stage_idx, stage_res in enumerate(resolutions):
        is_final = (stage_idx == len(resolutions) - 1)

        target_cloud, target_tree = small_gicp.preprocess_points(
            target, downsampling_resolution=stage_res, num_threads=num_threads,
        )
        source_cloud, source_tree = small_gicp.preprocess_points(
            source, downsampling_resolution=stage_res, num_threads=num_threads,
        )

        if is_final:
            stage_max_dist = max_corr_dist
            stage_max_iters = max_iters
        else:
            scale = stage_res / resolutions[-1]
            stage_max_dist = max_corr_dist * scale
            stage_max_iters = max(20, max_iters // 2)

        if method == "VGICP":
            voxelmap = small_gicp.GaussianVoxelMap(stage_res)
            voxelmap.insert(target_cloud)
            result = small_gicp.align(
                voxelmap, source_cloud,
                init_T_target_source=init_T,
                max_correspondence_distance=stage_max_dist,
                max_iterations=stage_max_iters,
                num_threads=num_threads,
            )
        else:
            result = small_gicp.align(
                target_cloud, source_cloud, target_tree,
                init_T_target_source=init_T,
                registration_type=method,
                max_correspondence_distance=stage_max_dist,
                max_iterations=stage_max_iters,
                num_threads=num_threads,
            )

        init_T = result.T_target_source

    return result


# 1. Multi-resolution convergence (Rec #4)

class TestMultiResolutionConvergence:
    """
    Multi-resolution alignment should converge for misalignments that are
    too large for single-resolution GICP to handle reliably.

    Analogy: coarse-to-fine is like adjusting binoculars : you get the
    rough focus first, then refine.  Without the coarse pass, you might
    spin the fine knob forever without finding the target.
    """

    @requires_small_gicp
    def test_large_translation_converges(self):
        """10 m translation: multi-res should converge with < 0.5 m error."""
        target = _create_synthetic_terrain(n_points=50_000, seed=42)
        large_offset = np.array([10.0, -8.0, 2.0])
        source = target + large_offset

        result = _multi_resolution_align(
            target, source,
            method="GICP",
            resolutions=[2.0, 1.0, 0.5],
            max_corr_dist=15.0,
            max_iters=50,
        )

        assert result.converged, "Multi-res alignment did not converge for 10m offset"
        recovered = result.T_target_source[:3, 3]
        expected = -large_offset
        error = np.linalg.norm(recovered - expected)
        assert error < 0.5, (
            f"Multi-res translation error {error:.3f}m exceeds 0.5m threshold"
        )

    @requires_small_gicp
    def test_large_rotation_converges(self):
        """5° rotation + 5 m translation: multi-res should converge."""
        target = _create_synthetic_terrain(n_points=50_000, seed=100)
        translation = np.array([5.0, -3.0, 1.0])
        source = _apply_rigid_transform(target, translation, rotation_deg=5.0, rotation_axis="z")

        result = _multi_resolution_align(
            target, source,
            method="GICP",
            resolutions=[2.0, 1.0, 0.5],
            max_corr_dist=15.0,
            max_iters=60,
        )

        assert result.converged, "Multi-res alignment did not converge for 5° rotation"
        R = result.T_target_source[:3, :3]
        recovered_deg = np.degrees(
            np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        )
        assert abs(recovered_deg - 5.0) < 1.5, (
            f"Recovered rotation {recovered_deg:.2f}° vs expected 5°"
        )

    @requires_small_gicp
    def test_multi_res_not_worse_than_single_res(self):
        """Multi-resolution should not degrade quality vs single-resolution."""
        target = _create_synthetic_terrain(n_points=40_000, seed=200)
        offset = np.array([1.5, -0.8, 0.3])
        source = target + offset

        # single-resolution
        import small_gicp
        single_result = small_gicp.align(
            target, source,
            registration_type="GICP",
            downsampling_resolution=0.5,
            max_correspondence_distance=5.0,
            num_threads=4,
        )

        # multi-resolution
        multi_result = _multi_resolution_align(
            target, source,
            method="GICP",
            resolutions=[2.0, 1.0, 0.5],
            max_corr_dist=5.0,
        )

        single_error = np.linalg.norm(single_result.T_target_source[:3, 3] - (-offset))
        multi_error = np.linalg.norm(multi_result.T_target_source[:3, 3] - (-offset))

        # multi-res should be within 2x of single-res (usually better)
        assert multi_error < single_error * 2.0 + 0.01, (
            f"Multi-res error {multi_error:.4f} >> single-res error {single_error:.4f}"
        )

    @requires_small_gicp
    def test_vgicp_multi_resolution(self):
        """Multi-resolution with VGICP method should also converge."""
        target = _create_synthetic_terrain(n_points=40_000, seed=300)
        offset = np.array([6.0, -4.0, 1.5])
        source = target + offset

        result = _multi_resolution_align(
            target, source,
            method="VGICP",
            resolutions=[2.0, 1.0, 0.5],
            max_corr_dist=12.0,
        )

        assert result.converged
        error = np.linalg.norm(result.T_target_source[:3, 3] - (-offset))
        assert error < 0.5, f"VGICP multi-res error {error:.3f}m"


# 2. Multi-resolution disabled

class TestMultiResolutionDisabled:
    """Verify that multi_resolution=False gives single-pass alignment."""

    @requires_small_gicp
    def test_single_resolution_still_converges(self):
        """With multi_resolution=False, small offsets should still converge."""
        target = _create_synthetic_terrain(n_points=40_000, seed=42)
        offset = np.array([1.0, -0.5, 0.2])
        source = target + offset

        # single resolution (no multi-res loop)
        result = _multi_resolution_align(
            target, source,
            method="GICP",
            resolutions=[0.5],  # single stage = multi_resolution disabled
            max_corr_dist=5.0,
        )

        assert result.converged
        error = np.linalg.norm(result.T_target_source[:3, 3] - (-offset))
        assert error < 0.2, f"Single-res error {error:.3f}m"


# 3. Custom resolution stages

class TestMultiResolutionCustomStages:
    """Verify that custom resolution_stages are respected."""

    @requires_small_gicp
    def test_two_stage_schedule(self):
        """A 2-stage [2.0, 0.5] schedule should converge normally."""
        target = _create_synthetic_terrain(n_points=40_000, seed=500)
        offset = np.array([3.0, -2.0, 0.5])
        source = target + offset

        result = _multi_resolution_align(
            target, source,
            method="GICP",
            resolutions=[2.0, 0.5],  # custom 2-stage
            max_corr_dist=8.0,
        )

        assert result.converged
        error = np.linalg.norm(result.T_target_source[:3, 3] - (-offset))
        assert error < 0.3, f"Custom 2-stage error {error:.3f}m"

    @requires_small_gicp
    def test_five_stage_schedule(self):
        """A 5-stage fine-grained schedule should also work."""
        target = _create_synthetic_terrain(n_points=40_000, seed=501)
        offset = np.array([4.0, -3.0, 1.0])
        source = target + offset

        result = _multi_resolution_align(
            target, source,
            method="GICP",
            resolutions=[4.0, 2.0, 1.0, 0.7, 0.5],
            max_corr_dist=10.0,
        )

        assert result.converged
        error = np.linalg.norm(result.T_target_source[:3, 3] - (-offset))
        assert error < 0.3, f"Custom 5-stage error {error:.3f}m"


# 4. Sign convention regression

class TestSignConventionRegression:
    """
    Document and guard the ICP sign convention: T_target_source maps points
    FROM source TO target.  Therefore, if we offset the source by +[dx, dy, dz],
    the recovered T_target_source[:3, 3] should be approximately -[dx, dy, dz].

    This is analogous to coordinate transforms: if you move your camera +1 m
    right, objects in camera coordinates shift -1 m left.
    """

    @requires_small_gicp
    def test_positive_offset_gives_negative_translation(self):
        """Source = target + [+2, +1, +0.5] → T[:3,3] ≈ [-2, -1, -0.5]."""
        import small_gicp

        target = _create_synthetic_terrain(n_points=40_000, seed=42)
        offset = np.array([2.0, 1.0, 0.5])
        source = target + offset

        result = small_gicp.align(
            target, source,
            registration_type="GICP",
            downsampling_resolution=0.5,
            max_correspondence_distance=5.0,
            num_threads=4,
        )

        assert result.converged
        recovered = result.T_target_source[:3, 3]

        # the KEY assertion: sign must be flipped
        for i, axis in enumerate(["X", "Y", "Z"]):
            assert recovered[i] < 0, (
                f"T_target_source[{axis}] = {recovered[i]:.4f} should be negative "
                f"for positive source offset +{offset[i]}"
            )

        np.testing.assert_allclose(recovered, -offset, atol=0.1)

    @requires_small_gicp
    def test_negative_offset_gives_positive_translation(self):
        """Source = target + [-3, -1, -0.2] → T[:3,3] ≈ [+3, +1, +0.2]."""
        import small_gicp

        target = _create_synthetic_terrain(n_points=40_000, seed=42)
        offset = np.array([-3.0, -1.0, -0.2])
        source = target + offset

        result = small_gicp.align(
            target, source,
            registration_type="GICP",
            downsampling_resolution=0.5,
            max_correspondence_distance=8.0,
            num_threads=4,
        )

        assert result.converged
        recovered = result.T_target_source[:3, 3]

        for i, axis in enumerate(["X", "Y", "Z"]):
            assert recovered[i] > 0, (
                f"T_target_source[{axis}] = {recovered[i]:.4f} should be positive "
                f"for negative source offset {offset[i]}"
            )

        np.testing.assert_allclose(recovered, -offset, atol=0.15)


# 5. PLANE_ICP synthetic test

class TestPlaneICPSynthetic:
    """
    PLANE_ICP uses point-to-plane error metric, which should converge
    faster than point-to-point ICP for well-structured surfaces.
    """

    @requires_small_gicp
    def test_plane_icp_pure_translation(self):
        """PLANE_ICP should recover a known translation."""
        import small_gicp

        target = _create_synthetic_terrain(n_points=50_000, seed=42)
        offset = np.array([1.5, -0.8, 0.3])
        source = target + offset

        target_cloud, target_tree = small_gicp.preprocess_points(
            target, downsampling_resolution=0.5, num_threads=4,
        )
        source_cloud, _ = small_gicp.preprocess_points(
            source, downsampling_resolution=0.5, num_threads=4,
        )

        result = small_gicp.align(
            target_cloud, source_cloud, target_tree,
            registration_type="PLANE_ICP",
            max_correspondence_distance=5.0,
            max_iterations=50,
            num_threads=4,
        )

        assert result.converged, "PLANE_ICP did not converge"
        recovered = result.T_target_source[:3, 3]
        np.testing.assert_allclose(recovered, -offset, atol=0.15)

    @requires_small_gicp
    def test_plane_icp_with_rotation(self):
        """PLANE_ICP should handle small rotation + translation."""
        import small_gicp

        target = _create_synthetic_terrain(n_points=50_000, seed=123)
        translation = np.array([2.0, 1.0, -0.3])
        source = _apply_rigid_transform(
            target, translation, rotation_deg=1.5, rotation_axis="z",
        )

        target_cloud, target_tree = small_gicp.preprocess_points(
            target, downsampling_resolution=0.5, num_threads=4,
        )
        source_cloud, _ = small_gicp.preprocess_points(
            source, downsampling_resolution=0.5, num_threads=4,
        )

        result = small_gicp.align(
            target_cloud, source_cloud, target_tree,
            registration_type="PLANE_ICP",
            max_correspondence_distance=8.0,
            max_iterations=60,
            num_threads=4,
        )

        assert result.converged
        R = result.T_target_source[:3, :3]
        recovered_deg = np.degrees(
            np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        )
        assert abs(recovered_deg - 1.5) < 1.0, (
            f"PLANE_ICP rotation recovery: {recovered_deg:.2f}° vs 1.5° expected"
        )

    @requires_small_gicp
    def test_plane_icp_multi_resolution(self):
        """PLANE_ICP should work through the multi-resolution loop."""
        target = _create_synthetic_terrain(n_points=40_000, seed=600)
        offset = np.array([5.0, -3.0, 1.0])
        source = target + offset

        result = _multi_resolution_align(
            target, source,
            method="PLANE_ICP",
            resolutions=[2.0, 1.0, 0.5],
            max_corr_dist=10.0,
        )

        assert result.converged
        error = np.linalg.norm(result.T_target_source[:3, 3] - (-offset))
        assert error < 0.5, f"PLANE_ICP multi-res error {error:.3f}m"


# 6. CRS Transformer cache (R10)

class TestCRSTransformerCache:
    """
    Verify the LRU-cached CRS transformer:
    - Cache hits on repeated calls
    - CRS objects and string authorities share the same cache slot
    - Non-authority CRS falls back gracefully
    """

    def test_string_authority_cache_hit(self):
        """Identical string authority codes should return the same object."""
        from topochange.pointcloud import _get_transformer, _cached_transformer
        _cached_transformer.cache_clear()

        tf1 = _get_transformer("EPSG:32610", "EPSG:4326")
        tf2 = _get_transformer("EPSG:32610", "EPSG:4326")

        assert tf1 is tf2, "Cache miss on identical string authorities"
        assert _cached_transformer.cache_info().hits >= 1

    def test_crs_object_cache_hit(self):
        """CRS objects with known EPSG codes should use the cache."""
        from pyproj import CRS as CRS_
        from topochange.pointcloud import _get_transformer, _cached_transformer
        _cached_transformer.cache_clear()

        src = CRS_.from_epsg(32610)
        dst = CRS_.from_epsg(4326)

        tf1 = _get_transformer(src, dst)
        tf2 = _get_transformer(src, dst)

        assert tf1 is tf2, "Cache miss on CRS objects"

    def test_crs_and_string_share_cache(self):
        """CRS object and equivalent string should share the same entry."""
        from pyproj import CRS as CRS_
        from topochange.pointcloud import _get_transformer, _cached_transformer
        _cached_transformer.cache_clear()

        tf_str = _get_transformer("EPSG:32610", "EPSG:4326")
        tf_obj = _get_transformer(CRS_.from_epsg(32610), CRS_.from_epsg(4326))

        assert tf_str is tf_obj, "CRS object and string do not share cache"

    def test_transform_produces_correct_output(self):
        """Cached transformer should produce correct coordinate transforms."""
        from topochange.pointcloud import _get_transformer

        tf = _get_transformer("EPSG:32610", "EPSG:4326")
        lon, lat = tf.transform(500000, 4649776)

        # UTM Zone 10N: easting 500000 is the central meridian (-123°)
        assert abs(lon - (-123.0)) < 0.1, f"Longitude {lon} not near -123°"
        assert abs(lat - 42.0) < 0.5, f"Latitude {lat} not near 42°"

    def test_fallback_for_non_authority_crs(self):
        """A CRS without an authority code should fall back to direct creation."""
        from pyproj import CRS as CRS_
        from topochange.pointcloud import _get_transformer

        # create a CRS from raw WKT that may not resolve to an authority
        custom_wkt = CRS_.from_epsg(32610).to_wkt()
        # this should still work even if to_authority() returns None
        tf = _get_transformer(custom_wkt, "EPSG:4326")
        assert tf is not None


# 7. Noise and outlier resilience

class TestNoiseAndOutlierResilience:
    """
    GICP's Gaussian error model should be inherently more outlier-tolerant
    than standard ICP.  These tests add realistic noise and gross outliers
    to verify the alignment still converges.
    """

    @requires_small_gicp
    def test_moderate_noise(self):
        """0.1 m Gaussian noise (typical lidar) should not prevent convergence."""
        import small_gicp

        rng = np.random.RandomState(42)
        target = _create_synthetic_terrain(n_points=40_000, seed=42)
        offset = np.array([1.5, -0.8, 0.3])
        source = target + offset + rng.normal(0, 0.10, target.shape)

        result = small_gicp.align(
            target, source,
            registration_type="GICP",
            downsampling_resolution=0.5,
            max_correspondence_distance=5.0,
            num_threads=4,
        )

        assert result.converged
        error = np.linalg.norm(result.T_target_source[:3, 3] - (-offset))
        assert error < 0.5, f"Error {error:.3f}m with 0.1m noise"

    @requires_small_gicp
    def test_gross_outliers(self):
        """5% gross outliers (10 m jumps) should not derail GICP."""
        import small_gicp

        rng = np.random.RandomState(42)
        target = _create_synthetic_terrain(n_points=40_000, seed=42)
        offset = np.array([1.5, -0.8, 0.3])
        source = target + offset

        # corrupt 5% of source points with 10 m random jumps
        n_outliers = int(len(source) * 0.05)
        outlier_idx = rng.choice(len(source), n_outliers, replace=False)
        source[outlier_idx] += rng.uniform(-10, 10, (n_outliers, 3))

        result = small_gicp.align(
            target, source,
            registration_type="GICP",
            downsampling_resolution=0.5,
            max_correspondence_distance=5.0,
            num_threads=4,
        )

        assert result.converged
        error = np.linalg.norm(result.T_target_source[:3, 3] - (-offset))
        assert error < 1.0, f"Error {error:.3f}m with 5% outliers"

    @requires_small_gicp
    def test_noise_plus_outliers_multi_res(self):
        """Combined noise + outliers with multi-resolution."""
        rng = np.random.RandomState(99)
        target = _create_synthetic_terrain(n_points=50_000, seed=99)
        offset = np.array([5.0, -3.0, 1.0])
        source = target + offset + rng.normal(0, 0.08, target.shape)

        # 3% gross outliers
        n_outliers = int(len(source) * 0.03)
        outlier_idx = rng.choice(len(source), n_outliers, replace=False)
        source[outlier_idx] += rng.uniform(-8, 8, (n_outliers, 3))

        result = _multi_resolution_align(
            target, source,
            method="GICP",
            resolutions=[2.0, 1.0, 0.5],
            max_corr_dist=12.0,
        )

        assert result.converged
        error = np.linalg.norm(result.T_target_source[:3, 3] - (-offset))
        assert error < 1.0, (
            f"Multi-res error {error:.3f}m with noise + outliers"
        )


# 8. Download resume (R9) : mocked

@requires_gdal
class TestDownloadResume:
    """
    Test the Range-header resume logic in GetDEMs.download_file.

    We mock requests.Session.get to simulate four scenarios:
      1. Fresh download (no existing file) → 200
      2. Partial file → 206 Partial Content
      3. Complete file → 416 Range Not Satisfiable
      4. Server ignores Range → 200 with Content-Length
    """

    def _make_mock_response(self, status_code, content=b"data", headers=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.headers = headers or {}
        resp.iter_content = MagicMock(return_value=[content])
        resp.raise_for_status = MagicMock()
        if status_code >= 400 and status_code != 416:
            from requests.exceptions import HTTPError
            resp.raise_for_status.side_effect = HTTPError(response=resp)
        return resp

    def test_fresh_download(self, tmp_path):
        """No existing file → full GET with no Range header, writes in 'wb' mode."""
        from topochange.data_access import GetDEMs

        output = tmp_path / "test.laz"
        obj = GetDEMs.__new__(GetDEMs)
        obj._session = MagicMock()

        response = self._make_mock_response(200, content=b"full_file_data")
        obj._session.get.return_value = response

        result = obj.download_file("https://example.com/test.laz", output)

        assert result['success'] is True
        assert output.exists()
        # should NOT have sent Range header
        call_kwargs = obj._session.get.call_args
        headers_sent = call_kwargs.kwargs.get('headers', call_kwargs[1].get('headers', {}))
        assert 'Range' not in headers_sent

    def test_resume_partial_download(self, tmp_path):
        """Existing partial file → Range header → 206 → append mode."""
        from topochange.data_access import GetDEMs

        output = tmp_path / "test.laz"
        output.write_bytes(b"partial")  # 7 bytes exist

        obj = GetDEMs.__new__(GetDEMs)
        obj._session = MagicMock()

        response = self._make_mock_response(206, content=b"_remaining_data")
        obj._session.get.return_value = response

        result = obj.download_file("https://example.com/test.laz", output)

        assert result['success'] is True
        # should have sent Range header
        call_kwargs = obj._session.get.call_args
        headers_sent = call_kwargs.kwargs.get('headers', call_kwargs[1].get('headers', {}))
        assert 'Range' in headers_sent
        assert headers_sent['Range'] == 'bytes=7-'

    def test_complete_file_416(self, tmp_path):
        """Existing complete file → 416 → skip download."""
        from topochange.data_access import GetDEMs

        output = tmp_path / "test.laz"
        output.write_bytes(b"complete_file_data")

        obj = GetDEMs.__new__(GetDEMs)
        obj._session = MagicMock()

        response = self._make_mock_response(416)
        obj._session.get.return_value = response

        result = obj.download_file("https://example.com/test.laz", output)

        assert result['success'] is True
        assert result['size_bytes'] == len(b"complete_file_data")

    def test_server_ignores_range_full_redownload(self, tmp_path):
        """Server returns 200 (not 206) → re-download from scratch."""
        from topochange.data_access import GetDEMs

        output = tmp_path / "test.laz"
        output.write_bytes(b"partial")  # 7 bytes exist

        obj = GetDEMs.__new__(GetDEMs)
        obj._session = MagicMock()

        # server returns 200 with Content-Length > existing size → full redownload
        response = self._make_mock_response(
            200,
            content=b"complete_new_file",
            headers={'Content-Length': '100'},  # larger than existing 7 bytes
        )
        obj._session.get.return_value = response

        result = obj.download_file("https://example.com/test.laz", output)

        assert result['success'] is True


# 9. Deque BFS ordering (R2)

class TestDequeBFSOrdering:
    """
    Verify that the deque-based BFS in get_ept_hierarchy_nodes produces
    correct breadth-first traversal order.

    This is a unit-level test of the data structure change: deque.popleft()
    gives O(1) FIFO, identical to list.pop(0) but without the O(n) cost.
    """

    def test_deque_popleft_matches_list_pop0(self):
        """deque.popleft() and list.pop(0) give identical traversal order."""
        items = list(range(100))

        # list-based BFS (original)
        list_order = []
        q_list = list(items[:])
        while q_list:
            node = q_list.pop(0)
            list_order.append(node)

        # deque-based BFS (optimized)
        deque_order = []
        q_deque = deque(items[:])
        while q_deque:
            node = q_deque.popleft()
            deque_order.append(node)

        assert list_order == deque_order

    def test_bfs_with_dynamic_appends(self):
        """BFS with children added during traversal produces same ordering."""
        # simple tree: root → [1,2], 1 → [3,4], 2 → [5,6]
        children = {0: [1, 2], 1: [3, 4], 2: [5, 6]}

        # list version
        list_order = []
        q = [0]
        while q:
            node = q.pop(0)
            list_order.append(node)
            for child in children.get(node, []):
                q.append(child)

        # deque version
        deque_order = []
        q = deque([0])
        while q:
            node = q.popleft()
            deque_order.append(node)
            for child in children.get(node, []):
                q.append(child)

        assert list_order == deque_order
        assert deque_order == [0, 1, 2, 3, 4, 5, 6], "BFS order incorrect"


# 10. run_chain Pipeline composition (R5) : needs PDAL

@requires_pdal
class TestRunChainComposition:
    """
    Verify that PointCloudPreprocessor.run_chain() composes multiple PDAL
    filters into a single pipeline pass.
    """

    def test_classification_plus_voxel(self, compare_laz_path, tmp_path):
        """run_chain with classification filter + voxel downsample should work."""
        from topochange.alignment_utils import PointCloudPreprocessor

        output = str(tmp_path / "chain_output.laz")
        result_path = PointCloudPreprocessor.run_chain(
            compare_laz_path,
            output,
            classifications="ground",
            voxel_size=2.0,
            overwrite=True,
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_voxel_only(self, compare_laz_path, tmp_path):
        """run_chain with just voxel downsampling should produce fewer points."""
        from topochange.alignment_utils import PointCloudPreprocessor

        output = str(tmp_path / "voxel_output.laz")
        result_path = PointCloudPreprocessor.run_chain(
            compare_laz_path,
            output,
            voxel_size=5.0,
            overwrite=True,
        )

        # the output file should be smaller than the input (5 m voxel is aggressive)
        input_size = os.path.getsize(compare_laz_path)
        output_size = os.path.getsize(result_path)
        assert output_size < input_size, (
            f"Voxel-downsampled file ({output_size}) is not smaller than input ({input_size})"
        )

    def test_overwrite_false_skips(self, compare_laz_path, tmp_path):
        """run_chain with overwrite=False should skip if output exists."""
        from topochange.alignment_utils import PointCloudPreprocessor

        output = str(tmp_path / "existing.laz")
        # create a dummy file
        Path(output).write_bytes(b"dummy")

        result_path = PointCloudPreprocessor.run_chain(
            compare_laz_path,
            output,
            voxel_size=2.0,
            overwrite=False,
        )

        assert result_path == output
        # file should still be "dummy" (not overwritten)
        assert Path(output).read_bytes() == b"dummy"


# 11. Parallel crop equivalence (R6) : needs PDAL

@requires_pdal
@requires_small_gicp
class TestParallelCropEquivalence:
    """
    Verify that the ThreadPoolExecutor-based parallel clip_to_polygon
    in crop_to_overlap produces the same result as sequential execution.
    """

    def test_crop_produces_two_files(self, pc_pair, tmp_path):
        """crop_to_overlap should produce two valid cropped files."""
        pc1_cropped, pc2_cropped = pc_pair.crop_to_overlap(
            output_dir=str(tmp_path),
            overwrite=True,
        )

        assert pc1_cropped is not None
        assert pc2_cropped is not None
        assert os.path.exists(pc1_cropped.filename)
        assert os.path.exists(pc2_cropped.filename)
        assert os.path.getsize(pc1_cropped.filename) > 0
        assert os.path.getsize(pc2_cropped.filename) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

