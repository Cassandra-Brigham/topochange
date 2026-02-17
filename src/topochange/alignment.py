"""Point cloud registration using ICP-based methods via small_gicp.

This module provides the core alignment function and configuration for
registering point clouds using ICP, GICP, VGICP, or Plane-ICP methods.
All downsampling during alignment is handled by small_gicp.preprocess_points(),
ensuring a single, consistent downsampling path.

The main entry point is align_point_clouds(), which takes centered Nx3 arrays
and a RegistrationConfig, runs multi-resolution coarse-to-fine registration,
and returns a RegistrationResult with quality metrics.

For file I/O utilities (loading, saving, preprocessing), see alignment_utils.py.

References
----------
- Besl & McKay (1992). A Method for Registration of 3-D Shapes.
- Koide et al. (2021). Voxelized GICP for Fast and Accurate 3D Point Cloud
  Registration. https://doi.org/10.1109/ICRA48506.2021.9560835
- Koide (2024). small_gicp: Efficient and parallel algorithms for point cloud
  registration. JOSS 9(103), 6948. https://doi.org/10.21105/joss.06948
- Scott et al. (2021). Measuring change at Earth's surface: On-demand vertical
  and three-dimensional topographic differencing. Geosphere 18(1), 210-232.
  https://doi.org/10.1130/GES02259.1
"""

import gc
import logging
import os
import sys
import warnings
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union, Any, TYPE_CHECKING

from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree

# small_gicp import (optional dependency)
try:
    import small_gicp
    _HAS_SMALL_GICP = True
except ImportError:
    small_gicp = None
    _HAS_SMALL_GICP = False

# Import shared utilities (also used by pointcloudpair.py)
from .alignment_utils import (
    run_pdal_pipeline,
    load_points_from_las,
    save_transformed_las,
    compute_alignment_quality,
    PointCloudPreprocessor,
    AlignmentQualityMetrics,
    require_small_gicp,
    has_small_gicp,
    decompose_transformation,
)

if TYPE_CHECKING:
    from .pointcloud import PointCloud

logger = logging.getLogger(__name__)

# Valid registration methods (maps to small_gicp registration_type strings)
VALID_METHODS = {"icp", "plane_icp", "gicp", "vgicp"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegistrationConfig:
    """Configuration for point cloud registration.

    This dataclass controls all aspects of the alignment pipeline. Fields are
    organized into two groups:

    **Core registration** parameters are consumed directly by
    ``align_point_clouds()`` (the pure-array function).

    **Orchestration** parameters (point_filter, crop_*, max_points) are used
    by ``PointCloudPair.align_point_clouds()`` to prepare data before calling
    the core function.

    Parameters
    ----------
    method : str
        Registration method: "vgicp" (default), "gicp", "icp", or "plane_icp".
    max_correspondence_distance : float
        Maximum distance for point correspondences in meters (default 1.0).
    max_iterations : int
        Maximum ICP iterations per resolution stage (default 50).
    num_threads : int
        Number of threads for parallel processing (default: up to 8 cores).
    voxel_resolution : float
        Finest voxel resolution in meters for small_gicp preprocessing
        (default 0.5). Coarser stages are derived automatically.
    multi_resolution : bool
        Enable coarse-to-fine multi-resolution alignment (default True).
    resolution_stages : list of float, optional
        Custom resolution schedule ordered coarse-to-fine. If None and
        multi_resolution is True, stages are auto-computed as [4x, 2x, 1x]
        of voxel_resolution.
    point_filter : str or list of int
        Classification filter for loading: "ground", "all", or list of codes.
    max_points : int, optional
        Hard cap on points per cloud (random subsample after filtering).
    crop_to_overlap : bool
        Crop both clouds to their overlap area before alignment (default True).
    crop_buffer : float
        Buffer in meters added to the compare cloud's overlap crop (default 10).
    crop_box_size : tuple of (float, float), optional
        If set, further restrict alignment to an NxN meter box centered on the
        overlap centroid. Transform is still applied to the full cloud.
    crop_box_buffer : float
        Additional buffer in meters for the compare cloud's box crop (default 0).
    auto_revert : bool
        Automatically revert to identity if alignment increases RMSE (default True).
    apply_transform : bool
        Whether to apply and save the aligned point cloud (default True).
    output_path : str, optional
        Output path for the aligned point cloud (auto-generated if None).
    overwrite : bool
        Overwrite existing output files (default True).
    verbose : bool
        Print progress information (default True).

    Examples
    --------
    >>> config = RegistrationConfig(method="vgicp", voxel_resolution=0.5)
    >>> result = align_point_clouds(source_pts, target_pts, config)

    >>> # With custom multi-resolution schedule
    >>> config = RegistrationConfig(
    ...     method="gicp",
    ...     resolution_stages=[4.0, 2.0, 1.0, 0.5],
    ...     max_correspondence_distance=2.0,
    ... )
    """

    # === Core Registration Parameters ===
    method: str = "vgicp"
    max_correspondence_distance: float = 1.0
    max_iterations: int = 50
    num_threads: int = field(default_factory=lambda: min(os.cpu_count() or 4, 8))

    # === Multi-Resolution Parameters ===
    voxel_resolution: float = 0.5
    multi_resolution: bool = True
    resolution_stages: Optional[List[float]] = None

    # === Point Loading (orchestration layer) ===
    point_filter: Union[str, List[int]] = "ground"
    max_points: Optional[int] = None

    # === Spatial Constraints (orchestration layer) ===
    crop_to_overlap: bool = True
    crop_buffer: float = 10.0
    crop_box_size: Optional[Tuple[float, float]] = None
    crop_box_buffer: float = 0.0

    # === Quality Gate ===
    auto_revert: bool = True

    # === Output Control ===
    apply_transform: bool = True
    output_path: Optional[str] = None
    overwrite: bool = True
    verbose: bool = True

    def __post_init__(self):
        """Validate and normalize configuration."""
        # normalize method
        self.method = self.method.lower()
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                f"Valid options: {', '.join(sorted(VALID_METHODS))}"
            )

        # normalize point_filter
        if isinstance(self.point_filter, str):
            self.point_filter = self.point_filter.lower()
            if self.point_filter not in ("ground", "all"):
                raise ValueError(
                    f"Invalid point_filter string: '{self.point_filter}'. "
                    "Use 'ground', 'all', or a list of classification codes."
                )

        # validate voxel_resolution
        if self.voxel_resolution <= 0:
            raise ValueError(f"voxel_resolution must be positive, got {self.voxel_resolution}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'method': self.method,
            'max_correspondence_distance': self.max_correspondence_distance,
            'max_iterations': self.max_iterations,
            'num_threads': self.num_threads,
            'voxel_resolution': self.voxel_resolution,
            'multi_resolution': self.multi_resolution,
            'resolution_stages': self.resolution_stages,
            'point_filter': self.point_filter,
            'max_points': self.max_points,
            'crop_to_overlap': self.crop_to_overlap,
            'crop_buffer': self.crop_buffer,
            'crop_box_size': self.crop_box_size,
            'crop_box_buffer': self.crop_box_buffer,
            'auto_revert': self.auto_revert,
            'verbose': self.verbose,
        }


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------

class RegistrationResult:
    """Container for point cloud registration results.

    Attributes
    ----------
    transformation : np.ndarray
        4x4 homogeneous transformation matrix (T_target_source: transforms
        source points into target frame).
    rmse : float
        Post-alignment Root Mean Square Error of inlier distances.
    pre_rmse : float
        Pre-alignment RMSE (for quality comparison).
    fitness : float
        Inlier ratio (fraction of source points with correspondences within
        max_correspondence_distance), range 0-1.
    num_inliers : int
        Number of inlier correspondences.
    converged : bool
        Whether the registration converged.
    iterations : int
        Total iterations across all resolution stages.
    method_used : str
        Registration method that was used.
    centroid : np.ndarray, optional
        Centroid used for centering (needed for applying transform to files).
    translation : np.ndarray
        Translation component (3,), computed from transformation.
    rotation_angle_deg : float
        Rotation angle in degrees, computed from transformation.
    source_path, target_path, output_path : str, optional
        File paths for context.
    reverted : bool
        True if the transform was reverted to identity due to quality gate.
    """

    def __init__(self):
        self.transformation: np.ndarray = np.eye(4)

        # quality metrics
        self.rmse: float = np.inf
        self.pre_rmse: float = np.inf
        self.fitness: float = 0.0
        self.num_inliers: int = 0
        self.num_correspondences: int = 0  # legacy alias

        # convergence info
        self.converged: bool = False
        self.iterations: int = 0
        self.method_used: str = ""
        self.reverted: bool = False

        # transformation components (computed on access)
        self._translation: Optional[np.ndarray] = None
        self._rotation_angle_deg: Optional[float] = None

        # context for proper transformation application
        self.centroid: Optional[np.ndarray] = None
        self.source_path: Optional[str] = None
        self.target_path: Optional[str] = None
        self.output_path: Optional[str] = None

        # additional metadata
        self.metadata: Dict[str, Any] = {}

    @property
    def translation(self) -> np.ndarray:
        """Translation component of the transformation."""
        if self._translation is None:
            self._translation = self.transformation[:3, 3].copy()
        return self._translation

    @property
    def rotation_angle_deg(self) -> float:
        """Rotation angle in degrees."""
        if self._rotation_angle_deg is None:
            R = self.transformation[:3, :3]
            trace = np.trace(R)
            cos_angle = np.clip((trace - 1) / 2, -1, 1)
            self._rotation_angle_deg = float(np.degrees(np.arccos(cos_angle)))
        return self._rotation_angle_deg

    def is_valid(self, config: RegistrationConfig) -> bool:
        """Check if result meets basic quality criteria."""
        return self.converged and self.fitness > 0.1 and self.rmse < np.inf

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'transformation': self.transformation.tolist(),
            'rmse': self.rmse,
            'pre_rmse': self.pre_rmse,
            'fitness': self.fitness,
            'num_inliers': self.num_inliers,
            'converged': self.converged,
            'iterations': self.iterations,
            'method_used': self.method_used,
            'reverted': self.reverted,
            'translation': self.translation.tolist(),
            'rotation_angle_deg': self.rotation_angle_deg,
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'source_path': self.source_path,
            'target_path': self.target_path,
            'output_path': self.output_path,
        }

    def __repr__(self) -> str:
        status = "REVERTED" if self.reverted else ("converged" if self.converged else "not converged")
        return (
            f"RegistrationResult(method={self.method_used}, "
            f"rmse={self.rmse:.4f}, pre_rmse={self.pre_rmse:.4f}, "
            f"fitness={self.fitness:.3f}, {status})"
        )


# ---------------------------------------------------------------------------
# Backward-compatibility shims
# ---------------------------------------------------------------------------

# Keep RegistrationMethod importable but mark as deprecated
class RegistrationMethod:
    """Deprecated: use string method names directly in RegistrationConfig."""
    ICP = "icp"
    PLANE_ICP = "plane_icp"
    GICP = "gicp"
    VGICP = "vgicp"

    def __init__(self, value: str = "vgicp"):
        self.value = value


class LandscapeAligner:
    """Deprecated: use align_point_clouds() function directly.

    This shim exists for backward compatibility with tests and notebooks
    that reference LandscapeAligner. It delegates to the standalone
    align_point_clouds() function.
    """

    def __init__(self, config: Optional[RegistrationConfig] = None):
        self.config = config or RegistrationConfig()

    def align(self, source, target, method=None, initial_transform=None,
              apply_transform=False, output_path=None) -> RegistrationResult:
        warnings.warn(
            "LandscapeAligner is deprecated. Use align_point_clouds() directly.",
            DeprecationWarning, stacklevel=2,
        )
        if method is not None:
            method_str = method.value if hasattr(method, 'value') else str(method)
            self.config.method = method_str.lower()

        # handle numpy arrays directly
        if isinstance(source, np.ndarray) and isinstance(target, np.ndarray):
            return align_point_clouds(source, target, self.config)

        # handle file paths / PointCloud objects
        src_path = source.filename if hasattr(source, 'filename') else str(source)
        tgt_path = target.filename if hasattr(target, 'filename') else str(target)

        src_pts = load_points_from_las(src_path, point_filter=self.config.point_filter)
        tgt_pts = load_points_from_las(tgt_path, point_filter=self.config.point_filter)

        result = align_point_clouds(src_pts, tgt_pts, self.config)

        if apply_transform and result.converged:
            if output_path is None:
                p = Path(src_path)
                output_path = str(p.with_name(p.stem + "_aligned" + p.suffix))
            save_transformed_las(src_path, output_path, result.transformation, result.centroid)
            result.output_path = str(output_path)

        return result


# ---------------------------------------------------------------------------
# Core Alignment Function
# ---------------------------------------------------------------------------

def _compute_rmse_fitness(
    source_points: np.ndarray,
    target_points: np.ndarray,
    transformation: np.ndarray,
    max_distance: float,
) -> Tuple[float, float, int]:
    """Compute RMSE and fitness by transforming source and querying target KD-tree.

    Parameters
    ----------
    source_points : np.ndarray
        Nx3 source points (centered).
    target_points : np.ndarray
        Nx3 target points (centered).
    transformation : np.ndarray
        4x4 transformation matrix.
    max_distance : float
        Maximum correspondence distance for inlier filtering.

    Returns
    -------
    rmse : float
    fitness : float (inlier ratio)
    num_inliers : int
    """
    source_h = np.hstack([source_points, np.ones((len(source_points), 1))])
    source_transformed = (source_h @ transformation.T)[:, :3]

    tree = cKDTree(target_points)
    distances, _ = tree.query(source_transformed)

    inlier_mask = distances < max_distance
    inlier_dists = distances[inlier_mask]

    if len(inlier_dists) == 0:
        return float('inf'), 0.0, 0

    rmse = float(np.sqrt(np.mean(inlier_dists ** 2)))
    fitness = len(inlier_dists) / len(source_points) if len(source_points) > 0 else 0.0
    return rmse, fitness, len(inlier_dists)


def align_point_clouds(
    source_points: np.ndarray,
    target_points: np.ndarray,
    config: Optional[RegistrationConfig] = None,
) -> RegistrationResult:
    """Align source point cloud to target using multi-resolution ICP registration.

    This is the single source of truth for all ICP-based registration. It takes
    Nx3 numpy arrays (which may or may not already be centered) and runs a
    multi-resolution coarse-to-fine alignment using small_gicp.

    All downsampling is handled internally by ``small_gicp.preprocess_points()``
    at each resolution stage. The caller should NOT pre-downsample the data.

    Parameters
    ----------
    source_points : np.ndarray
        Nx3 array of source (compare) point coordinates.
    target_points : np.ndarray
        Nx3 array of target (reference) point coordinates.
    config : RegistrationConfig, optional
        Registration configuration. Uses defaults if None.

    Returns
    -------
    RegistrationResult
        Contains transformation matrix, RMSE (pre and post), fitness, centroid,
        and convergence information. If auto_revert triggered, the transformation
        is identity and result.reverted is True.

    Notes
    -----
    The function centers both point clouds to a shared centroid before
    registration. The centroid is stored in the result so that the transform
    can be correctly applied to the original (uncentered) data via::

        save_transformed_las(source_file, output_file,
                             result.transformation, result.centroid)
    """
    require_small_gicp()
    import small_gicp as sgicp

    if config is None:
        config = RegistrationConfig()

    result = RegistrationResult()
    result.method_used = config.method

    verbose = config.verbose
    n_source = len(source_points)
    n_target = len(target_points)

    if n_source == 0 or n_target == 0:
        warnings.warn("Empty point cloud(s) passed to align_point_clouds.")
        return result

    # ── Step 1: Center both clouds ──────────────────────────────────────
    src_centroid = source_points.mean(axis=0)
    tgt_centroid = target_points.mean(axis=0)
    n_total = n_source + n_target
    centroid = (src_centroid * n_source + tgt_centroid * n_target) / n_total
    result.centroid = centroid

    src_pts = source_points - centroid
    tgt_pts = target_points - centroid

    if verbose:
        print(f"\nCentering point clouds (centroid: "
              f"[{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}])",
              file=sys.stderr)
        print(f"Source points: {n_source:,}", file=sys.stderr)
        print(f"Target points: {n_target:,}", file=sys.stderr)

    # ── Step 2: Build resolution schedule ───────────────────────────────
    base_res = config.voxel_resolution

    if not config.multi_resolution:
        resolutions = [base_res]
    elif config.resolution_stages is not None:
        resolutions = sorted(config.resolution_stages, reverse=True)
    else:
        # default: 3 stages at 4x, 2x, 1x the target resolution
        resolutions = [base_res * 4.0, base_res * 2.0, base_res]

    # deduplicate stages that are too close together (<10% finer)
    resolutions = [r for i, r in enumerate(resolutions)
                   if i == 0 or r < resolutions[i - 1] * 0.9]

    if verbose:
        if len(resolutions) > 1:
            sched = " -> ".join(f"{r:.2f}m" for r in resolutions)
            print(f"Coarse-to-fine schedule: {sched}", file=sys.stderr)
        else:
            print(f"Single-resolution alignment at {resolutions[0]:.2f}m", file=sys.stderr)

    # ── Step 3: Multi-resolution registration ───────────────────────────
    method_upper = config.method.upper()
    init_T = np.eye(4)
    total_iterations = 0

    for stage_idx, stage_res in enumerate(resolutions):
        is_final = (stage_idx == len(resolutions) - 1)
        stage_label = "Final" if is_final else f"Stage {stage_idx + 1}/{len(resolutions)}"

        if verbose:
            print(f"\n  [{stage_label}] Resolution {stage_res:.3f}m ...", file=sys.stderr)

        # preprocess: downsample + estimate normals/covariances
        # This is the ONLY downsampling in the entire pipeline.
        with ThreadPoolExecutor(max_workers=2) as executor:
            tgt_future = executor.submit(
                sgicp.preprocess_points, tgt_pts,
                downsampling_resolution=stage_res,
                num_threads=max(1, config.num_threads // 2),
            )
            src_future = executor.submit(
                sgicp.preprocess_points, src_pts,
                downsampling_resolution=stage_res,
                num_threads=max(1, config.num_threads // 2),
            )
            tgt_cloud, tgt_tree = tgt_future.result()
            src_cloud, src_tree = src_future.result()

        if verbose:
            print(f"    Points: {src_cloud.size():,} source, "
                  f"{tgt_cloud.size():,} target", file=sys.stderr)

        # scale correspondence distance and iterations for coarser stages
        if is_final:
            stage_max_dist = config.max_correspondence_distance
            stage_max_iters = config.max_iterations
        else:
            scale = stage_res / resolutions[-1]
            stage_max_dist = config.max_correspondence_distance * scale
            stage_max_iters = max(20, config.max_iterations // 2)

        # run registration
        align_kwargs = dict(
            init_T_target_source=init_T,
            max_correspondence_distance=stage_max_dist,
            max_iterations=stage_max_iters,
            num_threads=config.num_threads,
        )

        if method_upper == "VGICP":
            # VGICP uses GaussianVoxelMap for the target
            voxelmap = sgicp.GaussianVoxelMap(stage_res)
            voxelmap.insert(tgt_cloud)
            gicp_result = sgicp.align(voxelmap, src_cloud, **align_kwargs)
        elif method_upper == "ICP":
            align_kwargs["registration_type"] = "ICP"
            gicp_result = sgicp.align(tgt_cloud, src_cloud, tgt_tree, **align_kwargs)
        else:
            # GICP or PLANE_ICP
            align_kwargs["registration_type"] = method_upper
            gicp_result = sgicp.align(tgt_cloud, src_cloud, tgt_tree, **align_kwargs)

        init_T = gicp_result.T_target_source
        total_iterations += getattr(gicp_result, 'iterations', 0)

        if verbose:
            t = init_T[:3, 3]
            print(f"    Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m",
                  file=sys.stderr)

        # clean up stage clouds
        del tgt_cloud, tgt_tree, src_cloud, src_tree
        gc.collect()

    # ── Step 4: Compute pre/post RMSE on full-resolution points ─────────
    max_dist = config.max_correspondence_distance

    if verbose:
        print(f"\n  Computing alignment quality on full-resolution points ...",
              file=sys.stderr)

    pre_rmse, pre_fitness, pre_inliers = _compute_rmse_fitness(
        src_pts, tgt_pts, np.eye(4), max_dist,
    )
    post_rmse, post_fitness, post_inliers = _compute_rmse_fitness(
        src_pts, tgt_pts, init_T, max_dist,
    )

    result.pre_rmse = pre_rmse
    result.rmse = post_rmse
    result.fitness = post_fitness
    result.num_inliers = post_inliers
    result.num_correspondences = post_inliers
    result.transformation = init_T
    result.converged = getattr(gicp_result, 'converged', True)
    result.iterations = total_iterations

    if verbose:
        print(f"  Pre-alignment:  RMSE={pre_rmse:.4f} m, fitness={pre_fitness:.4f}",
              file=sys.stderr)
        print(f"  Post-alignment: RMSE={post_rmse:.4f} m, fitness={post_fitness:.4f}",
              file=sys.stderr)

    # ── Step 5: Quality gate — auto-revert if alignment worsened ────────
    if config.auto_revert and post_rmse >= pre_rmse and pre_fitness > 0.3:
        logger.warning(
            f"Alignment INCREASED RMSE ({pre_rmse:.4f} -> {post_rmse:.4f} m). "
            "Reverting to identity transform."
        )
        if verbose:
            print(f"  WARNING: Alignment worsened RMSE. Reverting to identity.",
                  file=sys.stderr)
        result.transformation = np.eye(4)
        result.rmse = pre_rmse
        result.fitness = pre_fitness
        result.num_inliers = pre_inliers
        result.num_correspondences = pre_inliers
        result.reverted = True
        # reset cached properties
        result._translation = None
        result._rotation_angle_deg = None

    if verbose:
        print(f"\nAlignment Results:", file=sys.stderr)
        print(f"  Converged: {result.converged}", file=sys.stderr)
        print(f"  Fitness (inlier ratio): {result.fitness:.4f}", file=sys.stderr)
        print(f"  RMSE: {result.rmse:.4f} m (pre: {result.pre_rmse:.4f} m)",
              file=sys.stderr)
        print(f"  Inlier correspondences: {result.num_inliers:,}", file=sys.stderr)
        print(f"  Translation: [{result.translation[0]:.4f}, "
              f"{result.translation[1]:.4f}, {result.translation[2]:.4f}] m",
              file=sys.stderr)
        print(f"  Rotation: {result.rotation_angle_deg:.4f} deg", file=sys.stderr)
        if result.reverted:
            print(f"  *** Transform REVERTED to identity (quality gate) ***",
                  file=sys.stderr)

    # clean up
    del src_pts, tgt_pts
    gc.collect()

    return result
