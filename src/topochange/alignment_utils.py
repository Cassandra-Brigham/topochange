"""
Shared utilities for point cloud alignment operations.

This module consolidates common functionality from pointcloudpair.py and alignment.py
to eliminate code duplication and provide a single source of truth for:
- PDAL pipeline execution
- Point cloud loading with filtering/downsampling
- Transformation application to LAS files
- Alignment quality metrics computation
- Point cloud preprocessing utilities

References
----------
- Besl, P.J. and McKay, N.D. (1992). A Method for Registration of 3-D Shapes.
  IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Koide, K. et al. (2021). Voxelized GICP for Fast and Accurate 3D Point Cloud Registration.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# PDAL Pipeline Utilities
# ============================================================================

def run_pdal_pipeline(
    steps: List[Dict[str, Any]],
    arrays: Optional[np.ndarray] = None,
    need_arrays: bool = True,
    streaming: bool = False,
    chunk_size: int = 1_000_000,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Execute a PDAL pipeline with unified error handling.

    This function consolidates PDAL pipeline execution from multiple modules
    into a single, well-tested implementation.

    Parameters
    ----------
    steps : list of dict
        List of PDAL pipeline stages (readers, filters, writers)
    arrays : np.ndarray, optional
        Input array for filters.python or similar stages
    need_arrays : bool, default True
        Whether to return point arrays from the pipeline
    streaming : bool, default False
        Use streaming execution for memory efficiency (useful for large files)
    chunk_size : int, default 1_000_000
        Chunk size for streaming execution

    Returns
    -------
    tuple of (np.ndarray or None, dict)
        Point array (if need_arrays=True and not streaming) and pipeline metadata

    Raises
    ------
    RuntimeError
        If pipeline execution fails

    Examples
    --------
    >>> steps = [
    ...     {"type": "readers.las", "filename": "input.laz"},
    ...     {"type": "filters.range", "limits": "Classification[2:2]"},
    ...     {"type": "writers.las", "filename": "output.laz"}
    ... ]
    >>> arr, meta = run_pdal_pipeline(steps, need_arrays=False)
    """
    import pdal

    pipeline_spec = {"pipeline": steps}

    if arrays is not None:
        pipe = pdal.Pipeline(json.dumps(pipeline_spec), arrays=[arrays])
    else:
        pipe = pdal.Pipeline(json.dumps(pipeline_spec))

    try:
        if streaming:
            count = pipe.execute_streaming(chunk_size=chunk_size)
            logger.debug(f"Streaming pipeline executed: {count} points processed")
        else:
            count = pipe.execute()
            logger.debug(f"Pipeline executed: {count} points")
    except Exception as e:
        logger.error(f"PDAL pipeline failed: {e}")
        raise RuntimeError(f"PDAL pipeline execution failed: {e}") from e

    metadata = pipe.metadata.get("metadata", {})

    if need_arrays and not streaming:
        if pipe.arrays and len(pipe.arrays) > 0:
            return pipe.arrays[0], metadata
        return None, metadata

    return None, metadata


def load_points_from_las(
    filename: Union[str, Path],
    max_points: Optional[int] = None,
    voxel_size: Optional[float] = None,
    point_filter: Optional[Union[str, List[int]]] = None,
    crop_bounds: Optional[Tuple[float, float, float, float]] = None,
    streaming: bool = False,
    return_colors: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Load points from LAS/LAZ file with optional filtering and downsampling.

    This function provides a unified interface for loading point clouds with
    various preprocessing options applied at load time for memory efficiency.

    Parameters
    ----------
    filename : str or Path
        Path to LAS/LAZ file
    max_points : int, optional
        Maximum number of points to load (applied after other filters)
    voxel_size : float, optional
        Voxel size for downsampling during load. If None, no downsampling.
    point_filter : str or list of int, optional
        Classification filter:
        - "ground" or "Ground": Only ground points (classification 2)
        - "all" or None: All points (no filtering)
        - List of ints: Points with these classification codes
    crop_bounds : tuple of (minx, miny, maxx, maxy), optional
        Spatial bounding box for cropping
    streaming : bool, default False
        Use streaming execution (returns None, writes to temp file first)
    return_colors : bool, default False
        Also return RGB colors if available

    Returns
    -------
    np.ndarray or tuple
        Nx3 array of XYZ coordinates, or (XYZ, RGB) if return_colors=True

    Raises
    ------
    RuntimeError
        If no points are loaded from the file
    FileNotFoundError
        If the input file does not exist

    Examples
    --------
    >>> # Load ground points with 0.5m voxel downsampling
    >>> points = load_points_from_las(
    ...     "terrain.laz",
    ...     point_filter="ground",
    ...     voxel_size=0.5,
    ...     max_points=1_000_000
    ... )
    >>> points.shape
    (1000000, 3)

    >>> # Load points within a bounding box
    >>> points = load_points_from_las(
    ...     "terrain.laz",
    ...     crop_bounds=(500000, 4000000, 501000, 4001000)
    ... )
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"Point cloud file not found: {filename}")

    steps = [{"type": "readers.las", "filename": str(filename)}]

    # Classification filter
    if point_filter and point_filter not in ("all", "All", None):
        if point_filter in ("ground", "Ground"):
            steps.append({"type": "filters.range", "limits": "Classification[2:2]"})
        elif isinstance(point_filter, (list, tuple)):
            # Support multiple classification codes
            ranges = ",".join(f"Classification[{c}:{c}]" for c in point_filter)
            steps.append({"type": "filters.range", "limits": ranges})
        elif isinstance(point_filter, int):
            steps.append({"type": "filters.range", "limits": f"Classification[{point_filter}:{point_filter}]"})
        else:
            logger.warning(f"Unknown point_filter value: {point_filter}, using all points")

    # Spatial crop
    if crop_bounds is not None:
        minx, miny, maxx, maxy = crop_bounds
        steps.append({
            "type": "filters.crop",
            "bounds": f"([{minx},{maxx}],[{miny},{maxy}])"
        })

    # Voxel downsampling
    if voxel_size is not None and voxel_size > 0:
        steps.append({
            "type": "filters.voxeldownsize",
            "cell": float(voxel_size),
            "mode": "center"
        })

    # Point limit (applied last)
    if max_points is not None and max_points > 0:
        steps.append({"type": "filters.head", "count": int(max_points)})

    # Execute pipeline
    arr, metadata = run_pdal_pipeline(steps, need_arrays=True, streaming=False)

    if arr is None or len(arr) == 0:
        raise RuntimeError(f"No points loaded from {filename}")

    # Extract XYZ
    xyz = np.column_stack((arr['X'], arr['Y'], arr['Z']))

    if return_colors:
        # Try to extract RGB colors
        try:
            if 'Red' in arr.dtype.names and 'Green' in arr.dtype.names and 'Blue' in arr.dtype.names:
                rgb = np.column_stack((arr['Red'], arr['Green'], arr['Blue']))
                # Normalize to 0-1 if values are 16-bit
                if rgb.max() > 255:
                    rgb = rgb / 65535.0
                elif rgb.max() > 1:
                    rgb = rgb / 255.0
                return xyz, rgb
        except Exception:
            pass
        return xyz, None

    return xyz


def save_transformed_las(
    source_filename: Union[str, Path],
    output_filename: Union[str, Path],
    transformation_matrix: np.ndarray,
    centroid: Optional[np.ndarray] = None,
    output_crs: Optional[str] = None,
    streaming: bool = True,
    chunk_size: int = 1_000_000,
) -> str:
    """
    Apply 4x4 transformation matrix to LAS file and save result.

    This function handles the complexity of applying transformations that were
    computed in centered coordinates (common for ICP registration) back to
    the original coordinate system.

    Parameters
    ----------
    source_filename : str or Path
        Input LAS/LAZ file
    output_filename : str or Path
        Output LAS/LAZ file
    transformation_matrix : np.ndarray
        4x4 homogeneous transformation matrix
    centroid : np.ndarray, optional
        If provided, the transformation is applied in centered coordinates:
        1. Subtract centroid from points
        2. Apply transformation
        3. Add centroid back to points
        This is necessary when the transformation was computed on centered data.
    output_crs : str, optional
        CRS for output file (WKT or PROJ string). If None, preserves source CRS.
    streaming : bool, default True
        Use streaming execution for memory efficiency
    chunk_size : int, default 1_000_000
        Chunk size for streaming execution

    Returns
    -------
    str
        Path to output file

    Examples
    --------
    >>> # Apply ICP result computed on centered data
    >>> T = np.array([
    ...     [1, 0, 0, 0.5],
    ...     [0, 1, 0, 0.3],
    ...     [0, 0, 1, 15.0],
    ...     [0, 0, 0, 1]
    ... ])
    >>> centroid = np.array([500000, 4000000, 1000])
    >>> save_transformed_las("source.laz", "aligned.laz", T, centroid=centroid)
    """
    source_filename = Path(source_filename)
    output_filename = Path(output_filename)

    # Ensure output directory exists
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    # Build the full transformation matrix
    if centroid is not None:
        # Create centering matrices
        T_center = np.eye(4)
        T_center[:3, 3] = -centroid

        T_uncenter = np.eye(4)
        T_uncenter[:3, 3] = centroid

        # Combined transformation: uncenter @ transform @ center
        # This applies: (1) subtract centroid, (2) apply T, (3) add centroid
        full_matrix = T_uncenter @ transformation_matrix @ T_center
    else:
        full_matrix = transformation_matrix

    # Format matrix for PDAL (row-major, space-separated)
    matrix_str = " ".join(f"{v:.12g}" for v in full_matrix.flatten())

    # Build pipeline
    steps = [
        {"type": "readers.las", "filename": str(source_filename)},
        {"type": "filters.transformation", "matrix": matrix_str},
    ]

    writer = {"type": "writers.las", "filename": str(output_filename)}
    if output_crs:
        writer["a_srs"] = output_crs
    steps.append(writer)

    # Execute
    run_pdal_pipeline(steps, need_arrays=False, streaming=streaming, chunk_size=chunk_size)

    logger.info(f"Saved transformed point cloud to {output_filename}")
    return str(output_filename)


# ============================================================================
# Point Cloud Preprocessing
# ============================================================================

class PointCloudPreprocessor:
    """
    Unified point cloud preprocessing utilities.

    This class provides static methods for common preprocessing operations
    that can be applied to point cloud files before alignment.

    All methods operate on files (not arrays) for memory efficiency with
    large point clouds, using PDAL pipelines under the hood.

    Examples
    --------
    >>> # Chain preprocessing operations
    >>> filtered = PointCloudPreprocessor.filter_by_classification(
    ...     "input.laz", "ground", "ground_only.laz"
    ... )
    >>> cleaned = PointCloudPreprocessor.filter_outliers(
    ...     filtered, k_neighbors=20, std_multiplier=2.0
    ... )
    >>> downsampled = PointCloudPreprocessor.downsample_voxel(
    ...     cleaned, voxel_size=0.5
    ... )
    """

    @staticmethod
    def filter_by_classification(
        filename: Union[str, Path],
        classifications: Union[str, List[int], int],
        output_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> str:
        """
        Filter points by classification code(s).

        Parameters
        ----------
        filename : str or Path
            Input LAS/LAZ file
        classifications : str, int, or list of int
            Classification filter:
            - "ground": Ground points (class 2)
            - "all" or None: No filtering (returns input path)
            - int: Single classification code
            - list of int: Multiple classification codes
        output_path : str, optional
            Output file path. If None, auto-generated.
        overwrite : bool, default True
            Overwrite existing output file

        Returns
        -------
        str
            Path to filtered file (may be same as input if no filtering needed)
        """
        filename = Path(filename)

        # No filtering needed
        if classifications in ("all", "All", None):
            return str(filename)

        # Determine filter limits
        if classifications in ("ground", "Ground"):
            limits = "Classification[2:2]"
        elif isinstance(classifications, int):
            limits = f"Classification[{classifications}:{classifications}]"
        elif isinstance(classifications, (list, tuple)):
            ranges = ",".join(f"Classification[{c}:{c}]" for c in classifications)
            limits = ranges
        else:
            raise ValueError(f"Invalid classification filter: {classifications}")

        # Determine output path
        if output_path is None:
            output_path = str(filename.with_name(filename.stem + "_filtered" + filename.suffix))

        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            logger.info(f"Using existing filtered file: {output_path}")
            return str(output_path)

        # Execute pipeline
        steps = [
            {"type": "readers.las", "filename": str(filename)},
            {"type": "filters.range", "limits": limits},
            {"type": "writers.las", "filename": str(output_path)}
        ]
        run_pdal_pipeline(steps, need_arrays=False, streaming=True)

        logger.info(f"Filtered point cloud saved to {output_path}")
        return str(output_path)

    @staticmethod
    def downsample_voxel(
        filename: Union[str, Path],
        voxel_size: float,
        output_path: Optional[str] = None,
        mode: str = "center",
        overwrite: bool = True,
    ) -> str:
        """
        Downsample point cloud using voxel grid.

        Parameters
        ----------
        filename : str or Path
            Input LAS/LAZ file
        voxel_size : float
            Size of voxel cells in coordinate units (usually meters)
        output_path : str, optional
            Output file path. If None, auto-generated.
        mode : str, default "center"
            Point selection mode: "center", "first", or "last"
        overwrite : bool, default True
            Overwrite existing output file

        Returns
        -------
        str
            Path to downsampled file
        """
        filename = Path(filename)

        if output_path is None:
            output_path = str(filename.with_name(
                filename.stem + f"_voxel{voxel_size:.2f}" + filename.suffix
            ))

        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            logger.info(f"Using existing downsampled file: {output_path}")
            return str(output_path)

        steps = [
            {"type": "readers.las", "filename": str(filename)},
            {"type": "filters.voxeldownsize", "cell": float(voxel_size), "mode": mode},
            {"type": "writers.las", "filename": str(output_path)}
        ]
        run_pdal_pipeline(steps, need_arrays=False, streaming=True)

        logger.info(f"Downsampled point cloud saved to {output_path}")
        return str(output_path)

    @staticmethod
    def filter_outliers(
        filename: Union[str, Path],
        k_neighbors: int = 20,
        std_multiplier: float = 2.0,
        output_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> str:
        """
        Remove statistical outliers from point cloud.

        Uses the statistical outlier removal algorithm which identifies points
        whose average distance to their k nearest neighbors is above a threshold.

        Parameters
        ----------
        filename : str or Path
            Input LAS/LAZ file
        k_neighbors : int, default 20
            Number of neighbors for distance calculation
        std_multiplier : float, default 2.0
            Standard deviation multiplier for outlier threshold
        output_path : str, optional
            Output file path. If None, auto-generated.
        overwrite : bool, default True
            Overwrite existing output file

        Returns
        -------
        str
            Path to cleaned file
        """
        filename = Path(filename)

        if output_path is None:
            output_path = str(filename.with_name(filename.stem + "_cleaned" + filename.suffix))

        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            logger.info(f"Using existing cleaned file: {output_path}")
            return str(output_path)

        steps = [
            {"type": "readers.las", "filename": str(filename)},
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": int(k_neighbors),
                "multiplier": float(std_multiplier)
            },
            {"type": "writers.las", "filename": str(output_path)}
        ]
        run_pdal_pipeline(steps, need_arrays=False, streaming=True)

        logger.info(f"Cleaned point cloud saved to {output_path}")
        return str(output_path)

    @staticmethod
    def apply_ground_filter(
        filename: Union[str, Path],
        output_path: Optional[str] = None,
        cell_size: float = 1.0,
        slope: float = 0.15,
        initial_distance: float = 0.15,
        max_distance: float = 2.5,
        max_window_size: float = 18.0,
        overwrite: bool = True,
    ) -> str:
        """
        Apply SMRF ground classification filter.

        Simple Morphological Filter (SMRF) for identifying ground points
        in point clouds with vegetation or buildings.

        Parameters
        ----------
        filename : str or Path
            Input LAS/LAZ file
        output_path : str, optional
            Output file path. If None, auto-generated.
        cell_size : float, default 1.0
            Cell size for the morphological filter
        slope : float, default 0.15
            Slope threshold for ground classification
        initial_distance : float, default 0.15
            Initial distance threshold
        max_distance : float, default 2.5
            Maximum distance threshold
        max_window_size : float, default 18.0
            Maximum window size for morphological operations
        overwrite : bool, default True
            Overwrite existing output file

        Returns
        -------
        str
            Path to file with updated ground classification
        """
        filename = Path(filename)

        if output_path is None:
            output_path = str(filename.with_name(filename.stem + "_smrf" + filename.suffix))

        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            logger.info(f"Using existing SMRF-filtered file: {output_path}")
            return str(output_path)

        steps = [
            {"type": "readers.las", "filename": str(filename)},
            {
                "type": "filters.smrf",
                "cell": float(cell_size),
                "slope": float(slope),
                "initial_distance": float(initial_distance),
                "max_distance": float(max_distance),
                "max_window_size": float(max_window_size),
            },
            {"type": "writers.las", "filename": str(output_path)}
        ]
        run_pdal_pipeline(steps, need_arrays=False, streaming=True)

        logger.info(f"SMRF-filtered point cloud saved to {output_path}")
        return str(output_path)

    @staticmethod
    def crop_to_bounds(
        filename: Union[str, Path],
        bounds: Tuple[float, float, float, float],
        output_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> str:
        """
        Crop point cloud to bounding box.

        Parameters
        ----------
        filename : str or Path
            Input LAS/LAZ file
        bounds : tuple of (minx, miny, maxx, maxy)
            Bounding box for cropping
        output_path : str, optional
            Output file path. If None, auto-generated.
        overwrite : bool, default True
            Overwrite existing output file

        Returns
        -------
        str
            Path to cropped file
        """
        filename = Path(filename)
        minx, miny, maxx, maxy = bounds

        if output_path is None:
            output_path = str(filename.with_name(filename.stem + "_cropped" + filename.suffix))

        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            logger.info(f"Using existing cropped file: {output_path}")
            return str(output_path)

        steps = [
            {"type": "readers.las", "filename": str(filename)},
            {"type": "filters.crop", "bounds": f"([{minx},{maxx}],[{miny},{maxy}])"},
            {"type": "writers.las", "filename": str(output_path)}
        ]
        run_pdal_pipeline(steps, need_arrays=False, streaming=True)

        logger.info(f"Cropped point cloud saved to {output_path}")
        return str(output_path)

    @staticmethod
    def estimate_optimal_voxel_size(
        filename: Union[str, Path],
        target_points: int = 500_000,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> float:
        """
        Estimate optimal voxel size based on point density.

        Calculates the voxel size needed to reduce a point cloud to approximately
        the target number of points, assuming uniform distribution.

        Parameters
        ----------
        filename : str or Path
            Input LAS/LAZ file
        target_points : int, default 500_000
            Target number of points after downsampling
        bounds : tuple of (minx, miny, maxx, maxy), optional
            Custom bounds for area calculation. If None, uses file header.

        Returns
        -------
        float
            Recommended voxel size in coordinate units (usually meters)

        Notes
        -----
        The estimation assumes a roughly 2D distribution (typical for terrain data).
        For truly 3D data, the actual reduction may differ.
        """
        try:
            import laspy
            with laspy.open(str(filename)) as f:
                point_count = f.header.point_count
                if bounds is None:
                    mins = f.header.mins
                    maxs = f.header.maxs
                    bounds = (mins[0], mins[1], maxs[0], maxs[1])
        except Exception:
            # Fallback: use PDAL to get info
            steps = [{"type": "readers.las", "filename": str(filename)}]
            _, metadata = run_pdal_pipeline(steps, need_arrays=False)
            reader_meta = metadata.get("readers.las", {})
            point_count = reader_meta.get("count", 1_000_000)
            if bounds is None:
                bounds = (
                    reader_meta.get("minx", 0),
                    reader_meta.get("miny", 0),
                    reader_meta.get("maxx", 1000),
                    reader_meta.get("maxy", 1000),
                )

        # Calculate area
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area = max(width * height, 1.0)  # Avoid division by zero

        if point_count <= target_points:
            return 0.1  # Minimal downsampling

        # Target density (points per square unit)
        target_density = target_points / area

        # Voxel size ≈ sqrt(1 / target_density) for 2D distribution
        voxel_size = np.sqrt(1.0 / target_density)

        # Clamp to reasonable range
        voxel_size = max(0.1, min(voxel_size, 10.0))

        logger.debug(
            f"Estimated voxel size: {voxel_size:.2f}m "
            f"(from {point_count:,} points over {area:,.0f} m² to target {target_points:,})"
        )

        return voxel_size

    @classmethod
    def run_chain(
        cls,
        filename: Union[str, Path],
        output_path: Union[str, Path],
        *,
        classifications: Optional[Union[str, List[int], int]] = None,
        crop_bounds: Optional[Tuple[float, float, float, float]] = None,
        smrf: bool = False,
        smrf_params: Optional[Dict] = None,
        outlier_k: Optional[int] = None,
        outlier_std: float = 2.0,
        voxel_size: Optional[float] = None,
        voxel_mode: str = "center",
        max_points: Optional[int] = None,
        streaming: bool = True,
        chunk_size: int = 1_000_000,
        overwrite: bool = True,
    ) -> str:
        """
        Apply multiple preprocessing steps in a single PDAL pipeline pass.

        This avoids the intermediate file writes that occur when chaining
        individual methods (filter_by_classification → filter_outliers →
        downsample_voxel). All requested filters are composed into one
        read → filter → … → filter → write pipeline executed in streaming
        mode.

        Parameters
        ----------
        filename : str or Path
            Input LAS/LAZ file.
        output_path : str or Path
            Output LAS/LAZ file.
        classifications : str, int, or list of int, optional
            Classification filter (same semantics as filter_by_classification).
            None or "all" means no classification filtering.
        crop_bounds : tuple of (minx, miny, maxx, maxy), optional
            Spatial bounding box for cropping.
        smrf : bool, default False
            Apply SMRF ground classification filter.
        smrf_params : dict, optional
            Override default SMRF parameters (cell, slope, initial_distance,
            max_distance, max_window_size).
        outlier_k : int, optional
            If provided, apply statistical outlier removal with this many
            neighbors.
        outlier_std : float, default 2.0
            Standard deviation multiplier for outlier threshold.
        voxel_size : float, optional
            Voxel size for downsampling. None means no downsampling.
        voxel_mode : str, default "center"
            Point selection mode for voxel filter: "center", "first", "last".
        max_points : int, optional
            Hard cap on output points (applied last).
        streaming : bool, default True
            Use streaming execution for memory efficiency.
        chunk_size : int, default 1_000_000
            Chunk size for streaming execution.
        overwrite : bool, default True
            Overwrite existing output file.

        Returns
        -------
        str
            Path to the output file.
        """
        filename = Path(filename)
        output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            logger.info(f"Using existing preprocessed file: {output_path}")
            return str(output_path)

        steps: List[Dict[str, Any]] = [
            {"type": "readers.las", "filename": str(filename)}
        ]

        # 1. Classification filter (early = fewer points for subsequent stages)
        if classifications and classifications not in ("all", "All"):
            if classifications in ("ground", "Ground"):
                limits = "Classification[2:2]"
            elif isinstance(classifications, int):
                limits = f"Classification[{classifications}:{classifications}]"
            elif isinstance(classifications, (list, tuple)):
                limits = ",".join(f"Classification[{c}:{c}]" for c in classifications)
            else:
                raise ValueError(f"Invalid classification filter: {classifications}")
            steps.append({"type": "filters.range", "limits": limits})

        # 2. Spatial crop
        if crop_bounds is not None:
            minx, miny, maxx, maxy = crop_bounds
            steps.append({
                "type": "filters.crop",
                "bounds": f"([{minx},{maxx}],[{miny},{maxy}])",
            })

        # 3. SMRF ground classification
        if smrf:
            params = {
                "type": "filters.smrf",
                "cell": 1.0,
                "slope": 0.15,
                "initial_distance": 0.15,
                "max_distance": 2.5,
                "max_window_size": 18.0,
            }
            if smrf_params:
                params.update(smrf_params)
            steps.append(params)

        # 4. Statistical outlier removal
        if outlier_k is not None:
            steps.append({
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": int(outlier_k),
                "multiplier": float(outlier_std),
            })

        # 5. Voxel downsampling
        if voxel_size is not None and voxel_size > 0:
            steps.append({
                "type": "filters.voxeldownsize",
                "cell": float(voxel_size),
                "mode": voxel_mode,
            })

        # 6. Point limit (last)
        if max_points is not None and max_points > 0:
            steps.append({"type": "filters.head", "count": int(max_points)})

        # Writer
        steps.append({"type": "writers.las", "filename": str(output_path)})

        run_pdal_pipeline(
            steps, need_arrays=False, streaming=streaming, chunk_size=chunk_size
        )

        logger.info(f"Preprocessed point cloud saved to {output_path}")
        return str(output_path)


# ============================================================================
# Alignment Quality Metrics
# ============================================================================

@dataclass
class AlignmentQualityMetrics:
    """
    Container for alignment quality assessment metrics.

    Attributes
    ----------
    rmse : float
        Root Mean Square Error of correspondence distances
    mae : float
        Mean Absolute Error of correspondence distances
    median : float
        Median correspondence distance
    nmad : float
        Normalized Median Absolute Deviation (robust std estimate)
    inlier_ratio : float
        Fraction of points with correspondence distance <= max_distance
    inlier_count : int
        Number of inlier correspondences
    total_points : int
        Total number of source points evaluated
    percentiles : dict
        Distance percentiles (50th, 90th, 95th, 99th)
    max_distance_used : float
        Maximum correspondence distance used for inlier determination
    """
    rmse: float
    mae: float
    median: float
    nmad: float
    inlier_ratio: float
    inlier_count: int
    total_points: int
    percentiles: Dict[str, float]
    max_distance_used: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'median': self.median,
            'nmad': self.nmad,
            'inlier_ratio': self.inlier_ratio,
            'inlier_count': self.inlier_count,
            'total_points': self.total_points,
            'percentiles': self.percentiles,
            'max_distance_used': self.max_distance_used,
        }

    def __repr__(self) -> str:
        return (
            f"AlignmentQualityMetrics(rmse={self.rmse:.4f}, "
            f"inlier_ratio={self.inlier_ratio:.2%}, "
            f"nmad={self.nmad:.4f})"
        )


def compute_alignment_quality(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_distance: float = 1.0,
    sample_size: Optional[int] = None,
) -> AlignmentQualityMetrics:
    """
    Compute detailed alignment quality metrics using KD-tree nearest neighbor search.

    This function assesses how well the source point cloud aligns to the target
    by computing various distance statistics between corresponding points.

    Parameters
    ----------
    source_points : np.ndarray
        Nx3 array of aligned source points (after applying transformation)
    target_points : np.ndarray
        Mx3 array of target points (reference)
    max_distance : float, default 1.0
        Maximum correspondence distance for inlier calculation.
        Points with nearest neighbor farther than this are outliers.
    sample_size : int, optional
        If specified, randomly sample this many source points for efficiency.
        Useful for very large point clouds.

    Returns
    -------
    AlignmentQualityMetrics
        Quality metrics including RMSE, MAE, percentiles, inlier ratio

    Notes
    -----
    The metrics are computed as follows:
    - RMSE: sqrt(mean(d²)) for inlier distances d
    - MAE: mean(|d|) for inlier distances
    - NMAD: 1.4826 * median(|d - median(d)|), a robust std estimate
    - Inlier ratio: fraction of source points with d <= max_distance

    Examples
    --------
    >>> source = np.random.rand(10000, 3) * 100
    >>> target = source + np.random.randn(10000, 3) * 0.1  # Small perturbation
    >>> metrics = compute_alignment_quality(source, target, max_distance=0.5)
    >>> print(f"RMSE: {metrics.rmse:.4f}, Inliers: {metrics.inlier_ratio:.1%}")
    """
    from scipy.spatial import cKDTree

    # Optional sampling for large clouds
    if sample_size is not None and len(source_points) > sample_size:
        indices = np.random.choice(len(source_points), sample_size, replace=False)
        source_sample = source_points[indices]
    else:
        source_sample = source_points

    # Build KD-tree on target (this is O(M log M))
    tree = cKDTree(target_points)

    # Query nearest neighbors for all source points (this is O(N log M))
    distances, _ = tree.query(source_sample, k=1)

    # Compute inlier mask
    inlier_mask = distances <= max_distance
    inlier_distances = distances[inlier_mask]

    # Handle edge case of no inliers
    if len(inlier_distances) == 0:
        return AlignmentQualityMetrics(
            rmse=np.inf,
            mae=np.inf,
            median=np.inf,
            nmad=np.inf,
            inlier_ratio=0.0,
            inlier_count=0,
            total_points=len(source_sample),
            percentiles={'50': np.inf, '90': np.inf, '95': np.inf, '99': np.inf},
            max_distance_used=max_distance,
        )

    # Compute metrics
    rmse = np.sqrt(np.mean(inlier_distances ** 2))
    mae = np.mean(inlier_distances)
    median = np.median(inlier_distances)
    nmad = 1.4826 * np.median(np.abs(inlier_distances - median))

    return AlignmentQualityMetrics(
        rmse=rmse,
        mae=mae,
        median=median,
        nmad=nmad,
        inlier_ratio=len(inlier_distances) / len(source_sample),
        inlier_count=len(inlier_distances),
        total_points=len(source_sample),
        percentiles={
            '50': np.percentile(inlier_distances, 50),
            '90': np.percentile(inlier_distances, 90),
            '95': np.percentile(inlier_distances, 95),
            '99': np.percentile(inlier_distances, 99),
        },
        max_distance_used=max_distance,
    )


# ============================================================================
# Dependency Checks
# ============================================================================

def require_small_gicp():
    """
    Ensure small_gicp is available, raise informative error if not.

    Returns
    -------
    module
        The small_gicp module

    Raises
    ------
    ImportError
        If small_gicp is not installed
    """
    try:
        import small_gicp
        return small_gicp
    except ImportError:
        raise ImportError(
            "small_gicp is required for point cloud alignment. "
            "Install with: pip install small_gicp\n"
            "For GPU acceleration, see: https://github.com/koide3/small_gicp"
        )


def has_small_gicp() -> bool:
    """
    Check if small_gicp is available.

    Returns
    -------
    bool
        True if small_gicp is importable
    """
    try:
        import small_gicp
        return True
    except ImportError:
        return False


# ============================================================================
# Transformation Utilities
# ============================================================================

def decompose_transformation(T: np.ndarray) -> Dict[str, Any]:
    """
    Decompose a 4x4 transformation matrix into translation, rotation, and scale.

    Parameters
    ----------
    T : np.ndarray
        4x4 homogeneous transformation matrix

    Returns
    -------
    dict
        Dictionary with keys:
        - 'translation': np.ndarray of shape (3,)
        - 'rotation_matrix': np.ndarray of shape (3, 3)
        - 'rotation_angle_deg': float (angle in degrees)
        - 'rotation_axis': np.ndarray of shape (3,) (unit axis)
        - 'scale': float (uniform scale factor)
    """
    # Extract translation
    translation = T[:3, 3].copy()

    # Extract rotation matrix
    R = T[:3, :3].copy()

    # Compute scale (assuming uniform scaling)
    scale = np.cbrt(np.linalg.det(R))
    if scale != 0:
        R = R / scale

    # Compute rotation angle using trace
    # For a rotation matrix: trace(R) = 1 + 2*cos(theta)
    trace = np.trace(R)
    cos_angle = (trace - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    # Compute rotation axis (eigenvector with eigenvalue 1)
    if angle_rad < 1e-6:
        # Near-identity rotation
        axis = np.array([0, 0, 1])
    elif abs(angle_rad - np.pi) < 1e-6:
        # 180-degree rotation: find axis from R + I
        RpI = R + np.eye(3)
        axis = RpI[:, np.argmax(np.sum(RpI ** 2, axis=0))]
        axis = axis / np.linalg.norm(axis)
    else:
        # General case: axis from skew-symmetric part
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        axis = axis / (2 * np.sin(angle_rad))

    return {
        'translation': translation,
        'rotation_matrix': R,
        'rotation_angle_deg': angle_deg,
        'rotation_axis': axis,
        'scale': scale,
    }


def compose_transformation(
    translation: Optional[np.ndarray] = None,
    rotation_matrix: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Compose a 4x4 transformation matrix from components.

    Parameters
    ----------
    translation : np.ndarray, optional
        Translation vector of shape (3,). Default is [0, 0, 0].
    rotation_matrix : np.ndarray, optional
        3x3 rotation matrix. Default is identity.
    scale : float, default 1.0
        Uniform scale factor.

    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)

    if rotation_matrix is not None:
        T[:3, :3] = rotation_matrix * scale
    elif scale != 1.0:
        T[:3, :3] *= scale

    if translation is not None:
        T[:3, 3] = translation

    return T
