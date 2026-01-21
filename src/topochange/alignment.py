"""
Automatic Point Cloud Registration for Landscape Data
Integrates with PointCloud and PointCloudPair classes
Uses small_gicp for registration and PDAL for I/O and filtering
"""

import json
import numpy as np
try:
    import small_gicp
    _HAS_SMALL_GICP = True
except ImportError:
    small_gicp = None
    _HAS_SMALL_GICP = False
# Use pdal_wrapper for Colab compatibility (falls back to native pdal locally)
try:
    from .pdal_wrapper import pdal
except ImportError:
    import pdal
from typing import Optional, Dict, Tuple, List, Union, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path
from scipy.spatial import cKDTree
import logging
import tempfile
import os


def _require_small_gicp():
    """Raise ImportError if small_gicp is not available."""
    if not _HAS_SMALL_GICP:
        raise ImportError(
            "small_gicp is required for point cloud alignment. "
            "Install with: pip install topochange[alignment]"
        )

if TYPE_CHECKING:
    from .pointcloud import PointCloud

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_pdal_pipeline(steps, arrays=None, need_arrays=None):
    """
    Execute a PDAL pipeline from a list of step dictionaries.

    Args:
        steps: List of PDAL stage configurations
        arrays: Optional numpy arrays to pass to the pipeline
        need_arrays: If True, use execute() to get arrays back. If False, use
                    execute_streaming() for memory efficiency. If None (default),
                    auto-detect based on whether pipeline has a writer.

    Returns:
        Executed pipeline object
    """
    pipeline_json = json.dumps({"pipeline": steps})
    if arrays is not None:
        pipeline = pdal.Pipeline(pipeline_json, arrays=arrays)
    else:
        pipeline = pdal.Pipeline(pipeline_json)

    # Auto-detect if we need arrays returned
    if need_arrays is None:
        # If pipeline ends with a writer, we don't need arrays back
        has_writer = any(
            step.get("type", "").startswith("writers.")
            for step in steps
        )
        need_arrays = not has_writer

    if need_arrays:
        pipeline.execute()
    else:
        # Use streaming for file-to-file operations (memory efficient)
        pipeline.execute_streaming(chunk_size=1000000)

    return pipeline


class RegistrationMethod(Enum):
    """Available registration methods (maps to small_gicp registration_type)"""
    ICP = "icp"  # Standard ICP (point-to-point)
    PLANE_ICP = "plane_icp"  # Point-to-plane ICP (uses normals)
    GICP = "gicp"  # Generalized ICP (uses covariances)
    VGICP = "vgicp"  # Voxelized GICP (fastest for large clouds)


@dataclass
class RegistrationConfig:
    """Configuration for point cloud registration"""

    # General parameters
    method: str = "gicp"  # Registration type: "icp", "plane_icp", "gicp", "vgicp"
    max_correspondence_distance: Optional[float] = None  # Auto-compute if None
    max_iterations: int = 50

    # Centering and cropping parameters
    center_to_origin: bool = True  # Center both clouds to (0,0,0) before registration
    crop_dimensions: Optional[Tuple[float, float]] = None  # (x, y) crop rectangle centered on origin

    # Downsampling parameters
    downsample: bool = False  # Disable downsampling by default (use full point density)
    voxel_size: Optional[float] = None  # Auto-compute if None (when downsample=True)
    target_points: int = 100000  # Target number of points after downsampling (when downsample=True)

    # Coarse alignment parameters
    perform_coarse_alignment: bool = True
    use_ground_plane_constraint: bool = True  # Constrains rotation for landscape data

    # Filtering parameters
    point_filter: str = "ground"  # "ground" (default), "all", or "custom" (uses classification_filter)
    use_ground_filter: bool = False  # Use SMRF to classify ground points (for unclassified data)
    ground_filter_params: Optional[Dict[str, Any]] = None
    outlier_removal: bool = True
    outlier_k_neighbors: int = 20
    outlier_std_multiplier: float = 2.0
    classification_filter: Optional[Union[List[int], str]] = None  # Custom list when point_filter="custom"

    # Validation
    min_fitness_score: float = 0.3  # Minimum acceptable fitness score
    max_rmse: Optional[float] = None  # Maximum acceptable RMSE (auto if None)

    # Auto-retry parameters
    enable_auto_retry: bool = True
    max_retries: int = 3
    retry_strategies: List[str] = field(default_factory=lambda: ["increase_correspondence", "change_method", "adjust_filtering"])


class PointCloudProcessor:
    """Preprocessing utilities for point clouds using PDAL"""
    
    @staticmethod
    def estimate_optimal_voxel_size(bounds: Tuple[float, float, float, float], 
                                   total_points: int,
                                   target_points: int = 100000) -> float:
        """
        Estimate optimal voxel size for downsampling to target number of points
        
        Args:
            bounds: (minx, miny, maxx, maxy) bounds
            total_points: Total number of points
            target_points: Target number of points after downsampling
            
        Returns:
            Estimated voxel size
        """
        if total_points <= target_points:
            return 0.0  # No downsampling needed
        
        # Estimate point cloud extent
        minx, miny, maxx, maxy = bounds
        area = (maxx - minx) * (maxy - miny)
        
        if area <= 0:
            return 1.0  # Default voxel size
        
        # Points per square meter
        density = total_points / area
        
        # Target density
        target_density = target_points / area
        
        # Voxel size to achieve target density
        voxel_size = np.sqrt(1.0 / target_density) if target_density > 0 else 1.0
        
        return float(voxel_size)
    
    @staticmethod
    def filter_outliers_pdal(input_path: str, output_path: str,
                            k_neighbors: int = 20,
                            std_multiplier: float = 2.0) -> str:
        """
        Remove statistical outliers using PDAL

        Args:
            input_path: Input LAS/LAZ file
            output_path: Output LAS/LAZ file
            k_neighbors: Number of neighbors for outlier detection
            std_multiplier: Standard deviation multiplier

        Returns:
            Path to filtered file
        """
        _run_pdal_pipeline([
            {"type": "readers.las", "filename": input_path},
            {"type": "filters.outlier", "method": "statistical",
             "mean_k": k_neighbors, "multiplier": std_multiplier},
            {"type": "writers.las", "filename": output_path}
        ])
        return output_path
    
    @staticmethod
    def apply_ground_filter(input_path: str, output_path: str,
                          smrf_params: Optional[Dict[str, Any]] = None,
                          keep_only_ground: bool = True) -> str:
        """
        Apply SMRF ground classification filter

        Args:
            input_path: Input LAS/LAZ file
            output_path: Output LAS/LAZ file
            smrf_params: SMRF parameters
            keep_only_ground: If True, keep only ground points; if False, just classify

        Returns:
            Path to filtered file
        """
        defaults = {
            "cell": 1.0, "scalar": 1.25, "slope": 0.15,
            "threshold": 0.5, "window": 18.0
        }
        params = {**defaults, **(smrf_params or {})}

        steps = [
            {"type": "readers.las", "filename": input_path},
            {"type": "filters.smrf", **params}
        ]
        if keep_only_ground:
            steps.append({"type": "filters.range", "limits": "Classification[2:2]"})
        steps.append({"type": "writers.las", "filename": output_path})

        _run_pdal_pipeline(steps)
        return output_path
    
    @staticmethod
    def filter_by_classification(input_path: str, output_path: str,
                                classifications: List[int]) -> str:
        """
        Filter points by classification codes

        Args:
            input_path: Input LAS/LAZ file
            output_path: Output LAS/LAZ file
            classifications: List of classification codes to keep

        Returns:
            Path to filtered file
        """
        limits = ",".join([f"Classification[{c}:{c}]" for c in classifications])
        _run_pdal_pipeline([
            {"type": "readers.las", "filename": input_path},
            {"type": "filters.range", "limits": limits},
            {"type": "writers.las", "filename": output_path}
        ])
        return output_path
    
    @staticmethod
    def downsample_voxel_pdal(input_path: str, output_path: str,
                             voxel_size: float) -> str:
        """
        Downsample using voxel grid with PDAL

        Args:
            input_path: Input LAS/LAZ file
            output_path: Output LAS/LAZ file
            voxel_size: Voxel size for downsampling

        Returns:
            Path to downsampled file
        """
        _run_pdal_pipeline([
            {"type": "readers.las", "filename": input_path},
            {"type": "filters.voxelcenternearestneighbor", "cell": voxel_size},
            {"type": "writers.las", "filename": output_path}
        ])
        return output_path
    
    @staticmethod
    def extract_points_and_colors(las_path: str,
                                 max_points: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract points and optionally colors from LAS/LAZ file

        Args:
            las_path: Path to LAS/LAZ file
            max_points: Maximum number of points to load

        Returns:
            Points array (Nx3) and optional colors array (Nx3)
        """
        steps = [{"type": "readers.las", "filename": las_path}]
        if max_points is not None:
            steps.extend([
                {"type": "filters.randomize"},
                {"type": "filters.head", "count": max_points}
            ])

        pipeline = _run_pdal_pipeline(steps)
        arrays = pipeline.arrays
        if not arrays:
            raise RuntimeError(f"No points loaded from {las_path}")

        arr = arrays[0]
        points = np.column_stack([arr["X"], arr["Y"], arr["Z"]])

        # Check for color channels
        colors = None
        field_names = arr.dtype.names or ()
        if all(field in field_names for field in ["Red", "Green", "Blue"]):
            # Normalize colors to 0-1 range (assuming 16-bit color)
            colors = np.column_stack([
                arr["Red"] / 65535.0,
                arr["Green"] / 65535.0,
                arr["Blue"] / 65535.0
            ])

        return points, colors


class RegistrationResult:
    """Container for registration results"""

    def __init__(self):
        self.transformation: np.ndarray = np.eye(4)
        self.rmse: float = np.inf
        self.fitness: float = 0.0
        self.num_correspondences: int = 0
        self.converged: bool = False
        self.iterations: int = 0
        self.scale: float = 1.0
        self.method_used: str = ""
        self.retry_count: int = 0
        self.centroid: Optional[np.ndarray] = None  # Centroid used for centering (if any)
        self.metadata: Dict[str, Any] = {}
    
    def is_valid(self, config: RegistrationConfig) -> bool:
        """Check if registration result meets quality criteria"""
        max_rmse = config.max_rmse
        if max_rmse is None:
            # Auto-compute based on typical landscape scale
            max_rmse = 5.0  # meters, reasonable for landscape data
        
        return (self.fitness >= config.min_fitness_score and 
                self.rmse <= max_rmse and
                self.converged)
    
    def __repr__(self) -> str:
        return (f"RegistrationResult(method={self.method_used}, "
                f"rmse={self.rmse:.3f}, "
                f"fitness={self.fitness:.3f}, "
                f"converged={self.converged}, "
                f"iterations={self.iterations}, "
                f"retries={self.retry_count})")


class LandscapeAligner:
    """Main class for landscape point cloud alignment with PointCloud integration"""
    
    def __init__(self, config: Optional[RegistrationConfig] = None):
        """
        Initialize aligner with configuration
        
        Args:
            config: Registration configuration (uses defaults if None)
        """
        self.config = config or RegistrationConfig()
        self.processor = PointCloudProcessor()
        
    def align(self,
             source: Union[str, 'PointCloud', np.ndarray],
             target: Union[str, 'PointCloud', np.ndarray],
             method: Optional[RegistrationMethod] = None,
             initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        """
        Align source point cloud to target with automatic retry on failure

        Args:
            source: Source point cloud (file path, PointCloud object, or Nx3 array)
            target: Target point cloud (file path, PointCloud object, or Nx3 array)
            method: Registration method to use. If None, uses config.method.
            initial_transform: Optional initial transformation

        Returns:
            Registration result
        """
        # Use config.method if method not specified
        if method is None:
            method = RegistrationMethod(self.config.method)

        # Extract paths and metadata
        source_path, source_meta = self._extract_path_and_metadata(source)
        target_path, target_meta = self._extract_path_and_metadata(target)
        
        logger.info(f"Starting registration: {source_meta['name']} -> {target_meta['name']}")
        
        # Auto-compute max correspondence distance if needed
        if self.config.max_correspondence_distance is None:
            self._auto_compute_correspondence_distance(source_meta, target_meta)
        
        # Main registration with retry logic
        if self.config.enable_auto_retry:
            result = self._align_with_retry(
                source_path, target_path, 
                source_meta, target_meta,
                method, initial_transform
            )
        else:
            result = self._align_single_attempt(
                source_path, target_path,
                source_meta, target_meta, 
                method, initial_transform
            )
        
        return result
    
    def _extract_path_and_metadata(self, 
                                  data: Union[str, 'PointCloud', np.ndarray]) -> Tuple[str, Dict[str, Any]]:
        """Extract file path and metadata from various input formats"""
        metadata = {
            "name": "unknown",
            "bounds": None,
            "total_points": None,
            "units": "meters",
            "crs": None,
            "epoch": None
        }
        
        if isinstance(data, str):
            # Direct file path
            path = data
            metadata["name"] = Path(path).stem

            # Quick metadata extraction with PDAL
            pipeline = _run_pdal_pipeline([
                {"type": "readers.las", "filename": path, "count": 0}
            ])
            raw_meta = pipeline.metadata
            meta = json.loads(raw_meta) if isinstance(raw_meta, (str, bytes)) else raw_meta
            las_meta = meta.get("metadata", {}).get("readers.las", {})
            
            metadata["bounds"] = (
                las_meta.get("minx"), las_meta.get("miny"),
                las_meta.get("maxx"), las_meta.get("maxy")
            )
            metadata["total_points"] = las_meta.get("count")
            
        elif hasattr(data, 'filename'):
            # PointCloud object
            path = data.filename
            metadata["name"] = Path(path).stem
            metadata["bounds"] = getattr(data, 'bounds', None)
            metadata["total_points"] = getattr(data, 'total_points', None)
            metadata["units"] = getattr(data, 'horizontal_units', 'meters')
            metadata["crs"] = getattr(data, 'current_compound_crs', None) or getattr(data, 'original_compound_crs', None)
            metadata["epoch"] = getattr(data, 'epoch', None)
            
        elif isinstance(data, np.ndarray):
            # NumPy array - need to save to temp file
            with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as tmp:
                path = tmp.name

            # Write array to LAS using PDAL
            structured_array = np.zeros(len(data), dtype=[('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')])
            structured_array['X'] = data[:, 0]
            structured_array['Y'] = data[:, 1]
            structured_array['Z'] = data[:, 2]

            _run_pdal_pipeline(
                [{"type": "writers.las", "filename": path}],
                arrays=[structured_array]
            )
            
            metadata["name"] = "numpy_array"
            metadata["total_points"] = len(data)
            metadata["bounds"] = (
                data[:, 0].min(), data[:, 1].min(),
                data[:, 0].max(), data[:, 1].max()
            )
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
        
        return path, metadata
    
    def _auto_compute_correspondence_distance(self, 
                                             source_meta: Dict[str, Any],
                                             target_meta: Dict[str, Any]) -> None:
        """Auto-compute max correspondence distance based on data scale and units"""
        bounds_list = []
        for meta in [source_meta, target_meta]:
            if meta["bounds"] and all(b is not None for b in meta["bounds"]):
                bounds_list.append(meta["bounds"])
        
        if not bounds_list:
            # Default for landscape data in meters
            self.config.max_correspondence_distance = 10.0
            return
        
        # Compute based on extent
        max_extent = 0
        for bounds in bounds_list:
            minx, miny, maxx, maxy = bounds
            extent = max(maxx - minx, maxy - miny)
            max_extent = max(max_extent, extent)
        
        # Adjust for units (assume meters unless specified otherwise)
        unit_scale = 1.0
        for meta in [source_meta, target_meta]:
            if meta["units"] and "foot" in meta["units"].lower():
                unit_scale = 0.3048  # Convert to meters
                break
        
        max_extent *= unit_scale
        
        # Set to 1% of max extent, clamped between 0.5 and 50 meters
        self.config.max_correspondence_distance = np.clip(
            max_extent * 0.01, 0.5, 50.0
        )
        logger.info(f"Auto-computed max correspondence distance: {self.config.max_correspondence_distance:.2f} meters")
    
    def _align_with_retry(self,
                         source_path: str,
                         target_path: str,
                         source_meta: Dict[str, Any],
                         target_meta: Dict[str, Any],
                         method: RegistrationMethod,
                         initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        """
        Align with automatic retry using different strategies
        """
        best_result = None
        strategies_tried = []
        original_method = method
        
        for retry in range(self.config.max_retries):
            # Adjust config based on retry strategy
            if retry > 0:
                strategy = self.config.retry_strategies[min(retry - 1, len(self.config.retry_strategies) - 1)]
                self._apply_retry_strategy(strategy, retry)
                strategies_tried.append(strategy)
                
                # Try different methods on retry
                if retry == 1 and method == RegistrationMethod.VGICP:
                    method = RegistrationMethod.GICP
                    logger.info("Switching from VGICP to GICP for retry")
                elif retry == 2:
                    method = RegistrationMethod.ICP
                    logger.info("Switching to standard ICP for retry")
            
            try:
                result = self._align_single_attempt(
                    source_path, target_path,
                    source_meta, target_meta,
                    method, initial_transform
                )
                result.retry_count = retry
                
                if result.is_valid(self.config):
                    logger.info(f"Registration succeeded on attempt {retry + 1}")
                    return result
                
                if best_result is None or result.rmse < best_result.rmse:
                    best_result = result
                
                logger.warning(f"Registration attempt {retry + 1} failed validation. "
                             f"RMSE: {result.rmse:.3f}, Fitness: {result.fitness:.3f}")
                
            except Exception as e:
                logger.error(f"Registration attempt {retry + 1} failed: {e}")
        
        if best_result is None:
            best_result = RegistrationResult()
            best_result.metadata["error"] = "All registration attempts failed"
            best_result.metadata["strategies_tried"] = strategies_tried
        
        logger.warning(f"Registration failed after {self.config.max_retries} attempts. "
                      f"Best RMSE: {best_result.rmse:.3f}")
        return best_result
    
    def _apply_retry_strategy(self, strategy: str, retry_num: int) -> None:
        """Apply a retry strategy by modifying config"""
        if strategy == "increase_correspondence":
            # Increase correspondence distance
            self.config.max_correspondence_distance *= 1.5
            logger.info(f"Increased max correspondence distance to {self.config.max_correspondence_distance:.2f}")
            
        elif strategy == "change_method":
            # Method change handled in calling function
            pass
            
        elif strategy == "adjust_filtering":
            # Relax filtering parameters
            if self.config.outlier_removal:
                self.config.outlier_std_multiplier *= 1.5
                logger.info(f"Relaxed outlier threshold to {self.config.outlier_std_multiplier:.1f} std")
            
            # Increase target points for better coverage
            self.config.target_points = int(self.config.target_points * 1.5)
            logger.info(f"Increased target points to {self.config.target_points}")
    
    def _align_single_attempt(self,
                             source_path: str,
                             target_path: str,
                             source_meta: Dict[str, Any],
                             target_meta: Dict[str, Any],
                             method: RegistrationMethod,
                             initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        """
        Single registration attempt with preprocessing
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Preprocessing pipeline
            processed_source = self._preprocess_pointcloud(source_path, tmpdir, "source", source_meta)
            processed_target = self._preprocess_pointcloud(target_path, tmpdir, "target", target_meta)

            # Load processed points
            source_points, _ = self.processor.extract_points_and_colors(processed_source)
            target_points, _ = self.processor.extract_points_and_colors(processed_target)

            logger.info(f"After preprocessing: {len(source_points)} source, {len(target_points)} target points")

            # Center both point clouds to common origin if requested
            centroid = None
            if self.config.center_to_origin:
                # Compute shared centroid (weighted average)
                n_source = len(source_points)
                n_target = len(target_points)
                n_total = n_source + n_target

                source_centroid = np.mean(source_points, axis=0)
                target_centroid = np.mean(target_points, axis=0)
                centroid = (source_centroid * n_source + target_centroid * n_target) / n_total

                logger.info(f"Centering to origin (centroid: [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}])")

                # Center both point clouds in-place
                source_points -= centroid
                target_points -= centroid

            # Crop to rectangular extent centered on origin if requested
            if self.config.crop_dimensions is not None:
                crop_x, crop_y = self.config.crop_dimensions
                half_x, half_y = crop_x / 2.0, crop_y / 2.0

                # Filter source points
                source_mask = (
                    (source_points[:, 0] >= -half_x) & (source_points[:, 0] <= half_x) &
                    (source_points[:, 1] >= -half_y) & (source_points[:, 1] <= half_y)
                )
                source_points = source_points[source_mask]

                # Filter target points
                target_mask = (
                    (target_points[:, 0] >= -half_x) & (target_points[:, 0] <= half_x) &
                    (target_points[:, 1] >= -half_y) & (target_points[:, 1] <= half_y)
                )
                target_points = target_points[target_mask]

                logger.info(f"Cropped to {crop_x}x{crop_y}m rectangle: "
                           f"{len(source_points)} source, {len(target_points)} target points")

            # Coarse alignment if requested
            if self.config.perform_coarse_alignment and initial_transform is None:
                initial_transform = self._coarse_alignment(source_points, target_points)
                logger.info("Coarse alignment completed")

            # Fine registration
            result = self._fine_registration(
                source_points, target_points, method, initial_transform
            )

            result.method_used = method.value
            result.centroid = centroid

            return result
    
    def _preprocess_pointcloud(self,
                              input_path: str,
                              tmpdir: str,
                              prefix: str,
                              metadata: Dict[str, Any]) -> str:
        """
        Apply preprocessing pipeline to point cloud
        """
        current_path = input_path
        step = 0
        
        # 1. Point filtering based on classification
        classifications = None
        if self.config.point_filter == "ground":
            classifications = [2]  # Ground class only
        elif self.config.point_filter == "custom" and self.config.classification_filter is not None:
            if self.config.classification_filter == "ground":
                classifications = [2]
            elif isinstance(self.config.classification_filter, (list, tuple)):
                classifications = list(self.config.classification_filter)
        # point_filter == "all" means no filtering

        if classifications:
            step += 1
            output_path = os.path.join(tmpdir, f"{prefix}_step{step}_classified.laz")
            current_path = self.processor.filter_by_classification(
                current_path, output_path, classifications
            )
            logger.info(f"Filtered {prefix} to classifications: {classifications}")
        
        # 2. Ground filtering with SMRF
        if self.config.use_ground_filter:
            step += 1
            output_path = os.path.join(tmpdir, f"{prefix}_step{step}_ground.laz")
            current_path = self.processor.apply_ground_filter(
                current_path, output_path, self.config.ground_filter_params
            )
            logger.info(f"Applied ground filter to {prefix}")
        
        # 3. Outlier removal
        if self.config.outlier_removal:
            step += 1
            output_path = os.path.join(tmpdir, f"{prefix}_step{step}_cleaned.laz")
            current_path = self.processor.filter_outliers_pdal(
                current_path, output_path,
                self.config.outlier_k_neighbors,
                self.config.outlier_std_multiplier
            )
            logger.info(f"Removed outliers from {prefix}")
        
        # 4. Downsampling (only if enabled)
        if self.config.downsample:
            if self.config.voxel_size is None and metadata["bounds"] and metadata["total_points"]:
                # Auto-compute voxel size
                voxel_size = self.processor.estimate_optimal_voxel_size(
                    metadata["bounds"],
                    metadata["total_points"],
                    self.config.target_points
                )
            else:
                voxel_size = self.config.voxel_size or 1.0

            if voxel_size > 0:
                step += 1
                output_path = os.path.join(tmpdir, f"{prefix}_step{step}_downsampled.laz")
                current_path = self.processor.downsample_voxel_pdal(
                    current_path, output_path, voxel_size
                )
                logger.info(f"Downsampled {prefix} with voxel size {voxel_size:.3f}")

        return current_path
    
    def _coarse_alignment(self, 
                         source: np.ndarray, 
                         target: np.ndarray) -> np.ndarray:
        """
        Perform coarse alignment using centroids and PCA
        """
        # Centroid alignment
        source_centroid = source.mean(axis=0)
        target_centroid = target.mean(axis=0)
        
        initial_transform = np.eye(4)
        initial_transform[:3, 3] = target_centroid - source_centroid
        
        # Optional: PCA alignment for rotation (for landscapes, often not needed)
        if self.config.use_ground_plane_constraint:
            # For landscapes, we typically don't want to rotate around vertical axis
            # Just use translation
            pass
        elif len(source) > 1000 and len(target) > 1000:
            # Use subset for PCA
            source_subset = source[::max(1, len(source)//1000)]
            target_subset = target[::max(1, len(target)//1000)]
            
            # Center points
            source_centered = source_subset - source_centroid
            target_centered = target_subset - target_centroid
            
            # Compute principal axes
            _, _, source_axes = np.linalg.svd(source_centered.T @ source_centered)
            _, _, target_axes = np.linalg.svd(target_centered.T @ target_centered)
            
            # Compute rotation to align principal axes
            rotation = target_axes.T @ source_axes
            
            # Check for reflection
            if np.linalg.det(rotation) < 0:
                target_axes[-1] *= -1
                rotation = target_axes.T @ source_axes
            
            initial_transform[:3, :3] = rotation
        
        return initial_transform
    
    def _fine_registration(self,
                          source: np.ndarray,
                          target: np.ndarray,
                          method: RegistrationMethod,
                          initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        """
        Perform fine registration using small_gicp
        """
        _require_small_gicp()
        result = RegistrationResult()

        # Set initial transformation
        if initial_transform is None:
            initial_transform = np.eye(4)

        try:
            # Perform registration
            logger.info(f"Running {method.value} registration...")

            max_corr_dist = self.config.max_correspondence_distance or 1.0
            num_threads = 4

            # Methods requiring preprocessing (normals/covariances)
            needs_preprocessing = method in [
                RegistrationMethod.PLANE_ICP,
                RegistrationMethod.GICP,
                RegistrationMethod.VGICP,
            ]

            if needs_preprocessing:
                # PLANE_ICP, GICP, VGICP require preprocessing for normals/covariances
                # See: https://github.com/koide3/small_gicp

                # Build preprocessing kwargs
                preprocess_kwargs = {"num_threads": num_threads}
                if self.config.downsample:
                    downsample_resolution = self.config.voxel_size or 0.25
                    preprocess_kwargs["downsampling_resolution"] = downsample_resolution
                    logger.info(
                        f"Preprocessing (downsample={downsample_resolution}m + "
                        "covariance estimation)..."
                    )
                else:
                    logger.info(
                        "Preprocessing (covariance estimation, no downsampling)..."
                    )

                target_cloud, target_tree = small_gicp.preprocess_points(
                    target, **preprocess_kwargs
                )
                source_cloud, source_tree = small_gicp.preprocess_points(
                    source, **preprocess_kwargs
                )

                logger.info(
                    f"After preprocessing: {source_cloud.size()} source, "
                    f"{target_cloud.size()} target points"
                )

                # Map method to small_gicp registration_type
                # VGICP with preprocessed points is effectively GICP
                reg_type = "GICP" if method == RegistrationMethod.VGICP else method.value.upper()

                reg_result = small_gicp.align(
                    target_cloud,
                    source_cloud,
                    target_tree,
                    init_T_target_source=initial_transform,
                    registration_type=reg_type,
                    max_correspondence_distance=max_corr_dist,
                    max_iterations=self.config.max_iterations,
                    num_threads=num_threads,
                )
            else:
                # Plain ICP can use raw numpy arrays directly
                align_kwargs = {
                    "init_T_target_source": initial_transform,
                    "registration_type": "ICP",
                    "max_correspondence_distance": max_corr_dist,
                    "max_iterations": self.config.max_iterations,
                    "num_threads": num_threads,
                }
                if self.config.downsample:
                    downsample_resolution = self.config.voxel_size or 0.25
                    align_kwargs["downsampling_resolution"] = downsample_resolution
                    logger.info(f"Using downsampling at {downsample_resolution}m")

                reg_result = small_gicp.align(target, source, **align_kwargs)

            # Extract results - use T_target_source attribute
            result.transformation = reg_result.T_target_source
            result.converged = getattr(reg_result, 'converged', True)
            result.iterations = getattr(reg_result, 'iterations', 0)
            
            # Compute fitness metrics
            result.rmse, result.fitness, result.num_correspondences = \
                self._compute_fitness(source, target, result.transformation)
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            result.metadata['error'] = str(e)
        
        return result
    
    def _compute_fitness(self, 
                        source: np.ndarray, 
                        target: np.ndarray,
                        transformation: np.ndarray) -> Tuple[float, float, int]:
        """
        Compute registration fitness metrics
        """
        # Transform source points
        source_h = np.hstack([source, np.ones((len(source), 1))])
        source_transformed = (source_h @ transformation.T)[:, :3]
        
        # Find correspondences
        tree = cKDTree(target)
        distances, _ = tree.query(source_transformed)
        
        # Filter by max correspondence distance
        max_dist = self.config.max_correspondence_distance or 10.0
        mask = distances < max_dist
        valid_distances = distances[mask]
        
        if len(valid_distances) == 0:
            return np.inf, 0.0, 0
        
        # Compute metrics
        rmse = np.sqrt(np.mean(valid_distances ** 2))
        fitness = len(valid_distances) / len(source)
        num_correspondences = len(valid_distances)
        
        return rmse, fitness, num_correspondences
