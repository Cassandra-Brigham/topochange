# pointcloudpair.py
"""Compare, transform, align, and difference two point clouds.

Provides tools for:
- Comparing CRS, epoch, geoid, and other parameters between point clouds
- Transforming pc1 to match pc2's reference frame
- ICP-based alignment using small_gicp
- 3D point cloud differencing
- 2D DEM-based differencing via RasterPair

Transformation order:
1. Dynamic epoch transformation
2. Horizontal CRS reprojection
3. Vertical datum transformation
4. ICP alignment (optional)
5. DEM creation and differencing
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import numpy as np
# Use pdal_wrapper for Colab compatibility (falls back to native pdal locally)
try:
    from .pdal_wrapper import pdal
except ImportError:
    import pdal
from pyproj import CRS as _CRS
from pyproj.crs import CompoundCRS

# Handle imports
from .pointcloud import PointCloud
from .raster import Raster
from .rasterpair import RasterPair
from .crs_utils import (
    _ensure_crs_obj,
    is_3d_geographic_crs,
    extract_ellipsoidal_height_as_vertical_crs,
)
try:
    from .unit_utils import (
        UnitInfo,
        UNKNOWN_UNIT,
        METER,
        lookup_unit,
        get_conversion_factor,
    )
    _UNIT_UTILS_AVAILABLE = True
except ImportError:
    _UNIT_UTILS_AVAILABLE = False
    UnitInfo = None
    UNKNOWN_UNIT = None
    METER = None


# =============================================================================
# Utility Functions
# =============================================================================

def _has_small_gicp() -> bool:
    """Check if small_gicp is available."""
    try:
        import small_gicp
        return True
    except ImportError:
        return False


def _crs_equivalent(crs1: Any, crs2: Any) -> bool:
    """
    Check if two CRS are equivalent.
    
    Uses EPSG comparison first, then falls back to pyproj equals().
    """
    if crs1 is None and crs2 is None:
        return True
    if crs1 is None or crs2 is None:
        return False
    
    try:
        obj1 = _ensure_crs_obj(crs1)
        obj2 = _ensure_crs_obj(crs2)
        
        # Try EPSG comparison first (fastest)
        epsg1 = obj1.to_epsg()
        epsg2 = obj2.to_epsg()
        if epsg1 is not None and epsg2 is not None:
            return epsg1 == epsg2
        
        return obj1.equals(obj2)
    except Exception:
        return str(crs1) == str(crs2)


def _geoid_equivalent(geoid1: Optional[str], geoid2: Optional[str]) -> bool:
    """
    Check if two geoid model names are equivalent.
    
    Handles case-insensitive comparison and common naming variations.
    """
    if geoid1 is None and geoid2 is None:
        return True
    if geoid1 is None or geoid2 is None:
        return False
    
    def normalize(g):
        g = str(g).lower().strip()
        # Remove common prefixes
        for prefix in ['us_noaa_', 'noaa_', 'ngs_', 'egm', 'geoid']:
            if g.startswith(prefix):
                g = g[len(prefix):]
        # Remove file extensions
        for ext in ['.tif', '.tiff', '.gtx', '.bin']:
            if g.endswith(ext):
                g = g[:-len(ext)]
        return g
    
    return normalize(geoid1) == normalize(geoid2)


def _units_equivalent(unit1: Any, unit2: Any) -> Tuple[bool, Optional[float]]:
    """
    Check if two units are equivalent and compute conversion factor if not.
    
    Returns
    -------
    tuple[bool, float or None]
        (are_equivalent, conversion_factor)
    """
    if unit1 is None and unit2 is None:
        return True, None
    if unit1 is None or unit2 is None:
        return False, None
    
    # Get unit names
    if hasattr(unit1, 'name'):
        name1 = unit1.name
    else:
        name1 = str(unit1).lower()
    
    if hasattr(unit2, 'name'):
        name2 = unit2.name
    else:
        name2 = str(unit2).lower()
    
    if name1 == name2:
        return True, None
    
    if name1 == "unknown" or name2 == "unknown":
        return False, None
    
    # Try to get conversion factor
    try:
        unit1_obj = lookup_unit(name1) if isinstance(unit1, str) else unit1
        unit2_obj = lookup_unit(name2) if isinstance(unit2, str) else unit2
        if unit1_obj and unit2_obj:
            factor = get_conversion_factor(unit1_obj, unit2_obj)
            return False, factor
    except Exception:
        pass
    
    return False, None


def _load_points_from_las(
    filename: str,
    max_points: Optional[int] = None,
    voxel_size: Optional[float] = None,
    streaming: bool = True,
) -> np.ndarray:
    """
    Load XYZ points from a LAS/LAZ file using PDAL.

    Parameters
    ----------
    filename : str
        Path to LAS/LAZ file
    max_points : int, optional
        Maximum number of points to load (random sampling after load).
        For memory efficiency in Colab, prefer using voxel_size instead.
    voxel_size : float, optional
        Voxel grid downsampling size in meters. Applied via PDAL filters.voxeldownsize
        which is more memory efficient than loading all points then sampling.
    streaming : bool, default True
        If True and voxel_size is set, use streaming mode to process points
        in chunks. This prevents memory spikes for large point clouds.

    Returns
    -------
    np.ndarray
        Nx3 array of XYZ coordinates
    """
    import tempfile
    import os as _os

    # For streaming with voxel downsampling, write to temp file first
    # This avoids loading all points into Python memory
    if streaming and voxel_size is not None:
        # Create temp file for downsampled output
        with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Pipeline: read -> voxel downsample -> write to temp file
            downsample_spec = {
                "pipeline": [
                    {"type": "readers.las", "filename": str(filename)},
                    {
                        "type": "filters.voxeldownsize",
                        "cell": voxel_size,
                        "mode": "center",
                    },
                    {"type": "writers.las", "filename": tmp_path},
                ]
            }
            pipe = pdal.Pipeline(json.dumps(downsample_spec))
            # Use streaming execution - processes in chunks
            if hasattr(pipe, 'execute_streaming'):
                pipe.execute_streaming(chunk_size=100000)
            else:
                pipe.execute()

            # Now read the (much smaller) downsampled file
            read_spec = {
                "pipeline": [
                    {"type": "readers.las", "filename": tmp_path},
                ]
            }
            read_pipe = pdal.Pipeline(json.dumps(read_spec))
            read_pipe.execute()

            arrays = read_pipe.arrays
            if not arrays or len(arrays) == 0:
                raise ValueError(f"No points after downsampling from {filename}")

            arr = arrays[0]
            points = np.column_stack([arr['X'], arr['Y'], arr['Z']]).astype(np.float64)

        finally:
            # Clean up temp file
            if _os.path.exists(tmp_path):
                _os.remove(tmp_path)

    else:
        # Standard non-streaming approach
        pipeline_spec = {
            "pipeline": [
                {"type": "readers.las", "filename": str(filename)},
            ]
        }

        # Use voxel grid downsampling if specified
        if voxel_size is not None:
            pipeline_spec["pipeline"].append({
                "type": "filters.voxeldownsize",
                "cell": voxel_size,
                "mode": "center",
            })

        pipe = pdal.Pipeline(json.dumps(pipeline_spec))
        pipe.execute()

        arrays = pipe.arrays
        if not arrays or len(arrays) == 0:
            raise ValueError(f"No points loaded from {filename}")

        arr = arrays[0]
        points = np.column_stack([arr['X'], arr['Y'], arr['Z']]).astype(np.float64)

    # Additional random sampling if max_points specified and still too many
    if max_points is not None and len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    return points


def _save_transformed_las(
    source_filename: str,
    output_filename: str,
    transformation_matrix: np.ndarray,
    centroid: np.ndarray = None,
) -> None:
    """
    Apply a 4x4 transformation matrix to a LAS file and save.

    Parameters
    ----------
    source_filename : str
        Input LAS/LAZ file
    output_filename : str
        Output LAS/LAZ file
    transformation_matrix : np.ndarray
        4x4 homogeneous transformation matrix
    centroid : np.ndarray, optional
        If provided, the transformation is applied in centered coordinates:
        1. Translate points to origin (subtract centroid)
        2. Apply transformation matrix
        3. Translate back (add centroid)
        This is required when the transformation was computed on centered data.
    """
    # Build PDAL transformation matrix string (row-major, 16 values)
    matrix_str = " ".join(str(x) for x in transformation_matrix.flatten())

    pipeline_steps = [
        {"type": "readers.las", "filename": str(source_filename)},
    ]

    if centroid is not None:
        # Step 1: Translate to origin (subtract centroid)
        T_to_origin = f"1 0 0 {-centroid[0]} 0 1 0 {-centroid[1]} 0 0 1 {-centroid[2]} 0 0 0 1"
        pipeline_steps.append({
            "type": "filters.transformation",
            "matrix": T_to_origin,
        })

        # Step 2: Apply the ICP transformation
        pipeline_steps.append({
            "type": "filters.transformation",
            "matrix": matrix_str,
        })

        # Step 3: Translate back (add centroid)
        T_from_origin = f"1 0 0 {centroid[0]} 0 1 0 {centroid[1]} 0 0 1 {centroid[2]} 0 0 0 1"
        pipeline_steps.append({
            "type": "filters.transformation",
            "matrix": T_from_origin,
        })
    else:
        # Direct transformation (no centering)
        pipeline_steps.append({
            "type": "filters.transformation",
            "matrix": matrix_str,
        })

    pipeline_steps.append({"type": "writers.las", "filename": str(output_filename)})

    pipeline_spec = {"pipeline": pipeline_steps}

    pipe = pdal.Pipeline(json.dumps(pipeline_spec))
    pipe.execute()


# =============================================================================
# PointCloudPair Class
# =============================================================================

@dataclass
class PointCloudPair:
    """
    Pair of point clouds for comparison, transformation, alignment, and differencing.
    
    Attributes
    ----------
    pc1 : PointCloud
        The "compare" point cloud (to be transformed)
    pc2 : PointCloud
        The "reference" point cloud (target reference frame)
    
    Notes
    -----
    By convention:
    - pc1 is the "compare" or "source" point cloud
    - pc2 is the "reference" or "target" point cloud
    - Transformations are applied to pc1 to match pc2
    - Differences are computed as pc2 - pc1 (positive = gain)
    """
    
    pc1: PointCloud  # Compare
    pc2: PointCloud  # Reference
    
    # Internal state
    _transformation_history: List[Dict[str, Any]] = field(default_factory=list)
    _pc1_transformed: Optional[PointCloud] = field(default=None, repr=False)
    _pc1_cropped: Optional[PointCloud] = field(default=None, repr=False)
    _pc1_cropped_original: Optional[PointCloud] = field(default=None, repr=False)  # Preserved unaligned version
    _pc2_cropped: Optional[PointCloud] = field(default=None, repr=False)
    _alignment_result: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize internal state."""
        self._transformation_history = []
        self._pc1_transformed = None
        self._pc1_cropped = None
        self._pc1_cropped_original = None
        self._pc2_cropped = None
        self._alignment_result = None
    
    # =========================================================================
    # Comparison Methods
    # =========================================================================
    
    def check_all_match(self) -> Dict[str, Any]:
        """
        Check if all CRS/metadata parameters match between pc1 and pc2.
        
        Returns
        -------
        dict
            Dictionary with match status for each parameter:
            - compound_crs, horizontal_crs, vertical_crs
            - geoid, epoch, vertical_units
            - transformations_needed: list of required transformations
        """
        result = {
            'compound_crs': {'match': False, 'pc1': None, 'pc2': None},
            'horizontal_crs': {'match': False, 'pc1': None, 'pc2': None},
            'vertical_crs': {'match': False, 'pc1': None, 'pc2': None},
            'geoid': {'match': False, 'pc1': None, 'pc2': None},
            'epoch': {'match': False, 'pc1': None, 'pc2': None},
            'vertical_units': {'match': False, 'pc1': None, 'pc2': None},
            'transformations_needed': [],
        }
        
        # Compound/3D CRS
        pc1_comp = (
            getattr(self.pc1, 'current_compound_crs', None) or
            getattr(self.pc1, 'original_compound_crs', None)
        )
        pc2_comp = (
            getattr(self.pc2, 'current_compound_crs', None) or
            getattr(self.pc2, 'original_compound_crs', None)
        )
        result['compound_crs']['pc1'] = (
            pc1_comp[:100] + '...' if pc1_comp and len(str(pc1_comp)) > 100 else pc1_comp
        )
        result['compound_crs']['pc2'] = (
            pc2_comp[:100] + '...' if pc2_comp and len(str(pc2_comp)) > 100 else pc2_comp
        )
        result['compound_crs']['match'] = _crs_equivalent(pc1_comp, pc2_comp)
        
        # Horizontal CRS
        pc1_horiz = (
            getattr(self.pc1, 'current_horizontal_crs', None) or
            getattr(self.pc1, 'original_horizontal_crs', None)
        )
        pc2_horiz = (
            getattr(self.pc2, 'current_horizontal_crs', None) or
            getattr(self.pc2, 'original_horizontal_crs', None)
        )
        result['horizontal_crs']['pc1'] = (
            pc1_horiz[:100] + '...' if pc1_horiz and len(str(pc1_horiz)) > 100 else pc1_horiz
        )
        result['horizontal_crs']['pc2'] = (
            pc2_horiz[:100] + '...' if pc2_horiz and len(str(pc2_horiz)) > 100 else pc2_horiz
        )
        result['horizontal_crs']['match'] = _crs_equivalent(pc1_horiz, pc2_horiz)
        if not result['horizontal_crs']['match']:
            result['transformations_needed'].append('horizontal_crs')
        
        # Vertical CRS
        pc1_vert = (
            getattr(self.pc1, 'current_vertical_crs', None) or
            getattr(self.pc1, 'original_vertical_crs', None)
        )
        pc2_vert = (
            getattr(self.pc2, 'current_vertical_crs', None) or
            getattr(self.pc2, 'original_vertical_crs', None)
        )
        result['vertical_crs']['pc1'] = (
            pc1_vert[:100] + '...' if pc1_vert and len(str(pc1_vert)) > 100 else pc1_vert
        )
        result['vertical_crs']['pc2'] = (
            pc2_vert[:100] + '...' if pc2_vert and len(str(pc2_vert)) > 100 else pc2_vert
        )
        result['vertical_crs']['match'] = _crs_equivalent(pc1_vert, pc2_vert)
        
        # Geoid model
        pc1_geoid = getattr(self.pc1, 'geoid_model', None)
        pc2_geoid = getattr(self.pc2, 'geoid_model', None)
        result['geoid']['pc1'] = pc1_geoid
        result['geoid']['pc2'] = pc2_geoid
        result['geoid']['match'] = _geoid_equivalent(pc1_geoid, pc2_geoid)
        
        # Check if vertical datum transformation is needed
        pc1_ortho = getattr(self.pc1, 'is_orthometric', None)
        pc2_ortho = getattr(self.pc2, 'is_orthometric', None)
        if not result['vertical_crs']['match'] or not result['geoid']['match']:
            if pc1_ortho != pc2_ortho or not result['geoid']['match']:
                result['transformations_needed'].append('vertical_datum')
        
        # Epoch
        pc1_epoch = getattr(self.pc1, 'epoch', None)
        pc2_epoch = getattr(self.pc2, 'epoch', None)
        result['epoch']['pc1'] = pc1_epoch
        result['epoch']['pc2'] = pc2_epoch
        if pc1_epoch is not None and pc2_epoch is not None:
            result['epoch']['match'] = abs(pc1_epoch - pc2_epoch) < 0.001  # ~8 hours
        else:
            result['epoch']['match'] = pc1_epoch is None and pc2_epoch is None
        if not result['epoch']['match'] and pc1_epoch is not None and pc2_epoch is not None:
            result['transformations_needed'].append('epoch')
        
        # Vertical units
        pc1_vunit = getattr(self.pc1, 'vertical_unit', UNKNOWN_UNIT)
        pc2_vunit = getattr(self.pc2, 'vertical_unit', UNKNOWN_UNIT)
        pc1_vunit_name = pc1_vunit.name if hasattr(pc1_vunit, 'name') else str(pc1_vunit)
        pc2_vunit_name = pc2_vunit.name if hasattr(pc2_vunit, 'name') else str(pc2_vunit)
        result['vertical_units']['pc1'] = pc1_vunit_name
        result['vertical_units']['pc2'] = pc2_vunit_name
        units_match, _ = _units_equivalent(pc1_vunit, pc2_vunit)
        result['vertical_units']['match'] = units_match
        if not units_match:
            result['transformations_needed'].append('vertical_units')
        
        return result
    
    def print_comparison(self) -> None:
        """Print a human-readable comparison of the two point clouds."""
        comparison = self.check_all_match()

        def match_sym(val):
            return "Yes" if val else "No"

        print("\n--- PointCloudPair Comparison ---")
        print(f"\nCompare (pc1):   {Path(self.pc1.filename).name}")
        print(f"Reference (pc2): {Path(self.pc2.filename).name}")

        print(f"\n{'Parameter':<20} {'Match':<8} {'PC1':<20} {'PC2':<20}")
        print("-" * 70)
        
        # Horizontal CRS - get directly from point clouds, not from truncated comparison dict
        pc1_horiz = (
            getattr(self.pc1, 'current_horizontal_crs', None) or
            getattr(self.pc1, 'original_horizontal_crs', None)
        )
        pc2_horiz = (
            getattr(self.pc2, 'current_horizontal_crs', None) or
            getattr(self.pc2, 'original_horizontal_crs', None)
        )
        
        pc1_str = "None"
        pc2_str = "None"
        
        if pc1_horiz is not None:
            try:
                crs_obj = _ensure_crs_obj(pc1_horiz)
                epsg = crs_obj.to_epsg()
                if epsg:
                    pc1_str = f"EPSG:{epsg}"
                elif crs_obj.name:
                    # Truncate long names
                    name = crs_obj.name
                    pc1_str = name[:18] + ".." if len(name) > 20 else name
                else:
                    pc1_str = "Custom"
            except Exception:
                pc1_str = "Unknown"
        
        if pc2_horiz is not None:
            try:
                crs_obj = _ensure_crs_obj(pc2_horiz)
                epsg = crs_obj.to_epsg()
                if epsg:
                    pc2_str = f"EPSG:{epsg}"
                elif crs_obj.name:
                    name = crs_obj.name
                    pc2_str = name[:18] + ".." if len(name) > 20 else name
                else:
                    pc2_str = "Custom"
            except Exception:
                pc2_str = "Unknown"
        
        print(f"{'Horizontal CRS':<20} {match_sym(comparison['horizontal_crs']['match']):<8} {pc1_str:<20} {pc2_str:<20}")
        
        # Vertical CRS - get directly from point clouds
        pc1_vert = (
            getattr(self.pc1, 'current_vertical_crs', None) or
            getattr(self.pc1, 'original_vertical_crs', None)
        )
        pc2_vert = (
            getattr(self.pc2, 'current_vertical_crs', None) or
            getattr(self.pc2, 'original_vertical_crs', None)
        )
        
        # Determine vertical type
        def get_vertical_type(vert_crs, pc) -> str:
            if vert_crs is None:
                return "None"
            
            vert_str = str(vert_crs).lower()
            
            # Check for ellipsoidal
            if "ellipsoidal" in vert_str:
                return "Ellipsoidal"
            
            # Check is_orthometric attribute
            is_ortho = getattr(pc, 'is_orthometric', None)
            if is_ortho is True:
                return "Orthometric"
            elif is_ortho is False:
                return "Ellipsoidal"
            
            # Try to extract from CRS
            try:
                crs_obj = _ensure_crs_obj(vert_crs)
                epsg = crs_obj.to_epsg()
                if epsg:
                    # Known orthometric EPSG codes
                    if epsg in (5703, 5866, 6647, 8228):  # NAVD88, CGVD2013, etc.
                        return "Orthometric"
                    return f"EPSG:{epsg}"
                if crs_obj.name:
                    name = crs_obj.name
                    if "navd" in name.lower() or "orthometric" in name.lower():
                        return "Orthometric"
                    return name[:18] + ".." if len(name) > 20 else name
            except Exception:
                pass
            
            return "Unknown"
        
        pc1_vert_str = get_vertical_type(pc1_vert, self.pc1)
        pc2_vert_str = get_vertical_type(pc2_vert, self.pc2)
        
        print(f"{'Vertical CRS':<20} {match_sym(comparison['vertical_crs']['match']):<8} {pc1_vert_str:<20} {pc2_vert_str:<20}")
        
        # Geoid
        pc1_geoid = comparison['geoid']['pc1'] or "None"
        pc2_geoid = comparison['geoid']['pc2'] or "None"
        pc1_geoid = pc1_geoid[:18] + ".." if len(pc1_geoid) > 20 else pc1_geoid
        pc2_geoid = pc2_geoid[:18] + ".." if len(pc2_geoid) > 20 else pc2_geoid
        print(f"{'Geoid Model':<20} {match_sym(comparison['geoid']['match']):<8} {pc1_geoid:<20} {pc2_geoid:<20}")

        # Epoch
        pc1_epoch = f"{comparison['epoch']['pc1']:.4f}" if comparison['epoch']['pc1'] else "None"
        pc2_epoch = f"{comparison['epoch']['pc2']:.4f}" if comparison['epoch']['pc2'] else "None"
        print(f"{'Epoch':<20} {match_sym(comparison['epoch']['match']):<8} {pc1_epoch:<20} {pc2_epoch:<20}")

        # Units
        print(f"{'Vertical Units':<20} {match_sym(comparison['vertical_units']['match']):<8} {comparison['vertical_units']['pc1']:<20} {comparison['vertical_units']['pc2']:<20}")

        print("-" * 70)

        if comparison['transformations_needed']:
            print(f"\nTransformations needed: {', '.join(comparison['transformations_needed'])}")
        else:
            print(f"\nPoint clouds are fully aligned.")

        if self._transformation_history:
            print(f"\nTransformation steps applied:")
            for i, step in enumerate(self._transformation_history, 1):
                print(f"  {i}. {step.get('step', 'unknown')}")

        print("")

    # =========================================================================
    # Overlap and Cropping Methods
    # =========================================================================

    def get_bounding_polygons(
        self,
        use_transformed: bool = True,
    ) -> Dict[str, Any]:
        """
        Get bounding polygons for both point clouds.

        Parameters
        ----------
        use_transformed : bool, default True
            If True and pc1 has been transformed, use the transformed version.

        Returns
        -------
        dict
            {
                'pc1_polygon': shapely.Polygon (in UTM),
                'pc2_polygon': shapely.Polygon (in UTM),
                'pc1_polygon_4326': shapely.Polygon (in WGS84),
                'pc2_polygon_4326': shapely.Polygon (in WGS84),
                'pc1_bounds': tuple (minx, miny, maxx, maxy),
                'pc2_bounds': tuple (minx, miny, maxx, maxy),
                'epsg_utm': str,
            }
        """
        # Select pc1 source
        pc1_source = self._pc1_transformed if use_transformed and self._pc1_transformed else self.pc1

        # Get polygons from point clouds (set during from_file())
        pc1_poly_utm = getattr(pc1_source, 'poly_utm', None)
        pc2_poly_utm = getattr(self.pc2, 'poly_utm', None)
        pc1_poly_4326 = getattr(pc1_source, 'poly_4326', None)
        pc2_poly_4326 = getattr(self.pc2, 'poly_4326', None)
        epsg_utm = getattr(self.pc2, 'epsg_utm', None)

        return {
            'pc1_polygon': pc1_poly_utm,
            'pc2_polygon': pc2_poly_utm,
            'pc1_polygon_4326': pc1_poly_4326,
            'pc2_polygon_4326': pc2_poly_4326,
            'pc1_bounds': pc1_poly_utm.bounds if pc1_poly_utm else None,
            'pc2_bounds': pc2_poly_utm.bounds if pc2_poly_utm else None,
            'epsg_utm': epsg_utm,
        }

    def compute_overlap_polygon(
        self,
        use_transformed: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute the intersection polygon of pc1 and pc2 bounding areas (Area A).

        Both point clouds should be in the same CRS for meaningful results.
        Uses the UTM polygons computed during from_file().

        Parameters
        ----------
        use_transformed : bool, default True
            If True and pc1 has been transformed, use the transformed version.

        Returns
        -------
        dict
            {
                'overlap_polygon': shapely.Polygon or None,
                'overlap_bounds': tuple (minx, miny, maxx, maxy) or None,
                'overlap_area': float,
                'pc1_area': float,
                'pc2_area': float,
                'overlap_fraction_pc1': float,
                'overlap_fraction_pc2': float,
                'has_overlap': bool,
            }
        """
        polys = self.get_bounding_polygons(use_transformed=use_transformed)
        pc1_poly = polys['pc1_polygon']
        pc2_poly = polys['pc2_polygon']

        if pc1_poly is None or pc2_poly is None:
            return {
                'overlap_polygon': None,
                'overlap_bounds': None,
                'overlap_area': 0.0,
                'pc1_area': pc1_poly.area if pc1_poly else 0.0,
                'pc2_area': pc2_poly.area if pc2_poly else 0.0,
                'overlap_fraction_pc1': 0.0,
                'overlap_fraction_pc2': 0.0,
                'has_overlap': False,
            }

        overlap = pc1_poly.intersection(pc2_poly)

        if overlap.is_empty:
            return {
                'overlap_polygon': None,
                'overlap_bounds': None,
                'overlap_area': 0.0,
                'pc1_area': pc1_poly.area,
                'pc2_area': pc2_poly.area,
                'overlap_fraction_pc1': 0.0,
                'overlap_fraction_pc2': 0.0,
                'has_overlap': False,
            }

        return {
            'overlap_polygon': overlap,
            'overlap_bounds': overlap.bounds,
            'overlap_area': overlap.area,
            'pc1_area': pc1_poly.area,
            'pc2_area': pc2_poly.area,
            'overlap_fraction_pc1': overlap.area / pc1_poly.area if pc1_poly.area > 0 else 0,
            'overlap_fraction_pc2': overlap.area / pc2_poly.area if pc2_poly.area > 0 else 0,
            'has_overlap': True,
        }

    def compute_buffered_overlap_polygon(
        self,
        buffer_distance: float = 10.0,
        use_transformed: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute overlap polygon with buffer for alignment preparation (Area B).

        Area B = overlap polygon + buffer. This is used for the compare cloud
        during alignment to provide context beyond the strict overlap area.

        Parameters
        ----------
        buffer_distance : float, default 10.0
            Buffer distance in map units (typically meters).
        use_transformed : bool, default True
            If True and pc1 has been transformed, use the transformed version.

        Returns
        -------
        dict
            {
                'buffered_polygon': shapely.Polygon,
                'buffered_bounds': tuple,
                'buffered_area': float,
                'overlap_polygon': shapely.Polygon,
                'overlap_area': float,
                'buffer_distance': float,
            }
        """
        overlap_result = self.compute_overlap_polygon(use_transformed=use_transformed)

        if not overlap_result['has_overlap']:
            return {
                'buffered_polygon': None,
                'buffered_bounds': None,
                'buffered_area': 0.0,
                'overlap_polygon': None,
                'overlap_area': 0.0,
                'buffer_distance': buffer_distance,
            }

        overlap_poly = overlap_result['overlap_polygon']
        buffered_poly = overlap_poly.buffer(buffer_distance)

        return {
            'buffered_polygon': buffered_poly,
            'buffered_bounds': buffered_poly.bounds,
            'buffered_area': buffered_poly.area,
            'overlap_polygon': overlap_poly,
            'overlap_area': overlap_result['overlap_area'],
            'buffer_distance': buffer_distance,
        }

    def crop_to_overlap(
        self,
        output_dir: Optional[str] = None,
        use_transformed: bool = True,
        target_crs: Optional[Any] = None,
        interior_buffer: Optional[float] = None,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> Tuple[PointCloud, PointCloud]:
        """
        Crop both point clouds to their overlap area.

        The workflow:
        1. Get outline polygons from each point cloud (stored in WGS84 as poly_4326)
        2. Transform both polygons to a common CRS (pc2's CRS or target_crs)
        3. Compute intersection in that common CRS
        4. Optionally apply interior buffer to compare cloud's clip polygon
        5. Transform clip polygons to each point cloud's native CRS for clipping

        Parameters
        ----------
        output_dir : str, optional
            Output directory for cropped files. If None, uses same directory as inputs.
        use_transformed : bool, default True
            If True and pc1 has been transformed, use the transformed version.
        target_crs : CRS, optional
            Target CRS for computing intersection. If None, uses pc2's CRS.
        interior_buffer : float, optional
            If provided, shrinks the compare cloud's clip polygon inward by this
            distance (in meters). This ensures the compare cloud is smaller on all
            sides than the reference, which is useful for alignment. The reference
            cloud still uses the full intersection polygon.
        overwrite : bool, default True
            Whether to overwrite existing output files.
        verbose : bool, default True
            Print progress messages.

        Returns
        -------
        tuple[PointCloud, PointCloud]
            (pc1_cropped, pc2_cropped) - pc1 cropped to intersection (minus interior
            buffer if specified), pc2 cropped to full intersection.
        """
        import sys
        from pyproj import CRS as CRS_, Transformer
        from shapely.ops import transform as shapely_transform

        # Select pc1 source
        pc1_source = self._pc1_transformed if use_transformed and self._pc1_transformed else self.pc1

        # Get outline polygons in WGS84 (set during from_file())
        pc1_poly_4326 = getattr(pc1_source, 'poly_4326', None)
        pc2_poly_4326 = getattr(self.pc2, 'poly_4326', None)

        if pc1_poly_4326 is None or pc2_poly_4326 is None:
            raise ValueError("Point cloud outline polygons not available. Ensure from_file() was called.")

        # Get native CRS for each point cloud (for clipping)
        pc1_native_crs_wkt = (
            getattr(pc1_source, 'current_horizontal_crs', None) or
            getattr(pc1_source, 'original_horizontal_crs', None) or
            getattr(pc1_source, 'current_compound_crs', None) or
            getattr(pc1_source, 'original_compound_crs', None)
        )
        pc2_native_crs_wkt = (
            getattr(self.pc2, 'current_horizontal_crs', None) or
            getattr(self.pc2, 'original_horizontal_crs', None) or
            getattr(self.pc2, 'current_compound_crs', None) or
            getattr(self.pc2, 'original_compound_crs', None)
        )

        # Determine target CRS for intersection computation
        if target_crs is not None:
            common_crs = CRS_.from_user_input(target_crs)
        elif pc2_native_crs_wkt:
            common_crs = CRS_.from_user_input(pc2_native_crs_wkt)
        else:
            # Fallback to WGS84
            common_crs = CRS_.from_epsg(4326)

        wgs84 = CRS_.from_epsg(4326)

        # Transform both polygons from WGS84 to common CRS
        if not common_crs.equals(wgs84):
            transformer_to_common = Transformer.from_crs(wgs84, common_crs, always_xy=True)
            pc1_poly_common = shapely_transform(transformer_to_common.transform, pc1_poly_4326)
            pc2_poly_common = shapely_transform(transformer_to_common.transform, pc2_poly_4326)
        else:
            pc1_poly_common = pc1_poly_4326
            pc2_poly_common = pc2_poly_4326

        # Compute intersection in common CRS
        overlap_poly_common = pc1_poly_common.intersection(pc2_poly_common)

        if overlap_poly_common.is_empty:
            raise ValueError("Point clouds do not overlap. Cannot crop to overlap area.")

        overlap_area = overlap_poly_common.area
        pc1_overlap_frac = overlap_area / pc1_poly_common.area if pc1_poly_common.area > 0 else 0
        pc2_overlap_frac = overlap_area / pc2_poly_common.area if pc2_poly_common.area > 0 else 0

        # Apply interior buffer to compare cloud's clip polygon if requested
        # This shrinks the compare polygon inward so it's smaller than reference on all sides
        if interior_buffer is not None and interior_buffer > 0:
            # Negative buffer shrinks the polygon inward
            pc1_clip_poly_common = overlap_poly_common.buffer(-interior_buffer)
            if pc1_clip_poly_common.is_empty:
                raise ValueError(
                    f"Interior buffer of {interior_buffer}m resulted in empty polygon. "
                    "Try a smaller buffer value."
                )
            pc1_buffered_area = pc1_clip_poly_common.area
        else:
            pc1_clip_poly_common = overlap_poly_common
            pc1_buffered_area = None

        # Reference cloud always uses full intersection polygon
        pc2_clip_poly_common = overlap_poly_common

        if verbose:
            print(f"\n--- Cropping to Overlap Area ---", file=sys.stderr)
            print(f"PC1 poly_4326 bounds: {pc1_poly_4326.bounds}", file=sys.stderr)
            print(f"PC2 poly_4326 bounds: {pc2_poly_4326.bounds}", file=sys.stderr)
            print(f"Common CRS: {common_crs.to_epsg() or common_crs.name}", file=sys.stderr)
            print(f"PC1 native CRS: {pc1_native_crs_wkt[:80] if pc1_native_crs_wkt else 'None'}...", file=sys.stderr)
            print(f"PC2 native CRS: {pc2_native_crs_wkt[:80] if pc2_native_crs_wkt else 'None'}...", file=sys.stderr)
            print(f"Intersection area: {overlap_area:,.0f} m²", file=sys.stderr)
            print(f"Overlap fraction pc1: {pc1_overlap_frac:.1%}", file=sys.stderr)
            print(f"Overlap fraction pc2: {pc2_overlap_frac:.1%}", file=sys.stderr)
            if interior_buffer is not None and interior_buffer > 0:
                print(f"Interior buffer for compare: {interior_buffer} m", file=sys.stderr)
                print(f"Compare clip area (after buffer): {pc1_buffered_area:,.0f} m²", file=sys.stderr)

        # Transform clip polygons to each point cloud's native CRS for clipping
        if pc1_native_crs_wkt:
            pc1_native_crs = CRS_.from_user_input(pc1_native_crs_wkt)
            if not pc1_native_crs.equals(common_crs):
                transformer_to_pc1 = Transformer.from_crs(common_crs, pc1_native_crs, always_xy=True)
                clip_poly_pc1 = shapely_transform(transformer_to_pc1.transform, pc1_clip_poly_common)
            else:
                clip_poly_pc1 = pc1_clip_poly_common
        else:
            clip_poly_pc1 = pc1_clip_poly_common

        if pc2_native_crs_wkt:
            pc2_native_crs = CRS_.from_user_input(pc2_native_crs_wkt)
            if not pc2_native_crs.equals(common_crs):
                transformer_to_pc2 = Transformer.from_crs(common_crs, pc2_native_crs, always_xy=True)
                clip_poly_pc2 = shapely_transform(transformer_to_pc2.transform, pc2_clip_poly_common)
            else:
                clip_poly_pc2 = pc2_clip_poly_common
        else:
            clip_poly_pc2 = pc2_clip_poly_common

        # Determine output paths
        if output_dir is None:
            out_dir1 = Path(pc1_source.filename).parent
            out_dir2 = Path(self.pc2.filename).parent
        else:
            out_dir1 = out_dir2 = Path(output_dir)
            out_dir1.mkdir(parents=True, exist_ok=True)

        # Use different suffix if interior buffer applied
        if interior_buffer is not None and interior_buffer > 0:
            pc1_suffix = "_intersection_buffered"
        else:
            pc1_suffix = "_intersection"
        pc2_suffix = "_intersection"

        out_path1 = out_dir1 / (Path(pc1_source.filename).stem + pc1_suffix + Path(pc1_source.filename).suffix)
        out_path2 = out_dir2 / (Path(self.pc2.filename).stem + pc2_suffix + Path(self.pc2.filename).suffix)

        if verbose:
            print(f"PC1 bounds: ({pc1_source.minx:.2f}, {pc1_source.miny:.2f}) to ({pc1_source.maxx:.2f}, {pc1_source.maxy:.2f})", file=sys.stderr)
            print(f"PC1 clip polygon bounds: {clip_poly_pc1.bounds}", file=sys.stderr)
            print(f"PC1 CRS match common? {pc1_native_crs.equals(common_crs) if pc1_native_crs_wkt else 'N/A'}", file=sys.stderr)
            print(f"PC2 bounds: ({self.pc2.minx:.2f}, {self.pc2.miny:.2f}) to ({self.pc2.maxx:.2f}, {self.pc2.maxy:.2f})", file=sys.stderr)
            print(f"PC2 clip polygon bounds: {clip_poly_pc2.bounds}", file=sys.stderr)
            print(f"PC2 CRS match common? {pc2_native_crs.equals(common_crs) if pc2_native_crs_wkt else 'N/A'}", file=sys.stderr)
            print(f"Cropping pc1 to: {out_path1.name}", file=sys.stderr)

        pc1_cropped = pc1_source.clip_to_polygon(
            polygon=clip_poly_pc1,
            output_path=out_path1,
            overwrite=overwrite,
        )

        if verbose:
            print(f"Cropping pc2 to: {out_path2.name}", file=sys.stderr)

        pc2_cropped = self.pc2.clip_to_polygon(
            polygon=clip_poly_pc2,
            output_path=out_path2,
            overwrite=overwrite,
        )

        if verbose:
            print(f"Cropping complete.", file=sys.stderr)

        # Store cropped clouds internally for use by alignment
        self._pc1_cropped = pc1_cropped
        self._pc1_cropped_original = pc1_cropped  # Preserve unaligned version
        self._pc2_cropped = pc2_cropped

        return pc1_cropped, pc2_cropped

    def crop_compare_with_buffer(
        self,
        buffer_distance: float = 10.0,
        output_dir: Optional[str] = None,
        use_transformed: bool = True,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> PointCloud:
        """
        Crop compare point cloud (pc1) to overlap area plus buffer (Area B).

        Area B is used for the compare cloud during alignment to provide
        spatial context beyond the strict overlap region.

        Parameters
        ----------
        buffer_distance : float, default 10.0
            Buffer distance around overlap area in map units (typically meters).
        output_dir : str, optional
            Output directory for cropped file. If None, uses same directory as input.
        use_transformed : bool, default True
            If True and pc1 has been transformed, use the transformed version.
        overwrite : bool, default True
            Whether to overwrite existing output file.
        verbose : bool, default True
            Print progress messages.

        Returns
        -------
        PointCloud
            pc1 cropped to Area B (overlap + buffer).
        """
        import sys
        from pyproj import CRS as CRS_, Transformer
        from shapely.ops import transform as shapely_transform

        buffered_result = self.compute_buffered_overlap_polygon(
            buffer_distance=buffer_distance,
            use_transformed=use_transformed,
        )

        if buffered_result['buffered_polygon'] is None:
            raise ValueError("Point clouds do not overlap. Cannot create buffered crop.")

        buffered_poly_utm = buffered_result['buffered_polygon']
        epsg_utm = getattr(self.pc2, 'epsg_utm', None)

        if verbose:
            print(f"\n--- Cropping Compare Cloud with Buffer (Area B) ---", file=sys.stderr)
            print(f"Buffer distance: {buffer_distance} m", file=sys.stderr)
            print(f"Overlap area: {buffered_result['overlap_area']:,.0f} m²", file=sys.stderr)
            print(f"Buffered area: {buffered_result['buffered_area']:,.0f} m²", file=sys.stderr)

        # Select pc1 source
        pc1_source = self._pc1_transformed if use_transformed and self._pc1_transformed else self.pc1

        # Reproject polygon from UTM to point cloud's native CRS
        pc_crs_wkt = (
            getattr(pc1_source, 'current_horizontal_crs', None) or
            getattr(pc1_source, 'original_horizontal_crs', None) or
            getattr(pc1_source, 'current_compound_crs', None) or
            getattr(pc1_source, 'original_compound_crs', None)
        )

        buffered_poly = buffered_poly_utm  # Default to UTM polygon
        if pc_crs_wkt and epsg_utm:
            pc_crs = CRS_.from_user_input(pc_crs_wkt)
            utm_crs = CRS_.from_epsg(epsg_utm)
            if not pc_crs.equals(utm_crs):
                transformer = Transformer.from_crs(utm_crs, pc_crs, always_xy=True)
                buffered_poly = shapely_transform(transformer.transform, buffered_poly_utm)

        # Determine output path
        if output_dir is None:
            out_dir = Path(pc1_source.filename).parent
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / (Path(pc1_source.filename).stem + "_areaB" + Path(pc1_source.filename).suffix)

        if verbose:
            print(f"Cropping pc1 to: {out_path.name}", file=sys.stderr)

        pc1_buffered = pc1_source.clip_to_polygon(
            polygon=buffered_poly,
            output_path=out_path,
            overwrite=overwrite,
        )

        if verbose:
            print(f"Cropping complete.", file=sys.stderr)

        return pc1_buffered

    # =========================================================================
    # Transformation Methods
    # =========================================================================

    def transform_compare_to_match_reference(
        self,
        skip_epoch: bool = False,
        skip_horizontal: bool = False,
        skip_vertical: bool = False,
        skip_units: bool = False,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> PointCloud:
        """
        Transform pc1 (compare) to match pc2 (reference)'s reference frame.
        
        All transformations are composed into a SINGLE PDAL pipeline pass
        for efficiency.
        """
        import sys
        
        comparison = self.check_all_match()
        
        if verbose:
            print(f"\n--- Transform compare to match reference ---", file=sys.stderr)
            print(f"Transformations needed: {comparison['transformations_needed']}", file=sys.stderr)
        
        self._transformation_history = []
        
        # Get target parameters from reference (pc2)
        target_epoch = getattr(self.pc2, 'epoch', None)
        target_horiz_crs = (
            getattr(self.pc2, 'current_horizontal_crs', None) or
            getattr(self.pc2, 'original_horizontal_crs', None)
        )
        target_vert_crs = (
            getattr(self.pc2, 'current_vertical_crs', None) or
            getattr(self.pc2, 'original_vertical_crs', None)
        )
        target_geoid = getattr(self.pc2, 'geoid_model', None)
        
        # Determine vertical kinds
        source_is_ortho = getattr(self.pc1, 'is_orthometric', None)
        target_is_ortho = getattr(self.pc2, 'is_orthometric', None)
        
        source_vertical_kind = (
            "orthometric" if source_is_ortho else
            "ellipsoidal" if source_is_ortho is False else
            None
        )
        target_vertical_kind = (
            "orthometric" if target_is_ortho else
            "ellipsoidal" if target_is_ortho is False else
            None
        )
        source_geoid = getattr(self.pc1, 'geoid_model', None)
        
        # Determine what's actually needed
        needs_epoch = (
            not skip_epoch and 
            'epoch' in comparison['transformations_needed'] and
            target_epoch is not None
        )
        needs_vertical = (
            not skip_vertical and
            'vertical_datum' in comparison['transformations_needed']
        )
        needs_horizontal = (
            not skip_horizontal and
            'horizontal_crs' in comparison['transformations_needed']
        )
        
        if verbose:
            src_epoch = getattr(self.pc1, 'epoch', None)
            if needs_epoch:
                print(f"  Epoch: {src_epoch:.4f} -> {target_epoch:.4f}", file=sys.stderr)
            if needs_vertical:
                print(f"  Vertical: {source_vertical_kind} -> {target_vertical_kind}", file=sys.stderr)
                print(f"  Geoid: {source_geoid} -> {target_geoid}", file=sys.stderr)
            if needs_horizontal:
                print(f"  Horizontal CRS reprojection needed", file=sys.stderr)

        # SINGLE warp_pointcloud call with ALL parameters
        if needs_epoch or needs_vertical or needs_horizontal:
            
            # Build output filename
            src_path = Path(self.pc1.filename)
            suffix_parts = []
            if needs_epoch:
                suffix_parts.append(f"epoch{target_epoch:.2f}".replace(".", "p"))
            if needs_vertical:
                suffix_parts.append(f"{target_vertical_kind[:4]}")
            if needs_horizontal:
                suffix_parts.append("reproj")
            suffix = "_".join(suffix_parts) if suffix_parts else "transformed"
            output_path = src_path.with_name(src_path.stem + f"_{suffix}" + src_path.suffix)
            
            current = self.pc1.warp_pointcloud(
                # Epoch parameters
                dynamic_target_epoch=target_epoch if needs_epoch else None,
                # Vertical parameters  
                source_vertical_kind=source_vertical_kind if needs_vertical else None,
                target_vertical_kind=target_vertical_kind if needs_vertical else None,
                source_geoid_model=source_geoid if needs_vertical else None,
                target_geoid_model=target_geoid if needs_vertical else None,
                # Horizontal parameters
                target_horizontal_crs=target_horiz_crs if needs_horizontal else None,
                # Output
                output_path=output_path,
                overwrite=overwrite,
            )
            
            self._transformation_history.append({
                'step': 'combined_transform',
                'needs_epoch': needs_epoch,
                'needs_vertical': needs_vertical,
                'needs_horizontal': needs_horizontal,
                'source_epoch': getattr(self.pc1, 'epoch', None),
                'target_epoch': target_epoch,
                'source_vertical_kind': source_vertical_kind,
                'target_vertical_kind': target_vertical_kind,
                'output_file': current.filename,
            })
            
            if verbose:
                print(f"  Combined transformation [done]", file=sys.stderr)
        else:
            current = self.pc1
            if verbose:
                print(f"No transformations needed.", file=sys.stderr)
        
        # Update metadata to match reference (only update fields that were transformed)
        current.add_metadata(
            horizontal_CRS=target_horiz_crs if needs_horizontal else None,
            vertical_CRS=target_vert_crs if needs_vertical else None,
            geoid_model=target_geoid if needs_vertical else None,
            epoch=target_epoch if needs_epoch else None,
        )
        
        # Cache result
        self._pc1_transformed = current
        
        if verbose:
            print(f"\nOutput: {current.filename}", file=sys.stderr)
        
        return current
    
    def warp_pointclouds_to_common_crs(
        self,
        target_pc: str = "pc2",
        target_crs_proj: Optional[Any] = None,
        verbose: bool = True,
    ) -> "PointCloudPair":
        """
        Warp point clouds to a common reference frame.
        
        By default, transforms pc1 to match pc2's reference frame.
        
        Parameters
        ----------
        target_pc : str, {"pc1", "pc2"}
            Which point cloud to use as the reference
            - "pc1": Transform pc2 to match pc1
            - "pc2": Transform pc1 to match pc2 (default)
        target_crs_proj : Any, optional
            If provided, warp both to this CRS (horizontal only)
        verbose : bool
            Print progress messages
            
        Returns
        -------
        PointCloudPair
            New PointCloudPair with transformed point clouds
        """
        if target_crs_proj is not None:
            # Warp both to explicit target CRS (horizontal only)
            warped_pc1 = self.pc1.warp_pointcloud(target_horizontal_crs=target_crs_proj)
            warped_pc2 = self.pc2.warp_pointcloud(target_horizontal_crs=target_crs_proj)
            return PointCloudPair(warped_pc1, warped_pc2)
        
        if target_pc == "pc2":
            # Transform pc1 to match pc2 (reference)
            transformed_pc1 = self.transform_compare_to_match_reference(verbose=verbose)
            return PointCloudPair(transformed_pc1, self.pc2)
        
        elif target_pc == "pc1":
            # Swap and transform pc2 to match pc1
            swapped_pair = PointCloudPair(self.pc2, self.pc1)
            transformed_pc2 = swapped_pair.transform_compare_to_match_reference(verbose=verbose)
            return PointCloudPair(self.pc1, transformed_pc2)
        
        else:
            raise ValueError(f"target_pc must be 'pc1' or 'pc2', got {target_pc!r}.")
    
    # =========================================================================
    # Alignment Methods (ICP via small_gicp)
    # =========================================================================
    
    def align_point_clouds(
        self,
        method: str = "vgicp",
        downsample_resolution: Optional[float] = None,
        max_correspondence_distance: float = 1.0,
        max_iterations: int = 50,
        transformation_epsilon: float = 1e-6,
        num_threads: int = 4,
        apply_transform: bool = True,
        output_path: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
        initial_voxel_size: Optional[float] = None,
        max_points: Optional[int] = 2_000_000,
        auto_downsample: bool = True,
        target_points: int = 2_000_000,
        source_cloud: Optional[PointCloud] = None,
        target_cloud: Optional[PointCloud] = None,
        alignment_buffer: float = 10.0,
        use_cropped_clouds: bool = False,
    ) -> Dict[str, Any]:
        """
        Align pc1 to pc2 using ICP registration via small_gicp.

        Point clouds are automatically centered before registration to avoid
        numerical issues with large UTM coordinates.

        Parameters
        ----------
        method : str
            Registration method: 'gicp', 'vgicp', 'icp', or 'plane_icp'.
            Default is 'vgicp' for best accuracy.
        downsample_resolution : float, optional
            Voxel size for small_gicp preprocessing. If None and auto_downsample
            is True, automatically calculated based on point density.
        max_correspondence_distance : float
            Maximum distance for point correspondences (default 1.0m)
        max_iterations : int
            Maximum ICP iterations
        transformation_epsilon : float
            Convergence threshold
        num_threads : int
            Number of threads for parallel processing
        apply_transform : bool
            Whether to apply the transform and update internal state
        output_path : str, optional
            Path to save aligned point cloud
        overwrite : bool
            Whether to overwrite existing output file
        verbose : bool
            Print progress information
        initial_voxel_size : float, optional
            Voxel size for initial downsampling during point cloud loading.
            If None, uses downsample_resolution * 2 for memory efficiency.
            Set to 0 to disable initial downsampling (loads all points).
        max_points : int, optional
            Maximum points to load per cloud (default 2M). Applied after
            voxel downsampling. Set to None for no limit.
        auto_downsample : bool, default True
            If True and downsample_resolution is None, automatically calculate
            optimal voxel size to achieve approximately target_points points.
        target_points : int, default 2_000_000
            Target number of points for auto-downsampling. Uses minimum
            downsampling needed to stay below this threshold.
        source_cloud : PointCloud, optional
            Custom source (compare) point cloud to use for alignment.
            If None, uses pc1 (or transformed pc1 if available).
        target_cloud : PointCloud, optional
            Custom target (reference) point cloud to use for alignment.
            If None, uses pc2.
        alignment_buffer : float, default 10.0
            Buffer distance (meters) for Area B when use_cropped_clouds=True.
        use_cropped_clouds : bool, default False
            If True, automatically crop clouds before alignment:
            - Compare cloud (pc1) cropped to Area B (overlap + buffer)
            - Reference cloud (pc2) cropped to Area A (overlap only)
        """
        if not _has_small_gicp():
            raise ImportError(
                "small_gicp is required for point cloud alignment. "
                "Install with: pip install small_gicp"
            )

        import small_gicp
        import sys

        if verbose:
            print(f"\n--- Point Cloud Alignment (small_gicp) ---", file=sys.stderr)
            print(f"Method: {method.upper()}", file=sys.stderr)

        # =========================================================================
        # Determine source and target point clouds
        # =========================================================================
        # Priority: custom clouds > stored cropped clouds > use_cropped_clouds > default
        if source_cloud is not None:
            source_pc = source_cloud
        elif self._pc1_cropped is not None:
            # Use previously cropped cloud (from crop_to_overlap with interior_buffer)
            source_pc = self._pc1_cropped
            if verbose:
                print(f"Using stored cropped compare cloud: {source_pc.filename}", file=sys.stderr)
        elif use_cropped_clouds:
            # Crop compare cloud to Area B (overlap + buffer)
            if verbose:
                print(f"Cropping compare cloud to Area B (buffer={alignment_buffer}m)...",
                      file=sys.stderr)
            source_pc = self.crop_compare_with_buffer(
                buffer_distance=alignment_buffer,
                use_transformed=True,
                verbose=False,
            )
        else:
            # Use transformed pc1 if available, otherwise original
            source_pc = self._pc1_transformed or self.pc1

        if target_cloud is not None:
            target_pc = target_cloud
        elif self._pc2_cropped is not None:
            # Use previously cropped cloud (from crop_to_overlap)
            target_pc = self._pc2_cropped
            if verbose:
                print(f"Using stored cropped reference cloud: {target_pc.filename}", file=sys.stderr)
        elif use_cropped_clouds:
            # Crop reference cloud to Area A (overlap only)
            if verbose:
                print(f"Cropping reference cloud to Area A...", file=sys.stderr)
            _, target_pc = self.crop_to_overlap(
                use_transformed=True,
                verbose=False,
            )
        else:
            target_pc = self.pc2

        # =========================================================================
        # Auto-downsampling: calculate optimal voxel size
        # =========================================================================
        if downsample_resolution is None:
            if auto_downsample:
                # Get point counts from metadata
                source_count = getattr(source_pc, 'point_count', None)
                target_count = getattr(target_pc, 'point_count', None)

                if source_count is None or target_count is None:
                    # Estimate from file if not available
                    if verbose:
                        print("Estimating point counts...", file=sys.stderr)
                    import laspy
                    if source_count is None:
                        with laspy.open(source_pc.filename) as f:
                            source_count = f.header.point_count
                    if target_count is None:
                        with laspy.open(target_pc.filename) as f:
                            target_count = f.header.point_count

                total_points = source_count + target_count

                if total_points > target_points:
                    # Calculate voxel size needed to achieve target point count
                    # Approximate: points_after = points_before * (orig_density / new_density)
                    # For uniform distribution: new_density ~ 1/voxel_size^2 (2D) or ^3 (3D)
                    # Use conservative 2D estimate since lidar is mostly planar
                    reduction_factor = total_points / target_points

                    # Get area from overlap polygon for density estimation
                    overlap_info = self.compute_overlap_polygon(use_transformed=True)
                    if overlap_info['has_overlap']:
                        area = overlap_info['overlap_area']
                        # Current point density (points per m²)
                        current_density = total_points / area if area > 0 else 1.0
                        # Target density
                        target_density = target_points / area if area > 0 else 1.0
                        # Voxel size ~ sqrt(1/target_density)
                        downsample_resolution = max(0.1, np.sqrt(1.0 / target_density))
                    else:
                        # Fallback: simple scaling based on reduction factor
                        downsample_resolution = max(0.1, 0.5 * np.sqrt(reduction_factor))

                    if verbose:
                        print(f"Auto-downsample: {total_points:,} points -> "
                              f"target {target_points:,}", file=sys.stderr)
                        print(f"Calculated voxel size: {downsample_resolution:.2f} m",
                              file=sys.stderr)
                else:
                    # No downsampling needed
                    downsample_resolution = 0.25  # Minimal for preprocessing
                    if verbose:
                        print(f"Point count ({total_points:,}) below target "
                              f"({target_points:,}), minimal downsampling",
                              file=sys.stderr)
            else:
                # Default if auto_downsample is False and no resolution provided
                downsample_resolution = 0.5

        if verbose:
            print(f"Downsample resolution: {downsample_resolution} m", file=sys.stderr)
            print(f"Max correspondence distance: {max_correspondence_distance} m",
                  file=sys.stderr)

        # Determine initial voxel size for memory-efficient loading
        # Default to 2x the downsample resolution (will be further downsampled)
        if initial_voxel_size is None:
            load_voxel_size = downsample_resolution * 2.0
        elif initial_voxel_size == 0:
            load_voxel_size = None  # Disable voxel downsampling, load all points
        else:
            load_voxel_size = initial_voxel_size

        if verbose:
            if load_voxel_size:
                print(f"Initial voxel downsampling: {load_voxel_size} m", file=sys.stderr)
            if max_points:
                print(f"Max points per cloud: {max_points:,}", file=sys.stderr)

        # Load points with optional downsampling for memory efficiency
        if verbose:
            print(f"\nLoading source points from: {Path(source_pc.filename).name}",
                  file=sys.stderr)
        source_points = _load_points_from_las(
            source_pc.filename,
            max_points=max_points,
            voxel_size=load_voxel_size,
        )

        if verbose:
            print(f"Loading target points from: {Path(target_pc.filename).name}",
                  file=sys.stderr)
        target_points_arr = _load_points_from_las(
            target_pc.filename,
            max_points=max_points,
            voxel_size=load_voxel_size,
        )

        if verbose:
            print(f"Source points: {len(source_points):,}", file=sys.stderr)
            print(f"Target points: {len(target_points_arr):,}", file=sys.stderr)

        # =========================================================================
        # CENTER POINT CLOUDS to avoid voxel coordinate overflow
        # =========================================================================
        # Compute centroid WITHOUT concatenating arrays (memory efficient)
        # Combined centroid = weighted average of individual centroids
        n_source = len(source_points)
        n_target = len(target_points_arr)
        n_total = n_source + n_target

        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points_arr, axis=0)
        centroid = (source_centroid * n_source + target_centroid * n_target) / n_total

        if verbose:
            print(f"\nCentering point clouds (centroid: "
                  f"[{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}])",
                  file=sys.stderr)

        # Center both point clouds (in-place to save memory)
        source_points -= centroid
        target_points_arr -= centroid

        if verbose:
            src_range = np.ptp(source_points, axis=0)
            tgt_range = np.ptp(target_points_arr, axis=0)
            src_min = np.min(source_points, axis=0)
            src_max = np.max(source_points, axis=0)
            tgt_min = np.min(target_points_arr, axis=0)
            tgt_max = np.max(target_points_arr, axis=0)
            print(f"Source range after centering: "
                  f"X={src_range[0]:.1f}, Y={src_range[1]:.1f}, Z={src_range[2]:.1f}",
                  file=sys.stderr)
            print(f"Target range after centering: "
                  f"X={tgt_range[0]:.1f}, Y={tgt_range[1]:.1f}, Z={tgt_range[2]:.1f}",
                  file=sys.stderr)
            # Check overlap
            overlap_min = np.maximum(src_min, tgt_min)
            overlap_max = np.minimum(src_max, tgt_max)
            overlap_size = np.maximum(overlap_max - overlap_min, 0)
            print(f"XY overlap region: {overlap_size[0]:.1f} x {overlap_size[1]:.1f} m",
                  file=sys.stderr)
            if overlap_size[0] <= 0 or overlap_size[1] <= 0:
                print("WARNING: Point clouds do not overlap in XY!", file=sys.stderr)

        # Select registration type
        method_upper = method.upper()
        if method_upper not in ["GICP", "VGICP", "ICP", "PLANE_ICP"]:
            raise ValueError(
                f"Unknown method: {method}. Use 'gicp', 'vgicp', 'icp', or 'plane_icp'."
            )

        if verbose:
            print(f"\nPoint data check:", file=sys.stderr)
            print(f"  Source dtype: {source_points.dtype}, shape: {source_points.shape}",
                  file=sys.stderr)
            print(f"  Target dtype: {target_points_arr.dtype}, "
                  f"shape: {target_points_arr.shape}", file=sys.stderr)
            print(f"  Source sample: {source_points[:3]}", file=sys.stderr)
            print(f"  Target sample: {target_points_arr[:3]}", file=sys.stderr)

        # Run registration
        if verbose:
            print(f"\nRunning {method_upper} registration...", file=sys.stderr)
            print(f"  downsample_resolution: {downsample_resolution}", file=sys.stderr)
            print(f"  max_correspondence_distance: {max_correspondence_distance}",
                  file=sys.stderr)
            print(f"  max_iterations: {max_iterations}", file=sys.stderr)

        # Provide identity as explicit initial guess
        init_T = np.eye(4)

        # IMPORTANT: The raw numpy array API in small_gicp only supports ICP, PLANE_ICP,
        # and GICP. For VGICP, we must use preprocess_points() first to estimate
        # covariances, then align. For GICP and PLANE_ICP, we also need preprocessing
        # for normals/covariances. Only plain ICP works directly with raw arrays.

        if method_upper in ["VGICP", "GICP", "PLANE_ICP"]:
            # Preprocess point clouds: downsampling, normal/covariance estimation, KdTree
            if verbose:
                print("  Preprocessing point clouds (downsampling + covariance)...",
                      file=sys.stderr)

            target_cloud_gicp, target_tree = small_gicp.preprocess_points(
                target_points_arr,
                downsampling_resolution=downsample_resolution,
                num_threads=num_threads,
            )
            source_cloud_gicp, source_tree = small_gicp.preprocess_points(
                source_points,
                downsampling_resolution=downsample_resolution,
                num_threads=num_threads,
            )

            if verbose:
                print(f"  After preprocessing: {source_cloud_gicp.size()} source, "
                      f"{target_cloud_gicp.size()} target points", file=sys.stderr)

            # Free raw numpy arrays now that we have preprocessed clouds
            del source_points, target_points_arr
            import gc
            gc.collect()

            # When using preprocessed PointCloud objects with covariances, use GICP.
            # PLANE_ICP uses only normals, GICP uses full covariance matrices.
            # VGICP is essentially GICP with preprocessing - the "V" refers to
            # voxelization which we've already handled via preprocess_points().
            reg_type = "GICP" if method_upper in ["VGICP", "GICP"] else "PLANE_ICP"

            result = small_gicp.align(
                target_cloud_gicp,
                source_cloud_gicp,
                target_tree,
                init_T_target_source=init_T,
                registration_type=reg_type,
                max_correspondence_distance=max_correspondence_distance,
                max_iterations=max_iterations,
                num_threads=num_threads,
                verbose=verbose,
            )

            # Clean up preprocessed clouds
            del target_cloud_gicp, target_tree, source_cloud_gicp, source_tree
            gc.collect()
        else:
            # Plain ICP can use raw numpy arrays with the simple API
            # Note: The raw numpy API actually supports ICP, PLANE_ICP, and GICP,
            # but for consistency and to ensure proper covariance estimation,
            # we use preprocessing for GICP/PLANE_ICP above.
            result = small_gicp.align(
                target_points_arr,
                source_points,
                init_T_target_source=init_T,
                registration_type="ICP",
                downsampling_resolution=downsample_resolution,
                max_correspondence_distance=max_correspondence_distance,
                max_iterations=max_iterations,
                num_threads=num_threads,
                verbose=verbose,
            )

            # Free numpy arrays after align() has processed them
            del source_points, target_points_arr
            import gc
            gc.collect()

        if verbose:
            # Debug: check result attributes
            print(f"\nRegistration result attributes: {dir(result)}", file=sys.stderr)
            if hasattr(result, 'iterations'):
                print(f"  Iterations performed: {result.iterations}", file=sys.stderr)
            if hasattr(result, 'converged'):
                print(f"  Converged: {result.converged}", file=sys.stderr)
            if hasattr(result, 'error'):
                print(f"  Final error: {result.error}", file=sys.stderr)
            if hasattr(result, 'num_inliers'):
                print(f"  Num inliers: {result.num_inliers}", file=sys.stderr)
            if hasattr(result, 'H'):
                print(f"  Hessian (H) shape: "
                      f"{np.array(result.H).shape if result.H is not None else 'None'}",
                      file=sys.stderr)

        # =========================================================================
        # Extract transformation result
        # =========================================================================
        # T_centered is the transformation computed in centered coordinates.
        # We apply it by: center points -> apply T_centered -> uncenter points
        # This is handled in _save_transformed_las with the centroid parameter.
        T_centered = result.T_target_source  # 4x4 transformation in centered coords

        if verbose:
            print(f"\nTransformation (in centered coordinates):", file=sys.stderr)
            print(T_centered, file=sys.stderr)

        # Use metrics from small_gicp result
        # Note: num_inliers is based on the downsampled point count
        num_inliers = result.num_inliers if hasattr(result, 'num_inliers') else 0
        # Estimate total points (we don't have exact count after internal downsampling)
        # Use error as a proxy for quality
        final_error = result.error if hasattr(result, 'error') else float('inf')

        # Fitness is approximate since we don't know exact point counts after internal downsampling
        # If we had n_source points and num_inliers correspondences, fitness ~ num_inliers / n_source
        # For now, just report the raw inlier count
        fitness = num_inliers / n_source if n_source > 0 else 0.0
        # small_gicp error is the sum of squared errors, convert to RMSE
        rmse = np.sqrt(final_error / num_inliers) if num_inliers > 0 else float('inf')

        # Extract rotation and translation from T_centered (the actual applied transformation)
        R = T_centered[:3, :3]
        t = T_centered[:3, 3]

        # Compute rotation angle (magnitude of axis-angle representation)
        rotation_angle_rad = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        rotation_angle_deg = np.degrees(rotation_angle_rad)

        alignment_result = {
            'transformation': T_centered,  # The actual transformation applied
            'transformation_centered': T_centered,  # Keep for backwards compatibility
            'centroid': centroid,
            'converged': result.converged if hasattr(result, 'converged') else True,
            'iterations': result.iterations if hasattr(result, 'iterations') else None,
            'fitness': fitness,
            'rmse': rmse,
            'num_correspondences': num_inliers,
            'method': method,
            'downsample_resolution': downsample_resolution,
            'max_correspondence_distance': max_correspondence_distance,
            'translation': t,
            'rotation_angle_deg': rotation_angle_deg,
        }

        if verbose:
            print(f"\nAlignment Results:", file=sys.stderr)
            print(f"  Converged: {alignment_result['converged']}", file=sys.stderr)
            print(f"  Fitness (inlier ratio): {fitness:.4f}", file=sys.stderr)
            print(f"  RMSE: {rmse:.4f} m", file=sys.stderr)
            print(f"  Inlier correspondences: {alignment_result['num_correspondences']:,}",
                  file=sys.stderr)
            print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m", file=sys.stderr)
            print(f"  Rotation: {rotation_angle_deg:.4f}°", file=sys.stderr)

        # Apply transformation if requested
        if apply_transform:
            if output_path is None:
                src_path = Path(source_pc.filename)
                output_path = str(
                    src_path.with_name(src_path.stem + "_aligned" + src_path.suffix)
                )

            if os.path.exists(output_path) and not overwrite:
                raise FileExistsError(
                    f"Output file exists and overwrite=False: {output_path}"
                )

            if verbose:
                print(f"\nApplying transformation to: {output_path}", file=sys.stderr)

            # Use T_centered with centroid for correct transformation application
            # The transformation was computed on centered data, so we must:
            # 1. Center the points (subtract centroid)
            # 2. Apply T_centered
            # 3. Uncenter (add centroid back)
            _save_transformed_las(source_pc.filename, output_path, T_centered, centroid)
            
            # Load aligned point cloud
            aligned_pc = PointCloud(output_path)
            aligned_pc.from_file()

            # Copy metadata from source
            aligned_pc.add_metadata(
                compound_CRS=source_pc.current_compound_crs or source_pc.original_compound_crs,
                horizontal_CRS=source_pc.current_horizontal_crs or source_pc.original_horizontal_crs,
                vertical_CRS=source_pc.current_vertical_crs or source_pc.original_vertical_crs,
                geoid_model=source_pc.geoid_model,
                epoch=source_pc.epoch,
            )

            # Copy unit information from source
            aligned_pc.horizontal_unit = source_pc.horizontal_unit
            aligned_pc.vertical_unit = source_pc.vertical_unit
            aligned_pc.horizontal_units = source_pc.horizontal_units
            aligned_pc.vertical_units = source_pc.vertical_units
            
            alignment_result['aligned_pc'] = aligned_pc
            alignment_result['output_file'] = output_path

            # Update internal state
            # Update _pc1_cropped so iterative alignment passes use the aligned cloud
            # Note: _pc1_cropped_original preserves the unaligned cropped version
            if self._pc1_cropped is not None:
                self._pc1_cropped = aligned_pc
            self._pc1_transformed = aligned_pc

            self._transformation_history.append({
                'step': 'icp_alignment',
                'method': method,
                'fitness': fitness,
                'rmse': rmse,
                'translation': t.tolist(),
                'rotation_deg': rotation_angle_deg,
                'output_file': output_path,
            })
        
        self._alignment_result = alignment_result

        if verbose:
            print(f"\n{'=' * 60}\n", file=sys.stderr)

        return alignment_result

    def compute_alignment_quality(
        self,
        max_distance: float = 1.0,
        sample_size: Optional[int] = 100000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute detailed goodness-of-fit metrics after alignment.

        Compares the aligned compare cloud against the reference cloud to
        assess alignment quality through various statistical measures.

        Parameters
        ----------
        max_distance : float, default 1.0
            Maximum distance (meters) to consider for correspondences.
            Points farther than this are considered outliers.
        sample_size : int, optional, default 100000
            Number of points to sample for quality assessment.
            Set to None to use all points (slower for large clouds).
        verbose : bool, default True
            Print quality metrics.

        Returns
        -------
        dict
            {
                'rmse': float - Root mean square error of distances,
                'mae': float - Mean absolute error,
                'median_distance': float - Median point-to-point distance,
                'std_distance': float - Standard deviation of distances,
                'nmad': float - Normalized median absolute deviation,
                'inlier_ratio': float - Fraction of points within max_distance,
                'inlier_count': int - Number of inlier correspondences,
                'total_count': int - Total points compared,
                'percentiles': dict - Distance percentiles (5, 25, 50, 75, 95),
                'max_observed': float - Maximum observed distance,
            }

        Raises
        ------
        ValueError
            If alignment has not been performed yet.
        """
        import sys
        from scipy.spatial import cKDTree

        if self._pc1_transformed is None and self._alignment_result is None:
            raise ValueError(
                "Alignment has not been performed. Call align_point_clouds() first."
            )

        if verbose:
            print(f"\n--- Computing Alignment Quality ---", file=sys.stderr)

        # Get the aligned source and target clouds
        aligned_pc = self._pc1_transformed or self.pc1
        target_pc = self.pc2

        # Load points (with optional sampling for speed)
        if verbose:
            print(f"Loading points for quality assessment...", file=sys.stderr)

        source_pts = _load_points_from_las(
            aligned_pc.filename,
            max_points=sample_size,
            voxel_size=None,
        )
        target_pts = _load_points_from_las(
            target_pc.filename,
            max_points=sample_size * 2 if sample_size else None,  # More target for matching
            voxel_size=None,
        )

        if verbose:
            print(f"  Aligned cloud: {len(source_pts):,} points", file=sys.stderr)
            print(f"  Reference cloud: {len(target_pts):,} points", file=sys.stderr)

        # Build KD-tree on target (reference) cloud
        if verbose:
            print(f"Building KD-tree...", file=sys.stderr)
        tree = cKDTree(target_pts)

        # Query nearest neighbors
        if verbose:
            print(f"Computing nearest neighbor distances...", file=sys.stderr)
        distances, _ = tree.query(source_pts, k=1, workers=-1)

        # Compute statistics
        inlier_mask = distances <= max_distance
        inlier_distances = distances[inlier_mask]
        inlier_count = int(np.sum(inlier_mask))
        total_count = len(distances)
        inlier_ratio = inlier_count / total_count if total_count > 0 else 0.0

        # Core metrics (on inliers only for meaningful statistics)
        if inlier_count > 0:
            rmse = float(np.sqrt(np.mean(inlier_distances ** 2)))
            mae = float(np.mean(inlier_distances))
            median_dist = float(np.median(inlier_distances))
            std_dist = float(np.std(inlier_distances))
            nmad = float(1.4826 * np.median(np.abs(inlier_distances - median_dist)))
            percentiles = np.percentile(inlier_distances, [5, 25, 50, 75, 95])
        else:
            rmse = mae = median_dist = std_dist = nmad = float('nan')
            percentiles = [float('nan')] * 5

        quality_result = {
            'rmse': rmse,
            'mae': mae,
            'median_distance': median_dist,
            'std_distance': std_dist,
            'nmad': nmad,
            'inlier_ratio': inlier_ratio,
            'inlier_count': inlier_count,
            'total_count': total_count,
            'percentiles': {
                'p5': float(percentiles[0]),
                'p25': float(percentiles[1]),
                'p50': float(percentiles[2]),
                'p75': float(percentiles[3]),
                'p95': float(percentiles[4]),
            },
            'max_observed': float(np.max(distances)),
            'max_distance_threshold': max_distance,
        }

        if verbose:
            print(f"\nAlignment Quality Metrics:", file=sys.stderr)
            print(f"  RMSE: {rmse:.4f} m", file=sys.stderr)
            print(f"  MAE: {mae:.4f} m", file=sys.stderr)
            print(f"  Median distance: {median_dist:.4f} m", file=sys.stderr)
            print(f"  NMAD: {nmad:.4f} m", file=sys.stderr)
            print(f"  Inlier ratio: {inlier_ratio:.1%} "
                  f"({inlier_count:,}/{total_count:,})", file=sys.stderr)
            print(f"  95th percentile: {percentiles[4]:.4f} m", file=sys.stderr)

        return quality_result

    # =========================================================================
    # DEM Creation Methods
    # =========================================================================
    
    def create_dem_pair(
        self,
        dem_type: str = "dtm",
        resolution: float = 1.0,
        interpolation: str = "idw",
        use_transformed: bool = True,
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
        classifications_pc1: Optional[Union[str, List[int], Set[int]]] = "auto",
        classifications_pc2: Optional[Union[str, List[int], Set[int]]] = "auto",
        **dem_kwargs,
    ) -> Tuple[Raster, Raster]:
        """
        Create DEMs from both point clouds.

        Parameters
        ----------
        dem_type : str, {"dtm", "dsm"}
            Type of DEM to create. Used when classifications are "auto".
        resolution : float
            Output resolution in map units (typically meters)
        interpolation : str
            Interpolation method ("tin", "idw", "min", "max", "mean", etc.)
        use_transformed : bool
            If True, use the transformed pc1 (if available)
        output_dir : str, optional
            Directory for output files (default: same as input files)
        overwrite : bool
            Overwrite existing output files
        verbose : bool
            Print progress messages
        classifications_pc1 : str, list, or set, default "auto"
            Classification filter for pc1 (compare cloud):
            - "auto": Use dem_type to determine (ground for DTM, first returns for DSM)
            - list/set of ints: Specific classification codes (e.g., [2] for ground)
            - None: No classification filtering (use all points)
        classifications_pc2 : str, list, or set, default "auto"
            Classification filter for pc2 (reference cloud). Same options as pc1.
        **dem_kwargs
            Additional arguments passed to PointCloud.create_dem()

        Returns
        -------
        tuple[Raster, Raster]
            (dem1, dem2) - DEMs created from pc1 and pc2
        """
        import sys

        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"Creating {dem_type.upper()} pair", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)
            print(f"Resolution: {resolution} m", file=sys.stderr)
            print(f"Interpolation: {interpolation}", file=sys.stderr)

        # Select source for pc1
        pc1_source = self._pc1_transformed if use_transformed and self._pc1_transformed else self.pc1

        # Determine output paths
        if output_dir is None:
            dir1 = Path(pc1_source.filename).parent
            dir2 = Path(self.pc2.filename).parent
        else:
            dir1 = dir2 = Path(output_dir)
            dir1.mkdir(parents=True, exist_ok=True)

        out1 = dir1 / f"{Path(pc1_source.filename).stem}_{dem_type}_{int(resolution)}m.tif"
        out2 = dir2 / f"{Path(self.pc2.filename).stem}_{dem_type}_{int(resolution)}m.tif"

        # Prepare classification filters
        # If "auto", pass dem_type and let create_dem handle it
        # Otherwise pass the explicit classification list
        dem1_kwargs = dict(dem_kwargs)
        dem2_kwargs = dict(dem_kwargs)

        if classifications_pc1 != "auto":
            dem1_kwargs["classification_filter"] = classifications_pc1
        if classifications_pc2 != "auto":
            dem2_kwargs["classification_filter"] = classifications_pc2

        if verbose:
            cls1_str = classifications_pc1 if classifications_pc1 != "auto" else f"auto ({dem_type})"
            cls2_str = classifications_pc2 if classifications_pc2 != "auto" else f"auto ({dem_type})"
            print(f"PC1 classifications: {cls1_str}", file=sys.stderr)
            print(f"PC2 classifications: {cls2_str}", file=sys.stderr)
            print(f"\nCreating DEM from pc1: {Path(pc1_source.filename).name}", file=sys.stderr)

        dem1 = pc1_source.create_dem(
            output_path=str(out1),
            dem_type=dem_type,
            resolution=resolution,
            interpolation=interpolation,
            **dem1_kwargs,
        )

        if verbose:
            print(f"Creating DEM from pc2: {Path(self.pc2.filename).name}", file=sys.stderr)

        dem2 = self.pc2.create_dem(
            output_path=str(out2),
            dem_type=dem_type,
            resolution=resolution,
            interpolation=interpolation,
            **dem2_kwargs,
        )
        
        # Copy epoch and CRS info to DEMs
        dem1.epoch = getattr(pc1_source, 'epoch', None)
        dem2.epoch = getattr(self.pc2, 'epoch', None)

        if verbose:
            print(f"\nDEM1: {out1}", file=sys.stderr)
            print(f"DEM2: {out2}", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)

        return dem1, dem2
    
    def create_dtm_pair(self, **kwargs) -> Tuple[Raster, Raster]:
        """Create DTM (bare earth) pair. Convenience wrapper for create_dem_pair()."""
        return self.create_dem_pair(dem_type="dtm", **kwargs)
    
    def create_dsm_pair(self, **kwargs) -> Tuple[Raster, Raster]:
        """Create DSM (surface) pair. Convenience wrapper for create_dem_pair()."""
        return self.create_dem_pair(dem_type="dsm", **kwargs)
    
    # =========================================================================
    # 3D Differencing Methods
    # =========================================================================

    def compute_3d_difference(
        self,
        max_distance: float = 1.0,
        use_transformed: bool = True,
        max_points: Optional[int] = 5_000_000,
        voxel_size: Optional[float] = None,
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute point-to-point 3D differences between point clouds.

        Both clouds are automatically cropped to their overlap area (Area A)
        before computing differences. For each point in pc1, finds the nearest
        point in pc2 and computes the signed vertical (Z) difference.

        Parameters
        ----------
        max_distance : float, default 1.0
            Maximum 3D distance for valid correspondences (meters).
            Points farther than this are considered outliers.
        use_transformed : bool, default True
            If True, use the transformed/aligned pc1.
        max_points : int, optional, default 5_000_000
            Maximum points to load per cloud. Set to None for no limit.
        voxel_size : float, optional
            Voxel size for downsampling during loading. If None, no
            voxel downsampling is applied (only max_points limit).
        output_dir : str, optional
            Directory for cropped cloud outputs. If None, uses input directories.
        overwrite : bool, default True
            Overwrite existing cropped files.
        verbose : bool, default True
            Print progress messages.

        Returns
        -------
        dict
            Results including:
            - differences: array of Z differences for each pc1 point
            - distances_3d: 3D distances to nearest pc2 point
            - valid_mask: boolean mask for points within max_distance
            - statistics: dict with mean, std, median, nmad, percentiles, etc.
            - pc1_cropped: PointCloud cropped to Area A
            - pc2_cropped: PointCloud cropped to Area A
        """
        import sys
        from scipy.spatial import cKDTree

        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print("Computing 3D Point Cloud Difference", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)

        # =====================================================================
        # Step 1: Crop both clouds to overlap area (Area A)
        # =====================================================================
        if verbose:
            print(f"\nCropping clouds to overlap area (Area A)...", file=sys.stderr)

        pc1_cropped, pc2_cropped = self.crop_to_overlap(
            output_dir=output_dir,
            use_transformed=use_transformed,
            overwrite=overwrite,
            verbose=verbose,
        )

        # =====================================================================
        # Step 2: Load points from cropped clouds
        # =====================================================================
        if verbose:
            print(f"\nLoading points from cropped clouds...", file=sys.stderr)
            if voxel_size:
                print(f"  Voxel downsampling: {voxel_size} m", file=sys.stderr)
            if max_points:
                print(f"  Max points per cloud: {max_points:,}", file=sys.stderr)

        points1 = _load_points_from_las(
            pc1_cropped.filename,
            max_points=max_points,
            voxel_size=voxel_size,
        )
        points2 = _load_points_from_las(
            pc2_cropped.filename,
            max_points=max_points,
            voxel_size=voxel_size,
        )

        if verbose:
            print(f"  PC1 (compare): {len(points1):,} points", file=sys.stderr)
            print(f"  PC2 (reference): {len(points2):,} points", file=sys.stderr)

        # =====================================================================
        # Step 3: Build KD-tree and find correspondences
        # =====================================================================
        if verbose:
            print(f"\nBuilding KD-tree on reference cloud...", file=sys.stderr)

        tree2 = cKDTree(points2)

        if verbose:
            print(f"Finding nearest neighbors...", file=sys.stderr)

        distances_3d, indices = tree2.query(points1, k=1, workers=-1)

        # =====================================================================
        # Step 4: Compute Z differences (pc2 - pc1, positive = gain)
        # =====================================================================
        z1 = points1[:, 2]
        z2_nearest = points2[indices, 2]
        z_differences = z2_nearest - z1

        # Apply distance filter
        valid_mask = distances_3d <= max_distance
        valid_differences = z_differences[valid_mask]

        # =====================================================================
        # Step 5: Compute statistics
        # =====================================================================
        if len(valid_differences) > 0:
            median_val = float(np.median(valid_differences))
            statistics = {
                'count': len(valid_differences),
                'total_count': len(z_differences),
                'mean': float(np.mean(valid_differences)),
                'std': float(np.std(valid_differences)),
                'median': median_val,
                'min': float(np.min(valid_differences)),
                'max': float(np.max(valid_differences)),
                'q05': float(np.percentile(valid_differences, 5)),
                'q25': float(np.percentile(valid_differences, 25)),
                'q75': float(np.percentile(valid_differences, 75)),
                'q95': float(np.percentile(valid_differences, 95)),
                'iqr': float(np.percentile(valid_differences, 75) - np.percentile(valid_differences, 25)),
                'nmad': float(1.4826 * np.median(np.abs(valid_differences - median_val))),
                'valid_ratio': float(np.sum(valid_mask) / len(valid_mask)),
            }
        else:
            statistics = {
                'count': 0,
                'total_count': len(z_differences),
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'q05': np.nan,
                'q25': np.nan,
                'q75': np.nan,
                'q95': np.nan,
                'iqr': np.nan,
                'nmad': np.nan,
                'valid_ratio': 0.0,
            }

        if verbose:
            print(f"\n3D Difference Statistics:", file=sys.stderr)
            print(f"  Valid points: {statistics['count']:,} / {statistics['total_count']:,} "
                  f"({statistics['valid_ratio']:.1%})", file=sys.stderr)
            print(f"  Mean:   {statistics['mean']:.4f} m", file=sys.stderr)
            print(f"  Std:    {statistics['std']:.4f} m", file=sys.stderr)
            print(f"  Median: {statistics['median']:.4f} m", file=sys.stderr)
            print(f"  NMAD:   {statistics['nmad']:.4f} m", file=sys.stderr)
            print(f"  IQR:    {statistics['iqr']:.4f} m", file=sys.stderr)
            print(f"  Range:  [{statistics['min']:.4f}, {statistics['max']:.4f}] m",
                  file=sys.stderr)
            print(f"  5-95%:  [{statistics['q05']:.4f}, {statistics['q95']:.4f}] m",
                  file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)

        return {
            'differences': z_differences,
            'distances_3d': distances_3d,
            'valid_mask': valid_mask,
            'statistics': statistics,
            'max_distance': max_distance,
            'pc1_cropped': pc1_cropped,
            'pc2_cropped': pc2_cropped,
        }
    
    # =========================================================================
    # 2D Differencing Methods (via RasterPair)
    # =========================================================================

    # Valid DEM source options for dem1 (compare/older)
    _DEM1_OPTIONS = {
        "dtm": "dtm",
        "dsm": "dsm",
        "dtm_transformed": "dtm",
        "dsm_transformed": "dsm",
        "dtm_aligned": "dtm",
        "dsm_aligned": "dsm",
        "dtm_transformed_aligned": "dtm",
        "dsm_transformed_aligned": "dsm",
    }

    # Valid DEM source options for dem2 (reference/younger)
    _DEM2_OPTIONS = {
        "dtm": "dtm",
        "dsm": "dsm",
    }

    def _resolve_pc1_source(self, dem1_option: str) -> Tuple[PointCloud, str]:
        """
        Resolve which pc1 variant to use based on dem1 option string.

        Parameters
        ----------
        dem1_option : str
            One of: "dtm", "dsm", "dtm_transformed", "dsm_transformed",
            "dtm_aligned", "dsm_aligned", "dtm_transformed_aligned", "dsm_transformed_aligned"

        Returns
        -------
        tuple[PointCloud, str]
            (point_cloud_source, dem_type)

        Raises
        ------
        ValueError
            If dem1_option is invalid or required point cloud variant is not available.
        """
        if dem1_option not in self._DEM1_OPTIONS:
            valid = ", ".join(sorted(self._DEM1_OPTIONS.keys()))
            raise ValueError(f"Invalid dem1 option '{dem1_option}'. Valid options: {valid}")

        dem_type = self._DEM1_OPTIONS[dem1_option]

        # Determine which point cloud source to use
        # Options containing "aligned" or "transformed" use the aligned/transformed version
        # Options without these use the original cropped (unaligned) version
        needs_aligned = "aligned" in dem1_option or "transformed" in dem1_option

        if needs_aligned:
            # Use aligned/transformed version
            if self._pc1_transformed is not None:
                return self._pc1_transformed, dem_type
            elif self._pc1_cropped is not None:
                # _pc1_cropped gets updated to aligned version after alignment
                return self._pc1_cropped, dem_type
            else:
                raise ValueError(
                    f"dem1='{dem1_option}' requires an aligned/transformed point cloud, "
                    "but no alignment has been performed. Run align_point_clouds() first, "
                    "or use dem1='dtm' or dem1='dsm' for the unaligned version."
                )
        else:
            # Use original cropped (unaligned) version
            if self._pc1_cropped_original is not None:
                return self._pc1_cropped_original, dem_type
            elif self._pc1_cropped is not None and self._pc1_transformed is None:
                # No alignment done yet, _pc1_cropped is still the original
                return self._pc1_cropped, dem_type
            elif self.pc1 is not None:
                # Fall back to original pc1 (not cropped)
                import warnings
                warnings.warn(
                    f"No cropped point cloud available for dem1='{dem1_option}'. "
                    "Using original pc1. Run crop_to_overlap() first for proper results."
                )
                return self.pc1, dem_type
            else:
                raise ValueError("No point cloud available for dem1.")

    def _resolve_pc2_source(self, dem2_option: str) -> Tuple[PointCloud, str]:
        """
        Resolve which pc2 variant to use based on dem2 option string.

        Parameters
        ----------
        dem2_option : str
            One of: "dtm", "dsm"

        Returns
        -------
        tuple[PointCloud, str]
            (point_cloud_source, dem_type)
        """
        if dem2_option not in self._DEM2_OPTIONS:
            valid = ", ".join(sorted(self._DEM2_OPTIONS.keys()))
            raise ValueError(f"Invalid dem2 option '{dem2_option}'. Valid options: {valid}")

        dem_type = self._DEM2_OPTIONS[dem2_option]

        # pc2 is always the reference - use cropped if available
        if self._pc2_cropped is not None:
            return self._pc2_cropped, dem_type
        else:
            return self.pc2, dem_type

    def compute_2d_difference(
        self,
        dem1: Optional[str] = None,
        dem2: Optional[str] = None,
        dem_type: str = "dtm",
        resolution: float = 1.0,
        interpolation: str = "idw",
        transform_first: bool = True,
        use_transformed: bool = True,
        output_dir: Optional[str] = None,
        diff_output_path: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
        **dem_kwargs,
    ) -> Dict[str, Any]:
        """
        Compute 2D (raster-based) elevation difference.

        This method:
        1. Creates DEMs from both point clouds
        2. Creates a RasterPair
        3. Uses RasterPair.compute_difference() for differencing

        Parameters
        ----------
        dem1 : str, optional
            DEM source for compare/older point cloud (pc1). Options:
            - "dtm" or "dsm": Use original cropped (unaligned) point cloud
            - "dtm_transformed", "dsm_transformed": Use transformed point cloud
            - "dtm_aligned", "dsm_aligned": Use aligned point cloud
            - "dtm_transformed_aligned", "dsm_transformed_aligned": Use fully processed cloud
            If not specified, falls back to dem_type + use_transformed behavior.
        dem2 : str, optional
            DEM source for reference/younger point cloud (pc2). Options:
            - "dtm": Create DTM from reference cloud (cropped to overlap)
            - "dsm": Create DSM from reference cloud (cropped to overlap)
            If not specified, falls back to dem_type.
        dem_type : str, {"dtm", "dsm"}, default "dtm"
            Type of DEM to create. Only used if dem1/dem2 are not specified.
            Deprecated: Use dem1 and dem2 parameters instead.
        resolution : float
            DEM resolution in map units
        interpolation : str
            DEM interpolation method
        transform_first : bool
            Transform compare DEM to match reference before differencing.
            Set to False when using unaligned dem1 options to preserve raw differences.
        use_transformed : bool
            Use transformed pc1 for DEM creation. Only used if dem1 is not specified.
            Deprecated: Use dem1 parameter instead.
        output_dir : str, optional
            Directory for output files
        overwrite : bool
            Overwrite existing files
        verbose : bool
            Print progress messages
        **dem_kwargs
            Additional arguments for DEM creation

        Returns
        -------
        dict
            Results from RasterPair.compute_difference() plus:
            - dem1_source: str describing which pc1 variant was used
            - dem2_source: str describing which pc2 variant was used
            - dem_type: str (for backward compatibility)
            - raster_pair: RasterPair object

        Examples
        --------
        # Use aligned DTM for compare, DTM for reference (typical workflow)
        >>> result = pc_pair.compute_2d_difference(
        ...     dem1="dtm_transformed_aligned",
        ...     dem2="dtm",
        ...     resolution=1.0,
        ... )

        # Use unaligned DTM for compare to see raw differences
        >>> result = pc_pair.compute_2d_difference(
        ...     dem1="dtm",
        ...     dem2="dtm",
        ...     resolution=1.0,
        ...     transform_first=False,  # Don't re-align during differencing
        ... )

        # Legacy usage (deprecated but still supported)
        >>> result = pc_pair.compute_2d_difference(
        ...     dem_type="dtm",
        ...     use_transformed=True,
        ... )
        """
        import sys

        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print("Computing 2D (DEM-based) Difference", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)

        # Resolve dem1 and dem2 sources
        if dem1 is not None:
            # New API: use explicit dem1 option
            pc1_source, dem1_type = self._resolve_pc1_source(dem1)
            dem1_source_desc = dem1
        else:
            # Legacy API: use dem_type and use_transformed
            dem1_type = dem_type
            if use_transformed and self._pc1_transformed is not None:
                pc1_source = self._pc1_transformed
                dem1_source_desc = f"{dem_type}_transformed"
            elif self._pc1_cropped is not None:
                pc1_source = self._pc1_cropped
                dem1_source_desc = f"{dem_type}_cropped"
            else:
                pc1_source = self.pc1
                dem1_source_desc = f"{dem_type}_original"

        if dem2 is not None:
            # New API: use explicit dem2 option
            pc2_source, dem2_type = self._resolve_pc2_source(dem2)
            dem2_source_desc = dem2
        else:
            # Legacy API: use dem_type
            dem2_type = dem_type
            if self._pc2_cropped is not None:
                pc2_source = self._pc2_cropped
                dem2_source_desc = f"{dem_type}_cropped"
            else:
                pc2_source = self.pc2
                dem2_source_desc = f"{dem_type}_original"

        if verbose:
            print(f"DEM1 source: {dem1_source_desc} ({Path(pc1_source.filename).name})", file=sys.stderr)
            print(f"DEM2 source: {dem2_source_desc} ({Path(pc2_source.filename).name})", file=sys.stderr)
            print(f"DEM1 type: {dem1_type.upper()}", file=sys.stderr)
            print(f"DEM2 type: {dem2_type.upper()}", file=sys.stderr)

        # Determine output paths
        if output_dir is None:
            dir1 = Path(pc1_source.filename).parent
            dir2 = Path(pc2_source.filename).parent
        else:
            dir1 = dir2 = Path(output_dir)
            dir1.mkdir(parents=True, exist_ok=True)

        out1 = dir1 / f"{Path(pc1_source.filename).stem}_{dem1_type}_{int(resolution)}m.tif"
        out2 = dir2 / f"{Path(pc2_source.filename).stem}_{dem2_type}_{int(resolution)}m.tif"

        if verbose:
            print(f"\nCreating DEM from pc1: {Path(pc1_source.filename).name}", file=sys.stderr)

        # Create DEMs
        raster_dem1 = pc1_source.create_dem(
            output_path=str(out1),
            dem_type=dem1_type,
            resolution=resolution,
            interpolation=interpolation,
            **dem_kwargs,
        )

        if verbose:
            print(f"Creating DEM from pc2: {Path(pc2_source.filename).name}", file=sys.stderr)

        raster_dem2 = pc2_source.create_dem(
            output_path=str(out2),
            dem_type=dem2_type,
            resolution=resolution,
            interpolation=interpolation,
            **dem_kwargs,
        )

        # Copy epoch info
        raster_dem1.epoch = getattr(pc1_source, 'epoch', None)
        raster_dem2.epoch = getattr(pc2_source, 'epoch', None)

        if verbose:
            print(f"\nDEM1: {out1}", file=sys.stderr)
            print(f"DEM2: {out2}", file=sys.stderr)

        # Create RasterPair (dem1 = compare, dem2 = reference)
        raster_pair = RasterPair(raster_dem1, raster_dem2)

        if verbose:
            print("\nRasterPair comparison:", file=sys.stderr)
            raster_pair.print_summary()

        # Compute difference using RasterPair
        result = raster_pair.compute_difference(
            transform_first=transform_first,
            interpolation_method="bilinear",
            clip_to_overlap=True,
            output_path=diff_output_path,
            overwrite=overwrite,
            verbose=verbose,
        )

        # Add context to result
        result['dem1_source'] = dem1_source_desc
        result['dem2_source'] = dem2_source_desc
        result['dem_type'] = dem1_type  # For backward compatibility
        result['dem_resolution'] = resolution
        result['dem_interpolation'] = interpolation
        result['pc1_file'] = self.pc1.filename
        result['pc2_file'] = self.pc2.filename
        result['raster_pair'] = raster_pair

        return result
    
    def compute_dtm_difference(self, **kwargs) -> Dict[str, Any]:
        """Compute DTM-based difference. Convenience wrapper."""
        return self.compute_2d_difference(dem_type="dtm", **kwargs)
    
    def compute_dsm_difference(self, **kwargs) -> Dict[str, Any]:
        """Compute DSM-based difference. Convenience wrapper."""
        return self.compute_2d_difference(dem_type="dsm", **kwargs)

    def compute_mixed_2d_difference(
        self,
        dem_type_pc1: str = "dtm",
        dem_type_pc2: str = "dsm",
        resolution: float = 1.0,
        interpolation: str = "idw",
        classifications_pc1: Optional[Union[str, List[int], Set[int]]] = "auto",
        classifications_pc2: Optional[Union[str, List[int], Set[int]]] = "auto",
        transform_first: bool = True,
        use_transformed: bool = True,
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
        **dem_kwargs,
    ) -> Dict[str, Any]:
        """
        Compute 2D difference with different DEM types for each cloud.

        This method allows mixing DTM and DSM between the compare and reference
        clouds. Useful for scenarios like:
        - DTM from newer survey vs DSM from older survey (vegetation change)
        - DSM from drone vs DTM from lidar (canopy height estimation)

        Parameters
        ----------
        dem_type_pc1 : str, {"dtm", "dsm"}, default "dtm"
            DEM type for pc1 (compare cloud). Only used when classifications_pc1="auto".
        dem_type_pc2 : str, {"dtm", "dsm"}, default "dsm"
            DEM type for pc2 (reference cloud). Only used when classifications_pc2="auto".
        resolution : float
            DEM resolution in map units
        interpolation : str
            DEM interpolation method
        classifications_pc1 : str, list, or set, default "auto"
            Classification filter for pc1. "auto" uses dem_type_pc1.
        classifications_pc2 : str, list, or set, default "auto"
            Classification filter for pc2. "auto" uses dem_type_pc2.
        transform_first : bool
            Transform compare DEM to match reference before differencing
        use_transformed : bool
            Use transformed pc1 for DEM creation
        output_dir : str, optional
            Directory for output files
        overwrite : bool
            Overwrite existing files
        verbose : bool
            Print progress messages
        **dem_kwargs
            Additional arguments for DEM creation

        Returns
        -------
        dict
            Results from RasterPair.compute_difference() plus metadata
        """
        import sys

        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print("Computing Mixed 2D (DEM-based) Difference", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)
            print(f"PC1 DEM type: {dem_type_pc1.upper()}", file=sys.stderr)
            print(f"PC2 DEM type: {dem_type_pc2.upper()}", file=sys.stderr)

        # Select source for pc1
        pc1_source = self._pc1_transformed if use_transformed and self._pc1_transformed else self.pc1

        # Determine output paths
        if output_dir is None:
            dir1 = Path(pc1_source.filename).parent
            dir2 = Path(self.pc2.filename).parent
        else:
            dir1 = dir2 = Path(output_dir)
            dir1.mkdir(parents=True, exist_ok=True)

        out1 = dir1 / f"{Path(pc1_source.filename).stem}_{dem_type_pc1}_{int(resolution)}m.tif"
        out2 = dir2 / f"{Path(self.pc2.filename).stem}_{dem_type_pc2}_{int(resolution)}m.tif"

        # Prepare classification filters
        dem1_kwargs = dict(dem_kwargs)
        dem2_kwargs = dict(dem_kwargs)

        if classifications_pc1 != "auto":
            dem1_kwargs["classification_filter"] = classifications_pc1
        if classifications_pc2 != "auto":
            dem2_kwargs["classification_filter"] = classifications_pc2

        # Create DEM from pc1
        if verbose:
            print(f"\nCreating {dem_type_pc1.upper()} from pc1: {Path(pc1_source.filename).name}",
                  file=sys.stderr)

        dem1 = pc1_source.create_dem(
            output_path=str(out1),
            dem_type=dem_type_pc1,
            resolution=resolution,
            interpolation=interpolation,
            **dem1_kwargs,
        )

        # Create DEM from pc2
        if verbose:
            print(f"Creating {dem_type_pc2.upper()} from pc2: {Path(self.pc2.filename).name}",
                  file=sys.stderr)

        dem2 = self.pc2.create_dem(
            output_path=str(out2),
            dem_type=dem_type_pc2,
            resolution=resolution,
            interpolation=interpolation,
            **dem2_kwargs,
        )

        # Copy epoch info
        dem1.epoch = getattr(pc1_source, 'epoch', None)
        dem2.epoch = getattr(self.pc2, 'epoch', None)

        # Create RasterPair (dem1 = compare, dem2 = reference)
        raster_pair = RasterPair(dem1, dem2)

        if verbose:
            print("\nRasterPair comparison:", file=sys.stderr)
            raster_pair.print_summary()

        # Compute difference using RasterPair
        result = raster_pair.compute_difference(
            transform_first=transform_first,
            interpolation_method="bilinear",
            clip_to_overlap=True,
            overwrite=overwrite,
            verbose=verbose,
        )

        # Add point cloud context to result
        result['dem_type_pc1'] = dem_type_pc1
        result['dem_type_pc2'] = dem_type_pc2
        result['dem_resolution'] = resolution
        result['dem_interpolation'] = interpolation
        result['pc1_file'] = self.pc1.filename
        result['pc2_file'] = self.pc2.filename
        result['mixed_difference'] = True

        return result

    # =========================================================================
    # Full Pipeline Methods
    # =========================================================================
    
    def full_differencing_pipeline(
        self,
        dem_type: str = "dtm",
        resolution: float = 1.0,
        interpolation: str = "idw",
        align_icp: bool = True,
        icp_method: str = "gicp",
        icp_downsample: float = 0.5,
        skip_epoch: bool = False,
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full differencing pipeline.
        
        Pipeline steps:
        1. Transform pc1 to match pc2's reference frame
        2. (Optional) ICP alignment for fine registration
        3. Create DEMs
        4. Compute difference
        
        Parameters
        ----------
        dem_type : str
            Type of DEM ("dtm" or "dsm")
        resolution : float
            DEM resolution in meters
        interpolation : str
            DEM interpolation method
        align_icp : bool
            Whether to perform ICP alignment
        icp_method : str
            ICP method ("gicp", "vgicp", "icp")
        icp_downsample : float
            Downsampling resolution for ICP
        skip_epoch : bool
            Skip epoch transformation (faster, but less accurate if epochs differ significantly)
        output_dir : str, optional
            Output directory
        overwrite : bool
            Overwrite existing files
        verbose : bool
            Print progress messages
            
        Returns
        -------
        dict
            Complete results including:
            - comparison: initial comparison results
            - transformation_history: list of applied transformations
            - alignment_result: ICP alignment results (if performed)
            - difference_result: DEM differencing results
        """
        if verbose:
            print(f"\n{'#' * 60}")
            print("# Full Differencing Pipeline")
            print(f"{'#' * 60}")
        
        results = {
            'comparison': self.check_all_match(),
            'transformation_history': [],
            'alignment_result': None,
            'difference_result': None,
        }
        
        # Step 1: Transform pc1 to match pc2
        if verbose:
            print("\n[Pipeline Step 1/4] CRS/Datum Transformation")
        
        if results['comparison']['transformations_needed']:
            self.transform_compare_to_match_reference(
                skip_epoch=skip_epoch,
                overwrite=overwrite,
                verbose=verbose,
            )
            results['transformation_history'] = self._transformation_history.copy()
        else:
            if verbose:
                print("  No transformations needed - point clouds already aligned")
        
        # Step 2: ICP Alignment
        if align_icp:
            if verbose:
                print("\n[Pipeline Step 2/4] ICP Fine Alignment")
            
            if _has_small_gicp():
                alignment = self.align_point_clouds(
                    method=icp_method,
                    downsample_resolution=icp_downsample,
                    apply_transform=True,
                    overwrite=overwrite,
                    verbose=verbose,
                )
                results['alignment_result'] = alignment
            else:
                if verbose:
                    print("  Skipping ICP - small_gicp not installed")
        else:
            if verbose:
                print("\n[Pipeline Step 2/4] ICP Alignment - Skipped")
        
        # Step 3 & 4: DEM creation and differencing
        if verbose:
            print("\n[Pipeline Step 3-4/4] DEM Creation and Differencing")
        
        diff_result = self.compute_2d_difference(
            dem_type=dem_type,
            resolution=resolution,
            interpolation=interpolation,
            use_transformed=True,
            output_dir=output_dir,
            overwrite=overwrite,
            verbose=verbose,
        )
        results['difference_result'] = diff_result
        
        if verbose:
            print(f"\n{'#' * 60}")
            print("# Pipeline Complete")
            print(f"{'#' * 60}")
            
            stats = diff_result.get('stats', {})
            print(f"\nFinal Difference Statistics:")
            print(f"  Mean:   {stats.get('mean', np.nan):.4f} m")
            print(f"  Std:    {stats.get('std', np.nan):.4f} m")
            print(f"  Median: {stats.get('median', np.nan):.4f} m")
            print(f"  NMAD:   {stats.get('nmad', np.nan):.4f} m")
            print(f"\nDifference raster: {diff_result.get('difference_raster', {}).filename}")
            print(f"{'#' * 60}\n")
        
        return results
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_transformed_pc1(self) -> Optional[PointCloud]:
        """Get the transformed pc1, if available."""
        return self._pc1_transformed
    
    def get_alignment_result(self) -> Optional[Dict[str, Any]]:
        """Get the ICP alignment result, if available."""
        return self._alignment_result
    
    def get_transformation_history(self) -> List[Dict[str, Any]]:
        """Get the list of transformations applied."""
        return self._transformation_history.copy()
    
    def reset(self) -> None:
        """Reset internal state (clear cached transformations)."""
        self._transformation_history = []
        self._pc1_transformed = None
        self._alignment_result = None

    def process_point_cloud_pair(
        self,
        # CRS/Transformation options
        warp_to_reference: bool = True,
        skip_epoch: bool = False,
        skip_vertical: bool = False,
        # Cropping options
        crop_to_overlap: bool = True,
        alignment_buffer: float = 10.0,
        # Alignment options
        align_icp: bool = True,
        icp_method: str = "vgicp",
        auto_downsample: bool = True,
        target_points: int = 2_000_000,
        max_correspondence_distance: float = 1.0,
        compute_quality: bool = True,
        # DEM options
        dem_type: str = "dtm",
        dem_type_pc1: Optional[str] = None,
        dem_type_pc2: Optional[str] = None,
        resolution: float = 1.0,
        interpolation: str = "idw",
        classifications_pc1: Optional[Union[str, List[int], Set[int]]] = "auto",
        classifications_pc2: Optional[Union[str, List[int], Set[int]]] = "auto",
        # Output options
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive point cloud processing workflow.

        This method executes a full point cloud comparison workflow:
        1. Compare metadata between clouds (CRS, epoch, geoid, units)
        2. Warp compare cloud CRS to match reference
        3. Crop both clouds to overlap area (Area A)
        4. Crop compare cloud with buffer for alignment (Area B)
        5. ICP alignment using Area B (compare) vs Area A (reference)
        6. Compute alignment quality metrics
        7. Create DEMs (DTM/DSM with optional custom classifications)
        8. Compute 2D difference via RasterPair

        Parameters
        ----------
        warp_to_reference : bool, default True
            Transform pc1 to match pc2's CRS/datum/epoch.
        skip_epoch : bool, default False
            Skip epoch transformation even if epochs differ.
        skip_vertical : bool, default False
            Skip vertical datum transformation.
        crop_to_overlap : bool, default True
            Crop clouds to overlap area before alignment/differencing.
        alignment_buffer : float, default 10.0
            Buffer distance (meters) for Area B around overlap polygon.
        align_icp : bool, default True
            Perform ICP alignment for fine registration.
        icp_method : str, default "vgicp"
            ICP method: "vgicp", "gicp", "icp", or "plane_icp".
        auto_downsample : bool, default True
            Automatically calculate optimal voxel size for alignment.
        target_points : int, default 2_000_000
            Target point count for auto-downsampling.
        max_correspondence_distance : float, default 1.0
            Maximum distance for ICP correspondences (meters).
        compute_quality : bool, default True
            Compute detailed alignment quality metrics after ICP.
        dem_type : str, default "dtm"
            Default DEM type for both clouds. Overridden by dem_type_pc1/pc2.
        dem_type_pc1 : str, optional
            DEM type for pc1. If None, uses dem_type.
        dem_type_pc2 : str, optional
            DEM type for pc2. If None, uses dem_type.
        resolution : float, default 1.0
            DEM resolution in meters.
        interpolation : str, default "idw"
            DEM interpolation method.
        classifications_pc1 : str, list, or set, default "auto"
            Classification filter for pc1 DEM creation.
        classifications_pc2 : str, list, or set, default "auto"
            Classification filter for pc2 DEM creation.
        output_dir : str, optional
            Directory for all output files. If None, uses input directories.
        overwrite : bool, default True
            Overwrite existing output files.
        verbose : bool, default True
            Print progress messages.

        Returns
        -------
        dict
            Comprehensive results including:
            - comparison: Initial metadata comparison
            - transformation_history: List of CRS transformations applied
            - overlap_info: Overlap polygon and area statistics
            - alignment_result: ICP alignment results (if performed)
            - alignment_quality: Detailed quality metrics (if computed)
            - dem1, dem2: Created Raster objects
            - difference_result: 2D differencing results with statistics
        """
        import sys

        if verbose:
            print(f"\n{'#' * 70}", file=sys.stderr)
            print("# Point Cloud Pair Processing Workflow", file=sys.stderr)
            print(f"{'#' * 70}", file=sys.stderr)
            print(f"Compare:   {Path(self.pc1.filename).name}", file=sys.stderr)
            print(f"Reference: {Path(self.pc2.filename).name}", file=sys.stderr)

        results: Dict[str, Any] = {
            'comparison': None,
            'transformation_history': [],
            'overlap_info': None,
            'alignment_result': None,
            'alignment_quality': None,
            'dem1': None,
            'dem2': None,
            'difference_result': None,
        }

        # =====================================================================
        # Step 1: Compare metadata
        # =====================================================================
        if verbose:
            print(f"\n[Step 1/7] Comparing metadata...", file=sys.stderr)

        results['comparison'] = self.check_all_match()

        if verbose:
            self.print_comparison()

        # =====================================================================
        # Step 2: Warp CRS to match reference
        # =====================================================================
        if warp_to_reference and results['comparison']['transformations_needed']:
            if verbose:
                print(f"\n[Step 2/7] Warping compare cloud to reference frame...",
                      file=sys.stderr)

            self.transform_compare_to_match_reference(
                skip_epoch=skip_epoch,
                skip_vertical=skip_vertical,
                overwrite=overwrite,
                verbose=verbose,
            )
            results['transformation_history'] = self._transformation_history.copy()
        else:
            if verbose:
                print(f"\n[Step 2/7] CRS transformation - Skipped", file=sys.stderr)
                if not results['comparison']['transformations_needed']:
                    print("  (Point clouds already in same reference frame)",
                          file=sys.stderr)

        # =====================================================================
        # Step 3: Compute overlap and crop
        # =====================================================================
        if verbose:
            print(f"\n[Step 3/7] Computing overlap area...", file=sys.stderr)

        results['overlap_info'] = self.compute_overlap_polygon(use_transformed=True)

        if not results['overlap_info']['has_overlap']:
            raise ValueError("Point clouds do not overlap. Cannot proceed with workflow.")

        if verbose:
            oi = results['overlap_info']
            print(f"  Overlap area: {oi['overlap_area']:,.0f} m²", file=sys.stderr)
            print(f"  PC1 coverage: {oi['overlap_fraction_pc1']:.1%}", file=sys.stderr)
            print(f"  PC2 coverage: {oi['overlap_fraction_pc2']:.1%}", file=sys.stderr)

        # =====================================================================
        # Step 4: ICP Alignment
        # =====================================================================
        if align_icp:
            if verbose:
                print(f"\n[Step 4/7] ICP alignment ({icp_method.upper()})...",
                      file=sys.stderr)

            if not _has_small_gicp():
                if verbose:
                    print("  WARNING: small_gicp not installed, skipping alignment",
                          file=sys.stderr)
            else:
                alignment = self.align_point_clouds(
                    method=icp_method,
                    auto_downsample=auto_downsample,
                    target_points=target_points,
                    max_correspondence_distance=max_correspondence_distance,
                    alignment_buffer=alignment_buffer,
                    use_cropped_clouds=crop_to_overlap,
                    apply_transform=True,
                    overwrite=overwrite,
                    verbose=verbose,
                )
                results['alignment_result'] = alignment

                # Step 5: Compute alignment quality
                if compute_quality:
                    if verbose:
                        print(f"\n[Step 5/7] Computing alignment quality...",
                              file=sys.stderr)

                    results['alignment_quality'] = self.compute_alignment_quality(
                        max_distance=max_correspondence_distance,
                        verbose=verbose,
                    )
                else:
                    if verbose:
                        print(f"\n[Step 5/7] Alignment quality - Skipped", file=sys.stderr)
        else:
            if verbose:
                print(f"\n[Step 4/7] ICP alignment - Skipped", file=sys.stderr)
                print(f"\n[Step 5/7] Alignment quality - Skipped", file=sys.stderr)

        # =====================================================================
        # Step 6: Create DEMs
        # =====================================================================
        if verbose:
            print(f"\n[Step 6/7] Creating DEMs...", file=sys.stderr)

        # Determine DEM types
        dt_pc1 = dem_type_pc1 if dem_type_pc1 is not None else dem_type
        dt_pc2 = dem_type_pc2 if dem_type_pc2 is not None else dem_type
        is_mixed = (dt_pc1 != dt_pc2)

        if is_mixed:
            # Use mixed method (different DEM types)
            if verbose:
                print(f"  Mixed mode: PC1={dt_pc1.upper()}, PC2={dt_pc2.upper()}",
                      file=sys.stderr)
        else:
            if verbose:
                print(f"  DEM type: {dt_pc1.upper()}", file=sys.stderr)

        dem1, dem2 = self.create_dem_pair(
            dem_type=dt_pc1,  # For pc1
            resolution=resolution,
            interpolation=interpolation,
            use_transformed=True,
            output_dir=output_dir,
            overwrite=overwrite,
            verbose=verbose,
            classifications_pc1=classifications_pc1,
            classifications_pc2=classifications_pc2,
        )

        # If mixed, recreate dem2 with different type
        if is_mixed:
            if verbose:
                print(f"  Recreating PC2 DEM as {dt_pc2.upper()}...", file=sys.stderr)

            if output_dir is None:
                dir2 = Path(self.pc2.filename).parent
            else:
                dir2 = Path(output_dir)

            out2 = dir2 / f"{Path(self.pc2.filename).stem}_{dt_pc2}_{int(resolution)}m.tif"

            dem2_kwargs = {}
            if classifications_pc2 != "auto":
                dem2_kwargs["classification_filter"] = classifications_pc2

            dem2 = self.pc2.create_dem(
                output_path=str(out2),
                dem_type=dt_pc2,
                resolution=resolution,
                interpolation=interpolation,
                **dem2_kwargs,
            )
            dem2.epoch = getattr(self.pc2, 'epoch', None)

        results['dem1'] = dem1
        results['dem2'] = dem2

        # =====================================================================
        # Step 7: Compute 2D difference
        # =====================================================================
        if verbose:
            print(f"\n[Step 7/7] Computing 2D difference...", file=sys.stderr)

        raster_pair = RasterPair(dem1, dem2)
        diff_result = raster_pair.compute_difference(
            transform_first=True,
            interpolation_method="bilinear",
            clip_to_overlap=True,
            overwrite=overwrite,
            verbose=verbose,
        )

        # Add metadata
        diff_result['dem_type_pc1'] = dt_pc1
        diff_result['dem_type_pc2'] = dt_pc2
        diff_result['dem_resolution'] = resolution
        diff_result['pc1_file'] = self.pc1.filename
        diff_result['pc2_file'] = self.pc2.filename

        results['difference_result'] = diff_result

        # =====================================================================
        # Summary
        # =====================================================================
        if verbose:
            print(f"\n{'#' * 70}", file=sys.stderr)
            print("# Workflow Complete", file=sys.stderr)
            print(f"{'#' * 70}", file=sys.stderr)

            stats = diff_result.get('stats', {})
            print(f"\nDifference Statistics:", file=sys.stderr)
            print(f"  Mean:   {stats.get('mean', np.nan):.4f} m", file=sys.stderr)
            print(f"  Std:    {stats.get('std', np.nan):.4f} m", file=sys.stderr)
            print(f"  Median: {stats.get('median', np.nan):.4f} m", file=sys.stderr)
            print(f"  NMAD:   {stats.get('nmad', np.nan):.4f} m", file=sys.stderr)

            if results['alignment_quality']:
                aq = results['alignment_quality']
                print(f"\nAlignment Quality:", file=sys.stderr)
                print(f"  RMSE:   {aq['rmse']:.4f} m", file=sys.stderr)
                print(f"  Inlier: {aq['inlier_ratio']:.1%}", file=sys.stderr)

            diff_raster = diff_result.get('difference_raster')
            if diff_raster:
                print(f"\nOutput: {diff_raster.filename}", file=sys.stderr)

            print(f"{'#' * 70}\n", file=sys.stderr)

        return results