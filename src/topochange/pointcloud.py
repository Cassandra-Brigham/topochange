"""Point cloud loading, metadata extraction, and transformation.

Provides the PointCloud class for loading LAS/LAZ files, extracting CRS and
time metadata, and performing coordinate transformations via PDAL pipelines.
"""
from __future__ import annotations

import datetime
import gc
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import numpy as np
# Use pdal_wrapper for Colab compatibility (falls back to native pdal locally)
try:
    from .pdal_wrapper import pdal
except ImportError:
    import pdal
import pyproj
import rasterio
import scipy  # noqa: F401  (needed if scipy is installed, used via has_scipy / generic_filter)
import shapely.geometry
from pyproj import CRS as CRS_
from pyproj import Proj, Transformer
from pyproj.crs import CompoundCRS
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from shapely.geometry import Polygon
from shapely.ops import transform

from .crs_utils import (
    _ensure_crs_obj,
    apply_dynamic_transform,
    crs_to_projjson,
    crs_to_wkt2_2019,
    is_orthometric,
    is_3d_geographic_crs,
    extract_ellipsoidal_height_as_vertical_crs,
    make_coordinate_metadata_projjson,
    wrap_coordinate_metadata_wkt,
)
from .unit_utils import (
    UnitInfo,
    UNKNOWN_UNIT,
    METER,
    FOOT,
    US_SURVEY_FOOT,
    parse_pdal_units,
    get_horizontal_unit,
    get_vertical_unit,
    get_crs_units,
    convert_length,
    convert_to_meters,
    get_conversion_factor,
    lookup_unit,
    parse_unit_string,
    format_value_with_unit,
    describe_unit,
    # Backward-compatible functions
    horizontal_unit_scale,
    vertical_unit_scale,
)
from .geoid_utils import parse_geoid_info, select_geoid_grid
from .time_utils import (
    _datetime_to_decimal_year,
    _guess_in_time_from_stats,
    _parse_epoch_string_to_decimal,
    gps_seconds_to_decimal_year_utc,
)

from .pipeline_builder import CRSState, ProjError, build_complete_pipeline
from .deformation_utils import select_velocity_model

if TYPE_CHECKING:
    from .raster import Raster


def has_rasterio() -> bool:
    try:
        import rasterio  # noqa: F401
        return True
    except Exception:
        return False


def has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


class DependencyMissingError(ImportError):
    def __init__(self, package: str, where: str = ""):
        msg = f"Required dependency '{package}' is not available"
        if where:
            msg += f" (needed by {where})"
        super().__init__(msg)


def _determine_utm_epsg(poly4326: Polygon) -> str:
    """
    Return the EPSG code string (e.g. '32611') of the centroid's UTM zone.
    """
    lon, lat = poly4326.centroid.xy
    zone = int((lon[0] + 180) / 6) + 1
    hemi = "north" if lat[0] >= 0 else "south"
    proj = Proj(f"+proj=utm +zone={zone} +{hemi} +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    epsg = CRS_(proj.srs).to_epsg()
    if epsg is not None:
        return str(epsg)
    # Fallback if EPSG cannot be determined from PROJ
    return str(32600 + zone if hemi == "north" else 32700 + zone)


def _reproject_poly(poly: Polygon, src_epsg: Union[str, int], dst_epsg: Union[str, int]) -> Polygon:
    """
    Reproject a polygon between coordinate reference systems.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Geometry to reproject.
    src_epsg : int or str
        EPSG code of the source CRS.
    dst_epsg : int or str
        EPSG code of the destination CRS.

    Returns
    -------
    shapely.geometry.Polygon
        The polygon transformed into the target CRS.
    """
    tf = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
    return transform(lambda x, y, z=None: tf.transform(x, y), poly)


def get_true_extent(pc: "PointCloud", edge_size: float = 5.0) -> Tuple[int, Polygon, Polygon]:
    """
    Extract point cloud true data boundary using hexbin filter.

    This computes the actual data footprint (not just bounding box) by using
    PDAL's hexbin filter which creates a polygon from occupied hex cells.
    The resulting polygon is shrunk inward by half the edge_size to compensate
    for hex cells extending beyond actual point locations.

    Parameters
    ----------
    pc : PointCloud
        Point cloud object with filename attribute.
    edge_size : float, default 5.0
        Size of hexbin cells in meters. Smaller values give more accurate
        boundaries but take longer to compute.

    Returns
    -------
    tuple[int, Polygon, Polygon]
        (epsg_utm, poly_utm, poly_4326) - UTM EPSG code, polygon in UTM,
        and polygon in WGS84.

    Raises
    ------
    RuntimeError
        If true extent cannot be computed.
    """
    import json
    import pdal
    from pyproj import CRS as CRS_
    from shapely import wkt
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

    # Try hexbin approach (streaming-compatible, gives true data polygon)
    try:
        hexbin_pipe = pdal.Pipeline(
            json.dumps(
                {
                    "pipeline": [
                        {"type": "readers.las", "filename": str(pc.filename)},
                        {
                            "type": "filters.hexbin",
                            "edge_size": edge_size,
                            "threshold": 1,   # Include cells with at least 1 point
                        },
                    ]
                }
            )
        )
        # Use streaming execution if available
        if hasattr(hexbin_pipe, 'execute_streaming_metadata'):
            hexbin_pipe.execute_streaming_metadata(chunk_size=100000)
        elif hasattr(hexbin_pipe, 'execute_metadata_only'):
            hexbin_pipe.execute_metadata_only()
        else:
            hexbin_pipe.execute()

        hexbin_md = hexbin_pipe.metadata.get("metadata", {}).get("filters.hexbin", {})
        boundary_wkt = hexbin_md.get("boundary")

        if boundary_wkt:
            # Parse WKT boundary (in native CRS)
            poly_native = wkt.loads(boundary_wkt)

            # Handle MultiPolygon by taking the largest polygon
            if isinstance(poly_native, MultiPolygon):
                poly_native = max(poly_native.geoms, key=lambda p: p.area)

            # Shrink the polygon inward by half the edge_size to compensate for
            # hex cells extending beyond actual point locations
            shrink_distance = edge_size / 2.0
            poly_native_shrunk = poly_native.buffer(-shrink_distance)

            # Handle case where shrinking results in MultiPolygon or empty
            if poly_native_shrunk.is_empty:
                # Fall back to original if shrinking made it empty
                poly_native_shrunk = poly_native
            elif isinstance(poly_native_shrunk, MultiPolygon):
                poly_native_shrunk = max(poly_native_shrunk.geoms, key=lambda p: p.area)

            # Transform to EPSG:4326
            src_crs_wkt = (
                getattr(pc, "current_horizontal_crs", None)
                or getattr(pc, "original_horizontal_crs", None)
                or getattr(pc, "current_compound_crs", None)
                or getattr(pc, "original_compound_crs", None)
            )

            if src_crs_wkt:
                src_crs = CRS_.from_user_input(src_crs_wkt)
                dst_crs = CRS_.from_epsg(4326)
                tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                poly_4326 = transform(lambda x, y, z=None: tf.transform(x, y), poly_native_shrunk)
            else:
                poly_4326 = poly_native_shrunk

            if not poly_4326.is_empty:
                epsg_utm = _determine_utm_epsg(poly_4326)
                poly_utm = _reproject_poly(poly_4326, 4326, epsg_utm)
                return epsg_utm, poly_utm, poly_4326

    except Exception as e:
        raise RuntimeError(f"Could not compute true extent using hexbin: {e}")

    raise RuntimeError("Hexbin filter did not return a boundary polygon")


@dataclass
class PointCloud:
    """
    Point cloud with CRS and metadata tracking.
    
    Attributes
    ----------
    filename : str
        Path to the LAS/LAZ file.
    horizontal_unit : UnitInfo
        Full unit metadata for horizontal coordinates.
    vertical_unit : UnitInfo
        Full unit metadata for vertical (Z) coordinates.
    horizontal_units : str
        Display name of horizontal units (for backward compatibility).
    vertical_units : str
        Display name of vertical units (for backward compatibility).
    """

    filename: str

    def __init__(self, filename: str):
        self.filename = filename
        # Will be initialized in from_file() once metadata is known
        self.crs_history = None
        
        # Unit info objects - initialized to unknown until from_file() is called
        self.horizontal_unit: UnitInfo = UNKNOWN_UNIT
        self.vertical_unit: UnitInfo = UNKNOWN_UNIT

    # -------------------------------------------------------------------------
    # Metadata loading
    # -------------------------------------------------------------------------
    def from_file(self, lightweight: bool = False, bbox_only: bool = False) -> None:
        """
        Load point cloud from LAS/LAZ file with metadata extraction.

        Parameters
        ----------
        lightweight : bool, default False
            If True, only load essential metadata from the LAS header without
            running additional PDAL pipelines. This is much faster and uses
            minimal memory, but skips:
            - GPS time statistics and epoch calculation
            - Classification statistics

            Use lightweight=True for large files in memory-constrained
            environments like Google Colab, or when you only need basic
            CRS and bounds information.

        bbox_only : bool, default False
            Only applies when lightweight=True.
            - If True: Use simple bounding box for polygon (fastest, no point loading)
            - If False: Use hexbin filter for true data footprint polygon
              (more accurate boundary, captures actual data extent)

        Optimized for memory efficiency, especially in Google Colab:
        - Uses metadata-only execution where possible (skips array serialization)
        - Uses streaming mode with hexbin for boundary polygon extraction
        - Includes garbage collection between pipeline executions
        """

        # ------------------------------------------------------------------
        # 1) PDAL metadata: basic CRS, units, counts, bounds, etc.
        #    (count=0 means no points loaded - very efficient)
        # ------------------------------------------------------------------
        pipeline_meta = pdal.Pipeline(
            json.dumps(
                {
                    "pipeline": [
                        {
                            "type": "readers.las",
                            "filename": self.filename,
                            "count": 0,  # Just get metadata, don't load points
                        }
                    ]
                }
            )
        )
        pipeline_meta.execute()
        meta_root = pipeline_meta.metadata
        md = meta_root.get("metadata", {})

        las_md = md.get("readers.las", {})

        # Extract CRS information
        srs_md = las_md.get("srs", {}) or {}

        self.original_compound_crs = srs_md.get("compoundwkt")
        self.original_horizontal_crs = srs_md.get("horizontal")
        self.original_vertical_crs = srs_md.get("vertical")
        self.original_pretty_wkt = srs_md.get("prettywkt")
        self.original_proj_string = srs_md.get("proj4")

        # Set current CRS to original
        self.current_compound_crs = self.original_compound_crs
        self.current_horizontal_crs = self.original_horizontal_crs
        self.current_vertical_crs = self.original_vertical_crs
        self.current_pretty_wkt = self.original_pretty_wkt
        self.current_proj_string = self.original_proj_string

        # Orthometric or ellipsoidal heights?
        if self.original_vertical_crs is not None:
            self.is_orthometric = is_orthometric(self.original_vertical_crs)
        else:
            self.is_orthometric = False

        # Geoid
        geoid_info = parse_geoid_info(md)
        self.geoid_model = geoid_info.get("geoid_model")

        # Get point count and bounds
        self.total_points = las_md.get("count")
        self.maxx = las_md.get("maxx")
        self.maxy = las_md.get("maxy")
        self.minx = las_md.get("minx")
        self.miny = las_md.get("miny")
        self.bounds = (self.minx, self.miny, self.maxx, self.maxy)

        # Creation date (needed for GPS time conversion)
        self.creation_doy = las_md.get("creation_doy")
        self.creation_year = las_md.get("creation_year")

        # Save srs_md for unit parsing (needed after cleanup)
        _srs_md = srs_md

        # Clean up Pipeline 1
        del pipeline_meta, meta_root, md, las_md, srs_md
        gc.collect()

        # ------------------------------------------------------------------
        # Helper function to extract point cloud extent polygon
        # Defined here so both lightweight and full modes can use it
        # ------------------------------------------------------------------
        def _get_pointcloud_extent(pc: PointCloud) -> Tuple[str, Polygon, Polygon]:
            """
            Extract point cloud boundary polygon using streaming hexbin filter.

            This approach is memory-efficient because hexbin works incrementally
            and only stores hex cell occupancy, not all point coordinates.
            """
            # Try hexbin approach first (streaming-compatible, gives true data polygon)
            try:
                hexbin_pipe = pdal.Pipeline(
                    json.dumps(
                        {
                            "pipeline": [
                                {"type": "readers.las", "filename": str(pc.filename)},
                                {
                                    "type": "filters.hexbin",
                                    "edge_size": 10,  # 10m hex cells - balance detail vs memory
                                    "threshold": 1,   # Include cells with at least 1 point
                                },
                            ]
                        }
                    )
                )
                # Use streaming execution - processes points in chunks, very memory efficient
                if hasattr(hexbin_pipe, 'execute_streaming_metadata'):
                    hexbin_pipe.execute_streaming_metadata(chunk_size=100000)
                elif hasattr(hexbin_pipe, 'execute_metadata_only'):
                    hexbin_pipe.execute_metadata_only()
                else:
                    hexbin_pipe.execute()

                hexbin_md = hexbin_pipe.metadata.get("metadata", {}).get("filters.hexbin", {})
                boundary_wkt = hexbin_md.get("boundary")

                if boundary_wkt:
                    # Parse WKT boundary (in native CRS)
                    from shapely import wkt
                    poly_native = wkt.loads(boundary_wkt)

                    # Transform to EPSG:4326
                    src_crs_wkt = (
                        getattr(pc, "current_horizontal_crs", None)
                        or getattr(pc, "original_horizontal_crs", None)
                        or getattr(pc, "current_compound_crs", None)
                        or getattr(pc, "original_compound_crs", None)
                    )

                    if src_crs_wkt:
                        src_crs = CRS_.from_user_input(src_crs_wkt)
                        dst_crs = CRS_.from_epsg(4326)
                        tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                        poly_4326 = transform(lambda x, y, z=None: tf.transform(x, y), poly_native)
                    else:
                        # Assume already in 4326 if no CRS info
                        poly_4326 = poly_native

                    if not poly_4326.is_empty:
                        epsg_utm = _determine_utm_epsg(poly_4326)
                        poly_utm = _reproject_poly(poly_4326, 4326, epsg_utm)
                        return epsg_utm, poly_utm, poly_4326

            except Exception:
                pass  # Fall through to fallback methods

            # Fallback: Try filters.stats for EPSG:4326 boundary (loads all points)
            try:
                meta_pipe = pdal.Pipeline(
                    json.dumps(
                        {
                            "pipeline": [
                                {"type": "readers.las", "filename": str(pc.filename)},
                                {"type": "filters.stats", "dimensions": "X,Y,Z"},
                                {"type": "filters.info"},
                            ]
                        }
                    )
                )
                # Use metadata-only execution to avoid array serialization
                if hasattr(meta_pipe, 'execute_metadata_only'):
                    meta_pipe.execute_metadata_only()
                else:
                    meta_pipe.execute()

                meta_root2 = meta_pipe.metadata
                md2 = meta_root2.get("metadata", {})
                stats_md = md2.get("filters.stats", {})

                # Try to use PDAL's EPSG:4326 boundary if available
                bbox = stats_md.get("bbox", {})
                bbox_4326 = bbox.get("EPSG:4326", {})
                boundary = bbox_4326.get("boundary", {})
                coords_list = boundary.get("coordinates", [])
                coords = coords_list[0] if coords_list else []

                if coords:
                    poly_4326 = Polygon([(float(pt[0]), float(pt[1])) for pt in coords])
                    if not poly_4326.is_empty:
                        epsg_utm = _determine_utm_epsg(poly_4326)
                        poly_utm = _reproject_poly(poly_4326, 4326, epsg_utm)
                        return epsg_utm, poly_utm, poly_4326
            except Exception:
                pass  # Fall through to bounding box fallback

            # Final fallback: construct rectangle from LAS header bounds (no point loading)
            if pc.bounds is None:
                raise ValueError(
                    "No EPSG:4326 bbox in PDAL metadata and pc.bounds is not set; "
                    "cannot determine point cloud extent."
                )

            minx, miny, maxx, maxy = pc.bounds

            src_crs_wkt = (
                getattr(pc, "current_horizontal_crs", None)
                or getattr(pc, "original_horizontal_crs", None)
                or getattr(pc, "current_compound_crs", None)
                or getattr(pc, "original_compound_crs", None)
            )

            if src_crs_wkt:
                src_crs = CRS_.from_user_input(src_crs_wkt)
            else:
                src_crs = CRS_.from_epsg(4326)

            dst_crs = CRS_.from_epsg(4326)
            tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

            xs = [minx, maxx, maxx, minx, minx]
            ys = [miny, miny, maxy, maxy, miny]
            lon, lat = tf.transform(xs, ys)
            poly_4326 = Polygon(zip(lon, lat))

            if poly_4326.is_empty:
                raise ValueError(
                    "Failed to build EPSG:4326 extent polygon from metadata or bounds."
                )

            epsg_utm = _determine_utm_epsg(poly_4326)
            poly_utm = _reproject_poly(poly_4326, 4326, epsg_utm)
            return epsg_utm, poly_utm, poly_4326

        # ------------------------------------------------------------------
        # LIGHTWEIGHT MODE: Skip expensive pipelines
        # ------------------------------------------------------------------
        if lightweight:
            # Parse units from saved srs_md
            self.horizontal_unit, self.vertical_unit = parse_pdal_units(_srs_md)
            if self.horizontal_unit.name == "unknown" and self.original_horizontal_crs:
                self.horizontal_unit = get_horizontal_unit(self.original_horizontal_crs)
            if self.vertical_unit.name == "unknown" and self.original_vertical_crs:
                self.vertical_unit = get_vertical_unit(self.original_vertical_crs)
            self.horizontal_units = self.horizontal_unit.display_name
            self.vertical_units = self.vertical_unit.display_name

            if bbox_only:
                # Build bounding box polygon from header bounds (no point loading)
                src_crs_wkt = (
                    self.original_horizontal_crs
                    or self.original_compound_crs
                )
                if src_crs_wkt:
                    src_crs = CRS_.from_user_input(src_crs_wkt)
                else:
                    src_crs = CRS_.from_epsg(4326)

                dst_crs = CRS_.from_epsg(4326)
                tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

                xs = [self.minx, self.maxx, self.maxx, self.minx, self.minx]
                ys = [self.miny, self.miny, self.maxy, self.maxy, self.miny]
                lon, lat = tf.transform(xs, ys)
                self.poly_4326 = Polygon(zip(lon, lat))
                self.bbox_4326 = self.poly_4326.bounds

                self.epsg_utm = _determine_utm_epsg(self.poly_4326)
                self.poly_utm = _reproject_poly(self.poly_4326, 4326, self.epsg_utm)
            else:
                # Use hexbin for true data polygon (loads points but accurate)
                self.epsg_utm, self.poly_utm, self.poly_4326 = _get_pointcloud_extent(self)
                self.bbox_4326 = self.poly_4326.bounds

            # Set GPS/epoch attributes to None (not computed in lightweight mode)
            self.gps_time_mean_raw = None
            self.gps_time_min_raw = None
            self.gps_time_max_raw = None
            self.gps_stddev_raw = None
            self.gps_time_mean = None
            self.gps_time_min = None
            self.gps_time_max = None
            self.gps_stddev = None
            self.decimal_year_mean_utc = None
            self.decimal_year_min_utc = None
            self.decimal_year_max_utc = None
            self.epoch = None

            # Set classification attributes to empty (not computed in lightweight mode)
            self.classification = {}
            self.class_values = []
            self.class_counts = []
            self.has_ground_class = False

            # Initialize CRS history
            try:
                from .crs_history import CRSHistory
                if getattr(self, "crs_history", None) is None:
                    self.crs_history = CRSHistory(self)
            except Exception:
                self.crs_history = None

            del _srs_md
            gc.collect()
            return  # Early return for lightweight mode

        # ------------------------------------------------------------------
        # 2) Point cloud extent polygon (in UTM)
        #    Uses streaming hexbin filter for memory efficiency while still
        #    capturing the true data footprint (not just bounding box)
        #    (function defined earlier in from_file for use by lightweight mode)
        # ------------------------------------------------------------------
        self.epsg_utm, self.poly_utm, self.poly_4326 = _get_pointcloud_extent(self)
        self.bbox_4326 = self.poly_4326.bounds  # (min_lon, min_lat, max_lon, max_lat)

        # Clean up Pipeline 2
        gc.collect()

        # ------------------------------------------------------------------
        # Units - Enhanced with UnitInfo objects
        # ------------------------------------------------------------------
        # Parse units from PDAL metadata (srs.units.horizontal/vertical)
        self.horizontal_unit, self.vertical_unit = parse_pdal_units(_srs_md)

        # If PDAL didn't provide units, try to extract from CRS
        if self.horizontal_unit.name == "unknown" and self.original_horizontal_crs:
            self.horizontal_unit = get_horizontal_unit(self.original_horizontal_crs)

        if self.vertical_unit.name == "unknown" and self.original_vertical_crs:
            self.vertical_unit = get_vertical_unit(self.original_vertical_crs)

        # Backward compatible string properties
        self.horizontal_units = self.horizontal_unit.display_name
        self.vertical_units = self.vertical_unit.display_name

        # Clean up _srs_md
        del _srs_md

        # ------------------------------------------------------------------
        # 3) GPS time stats (and conversion to GPS seconds if needed)
        #    Uses execute_metadata_only() to avoid array serialization overhead
        # ------------------------------------------------------------------
        #
        # For original survey LAS files, we expect valid GpsTime and derive
        # an epoch. For derived products (e.g., after PROJ pipelines), GpsTime
        # may be missing, NaN, or nonsense. In that case, we *gracefully*
        # fall back to epoch=None instead of raising.
        try:
            pipeline_gps_time = pdal.Pipeline(
                json.dumps(
                    {
                        "pipeline": [
                            self.filename,  # PDAL infers readers.las
                            {
                                "type": "filters.stats",
                                "dimensions": "GpsTime",
                                "count": "true",
                            },
                        ]
                    }
                )
            )
            # Use streaming execution for memory efficiency with large files
            if hasattr(pipeline_gps_time, 'execute_streaming_metadata'):
                pipeline_gps_time.execute_streaming_metadata(chunk_size=100000)
            elif hasattr(pipeline_gps_time, 'execute_metadata_only'):
                pipeline_gps_time.execute_metadata_only()
            else:
                pipeline_gps_time.execute()

            gps_root = pipeline_gps_time.metadata
            gps_md = gps_root.get("metadata", {}).get("filters.stats", {})
            stats_list = gps_md.get("statistic", [])

            if not stats_list:
                # No GPS stats at all – treat as "no GPS time"
                self.gps_time_mean_raw = None
                self.gps_time_min_raw = None
                self.gps_time_max_raw = None
                self.gps_stddev_raw = None

                self.gps_time_mean = None
                self.gps_time_min = None
                self.gps_time_max = None
                self.gps_stddev = None

                self.decimal_year_mean_utc = None
                self.decimal_year_min_utc = None
                self.decimal_year_max_utc = None
                self.epoch = None
                gps_time_type = None

            else:
                gps_stats = stats_list[0]

                self.gps_time_mean_raw = gps_stats.get("average")
                self.gps_time_min_raw = gps_stats.get("minimum")
                self.gps_time_max_raw = gps_stats.get("maximum")
                self.gps_stddev_raw = gps_stats.get("stddev")

                raw_vals = [
                    self.gps_time_min_raw,
                    self.gps_time_max_raw,
                    self.gps_time_mean_raw,
                ]

                # If any are non-finite, treat as "no usable GPS"
                if any(
                    (v is None) or not math.isfinite(float(v))
                    for v in raw_vals
                ):
                    self.gps_time_mean = None
                    self.gps_time_min = None
                    self.gps_time_max = None
                    self.gps_stddev = None

                    self.decimal_year_mean_utc = None
                    self.decimal_year_min_utc = None
                    self.decimal_year_max_utc = None
                    self.epoch = None
                    gps_time_type = None

                else:
                    # Now we know we have finite numbers
                    self.gps_time_min_raw = float(self.gps_time_min_raw)
                    self.gps_time_max_raw = float(self.gps_time_max_raw)
                    self.gps_time_mean_raw = float(self.gps_time_mean_raw)

                    gps_time_type = _guess_in_time_from_stats(
                        vmin=self.gps_time_min_raw,
                        vmax=self.gps_time_max_raw,
                        vmean=self.gps_time_mean_raw,
                    )

                    if gps_time_type not in ("gt", "gst", "gws"):
                        # Unknown format → treat as "no epoch"
                        self.gps_time_mean = None
                        self.gps_time_min = None
                        self.gps_time_max = None
                        self.gps_stddev = None

                        self.decimal_year_mean_utc = None
                        self.decimal_year_min_utc = None
                        self.decimal_year_max_utc = None
                        self.epoch = None
                        gps_time_type = None

                    elif gps_time_type == "gws":
                        # Need to set start_date for week seconds
                        creation_year = self.creation_year
                        creation_doy = self.creation_doy
                        if creation_year is None or creation_doy is None:
                            raise ValueError(
                                "Creation year/day metadata required for GPS week seconds conversion."
                            )
                        start_date = datetime.datetime(creation_year, 1, 1) + datetime.timedelta(
                            days=creation_doy - 1
                        )
                        start_date_str = start_date.strftime("%Y-%m-%d")

                        # Convert GPS time in-place: write to temp file, then replace original
                        file_ext = os.path.splitext(self.filename)[1]
                        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                            temp_filename = tmp.name

                        try:
                            pipeline_gps_convert = pdal.Pipeline(
                                json.dumps(
                                    {
                                        "pipeline": [
                                            self.filename,
                                            {
                                                "type": "filters.sort",
                                                "dimension": "GpsTime",
                                                "order": "ASC",
                                            },
                                            {
                                                "type": "filters.gpstimeconvert",
                                                "conversion": "gws2gt",
                                                "start_date": start_date_str,
                                            },
                                            temp_filename,
                                        ]
                                    }
                                )
                            )
                            pipeline_gps_convert.execute()

                            # Replace original file with converted file
                            shutil.move(temp_filename, self.filename)

                            # Get stats from the updated file
                            pipeline_gps_converted = pdal.Pipeline(
                                json.dumps(
                                    {
                                        "pipeline": [
                                            self.filename,
                                            {
                                                "type": "filters.stats",
                                                "dimensions": "GpsTime",
                                            },
                                        ]
                                    }
                                )
                            )
                            pipeline_gps_converted.execute()
                            gps_root2 = pipeline_gps_converted.metadata
                            gps_md2 = gps_root2.get("metadata", {}).get("filters.stats", {})
                            stats_list2 = gps_md2.get("statistic", [])
                            if not stats_list2:
                                raise ValueError("No GPS time statistics found. Check GpsTime dimension.")
                            gps_stats2 = stats_list2[0]

                            self.gps_time_mean = float(gps_stats2.get("average"))
                            self.gps_time_min = float(gps_stats2.get("minimum"))
                            self.gps_time_max = float(gps_stats2.get("maximum"))
                            self.gps_stddev = float(gps_stats2.get("stddev"))
                        finally:
                            # Clean up temp file if it still exists
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)

                    elif gps_time_type == "gst":
                        # Convert GPS time in-place: write to temp file, then replace original
                        file_ext = os.path.splitext(self.filename)[1]
                        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                            temp_filename = tmp.name

                        try:
                            pipeline_gps_convert = pdal.Pipeline(
                                json.dumps(
                                    {
                                        "pipeline": [
                                            self.filename,
                                            {
                                                "type": "filters.sort",
                                                "dimension": "GpsTime",
                                                "order": "ASC",
                                            },
                                            {
                                                "type": "filters.gpstimeconvert",
                                                "conversion": "gst2gt",
                                            },
                                            temp_filename,
                                        ]
                                    }
                                )
                            )
                            pipeline_gps_convert.execute()

                            # Replace original file with converted file
                            shutil.move(temp_filename, self.filename)

                            # Get stats from the updated file
                            pipeline_gps_converted = pdal.Pipeline(
                                json.dumps(
                                    {
                                        "pipeline": [
                                            self.filename,
                                            {
                                                "type": "filters.stats",
                                                "dimensions": "GpsTime",
                                            },
                                        ]
                                    }
                                )
                            )
                            pipeline_gps_converted.execute()
                            gps_root2 = pipeline_gps_converted.metadata
                            gps_md2 = gps_root2.get("metadata", {}).get("filters.stats", {})
                            stats_list2 = gps_md2.get("statistic", [])
                            if not stats_list2:
                                raise ValueError("No GPS time statistics found. Check GpsTime dimension.")
                            gps_stats2 = stats_list2[0]

                            self.gps_time_mean = float(gps_stats2.get("average"))
                            self.gps_time_min = float(gps_stats2.get("minimum"))
                            self.gps_time_max = float(gps_stats2.get("maximum"))
                            self.gps_stddev = float(gps_stats2.get("stddev"))
                        finally:
                            # Clean up temp file if it still exists
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)

                    else:  # gps_time_type == "gt"
                        self.gps_time_mean = float(self.gps_time_mean_raw)
                        self.gps_time_min = float(self.gps_time_min_raw)
                        self.gps_time_max = float(self.gps_time_max_raw)
                        self.gps_stddev = float(self.gps_stddev_raw)

                    # If we successfully identified a GPS type, compute decimal years
                    if gps_time_type in ("gt", "gst", "gws"):
                        self.decimal_year_mean_utc = gps_seconds_to_decimal_year_utc(self.gps_time_mean)
                        self.decimal_year_min_utc = gps_seconds_to_decimal_year_utc(self.gps_time_min)
                        self.decimal_year_max_utc = gps_seconds_to_decimal_year_utc(self.gps_time_max)
                        self.epoch = self.decimal_year_mean_utc

        except Exception:
            # If anything in the GPS/epoch pipeline fails (missing GpsTime,
            # NaNs, infinities, conversion errors), fall back to "no epoch".
            self.gps_time_mean = None
            self.gps_time_min = None
            self.gps_time_max = None
            self.gps_stddev = None

            self.decimal_year_mean_utc = None
            self.decimal_year_min_utc = None
            self.decimal_year_max_utc = None
            self.epoch = None

        # Clean up GPS time pipeline
        gc.collect()

        # ------------------------------------------------------------------
        # 4) Classification stats
        #    Uses execute_metadata_only() to avoid array serialization overhead
        # ------------------------------------------------------------------
        pipeline_classification = pdal.Pipeline(
            json.dumps(
                {
                    "pipeline": [
                        self.filename,
                        {
                            "type": "filters.stats",
                            "dimensions": "Classification",
                            "enumerate": "Classification",
                            "count": "Classification",
                        },
                    ]
                }
            )
        )
        # Use streaming execution for memory efficiency with large files
        if hasattr(pipeline_classification, 'execute_streaming_metadata'):
            pipeline_classification.execute_streaming_metadata(chunk_size=100000)
        elif hasattr(pipeline_classification, 'execute_metadata_only'):
            pipeline_classification.execute_metadata_only()
        else:
            pipeline_classification.execute()
        class_root = pipeline_classification.metadata
        class_md = class_root.get("metadata", {}).get("filters.stats", {})
        class_stats_list = class_md.get("statistic", [])
        if class_stats_list:
            bins = class_stats_list[0].get("bins", {})
        else:
            bins = {}

        class_values = [int(float(k)) for k in bins.keys()]
        class_counts = list(bins.values())

        self.classification = bins
        self.class_values = class_values
        self.class_counts = class_counts

        # Determine if ground points have been classified
        self.has_ground_class = 2 in self.class_values

        # Clean up Classification pipeline
        del pipeline_classification, class_root, class_md, class_stats_list, bins
        gc.collect()

        # ------------------------------------------------------------------
        # 5) Initialize CRS history object for this point cloud
        # ------------------------------------------------------------------
        try:
            from .crs_history import CRSHistory  # local import to avoid circulars

            if getattr(self, "crs_history", None) is None:
                self.crs_history = CRSHistory(self)
        except Exception:
            # Don't break loading if CRSHistory construction fails
            self.crs_history = None

    # -------------------------------------------------------------------------
    # Unit conversion methods
    # -------------------------------------------------------------------------
    def convert_z_to_meters(self, z_values: np.ndarray) -> np.ndarray:
        """
        Convert Z values from the point cloud's vertical unit to meters.
        
        Parameters
        ----------
        z_values : np.ndarray
            Z values in the point cloud's vertical unit
            
        Returns
        -------
        np.ndarray
            Z values converted to meters
            
        Examples
        --------
        >>> pc = PointCloud("lidar.las")
        >>> pc.from_file()
        >>> z_meters = pc.convert_z_to_meters(pc.get_z_values())
        """
        if self.vertical_unit.name == "unknown":
            # If unit is unknown, assume meters and warn
            import warnings
            warnings.warn(
                "Vertical unit is unknown, assuming meters. "
                "Use pc.vertical_unit = unit_utils.lookup_unit('foot') to set manually."
            )
            return z_values
        return convert_length(z_values, self.vertical_unit, METER)
    
    def convert_z_from_meters(self, z_meters: np.ndarray) -> np.ndarray:
        """
        Convert Z values from meters to the point cloud's vertical unit.
        
        Parameters
        ----------
        z_meters : np.ndarray
            Z values in meters
            
        Returns
        -------
        np.ndarray
            Z values in the point cloud's vertical unit
        """
        if self.vertical_unit.name == "unknown":
            import warnings
            warnings.warn("Vertical unit is unknown, assuming meters.")
            return z_meters
        return convert_length(z_meters, METER, self.vertical_unit)
    
    def get_z_conversion_factor(self, target_unit: Union[str, UnitInfo] = "meter") -> float:
        """
        Get the factor to multiply Z values by to convert to target unit.
        
        Parameters
        ----------
        target_unit : str or UnitInfo
            Target unit (default: "meter")
            
        Returns
        -------
        float
            Conversion factor
            
        Examples
        --------
        >>> pc.get_z_conversion_factor("meter")
        0.3048  # if vertical unit is feet
        """
        if isinstance(target_unit, str):
            target = lookup_unit(target_unit)
            if target is None:
                raise ValueError(f"Unknown unit: {target_unit}. Use 'meter', 'foot', etc.")
        else:
            target = target_unit
        
        if self.vertical_unit.name == "unknown":
            return 1.0
        
        return get_conversion_factor(self.vertical_unit, target)
    
    def are_units_metric(self) -> Tuple[bool, bool]:
        """
        Check if horizontal and vertical units are metric (meter-based).
        
        Returns
        -------
        tuple[bool, bool]
            (horizontal_is_metric, vertical_is_metric)
        """
        h_metric = self.horizontal_unit.name in ("meter", "kilometer", "centimeter", "millimeter")
        v_metric = self.vertical_unit.name in ("meter", "kilometer", "centimeter", "millimeter")
        return h_metric, v_metric

    # -------------------------------------------------------------------------
    # Pretty printing
    # -------------------------------------------------------------------------
    def print_metadata(self) -> None:
        """Print point cloud metadata in table format."""
        print("\n--- CRS Information ---")
        print(f"{'Property':<25} {'Value':<50}")
        print("-" * 75)
        print(f"{'Compound CRS':<25} {str(self.original_compound_crs)[:48]}")
        print(f"{'Horizontal CRS':<25} {str(self.original_horizontal_crs)[:48]}")
        print(f"{'Vertical CRS':<25} {str(self.original_vertical_crs)[:48]}")
        print(f"{'Geoid model':<25} {str(self.geoid_model)[:48]}")

        print("\n--- Point Cloud Metadata ---")
        print(f"{'Property':<25} {'Value':<50}")
        print("-" * 75)
        print(f"{'Total points':<25} {self.total_points:,}")
        print(f"{'Bounds':<25} {self.bounds}")
        h_unit_str = f"{self.horizontal_unit}"
        if self.horizontal_unit.epsg_code:
            h_unit_str += f" (EPSG:{self.horizontal_unit.epsg_code})"
        print(f"{'Horizontal units':<25} {h_unit_str}")
        v_unit_str = f"{self.vertical_unit}"
        if self.vertical_unit.epsg_code:
            v_unit_str += f" (EPSG:{self.vertical_unit.epsg_code})"
        print(f"{'Vertical units':<25} {v_unit_str}")

        print("\n--- Time Information ---")
        print(f"{'Property':<25} {'Value':<50}")
        print("-" * 75)
        print(f"{'Creation year':<25} {self.creation_year}")
        print(f"{'Creation DOY':<25} {self.creation_doy}")
        print(f"{'GPS time mean (raw)':<25} {self.gps_time_mean_raw}")
        print(f"{'GPS time min (raw)':<25} {self.gps_time_min_raw}")
        print(f"{'GPS time max (raw)':<25} {self.gps_time_max_raw}")
        print(f"{'GPS stddev (raw)':<25} {self.gps_stddev_raw}")
        print(f"{'GPS time mean':<25} {self.gps_time_mean}")
        print(f"{'GPS time min':<25} {self.gps_time_min}")
        print(f"{'GPS time max':<25} {self.gps_time_max}")
        print(f"{'GPS stddev':<25} {self.gps_stddev}")
        print(f"{'Decimal year mean (UTC)':<25} {self.decimal_year_mean_utc}")
        print(f"{'Decimal year min (UTC)':<25} {self.decimal_year_min_utc}")
        print(f"{'Decimal year max (UTC)':<25} {self.decimal_year_max_utc}")
        print(f"{'Epoch':<25} {self.epoch}")

        print("\n--- Classification Information ---")
        print(f"{'Property':<25} {'Value':<50}")
        print("-" * 75)
        print(f"{'Classification bins':<25} {self.classification}")
        print(f"{'Class values':<25} {self.class_values}")
        print(f"{'Class counts':<25} {self.class_counts}")

    def print_unit_info(self) -> None:
        """Print detailed information about the point cloud's units."""
        print("\n--- Unit Information ---")
        print(f"Horizontal: {describe_unit(self.horizontal_unit)}")
        print(f"Vertical:   {describe_unit(self.vertical_unit)}")

        h_metric, v_metric = self.are_units_metric()
        print(f"Horizontal is metric: {h_metric}")
        print(f"Vertical is metric:   {v_metric}")

        if self.vertical_unit.name != "meter":
            factor = self.get_z_conversion_factor("meter")
            print(f"Z to meters factor:   {factor:.10f}")

    # -------------------------------------------------------------------------
    # Metadata editing
    # -------------------------------------------------------------------------
    def add_metadata(
        self,
        compound_CRS: Optional[Any] = None,
        horizontal_CRS: Optional[Any] = None,
        vertical_CRS: Optional[Any] = None,
        geoid_model: Optional[str] = None,
        epoch: Optional[Any] = None,
    ) -> None:
        """
        Add or update CRS, geoid, and temporal metadata.

        You can pass:
          - compound_CRS: a full compound CRS, or a purely horizontal or purely
            vertical CRS. The function will inspect it and decide whether it is
            horizontal, vertical, or truly compound.
          - horizontal_CRS: horizontal CRS only
          - vertical_CRS: vertical CRS only

        Any combination is allowed:
          - compound only
          - horizontal only
          - vertical only
          - compound + horizontal (horizontal overrides compound's horizontal)
          - compound + vertical (vertical overrides compound's vertical)
          - horizontal + vertical (compound is built from them)
          - compound + horizontal + vertical (horizontal/vertical override)

        If only horizontal and/or vertical are given, a compound CRS is
        generated from them when possible.

        `epoch` can be:
          - float/int: decimal year
          - datetime/date: a single epoch date
          - string: single date ("2006-04-06", "04/06/2006", etc.)
          - string range: "04/06/2006 - 05/01/2006"
          - (start, end): 2-element iterable of dates/datetimes/decimal years
        """

        # -------------------------------
        # Helper: safely coerce to CRS
        # -------------------------------
        def _crs_or_none(value: Any) -> Optional[CRS_]:
            if value is None:
                return None
            try:
                return _ensure_crs_obj(value)
            except Exception:
                return None

        # -------------------------------
        # 1. Start from existing state
        # -------------------------------
        existing_horiz = _crs_or_none(
            getattr(self, "current_horizontal_crs", None)
            or getattr(self, "original_horizontal_crs", None)
        )
        existing_vert = _crs_or_none(
            getattr(self, "current_vertical_crs", None)
            or getattr(self, "original_vertical_crs", None)
        )
        existing_comp = _crs_or_none(
            getattr(self, "current_compound_crs", None)
            or getattr(self, "original_compound_crs", None)
        )

        new_horiz = existing_horiz
        new_vert = existing_vert
        new_comp = existing_comp

        crs_changed = False
        geoid_changed = False
        epoch_changed = False

        # -------------------------------
        # 2. Interpret compound_CRS, if any
        # -------------------------------
        # Helper to check if value is meaningful (not None, not empty string)
        def _is_valid_crs_input(val: Any) -> bool:
            if val is None:
                return False
            if isinstance(val, str) and not val.strip():
                return False
            return True

        if _is_valid_crs_input(compound_CRS):
            comp = _ensure_crs_obj(compound_CRS)

            if comp.is_compound:
                # Try to split into horizontal + vertical
                sub = getattr(comp, "sub_crs_list", None) or []
                horiz_candidate = sub[0] if len(sub) >= 1 else None
                vert_candidate = sub[1] if len(sub) >= 2 else None

                if horiz_candidate is not None:
                    new_horiz = horiz_candidate
                if vert_candidate is not None:
                    new_vert = vert_candidate

                new_comp = comp
                crs_changed = True

            elif is_3d_geographic_crs(comp):
                # 3D geographic CRS (e.g., EPSG:4979) - use directly as full CRS
                # and derive a synthetic 1D vertical CRS for the vertical_crs attribute
                new_comp = comp
                new_vert = extract_ellipsoidal_height_as_vertical_crs(comp)
                # Horizontal stays as-is (user may have set it separately)
                crs_changed = True

            else:
                # Non-compound: decide whether horizontal or vertical
                if getattr(comp, "is_vertical", False):
                    new_vert = comp
                else:
                    # Geographic or projected -> horizontal
                    new_horiz = comp
                # Compound will be rebuilt later from horiz/vert
                new_comp = None
                crs_changed = True

        # -------------------------------
        # 3. Explicit horizontal / vertical overrides
        # -------------------------------
        if _is_valid_crs_input(horizontal_CRS):
            new_horiz = _ensure_crs_obj(horizontal_CRS)
            new_comp = None
            crs_changed = True

        if _is_valid_crs_input(vertical_CRS):
            vert_candidate = _ensure_crs_obj(vertical_CRS)
            
            # Check if user passed a 3D geographic CRS (e.g., EPSG:4979) as vertical
            if is_3d_geographic_crs(vert_candidate):
                # Use the 3D CRS as the full CRS, derive synthetic 1D vertical
                new_comp = vert_candidate
                new_vert = extract_ellipsoidal_height_as_vertical_crs(vert_candidate)
            else:
                new_vert = vert_candidate
                new_comp = None
            crs_changed = True

        # -------------------------------
        # 4. Rebuild compound if needed
        # -------------------------------
        if new_comp is None:
            if new_horiz is not None and new_vert is not None:
                # Check if new_vert is actually a 3D geographic CRS
                # (shouldn't happen after section 3, but defensive check)
                if is_3d_geographic_crs(new_vert):
                    # Use the 3D CRS directly as the full CRS
                    new_comp = new_vert
                    new_vert = extract_ellipsoidal_height_as_vertical_crs(new_vert)
                else:
                    # Normal case: build true compound from 2D + 1D
                    comp_name = f"{new_horiz.name} + {new_vert.name}"
                    new_comp = CompoundCRS(name=comp_name, components=[new_horiz, new_vert])
            elif new_horiz is not None:
                # Horizontal only
                new_comp = new_horiz
            elif new_vert is not None:
                # Vertical only - check if it's actually a 3D CRS
                if is_3d_geographic_crs(new_vert):
                    new_comp = new_vert
                    new_vert = extract_ellipsoidal_height_as_vertical_crs(new_vert)
                else:
                    new_comp = new_vert
            else:
                new_comp = existing_comp  # nothing better to do

        # -------------------------------
        # 5. Write back CRS to PointCloud
        # -------------------------------
        if new_comp is not None:
            self.current_compound_crs = new_comp.to_wkt()
        if new_horiz is not None:
            self.current_horizontal_crs = new_horiz.to_wkt()
            # Update horizontal unit from new CRS
            self.horizontal_unit = get_horizontal_unit(new_horiz)
            self.horizontal_units = self.horizontal_unit.display_name
        if new_vert is not None:
            self.current_vertical_crs = new_vert.to_wkt()
            # Update vertical unit from new CRS
            self.vertical_unit = get_vertical_unit(new_vert)
            self.vertical_units = self.vertical_unit.display_name

        # Orthometric flag can update when vertical changes
        if new_vert is not None:
            self.is_orthometric = is_orthometric(new_vert.to_wkt())

        # -------------------------------
        # 6. Geoid model
        # -------------------------------
        if geoid_model is not None:
            gm_str = str(geoid_model)

            # Case A: looks like a file path or filename (e.g., "us_noaa_geoid03_conus.tif")
            #         -> just store the basename, do NOT call select_geoid_grid again.
            # Case B: looks like an alias (e.g., "GEOID03", "GEOID18")
            #         -> resolve to a grid path via select_geoid_grid.
            if gm_str.lower().endswith(".tif") or "/" in gm_str or "\\" in gm_str:
                self.geoid_model = Path(gm_str).name
            else:
                selected_geoid, _ = select_geoid_grid(gm_str, verbose=False)
                self.geoid_model = Path(selected_geoid).name

            geoid_changed = True

        # -------------------------------
        # 7. Epoch handling
        # -------------------------------
        if epoch is not None:

            def _epoch_to_decimal_year(value: Any) -> float:
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, datetime.datetime):
                    return _datetime_to_decimal_year(value)
                if isinstance(value, datetime.date):
                    dt = datetime.datetime.combine(value, datetime.time())
                    return _datetime_to_decimal_year(dt)
                if isinstance(value, str):
                    parsed = _parse_epoch_string_to_decimal(value)
                    if isinstance(parsed, tuple):
                        # If string itself is a range, we take the mid-point.
                        return 0.5 * (parsed[0] + parsed[1])
                    return float(parsed)
                raise TypeError(
                    "epoch must be a float/int (decimal year), a datetime/date, "
                    "a string (date or 'start - end'), or a 2-element range of those."
                )

            # String range: "start - end"
            if isinstance(epoch, str):
                parsed = _parse_epoch_string_to_decimal(epoch)
                if isinstance(parsed, tuple):
                    start_dec, end_dec = parsed
                    self.epoch_start = start_dec
                    self.epoch_end = end_dec
                    self.epoch = 0.5 * (start_dec + end_dec)
                else:
                    epoch_dec = float(parsed)
                    self.epoch = epoch_dec
                    self.epoch_start = epoch_dec
                    self.epoch_end = epoch_dec
            # Tuple/list range: (start, end)
            elif isinstance(epoch, (list, tuple)) and len(epoch) == 2:
                start_dec = _epoch_to_decimal_year(epoch[0])
                end_dec = _epoch_to_decimal_year(epoch[1])
                self.epoch_start = min(start_dec, end_dec)
                self.epoch_end = max(start_dec, end_dec)
                self.epoch = 0.5 * (self.epoch_start + self.epoch_end)
            else:
                # Single value (numeric, date, datetime, etc.)
                epoch_dec = _epoch_to_decimal_year(epoch)
                self.epoch = epoch_dec
                self.epoch_start = epoch_dec
                self.epoch_end = epoch_dec

            epoch_changed = True

        # -------------------------------
        # 8. Record a single CRSHistory entry summarizing all changes
        # -------------------------------
        if getattr(self, "crs_history", None) is not None:
            # Only pass updated pieces; CRSHistory keeps its own current state.
            self.crs_history.add_manual_change_entry(
                new_compound_crs_proj=new_comp if crs_changed else None,
                new_horizontal_crs_proj=new_horiz if crs_changed else None,
                new_vertical_crs_proj=new_vert if crs_changed else None,
                geoid_model=self.geoid_model if geoid_changed else None,
                epoch=self.epoch if epoch_changed else None,
                note="PointCloud.add_metadata manual update.",
            )

    def set_units(
        self,
        horizontal_unit: Optional[Union[str, UnitInfo]] = None,
        vertical_unit: Optional[Union[str, UnitInfo]] = None,
    ) -> None:
        """
        Manually set the horizontal and/or vertical units.
        
        Use this when the units aren't correctly detected from the file metadata.
        
        Parameters
        ----------
        horizontal_unit : str or UnitInfo, optional
            Horizontal unit name (e.g., "meter", "us_survey_foot") or UnitInfo object
        vertical_unit : str or UnitInfo, optional
            Vertical unit name or UnitInfo object
            
        Examples
        --------
        >>> pc.set_units(vertical_unit="us_survey_foot")
        >>> pc.set_units(horizontal_unit="foot", vertical_unit="foot")
        """
        if horizontal_unit is not None:
            if isinstance(horizontal_unit, str):
                unit = lookup_unit(horizontal_unit)
                if unit is None:
                    raise ValueError(f"Unknown horizontal unit: {horizontal_unit}")
                self.horizontal_unit = unit
            else:
                self.horizontal_unit = horizontal_unit
            self.horizontal_units = self.horizontal_unit.display_name
        
        if vertical_unit is not None:
            if isinstance(vertical_unit, str):
                unit = lookup_unit(vertical_unit)
                if unit is None:
                    raise ValueError(f"Unknown vertical unit: {vertical_unit}")
                self.vertical_unit = unit
            else:
                self.vertical_unit = vertical_unit
            self.vertical_units = self.vertical_unit.display_name

    # -------------------------------------------------------------------------
    # Clipping / Cropping
    # -------------------------------------------------------------------------
    def clip_to_polygon(
        self,
        polygon: Union["Polygon", str],
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = True,
    ) -> "PointCloud":
        """
        Clip point cloud to a polygon boundary.

        Uses PDAL filters.crop to extract points within the polygon.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon or str
            Polygon to clip to. Can be a shapely Polygon object or a WKT string.
        output_path : str or Path, optional
            Output file path. If None, generates path based on input filename.
        overwrite : bool, default True
            Whether to overwrite existing output file.

        Returns
        -------
        PointCloud
            New PointCloud containing only points within the polygon.

        Examples
        --------
        >>> from shapely.geometry import box
        >>> bbox = box(500000, 4000000, 501000, 4001000)
        >>> clipped = pc.clip_to_polygon(bbox)
        """
        from shapely.geometry import Polygon as ShapelyPolygon

        # Convert polygon to WKT if needed
        if isinstance(polygon, str):
            polygon_wkt = polygon
        elif hasattr(polygon, 'wkt'):
            polygon_wkt = polygon.wkt
        else:
            raise TypeError(
                f"polygon must be a shapely Polygon or WKT string, got {type(polygon)}"
            )

        # Generate output path if not provided
        src_path = Path(self.filename)
        if output_path is None:
            output_path = src_path.with_name(src_path.stem + "_clipped" + src_path.suffix)
        else:
            output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists and overwrite=False: {output_path}")

        # Build PDAL pipeline
        pipeline_spec = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": str(self.filename),
                },
                {
                    "type": "filters.crop",
                    "polygon": polygon_wkt,
                },
                {
                    "type": "writers.las",
                    "filename": str(output_path),
                },
            ]
        }

        pipe = pdal.Pipeline(json.dumps(pipeline_spec))
        # Use streaming execution for memory efficiency - this pipeline writes to
        # a file and doesn't need array data returned to Python
        count = pipe.execute_streaming(chunk_size=1000000)

        if count == 0:
            import warnings
            warnings.warn(
                f"No points found within the clip polygon. Output file may be empty."
            )

        # Create new PointCloud from output
        clipped_pc = PointCloud(str(output_path))
        clipped_pc.from_file(lightweight=True)

        # Copy metadata from source
        clipped_pc.current_compound_crs = self.current_compound_crs
        clipped_pc.current_horizontal_crs = self.current_horizontal_crs
        clipped_pc.current_vertical_crs = self.current_vertical_crs
        clipped_pc.geoid_model = getattr(self, 'geoid_model', None)
        clipped_pc.epoch = getattr(self, 'epoch', None)
        clipped_pc.is_orthometric = getattr(self, 'is_orthometric', None)
        clipped_pc.horizontal_unit = self.horizontal_unit
        clipped_pc.vertical_unit = self.vertical_unit

        return clipped_pc

    # -------------------------------------------------------------------------
    # DEM creation
    # -------------------------------------------------------------------------
    def create_dem(
        self,
        output_path: Union[str, os.PathLike],
        dem_type: str = "dtm",
        resolution: float = 1.0,
        interpolation: str = "idw",
        classification_filter: Optional[Union[str, List[int], Set[int]]] = "auto",
        use_smrf: bool = False,
        smrf_params: Optional[Dict[str, Any]] = None,
        hole_filling: bool = False,
        hole_filling_method: str = "interpolation",
        output_crs: Optional[Union[str, CRS_]] = None,
        create_cog: bool = False,
        gdal_options: Optional[Dict[str, Any]] = None,
        window_size: Optional[int] = None,
        power: float = 2.0,
        radius: Optional[float] = None,
        overwrite: bool = False,
    ) -> "Raster":
        """
        Create advanced DEM with comprehensive options.

        This method provides full control over DEM generation including:
        - DTM vs DSM creation
        - Multiple interpolation methods (TIN, IDW, etc.)
        - SMRF ground classification
        - Hole filling
        - COG output
        - Custom output CRS
        """

        from .raster import Raster  # local import to avoid circulars

        # Caching: if output file exists and overwrite=False, load and return it
        if os.path.exists(output_path) and not overwrite:
            import sys
            print(f"Loading existing DEM: {os.path.basename(str(output_path))}", file=sys.stderr)
            return Raster.from_file(str(output_path), rtype=dem_type, metadata={})

        # Known interpolation keywords (not enforced, kept for reference)
        valid_interpolations = {
            "tin",
            "idw",
            "bilinear",
            "min",
            "max",
            "mean",
            "count",
            "stdev",
            "stddev",
            "variance",
            "range",
        }

        # Build pipeline
        pipeline_steps: List[Any] = []
        pipeline_steps.append(
            {
                "type": "readers.las",
                "filename": self.filename,
            }
        )

        # Apply SMRF if requested
        if use_smrf:
            defaults = {
                "cell": 1.0,
                "scalar": 1.25,
                "slope": 0.15,
                "threshold": 0.5,
                "window": 18.0,
            }
            params = {**defaults, **(smrf_params or {})}
            smrf_filter = {"type": "filters.smrf", **params}
            pipeline_steps.append(smrf_filter)

        # Classification filtering
        if classification_filter is not None:
            if classification_filter == "auto":
                if dem_type == "dtm":
                    # Ground only
                    filter_expr = "Classification[2:2]"
                elif dem_type == "dsm":
                    # First returns for any class 1–65
                    filter_expr = "Classification[1:65],ReturnNumber[1:1]"
                else:
                    filter_expr = None
            elif isinstance(classification_filter, (int, list, set, tuple)):
                # Normalize to a list of unique integers (preserve order)
                classes_seq = (
                    [classification_filter]
                    if isinstance(classification_filter, int)
                    else list(classification_filter)
                )
                seen_cls: Set[int] = set()
                classes: List[int] = []
                for c in classes_seq:
                    try:
                        ci = int(c)
                    except (TypeError, ValueError):
                        continue
                    if ci not in seen_cls:
                        seen_cls.add(ci)
                        classes.append(ci)
                filter_expr = (
                    ",".join(f"Classification[{c}:{c}]" for c in classes)
                    if classes
                    else None
                )
            else:
                filter_expr = None

            if filter_expr:
                pipeline_steps.append(
                    {
                        "type": "filters.range",
                        "limits": filter_expr,
                    }
                )

        # Reprojection if needed
        if output_crs is not None:
            target_crs = CRS_.from_user_input(output_crs)
            # Compare using CRS objects (current_horizontal_crs is WKT/string)
            try:
                current_horiz = (
                    CRS_.from_user_input(self.current_horizontal_crs)
                    if self.current_horizontal_crs
                    else None
                )
            except Exception:
                current_horiz = None

            if current_horiz is None or target_crs != current_horiz:
                pipeline_steps.append(
                    {
                        "type": "filters.reprojection",
                        "out_srs": target_crs.to_wkt(),
                    }
                )

        # Decide output path used by PDAL writer
        output_path = Path(output_path)
        temp_path = (
            str(output_path)
            if not (hole_filling or create_cog)
            else str(output_path) + ".tmp.tif"
        )

        # Consistent NoData used for outputs we create directly here
        nodata_value = -9999.0

        # ------------------------------------------------------------------
        # TIN path: use filters.delaunay + filters.faceraster + writers.raster
        # ------------------------------------------------------------------
        if interpolation == "tin":
            pipeline_steps.append({"type": "filters.delaunay"})
            pipeline_steps.append(
                {
                    "type": "filters.faceraster",
                    "resolution": float(resolution),
                    "max_triangle_edge_length": 2 * float(resolution),
                    "nodata": nodata_value,
                }
            )
            pipeline_steps.append(
                {
                    "type": "writers.raster",
                    "filename": temp_path,
                    "gdaldriver": "GTiff",
                    "data_type": "float32",
                    "nodata": nodata_value,
                    "gdalopts": "TILED=YES,COMPRESS=DEFLATE,BIGTIFF=IF_SAFER",
                }
            )

        else:
            # ------------------------------------------------------------------
            # Non-TIN path: writers.gdal
            # ------------------------------------------------------------------
            writer_options: Dict[str, Any] = {
                "type": "writers.gdal",
                "filename": temp_path,
                "resolution": float(resolution),
                "output_type": interpolation,
                "data_type": "float32",
                # Ensure NoData tag is present so post-processing can act on it
                "nodata": nodata_value,
            }

            # IDW parameters
            if interpolation == "idw":
                if window_size is not None:
                    writer_options["window_size"] = int(window_size)
                if power != 2.0:
                    writer_options["power"] = float(power)
                if radius is not None:
                    writer_options["radius"] = float(radius)
            elif window_size is not None:
                writer_options["window_size"] = int(window_size)

            # Stats-type interpolations
            if interpolation in {
                "min",
                "max",
                "mean",
                "count",
                "stdev",
                "stddev",
                "variance",
                "range",
            }:
                writer_options["output_type"] = interpolation
                if window_size is not None:
                    writer_options["window_size"] = int(window_size)
            elif interpolation == "bilinear":
                # NOTE: This assumes PDAL/writers.gdal accepts 'bilinear' as output_type.
                # If PDAL version does not, this will raise at pipeline execution.
                writer_options["output_type"] = "bilinear"
                if window_size is not None:
                    writer_options["window_size"] = int(window_size)
                else:
                    writer_options["window_size"] = 2

            # GDAL options
            base_gdal_options = {
                "TILED": "YES",
                "COMPRESS": "DEFLATE",
                "PREDICTOR": "2",
                "ZLEVEL": "9",
                "BIGTIFF": "IF_SAFER",
            }
            if gdal_options:
                base_gdal_options.update(gdal_options)

            writer_options["gdalopts"] = ",".join(
                f"{k}={v}" for k, v in base_gdal_options.items()
            )

            pipeline_steps.append(writer_options)

        # Execute pipeline
        try:
            p = pdal.Pipeline(json.dumps({"pipeline": pipeline_steps}))
            points_processed = p.execute()
        except Exception as e:
            raise RuntimeError(f"PDAL pipeline failed: {e}")

        # ------------------------------------------------------------------
        # Normalize NaN/NoData for non-TIN outputs when not delegating to
        # _postprocess_dem.
        # ------------------------------------------------------------------
        if interpolation != "tin" and not (hole_filling or create_cog):
            if has_rasterio():
                with rasterio.open(temp_path) as src:
                    data = src.read(1).astype("float32", copy=False)
                    profile = src.profile.copy()
                    existing_nodata = src.nodata

                # Choose the output nodata we will enforce
                out_nodata = (
                    existing_nodata
                    if (existing_nodata is not None and not np.isnan(existing_nodata))
                    else nodata_value
                )

                # Replace NaNs with NoData
                if np.issubdtype(data.dtype, np.floating):
                    nan_mask = np.isnan(data)
                    if np.any(nan_mask):
                        data = data.copy()
                        data[nan_mask] = out_nodata

                # Ensure profile nodata is set
                profile["nodata"] = out_nodata
                # Enable BIGTIFF for large files
                profile["BIGTIFF"] = "IF_SAFER"

                # Rewrite to the same path atomically
                tmp_fix = f"{temp_path}.nodatatmp.tif"
                with rasterio.open(tmp_fix, "w", **profile) as dst:
                    dst.write(data, 1)
                os.replace(tmp_fix, temp_path)
            else:
                # If rasterio is not available, writers.gdal at least wrote with 'nodata'
                pass

        # Post-processing (hole filling / COG)
        if hole_filling or create_cog:
            self._postprocess_dem(
                temp_path,
                str(output_path),
                hole_filling=hole_filling,
                hole_filling_method=hole_filling_method,
                create_cog=create_cog,
            )

        # Create raster object with metadata
        dem = Raster.from_file(
            str(output_path),
            rtype=dem_type,
            metadata={
                "source_pointcloud": self.filename,
                "boundary_polygon": self.poly_utm,
                "utm_crs": self.epsg_utm,
                "interpolation_method": interpolation,
                "classification_filter": str(classification_filter),
                "used_smrf": use_smrf,
                "hole_filled": hole_filling,
                "is_cog": create_cog,
                "points_processed": points_processed,
                "resolution": resolution,
            },
        )

        # Inherit units from source point cloud
        if hasattr(self, 'vertical_unit') and self.vertical_unit is not None:
            dem.current_vertical_unit = self.vertical_unit
            dem.original_vertical_unit = self.vertical_unit
            if hasattr(self, 'vertical_units') and self.vertical_units:
                dem.current_vertical_units = self.vertical_units
                dem.original_vertical_units = self.vertical_units
            else:
                dem.current_vertical_units = self.vertical_unit.display_name
                dem.original_vertical_units = self.vertical_unit.display_name
        if hasattr(self, 'horizontal_unit') and self.horizontal_unit is not None:
            dem.current_horizontal_unit = self.horizontal_unit
            dem.original_horizontal_unit = self.horizontal_unit
            if hasattr(self, 'horizontal_units') and self.horizontal_units:
                dem.current_horizontal_units = self.horizontal_units
                dem.original_horizontal_units = self.horizontal_units
            else:
                dem.current_horizontal_units = self.horizontal_unit.display_name
                dem.original_horizontal_units = self.horizontal_unit.display_name

        # Inherit additional metadata from source point cloud
        dem.epoch = getattr(self, 'epoch', None)
        dem.current_geoid_model = getattr(self, 'geoid_model', None)
        dem.original_geoid_model = getattr(self, 'geoid_model', None)
        dem.current_vertical_crs = getattr(self, 'current_vertical_crs', None)
        dem.original_vertical_crs = getattr(self, 'original_vertical_crs', None)
        dem.is_orthometric = getattr(self, 'is_orthometric', None)

        return dem

    # -------------------------------------------------------------------------
    # DEM post-processing
    # -------------------------------------------------------------------------
    def _postprocess_dem(
        self,
        input_path: str,
        output_path: str,
        hole_filling: bool = False,
        hole_filling_method: str = "interpolation",
        create_cog: bool = False,
    ) -> None:
        """Post-process DEM for hole filling and COG creation."""

        with rasterio.open(input_path) as src:
            data = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata

        # Normalize NoData to NaN for processing
        data = data.astype("float32", copy=False)
        if nodata is not None and not np.isnan(nodata):
            nodata_mask = data == nodata
            if np.any(nodata_mask):
                data = data.copy()
                data[nodata_mask] = np.nan

        # Hole filling
        if hole_filling:
            if hole_filling_method == "interpolation":
                # rasterio.fill.fillnodata expects mask=True where values are valid
                valid_mask = ~np.isnan(data)
                filled = fillnodata(data, mask=valid_mask, max_search_distance=100.0)
                data = filled
            elif hole_filling_method in ["mean", "median", "min", "max"] and has_scipy():
                from scipy.ndimage import generic_filter

                def fill_func(values: np.ndarray) -> float:
                    valid = values[~np.isnan(values)]
                    if len(valid) == 0:
                        return float("nan")
                    if hole_filling_method == "mean":
                        return float(np.mean(valid))
                    if hole_filling_method == "median":
                        return float(np.median(valid))
                    if hole_filling_method == "min":
                        return float(np.min(valid))
                    # max
                    return float(np.max(valid))

                mask = np.isnan(data)
                if np.any(mask):
                    filled = generic_filter(
                        data,
                        fill_func,
                        size=3,
                        mode="constant",
                        cval=np.nan,
                    )
                    data[mask] = filled[mask]

        # Update profile for COG
        if create_cog:
            profile.update(
                {
                    "driver": "GTiff",
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                    "compress": "deflate",
                    "predictor": 2,
                    "ZLEVEL": 9,
                    "BIGTIFF": "IF_SAFER",
                }
            )

        # Convert NaNs back to nodata value if nodata is defined
        out_data = data
        if nodata is not None and not np.isnan(nodata):
            nan_mask = np.isnan(out_data)
            if np.any(nan_mask):
                out_data = out_data.copy()
                out_data[nan_mask] = nodata
            profile["nodata"] = nodata  # ensure nodata is preserved in output

        # Write output
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out_data, 1)

            # Build overviews for COG
            if create_cog:
                factors = [2, 4, 8, 16, 32]
                dst.build_overviews(factors, Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")

    def _copy_metadata_attributes(self, target_pc: 'PointCloud') -> None:
        """
        Copy metadata attributes from this point cloud to a target point cloud.
        Used after transformations to preserve metadata that isn't automatically
        transferred by PDAL operations.
        
        Parameters
        ----------
        target_pc : PointCloud
            The target point cloud to copy attributes to.
        """
        # GPS time and epoch information
        for attr in ['gps_time_mean', 'gps_time_min', 'gps_time_max', 'gps_stddev',
                     'gps_time_mean_raw', 'gps_time_min_raw', 'gps_time_max_raw', 'gps_stddev_raw',
                     'decimal_year_mean_utc', 'decimal_year_min_utc', 'decimal_year_max_utc',
                     'creation_doy', 'creation_year']:
            if hasattr(self, attr):
                setattr(target_pc, attr, getattr(self, attr))
        
        # Classification information
        for attr in ['classification', 'class_values', 'class_counts', 'has_ground_class']:
            if hasattr(self, attr):
                setattr(target_pc, attr, getattr(self, attr))
        
        # Preserve is_orthometric flag if not already set correctly
        if hasattr(self, 'is_orthometric') and not hasattr(target_pc, 'is_orthometric'):
            target_pc.is_orthometric = self.is_orthometric
        
        # Preserve unit info objects - source is authoritative for transformations
        # that don't explicitly change units. Always copy if source has known units.
        if hasattr(self, 'horizontal_unit') and self.horizontal_unit.name != "unknown":
            target_pc.horizontal_unit = self.horizontal_unit
            target_pc.horizontal_units = self.horizontal_unit.display_name
        if hasattr(self, 'vertical_unit') and self.vertical_unit.name != "unknown":
            target_pc.vertical_unit = self.vertical_unit
            target_pc.vertical_units = self.vertical_unit.display_name

        # Preserve CRS attributes if not set on target (fallback for metadata propagation)
        # Only copy if source has a valid value and target doesn't
        def _has_valid_crs(obj, attr):
            val = getattr(obj, attr, None)
            return val is not None and (not isinstance(val, str) or val.strip())

        crs_attrs = [
            'current_vertical_crs', 'original_vertical_crs',
            'current_horizontal_crs', 'original_horizontal_crs',
            'current_compound_crs', 'original_compound_crs',
            'geoid_model',
        ]
        for attr in crs_attrs:
            if _has_valid_crs(self, attr) and not _has_valid_crs(target_pc, attr):
                setattr(target_pc, attr, getattr(self, attr))

    def warp_pointcloud(
        self,
        target_horizontal_crs: Optional[Any] = None,
        target_compound_crs: Optional[Any] = None,
        target_horizontal_units: Optional[str] = None,
        target_vertical_units: Optional[str] = None,
        source_vertical_kind: Optional[str] = None,
        target_vertical_kind: Optional[str] = None,
        source_geoid_model: Optional[str] = None,
        target_geoid_model: Optional[str] = None,
        dynamic_target_epoch: Optional[float] = None,
        dynamic_target_crs_proj: Optional[Any] = None,
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        return_pipeline: bool = False,
        velocity_model_path: Optional[Union[str, Path]] = None,
    ):
        """
        Unified warp method that handles horizontal, vertical, and epoch transforms.

        When multiple transformation types are requested, they are composed into
        a SINGLE PROJ pipeline for efficiency.

        Parameters
        ----------
        velocity_model_path : str or Path, optional
            Path to a custom velocity model GeoTIFF for epoch transformation.
            See `warp_dynamic_epoch` docstring for file format requirements.
        """
        from pyproj import CRS as _CRS
        
        # Determine what transformations are needed
        needs_epoch = dynamic_target_epoch is not None
        needs_vertical = (
            source_vertical_kind is not None or 
            target_vertical_kind is not None or
            source_geoid_model is not None or
            target_geoid_model is not None
        )
        needs_horizontal = (
            target_horizontal_crs is not None or 
            target_compound_crs is not None
        )
        
        # Count how many transformation types are requested
        transform_count = sum([needs_epoch, needs_vertical, needs_horizontal])
        
        # If multiple transforms needed, use combined pipeline
        if transform_count > 1 or (needs_epoch and (needs_vertical or needs_horizontal)):
            return self._warp_combined(
                target_horizontal_crs=target_horizontal_crs,
                target_compound_crs=target_compound_crs,
                source_vertical_kind=source_vertical_kind,
                target_vertical_kind=target_vertical_kind,
                source_geoid_model=source_geoid_model,
                target_geoid_model=target_geoid_model,
                dynamic_target_epoch=dynamic_target_epoch,
                output_path=output_path,
                overwrite=overwrite,
                return_pipeline=return_pipeline,
                velocity_model_path=velocity_model_path,
            )

        # Single transformation type - use existing specialized methods
        if needs_epoch:
            return self._warp_dynamic_epoch_core(
                target_epoch=dynamic_target_epoch,
                target_crs_proj=dynamic_target_crs_proj,
                source_vertical_kind=source_vertical_kind,
                target_vertical_kind=target_vertical_kind,
                source_geoid_model=source_geoid_model,
                target_geoid_model=target_geoid_model,
                output_path=output_path,
                overwrite=overwrite,
                return_pipeline=return_pipeline,
                velocity_model_path=velocity_model_path,
            )
        
        if needs_vertical:
            return self._warp_vertical_datum_core(
                source_kind=source_vertical_kind or "ellipsoidal",
                target_kind=target_vertical_kind or "orthometric",
                source_geoid_model=source_geoid_model,
                target_geoid_model=target_geoid_model,
                target_crs_proj=target_compound_crs,
                output_path=output_path,
                overwrite=overwrite,
                return_pipeline=return_pipeline,
            )
        
        if needs_horizontal:
            # Route horizontal-only transforms through _warp_combined
            return self._warp_combined(
                target_horizontal_crs=target_horizontal_crs,
                target_compound_crs=target_compound_crs,
                output_path=output_path,
                overwrite=overwrite,
                return_pipeline=return_pipeline,
            )

        # No transformation needed
        return self

    def warp_vertical_datum(
        self,
        source_kind: str,
        target_kind: str,
        source_geoid_model: Optional[str] = None,
        target_geoid_model: Optional[str] = None,
        target_crs_proj: Optional[Any] = None,
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = True,
        return_pipeline: bool = False,
    ):
        """
        Convenience wrapper around warp_pointcloud for vertical datum changes.
        """
        return self.warp_pointcloud(
            source_vertical_kind=source_kind,
            target_vertical_kind=target_kind,
            source_geoid_model=source_geoid_model,
            target_geoid_model=target_geoid_model,
            target_compound_crs=target_crs_proj,
            output_path=output_path,
            overwrite=overwrite,
            return_pipeline=return_pipeline,
        )

    def _warp_combined(
        self,
        target_horizontal_crs: Optional[Any] = None,
        target_compound_crs: Optional[Any] = None,
        source_vertical_kind: Optional[str] = None,
        target_vertical_kind: Optional[str] = None,
        source_geoid_model: Optional[str] = None,
        target_geoid_model: Optional[str] = None,
        dynamic_target_epoch: Optional[float] = None,
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        return_pipeline: bool = False,
        velocity_model_path: Optional[Union[str, Path]] = None,
    ):
        """
        Combined transformation: epoch + vertical + horizontal in a single PDAL pass.

        This is the most efficient path when multiple transformation types are needed.

        Parameters
        ----------
        velocity_model_path : str or Path, optional
            Path to a custom velocity model GeoTIFF. If provided, bypasses
            automatic model selection for epoch transformation.
        """
        from pyproj import CRS as _CRS
        
        # Determine source CRS
        src_crs_wkt = self.current_compound_crs or self.original_compound_crs
        if not src_crs_wkt:
            src_crs_wkt = self.current_horizontal_crs or self.original_horizontal_crs
        if not src_crs_wkt:
            raise ValueError("No CRS found on this point cloud")
        
        src_crs_obj = _CRS.from_user_input(src_crs_wkt)
        
        # Extract horizontal CRS from source
        if src_crs_obj.is_compound and hasattr(src_crs_obj, 'sub_crs_list'):
            src_horiz_crs = src_crs_obj.sub_crs_list[0]
        else:
            src_horiz_crs = src_crs_obj
        src_horiz_str = src_horiz_crs.to_string()
        
        # Determine target horizontal CRS
        if target_horizontal_crs is not None:
            dst_horiz_obj = _CRS.from_user_input(target_horizontal_crs)
            dst_horiz_str = dst_horiz_obj.to_string()
        elif target_compound_crs is not None:
            dst_crs_obj = _CRS.from_user_input(target_compound_crs)
            if dst_crs_obj.is_compound and hasattr(dst_crs_obj, 'sub_crs_list'):
                dst_horiz_str = dst_crs_obj.sub_crs_list[0].to_string()
            else:
                dst_horiz_str = dst_crs_obj.to_string()
        else:
            dst_horiz_str = src_horiz_str
        
        # Epochs
        src_epoch = getattr(self, "epoch", None)
        dst_epoch = float(dynamic_target_epoch) if dynamic_target_epoch is not None else None
        
        # Vertical parameters
        src_vertical_kind = source_vertical_kind
        if src_vertical_kind is None:
            is_ortho = getattr(self, 'is_orthometric', None)
            if is_ortho is True:
                src_vertical_kind = "orthometric"
            elif is_ortho is False:
                src_vertical_kind = "ellipsoidal"
        
        dst_vertical_kind = target_vertical_kind or src_vertical_kind
        
        src_geoid = source_geoid_model or getattr(self, "geoid_model", None)
        dst_geoid = target_geoid_model or src_geoid
        
        # Build output path
        src_path = Path(self.filename)
        if output_path is None:
            parts = []
            if dst_epoch and src_epoch and abs(dst_epoch - src_epoch) > 0.001:
                parts.append(f"epoch{dst_epoch:.2f}".replace(".", "p"))
            if dst_vertical_kind and dst_vertical_kind != src_vertical_kind:
                parts.append(dst_vertical_kind[:4])
            if dst_horiz_str != src_horiz_str:
                parts.append("reproj")
            tag = "_".join(parts) if parts else "warped"
            # Always add "_transformed" suffix to indicate this is a transformed point cloud
            output_path = src_path.with_name(src_path.stem + f"_{tag}_transformed" + src_path.suffix)
        else:
            output_path = Path(output_path)
        
        if output_path.exists() and not overwrite:
            raise ValueError(f"Output file exists and overwrite=False: {output_path}")
        
        # Build CRSState objects
        src_state = CRSState(
            crs=src_horiz_str,
            epoch=src_epoch,
            vertical_kind=src_vertical_kind,
            geoid_alias=src_geoid,
        )
        dst_state = CRSState(
            crs=dst_horiz_str,
            epoch=dst_epoch,
            vertical_kind=dst_vertical_kind,
            geoid_alias=dst_geoid,
        )
        
        # Get deformation grids if epoch transform needed
        deformation_grids = None
        central_epoch = None
        if dst_epoch is not None and src_epoch is not None and abs(dst_epoch - src_epoch) > 0.001:
            if velocity_model_path is not None:
                # User provided a custom velocity model
                velocity_model_path = Path(velocity_model_path)
                if not velocity_model_path.exists():
                    raise FileNotFoundError(
                        f"Custom velocity model not found: {velocity_model_path}"
                    )
                deformation_grids = str(velocity_model_path)
                central_epoch = (src_epoch + dst_epoch) / 2.0
                print(f"Using custom velocity model: {velocity_model_path}", file=sys.stderr)
            else:
                # Automatic velocity model selection
                try:
                    bbox_4326 = self.bbox_4326
                except Exception as e:
                    raise ValueError(
                        "bbox_4326 not available. Ensure from_file() was called."
                    ) from e

                vm, _ = select_velocity_model(
                    bbox_4326=bbox_4326,
                    src_epoch=float(src_epoch),
                    dst_epoch=float(dst_epoch),
                    choice=None,
                    verbose=True,
                )
                deformation_grids = vm.filepath
                central_epoch = vm.central_epoch if vm.central_epoch is not None else src_epoch
        
        # Build the combined pipeline
        try:
            coord_op = build_complete_pipeline(
                src_state,
                dst_state,
                deformation_grids=deformation_grids,
                deformation_central_epoch=central_epoch,
            )
        except ProjError as e:
            raise RuntimeError(f"Failed to build combined PROJ pipeline: {e}")
        
        # Run PDAL
        pipeline_spec = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": str(self.filename),
                },
                {
                    "type": "filters.projpipeline",
                    "coord_op": coord_op,
                    "out_srs": dst_horiz_str,
                },
                {
                    "type": "writers.las",
                    "filename": str(output_path),
                    "a_srs": dst_horiz_str,
                },
            ]
        }
        
        pipe = pdal.Pipeline(json.dumps(pipeline_spec))
        # Use streaming execution for memory efficiency - writes to file
        execute_count = pipe.execute_streaming(chunk_size=1000000)

        # Verify output - try multiple methods to get point count
        out_count_val = 0

        # Method 1: Check execute() return value
        if execute_count and execute_count > 0:
            out_count_val = execute_count

        # Method 2: Check metadata (arrays not available in streaming mode)
        if out_count_val == 0:
            md = pipe.metadata.get("metadata", {})
            writer_keys = [k for k in md.keys() if k.startswith("writers.las")]
            writer_md = md.get(writer_keys[0], {}) if writer_keys else {}

            out_count = (
                writer_md.get("num_points") or
                writer_md.get("count") or
                writer_md.get("points")
            )
            try:
                out_count_val = int(out_count) if out_count is not None else 0
            except Exception:
                pass

        # Method 4: Check if output file exists and has size
        if out_count_val == 0 and output_path.exists():
            if output_path.stat().st_size > 0:
                # File exists with data - trust it worked
                out_count_val = 1  # Placeholder, will be updated when loading

        if out_count_val == 0:
            log_text = getattr(pipe, "log", "")
            # Get more diagnostic info
            import os
            proj_lib = os.environ.get('PROJ_LIB', 'not set')
            output_exists = output_path.exists() if output_path else False
            output_size = output_path.stat().st_size if output_exists else 0
            raise RuntimeError(
                f"Combined warp produced zero output points.\n"
                f"Pipeline: {coord_op}\n"
                f"PDAL log: {log_text}\n"
                f"PROJ_LIB: {proj_lib}\n"
                f"Output file exists: {output_exists}, size: {output_size}\n"
                f"execute() returned: {execute_count}"
            )
        
        # Load output and update metadata
        out_pc = PointCloud(str(output_path))
        out_pc.from_file()
        
        # Propagate CRS metadata to output
        # Use source vertical CRS if no vertical transformation was done
        src_vert_crs = (
            getattr(self, 'current_vertical_crs', None) or
            getattr(self, 'original_vertical_crs', None)
        )

        out_pc.add_metadata(
            horizontal_CRS=dst_horiz_str,
            vertical_CRS=src_vert_crs,  # Preserve source vertical CRS
            epoch=dst_epoch,
            geoid_model=dst_geoid,
        )

        # Update vertical kind tracking
        if dst_vertical_kind:
            out_pc.is_orthometric = (dst_vertical_kind.lower() == "orthometric")

        self._copy_metadata_attributes(out_pc)
        
        # Record in CRS history
        if getattr(self, "crs_history", None) is not None:
            try:
                self.crs_history.record_transformation_entry(
                    transformation_type="Combined warp (epoch + vertical + horizontal)",
                    source_crs_proj=src_horiz_str,
                    target_crs_proj=dst_horiz_str,
                    method="PROJ pipeline via PDAL filters.projpipeline",
                    src_epoch=src_epoch,
                    dst_epoch=dst_epoch,
                    geoid_model=dst_geoid,
                    source_file=str(self.filename),
                    target_file=str(output_path),
                )
            except Exception:
                pass
        
        if return_pipeline:
            return out_pc, coord_op
        return out_pc

    def _warp_vertical_datum_core(
        self,
        source_kind: str,
        target_kind: str,
        source_geoid_model: Optional[str],
        target_geoid_model: Optional[str],
        target_crs_proj: Optional[Any],
        output_path: Optional[Union[str, Path]],
        overwrite: bool,
        return_pipeline: bool = False,
    ):
        """
        Internal implementation of vertical datum transformation.

        This version builds a PROJ 9 '+proj=pipeline' string via build_complete_pipeline()
        and applies it using PDAL's filters.projpipeline. All vertical/geoid logic
        is handled by PROJ; this function only orchestrates I/O and metadata.
        """
        from pyproj import CRS as _CRS

        # Validate vertical kinds
        source_kind = (source_kind or "").lower()
        target_kind = (target_kind or "").lower()
        if source_kind not in ("orthometric", "ellipsoidal"):
            raise ValueError("source_kind must be 'orthometric' or 'ellipsoidal'.")
        if target_kind not in ("orthometric", "ellipsoidal"):
            raise ValueError("target_kind must be 'orthometric' or 'ellipsoidal'.")

        # Determine output filename
        src_path = Path(self.filename)

        if output_path is None:
            tag = f"{source_kind}_to_{target_kind}"
            if (
                source_geoid_model
                and target_geoid_model
                and source_geoid_model != target_geoid_model
            ):
                tag += f"_{source_geoid_model}_to_{target_geoid_model}"
            # Always add "_transformed" suffix to indicate this is a transformed point cloud
            output_path = src_path.with_name(src_path.stem + f"_{tag}_transformed" + src_path.suffix)
        else:
            output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise ValueError(f"Output file already exists and overwrite=False: {output_path}")

        # Determine a base horizontal CRS for PROJ
        if self.current_horizontal_crs:
            horiz_crs_obj = _CRS.from_user_input(self.current_horizontal_crs)
        elif self.current_compound_crs:
            comp = _CRS.from_user_input(self.current_compound_crs)
            if getattr(comp, "sub_crs_list", None):
                horiz_crs_obj = comp.sub_crs_list[0]
            else:
                horiz_crs_obj = comp
        elif self.original_compound_crs:
            comp = _CRS.from_user_input(self.original_compound_crs)
            if getattr(comp, "sub_crs_list", None):
                horiz_crs_obj = comp.sub_crs_list[0]
            else:
                horiz_crs_obj = comp
        else:
            raise ValueError(
                "Could not determine a horizontal CRS for this point cloud; "
                "vertical datum transformation requires a known horizontal CRS."
            )

        base_crs_str = horiz_crs_obj.to_string()

        # Determine target CRS string
        if target_crs_proj is None:
            dst_crs_str = base_crs_str
        else:
            if isinstance(target_crs_proj, _CRS):
                dst_crs_obj = target_crs_proj
            else:
                dst_crs_obj = _CRS.from_user_input(target_crs_proj)
            dst_crs_str = dst_crs_obj.to_string()

        # Build CRSState objects and call build_complete_pipeline
        src_state = CRSState(
            crs=base_crs_str,
            epoch=None,
            vertical_kind=source_kind,
            geoid_alias=str(source_geoid_model) if source_geoid_model else None,
        )
        dst_state = CRSState(
            crs=dst_crs_str,
            epoch=None,
            vertical_kind=target_kind,
            geoid_alias=str(target_geoid_model) if target_geoid_model else None,
        )

        try:
            coord_op = build_complete_pipeline(src_state, dst_state)
        except ProjError as e:
            raise RuntimeError(
                f"Failed to build PROJ pipeline for vertical datum transformation: {e}"
            )

        # Run PDAL with filters.projpipeline
        pipeline_spec = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": str(src_path),
                },
                {
                    "type": "filters.projpipeline",
                    "coord_op": coord_op,
                    "out_srs": dst_crs_str,
                },
                {
                    "type": "writers.las",
                    "filename": str(output_path),
                    "a_srs": dst_crs_str,
                },
            ]
        }

        pipe = pdal.Pipeline(json.dumps(pipeline_spec))
        # Use streaming execution for memory efficiency - writes to file
        execute_count = pipe.execute_streaming(chunk_size=1000000)

        # Determine how many points we actually wrote
        out_count_val = execute_count if execute_count and execute_count > 0 else 0

        # Fallback: check metadata
        if out_count_val == 0:
            md = pipe.metadata.get("metadata", {})
            writer_keys = [k for k in md.keys() if k.startswith("writers.las")]
            writer_md = md.get(writer_keys[0], {}) if writer_keys else {}

            out_count = (
                writer_md.get("num_points")
                or writer_md.get("count")
                or writer_md.get("points")
            )
            try:
                out_count_val = int(out_count) if out_count is not None else 0
            except Exception:
                out_count_val = 0

        # Fallback: check if output file exists with data
        if out_count_val == 0 and output_path.exists() and output_path.stat().st_size > 0:
            out_count_val = 1  # File has data, trust it worked

        if out_count_val == 0:
            log_text = getattr(pipe, "log", "")
            raise RuntimeError(
                "Vertical datum warp produced zero output points.\n"
                "This often indicates a geoid / PROJ issue (grid not found, "
                "outside coverage, or pipeline parse error).\n\n"
                f"PDAL log:\n{log_text}"
            )

        # Load transformed file and update metadata
        out_pc = PointCloud(str(output_path))
        out_pc.from_file()

        # Propagate CRS metadata - preserve source vertical CRS
        src_vert_crs = (
            getattr(self, 'current_vertical_crs', None) or
            getattr(self, 'original_vertical_crs', None)
        )

        out_pc.add_metadata(
            horizontal_CRS=dst_crs_str,
            vertical_CRS=src_vert_crs,
            geoid_model=target_geoid_model,
        )

        # Update vertical kind tracking
        out_pc.is_orthometric = (target_kind.lower() == "orthometric")

        self._copy_metadata_attributes(out_pc)

        if return_pipeline:
            return out_pc, coord_op
        return out_pc

    def _warp_dynamic_epoch_core(
        self,
        target_epoch: float,
        target_crs_proj: Optional[Any],
        source_vertical_kind: Optional[str],
        target_vertical_kind: Optional[str],
        source_geoid_model: Optional[str],
        target_geoid_model: Optional[str],
        output_path: Optional[Union[str, Path]],
        overwrite: bool,
        return_pipeline: bool = False,
        velocity_model_path: Optional[Union[str, Path]] = None,
    ):
        """
        Internal implementation of dynamic epoch transformation.

        This version:
          - Uses PROJ 9 via build_complete_pipeline (no pyproj transforms),
          - Automatically selects a deformation / velocity model based on
            geographic extent (EPSG:4326 bbox) and [src_epoch, dst_epoch],
            OR uses a user-provided velocity model if velocity_model_path is given,
          - Can combine:
                * horizontal CRS change,
                * epoch change,
                * vertical/geoid change
            into a single PROJ pipeline,
          - Runs the '+proj=pipeline' string with PDAL filters.projpipeline.

        Parameters
        ----------
        velocity_model_path : str or Path, optional
            Path to a custom velocity model GeoTIFF. If provided, bypasses
            automatic model selection. See warp_dynamic_epoch docstring for
            required file format (3-band GeoTIFF with E/N/U velocities in mm/yr).
        """
        from pyproj import CRS as _CRS

        # Source CRS and epochs
        src_crs_wkt = self.current_compound_crs or self.original_compound_crs
        if not src_crs_wkt:
            raise ValueError(
                "No compound CRS found on this point cloud; "
                "cannot perform dynamic epoch transformation."
            )

        src_crs_obj = _CRS.from_user_input(src_crs_wkt)
        src_crs_str = src_crs_obj.to_string()

        src_epoch = getattr(self, "epoch", None)
        if src_epoch is None:
            raise ValueError(
                "PointCloud.epoch is not set; dynamic epoch transformation "
                "requires a known source epoch."
            )

        dst_epoch = float(target_epoch)

        # Destination CRS string
        if target_crs_proj is None:
            dst_crs_str = src_crs_str
        else:
            if isinstance(target_crs_proj, _CRS):
                dst_crs_str = target_crs_proj.to_string()
            else:
                dst_crs_str = str(target_crs_proj)

        # Output filename
        src_path = Path(self.filename)
        if output_path is None:
            tag = f"epoch{dst_epoch:.3f}".replace(".", "p")
            # Always add "_transformed" suffix to indicate this is a transformed point cloud
            output_path = src_path.with_name(src_path.stem + f"_{tag}_transformed" + src_path.suffix)
        else:
            output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            raise ValueError(f"Output file already exists and overwrite=False: {output_path}")

        # Geographic bbox in EPSG:4326
        try:
            bbox_4326 = self.bbox_4326
        except Exception as e:
            raise ValueError(
                "bbox_4326 is not available. Ensure from_file() was called "
                "and _get_pointcloud_extent stores poly_4326."
            ) from e

        # Velocity / deformation model selection
        if velocity_model_path is not None:
            # User provided a custom velocity model
            velocity_model_path = Path(velocity_model_path)
            if not velocity_model_path.exists():
                raise FileNotFoundError(
                    f"Custom velocity model not found: {velocity_model_path}"
                )
            deformation_grids = str(velocity_model_path)
            # For custom models, assume central epoch is the midpoint of transformation
            central_epoch = (src_epoch + dst_epoch) / 2.0
            print(f"Using custom velocity model: {velocity_model_path}", file=sys.stderr)
        else:
            # Automatic velocity model selection from registry
            vm, vm_candidates = select_velocity_model(
                bbox_4326=bbox_4326,
                src_epoch=float(src_epoch),
                dst_epoch=float(dst_epoch),
                choice=None,
                verbose=True,
            )
            deformation_grids = vm.filepath
            central_epoch = vm.central_epoch if vm.central_epoch is not None else src_epoch

        # Normalize vertical kind / geoid aliases
        def _norm_kind(k: Optional[str]) -> Optional[str]:
            if k is None:
                return None
            k = k.lower()
            return k if k in ("orthometric", "ellipsoidal") else None

        src_vertical_kind = _norm_kind(source_vertical_kind)
        dst_vertical_kind = _norm_kind(target_vertical_kind) or src_vertical_kind

        src_geoid_alias = source_geoid_model or getattr(self, "geoid_model", None)
        dst_geoid_alias = (
            target_geoid_model
            or source_geoid_model
            or getattr(self, "geoid_model", None)
        )

        # Build CRSState for source and destination
        src_state = CRSState(
            crs=src_crs_str,
            epoch=float(src_epoch),
            vertical_kind=src_vertical_kind,
            geoid_alias=src_geoid_alias,
        )
        dst_state = CRSState(
            crs=dst_crs_str,
            epoch=float(dst_epoch),
            vertical_kind=dst_vertical_kind,
            geoid_alias=dst_geoid_alias,
        )

        # Build full pipeline
        try:
            coord_op = build_complete_pipeline(
                src_state,
                dst_state,
                deformation_grids=deformation_grids,
                deformation_central_epoch=central_epoch,
            )
        except ProjError as e:
            raise RuntimeError(
                f"Failed to build PROJ pipeline for dynamic epoch transformation: {e}"
            )

        # Run PDAL with filters.projpipeline
        pipeline_spec = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": str(self.filename),
                },
                {
                    "type": "filters.projpipeline",
                    "coord_op": coord_op,
                    "out_srs": dst_crs_str,
                },
                {
                    "type": "writers.las",
                    "filename": str(output_path),
                    "a_srs": dst_crs_str,
                },
            ]
        }

        pipe = pdal.Pipeline(json.dumps(pipeline_spec))
        # Use streaming execution for memory efficiency - writes to file
        execute_count = pipe.execute_streaming(chunk_size=1000000)

        # Sanity check: did we actually write any points?
        out_count_val = execute_count if execute_count and execute_count > 0 else 0

        # Fallback: check metadata
        if out_count_val == 0:
            meta = pipe.metadata
            md = meta.get("metadata", {})
            writer_md = md.get("writers.las", {})

            out_count = (
                writer_md.get("num_points")
                or writer_md.get("count")
                or writer_md.get("points")
            )
            try:
                out_count_val = int(out_count) if out_count is not None else 0
            except Exception:
                out_count_val = 0

        # Fallback: check if output file exists with data
        if out_count_val == 0 and output_path.exists() and output_path.stat().st_size > 0:
            out_count_val = 1  # File has data, trust it worked

        if out_count_val == 0:
            log_text = getattr(pipe, "log", "")
            raise RuntimeError(
                "Vertical datum warp produced zero output points.\n"
                "This often indicates a geoid / PROJ issue (grid not found, "
                "outside coverage, or pipeline parse error).\n\n"
                f"PDAL log:\n{log_text}"
            )
            
        # Load as new PointCloud and update metadata
        out_pc = PointCloud(str(output_path))
        out_pc.from_file()

        final_geoid = dst_state.geoid_alias

        # Propagate CRS metadata - preserve source vertical CRS
        src_vert_crs = (
            getattr(self, 'current_vertical_crs', None) or
            getattr(self, 'original_vertical_crs', None)
        )

        out_pc.add_metadata(
            compound_CRS=dst_crs_str,
            vertical_CRS=src_vert_crs,  # Preserve vertical CRS through epoch transform
            epoch=dst_epoch,
            geoid_model=final_geoid,
        )

        self._copy_metadata_attributes(out_pc)

        # CRSHistory entry
        if getattr(self, "crs_history", None) is not None:
            try:
                self.crs_history.record_transformation_entry(
                    transformation_type="Dynamic epoch transformation",
                    source_crs_proj=src_crs_str,
                    target_crs_proj=dst_crs_str,
                    method=(
                        "PROJ9 pipeline via PDAL filters.projpipeline "
                        f"with velocity model '{vm.name}' ({vm.filename})"
                    ),
                    src_epoch=src_epoch,
                    dst_epoch=dst_epoch,
                    note="Coordinates moved using time-dependent CRS transformation.",
                    source_file=str(self.filename),
                    target_file=str(output_path),
                )

                if getattr(out_pc, "crs_history", None) is not None:
                    out_pc.crs_history.add_manual_change_entry(
                        note=(
                            f"Derived from {self.filename} via dynamic epoch "
                            f"transformation using velocity model '{vm.name}'."
                        ),
                        epoch=out_pc.epoch,
                        geoid_model=out_pc.geoid_model,
                    )
            except Exception:
                pass

        if return_pipeline:
            return out_pc, coord_op
        return out_pc

    def warp_dynamic_epoch(
        self,
        target_epoch: float,
        target_crs_proj: Optional[Any] = None,
        source_vertical_kind: Optional[str] = None,
        target_vertical_kind: Optional[str] = None,
        source_geoid_model: Optional[str] = None,
        target_geoid_model: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = True,
        return_pipeline: bool = False,
        velocity_model_path: Optional[Union[str, Path]] = None,
    ):
        """
        Transform point cloud coordinates from current epoch to target epoch.

        This applies a velocity/deformation model to account for crustal motion
        between epochs. By default, an appropriate velocity model is automatically
        selected based on geographic location and time span.

        Parameters
        ----------
        target_epoch : float
            Target epoch as decimal year (e.g., 2020.5 for July 1, 2020).
        target_crs_proj : CRS or str, optional
            Target CRS if also changing coordinate reference system.
        source_vertical_kind : str, optional
            Source vertical datum type ('orthometric' or 'ellipsoidal').
        target_vertical_kind : str, optional
            Target vertical datum type ('orthometric' or 'ellipsoidal').
        source_geoid_model : str, optional
            Source geoid model name (e.g., 'geoid18').
        target_geoid_model : str, optional
            Target geoid model name.
        output_path : str or Path, optional
            Output file path. If None, auto-generated with '_transformed' suffix.
        overwrite : bool, default True
            Whether to overwrite existing output file.
        return_pipeline : bool, default False
            If True, also return the PROJ pipeline string.
        velocity_model_path : str or Path, optional
            Path to a custom velocity model file. If provided, this model is used
            instead of automatic selection from the registry.

            **Velocity Model File Format:**

            The file must be a GeoTIFF with 3 bands containing velocity components
            in the local East-North-Up (ENU) coordinate system:

            - **Band 1**: East velocity (positive = eastward motion)
            - **Band 2**: North velocity (positive = northward motion)
            - **Band 3**: Up velocity (positive = uplift)

            **Units**: All velocities must be in **millimeters per year (mm/yr)**.

            **Coordinate System**: The GeoTIFF must be georeferenced in **EPSG:4326**
            (WGS84 geographic coordinates, longitude/latitude in degrees).

            **Coverage**: The grid must cover the full extent of your point cloud.
            PROJ will fail if the point cloud extends beyond the velocity grid.

            **Example creation with rasterio**::

                import rasterio
                import numpy as np
                from rasterio.transform import from_bounds

                # Create velocity grids (example: 0.1 degree resolution)
                ve = np.zeros((100, 100), dtype=np.float32)  # East velocity (mm/yr)
                vn = np.zeros((100, 100), dtype=np.float32)  # North velocity (mm/yr)
                vu = np.zeros((100, 100), dtype=np.float32)  # Up velocity (mm/yr)

                transform = from_bounds(-110, 35, -100, 45, 100, 100)

                with rasterio.open(
                    'my_velocity_model.tif', 'w',
                    driver='GTiff',
                    height=100, width=100,
                    count=3,
                    dtype=rasterio.float32,
                    crs='EPSG:4326',
                    transform=transform,
                ) as dst:
                    dst.write(ve, 1)
                    dst.write(vn, 2)
                    dst.write(vu, 3)

        Returns
        -------
        PointCloud
            New PointCloud object with transformed coordinates.
        tuple
            If return_pipeline=True, returns (PointCloud, pipeline_string).

        See Also
        --------
        warp_pointcloud : Unified transformation method.
        velocity_model_converters : Tools to create velocity model GeoTIFFs.

        References
        ----------
        - PROJ deformation grid format: https://proj.org/operations/transformations/deformation.html
        - EarthScope/GAGE velocity data: https://www.unavco.org/data/gps-gnss/derived-products/
        """
        return self.warp_pointcloud(
            dynamic_target_epoch=target_epoch,
            dynamic_target_crs_proj=target_crs_proj,
            source_vertical_kind=source_vertical_kind,
            target_vertical_kind=target_vertical_kind,
            source_geoid_model=source_geoid_model,
            target_geoid_model=target_geoid_model,
            output_path=output_path,
            overwrite=overwrite,
            return_pipeline=return_pipeline,
            velocity_model_path=velocity_model_path,
        )

    def check_transformation_metadata(
        self,
        expected_epoch: Optional[float] = None,
        expected_horizontal_crs: Optional[str] = None,
        expected_vertical_crs: Optional[str] = None,
        expected_vertical_kind: Optional[str] = None,
        expected_geoid_model: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Check and validate the metadata of a (transformed) point cloud.

        This function inspects the point cloud's CRS, epoch, geoid model, and
        transformation history to verify that transformations were applied correctly.

        Parameters
        ----------
        expected_epoch : float, optional
            Expected epoch after transformation (decimal year).
        expected_horizontal_crs : str, optional
            Expected horizontal CRS (EPSG code like "EPSG:32610" or WKT).
        expected_vertical_crs : str, optional
            Expected vertical CRS (EPSG code or WKT).
        expected_vertical_kind : str, optional
            Expected vertical datum type ("orthometric" or "ellipsoidal").
        expected_geoid_model : str, optional
            Expected geoid model name (e.g., "geoid18").
        verbose : bool, default True
            Print detailed report to stderr.

        Returns
        -------
        dict
            Dictionary containing:
            - 'valid': bool - True if all checks passed
            - 'errors': list - List of validation error messages
            - 'warnings': list - List of warning messages
            - 'metadata': dict - Current metadata values
            - 'original': dict - Original metadata values
            - 'transformations': list - List of applied transformations
            - 'checks': dict - Results of each validation check

        Example
        -------
        >>> # Check metadata after epoch transformation
        >>> result = transformed_pc.check_transformation_metadata(
        ...     expected_epoch=2018.5,
        ...     expected_vertical_kind="orthometric",
        ...     verbose=True
        ... )
        >>> if result['valid']:
        ...     print("Transformation verified!")
        ... else:
        ...     print("Errors:", result['errors'])
        """
        import sys
        from pyproj import CRS as CRS_

        errors = []
        warnings = []
        checks = {}

        # Helper to safely get CRS EPSG code
        def _get_epsg(crs_val):
            if crs_val is None:
                return None
            try:
                crs_obj = _ensure_crs_obj(crs_val)
                epsg = crs_obj.to_epsg()
                return f"EPSG:{epsg}" if epsg else crs_obj.to_string()[:50]
            except Exception:
                return str(crs_val)[:50] if crs_val else None

        # Helper to compare CRS
        def _crs_match(crs1, crs2):
            if crs1 is None or crs2 is None:
                return crs1 == crs2
            try:
                obj1 = _ensure_crs_obj(crs1)
                obj2 = _ensure_crs_obj(crs2)
                return obj1.equals(obj2)
            except Exception:
                return str(crs1) == str(crs2)

        # Gather current metadata
        current_metadata = {
            'filename': str(self.filename),
            'point_count': getattr(self, 'point_count', None),
            'epoch': getattr(self, 'epoch', None),
            'geoid_model': getattr(self, 'geoid_model', None),
            'is_orthometric': getattr(self, 'is_orthometric', None),
            'horizontal_crs': _get_epsg(getattr(self, 'current_horizontal_crs', None)),
            'vertical_crs': _get_epsg(getattr(self, 'current_vertical_crs', None)),
            'compound_crs': _get_epsg(getattr(self, 'current_compound_crs', None)),
            'horizontal_unit': str(getattr(self, 'horizontal_unit', None)),
            'vertical_unit': str(getattr(self, 'vertical_unit', None)),
        }

        # Gather original metadata
        original_metadata = {
            'epoch': getattr(self, 'original_epoch', None) if hasattr(self, 'original_epoch') else None,
            'geoid_model': getattr(self, 'original_geoid_model', None) if hasattr(self, 'original_geoid_model') else None,
            'horizontal_crs': _get_epsg(getattr(self, 'original_horizontal_crs', None)),
            'vertical_crs': _get_epsg(getattr(self, 'original_vertical_crs', None)),
            'compound_crs': _get_epsg(getattr(self, 'original_compound_crs', None)),
        }

        # Determine vertical kind from is_orthometric
        current_vertical_kind = None
        if current_metadata['is_orthometric'] is True:
            current_vertical_kind = "orthometric"
        elif current_metadata['is_orthometric'] is False:
            current_vertical_kind = "ellipsoidal"
        current_metadata['vertical_kind'] = current_vertical_kind

        # Get transformation history
        transformations = []
        if hasattr(self, 'crs_history') and self.crs_history is not None:
            try:
                history_dict = self.crs_history.to_dict()
                if 'history' in history_dict:
                    for entry in history_dict['history']:
                        transformations.append({
                            'type': entry.get('entry_type', 'unknown'),
                            'timestamp': entry.get('timestamp', ''),
                            'note': entry.get('note', ''),
                        })
            except Exception:
                pass

        # =====================================================================
        # Validation checks
        # =====================================================================

        # Check 1: Epoch
        if expected_epoch is not None:
            actual_epoch = current_metadata['epoch']
            if actual_epoch is None:
                errors.append(f"Epoch not set (expected {expected_epoch})")
                checks['epoch'] = {'status': 'FAIL', 'expected': expected_epoch, 'actual': None}
            elif abs(actual_epoch - expected_epoch) > 0.01:
                errors.append(
                    f"Epoch mismatch: expected {expected_epoch}, got {actual_epoch}"
                )
                checks['epoch'] = {'status': 'FAIL', 'expected': expected_epoch, 'actual': actual_epoch}
            else:
                checks['epoch'] = {'status': 'PASS', 'expected': expected_epoch, 'actual': actual_epoch}
        else:
            if current_metadata['epoch'] is None:
                warnings.append("Epoch not set on point cloud")
            checks['epoch'] = {'status': 'SKIP', 'actual': current_metadata['epoch']}

        # Check 2: Horizontal CRS
        if expected_horizontal_crs is not None:
            actual_horiz = getattr(self, 'current_horizontal_crs', None)
            if _crs_match(actual_horiz, expected_horizontal_crs):
                checks['horizontal_crs'] = {
                    'status': 'PASS',
                    'expected': expected_horizontal_crs,
                    'actual': current_metadata['horizontal_crs']
                }
            else:
                errors.append(
                    f"Horizontal CRS mismatch: expected {expected_horizontal_crs}, "
                    f"got {current_metadata['horizontal_crs']}"
                )
                checks['horizontal_crs'] = {
                    'status': 'FAIL',
                    'expected': expected_horizontal_crs,
                    'actual': current_metadata['horizontal_crs']
                }
        else:
            checks['horizontal_crs'] = {'status': 'SKIP', 'actual': current_metadata['horizontal_crs']}

        # Check 3: Vertical CRS
        if expected_vertical_crs is not None:
            actual_vert = getattr(self, 'current_vertical_crs', None)
            if _crs_match(actual_vert, expected_vertical_crs):
                checks['vertical_crs'] = {
                    'status': 'PASS',
                    'expected': expected_vertical_crs,
                    'actual': current_metadata['vertical_crs']
                }
            else:
                errors.append(
                    f"Vertical CRS mismatch: expected {expected_vertical_crs}, "
                    f"got {current_metadata['vertical_crs']}"
                )
                checks['vertical_crs'] = {
                    'status': 'FAIL',
                    'expected': expected_vertical_crs,
                    'actual': current_metadata['vertical_crs']
                }
        else:
            checks['vertical_crs'] = {'status': 'SKIP', 'actual': current_metadata['vertical_crs']}

        # Check 4: Vertical kind (orthometric/ellipsoidal)
        if expected_vertical_kind is not None:
            expected_kind_lower = expected_vertical_kind.lower()
            if current_vertical_kind is None:
                errors.append(
                    f"Vertical kind not determined (expected {expected_vertical_kind})"
                )
                checks['vertical_kind'] = {
                    'status': 'FAIL',
                    'expected': expected_vertical_kind,
                    'actual': None
                }
            elif current_vertical_kind != expected_kind_lower:
                errors.append(
                    f"Vertical kind mismatch: expected {expected_vertical_kind}, "
                    f"got {current_vertical_kind}"
                )
                checks['vertical_kind'] = {
                    'status': 'FAIL',
                    'expected': expected_vertical_kind,
                    'actual': current_vertical_kind
                }
            else:
                checks['vertical_kind'] = {
                    'status': 'PASS',
                    'expected': expected_vertical_kind,
                    'actual': current_vertical_kind
                }
        else:
            checks['vertical_kind'] = {'status': 'SKIP', 'actual': current_vertical_kind}

        # Check 5: Geoid model
        if expected_geoid_model is not None:
            actual_geoid = current_metadata['geoid_model']
            if actual_geoid is None:
                errors.append(f"Geoid model not set (expected {expected_geoid_model})")
                checks['geoid_model'] = {
                    'status': 'FAIL',
                    'expected': expected_geoid_model,
                    'actual': None
                }
            elif actual_geoid.lower() != expected_geoid_model.lower():
                errors.append(
                    f"Geoid model mismatch: expected {expected_geoid_model}, "
                    f"got {actual_geoid}"
                )
                checks['geoid_model'] = {
                    'status': 'FAIL',
                    'expected': expected_geoid_model,
                    'actual': actual_geoid
                }
            else:
                checks['geoid_model'] = {
                    'status': 'PASS',
                    'expected': expected_geoid_model,
                    'actual': actual_geoid
                }
        else:
            checks['geoid_model'] = {'status': 'SKIP', 'actual': current_metadata['geoid_model']}

        # Check 6: Transformation history exists
        if len(transformations) > 1:  # More than just 'initial' entry
            checks['has_transformations'] = {'status': 'PASS', 'count': len(transformations)}
        else:
            warnings.append("No transformation history recorded")
            checks['has_transformations'] = {'status': 'WARN', 'count': len(transformations)}

        # Build result
        is_valid = len(errors) == 0
        result = {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'metadata': current_metadata,
            'original': original_metadata,
            'transformations': transformations,
            'checks': checks,
        }

        # Print report if verbose
        if verbose:
            print("=" * 70, file=sys.stderr)
            print("POINT CLOUD TRANSFORMATION METADATA CHECK", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            print(f"File: {current_metadata['filename']}", file=sys.stderr)
            print(f"Points: {current_metadata['point_count']}", file=sys.stderr)
            print(file=sys.stderr)

            print("CURRENT METADATA:", file=sys.stderr)
            print(f"  Epoch:           {current_metadata['epoch']}", file=sys.stderr)
            print(f"  Horizontal CRS:  {current_metadata['horizontal_crs']}", file=sys.stderr)
            print(f"  Vertical CRS:    {current_metadata['vertical_crs']}", file=sys.stderr)
            print(f"  Vertical Kind:   {current_vertical_kind}", file=sys.stderr)
            print(f"  Geoid Model:     {current_metadata['geoid_model']}", file=sys.stderr)
            print(f"  Horizontal Unit: {current_metadata['horizontal_unit']}", file=sys.stderr)
            print(f"  Vertical Unit:   {current_metadata['vertical_unit']}", file=sys.stderr)
            print(file=sys.stderr)

            print("ORIGINAL METADATA:", file=sys.stderr)
            print(f"  Epoch:           {original_metadata['epoch']}", file=sys.stderr)
            print(f"  Horizontal CRS:  {original_metadata['horizontal_crs']}", file=sys.stderr)
            print(f"  Vertical CRS:    {original_metadata['vertical_crs']}", file=sys.stderr)
            print(file=sys.stderr)

            print("VALIDATION CHECKS:", file=sys.stderr)
            for check_name, check_result in checks.items():
                status = check_result['status']
                if status == 'PASS':
                    icon = '✓'
                elif status == 'FAIL':
                    icon = '✗'
                elif status == 'WARN':
                    icon = '⚠'
                else:
                    icon = '-'

                if 'expected' in check_result:
                    print(f"  {icon} {check_name}: {status} "
                          f"(expected={check_result.get('expected')}, "
                          f"actual={check_result.get('actual')})", file=sys.stderr)
                else:
                    print(f"  {icon} {check_name}: {status} "
                          f"(actual={check_result.get('actual', check_result.get('count', 'N/A'))})",
                          file=sys.stderr)
            print(file=sys.stderr)

            if transformations:
                print(f"TRANSFORMATION HISTORY ({len(transformations)} entries):", file=sys.stderr)
                for i, t in enumerate(transformations):
                    print(f"  [{i}] {t['type']}: {t['note'][:60]}...", file=sys.stderr)
                print(file=sys.stderr)

            if errors:
                print("ERRORS:", file=sys.stderr)
                for err in errors:
                    print(f"  ✗ {err}", file=sys.stderr)
                print(file=sys.stderr)

            if warnings:
                print("WARNINGS:", file=sys.stderr)
                for warn in warnings:
                    print(f"  ⚠ {warn}", file=sys.stderr)
                print(file=sys.stderr)

            print("=" * 70, file=sys.stderr)
            if is_valid:
                print("RESULT: ✓ ALL CHECKS PASSED", file=sys.stderr)
            else:
                print(f"RESULT: ✗ VALIDATION FAILED ({len(errors)} errors)", file=sys.stderr)
            print("=" * 70, file=sys.stderr)

        return result

    def get_metadata_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the point cloud's current metadata.

        Returns
        -------
        dict
            Dictionary containing all current metadata values.
        """
        return self.check_transformation_metadata(verbose=False)['metadata']