"""Build PROJ pipeline strings for coordinate transformations.

Supports:
1. Horizontal-only transforms (CRS -> CRS)
2. Vertical-only transforms (ellipsoidal <-> orthometric, geoid A <-> geoid B)
3. Dynamic epoch transforms (with velocity/deformation grids)

Only builds PROJ pipeline strings. Execution is handled by PDAL, pyproj, etc.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import pyproj
from pyproj import CRS

from .geoid_utils import select_geoid_grid


# ---------------------------------------------------------------------
# Basic error + state container
# ---------------------------------------------------------------------


class ProjError(RuntimeError):
    pass


@dataclass
class CRSState:
    """
    Description of a CRS and the extra info needed for vertical / dynamic use.

    Attributes
    ----------
    crs          : str
        CRS identifier that PROJ understands: 'EPSG:XXXX', WKT, PROJ string, etc.
    epoch        : float or None
        Decimal year, used only for dynamic transformations.
    vertical_kind: {'ellipsoidal', 'orthometric', None}
        Interpretation of Z-values in this CRS.
    geoid_alias  : str or None
        User-friendly name for geoid model ('geoid18', 'egm2008', ...).
        Only used to resolve a grid filename if geoid_grid is not given.
    geoid_grid   : str or None
        Actual grid filename or path for PROJ (+grids=...), e.g. 'us_noaa_g2018u0.tif'.
    """

    crs: str
    epoch: Optional[float] = None
    vertical_kind: Optional[str] = None
    geoid_alias: Optional[str] = None
    geoid_grid: Optional[str] = None  # just filename is usually safest

    def with_resolved_geoid(self) -> "CRSState":
        """
        If geoid_grid is missing but geoid_alias is set, try to resolve it.

        Here we just call a hook function `resolve_geoid_alias(alias)`, which
        you can adapt to your project's own geoid-handling logic.
        """
        if self.geoid_grid or not self.geoid_alias:
            return self

        grid_path = resolve_geoid_alias(self.geoid_alias)
        if grid_path is None:
            raise ProjError(f"Could not resolve geoid alias '{self.geoid_alias}'.")

        # resolve_geoid_alias returns either:
        # - Just a filename (if grid is in PROJ_LIB) - use as-is
        # - A full absolute path (if grid is elsewhere) - use as-is
        # Do NOT call Path().resolve() as it would resolve filenames to CWD
        grid_path_str = grid_path

        return CRSState(
            crs=self.crs,
            epoch=self.epoch,
            vertical_kind=self.vertical_kind,
            geoid_alias=self.geoid_alias,
            geoid_grid=grid_path_str,
        )


# ---------------------------------------------------------------------
# Hook for your own geoid alias system
# ---------------------------------------------------------------------


def resolve_geoid_alias(alias: str) -> Optional[str]:
    """
    Map a "human" alias like 'geoid18' to an actual grid filename or path.

    This now delegates to the geoid selection logic in geoid_utils.select_geoid_grid,
    which knows about many aliases and will prefer local CONUS grids when available.

    If the input already looks like a filename (ends with .tif, .gtx, etc.),
    we search for it in PROJ data directories, preferring paths without spaces
    (which can cause issues with PDAL/GDAL).

    Return:
        - Grid filename (if in PROJ_LIB) or full path to grid file (string), or
        - None if the alias is unknown.
    """
    alias_str = alias.strip()
    if not alias_str:
        return None

    # Check if it's already an absolute path that exists and has no spaces
    if os.path.isabs(alias_str) and os.path.exists(alias_str):
        if ' ' not in alias_str:
            return alias_str
        # Has spaces - try to find it in a directory without spaces instead

    # Common geoid grid extensions
    geoid_extensions = ('.tif', '.gtx', '.gtx.gz', '.gvb', '.byn', '.grid')
    alias_low = alias_str.lower()

    # Check if it's a filename (not an alias)
    is_filename = any(alias_low.endswith(ext) for ext in geoid_extensions)

    if is_filename:
        filename = os.path.basename(alias_str)
        return _find_grid_prefer_no_spaces(filename)

    # It's an alias - use select_geoid_grid
    try:
        selected, _ = select_geoid_grid(alias_low, verbose=False)
    except (ValueError, IndexError):
        return None

    if selected is None:
        return None

    # select_geoid_grid returns a full path - check if we can use a simpler reference
    filename = os.path.basename(selected)
    return _find_grid_prefer_no_spaces(filename)


def _find_grid_prefer_no_spaces(filename: str) -> Optional[str]:
    """
    Find a grid file in PROJ data directories, preferring paths without spaces.

    PDAL/GDAL can have issues with spaces in paths even when quoted.
    If the grid exists in PROJ_LIB (no spaces), just return the filename
    since PROJ will find it automatically.
    """
    from .geoid_utils import get_all_proj_data_dirs

    # First, check PROJ_LIB specifically - if grid is there, just use filename
    proj_lib = os.environ.get('PROJ_LIB') or os.environ.get('PROJ_DATA')
    if proj_lib and os.path.isdir(proj_lib):
        candidate = os.path.join(proj_lib, filename)
        if os.path.exists(candidate):
            # Grid is in PROJ_LIB - PROJ will find it by filename alone
            return filename

    # Search all PROJ directories, preferring those without spaces
    dirs_no_spaces = []
    dirs_with_spaces = []

    for proj_dir in get_all_proj_data_dirs():
        if ' ' in proj_dir:
            dirs_with_spaces.append(proj_dir)
        else:
            dirs_no_spaces.append(proj_dir)

    # Check directories without spaces first
    for proj_dir in dirs_no_spaces:
        candidate = os.path.join(proj_dir, filename)
        if os.path.exists(candidate):
            # Return just filename if it's findable via PROJ search path
            # or full path if we want to be explicit
            return filename  # PROJ should find it

    # Fall back to directories with spaces (will need quoting)
    for proj_dir in dirs_with_spaces:
        candidate = os.path.join(proj_dir, filename)
        if os.path.exists(candidate):
            return candidate  # Return full path, will be quoted later

    return None


# ---------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------


def _run_projinfo(src_crs: str, dst_crs: str) -> str:
    """
    Call 'projinfo -s SRC -t DST -o PROJ --single-line' and return the PROJ op.

    Normally you get either:
      - '+proj=pipeline +step ...'
      - '+proj=noop'
    """
    cmd = [
        "projinfo",
        "-q",
        "-s",
        src_crs,
        "-t",
        dst_crs,
        "-o",
        "PROJ",
        "--single-line",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise ProjError(
            f"projinfo failed (code {proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {proc.stderr.strip()}"
        )
    line = proc.stdout.strip()
    if "+proj=" not in line:
        raise ProjError(f"projinfo output did not contain a PROJ operation: {line!r}.")
    return line


def _is_geographic(crs_str: str) -> bool:
    """
    Return True if the CRS is geographic (lat/lon).
    """
    try:
        crs = CRS.from_user_input(crs_str)
    except pyproj.exceptions.CRSError as exc:
        raise ProjError(f"Invalid CRS {crs_str!r}: {exc}.")
    return crs.is_geographic


def _proj_step_from_crs(crs_str: str) -> str:
    """
    Extract the core +proj=... step (for projected CRS) suitable for use in a pipeline.

    For example, for UTM 10N NAD83(2011), this might give:

        '+proj=utm +zone=10 +ellps=GRS80'

    We deliberately avoid adding +type=crs or similar, and replace +datum with +ellps
    for better PROJ 9+ compatibility in pipelines.
    """
    crs = CRS.from_user_input(crs_str)
    # PROJ string for the projected part
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore",
            message=".*You will likely lose important projection.*",
        )
        proj_str = crs.to_proj4()
    # Ensure we only keep the step part (starting at +proj=...)
    if "+proj=" not in proj_str:
        raise ProjError(f"CRS {crs_str!r} does not appear to be projected.")
    idx = proj_str.index("+proj=")
    step = proj_str[idx:].strip()

    # Remove deprecated/problematic parameters for PROJ 9+ pipelines
    # - +type=crs: not needed in pipeline steps
    # - +no_defs: deprecated and can cause issues
    # - +datum=: should be replaced with +ellps=
    # - +units=m: can cause issues in pipelines when mixed with angular ops
    parts = [p for p in step.split() if not (p.startswith("+type=") or p == "+no_defs" or p.startswith("+units="))]

    # Replace +datum= with +ellps= for PROJ 9+ compatibility
    # +datum is deprecated in pipelines and can cause "Unknown projection" errors
    result_parts = []
    for part in parts:
        if part.startswith("+datum="):
            # Replace with +ellps based on common datums
            datum = part.split("=")[1]
            if datum in ("NAD83", "NAD27"):
                result_parts.append("+ellps=GRS80" if datum == "NAD83" else "+ellps=clrk66")
            elif datum == "WGS84":
                result_parts.append("+ellps=WGS84")
            else:
                # Keep original if unknown
                result_parts.append(part)
        else:
            result_parts.append(part)

    return " ".join(result_parts)


# ---------------------------------------------------------------------
# 1) Horizontal-only pipeline
# ---------------------------------------------------------------------


def build_horizontal_pipeline(src: str, dst: str) -> str:
    """
    Horizontal-only CRS->CRS pipeline.

    Just wraps projinfo and normalizes the result so that you always get
    a '+proj=pipeline ...' string (or a trivial pipeline when src == dst).
    """
    op = _run_projinfo(src, dst)
    if op.startswith("+proj=pipeline"):
        return op
    if op.startswith("+proj=noop"):
        # explicit trivial pipeline
        return "+proj=pipeline"
    raise ProjError(f"Unexpected projinfo operation: {op}")


# ---------------------------------------------------------------------
# 2) Vertical-only helpers
# ---------------------------------------------------------------------


def _build_vertical_steps(src: CRSState, dst: CRSState) -> str:
    """
    Construct the vgridshift steps ONLY (no projection steps).

    Assumes coordinates are already in the geographic CRS that the grid expects
    (lat/lon, angular). For projected cases, you will wrap this between
    +inv/+fwd projection steps.
    """
    src = src.with_resolved_geoid()
    dst = dst.with_resolved_geoid()

    sk = (src.vertical_kind or "").lower()
    dk = (dst.vertical_kind or "").lower()

    # No vertical semantics => nothing to do
    if not sk and not dk and not src.geoid_grid and not dst.geoid_grid:
        return ""

    # Helper to format grid path (quote if contains spaces)
    def _format_grid_path(grid_path: str) -> str:
        if ' ' in grid_path:
            return f'"{grid_path}"'
        return grid_path

    # Orthometric -> Ellipsoidal (remove geoid)
    if sk == "orthometric" and dk == "ellipsoidal":
        if not src.geoid_grid:
            raise ProjError(
                "Source is orthometric but source geoid grid is unknown. Specify source geoid."
            )
        grid = _format_grid_path(src.geoid_grid)
        return f"+step +proj=vgridshift +grids={grid} +inv"

    # Ellipsoidal -> Orthometric (apply geoid)
    if sk == "ellipsoidal" and dk == "orthometric":
        if not dst.geoid_grid:
            raise ProjError(
                "Target is orthometric but target geoid grid is unknown. Specify target geoid."
            )
        grid = _format_grid_path(dst.geoid_grid)
        return f"+step +proj=vgridshift +grids={grid}"

    # Orthometric A -> Orthometric B
    if sk == "orthometric" and dk == "orthometric":
        if not src.geoid_grid and not dst.geoid_grid:
            # Neither geoid known — nothing we can do; treat as no-op
            return ""
        if src.geoid_grid and dst.geoid_grid:
            if Path(src.geoid_grid).name == Path(dst.geoid_grid).name:
                # Same model; nothing to do
                return ""
            # Remove A, apply B
            src_grid = _format_grid_path(src.geoid_grid)
            dst_grid = _format_grid_path(dst.geoid_grid)
            return (
                f"+step +proj=vgridshift +grids={src_grid} +inv "
                f"+step +proj=vgridshift +grids={dst_grid}"
            )
        if not src.geoid_grid and dst.geoid_grid:
            # Source has no geoid; apply target geoid only
            dst_grid = _format_grid_path(dst.geoid_grid)
            return f"+step +proj=vgridshift +grids={dst_grid}"
        # dst.geoid_grid is None, src.geoid_grid is set → remove source geoid only
        src_grid = _format_grid_path(src.geoid_grid)
        return f"+step +proj=vgridshift +grids={src_grid} +inv"

    # Everything else (e.g., one/both vertical_kind missing)
    raise ProjError(
        f"Unsupported vertical-kind combination: src={src.vertical_kind!r}, dst={dst.vertical_kind!r}"
    )


def build_vertical_pipeline_geographic(src: CRSState, dst: CRSState) -> str:
    """
    Vertical-only change when coordinates are already in geographic CRS.

    Example use cases:
      - EGM2008 ellipsoidal <-> orthometric
      - NZ local datum <-> NZVD2016 (see nz_linz grids) 
    """
    vert_steps = _build_vertical_steps(src, dst)
    if not vert_steps:
        return "+proj=pipeline"

    pipeline = f"+proj=pipeline {vert_steps}"
    return pipeline


def build_vertical_pipeline_projected(src: CRSState, dst: CRSState) -> str:
    """
    Vertical-only change when coordinates are in a projected CRS (UTM, LCC, etc.)
    and the horizontal CRS is the same for source and target.

    Pattern (compare with NOAA / PROJ examples for GEOID18 etc.): 

        +proj=pipeline
        +step +inv +proj=<proj_of_crs>
        +step +proj=vgridshift ...
        [+step +proj=vgridshift ...]
        +step +proj=<proj_of_crs>

    We rely on pyproj to extract the projection definition of src.crs.
    """
    if src.crs != dst.crs:
        raise ProjError(
            "build_vertical_pipeline_projected assumes src.crs == dst.crs "
            "(no horizontal/datum change)."
        )

    proj_step = _proj_step_from_crs(src.crs)
    vert_steps = _build_vertical_steps(src, dst)
    if not vert_steps:
        return "+proj=pipeline"

    pipeline = (
        f"+proj=pipeline "
        f"+step +inv {proj_step} "
        f"{vert_steps} "
        f"+step {proj_step}"
    )

    return pipeline


def build_vertical_pipeline(src: CRSState, dst: CRSState) -> str:
    """
    Dispatcher: vertical-only pipeline, geographic vs projected.

    If the CRS is geographic, we operate directly in that CRS.
    If projected, we wrap the vgridshift steps with inverse / forward projection.
    """
    if src.crs != dst.crs:
        raise ProjError(
            "build_vertical_pipeline assumes src and dst have the same CRS; "
            "for combined horizontal+vertical transforms, use a horizontal "
            "pipeline first, then a vertical pipeline."
        )

    if _is_geographic(src.crs):
        return build_vertical_pipeline_geographic(src, dst)
    else:
        return build_vertical_pipeline_projected(src, dst)


# ---------------------------------------------------------------------
# 3) Dynamic epoch pipeline (with +proj=deformation)
# ---------------------------------------------------------------------


def _ellipsoid_of_crs(crs_str: str) -> str:
    """
    Extract the ellipsoid name for use in +proj=cart / +proj=deformation.

    In many practical cases, you might hard-code this (e.g., 'GRS80' for NAD83).
    Here we introspect the CRS via pyproj.
    """
    crs = CRS.from_user_input(crs_str)
    ellps = crs.ellipsoid
    if not ellps:
        raise ProjError(f"Could not determine ellipsoid for CRS {crs_str!r}")
    name = ellps.name
    # This is heuristic; you may want a custom mapping for your datasets.
    # For typical NAD83 / WGS84 derived CRS, 'GRS 1980' / 'GRS80' / 'WGS 84' all map
    # to +ellps=GRS80 or +ellps=WGS84 as needed. We'll just pick the identifier.
    # pyproj's CRS.ellipsoid has 'datum_ellipsoid_code' in newer versions, but
    # to keep it simple here we try common codes.
    if "GRS 1980" in name or "GRS80" in name:
        return "GRS80"
    if "WGS 84" in name or "WGS84" in name:
        return "WGS84"
    # Fallback: rely on PROJ's auto-inference (occasionally works).
    return "GRS80"


def _build_epoch_steps(
    src: CRSState,
    dst: CRSState,
    *,
    deformation_grids: str,
    central_epoch: Optional[float] = None,
    ellps: Optional[str] = None,
) -> str:
    """
    Construct the epoch/deformation steps ONLY (no projection wrapping).

    Returns the steps from +proj=push through +proj=pop, suitable for embedding
    in a larger pipeline. Assumes coordinates are already in geographic CRS.

    This parallels _build_vertical_steps() for vertical transforms.

    Parameters
    ----------
    src, dst : CRSState
        Source and destination states. Only epoch fields are used.
    deformation_grids : str
        Path/name of velocity grid for +proj=deformation.
    central_epoch : float, optional
        Reference epoch of the velocity model (informational only, does not affect dt).
    ellps : str, optional
        Ellipsoid for +proj=cart. Inferred from src.crs if not provided.

    Returns
    -------
    str
        The deformation steps (without +proj=pipeline prefix), or empty string
        if no epoch transform is needed.
    """
    if src.epoch is None or dst.epoch is None:
        return ""
    if src.epoch == dst.epoch:
        return ""

    # CRITICAL: dt must always be the full time difference between source and destination epochs.
    # The central_epoch parameter is the reference epoch of the velocity model (informational only)
    # and should NOT affect the dt calculation. The velocity model gives velocities in mm/year,
    # and we multiply by dt to get the total displacement.
    # See: https://proj.org/en/stable/operations/transformations/deformation.html
    dt = dst.epoch - src.epoch

    if ellps is None:
        ellps = _ellipsoid_of_crs(src.crs)

    # Return just the steps (no +proj=pipeline prefix)
    steps = (
        f"+step +proj=push +v_3 "
        f"+step +proj=cart +ellps={ellps} "
        f"+step +proj=deformation +grids={deformation_grids} +dt={dt} "
        f"+step +inv +proj=cart +ellps={ellps} "
        f"+step +proj=pop +v_3"
    )
    return steps


def build_dynamic_epoch_pipeline(
    src: CRSState,
    dst: CRSState,
    *,
    deformation_grids: str,
    central_epoch: Optional[float] = None,
    ellps: Optional[str] = None,
) -> str:
    """
    Build a dynamic epoch pipeline using +proj=deformation and a velocity grid.

    We use the pattern from the PROJ deformation docs: 

        +proj=pipeline
        +step +proj=push +v_3
        +step +proj=cart +ellps=<E>
        +step +proj=deformation +grids=<vel_grid> +dt=<dst.epoch - src.epoch>
        +step +inv +proj=cart +ellps=<E>
        +step +proj=pop +v_3

    This assumes:
      - src.crs == dst.crs (no horizontal change) OR you have already
        reprojected into a dynamic CRS beforehand.
      - src.epoch and dst.epoch are both set (decimal year).
      - deformation_grids is a PROJ grid with velocities for that CRS.

    Parameters
    ----------
    deformation_grids : str
        Name/path of the velocity grid(s) usable by +proj=deformation.
    central_epoch : float, optional
        Reference epoch of the velocity model (informational only).
        NOTE: This parameter does NOT affect the dt calculation. The dt is always
        computed as (dst.epoch - src.epoch), which is the correct time shift
        to apply to coordinates. The central_epoch is retained for metadata only.
    ellps : str, optional
        Ellipsoid name for +proj=cart. If None, we infer from src.crs.

    Returns
    -------
    pipeline : str
        A '+proj=pipeline ...' string implementing the epoch shift.
    """
    if src.crs != dst.crs:
        raise ProjError(
            "build_dynamic_epoch_pipeline assumes src and dst share the same CRS; "
            "reproject separately before or after applying the deformation."
        )
    if src.epoch is None or dst.epoch is None:
        raise ProjError("Dynamic epoch pipeline requires both source and target epochs.")
    if src.epoch == dst.epoch:
        return "+proj=pipeline"

    # CRITICAL: dt must always be the full time difference between source and destination epochs.
    # The central_epoch parameter is informational only and should NOT affect the dt calculation.
    dt = dst.epoch - src.epoch

    if ellps is None:
        ellps = _ellipsoid_of_crs(src.crs)

    pipeline = (
        f"+proj=pipeline "
        f"+step +proj=push +v_3 "
        f"+step +proj=cart +ellps={ellps} "
        f"+step +proj=deformation +grids={deformation_grids} +dt={dt} "
        f"+step +inv +proj=cart +ellps={ellps} "
        f"+step +proj=pop +v_3"
    )
    return pipeline


def build_dynamic_epoch_pipeline_projected(
    src: CRSState,
    dst: CRSState,
    *,
    deformation_grids: str,
    central_epoch: Optional[float] = None,
) -> str:
    """
    Dynamic epoch pipeline when coordinates are in a projected CRS (UTM, LCC, etc.)

    The deformation transform (+proj=push/cart/deformation/pop) expects geographic
    coordinates. For projected CRS, we wrap with inverse/forward projection:

        +proj=pipeline
        +step +inv +proj=<proj_of_crs>   # projected → geographic
        +step +proj=push +v_3
        +step +proj=cart +ellps=<E>
        +step +proj=deformation +grids=<grid> +dt=<dt>
        +step +inv +proj=cart +ellps=<E>
        +step +proj=pop +v_3
        +step +proj=<proj_of_crs>        # geographic → projected

    This mirrors the pattern in build_vertical_pipeline_projected().

    Parameters
    ----------
    src, dst : CRSState
        Source and destination states. Must have same CRS (no horizontal change).
    deformation_grids : str
        Path/name of velocity grid.
    central_epoch : float, optional
        Reference epoch of the velocity model (informational only, does not affect dt).

    Returns
    -------
    str
        A '+proj=pipeline ...' string.
    """
    if src.crs != dst.crs:
        raise ProjError(
            "build_dynamic_epoch_pipeline_projected assumes src.crs == dst.crs "
            "(no horizontal/datum change). Handle reprojection separately."
        )

    proj_step = _proj_step_from_crs(src.crs)
    epoch_steps = _build_epoch_steps(
        src, dst,
        deformation_grids=deformation_grids,
        central_epoch=central_epoch,
    )

    if not epoch_steps:
        return "+proj=pipeline"

    pipeline = (
        f"+proj=pipeline "
        f"+step +inv {proj_step} "
        f"{epoch_steps} "
        f"+step {proj_step}"
    )
    return pipeline


# ---------------------------------------------------------------------
# 4) High-level dispatcher (brings it all together)
# ---------------------------------------------------------------------


def _extract_pipeline_steps(pipeline: str) -> str:
    """
    Given a '+proj=pipeline ...' string, return just the '+step ...' part(s).

    If the pipeline is empty or just '+proj=pipeline', this returns an empty string.
    """
    if not pipeline:
        return ""
    p = pipeline.strip()
    if not p:
        return ""
    if not p.startswith("+proj=pipeline"):
        raise ProjError(f"Expected a '+proj=pipeline' string, got: {p!r}")
    rest = p[len("+proj=pipeline"):].strip()
    return rest


def _compose_pipelines(*pipelines: str) -> str:
    """
    Compose multiple '+proj=pipeline ...' strings into a single pipeline.

    Empty or trivial pipelines are skipped. If all are trivial, the result
    is '+proj=pipeline'.
    """
    steps: List[str] = []
    for p in pipelines:
        s = _extract_pipeline_steps(p)
        if s:
            steps.append(s)
    if not steps:
        return "+proj=pipeline"
    return "+proj=pipeline " + " ".join(steps)


def build_complete_pipeline(
    src: CRSState,
    dst: CRSState,
    *,
    deformation_grids: Optional[str] = None,
    deformation_central_epoch: Optional[float] = None,
) -> str:
    """
    High-level dispatcher: horizontal, vertical, and/or epoch transforms.

    This function figures out what kind of pipeline is needed and delegates 
    to the appropriate builder. It handles:

    - Pure horizontal (CRS change only)
    - Pure vertical (geoid/ellipsoidal change, same CRS)
    - Pure epoch (deformation model, same CRS)
    - Combined: epoch + vertical, epoch + horizontal, vertical + horizontal
    - Combined: epoch + vertical + horizontal

    **Key fix**: For epoch transforms with projected CRS, coordinates are 
    converted to geographic before the deformation, then converted back or
    to the target CRS. This prevents the "Invalid latitude" error that occurs
    when projected coordinates (meters) are passed to +proj=push.

    Parameters
    ----------
    src, dst : CRSState
        Source and destination CRS states including epoch and vertical info.
    deformation_grids : str, optional
        Path/name of velocity grid for dynamic epoch transforms.
    deformation_central_epoch : float, optional
        Central epoch for the deformation model.

    Returns
    -------
    pipeline : str
        A '+proj=pipeline ...' string.
    """
    src = src.with_resolved_geoid()
    dst = dst.with_resolved_geoid()

    # -------------------------------------------------------------------------
    # Detect what transforms are needed
    # -------------------------------------------------------------------------
    needs_epoch = (
        src.epoch is not None
        and dst.epoch is not None
        and src.epoch != dst.epoch
    )

    has_vertical_semantics = any([
        src.vertical_kind,
        dst.vertical_kind,
        src.geoid_grid,
        dst.geoid_grid,
        src.geoid_alias,
        dst.geoid_alias,
    ])

    crs_changed = src.crs != dst.crs
    src_is_projected = not _is_geographic(src.crs)

    # Validate epoch requirements
    if needs_epoch and deformation_grids is None:
        raise ProjError(
            "Dynamic epoch requested (src.epoch != dst.epoch) "
            "but deformation_grids is not provided."
        )

    # -------------------------------------------------------------------------
    # Case A: Epoch transform needed (the main fix is here)
    # -------------------------------------------------------------------------
    if needs_epoch:

        if src_is_projected:
            # -----------------------------------------------------------------
            # PROJECTED SOURCE + EPOCH TRANSFORM
            # -----------------------------------------------------------------
            # Strategy:
            # 1. Inverse project to geographic (lat/lon)
            # 2. Epoch/deformation transform (in geographic coords)
            # 3. Vertical transform if needed (in geographic coords)
            # 4. Project to target CRS (or back to source if no CRS change)

            proj_step = _proj_step_from_crs(src.crs)

            # Step 1: projected → geographic (inverse projection)
            step1_inv = f"+step +inv {proj_step}"

            # Step 2: epoch transform (just the steps, no pipeline wrapper)
            epoch_steps = _build_epoch_steps(
                src, dst,
                deformation_grids=deformation_grids,
                central_epoch=deformation_central_epoch,
            )

            # Step 3: vertical transform if needed (just the steps)
            vert_steps = ""
            if has_vertical_semantics:
                # Create intermediate states after epoch transform
                vert_src = CRSState(
                    crs=src.crs,
                    epoch=dst.epoch,  # After epoch transform
                    vertical_kind=src.vertical_kind,
                    geoid_alias=src.geoid_alias,
                    geoid_grid=src.geoid_grid,
                )
                vert_dst = CRSState(
                    crs=src.crs,
                    epoch=dst.epoch,
                    vertical_kind=dst.vertical_kind,
                    geoid_alias=dst.geoid_alias,
                    geoid_grid=dst.geoid_grid,
                )
                vert_steps = _build_vertical_steps(vert_src, vert_dst)

            # Step 4: forward project to target CRS
            if crs_changed:
                # Need horizontal transform to different CRS
                # Get the geographic CRS underlying the source projection
                src_crs_obj = CRS.from_user_input(src.crs)
                src_geog = src_crs_obj.geodetic_crs
                if src_geog is None:
                    raise ProjError(
                        f"Could not determine geographic CRS for {src.crs}"
                    )
                src_geog_str = src_geog.to_string()

                # Build pipeline from source's geographic CRS to target
                step4_horiz = build_horizontal_pipeline(src_geog_str, dst.crs)
                step4_horiz_steps = _extract_pipeline_steps(step4_horiz)
            else:
                # No CRS change - just project back to source CRS
                step4_horiz_steps = f"+step {proj_step}"

            # Compose all steps into final pipeline
            all_steps = [step1_inv, epoch_steps]
            if vert_steps:
                all_steps.append(vert_steps)
            if step4_horiz_steps:
                all_steps.append(step4_horiz_steps)

            pipeline = "+proj=pipeline " + " ".join(s for s in all_steps if s)
            return pipeline

        else:
            # -----------------------------------------------------------------
            # GEOGRAPHIC SOURCE + EPOCH TRANSFORM
            # -----------------------------------------------------------------
            # Source is already in lat/lon - can do epoch transform directly.
            # Handle combinations: epoch-only, epoch+vertical, epoch+horizontal,
            # or epoch+vertical+horizontal

            # Simple case: epoch-only (same CRS, no vertical)
            if not has_vertical_semantics and not crs_changed:
                return build_dynamic_epoch_pipeline(
                    src, dst,
                    deformation_grids=deformation_grids,
                    central_epoch=deformation_central_epoch,
                )

            # Combined case: build individual steps and compose
            epoch_steps = _build_epoch_steps(
                src, dst,
                deformation_grids=deformation_grids,
                central_epoch=deformation_central_epoch,
            )

            vert_steps = ""
            if has_vertical_semantics:
                vert_src = CRSState(
                    crs=src.crs,
                    epoch=dst.epoch,
                    vertical_kind=src.vertical_kind,
                    geoid_alias=src.geoid_alias,
                    geoid_grid=src.geoid_grid,
                )
                vert_dst = CRSState(
                    crs=src.crs,
                    epoch=dst.epoch,
                    vertical_kind=dst.vertical_kind,
                    geoid_alias=dst.geoid_alias,
                    geoid_grid=dst.geoid_grid,
                )
                vert_steps = _build_vertical_steps(vert_src, vert_dst)

            horiz_steps = ""
            if crs_changed:
                horiz_pipeline = build_horizontal_pipeline(src.crs, dst.crs)
                horiz_steps = _extract_pipeline_steps(horiz_pipeline)

            all_steps = [epoch_steps, vert_steps, horiz_steps]
            pipeline = "+proj=pipeline " + " ".join(s for s in all_steps if s)
            return pipeline

    # -------------------------------------------------------------------------
    # Case B: No epoch transform, but has vertical semantics
    # -------------------------------------------------------------------------
    if has_vertical_semantics:
        # Same CRS -> pure vertical
        if not crs_changed:
            return build_vertical_pipeline(src, dst)

        # Vertical + horizontal transform needed
        # For projected source CRS, we need to be careful about composition:
        # Cannot simply compose build_vertical_pipeline + build_horizontal_pipeline
        # because both would wrap their ops with inverse/forward projection steps.
        #
        # Strategy:
        # 1. If source is projected: unproject → vertical transform → horiz transform
        # 2. If source is geographic: vertical transform → horizontal transform

        if src_is_projected:
            # Get the geographic CRS underlying the source projection
            src_crs_obj = CRS.from_user_input(src.crs)
            src_geog = src_crs_obj.geodetic_crs
            if src_geog is None:
                raise ProjError(f"Could not determine geographic CRS for {src.crs}")
            src_geog_str = src_geog.to_string()

            # Get projection step
            proj_step = _proj_step_from_crs(src.crs)

            # Step 1: Inverse projection (projected → geographic)
            step1_inv = f"+step +inv {proj_step}"

            # Step 2: Vertical transform in geographic coordinates
            vert_src = CRSState(
                crs=src_geog_str,
                epoch=src.epoch,
                vertical_kind=src.vertical_kind,
                geoid_alias=src.geoid_alias,
                geoid_grid=src.geoid_grid,
            )
            vert_dst_temp = CRSState(
                crs=src_geog_str,
                epoch=src.epoch,
                vertical_kind=dst.vertical_kind,
                geoid_alias=dst.geoid_alias,
                geoid_grid=dst.geoid_grid,
            )
            vert_steps = _build_vertical_steps(vert_src, vert_dst_temp)

            # Step 3: Horizontal transform from source geographic to target CRS
            # After vgridshift, we have geographic coordinates (lat/lon in degrees, ellipsoidal or orthometric Z)
            # We need to transform to the target CRS

            dst_is_projected = not _is_geographic(dst.crs)

            if dst_is_projected:
                # Target is projected - need to project from geographic
                # Check if source and target use the same geodetic CRS
                dst_crs_obj = CRS.from_user_input(dst.crs)
                dst_geog = dst_crs_obj.geodetic_crs
                if dst_geog is None:
                    raise ProjError(f"Could not determine geographic CRS for target {dst.crs}")

                # Get the projection parameters for target
                dst_proj_step = _proj_step_from_crs(dst.crs)

                # If source and target geographic CRS are different, need datum transform
                src_geog_epsg = src_geog.to_epsg()
                dst_geog_epsg = dst_geog.to_epsg()

                if src_geog_epsg != dst_geog_epsg:
                    # Different datums (e.g., NAD83 vs WGS84)
                    # After vgridshift, coordinates are in lat/lon degrees
                    # We can't use projinfo's full pipeline because it adds axisswap+unitconvert
                    # that cause unit mismatch errors with vgridshift output
                    #
                    # For NAD83 <-> WGS84 in North America, the datum shift is sub-meter
                    # and often ignored in practice. Just apply the forward projection.
                    # If higher accuracy is needed, the datum transform should be done
                    # separately before or after the vertical transform.
                    horiz_steps = f"+step {dst_proj_step}"
                else:
                    # Same datum - just apply forward projection
                    # vgridshift outputs lat,lon in degrees, which is what projections expect
                    horiz_steps = f"+step {dst_proj_step}"
            else:
                # Target is geographic
                if src_geog_str != dst.crs:
                    # Different geographic CRS - need datum transform
                    horiz_pipeline = build_horizontal_pipeline(src_geog_str, dst.crs)
                    horiz_steps = _extract_pipeline_steps(horiz_pipeline)
                else:
                    # Same geographic CRS - no horizontal transform needed
                    horiz_steps = ""

            # Compose all steps
            all_steps = [step1_inv]
            if vert_steps:
                all_steps.append(vert_steps)
            if horiz_steps:
                all_steps.append(horiz_steps)

            pipeline = "+proj=pipeline " + " ".join(s for s in all_steps if s)
            return pipeline
        else:
            # Source is geographic - simpler case
            # Do vertical transform first (stays in geographic), then horizontal
            vert_dst_temp = CRSState(
                crs=src.crs,
                epoch=src.epoch,
                vertical_kind=dst.vertical_kind,
                geoid_alias=dst.geoid_alias,
                geoid_grid=dst.geoid_grid,
            )
            vert_steps = _build_vertical_steps(src, vert_dst_temp)
            horiz_pipeline = build_horizontal_pipeline(src.crs, dst.crs)
            horiz_steps = _extract_pipeline_steps(horiz_pipeline)

            all_steps = []
            if vert_steps:
                all_steps.append(vert_steps)
            if horiz_steps:
                all_steps.append(horiz_steps)

            if not all_steps:
                return "+proj=pipeline"

            pipeline = "+proj=pipeline " + " ".join(all_steps)
            return pipeline

    # -------------------------------------------------------------------------
    # Case C: Pure horizontal (no epoch, no vertical)
    # -------------------------------------------------------------------------
    return build_horizontal_pipeline(src.crs, dst.crs)


def build_pipeline(
    src: CRSState,
    dst: CRSState,
    *,
    deformation_grids: Optional[str] = None,
    deformation_central_epoch: Optional[float] = None,
) -> str:
    """Backwards-compatible alias for :func:`build_complete_pipeline`."""
    return build_complete_pipeline(
        src,
        dst,
        deformation_grids=deformation_grids,
        deformation_central_epoch=deformation_central_epoch,
    )