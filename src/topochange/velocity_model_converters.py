"""format converters for velocity/deformation models to PROJ-compatible GeoTIFF."""

from __future__ import annotations

import json
import os
import re
import struct
import warnings
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS as RioCRS
except ImportError:
    rasterio = None

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    from scipy.interpolate import griddata
except ImportError:
    griddata = None


# base converter class

class VelocityConverter:
    """Base class for velocity model format converters."""
    
    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        **kwargs,
    ) -> Path:
        """
        Convert source format to PROJ-compatible GeoTIFF.
        
        Args:
            input_path: Source file/directory
            output_path: Output GeoTIFF path
            **kwargs: Format-specific options
            
        Returns:
            Path to created GeoTIFF
        """
        raise NotImplementedError


# uCERF3 OpenSHA converter

class UCERF3Converter(VelocityConverter):
    """
    Convert UCERF3 deformation model to GeoTIFF.
    
    UCERF3 provides:
      - Fault slip rates (in OpenSHA FaultSystemSolution format)
      - Off-fault strain rates on 0.1° grid
      
    We extract the gridded strain rates and convert to velocities.
    """
    
    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        model_variant: str = "MeanUCERF3",
        **kwargs,
    ) -> Path:
        """
        Convert UCERF3 to GeoTIFF velocity grid.
        
        Args:
            input_path: Path to UCERF3 zip or extracted directory
            output_path: Output GeoTIFF path
            model_variant: Which UCERF3 variant ('MeanUCERF3', 'GEOLOGIC', etc.)
        """
        if rasterio is None:
            raise ImportError("rasterio required for UCERF3 conversion")
        
        input_path = Path(input_path)
        
        # extract if zip
        if input_path.suffix == '.zip':
            extract_dir = input_path.parent / input_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(input_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            work_dir = extract_dir
        else:
            work_dir = input_path
        
        # find strain rate files
        # uCERF3 format: off-fault strain on 0.1 degree grid
        # typically: *_strain_rates.txt or similar
        strain_files = list(work_dir.glob(f"*{model_variant}*strain*.txt"))
        if not strain_files:
            strain_files = list(work_dir.glob("*strain*.txt"))
        
        if not strain_files:
            raise FileNotFoundError(
                f"No strain rate files found in {work_dir}. "
                f"Expected *strain*.txt"
            )
        
        strain_file = strain_files[0]
        
        # parse UCERF3 strain rate file
        # format: lon, lat, exx, eyy, exy (strain rate tensor components)
        data = np.loadtxt(strain_file, skiprows=1)
        
        if data.shape[1] < 5:
            raise ValueError(
                f"Expected at least 5 columns (lon, lat, exx, eyy, exy), "
                f"got {data.shape[1]}"
            )
        
        lons = data[:, 0]
        lats = data[:, 1]
        exx = data[:, 2]  # E-W strain rate (1/yr)
        eyy = data[:, 3]  # N-S strain rate (1/yr)
        exy = data[:, 4]  # Shear strain rate (1/yr)
        
        # convert strain rates to velocities
        # this is simplified - proper conversion requires:
        # 1. Integration of strain field
        # 2. Boundary conditions (plate motion)
        # 3. Rotation consideration
        
        # for now, approximate using strain × characteristic length
        # this gives relative velocities, not absolute
        R_earth = 6371000  # meters
        deg_to_rad = np.pi / 180
        lat_rad = lats * deg_to_rad
        
        # approximate velocity from strain (simplified)
        # v = strain_rate × distance
        # use local earth radius for characteristic scale
        dx = 0.1 * deg_to_rad * R_earth * np.cos(lat_rad)  # E-W distance
        dy = 0.1 * deg_to_rad * R_earth  # N-S distance
        
        # velocity components (mm/yr)
        ve = exx * dx * 1000  # East velocity
        vn = eyy * dy * 1000  # North velocity
        
        # for vertical: assume negligible from horizontal strain
        # (would need full 3D model for proper vertical)
        vu = np.zeros_like(ve)
        
        # create regular grid
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        # uCERF3 uses 0.1 degree spacing
        resolution = 0.1
        nx = int((lon_max - lon_min) / resolution) + 1
        ny = int((lat_max - lat_min) / resolution) + 1
        
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # interpolate to regular grid
        if griddata is None:
            raise ImportError("scipy required for gridding")
        
        print(f"Gridding UCERF3 data: {len(lons)} points → {nx}x{ny} grid")
        
        ve_grid = griddata((lons, lats), ve, (lon_mesh, lat_mesh), method='linear')
        vn_grid = griddata((lons, lats), vn, (lon_mesh, lat_mesh), method='linear')
        vu_grid = np.zeros_like(ve_grid)
        
        # write to GeoTIFF
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=3,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='deflate',
        ) as dst:
            dst.write(ve_grid.astype(np.float32), 1)
            dst.write(vn_grid.astype(np.float32), 2)
            dst.write(vu_grid.astype(np.float32), 3)
            
            dst.set_band_description(1, 'east_velocity')
            dst.set_band_description(2, 'north_velocity')
            dst.set_band_description(3, 'up_velocity')
            
            dst.update_tags(1, units='mm/year')
            dst.update_tags(2, units='mm/year')
            dst.update_tags(3, units='mm/year')
            
            dst.update_tags(
                source='UCERF3',
                model_variant=model_variant,
                conversion_note='Velocities approximated from strain rates',
            )
        
        print(f"Created UCERF3 velocity grid: {output_path}")
        return Path(output_path)


# uSGS NSHM NetCDF converter

class USGSNSHMConverter(VelocityConverter):
    """
    Convert USGS NSHM 2023 geodetic models to GeoTIFF.
    
    NSHM data release includes multiple model variants:
      - Pollitz block model
      - Zeng smoothed model
      - Shen model
      - Evans model
    """
    
    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        model_variant: str = "pollitz",
        **kwargs,
    ) -> Path:
        """
        Convert USGS NSHM NetCDF to GeoTIFF.
        
        Args:
            input_path: NetCDF file or directory
            output_path: Output GeoTIFF
            model_variant: Which model ('pollitz', 'zeng', 'shen', 'evans')
        """
        if xr is None:
            raise ImportError("xarray required for NetCDF conversion")
        if rasterio is None:
            raise ImportError("rasterio required for GeoTIFF writing")
        
        input_path = Path(input_path)
        
        # find NetCDF file
        if input_path.is_dir():
            nc_files = list(input_path.glob(f"*{model_variant}*.nc"))
            if not nc_files:
                nc_files = list(input_path.glob("*.nc"))
            if not nc_files:
                raise FileNotFoundError(f"No NetCDF files in {input_path}")
            nc_file = nc_files[0]
        else:
            nc_file = input_path
        
        # load NetCDF
        print(f"Loading {nc_file}")
        ds = xr.open_dataset(nc_file)
        
        # extract velocity components
        # nSHM format may vary - try common variable names
        var_names = {
            've': ['velocity_east', 've', 'east_velocity', 'veast'],
            'vn': ['velocity_north', 'vn', 'north_velocity', 'vnorth'],
            'vu': ['velocity_up', 'vu', 'up_velocity', 'vup', 'vertical_velocity'],
        }
        
        def find_var(names: List[str]) -> Optional[str]:
            for name in names:
                if name in ds.variables:
                    return name
            return None
        
        ve_var = find_var(var_names['ve'])
        vn_var = find_var(var_names['vn'])
        vu_var = find_var(var_names['vu'])
        
        if not ve_var or not vn_var:
            raise ValueError(
                f"Could not find velocity variables in NetCDF. "
                f"Available: {list(ds.variables.keys())}"
            )
        
        ve = ds[ve_var].values
        vn = ds[vn_var].values
        vu = ds[vu_var].values if vu_var else np.zeros_like(ve)
        
        # get coordinates
        # try common names
        lon_var = find_var(['lon', 'longitude', 'x'])
        lat_var = find_var(['lat', 'latitude', 'y'])
        
        if not lon_var or not lat_var:
            raise ValueError(
                f"Could not find coordinate variables. "
                f"Available: {list(ds.variables.keys())}"
            )
        
        lons = ds[lon_var].values
        lats = ds[lat_var].values
        
        # convert to mm/year if needed
        # check units
        ve_units = ds[ve_var].attrs.get('units', 'm/year')
        if 'm/year' in ve_units.lower() or 'm/yr' in ve_units.lower():
            print(f"Converting from m/year to mm/year")
            ve *= 1000
            vn *= 1000
            vu *= 1000
        
        # determine grid spacing and create GeoTIFF
        if lons.ndim == 1 and lats.ndim == 1:
            # 1D coordinate arrays - create mesh
            ny, nx = len(lats), len(lons)
            transform = from_bounds(lons[0], lats[0], lons[-1], lats[-1], nx, ny)
        else:
            # 2D coordinate arrays
            ny, nx = lons.shape
            lon_min, lon_max = lons.min(), lons.max()
            lat_min, lat_max = lats.min(), lats.max()
            transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
        
        # write GeoTIFF
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=3,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='deflate',
        ) as dst:
            dst.write(ve.astype(np.float32), 1)
            dst.write(vn.astype(np.float32), 2)
            dst.write(vu.astype(np.float32), 3)
            
            dst.set_band_description(1, 'east_velocity')
            dst.set_band_description(2, 'north_velocity')
            dst.set_band_description(3, 'up_velocity')
            
            dst.update_tags(1, units='mm/year')
            dst.update_tags(2, units='mm/year')
            dst.update_tags(3, units='mm/year')
            
            dst.update_tags(
                source='USGS_NSHM_2023',
                model_variant=model_variant,
            )
        
        ds.close()
        
        print(f"Created USGS NSHM velocity grid: {output_path}")
        return Path(output_path)


# gEM GSRM ASCII converter

class GSRMConverter(VelocityConverter):
    """
    Convert GEM GSRM ASCII strain rate grids to velocity GeoTIFF.
    
    GSRM provides strain rate tensors globally.
    """
    
    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        **kwargs,
    ) -> Path:
        """
        Convert GSRM ASCII to GeoTIFF.
        
        Similar approach to UCERF3 - convert strain rates to velocities.
        """
        if rasterio is None:
            raise ImportError("rasterio required")
        
        input_path = Path(input_path)
        
        # find strain rate files
        if input_path.is_dir():
            ascii_files = list(input_path.glob("*.txt"))
            if not ascii_files:
                ascii_files = list(input_path.glob("*.dat"))
            if not ascii_files:
                raise FileNotFoundError(f"No ASCII files in {input_path}")
            ascii_file = ascii_files[0]
        else:
            ascii_file = input_path
        
        # parse GSRM format
        # typical: lon, lat, exx, eyy, exy, [uncertainties]
        data = np.loadtxt(ascii_file, skiprows=1)
        
        lons = data[:, 0]
        lats = data[:, 1]
        exx = data[:, 2]
        eyy = data[:, 3]
        exy = data[:, 4]
        
        # convert to velocities (same approach as UCERF3)
        R_earth = 6371000
        deg_to_rad = np.pi / 180
        lat_rad = lats * deg_to_rad
        
        # estimate grid spacing
        unique_lons = np.unique(lons)
        resolution = np.median(np.diff(unique_lons))
        
        dx = resolution * deg_to_rad * R_earth * np.cos(lat_rad)
        dy = resolution * deg_to_rad * R_earth
        
        ve = exx * dx * 1000
        vn = eyy * dy * 1000
        vu = np.zeros_like(ve)
        
        # grid
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        nx = int((lon_max - lon_min) / resolution) + 1
        ny = int((lat_max - lat_min) / resolution) + 1
        
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        if griddata is None:
            raise ImportError("scipy required")
        
        ve_grid = griddata((lons, lats), ve, (lon_mesh, lat_mesh), method='linear')
        vn_grid = griddata((lons, lats), vn, (lon_mesh, lat_mesh), method='linear')
        vu_grid = np.zeros_like(ve_grid)
        
        # write GeoTIFF
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=3,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='deflate',
        ) as dst:
            dst.write(ve_grid.astype(np.float32), 1)
            dst.write(vn_grid.astype(np.float32), 2)
            dst.write(vu_grid.astype(np.float32), 3)
            
            dst.set_band_description(1, 'east_velocity')
            dst.set_band_description(2, 'north_velocity')
            dst.set_band_description(3, 'up_velocity')
            
            dst.update_tags(1, units='mm/year')
            dst.update_tags(2, units='mm/year')
            dst.update_tags(3, units='mm/year')
            
            dst.update_tags(source='GEM_GSRM_v2.1')
        
        print(f"Created GSRM velocity grid: {output_path}")
        return Path(output_path)


# earthScope point data converter

class EarthScopeConverter(VelocityConverter):
    """
    Convert EarthScope/UNAVCO GNSS point velocities to gridded GeoTIFF.

    Handles the GAGE/PBO velocity file format (.vel files) which contain:
    - Station coordinates (XYZ and lat/lon)
    - Cartesian velocities (dX/dt, dY/dt, dZ/dt) in meters/year
    """

    @staticmethod
    def _xyz_to_enu(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                    lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Cartesian (ECEF) velocities to local East-North-Up.

        Args:
            vx, vy, vz: Cartesian velocity components (m/yr)
            lat, lon: Station positions (degrees)

        Returns:
            ve, vn, vu: East, North, Up velocity components (m/yr)
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # rotation matrix from ECEF to ENU
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # east component: -sin(lon)*vx + cos(lon)*vy
        ve = -sin_lon * vx + cos_lon * vy

        # north component: -sin(lat)*cos(lon)*vx - sin(lat)*sin(lon)*vy + cos(lat)*vz
        vn = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz

        # up component: cos(lat)*cos(lon)*vx + cos(lat)*sin(lon)*vy + sin(lat)*vz
        vu = cos_lat * cos_lon * vx + cos_lat * sin_lon * vy + sin_lat * vz

        return ve, vn, vu

    @staticmethod
    def _parse_vel_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse GAGE/PBO .vel file format.

        Returns:
            lons, lats, ve, vn, vu (velocities in mm/year)
        """
        lons = []
        lats = []
        vx_list = []
        vy_list = []
        vz_list = []

        with open(filepath, 'r') as f:
            in_data = False
            for line in f:
                line = line.strip()

                # skip empty lines and comments
                if not line or line.startswith('#') or line.startswith('*'):
                    # check for data section start
                    if 'Dot#' in line or 'dX/dt' in line:
                        in_data = True
                    continue

                # try to parse data line
                parts = line.split()
                if len(parts) < 15:
                    continue

                try:
                    # column indices (0-based):
                    # 0: Dot#, 1: Name, 2: Ref_epoch, 3: Ref_jday
                    # 4: Ref_X, 5: Ref_Y, 6: Ref_Z
                    # 7: Ref_Nlat, 8: Ref_Elong, 9: Ref_Up
                    # 10: dX/dt, 11: dY/dt, 12: dZ/dt
                    # 13: SXd, 14: SYd, 15: SZd

                    lat = float(parts[7])
                    lon = float(parts[8])
                    vx = float(parts[10])  # m/yr
                    vy = float(parts[11])  # m/yr
                    vz = float(parts[12])  # m/yr

                    # validate reasonable values
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        lons.append(lon)
                        lats.append(lat)
                        vx_list.append(vx)
                        vy_list.append(vy)
                        vz_list.append(vz)

                except (ValueError, IndexError):
                    continue

        if not lons:
            raise ValueError(f"No valid velocity data found in {filepath}")

        lons = np.array(lons)
        lats = np.array(lats)
        vx = np.array(vx_list)
        vy = np.array(vy_list)
        vz = np.array(vz_list)

        # convert XYZ to ENU
        ve, vn, vu = EarthScopeConverter._xyz_to_enu(vx, vy, vz, lats, lons)

        # convert from m/yr to mm/yr
        ve *= 1000
        vn *= 1000
        vu *= 1000

        return lons, lats, ve, vn, vu

    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        resolution: float = 0.1,
        method: str = 'linear',
        bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Path:
        """
        Grid EarthScope/GAGE point velocities to GeoTIFF.

        Args:
            input_path: ASCII velocity file (.vel format)
            output_path: Output GeoTIFF
            resolution: Grid spacing in degrees (default 0.1°)
            method: Interpolation method ('linear', 'cubic', 'nearest')
            bbox: Optional bounding box to limit output (min_lon, min_lat, max_lon, max_lat)
        """
        if griddata is None:
            raise ImportError("scipy required for gridding")
        if rasterio is None:
            raise ImportError("rasterio required")

        input_path = Path(input_path)

        # parse the velocity file
        print(f"Parsing velocity file: {input_path}")
        lons, lats, ve, vn, vu = EarthScopeConverter._parse_vel_file(input_path)

        print(f"Loaded {len(lons)} GNSS stations")
        print(f"  Longitude range: {lons.min():.2f} to {lons.max():.2f}")
        print(f"  Latitude range: {lats.min():.2f} to {lats.max():.2f}")
        print(f"  East velocity range: {ve.min():.2f} to {ve.max():.2f} mm/yr")
        print(f"  North velocity range: {vn.min():.2f} to {vn.max():.2f} mm/yr")

        # determine grid bounds
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
        else:
            lon_min, lon_max = lons.min(), lons.max()
            lat_min, lat_max = lats.min(), lats.max()

        # add small buffer
        buffer = resolution / 2
        lon_min -= buffer
        lon_max += buffer
        lat_min -= buffer
        lat_max += buffer

        # create regular grid
        nx = int((lon_max - lon_min) / resolution) + 1
        ny = int((lat_max - lat_min) / resolution) + 1

        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        print(f"Creating {nx}x{ny} grid ({resolution}° resolution)")

        # interpolate
        print(f"Interpolating using '{method}' method...")
        ve_grid = griddata((lons, lats), ve, (lon_mesh, lat_mesh), method=method)
        vn_grid = griddata((lons, lats), vn, (lon_mesh, lat_mesh), method=method)
        vu_grid = griddata((lons, lats), vu, (lon_mesh, lat_mesh), method=method)

        # fill NaN values at edges with nearest neighbor
        mask = np.isnan(ve_grid)
        if mask.any():
            print(f"Filling {mask.sum()} NaN values with nearest neighbor...")
            ve_nn = griddata((lons, lats), ve, (lon_mesh, lat_mesh), method='nearest')
            vn_nn = griddata((lons, lats), vn, (lon_mesh, lat_mesh), method='nearest')
            vu_nn = griddata((lons, lats), vu, (lon_mesh, lat_mesh), method='nearest')
            ve_grid[mask] = ve_nn[mask]
            vn_grid[mask] = vn_nn[mask]
            vu_grid[mask] = vu_nn[mask]

        # write GeoTIFF
        # note: rasterio expects data in (rows, cols) = (lat, lon) order
        # with lat decreasing from top to bottom
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)

        # flip latitude axis so north is up
        ve_grid = np.flipud(ve_grid)
        vn_grid = np.flipud(vn_grid)
        vu_grid = np.flipud(vu_grid)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=3,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='deflate',
        ) as dst:
            dst.write(ve_grid.astype(np.float32), 1)
            dst.write(vn_grid.astype(np.float32), 2)
            dst.write(vu_grid.astype(np.float32), 3)

            dst.set_band_description(1, 'east_velocity')
            dst.set_band_description(2, 'north_velocity')
            dst.set_band_description(3, 'up_velocity')

            dst.update_tags(1, units='mm/year')
            dst.update_tags(2, units='mm/year')
            dst.update_tags(3, units='mm/year')

            dst.update_tags(
                source='EarthScope_GAGE_GNSS',
                interpolation_method=method,
                n_stations=len(lons),
                reference_frame='NAM14',
            )

        print(f"Created gridded velocity model: {output_path}")
        return Path(output_path)


# converter registry and dispatcher

CONVERTERS = {
    'ucerf3': UCERF3Converter,
    'usgs_nshm': USGSNSHMConverter,
    'gsrm': GSRMConverter,
    'earthscope': EarthScopeConverter,
}


def convert_model(
    model_name: str,
    input_path: Path,
    output_path: Optional[Path] = None,
    **kwargs,
) -> Path:
    """
    Convert velocity model to PROJ-compatible GeoTIFF.
    
    Args:
        model_name: Model identifier (determines converter)
        input_path: Source file/directory
        output_path: Output GeoTIFF (auto-generated if None)
        **kwargs: Converter-specific options
        
    Returns:
        Path to created GeoTIFF
    """
    # determine converter
    converter_key = None
    for key in CONVERTERS:
        if key in model_name.lower():
            converter_key = key
            break
    
    if not converter_key:
        raise ValueError(
            f"No converter found for model '{model_name}'. "
            f"Available: {list(CONVERTERS.keys())}"
        )
    
    # auto-generate output path
    if output_path is None:
        output_path = input_path.parent / f"{model_name}_velocity.tif"
    
    # run converter
    converter = CONVERTERS[converter_key]
    return converter.convert(input_path, output_path, **kwargs)


# convenience exports

__all__ = [
    'VelocityConverter',
    'UCERF3Converter',
    'USGSNSHMConverter',
    'GSRMConverter',
    'EarthScopeConverter',
    'convert_model',
    'CONVERTERS',
]
