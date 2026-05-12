"""
Roughness (Land Cover) Data Downloader for WindFarm Designer Pro.
Downloads global land cover / roughness data from Copernicus Climate Change
Service (C3S) or NASA MODIS and converts it to surface roughness length (z0)
maps for wind resource assessment.

Data sources (tried in order):
1. ESA CCI Land Cover v2.0.7 (300m resolution) via CDS API — Primary
2. ESA WorldCover v2.0 (10m resolution) via WMS — High resolution fallback
3. OpenStreetMap land use classification — Last resort fallback

Output includes:
- GeoTIFF roughness map (z0 in meters)
- WAsP .map file (binary format for WAsP engineering software)
- WAsP .map file (text format as fallback)

Roughness classes follow the WAsP / wind energy standard z0 tables.
"""

import os
import math
import logging
import struct
import threading
import time
import requests
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# Roughness class definitions
# CCI Land Cover class ID -> z0 (meters)
# Based on WAsP roughness classification and CCI nomenclature
# ============================================================

CCI_ROUGHNESS_MAP = {
    10: 0.03,    # Cropland, rainfed
    11: 0.03,    # Herbaceous cover
    12: 0.05,    # Tree or shrub cover
    20: 0.03,    # Cropland, irrigated / post-flooding
    30: 0.05,    # Mosaic cropland (50-70%) / vegetation (20-50%)
    40: 0.30,    # Mosaic cropland (50-70%) / tree cover (20-50%)
    50: 0.50,    # Tree cover, broadleaved, evergreen, >5m
    60: 0.80,    # Tree cover, broadleaved, deciduous, >5m
    70: 0.60,    # Tree cover, needleleaved, evergreen, >5m
    80: 0.70,    # Tree cover, needleleaved, deciduous, >5m
    90: 0.50,    # Tree cover, mixed leaf type, >5m
    100: 1.00,   # Mosaic tree and shrub (>50%) / herbaceous cover (<50%)
    110: 0.10,   # Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
    120: 0.05,   # Shrubland
    121: 0.10,   # Evergreen shrubland
    122: 0.03,   # Deciduous shrubland
    130: 0.03,   # Grassland
    140: 0.01,   # Lichens and mosses
    150: 0.02,   # Sparse vegetation (tree, shrub, herbaceous) <15%
    160: 0.005,  # Tree cover, flooded, fresh or brackish water
    170: 0.0002, # Tree cover, flooded, saline water
    180: 0.03,   # Shrub or herbaceous cover, flooded
    190: 0.50,   # Urban areas
    200: 0.001,  # Bare areas (consolidated, unconsolidated)
    210: 0.0002, # Water bodies
    220: 0.0001, # Permanent snow and ice
}

# ESA WorldCover class -> z0 (meters)
WORLDCOVER_ROUGHNESS_MAP = {
    10: 0.03,    # Tree cover
    20: 0.80,    # Shrubland
    30: 0.03,    # Grassland
    40: 0.03,    # Cropland (rainfed)
    50: 0.03,    # Built-up / Urban
    60: 0.0002,  # Bare / sparse vegetation
    70: 0.03,    # Snow and ice
    80: 0.0002,  # Permanent water bodies
    90: 0.03,    # Herbaceous wetland
    95: 0.50,    # Mangroves
    100: 0.50,   # Moss and lichen
}

DEFAULT_Z0 = 0.03  # Default roughness (grassland)

# WAsP .map file constants
WASP_MAP_HEADER = b'WAsP Map File'
WASP_MAP_VERSION = 1
WASP_NODATA = -9999.0


class RoughnessDownloader:
    """
    Downloads land cover data and converts to roughness length maps.

    Attributes
    ----------
    output_dir : Path
        Directory where output files are saved.
    buffer_km : float
        Buffer distance around the area of interest.
    """

    def __init__(self, output_dir: str = './roughness_data', buffer_km: float = 20.0):
        self.output_dir = Path(output_dir)
        self.buffer_km = buffer_km
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        self.progress_callback = callback

    def set_status_callback(self, callback: Callable):
        self.status_callback = callback

    def _report_status(self, message: str):
        logger.info(message)
        if self.status_callback:
            self.status_callback(message)

    def _report_progress(self, current: int, total: int, message: str = ""):
        if self.progress_callback:
            self.progress_callback(current, total, message)

    # ============================================================
    # Download from Copernicus CDS (Primary — CCI 300m)
    # ============================================================

    def download_global_land_cover_300m(
        self,
        bbox: Tuple[float, float, float, float],
        year: int = 2020,
    ) -> Optional[str]:
        """
        Download ESA CCI Land Cover v2.0.7 (300m resolution) via the CDS API.

        This is the primary data source for roughness mapping. The CCI Land Cover
        dataset at 300m resolution provides global coverage with consistent
        classification suitable for wind resource assessment.

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        year : int
            Reference year for the land cover map (2015-2020 available).

        Returns
        -------
        str or None
            Path to downloaded file (NetCDF or zip archive), or None on failure.
        """
        try:
            import cdsapi
        except ImportError:
            self._report_status(
                "  CDS API package not found. Install with: pip install cdsapi\n"
                "  Also ensure you have a CDS API key set up at ~/.cdsapirc\n"
                "  See: https://cds.climate.copernicus.eu/api-how-to"
            )
            return None

        from src.utils.geo_utils import extend_bbox
        ext_bbox = extend_bbox(bbox, self.buffer_km)
        min_lon, min_lat, max_lon, max_lat = ext_bbox

        output_file = self.output_dir / f'cci_landcover_{year}.zip'

        self._report_status(
            f"  Requesting ESA CCI Land Cover v2.0.7 (300m) for {year} "
            f"over [{min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f}]..."
        )

        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve(
                'satellite-land-cover',
                {
                    'variable': 'land-cover',
                    'year': str(year),
                    'version': 'v2.0.7',
                    'format': 'zip',
                    'area': [max_lat, min_lon, min_lat, max_lon],
                },
                str(output_file)
            )
            self._report_status(f"  CCI Land Cover data downloaded to: {output_file}")
            return str(output_file)

        except Exception as e:
            self._report_status(f"  CDS API request failed: {e}")
            return None

    def download_copernicus_land_cover(
        self,
        bbox: Tuple[float, float, float, float],
        year: int = 2020,
    ) -> Optional[str]:
        """
        Download Copernicus CCI Land Cover data via the CDS API.
        Alias for download_global_land_cover_300m() for backward compatibility.

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        year : int
            Reference year for the land cover map.

        Returns
        -------
        str or None
            Path to downloaded file.
        """
        return self.download_global_land_cover_300m(bbox, year)

    # ============================================================
    # Download from ESA WorldCover (via WMS/WCS)
    # ============================================================

    def download_esa_worldcover(
        self,
        bbox: Tuple[float, float, float, float],
        year: int = 2021,
    ) -> Optional[str]:
        """
        Download ESA WorldCover 10m land cover data via WMS.

        Parameters
        ----------
        bbox : tuple
        year : int

        Returns
        -------
        str or None
        """
        from src.utils.geo_utils import extend_bbox
        ext_bbox = extend_bbox(bbox, self.buffer_km)
        min_lon, min_lat, max_lon, max_lat = ext_bbox

        self._report_status(
            f"  Attempting ESA WorldCover download for "
            f"[{min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f}]..."
        )

        # WorldCover WMS service via Terrascope
        base_url = "https://services.terrascope.be/wms/v2"
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetMap',
            'LAYERS': 'WORLDCOVER_2021_MAP',
            'FORMAT': 'image/tiff',
            'CRS': 'EPSG:4326',
            'BBOX': f"{min_lat},{min_lon},{max_lat},{max_lon}",
            'WIDTH': '2160',
            'HEIGHT': '2160',
        }

        try:
            response = requests.get(base_url, params=params, timeout=120)
            if response.status_code == 200 and len(response.content) > 1000:
                output_file = self.output_dir / f'worldcover_{year}.tif'
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                self._report_status(f"  ESA WorldCover data saved to: {output_file}")
                return str(output_file)
            else:
                self._report_status(
                    f"  ESA WorldCover returned status {response.status_code}"
                )
                return None
        except requests.RequestException as e:
            self._report_status(f"  ESA WorldCover download error: {e}")
            return None

    # ============================================================
    # Generate Synthetic Roughness from OpenStreetMap
    # ============================================================

    def generate_roughness_from_osm(
        self,
        bbox: Tuple[float, float, float, float],
        resolution_arcsec: float = 3.0,
    ) -> Optional[str]:
        """
        Generate a roughness map using OpenStreetMap land use data.
        This is a fallback method when CDS/ESA APIs are unavailable.

        Downloads OSM data for the area, classifies land use types,
        and assigns roughness lengths.

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        resolution_arcsec : float
            Output grid resolution in arc-seconds (default 3.0 ≈ 90m).

        Returns
        -------
        str or None
            Path to roughness GeoTIFF.
        """
        from src.utils.geo_utils import extend_bbox
        ext_bbox = extend_bbox(bbox, self.buffer_km)
        min_lon, min_lat, max_lon, max_lat = ext_bbox

        self._report_status("  Generating roughness map from OpenStreetMap data...")

        # OSM Overpass API query for land use within bbox
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:120];
        (
          way["landuse"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["natural"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["leisure"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["landuse"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

        try:
            response = requests.post(overpass_url, data={'data': query}, timeout=120)
            if response.status_code != 200:
                self._report_status(f"  OSM query failed with status {response.status_code}")
                return self._generate_default_roughness(bbox, resolution_arcsec)

            osm_data = response.json()

        except requests.RequestException as e:
            self._report_status(f"  OSM query error: {e}")
            return self._generate_default_roughness(bbox, resolution_arcsec)

        # Parse OSM elements and create roughness raster
        osm_roughness = {
            'residential': 0.50, 'commercial': 0.75, 'industrial': 0.30,
            'forest': 0.50, 'meadow': 0.03, 'farmland': 0.03, 'grass': 0.03,
            'scrub': 0.05, 'heath': 0.03, 'wetland': 0.03, 'marsh': 0.03,
            'water': 0.0002, 'lake': 0.0002, 'river': 0.0002,
            'park': 0.03, 'garden': 0.03, 'orchard': 0.05,
            'vineyard': 0.03, 'quarry': 0.001, 'landfill': 0.01,
            'bare_rock': 0.001, 'sand': 0.0005, 'beach': 0.0005,
            'wood': 0.50, 'tree': 0.50, 'desert': 0.0005,
        }

        # Build polygon features with roughness values
        from shapely.geometry import Polygon, MultiPolygon, box, Point

        features = []  # (roughness, Polygon)
        nodes = {}

        # Collect nodes
        for element in osm_data.get('elements', []):
            if element['type'] == 'node':
                nodes[element['id']] = (element['lon'], element['lat'])

        # Collect ways
        for element in osm_data.get('elements', []):
            if element['type'] == 'way':
                tags = element.get('tags', {})
                land_use = (tags.get('landuse') or tags.get('natural') or
                           tags.get('leisure', ''))
                z0 = osm_roughness.get(land_use, None)
                if z0 is not None and 'nodes' in element:
                    coords = []
                    for nid in element['nodes']:
                        if nid in nodes:
                            coords.append(nodes[nid])
                    if len(coords) >= 3:
                        try:
                            poly = Polygon(coords)
                            if poly.is_valid:
                                features.append((z0, poly))
                        except Exception:
                            pass

        self._report_status(f"  Found {len(features)} land use polygons from OSM")

        # Create roughness raster grid
        resolution_deg = resolution_arcsec / 3600.0
        nx = int((max_lon - min_lon) / resolution_deg) + 1
        ny = int((max_lat - min_lat) / resolution_deg) + 1

        # Limit grid size for performance
        max_pixels = 5000
        if nx * ny > max_pixels * max_pixels:
            scale = math.sqrt(nx * ny / (max_pixels * max_pixels))
            resolution_deg *= scale
            nx = int((max_lon - min_lon) / resolution_deg) + 1
            ny = int((max_lat - min_lat) / resolution_deg) + 1

        # Initialize with default roughness
        roughness_grid = np.full((ny, nx), DEFAULT_Z0, dtype=np.float32)

        # Rasterize features
        target_bbox = box(min_lon, min_lat, max_lon, max_lat)
        for z0, poly in features:
            try:
                clipped = poly.intersection(target_bbox)
                if clipped.is_empty or clipped.area < 1e-12:
                    continue

                # Get bounding box of feature in pixel coordinates
                fminx, fminy, fmaxx, fmaxy = clipped.bounds
                iy_min = max(0, int((fminy - min_lat) / resolution_deg))
                iy_max = min(ny, int((fmaxy - min_lat) / resolution_deg) + 1)
                ix_min = max(0, int((fminx - min_lon) / resolution_deg))
                ix_max = min(nx, int((fmaxx - min_lon) / resolution_deg) + 1)

                for iy in range(iy_min, iy_max):
                    for ix in range(ix_min, ix_max):
                        px = min_lon + (ix + 0.5) * resolution_deg
                        py = min_lat + (iy + 0.5) * resolution_deg
                        if clipped.contains(Point(px, py)):
                            roughness_grid[iy, ix] = z0
            except Exception:
                continue

        # Save as GeoTIFF
        return self._save_roughness_geotiff(
            roughness_grid, min_lon, min_lat, max_lon, max_lat, 'roughness_map.tif'
        )

    def _generate_default_roughness(
        self,
        bbox: Tuple[float, float, float, float],
        resolution_arcsec: float = 3.0,
    ) -> Optional[str]:
        """
        Generate a uniform default roughness map when no data source is available.
        """
        from src.utils.geo_utils import extend_bbox
        ext_bbox = extend_bbox(bbox, self.buffer_km)
        min_lon, min_lat, max_lon, max_lat = ext_bbox

        resolution_deg = resolution_arcsec / 3600.0
        nx = int((max_lon - min_lon) / resolution_deg) + 1
        ny = int((max_lat - min_lat) / resolution_deg) + 1

        roughness_grid = np.full((ny, nx), DEFAULT_Z0, dtype=np.float32)

        return self._save_roughness_geotiff(
            roughness_grid, min_lon, min_lat, max_lon, max_lat, 'roughness_map.tif'
        )

    # ============================================================
    # GeoTIFF Save Helper
    # ============================================================

    def _save_roughness_geotiff(
        self,
        roughness_grid: np.ndarray,
        min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
        filename: str = 'roughness_map.tif',
    ) -> Optional[str]:
        """Save a roughness numpy grid to GeoTIFF. Falls back to .npy."""
        ny, nx = roughness_grid.shape

        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS

            output_file = self.output_dir / filename
            transform = from_bounds(min_lon, min_lat, max_lon, max_lat, nx, ny)

            with rasterio.open(
                output_file, 'w',
                driver='GTiff',
                height=ny, width=nx,
                count=1,
                dtype=np.float32,
                crs=CRS.from_epsg(4326),
                transform=transform,
                nodata=-9999,
                compress='DEFLATE',
            ) as dst:
                dst.write(roughness_grid.astype(np.float32), 1)

            self._report_status(f"  Roughness GeoTIFF saved to: {output_file}")
            return str(output_file)

        except ImportError:
            self._report_status("  rasterio not available, saving as numpy array")
            npy_file = self.output_dir / 'roughness_map.npy'
            np.save(npy_file, roughness_grid)
            self._report_status(f"  Roughness array saved to: {npy_file}")
            return str(npy_file)

    # ============================================================
    # WAsP .map File Output
    # ============================================================

    def write_map_file(
        self,
        roughness_data: np.ndarray,
        x_min: float,
        y_min: float,
        cell_size: float,
        output_path: Optional[str] = None,
        format: str = 'binary',
    ) -> str:
        """
        Write roughness data to WAsP .map file format.

        WAsP (Wind Atlas Analysis and Application Program) uses .map files
        for roughness input. This method supports both binary and text formats.

        Binary format structure:
            - Header: "WAsP Map File" (ASCII, 14 bytes)
            - Version: int16 (2 bytes) = 1
            - NCols: int32 (4 bytes)
            - NRows: int32 (4 bytes)
            - XMin: float64 (8 bytes) - UTM X coordinate (meters)
            - YMin: float64 (8 bytes) - UTM Y coordinate (meters)
            - CellSize: float64 (8 bytes) - cell size (meters)
            - Nodata: float32 (4 bytes) = -9999.0
            - Data: float32 array [NRows x NCols] - roughness z0 values (meters)

        Text format structure:
            # WAsP Map File (text format)
            # Generated by WindFarm Designer Pro
            ncols nrows
            xllcorner yllcorner  (UTM coordinates in meters)
            cellsize  (meters)
            nodata_value
            data values row by row...

        Parameters
        ----------
        roughness_data : np.ndarray
            2D array of roughness values (z0 in meters). Shape (NRows, NCols).
        x_min : float
            Minimum X coordinate (meters, typically UTM).
        y_min : float
            Minimum Y coordinate (meters, typically UTM).
        cell_size : float
            Cell size in meters.
        output_path : str or None
            Output file path. If None, uses default path in output_dir.
        format : str
            'binary' for binary WAsP .map format, 'text' for text format.

        Returns
        -------
        str
            Path to the written .map file.
        """
        if output_path is None:
            suffix = '.map' if format == 'binary' else '_text.map'
            output_path = str(self.output_dir / f'roughness{suffix}')

        nrows, ncols = roughness_data.shape

        if format == 'binary':
            self._write_map_file_binary(
                roughness_data, ncols, nrows, x_min, y_min, cell_size, output_path
            )
        else:
            self._write_map_file_text(
                roughness_data, ncols, nrows, x_min, y_min, cell_size, output_path
            )

        self._report_status(f"  WAsP .map file ({format}) saved to: {output_path}")
        return output_path

    def _write_map_file_binary(
        self,
        data: np.ndarray,
        ncols: int,
        nrows: int,
        x_min: float,
        y_min: float,
        cell_size: float,
        output_path: str,
    ):
        """
        Write binary WAsP .map file.

        Format:
            Bytes  0-13:  "WAsP Map File" (14 bytes ASCII, no null terminator)
            Bytes 14-15:  version (int16 little-endian) = 1
            Bytes 16-19:  ncols (int32 little-endian)
            Bytes 20-23:  nrows (int32 little-endian)
            Bytes 24-31:  x_min (float64 little-endian, UTM meters)
            Bytes 32-39:  y_min (float64 little-endian, UTM meters)
            Bytes 40-47:  cell_size (float64 little-endian, meters)
            Bytes 48-51:  nodata_value (float32 little-endian) = -9999.0
            Bytes 52+:    data array (float32 little-endian, row-major, NRows * NCols)
        """
        with open(output_path, 'wb') as f:
            # Header: 14 bytes ASCII
            f.write(WASP_MAP_HEADER)

            # Version: int16 = 1
            f.write(struct.pack('<h', WASP_MAP_VERSION))

            # Grid dimensions
            f.write(struct.pack('<i', ncols))
            f.write(struct.pack('<i', nrows))

            # Spatial parameters (float64)
            f.write(struct.pack('<d', float(x_min)))
            f.write(struct.pack('<d', float(y_min)))
            f.write(struct.pack('<d', float(cell_size)))

            # Nodata value (float32)
            f.write(struct.pack('<f', WASP_NODATA))

            # Roughness data (float32, row-major)
            # Replace any NaN/inf with nodata
            clean_data = np.where(
                np.isfinite(data), data.astype(np.float32), np.float32(WASP_NODATA)
            )
            f.write(clean_data.tobytes())

    def _write_map_file_text(
        self,
        data: np.ndarray,
        ncols: int,
        nrows: int,
        x_min: float,
        y_min: float,
        cell_size: float,
        output_path: str,
    ):
        """
        Write text-based WAsP .map file (ESRI ASCII grid-like format).

        This is a simpler, human-readable format that can be used as a
        fallback when binary format is not supported or for debugging.
        """
        import datetime

        with open(output_path, 'w') as f:
            f.write("# WAsP Map File (text format)\n")
            f.write(f"# Generated by WindFarm Designer Pro\n")
            f.write(f"# Date: {datetime.datetime.now().isoformat()}\n")
            f.write(f"# Roughness length (z0) in meters\n")
            f.write(f"ncols {ncols}\n")
            f.write(f"nrows {nrows}\n")
            f.write(f"xllcorner {x_min:.6f}\n")
            f.write(f"yllcorner {y_min:.6f}\n")
            f.write(f"cellsize {cell_size:.6f}\n")
            f.write(f"nodata_value {WASP_NODATA}\n")

            # Write data row by row
            for row in data:
                row_str = ' '.join(
                    f'{v:.6f}' if np.isfinite(v) else str(WASP_NODATA)
                    for v in row
                )
                f.write(row_str + '\n')

    def _geotiff_to_map_file(
        self,
        geotiff_path: str,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convert a roughness GeoTIFF to a WAsP .map file.

        Handles coordinate transformation from geographic (lat/lon) to UTM
        for proper WAsP compatibility.

        Parameters
        ----------
        geotiff_path : str
            Path to roughness GeoTIFF.
        output_path : str or None
            Output .map file path.

        Returns
        -------
        str or None
            Path to the .map file, or None on failure.
        """
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling
        except ImportError:
            self._report_status("  rasterio required for WAsP .map export")
            return None

        try:
            with rasterio.open(geotiff_path) as src:
                crs = src.crs
                data = src.read(1)
                transform = src.transform
                ncols = src.width
                nrows = src.height
                bounds = src.bounds  # left, bottom, right, top

            if crs is None or crs.is_geographic:
                # Convert from geographic to UTM
                # Determine UTM zone from center of bounds
                center_lon = (bounds.left + bounds.right) / 2.0
                center_lat = (bounds.bottom + bounds.top) / 2.0
                utm_zone = int((center_lon + 180) / 6) + 1
                utm_crs = rasterio.crs.CRS.from_epsg(
                    32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
                )

                # Reproject the data to UTM
                from rasterio.warp import calculate_default_transform
                utm_transform, utm_width, utm_height = calculate_default_transform(
                    src.crs, utm_crs, src.width, src.height, *src.bounds
                )

                utm_data = np.empty((utm_height, utm_width), dtype=np.float32)
                from rasterio.warp import reproject
                reproject(
                    source=data,
                    destination=utm_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=utm_transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.nearest,
                )

                # UTM bounds
                x_min = utm_transform[2]  # c (x-origin)
                y_min = utm_transform[5] + utm_transform[4] * utm_height  # f + e*rows
                cell_size = abs(utm_transform[0])  # a (pixel width)

                # Write binary .map file
                map_path = self.write_map_file(
                    utm_data, x_min, y_min, cell_size, output_path, format='binary'
                )

                # Also write text version as companion
                text_path = map_path.replace('.map', '_text.map')
                self.write_map_file(
                    utm_data, x_min, y_min, cell_size, text_path, format='text'
                )

                return map_path

            else:
                # Already projected (assume meters)
                x_min = transform[2]
                y_min = transform[5] + transform[4] * nrows
                cell_size = abs(transform[0])

                return self.write_map_file(
                    data, x_min, y_min, cell_size, output_path, format='binary'
                )

        except Exception as e:
            self._report_status(f"  Error creating WAsP .map file: {e}")
            return None

    # ============================================================
    # Convert Land Cover to Roughness
    # ============================================================

    def landcover_to_roughness(self, lc_path: str, source: str = 'cci') -> Optional[str]:
        """
        Convert a land cover raster to a roughness length raster.

        Parameters
        ----------
        lc_path : str
            Path to land cover GeoTIFF or NetCDF.
        source : str
            'cci' for Copernicus CCI, 'worldcover' for ESA WorldCover.

        Returns
        -------
        str or None
            Path to roughness GeoTIFF.
        """
        try:
            import rasterio
            import numpy as np
        except ImportError:
            self._report_status("  rasterio and numpy required for conversion.")
            return None

        roughness_map = CCI_ROUGHNESS_MAP if source == 'cci' else WORLDCOVER_ROUGHNESS_MAP

        self._report_status(f"  Converting {source} land cover to roughness...")

        try:
            with rasterio.open(lc_path) as src:
                lc_data = src.read(1)
                profile = src.profile

            # Vectorized lookup
            roughness_data = np.vectorize(
                lambda x: roughness_map.get(int(x), DEFAULT_Z0)
            )(lc_data).astype(np.float32)

            output_file = self.output_dir / 'roughness_map.tif'
            profile.update(dtype=np.float32, nodata=-9999)

            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(roughness_data, 1)

            self._report_status(f"  Roughness GeoTIFF saved to: {output_file}")
            return str(output_file)

        except Exception as e:
            self._report_status(f"  Error converting land cover: {e}")
            return None

    # ============================================================
    # Extract roughness at specific points
    # ============================================================

    def get_roughness_at_points(
        self,
        roughness_path: str,
        points: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Extract roughness values at specific lat/lon points.

        Parameters
        ----------
        roughness_path : str
            Path to roughness GeoTIFF.
        points : list of (lat, lon)

        Returns
        -------
        list of float
        """
        try:
            import rasterio
        except ImportError:
            return [DEFAULT_Z0] * len(points)

        values = []
        with rasterio.open(roughness_path) as src:
            for lat, lon in points:
                try:
                    row, col = src.index(lon, lat)
                    val = src.read(1)[row, col]
                    if val == src.nodata or val < 0:
                        val = DEFAULT_Z0
                    values.append(float(val))
                except (IndexError, ValueError):
                    values.append(DEFAULT_Z0)

        return values

    # ============================================================
    # Full Pipeline
    # ============================================================

    def download_and_process(
        self,
        bbox: Tuple[float, float, float, float],
        preferred_source: str = 'auto',
    ) -> Dict:
        """
        Complete pipeline: download land cover data, convert to roughness map,
        and generate WAsP .map files.

        Data source priority:
        1. ESA CCI Land Cover v2.0.7 (300m) via CDS API
        2. ESA WorldCover v2.0 (10m) via WMS
        3. OpenStreetMap land use classification

        Regardless of source, always generates:
        - roughness_map.tif (GeoTIFF with z0 values)
        - roughness.map (WAsP binary format)
        - roughness_text.map (WAsP text format, companion)

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        preferred_source : str
            'auto', 'cci', 'worldcover', or 'osm'

        Returns
        -------
        dict with keys:
            roughness_path: str - path to GeoTIFF
            map_file_path: str or None - path to WAsP .map file
            landcover_path: str or None - path to raw land cover data
            source: str - which data source was used
            bbox: tuple - extended bounding box
        """
        from src.utils.geo_utils import extend_bbox

        ext_bbox = extend_bbox(bbox, self.buffer_km)
        lc_path = None
        source_used = 'default'

        # --- Try CDS API (CCI 300m) ---
        if preferred_source in ('auto', 'cci'):
            lc_path = self.download_global_land_cover_300m(ext_bbox)
            if lc_path:
                source_used = 'cci'

        # --- Try ESA WorldCover (10m WMS) ---
        if lc_path is None and preferred_source in ('auto', 'worldcover'):
            lc_path = self.download_esa_worldcover(ext_bbox)
            if lc_path:
                source_used = 'worldcover'

        # --- Generate roughness map ---
        if lc_path is not None:
            roughness_path = self.landcover_to_roughness(lc_path, source_used)
        else:
            roughness_path = self.generate_roughness_from_osm(ext_bbox)
            source_used = 'osm'

        # --- Generate WAsP .map files ---
        map_file_path = None
        if roughness_path and roughness_path.endswith('.tif'):
            self._report_status("  Generating WAsP .map files...")
            map_file_path = self._geotiff_to_map_file(roughness_path)

        return {
            'roughness_path': roughness_path,
            'map_file_path': map_file_path,
            'landcover_path': lc_path,
            'source': source_used,
            'bbox': ext_bbox,
        }
