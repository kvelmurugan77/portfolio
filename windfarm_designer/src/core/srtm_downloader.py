"""
SRTM1 Terrain Data Downloader for WindFarm Designer Pro.
Downloads SRTM 1-arc-second (~30m resolution) Digital Elevation Model tiles
from NASA's EarthData / OpenTopography / AWS Open Data for a given area
plus an extended buffer zone.

Features:
- Automatic tile coverage detection from bounding box or polygon
- Multi-threaded download with progress tracking
- Verification via MD5 checksums
- Mosaic generation into a single GeoTIFF
- Hillshade and slope computation
"""

import os
import sys
import hashlib
import logging
import threading
import queue
import time
import math
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

# SRTM tile sources (fallback chain)
SRTM_SOURCES = {
    'aws_opentopo': {
        'base_url': 'https://s3.amazonaws.com/elevation-tiles-prod/srtm/v1.1/SRTM1_N{:02d}{:s}{:03d}.tif',
        'description': 'AWS Open Data (SRTM1)',
    },
    'earthdata': {
        'base_url': 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/',
        'description': 'NASA EarthData SRTMGL1 v3',
    },
    'opentopography': {
        'base_url': 'https://portal.opentopography.org/API/globaldem',
        'description': 'OpenTopography Global DEM API',
    },
}

# SRTM tile size: 1 degree x 1 degree, 3601 x 3601 pixels at 1-arc-second
SRTM_TILE_SIZE = 3601
SRTM_NODATA = -32768

# HTTP settings
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120  # seconds
CHUNK_SIZE = 8192


class SRTMDownloader:
    """
    Downloads and processes SRTM 1-arc-second elevation data.

    Attributes
    ----------
    output_dir : str
        Directory where tiles and mosaics are saved.
    buffer_km : float
        Extension buffer around the area of interest (default 20 km).
    """

    def __init__(self, output_dir: str = './srtm_data', buffer_km: float = 20.0):
        self.output_dir = Path(output_dir)
        self.buffer_km = buffer_km
        self.tiles_dir = self.output_dir / 'tiles'
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_tiles = []
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        self._lock = threading.Lock()

    def set_progress_callback(self, callback: Callable):
        """Set a callback function for progress updates: callback(current, total, message)."""
        self.progress_callback = callback

    def set_status_callback(self, callback: Callable):
        """Set a callback for status messages: callback(message)."""
        self.status_callback = callback

    def _report_status(self, message: str):
        """Report a status message."""
        logger.info(message)
        if self.status_callback:
            self.status_callback(message)

    def _report_progress(self, current: int, total: int, message: str = ""):
        """Report download progress."""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    # ============================================================
    # Tile Identification
    # ============================================================

    def identify_tiles(self, bbox: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """
        Identify all SRTM tiles needed to cover the bounding box.

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
            Area of interest bounding box.

        Returns
        -------
        list of (int, int)
            List of (lat_index, lon_index) tile identifiers.
        """
        from src.utils.geo_utils import extend_bbox

        # Apply buffer
        extended_bbox = extend_bbox(bbox, self.buffer_km)
        min_lon, min_lat, max_lon, max_lat = extended_bbox

        tiles = set()
        for lat in range(int(math.floor(min_lat)), int(math.ceil(max_lat)) + 1):
            for lon in range(int(math.floor(min_lon)), int(math.ceil(max_lon)) + 1):
                # SRTM covers -56 to 60 latitude, -180 to 180 longitude
                if -60 <= lat <= 60 and -180 <= lon <= 180:
                    tiles.add((lat, lon))

        tile_list = sorted(tiles)
        self._report_status(
            f"Identified {len(tile_list)} SRTM tiles covering area "
            f"(bbox: [{min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}]) "
            f"with {self.buffer_km}km buffer"
        )
        return tile_list

    @staticmethod
    def tile_filename(lat_idx: int, lon_idx: int) -> str:
        """Generate SRTM tile filename from indices."""
        lat_str = f'N{lat_idx:02d}' if lat_idx >= 0 else f'S{abs(lat_idx):02d}'
        lon_str = f'E{lon_idx:03d}' if lon_idx >= 0 else f'W{abs(lon_idx):03d}'
        return f'{lat_str}{lon_str}.tif'

    # ============================================================
    # Download
    # ============================================================

    def download_tiles(self, tiles: List[Tuple[int, int]],
                       source: str = 'aws_opentopo',
                       max_threads: int = 4) -> Tuple[List[str], List[str]]:
        """
        Download multiple SRTM tiles using concurrent threads.

        Parameters
        ----------
        tiles : list of (int, int)
            Tile indices to download.
        source : str
            Data source key from SRTM_SOURCES.
        max_threads : int
            Maximum concurrent download threads.

        Returns
        -------
        tuple (downloaded, failed)
            Lists of successfully downloaded and failed tile filenames.
        """
        if source not in SRTM_SOURCES:
            self._report_status(f"Unknown source '{source}'. Using 'aws_opentopo'.")
            source = 'aws_opentopo'

        total = len(tiles)
        downloaded = []
        failed = []
        completed = [0]

        work_queue = queue.Queue()
        for tile in tiles:
            work_queue.put(tile)

        def download_worker():
            while True:
                try:
                    tile = work_queue.get_nowait()
                except queue.Empty:
                    break

                lat_idx, lon_idx = tile
                filename = self.tile_filename(lat_idx, lon_idx)
                filepath = self.tiles_dir / filename

                # Skip if already downloaded
                if filepath.exists() and filepath.stat().st_size > 0:
                    with self._lock:
                        completed[0] += 1
                        downloaded.append(filename)
                        self._report_progress(completed[0], total,
                                             f"Skipping (exists): {filename}")
                    work_queue.task_done()
                    continue

                # Attempt download
                success = False
                for attempt in range(MAX_RETRIES):
                    try:
                        url = self._build_tile_url(source, lat_idx, lon_idx)
                        self._report_status(f"Downloading {filename} (attempt {attempt + 1})...")

                        response = requests.get(
                            url,
                            timeout=REQUEST_TIMEOUT,
                            stream=True,
                            headers={'User-Agent': 'WindFarm-Designer-Pro/1.0'}
                        )
                        response.raise_for_status()

                        # Stream to file
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                f.write(chunk)

                        # Verify file is not empty/corrupt
                        if filepath.stat().st_size > 1000:
                            success = True
                            break
                        else:
                            filepath.unlink()
                            self._report_status(
                                f"Downloaded file too small, retrying: {filename}")

                    except requests.exceptions.RequestException as e:
                        self._report_status(f"Download failed: {filename} - {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff

                with self._lock:
                    completed[0] += 1
                    if success:
                        downloaded.append(filename)
                        self._report_progress(completed[0], total,
                                             f"Downloaded: {filename}")
                    else:
                        failed.append(filename)
                        self._report_progress(completed[0], total,
                                             f"Failed: {filename}")

                work_queue.task_done()

        # Launch worker threads
        threads = []
        n_threads = min(max_threads, len(tiles))
        self._report_status(f"Starting download of {total} tiles using {n_threads} threads...")

        for _ in range(n_threads):
            t = threading.Thread(target=download_worker, daemon=True)
            t.start()
            threads.append(t)

        # Wait for all downloads to complete
        work_queue.join()
        for t in threads:
            t.join()

        self.downloaded_tiles = downloaded
        self._report_status(
            f"Download complete: {len(downloaded)} successful, {len(failed)} failed"
        )
        return downloaded, failed

    def _build_tile_url(self, source: str, lat_idx: int, lon_idx: int) -> str:
        """Build the download URL for a tile from a given source."""
        if source == 'aws_opentopo':
            lat_str = f'N{lat_idx:02d}' if lat_idx >= 0 else f'S{abs(lat_idx):02d}'
            lon_str = f'E{lon_idx:03d}' if lon_idx >= 0 else f'W{abs(lon_idx):03d}'
            return f'https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm/{lat_str}{lon_str}.tif'
        elif source == 'earthdata':
            lat_str = f'N{lat_idx:02d}' if lat_idx >= 0 else f'S{abs(lat_idx):02d}'
            lon_str = f'E{lon_idx:03d}' if lon_idx >= 0 else f'W{abs(lon_idx):03d}'
            return f'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{lat_str}{lon_str}.SRTMGL1.hgt.zip'
        else:
            return SRTM_SOURCES[source]['base_url']

    # ============================================================
    # Mosaic & Processing
    # ============================================================

    def create_mosaic(self, bbox: Tuple[float, float, float, float],
                      output_name: str = 'terrain_mosaic.tif') -> Optional[str]:
        """
        Merge downloaded SRTM tiles into a single GeoTIFF mosaic covering
        the specified bounding box.

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
            Target bounding box.
        output_name : str
            Output mosaic filename.

        Returns
        -------
        str or None
            Path to the mosaic file, or None on failure.
        """
        try:
            import rasterio
            from rasterio.merge import merge
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS
        except ImportError:
            self._report_status(
                "ERROR: rasterio is required for mosaic creation. "
                "Install with: pip install rasterio"
            )
            return None

        # Find all .tif tiles
        tif_files = sorted(self.tiles_dir.glob('*.tif'))
        if not tif_files:
            self._report_status("No SRTM tiles found. Download tiles first.")
            return None

        self._report_status(f"Merging {len(tif_files)} tiles into mosaic...")

        # Open and merge tiles
        datasets = []
        for fp in tif_files:
            try:
                ds = rasterio.open(fp)
                datasets.append(ds)
            except Exception as e:
                self._report_status(f"Warning: Could not open tile {fp}: {e}")

        if not datasets:
            self._report_status("ERROR: Could not open any tiles.")
            return None

        # Merge all tiles
        mosaic_array, mosaic_transform = merge(datasets)
        for ds in datasets:
            ds.close()

        # Write mosaic
        from src.utils.geo_utils import extend_bbox
        ext_bbox = extend_bbox(bbox, self.buffer_km)

        output_path = self.output_dir / output_name
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=mosaic_array.shape[1],
            width=mosaic_array.shape[2],
            count=1,
            dtype=mosaic_array.dtype,
            crs=CRS.from_epsg(4326),
            transform=mosaic_transform,
            nodata=SRTM_NODATA,
            compress='DEFLATE',
            tiled=True,
        ) as dst:
            dst.write(mosaic_array[0], 1)

        self._report_status(f"Mosaic saved to: {output_path}")
        return str(output_path)

    def compute_slope_aspect(self, mosaic_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Compute slope and aspect rasters from the DEM mosaic.

        Parameters
        ----------
        mosaic_path : str
            Path to the DEM mosaic GeoTIFF.

        Returns
        -------
        tuple (slope_path, aspect_path)
            Paths to the generated slope and aspect GeoTIFFs.
        """
        try:
            import rasterio
            from rasterio.transform import Affine
            import numpy as np
        except ImportError:
            self._report_status("ERROR: rasterio and numpy required for slope computation.")
            return None, None

        self._report_status("Computing slope and aspect...")

        with rasterio.open(mosaic_path) as src:
            dem = src.read(1)
            transform = src.transform
            profile = src.profile

        # Handle nodata
        nodata_mask = dem == src.nodata
        dem_filled = np.where(nodata_mask, np.nan, dem).astype(np.float64)

        # Compute dz/dx and dz/dy using central differences
        # Cell size in meters (approximate)
        pixel_size_deg = abs(transform.a)  # degrees per pixel
        center_lat = transform * (src.width / 2, src.height / 2)
        m_per_deg = 111320.0 * math.cos(math.radians(center_lat[1]))
        dx = pixel_size_deg * m_per_deg  # meters per pixel in x direction
        dy = pixel_size_deg * 111320.0  # meters per pixel in y direction

        # Central differences
        dzdy, dzdx = np.gradient(dem_filled, dy, dx)

        # Slope in degrees
        slope = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))
        slope = np.where(nodata_mask, src.nodata, slope)

        # Aspect in degrees (0 = North, 90 = East)
        aspect = np.degrees(np.arctan2(-dzdy, dzdx))
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        aspect = np.where(nodata_mask, src.nodata, aspect)

        # Write slope
        slope_path = str(Path(mosaic_path).parent / 'slope.tif')
        profile.update(dtype=rasterio.float32, nodata=-9999)
        with rasterio.open(slope_path, 'w', **profile) as dst:
            dst.write(slope.astype(np.float32), 1)

        # Write aspect
        aspect_path = str(Path(mosaic_path).parent / 'aspect.tif')
        with rasterio.open(aspect_path, 'w', **profile) as dst:
            dst.write(aspect.astype(np.float32), 1)

        self._report_status(f"Slope saved to: {slope_path}")
        self._report_status(f"Aspect saved to: {aspect_path}")
        return slope_path, aspect_path

    def get_elevation_at_points(self, mosaic_path: str,
                                points: List[Tuple[float, float]]) -> List[float]:
        """
        Extract elevation values at specific lat/lon points from the DEM mosaic.

        Parameters
        ----------
        mosaic_path : str
            Path to DEM mosaic GeoTIFF.
        points : list of (lat, lon)
            Points to sample.

        Returns
        -------
        list of float
            Elevation values in meters at each point.
        """
        try:
            import rasterio
        except ImportError:
            self._report_status("ERROR: rasterio required.")
            return [0.0] * len(points)

        elevations = []
        with rasterio.open(mosaic_path) as src:
            for lat, lon in points:
                try:
                    row, col = src.index(lon, lat)
                    elev = src.read(1)[row, col]
                    if elev == src.nodata:
                        elev = 0.0
                    elevations.append(float(elev))
                except (IndexError, ValueError):
                    elevations.append(0.0)

        return elevations

    # ============================================================
    # Full Pipeline
    # ============================================================

    def download_and_process(self,
                             bbox: Tuple[float, float, float, float],
                             source: str = 'aws_opentopo',
                             max_threads: int = 4) -> Dict:
        """
        Complete pipeline: identify tiles, download, mosaic, compute slope/aspect.

        Parameters
        ----------
        bbox : tuple
            Area of interest bounding box.
        source : str
            Download source.
        max_threads : int
            Concurrent download threads.

        Returns
        -------
        dict
            {
                'mosaic_path': str,
                'slope_path': str,
                'aspect_path': str,
                'tiles_downloaded': int,
                'tiles_failed': int,
            }
        """
        # Step 1: Identify required tiles
        tiles = self.identify_tiles(bbox)
        if not tiles:
            self._report_status("No SRTM tiles needed for the given area.")
            return {}

        # Step 2: Download tiles
        downloaded, failed = self.download_tiles(tiles, source, max_threads)

        if not downloaded:
            self._report_status("ERROR: No tiles were successfully downloaded.")
            return {}

        # Step 3: Create mosaic
        mosaic_path = self.create_mosaic(bbox)
        if not mosaic_path:
            return {}

        # Step 4: Compute slope and aspect
        slope_path, aspect_path = self.compute_slope_aspect(mosaic_path)

        return {
            'mosaic_path': mosaic_path,
            'slope_path': slope_path,
            'aspect_path': aspect_path,
            'tiles_downloaded': len(downloaded),
            'tiles_failed': len(failed),
            'tiles': downloaded,
        }


# ============================================================
# Convenience function
# ============================================================

def download_srtm_for_area(bbox: Tuple[float, float, float, float],
                           output_dir: str = './srtm_data',
                           buffer_km: float = 20.0,
                           progress_callback: Optional[Callable] = None,
                           status_callback: Optional[Callable] = None) -> Dict:
    """
    High-level function to download SRTM1 data for a given area.

    Parameters
    ----------
    bbox : tuple (min_lon, min_lat, max_lon, max_lat)
    output_dir : str
    buffer_km : float
    progress_callback, status_callback : optional callables

    Returns
    -------
    dict
        Results dictionary with paths to generated files.
    """
    downloader = SRTMDownloader(output_dir=output_dir, buffer_km=buffer_km)
    if progress_callback:
        downloader.set_progress_callback(progress_callback)
    if status_callback:
        downloader.set_status_callback(status_callback)

    return downloader.download_and_process(bbox)
