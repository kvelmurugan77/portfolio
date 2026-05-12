"""
Geographic utility functions for WindFarm Designer Pro.
Provides coordinate transformations, distance calculations, bounding box operations,
and geospatial helper methods used throughout the application.
"""

import math
import numpy as np
from typing import Tuple, List, Optional
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.ops import unary_union


# ============================================================
# Constants
# ============================================================
EARTH_RADIUS_M = 6371000.0          # Mean Earth radius in meters
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
METERS_PER_DEG_LAT = 111320.0       # Approximate meters per degree latitude
# Roughness length lookup table (meters) - CCI Land Cover classes
# Class IDs based on CCI Land Cover v2.0.7 nomenclature
ROUGHNESS_TABLE = {
    10: 0.0002,   # Cropland, rainfed
    11: 0.0005,   # Cropland, rainfed (herbaceous cover)
    12: 0.0010,   # Cropland, rainfed (tree or shrub cover)
    20: 0.0003,   # Cropland, irrigated
    30: 0.0010,   # Mosaic cropland / vegetation
    40: 0.5000,   # Mosaic cropland / tree cover
    50: 0.5000,   # Tree cover, broadleaved, evergreen
    60: 0.8000,   # Tree cover, broadleaved, deciduous
    70: 0.6000,   # Tree cover, needleleaved, evergreen
    80: 0.7000,   # Tree cover, needleleaved, deciduous
    90: 0.5000,   # Tree cover, mixed leaf type
    100: 1.0000,  # Mosaic tree and shrub cover
    110: 0.5000,  # Mosaic grassland / tree or shrub cover
    120: 0.0300,  # Shrubland
    121: 0.1000,  # Shrubland, evergreen
    122: 0.0500,  # Shrubland, deciduous
    130: 0.0300,  # Grassland
    140: 0.0100,  # Lichens and mosses
    150: 0.0005,  # Sparse vegetation (tree, shrub, herbaceous cover)
    160: 0.0002,  # Tree cover, flooded, fresh/brackish water
    170: 0.0002,  # Tree cover, flooded, saline water
    180: 0.0002,  # Shrub or herb cover, flooded
    190: 0.0001,  # Urban areas
    200: 0.0002,  # Bare areas
    201: 0.0003,  # Bare areas, consolidated
    202: 0.0001,  # Bare areas, unconsolidated
    210: 0.0000,  # Water bodies
    220: 0.0002,  # Permanent snow and ice
    230: 0.0000,  # Open sea
}

# Default roughness length if class not found
DEFAULT_ROUGHNESS = 0.0300  # Grassland default

# Turbine power curve data format constants
WIND_SPEED_COL = 0
POWER_COL = 1
CT_COL = 2

# Standard air density at sea level (kg/m^3)
STANDARD_AIR_DENSITY = 1.225


# ============================================================
# Coordinate & Distance Functions
# ============================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth
    using the Haversine formula.

    Parameters
    ----------
    lat1, lon1 : float
        Coordinates of the first point in decimal degrees.
    lat2, lon2 : float
        Coordinates of the second point in decimal degrees.

    Returns
    -------
    float
        Distance in meters between the two points.
    """
    lat1_r, lat2_r = lat1 * DEG_TO_RAD, lat2 * DEG_TO_RAD
    dlat = (lat2 - lat1) * DEG_TO_RAD
    dlon = (lon2 - lon1) * DEG_TO_RAD

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing (forward azimuth) from point 1 to point 2.

    Parameters
    ----------
    lat1, lon1 : float
        Origin coordinates in decimal degrees.
    lat2, lon2 : float
        Destination coordinates in decimal degrees.

    Returns
    -------
    float
        Bearing in degrees (0 = North, 90 = East).
    """
    lat1_r = lat1 * DEG_TO_RAD
    lat2_r = lat2 * DEG_TO_RAD
    dlon = (lon2 - lon1) * DEG_TO_RAD

    x = math.sin(dlon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))

    bearing_rad = math.atan2(x, y)
    return (bearing_rad * RAD_TO_DEG) % 360.0


def destination_point(lat: float, lon: float, bearing_deg: float,
                      distance_m: float) -> Tuple[float, float]:
    """
    Calculate the destination point given a starting point, bearing, and distance.

    Parameters
    ----------
    lat, lon : float
        Starting point coordinates in decimal degrees.
    bearing_deg : float
        Bearing in degrees.
    distance_m : float
        Distance in meters.

    Returns
    -------
    tuple (float, float)
        Destination (latitude, longitude) in decimal degrees.
    """
    lat_r = lat * DEG_TO_RAD
    bearing_r = bearing_deg * DEG_TO_RAD
    angular_dist = distance_m / EARTH_RADIUS_M

    lat2_r = math.asin(
        math.sin(lat_r) * math.cos(angular_dist) +
        math.cos(lat_r) * math.sin(angular_dist) * math.cos(bearing_r)
    )
    lon2_r = lon_r + math.atan2(
        math.sin(bearing_r) * math.sin(angular_dist) * math.cos(lat_r),
        math.cos(angular_dist) - math.sin(lat_r) * math.sin(lat2_r)
    )

    return lat2_r * RAD_TO_DEG, lon2_r * RAD_TO_DEG


# ============================================================
# Bounding Box & Buffer Operations
# ============================================================

def extend_bbox(bbox: Tuple[float, float, float, float],
                buffer_km: float = 20.0) -> Tuple[float, float, float, float]:
    """
    Extend a bounding box by a buffer distance in kilometers.
    Accounts for the convergence of meridians at higher latitudes.

    Parameters
    ----------
    bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        Input bounding box.
    buffer_km : float
        Buffer distance in kilometers (default: 20 km).

    Returns
    -------
    tuple (min_lon, min_lat, max_lon, max_lat)
        Extended bounding box.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    buffer_m = buffer_km * 1000.0

    # Latitude buffer (approximately constant per degree)
    lat_buffer = buffer_m / METERS_PER_DEG_LAT

    # Longitude buffer varies with latitude (shrinks toward poles)
    mid_lat = (min_lat + max_lat) / 2.0
    lon_buffer = buffer_m / (METERS_PER_DEG_LAT * math.cos(mid_lat * DEG_TO_RAD))

    return (
        min_lon - lon_buffer,
        min_lat - lat_buffer,
        max_lon + lon_buffer,
        max_lat + lat_buffer
    )


def bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> Polygon:
    """
    Convert a bounding box to a Shapely Polygon.

    Parameters
    ----------
    bbox : tuple (min_lon, min_lat, max_lon, max_lat)

    Returns
    -------
    shapely.geometry.Polygon
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return box(min_lon, min_lat, max_lon, max_lat)


def polygon_to_bbox(polygon: Polygon) -> Tuple[float, float, float, float]:
    """
    Extract bounding box from a Shapely Polygon.

    Returns
    -------
    tuple (min_lon, min_lat, max_lon, max_lat)
    """
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    return bounds  # (min_lon, min_lat, max_lon, max_lat)


def buffer_polygon(polygon: Polygon, buffer_km: float) -> Polygon:
    """
    Buffer a Shapely Polygon by a distance in kilometers.
    Uses a local CRS approximation for reasonable accuracy.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
    buffer_km : float
        Buffer distance in kilometers.

    Returns
    -------
    shapely.geometry.Polygon
    """
    # Approximate degrees per km at the centroid latitude
    centroid = polygon.centroid
    lat = centroid.y
    deg_per_km_lon = 1.0 / (METERS_PER_DEG_LAT * math.cos(lat * DEG_TO_RAD) / 1000.0)
    deg_per_km_lat = 1.0 / (METERS_PER_DEG_LAT / 1000.0)

    return polygon.buffer(buffer_km * deg_per_km_lon)

    # Note: For production, use pyproj for accurate geodetic buffering


# ============================================================
# SRTM Tile Helpers
# ============================================================

def get_srtm_tile_indices(lat: float, lon: float) -> Tuple[int, int]:
    """
    Get the SRTM 1-arc-second tile index for a given coordinate.
    SRTM tiles are 1-degree x 1-degree.

    Parameters
    ----------
    lat, lon : float
        Coordinates in decimal degrees.

    Returns
    -------
    tuple (int, int)
        (row_index, col_index) for the SRTM tile.
    """
    row = int(math.floor(lat))
    col = int(math.floor(lon))
    return row, col


def get_srtm_tiles_for_bbox(bbox: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
    """
    Get all SRTM tile indices that intersect with a bounding box.

    Parameters
    ----------
    bbox : tuple (min_lon, min_lat, max_lon, max_lat)

    Returns
    -------
    list of (int, int)
        List of (lat_index, lon_index) for each required tile.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = set()

    for lat in range(int(math.floor(min_lat)), int(math.ceil(max_lat)) + 1):
        for lon in range(int(math.floor(min_lon)), int(math.ceil(max_lon)) + 1):
            tiles.add((lat, lon))

    return sorted(tiles)


def srtm_tile_name(lat_idx: int, lon_idx: int) -> str:
    """
    Generate the SRTM tile filename from indices.
    Example: (35, -100) -> 'N35W100'

    Parameters
    ----------
    lat_idx : int
        Latitude index (can be positive or negative).
    lon_idx : int
        Longitude index (can be positive or negative).

    Returns
    -------
    str
        SRTM tile filename without extension.
    """
    lat_str = f'{"N" if lat_idx >= 0 else "S"}{abs(lat_idx):02d}'
    lon_str = f'{"E" if lon_idx >= 0 else "W"}{abs(lon_idx):03d}'
    return f'{lat_str}{lon_str}'


# ============================================================
# Wind & Roughness Helpers
# ============================================================

def wind_speed_at_height(v_ref: float, z_ref: float, z: float,
                         roughness: float) -> float:
    """
    Extrapolate wind speed from reference height to target height
    using the power law profile.

        v(z) = v(z_ref) * (z / z_ref) ^ alpha

    where alpha = 1 / ln(z_ref / z0)

    Parameters
    ----------
    v_ref : float
        Wind speed at reference height (m/s).
    z_ref : float
        Reference height (m).
    z : float
        Target height (m).
    roughness : float
        Surface roughness length z0 (m).

    Returns
    -------
    float
        Wind speed at height z (m/s).
    """
    if z <= 0 or z_ref <= 0 or roughness <= 0:
        return v_ref
    alpha = 1.0 / math.log(z_ref / roughness)
    return v_ref * (z / z_ref) ** alpha


def weibull_params_from_mean_std(mean_speed: float, std_speed: float) -> Tuple[float, float]:
    """
    Estimate Weibull distribution parameters (A, k) from
    the mean and standard deviation of wind speed.

    Uses the method of moments:
        k = (std / mean)^(-1.086)
        A = mean / Gamma(1 + 1/k)

    Parameters
    ----------
    mean_speed : float
        Mean wind speed (m/s).
    std_speed : float
        Standard deviation of wind speed (m/s).

    Returns
    -------
    tuple (float, float)
        (scale parameter A, shape parameter k)
    """
    if mean_speed <= 0 or std_speed <= 0:
        return mean_speed, 2.0  # Fallback

    k = (std_speed / mean_speed) ** (-1.086)
    k = max(k, 1.0)  # Physical lower bound

    from scipy.special import gamma
    A = mean_speed / gamma(1.0 + 1.0 / k)
    return A, k


def weibull_pdf(v: float, A: float, k: float) -> float:
    """
    Weibull probability density function.

    Parameters
    ----------
    v : float or np.ndarray
        Wind speed (m/s).
    A : float
        Scale parameter (m/s).
    k : float
        Shape parameter (dimensionless).

    Returns
    -------
    float or np.ndarray
        Probability density.
    """
    if A <= 0 or k <= 0 or v < 0:
        return 0.0
    return (k / A) * (v / A) ** (k - 1) * math.exp(-(v / A) ** k)


def get_roughness_from_class(land_class_id: int) -> float:
    """
    Look up surface roughness length from a CCI Land Cover class ID.

    Parameters
    ----------
    land_class_id : int
        CCI Land Cover class identifier.

    Returns
    -------
    float
        Roughness length z0 in meters.
    """
    return ROUGHNESS_TABLE.get(land_class_id, DEFAULT_ROUGHNESS)


def air_density_adjustment(rho: float = 1.225) -> float:
    """
    Compute the power adjustment factor for non-standard air density.
    P_adj = rho / rho_0

    Parameters
    ----------
    rho : float
        Actual air density (kg/m^3). Default is standard sea-level.

    Returns
    -------
    float
        Multiplicative adjustment factor.
    """
    return rho / STANDARD_AIR_DENSITY


# ============================================================
# Unit Conversions
# ============================================================

def mps_to_kph(speed_mps: float) -> float:
    """Convert m/s to km/h."""
    return speed_mps * 3.6


def kph_to_mps(speed_kph: float) -> float:
    """Convert km/h to m/s."""
    return speed_kph / 3.6


def mps_to_mph(speed_mps: float) -> float:
    """Convert m/s to mph."""
    return speed_mps * 2.23694


def meters_to_feet(m: float) -> float:
    """Convert meters to feet."""
    return m * 3.28084


def feet_to_meters(ft: float) -> float:
    """Convert feet to meters."""
    return ft / 3.28084


def watts_to_mw(w: float) -> float:
    """Convert watts to megawatts."""
    return w / 1e6


def mw_to_watts(mw: float) -> float:
    """Convert megawatts to watts."""
    return mw * 1e6


def degrees_to_compass(deg: float) -> str:
    """
    Convert wind direction in degrees to compass direction string.

    Parameters
    ----------
    deg : float
        Direction in degrees (meteorological convention: 0=N, 90=E).

    Returns
    -------
    str
        Compass direction (e.g., 'N', 'NNE', 'NE', ...).
    """
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = round(deg / 22.5) % 16
    return dirs[idx]


# ============================================================
# Geometry Helpers
# ============================================================

def point_in_polygon(lat: float, lon: float, polygon: Polygon) -> bool:
    """
    Check if a point is inside a polygon.

    Parameters
    ----------
    lat, lon : float
        Point coordinates.
    polygon : shapely.geometry.Polygon

    Returns
    -------
    bool
    """
    return polygon.contains(Point(lon, lat))


def generate_grid_points(bbox: Tuple[float, float, float, float],
                         spacing_km: float) -> List[Tuple[float, float]]:
    """
    Generate a regular grid of points within a bounding box.

    Parameters
    ----------
    bbox : tuple (min_lon, min_lat, max_lon, max_lat)
    spacing_km : float
        Grid spacing in kilometers.

    Returns
    -------
    list of (float, float)
        List of (latitude, longitude) tuples.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    points = []

    mid_lat = (min_lat + max_lat) / 2.0
    dlat = spacing_km * 1000.0 / METERS_PER_DEG_LAT
    dlon = spacing_km * 1000.0 / (METERS_PER_DEG_LAT * math.cos(mid_lat * DEG_TO_RAD))

    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            points.append((lat, lon))
            lon += dlon
        lat += dlat

    return points


def compute_area_km2(polygon: Polygon) -> float:
    """
    Compute the approximate area of a polygon in square kilometers.
    Uses a simple spherical approximation.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon

    Returns
    -------
    float
        Area in km^2.
    """
    # For accurate area, use pyproj or a geodetic calculation
    # This is a reasonable approximation for small areas
    minx, miny, maxx, maxy = polygon.bounds
    mid_lat = (miny + maxy) / 2.0
    m_per_deg_lon = METERS_PER_DEG_LAT * math.cos(mid_lat * DEG_TO_RAD)
    m_per_deg_lat = METERS_PER_DEG_LAT

    # Approximate area from UTM-projected coordinates
    from pyproj import CRS, Transformer

    utm_crs = CRS.from_dict({
        'proj': 'utm',
        'zone': int((minx + 180) / 6) + 1,
        'south': mid_lat < 0,
        'ellps': 'WGS84'
    })
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84, utm_crs, always_xy=True)

    coords = list(polygon.exterior.coords)
    projected = [transformer.transform(x, y) for x, y in coords]
    projected_poly = Polygon(projected)
    return projected_poly.area / 1e6  # m^2 to km^2
