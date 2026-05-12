"""
Global Wind Atlas (GWA) Data Downloader for WindFarm Designer Pro.
Fetches mesoscale wind resource data from the Global Wind Atlas API
operated by DTU (Technical University of Denmark).

The GWA provides:
- Mean wind speed at multiple heights (10m, 50m, 100m, 150m, 200m)
- Wind rose data (directional distribution, 12 sectors)
- Weibull parameters (A and k) per sector
- Power density

The primary API endpoint is a POST request to:
    https://globalwindatlas.info/api/gwa/windresource/v2

with a JSON body containing lat, lon, and height.

If the API is unavailable, synthetic wind data is generated from a
simplified WAsP-like power-law model with realistic sector distributions.
The fallback is designed to degrade gracefully with minimal noise.
"""

import math
import logging
import threading
import time
import requests
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# GWA API Configuration
# ============================================================

# The known-working GWA v2 POST endpoint
GWA_API_ENDPOINT = "https://globalwindatlas.info/api/gwa/windresource/v2"

# Fallback GET endpoints (tried in order if POST fails)
GWA_API_GET_ENDPOINTS = [
    "https://globalwindatlas.info/api/gwa/windresource",
    "https://globalwindatlas.info/api/GWA/public/windresource",
    "https://globalwindatlas.info/api/windresource",
]

GWA_HEIGHTS = [10, 50, 100, 150, 200]
GWA_SECTORS = 12
GWA_SECTOR_CENTERS = [i * 30 for i in range(GWA_SECTORS)]

# Request settings
MAX_RETRIES = 2
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.3
GRID_SPACING_KM = 3.0  # Default grid spacing (3km to reduce API calls)

# Synthetic wind data defaults
DEFAULT_MEAN_WIND_SPEED = 7.5  # m/s at 100m
DEFAULT_WEIBULL_A = 8.5
DEFAULT_WEIBULL_K = 2.1
DEFAULT_POWER_DENSITY = 380.0  # W/m2

# Required HTTP headers for GWA API (without these, the API returns 400)
GWA_REQUEST_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://globalwindatlas.info/',
    'Origin': 'https://globalwindatlas.info',
    'Content-Type': 'application/json',
}


class GWADownloader:
    """
    Fetches wind resource data from the Global Wind Atlas.
    Falls back to synthetic data generation if the API is unavailable.

    The downloader tries multiple request strategies:
    1. POST to the GWA v2 wind resource endpoint (primary)
    2. GET to legacy GWA endpoints (fallback)
    3. Synthetic wind data generation (last resort)
    """

    def __init__(self, output_dir: str = './gwa_data',
                 buffer_km: float = 20.0,
                 grid_spacing_km: float = GRID_SPACING_KM):
        self.output_dir = Path(output_dir)
        self.buffer_km = buffer_km
        self.grid_spacing_km = grid_spacing_km
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        self.session = requests.Session()
        self.session.headers.update(GWA_REQUEST_HEADERS)

        # API availability tracking
        self._api_available: Optional[bool] = None
        self._use_post: bool = True  # Start with POST (primary method)
        self._get_endpoint_index: int = 0  # Track which GET endpoint to try next
        self._query_count: int = 0  # Track number of queries attempted

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
    # API Detection
    # ============================================================

    def _detect_working_endpoint(self) -> Optional[str]:
        """
        Probe GWA API endpoints and return a description of the working method.
        Tries POST first, then falls back to GET endpoints.

        Returns a string like 'post' or 'get:URL' or None if nothing works.
        """
        test_lat, test_lon = 55.0, 9.0  # Denmark - known GWA coverage

        self._report_status("Checking Global Wind Atlas API availability...")

        # --- Try POST (primary) ---
        try:
            payload = {'lat': test_lat, 'lon': test_lon, 'height': 100}
            resp = self.session.post(
                GWA_API_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                has_data = any(k in data for k in [
                    'meanWindSpeed', 'mean_wind_speed', 'windSpeed',
                    'general', 'properties', 'data', 'output'
                ])
                if has_data:
                    self._report_status("  GWA API available (POST /v2 endpoint)")
                    self._use_post = True
                    self._api_available = True
                    return 'post'
                else:
                    self._report_status(
                        "  POST endpoint responded but with unexpected data format"
                    )
            elif resp.status_code == 400:
                logger.debug("POST endpoint returned 400 for test query")
            elif resp.status_code == 429:
                self._report_status("  GWA API rate-limited during probe, will retry later")
                self._api_available = True
                self._use_post = True
                return 'post'
        except requests.RequestException as e:
            logger.debug(f"POST endpoint connection error: {e}")

        # --- Try GET endpoints (fallback) ---
        for i, url_base in enumerate(GWA_API_GET_ENDPOINTS):
            try:
                params = {'lat': test_lat, 'lon': test_lon, 'height': 100}
                resp = self.session.get(url_base, params=params, timeout=REQUEST_TIMEOUT)

                if resp.status_code == 200:
                    data = resp.json()
                    has_data = any(k in data for k in [
                        'meanWindSpeed', 'mean_wind_speed', 'windSpeed',
                        'general', 'properties', 'data'
                    ])
                    if has_data:
                        self._report_status(f"  GWA API available (GET: {url_base})")
                        self._use_post = False
                        self._get_endpoint_index = i
                        self._api_available = True
                        return f'get:{url_base}'
            except requests.RequestException:
                continue

        self._api_available = False
        self._report_status(
            "  GWA API not reachable. Will use modeled wind data instead."
        )
        return None

    # ============================================================
    # Grid Generation
    # ============================================================

    def generate_query_grid(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> List[Tuple[float, float]]:
        """
        Generate a grid of query points covering the bounding box.
        """
        from src.utils.geo_utils import extend_bbox

        ext_bbox = extend_bbox(bbox, self.buffer_km)
        min_lon, min_lat, max_lon, max_lat = ext_bbox

        points = []
        mid_lat = (min_lat + max_lat) / 2.0
        m_per_deg_lat = 111320.0
        m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(mid_lat))

        dlat = (self.grid_spacing_km * 1000.0) / m_per_deg_lat
        dlon = (self.grid_spacing_km * 1000.0) / m_per_deg_lon

        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                points.append((lat, lon))
                lon += dlon
            lat += dlat

        self._report_status(
            f"Generated {len(points)} query points at {self.grid_spacing_km}km spacing "
            f"for extended bbox [{min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}]"
        )
        return points

    # ============================================================
    # Single Point Query
    # ============================================================

    def query_point(self, lat: float, lon: float, height: int = 100) -> Optional[Dict]:
        """
        Query the GWA API for wind data at a single point and height.
        Returns None on failure.

        Strategy:
        1. If POST is enabled, try POST to /v2 endpoint
        2. If POST fails or is disabled, try GET endpoints
        3. If all fail, return None (caller should fall back to synthetic)
        """
        if self._api_available is False:
            return None

        self._query_count += 1

        # --- Strategy 1: POST request (primary) ---
        if self._use_post:
            result = self._query_point_post(lat, lon, height)
            if result is not None:
                return result
            # If POST returned 400 on the first attempt, switch to GET
            if self._query_count <= 1:
                self._report_status(
                    "  POST endpoint unavailable, trying GET fallback..."
                )
                self._use_post = False

        # --- Strategy 2: GET requests (fallback) ---
        for attempt in range(MAX_RETRIES):
            endpoint = GWA_API_GET_ENDPOINTS[
                min(self._get_endpoint_index, len(GWA_API_GET_ENDPOINTS) - 1)
            ]
            try:
                params = {'lat': lat, 'lon': lon, 'height': height}
                response = self.session.get(
                    endpoint, params=params, timeout=REQUEST_TIMEOUT
                )

                if response.status_code == 200:
                    return self._parse_gwa_response(response.json(), lat, lon, height)

                elif response.status_code == 429:
                    wait = (attempt + 1) * 5
                    self._report_status(f"  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                elif response.status_code == 400:
                    # Silent fallback — don't spam error messages
                    if self._query_count <= 2:
                        self._report_status(
                            f"  GWA API returned unexpected response for "
                            f"({lat:.4f}, {lon:.4f}). Using modeled data."
                        )
                    self._api_available = False
                    return None

                else:
                    if self._query_count <= 2:
                        self._report_status(
                            f"  GWA API returned HTTP {response.status_code}. "
                            f"Using modeled data."
                        )
                    return None

            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    if self._query_count <= 2:
                        self._report_status(
                            f"  GWA API connection issue at ({lat:.4f}, {lon:.4f}). "
                            f"Using modeled data."
                        )
                    return None

        return None

    def _query_point_post(
        self, lat: float, lon: float, height: int
    ) -> Optional[Dict]:
        """
        Query using POST to the GWA v2 wind resource endpoint.
        Returns parsed data dict or None on failure.
        """
        payload = {'lat': lat, 'lon': lon, 'height': height}

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    GWA_API_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT
                )

                if response.status_code == 200:
                    return self._parse_gwa_response(response.json(), lat, lon, height)

                elif response.status_code == 429:
                    wait = (attempt + 1) * 5
                    self._report_status(f"  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                elif response.status_code == 400:
                    # On first 400, log it once and mark POST as failed
                    if self._query_count <= 1:
                        body = response.text[:200]
                        logger.debug(
                            f"GWA POST 400 at ({lat:.4f}, {lon:.4f}): {body}"
                        )
                    self._use_post = False
                    return None

                else:
                    logger.debug(
                        f"GWA POST returned HTTP {response.status_code} "
                        f"at ({lat:.4f}, {lon:.4f})"
                    )
                    return None

            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.debug(f"GWA POST error at ({lat:.4f}, {lon:.4f}): {e}")
                    self._use_post = False
                    return None

        self._use_post = False
        return None

    def _parse_gwa_response(self, data: dict, lat: float, lon: float,
                           height: int) -> Dict:
        """
        Normalize a GWA API response into a standard dict.
        Handles multiple response formats (v1, v2, v3, GeoJSON, etc.).
        """
        result = {
            'lat': lat,
            'lon': lon,
            'height': height,
            'mean_wind_speed': 0.0,
            'mean_power_density': 0.0,
            'mean_weibull_A': 0.0,
            'mean_weibull_k': 0.0,
            'sectors': [],
        }

        # Try multiple response formats

        # Format 1: direct flat response
        if 'meanWindSpeed' in data or 'mean_wind_speed' in data:
            result['mean_wind_speed'] = float(
                data.get('meanWindSpeed', data.get('mean_wind_speed', 0.0))
            )
            result['mean_power_density'] = float(
                data.get('meanPowerDensity', data.get('mean_power_density', 0.0))
            )
            result['mean_weibull_A'] = float(
                data.get('weibullA', data.get('weibull_a', 0.0))
            )
            result['mean_weibull_k'] = float(
                data.get('weibullK', data.get('weibull_k', 0.0))
            )

        # Format 2: nested 'properties' (GeoJSON-like)
        elif 'properties' in data:
            props = data['properties']
            result['mean_wind_speed'] = float(props.get('meanWindSpeed', 0.0))
            result['mean_power_density'] = float(props.get('meanPowerDensity', 0.0))
            result['mean_weibull_A'] = float(props.get('weibullA', 0.0))
            result['mean_weibull_k'] = float(props.get('weibullK', 0.0))

        # Format 3: nested 'general' key
        elif 'general' in data:
            gen = data['general']
            result['mean_wind_speed'] = float(
                gen.get('meanWindSpeed', gen.get('ws', 0.0))
            )
            result['mean_power_density'] = float(
                gen.get('powerDensity', gen.get('pd', 0.0))
            )
            wr = data.get('windRose', data.get('frequency', {}))
            if isinstance(wr, dict):
                result['mean_weibull_A'] = float(wr.get('A', 0.0))
                result['mean_weibull_k'] = float(wr.get('k', 0.0))
            elif isinstance(wr, list) and wr:
                result['mean_weibull_A'] = float(wr[0].get('A', 0.0))
                result['mean_weibull_k'] = float(wr[0].get('k', 0.0))

        # Format 4: nested 'output' key (some v2 responses)
        elif 'output' in data:
            output = data['output']
            if isinstance(output, dict):
                result['mean_wind_speed'] = float(
                    output.get('meanWindSpeed', output.get('mean_wind_speed', 0.0))
                )
                result['mean_power_density'] = float(
                    output.get('meanPowerDensity', output.get('mean_power_density', 0.0))
                )
                result['mean_weibull_A'] = float(
                    output.get('weibullA', output.get('weibull_a', 0.0))
                )
                result['mean_weibull_k'] = float(
                    output.get('weibullK', output.get('weibull_k', 0.0))
                )

        # If we got a wind speed but no Weibull params, estimate them
        if result['mean_wind_speed'] > 0 and result['mean_weibull_A'] <= 0:
            result['mean_weibull_A'] = result['mean_wind_speed'] * 1.12
            result['mean_weibull_k'] = 2.1

        # Parse sector data
        sector_data = (
            data.get('sectors', [])
            or data.get('windRose', [])
            or data.get('frequency', [])
        )
        if isinstance(sector_data, dict):
            sector_data = sector_data.get('sectors', sector_data.get('bins', []))

        if isinstance(sector_data, list):
            for i, sector in enumerate(sector_data[:GWA_SECTORS]):
                result['sectors'].append({
                    'direction': GWA_SECTOR_CENTERS[i] if i < GWA_SECTORS else i * 30,
                    'frequency': float(sector.get('frequency', sector.get('freq', 0.0))),
                    'mean_speed': float(
                        sector.get('meanWindSpeed',
                                   sector.get('meanSpeed', sector.get('speed', 0.0))
                        ) if result['mean_wind_speed'] > 0 else 0.0
                    ),
                    'weibull_A': float(
                        sector.get('weibullA', sector.get('A',
                                  result['mean_weibull_A']))
                    ),
                    'weibull_k': float(
                        sector.get('weibullK', sector.get('k',
                                  result['mean_weibull_k']))
                    ),
                    'power_density': float(
                        sector.get('powerDensity', sector.get('pd', 0.0))
                    ),
                })

        # If no sectors returned, generate uniform distribution
        if not result['sectors'] and result['mean_wind_speed'] > 0:
            freq = 1.0 / GWA_SECTORS
            for s in range(GWA_SECTORS):
                result['sectors'].append({
                    'direction': GWA_SECTOR_CENTERS[s],
                    'frequency': freq,
                    'mean_speed': result['mean_wind_speed'],
                    'weibull_A': result['mean_weibull_A'],
                    'weibull_k': result['mean_weibull_k'],
                    'power_density': result['mean_power_density'],
                })

        return result

    # ============================================================
    # Synthetic Wind Data Generation
    # ============================================================

    def _generate_realistic_sector_distribution(
        self,
        prevailing_direction: float = 225.0,
        concentration: float = 2.5,
    ) -> np.ndarray:
        """
        Generate a realistic 12-sector wind frequency distribution using a
        wrapped normal (von Mises-like) distribution around a prevailing
        direction.

        The result is normalized so frequencies sum to 1.0.

        Parameters
        ----------
        prevailing_direction : float
            Dominant wind direction in degrees (meteorological: where wind comes FROM).
            Default 225° (southwest) is typical for many mid-latitude regions.
        concentration : float
            Concentration parameter (analogous to von Mises kappa).
            Higher values = more peaked distribution. Range 1.0–5.0.
            2.5 gives a moderate prevailing wind with some spread.

        Returns
        -------
        np.ndarray of shape (12,)
            Frequency for each of the 12 sectors (N, NNE, ..., NNW).
        """
        # Convert prevailing direction to the 0-11 sector index
        # GWA sectors: 0=N(0°), 1=NNE(30°), 2=ENE(60°), ...
        prevailing_sector = (prevailing_direction / 30.0) % GWA_SECTORS

        frequencies = np.zeros(GWA_SECTORS)
        for s in range(GWA_SECTORS):
            # Angular distance (circular, accounting for wraparound)
            delta = abs(s - prevailing_sector)
            delta = min(delta, GWA_SECTORS - delta)
            # Wrapped normal-like distribution
            frequencies[s] = math.exp(-0.5 * (delta / concentration) ** 2)

        # Normalize to sum to 1.0
        frequencies /= frequencies.sum()
        return frequencies

    def _estimate_regional_wind_params(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Dict:
        """
        Estimate reasonable wind parameters based on geographic location.
        Uses latitude-based heuristics for a first-order approximation.

        Returns dict with keys: mean_speed, weibull_k, prevailing_dir, concentration
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        mid_lat = (min_lat + max_lat) / 2.0
        mid_lon = (min_lon + max_lon) / 2.0

        # Base wind speed tends to increase with latitude (storm tracks)
        # and decrease near equator (doldrums). Simple sinusoidal model:
        # - Max around 50-60°N (North Atlantic storm track)
        # - Min around 10°N (ITCZ)
        lat_factor = math.sin(math.radians(mid_lat)) ** 0.8
        base_speed = 5.0 + 4.0 * lat_factor  # 5.0 at equator, ~9.0 at 50°N

        # Continental vs coastal effect: areas far from coast tend to have
        # lower wind speeds. Simple heuristic using mid_lon:
        # (Not perfectly accurate but better than a flat default)
        coastal_bonus = 0.5 * (1.0 + math.sin(math.radians(mid_lon * 2)))

        mean_speed = base_speed + coastal_bonus
        mean_speed = max(4.0, min(12.0, mean_speed))  # Clamp to realistic range

        # Weibull k tends to be higher (narrower distribution) at mid-latitudes
        # and lower (wider distribution) in tropical/subtropical regions
        weibull_k = 1.8 + 0.6 * lat_factor
        weibull_k = max(1.5, min(2.8, weibull_k))

        # Prevailing wind direction varies by region (rough heuristic)
        # Northern mid-latitudes: westerlies (~270°)
        # Tropics: trade winds (~60-90°NE or ~180°S)
        # Southern mid-latitudes: westerlies (~270°)
        if abs(mid_lat) < 15:
            # Tropical: NE trades (Northern) or SE trades (Southern)
            prevailing_dir = 45.0 if mid_lat >= 0 else 180.0
            concentration = 2.0  # Steady trade winds
        elif abs(mid_lat) < 30:
            # Subtropical: variable, slight westerly component
            prevailing_dir = 270.0 if mid_lat >= 0 else 270.0
            concentration = 1.8  # More variable
        else:
            # Mid-latitudes: prevailing westerlies
            prevailing_dir = 250.0  # WSW
            concentration = 2.5  # Moderately concentrated

        return {
            'mean_speed': mean_speed,
            'weibull_k': weibull_k,
            'prevailing_dir': prevailing_dir,
            'concentration': concentration,
        }

    def generate_synthetic_data(
        self,
        bbox: Tuple[float, float, float, float],
        height: int = 100,
        mean_speed: float = None,
        weibull_k: float = None,
    ) -> Dict:
        """
        Generate synthetic wind resource data using a simplified model.

        This creates realistic-looking wind data with:
        - Regional wind speed estimation based on latitude
        - Spatial variation using multi-frequency pseudo-noise
        - 12-sector wind rose with von Mises-like distribution
        - Per-sector Weibull parameters with realistic cross-sector variation
        - Proper extrapolation to the target height using power law

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        height : int
            Target hub height (m).
        mean_speed : float or None
            Override mean wind speed (m/s) at 100m. If None, estimated from location.
        weibull_k : float or None
            Override Weibull shape parameter. If None, estimated from location.
        """
        from src.utils.geo_utils import extend_bbox

        ext_bbox = extend_bbox(bbox, self.buffer_km)

        # Estimate regional parameters
        regional = self._estimate_regional_wind_params(bbox)

        if mean_speed is None:
            mean_speed = regional['mean_speed']
        if weibull_k is None:
            weibull_k = regional['weibull_k']

        points = self.generate_query_grid(bbox)
        total = len(points)

        self._report_status(
            f"Generating modeled wind data for {total} points at {height}m "
            f"(mean speed ≈ {mean_speed:.1f} m/s, k ≈ {weibull_k:.2f})..."
        )

        # Compute Weibull A from mean speed and k
        try:
            from scipy.special import gamma as gamma_func
            A = mean_speed / gamma_func(1.0 + 1.0 / weibull_k)
        except ImportError:
            # Fallback approximation: gamma(1+1/k) ≈ 0.886 for k=2
            gamma_approx = 0.886 if abs(weibull_k - 2.0) < 0.3 else 0.90
            A = mean_speed / gamma_approx

        # Generate sector wind rose distribution
        sector_freqs = self._generate_realistic_sector_distribution(
            prevailing_direction=regional['prevailing_dir'],
            concentration=regional['concentration'],
        )

        # Per-sector speed variation factors (sectors with higher frequency
        # often have slightly different speeds — calmer sectors are calmer)
        # This creates more realistic-looking sector data
        np.random.seed(42)  # Reproducible
        sector_speed_factors = np.array([
            1.0 + 0.20 * (freq * GWA_SECTORS - 1.0)
            + np.random.normal(0, 0.03)
            for freq in sector_freqs
        ])
        sector_speed_factors = np.clip(sector_speed_factors, 0.6, 1.4)

        # Per-sector Weibull k variation (±0.2 around the mean)
        sector_k_offsets = np.random.uniform(-0.15, 0.15, GWA_SECTORS)

        results = []
        for i, (lat, lon) in enumerate(points):
            # Multi-frequency spatial variation (±12% around mean)
            # Uses several sine/cosine terms to simulate terrain effects
            noise = (
                math.sin(lat * 7.3 + lon * 13.7) * 0.06
                + math.cos(lat * 23.1 - lon * 11.3) * 0.04
                + math.sin(lat * 31.7 + lon * 5.1) * 0.02
                + math.cos(lat * 47.3 + lon * 19.9) * 0.01
            )
            local_speed = mean_speed * (1.0 + noise)
            local_A = A * (1.0 + noise)
            local_k = weibull_k + 0.08 * noise

            # Extrapolate from reference height (100m) to target height
            # Using power law: v(z) = v(zref) * (z/zref)^alpha
            z_ref = 100.0
            if height != z_ref:
                z0 = 0.03  # Default roughness (grassland)
                alpha = 1.0 / math.log(z_ref / z0)
                height_factor = (height / z_ref) ** alpha
                local_speed *= height_factor
                local_A *= height_factor

            # Compute power density: P = 0.5 * rho * v^3
            local_pd = 0.5 * 1.225 * local_speed ** 3  # W/m² (rough)

            # Sector-specific data with directional variation
            sectors = []
            for s in range(GWA_SECTORS):
                sf = sector_speed_factors[s]
                sk = local_k + sector_k_offsets[s]
                sk = max(1.5, min(3.0, sk))  # Clamp to physical range
                sA = local_A * sf
                s_speed = local_speed * sf
                s_pd = 0.5 * 1.225 * s_speed ** 3

                sectors.append({
                    'direction': GWA_SECTOR_CENTERS[s],
                    'frequency': round(float(sector_freqs[s]), 4),
                    'mean_speed': round(s_speed, 2),
                    'weibull_A': round(sA, 2),
                    'weibull_k': round(sk, 3),
                    'power_density': round(s_pd, 1),
                })

            results.append({
                'lat': lat,
                'lon': lon,
                'height': height,
                'mean_wind_speed': round(local_speed, 2),
                'mean_power_density': round(local_pd, 1),
                'mean_weibull_A': round(local_A, 2),
                'mean_weibull_k': round(local_k, 3),
                'sectors': sectors,
            })

            if (i + 1) % 20 == 0 or (i + 1) == total:
                self._report_progress(i + 1, total,
                                     f"Generated ({lat:.4f}, {lon:.4f})")

        # Summary stats
        speeds = [r['mean_wind_speed'] for r in results]
        mean_ws = float(np.mean(speeds))
        mean_pd = float(np.mean([r['mean_power_density'] for r in results]))

        output_data = {
            'points': results,
            'mean_speed': mean_ws,
            'mean_power_density': mean_pd,
            'num_points': len(results),
            'height': height,
            'bbox': bbox,
            'source': 'synthetic',
            'synthetic_params': {
                'mean_speed': mean_speed,
                'weibull_k': weibull_k,
                'reference_height': 100,
                'prevailing_direction': regional['prevailing_dir'],
                'concentration': regional['concentration'],
            },
        }

        # Save
        output_file = self.output_dir / f'gwa_data_{height}m_synthetic.json'
        self._save_data(output_data, str(output_file))

        self._report_status(
            f"Modeled wind data generated: {len(results)} points, "
            f"mean speed = {mean_ws:.1f} m/s at {height}m, "
            f"mean power density = {mean_pd:.0f} W/m²"
        )
        self._report_status(
            "  Note: Using modeled estimates. For production assessments, "
            "import measured mast data or validated wind atlas files."
        )

        return output_data

    # ============================================================
    # Bulk Grid Download
    # ============================================================

    def download_grid(
        self,
        bbox: Tuple[float, float, float, float],
        height: int = 100,
        max_threads: int = 1,
    ) -> Dict:
        """
        Download wind resource data for all points in a grid.
        Falls back to synthetic data if the API is unavailable.
        """
        # First, check if API is available (probe once)
        if self._api_available is None:
            self._detect_working_endpoint()

        # If API is not available, generate synthetic data immediately
        if not self._api_available:
            self._report_status(
                "Global Wind Atlas API is not reachable."
            )
            self._report_status("Generating modeled wind data instead...")
            return self.generate_synthetic_data(bbox, height)

        points = self.generate_query_grid(bbox)
        total = len(points)
        if total == 0:
            return {}

        self._report_status(
            f"Downloading GWA data for {total} points at {height}m height..."
        )

        results = []
        completed = [0]
        consecutive_failures = [0]
        MAX_CONSECUTIVE_FAILURES = 3

        for i, (lat, lon) in enumerate(points):
            data = self.query_point(lat, lon, height)

            if data:
                results.append(data)
                consecutive_failures[0] = 0
            else:
                consecutive_failures[0] += 1

            completed[0] += 1
            if completed[0] % 5 == 0 or completed[0] == total:
                self._report_progress(completed[0], total,
                                     f"Queried ({lat:.4f}, {lon:.4f})")

            # Rate limiting
            time.sleep(REQUEST_DELAY)

            # If too many consecutive failures, switch to synthetic
            if consecutive_failures[0] >= MAX_CONSECUTIVE_FAILURES:
                self._report_status(
                    f"  API returned {MAX_CONSECUTIVE_FAILURES} consecutive errors. "
                    f"Switching to modeled data..."
                )
                return self.generate_synthetic_data(bbox, height)

        if not results:
            self._report_status(
                "  No GWA data retrieved. Generating modeled data..."
            )
            return self.generate_synthetic_data(bbox, height)

        # Compute summary statistics
        speeds = [r['mean_wind_speed'] for r in results if r['mean_wind_speed'] > 0]
        power_densities = [r['mean_power_density'] for r in results
                           if r['mean_power_density'] > 0]

        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        mean_pd = float(np.mean(power_densities)) if power_densities else 0.0

        output_data = {
            'points': results,
            'mean_speed': mean_speed,
            'mean_power_density': mean_pd,
            'num_points': len(results),
            'height': height,
            'bbox': bbox,
            'source': 'gwa_api',
        }

        # Save raw data
        output_file = self.output_dir / f'gwa_data_{height}m.json'
        self._save_data(output_data, str(output_file))

        self._report_status(
            f"GWA download complete: {len(results)} points, "
            f"mean speed = {mean_speed:.1f} m/s at {height}m, "
            f"mean power density = {mean_pd:.0f} W/m²"
        )

        return output_data

    # ============================================================
    # Resource Grid (numpy raster)
    # ============================================================

    def create_wind_speed_grid(
        self, gwa_data: Dict, bbox: Tuple[float, float, float, float]
    ) -> Optional[str]:
        """Convert GWA point data into a raster grid (GeoTIFF)."""
        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.crs import CRS
        except ImportError:
            self._report_status("  Note: rasterio not available, skipping grid creation.")
            return None

        from src.utils.geo_utils import extend_bbox
        ext_bbox = extend_bbox(bbox, self.buffer_km)
        min_lon, min_lat, max_lon, max_lat = ext_bbox

        points = gwa_data.get('points', [])
        if not points:
            return None

        lats = sorted(set(p['lat'] for p in points))
        lons = sorted(set(p['lon'] for p in points))

        if len(lats) < 2 or len(lons) < 2:
            return None

        ny = len(lats)
        nx = len(lons)

        speed_grid = np.full((ny, nx), -9999.0, dtype=np.float32)

        lat_idx = {lat: i for i, lat in enumerate(lats)}
        lon_idx = {lon: i for i, lon in enumerate(lons)}

        for p in points:
            iy = lat_idx.get(p['lat'])
            ix = lon_idx.get(p['lon'])
            if iy is not None and ix is not None:
                speed_grid[iy, ix] = p['mean_wind_speed']

        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, nx, ny)
        speed_file = self.output_dir / f'wind_speed_{gwa_data["height"]}m.tif'

        with rasterio.open(
            speed_file, 'w',
            driver='GTiff',
            height=ny, width=nx,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(4326),
            transform=transform,
            nodata=-9999,
            compress='DEFLATE',
        ) as dst:
            dst.write(speed_grid, 1)

        self._report_status(f"  Wind speed grid saved to: {speed_file}")
        return str(speed_file)

    # ============================================================
    # Interpolation to Arbitrary Points
    # ============================================================

    def interpolate_to_points(
        self,
        gwa_data: Dict,
        target_points: List[Tuple[float, float]],
    ) -> List[Dict]:
        """IDW interpolation of GWA wind data to target points."""
        source_points = gwa_data.get('points', [])
        if not source_points:
            return []

        results = []
        power = 2

        for tlat, tlon in target_points:
            distances = []
            for sp in source_points:
                dlat = tlat - sp['lat']
                dlon = tlon - sp['lon']
                dist = math.sqrt(dlat ** 2 + (dlon * math.cos(math.radians(tlat))) ** 2)
                if dist < 1e-10:
                    results.append(sp)
                    break
                distances.append((dist, sp))
            else:
                distances.sort(key=lambda x: x[0])
                n_neighbors = min(8, len(distances))
                neighbors = distances[:n_neighbors]

                weights = []
                vs, vp, vA, vk = [], [], [], []

                for d, sp in neighbors:
                    w = 1.0 / (d ** power)
                    weights.append(w)
                    vs.append(sp['mean_wind_speed'])
                    vp.append(sp['mean_power_density'])
                    vA.append(sp['mean_weibull_A'])
                    vk.append(sp['mean_weibull_k'])

                tw = sum(weights)
                if tw > 0:
                    results.append({
                        'lat': tlat, 'lon': tlon,
                        'mean_wind_speed': sum(w * v for w, v in zip(weights, vs)) / tw,
                        'mean_power_density': sum(w * v for w, v in zip(weights, vp)) / tw,
                        'mean_weibull_A': sum(w * v for w, v in zip(weights, vA)) / tw,
                        'mean_weibull_k': sum(w * v for w, v in zip(weights, vk)) / tw,
                        'sectors': self._interpolate_sectors(neighbors, weights, tw),
                    })
                else:
                    results.append({
                        'lat': tlat, 'lon': tlon,
                        'mean_wind_speed': 0.0, 'mean_power_density': 0.0,
                        'mean_weibull_A': 0.0, 'mean_weibull_k': 0.0,
                        'sectors': [],
                    })

        return results

    def _interpolate_sectors(self, neighbors, weights, total_w):
        """Interpolate sector data using IDW weights."""
        interpolated = []
        for s in range(GWA_SECTORS):
            sf, ss, sA, sk = [], [], [], []
            for _, sp in neighbors:
                sectors = sp.get('sectors', [])
                if s < len(sectors):
                    sec = sectors[s]
                    sf.append(sec.get('frequency', 0.0))
                    ss.append(sec.get('mean_speed', 0.0))
                    sA.append(sec.get('weibull_A', 0.0))
                    sk.append(sec.get('weibull_k', 0.0))
                else:
                    sf.append(0.0); ss.append(0.0)
                    sA.append(0.0); sk.append(0.0)

            interpolated.append({
                'direction': GWA_SECTOR_CENTERS[s],
                'frequency': sum(w * v for w, v in zip(weights, sf)) / total_w,
                'mean_speed': sum(w * v for w, v in zip(weights, ss)) / total_w,
                'weibull_A': sum(w * v for w, v in zip(weights, sA)) / total_w,
                'weibull_k': sum(w * v for w, v in zip(weights, sk)) / total_w,
            })
        return interpolated

    # ============================================================
    # Data I/O
    # ============================================================

    def _save_data(self, data: Dict, filepath: str):
        """Save data to JSON file."""
        import json
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj) if isinstance(obj, (np.float64, np.float32)) else int(obj)
            return obj
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=convert)

    def load_data(self, filepath: str) -> Dict:
        """Load previously saved GWA data."""
        import json
        with open(filepath, 'r') as f:
            return json.load(f)

    # ============================================================
    # Full Pipeline
    # ============================================================

    def download_and_process(
        self,
        bbox: Tuple[float, float, float, float],
        height: int = 100,
    ) -> Dict:
        """
        Complete pipeline: download GWA data (or fallback to synthetic)
        and create raster grids.
        """
        gwa_data = self.download_grid(bbox, height)

        if not gwa_data:
            self._report_status("No wind data produced.")
            return {}

        speed_grid_path = self.create_wind_speed_grid(gwa_data, bbox)

        return {
            'gwa_data': gwa_data,
            'wind_speed_grid': speed_grid_path,
            'mean_speed': gwa_data.get('mean_speed', 0.0),
            'mean_power_density': gwa_data.get('mean_power_density', 0.0),
            'num_points': gwa_data.get('num_points', 0),
            'source': gwa_data.get('source', 'unknown'),
        }


# ============================================================
# Convenience function
# ============================================================

def download_gwa_for_area(
    bbox: Tuple[float, float, float, float],
    output_dir: str = './gwa_data',
    buffer_km: float = 20.0,
    height: int = 100,
    grid_spacing_km: float = 3.0,
    status_callback: Optional[Callable] = None,
) -> Dict:
    """High-level function to download or generate wind resource data."""
    downloader = GWADownloader(
        output_dir=output_dir,
        buffer_km=buffer_km,
        grid_spacing_km=grid_spacing_km,
    )
    if status_callback:
        downloader.set_status_callback(status_callback)

    return downloader.download_and_process(bbox, height)
