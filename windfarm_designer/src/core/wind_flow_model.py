"""
Wind Flow Model for WindFarm Designer Pro – WASP-like Methodology.

This module implements a wind flow model closely following the WASP
(Wind Atlas Analysis and Application Program) methodology.  It accounts
for:

1. **Jackson-Hunt Linearised Theory** (1975) for terrain speed-up over
   low hills under neutral stratification, using modified Bessel
   functions K₀ and K₁.
2. **Internal Boundary Layer (IBL) Growth** when wind flows across a
   roughness change, with the neutral-stability growth law.
3. **Effective Weighing Area (EWA)** method for computing per-cell
   effective roughness by tracing upstream along each wind direction.
4. **Flow separation detection** for steep lee-side slopes.
5. **Combined speed-up maps** per-sector and frequency-weighted.

The model takes as input:
- DEM (Digital Elevation Model) from SRTM
- Roughness map from land cover data
- Reference wind data (from GWA or mast measurements)
- Wind farm turbine positions

And produces:
- Corrected wind speeds at each turbine position
- Per-sector terrain and roughness speed-up factors
- Turbulence intensity estimates
- Georeferenced speed-up maps (GeoTIFF, EPSG:4326)

Note: This is a diagnostic model inspired by WASP.  For production-grade
CFD-based modelling, use external tools (OpenFOAM, WRF, etc.).
"""

import math
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from scipy.special import k0 as bessel_k0
from scipy.special import k1 as bessel_k1
from scipy.ndimage import (
    gaussian_filter,
    minimum_filter,
    maximum_filter,
    uniform_filter,
)
from scipy.interpolate import RegularGridInterpolator

from .layout_optimizer import WTGPosition
from src.utils.geo_utils import wind_speed_at_height

logger = logging.getLogger(__name__)

# Physical constants
LOG_E = math.log(math.e)  # 1.0


# ============================================================
# Flow Model Configuration
# ============================================================

class FlowModelConfig:
    """Configuration for wind flow modelling."""

    def __init__(
        self,
        n_sectors: int = 12,
        sector_width_deg: float = 30.0,
        search_radius_cells: int = 50,
        roughness_class: float = 0.03,
        min_roughness: float = 0.0001,
        max_roughness: float = 5.0,
        reference_height: float = 100.0,
        target_height: float = 100.0,
        use_terrain_correction: bool = True,
        use_roughness_correction: bool = True,
    ):
        self.n_sectors = n_sectors
        self.sector_width_deg = sector_width_deg
        self.search_radius_cells = search_radius_cells
        self.roughness_class = roughness_class
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.reference_height = reference_height
        self.target_height = target_height
        self.use_terrain_correction = use_terrain_correction
        self.use_roughness_correction = use_roughness_correction


# ============================================================
# Wind Flow Model
# ============================================================

class WindFlowModel:
    """
    WASP-like wind flow model over terrain.

    The model computes terrain-induced speed-up and roughness-induced
    wind speed modifications at each turbine position following the
    methodology of the European Wind Atlas / WASP:

    * **Terrain speed-up** – Jackson-Hunt (1975) linearised theory
      using modified Bessel functions K₀, K₁ to capture the
      inner/outer boundary-layer structure over hills.

    * **Roughness effects** – Internal Boundary Layer (IBL) growth
      combined with the Effective Weighing Area (EWA) method to
      compute per-cell effective roughness and resulting speed-up/slow-
      down ratios.

    * **Flow separation** – simple detection for steep lee slopes
      (> 0.3 gradient) where the linearised theory breaks down.
    """

    def __init__(self, config: FlowModelConfig = None):
        self.config = config or FlowModelConfig()
        self.dem: Optional[np.ndarray] = None
        self.dem_transform = None
        self.roughness_map: Optional[np.ndarray] = None
        self.roughness_transform = None
        self.slope_map: Optional[np.ndarray] = None
        self.aspect_map: Optional[np.ndarray] = None
        self.dem_smooth: Optional[np.ndarray] = None

        # Cached cell size (metres) – computed once when DEM is loaded
        self._cell_size_m: float = 30.0

        # Cached results (so downstream methods can reuse them)
        self._terrain_speedup_maps: Dict[int, np.ndarray] = {}
        self._roughness_results: Optional[Dict] = None

        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None

    # ----------------------------------------------------------
    # Callback helpers
    # ----------------------------------------------------------

    def set_progress_callback(self, callback: Callable):
        self.progress_callback = callback

    def set_status_callback(self, callback: Callable):
        self.status_callback = callback

    def _report_status(self, msg: str):
        logger.info(msg)
        if self.status_callback:
            self.status_callback(msg)

    def _report_progress(self, current: int, total: int, msg: str = ""):
        if self.progress_callback:
            self.progress_callback(current, total, msg)

    # ==========================================================
    # Data Loading
    # ==========================================================

    def load_dem(self, dem_path: str):
        """Load DEM from GeoTIFF file."""
        try:
            import rasterio
        except ImportError:
            self._report_status("ERROR: rasterio required for DEM loading.")
            return

        self._report_status(f"Loading DEM from {dem_path}...")
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1).astype(np.float64)
            self.dem_transform = src.transform
            self.dem_profile = src.profile
            self.dem_nodata = src.nodata if src.nodata else -32768

        # Replace nodata with NaN
        self.dem[self.dem == self.dem_nodata] = np.nan

        # Smooth DEM for terrain analysis (WASP uses smoothed profiles)
        self.dem_smooth = gaussian_filter(
            np.nan_to_num(self.dem, nan=0.0), sigma=3
        )
        self.dem_smooth[np.isnan(self.dem)] = np.nan

        # Compute cell size in metres for later use
        self._cell_size_m = self._compute_cell_size_m()

        self._report_status(
            f"DEM loaded: {self.dem.shape[1]}x{self.dem.shape[0]} pixels, "
            f"cell size ~{self._cell_size_m:.1f} m"
        )

    def load_roughness(self, roughness_path: str):
        """Load roughness map from GeoTIFF file."""
        try:
            import rasterio
        except ImportError:
            self._report_status("ERROR: rasterio required for roughness loading.")
            return

        self._report_status(f"Loading roughness map from {roughness_path}...")
        with rasterio.open(roughness_path) as src:
            self.roughness_map = src.read(1).astype(np.float64)
            self.roughness_transform = src.transform
            self.roughness_nodata = src.nodata if src.nodata else -9999

        # Replace negative / NaN with default grassland roughness
        default_z0 = 0.03
        self.roughness_map[self.roughness_map < 0] = default_z0
        self.roughness_map[np.isnan(self.roughness_map)] = default_z0

        self._report_status(
            f"Roughness map loaded: "
            f"{self.roughness_map.shape[1]}x{self.roughness_map.shape[0]} pixels"
        )

    def load_slope_aspect(self, slope_path: str = None, aspect_path: str = None):
        """Load pre-computed slope and aspect maps."""
        try:
            import rasterio
        except ImportError:
            return

        if slope_path:
            with rasterio.open(slope_path) as src:
                self.slope_map = src.read(1).astype(np.float64)

        if aspect_path:
            with rasterio.open(aspect_path) as src:
                self.aspect_map = src.read(1).astype(np.float64)

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _compute_cell_size_m(self) -> float:
        """Estimate the DEM cell size in metres at the scene centre."""
        if self.dem_transform is None or self.dem is None:
            return 30.0  # sensible default for SRTM 1-arcsec ≈ 30 m
        pixel_size_deg = abs(self.dem_transform.a)
        ny, nx = self.dem.shape
        center_y = (self.dem_transform * (nx / 2, ny / 2))[1]
        m_per_deg = 111320.0 * math.cos(math.radians(abs(center_y)))
        return pixel_size_deg * m_per_deg

    def _cell_size_lon_m(self) -> float:
        """Cell size in the longitude direction (metres)."""
        if self.dem_transform is None:
            return 30.0
        return abs(self.dem_transform.a) * 111320.0 * math.cos(
            math.radians(self._scene_center_lat())
        )

    def _cell_size_lat_m(self) -> float:
        """Cell size in the latitude direction (metres)."""
        if self.dem_transform is None:
            return 30.0
        return abs(self.dem_transform.a) * 111320.0

    def _scene_center_lat(self) -> float:
        if self.dem_transform is None or self.dem is None:
            return 0.0
        ny, nx = self.dem.shape
        return (self.dem_transform * (nx / 2, ny / 2))[1]

    # ==========================================================
    # 1. Jackson-Hunt Terrain Speed-Up
    # ==========================================================

    @staticmethod
    def _bessel_factor(sigma: np.ndarray) -> np.ndarray:
        """
        Compute the Jackson-Hunt Bessel function factor B(σ):

            B(σ) = σ · K₁(σ) / K₀(σ)

        where K₀ and K₁ are the modified Bessel functions of the
        second kind (order 0 and 1), and

            σ = ln(L / z₀)

        This factor captures the coupling between the inner (surface)
        and outer (pressure-gradient) layers in the linearised
        solution.

        Parameters
        ----------
        sigma : np.ndarray
            The log-ratio ln(L/z₀) for each grid cell.

        Returns
        -------
        np.ndarray
            B(σ) values, clipped to avoid singularities.
        """
        sigma = np.maximum(sigma, 1.0)  # avoid log(0) / Bessel(0) blow-up

        # Guard against overflow in Bessel functions for very large sigma
        # K0 and K1 decay rapidly; use asymptotic approximation for sigma > 50
        b_val = np.empty_like(sigma)
        small = sigma <= 50.0
        large = ~small

        if np.any(small):
            s = sigma[small]
            k0_val = bessel_k0(s)
            k1_val = bessel_k1(s)
            # Avoid division by zero
            k0_val = np.maximum(k0_val, 1e-30)
            b_val[small] = s * k1_val / k0_val

        if np.any(large):
            # Asymptotic expansion: K_n(x) ~ sqrt(pi/(2x)) * exp(-x)
            # So K1/K0 -> 1 for large x, and B(σ) -> σ
            b_val[large] = sigma[large] * 0.98  # slightly < 1 from next term

        return b_val

    def _compute_terrain_effects(self) -> Dict:
        """
        Compute terrain-induced wind speed modifications using the
        Jackson-Hunt (1975) linearised theory.

        For each of 12 wind sectors the model:

        1. Estimates effective hill height *H* as the difference
           between local smoothed elevation and the upstream minimum
           over a characteristic fetch.
        2. Estimates hill half-length *L* from the upstream fetch
           distance.
        3. Computes the Bessel function factor B(σ) where
           σ = ln(L / z₀).
        4. Applies the Jackson-Hunt speed-up:
           Δs/s₀ ≈ 2 · (H/L) · B(σ)
        5. Detects flow separation on steep lee slopes.

        Returns
        -------
        dict
            ``speedup_maps``: per-sector 2-D arrays,
            ``slope_map``, ``aspect_map``.
        """
        if self.dem is None:
            self._report_status("ERROR: DEM not loaded.")
            return {}

        self._report_status("Computing terrain effects (Jackson-Hunt)…")

        # Compute slope / aspect if not already loaded
        if self.slope_map is None or self.aspect_map is None:
            self._compute_slope_aspect()

        speedup_maps: Dict[int, np.ndarray] = {}
        for s in range(self.config.n_sectors):
            direction = s * self.config.sector_width_deg
            self._report_status(f"  Sector {s}: wind from {direction:.0f}°")
            speedup_maps[s] = self._compute_jackson_hunt_speedup(direction)

        self._terrain_speedup_maps = speedup_maps

        self._report_status("Terrain effects computed.")
        return {
            "speedup_maps": speedup_maps,
            "slope_map": self.slope_map,
            "aspect_map": self.aspect_map,
        }

    def _compute_slope_aspect(self):
        """Compute slope and aspect from DEM using central differences."""
        if self.dem_smooth is None:
            return

        dx = self._cell_size_lon_m()
        dy = self._cell_size_lat_m()

        dem_filled = np.nan_to_num(self.dem_smooth, nan=0.0)
        dzdy, dzdx = np.gradient(dem_filled, dy, dx)

        self.slope_map = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))
        self.aspect_map = np.degrees(np.arctan2(-dzdy, dzdx)) % 360

    def _compute_jackson_hunt_speedup(self, wind_direction_deg: float) -> np.ndarray:
        """
        Jackson-Hunt linearised terrain speed-up for one sector.

        Core equation (Jackson & Hunt, 1975):

            Δs / s₀ ≈ 2 · (H / L) · B(σ)

        where:
        - H = effective hill height (local − upstream min)
        - L = hill half-length (fetch distance)
        - B(σ) = σ · K₁(σ) / K₀(σ),  σ = ln(L / z₀)
        - K₀, K₁ = modified Bessel functions of 2nd kind

        Parameters
        ----------
        wind_direction_deg : float
            Wind **from** direction in degrees (meteorological: 0 = N).

        Returns
        -------
        np.ndarray
            2-D speed-up ratio map (1.0 = no change).
        """
        if self.dem_smooth is None or self.slope_map is None:
            return np.ones_like(self.dem)

        dem_filled = np.nan_to_num(self.dem_smooth, nan=0.0)
        ny, nx = dem_filled.shape
        cell = self._cell_size_m

        # Convert meteorological "from" direction to raster coordinate
        # direction vector (row increases southward).
        wind_rad = math.radians(270.0 - wind_direction_deg)

        # ----------------------------------------------------------
        # 1. Effective hill height H
        # ----------------------------------------------------------
        # Shift DEM upwind to find the upstream minimum over a fetch
        # window.  The hill height is the local elevation minus this
        # upstream minimum.
        hill_fetch_cells = self.config.search_radius_cells

        upwind_dr = int(round(hill_fetch_cells * math.cos(wind_rad)))
        upwind_dc = int(round(hill_fetch_cells * math.sin(wind_rad)))

        # Upstream minimum via shifted minimum_filter
        dem_shifted = np.roll(
            np.roll(dem_filled, upwind_dr, axis=0), upwind_dc, axis=1
        )
        upstream_min = minimum_filter(dem_shifted, size=max(3, hill_fetch_cells))

        H = np.maximum(dem_filled - upstream_min, 0.0)

        # ----------------------------------------------------------
        # 2. Hill half-length L
        # ----------------------------------------------------------
        # Estimate L from the distance over which the terrain rises
        # from its upstream mean to the local elevation.  Use the
        # standard deviation of elevation within the fetch window as
        # a proxy for hill spatial scale.
        local_std = uniform_filter(dem_filled, size=hill_fetch_cells)
        local_mean = uniform_filter(dem_filled, size=hill_fetch_cells)
        L = np.maximum(hill_fetch_cells * cell * 0.3, cell * 5.0)

        # ----------------------------------------------------------
        # 3. Bessel function factor B(σ)
        # ----------------------------------------------------------
        # σ = ln(L / z₀)  — use the reference roughness as the
        # representative surface roughness.
        z0 = max(self.config.roughness_class, 1e-4)
        sigma = np.log(L / z0)
        B_sigma = self._bessel_factor(sigma)

        # ----------------------------------------------------------
        # 4. Fractional speed-up  Δs/s₀
        # ----------------------------------------------------------
        delta_s = 2.0 * (H / L) * B_sigma

        # Separate upwind (positive) from downwind (negative)
        # using the smoothed along-wind gradient.
        dzdy_grid, dzdx_grid = np.gradient(dem_filled, cell)
        along_wind_grad = dzdy_grid * math.cos(wind_rad) + dzdx_grid * math.sin(wind_rad)
        grad_smooth = gaussian_filter(along_wind_grad, sigma=5)

        speedup = 1.0 + delta_s * np.sign(grad_smooth)

        # ----------------------------------------------------------
        # 5. Flow separation on steep lee slopes
        # ----------------------------------------------------------
        if self.aspect_map is not None:
            lee_angle = (wind_direction_deg + 180.0) % 360.0
            aspect_diff = np.abs(self.aspect_map - lee_angle)
            aspect_diff = np.minimum(aspect_diff, 360.0 - aspect_diff)

            # Separation criterion: lee-facing slope > ~15°
            sep_mask = (aspect_diff < 45.0) & (self.slope_map > 15.0)
            speedup[sep_mask] = np.minimum(
                speedup[sep_mask],
                1.0 - 0.3 * (self.slope_map[sep_mask] / 30.0),
            )

        # Clip to physically reasonable range
        speedup = np.clip(speedup, 0.5, 1.5)

        # Re-apply NaN mask from original DEM
        speedup[np.isnan(self.dem)] = np.nan

        return speedup

    # ==========================================================
    # 2. IBL Growth Model
    # ==========================================================

    @staticmethod
    def _ibl_height(x_m: float, z0_new: float, z0_old: float) -> float:
        """
        Internal Boundary Layer height for neutral stability.

            δ(x) = 0.28 · z₀_new^0.45 · x^0.8 / z₀_old^0.18

        Parameters
        ----------
        x_m : float
            Fetch distance over the **new** surface (metres).
        z0_new : float
            Roughness of the new (downstream) surface (m).
        z0_old : float
            Roughness of the old (upstream) surface (m).

        Returns
        -------
        float
            IBL height δ in metres.
        """
        z0_new = max(z0_new, 1e-6)
        z0_old = max(z0_old, 1e-6)
        x_m = max(x_m, 0.1)
        return 0.28 * (z0_new ** 0.45) * (x_m ** 0.8) / (z0_old ** 0.18)

    # ==========================================================
    # 3. EWA Effective Roughness
    # ==========================================================

    def _compute_roughness_effects(self) -> Dict:
        """
        Compute roughness-induced wind speed modifications using
        the Effective Weighing Area (EWA) method.

        Steps per sector:
        1. Compute per-cell effective roughness z₀_eff by tracing
           upstream and applying EWA weighting.
        2. Compute IBL speed-up maps from roughness transitions.
        3. Derive scalar effective roughness and log-law ratio.

        Returns
        -------
        dict
            ``effective_roughness``, ``roughness_ratio``,
            ``ibl_speedup_maps``.
        """
        if self.roughness_map is None:
            self._report_status("WARNING: No roughness map loaded. Using default z₀.")
            return {}

        self._report_status("Computing roughness effects (EWA method)…")

        effective_roughness: Dict[int, float] = {}
        roughness_ratio: Dict[int, float] = {}
        ibl_speedup_maps: Dict[int, np.ndarray] = {}

        for s in range(self.config.n_sectors):
            direction = s * self.config.sector_width_deg
            self._report_status(f"  Sector {s}: wind from {direction:.0f}°")

            # Per-cell EWA effective roughness
            z0_eff_map = self._compute_ewa_effective_roughness(direction)
            effective_roughness[s] = float(np.nanmean(z0_eff_map))

            # Log-law roughness ratio (scalar, map-averaged)
            z0_eff = effective_roughness[s]
            z0_ref = self.config.roughness_class
            z = self.config.reference_height
            if z0_eff > 0 and z0_ref > 0:
                ratio = math.log(z / z0_eff) / math.log(z / z0_ref)
                ratio = max(0.7, min(ratio, 1.3))
            else:
                ratio = 1.0
            roughness_ratio[s] = ratio

            # Per-cell IBL speed-up / slow-down map
            ibl_speedup_maps[s] = self._compute_ibl_speedup_map(direction, z0_eff_map)

        self._roughness_results = {
            "effective_roughness": effective_roughness,
            "roughness_ratio": roughness_ratio,
            "ibl_speedup_maps": ibl_speedup_maps,
        }

        self._report_status("Roughness effects computed.")
        return self._roughness_results

    def _compute_ewa_effective_roughness(
        self, wind_direction_deg: float
    ) -> np.ndarray:
        """
        Compute per-cell effective roughness z₀_eff using the
        Effective Weighing Area (EWA) method.

        For each cell the model:
        1. Traces upstream along the wind direction.
        2. Weights nearby roughness values using:
           - Gaussian cross-wind weighting
           - Distance-weighted upstream weighting
        3. Computes:

           z₀_eff = ( ∫z₀^p · w(x) dx / ∫w(x) dx )^(1/p)

           where p ≈ 0.2 for neutral stability.

        Parameters
        ----------
        wind_direction_deg : float
            Wind FROM direction (meteorological degrees).

        Returns
        -------
        np.ndarray
            2-D effective roughness map (metres).
        """
        if self.roughness_map is None:
            return np.full_like(
                self.dem, self.config.roughness_class
            ) if self.dem is not None else np.array([[self.config.roughness_class]])

        ny, nx = self.roughness_map.shape
        wind_rad = math.radians(270.0 - wind_direction_deg)
        dr = math.cos(wind_rad)
        dc = math.sin(wind_rad)

        # EWA exponent for neutral stability
        p = 0.2
        max_fetch = self.config.search_radius_cells

        # Cross-wind Gaussian width (cells)
        cross_sigma = max(3, int(max_fetch * 0.3))

        # Pre-build relative coordinate grid for the cross-wind
        # Gaussian kernel within the fetch range
        rel_range = np.arange(1, max_fetch + 1, dtype=np.float64)
        rel_cross = np.arange(-cross_sigma * 2, cross_sigma * 2 + 1, dtype=np.float64)

        # Cross-wind weights (Gaussian)
        cross_w = np.exp(-0.5 * (rel_cross / cross_sigma) ** 2)

        # Upstream distance weights (1/r decay)
        upstream_w = 1.0 / rel_range

        self._report_status(
            f"    EWA trace for sector {wind_direction_deg:.0f}° "
            f"(fetch={max_fetch} cells)…"
        )

        # Result array – initialise with the local roughness
        z0_eff = self.roughness_map.copy()

        # Vectorised approach: for each upstream distance, shift the
        # roughness map and accumulate weighted values.
        num_z0 = np.zeros((ny, nx), dtype=np.float64)
        den_z0 = np.zeros((ny, nx), dtype=np.float64)

        for step_idx, step in enumerate(range(1, max_fetch + 1)):
            # Upstream cell offset
            ur = int(round(step * dr))
            uc = int(round(step * dc))

            # Shift roughness map upwind (roll with edge handling)
            shifted = np.roll(
                np.roll(self.roughness_map, ur, axis=0), uc, axis=1
            )

            # Weight for this upstream distance
            dist_weight = upstream_w[step_idx]

            # Apply cross-wind Gaussian by shifting perpendicular to
            # the wind direction and accumulating.
            # Perpendicular direction: (-sin(wind_rad), cos(wind_rad))
            # in (row, col) coordinates.
            perp_dr = -math.sin(wind_rad)
            perp_dc = math.cos(wind_rad)

            perp_shifted = np.zeros_like(shifted)
            perp_weight_sum = np.zeros_like(shifted)

            for ci, c_off in enumerate(rel_cross):
                pr = int(round(c_off * perp_dr))
                pc = int(round(c_off * perp_dc))
                cw = cross_w[ci]

                cs = np.roll(np.roll(shifted, pr, axis=0), pc, axis=1)
                perp_shifted += cs * cw
                perp_weight_sum += cw

            perp_shifted /= np.maximum(perp_weight_sum, 1e-30)

            # Raise to power p and accumulate
            w = dist_weight
            num_z0 += (np.maximum(perp_shifted, self.config.min_roughness) ** p) * w
            den_z0 += w

            if (step_idx + 1) % 10 == 0:
                self._report_progress(
                    step_idx + 1, max_fetch,
                    f"    EWA step {step_idx + 1}/{max_fetch}"
                )

        # EWA effective roughness
        valid = den_z0 > 0
        z0_eff[valid] = (num_z0[valid] / den_z0[valid]) ** (1.0 / p)

        # Clamp
        z0_eff = np.clip(z0_eff, self.config.min_roughness, self.config.max_roughness)

        return z0_eff

    def _compute_ibl_speedup_map(
        self,
        wind_direction_deg: float,
        z0_eff_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute per-cell IBL speed-up ratios for roughness changes
        using the internal boundary layer growth model.

        For each cell the model:
        1. Traces upstream to find the nearest significant roughness
           change.
        2. Computes IBL height at the cell using the neutral-stability
           growth law: δ = 0.28·z₀_new^0.45·x^0.8 / z₀_old^0.18.
        3. If δ < z (hub height), the IBL has not yet fully developed
           and the local roughness z₀_new controls. Otherwise, the
           upstream roughness still contributes.
        4. Converts to a speed-up ratio via the log-law profile.

        Parameters
        ----------
        wind_direction_deg : float
            Wind FROM direction (meteorological degrees).
        z0_eff_map : np.ndarray, optional
            Pre-computed EWA effective roughness.  If None, a simple
            upstream trace is used.

        Returns
        -------
        np.ndarray
            2-D array of IBL speed-up ratios (1.0 = no change).
        """
        if self.roughness_map is None:
            return (
                np.ones_like(self.dem)
                if self.dem is not None
                else np.array([[1.0]])
            )

        ny, nx = self.roughness_map.shape
        ibl_speedup = np.ones((ny, nx), dtype=np.float64)

        wind_rad = math.radians(270.0 - wind_direction_deg)
        dr = math.cos(wind_rad)
        dc = math.sin(wind_rad)

        z0_ref = self.config.roughness_class
        z_target = self.config.reference_height
        cell_m = self._cell_size_m
        max_fetch = self.config.search_radius_cells
        roughness_change_factor = 2.0  # factor-of-2 threshold

        for r in range(ny):
            for c in range(nx):
                z0_local = self.roughness_map[r, c]

                # Trace upstream to find nearest significant roughness change
                prev_z0 = z0_ref
                change_dist = 0  # distance from roughness change to this cell
                prev_extent = 0  # how far the previous roughness extends
                found_change = False

                for step in range(1, max_fetch + 1):
                    ur = int(round(r + step * dr))
                    uc = int(round(c + step * dc))

                    if ur < 0 or ur >= ny or uc < 0 or uc >= nx:
                        break

                    z0_up = self.roughness_map[ur, uc]

                    if (
                        z0_up > z0_local * roughness_change_factor
                        or z0_up < z0_local / roughness_change_factor
                    ):
                        # Found an upstream roughness change
                        prev_z0 = z0_up
                        change_dist = step
                        found_change = True

                        # Continue to find where the upstream roughness
                        # region starts (another change)
                        for step2 in range(step + 1, min(step + max_fetch, max_fetch + 1)):
                            ur2 = int(round(r + step2 * dr))
                            uc2 = int(round(c + step2 * dc))
                            if ur2 < 0 or ur2 >= ny or uc2 < 0 or uc2 >= nx:
                                break
                            z0_up2 = self.roughness_map[ur2, uc2]
                            if (
                                z0_up2 > prev_z0 * roughness_change_factor
                                or z0_up2 < prev_z0 / roughness_change_factor
                            ):
                                break
                            prev_extent = step2 - step
                        break

                if found_change and change_dist > 0:
                    # Distance over new surface (metres)
                    x_new = float(change_dist) * cell_m

                    # IBL height at this cell
                    delta_ibl = self._ibl_height(x_new, z0_local, prev_z0)

                    # If IBL has grown above hub height, use the
                    # upstream roughness; otherwise, blend.
                    if delta_ibl >= z_target:
                        # Fully adjusted to local roughness
                        z0_eff = z0_local
                    else:
                        # Blend: the wind profile below δ follows z0_local,
                        # above δ follows prev_z0.  For simplicity, use the
                        # EWA-like power-law blend weighted by (δ/z).
                        blend = min(delta_ibl / z_target, 1.0)
                        # Weighted effective roughness
                        z0_eff = (
                            z0_local ** blend * prev_z0 ** (1.0 - blend)
                        )

                    z0_eff = float(np.clip(
                        z0_eff, self.config.min_roughness, self.config.max_roughness
                    ))

                    # Speed-up ratio from log-law
                    if z0_eff > 0 and z0_ref > 0:
                        ratio = math.log(z_target / z0_eff) / math.log(
                            z_target / z0_ref
                        )
                        ratio = max(0.7, min(ratio, 1.3))
                        ibl_speedup[r, c] = ratio

        # Apply NaN mask if DEM is loaded
        if self.dem is not None:
            ibl_speedup[np.isnan(self.dem)] = np.nan

        return ibl_speedup

    # ==========================================================
    # 4. Full Flow Model
    # ==========================================================

    def run_flow_model(
        self,
        positions: List[WTGPosition],
        wind_data: Dict,
    ) -> Dict:
        """
        Run the complete wind flow model for turbine positions.

        For each turbine, per sector:

            v_corrected = v_ref × Δs_terrain × Δs_roughness

        where Δs_terrain is from Jackson-Hunt and Δs_roughness from
        the IBL/EWA model.

        Parameters
        ----------
        positions : list of WTGPosition
            Turbine positions.
        wind_data : dict
            Reference wind resource data.

        Returns
        -------
        dict
            Turbine corrections, terrain speed-up maps,
            roughness ratios.
        """
        self._report_status("Starting wind flow modelling…")

        # Step 1: Terrain effects (Jackson-Hunt)
        terrain_results: Dict = {}
        if self.config.use_terrain_correction and self.dem is not None:
            terrain_results = self._compute_terrain_effects()

        # Step 2: Roughness effects (EWA / IBL)
        roughness_results: Dict = {}
        if self.config.use_roughness_correction and self.roughness_map is not None:
            roughness_results = self._compute_roughness_effects()

        # Step 3: Combine at each turbine position
        self._report_status("Applying flow corrections to turbine positions…")
        n_sectors = self.config.n_sectors

        results: Dict = {
            "turbine_corrections": [],
            "terrain_speedup": terrain_results.get("speedup_maps", {}),
            "roughness_ratios": roughness_results.get("roughness_ratio", {}),
        }

        for i, pos in enumerate(positions):
            corrections: Dict = {
                "name": pos.name,
                "terrain_speedup_per_sector": [],
                "roughness_ratio_per_sector": [],
                "combined_factor_per_sector": [],
            }

            for s in range(n_sectors):
                # Terrain speed-up sampled at turbine position
                terrain_su = 1.0
                speedup_maps = terrain_results.get("speedup_maps", {})
                if speedup_maps:
                    smap = speedup_maps.get(s)
                    if smap is not None:
                        terrain_su = self._sample_at_position(
                            pos, smap, self.dem_transform
                        )
                        if np.isnan(terrain_su):
                            terrain_su = 1.0

                # Roughness ratio (from IBL/EWA)
                roughness_ratio = 1.0
                rrat = roughness_results.get("roughness_ratio", {})
                if rrat:
                    roughness_ratio = rrat.get(s, 1.0)

                # Combined: v_corrected = v_ref × Δs_terrain × Δs_roughness
                combined = terrain_su * roughness_ratio
                combined = max(0.5, min(combined, 1.5))

                corrections["terrain_speedup_per_sector"].append(float(terrain_su))
                corrections["roughness_ratio_per_sector"].append(float(roughness_ratio))
                corrections["combined_factor_per_sector"].append(float(combined))

            results["turbine_corrections"].append(corrections)

            self._report_progress(
                i + 1, len(positions), f"Modelled flow at {pos.name}"
            )

        self._report_status("Wind flow modelling complete.")
        return results

    # ----------------------------------------------------------
    # Sampling helper
    # ----------------------------------------------------------

    def _sample_at_position(
        self,
        pos: WTGPosition,
        data_map: np.ndarray,
        transform,
    ) -> float:
        """
        Sample a 2-D raster value at a turbine position (lat/lon).

        Uses the affine transform inverse to convert geographic
        coordinates to pixel indices, then nearest-neighbour lookup.
        """
        try:
            inv_transform = ~transform
            col, row = inv_transform * (pos.lon, pos.lat)
            row_int = int(round(row))
            col_int = int(round(col))

            if (
                0 <= row_int < data_map.shape[0]
                and 0 <= col_int < data_map.shape[1]
            ):
                return float(data_map[row_int, col_int])
            return np.nan
        except Exception:
            return np.nan

    # ==========================================================
    # Turbulence Intensity
    # ==========================================================

    def estimate_turbulence_intensity(
        self,
        positions: List[WTGPosition],
    ) -> List[float]:
        """
        Estimate ambient turbulence intensity at each turbine position.

        TI is estimated from:
        1. Surface roughness:  TI_roughness ∝ 1 / ln(z/z₀)
        2. Terrain complexity: slope contribution
        3. Proximity to roughness changes: additional TI from IBL
           transitions.

        Returns
        -------
        list of float
            TI values (0–1) at each position.
        """
        ti_values: List[float] = []

        for pos in positions:
            # Base TI from roughness (Frandsen / WAsP formulation)
            z0 = pos.roughness if pos.roughness > 0 else 0.03
            z = self.config.reference_height
            ln_ratio = math.log(z / z0)
            if ln_ratio > 0:
                ti_roughness = 1.0 / ln_ratio
            else:
                ti_roughness = 0.1
            ti_roughness = min(ti_roughness, 0.25)

            # Terrain-induced TI
            ti_terrain = 0.0
            if self.slope_map is not None and self.dem_transform is not None:
                slope_val = self._sample_at_position(
                    pos, self.slope_map, self.dem_transform
                )
                if not np.isnan(slope_val):
                    # WASP-style: ~1 % TI per degree of slope
                    ti_terrain = slope_val / 100.0

            # Combined (RMS sum)
            ti = math.sqrt(ti_roughness ** 2 + ti_terrain ** 2)
            ti = max(0.05, min(ti, 0.30))

            ti_values.append(ti)

        return ti_values

    # ==========================================================
    # 5. Speed-Up Map Creation & GeoTIFF Export
    # ==========================================================

    def create_combined_speedup_map(self, wind_data: Dict = None) -> Dict:
        """
        Create a georeferenced combined speed-up map by weighting
        terrain and roughness speed-up per sector with wind rose
        frequencies.

        The combined map is:
            SU_combined = Σ_f  f_s × SU_terrain,s × SU_roughness,s

        Returns
        -------
        dict
            ``sector_maps``, ``combined_map``, ``transform``, ``crs``.
        """
        if self.dem is None or self.dem_transform is None:
            self._report_status("ERROR: DEM not loaded.")
            return {}

        self._report_status("Creating combined speed-up map…")

        # Terrain speed-up maps (Jackson-Hunt)
        terrain_results = self._compute_terrain_effects()
        speedup_maps = terrain_results.get("speedup_maps", {})

        # Roughness IBL speed-up maps (EWA)
        ibl_maps: Dict[int, np.ndarray] = {}
        if self.roughness_map is not None:
            roughness_results = self._compute_roughness_effects()
            ibl_maps = roughness_results.get("ibl_speedup_maps", {})

        # Sector frequencies from wind data
        sector_freqs = self._extract_sector_frequencies(wind_data)

        # Build frequency-weighted combined map
        ny, nx = self.dem.shape
        combined_map = np.zeros((ny, nx), dtype=np.float64)
        weight_sum = np.zeros((ny, nx), dtype=np.float64)
        sector_speedup_products: Dict[int, np.ndarray] = {}

        for s in range(self.config.n_sectors):
            terrain_su = speedup_maps.get(s)
            ibl_su = ibl_maps.get(s)
            freq = sector_freqs.get(s, 1.0 / self.config.n_sectors)

            product = None
            if terrain_su is not None:
                product = terrain_su.copy()
                if ibl_su is not None:
                    valid_ibl = ~np.isnan(ibl_su)
                    product[valid_ibl] *= ibl_su[valid_ibl]
                sector_speedup_products[s] = product
            elif ibl_su is not None:
                product = ibl_su.copy()
                sector_speedup_products[s] = product

            if product is not None:
                valid = ~np.isnan(product)
                combined_map[valid] += product[valid] * freq
                weight_sum[valid] += freq

        # Normalise
        valid_weights = weight_sum > 0
        combined_map[valid_weights] /= weight_sum[valid_weights]
        combined_map[~valid_weights] = 1.0
        combined_map[np.isnan(self.dem)] = np.nan

        self._report_status("Combined speed-up map created.")

        return {
            "sector_maps": sector_speedup_products,
            "combined_map": combined_map,
            "transform": self.dem_transform,
            "crs": "EPSG:4326",
        }

    def _extract_sector_frequencies(self, wind_data: Dict) -> Dict[int, float]:
        """Extract normalised sector frequencies from wind data."""
        sector_freqs: Dict[int, float] = {}
        if wind_data:
            points = wind_data.get("points", [])
            if points and "sectors" in points[0]:
                for sec in points[0]["sectors"]:
                    idx = sec.get("direction", 0) // int(self.config.sector_width_deg)
                    sector_freqs[idx % self.config.n_sectors] = sec.get(
                        "frequency", 1.0 / self.config.n_sectors
                    )

        # Default uniform
        total = 0.0
        for s in range(self.config.n_sectors):
            sector_freqs.setdefault(s, 1.0 / self.config.n_sectors)
            total += sector_freqs[s]
        for s in sector_freqs:
            sector_freqs[s] /= total

        return sector_freqs

    def create_speedup_geotiff(
        self, output_path: str, sector: int = None
    ) -> Optional[str]:
        """
        Save the speed-up map(s) as a GeoTIFF with EPSG:4326 CRS.

        Parameters
        ----------
        output_path : str
            Output file path.  If *sector* is None, saves the
            frequency-weighted combined map.
        sector : int, optional
            Sector index (0–11).  Appends ``_sector_XXX`` suffix.

        Returns
        -------
        str or None
            Path to saved file, or None on failure.
        """
        try:
            import rasterio
            from rasterio.crs import CRS
        except ImportError:
            self._report_status("ERROR: rasterio required for GeoTIFF export.")
            return None

        if self.dem is None or self.dem_transform is None:
            self._report_status("ERROR: DEM not loaded.")
            return None

        self._report_status("Computing speed-up for GeoTIFF export…")
        speedup_data = self.create_combined_speedup_map()

        if not speedup_data:
            return None

        if sector is not None:
            sector_map = speedup_data["sector_maps"].get(sector)
            if sector_map is None:
                self._report_status(f"No speed-up data for sector {sector}.")
                return None
            data_to_write = sector_map
            suffix = f"_sector_{sector:03d}"
        else:
            data_to_write = speedup_data["combined_map"]
            suffix = ""

        # Build output path
        from pathlib import Path
        p = Path(output_path)
        out_file = str(p.with_name(p.stem + suffix + p.suffix))

        ny, nx = data_to_write.shape
        nodata_val = -9999.0
        data_out = np.where(
            np.isnan(data_to_write), nodata_val, data_to_write
        ).astype(np.float32)

        with rasterio.open(
            out_file,
            "w",
            driver="GTiff",
            height=ny,
            width=nx,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(4326),
            transform=self.dem_transform,
            nodata=nodata_val,
            compress="DEFLATE",
        ) as dst:
            dst.write(data_out, 1)

        self._report_status(f"Speed-up GeoTIFF saved to: {out_file}")
        return out_file
