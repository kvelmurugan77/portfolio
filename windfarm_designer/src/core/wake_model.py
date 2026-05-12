"""
Wind Turbine Wake Model for WindFarm Designer Pro.
Implements multiple wake models for estimating wake losses in wind farms.

Supported models:
1. Jensen (Park) Single Wake Model (1983)
2. Katic Multiple Wake Model (1986) - Superposition of Jensen wakes
3. Frandsen Single Wake Model (2006)
4. Ainslie Eddy Viscosity Model (simplified)

All models compute the velocity deficit downstream of turbines,
which is then used to calculate the effective wind speed at each
turbine position and the resulting wake losses in AEP.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .layout_optimizer import WTGPosition, TurbineModel


# ============================================================
# Wake Model Configuration
# ============================================================

@dataclass
class WakeModelConfig:
    """Configuration for wake model computation."""
    model_type: str = "jensen"       # 'jensen', 'katic', 'frandsen', 'ainslie'
    wake_decay_constant: float = 0.05  # Jensen k (typical: 0.04-0.075 for onshore)
    turbulence_intensity: float = 0.10  # Ambient TI at hub height
    wind_sectors: int = 12            # Number of directional sectors
    sector_width_deg: float = 30.0    # Degrees per sector
    ct_curve: Optional[np.ndarray] = None  # Ct as function of wind speed
    power_curve: Optional[np.ndarray] = None  # Power as function of wind speed
    wind_speeds: Optional[np.ndarray] = None  # Wind speed bins


# ============================================================
# Jensen (Park) Single Wake Model
# ============================================================

def jensen_wake_deficit(
    x_downwind: float,
    y_crosswind: float,
    rotor_diameter: float,
    ct: float,
    k: float = 0.05,
) -> float:
    """
    Compute the wake velocity deficit using the Jensen/Park model.

    The wake expands linearly downstream:
        r_wake(x) = r_rotor + k * x

    The velocity deficit at a point (x, y) downstream:
        delta_u/u_inf = (1 - sqrt(1 - Ct)) / (1 + k*x/r_rotor)^2

    The deficit is only applied within the wake cone radius.

    Parameters
    ----------
    x_downwind : float
        Downstream distance from the turbine (m).
    y_crosswind : float
        Crosswind (lateral) distance from the wake centerline (m).
    rotor_diameter : float
        Turbine rotor diameter (m).
    ct : float
        Thrust coefficient at the operating wind speed.
    k : float
        Wake decay constant (default 0.05 for onshore).

    Returns
    -------
    float
        Velocity deficit ratio (0 = no deficit, 1 = full wake).
    """
    if ct <= 0 or x_downwind <= 0:
        return 0.0

    r0 = rotor_diameter / 2.0
    r_wake = r0 + k * x_downwind

    # Check if the point is within the wake cone
    if abs(y_crosswind) > r_wake:
        return 0.0

    # Jensen velocity deficit
    a = 0.5 * (1 - math.sqrt(max(0.0, 1 - ct)))  # Axial induction factor
    deficit = (1 - a) / ((1 + k * x_downwind / r0) ** 2)

    # Apply radial profile (Gaussian smoothing within the wake)
    if r_wake > r0:
        sigma = r_wake * 0.5
        radial_factor = math.exp(-0.5 * (y_crosswind / sigma) ** 2)
    else:
        radial_factor = 1.0

    return deficit * radial_factor


# ============================================================
# Frandsen Wake Model
# ============================================================

def frandsen_wake_deficit(
    x_downwind: float,
    y_crosswind: float,
    rotor_diameter: float,
    ct: float,
    ambient_ti: float = 0.10,
) -> float:
    """
    Compute wake deficit using the Frandsen model (2006).

    The Frandsen model uses a different wake expansion formulation
    that accounts for turbulence intensity:
        r_wake = r_rotor * sqrt((1 + a) / (1 - a)) * sqrt(beta)
    where a is the induction factor and beta depends on TI.

    Parameters
    ----------
    x_downwind, y_crosswind : float
        Distances in meters.
    rotor_diameter : float
    ct : float
    ambient_ti : float

    Returns
    -------
    float
        Velocity deficit ratio.
    """
    if ct <= 0 or ct >= 1 or x_downwind <= 0:
        return 0.0

    r0 = rotor_diameter / 2.0
    a = 0.5 * (1 - math.sqrt(max(0.0, 1 - ct)))

    # Wake radius (Frandsen formulation)
    beta = 1.0 / math.sqrt(1.0 + ambient_ti)
    expansion = math.sqrt((1 + a) / max(1 - a, 0.01)) * beta
    r_wake = r0 * expansion * math.sqrt(x_downwind / rotor_diameter + 1)

    if abs(y_crosswind) > r_wake:
        return 0.0

    # Velocity deficit
    deficit_sq = (1.0 - expansion ** 2) ** 2 if expansion < 1.0 else 0.0
    deficit = math.sqrt(max(0.0, deficit_sq))

    return deficit


# ============================================================
# Ainslie Eddy Viscosity Model (Simplified)
# ============================================================

def ainslie_wake_deficit(
    x_downwind: float,
    y_crosswind: float,
    rotor_diameter: float,
    ct: float,
    ambient_ti: float = 0.10,
    u_inf: float = 8.0,
) -> float:
    """
    Compute wake deficit using a simplified Ainslie model.

    Uses a Gaussian wake profile with an eddy viscosity that
    depends on ambient turbulence intensity.

    Parameters
    ----------
    x_downwind, y_crosswind : float
    rotor_diameter : float
    ct : float
    ambient_ti : float
    u_inf : float
        Free-stream wind speed.

    Returns
    -------
    float
        Velocity deficit ratio.
    """
    if ct <= 0 or x_downwind <= 0:
        return 0.0

    r0 = rotor_diameter / 2.0
    a = 0.5 * (1 - math.sqrt(max(0.0, 1 - ct)))

    # Initial wake radius
    r_init = r0 * math.sqrt((1 + a) / 2.0)

    # Wake expansion rate based on TI
    k_star = 0.025 * ambient_ti * (2 * r0 / r_init) ** 2

    # Wake radius at distance x
    r_wake = r_init + k_star * x_downwind

    # Centerline deficit decay
    if x_downwind < 2 * rotor_diameter:
        deficit_center = 2 * a * (r0 / (r_init + k_star * x_downwind)) ** 2
    else:
        deficit_center = a * (r0 / r_wake) ** 2

    # Gaussian radial profile
    radial_factor = math.exp(-0.5 * (y_crosswind / r_wake) ** 2)

    return deficit_center * radial_factor


# ============================================================
# Multiple Wake Superposition
# ============================================================

def superpose_wakes(deficits: List[float], method: str = "katic") -> float:
    """
    Superpose multiple wake deficits at a single point.

    Methods:
    - 'katic' (sum of squares): total_deficit = sqrt(sum(di^2))
    - 'linear': total_deficit = sum(di)
    - 'max': total_deficit = max(di)

    Parameters
    ----------
    deficits : list of float
        Individual wake deficit ratios.
    method : str

    Returns
    -------
    float
        Combined velocity deficit ratio (0 to ~0.8).
    """
    if not deficits:
        return 0.0

    # Filter out zero deficits
    deficits = [d for d in deficits if d > 0]

    if not deficits:
        return 0.0

    if method == "katic":
        # Katic/Frandsen: sum of squares (RSS)
        combined = math.sqrt(sum(d ** 2 for d in deficits))
    elif method == "linear":
        combined = sum(deficits)
    elif method == "max":
        combined = max(deficits)
    else:
        combined = math.sqrt(sum(d ** 2 for d in deficits))

    return min(combined, 0.8)  # Physical limit on wake deficit


# ============================================================
# Full Wake Calculator
# ============================================================

class WakeCalculator:
    """
    Calculates wake effects for an entire wind farm layout.

    For each turbine, computes the cumulative wake deficit from
    all upstream turbines for each wind direction sector, then
    aggregates to produce overall wake loss estimates.
    """

    def __init__(self, config: WakeModelConfig = None):
        self.config = config or WakeModelConfig()

    def compute_wake_losses(
        self,
        positions: List[WTGPosition],
        turbine: TurbineModel,
        wind_data: Dict,
    ) -> Dict:
        """
        Compute wake losses for all turbines and wind sectors.

        Parameters
        ----------
        positions : list of WTGPosition
            Turbine positions in local coordinates.
        turbine : TurbineModel
        wind_data : dict
            Wind resource data including sector frequencies.

        Returns
        -------
        dict
            {
                'turbine_results': list of dict,
                'total_wake_loss_pct': float,
                'sector_wake_losses': list,
            }
        """
        n_turb = len(positions)
        n_sectors = self.config.wind_sectors
        sector_width = self.config.sector_width_deg

        # Initialize sector results
        sector_deficits = np.zeros((n_turb, n_sectors))
        sector_frequencies = np.zeros(n_sectors)

        # Get sector frequencies from wind data
        gwa_points = wind_data.get('points', [])
        if gwa_points and 'sectors' in gwa_points[0]:
            for i, sec in enumerate(gwa_points[0]['sectors'][:n_sectors]):
                sector_frequencies[i] = sec.get('frequency', 1.0 / n_sectors)
        else:
            sector_frequencies[:] = 1.0 / n_sectors

        # Normalize frequencies
        total_freq = sector_frequencies.sum()
        if total_freq > 0:
            sector_frequencies /= total_freq

        # Compute wake effects for each sector
        for s in range(n_sectors):
            wind_dir = s * sector_width  # Sector center direction (deg)
            wind_rad = math.radians(wind_dir)

            for i in range(n_turb):
                deficits = []

                for j in range(n_turb):
                    if i == j:
                        continue

                    # Vector from turbine j to turbine i
                    dx = positions[i].x_local - positions[j].x_local
                    dy = positions[i].y_local - positions[j].y_local
                    dist = math.sqrt(dx ** 2 + dy ** 2)

                    if dist < 1.0:
                        continue

                    # Decompose into downwind and crosswind components
                    downwind = dx * math.sin(wind_rad) + dy * math.cos(wind_rad)
                    crosswind = abs(-dx * math.cos(wind_rad) + dy * math.sin(wind_rad))

                    # Only consider upstream turbines (downwind > 0)
                    if downwind <= 0:
                        continue

                    # Get Ct at sector mean wind speed
                    sector_speed = self._get_sector_speed(s, wind_data)
                    ct = self._get_ct(sector_speed, turbine)

                    # Compute single wake deficit
                    if self.config.model_type == "jensen" or self.config.model_type == "katic":
                        deficit = jensen_wake_deficit(
                            downwind, crosswind,
                            turbine.rotor_diameter, ct,
                            self.config.wake_decay_constant
                        )
                    elif self.config.model_type == "frandsen":
                        deficit = frandsen_wake_deficit(
                            downwind, crosswind,
                            turbine.rotor_diameter, ct,
                            self.config.turbulence_intensity
                        )
                    elif self.config.model_type == "ainslie":
                        deficit = ainslie_wake_deficit(
                            downwind, crosswind,
                            turbine.rotor_diameter, ct,
                            self.config.turbulence_intensity,
                            sector_speed
                        )
                    else:
                        deficit = jensen_wake_deficit(
                            downwind, crosswind,
                            turbine.rotor_diameter, ct,
                            self.config.wake_decay_constant
                        )

                    deficits.append(deficit)

                # Superpose all wakes on turbine i for this sector
                combined_deficit = superpose_wakes(
                    deficits,
                    method="katic" if self.config.model_type in ("jensen", "katic") else "katic"
                )
                sector_deficits[i, s] = combined_deficit

        # Compute weighted average wake deficit per turbine
        avg_deficits = np.zeros(n_turb)
        for i in range(n_turb):
            avg_deficits[i] = np.sum(sector_deficits[i] * sector_frequencies)

        # Build results
        turbine_results = []
        for i, pos in enumerate(positions):
            wake_loss = avg_deficits[i] * 100  # Convert to percentage
            turbine_results.append({
                'name': pos.name,
                'avg_deficit': float(avg_deficits[i]),
                'wake_loss_pct': float(wake_loss),
                'sector_deficits': sector_deficits[i].tolist(),
            })

        # Total wake loss (energy-weighted)
        total_wake_loss = float(np.mean(avg_deficits) * 100)

        return {
            'turbine_results': turbine_results,
            'total_wake_loss_pct': total_wake_loss,
            'sector_wake_losses': sector_deficits.tolist(),
            'sector_frequencies': sector_frequencies.tolist(),
            'n_sectors': n_sectors,
        }

    def _get_sector_speed(self, sector_idx: int, wind_data: Dict) -> float:
        """Get the mean wind speed for a directional sector."""
        gwa_points = wind_data.get('points', [])
        if gwa_points and 'sectors' in gwa_points[0]:
            sectors = gwa_points[0]['sectors']
            if sector_idx < len(sectors):
                speed = sectors[sector_idx].get('mean_speed', 8.0)
                return max(speed, 1.0)
        return 8.0

    def _get_ct(self, wind_speed: float, turbine: TurbineModel) -> float:
        """Get thrust coefficient at a given wind speed."""
        if self.config.ct_curve is not None and self.config.wind_speeds is not None:
            # Interpolate from Ct curve
            ct = float(np.interp(wind_speed, self.config.wind_speeds, self.config.ct_curve))
            return max(0.0, min(ct, 0.99))

        # Default Ct estimation from turbine specs
        ci = turbine.cut_in_ms
        co = turbine.cut_out_ms
        vr = turbine.rated_speed_ms

        if wind_speed < ci or wind_speed > co:
            return 0.0
        elif wind_speed <= vr:
            frac = (wind_speed - ci) / (vr - ci)
            return max(0.0, min(4.0 * frac * (1 - frac) + 0.1, 0.9))
        else:
            return 0.05  # Low Ct at rated and above
