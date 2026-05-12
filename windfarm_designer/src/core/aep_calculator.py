"""
AEP (Annual Energy Production) Calculator for WindFarm Designer Pro.

Computes the expected annual energy production for each turbine position
and for the entire wind farm, accounting for:

1. Wind resource (Weibull distribution per sector from GWA or mast data)
2. Power curve (turbine-specific or default)
3. Air density correction
4. Roughness/terrain effects on wind profile
5. Wake losses from upstream turbines
6. Availability and electrical losses (user-configurable)

Output metrics:
- Gross AEP per turbine (no wake losses)
- Net AEP per turbine (with wake losses)
- Wake loss percentage
- Total wind farm AEP
- Capacity factor
- Full load hours
"""

import math
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.special import gamma as gamma_func
from scipy.integrate import trapezoid

from .layout_optimizer import WTGPosition, TurbineModel
from .wake_model import WakeCalculator, WakeModelConfig
from src.utils.data_utils import generate_default_power_curve, weibull_pdf
from src.utils.geo_utils import wind_speed_at_height, air_density_adjustment

logger = logging.getLogger(__name__)


# ============================================================
# Loss Factors Configuration
# ============================================================

@dataclass
class LossFactors:
    """User-configurable loss factors applied to AEP."""
    availability_pct: float = 97.0        # Turbine availability
    electrical_loss_pct: float = 2.0      # Electrical transmission losses
    curtailment_pct: float = 1.0          # Regulatory curtailment
    environmental_pct: float = 0.5        # Environmental (bat/bird) shutdown
    icing_pct: float = 0.5                # Icing losses
    other_losses_pct: float = 1.0         # Other miscellaneous losses
    wake_losses_included: bool = True     # Wake losses computed separately

    def total_loss_factor(self) -> float:
        """Calculate combined loss factor (multiplied with net AEP)."""
        factors = [
            self.availability_pct / 100.0,
            1.0 - self.electrical_loss_pct / 100.0,
            1.0 - self.curtailment_pct / 100.0,
            1.0 - self.environmental_pct / 100.0,
            1.0 - self.icing_pct / 100.0,
            1.0 - self.other_losses_pct / 100.0,
        ]
        return float(np.prod(factors))


# ============================================================
# AEP Calculator
# ============================================================

class AEPCalculator:
    """
    Computes Annual Energy Production for a wind farm.

    The calculation uses the following approach:

    1. For each turbine, obtain the wind resource at its position
       (Weibull parameters A, k for each directional sector, plus
       sector frequencies).

    2. Apply terrain and roughness corrections to the wind speed
       at hub height.

    3. For each wind speed bin, compute:
       - Probability of occurrence (Weibull PDF)
       - Power output from the power curve
       - Energy = P * probability * hours_in_year

    4. Sum over all speed bins and sectors to get Gross AEP.

    5. Apply wake deficits (from WakeCalculator) to get Net AEP.

    6. Apply additional loss factors.

    Parameters
    ----------
    turbine : TurbineModel
    power_curve : dict
        Power curve with 'wind_speeds', 'power', 'ct' arrays.
    loss_factors : LossFactors
    wake_config : WakeModelConfig
    """

    def __init__(
        self,
        turbine: TurbineModel,
        power_curve: Dict = None,
        loss_factors: LossFactors = None,
        wake_config: WakeModelConfig = None,
    ):
        self.turbine = turbine
        self.power_curve = power_curve or generate_default_power_curve(turbine.__dict__)
        self.loss_factors = loss_factors or LossFactors()
        self.wake_config = wake_config or WakeModelConfig()

        # Pass power curve data to wake config
        self.wake_config.wind_speeds = self.power_curve['wind_speeds']
        self.wake_config.power_curve = self.power_curve['power']
        self.wake_config.ct_curve = self.power_curve['ct']

        # Standard air density
        self.air_density = 1.225  # kg/m^3

        # Hours per year
        self.hours_per_year = 8760.0

    def compute_farm_aep(
        self,
        positions: List[WTGPosition],
        wind_data: Dict,
        include_wakes: bool = True,
    ) -> Dict:
        """
        Compute AEP for the entire wind farm.

        Parameters
        ----------
        positions : list of WTGPosition
        wind_data : dict
            GWA wind resource data with points, sectors, etc.
        include_wakes : bool
            Whether to include wake losses.

        Returns
        -------
        dict
            Comprehensive AEP results.
        """
        wake_str = "with" if include_wakes else "without"
        self._report_status(
            f"Computing AEP for {len(positions)} turbines "
            f"({wake_str} wakes)..."
        )
        n_turb = len(positions)

        # Step 1: Get wind resource at each turbine position
        self._report_status("Step 1: Interpolating wind resource to turbine positions...")
        turbine_wind = self._interpolate_wind_to_turbines(positions, wind_data)

        # Step 2: Apply terrain corrections
        self._report_status("Step 2: Applying terrain and roughness corrections...")
        self._apply_terrain_corrections(positions, turbine_wind)

        # Step 3: Compute gross AEP per turbine
        self._report_status("Step 3: Computing gross AEP per turbine...")
        for i, pos in enumerate(positions):
            wind = turbine_wind[i]
            gross_aep = self._compute_single_turbine_aep(wind)
            pos.gross_aep_gwh = gross_aep / 1e6  # Wh to GWh
            pos.mean_wind_speed_ms = wind.get('mean_speed', 0.0)

        # Step 4: Compute wake losses
        wake_results = None
        if include_wakes:
            self._report_status("Step 4: Computing wake losses...")
            wake_calc = WakeCalculator(self.wake_config)
            wake_results = wake_calc.compute_wake_losses(positions, self.turbine, wind_data)

            # Apply wake losses to each turbine
            for i, pos in enumerate(positions):
                if i < len(wake_results['turbine_results']):
                    wake_pct = wake_results['turbine_results'][i]['wake_loss_pct']
                    pos.wake_loss_pct = wake_pct
                    pos.net_aep_gwh = pos.gross_aep_gwh * (1.0 - wake_pct / 100.0)
                else:
                    pos.net_aep_gwh = pos.gross_aep_gwh
        else:
            for pos in positions:
                pos.wake_loss_pct = 0.0
                pos.net_aep_gwh = pos.gross_aep_gwh

        # Step 5: Apply additional loss factors
        total_loss_factor = self.loss_factors.total_loss_factor()
        for pos in positions:
            pos.aep_gwh = pos.net_aep_gwh * total_loss_factor

        # Step 6: Aggregate farm-level results
        total_gross_aep = sum(p.gross_aep_gwh for p in positions)
        total_net_aep = sum(p.net_aep_gwh for p in positions)
        total_aep = sum(p.aep_gwh for p in positions)
        installed_capacity_mw = len(positions) * self.turbine.rated_power_kw / 1000.0

        if installed_capacity_mw > 0:
            capacity_factor = (total_aep * 1e6) / (installed_capacity_mw * 1e6 * self.hours_per_year) * 100
            full_load_hours = capacity_factor / 100.0 * self.hours_per_year
        else:
            capacity_factor = 0.0
            full_load_hours = 0.0

        total_wake_loss_pct = wake_results['total_wake_loss_pct'] if wake_results else 0.0

        # Build per-turbine results
        wtg_results = []
        for pos in positions:
            wtg_results.append({
                'name': pos.name,
                'latitude': pos.lat,
                'longitude': pos.lon,
                'elevation_m': pos.elevation,
                'roughness_z0': pos.roughness,
                'hub_height_m': pos.hub_height_m,
                'mean_wind_speed_ms': pos.mean_wind_speed_ms,
                'gross_aep_gwh': round(pos.gross_aep_gwh, 4),
                'wake_loss_pct': round(pos.wake_loss_pct, 2),
                'net_aep_gwh': round(pos.net_aep_gwh, 4),
                'final_aep_gwh': round(pos.aep_gwh, 4),
                'capacity_factor_pct': round(
                    (pos.aep_gwh * 1e6) / (self.turbine.rated_power_kw * self.hours_per_year) * 100, 2
                ) if self.turbine.rated_power_kw > 0 else 0.0,
            })

        # Sort by name
        wtg_results.sort(key=lambda x: x['name'])

        results = {
            'summary': {
                'n_turbines': len(positions),
                'turbine_model': self.turbine.name,
                'rated_power_kw': self.turbine.rated_power_kw,
                'rotor_diameter_m': self.turbine.rotor_diameter_m,
                'hub_height_m': self.turbine.hub_height_m,
                'installed_capacity_mw': round(installed_capacity_mw, 2),
                'total_gross_aep_gwh': round(total_gross_aep, 4),
                'total_wake_loss_pct': round(total_wake_loss_pct, 2),
                'total_net_aep_gwh': round(total_net_aep, 4),
                'other_losses_factor': round(total_loss_factor, 4),
                'total_aep_gwh': round(total_aep, 4),
                'capacity_factor_pct': round(capacity_factor, 2),
                'full_load_hours': round(full_load_hours, 1),
            },
            'loss_factors': {
                'availability_pct': self.loss_factors.availability_pct,
                'electrical_loss_pct': self.loss_factors.electrical_loss_pct,
                'curtailment_pct': self.loss_factors.curtailment_pct,
                'environmental_pct': self.loss_factors.environmental_pct,
                'icing_pct': self.loss_factors.icing_pct,
                'other_losses_pct': self.loss_factors.other_losses_pct,
                'total_loss_factor_pct': round(total_loss_factor * 100, 2),
            },
            'wtg_results': wtg_results,
            'wake_details': wake_results,
        }

        self._report_status(
            f"AEP Calculation Complete:\n"
            f"  Installed Capacity: {installed_capacity_mw:.1f} MW\n"
            f"  Gross AEP: {total_gross_aep:.2f} GWh/yr\n"
            f"  Wake Loss: {total_wake_loss_pct:.1f}%\n"
            f"  Net AEP: {total_net_aep:.2f} GWh/yr\n"
            f"  Final AEP (with losses): {total_aep:.2f} GWh/yr\n"
            f"  Capacity Factor: {capacity_factor:.1f}%\n"
            f"  Full Load Hours: {full_load_hours:.0f} h/yr"
        )

        return results

    def _interpolate_wind_to_turbines(
        self,
        positions: List[WTGPosition],
        wind_data: Dict,
    ) -> List[Dict]:
        """
        Interpolate GWA wind data to each turbine position.
        """
        gwa_points = wind_data.get('points', [])

        if not gwa_points:
            # Return default wind resource
            return [{
                'mean_speed': 7.5,
                'weibull_A': 8.5,
                'weibull_k': 2.1,
                'sectors': [
                    {'direction': i * 30, 'frequency': 1.0 / 12,
                     'mean_speed': 7.5, 'weibull_A': 8.5, 'weibull_k': 2.1}
                    for i in range(12)
                ]
            } for _ in positions]

        # Simple nearest-neighbor interpolation
        # For production, use IDW or kriging
        turbine_wind = []

        for pos in positions:
            if pos.lat == 0.0 and pos.lon == 0.0:
                # Use local coordinates to estimate lat/lon
                from .layout_optimizer import local_to_latlon
                latlon = local_to_latlon(
                    [(pos.x_local, pos.y_local)],
                    getattr(self, '_origin', (0, 0))
                )
                lat, lon = latlon[0]
            else:
                lat, lon = pos.lat, pos.lon

            # Find nearest GWA point
            min_dist = float('inf')
            nearest = gwa_points[0]

            for p in gwa_points:
                dlat = lat - p['lat']
                dlon = lon - p['lon']
                d = math.sqrt(dlat ** 2 + dlon ** 2)
                if d < min_dist:
                    min_dist = d
                    nearest = p

            turbine_wind.append({
                'mean_speed': nearest.get('mean_wind_speed', 7.5),
                'weibull_A': nearest.get('mean_weibull_A', 8.5),
                'weibull_k': nearest.get('mean_weibull_k', 2.1),
                'power_density': nearest.get('mean_power_density', 400),
                'sectors': nearest.get('sectors', []),
            })

        return turbine_wind

    def _apply_terrain_corrections(
        self,
        positions: List[WTGPosition],
        turbine_wind: List[Dict],
    ):
        """
        Apply terrain elevation and roughness corrections to wind speed.

        Uses the power law to adjust from reference height to hub height:
            v(hub) = v(ref) * (hub/ref) ^ alpha
        where alpha depends on roughness and stability.
        """
        for i, pos in enumerate(positions):
            z0 = pos.roughness if pos.roughness > 0 else 0.03
            hub_h = pos.hub_height_m

            # Reference height from GWA data (typically 100m)
            ref_h = 100.0
            ref_speed = turbine_wind[i]['mean_speed']

            if hub_h != ref_h and z0 > 0:
                # Power law exponent
                alpha = 1.0 / math.log(ref_h / z0)
                corrected_speed = ref_speed * (hub_h / ref_h) ** alpha
                turbine_wind[i]['mean_speed'] = corrected_speed

            # Apply air density correction
            # (simplified: density decreases with elevation)
            if pos.elevation > 0:
                # Barometric formula approximation
                rho = self.air_density * math.exp(-pos.elevation / 8500.0)
                speed_correction = math.sqrt(rho / self.air_density)
                # This affects power, not wind speed directly
                turbine_wind[i]['air_density'] = rho
            else:
                turbine_wind[i]['air_density'] = self.air_density

    def _compute_single_turbine_aep(self, wind: Dict) -> float:
        """
        Compute AEP for a single turbine using sector-based Weibull distribution.

        AEP = sum over sectors: [freq_s * sum over speed bins: [P(v) * f_weibull(v) * 8760]]

        Parameters
        ----------
        wind : dict
            Wind resource data for this turbine position.

        Returns
        -------
        float
            AEP in Watt-hours (Wh).
        """
        sectors = wind.get('sectors', [])
        aep_wh = 0.0

        if sectors:
            # Sector-based calculation
            for sector in sectors:
                freq = sector.get('frequency', 0.0)
                A = sector.get('weibull_A', wind.get('weibull_A', 8.5))
                k = sector.get('weibull_k', wind.get('weibull_k', 2.1))

                # Integrate power over Weibull distribution
                sector_aep = self._integrate_power_weibull(A, k)
                aep_wh += freq * sector_aep
        else:
            # Omnidirectional calculation
            A = wind.get('weibull_A', 8.5)
            k = wind.get('weibull_k', 2.1)
            aep_wh = self._integrate_power_weibull(A, k)

        return aep_wh

    def _integrate_power_weibull(self, A: float, k: float) -> float:
        """
        Numerically integrate P(v) * f_weibull(v) dv over all wind speeds.

        Uses the trapezoidal rule with fine resolution (0.25 m/s steps).

        Parameters
        ----------
        A : float
            Weibull scale parameter (m/s).
        k : float
            Weibull shape parameter.

        Returns
        -------
        float
            Energy in Watt-hours (Wh) per year for this sector.
        """
        # Wind speed range
        v_min = 0.0
        v_max = self.power_curve['cut_out'] + 2.0
        dv = 0.25
        speeds = np.arange(v_min, v_max + dv, dv)

        # Weibull PDF at each speed
        pdf_values = np.zeros_like(speeds)
        for i, v in enumerate(speeds):
            if v > 0 and A > 0 and k > 0:
                pdf_values[i] = (k / A) * (v / A) ** (k - 1) * math.exp(-(v / A) ** k)

        # Power at each speed (from power curve)
        power_values = np.interp(
            speeds,
            self.power_curve['wind_speeds'],
            self.power_curve['power']
        )

        # Energy = integral of P(v) * f(v) * hours_per_year
        energy_density = power_values * pdf_values  # W per (m/s)
        aep_w = float(trapezoid(energy_density, speeds)) * self.hours_per_year

        # Air density correction
        rho = self.air_density
        if hasattr(self, '_current_rho'):
            rho = self._current_rho
        rho_factor = rho / 1.225

        return aep_w * rho_factor

    def _report_status(self, msg: str):
        logger.info(msg)



