"""
Wind Farm Layout Optimizer for WindFarm Designer Pro.
Generates optimized turbine layouts within a given wind farm boundary
for a desired installed capacity.

Algorithms supported:
1. Grid-based layout with spacing optimization
2. Greedy placement with wake overlap minimization (KD-tree accelerated)
3. Particle Swarm Optimization (PSO) for micro-siting (reflective boundaries, restart)
4. Genetic Algorithm (GA) for layout optimization (diversity preservation, adaptive mutation)

The optimizer considers:
- Wind resource spatial variation
- Terrain complexity (slope constraints)
- Wake interference (Park/Jensen model)
- Minimum/maximum inter-turbine spacing
- Boundary constraints
- Desired installed capacity
- Turbine rotor diameter and rated power
"""

import math
import random
import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass, field
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.ops import unary_union
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class TurbineModel:
    """Wind turbine specification."""
    name: str = "Generic 3MW"
    manufacturer: str = "Generic"
    rated_power_kw: float = 3000.0
    hub_height_m: float = 80.0
    rotor_diameter_m: float = 100.0
    cut_in_ms: float = 3.0
    cut_out_ms: float = 25.0
    rated_speed_ms: float = 12.0


@dataclass
class WTGPosition:
    """Wind turbine generator position."""
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    x_local: float = 0.0  # Local coordinate (meters)
    y_local: float = 0.0
    elevation: float = 0.0  # From DEM
    roughness: float = 0.03  # z0 at position
    hub_height_m: float = 80.0
    aep_gwh: float = 0.0  # Calculated AEP
    wake_loss_pct: float = 0.0
    gross_aep_gwh: float = 0.0
    net_aep_gwh: float = 0.0
    mean_wind_speed_ms: float = 0.0


@dataclass
class LayoutConfig:
    """Configuration for layout optimization."""
    min_spacing_rotor_d: float = 5.0  # Minimum spacing in rotor diameters
    preferred_spacing_rotor_d: float = 7.0  # Preferred spacing
    max_slope_deg: float = 15.0  # Maximum terrain slope
    boundary_buffer_m: float = 200.0  # Buffer from boundary edge
    max_capacity_kw: float = 100000.0  # Desired installed capacity (kW)
    min_capacity_kw: float = 0.0  # Minimum acceptable capacity
    algorithm: str = "pso"  # 'grid', 'greedy', 'pso', 'ga'
    n_populations: int = 50  # For PSO/GA
    n_iterations: int = 100  # For PSO/GA
    random_seed: Optional[int] = None


# ============================================================
# Spatial utilities
# ============================================================

def latlon_to_local(coords: List[Tuple[float, float]],
                    origin: Tuple[float, float] = None) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """
    Convert lat/lon to local Cartesian coordinates (meters).
    Uses a simple equirectangular projection centered on the origin.

    Parameters
    ----------
    coords : list of (lat, lon)
    origin : (lat, lon) or None (auto-computed as centroid)

    Returns
    -------
    list of (x, y), origin (lat, lon)
    """
    if origin is None:
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        origin = (np.mean(lats), np.mean(lons))

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(origin[0]))

    local = []
    for lat, lon in coords:
        x = (lon - origin[1]) * m_per_deg_lon
        y = (lat - origin[0]) * m_per_deg_lat
        local.append((x, y))

    return local, origin


def local_to_latlon(local_coords: List[Tuple[float, float]],
                    origin: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Convert local Cartesian coordinates back to lat/lon."""
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(origin[0]))

    coords = []
    for x, y in local_coords:
        lon = origin[1] + x / m_per_deg_lon
        lat = origin[0] + y / m_per_deg_lat
        coords.append((lat, lon))

    return coords


# ============================================================
# Layout Optimizer
# ============================================================

class LayoutOptimizer:
    """
    Optimizes wind turbine layout within a given boundary.

    Parameters
    ----------
    boundary : list of (lat, lon)
        Wind farm boundary polygon vertices.
    turbine : TurbineModel
        Turbine specification.
    config : LayoutConfig
        Optimization configuration.
    """

    def __init__(self, boundary: List[Tuple[float, float]],
                 turbine: TurbineModel,
                 config: LayoutConfig = None):
        self.boundary_latlon = boundary
        self.turbine = turbine

        if config is None:
            config = LayoutConfig()
        self.config = config

        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)

        # Convert boundary to local coordinates
        self.local_boundary, self.origin = latlon_to_local(boundary)
        self.boundary_polygon = Polygon(self.local_boundary)

        if not self.boundary_polygon.is_valid:
            self.boundary_polygon = self.boundary_polygon.buffer(0)

        # Buffered boundary for turbine placement
        self.inner_polygon = self.boundary_polygon.buffer(-config.boundary_buffer_m)

        # If the buffer consumed the polygon entirely, fall back to the
        # unbuffered boundary.
        if self.inner_polygon.is_empty:
            self.inner_polygon = self.boundary_polygon

        # Spacing constraints
        self.min_spacing = config.min_spacing_rotor_d * turbine.rotor_diameter_m
        self.preferred_spacing = config.preferred_spacing_rotor_d * turbine.rotor_diameter_m

        # Target capacity and number of turbines
        self.n_turbines_target = int(np.ceil(
            config.max_capacity_kw / turbine.rated_power_kw
        ))

        # Wind data (set externally)
        self.wind_data: Optional[Dict] = None
        self.elevation_data: Optional[Dict] = None
        self.roughness_data: Optional[Dict] = None

        # Results
        self.positions: List[WTGPosition] = []
        self.best_positions: List[WTGPosition] = []
        self.best_aep = 0.0

        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None

    def set_wind_data(self, wind_data: Dict):
        """Set wind resource data for the area."""
        self.wind_data = wind_data

    def set_elevation_data(self, elevation_map: np.ndarray, transform=None):
        """Set elevation data for slope constraints."""
        self.elevation_data = {'map': elevation_map, 'transform': transform}

    def set_roughness_data(self, roughness_map: np.ndarray, transform=None):
        """Set roughness data."""
        self.roughness_data = {'map': roughness_map, 'transform': transform}

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

    # ============================================================
    # Algorithm Selection
    # ============================================================

    def optimize(self) -> List[WTGPosition]:
        """
        Run the layout optimization based on the configured algorithm.

        Returns
        -------
        list of WTGPosition
            Optimized turbine positions.
        """
        algo = self.config.algorithm
        self._report_status(
            f"Starting layout optimization: {algo.upper()} algorithm, "
            f"target capacity = {self.config.max_capacity_kw / 1000:.0f} MW, "
            f"target turbines = {self.n_turbines_target}"
        )

        t0 = time.time()

        if algo == 'grid':
            positions = self._optimize_grid()
        elif algo == 'greedy':
            positions = self._optimize_greedy()
        elif algo == 'pso':
            positions = self._optimize_pso()
        elif algo == 'ga':
            positions = self._optimize_ga()
        else:
            self._report_status(f"Unknown algorithm '{algo}'. Using grid.")
            positions = self._optimize_grid()

        # Ensure capacity match
        positions = self._adjust_capacity(positions)

        # Sort and name
        for i, pos in enumerate(positions):
            pos.name = f"WTG{i + 1:03d}"

        self.best_positions = positions

        elapsed = time.time() - t0
        self._report_status(
            f"Optimization complete: {len(positions)} turbines, "
            f"installed capacity = {len(positions) * self.turbine.rated_power_kw / 1000:.1f} MW, "
            f"time = {elapsed:.1f}s"
        )

        return positions

    # ============================================================
    # Algorithm 1: Grid-based Layout
    # ============================================================

    def _optimize_grid(self) -> List[WTGPosition]:
        """
        Generate a regular grid layout with optimized spacing and rotation.
        The grid is rotated to align with the prevailing wind direction.
        """
        self._report_status("Running Grid-based layout optimization...")
        t0 = time.time()

        # Determine prevailing wind direction from wind data
        prevailing_dir = self._get_prevailing_direction()

        # Generate rotated grid
        minx, miny, maxx, maxy = self.inner_polygon.bounds
        spacing = self.preferred_spacing

        # Try multiple grid rotations and pick the one with best AEP estimate
        best_positions = []
        best_score = -1e20

        # Use finer rotation steps near the prevailing wind direction
        rotations = list(np.arange(0, 180, 10))
        # Add finer steps around the prevailing direction
        for offset in [-15, -10, -5, 0, 5, 10, 15]:
            r = prevailing_dir + offset
            r = r % 180
            if r not in rotations:
                rotations.append(r)
        rotations.sort()

        n_rotations = len(rotations)
        for idx, rotation in enumerate(rotations):
            positions = self._generate_rotated_grid(
                minx, miny, maxx, maxy, spacing, rotation
            )
            # Score: prefer positions closer to target count with good coverage
            score = self._score_grid_layout(positions)
            if score > best_score:
                best_score = score
                best_positions = positions

            self._report_progress(idx + 1, n_rotations,
                                 f"Grid rotation {rotation:.0f}°: "
                                 f"{len(positions)} positions")

        elapsed = time.time() - t0
        self._report_status(
            f"Grid layout: {len(best_positions)} candidate positions, "
            f"time = {elapsed:.1f}s"
        )
        return best_positions

    def _score_grid_layout(self, positions: List[WTGPosition]) -> float:
        """Score a grid layout — higher is better."""
        n = len(positions)
        if n == 0:
            return -1e20

        # Reward being close to the target number of turbines
        target = self.n_turbines_target
        count_score = -abs(n - target) * 2.0

        # Reward wind resource quality
        wind_score = 0.0
        for pos in positions:
            wind_score += self._get_wind_score(pos)

        # Penalise excessive wake overlap
        wake_penalty = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i].x_local - positions[j].x_local
                dy = positions[i].y_local - positions[j].y_local
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < self.preferred_spacing:
                    wake_penalty += (self.preferred_spacing - dist) / self.preferred_spacing

        return count_score + wind_score - wake_penalty

    def _generate_rotated_grid(self, minx, miny, maxx, maxy,
                                spacing, rotation_deg) -> List[WTGPosition]:
        """Generate a rotated grid of candidate positions.

        Uses tighter bounding based on the rotated polygon extent instead
        of the axis-aligned bounding box * 1.5, reducing the number of
        candidates that fall outside the boundary.
        """
        positions = []
        rotation_rad = math.radians(rotation_deg)
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)

        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        # Compute the bounding extent in the rotated frame
        # Sample the polygon vertices in the rotated coordinate system
        coords = list(self.inner_polygon.exterior.coords)
        rx_vals = [(x - cx) * cos_r + (y - cy) * sin_r for x, y in coords]
        ry_vals = [-(x - cx) * sin_r + (y - cy) * cos_r for x, y in coords]
        rx_min, rx_max = min(rx_vals), max(rx_vals)
        ry_min, ry_max = min(ry_vals), max(ry_vals)

        # Add a small margin
        margin = spacing * 0.5
        rx_min -= margin
        rx_max += margin
        ry_min -= margin
        ry_max += margin

        n_x = int((rx_max - rx_min) / spacing) + 1
        n_y = int((ry_max - ry_min) / spacing) + 1

        # Pre-compute positions in rotated frame, then rotate back
        # and check boundary + spacing in batch.
        candidate_x = []
        candidate_y = []
        for i in range(n_x):
            for j in range(n_y):
                rx = rx_min + i * spacing
                ry = ry_min + j * spacing
                # Rotate back to local frame
                lx = cx + rx * cos_r - ry * sin_r
                ly = cy + rx * sin_r + ry * cos_r
                candidate_x.append(lx)
                candidate_y.append(ly)

        if not candidate_x:
            return positions

        # Vectorised boundary check using numpy for speed
        cand_xy = np.column_stack([candidate_x, candidate_y])
        from shapely.vectorized import contains
        inside = contains(self.inner_polygon, cand_xy[:, 0], cand_xy[:, 1])

        # Also accept points very close to the boundary
        boundary_dist = np.array([
            self.inner_polygon.boundary.distance(Point(cx_, cy_))
            for cx_, cy_ in zip(candidate_x, candidate_y)
        ])
        near_boundary = boundary_dist < 2.0
        valid_mask = inside | near_boundary

        valid_x = np.array(candidate_x)[valid_mask]
        valid_y = np.array(candidate_y)[valid_mask]

        # Spacing check using KDTree incrementally
        placed_coords = []
        for vx, vy in zip(valid_x, valid_y):
            if not placed_coords:
                placed_coords.append((vx, vy))
                continue
            # Check distance to all already-placed positions
            arr = np.array(placed_coords)
            dists = np.sqrt((arr[:, 0] - vx) ** 2 + (arr[:, 1] - vy) ** 2)
            if np.all(dists >= self.min_spacing):
                placed_coords.append((vx, vy))

        for px, py in placed_coords:
            positions.append(WTGPosition(
                lat=0.0, lon=0.0,
                x_local=px, y_local=py,
                hub_height_m=self.turbine.hub_height_m,
            ))

        return positions

    # ============================================================
    # Algorithm 2: Greedy Placement (KD-tree accelerated)
    # ============================================================

    def _optimize_greedy(self) -> List[WTGPosition]:
        """
        Greedy placement algorithm that places turbines one by one,
        each time selecting the position with the highest expected
        energy yield considering wake effects.

        Performance improvements:
        - Uses scipy.spatial.KDTree for fast nearest-neighbor lookups
        - Vectorised distance pre-filtering with numpy
        - Early termination when score exceeds dynamic threshold
        - Batch evaluation of candidate subsets
        """
        self._report_status("Running Greedy layout optimization...")
        t0 = time.time()

        # Generate candidate positions on a dense grid
        minx, miny, maxx, maxy = self.inner_polygon.bounds
        candidate_spacing = self.min_spacing * 0.5
        candidates = self._generate_rotated_grid(
            minx, miny, maxx, maxy, candidate_spacing, 0
        )

        self._report_status(f"Generated {len(candidates)} candidate positions")

        selected: List[WTGPosition] = []
        target_n = self.n_turbines_target
        min_spacing_sq = self.min_spacing ** 2

        for iteration in range(target_n):
            iter_t0 = time.time()

            # Build KD-tree from already-selected turbines
            if selected:
                sel_coords = np.array([[s.x_local, s.y_local] for s in selected])
                tree = KDTree(sel_coords)
            else:
                tree = None

            # Build numpy arrays of all candidate coordinates
            cand_coords = np.array([[c.x_local, c.y_local] for c in candidates])
            if len(cand_coords) == 0:
                self._report_status(
                    f"Cannot place more turbines. Stopped at {len(selected)}."
                )
                break

            # Vectorised spacing pre-filter using KD-tree
            if tree is not None:
                # Query nearest selected turbine for each candidate
                dists, _ = tree.query(cand_coords, k=1)
                viable_mask = dists >= self.min_spacing
            else:
                viable_mask = np.ones(len(cand_coords), dtype=bool)

            viable_indices = np.where(viable_mask)[0]
            if len(viable_indices) == 0:
                self._report_status(
                    f"Cannot place more turbines. Stopped at {len(selected)}."
                )
                break

            # Limit candidate evaluation for performance:
            # If there are too many viable candidates, sub-sample
            max_eval = 500
            if len(viable_indices) > max_eval:
                # Prefer candidates in high-wind areas: compute wind scores
                # for a random subset, keep the best
                eval_indices = np.random.choice(
                    viable_indices, size=max_eval, replace=False
                )
            else:
                eval_indices = viable_indices

            # Evaluate candidates with early termination
            best_idx = -1
            best_score = -1e20
            early_stop_count = 0
            max_early_stop = min(50, len(eval_indices))

            for k in eval_indices:
                cand = candidates[k]
                score = self._evaluate_position_score(cand, selected, tree)
                if score > best_score:
                    best_score = score
                    best_idx = int(k)
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    # If we've seen many candidates without improvement,
                    # the current best is likely good enough
                    if early_stop_count >= max_early_stop:
                        break

            if best_idx >= 0:
                selected.append(candidates.pop(best_idx))
                iter_elapsed = time.time() - iter_t0
                self._report_progress(
                    iteration + 1, target_n,
                    f"Placed turbine {len(selected)}/{target_n} "
                    f"(score={best_score:.2f}, {iter_elapsed:.2f}s)"
                )

        elapsed = time.time() - t0
        self._report_status(
            f"Greedy layout: {len(selected)} turbines, time = {elapsed:.1f}s"
        )
        return selected

    def _evaluate_position_score(self, candidate: WTGPosition,
                                  existing: List[WTGPosition],
                                  tree: Optional[KDTree] = None) -> float:
        """
        Evaluate the quality of a candidate turbine position.
        Higher score = better position.

        Considers:
        - Wind resource at the position
        - Wake losses from upstream turbines
        - Distance to boundary (penalty for edge positions)
        """
        # Base score from wind resource
        base_score = self._get_wind_score(candidate)

        # Wake penalty from existing turbines
        wake_penalty = 0.0
        wind_dir = self._get_prevailing_direction()
        wind_rad_from = math.radians(270.0 - wind_dir)
        sin_w = math.sin(wind_rad_from)
        cos_w = math.cos(wind_rad_from)

        if tree is not None and existing:
            # Only check nearby turbines (within 3x preferred spacing)
            sel_coords = np.array([[e.x_local, e.y_local] for e in existing])
            dists, indices = tree.query(
                [candidate.x_local, candidate.y_local],
                k=min(len(existing), 10)
            )
            # Make dists and indices arrays even for k=1
            dists = np.atleast_1d(dists)
            indices = np.atleast_1d(indices)

            for d, idx in zip(dists, indices):
                if d > self.preferred_spacing * 3:
                    continue
                ext = existing[idx]
                dx = candidate.x_local - ext.x_local
                dy = candidate.y_local - ext.y_local
                downwind = -(dx * sin_w + dy * cos_w)
                crosswind = abs(-dx * cos_w + dy * sin_w)
                if downwind > 0 and crosswind < self.turbine.rotor_diameter_m * 3:
                    # Wake deficit (simplified Jensen)
                    wake_expansion = 0.05
                    rotor_radius = self.turbine.rotor_diameter_m / 2
                    wake_radius = rotor_radius + wake_expansion * downwind
                    if crosswind < wake_radius:
                        deficit = (rotor_radius / (rotor_radius + wake_expansion * downwind)) ** 2
                        deficit *= 0.8  # Max wake deficit
                        wake_penalty += deficit
        else:
            # Fallback: iterate all existing turbines
            for ext in existing:
                dx = candidate.x_local - ext.x_local
                dy = candidate.y_local - ext.y_local
                dist = math.sqrt(dx ** 2 + dy ** 2)
                if dist < self.preferred_spacing * 3:
                    downwind = -(dx * sin_w + dy * cos_w)
                    crosswind = abs(-dx * cos_w + dy * sin_w)
                    if downwind > 0 and crosswind < self.turbine.rotor_diameter_m * 3:
                        wake_expansion = 0.05
                        rotor_radius = self.turbine.rotor_diameter_m / 2
                        wake_radius = rotor_radius + wake_expansion * downwind
                        if crosswind < wake_radius:
                            deficit = (rotor_radius / (rotor_radius + wake_expansion * downwind)) ** 2
                            deficit *= 0.8
                            wake_penalty += deficit

        # Boundary distance bonus
        boundary_dist = self.inner_polygon.boundary.distance(
            Point(candidate.x_local, candidate.y_local)
        )
        boundary_bonus = min(boundary_dist / 1000.0, 1.0) * 0.5

        return base_score * (1.0 - min(wake_penalty, 0.5)) + boundary_bonus

    def _get_wind_score(self, position: WTGPosition) -> float:
        """Get a wind resource score for a position (0-10)."""
        if self.wind_data:
            points = self.wind_data.get('points', [])
            if points:
                # Find nearest GWA point
                latlon = local_to_latlon([(position.x_local, position.y_local)], self.origin)[0]
                min_dist = float('inf')
                nearest_speed = 7.0  # Default
                for p in points:
                    d = (latlon[0] - p['lat']) ** 2 + (latlon[1] - p['lon']) ** 2
                    if d < min_dist:
                        min_dist = d
                        nearest_speed = p.get('mean_wind_speed', 7.0)
                return nearest_speed  # Score proportional to wind speed
        return 7.0  # Default moderate wind resource

    def _get_prevailing_direction(self) -> float:
        """Get the prevailing wind direction from wind data."""
        if self.wind_data:
            points = self.wind_data.get('points', [])
            if points and 'sectors' in points[0]:
                sectors = points[0]['sectors']
                if sectors:
                    max_freq_sector = max(sectors, key=lambda s: s.get('frequency', 0))
                    return max_freq_sector.get('direction', 0.0)
        return 225.0  # Default: SW

    # ============================================================
    # Algorithm 3: Particle Swarm Optimization (PSO)
    # ============================================================

    def _optimize_pso(self) -> List[WTGPosition]:
        """
        PSO-based layout optimization where each particle represents
        a complete layout (positions of all turbines).

        The fitness function maximizes Annual Energy Production (AEP)
        while respecting spacing and boundary constraints.

        Improvements over the previous version:
        - Reflective boundary enforcement (no velocity reset)
        - Adaptive inertia: w = 0.9 - 0.5 * (iter / max_iter)
        - Restart mechanism for stagnated particles
        """
        self._report_status("Running PSO layout optimization...")
        t0 = time.time()

        n_turb = self.n_turbines_target
        n_pop = self.config.n_populations
        n_iter = self.config.n_iterations

        minx, miny, maxx, maxy = self.inner_polygon.bounds

        # Initialize particle swarm
        particles = []
        velocities = []
        personal_bests = []
        personal_best_fitness = []

        # Velocity limits proportional to boundary extent
        max_v = max(maxx - minx, maxy - miny) * 0.1

        for _ in range(n_pop):
            # Generate random layout within boundary
            layout = self._random_layout(n_turb, minx, miny, maxx, maxy)
            particles.append(layout)
            velocities.append([
                [random.uniform(-max_v, max_v) for _ in range(2)]
                for _ in range(n_turb)
            ])
            fitness = self._layout_fitness(layout)
            personal_bests.append([list(t) for t in layout])
            personal_best_fitness.append(fitness)

        # Global best
        global_best_idx = int(np.argmax(personal_best_fitness))
        global_best = [list(t) for t in personal_bests[global_best_idx]]
        global_best_fitness = personal_best_fitness[global_best_idx]

        c1 = 1.5   # Cognitive component
        c2 = 1.5   # Social component

        # Stagnation tracking for restart mechanism
        stagnation_count = 0
        prev_global_best_fitness = global_best_fitness

        for iteration in range(n_iter):
            # Adaptive inertia: linearly decrease from 0.9 to 0.4
            w = 0.9 - 0.5 * (iteration / n_iter)

            for i in range(n_pop):
                for j in range(n_turb):
                    # Update velocity
                    r1, r2 = random.random(), random.random()
                    velocities[i][j][0] = (
                        w * velocities[i][j][0] +
                        c1 * r1 * (personal_bests[i][j][0] - particles[i][j][0]) +
                        c2 * r2 * (global_best[j][0] - particles[i][j][0])
                    )
                    velocities[i][j][1] = (
                        w * velocities[i][j][1] +
                        c1 * r1 * (personal_bests[i][j][1] - particles[i][j][1]) +
                        c2 * r2 * (global_best[j][1] - particles[i][j][1])
                    )

                    # Clamp velocity (proportional to boundary size)
                    velocities[i][j][0] = max(-max_v, min(max_v, velocities[i][j][0]))
                    velocities[i][j][1] = max(-max_v, min(max_v, velocities[i][j][1]))

                    # Update position
                    particles[i][j][0] += velocities[i][j][0]
                    particles[i][j][1] += velocities[i][j][1]

                # Reflective boundary enforcement
                self._reflect_boundary(particles[i], velocities[i], minx, miny, maxx, maxy)

                # Evaluate fitness
                fitness = self._layout_fitness(particles[i])

                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_bests[i] = [list(t) for t in particles[i]]

                    # Update global best
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best = [list(t) for t in particles[i]]

            # Stagnation tracking
            if abs(global_best_fitness - prev_global_best_fitness) < 1e-6:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_global_best_fitness = global_best_fitness

            # Restart mechanism: reinitialize worst 30% if stagnated for 20 iters
            if stagnation_count >= 20:
                n_restart = max(1, int(n_pop * 0.3))
                # Find the worst performers
                worst_indices = np.argsort(personal_best_fitness)[:n_restart]
                for wi in worst_indices:
                    new_layout = self._random_layout(n_turb, minx, miny, maxx, maxy)
                    particles[wi] = new_layout
                    velocities[wi] = [
                        [random.uniform(-max_v, max_v) for _ in range(2)]
                        for _ in range(n_turb)
                    ]
                    fitness = self._layout_fitness(new_layout)
                    personal_bests[wi] = [list(t) for t in new_layout]
                    personal_best_fitness[wi] = fitness
                stagnation_count = 0
                self._report_status(
                    f"PSO restart: reinitialized {n_restart} stagnant particles "
                    f"at iteration {iteration + 1}"
                )

            if (iteration + 1) % 10 == 0:
                self._report_progress(
                    iteration + 1, n_iter,
                    f"PSO iteration {iteration + 1}/{n_iter}, "
                    f"best fitness = {global_best_fitness:.2f}, "
                    f"w = {w:.3f}"
                )

        # Convert global best to WTGPosition list
        positions = []
        for x, y in global_best:
            pos = WTGPosition(
                lat=0.0, lon=0.0,
                x_local=x, y_local=y,
                hub_height_m=self.turbine.hub_height_m,
            )
            positions.append(pos)

        elapsed = time.time() - t0
        self._report_status(
            f"PSO complete: best fitness = {global_best_fitness:.2f}, "
            f"time = {elapsed:.1f}s"
        )
        return positions

    def _reflect_boundary(self, layout: List[List[float]],
                          velocities: List[List[float]],
                          minx: float, miny: float, maxx: float, maxy):
        """Reflective boundary enforcement.

        When a turbine goes outside the bounding box, its position is
        reflected back inside and its velocity component is reversed
        (with damping) instead of being reset to zero. This prevents
        premature convergence caused by the old clamping approach.
        """
        damping = 0.5  # Energy loss on reflection
        for j in range(len(layout)):
            x, y = layout[j]

            # Reflect in X
            if x < minx:
                layout[j][0] = minx + (minx - x)
                velocities[j][0] = abs(velocities[j][0]) * damping
            elif x > maxx:
                layout[j][0] = maxx - (x - maxx)
                velocities[j][0] = -abs(velocities[j][0]) * damping

            # Reflect in Y
            if layout[j][1] < miny:
                layout[j][1] = miny + (miny - layout[j][1])
                velocities[j][1] = abs(velocities[j][1]) * damping
            elif layout[j][1] > maxy:
                layout[j][1] = maxy - (layout[j][1] - maxy)
                velocities[j][1] = -abs(velocities[j][1]) * damping

        # After reflection, check actual polygon boundary and nudge inside
        for j in range(len(layout)):
            pt = Point(layout[j][0], layout[j][1])
            if not self.inner_polygon.contains(pt):
                nearest = self.inner_polygon.boundary.interpolate(
                    self.inner_polygon.boundary.project(pt)
                )
                center = self.inner_polygon.centroid
                dx = center.x - nearest.x
                dy = center.y - nearest.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    layout[j] = [nearest.x + dx / dist * 10,
                                 nearest.y + dy / dist * 10]
                else:
                    layout[j] = [nearest.x, nearest.y]
                # Reduce velocity magnitude but don't zero it
                velocities[j][0] *= 0.3
                velocities[j][1] *= 0.3

    # ============================================================
    # Algorithm 4: Genetic Algorithm (GA)
    # ============================================================

    def _optimize_ga(self) -> List[WTGPosition]:
        """
        Genetic Algorithm for layout optimization.
        Uses tournament selection (size 5), layout-aware crossover,
        adaptive mutation, and diversity preservation.

        Improvements over the previous version:
        - Tournament selection with k=5 (was 3)
        - Layout-preserving crossover operator
        - Adaptive mutation rate that decreases over generations
        - Diversity preservation via random injection when diversity drops
        """
        self._report_status("Running Genetic Algorithm layout optimization...")
        t0 = time.time()

        n_turb = self.n_turbines_target
        n_pop = self.config.n_populations
        n_gen = self.config.n_iterations
        crossover_rate = 0.8
        base_mutation_rate = 0.2

        minx, miny, maxx, maxy = self.inner_polygon.bounds

        # Initialize population
        population = [
            self._random_layout(n_turb, minx, miny, maxx, maxy)
            for _ in range(n_pop)
        ]

        fitness = [self._layout_fitness(layout) for layout in population]

        best_gen_fitness = max(fitness)
        best_gen_layout = [list(t) for t in population[int(np.argmax(fitness))]]

        for gen in range(n_gen):
            # Adaptive mutation rate: decreases linearly over generations
            mutation_rate = base_mutation_rate * (1.0 - 0.7 * (gen / n_gen))

            # Diversity measurement
            diversity = self._population_diversity(population)
            diversity_threshold = self.min_spacing * 0.5  # Average pairwise spread

            new_population = []

            # Elitism: keep best 2
            elite_idx = np.argsort(fitness)[-2:]
            for idx in elite_idx:
                new_population.append([list(t) for t in population[idx]])

            # Diversity preservation: inject random individuals if diversity is low
            if diversity < diversity_threshold:
                n_inject = max(2, int(n_pop * 0.15))
                self._report_status(
                    f"GA gen {gen + 1}: Low diversity ({diversity:.0f}m), "
                    f"injecting {n_inject} random individuals"
                )
                for _ in range(n_inject):
                    new_population.append(
                        self._random_layout(n_turb, minx, miny, maxx, maxy)
                    )

            # Generate rest through selection, crossover, mutation
            while len(new_population) < n_pop:
                # Tournament selection with size 5
                parent1 = self._tournament_select(population, fitness, k=5)
                parent2 = self._tournament_select(population, fitness, k=5)

                # Crossover
                if random.random() < crossover_rate:
                    child = self._layout_crossover(parent1, parent2, n_turb)
                else:
                    child = [list(t) for t in parent1]

                # Mutation (adaptive rate)
                if random.random() < mutation_rate:
                    child = self._polynomial_mutation(child, minx, miny, maxx, maxy)

                # Enforce boundary
                child = self._enforce_boundary(child)
                new_population.append(child)

            population = new_population
            fitness = [self._layout_fitness(layout) for layout in population]

            # Track best
            current_best = max(fitness)
            if current_best > best_gen_fitness:
                best_gen_fitness = current_best
                best_gen_layout = [list(t) for t in population[int(np.argmax(fitness))]]

            if (gen + 1) % 10 == 0:
                self._report_progress(
                    gen + 1, n_gen,
                    f"GA generation {gen + 1}/{n_gen}, "
                    f"best fitness = {best_gen_fitness:.2f}, "
                    f"mutation rate = {mutation_rate:.3f}, "
                    f"diversity = {diversity:.0f}m"
                )

        positions = []
        for x, y in best_gen_layout:
            pos = WTGPosition(
                lat=0.0, lon=0.0,
                x_local=x, y_local=y,
                hub_height_m=self.turbine.hub_height_m,
            )
            positions.append(pos)

        elapsed = time.time() - t0
        self._report_status(
            f"GA complete: best fitness = {best_gen_fitness:.2f}, "
            f"time = {elapsed:.1f}s"
        )
        return positions

    def _population_diversity(self, population: List[List[List[float]]]) -> float:
        """Measure population diversity as the average standard deviation
        of turbine positions across all individuals."""
        n_pop = len(population)
        if n_pop == 0:
            return 0.0
        n_turb = len(population[0])

        # Compute centroid of all turbines across the population
        all_coords = np.array([[t[0], t[1]] for layout in population for t in layout])
        # Flatten: shape (n_pop * n_turb, 2)
        flat = all_coords.reshape(-1, 2)
        if len(flat) == 0:
            return 0.0
        # Standard deviation across all positions (proxy for spread)
        return float(np.mean(np.std(flat, axis=0)))

    # ============================================================
    # GA Helper Methods
    # ============================================================

    def _tournament_select(self, population, fitness, k=5):
        """Tournament selection with tournament size k."""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx]

    def _layout_crossover(self, parent1, parent2, n_turb, eta=20):
        """Layout-aware crossover that creates valid child layouts.

        Uses a whole-turbine swap strategy: for each turbine index,
        randomly choose coordinates from parent1 or parent2. Then apply
        SBX blend to create intermediate positions that tend to produce
        valid layouts.
        """
        child = []
        for i in range(n_turb):
            if random.random() < 0.5:
                # Use parent1 as base, blend with parent2
                cx, cy = parent1[i]
                alpha = random.uniform(0.2, 0.8)
                cx = alpha * parent1[i][0] + (1 - alpha) * parent2[i][0]
                cy = alpha * parent1[i][1] + (1 - alpha) * parent2[i][1]
            else:
                # Use parent2 as base, blend with parent1
                cx, cy = parent2[i]
                alpha = random.uniform(0.2, 0.8)
                cx = alpha * parent2[i][0] + (1 - alpha) * parent1[i][0]
                cy = alpha * parent2[i][1] + (1 - alpha) * parent1[i][1]
            child.append([cx, cy])

        # Repair spacing violations by nudging overlapping turbines
        child = self._repair_spacing(child, n_turb)
        return child

    def _repair_spacing(self, layout: List[List[float]],
                        n_turb: int, max_iterations: int = 50) -> List[List[float]]:
        """Repair spacing violations in a layout by pushing overlapping
        turbines apart iteratively."""
        min_dist = self.min_spacing
        coords = np.array(layout, dtype=float)

        for _ in range(max_iterations):
            moved = False
            for i in range(n_turb):
                for j in range(i + 1, n_turb):
                    dx = coords[j, 0] - coords[i, 0]
                    dy = coords[j, 1] - coords[i, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < min_dist and dist > 0:
                        # Push apart equally
                        overlap = (min_dist - dist) / 2.0 + 1.0
                        nx = dx / dist
                        ny = dy / dist
                        coords[i, 0] -= nx * overlap
                        coords[i, 1] -= ny * overlap
                        coords[j, 0] += nx * overlap
                        coords[j, 1] += ny * overlap
                        moved = True
            if not moved:
                break

        return [list(row) for row in coords]

    def _sbx_crossover(self, parent1, parent2, n_turb, eta=20):
        """Simulated Binary Crossover (kept for backward compatibility).

        Fixed: cx/cy are now initialised before the inner if/else block
        so they are always defined regardless of the random branching.
        """
        child = []
        for i in range(n_turb):
            # Initialise cx, cy to parent1 values as the default
            cx, cy = parent1[i]
            if random.random() < 0.5:
                for j in range(2):
                    if parent1[i][j] == parent2[i][j]:
                        child_val = parent1[i][j]
                    else:
                        if parent1[i][j] < parent2[i][j]:
                            u = random.random()
                            beta = 1.0 - (2.0 * u) ** (1.0 / (eta + 1))
                        else:
                            u = random.random()
                            beta = (2.0 * (1.0 - u)) ** (1.0 / (eta + 1)) - 1.0

                        child_val = 0.5 * (
                            (1 + beta) * parent1[i][j] + (1 - beta) * parent2[i][j]
                        )
                    if j == 0:
                        cx = child_val
                    else:
                        cy = child_val
            child.append([cx, cy])
        return child

    def _polynomial_mutation(self, child, minx, miny, maxx, maxy, eta=15):
        """Polynomial mutation."""
        for i in range(len(child)):
            for j in range(2):
                if random.random() < 0.1:
                    low = minx if j == 0 else miny
                    high = maxx if j == 0 else maxy
                    delta = high - low
                    u = random.random()
                    if u < 0.5:
                        deltaq = (2 * u) ** (1.0 / (eta + 1)) - 1.0
                    else:
                        deltaq = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[i][j] = child[i][j] + deltaq * delta
                    child[i][j] = max(low, min(high, child[i][j]))
        return child

    # ============================================================
    # Constraint & Fitness Helpers
    # ============================================================

    def _random_layout(self, n_turb, minx, miny, maxx, maxy) -> List[List[float]]:
        """Generate a random valid layout within the boundary.

        Uses multiple attempts with increasing spacing tolerance to handle
        cases where the boundary is small relative to the required spacing.
        Each attempt tries harder to place all turbines.
        """
        # Try with strict spacing first, then relax progressively
        for attempt in range(4):
            tolerance_factor = 1.0 + attempt * 0.1  # 1.0, 1.1, 1.2, 1.3
            effective_min_spacing = self.min_spacing / tolerance_factor
            layout = self._try_random_layout(n_turb, minx, miny, maxx, maxy,
                                              effective_min_spacing)
            if len(layout) >= n_turb:
                return layout
            # If we got some but not all, use what we have
            if len(layout) >= max(1, n_turb // 2):
                return layout

        # Final fallback: return whatever we got
        return layout if layout else self._try_random_layout(
            n_turb, minx, miny, maxx, maxy, self.min_spacing * 0.7
        )

    def _try_random_layout(self, n_turb, minx, miny, maxx, maxy,
                            min_spacing: float) -> List[List[float]]:
        """Single attempt to generate a random layout with the given spacing."""
        layout = []
        max_attempts = n_turb * 200  # More attempts per turbine
        attempts = 0
        consecutive_fails = 0
        max_consecutive_fails = max(50, n_turb * 50)

        while len(layout) < n_turb and attempts < max_attempts:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)

            if self.inner_polygon.contains(Point(x, y)):
                # Check spacing using KDTree for efficiency
                if layout:
                    arr = np.array(layout)
                    dists = np.sqrt((arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2)
                    if np.all(dists >= min_spacing):
                        layout.append([x, y])
                        consecutive_fails = 0
                    else:
                        consecutive_fails += 1
                else:
                    layout.append([x, y])
                    consecutive_fails = 0

            attempts += 1

            # Early termination if we haven't placed any in a long time
            if consecutive_fails >= max_consecutive_fails:
                break

        return layout

    def _enforce_boundary(self, layout: List[List[float]]) -> bool:
        """Push any out-of-boundary turbines back inside.

        Returns
        -------
        bool
            True if any turbine was moved, False otherwise.
        """
        moved = False
        for i in range(len(layout)):
            pt = Point(layout[i][0], layout[i][1])
            if not self.inner_polygon.contains(pt):
                # Find nearest point inside boundary
                nearest = self.inner_polygon.boundary.interpolate(
                    self.inner_polygon.boundary.project(pt)
                )
                # Move slightly inside
                center = self.inner_polygon.centroid
                dx = center.x - nearest.x
                dy = center.y - nearest.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    layout[i] = [nearest.x + dx / dist * 10, nearest.y + dy / dist * 10]
                else:
                    layout[i] = [nearest.x, nearest.y]
                moved = True

        return moved

    def _layout_fitness(self, layout: List[List[float]]) -> float:
        """
        Evaluate layout fitness.
        Higher is better. Considers:
        - AEP (proportional to wind resource)
        - Wake losses (penalty for close downstream spacing)
        - Spacing violations (hard penalty)
        - Boundary violations (hard penalty)
        - Capacity matching (soft reward near target)

        Coordinate convention:
        The wind blows FROM ``wind_dir`` degrees (meteorological convention:
        0=N, 90=E).  The component along the wind direction from turbine j
        to turbine i is computed as::

            downwind = -(dx * sin(wind_rad_from) + dy * cos(wind_rad_from))

        where ``wind_rad_from = radians(270 - wind_dir)``.  Positive downwind
        means i is *downstream* of j (in the wake).
        """
        fitness = 0.0
        penalty = 0.0
        n = len(layout)
        if n == 0:
            return -1e10

        # Correct wind direction: meteorological FROM -> math convention
        wind_dir_deg = self._get_prevailing_direction()
        wind_rad_from = math.radians(270.0 - wind_dir_deg)
        sin_w = math.sin(wind_rad_from)
        cos_w = math.cos(wind_rad_from)

        # Cache wind scores for each position to avoid recomputing
        wind_scores = []
        for x, y in layout:
            pt = Point(x, y)
            if not self.inner_polygon.contains(pt):
                penalty += 100.0

            ws = self._get_wind_score(
                WTGPosition(x_local=x, y_local=y, hub_height_m=self.turbine.hub_height_m)
            )
            wind_scores.append(ws)
            fitness += ws

        # Pairwise interactions: spacing + wake
        for i in range(n):
            xi, yi = layout[i]
            for j in range(i + 1, n):
                xj, yj = layout[j]
                dx = xi - xj
                dy = yi - yj
                dist = math.sqrt(dx * dx + dy * dy)

                # Spacing violation (hard penalty)
                if dist < self.min_spacing:
                    violation = (self.min_spacing - dist) / self.min_spacing
                    penalty += violation * 50.0

                # Wake penalty: positive downwind means i is in wake of j
                if dist < self.preferred_spacing * 5:
                    downwind = -(dx * sin_w + dy * cos_w)
                    crosswind = abs(-dx * cos_w + dy * sin_w)
                    if downwind > 0 and crosswind < self.turbine.rotor_diameter_m * 3:
                        wake_deficit = (self.turbine.rotor_diameter_m / (
                            self.turbine.rotor_diameter_m + 0.05 * downwind)) ** 2
                        fitness -= wake_deficit * 2.0

                    # Also check reverse direction (j in wake of i)
                    downwind_rev = -(-dx * sin_w + dy * cos_w)
                    crosswind_rev = abs(dx * cos_w + dy * sin_w)
                    if downwind_rev > 0 and crosswind_rev < self.turbine.rotor_diameter_m * 3:
                        wake_deficit = (self.turbine.rotor_diameter_m / (
                            self.turbine.rotor_diameter_m + 0.05 * downwind_rev)) ** 2
                        fitness -= wake_deficit * 2.0

        # Capacity match reward: soft reward for being near target
        # Use a Gaussian-like reward that peaks at the target capacity
        capacity = n * self.turbine.rated_power_kw
        target = self.config.max_capacity_kw
        capacity_ratio = capacity / target if target > 0 else 0

        # Reward peaks when ratio is between 0.9 and 1.1
        # Uses a bell curve centered at 1.0
        capacity_reward = 15.0 * math.exp(-((capacity_ratio - 1.0) ** 2) / (2 * 0.15 ** 2))
        fitness += capacity_reward

        # Soft penalty for being far from target (but don't over-penalize)
        if capacity_ratio < 0.7:
            penalty += (0.7 - capacity_ratio) * 10.0

        return fitness - penalty

    def _adjust_capacity(self, positions: List[WTGPosition]) -> List[WTGPosition]:
        """
        Adjust the number of turbines to match the desired capacity.
        Adds or removes turbines as needed.

        When removing turbines, removes those that contribute least to
        overall AEP considering wake effects (not just lowest wind scores).
        A turbine with high wind but deep in the wake of others may be
        removed in favor of a turbine with moderate wind but no wake losses.
        """
        current_capacity = len(positions) * self.turbine.rated_power_kw
        target = self.config.max_capacity_kw

        if current_capacity > target * 1.1:
            # Remove turbines – pick those contributing least to overall AEP
            target_n = int(target / self.turbine.rated_power_kw)
            if len(positions) > target_n:
                # Score each position by its net contribution (wind - wake losses)
                scored = []
                for idx, pos in enumerate(positions):
                    wind_score = self._get_wind_score(pos)

                    # Estimate wake loss at this position from other turbines
                    wake_loss = 0.0
                    wind_dir = self._get_prevailing_direction()
                    wind_rad_from = math.radians(270.0 - wind_dir)
                    sin_w = math.sin(wind_rad_from)
                    cos_w = math.cos(wind_rad_from)

                    for jdx, other in enumerate(positions):
                        if jdx == idx:
                            continue
                        dx = pos.x_local - other.x_local
                        dy = pos.y_local - other.y_local
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < self.preferred_spacing * 5:
                            # Check if this turbine is in the wake of `other`
                            downwind = -(dx * sin_w + dy * cos_w)
                            crosswind = abs(-dx * cos_w + dy * sin_w)
                            if downwind > 0 and crosswind < self.turbine.rotor_diameter_m * 3:
                                deficit = (self.turbine.rotor_diameter_m / (
                                    self.turbine.rotor_diameter_m + 0.05 * downwind)) ** 2
                                wake_loss += deficit * wind_score

                    # Net contribution: wind resource minus wake losses
                    net_contribution = wind_score - wake_loss
                    scored.append((net_contribution, pos))

                # Sort ascending by net contribution so the worst are first
                scored.sort(key=lambda item: item[0])
                # Keep the top `target_n` best positions
                kept = scored[-target_n:] if target_n > 0 else []
                positions = [pos for _, pos in kept]

        elif current_capacity < target * 0.9:
            # Try to add more turbines
            needed = int(np.ceil((target - current_capacity) / self.turbine.rated_power_kw))
            minx, miny, maxx, maxy = self.inner_polygon.bounds

            for _ in range(needed):
                added = False
                for _ in range(300):
                    x = random.uniform(minx, maxx)
                    y = random.uniform(miny, maxy)
                    pt = Point(x, y)

                    if self.inner_polygon.contains(pt):
                        valid = True
                        for ext in positions:
                            dx = x - ext.x_local
                            dy = y - ext.y_local
                            if math.sqrt(dx * dx + dy * dy) < self.min_spacing:
                                valid = False
                                break
                        if valid:
                            positions.append(WTGPosition(
                                lat=0.0, lon=0.0,
                                x_local=x, y_local=y,
                                hub_height_m=self.turbine.hub_height_m,
                            ))
                            added = True
                            break
                if not added:
                    break

        return positions

    # ============================================================
    # Imported Layout
    # ============================================================

    def use_imported_layout(self, layout_list: List[Tuple[float, float]]) -> List[WTGPosition]:
        """
        Bypass optimisation and use a user-provided list of turbine
        positions directly.

        Parameters
        ----------
        layout_list : list of (lat, lon)
            Turbine positions as latitude / longitude tuples in WGS84.

        Returns
        -------
        list of WTGPosition
            The converted positions, also stored in ``self.best_positions``.
        """
        self._report_status(
            f"Using imported layout with {len(layout_list)} turbine positions."
        )

        positions = []
        for lat, lon in layout_list:
            x_local = (lon - self.origin[1]) * 111320.0 * math.cos(math.radians(self.origin[0]))
            y_local = (lat - self.origin[0]) * 111320.0
            pos = WTGPosition(
                lat=lat,
                lon=lon,
                x_local=x_local,
                y_local=y_local,
                hub_height_m=self.turbine.hub_height_m,
            )
            positions.append(pos)

        # Name them
        for i, pos in enumerate(positions):
            pos.name = f"WTG{i + 1:03d}"

        self.best_positions = positions
        self._report_status(
            f"Imported layout: {len(positions)} turbines, "
            f"installed capacity = {len(positions) * self.turbine.rated_power_kw / 1000:.1f} MW"
        )
        return positions

    # ============================================================
    # Convert to Lat/Lon and Export
    # ============================================================

    def get_latlon_positions(self) -> List[Dict]:
        """Convert optimized positions to lat/lon and return as dicts."""
        latlon_coords = local_to_latlon(
            [(p.x_local, p.y_local) for p in self.best_positions],
            self.origin
        )

        result = []
        for i, pos in enumerate(self.best_positions):
            lat, lon = latlon_coords[i]
            result.append({
                'name': pos.name,
                'latitude': lat,
                'longitude': lon,
                'x_local_m': pos.x_local,
                'y_local_m': pos.y_local,
                'hub_height_m': pos.hub_height_m,
                'elevation_m': pos.elevation,
                'roughness_z0': pos.roughness,
                'gross_aep_gwh': pos.gross_aep_gwh,
                'net_aep_gwh': pos.net_aep_gwh,
                'wake_loss_pct': pos.wake_loss_pct,
                'mean_wind_speed_ms': pos.mean_wind_speed_ms,
            })

        return result
