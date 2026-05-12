"""
Data utility functions for WindFarm Designer Pro.
Handles file I/O for wind data, turbine definitions, CSV/JSON operations,
and mast data parsing.
"""

import csv
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ============================================================
# Turbine Power Curve I/O
# ============================================================

def load_power_curve(filepath: str) -> Dict:
    """
    Load a turbine power curve from a CSV file.

    Expected CSV format (header required):
        WindSpeed(m/s), Power(kW), Ct
        3.0, 0.0, 0.0
        4.0, 80.0, 0.82
        ...
        25.0, 0.0, 0.0

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    dict
        {
            'wind_speeds': np.array,   # m/s
            'power': np.array,         # kW
            'ct': np.array,            # thrust coefficient
            'rated_power_kw': float,
            'cut_in': float,
            'cut_out': float,
            'rated_speed': float,
            'hub_height': float,       # If provided in file or defaults
        }
    """
    wind_speeds = []
    power = []
    ct = []
    hub_height = 80.0  # default

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)

        # Check for extra metadata rows at the top
        for row in reader:
            ws = float(row.get('WindSpeed(m/s)', row.get('wind_speed', 0)))
            pw = float(row.get('Power(kW)', row.get('power', 0)))
            ct_val = float(row.get('Ct', row.get('ct', 0.0)))
            wind_speeds.append(ws)
            power.append(pw)
            ct.append(ct_val)

    wind_speeds = np.array(wind_speeds)
    power = np.array(power)
    ct = np.array(ct)

    # Determine key parameters from the power curve
    rated_power_kw = float(np.max(power))
    rated_idx = np.argmax(power)
    rated_speed = float(wind_speeds[rated_idx])
    cut_in = float(wind_speeds[0]) if power[0] == 0 else float(wind_speeds[np.where(power > 0)[0][0]])
    cut_out = float(wind_speeds[-1])

    return {
        'wind_speeds': wind_speeds,
        'power': power,
        'ct': ct,
        'rated_power_kw': rated_power_kw,
        'cut_in': cut_in,
        'cut_out': cut_out,
        'rated_speed': rated_speed,
        'hub_height': hub_height,
    }


def load_turbine_spec(filepath: str) -> Dict:
    """
    Load a full turbine specification from a JSON file.

    Expected JSON format:
    {
        "name": "Vestas V150-4.2",
        "manufacturer": "Vestas",
        "rated_power_kw": 4200,
        "hub_height_m": 105,
        "rotor_diameter_m": 150,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 12.5,
        "power_curve": {
            "wind_speeds": [3.0, 4.0, ...],
            "power_kw": [0.0, 80.0, ...],
            "ct": [0.0, 0.82, ...]
        }
    }

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict
    """
    with open(filepath, 'r') as f:
        spec = json.load(f)

    spec['power_curve']['wind_speeds'] = np.array(spec['power_curve']['wind_speeds'])
    spec['power_curve']['power_kw'] = np.array(spec['power_curve']['power_kw'])
    spec['power_curve']['ct'] = np.array(spec['power_curve']['ct'])

    return spec


# ============================================================
# Built-in Turbine Library
# ============================================================

BUILTIN_TURBINES = {
    "Vestas V150-4.2": {
        "manufacturer": "Vestas",
        "rated_power_kw": 4200,
        "hub_height_m": 105,
        "rotor_diameter_m": 150,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 12.5,
    },
    "Siemens Gamesa SG 14-222 DD": {
        "manufacturer": "Siemens Gamesa",
        "rated_power_kw": 14000,
        "hub_height_m": 115,
        "rotor_diameter_m": 222,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 11.5,
    },
    "GE Haliade-X 13": {
        "manufacturer": "GE",
        "rated_power_kw": 13000,
        "hub_height_m": 138,
        "rotor_diameter_m": 220,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 11.0,
    },
    "NREL 5MW Reference": {
        "manufacturer": "NREL",
        "rated_power_kw": 5000,
        "hub_height_m": 90,
        "rotor_diameter_m": 126,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 11.4,
    },
    "Enercon E-115 E4": {
        "manufacturer": "Enercon",
        "rated_power_kw": 3200,
        "hub_height_m": 92,
        "rotor_diameter_m": 115,
        "cut_in_ms": 2.0,
        "cut_out_ms": 34.0,
        "rated_speed_ms": 13.0,
    },
    "Goldwind GW 154-6.7": {
        "manufacturer": "Goldwind",
        "rated_power_kw": 6700,
        "hub_height_m": 100,
        "rotor_diameter_m": 154,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 11.0,
    },
    "Nordex N163-5.X": {
        "manufacturer": "Nordex",
        "rated_power_kw": 5000,
        "hub_height_m": 100,
        "rotor_diameter_m": 163,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 11.5,
    },
    "Vestas V164-9.5": {
        "manufacturer": "Vestas",
        "rated_power_kw": 9500,
        "hub_height_m": 105,
        "rotor_diameter_m": 164,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 12.5,
    },
    "GE 2.75-120": {
        "manufacturer": "GE",
        "rated_power_kw": 2750,
        "hub_height_m": 85,
        "rotor_diameter_m": 120,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 11.0,
    },
    "Siemens SWT-3.6-120": {
        "manufacturer": "Siemens",
        "rated_power_kw": 3600,
        "hub_height_m": 90,
        "rotor_diameter_m": 120,
        "cut_in_ms": 3.0,
        "cut_out_ms": 25.0,
        "rated_speed_ms": 12.0,
    },
}


def generate_default_power_curve(turbine_spec: Dict) -> Dict:
    """
    Generate a synthetic power curve for a turbine using standard
    aerodynamic approximations when no measured curve is available.

    Uses a simplified model:
    - Cubic region: P = P_rated * ((v - v_ci) / (v_rated - v_ci))^3
    - Rated region: P = P_rated
    - Zero below cut-in and above cut-out

    Parameters
    ----------
    turbine_spec : dict
        Turbine specification dictionary.

    Returns
    -------
    dict
        Power curve with wind_speeds, power_kw, ct arrays.
    """
    ci = turbine_spec['cut_in_ms']
    co = turbine_spec['cut_out_ms']
    vr = turbine_spec['rated_speed_ms']
    pr = turbine_spec['rated_power_kw']

    speeds = np.arange(0, co + 1, 0.5)
    power = np.zeros_like(speeds)
    ct = np.zeros_like(speeds)

    for i, v in enumerate(speeds):
        if v < ci:
            power[i] = 0.0
            ct[i] = 0.0
        elif v <= vr:
            frac = (v - ci) / (vr - ci)
            power[i] = pr * (frac ** 3)
            ct[i] = min(0.8, 4.0 * (1.0 - frac) * frac + 0.1)  # Approximate Ct
        elif v <= co:
            power[i] = pr
            ct[i] = 0.05  # Low Ct at rated
        else:
            power[i] = 0.0
            ct[i] = 0.0

    return {
        'wind_speeds': speeds,
        'power': power,
        'ct': ct,
        'rated_power_kw': pr,
        'cut_in': ci,
        'cut_out': co,
        'rated_speed': vr,
    }


def get_turbine_names() -> List[str]:
    """Return list of built-in turbine names."""
    return sorted(BUILTIN_TURBINES.keys())


def get_turbine_spec(name: str) -> Optional[Dict]:
    """Get a built-in turbine specification by name."""
    return BUILTIN_TURBINES.get(name)


# ============================================================
# Mast Data I/O
# ============================================================

def load_mast_data(filepath: str) -> Dict:
    """
    Load wind mast measurement data from a CSV file.

    Expected CSV formats accepted:

    Format 1 - Time series:
        Timestamp, Speed_80m(m/s), Dir_80m(deg), Speed_60m(m/s), ...
        2023-01-01 00:00, 8.5, 220, 7.8, ...

    Format 2 - Frequency table (wind rose):
        Dir_Bin(deg), Speed_Bin(m/s), Frequency(%)
        0, 3.5, 1.2
        0, 5.5, 2.1
        ...

    Format 3 - Weibull parameters by sector:
        Sector(deg), A(m/s), k, Frequency(%)
        0, 9.2, 2.3, 5.1
        30, 8.8, 2.1, 4.5
        ...

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict
        {
            'format': str,  # 'timeseries', 'frequency', 'weibull'
            'data': dict or pd.DataFrame,
            'heights': list,  # Measurement heights
            'lat': float, 'lon': float,  # If provided
        }
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        first_data = next(reader)

    # Auto-detect format
    headers_lower = [h.lower().strip() for h in headers]

    # Check for frequency table format
    if any('frequency' in h or 'freq' in h or 'probability' in h for h in headers_lower):
        if any('sector' in h or 'dir' in h for h in headers_lower):
            if any('weibull' in h or '_a' in h or '_k' in h for h in headers_lower):
                return _load_weibull_sectors(filepath)
            else:
                return _load_frequency_table(filepath)

    # Check for time series
    if any('timestamp' in h or 'date' in h or 'time' in h for h in headers_lower):
        return _load_timeseries(filepath)

    # Default: try as frequency table
    return _load_frequency_table(filepath)


def _load_timeseries(filepath: str) -> Dict:
    """Load time-series mast data from CSV."""
    import pandas as pd

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Detect measurement heights from column names
    heights = set()
    for col in df.columns:
        for part in col.split('_'):
            try:
                h = int(part.replace('m', '').strip())
                if 10 <= h <= 200:
                    heights.add(h)
            except ValueError:
                continue

    return {
        'format': 'timeseries',
        'data': df,
        'heights': sorted(heights),
        'lat': 0.0,
        'lon': 0.0,
    }


def _load_frequency_table(filepath: str) -> Dict:
    """Load frequency table mast data from CSV."""
    import pandas as pd

    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    dir_col = [c for c in df.columns if 'dir' in c][0]
    speed_col = [c for c in df.columns if 'speed' in c or 'ws' in c][0]
    freq_col = [c for c in df.columns if 'freq' in c or 'prob' in c or 'pct' in c][0]

    return {
        'format': 'frequency',
        'data': df,
        'dir_column': dir_col,
        'speed_column': speed_col,
        'freq_column': freq_col,
        'heights': [],
        'lat': 0.0,
        'lon': 0.0,
    }


def _load_weibull_sectors(filepath: str) -> Dict:
    """Load Weibull sector data from CSV."""
    import pandas as pd

    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    return {
        'format': 'weibull',
        'data': df,
        'heights': [],
        'lat': 0.0,
        'lon': 0.0,
    }


# ============================================================
# Wind Farm Boundary & Layout I/O
# ============================================================

def load_boundary(filepath: str) -> List[Tuple[float, float]]:
    """
    Load a wind farm boundary from a file.
    Supported formats: CSV (lat,lon), GeoJSON, KML (simplified).

    CSV format:
        latitude,longitude
        28.5,77.2
        28.5,77.5
        28.7,77.5
        28.7,77.2

    Parameters
    ----------
    filepath : str

    Returns
    -------
    list of (float, float)
        List of (latitude, longitude) vertices.
    """
    ext = Path(filepath).suffix.lower()

    if ext == '.geojson' or ext == '.json':
        return _load_geojson_boundary(filepath)
    elif ext == '.kml':
        return _load_kml_boundary(filepath)
    elif ext == '.csv':
        return _load_csv_boundary(filepath)
    else:
        return _load_csv_boundary(filepath)  # Try CSV as default


def _load_csv_boundary(filepath: str) -> List[Tuple[float, float]]:
    """Load boundary from CSV."""
    points = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    lat = float(row[0].strip())
                    lon = float(row[1].strip())
                    points.append((lat, lon))
                except ValueError:
                    continue
    return points


def _load_geojson_boundary(filepath: str) -> List[Tuple[float, float]]:
    """Load boundary from GeoJSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    points = []
    if data['type'] == 'Feature':
        geom = data['geometry']
    elif data['type'] == 'FeatureCollection':
        geom = data['features'][0]['geometry']
    else:
        geom = data

    if geom['type'] in ('Polygon', 'MultiPolygon'):
        coords = geom['coordinates'][0] if geom['type'] == 'Polygon' else geom['coordinates'][0][0]
        for lon, lat in coords:
            points.append((lat, lon))

    return points


def _load_kml_boundary(filepath: str) -> List[Tuple[float, float]]:
    """Load boundary from KML (simplified parser)."""
    from xml.etree import ElementTree as ET

    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    points = []
    for coord_text in root.findall('.//kml:coordinates', ns):
        coords = coord_text.text.strip().split()
        for coord in coords:
            parts = coord.split(',')
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                points.append((lat, lon))

    return points


def load_wtg_layout(filepath: str) -> List[Dict]:
    """
    Load existing WTG layout from a file.

    CSV format:
        Name,Latitude,Longitude,HubHeight(m)
        WTG01,28.55,77.25,105
        WTG02,28.55,77.35,105

    Parameters
    ----------
    filepath : str

    Returns
    -------
    list of dict
        Each dict: {'name': str, 'lat': float, 'lon': float, 'hub_height': float}
    """
    wtgs = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wtgs.append({
                'name': row.get('Name', row.get('name', 'WTG')),
                'lat': float(row.get('Latitude', row.get('lat', row.get('Lat', 0)))),
                'lon': float(row.get('Longitude', row.get('lon', row.get('Lon', 0)))),
                'hub_height': float(row.get('HubHeight(m)', row.get('hub_height', 80))),
            })
    return wtgs


# ============================================================
# Export Functions
# ============================================================

def export_results_to_csv(results: Dict, filepath: str):
    """
    Export AEP results to a CSV file.

    Parameters
    ----------
    results : dict
        Computed AEP results.
    filepath : str
        Output file path.
    """
    wtg_data = results.get('wtg_results', [])
    if not wtg_data:
        return

    fieldnames = list(wtg_data[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wtg_data)


def export_layout_to_csv(wtg_positions: List[Dict], filepath: str):
    """Export WTG layout positions to CSV."""
    if not wtg_positions:
        return

    fieldnames = ['name', 'latitude', 'longitude', 'hub_height_m', 'x_utm', 'y_utm']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wtg_positions)


def save_project(project_data: Dict, filepath: str):
    """Save project state to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    with open(filepath, 'w') as f:
        json.dump(project_data, f, indent=2, default=convert)


def load_project(filepath: str) -> Dict:
    """Load project state from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
