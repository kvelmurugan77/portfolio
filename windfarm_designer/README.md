# WindFarm Designer Pro

## Wind Farm Design & Energy Assessment Desktop Application

**Version:** 1.0.0  
**License:** MIT  
**Platform:** Windows / macOS / Linux

---

## Overview

WindFarm Designer Pro is a comprehensive desktop application for wind energy
professionals that streamlines the entire wind farm development workflow. From
site assessment and data acquisition to layout optimization and energy yield
prediction, it provides an integrated environment similar to WindPRO.

### Key Features

| Feature | Description |
|---------|-------------|
| **Boundary Import** | Import wind farm boundaries from CSV, GeoJSON, or KML files, or define by coordinates |
| **SRTM1 Terrain Data** | Download 30m-resolution DEM tiles from NASA/AWS for your site + 20km buffer |
| **Roughness Data** | Download land cover and convert to surface roughness (z0) maps from Copernicus, ESA, or OSM |
| **Global Wind Atlas** | Fetch mesoscale wind resource data from DTU's Global Wind Atlas API |
| **Mast Data Import** | Import time-series, frequency table, or Weibull-parameter wind mast measurements |
| **Layout Optimization** | Auto-generate optimized turbine layouts using Grid, Greedy, PSO, or Genetic Algorithms |
| **Wind Flow Modeling** | Compute terrain-induced speed-up and roughness corrections at each turbine position |
| **Wake Effects** | Calculate wake losses using Jensen (Park), Frandsen, or Ainslie models with 12-sector analysis |
| **AEP Calculation** | Compute Annual Energy Production with Weibull-based energy integration |
| **Loss Factors** | Configure availability, electrical, curtailment, environmental, and other losses |
| **Results Export** | Export detailed per-turbine AEP results to CSV or PDF reports |

### Built-in Turbine Library

- Vestas V150-4.2 (4.2 MW)
- Siemens Gamesa SG 14-222 DD (14 MW)
- GE Haliade-X 13 (13 MW)
- NREL 5MW Reference
- Enercon E-115 E4 (3.2 MW)
- Goldwind GW 154-6.7 (6.7 MW)
- Nordex N163-5.X (5 MW)
- Vestas V164-9.5 (9.5 MW)
- GE 2.75-120 (2.75 MW)
- Siemens SWT-3.6-120 (3.6 MW)

---

## Quick Start

### 1. Install Dependencies

```bash
cd windfarm_designer
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

### 3. Workflow

1. **Project Setup** — Define your wind farm boundary, desired capacity, and select a turbine model
2. **Terrain Data** — Download SRTM1 elevation data (DEM, slope, aspect)
3. **Roughness Data** — Download land cover data and generate roughness maps
4. **Wind Resource** — Download Global Wind Atlas data or import mast measurements
5. **Layout Optimizer** — Choose an algorithm and generate an optimized turbine layout
6. **Wind Flow Model** — Run terrain and roughness corrections
7. **AEP Results** — Calculate energy production, configure losses, and export results

---

## Building the .exe

### Prerequisites
- Python 3.8+ with all dependencies installed
- PyInstaller: `pip install pyinstaller`

### Build Commands

```bash
# Default build (directory mode, recommended)
python build_exe.py

# Single-file executable
python build_exe.py --onefile

# Include console window (for debugging)
python build_exe.py --console

# Clean previous build first
python build_exe.py --clean

# Debug build with symbols
python build_exe.py --debug
```

### Output
- `dist/WindFarm_Designer_Pro/` — Application directory
- `dist/WindFarm_Designer_Pro.exe` — Main executable (Windows)

---

## Project Structure

```
windfarm_designer/
├── main.py                     # Application entry point
├── build_exe.py                # PyInstaller build script
├── requirements.txt            # Python dependencies
├── requirements-optional.txt   # Optional dependencies
├── version.txt                 # Version info for .exe
└── src/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   ├── srtm_downloader.py      # SRTM1 terrain data download
    │   ├── roughness_downloader.py # Land cover & roughness data
    │   ├── gwa_downloader.py       # Global Wind Atlas data
    │   ├── layout_optimizer.py     # Layout optimization algorithms
    │   ├── wake_model.py           # Wake effect models
    │   ├── aep_calculator.py       # AEP calculation engine
    │   └── wind_flow_model.py      # Wind flow modeling
    ├── gui/
    │   ├── __init__.py
    │   ├── main_window.py          # Main application window
    │   ├── project_tab.py          # Project setup tab
    │   ├── terrain_tab.py          # Terrain data tab
    │   ├── roughness_tab.py        # Roughness data tab
    │   ├── wind_tab.py             # Wind resource tab
    │   ├── layout_tab.py           # Layout optimizer tab
    │   ├── flow_tab.py             # Wind flow model tab
    │   └── results_tab.py          # AEP results tab
    └── utils/
        ├── __init__.py
        ├── geo_utils.py            # Geographic utilities
        └── data_utils.py           # Data I/O utilities
```

---

## Data Sources

| Data | Source | Resolution | Coverage |
|------|--------|------------|----------|
| SRTM1 DEM | NASA/USGS via AWS | ~30m (1 arc-sec) | Global (56S-60N) |
| Land Cover | Copernicus CCI / ESA WorldCover | 300m / 10m | Global |
| Roughness | OpenStreetMap (fallback) | ~90m | Global |
| Wind Resource | DTU Global Wind Atlas | ~1km grid | Global |

---

## Technical Details

### Wake Models
- **Jensen (Park)**: Linear wake expansion with configurable decay constant (k = 0.04-0.075)
- **Frandsen**: Wake model accounting for turbulence intensity effects
- **Ainslie**: Eddy viscosity model with Gaussian wake profile

### Optimization Algorithms
- **Grid**: Regular grid with optimal rotation angle
- **Greedy**: Sequential placement with wake-aware scoring
- **PSO**: Particle Swarm Optimization with adaptive inertia
- **GA**: Genetic Algorithm with tournament selection and SBX crossover

### Wind Flow Model
- Jackson-Hunt linearized theory for terrain speed-up
- Internal boundary layer model for roughness changes
- 12 or 36 sector directional analysis
- Flow separation detection for steep terrain

---

## License

MIT License. See LICENSE file for details.

## Acknowledgments

- SRTM data: NASA Jet Propulsion Laboratory / USGS
- Land Cover: Copernicus Climate Change Service / ESA
- Wind Resource: DTU Wind Energy (Global Wind Atlas)
- Terrain visualization: NASA Blue Marble
