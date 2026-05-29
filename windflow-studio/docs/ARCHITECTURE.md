# Architecture

Data flow:

1. Layout -> turbine coordinates
2. Terrain/roughness downloads -> local site effects
3. GWA or ERA5 -> wind climate
4. WAsP-like site ratio = roughness ratio × directional terrain speed-up
5. Wake model -> gross/wake/net AEP
6. ERA5 time series -> hourly/period production
7. Report/CSV export

Future upgrade points:
- WAsP .map/.lib import/export
- directional roughness fetch integration
- shelter obstacles
- sector-wise RIX
- desktop EXE wrapper with Tauri/Electron
