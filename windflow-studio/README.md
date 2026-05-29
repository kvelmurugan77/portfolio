# WindFlow Studio v1.6 Mast Time-Series

A production-structured browser MVP with improved sector roughness and terrain flow for an open WAsP-like / WindPRO-like wind resource workflow.

## Run
Open `index.html` in a browser. For live data downloads, internet access is required.

## Modules
- `src/js/gwa.js` — Global Wind Atlas GWC download and sector Weibull interpolation
- `src/js/era5.js` — ERA5/ERA5T time-series download via Open-Meteo
- `src/js/terrain.js` — optimized DEM download and contour generation
- `src/js/roughness.js` — OSM roughness download and local z0 lookup
- `src/js/flow.js` — WAsP-like vertical/horizontal extrapolation and AEP
- `src/js/wake.js` — Jensen/PARK wake model
- `src/js/timeseries.js` — WindPRO-like production time series
- `src/js/report.js` — HTML/CSV exports

## Scope
This is not a reverse-engineered copy of WAsP/WindPRO. It is an open implementation of a similar engineering workflow for screening and pre-feasibility.

## v1.1 quality upgrades
- Sector-wise upstream effective roughness.
- Multi-scale directional terrain speed-up.
- Optional calibration hooks for WAsP/mast/LiDAR benchmarking.
- RIX calculation.


## Clean version note
This package intentionally removes the later Option C experimental changes:
- no FFT/spectral flow solver
- no WAsP benchmark calibration workflow
- no calibration.json requirement

It keeps the practical WAsP-like browser workflow:
- GWA point climate
- ERA5 time series
- terrain download and contours
- OSM roughness
- upstream sector roughness approximation
- multi-scale directional terrain speed-up
- AEP and time-series modules

## v1.4 six WAsP-closeness adjustments
- LT mast climate CSV import.
- mast-to-hub vertical extrapolation.
- direction/turbine horizontal site ratios.
- upstream effective roughness fetch.
- multi-scale orographic speed-up.
- obstacle shelter and wake model options.

## v1.5 five closeness upgrades
- improved orographic slope+curvature response
- roughness-change internal-boundary-layer approximation
- explicit OWC/GWC reporting metadata
- improved shelter attenuation
- power/CT curve import and wake-combination options

## v1.6 mast/Windographer time-series
- Flexible mast time-series CSV import.
- Uses imported mast TS as preferred source for WindPRO-like time-series.
- Calculates per-WTG and wind-farm time-series production.
- Exports per-WTG time-series CSV.
