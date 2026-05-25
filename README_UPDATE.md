# WindFlow Pro GitHub Update

This package contains the updated `windflow-pro.html` for the portfolio GitHub Pages site.

## What is included

- WAsP-like modelling fixes from the previous update:
  - per-sector / per-turbine site ratios
  - local OSM roughness use
  - bilinear terrain interpolation
  - directional terrain speed-up approximation
  - improved OWC → GWC → SWC chain
  - slope-based RIX estimate
  - corrected density handling in power LUT
  - time-series uses direction-dependent site ratios

- Terrain download and contour performance optimisations:
  - elevation request cache for repeated downloads
  - bounded concurrent Open-Meteo elevation requests with progress
  - retries for failed elevation batches
  - contour level cap to avoid excessive rendering workload
  - batched canvas contour drawing per level instead of per segment
  - removed continuous redraw during map pan/zoom; redraw now happens after move/zoom ends
  - UI yields during terrain processing to reduce freezing

## How to update GitHub

Replace the existing repository file:

```text
windflow-pro.html
```

with the `windflow-pro.html` in this package, then commit to `main`.

Example:

```bash
git clone https://github.com/kvelmurugan77/portfolio.git
cd portfolio
cp /path/to/this/package/windflow-pro.html ./windflow-pro.html
git add windflow-pro.html
git commit -m "Improve WAsP modelling and optimise terrain contours"
git push origin main
```

GitHub Pages should update the live URL after a short delay.
