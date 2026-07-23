# Wind Farm AEP Studio

A **focused clone** of the WindFlow Pro workflow for quick wind-farm AEP screening.

## What you can do

1. **Define the farm area**
   - Upload **boundary**: KML / KMZ / GeoJSON / CSV (`lon,lat`)
   - Upload **layout**: CSV / KML / KMZ / GeoJSON turbine points  
   - Or enter **SW–NE corners** (left-bottom & top-right)  
   - Or **center lat/lon + radius km**
2. **Choose WTG**
   - Presets (EN-156 140/120 m, V150, V163, N163, GE158, SG170)  
   - Or upload **power curve CSV** (`ws_mps, power_kW [, ct]`)
3. **Auto maps**
   - **Terrain / contours** via Open-Meteo (+ OpenTopo fallback)  
   - **Roughness** via OSM land use → z₀ + directional rose  
4. **Wind climate**
   - **ERA5 / ERA5T** (Open-Meteo archive)  
   - **GWA 4.0** `.lib` API (direct + CORS fallbacks; or upload `.lib`)  
   - **Site mast / other mesoscale** CSV upload  
5. **AEP**
   - Log-law hub-height extrapolation  
   - Orography (WFP61 spectral BZ when available)  
   - Bastankhah Gaussian wake + loss tree  
   - Export report, per-turbine CSV, sectors, layout, boundary  

## Run locally

```bash
cd windfarm-aep-studio
python3 -m http.server 8855
# open http://localhost:8855/
```

Files needed in the same folder:

- `index.html`
- `app.js`
- `wasp_bz_engine_v61.js` (spectral orography engine)

## Typical workflow

1. Click **Load Hatalageri demo boundary** (or upload your KML/CSV).  
2. Set WTG preset / HH / power curve.  
3. Optional: generate grid if you have no layout.  
4. **Run full AEP** (auto-downloads terrain, roughness, ERA5 if wind missing).  
5. **Export results**.

Or step through: Download terrain → roughness → ERA5 or GWA → Run full AEP.

## Notes / limits

| Item | Behaviour |
|------|-----------|
| GWA in browser | May hit CORS; app tries direct API then proxies. Upload `.lib` if blocked. |
| Accuracy | **Screening only**. Not bankable without on-site mast + certified tools. |
| OSM roughness | Good default; edit z₀ default if land cover is wrong. |
| vs full WindFlow Pro | Leaner UI; same class of physics (maps + climate + oro + wake). |

## Deploy (GitHub Pages)

Copy the three files into your portfolio repo (e.g. `aep-studio/`) and link from `index.html`.

```text
windfarm-aep-studio/
  index.html
  app.js
  wasp_bz_engine_v61.js
  README.md
```
