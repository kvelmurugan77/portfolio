# WindFlow Pro 61.1 — WAsP-class free alternative (tested)

Local, improved build of [WindFlow Pro](https://kvelmurugan77.github.io/portfolio/windflow-pro.html) with a **spectral WAsP-style BZ flow engine**, bug fixes, and automated end-to-end tests.

> **Honest scope:** This is a **practical WAsP-class workflow** (OWC→GWC→SWC, oro, roughness, wake, AEP). It is **not** a licensed DTU WAsP binary and will not bit-match commercial WAsP on every site. For investment/bankable work, still validate with measurements and, where required, a certified consultant tool.

---

## Quick start

```bash
cd windflow-pro-improved
python3 -m http.server 8765
# open http://localhost:8765/windflow-pro.html
```

Look for the green badge **WFP61 Spectral BZ** and log lines `WFP61…`.

### Deploy to GitHub Pages

Copy into your portfolio repo (same folder as the HTML):

```text
windflow-pro.html
wasp_bz_engine_v61.js
```

Commit & push. The HTML loads the engine via:

```html
<script src="wasp_bz_engine_v61.js"></script>
```

---

## What was tested (E2E)

Command:

```bash
python3 -m http.server 8765 &
python3 tests/e2e_windflow_pro.py
node tests/test_wfp61_flow.js
```

**Latest result: 16/16 E2E passed, physics unit tests passed.**

| Step | Result |
|------|--------|
| Page load | Pass |
| Core globals (`S`, `BZ`, `calcAEP`, `runWAsPAnalysis`) | Pass |
| WFP61 engine install (`runBZModel` → `runBZModelV61`) | Pass |
| Synthetic site (9 WTGs, 2000 wind samples, hill DEM) | Pass |
| Gaussian-hill spectral self-test | Pass |
| Spectral `runBZModel` | Pass |
| `calcAEP` gross/net/wake/CF | Pass |
| Full `runWAsPAnalysis` | Pass |
| Step navigation 1–5 | Pass |
| Power curve present | Pass |
| Export/save helpers present | Pass |
| **Flat terrain identity** (all SU=1, identical free WS) | Pass |
| No fatal page JS errors | Pass |

Console `400` noise is from optional map/tile/API calls in headless mode — not fatal.

---

## Bugs fixed in 61.1

| Issue | Impact | Fix |
|-------|--------|-----|
| `runWAsPFlow` used `await` but was **not** `async` | Could throw at runtime | Declared `async function runWAsPFlow` |
| Terrain gate checked only `S.terrain.elev` while download stored **`grid`** | WAsP Analysis blocked after terrain download | Gate accepts `grid` / `elev` / `elevations` / `flat` |
| Terrain object missing `elev` flat array | Same gate / helpers fragile | Assignments now set `elev` from grid |
| Spectral BZ helpers existed but **`runBZModel` never used FFT** | Not WAsP-class oro | `wasp_bz_engine_v61.js` overrides with spectral IBZ |
| Host `bzG` sign could invert crest response | Hills → slowdown | Sign-corrected \(G(K,z)\) |
| Engine comment contained raw `</script>` | If inlined, HTML parser cut script | Load as **external** file; comment sanitized |
| Wrong inject at first `</body>` (inside report template string) | Entire app JS truncated | Inject only before **last** `</body>` |
| Engine looked for `window.S` / `window.BZ` (`let`/`const` not on window) | Install never ran | Install keys off `runBZModel`; scope-safe access |

---

## WAsP-like capability map

| WAsP-style feature | In this build |
|--------------------|---------------|
| Site / layout / PC | Yes |
| Wind import + reanalysis download | Yes (ERA5, GWA, MERRA2, …) |
| MCP / vertical extrapolation | Yes |
| Terrain DEM | Yes (Open-Meteo / OpenTopo / SRTM paths) |
| Roughness rose / IBL | Yes |
| **Spectral oro BZ (IBZ-class)** | **Yes — WFP61 primary** |
| OWC → GWC → SWC | Yes |
| Wake (Jensen / Bastankhah) | Yes |
| AEP + losses + uncertainty UI | Yes |
| Export tab/WRG/project/report | Yes |
| Certified WAsP project file identity | No |
| Full obstacle / forest canopy like WAsP | Partial |
| Bankable without mast | **No** (same as physics requires) |

---

## Files

| File | Role |
|------|------|
| `windflow-pro.html` | App v61.1 + fixes + engine script tag |
| `wasp_bz_engine_v61.js` | Spectral BZ engine (must sit next to HTML) |
| `tests/e2e_windflow_pro.py` | Playwright full pipeline test |
| `tests/test_wfp61_flow.js` | Node physics tests |
| `tests/e2e_report.json` | Last E2E report |
| `README_WFP61.md` | This file |

---

## Recommended workflow (closest to WAsP)

1. **Step 1** — Boundary + turbine layout + power curve  
2. **Step 2** — Mast CSV if you have it; else ERA5 + plan MCP  
3. **Step 3** — VE to hub height; MCP to long-term  
4. **Step 4** — Download terrain + land-cover roughness rose  
5. **Step 5** — **RUN WAsP ANALYSIS** (WFP61 spectral BZ runs inside)  
6. Check per-WTG free WS, RIX, wake, net AEP  
7. Export WRG/tab/report for records  

On **flat + uniform z₀**, free-stream WS should be essentially identical across WTGs (E2E verified). On hills, ridge WTGs should exceed valley WTGs.

---

## Limits (please read)

- Not a legal/technical substitute for **DTU WAsP** where contracts require it.  
- Linearized BZ **degrades** when RIX is high (steep complex terrain) — same class limit as WAsP IBZ.  
- Quality of **roughness + mast** dominates accuracy more than solver branding.  
- Always keep a measurement campaign for decisions that move money.

---

*WindFlow Pro 61.1 · WFP61 spectral BZ · E2E 16/16 pass*


## Terrain & Roughness status (live-tested)

| Component | Status | Notes |
|-----------|--------|-------|
| Terrain Open-Meteo | **Works** | 20×20 Chennai test: 400/400 pts, elev 1–87 m |
| Terrain OpenTopo fallback | **Works** | mapzen/srtm30m OK |
| ESA CCI WMS | **Offline (404)** | App falls back to OSM |
| OSM Overpass roughness | **Works in browser** | Multi-endpoint fallback added in v61.1 |
| Roughness rose | **Fixed** | `xs is not defined` bug fixed; fetch distances in metres |
| Used by spectral BZ | **Yes** | terrain grid + sectorRoughRC |

Not identical to WAsP `.map` digitising: coarser point DEM + OSM landuse z0 lookup vs surveyed contour/roughness maps.
