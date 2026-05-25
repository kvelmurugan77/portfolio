# Quality and Validation Policy

## 99.99% agreement statement
A deterministic 99.99% agreement with WAsP cannot be guaranteed by any independent open tool unless:
1. WAsP's exact proprietary algorithms are licensed/used, or
2. the open model is calibrated against WAsP outputs for the same terrain/roughness/wind-climate cases, or
3. the result is calibrated against measured mast/LiDAR/SCADA data.

WindFlow Studio v1.1 therefore targets **high-quality, traceable, WAsP-like screening** rather than claiming exact WAsP replication.

## Improvements in v1.1
- Sector-wise effective roughness from upstream fetch sampling.
- Multi-scale directional terrain speed-up approximation.
- Optional sector calibration multiplier hook: `S.calibration.sectorSR[0..11]`.
- RIX calculation and warning.

## Recommended validation workflow
1. Run WAsP for 3-5 benchmark cases: flat, mixed roughness, ridge, complex terrain.
2. Export WAsP per-sector SWC/hub wind speed by WTG.
3. Calculate sector calibration multipliers:
   `cal_sector = WAsP_sector_WS / WindFlow_sector_WS`.
4. Store these in `S.calibration.sectorSR`.
5. Re-run AEP and verify wind-speed residuals.

## Realistic target after calibration
- Simple terrain: within 1-3% wind speed.
- Moderate terrain: within 2-5% wind speed.
- Complex terrain: site-specific; may require CFD or measured calibration.
