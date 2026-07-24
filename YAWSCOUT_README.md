# YawScout — SCADA Yaw Screening Studio

A browser-only desktop study that ranks wind turbine generators (WTGs) for **nacelle-mounted LiDAR confirmation** of a suspected yaw issue.

Launch from the portfolio tools page or open `yaw-screening-studio.html` directly in a current browser. SCADA files are parsed locally; no SCADA data is uploaded to a server.

## Purpose

Use the tool before a limited nacelle-LiDAR campaign to identify the WTG with the strongest and most credible **SCADA yaw-related signal**. Its output is a LiDAR targeting priority list, not a yaw correction.

## Required SCADA signals

| Tool field | Typical signal names | Unit |
|---|---|---:|
| Timestamp | `Timestamp`, `PCTimeStamp`, `DateTime` | date/time |
| WTG / Asset ID | `AssetName`, `Turbine_ID`, `WTG` | text |
| Wind speed | `Amb_WindSpeed_Avg`, `WindSpeed` | m/s |
| Active power | `Grd_Prod_Pwr_Avg`, `Active_Power` | kW |
| Wind direction | `Amb_WindDir_Abs_Avg`, `WindDirection` | degrees |
| Nacelle direction | `Nac_Direction_Avg`, `NacelleDirection` | degrees |

Recommended optional fields:

- **Status code** — define normal/full-production values in the configuration (for example `RUN`, `1`, or an OEM-specific production state).
- **Curtailment / derate flag** — records with non-zero / true values are excluded.
- **Pitch angle** — high-pitch records can be excluded to retain below-rated operation.

## Recommended preparation

1. Use at least several months of 10-minute SCADA where possible.
2. Confirm the OEM's direction reference and status-code definitions before running the tool.
3. Select a below-rated, normal-operation wind-speed band. The default is 5–10 m/s.
4. Add normal/full operation status values and map a curtailment flag if they are available.
5. Review data coverage, turbine outages, known wakes, terrain-flow effects and sensor maintenance history independently.

## How the ranking works

For every WTG, YawScout evaluates:

1. **Persistent SCADA direction offset** — circular mean of `nacelle direction − wind direction` during filtered records.
2. **Power-versus-offset response** — power is normalised within each turbine's 0.5 m/s wind-speed bin, then assessed against direction offset. A fitted or observed power peak is used only as corroboration when enough offset-bin data exists.
3. **Persistence and data support** — record count, multi-month repeatability, coverage of offset bins, and agreement between static and power-response signals.
4. **Confounding checks** — wind-speed bias relative to concurrent fleet measurements and directional-sector residual spread. These reduce the confidence/priority when large.

The result is a **screening score** and an evidence-confidence score. `High priority` means the configured yaw threshold is exceeded and the SCADA signals have sufficient corroboration. `Review` means there is a material indication but data quality, signal disagreement, or a confounder needs engineering review. `Low priority` is not proof of correct alignment.

## Important limitations

- A SCADA angle is **not** a physical rotor-to-inflow yaw correction. Wind-vane zero error, nacelle-position reference error, wake, terrain, curtailment and control behaviour can create similar patterns.
- Directional power effects can be aerodynamic, electrical, site-specific, or wake-related rather than yaw-related.
- Do not change yaw-controller offsets based solely on this desktop study.
- Confirm the selected WTG with nacelle-mounted LiDAR and follow the OEM/site safety, warranty and control-change process.

## Outputs

- Ranked turbine table with yaw-related signal, confidence, wind-speed cross-check, data flags and LiDAR priority.
- Power-residual versus SCADA-direction-offset plot for the selected WTG.
- Direction-sector residual plot to highlight potential wake / terrain confounding.
- CSV ranking export.
- Self-contained HTML **LiDAR Targeting Brief** for the campaign decision record.

## Demo

Click **Load synthetic demo**. It contains injected direction-offset signals on `WTG05` and `WTG06`; `WTG05` should normally rank first under the default settings.
