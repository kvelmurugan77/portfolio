# YawScout — SCADA Yaw Screening Studio

A browser-only desktop study that ranks wind turbine generators (WTGs) for **nacelle-mounted LiDAR confirmation** of a suspected yaw issue.

Launch from the portfolio tools page or open `yaw-screening-studio.html` directly in a current browser. SCADA files are parsed locally; no SCADA data is uploaded to a server.

## Purpose

Use the tool before a limited nacelle-LiDAR campaign to identify the WTG with the strongest and most credible **SCADA yaw-related signal**. Its output is a LiDAR targeting priority list, not a yaw correction.

## OEM and SCADA export compatibility

YawScout is designed around a **canonical SCADA schema**, not a single OEM export. It has header-alias profiles for common Vestas, Envision, Nordex / Acciona, Siemens Gamesa, Suzlon, Goldwind, Inox Wind and GE Vernova / GE signal families, plus a generic profile for other OEMs and customer historians.

Profiles only accelerate the initial mapping. Tag names, status codes, aggregation conventions, unit scaling and direction references vary by turbine generation, SCADA release, owner data lake and site configuration, so the mapping screen must always be checked against the OEM signal dictionary.

### Supported export layouts

- **Long / tall export:** one turbine-interval record per row, with an Asset / WTG column.
- **One turbine per file:** upload multiple files; if the asset column is absent, YawScout can derive a WTG label from the filename or use a supplied fallback label.
- **Wide matrix export:** one timestamp per row with signals repeated in columns such as `WTG01_Active_Power`, `WTG01_Wind_Speed`, etc. The tool expands this into canonical long records. The editable WTG-ID pattern supports non-standard site naming.
- **Delimited files:** comma, semicolon, tab and pipe delimiters; configurable header row for exports containing metadata lines.
- **Numeric / date variations:** decimal-comma and thousands-separated values; Excel serial date/time; ISO, day-first or month-first date formats; power in W/kW/MW; wind speed in m/s or km/h; directions in degrees or radians.

Auto-detection is intended as a convenience. If the automatic profile, data layout or units are not unequivocally correct, explicitly select the OEM profile / unit and remap the columns.

## Large SCADA / million-row processing

YawScout now includes a **bounded-memory streaming engine** for large browser-only studies. By default it switches to this mode when the selected files exceed **40 MB**; it can also be forced from the import panel.

### How it works

1. The browser reads only a small local preview so the engineer can select the OEM profile, verify the schema, units, filters and date convention.
2. A dedicated Web Worker reads the local file(s) in 4 MB chunks. It makes two passes: first, it creates turbine and fleet wind-speed-bin reference aggregates; second, it accumulates the yaw, power-response, directional-sector, monthly-persistence and available fleet-consensus evidence.
3. The worker returns compact per-WTG aggregates only. It does **not** send or retain millions of raw records in the visible browser page.

This makes full-farm million-row and GB-scale **desktop screening** practical without uploading SCADA data. It is deliberately an aggregate screening workflow: it cannot offer a raw-row table, raw scatter plot or a forensic re-filter without reading the local files again.

### Practical operating guidance

- Use a current 64-bit Chrome, Edge or Firefox browser on a workstation with adequate available RAM; close memory-intensive tabs before loading GB-scale data.
- Keep each export in a consistent encoding and SCADA schema. For wide files, choose the OEM profile and WTG-ID extraction pattern before upload.
- A long/tall export sorted by timestamp is best for the fleet wind-speed cross-check. One-WTG-per-file exports still run, but a fleet timestamp-consensus bias may be unavailable; the output flags this rather than silently inventing a comparison.
- The streaming power-response baseline uses mean values within wind-speed bins, whereas small in-memory studies use medians. It is suitable for WTG prioritisation, not a replacement for an IEC power-performance analysis.
- A browser may still be constrained by corporate endpoint policy, available browser memory or exceptionally malformed/heterogeneous files. For a failed stream, split by month/year or prepare a consistent long-format export — do not force raw GB data into in-memory mode.

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
3. Select the matching OEM profile (or retain generic) and verify the suggested signal mappings against the SCADA tag list.
4. Confirm export units. Explicitly choose W/kW/MW, m/s/km/h or radians when auto-detection is uncertain.
5. Select a below-rated, normal-operation wind-speed band. The default is 5–10 m/s.
6. Add normal/full operation status values and map a curtailment flag if they are available.
7. Review data coverage, turbine outages, known wakes, terrain-flow effects and sensor maintenance history independently.

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
