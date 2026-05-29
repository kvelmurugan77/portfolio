# Mast / Windographer Time-Series Import — v1.6

The tool can now import long-term corrected mast time series exported from Windographer or similar tools.

## Accepted columns
The parser is flexible and searches for:
- time: `Timestamp`, `DateTime`, or separate `Date` + `Time`
- wind speed: `Wind Speed`, `WS`, `Speed`
- wind direction: `Wind Direction`, `WD`, `Dir`, `Direction`
- optional: `Temperature`, `Pressure`

## Workflow
1. Set mast height in Project panel.
2. Import mast/Windographer time series.
3. Load/generate layout.
4. Download terrain/roughness if required.
5. Run WindPRO-like Time Series.
6. Export per-WTG hourly/step CSV.

## Output
The model calculates:
- per-WTG free wind speed
- wake-adjusted wind speed
- gross power
- net power
- per-WTG energy summary
- wind-farm total energy summary
