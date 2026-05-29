# Six WAsP-closeness Adjustments Added in v1.4

1. **LT mast climate import**: use sector frequency, Weibull A/k from long-term corrected mast data instead of only GWA/ERA5.
2. **Vertical extrapolation**: mast sector Weibull A is extrapolated from mast height to hub height using roughness/log-law.
3. **Horizontal extrapolation**: site ratio remains direction-dependent by turbine.
4. **Roughness-change approximation**: upstream effective roughness by directional fetch sampling is used.
5. **Orographic speed-up approximation**: multi-scale directional terrain speed-up remains active.
6. **Shelter + wake options**: obstacle shelter CSV and wake model options (Jensen/PARK, Eddy, PARK2-style) added.

This improves consistency when using the same LT mast climate and layout as WAsP, but it is still not proprietary WAsP.
