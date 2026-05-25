# Five Additional WAsP-Closeness Adjustments — v1.5

This version adds the five items previously identified as the main remaining gaps.

1. **Orographic flow**: improved clean-mode multi-scale slope + curvature terrain response.
2. **Roughness-change model**: upstream effective roughness plus a simple internal-boundary-layer relaxation factor.
3. **Generalized wind climate chain**: AEP now uses imported LT mast climate when present, with explicit OWC/GWC reporting metadata.
4. **Shelter/obstacle model**: obstacle shelter now includes hub-height attenuation, lateral width, porosity and downstream decay.
5. **Wake implementation**: imported power/CT curve support, selectable Jensen/Eddy/PARK2-style wake models and RSS/linear wake combination.

These changes improve consistency for same-input comparison with WAsP, but do not make the tool proprietary-WAsP equivalent.
