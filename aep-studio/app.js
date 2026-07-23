/* Wind Farm AEP Studio — focused WAsP-class workflow */
(() => {
  'use strict';

  // ─── State ───────────────────────────────────────────────────────────────
  const S = {
    project: 'My Wind Farm',
    boundary: [],      // [{lat,lon}, ...] ring (closed optional)
    turbines: [],      // [{lat,lon,hh,elev?}]
    terrain: null,     // {grid,ny,nx,lat0,lat1,lon0,lon1,minE,maxE,meanE}
    roughnessZones: [],
    roughnessRose: null,
    wind: null,        // {speeds, dirs, times?, height, source, meanWS}
    windSources: {},
    pc: null,
    results: null,
    map: null,
    layers: {
      boundary: null, turbines: null, elev: null, rough: null,
      speed: null, windPt: null, labels: null, elevControl: null
    },
    speedField: null, // per-turbine free WS after AEP
    windPoint: null,  // {lat,lon,source,height,meanWS}

  };

  // Expose for WFP61 spectral engine compatibility
  window.S = S;
  window.BZ = window.BZ || {
    enabled: false, speedups: [], sectorSpeedups: [], mastSectorSU: [],
    mastSU: 1, rix: 0, sectorRoughRC: [], turbRoughRC: [], engine: null,
  };
  // Minimal host shims used by wasp_bz_engine_v61.js
  window.$ = (id) => document.getElementById(id);
  window.log = (m, t) => addLog(m, t || 'i');
  window.getStabilityL = () => 1e8;
  window.psi_m = () => 0;
  window.computeRIX = (terrain) => {
    if (!terrain || !terrain.grid) return 0;
    const g = terrain.grid, ny = g.length, nx = g[0].length;
    const dLat = (terrain.lat1 - terrain.lat0) / Math.max(1, ny - 1);
    const dLon = (terrain.lon1 - terrain.lon0) / Math.max(1, nx - 1);
    const clat = (terrain.lat0 + terrain.lat1) / 2;
    const dy = dLat * 111320, dx = dLon * 111320 * Math.cos(clat * Math.PI / 180);
    let steep = 0, n = 0;
    for (let i = 1; i < ny - 1; i += 2) {
      for (let j = 1; j < nx - 1; j += 2) {
        const dzdx = (g[i][j + 1] - g[i][j - 1]) / (2 * dx);
        const dzdy = (g[i + 1][j] - g[i - 1][j]) / (2 * dy);
        const s = Math.hypot(dzdx, dzdy);
        n++; if (s > 0.3) steep++;
      }
    }
    return n ? (100 * steep / n) : 0;
  };
  window.getTerrainElevBilinear = (lat, lon) => elevAt(lat, lon);
  window.buildRoughnessSequence = (z0m, z0s) => ({ ratio: Math.sqrt(Math.max(1e-6, z0m) / Math.max(1e-6, z0s)) });
  window.getCalibrationFactors = () => ({ oroCorrection: 1, name: 'default' });
  window.setProg = (id, p, t) => { setProgress(p); if (t) addLog(t, 'i'); };

  // ─── Turbine presets ─────────────────────────────────────────────────────
  const PRESETS = {
    en156_140: {
      name: 'EN-156/3.3MW 140m', rated: 3300, D: 156, hh: 140, cutIn: 3, cutOut: 25,
      ws: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
      pw: [0,0,0,22,130,340,690,1200,1900,2620,3150,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,0],
      ct: [0,0,0,.88,.86,.84,.82,.79,.75,.67,.54,.42,.32,.25,.20,.16,.13,.11,.09,.08,.07,.06,.05,.04,.04,0],
    },
    en156_120: {
      name: 'EN-156/3.3MW 120m', rated: 3300, D: 156, hh: 120, cutIn: 3, cutOut: 25,
      ws: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
      pw: [0,0,0,22,130,340,690,1200,1900,2620,3150,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,0],
      ct: [0,0,0,.88,.86,.84,.82,.79,.75,.67,.54,.42,.32,.25,.20,.16,.13,.11,.09,.08,.07,.06,.05,.04,.04,0],
    },
    v150: {
      name: 'V150-5.6MW', rated: 5600, D: 150, hh: 120, cutIn: 3, cutOut: 25,
      ws: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
      pw: [0,0,0,85,310,680,1200,1850,2600,3400,4100,4800,5300,5550,5600,5600,5600,5600,5600,5600,5600,5600,5600,5600,5600,0],
      ct: [0,0,0,.90,.87,.84,.81,.78,.73,.66,.56,.47,.38,.31,.25,.20,.16,.13,.11,.09,.08,.07,.06,.05,.04,0],
    },
    v163: {
      name: 'V163-4.5MW', rated: 4500, D: 163, hh: 113, cutIn: 3, cutOut: 25,
      ws: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
      pw: [0,0,0,30,155,380,750,1280,1980,2780,3450,4000,4320,4480,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,0],
      ct: [0,0,0,.89,.86,.84,.81,.78,.74,.67,.57,.48,.39,.32,.26,.21,.17,.14,.12,.10,.09,.07,.06,.05,.04,0],
    },
    n163: {
      name: 'N163/6.X', rated: 6800, D: 163, hh: 148, cutIn: 3, cutOut: 25,
      ws: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
      pw: [0,0,0,110,400,870,1530,2380,3370,4450,5450,6150,6550,6750,6800,6800,6800,6800,6800,6800,6800,6800,6800,6800,6800,0],
      ct: [0,0,0,.90,.87,.84,.81,.78,.73,.66,.56,.47,.38,.31,.25,.20,.16,.13,.11,.09,.08,.07,.06,.05,.04,0],
    },
    ge158: {
      name: 'GE158-6.1MW', rated: 6100, D: 158, hh: 128, cutIn: 3, cutOut: 25,
      ws: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
      pw: [0,0,0,100,360,780,1380,2150,3050,4050,4950,5650,6000,6100,6100,6100,6100,6100,6100,6100,6100,6100,6100,6100,6100,0],
      ct: [0,0,0,.89,.86,.83,.80,.77,.72,.65,.56,.47,.38,.31,.25,.20,.16,.13,.11,.09,.08,.07,.06,.05,.04,0],
    },
    sg170: {
      name: 'SG170-7.0MW', rated: 7000, D: 170, hh: 125, cutIn: 3, cutOut: 25,
      ws: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
      pw: [0,0,0,105,380,820,1450,2250,3200,4250,5200,6050,6600,6900,7000,7000,7000,7000,7000,7000,7000,7000,7000,7000,7000,0],
      ct: [0,0,0,.89,.86,.83,.80,.77,.72,.65,.56,.47,.38,.31,.25,.20,.16,.13,.11,.09,.08,.07,.06,.05,.04,0],
    },
  };

  // ─── Utils ───────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  function addLog(msg, type = 'i') {
    const el = $('log');
    const d = document.createElement('div');
    d.className = 'log-' + type;
    d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    el.appendChild(d);
    el.scrollTop = el.scrollHeight;
  }
  function setProgress(p) { $('pb').style.width = clamp(p, 0, 100) + '%'; }
  function setStep(name, state) {
    document.querySelectorAll('#steps .step').forEach((el) => {
      if (el.dataset.s === name) {
        el.classList.remove('done', 'run');
        if (state) el.classList.add(state);
      }
    });
  }
  function chipBox(id, items) {
    $(id).innerHTML = items.map(([t, on]) => `<span class="chip ${on ? 'on' : ''}">${t}</span>`).join('');
  }
  function downloadText(filename, text, mime = 'text/plain') {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: mime }));
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }
  function mean(a) { return a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0; }
  function gamma(z) {
    // Lanczos approx for Γ(z), z>0
    const p = [676.5203681218851, -1259.1392167224028, 771.3234287776531,
      -176.6150291621406, 12.507343278686905, -0.13857109526572012, 9.984369654078991e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
    z -= 1;
    let x = 0.99999999999980993;
    for (let i = 0; i < p.length; i++) x += p[i] / (z + i + 1);
    const t = z + p.length - 0.5;
    return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
  }

  // ─── Geometry ────────────────────────────────────────────────────────────
  function pip(lon, lat, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i].lon, yi = ring[i].lat, xj = ring[j].lon, yj = ring[j].lat;
      const hit = ((yi > lat) !== (yj > lat)) && (lon < (xj - xi) * (lat - yi) / ((yj - yi) || 1e-15) + xi);
      if (hit) inside = !inside;
    }
    return inside;
  }
  function bboxOf(pts) {
    const lats = pts.map((p) => p.lat), lons = pts.map((p) => p.lon);
    return { minLat: Math.min(...lats), maxLat: Math.max(...lats), minLon: Math.min(...lons), maxLon: Math.max(...lons) };
  }
  function centerOf(pts) {
    const b = bboxOf(pts);
    return { lat: (b.minLat + b.maxLat) / 2, lon: (b.minLon + b.maxLon) / 2 };
  }
  function convexHull(points) {
    const pts = points.map((p) => ({ lat: p.lat, lon: p.lon }))
      .sort((a, b) => a.lon - b.lon || a.lat - b.lat);
    if (pts.length <= 2) return pts;
    const cross = (o, a, b) => (a.lon - o.lon) * (b.lat - o.lat) - (a.lat - o.lat) * (b.lon - o.lon);
    const lower = [];
    for (const p of pts) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) {
      const p = pts[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    upper.pop(); lower.pop();
    return lower.concat(upper);
  }
  function latLonToXY(lat, lon, lat0, lon0) {
    const x = (lon - lon0) * 111320 * Math.cos(lat0 * Math.PI / 180);
    const y = (lat - lat0) * 111320;
    return { x, y };
  }

  // ─── File parsers ────────────────────────────────────────────────────────
  function parseCSVPoints(text) {
    const lines = text.trim().split(/\r?\n/).filter((l) => l.trim() && !l.trim().startsWith('#'));
    if (!lines.length) return [];
    const sep = lines[0].includes(';') ? ';' : ',';
    let start = 0;
    let headers = lines[0].split(sep).map((h) => h.trim().toLowerCase().replace(/['"]/g, ''));
    const hasHeader = headers.some((h) => /lat|lon|lng|x|y|easting|northing/.test(h));
    if (hasHeader) start = 1; else headers = [];
    const pts = [];
    for (let i = start; i < lines.length; i++) {
      const c = lines[i].split(sep).map((x) => x.trim().replace(/['"]/g, ''));
      if (c.length < 2) continue;
      let lat, lon;
      if (hasHeader) {
        const li = headers.findIndex((h) => h === 'lat' || h === 'latitude' || h === 'y');
        const lo = headers.findIndex((h) => h === 'lon' || h === 'lng' || h === 'longitude' || h === 'x' || h === 'long');
        if (li >= 0 && lo >= 0) { lat = +c[li]; lon = +c[lo]; }
        else { lon = +c[0]; lat = +c[1]; }
      } else {
        const a = +c[0], b = +c[1];
        // Heuristic: |val|>30 more likely lon if other is lat-like, else lon,lat if first in [-180,180] and |a|>|b| often lon first for India
        if (Math.abs(a) <= 90 && Math.abs(b) <= 180 && Math.abs(b) > Math.abs(a)) { lat = a; lon = b; }
        else { lon = a; lat = b; }
      }
      if (isFinite(lat) && isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180) pts.push({ lat, lon });
    }
    return pts;
  }

  function parseKMLCoords(text) {
    const pts = [];
    const re = /<coordinates[^>]*>([\s\S]*?)<\/coordinates>/gi;
    let m;
    while ((m = re.exec(text))) {
      const block = m[1].trim().split(/[\s\n]+/).filter(Boolean);
      for (const triple of block) {
        const p = triple.split(',').map(Number);
        if (p.length >= 2 && isFinite(p[0]) && isFinite(p[1])) pts.push({ lon: p[0], lat: p[1] });
      }
    }
    return pts;
  }

  async function parseKMZ(file) {
    const zip = await JSZip.loadAsync(await file.arrayBuffer());
    const name = Object.keys(zip.files).find((n) => n.toLowerCase().endsWith('.kml'));
    if (!name) throw new Error('No KML inside KMZ');
    return zip.files[name].async('text');
  }

  function parseGeoJSON(obj) {
    const pts = [];
    const walk = (g) => {
      if (!g) return;
      if (g.type === 'FeatureCollection') g.features.forEach((f) => walk(f));
      else if (g.type === 'Feature') walk(g.geometry);
      else if (g.type === 'Point') pts.push({ lon: g.coordinates[0], lat: g.coordinates[1] });
      else if (g.type === 'MultiPoint') g.coordinates.forEach((c) => pts.push({ lon: c[0], lat: c[1] }));
      else if (g.type === 'LineString') g.coordinates.forEach((c) => pts.push({ lon: c[0], lat: c[1] }));
      else if (g.type === 'Polygon') g.coordinates[0].forEach((c) => pts.push({ lon: c[0], lat: c[1] }));
      else if (g.type === 'MultiPolygon') g.coordinates.forEach((poly) => poly[0].forEach((c) => pts.push({ lon: c[0], lat: c[1] })));
    };
    walk(obj);
    return pts;
  }

  async function readPointsFile(file, mode /* boundary|layout */) {
    const name = file.name.toLowerCase();
    let text = '';
    if (name.endsWith('.kmz')) text = await parseKMZ(file);
    else text = await file.text();

    let pts = [];
    if (name.endsWith('.geojson') || name.endsWith('.json') || text.trim().startsWith('{')) {
      pts = parseGeoJSON(JSON.parse(text));
    } else if (name.endsWith('.kml') || name.endsWith('.kmz') || name.endsWith('.xml') || text.includes('<coordinates')) {
      pts = parseKMLCoords(text);
    } else {
      pts = parseCSVPoints(text);
    }
    if (!pts.length) throw new Error('No coordinates found in ' + file.name);

    if (mode === 'boundary') {
      // If first==last ok; else close
      S.boundary = pts.slice();
      const a = S.boundary[0], b = S.boundary[S.boundary.length - 1];
      if (a.lat !== b.lat || a.lon !== b.lon) S.boundary.push({ ...a });
      addLog(`Boundary loaded: ${S.boundary.length} vertices from ${file.name}`, 'o');
    } else {
      S.turbines = pts.map((p) => ({ lat: p.lat, lon: p.lon, hh: +$('hh').value || 140 }));
      if (S.boundary.length < 3) {
        S.boundary = convexHull(S.turbines);
        const a = S.boundary[0];
        S.boundary.push({ ...a });
        addLog('Boundary auto-built as convex hull of layout', 'i');
      }
      addLog(`Layout loaded: ${S.turbines.length} turbines from ${file.name}`, 'o');
    }
    refreshSiteUI();
    redrawMap();
  }

  function parsePowerCurveCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter((l) => l.trim());
    const sep = lines[0].includes(';') ? ';' : ',';
    let i0 = 0;
    const h = lines[0].toLowerCase();
    if (/[a-z]/.test(h)) i0 = 1;
    const ws = [], pw = [], ct = [];
    for (let i = i0; i < lines.length; i++) {
      const c = lines[i].split(sep).map((x) => +x.trim());
      if (c.length >= 2 && isFinite(c[0]) && isFinite(c[1])) {
        ws.push(c[0]); pw.push(c[1]); ct.push(isFinite(c[2]) ? c[2] : guessCt(c[0], c[1]));
      }
    }
    if (ws.length < 5) throw new Error('Power curve needs ≥5 rows');
    const rated = Math.max(...pw);
    const cutIn = ws.find((v, i) => pw[i] > 0) || 3;
    const cutOut = [...ws].reverse().find((v, i) => pw[pw.length - 1 - i] > 0) || 25;
    return { name: 'Custom', rated, D: +$('D').value || 150, hh: +$('hh').value || 120, cutIn, cutOut, ws, pw, ct };
  }
  function guessCt(ws, pw) {
    if (ws < 3 || pw <= 0) return 0;
    if (ws < 8) return 0.85;
    if (ws < 12) return 0.6;
    if (ws < 18) return 0.25;
    return 0.1;
  }

  // ─── Map ─────────────────────────────────────────────────────────────────
  function initMap() {
    S.map = L.map('map', { zoomControl: true }).setView([20, 78], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18, attribution: '© OpenStreetMap',
    }).addTo(S.map);
    S.layers.elev = L.layerGroup().addTo(S.map);
    S.layers.rough = L.layerGroup().addTo(S.map);
    S.layers.boundary = L.layerGroup().addTo(S.map);
    S.layers.speed = L.layerGroup().addTo(S.map);
    S.layers.turbines = L.layerGroup().addTo(S.map);
    S.layers.windPt = L.layerGroup().addTo(S.map);
    S.layers.labels = L.layerGroup(); // off by default
    bindLayerToggles();
  }

  function bindLayerToggles() {
    const bind = (id, layer, fit) => {
      const el = $(id); if (!el) return;
      el.addEventListener('change', () => {
        if (el.checked) {
          if (!S.map.hasLayer(layer)) layer.addTo(S.map);
        } else if (S.map.hasLayer(layer)) {
          S.map.removeLayer(layer);
        }
        updateLegend();
      });
    };
    bind('lyrBoundary', S.layers.boundary);
    bind('lyrTurbines', S.layers.turbines);
    bind('lyrElev', S.layers.elev);
    bind('lyrRough', S.layers.rough);
    bind('lyrSpeed', S.layers.speed);
    bind('lyrWindPt', S.layers.windPt);
    bind('lyrLabels', S.layers.labels);
  }

  function elevColor(t) {
    // t 0..1 blue→cyan→green→yellow→red
    const stops = [
      [0.0, [11, 61, 145]],
      [0.25, [46, 204, 113]],
      [0.5, [241, 196, 15]],
      [0.75, [230, 126, 34]],
      [1.0, [192, 57, 43]],
    ];
    t = clamp(t, 0, 1);
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const t0 = stops[i - 1][0], t1 = stops[i][0];
        const u = (t - t0) / (t1 - t0 || 1);
        const c0 = stops[i - 1][1], c1 = stops[i][1];
        const r = Math.round(c0[0] + (c1[0] - c0[0]) * u);
        const g = Math.round(c0[1] + (c1[1] - c0[1]) * u);
        const b = Math.round(c0[2] + (c1[2] - c0[2]) * u);
        return `rgb(${r},${g},${b})`;
      }
    }
    return 'rgb(192,57,43)';
  }

  function z0Color(z0) {
    // water → open → farm → scrub → forest/urban
    if (z0 < 0.001) return '#1e90ff';
    if (z0 < 0.02) return '#f4d03f';
    if (z0 < 0.08) return '#58d68d';
    if (z0 < 0.3) return '#d4ac0d';
    if (z0 < 0.8) return '#af601a';
    return '#1e8449';
  }

  function wsColor(ws, wmin, wmax) {
    const t = (ws - wmin) / Math.max(0.2, wmax - wmin);
    return elevColor(clamp(t, 0, 1)); // reuse ramp
  }

  function drawElevationLayer() {
    S.layers.elev.clearLayers();
    const T = S.terrain;
    if (!T || !T.grid) return;
    const { grid, ny, nx, lat0, lat1, lon0, lon1, minE, maxE } = T;
    const dLat = (lat1 - lat0) / Math.max(1, ny - 1);
    const dLon = (lon1 - lon0) / Math.max(1, nx - 1);
    const range = Math.max(1, maxE - minE);
    // Cell heatmap (skip every other cell if large for performance)
    const step = ny > 50 ? 2 : 1;
    for (let i = 0; i < ny - 1; i += step) {
      for (let j = 0; j < nx - 1; j += step) {
        const v = grid[i][j];
        if (v == null || !isFinite(v)) continue;
        const la0 = lat0 + i * dLat, la1 = lat0 + (i + step) * dLat;
        const lo0 = lon0 + j * dLon, lo1 = lon0 + (j + step) * dLon;
        const col = elevColor((v - minE) / range);
        L.rectangle([[la0, lo0], [la1, lo1]], {
          stroke: false, fillColor: col, fillOpacity: 0.45,
          interactive: false,
        }).addTo(S.layers.elev);
      }
    }
    // Contour-like isolines via simple marching on coarse grid
    const nLevels = 8;
    const levels = [];
    for (let k = 1; k < nLevels; k++) levels.push(minE + (range * k) / nLevels);
    levels.forEach((lev) => {
      const segs = [];
      for (let i = 0; i < ny - 1; i++) {
        for (let j = 0; j < nx - 1; j++) {
          const v00 = grid[i][j], v10 = grid[i][j + 1], v01 = grid[i + 1][j], v11 = grid[i + 1][j + 1];
          if (![v00, v10, v01, v11].every((v) => v != null && isFinite(v))) continue;
          const corners = [
            [0, 0, v00], [1, 0, v10], [1, 1, v11], [0, 1, v01],
          ];
          const pts = [];
          for (let e = 0; e < 4; e++) {
            const [x1, y1, a] = corners[e];
            const [x2, y2, b] = corners[(e + 1) % 4];
            if ((a < lev && b >= lev) || (a >= lev && b < lev)) {
              const t = (lev - a) / ((b - a) || 1e-9);
              const x = x1 + (x2 - x1) * t;
              const y = y1 + (y2 - y1) * t;
              pts.push([lat0 + (i + y) * dLat, lon0 + (j + x) * dLon]);
            }
          }
          if (pts.length >= 2) segs.push(pts.slice(0, 2));
        }
      }
      segs.forEach((pair) => {
        L.polyline(pair, { color: 'rgba(255,255,255,0.55)', weight: 1.2, opacity: 0.85, interactive: false })
          .addTo(S.layers.elev);
      });
    });
    // outline domain
    L.rectangle([[lat0, lon0], [lat1, lon1]], {
      color: '#5dade2', weight: 1, dashArray: '4 3', fill: false, interactive: false,
    }).addTo(S.layers.elev);
  }

  function drawRoughnessLayer() {
    S.layers.rough.clearLayers();
    if (!S.roughnessZones || !S.roughnessZones.length) return;
    S.roughnessZones.forEach((z) => {
      if (!z.pts || z.pts.length < 3) return;
      const latlngs = z.pts.map((p) => [p.lat, p.lon]);
      const col = z0Color(z.z0);
      L.polygon(latlngs, {
        color: col, weight: 1, fillColor: col, fillOpacity: 0.28,
      }).bindTooltip(`${z.lu || 'zone'}<br>z₀ = ${Number(z.z0).toPrecision(3)} m`)
        .addTo(S.layers.rough);
    });
  }

  function drawWindPointLayer() {
    S.layers.windPt.clearLayers();
    const wp = S.windPoint || (S.wind && S.wind.lat != null ? {
      lat: S.wind.lat, lon: S.wind.lon, source: S.wind.source,
      height: S.wind.height, meanWS: S.wind.meanWS,
    } : null);
    if (!wp || wp.lat == null || wp.lon == null || !isFinite(wp.lat) || !isFinite(wp.lon)) return;
    const icon = L.divIcon({
      className: '',
      html: `<div style="width:16px;height:16px;border-radius:50%;background:#f5b942;border:2px solid #fff;box-shadow:0 0 0 3px rgba(245,185,66,.35)"></div>`,
      iconSize: [16, 16], iconAnchor: [8, 8],
    });
    L.marker([wp.lat, wp.lon], { icon, zIndexOffset: 1000 })
      .bindPopup(
        `<b>Wind data location</b><br>Source: ${wp.source || '—'}<br>` +
        `Lat/Lon: ${Number(wp.lat).toFixed(5)}, ${Number(wp.lon).toFixed(5)}<br>` +
        `Height: ${wp.height != null ? wp.height + ' m' : '—'}<br>` +
        `Mean WS: ${wp.meanWS != null ? Number(wp.meanWS).toFixed(2) + ' m/s' : '—'}`
      )
      .addTo(S.layers.windPt);
    L.circle([wp.lat, wp.lon], {
      radius: 400, color: '#f5b942', weight: 1, dashArray: '4 4',
      fillColor: '#f5b942', fillOpacity: 0.06, interactive: false,
    }).addTo(S.layers.windPt);
  }

  function drawSpeedLayer() {
    S.layers.speed.clearLayers();
    S.layers.labels.clearLayers();
    if (!S.speedField || !S.speedField.length || !S.turbines.length) return;
    const vals = S.speedField.filter((v) => v != null && isFinite(v));
    if (!vals.length) return;
    const wmin = Math.min(...vals), wmax = Math.max(...vals);
    S.turbines.forEach((t, i) => {
      const ws = S.speedField[i];
      if (ws == null || !isFinite(ws)) return;
      const col = wsColor(ws, wmin, wmax);
      const r = 6 + 10 * ((ws - wmin) / Math.max(0.2, wmax - wmin));
      L.circleMarker([t.lat, t.lon], {
        radius: r, color: '#fff', weight: 1, fillColor: col, fillOpacity: 0.85,
      }).bindTooltip(
        `<b>T${i + 1}</b><br>Hub WS: <b>${ws.toFixed(2)} m/s</b>` +
        (t.elev != null ? `<br>Elev: ${Number(t.elev).toFixed(0)} m` : '')
      ).addTo(S.layers.speed);

      const lab = L.divIcon({
        className: 'wtg-label',
        html: `${ws.toFixed(1)}`,
        iconSize: [28, 14], iconAnchor: [-6, 8],
      });
      L.marker([t.lat, t.lon], { icon: lab, interactive: false }).addTo(S.layers.labels);
    });
  }

  function updateMapStats() {
    const el = $('mapStats'); if (!el) return;
    const bits = [];
    if (S.terrain) bits.push(`Elev ${S.terrain.minE.toFixed(0)}–${S.terrain.maxE.toFixed(0)} m (${S.terrain.ny}×${S.terrain.nx})`);
    if (S.roughnessZones?.length) bits.push(`Roughness ${S.roughnessZones.length} zones`);
    else if (S.roughnessRose) bits.push(`Roughness rose (uniform/default z₀)`);
    if (S.windPoint || (S.wind && S.wind.lat != null)) {
      const wp = S.windPoint || S.wind;
      bits.push(`Wind pt: ${Number(wp.lat).toFixed(3)}, ${Number(wp.lon).toFixed(3)} (${wp.source || ''})`);
    }
    if (S.speedField?.length) {
      const v = S.speedField.filter(isFinite);
      bits.push(`Hub WS ${Math.min(...v).toFixed(2)}–${Math.max(...v).toFixed(2)} m/s`);
    }
    el.innerHTML = bits.length ? bits.map((b) => `• ${b}`).join('<br>') : 'No layers yet — download terrain / run AEP';
  }

  function updateLegend() {
    const body = $('legendBody'); if (!body) return;
    let html = '';
    if ($('lyrElev')?.checked && S.terrain) {
      html += `<div class="title">Elevation (m)</div><div class="bar"></div>` +
        `<div style="display:flex;justify-content:space-between"><span>${S.terrain.minE.toFixed(0)}</span><span>${S.terrain.maxE.toFixed(0)}</span></div>`;
    }
    if ($('lyrRough')?.checked) {
      html += `<div class="title" style="margin-top:6px">Roughness z₀</div>` +
        [['#1e90ff','water &lt;0.001'],['#f4d03f','open 0.001–0.02'],['#58d68d','farm 0.02–0.08'],
         ['#d4ac0d','scrub 0.08–0.3'],['#af601a','suburban'],['#1e8449','forest/urban']].map(
          ([c,l]) => `<div class="row"><span class="sw" style="background:${c}"></span>${l}</div>`
        ).join('');
    }
    if ($('lyrSpeed')?.checked && S.speedField?.length) {
      const v = S.speedField.filter(isFinite);
      html += `<div class="title" style="margin-top:6px">Hub wind speed (m/s)</div><div class="bar"></div>` +
        `<div style="display:flex;justify-content:space-between"><span>${Math.min(...v).toFixed(1)}</span><span>${Math.max(...v).toFixed(1)}</span></div>`;
    }
    if ($('lyrWindPt')?.checked && (S.windPoint || (S.wind && S.wind.lat != null))) {
      html += `<div class="row" style="margin-top:6px"><span class="sw" style="background:#f5b942;border-radius:50%"></span>Wind data location</div>`;
    }
    if ($('lyrTurbines')?.checked) {
      html += `<div class="row" style="margin-top:4px"><span class="sw" style="background:#1b7a4a;border-radius:50%"></span>Turbines</div>`;
    }
    body.innerHTML = html || '<span style="color:var(--muted)">Toggle layers above</span>';
    updateMapStats();
  }

  function redrawMap(opts = {}) {
    const fit = opts.fit !== false;
    S.layers.boundary.clearLayers();
    S.layers.turbines.clearLayers();
    if (S.boundary.length >= 3) {
      const latlngs = S.boundary.map((p) => [p.lat, p.lon]);
      L.polygon(latlngs, { color: '#3d8bfd', weight: 2, fillOpacity: 0.08 }).addTo(S.layers.boundary);
    }
    S.turbines.forEach((t, i) => {
      const elevTxt = t.elev != null ? `<br>Elev: ${Number(t.elev).toFixed(0)} m` : '';
      const wsTxt = S.speedField && S.speedField[i] != null ? `<br>WS: <b>${Number(S.speedField[i]).toFixed(2)} m/s</b>` : '';
      L.circleMarker([t.lat, t.lon], {
        radius: 5, color: '#3dd68c', weight: 1, fillColor: '#1b7a4a', fillOpacity: 0.9,
      }).bindTooltip(`T${i + 1}<br>${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}${elevTxt}${wsTxt}`)
        .addTo(S.layers.turbines);
    });
    drawElevationLayer();
    drawRoughnessLayer();
    drawWindPointLayer();
    drawSpeedLayer();
    // respect checkbox state
    [['lyrBoundary', S.layers.boundary], ['lyrTurbines', S.layers.turbines],
     ['lyrElev', S.layers.elev], ['lyrRough', S.layers.rough],
     ['lyrSpeed', S.layers.speed], ['lyrWindPt', S.layers.windPt],
     ['lyrLabels', S.layers.labels]].forEach(([id, layer]) => {
      const on = $(id)?.checked !== false;
      if (id === 'lyrLabels') {
        // labels default off
        const labOn = !!$('lyrLabels')?.checked;
        if (labOn && !S.map.hasLayer(layer)) layer.addTo(S.map);
        if (!labOn && S.map.hasLayer(layer)) S.map.removeLayer(layer);
        return;
      }
      if (on && !S.map.hasLayer(layer)) layer.addTo(S.map);
      if (!on && S.map.hasLayer(layer)) S.map.removeLayer(layer);
    });
    if (fit) {
      const all = S.boundary.concat(S.turbines);
      if (all.length) {
        const b = bboxOf(all);
        S.map.fitBounds([[b.minLat, b.minLon], [b.maxLat, b.maxLon]], { padding: [40, 40] });
      }
    }
    updateLegend();
  }
  function refreshSiteUI() {
    const nB = Math.max(0, S.boundary.length - (S.boundary.length > 1 ? 1 : 0));
    const nT = S.turbines.length;
    const cap = nT * (+$('rated').value || 0) / 1000;
    chipBox('siteChips', [
      [`Boundary: ${nB} pts`, nB >= 3],
      [`Turbines: ${nT}`, nT > 0],
      [`Capacity: ${cap.toFixed(1)} MW`, nT > 0],
    ]);
    $('kCap').textContent = nT ? cap.toFixed(1) + ' MW' : '—';
    setStep('site', nB >= 3 || nT > 0 ? 'done' : '');
    setStep('wtg', S.pc ? 'done' : '');
  }

  // ─── Presets / PC ────────────────────────────────────────────────────────
  function applyPreset() {
    const key = $('preset').value;
    if (key === 'custom' && S.pc && S.pc.name === 'Custom') return;
    const p = PRESETS[key];
    if (!p) return;
    S.pc = JSON.parse(JSON.stringify(p));
    $('hh').value = p.hh; $('D').value = p.D; $('rated').value = p.rated;
    S.turbines.forEach((t) => { if (!t._customHH) t.hh = p.hh; });
    addLog(`WTG preset: ${p.name}`, 'o');
    refreshSiteUI();
  }
  function currentPC() {
    if (!S.pc) applyPreset();
    // sync editable fields
    S.pc.hh = +$('hh').value || S.pc.hh;
    S.pc.D = +$('D').value || S.pc.D;
    S.pc.rated = +$('rated').value || S.pc.rated;
    return S.pc;
  }
  function powerAt(pc, ws) {
    if (ws < pc.cutIn || ws >= pc.cutOut) return 0;
    const x = pc.ws;
    for (let i = 1; i < x.length; i++) {
      if (ws <= x[i]) {
        const t = (ws - x[i - 1]) / (x[i] - x[i - 1] || 1e-9);
        return pc.pw[i - 1] * (1 - t) + pc.pw[i] * t;
      }
    }
    return 0;
  }
  function ctAt(pc, ws) {
    if (ws < pc.cutIn || ws >= pc.cutOut) return 0.05;
    const x = pc.ws;
    for (let i = 1; i < x.length; i++) {
      if (ws <= x[i]) {
        const t = (ws - x[i - 1]) / (x[i] - x[i - 1] || 1e-9);
        return (pc.ct[i - 1] || 0) * (1 - t) + (pc.ct[i] || 0) * t;
      }
    }
    return 0.05;
  }

  // ─── Layout grid ─────────────────────────────────────────────────────────
  function generateGrid() {
    if (S.boundary.length < 3) { alert('Define a boundary first'); return; }
    const pc = currentPC();
    const nWant = +$('nWtg').value || 20;
    const spA = (+$('spD').value || 5) * pc.D;
    const spC = (+$('spC').value || 3.5) * pc.D;
    const c = centerOf(S.boundary);
    const b = bboxOf(S.boundary);
    const mLat = 111320, mLon = 111320 * Math.cos(c.lat * Math.PI / 180);
    const theta = 80 * Math.PI / 180;
    const ux = Math.sin(theta), uy = Math.cos(theta);
    const vx = Math.sin(theta + Math.PI / 2), vy = Math.cos(theta + Math.PI / 2);
    const span = Math.hypot((b.maxLon - b.minLon) * mLon, (b.maxLat - b.minLat) * mLat);
    const nA = Math.ceil(span / spA) + 3, nC = Math.ceil(span / spC) + 3;
    const cand = [];
    for (let ia = -nA; ia <= nA; ia++) {
      for (let ic = -nC; ic <= nC; ic++) {
        const along = ia * spA + ((ic % 2 === 0) ? 0 : spA * 0.5);
        const cross = ic * spC;
        const lon = c.lon + (along * ux + cross * vx) / mLon;
        const lat = c.lat + (along * uy + cross * vy) / mLat;
        if (pip(lon, lat, S.boundary)) cand.push({ lat, lon, hh: pc.hh });
      }
    }
    let turbines = cand;
    if (cand.length > nWant) {
      const stride = cand.length / nWant;
      turbines = [];
      const seen = new Set();
      for (let i = 0; i < nWant; i++) {
        const t = cand[Math.min(cand.length - 1, Math.floor(i * stride))];
        const k = t.lat.toFixed(5) + ',' + t.lon.toFixed(5);
        if (!seen.has(k)) { seen.add(k); turbines.push(t); }
      }
    }
    S.turbines = turbines.slice(0, nWant);
    addLog(`Generated ${S.turbines.length} WTG grid (requested ${nWant}, candidates ${cand.length})`, 'o');
    refreshSiteUI(); redrawMap();
  }

  function applyBBox() {
    const swLon = +$('swLon').value, swLat = +$('swLat').value;
    const neLon = +$('neLon').value, neLat = +$('neLat').value;
    if (![swLon, swLat, neLon, neLat].every(isFinite)) { alert('Enter all four corner coordinates'); return; }
    S.boundary = [
      { lon: swLon, lat: swLat }, { lon: neLon, lat: swLat },
      { lon: neLon, lat: neLat }, { lon: swLon, lat: neLat },
      { lon: swLon, lat: swLat },
    ];
    addLog(`Boundary set from SW–NE box`, 'o');
    refreshSiteUI(); redrawMap();
  }
  function applyCenter() {
    const lon = +$('cLon').value, lat = +$('cLat').value, rkm = +$('cRad').value || 5;
    if (!isFinite(lon) || !isFinite(lat)) { alert('Enter center lat/lon'); return; }
    const dLat = rkm / 111.32, dLon = rkm / (111.32 * Math.cos(lat * Math.PI / 180));
    S.boundary = [
      { lon: lon - dLon, lat: lat - dLat }, { lon: lon + dLon, lat: lat - dLat },
      { lon: lon + dLon, lat: lat + dLat }, { lon: lon - dLon, lat: lat + dLat },
      { lon: lon - dLon, lat: lat - dLat },
    ];
    addLog(`Boundary set from center ${lat.toFixed(4)}, ${lon.toFixed(4)} ± ${rkm} km`, 'o');
    refreshSiteUI(); redrawMap();
  }

  // ─── Terrain download ────────────────────────────────────────────────────
  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  async function fetchJsonWithRetry(url, opts = {}, retries = 5) {
    let lastErr = null;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const r = await fetch(url, { ...opts, signal: AbortSignal.timeout(opts.timeoutMs || 20000) });
        if (r.status === 429 || r.status === 503) {
          const wait = Math.min(12000, 800 * Math.pow(2, attempt) + Math.random() * 400);
          addLog(`Rate limited (HTTP ${r.status}) — retry in ${(wait / 1000).toFixed(1)}s…`, 'w');
          await sleep(wait);
          lastErr = new Error('HTTP ' + r.status);
          continue;
        }
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return await r.json();
      } catch (e) {
        lastErr = e;
        if (attempt < retries && /timeout|network|fetch|429|503/i.test(String(e.message || e))) {
          const wait = Math.min(10000, 600 * Math.pow(2, attempt));
          await sleep(wait);
          continue;
        }
        throw e;
      }
    }
    throw lastErr || new Error('fetch failed');
  }

  async function fetchElevOpenMeteo(lats, lons) {
    // Smaller chunks + delay between batches to avoid Open-Meteo HTTP 429
    const CHUNK = 20, elevs = [];
    const nBatch = Math.ceil(lats.length / CHUNK);
    for (let c = 0, bi = 0; c < lats.length; c += CHUNK, bi++) {
      const la = lats.slice(c, c + CHUNK), lo = lons.slice(c, c + CHUNK);
      const url = `https://api.open-meteo.com/v1/elevation?latitude=${la.join(',')}&longitude=${lo.join(',')}`;
      try {
        const d = await fetchJsonWithRetry(url, { timeoutMs: 20000 }, 6);
        elevs.push(...(d.elevation || la.map(() => null)));
      } catch (e) {
        addLog(`Open-Meteo batch ${bi + 1}/${nBatch} failed: ${e.message}`, 'w');
        elevs.push(...la.map(() => null));
      }
      if (c + CHUNK < lats.length) await sleep(350); // pace requests
      if (bi % 5 === 0) setProgress(5 + Math.round(50 * (bi + 1) / nBatch));
    }
    return elevs;
  }

  async function fetchElevOpenTopo(lats, lons, ds = 'mapzen') {
    // OpenTopo often blocks browser CORS ("Failed to fetch"). Try direct, then CORS proxies.
    const CHUNK = 50, elevs = [];
    const nBatch = Math.ceil(lats.length / CHUNK);
    for (let c = 0, bi = 0; c < lats.length; c += CHUNK, bi++) {
      const la = lats.slice(c, c + CHUNK), lo = lons.slice(c, c + CHUNK);
      const locs = la.map((a, i) => `${a},${lo[i]}`).join('|');
      const base = `https://api.opentopodata.org/v1/${ds}?locations=${locs}`;
      const urls = [
        base,
        'https://corsproxy.io/?' + encodeURIComponent(base),
        'https://api.allorigins.win/raw?url=' + encodeURIComponent(base),
      ];
      let ok = false;
      let lastErr = null;
      for (const url of urls) {
        try {
          const d = await fetchJsonWithRetry(url, { timeoutMs: 25000 }, 2);
          const results = d.results || null;
          if (results && results.length) {
            results.forEach((res) => elevs.push(res.elevation != null ? res.elevation : null));
            ok = true;
            break;
          }
          // allorigins may wrap
          if (d && d.contents) {
            const inner = typeof d.contents === 'string' ? JSON.parse(d.contents) : d.contents;
            (inner.results || []).forEach((res) => elevs.push(res.elevation != null ? res.elevation : null));
            ok = true;
            break;
          }
        } catch (e) {
          lastErr = e;
        }
      }
      if (!ok) {
        addLog(`OpenTopo batch ${bi + 1}/${nBatch} failed (CORS/rate) — will fill gaps`, 'w');
        elevs.push(...la.map(() => null));
      }
      if (c + CHUNK < lats.length) await sleep(1200); // OpenTopo ~1 req/s
    }
    return elevs;
  }

  /** Fill null elevations by nearest-neighbor / bilinear from valid cells */
  function fillElevationNulls(elevs, ny, nx) {
    const a = elevs.slice();
    const valid = a.filter((v) => v != null && isFinite(v));
    if (!valid.length) return a;
    const fallback = mean(valid);
    // multi-pass nearest valid
    for (let pass = 0; pass < 4; pass++) {
      let changed = 0;
      for (let i = 0; i < ny; i++) {
        for (let j = 0; j < nx; j++) {
          const k = i * nx + j;
          if (a[k] != null && isFinite(a[k])) continue;
          let s = 0, n = 0;
          for (let di = -2; di <= 2; di++) for (let dj = -2; dj <= 2; dj++) {
            const ii = i + di, jj = j + dj;
            if (ii < 0 || jj < 0 || ii >= ny || jj >= nx) continue;
            const v = a[ii * nx + jj];
            if (v != null && isFinite(v)) { s += v; n++; }
          }
          if (n) { a[k] = s / n; changed++; }
        }
      }
      if (!changed) break;
    }
    for (let k = 0; k < a.length; k++) if (a[k] == null || !isFinite(a[k])) a[k] = fallback;
    return a;
  }

  async function downloadTerrain() {
    const pts = S.boundary.length ? S.boundary : S.turbines;
    if (pts.length < 1) { alert('Define site area or layout first'); return; }
    setStep('maps', 'run');
    const c = centerOf(pts);
    const b = bboxOf(pts);
    const spanKm = Math.max(
      (b.maxLat - b.minLat) * 111.32,
      (b.maxLon - b.minLon) * 111.32 * Math.cos(c.lat * Math.PI / 180)
    );
    const rad = Math.max(+$('terrR').value || 12, spanKm / 2 + 3);
    const ng = +$('terrG').value || 40;
    addLog(`Terrain: center ${c.lat.toFixed(4)}, ${c.lon.toFixed(4)} radius ${rad.toFixed(1)} km grid ${ng}×${ng}`, 'i');
    setProgress(5);

    const dLat = rad / 111.32, dLon = rad / (111.32 * Math.cos(c.lat * Math.PI / 180));
    const lat0 = c.lat - dLat, lat1 = c.lat + dLat, lon0 = c.lon - dLon, lon1 = c.lon + dLon;
    const gLats = [], gLons = [];
    for (let i = 0; i < ng; i++) for (let j = 0; j < ng; j++) {
      gLats.push(+(lat0 + i * (lat1 - lat0) / (ng - 1)).toFixed(4));
      gLons.push(+(lon0 + j * (lon1 - lon0) / (ng - 1)).toFixed(4));
    }

    let elevs = await fetchElevOpenMeteo(gLats, gLons);
    let valid = elevs.filter((e) => e != null && isFinite(e)).length;
    let source = 'Open-Meteo';
    addLog(`Open-Meteo: ${valid}/${gLats.length} valid points`, valid > 0 ? 'i' : 'w');
    if (valid < gLats.length * 0.85) {
      addLog('Filling gaps / trying OpenTopoData fallback…', 'w');
      try {
        const e2 = await fetchElevOpenTopo(gLats, gLons, 'mapzen');
        // merge: prefer Open-Meteo where valid, else OpenTopo
        for (let i = 0; i < elevs.length; i++) {
          if ((elevs[i] == null || !isFinite(elevs[i])) && e2[i] != null && isFinite(e2[i])) elevs[i] = e2[i];
        }
        const v2 = elevs.filter((e) => e != null && isFinite(e)).length;
        if (v2 > valid) { valid = v2; source = valid > gLats.length * 0.5 ? 'Open-Meteo+OpenTopo' : 'OpenTopo/mapzen'; }
      } catch (e) {
        addLog('OpenTopo fallback skipped: ' + e.message, 'w');
      }
    }
    // Never leave nulls as 0 (ocean/false flat) — interpolate gaps
    elevs = fillElevationNulls(elevs, ng, ng);
    valid = elevs.filter((e) => e != null && isFinite(e)).length;
    setProgress(70);
    const grid = [];
    for (let i = 0; i < ng; i++) {
      grid[i] = [];
      for (let j = 0; j < ng; j++) {
        const v = elevs[i * ng + j];
        grid[i][j] = (v != null && isFinite(v)) ? v : 0;
      }
    }
    const vals = elevs.filter((e) => e != null && isFinite(e));
    if (!vals.length) { addLog('Terrain download failed — no elevation values', 'e'); return false; }
    S.terrain = {
      grid, ny: ng, nx: ng, lat0, lat1, lon0, lon1,
      minE: Math.min(...vals), maxE: Math.max(...vals), meanE: mean(vals),
      elev: vals, source,
    };
    // turbine elevations
    S.turbines.forEach((t) => { const e = elevAt(t.lat, t.lon); if (e != null) t.elev = e; });
    setProgress(100);
    addLog(`Terrain OK (${source}): ${ng}×${ng}, elev ${S.terrain.minE.toFixed(0)}–${S.terrain.maxE.toFixed(0)} m`, 'o');
    chipBox('mapChips', [
      [`Terrain ${ng}×${ng}`, true],
      [`${S.terrain.minE.toFixed(0)}–${S.terrain.maxE.toFixed(0)} m`, true],
      [`Roughness ${S.roughnessZones.length || 0} zones`, S.roughnessZones.length > 0],
    ]);
    setStep('maps', S.roughnessZones.length ? 'done' : 'run');
    redrawMap({ fit: true });
    addLog('Map: elevation heatmap + contours drawn', 'i');
    return true;
  }

  function elevAt(lat, lon) {
    const T = S.terrain; if (!T) return null;
    const gy = (lat - T.lat0) / (T.lat1 - T.lat0) * (T.ny - 1);
    const gx = (lon - T.lon0) / (T.lon1 - T.lon0) * (T.nx - 1);
    const j0 = clamp(Math.floor(gy), 0, T.ny - 2), i0 = clamp(Math.floor(gx), 0, T.nx - 2);
    const ty = gy - j0, tx = gx - i0;
    const v00 = T.grid[j0][i0], v10 = T.grid[j0][i0 + 1], v01 = T.grid[j0 + 1][i0], v11 = T.grid[j0 + 1][i0 + 1];
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11;
  }

  // ─── Roughness ───────────────────────────────────────────────────────────
  async function fetchOverpassJSON(query, timeoutMs = 45000) {
    const endpoints = [
      'https://overpass-api.de/api/interpreter',
      'https://overpass.kumi.systems/api/interpreter',
      'https://overpass.openstreetmap.ru/cgi/interpreter',
    ];
    let last = null;
    for (const ep of endpoints) {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          addLog(`OSM Overpass: ${ep.split('/')[2]} (try ${attempt + 1})…`, 'i');
          const r = await fetch(ep, {
            method: 'POST',
            body: 'data=' + encodeURIComponent(query),
            headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json' },
            signal: AbortSignal.timeout(timeoutMs),
          });
          if (r.status === 429 || r.status === 504 || r.status === 503) {
            last = new Error('HTTP ' + r.status);
            await sleep(1500 * (attempt + 1));
            continue;
          }
          if (!r.ok) { last = new Error('HTTP ' + r.status); break; }
          const d = await r.json();
          if (d && (d.elements || d.remark)) return d;
          last = new Error('empty Overpass response');
        } catch (e) {
          last = e;
          addLog('Overpass fail ' + ep.split('/')[2] + ': ' + e.message, 'w');
          await sleep(800);
        }
      }
    }
    throw last || new Error('Overpass failed');
  }

  async function downloadRoughness() {
    if (!S.terrain && S.boundary.length < 3) { alert('Download terrain or set boundary first'); return; }
    setStep('maps', 'run');
    const b = S.terrain
      ? { minLat: S.terrain.lat0, maxLat: S.terrain.lat1, minLon: S.terrain.lon0, maxLon: S.terrain.lon1 }
      : bboxOf(S.boundary.length ? S.boundary : S.turbines);
    addLog('Roughness: querying OSM land use…', 'i');
    // Keep query lighter for reliability; out center instead of full geom when possible
    const q = `[out:json][timeout:60][bbox:${b.minLat},${b.minLon},${b.maxLat},${b.maxLon}];(
      way["landuse"];
      way["natural"~"wood|water|scrub|grassland|wetland|forest"];
    );out body center;`;
    const z0Map = {
      farmland: .03, grass: .03, meadow: .03, grassland: .03, scrub: .05, heath: .05,
      forest: 1.0, wood: 1.0, residential: .4, commercial: .4, industrial: .5, retail: .3,
      water: .0002, wetland: .0002, reservoir: .0002, sand: .002, orchard: .1, vineyard: .1,
      railway: .05, construction: .1, military: .3, recreation_ground: .03, park: .1,
    };
    try {
      const d = await fetchOverpassJSON(q, 55000);
      const zones = [];
      (d.elements || []).forEach((el) => {
        const lu = el.tags?.landuse || el.tags?.natural || el.tags?.leisure || 'other';
        const z0 = z0Map[lu] ?? 0.1;
        let pts = null;
        if (el.geometry && el.geometry.length >= 3) {
          pts = el.geometry.map((p) => ({ lat: p.lat, lon: p.lon }));
        } else if (el.center && el.center.lat != null) {
          // out center response — small square around centroid
          const d = 0.002;
          const la = el.center.lat, lo = el.center.lon;
          pts = [
            { lat: la - d, lon: lo - d }, { lat: la - d, lon: lo + d },
            { lat: la + d, lon: lo + d }, { lat: la + d, lon: lo - d },
          ];
        } else return;
        zones.push({ lu, z0, pts });
      });
      // cap extremes
      if (zones.length > 1) {
        const all = zones.map((z) => z.z0).sort((a, b) => a - b);
        const med = all[all.length >> 1];
        zones.forEach((z) => { z.z0 = clamp(z.z0, med * 0.1, med * 10); });
      }
      S.roughnessZones = zones;
      buildRoughnessRose();
      addLog(`Roughness OK: ${zones.length} OSM zones, rose built`, 'o');
    } catch (e) {
      addLog('Roughness OSM failed — using uniform z0=' + $('z0').value, 'w');
      S.roughnessZones = [];
      buildRoughnessRose(true);
    }
    chipBox('mapChips', [
      [`Terrain ${S.terrain ? S.terrain.ny + '×' + S.terrain.nx : 'no'}`, !!S.terrain],
      [`Roughness ${S.roughnessZones.length} zones`, true],
      [`z₀ default ${$('z0').value}`, true],
    ]);
    setStep('maps', S.terrain ? 'done' : 'run');
    redrawMap({ fit: false });
    addLog(S.roughnessZones.length ? 'Map: roughness polygons drawn' : 'Map: uniform z₀ (no OSM polygons)', 'i');
    return true;
  }

  function buildRoughnessRose(uniform = false) {
    const nSec = +$('nSec').value || 16;
    const z0def = +$('z0').value || 0.03;
    const c = centerOf(S.boundary.length ? S.boundary : S.turbines);
    const rose = [];
    for (let s = 0; s < nSec; s++) {
      if (uniform || !S.roughnessZones.length) {
        rose.push({ z0: [z0def, z0def, z0def], x: [0, 2000, 8000] });
        continue;
      }
      const dir = (s + 0.5) * 360 / nSec * Math.PI / 180;
      const dists = [2, 6, 12]; // km
      const z0s = dists.map((dkm) => {
        const lat = c.lat + (dkm / 111.32) * Math.cos(dir);
        const lon = c.lon + (dkm / (111.32 * Math.cos(c.lat * Math.PI / 180))) * Math.sin(dir);
        let best = z0def, minD = Infinity;
        for (const z of S.roughnessZones) {
          const zlat = z.pts.reduce((s, p) => s + p.lat, 0) / z.pts.length;
          const zlon = z.pts.reduce((s, p) => s + p.lon, 0) / z.pts.length;
          const d = Math.hypot(zlat - lat, zlon - lon);
          if (d < minD) { minD = d; best = z.z0; }
        }
        return best;
      });
      rose.push({ z0: z0s, x: [0, 2000, 8000] });
    }
    S.roughnessRose = rose;
    S.autoRoughnessRose = rose;
  }

  // ─── Wind data ───────────────────────────────────────────────────────────
  function weibullFit(ws) {
    const v = ws.filter((x) => x > 0.5);
    if (v.length < 50) return { A: mean(v) / 0.886 || 6, k: 2 };
    const m = mean(v);
    const m2 = mean(v.map((x) => x * x));
    const s2 = Math.max(1e-6, m2 - m * m);
    // k ≈ (σ/μ)^(-1.086)
    const k = clamp(Math.pow(Math.sqrt(s2) / m, -1.086), 1.2, 3.5);
    const A = m / gamma(1 + 1 / k);
    return { A, k };
  }

  function setWind(speeds, dirs, meta) {
    const valid = speeds.map((v, i) => ({ v, d: dirs[i] })).filter((x) => x.v > 0.2 && x.v < 50);
    const sp = valid.map((x) => x.v), dr = valid.map((x) => ((x.d % 360) + 360) % 360);
    const wb = weibullFit(sp);
    const w = {
      speeds: sp, dirs: dr, times: meta.times || [],
      height: meta.height || 100, source: meta.source || 'WIND',
      meanWS: mean(sp.filter((v) => v > 0.5)),
      weibullA: wb.A, weibullK: wb.k,
      lat: meta.lat, lon: meta.lon,
    };
    S.wind = w;
    S.windSources[w.source] = { ...w, speeds: [...sp], dirs: [...dr], sourceHeight: w.height, mastH: w.height, meanWS: w.meanWS };
    // Wind data location for map marker
    if (w.lat != null && w.lon != null && isFinite(w.lat) && isFinite(w.lon)) {
      S.windPoint = { lat: w.lat, lon: w.lon, source: w.source, height: w.height, meanWS: w.meanWS };
    } else if (S.turbines.length || S.boundary.length) {
      const c = centerOf(S.boundary.length ? S.boundary : S.turbines);
      S.windPoint = { lat: c.lat, lon: c.lon, source: w.source, height: w.height, meanWS: w.meanWS };
      w.lat = c.lat; w.lon = c.lon;
    }
    chipBox('windChips', [
      [`${w.source}`, true],
      [`n=${sp.length.toLocaleString()}`, true],
      [`mean ${w.meanWS.toFixed(2)} m/s @ ${w.height} m`, true],
      [`A=${wb.A.toFixed(2)} k=${wb.k.toFixed(2)}`, true],
      [w.lat != null ? `pt ${Number(w.lat).toFixed(3)}, ${Number(w.lon).toFixed(3)}` : 'pt n/a', w.lat != null],
    ]);
    setStep('wind', 'done');
    addLog(`Wind ${w.source}: ${sp.length} samples, mean ${w.meanWS.toFixed(2)} m/s @ ${w.height} m`, 'o');
    redrawMap({ fit: false });
    return w;
  }

  async function downloadERA5() {
    const pts = S.boundary.length ? S.boundary : S.turbines;
    if (!pts.length) { alert('Set site first'); return; }
    const c = centerOf(pts);
    const y0 = +$('y0').value || 2019, y1 = +$('y1').value || 2023;
    const h = +$('dataH').value || 100;
    const wsV = h >= 80 ? 'wind_speed_100m' : 'wind_speed_10m';
    const wdV = h >= 80 ? 'wind_direction_100m' : 'wind_direction_10m';
    const dataH = h >= 80 ? 100 : 10;
    setStep('wind', 'run');
    addLog(`ERA5: ${c.lat.toFixed(3)}, ${c.lon.toFixed(3)} years ${y0}–${y1} (${wsV})`, 'i');
    setProgress(10);
    const speeds = [], dirs = [], times = [];
    for (let y = y0; y <= y1; y++) {
      const url = `https://archive-api.open-meteo.com/v1/archive?latitude=${c.lat.toFixed(4)}&longitude=${c.lon.toFixed(4)}&start_date=${y}-01-01&end_date=${y}-12-31&hourly=${wsV},${wdV}&wind_speed_unit=ms&timezone=UTC`;
      try {
        const r = await fetch(url, { signal: AbortSignal.timeout(60000) });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const d = await r.json();
        const ws = d.hourly?.[wsV] || [], wd = d.hourly?.[wdV] || [], tm = d.hourly?.time || [];
        for (let i = 0; i < ws.length; i++) {
          if (ws[i] != null && ws[i] >= 0 && ws[i] < 80) {
            speeds.push(ws[i]); dirs.push(((wd[i] || 0) % 360 + 360) % 360); times.push(tm[i] || '');
          }
        }
        setProgress(10 + 80 * (y - y0 + 1) / (y1 - y0 + 1));
        addLog(`ERA5 year ${y}: +${ws.filter((v)=>v!=null).length} hours`, 'i');
        if (y < y1) await sleep(400);
      } catch (e) {
        addLog(`ERA5 ${y} failed: ${e.message} — retrying once…`, 'w');
        try {
          await sleep(1500);
          const r2 = await fetch(url, { signal: AbortSignal.timeout(90000) });
          if (!r2.ok) throw new Error('HTTP ' + r2.status);
          const d2 = await r2.json();
          const ws2 = d2.hourly?.[wsV] || [], wd2 = d2.hourly?.[wdV] || [], tm2 = d2.hourly?.time || [];
          for (let i = 0; i < ws2.length; i++) {
            if (ws2[i] != null && ws2[i] >= 0 && ws2[i] < 80) {
              speeds.push(ws2[i]); dirs.push(((wd2[i] || 0) % 360 + 360) % 360); times.push(tm2[i] || '');
            }
          }
          addLog(`ERA5 year ${y} retry OK`, 'o');
        } catch (e2) {
          addLog(`ERA5 ${y} failed: ${e2.message}`, 'e');
        }
      }
    }
    if (speeds.length < 100) { addLog('ERA5 download failed or too short', 'e'); return false; }
    setWind(speeds, dirs, { source: 'ERA5', height: dataH, lat: c.lat, lon: c.lon, times });
    setProgress(100);
    return true;
  }

  function parseGWALib(text, targetHH = 140, z0target = 0.03) {
    const lines = text.replace(/\r/g, '\n').split('\n').filter((l) => l.trim());
    let metaIdx = -1, nRough, nHeights, nSectors;
    for (let i = 0; i < Math.min(6, lines.length); i++) {
      const p = lines[i].trim().split(/\s+/);
      if (p.length === 3 && p.every((x) => isFinite(+x) && +x > 0)) {
        metaIdx = i;
        // GWA4: nRough nHeights nSectors
        const z0Line = lines[i + 1].trim().split(/\s+/).map(Number);
        const hLine = lines[i + 2].trim().split(/\s+/).map(Number);
        nRough = z0Line.length; nHeights = hLine.length;
        nSectors = [ +p[0], +p[1], +p[2] ].find((v) => v !== nRough && v !== nHeights);
        break;
      }
    }
    if (metaIdx < 0) throw new Error('Not a GWA/WAsP .lib');
    const z0s = lines[metaIdx + 1].trim().split(/\s+/).map(Number);
    const heights = lines[metaIdx + 2].trim().split(/\s+/).map(Number);
    const data = [];
    for (let i = metaIdx + 3; i < lines.length; i++) {
      const nums = lines[i].trim().split(/\s+/).map(Number).filter((v) => isFinite(v));
      if (nums.length >= nSectors) data.push(nums.slice(0, nSectors));
    }
    const rIdx = z0s.reduce((bi, z, i) => Math.abs(z - z0target) < Math.abs(z0s[bi] - z0target) ? i : bi, 0);
    const rowsPer = 1 + 2 * nHeights;
    const base = rIdx * rowsPer;
    const freq = data[base];
    const fsum = freq.reduce((a, b) => a + b, 0) || 1;
    // interleaved A,k
    const nearestH = (h) => heights.reduce((bi, hh, i) => Math.abs(hh - h) < Math.abs(heights[bi] - h) ? i : bi, 0);
    const hLo = nearestH(Math.min(...heights.filter((h) => h <= targetHH).concat([heights[0]])));
    const hHi = nearestH(Math.max(...heights.filter((h) => h >= targetHH).concat([heights[heights.length - 1]])));
    const getAK = (hIdx) => ({ A: data[base + 1 + 2 * hIdx], k: data[base + 2 + 2 * hIdx] });
    const lo = getAK(hLo), hi = getAK(hHi);
    const t = heights[hHi] === heights[hLo] ? 0 :
      (Math.log(targetHH) - Math.log(heights[hLo])) / (Math.log(heights[hHi]) - Math.log(heights[hLo]));
    const sectors = [];
    let meanWS = 0;
    for (let s = 0; s < nSectors; s++) {
      const A = lo.A[s] * (1 - t) + hi.A[s] * t;
      const k = Math.max(0.8, lo.k[s] * (1 - t) + hi.k[s] * t);
      const f = freq[s] / fsum;
      const ws = A * gamma(1 + 1 / k);
      meanWS += f * ws;
      sectors.push({ dir: (s + 0.5) * 360 / nSectors, freq: f, A, k, WS: ws });
    }
    // synthetic series
    const speeds = [], dirs = [];
    const N = 20000;
    for (const sec of sectors) {
      const n = Math.max(1, Math.round(N * sec.freq));
      for (let j = 0; j < n; j++) {
        const u = Math.min(Math.max(Math.random(), 1e-9), 1 - 1e-9);
        speeds.push(Math.min(35, Math.max(0.05, sec.A * Math.pow(-Math.log(u), 1 / sec.k))));
        dirs.push(((sec.dir + (Math.random() - 0.5) * 30) % 360 + 360) % 360);
      }
    }
    return { speeds: speeds.slice(0, N), dirs: dirs.slice(0, N), meanWS, sectors, z0: z0s[rIdx], height: targetHH };
  }

  async function downloadGWA() {
    const pts = S.boundary.length ? S.boundary : S.turbines;
    if (!pts.length) { alert('Set site first'); return; }
    const c = centerOf(pts);
    const hh = +$('hh').value || 140;
    setStep('wind', 'run');
    addLog(`GWA 4.0 .lib @ ${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}`, 'i');
    const libUrl = `https://globalwindatlas.info/api/gwa/custom/Lib?lat=${c.lat.toFixed(4)}&long=${c.lon.toFixed(4)}`;
    const headers = { 'X-Requested-With': 'XMLHttpRequest', 'Accept': '*/*', 'Referer': 'https://globalwindatlas.info/' };
    const tryUrls = [
      libUrl,
      'https://corsproxy.io/?' + encodeURIComponent(libUrl),
      'https://api.allorigins.win/raw?url=' + encodeURIComponent(libUrl),
    ];
    let text = null, err = null;
    for (const u of tryUrls) {
      try {
        const r = await fetch(u, { headers, signal: AbortSignal.timeout(30000) });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const t = await r.text();
        if (t && t.length > 50 && !t.startsWith('<!')) { text = t; addLog('GWA fetched via ' + u.slice(0, 48), 'i'); break; }
        err = 'empty/html response';
      } catch (e) { err = e.message; addLog('GWA try fail: ' + e.message, 'w'); }
    }
    if (!text) {
      addLog('GWA browser fetch blocked (' + err + '). Upload a .lib file or use ERA5.', 'e');
      alert('GWA API blocked by browser CORS.\n\nOptions:\n1) Use ERA5 button\n2) Download .lib from globalwindatlas.info and upload as wind file\n3) Run from a local helper that injects GWA');
      return false;
    }
    try {
      const g = parseGWALib(text, hh, +$('z0').value || 0.03);
      setWind(g.speeds, g.dirs, { source: 'GWA', height: hh, lat: c.lat, lon: c.lon });
      S.gwaMeta = { meanWS: g.meanWS, sectors: g.sectors, z0: g.z0 };
      addLog(`GWA climate mean ≈ ${g.meanWS.toFixed(2)} m/s @ ${hh} m (z0=${g.z0})`, 'o');
      setProgress(100);
      return true;
    } catch (e) {
      addLog('GWA parse failed: ' + e.message, 'e');
      return false;
    }
  }

  function loadWindFile(file) {
    return file.text().then((text) => {
      if (file.name.toLowerCase().endsWith('.lib') || text.includes('Generalized Wind Climate') || /^\s*\d+\s+\d+\s+\d+/m.test(text)) {
        const g = parseGWALib(text, +$('hh').value || 140, +$('z0').value || 0.03);
        setWind(g.speeds, g.dirs, { source: 'GWA_FILE', height: g.height });
        S.gwaMeta = { meanWS: g.meanWS, sectors: g.sectors, z0: g.z0 };
        return;
      }
      // CSV ws,wd
      const lines = text.trim().split(/\r?\n/).filter(Boolean);
      const sep = lines[0].includes(';') ? ';' : ',';
      let i0 = 0;
      const h = lines[0].toLowerCase();
      let iws = 0, iwd = 1;
      if (/[a-z]/.test(h)) {
        i0 = 1;
        const hs = h.split(sep).map((x) => x.trim());
        iws = hs.findIndex((x) => /ws|speed|wind_speed|wspd/.test(x));
        iwd = hs.findIndex((x) => /wd|dir|direction|wdir/.test(x));
        if (iws < 0) iws = 0; if (iwd < 0) iwd = 1;
      }
      const speeds = [], dirs = [];
      for (let i = i0; i < lines.length; i++) {
        const c = lines[i].split(sep);
        const v = +c[iws], d = +c[iwd];
        if (isFinite(v) && v >= 0 && v < 80) {
          speeds.push(v); dirs.push(isFinite(d) ? ((d % 360) + 360) % 360 : 0);
        }
      }
      if (speeds.length < 50) throw new Error('Need ≥50 wind samples');
      const src = $('windSrc').value === 'SITE' ? 'SITE' : 'MESO';
      setWind(speeds, dirs, { source: src, height: +$('dataH').value || 100 });
    });
  }

  // ─── Vertical extrapolation ──────────────────────────────────────────────
  function logLaw(ws, z1, z2, z0) {
    z0 = Math.max(z0, 1e-5);
    const a = Math.log(Math.max(z1, z0 * 1.1) / z0);
    const b = Math.log(Math.max(z2, z0 * 1.1) / z0);
    if (a <= 0) return ws;
    return ws * b / a;
  }

  // ─── Orography (WFP61 or fallback) ───────────────────────────────────────
  async function runOrography() {
    const pc = currentPC();
    // Prepare BZ host state
    window.BZ = window.BZ || {};
    Object.assign(window.BZ, {
      enabled: false, speedups: [], sectorSpeedups: [], mastSectorSU: [],
      mastSU: 1, rix: 0, sectorRoughRC: [], turbRoughRC: [], engine: null,
    });
    // Ensure rose
    if (!S.roughnessRose) buildRoughnessRose(!S.roughnessZones.length);

    if (S.terrain && S.terrain.grid && window.WFP61 && typeof window.WFP61.install === 'function') {
      try {
        // shims expected by engine
        if (!window.runBZModel) {
          // engine install patches runBZModel — create dummy first
          window.runBZModel = async () => {};
        }
        window.WFP61.install();
        // Set form fields engine reads
        if ($('hubH')) $('hubH').value = pc.hh;
        // engine uses $('hubH'), $('sects'), $('bzSz0'), $('z0val'), mast lat/lon
        // create hidden fields if missing
        ensureHiddenFields(pc);
        if (typeof window.runBZModel === 'function') {
          await window.runBZModel();
          addLog(`Orography: ${window.BZ.engine || 'BZ'} mastSU=${(window.BZ.mastSU || 1).toFixed(3)} RIX=${(window.BZ.rix || 0).toFixed(1)}%`, 'o');
          return true;
        }
      } catch (e) {
        addLog('Spectral BZ failed, using slope fallback: ' + e.message, 'w');
      }
    }
    // Fallback: simple elevation-based SU
    const nSec = +$('nSec').value || 16;
    const z0 = +$('z0').value || 0.03;
    const c = centerOf(S.boundary.length ? S.boundary : S.turbines);
    const mastE = elevAt(c.lat, c.lon) || 0;
    window.BZ.sectorSpeedups = [];
    window.BZ.mastSectorSU = [];
    for (let s = 0; s < nSec; s++) {
      window.BZ.mastSectorSU.push(1);
      const row = S.turbines.map((t) => {
        const e = t.elev != null ? t.elev : (elevAt(t.lat, t.lon) || mastE);
        const dE = e - mastE;
        const su = clamp(1 + 0.4 * dE / Math.max(80, pc.hh), 0.75, 1.35);
        return su;
      });
      window.BZ.sectorSpeedups.push(row);
    }
    window.BZ.speedups = S.turbines.map((_, ti) => mean(window.BZ.sectorSpeedups.map((r) => r[ti])));
    window.BZ.mastSU = 1;
    window.BZ.enabled = true;
    window.BZ.engine = 'elev-fallback';
    window.BZ.rix = window.computeRIX(S.terrain);
    window.BZ.sectorRoughRC = Array(nSec).fill(1);
    addLog('Orography: elevation fallback speed-ups', 'w');
    return true;
  }

  function ensureHiddenFields(pc) {
    const need = {
      hubH: pc.hh, sects: $('nSec').value, z0val: $('z0').value,
      bzSz0: $('z0').value, bzMz0: $('z0').value, bzRixThr: 0.3,
      mastLat: centerOf(S.boundary.length ? S.boundary : S.turbines).lat,
      mastLon: centerOf(S.boundary.length ? S.boundary : S.turbines).lon,
      mastH: S.wind?.height || 100,
      innerLayerScale: 1,
    };
    for (const [id, val] of Object.entries(need)) {
      let el = document.getElementById(id);
      if (!el) {
        el = document.createElement('input');
        el.type = 'hidden'; el.id = id;
        document.body.appendChild(el);
      }
      el.value = val;
    }
  }

  // ─── Wake: Bastankhah Gaussian ───────────────────────────────────────────
  function bastankhahDeficit(dx, dy, D, ct, k) {
    // dx downwind >0, dy crosswind
    if (dx <= 0) return 0;
    ct = clamp(ct, 0.01, 0.95);
    const beta = 0.5 * (1 + Math.sqrt(1 - ct)) / Math.sqrt(1 - ct);
    const eps = 0.2 * Math.sqrt(beta);
    const sig = k * dx / D + eps;
    const a = 1 - Math.sqrt(Math.max(0, 1 - ct / (8 * sig * sig)));
    const r2 = (dy / D) * (dy / D);
    return a * Math.exp(-0.5 * r2 / (sig * sig));
  }

  // ─── AEP core ────────────────────────────────────────────────────────────
  async function runAEP() {
    try {
      $('btnRun').disabled = true;
      setProgress(2);
      const pc = currentPC();
      if (!S.turbines.length) { alert('Add turbines (layout or grid)'); return; }
      if (S.boundary.length < 3) {
        S.boundary = convexHull(S.turbines); S.boundary.push({ ...S.boundary[0] });
      }

      // Maps
      if (!S.terrain) { addLog('Auto terrain…', 'i'); await downloadTerrain(); }
      setProgress(25);
      if (!S.roughnessRose) { addLog('Auto roughness…', 'i'); await downloadRoughness(); }
      setProgress(40);

      // Wind
      if (!S.wind || !S.wind.speeds?.length) {
        const src = $('windSrc').value;
        addLog('Auto wind: ' + src, 'i');
        if (src === 'GWA') await downloadGWA();
        else if (src === 'SITE' || src === 'MESO') { alert('Upload site/mesoscale wind CSV first'); return; }
        else await downloadERA5();
      }
      if (!S.wind?.speeds?.length) { addLog('No wind data', 'e'); return; }
      setProgress(55);

      // Extrapolate to hub
      const z0 = +$('z0').value || 0.03;
      const dataH = S.wind.height || +$('dataH').value || 100;
      const hubSp = S.wind.speeds.map((v) => logLaw(v, dataH, pc.hh, z0));
      const hubDir = S.wind.dirs.slice();
      addLog(`Vertical extrap ${dataH}→${pc.hh} m (log-law z0=${z0})`, 'i');

      // Orography
      setProgress(65);
      await runOrography();
      setProgress(75);

      // Sectorize
      const nSec = +$('nSec').value || 16;
      const secW = 360 / nSec;
      const sec = Array.from({ length: nSec }, () => ({ n: 0, sum: 0, ws: [] }));
      for (let i = 0; i < hubSp.length; i++) {
        if (hubSp[i] < 0.5) continue;
        const si = Math.floor((((hubDir[i] % 360) + 360) % 360) / secW) % nSec;
        sec[si].n++; sec[si].sum += hubSp[i]; sec[si].ws.push(hubSp[i]);
      }
      const nTot = sec.reduce((s, x) => s + x.n, 0) || 1;
      const sectors = sec.map((s, i) => {
        const wb = weibullFit(s.ws.length ? s.ws : [1]);
        return {
          dir: (i + 0.5) * secW,
          freq: s.n / nTot,
          A: wb.A, k: wb.k,
          WS: s.n ? s.sum / s.n : 0,
        };
      });

      // Turbine positions XY
      const lat0 = centerOf(S.turbines).lat, lon0 = centerOf(S.turbines).lon;
      const xy = S.turbines.map((t) => latLonToXY(t.lat, t.lon, lat0, lon0));
      const kWake = +$('wakeK').value || 0.04;
      const lossOther = +$('loss').value || 8;
      const avail = (+$('avail').value || 97) / 100;
      const elec = 1 - (+$('elec').value || 2) / 100;
      const rho = 1.225;

      // AEP integration
      const per = S.turbines.map(() => ({ gross: 0, net: 0, wSum: 0, fwsSum: 0, wakeSum: 0, fwsN: 0 }));
      const hours = 8760;

      for (let si = 0; si < nSec; si++) {
        const f = sectors[si].freq;
        if (f < 1e-6) continue;
        const dir = sectors[si].dir; // FROM
        const th = (dir + 180) * Math.PI / 180; // TO direction for downwind
        const ux = Math.sin(th), uy = Math.cos(th); // downwind unit (east,north)
        const vx = Math.sin(th + Math.PI / 2), vy = Math.cos(th + Math.PI / 2);

        const mastSU = (window.BZ.mastSectorSU && window.BZ.mastSectorSU[si]) || window.BZ.mastSU || 1;
        const secSU = (window.BZ.sectorSpeedups && window.BZ.sectorSpeedups[si]) || null;
        const rc = (window.BZ.sectorRoughRC && window.BZ.sectorRoughRC[si]) || 1;

        // Weibull discrete bins
        const A = sectors[si].A, k = Math.max(sectors[si].k, 1.1);
        for (let b = 1; b <= 30; b++) {
          const u0 = b - 0.5; // bin center free-stream at mast/climate
          // Weibull pdf bin probability
          const cdf = (u) => 1 - Math.exp(-Math.pow(u / A, k));
          const p = Math.max(0, cdf(b) - cdf(b - 1));
          if (p < 1e-9) continue;

          // Free WS at each turbine after oro/roughness relative to climate
          const free = S.turbines.map((_, ti) => {
            const su = secSU ? secSU[ti] : (window.BZ.speedups?.[ti] || 1);
            // WAsP-like relative: climate already generalized-ish; apply SU_rel = su/mastSU * rc
            const rel = (su / Math.max(1e-6, mastSU)) * (isFinite(rc) ? rc : 1);
            return Math.max(0.1, u0 * rel);
          });

          // Wake deficits (energy combination)
          const deficits2 = free.map(() => 0);
          for (let i = 0; i < xy.length; i++) {
            for (let j = 0; j < xy.length; j++) {
              if (i === j) continue;
              const dx = (xy[j].x - xy[i].x) * ux + (xy[j].y - xy[i].y) * uy; // downwind dist from i to j
              const dy = (xy[j].x - xy[i].x) * vx + (xy[j].y - xy[i].y) * vy;
              if (dx < 0.5 * pc.D) continue;
              const ct = ctAt(pc, free[i]);
              const def = bastankhahDeficit(dx, dy, pc.D, ct, kWake);
              deficits2[j] += def * def;
            }
          }
          const netU = free.map((u, ti) => u * (1 - Math.min(0.7, Math.sqrt(deficits2[ti]))));

          for (let ti = 0; ti < S.turbines.length; ti++) {
            const pg = powerAt(pc, free[ti]); // density ~1.225 baseline
            const pn = powerAt(pc, netU[ti]);
            // density correction light: none (user can extend)
            per[ti].gross += f * p * pg * hours / 1000; // MWh → later GWh
            per[ti].net += f * p * pn * hours / 1000;
            per[ti].fwsSum += f * p * free[ti];
            per[ti].fwsN += f * p;
            per[ti].wakeSum += f * p * (free[ti] > 0.1 ? (1 - netU[ti] / free[ti]) : 0);
            per[ti].wSum += f * p;
          }
        }
      }

      // MWh → GWh, apply losses on net
      const lossFac = (1 - lossOther / 100) * avail * elec;
      let gross = 0, net = 0;
      const perTurbine = per.map((p, i) => {
        const g = p.gross / 1000; // GWh
        const nWake = p.net / 1000;
        const n = nWake * lossFac;
        gross += g; net += n;
        const fws = p.fwsN > 0 ? p.fwsSum / p.fwsN : 0;
        const wakePct = p.wSum > 0 ? 100 * p.wakeSum / p.wSum : 0;
        const su = window.BZ.speedups?.[i] || 1;
        const cf = (n * 1000) / (pc.rated * 8760 / 1000) * 100; // %
        return {
          id: i + 1, lat: S.turbines[i].lat, lon: S.turbines[i].lon, elev: S.turbines[i].elev,
          freeWS: fws, wakePct, grossGWh: g, netGWh: n, CF: cf, SU: su,
        };
      });

      const capMW = S.turbines.length * pc.rated / 1000;
      const CF = net * 1000 / (capMW * 8760) * 100;
      const wakeLoss = gross > 0 ? (1 - (per.reduce((s, p) => s + p.net / 1000, 0) / gross)) * 100 : 0;
      const hubMean = mean(perTurbine.map((t) => t.freeWS));

      S.results = {
        grossAEP: gross,
        netAEP: net,
        wakeLoss,
        CF,
        hubMean,
        capacityMW: capMW,
        n: S.turbines.length,
        perTurbine,
        sectors,
        windSource: S.wind.source,
        engine: window.BZ.engine,
        rix: window.BZ.rix,
        pc: { name: pc.name, rated: pc.rated, D: pc.D, hh: pc.hh },
        losses: { other: lossOther, avail: avail * 100, elec: (1 - elec) * 100 },
      };

      // UI
      $('kGross').textContent = gross.toFixed(2) + ' GWh/y';
      $('kNet').textContent = net.toFixed(2) + ' GWh/y';
      $('kCF').textContent = CF.toFixed(1) + ' %';
      $('kWake').textContent = wakeLoss.toFixed(1) + ' %';
      $('kWS').textContent = hubMean.toFixed(2) + ' m/s';
      $('kCap').textContent = capMW.toFixed(1) + ' MW';
      chipBox('resChips', [
        [`Wind: ${S.wind.source}`, true],
        [`Engine: ${window.BZ.engine || 'n/a'}`, true],
        [`RIX ${(+window.BZ.rix || 0).toFixed(1)}%`, true],
        [`${S.turbines.length} WTGs`, true],
      ]);
      $('tbl').innerHTML = perTurbine.map((t) =>
        `<tr><td>${t.id}</td><td>${t.freeWS.toFixed(2)}</td><td>${(+t.SU).toFixed(3)}</td><td>${t.wakePct.toFixed(1)}</td><td>${t.netGWh.toFixed(2)}</td><td>${t.CF.toFixed(1)}</td></tr>`
      ).join('');
      $('sectbl').innerHTML = sectors.map((s) =>
        `<tr><td>${s.dir.toFixed(0)}°</td><td>${(100 * s.freq).toFixed(1)}</td><td>${s.A.toFixed(2)}</td><td>${s.k.toFixed(2)}</td><td>${s.WS.toFixed(2)}</td></tr>`
      ).join('');

      $('btnExport').disabled = false;
      setStep('aep', 'done');
      setProgress(100);
      // Speed map from free-stream hub WS
      S.speedField = perTurbine.map((t) => t.freeWS);
      redrawMap({ fit: false });
      addLog('Map: hub-height wind speed layer updated at each WTG', 'i');
      addLog(`AEP complete: Gross ${gross.toFixed(2)} · Net ${net.toFixed(2)} GWh/y · CF ${CF.toFixed(1)}% · Wake ${wakeLoss.toFixed(1)}%`, 'o');
      addLog('INDICATIVE only — not bankable without site mast validation', 'w');
    } catch (e) {
      console.error(e);
      addLog('AEP failed: ' + e.message, 'e');
      alert('AEP failed: ' + e.message);
    } finally {
      $('btnRun').disabled = false;
    }
  }

  function exportResults() {
    const R = S.results; if (!R) return;
    const stamp = new Date().toISOString().slice(0, 10);
    const base = (S.project || 'windfarm').replace(/\s+/g, '_');

    // summary md
    let md = `# ${S.project} — AEP Report\n\n`;
    md += `Generated: ${new Date().toISOString()}\n\n`;
    md += `## Configuration\n`;
    md += `- WTG: ${R.pc.name} · ${R.pc.rated} kW · D=${R.pc.D} m · HH=${R.pc.hh} m\n`;
    md += `- Count: ${R.n} · Capacity: ${R.capacityMW.toFixed(2)} MW\n`;
    md += `- Wind: ${R.windSource} · Engine: ${R.engine} · RIX: ${(+R.rix || 0).toFixed(2)}%\n`;
    md += `- Losses: other ${R.losses.other}% · avail ${R.losses.avail}% · elec ${R.losses.elec}%\n\n`;
    md += `## AEP\n`;
    md += `| Metric | Value |\n|--------|-------|\n`;
    md += `| Gross AEP | ${R.grossAEP.toFixed(3)} GWh/y |\n`;
    md += `| Net AEP | ${R.netAEP.toFixed(3)} GWh/y |\n`;
    md += `| Wake loss | ${R.wakeLoss.toFixed(2)} % |\n`;
    md += `| CF | ${R.CF.toFixed(2)} % |\n`;
    md += `| Mean hub WS | ${R.hubMean.toFixed(3)} m/s |\n\n`;
    md += `## Disclaimer\nIndicative Wind Farm AEP Studio result. Not a substitute for met-mast + certified WAsP/OpenWind assessment.\n`;
    downloadText(`${base}_AEP_Report_${stamp}.md`, md);

    let csv = 'ID,Lat,Lon,Elev_m,FreeWS_mps,SU,WakePct,Gross_GWh,Net_GWh,CF_pct\n';
    R.perTurbine.forEach((t) => {
      csv += [t.id, t.lat, t.lon, t.elev ?? '', t.freeWS.toFixed(3), (+t.SU).toFixed(4), t.wakePct.toFixed(2), t.grossGWh.toFixed(4), t.netGWh.toFixed(4), t.CF.toFixed(2)].join(',') + '\n';
    });
    downloadText(`${base}_AEP_per_turbine_${stamp}.csv`, csv, 'text/csv');

    let sc = 'Dir_deg,Freq,A,k,WS_mps\n';
    R.sectors.forEach((s) => { sc += [s.dir.toFixed(1), s.freq.toFixed(5), s.A.toFixed(3), s.k.toFixed(3), s.WS.toFixed(3)].join(',') + '\n'; });
    downloadText(`${base}_sectors_${stamp}.csv`, sc, 'text/csv');

    let lc = 'Name,Lat,Lon,HH_m\n';
    S.turbines.forEach((t, i) => { lc += `T${i + 1},${t.lat},${t.lon},${t.hh || R.pc.hh}\n`; });
    downloadText(`${base}_layout_${stamp}.csv`, lc, 'text/csv');

    if (S.boundary.length) {
      let bc = 'lon,lat\n';
      S.boundary.forEach((p) => { bc += `${p.lon},${p.lat}\n`; });
      downloadText(`${base}_boundary_${stamp}.csv`, bc, 'text/csv');
    }

    downloadText(`${base}_summary_${stamp}.json`, JSON.stringify({
      project: S.project, results: { ...R, perTurbine: undefined },
      nTurbines: R.n, windMean: S.wind?.meanWS, windSource: R.windSource,
    }, null, 2), 'application/json');

    addLog('Exported report, per-turbine CSV, sectors, layout, boundary, summary JSON', 'o');
  }

  // ─── Demo ────────────────────────────────────────────────────────────────
  function loadDemo() {
    // Hatalageri-like boundary
    const pts = [
      [75.75029, 15.49889], [75.72187, 15.49962], [75.72002, 15.47602], [75.74987, 15.47467],
      [75.75025, 15.45026], [75.77085, 15.45078], [75.77102, 15.4361], [75.7875, 15.4363],
      [75.78785, 15.42455], [75.76721, 15.42426], [75.76733, 15.40248], [75.89329, 15.40173],
      [75.91628, 15.40217], [75.91597, 15.41758], [75.83341, 15.41745], [75.8324, 15.49994],
      [75.75029, 15.49889],
    ].map(([lon, lat]) => ({ lon, lat }));
    S.boundary = pts;
    $('projName').value = 'Hatalageri GO Demo';
    S.project = 'Hatalageri GO Demo';
    $('preset').value = 'en156_140';
    applyPreset();
    $('nWtg').value = 25;
    generateGrid();
    addLog('Demo boundary + 25-WTG grid loaded (Hatalageri area)', 'o');
    refreshSiteUI(); redrawMap();
  }

  // ─── Drag/drop helpers ───────────────────────────────────────────────────
  function bindDrop(zoneId, inputId, handler) {
    const z = $(zoneId), inp = $(inputId);
    z.addEventListener('click', () => inp.click());
    z.addEventListener('dragover', (e) => { e.preventDefault(); z.classList.add('drag'); });
    z.addEventListener('dragleave', () => z.classList.remove('drag'));
    z.addEventListener('drop', async (e) => {
      e.preventDefault(); z.classList.remove('drag');
      const f = e.dataTransfer.files?.[0]; if (f) handler(f);
    });
    inp.addEventListener('change', () => { const f = inp.files?.[0]; if (f) handler(f); });
  }

  // ─── Init ────────────────────────────────────────────────────────────────
  function init() {
    initMap();
    applyPreset();
    bindDrop('dropBoundary', 'fileBoundary', (f) => readPointsFile(f, 'boundary').catch((e) => alert(e.message)));
    bindDrop('dropLayout', 'fileLayout', (f) => readPointsFile(f, 'layout').catch((e) => alert(e.message)));
    bindDrop('dropPC', 'filePC', async (f) => {
      try {
        S.pc = parsePowerCurveCSV(await f.text());
        $('preset').value = 'custom';
        $('rated').value = S.pc.rated;
        addLog(`Power curve uploaded: ${S.pc.ws.length} points, rated ${S.pc.rated} kW`, 'o');
        refreshSiteUI();
      } catch (e) { alert(e.message); }
    });
    bindDrop('dropWind', 'fileWind', (f) => loadWindFile(f).catch((e) => alert(e.message)));

    $('preset').addEventListener('change', applyPreset);
    $('btnBBox').onclick = applyBBox;
    $('btnCenter').onclick = applyCenter;
    $('btnGrid').onclick = generateGrid;
    $('btnTerrain').onclick = () => downloadTerrain().catch((e) => addLog(e.message, 'e'));
    $('btnRough').onclick = () => downloadRoughness().catch((e) => addLog(e.message, 'e'));
    $('btnERA5').onclick = () => downloadERA5().catch((e) => addLog(e.message, 'e'));
    $('btnGWA').onclick = () => downloadGWA().catch((e) => addLog(e.message, 'e'));
    $('btnRun').onclick = () => runAEP();
    $('btnExport').onclick = exportResults;
    $('btnDemo').onclick = loadDemo;
    $('projName').addEventListener('change', () => { S.project = $('projName').value; });

    // Install spectral engine if present
    setTimeout(() => {
      if (window.WFP61 && typeof window.WFP61.install === 'function') {
        try {
          if (!window.runBZModel) window.runBZModel = async () => {};
          window.WFP61.install();
          addLog('WFP61 spectral BZ engine ready', 'o');
        } catch (e) { addLog('WFP61 install: ' + e.message, 'w'); }
      } else {
        addLog('WFP61 engine not found — using elevation orography fallback', 'w');
      }
    }, 300);

    addLog('Wind Farm AEP Studio ready. Load boundary/layout, choose WTG, then Run full AEP.', 'i');
    refreshSiteUI();

    // Public API for automation / external scripts
    window.AEPStudio = {
      S, downloadTerrain, downloadRoughness, downloadERA5, downloadGWA,
      runAEP, exportResults, applyPreset, generateGrid, currentPC,
      loadDemo, redrawMap, refreshSiteUI, updateLegend,
      drawElevationLayer, drawRoughnessLayer, drawSpeedLayer, drawWindPointLayer
    };
    // also bind commonly used names
    window.downloadTerrain = downloadTerrain;
    window.downloadRoughness = downloadRoughness;
    window.downloadERA5 = downloadERA5;
    window.downloadGWA = downloadGWA;
    window.runAEP = runAEP;
    window.exportResults = exportResults;
  }

  document.addEventListener('DOMContentLoaded', init);
})();
