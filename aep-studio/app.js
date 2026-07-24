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
    masts: [],
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
  // ─── CRS: UTM ↔ WGS84 (compact, no external deps) ───────────────────────
  function utmToLatLon(easting, northing, zone, northern = true) {
    // WGS84
    const a = 6378137.0, f = 1 / 298.257223563, k0 = 0.9996;
    const e = Math.sqrt(f * (2 - f));
    const e1sq = e * e / (1 - e * e);
    const x = easting - 500000.0;
    let y = northing;
    if (!northern) y -= 10000000.0;
    const m = y / k0;
    const mu = m / (a * (1 - Math.pow(e, 2) / 4 - 3 * Math.pow(e, 4) / 64 - 5 * Math.pow(e, 6) / 256));
    const e1 = (1 - Math.sqrt(1 - e * e)) / (1 + Math.sqrt(1 - e * e));
    const j1 = 3 * e1 / 2 - 27 * Math.pow(e1, 3) / 32;
    const j2 = 21 * Math.pow(e1, 2) / 16 - 55 * Math.pow(e1, 4) / 32;
    const j3 = 151 * Math.pow(e1, 3) / 96;
    const j4 = 1097 * Math.pow(e1, 4) / 512;
    const fp = mu + j1 * Math.sin(2 * mu) + j2 * Math.sin(4 * mu) + j3 * Math.sin(6 * mu) + j4 * Math.sin(8 * mu);
    const sinfp = Math.sin(fp), cosfp = Math.cos(fp);
    const tanfp = Math.tan(fp);
    const c1 = e1sq * cosfp * cosfp;
    const t1 = tanfp * tanfp;
    const r1 = a * (1 - e * e) / Math.pow(1 - e * e * sinfp * sinfp, 1.5);
    const n1 = a / Math.sqrt(1 - e * e * sinfp * sinfp);
    const d = x / (n1 * k0);
    const q1 = n1 * tanfp / r1;
    const q2 = d * d / 2;
    const q3 = (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * e1sq) * Math.pow(d, 4) / 24;
    const q4 = (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * e1sq - 3 * c1 * c1) * Math.pow(d, 6) / 720;
    const lat = (fp - q1 * (q2 - q3 + q4)) * 180 / Math.PI;
    const q5 = d;
    const q6 = (1 + 2 * t1 + c1) * Math.pow(d, 3) / 6;
    const q7 = (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * e1sq + 24 * t1 * t1) * Math.pow(d, 5) / 120;
    const lon0 = (zone - 1) * 6 - 180 + 3;
    const lon = lon0 + (q5 - q6 + q7) / cosfp * 180 / Math.PI;
    return { lat, lon };
  }

  function latLonToUtm(lat, lon) {
    const zone = Math.floor((lon + 180) / 6) + 1;
    const a = 6378137.0, f = 1 / 298.257223563, k0 = 0.9996;
    const e2 = f * (2 - f);
    const ep2 = e2 / (1 - e2);
    const latr = lat * Math.PI / 180, lonr = lon * Math.PI / 180;
    const lon0 = ((zone - 1) * 6 - 180 + 3) * Math.PI / 180;
    const N = a / Math.sqrt(1 - e2 * Math.sin(latr) ** 2);
    const T = Math.tan(latr) ** 2;
    const C = ep2 * Math.cos(latr) ** 2;
    const A = Math.cos(latr) * (lonr - lon0);
    const M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * latr
      - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * Math.sin(2 * latr)
      + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * Math.sin(4 * latr)
      - (35 * e2 ** 3 / 3072) * Math.sin(6 * latr));
    const easting = k0 * N * (A + (1 - T + C) * A ** 3 / 6
      + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000;
    let northing = k0 * (M + N * Math.tan(latr) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
      + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720));
    const northern = lat >= 0;
    if (!northern) northing += 10000000;
    return { easting, northing, zone, northern };
  }

  function detectSep(line) {
    if (line.includes('\t')) return '\t';
    if (line.includes(';')) return ';';
    if (line.includes(',')) return ',';
    return /\s+/;
  }

  function splitLine(line, sep) {
    if (sep instanceof RegExp) return line.trim().split(sep).filter(Boolean);
    // CSV with quotes
    const out = []; let cur = '', q = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') { q = !q; continue; }
      if (!q && ch === sep) { out.push(cur.trim()); cur = ''; continue; }
      cur += ch;
    }
    out.push(cur.trim());
    return out;
  }

  function normalizeHeader(h) {
    return String(h || '').trim().toLowerCase()
      .replace(/^\uFEFF/, '')
      .replace(/["']/g, '')
      .replace(/\s+/g, '_')
      .replace(/[()]/g, '');
  }

  function findCol(headers, patterns) {
    for (let i = 0; i < headers.length; i++) {
      const h = headers[i];
      for (const p of patterns) {
        if (typeof p === 'string' ? h === p : p.test(h)) return i;
      }
    }
    return -1;
  }

  function looksLikeLatLon(a, b) {
    return Math.abs(a) <= 90 && Math.abs(b) <= 180;
  }
  function looksLikeLonLat(a, b) {
    return Math.abs(a) <= 180 && Math.abs(b) <= 90;
  }
  function looksLikeUtm(a, b) {
    // typical UTM easting 100k–900k, northing 0–10e6
    return a > 50000 && a < 900000 && b > 0 && b < 10000000;
  }

  function resolveUtmZone(lonHint) {
    const zSel = $('utmZone')?.value || 'auto';
    if (zSel !== 'auto') return +zSel;
    if (lonHint != null && isFinite(lonHint)) return Math.floor((lonHint + 180) / 6) + 1;
    // India default often 43/44 — use 43 if unknown
    return 43;
  }

  function convertXY(x, y, mode) {
    // returns {lat, lon} or null
    const crs = $('inCrs')?.value || 'auto';
    const order = $('colOrder')?.value || 'auto';
    const hem = ($('utmHem')?.value || 'N') === 'N';

    let a = x, b = y;
    // apply forced order
    if (order === 'latlon') { /* a=lat b=lon later */ }
    if (order === 'lonlat') { /* a=lon b=lat */ }
    if (order === 'ne') { const t = a; a = b; b = t; } // swap to E,N

    const forceUtm = crs === 'utm' || order === 'en' || order === 'ne';
    const forceLl = crs === 'wgs84' || order === 'latlon' || order === 'lonlat';

    if (!forceUtm && (forceLl || looksLikeLatLon(a, b) || looksLikeLonLat(a, b))) {
      let lat, lon;
      if (order === 'latlon') { lat = a; lon = b; }
      else if (order === 'lonlat') { lon = a; lat = b; }
      else if (looksLikeLatLon(a, b) && !(looksLikeLonLat(a, b) && Math.abs(a) > 90)) {
        // if a is clearly lat
        if (Math.abs(a) <= 90 && (Math.abs(b) > 90 || Math.abs(b) >= Math.abs(a))) { lat = a; lon = b; }
        else { lon = a; lat = b; }
      } else if (looksLikeLonLat(a, b)) {
        lon = a; lat = b;
      } else {
        lon = a; lat = b;
      }
      if (isFinite(lat) && isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180) return { lat, lon, crs: 'wgs84' };
    }

    if (forceUtm || looksLikeUtm(a, b) || looksLikeUtm(b, a)) {
      let E = a, N = b;
      if (order === 'ne' || (!forceUtm && looksLikeUtm(b, a) && !looksLikeUtm(a, b))) { E = b; N = a; }
      // if easting/northing swapped by magnitude (N often larger)
      if (E > 1000000 && N < 1000000) { const t = E; E = N; N = t; }
      const zone = resolveUtmZone(null);
      const northern = hem;
      try {
        const ll = utmToLatLon(E, N, zone, northern);
        if (isFinite(ll.lat) && isFinite(ll.lon)) {
          return { lat: ll.lat, lon: ll.lon, crs: 'utm', zone, easting: E, northing: N };
        }
      } catch (e) { /* fall through */ }
    }
    return null;
  }

  function parseCoordinateTable(text, fileHint = '') {
    const rawLines = text.replace(/^\uFEFF/, '').split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith('#') && !l.startsWith('//'));
    if (!rawLines.length) return [];
    const sep = detectSep(rawLines[0]);
    let headers = splitLine(rawLines[0], sep).map(normalizeHeader);
    const headerLike = headers.some((h) => /lat|lon|lng|long|east|north|utm|zone|name|turb|id|x|y|hh|hub/.test(h));
    let start = 0;
    if (headerLike) start = 1; else headers = [];

    const iLat = findCol(headers, ['lat', 'latitude', 'y_lat', 'ylat', /^lat_/]);
    const iLon = findCol(headers, ['lon', 'lng', 'long', 'longitude', 'x_lon', 'xlon', /^lon_/, /^long_/]);
    const iE = findCol(headers, ['easting', 'east', 'utm_e', 'utm_x', 'x_utm', 'e', 'x']);
    const iN = findCol(headers, ['northing', 'north', 'utm_n', 'utm_y', 'y_utm', 'n', 'y']);
    const iZone = findCol(headers, ['zone', 'utm_zone', 'utmzone']);
    const iName = findCol(headers, ['name', 'id', 'turbine', 'wtg', 'label', 'turb']);
    const iHH = findCol(headers, ['hh', 'hub', 'hub_height', 'hubheight', 'h']);

    const pts = [];
    for (let i = start; i < rawLines.length; i++) {
      const cols = splitLine(rawLines[i], sep);
      if (cols.length < 2) continue;
      let lat, lon, easting, northing, zone, name, hh;
      if (headerLike && iLat >= 0 && iLon >= 0) {
        lat = +cols[iLat]; lon = +cols[iLon];
      } else if (headerLike && iE >= 0 && iN >= 0) {
        easting = +cols[iE]; northing = +cols[iN];
        zone = iZone >= 0 ? parseInt(cols[iZone], 10) : resolveUtmZone(null);
        const hem = ($('utmHem')?.value || 'N') === 'N';
        // if zone like 43N
        if (iZone >= 0 && /s$/i.test(cols[iZone])) { /* south */ }
        const ll = utmToLatLon(easting, northing, zone, !(/s/i.test(String(cols[iZone] || '')) || ($('utmHem')?.value === 'S')));
        lat = ll.lat; lon = ll.lon;
      } else {
        // numeric first two (or skip leading id)
        let c0 = 0;
        if (cols.length >= 3 && !isFinite(+cols[0]) && isFinite(+cols[1])) c0 = 1;
        // id,e,n pattern
        if (cols.length >= 3 && isFinite(+cols[0]) && isFinite(+cols[1]) && isFinite(+cols[2])
            && looksLikeUtm(+cols[1], +cols[2])) {
          // id, easting, northing
          const ll = convertXY(+cols[1], +cols[2], 'utm');
          if (ll) { lat = ll.lat; lon = ll.lon; easting = ll.easting; northing = ll.northing; zone = ll.zone; name = cols[0]; }
        } else {
          const x = +cols[c0], y = +cols[c0 + 1];
          if (!isFinite(x) || !isFinite(y)) continue;
          const ll = convertXY(x, y);
          if (!ll) continue;
          lat = ll.lat; lon = ll.lon; easting = ll.easting; northing = ll.northing; zone = ll.zone;
        }
      }
      if (iName >= 0) name = cols[iName];
      if (iHH >= 0 && isFinite(+cols[iHH])) hh = +cols[iHH];
      if (!isFinite(lat) || !isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) continue;
      pts.push({ lat, lon, easting, northing, zone, name, hh });
    }
    return pts;
  }

  function parseKMLCoords(text) {
    const pts = [];
    // points from Point or coordinates blocks
    const re = /<coordinates[^>]*>([\s\S]*?)<\/coordinates>/gi;
    let m;
    while ((m = re.exec(text))) {
      const block = m[1].trim().split(/[\s\n]+/).filter(Boolean);
      for (const triple of block) {
        const p = triple.split(',').map(Number);
        if (p.length >= 2 && isFinite(p[0]) && isFinite(p[1])) pts.push({ lon: p[0], lat: p[1] });
      }
    }
    // also gx:coord
    const re2 = /<gx:coord>([\s\S]*?)<\/gx:coord>/gi;
    while ((m = re2.exec(text))) {
      const p = m[1].trim().split(/\s+/).map(Number);
      if (p.length >= 2) pts.push({ lon: p[0], lat: p[1] });
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
      else if (g.type === 'Feature') {
        const nm = g.properties?.name || g.properties?.Name || g.properties?.id;
        const hh = g.properties?.hh || g.properties?.hub_height || g.properties?.HH;
        const before = pts.length;
        walk(g.geometry);
        if (nm != null || hh != null) {
          for (let i = before; i < pts.length; i++) {
            if (nm != null) pts[i].name = String(nm);
            if (hh != null && isFinite(+hh)) pts[i].hh = +hh;
          }
        }
      } else if (g.type === 'Point') pts.push({ lon: g.coordinates[0], lat: g.coordinates[1] });
      else if (g.type === 'MultiPoint') g.coordinates.forEach((c) => pts.push({ lon: c[0], lat: c[1] }));
      else if (g.type === 'LineString') g.coordinates.forEach((c) => pts.push({ lon: c[0], lat: c[1] }));
      else if (g.type === 'MultiLineString') g.coordinates.forEach((line) => line.forEach((c) => pts.push({ lon: c[0], lat: c[1] })));
      else if (g.type === 'Polygon') g.coordinates[0].forEach((c) => pts.push({ lon: c[0], lat: c[1] }));
      else if (g.type === 'MultiPolygon') g.coordinates.forEach((poly) => poly[0].forEach((c) => pts.push({ lon: c[0], lat: c[1] })));
    };
    walk(obj);
    return pts;
  }

  async function parseShapefile(files) {
    // files: FileList or File — .zip or .shp(+sidecars)
    if (typeof shp === 'undefined') throw new Error('Shapefile library (shpjs) not loaded');
    let geo;
    if (files instanceof File) {
      const n = files.name.toLowerCase();
      if (n.endsWith('.zip')) {
        geo = await shp(await files.arrayBuffer());
      } else if (n.endsWith('.shp')) {
        throw new Error('Please zip .shp+.dbf+.shx together, or upload the .zip shapefile');
      } else {
        throw new Error('Unsupported shapefile input');
      }
    } else {
      // multiple files
      const arr = Array.from(files);
      const zipFile = arr.find((f) => f.name.toLowerCase().endsWith('.zip'));
      if (zipFile) geo = await shp(await zipFile.arrayBuffer());
      else {
        // build a zip in memory from sidecars
        const z = new JSZip();
        for (const f of arr) z.file(f.name, await f.arrayBuffer());
        const buf = await z.generateAsync({ type: 'arraybuffer' });
        geo = await shp(buf);
      }
    }
    // shp may return FeatureCollection or object of layers
    if (geo.type === 'FeatureCollection') return parseGeoJSON(geo);
    if (Array.isArray(geo)) {
      const pts = [];
      geo.forEach((g) => pts.push(...parseGeoJSON(g)));
      return pts;
    }
    if (geo && typeof geo === 'object') {
      const pts = [];
      Object.values(geo).forEach((g) => { if (g) pts.push(...parseGeoJSON(g)); });
      return pts;
    }
    throw new Error('Could not parse shapefile');
  }

  async function readPointsFile(fileOrList, mode /* boundary|layout */) {
    const files = (fileOrList instanceof FileList || Array.isArray(fileOrList))
      ? Array.from(fileOrList) : [fileOrList];
    const file = files[0];
    if (!file) return;
    const name = file.name.toLowerCase();
    let pts = [];

    try {
      if (name.endsWith('.zip') || name.endsWith('.shp') || files.some((f) => /\.shp$/i.test(f.name))) {
        addLog('Parsing shapefile…', 'i');
        pts = await parseShapefile(files.length > 1 ? files : file);
      } else if (name.endsWith('.kmz')) {
        const text = await parseKMZ(file);
        pts = parseKMLCoords(text);
      } else {
        const text = await file.text();
        if (name.endsWith('.geojson') || name.endsWith('.json') || text.trim().startsWith('{') || text.trim().startsWith('[')) {
          pts = parseGeoJSON(JSON.parse(text));
        } else if (name.endsWith('.kml') || name.endsWith('.xml') || text.includes('<coordinates') || text.includes('<kml')) {
          pts = parseKMLCoords(text);
        } else {
          // csv / txt / dat
          pts = parseCoordinateTable(text, name);
        }
      }
    } catch (e) {
      throw new Error((e && e.message) || String(e));
    }

    // Deduplicate consecutive identical points
    const cleaned = [];
    for (const p of pts) {
      if (!isFinite(p.lat) || !isFinite(p.lon)) continue;
      const prev = cleaned[cleaned.length - 1];
      if (prev && Math.abs(prev.lat - p.lat) < 1e-10 && Math.abs(prev.lon - p.lon) < 1e-10) continue;
      cleaned.push(p);
    }
    pts = cleaned;
    if (!pts.length) throw new Error('No coordinates found in ' + file.name + '. Check CRS/UTM zone settings.');

    // Report CRS detection
    const sample = pts[0];
    const utmN = pts.filter((p) => p.easting != null).length;
    addLog(
      `Parsed ${pts.length} points from ${file.name}` +
      (utmN ? ` (UTM converted, zone≈${sample.zone || $('utmZone')?.value || '?'})` : ' (lon/lat)'),
      'o'
    );

    if (mode === 'boundary') {
      S.boundary = pts.map((p) => ({ lat: p.lat, lon: p.lon }));
      const a = S.boundary[0], b = S.boundary[S.boundary.length - 1];
      if (a.lat !== b.lat || a.lon !== b.lon) S.boundary.push({ ...a });
      addLog(`Boundary loaded: ${S.boundary.length} vertices`, 'o');
    } else {
      const defHH = +$('hh').value || 140;
      S.turbines = pts.map((p, i) => ({
        lat: p.lat, lon: p.lon,
        hh: p.hh != null ? p.hh : defHH,
        name: p.name || `T${i + 1}`,
        easting: p.easting, northing: p.northing, zone: p.zone,
        _customHH: p.hh != null,
      }));
      if (S.boundary.length < 3) {
        S.boundary = convexHull(S.turbines);
        S.boundary.push({ ...S.boundary[0] });
        addLog('Boundary auto-built as convex hull of layout', 'i');
      }
      addLog(`Layout loaded: ${S.turbines.length} turbines`, 'o');
    }
    refreshSiteUI();
    redrawMap({ fit: true });
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
      
      const m = L.marker([t.lat, t.lon], {
        draggable: true,
        icon: L.divIcon({
          className: 'wtg-marker',
          html: `<div style="background:#1b7a4a; width:12px; height:12px; border-radius:50%; border:2px solid #3dd68c; transform:translate(-4px,-4px); box-shadow: 0 0 4px rgba(0,0,0,0.55); cursor: grab;" title="Drag to Move T${i+1}"></div>`,
          iconSize: [12, 12]
        })
      });

      m.bindTooltip(`T${i + 1}: ${t.name || `WTG ${i + 1}`}<br>${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}${elevTxt}${wsTxt}`);
      
      m.bindPopup(`
        <div style="color:var(--text); font-size:11px; font-family:sans-serif; min-width: 140px; padding: 4px 0;">
          <b style="color:var(--ok)">T${i + 1}: ${t.name || `WTG ${i + 1}`}</b><br>
          <b>Lat:</b> ${t.lat.toFixed(5)}<br>
          <b>Lon:</b> ${t.lon.toFixed(5)}<br>
          <b>Elev:</b> ${(t.elev || 0).toFixed(0)} m<br>
          <b>HH:</b> ${(t.hh || 140)} m<br>
          <button class="btn btn-ghost" onclick="window.AEPStudio.deleteTurbine(${i})" style="padding:4px 6px; font-size:9.5px; margin-top:8px; width:100%; border-radius:6px; background:#7a2412; color:#ffebe8; border:1px solid #9e3626; cursor: pointer;">🗑️ Delete WTG</button>
        </div>
      `);

      m.on('dragend', (e) => {
        const newLatLng = e.target.getLatLng();
        t.lat = newLatLng.lat;
        t.lon = newLatLng.lng;
        t.elev = elevAt(t.lat, t.lon) || 0;
        const utm = latLonToUtm(t.lat, t.lon);
        t.easting = utm.easting; t.northing = utm.northing; t.zone = utm.zone;
        addLog(`Moved T${i + 1} to ${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}`, 'i');
        refreshSiteUI();
        redrawMap({ fit: false });
      });

      m.addTo(S.layers.turbines);
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

  // Global pacing so we never stampede free elevation APIs (browser 429 root cause)
  let _elevQueue = Promise.resolve();
  let _lastElevAt = 0;
  const ELEV_MIN_GAP_MS = 700; // min spacing between elevation HTTP calls

  function enqueueElev(fn) {
    const run = _elevQueue.then(async () => {
      const wait = Math.max(0, ELEV_MIN_GAP_MS - (Date.now() - _lastElevAt));
      if (wait) await sleep(wait);
      try {
        return await fn();
      } finally {
        _lastElevAt = Date.now();
      }
    });
    // keep queue alive even on failure
    _elevQueue = run.then(() => {}, () => {});
    return run;
  }

  async function fetchJsonWithRetry(url, opts = {}, retries = 4) {
    let lastErr = null;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const r = await fetch(url, { ...opts, signal: AbortSignal.timeout(opts.timeoutMs || 25000) });
        if (r.status === 429 || r.status === 503) {
          // Honour Retry-After if present; otherwise long backoff (free APIs need this)
          let wait = 15000 * (attempt + 1);
          const ra = r.headers.get('Retry-After');
          if (ra && isFinite(+ra)) wait = Math.max(wait, (+ra) * 1000);
          wait = Math.min(60000, wait + Math.random() * 500);
          addLog(`Rate limited (HTTP ${r.status}) — waiting ${(wait / 1000).toFixed(0)}s before retry ${attempt + 1}/${retries}…`, 'w');
          await sleep(wait);
          lastErr = new Error('HTTP ' + r.status);
          continue;
        }
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const ct = r.headers.get('content-type') || '';
        if (ct.includes('application/json') || ct.includes('text') || ct.includes('json')) {
          return await r.json();
        }
        // try json anyway
        return await r.json();
      } catch (e) {
        lastErr = e;
        const msg = String(e && e.message || e);
        if (attempt < retries && /timeout|network|fetch|429|503|Failed/i.test(msg)) {
          const wait = Math.min(20000, 2000 * Math.pow(1.6, attempt));
          addLog(`Request error (${msg}) — retry in ${(wait / 1000).toFixed(1)}s…`, 'w');
          await sleep(wait);
          continue;
        }
        throw e;
      }
    }
    throw lastErr || new Error('fetch failed');
  }

  async function fetchElevOpenMeteo(lats, lons) {
    // Very small chunks + global queue → strongly reduces HTTP 429
    const CHUNK = 12, elevs = [];
    const nBatch = Math.ceil(lats.length / CHUNK);
    addLog(`Elevation: Open-Meteo ${lats.length} pts in ${nBatch} paced batches…`, 'i');
    for (let c = 0, bi = 0; c < lats.length; c += CHUNK, bi++) {
      const la = lats.slice(c, c + CHUNK), lo = lons.slice(c, c + CHUNK);
      const url = `https://api.open-meteo.com/v1/elevation?latitude=${la.join(',')}&longitude=${lo.join(',')}`;
      try {
        const d = await enqueueElev(() => fetchJsonWithRetry(url, { timeoutMs: 25000 }, 5));
        elevs.push(...(d.elevation || la.map(() => null)));
      } catch (e) {
        addLog(`Open-Meteo batch ${bi + 1}/${nBatch} failed: ${e.message}`, 'w');
        elevs.push(...la.map(() => null));
      }
      if (bi % 3 === 0) setProgress(5 + Math.round(55 * (bi + 1) / nBatch));
    }
    return elevs;
  }

  async function fetchElevOpenElevation(lats, lons) {
    // Alternate free API (POST JSON) — useful when Open-Meteo is 429'd out
    const CHUNK = 40, elevs = [];
    const nBatch = Math.ceil(lats.length / CHUNK);
    addLog(`Elevation: Open-Elevation fallback ${nBatch} batches…`, 'i');
    for (let c = 0, bi = 0; c < lats.length; c += CHUNK, bi++) {
      const la = lats.slice(c, c + CHUNK), lo = lons.slice(c, c + CHUNK);
      const locations = la.map((a, i) => ({ latitude: a, longitude: lo[i] }));
      try {
        const d = await enqueueElev(async () => {
          const r = await fetch('https://api.open-elevation.com/api/v1/lookup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ locations }),
            signal: AbortSignal.timeout(30000),
          });
          if (r.status === 429 || r.status === 503) {
            await sleep(12000);
            throw new Error('HTTP ' + r.status);
          }
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        });
        const arr = d.results || [];
        if (arr.length === la.length) {
          arr.forEach((res) => elevs.push(res.elevation != null ? res.elevation : null));
        } else {
          // partial / unexpected
          for (let i = 0; i < la.length; i++) elevs.push(arr[i] != null ? arr[i].elevation : null);
        }
      } catch (e) {
        addLog(`Open-Elevation batch ${bi + 1}/${nBatch} failed: ${e.message}`, 'w');
        elevs.push(...la.map(() => null));
      }
    }
    return elevs;
  }

  async function fetchElevOpenTopo(lats, lons, ds = 'mapzen') {
    // Often CORS-blocked in browsers; try briefly then give up (don't burn 30s × N)
    const CHUNK = 30, elevs = [];
    const nBatch = Math.ceil(lats.length / CHUNK);
    let corsBlocked = 0;
    for (let c = 0, bi = 0; c < lats.length; c += CHUNK, bi++) {
      if (corsBlocked >= 2) {
        // stop hammering after repeated CORS failures
        elevs.push(...lats.slice(c).map(() => null));
        break;
      }
      const la = lats.slice(c, c + CHUNK), lo = lons.slice(c, c + CHUNK);
      const locs = la.map((a, i) => `${a},${lo[i]}`).join('|');
      const base = `https://api.opentopodata.org/v1/${ds}?locations=${locs}`;
      try {
        const d = await enqueueElev(() => fetchJsonWithRetry(base, { timeoutMs: 15000 }, 1));
        (d.results || []).forEach((res) => elevs.push(res.elevation != null ? res.elevation : null));
      } catch (e) {
        corsBlocked++;
        addLog(`OpenTopo batch failed: ${e.message}`, 'w');
        elevs.push(...la.map(() => null));
      }
      if (c + CHUNK < lats.length) await sleep(1100);
    }
    return elevs;
  }

  /** Fill null elevations by nearest-neighbor from valid cells */
  function fillElevationNulls(elevs, ny, nx) {
    const a = elevs.slice();
    const valid = a.filter((v) => v != null && isFinite(v));
    if (!valid.length) return a;
    const fallback = mean(valid);
    for (let pass = 0; pass < 5; pass++) {
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

  function terrainCacheKey(lat0, lon0, lat1, lon1, ng) {
    return `aep_terr_v2_${lat0.toFixed(3)}_${lon0.toFixed(3)}_${lat1.toFixed(3)}_${lon1.toFixed(3)}_${ng}`;
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
    let ng = +$('terrG').value || 30;
    if (ng > 40) {
      addLog(`Grid ${ng}×${ng} is heavy on free APIs — capping to 40 to avoid HTTP 429`, 'w');
      ng = 40;
      if ($('terrG')) $('terrG').value = '40';
    }
    addLog(`Terrain: center ${c.lat.toFixed(4)}, ${c.lon.toFixed(4)} radius ${rad.toFixed(1)} km grid ${ng}×${ng}`, 'i');
    setProgress(5);

    const dLat = rad / 111.32, dLon = rad / (111.32 * Math.cos(c.lat * Math.PI / 180));
    const lat0 = c.lat - dLat, lat1 = c.lat + dLat, lon0 = c.lon - dLon, lon1 = c.lon + dLon;
    const gLats = [], gLons = [];
    for (let i = 0; i < ng; i++) for (let j = 0; j < ng; j++) {
      gLats.push(+(lat0 + i * (lat1 - lat0) / (ng - 1)).toFixed(4));
      gLons.push(+(lon0 + j * (lon1 - lon0) / (ng - 1)).toFixed(4));
    }

    // Session cache — re-running AEP won't re-hit APIs for same bbox/grid
    const ck = terrainCacheKey(lat0, lon0, lat1, lon1, ng);
    let elevs = null, source = 'Open-Meteo';
    try {
      const cached = sessionStorage.getItem(ck);
      if (cached) {
        const obj = JSON.parse(cached);
        if (obj && obj.elevs && obj.elevs.length === gLats.length) {
          elevs = obj.elevs; source = (obj.source || 'cache') + ' (cached)';
          addLog(`Terrain cache hit — ${gLats.length} pts (${source})`, 'o');
        }
      }
    } catch (e) {}

    if (!elevs) {
      elevs = await fetchElevOpenMeteo(gLats, gLons);
      let valid = elevs.filter((e) => e != null && isFinite(e)).length;
      addLog(`Open-Meteo: ${valid}/${gLats.length} valid points`, valid > 0 ? 'i' : 'w');

      if (valid < gLats.length * 0.7) {
        addLog('Trying Open-Elevation API for gaps…', 'w');
        try {
          const e2 = await fetchElevOpenElevation(gLats, gLons);
          for (let i = 0; i < elevs.length; i++) {
            if ((elevs[i] == null || !isFinite(elevs[i])) && e2[i] != null && isFinite(e2[i])) elevs[i] = e2[i];
          }
          const v2 = elevs.filter((e) => e != null && isFinite(e)).length;
          if (v2 > valid) { valid = v2; source = 'Open-Meteo+Open-Elevation'; }
        } catch (e) {
          addLog('Open-Elevation skipped: ' + e.message, 'w');
        }
      }

      if (valid < gLats.length * 0.5) {
        addLog('Trying OpenTopoData (may fail in browser due to CORS)…', 'w');
        try {
          const e3 = await fetchElevOpenTopo(gLats, gLons, 'mapzen');
          for (let i = 0; i < elevs.length; i++) {
            if ((elevs[i] == null || !isFinite(elevs[i])) && e3[i] != null && isFinite(e3[i])) elevs[i] = e3[i];
          }
          const v3 = elevs.filter((e) => e != null && isFinite(e)).length;
          if (v3 > valid) { valid = v3; source = source.includes('+') ? source + '+OpenTopo' : 'Open-Meteo+OpenTopo'; }
        } catch (e) {
          addLog('OpenTopo skipped: ' + e.message, 'w');
        }
      }

      // Never leave nulls as 0 (false flat) — interpolate gaps
      elevs = fillElevationNulls(elevs, ng, ng);
      try {
        sessionStorage.setItem(ck, JSON.stringify({ elevs, source, ts: Date.now() }));
      } catch (e) {}
    } else {
      // still ensure no nulls
      elevs = fillElevationNulls(elevs, ng, ng);
    }
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
  async function fetchOverpassJSON(query, timeoutMs = 18000) {
    const endpoints = [
      'https://overpass-api.de/api/interpreter',
      'https://overpass.kumi.systems/api/interpreter',
    ];
    let last = null;
    for (const ep of endpoints) {
      try {
        addLog(`OSM Overpass: ${ep.split('/')[2]}…`, 'i');
        const r = await fetch(ep, {
          method: 'POST',
          body: 'data=' + encodeURIComponent(query),
          headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json' },
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!r.ok) { last = new Error('HTTP ' + r.status); addLog('Overpass ' + ep.split('/')[2] + ' HTTP ' + r.status, 'w'); continue; }
        const d = await r.json();
        if (d && d.elements) return d;
        last = new Error('empty Overpass response');
      } catch (e) {
        last = e;
        addLog('Overpass fail ' + ep.split('/')[2] + ': ' + e.message, 'w');
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
    addLog('Roughness: querying OSM (short timeout — will use z₀=' + $('z0').value + ' if busy)…', 'i');
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
      const d = await fetchOverpassJSON(q, 18000);
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

  function sectorizeWindSeries(speeds, dirs, nSec) {
    const secW = 360 / nSec;
    const sec = Array.from({ length: nSec }, () => ({ n: 0, sum: 0, ws: [] }));
    for (let i = 0; i < speeds.length; i++) {
      if (speeds[i] < 0.5) continue;
      const si = Math.floor((((dirs[i] % 360) + 360) % 360) / secW) % nSec;
      sec[si].n++; sec[si].sum += speeds[i]; sec[si].ws.push(speeds[i]);
    }
    const nTot = sec.reduce((s, x) => s + x.n, 0) || 1;
    return sec.map((s, i) => {
      const wb = weibullFit(s.ws.length ? s.ws : [1]);
      return {
        dir: (i + 0.5) * secW,
        freq: s.n / nTot,
        A: wb.A, k: wb.k,
        WS: s.n ? s.sum / s.n : 0,
      };
    });
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

    // Add or update mast in Multi-Mast list S.masts
    if (!S.masts) S.masts = [];
    const existingIdx = S.masts.findIndex(m => m.name === w.source);
    const nSec = +$('nSec').value || 16;
    const mastSectors = sectorizeWindSeries(sp, dr, nSec);
    const newMast = {
      name: w.source,
      lat: w.lat,
      lon: w.lon,
      height: w.height,
      meanWS: w.meanWS,
      speeds: [...sp],
      dirs: [...dr],
      sectors: mastSectors,
      active: true
    };
    if (existingIdx >= 0) {
      S.masts[existingIdx] = newMast;
    } else {
      S.masts.push(newMast);
    }
    refreshMastsUI();

    chipBox('windChips', [
      [`${w.source}`, true],
      [`n=${sp.length.toLocaleString()}`, true],
      [`mean ${w.meanWS.toFixed(2)} m/s @ ${w.height} m`, true],
      [`A=${wb.A.toFixed(2)} k=${wb.k.toFixed(2)}`, true],
      [w.lat != null ? `pt ${Number(w.lat).toFixed(3)}, ${Number(w.lon).toFixed(3)}` : 'pt n/a', w.lat != null],
    ]);
    setStep('wind', 'done');
    addLog(`Wind ${w.source}: ${sp.length} samples, mean ${w.meanWS.toFixed(2)} m/s @ ${w.height} m`, 'o');
    updateWindStatusUI();
    redrawMap({ fit: false });
    return w;
  }

  function updateWindStatusUI() {
    const el = $('windStatus');
    if (!el) return;
    if (S.wind) {
      el.style.color = 'var(--ok)';
      el.style.background = '#14301f';
      el.style.borderColor = '#1f5a3a';
      el.innerHTML = `🟢 <b>Active in State:</b> ${S.wind.source} (Mean ${S.wind.meanWS.toFixed(2)} m/s @ ${S.wind.height}m)<br>` +
                     `<span style="color:var(--muted); font-size:10px;">AEP will use this dataset unless you switch primary source and run again.</span>`;
    } else {
      el.style.color = 'var(--muted)';
      el.style.background = 'var(--input)';
      el.style.borderColor = 'var(--line)';
      el.innerHTML = `⚪ <b>Active in State:</b> None (will download/load the selected primary source on run)`;
    }
  }

  function refreshMastsUI() {
    const el = $('mastsTableBody');
    if (!el) return;
    if (!S.masts || !S.masts.length) {
      el.innerHTML = '<tr><td colspan="5" class="muted text-center" style="font-size:10px; padding:8px 0;">No met masts loaded. Add GWA/ERA5 or upload CSV.</td></tr>';
      return;
    }
    el.innerHTML = S.masts.map((m, idx) => `
      <tr style="border-bottom: 1px solid var(--line);">
        <td><b style="color:var(--ok)">${m.name}</b></td>
        <td>${m.lat.toFixed(4)}, ${m.lon.toFixed(4)}</td>
        <td>${m.height} m</td>
        <td>${m.meanWS.toFixed(2)} m/s</td>
        <td style="text-align: right;">
          <input type="checkbox" id="mast_active_${idx}" ${m.active ? 'checked' : ''} onchange="window.AEPStudio.toggleMast(${idx})" style="width: auto; display: inline-block; margin-right: 8px; cursor: pointer;" />
          <button class="btn btn-ghost" onclick="window.AEPStudio.deleteMast(${idx})" style="padding: 2px 6px; font-size: 10px; background: #5a1212; color: #ffe8e8; border: 1px solid #7a2222; border-radius: 4px; cursor: pointer;">🗑️</button>
        </td>
      </tr>
    `).join('');
  }

  function toggleMast(idx) {
    if (S.masts && S.masts[idx]) {
      S.masts[idx].active = !S.masts[idx].active;
      addLog(`Toggled met mast ${S.masts[idx].name}: ${S.masts[idx].active ? 'Active' : 'Disabled'}`, 'i');
      refreshMastsUI();
    }
  }

  function deleteMast(idx) {
    if (S.masts && S.masts[idx]) {
      const name = S.masts[idx].name;
      S.masts.splice(idx, 1);
      addLog(`Deleted met mast: ${name}`, 'w');
      refreshMastsUI();
    }
  }

  function getDistanceM(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  async function downloadERA5() {
    const pts = S.boundary.length ? S.boundary : S.turbines;
    if (!pts.length) { alert('Set site first'); return; }
    const c = centerOf(pts);
    const y0 = +$('y0').value || 2019, y1 = +$('y1').value || 2023;
    if (y0 < 2000 || y1 < 2000) {
      alert('ERA5 data downloads are only allowed from year 2000 onwards.');
      return;
    }
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
        if (y < y1) await sleep(1200);
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

  function triggerTextDownload(filename, text) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  async function fetchGwaLibText(lat, lon) {
    const libUrl = `https://globalwindatlas.info/api/gwa/custom/Lib?lat=${Number(lat).toFixed(4)}&long=${Number(lon).toFixed(4)}`;
    // Direct browser navigation to this URL returns HTTP 400 (API wants XHR-style headers).
    // From JS we must use proxies or a manual PowerShell/curl download.
    const tryList = [
      { label: 'corsproxy.io', url: 'https://corsproxy.io/?' + encodeURIComponent(libUrl) },
      { label: 'allorigins', url: 'https://api.allorigins.win/raw?url=' + encodeURIComponent(libUrl) },
      { label: 'codetabs', url: 'https://api.codetabs.com/v1/proxy?quest=' + encodeURIComponent(libUrl) },
      { label: 'direct', url: libUrl, headers: {
          'X-Requested-With': 'XMLHttpRequest',
          'Accept': '*/*',
          'Referer': 'https://globalwindatlas.info/',
        } },
    ];
    let lastErr = null;
    for (const item of tryList) {
      try {
        addLog(`GWA fetch try: ${item.label}…`, 'i');
        const r = await fetch(item.url, {
          headers: item.headers || { 'Accept': '*/*' },
          signal: AbortSignal.timeout(40000),
        });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        let body = await r.text();
        if (body.trim().startsWith('{') && body.includes('"contents"')) {
          try {
            const j = JSON.parse(body);
            if (j.contents) body = j.contents;
          } catch (_) {}
        }
        if (body && body.length > 80
            && !/^\s*<!DOCTYPE/i.test(body)
            && !/^\s*<html/i.test(body)
            && (body.includes('Generalized Wind Climate') || body.includes('<coordinates>') || /\b\d+\s+\d+\s+\d+\b/.test(body))) {
          addLog('GWA .lib received via ' + item.label, 'o');
          return { text: body, libUrl, via: item.label };
        }
        lastErr = item.label + ': unexpected body';
      } catch (e) {
        lastErr = (item.label + ': ' + (e.message || e));
        addLog('GWA try fail (' + item.label + '): ' + (e.message || e), 'w');
      }
    }
    throw new Error(lastErr || 'All GWA fetch routes failed');
  }

  function showGwaHelpPanel(libUrl, lat, lon, lastErr) {
    let panel = document.getElementById('gwaHelpPanel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'gwaHelpPanel';
      panel.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:99999;display:flex;align-items:center;justify-content:center;padding:16px';
      document.body.appendChild(panel);
    }
    const la = Number(lat).toFixed(4);
    const lo = Number(lon).toFixed(4);
    const gwaMap = 'https://globalwindatlas.info/en/' + la + '/' + lo;
    // IMPORTANT: do not put $vars into innerHTML (can break on copy). Build plain strings.
    const fname = 'gwa_' + la + '_' + lo + '.lib';
    // PowerShell: use single-quoted URL so & is safe; use ${} only in JS below when assigning .value
    const psLines = [
      '$ErrorActionPreference = \'Stop\'',
      '$url = \'' + libUrl.replace(/'/g, "''") + '\'',
      '$out = Join-Path $env:USERPROFILE (Join-Path \'Downloads\' \'' + fname + '\')',
      '$headers = @{',
      '  \'X-Requested-With\' = \'XMLHttpRequest\'',
      '  \'Referer\' = \'https://globalwindatlas.info/\'',
      '  \'User-Agent\' = \'Mozilla/5.0\'',
      '}',
      'Invoke-WebRequest -Uri $url -Headers $headers -OutFile $out',
      'Write-Host ("Saved: " + $out)',
      'Get-Item $out | Format-List FullName, Length',
    ].join('\n');
    const curlCmd = 'curl -L -H "X-Requested-With: XMLHttpRequest" -H "Referer: https://globalwindatlas.info/" -A "Mozilla/5.0" "' +
      libUrl.replace(/"/g, '\\"') + '" -o "' + fname + '"';

    const errHtml = lastErr
      ? '<br><span style="color:#f5b942">Last error: ' +
        String(lastErr).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') +
        '</span>'
      : '';

    panel.innerHTML =
      '<div style="background:#121a2b;border:1px solid #35507a;border-radius:14px;max-width:560px;width:100%;padding:18px;color:#e8eefc;box-shadow:0 12px 40px rgba(0,0,0,.45);max-height:90vh;overflow:auto">' +
      '<div style="font-weight:700;font-size:1rem;margin-bottom:8px">Get GWA .lib for this site</div>' +
      '<p style="margin:0 0 10px;color:#9aabcc;font-size:.82rem;line-height:1.45">' +
      'Do <b>not</b> open the API URL in the browser address bar (returns <b>400 Bad Request</b>).' +
      errHtml +
      '</p>' +
      '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px">' +
      '<button type="button" id="gwaAutoDlBtn" style="background:#3d8bfd;color:#fff;border:0;padding:9px 12px;border-radius:9px;font-weight:600;font-size:.82rem;cursor:pointer">1) Auto-download .lib</button>' +
      '<button type="button" id="gwaUploadBtn" style="background:#1b7a4a;color:#fff;border:0;padding:9px 12px;border-radius:9px;font-weight:600;font-size:.82rem;cursor:pointer">2) Upload .lib file</button>' +
      '<button type="button" id="gwaUseEra5Btn" style="background:#1c2940;color:#e8eefc;border:1px solid #35507a;padding:9px 12px;border-radius:9px;font-size:.82rem;cursor:pointer">3) Use ERA5 instead</button>' +
      '</div>' +
      '<div id="gwaAutoStatus" style="font-size:.78rem;color:#9aabcc;min-height:1.2em;margin-bottom:10px"></div>' +
      '<details open style="margin-bottom:10px">' +
      '<summary style="cursor:pointer;color:#9ec5ff;font-size:.8rem">Windows PowerShell (recommended if auto fails)</summary>' +
      '<p style="font-size:.72rem;color:#9aabcc;margin:8px 0">Click <b>Copy PowerShell</b>, paste into PowerShell, press Enter. Then click <b>Upload .lib file</b> and choose the file from your Downloads folder.</p>' +
      '<textarea id="gwaPsCmd" readonly style="width:100%;min-height:120px;background:#0e1626;border:1px solid #35507a;color:#e8eefc;border-radius:8px;padding:8px;font-size:.72rem;font-family:Consolas,monospace"></textarea>' +
      '<button type="button" id="gwaCopyPs" style="margin-top:6px;background:#1c2940;color:#e8eefc;border:1px solid #35507a;padding:6px 10px;border-radius:8px;cursor:pointer;font-size:.75rem">Copy PowerShell</button>' +
      '</details>' +
      '<details style="margin-bottom:10px">' +
      '<summary style="cursor:pointer;color:#9ec5ff;font-size:.8rem">curl (Mac / Linux / Git Bash)</summary>' +
      '<textarea id="gwaCurlCmd" readonly style="width:100%;min-height:56px;background:#0e1626;border:1px solid #35507a;color:#e8eefc;border-radius:8px;padding:8px;font-size:.72rem;font-family:Consolas,monospace;margin-top:8px"></textarea>' +
      '<button type="button" id="gwaCopyCurl" style="margin-top:6px;background:#1c2940;color:#e8eefc;border:1px solid #35507a;padding:6px 10px;border-radius:8px;cursor:pointer;font-size:.75rem">Copy curl</button>' +
      '</details>' +
      '<p style="margin:0;color:#9aabcc;font-size:.72rem">Site: <code>' + la + ', ' + lo + '</code> · ' +
      '<a href="' + gwaMap + '" target="_blank" rel="noopener" style="color:#9ec5ff">Open GWA map</a></p>' +
      '<input type="file" id="gwaLibFile" accept=".lib,.txt,.dat,text/plain" style="display:none"/>' +
      '<div style="text-align:right;margin-top:12px">' +
      '<button type="button" id="gwaHelpClose" style="background:transparent;border:1px solid #35507a;color:#9aabcc;padding:7px 12px;border-radius:8px;cursor:pointer">Close</button>' +
      '</div></div>';

    panel.style.display = 'flex';

    // Set commands via .value so & and $ are never HTML-escaped/corrupted
    panel.querySelector('#gwaPsCmd').value = psLines;
    panel.querySelector('#gwaCurlCmd').value = curlCmd;

    const close = () => { panel.style.display = 'none'; };
    const statusEl = () => panel.querySelector('#gwaAutoStatus');

    panel.querySelector('#gwaHelpClose').onclick = close;
    panel.onclick = (e) => { if (e.target === panel) close(); };

    panel.querySelector('#gwaAutoDlBtn').onclick = async () => {
      const st = statusEl();
      st.style.color = '#9ec5ff';
      st.textContent = 'Downloading via proxy (up to ~40s)…';
      try {
        const res = await fetchGwaLibText(lat, lon);
        triggerTextDownload(fname, res.text);
        st.style.color = '#3dd68c';
        st.textContent = 'Got .lib via ' + res.via + ' — loading into app…';
        await ingestGwaLibText(res.text, { lat: lat, lon: lon, source: 'GWA', fileName: fname });
        st.textContent = 'GWA loaded. Close this panel and click Run full AEP.';
      } catch (e) {
        st.style.color = '#ff6b7a';
        st.textContent = 'Auto-download failed: ' + e.message + ' — use Copy PowerShell below, then Upload .lib file.';
        addLog('GWA auto-download failed: ' + e.message, 'e');
      }
    };

    panel.querySelector('#gwaUploadBtn').onclick = () => panel.querySelector('#gwaLibFile').click();
    panel.querySelector('#gwaLibFile').onchange = async (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      try {
        statusEl().textContent = 'Parsing ' + f.name + '…';
        await ingestGwaLibText(await f.text(), { lat: lat, lon: lon, source: 'GWA_FILE', fileName: f.name });
        statusEl().style.color = '#3dd68c';
        statusEl().textContent = 'GWA loaded from file.';
        close();
      } catch (err) {
        statusEl().style.color = '#ff6b7a';
        statusEl().textContent = 'Parse failed: ' + err.message;
        alert('Could not parse file.\\n\\n' + err.message + '\\n\\nValid GWA files start with: GWA4 Generalized Wind Climate\\n(Do not upload a 400 Bad Request HTML page.)');
      }
    };

    panel.querySelector('#gwaUseEra5Btn').onclick = async () => {
      close();
      if ($('windSrc')) $('windSrc').value = 'ERA5';
      addLog('Switching to ERA5 download…', 'i');
      try { await downloadERA5(); } catch (e) { addLog('ERA5 failed: ' + e.message, 'e'); }
    };

    function copyArea(sel, btn, resetLabel) {
      const el = panel.querySelector(sel);
      const text = el.value;
      const done = () => { btn.textContent = 'Copied!'; setTimeout(() => { btn.textContent = resetLabel; }, 1500); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(() => {
          el.focus(); el.select(); document.execCommand('copy'); done();
        });
      } else {
        el.focus(); el.select(); document.execCommand('copy'); done();
      }
    }
    panel.querySelector('#gwaCopyPs').onclick = function () { copyArea('#gwaPsCmd', this, 'Copy PowerShell'); };
    panel.querySelector('#gwaCopyCurl').onclick = function () { copyArea('#gwaCurlCmd', this, 'Copy curl'); };
  }

  async function ingestGwaLibText(text, meta = {}) {
    const hh = +$('hh').value || 140;
    const z0 = +$('z0').value || 0.03;
    if (!text || text.length < 50) throw new Error('File too short / empty');
    if (text.trim().startsWith('<!DOCTYPE') || text.trim().startsWith('<html')) {
      throw new Error('File looks like HTML, not a GWA .lib');
    }
    const g = parseGWALib(text, hh, z0);
    let lat = meta.lat, lon = meta.lon;
    // Try coordinates embedded in GWA header
    const m = text.match(/<coordinates>\s*([-\d.]+)\s*,\s*([-\d.]+)/i);
    if (m) { lon = +m[1]; lat = +m[2]; }
    if (lat == null || lon == null || !isFinite(lat) || !isFinite(lon)) {
      const pts = S.boundary.length ? S.boundary : S.turbines;
      if (pts.length) {
        const c = centerOf(pts);
        lat = c.lat; lon = c.lon;
      }
    }
    setWind(g.speeds, g.dirs, {
      source: meta.source || 'GWA',
      height: hh,
      lat, lon,
    });
    S.gwaMeta = { meanWS: g.meanWS, sectors: g.sectors, z0: g.z0, fileName: meta.fileName || null };
    addLog(
      `GWA climate loaded${meta.fileName ? ' from ' + meta.fileName : ''}: mean ≈ ${g.meanWS.toFixed(2)} m/s @ ${hh} m (z0=${g.z0})`,
      'o'
    );
    setStep('wind', 'done');
    setProgress(100);
    return true;
  }

  async function downloadGWA() {
    const pts = S.boundary.length ? S.boundary : S.turbines;
    if (!pts.length) { alert('Set site first'); return false; }
    const c = centerOf(pts);
    const libUrl = `https://globalwindatlas.info/api/gwa/custom/Lib?lat=${c.lat.toFixed(4)}&long=${c.lon.toFixed(4)}`;
    setStep('wind', 'run');
    addLog(`GWA 4.0 .lib @ ${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}`, 'i');
    addLog('Note: opening the API URL in a browser tab gives HTTP 400 — use proxy download or PowerShell.', 'i');

    try {
      const res = await fetchGwaLibText(c.lat, c.lon);
      const fname = `gwa_${c.lat.toFixed(4)}_${c.lon.toFixed(4)}.lib`;
      // Also offer file save for user records
      try { triggerTextDownload(fname, res.text); } catch (_) {}
      await ingestGwaLibText(res.text, { lat: c.lat, lon: c.lon, source: 'GWA', fileName: fname });
      return true;
    } catch (e) {
      addLog('GWA auto-fetch failed: ' + e.message, 'e');
      showGwaHelpPanel(libUrl, c.lat, c.lon, e.message);
      setStep('wind', '');
      return false;
    }
  }

  function loadWindFile(file) {
    return file.text().then(async (text) => {
      if (file.name.toLowerCase().endsWith('.lib') || text.includes('Generalized Wind Climate') || /^\s*\d+\s+\d+\s+\d+/m.test(text)) {
        const pts = S.boundary.length ? S.boundary : S.turbines;
        const c = pts.length ? centerOf(pts) : { lat: null, lon: null };
        await ingestGwaLibText(text, { lat: c.lat, lon: c.lon, source: 'GWA_FILE', fileName: file.name });
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

  function powerLaw(ws, z1, z2, alpha) {
    if (z1 <= 0 || z2 <= 0) return ws;
    return ws * Math.pow(z2 / z1, alpha);
  }

  function sectorZ0(dirDeg) {
    const rose = S.autoRoughnessRose || S.roughnessRose;
    const z0def = +$('z0').value || 0.03;
    if (!rose || !rose.length) return z0def;
    const n = rose.length;
    const si = Math.floor((((dirDeg % 360) + 360) % 360) / (360 / n)) % n;
    const z = rose[si]?.z0;
    if (Array.isArray(z) && z.length) return Math.max(1e-5, +z[0] || z0def);
    if (typeof z === 'number') return Math.max(1e-5, z);
    return z0def;
  }

  /** Vertical extrapolation of a speed sample with method + optional direction for sector z0 */
  function verticalExtrapolate(ws, zData, zHub, dirDeg) {
    const method = $('vertMethod')?.value || 'log';
    if (Math.abs(zData - zHub) < 0.5) return ws;
    if (method === 'power') {
      const a = +$('shearAlpha').value || 0.14;
      return powerLaw(ws, zData, zHub, a);
    }
    let z0 = +$('z0').value || 0.03;
    if (method === 'log_sector' && dirDeg != null) z0 = sectorZ0(dirDeg);
    return logLaw(ws, zData, zHub, z0);
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

  // ─── Wake: Eddy Viscosity (Ainslie-Gaussian) ──────────────────────────────
  function ainslieDeficit(dx, dy, D, ct, TI) {
    const xPrime = dx / D;
    if (xPrime <= 0) return 0;
    
    // Ainslie model is initialized at 2.0 D downstream
    let Dm0 = ct - 0.05 - (16 * ct - 0.5) * (TI * 100) / 1000;
    Dm0 = Math.max(0.01, Dm0);
    
    const Dm_rotor = 1 - Math.sqrt(Math.max(0.01, 1 - ct));
    
    if (xPrime < 2.0) {
      // Linear interpolation in the near wake (0 to 2D)
      const t = xPrime / 2.0;
      const Dm = Dm_rotor * (1 - t) + Dm0 * t;
      const Bw = Math.sqrt((0.445 * ct) / Math.max(1e-4, Dm * (1 - 0.5 * Dm)));
      const rPrime = dy / D;
      return Dm * Math.exp(-3.56 * rPrime * rPrime / (Bw * Bw));
    }
    
    // Integrate ODE from xPrime = 2.0 to the target xPrime
    let currentX = 2.0;
    let Dm = Dm0;
    const step = 0.5; // step size in rotor diameters
    
    while (currentX < xPrime) {
      const h = Math.min(step, xPrime - currentX);
      const Bw = Math.sqrt((0.445 * ct) / Math.max(1e-4, Dm * (1 - 0.5 * Dm)));
      const eps_amb = 0.16 * TI;
      const eps_shear = 0.015 * Bw * Dm;
      const eps = eps_amb + eps_shear;
      
      const dDm_dx = - (16 * eps * Dm) / Math.max(1e-4, Bw * Bw * (2 - Dm));
      
      Dm += dDm_dx * h;
      Dm = Math.max(0.01, Dm);
      currentX += h;
    }
    
    const Bw = Math.sqrt((0.445 * ct) / Math.max(1e-4, Dm * (1 - 0.5 * Dm)));
    const rPrime = dy / D;
    return Dm * Math.exp(-3.56 * rPrime * rPrime / (Bw * Bw));
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
      const desiredSrc = $('windSrc').value;
      if (S.windSources[desiredSrc]) {
        S.wind = S.windSources[desiredSrc];
        addLog(`Using cached wind source: ${desiredSrc}`, 'i');
      } else if (desiredSrc === 'GWA' && S.windSources['GWA_FILE']) {
        S.wind = S.windSources['GWA_FILE'];
        addLog(`Using cached wind source: GWA_FILE`, 'i');
      } else if (S.wind && S.wind.source !== desiredSrc && !(desiredSrc === 'GWA' && S.wind.source === 'GWA_FILE')) {
        S.wind = null;
      }

      if (!S.wind || !S.wind.speeds?.length) {
        addLog('Auto wind: ' + desiredSrc, 'i');
        if (desiredSrc === 'GWA') await downloadGWA();
        else if (desiredSrc === 'SITE' || desiredSrc === 'MESO') { alert(`Upload ${desiredSrc === 'SITE' ? 'site mast' : 'mesoscale'} wind CSV first`); return; }
        else await downloadERA5();
      }
      if (!S.wind?.speeds?.length) { addLog('No wind data', 'e'); return; }
      setProgress(55);

      // Extrapolate to hub
      const z0 = +$('z0').value || 0.03;
      const dataH = S.wind.height || +$('dataH').value || 100;
      const hubDir = S.wind.dirs.slice();
      const vMethod = $('vertMethod')?.value || 'log';
      const hubSp = S.wind.speeds.map((v, i) => verticalExtrapolate(v, dataH, pc.hh, hubDir[i]));
      addLog(`Vertical extrap ${dataH}→${pc.hh} m (${vMethod}${vMethod === 'power' ? ' α=' + ($('shearAlpha')?.value || 0.14) : ', z0≈' + z0})`, 'i');
      addLog('Horizontal: per-sector orographic SU (BZ) × roughness change, then Bastankhah wakes', 'i');

      // Orography (turbines + active masts to get speedups at both!)
      setProgress(65);
      const origTurbines = S.turbines;
      const activeMasts = (S.masts || []).filter(m => m.active);
      const flowPoints = S.turbines.concat(activeMasts.map(m => ({ lat: m.lat, lon: m.lon, name: m.name })));
      S.turbines = flowPoints;
      await runOrography();
      S.turbines = origTurbines;
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
      const useEV = ($('wakeModel')?.value || 'ev') === 'ev';
      const TI_val = (+$('wakeTI')?.value || 12) / 100.0;
      const rho = 1.225;

      // Resolve IEC 61400-15-2 Losses & Markov Availability
      const lossElec = +$('lossElec').value || 0;
      const lossEnv = +$('lossEnv').value || 0;
      const markovFail = +$('markovFail').value || 0;
      const markovMTTR = +$('markovMTTR').value || 0;
      const markovMTTD = +$('markovMTTD').value || 0;
      
      const lambda = markovFail;
      const mu = 8760 / Math.max(0.1, markovMTTR);
      const gamma_val = 8760 / Math.max(0.1, markovMTTD);
      const availRatio = 1 / (1 + lambda / mu + lambda / gamma_val);
      
      // Combined Downtime Loss Factor (ratio)
      const L_downtime = (1 - lossElec / 100) * (1 - lossEnv / 100) * availRatio;
      
      // Non-Downtime Loss Factor (ratio)
      const lossDegradation = +$('lossDegradation').value || 0;
      const lossSuboptimal = +$('lossSuboptimal').value || 0;
      const L_non_down = (1 - lossDegradation / 100) * (1 - lossSuboptimal / 100);

      // Setup rating first
      const ur = getRatedWS(pc);

      const per = S.turbines.map(() => ({ gross: 0, net: 0, wSum: 0, fwsSum: 0, wakeSum: 0, fwsN: 0 }));
      const hours = 8760;

      const calcMethod = $('calcMethod')?.value || 'time_series';
      addLog(`Calculation method: ${calcMethod === 'time_series' ? 'Time Series (Step-by-step)' : 'Statistical (Weibull Bins)'}`, 'i');
      addLog(`IEC 61400-15-2 Losses: Downtime factor=${(L_downtime * 100).toFixed(2)}% | Non-downtime partial-load factor=${(L_non_down * 100).toFixed(2)}%`, 'i');

      const isMultiMast = activeMasts.length > 1;

      // Precompute IDW weights for each turbine to all active masts
      const turbineWeights = S.turbines.map((t) => {
        if (!isMultiMast) return [];
        const weights = activeMasts.map((m) => {
          const dist = getDistanceM(t.lat, t.lon, m.lat, m.lon);
          return 1 / Math.pow(Math.max(10, dist), 2);
        });
        const sum = weights.reduce((a,b) => a+b, 0) || 1;
        return weights.map(w => w / sum); // normalize
      });

      if (calcMethod === 'time_series') {
        const N = hubSp.length;
        const scaleFactor = 8760 / N;
        addLog(`Time Series running hour-by-hour over ${N} steps…`, 'i');

        // Precompute sector-wise unit vectors
        const uxs = [], uys = [], vxs = [], vys = [];
        for (let s = 0; s < nSec; s++) {
          const dir = (s + 0.5) * secW;
          const th = (dir + 180) * Math.PI / 180;
          uxs.push(Math.sin(th)); uys.push(Math.cos(th));
          vxs.push(Math.sin(th + Math.PI / 2)); vys.push(Math.cos(th + Math.PI / 2));
        }

        for (let t = 0; t < N; t++) {
          const U = hubSp[t];
          const Dir = hubDir[t];
          const si = Math.floor((((Dir % 360) + 360) % 360) / secW) % nSec;

          const mastSU = (window.BZ.mastSectorSU && window.BZ.mastSectorSU[si]) || window.BZ.mastSU || 1;
          const rc = (window.BZ.sectorRoughRC && window.BZ.sectorRoughRC[si]) || 1;

          // Free wind speed at each turbine
          const free = S.turbines.map((_, ti) => {
            if (isMultiMast) {
              let sum_u = 0;
              const weights = turbineWeights[ti];
              for (let mIdx = 0; mIdx < activeMasts.length; mIdx++) {
                const m = activeMasts[mIdx];
                const mSU = window.BZ.sectorSpeedups[si][S.turbines.length + mIdx] || 1;
                const tSU = window.BZ.sectorSpeedups[si][ti] || 1;
                const rel = (tSU / Math.max(1e-6, mSU)) * rc;
                const mSp = m.speeds[t] ?? U;
                const mSp_extrap = verticalExtrapolate(mSp, m.height, pc.hh, Dir);
                sum_u += weights[mIdx] * mSp_extrap * rel;
              }
              return Math.max(0.1, sum_u);
            } else {
              const tSU = window.BZ.sectorSpeedups[si][ti] || 1;
              const rel = (tSU / Math.max(1e-6, mastSU)) * rc;
              return Math.max(0.1, U * rel);
            }
          });

          // Sort turbines from upwind to downwind for Quarton-Ainslie Added Turbulence
          const ux = uxs[si], uy = uys[si], vx = vxs[si], vy = vys[si];
          const sortedIdx = Array.from({length: S.turbines.length}, (_, i) => i)
            .sort((a,b) => xy[a].x * ux + xy[a].y * uy - (xy[b].x * ux + xy[b].y * uy));

          // Quarton-Ainslie local turbulence intensity tracking
          const TI_local = Array(S.turbines.length).fill(TI_val);

          // Wake deficits (energy combination)
          const deficits2 = free.map(() => 0);
          for (let sIdx = 0; sIdx < sortedIdx.length; sIdx++) {
            const i = sortedIdx[sIdx];
            for (let dIdx = sIdx + 1; dIdx < sortedIdx.length; dIdx++) {
              const j = sortedIdx[dIdx];
              const dx = (xy[j].x - xy[i].x) * ux + (xy[j].y - xy[i].y) * uy;
              const dy = (xy[j].x - xy[i].x) * vx + (xy[j].y - xy[i].y) * vy;
              if (dx < 0.5 * pc.D) continue;
              
              const ct = ctAt(pc, free[i]);
              
              // Calculate wake deficit using local waked turbulence intensity (TI_local[i])
              const def = useEV ? ainslieDeficit(dx, dy, pc.D, ct, TI_local[i]) : bastankhahDeficit(dx, dy, pc.D, ct, kWake);
              deficits2[j] += def * def;

              // Quarton-Ainslie Added Turbulence accumulation
              if (useEV && dx >= 2.0 * pc.D && Math.abs(dy) < 1.5 * pc.D) {
                const I_add = 5.7 * Math.pow(ct, 0.7) * Math.pow(TI_val, 0.68) * Math.pow(dx / pc.D, -0.96);
                const rPrime = dy / pc.D;
                const Bw = Math.sqrt((0.445 * ct) / Math.max(1e-4, def * (1 - 0.5 * def)));
                const addedTurb2 = Math.pow(I_add * Math.exp(-3.56 * rPrime * rPrime / Math.max(0.1, Bw * Bw)), 2);
                TI_local[j] = Math.sqrt(TI_local[j] * TI_local[j] + addedTurb2);
              }
            }
          }
          const netU = free.map((u, ti) => u * (1 - Math.min(0.7, Math.sqrt(deficits2[ti]))));

          for (let ti = 0; ti < S.turbines.length; ti++) {
            let pg = powerAt(pc, free[ti]);
            let pn = powerAt(pc, netU[ti]);

            // Elevation-based Air Density Correction
            const z = S.turbines[ti].elev || 0;
            const rho_ratio = Math.exp(-z / 8400);
            pg = pg * rho_ratio;
            pn = pn * rho_ratio;

            // Apply non-downtime partial load losses
            if (free[ti] < ur) pg = pg * L_non_down;
            if (netU[ti] < ur) pn = pn * L_non_down;

            // Scale power output with downtime losses
            pg = pg * L_downtime;
            pn = pn * L_downtime;

            per[ti].gross += pg * scaleFactor / 1000; // accumulate in MWh/y
            per[ti].net += pn * scaleFactor / 1000;
            per[ti].fwsSum += free[ti];
            per[ti].fwsN += 1;
            per[ti].wakeSum += (free[ti] > 0.1 ? (1 - netU[ti] / free[ti]) : 0);
            per[ti].wSum += 1;
          }
        }
      } else {
        // Statistical Weibull sectors calculation with Multi-Mast support
        addLog(`Statistical calculation running over sectors & Weibull bins…`, 'i');
        for (let si = 0; si < nSec; si++) {
          const f_dir = sectors[si].freq;
          if (f_dir < 1e-6) continue;
          const dir = sectors[si].dir; // FROM
          const th = (dir + 180) * Math.PI / 180; // TO
          const ux = Math.sin(th), uy = Math.cos(th);
          const vx = Math.sin(th + Math.PI / 2), vy = Math.cos(th + Math.PI / 2);

          const mastSU = (window.BZ.mastSectorSU && window.BZ.mastSectorSU[si]) || window.BZ.mastSU || 1;
          const rc = (window.BZ.sectorRoughRC && window.BZ.sectorRoughRC[si]) || 1;

          // Process Weibull bins on interpolated sector parameters for each turbine!
          for (let ti = 0; ti < S.turbines.length; ti++) {
            let A = sectors[si].A, k_val = Math.max(sectors[si].k, 1.1), f = f_dir;
            
            if (isMultiMast) {
              let sum_A = 0, sum_k = 0, sum_f = 0;
              const weights = turbineWeights[ti];
              for (let mIdx = 0; mIdx < activeMasts.length; mIdx++) {
                const m = activeMasts[mIdx];
                const mSU = window.BZ.sectorSpeedups[si][S.turbines.length + mIdx] || 1;
                const tSU = window.BZ.sectorSpeedups[si][ti] || 1;
                const rel = (tSU / Math.max(1e-6, mSU)) * rc;
                const mSector = m.sectors[si] || { A: 6, k: 2, freq: 1/nSec };
                sum_A += weights[mIdx] * mSector.A * rel;
                sum_k += weights[mIdx] * mSector.k;
                sum_f += weights[mIdx] * mSector.freq;
              }
              A = sum_A;
              k_val = Math.max(sum_k, 1.1);
              f = sum_f;
            } else {
              const tSU = window.BZ.sectorSpeedups[si][ti] || 1;
              const rel = (tSU / Math.max(1e-6, mastSU)) * rc;
              A = A * rel;
            }

            for (let b = 1; b <= 30; b++) {
              const u0 = b - 0.5; // bin center free-stream
              const cdf = (u) => 1 - Math.exp(-Math.pow(u / A, k_val));
              const p = Math.max(0, cdf(b) - cdf(b - 1));
              if (p < 1e-9) continue;

              const u_free = u0; // local free speed for this turbine bin

              // Wake deficits (energy combination for this bin)
              let deficits2_j = 0;
              for (let i = 0; i < xy.length; i++) {
                if (i === ti) continue;
                const dx = (xy[ti].x - xy[i].x) * ux + (xy[ti].y - xy[i].y) * uy;
                const dy = (xy[ti].x - xy[i].x) * vx + (xy[ti].y - xy[i].y) * vy;
                if (dx < 0.5 * pc.D) continue;
                const ct = ctAt(pc, u_free);
                const def = useEV ? ainslieDeficit(dx, dy, pc.D, ct, TI_val) : bastankhahDeficit(dx, dy, pc.D, ct, kWake);
                deficits2_j += def * def;
              }
              const u_net = u_free * (1 - Math.min(0.7, Math.sqrt(deficits2_j)));

              let pg = powerAt(pc, u_free);
              let pn = powerAt(pc, u_net);

              // Elevation-based Air Density Correction
              const z = S.turbines[ti].elev || 0;
              const rho_ratio = Math.exp(-z / 8400);
              pg = pg * rho_ratio;
              pn = pn * rho_ratio;

              // Apply non-downtime partial load losses
              if (u_free < ur) pg = pg * L_non_down;
              if (u_net < ur) pn = pn * L_non_down;

              // Apply downtime losses
              pg = pg * L_downtime;
              pn = pn * L_downtime;

              per[ti].gross += f * p * pg * hours / 1000; // GWh/y
              per[ti].net += f * p * pn * hours / 1000;
              per[ti].fwsSum += f * p * u_free;
              per[ti].fwsN += f * p;
              per[ti].wakeSum += f * p * (u_free > 0.1 ? (1 - u_net / u_free) : 0);
              per[ti].wSum += f * p;
            }
          }
        }
      }

      // MWh → GWh
      let gross = 0, net = 0;
      const perTurbine = per.map((p, i) => {
        const g = p.gross / 1000; // GWh
        const n = p.net / 1000; // GWh
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

      const lossOther = lossEnv;
      const avail = availRatio;
      const elec = 1 - lossElec / 100;

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
        wakeModel: useEV ? 'ev' : 'bastankhah',
        wakeTI: TI_val * 100,
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
    const R = S.results;
    if (!R) { alert('Run AEP first'); return; }
    const stamp = new Date().toISOString().slice(0, 10);
    const base = (S.project || 'windfarm').replace(/\s+/g, '_');
    exportReportMd(base, stamp);
    exportPerTurbineCsv(base, stamp);
    exportSectorsCsv(base, stamp);
    exportLayoutCsv(base, stamp);
    exportBoundaryCsv(base, stamp);
    exportWindCsv(base, stamp);
    exportTerrainCsv(base, stamp);
    exportRoughnessCsv(base, stamp);
    exportWaspTab(base, stamp);
    exportLayoutKml(base, stamp);
    exportSummaryJson(base, stamp);
    addLog('Exported full package (report, AEP, wind, terrain, roughness, maps data, KML, WAsP .tab)', 'o');
  }

  function exportReportMd(base, stamp) {
    const R = S.results; if (!R) return;
    const vMethod = $('vertMethod')?.value || 'log';
    let md = `# ${S.project} — AEP Report\n\n`;
    md += `Generated: ${new Date().toISOString()}\n\n`;
    md += `## Configuration\n`;
    md += `- WTG: ${R.pc.name} · ${R.pc.rated} kW · D=${R.pc.D} m · HH=${R.pc.hh} m\n`;
    md += `- Count: ${R.n} · Capacity: ${R.capacityMW.toFixed(2)} MW\n`;
    md += `- Wind: ${R.windSource} @ ${S.wind?.height || '?'} m → hub via **${vMethod}**\n`;
    md += `- Horizontal: spectral BZ orography + roughness rose/IBL · Engine: ${R.engine} · RIX: ${(+R.rix || 0).toFixed(2)}%\n`;
    md += `- Losses: other ${R.losses.other}% · avail ${R.losses.avail}% · elec ${R.losses.elec}%\n`;
    if (S.terrain) md += `- Terrain: ${S.terrain.ny}×${S.terrain.nx}, ${S.terrain.minE.toFixed(0)}–${S.terrain.maxE.toFixed(0)} m (${S.terrain.source || ''})\n`;
    md += `- Roughness zones: ${(S.roughnessZones || []).length}\n\n`;
    md += `## AEP\n| Metric | Value |\n|--------|-------|\n`;
    md += `| Gross AEP | ${R.grossAEP.toFixed(3)} GWh/y |\n| Net AEP | ${R.netAEP.toFixed(3)} GWh/y |\n`;
    md += `| Wake loss | ${R.wakeLoss.toFixed(2)} % |\n| CF | ${R.CF.toFixed(2)} % |\n| Mean hub WS | ${R.hubMean.toFixed(3)} m/s |\n\n`;
    md += `## Disclaimer\nIndicative Wind Farm AEP Studio result. Not a substitute for met-mast + certified WAsP/OpenWind assessment.\n`;
    downloadText(`${base}_AEP_Report_${stamp}.md`, md);
  }

  function exportPerTurbineCsv(base, stamp) {
    const R = S.results; if (!R) return;
    let csv = 'ID,Name,Lat,Lon,Easting,Northing,UTM_Zone,Elev_m,HH_m,FreeWS_mps,SU,WakePct,Gross_GWh,Net_GWh,CF_pct\n';
    R.perTurbine.forEach((t, i) => {
      const turb = S.turbines[i] || {};
      const utm = (turb.easting != null) ? { easting: turb.easting, northing: turb.northing, zone: turb.zone }
        : latLonToUtm(t.lat, t.lon);
      csv += [
        t.id, turb.name || `T${t.id}`, t.lat, t.lon,
        (utm.easting ?? '').toFixed ? utm.easting.toFixed(1) : '',
        (utm.northing ?? '').toFixed ? utm.northing.toFixed(1) : '',
        utm.zone ?? '',
        t.elev ?? '', turb.hh ?? R.pc.hh,
        t.freeWS.toFixed(3), (+t.SU).toFixed(4), t.wakePct.toFixed(2),
        t.grossGWh.toFixed(4), t.netGWh.toFixed(4), t.CF.toFixed(2),
      ].join(',') + '\n';
    });
    downloadText(`${base}_AEP_per_turbine_${stamp}.csv`, csv, 'text/csv');
  }

  function exportSectorsCsv(base, stamp) {
    const R = S.results; if (!R?.sectors) return;
    let sc = 'Dir_deg,Freq,A,k,WS_mps\n';
    R.sectors.forEach((s) => {
      sc += [s.dir.toFixed(1), s.freq.toFixed(5), s.A.toFixed(3), s.k.toFixed(3), s.WS.toFixed(3)].join(',') + '\n';
    });
    downloadText(`${base}_sectors_${stamp}.csv`, sc, 'text/csv');
  }

  function exportLayoutCsv(base, stamp) {
    let lc = 'Name,Lat,Lon,Easting,Northing,UTM_Zone,HH_m,Elev_m\n';
    S.turbines.forEach((t, i) => {
      const utm = t.easting != null ? { easting: t.easting, northing: t.northing, zone: t.zone } : latLonToUtm(t.lat, t.lon);
      lc += [
        t.name || `T${i + 1}`, t.lat, t.lon,
        utm.easting.toFixed(2), utm.northing.toFixed(2), utm.zone,
        t.hh || '', t.elev ?? '',
      ].join(',') + '\n';
    });
    downloadText(`${base}_layout_${stamp}.csv`, lc, 'text/csv');
  }

  function exportBoundaryCsv(base, stamp) {
    if (!S.boundary.length) return;
    let bc = 'lon,lat\n';
    S.boundary.forEach((p) => { bc += `${p.lon},${p.lat}\n`; });
    downloadText(`${base}_boundary_${stamp}.csv`, bc, 'text/csv');
  }

  function exportWindCsv(base, stamp) {
    if (!S.wind?.speeds?.length) { addLog('No wind series to export', 'w'); return; }
    const w = S.wind;
    let csv = 'index,timestamp,ws_data_mps,wd_deg,ws_hub_mps,height_data_m,height_hub_m,source\n';
    const hh = currentPC().hh;
    const dataH = w.height || +$('dataH').value || 100;
    const n = w.speeds.length;
    // export full series (may be large)
    for (let i = 0; i < n; i++) {
      const hub = verticalExtrapolate(w.speeds[i], dataH, hh, w.dirs[i]);
      const ts = (w.times && w.times[i]) || '';
      csv += [i, ts, w.speeds[i], w.dirs[i], hub.toFixed(4), dataH, hh, w.source].join(',') + '\n';
    }
    downloadText(`${base}_wind_series_${stamp}.csv`, csv, 'text/csv');
    // meta
    const meta = {
      source: w.source, height_m: dataH, hub_height_m: hh,
      meanWS_data: w.meanWS, meanWS_hub: mean(w.speeds.map((v, i) => verticalExtrapolate(v, dataH, hh, w.dirs[i]))),
      n, lat: w.lat, lon: w.lon, weibullA: w.weibullA, weibullK: w.weibullK,
      gwaMeta: S.gwaMeta || null, verticalMethod: $('vertMethod')?.value || 'log',
    };
    downloadText(`${base}_wind_meta_${stamp}.json`, JSON.stringify(meta, null, 2), 'application/json');
    addLog(`Exported wind series (${n} rows) + meta`, 'o');
  }

  function exportTerrainCsv(base, stamp) {
    const T = S.terrain;
    if (!T?.grid) { addLog('No terrain grid to export', 'w'); return; }
    let csv = 'row,col,lat,lon,elev_m\n';
    for (let i = 0; i < T.ny; i++) {
      const lat = T.lat0 + i * (T.lat1 - T.lat0) / Math.max(1, T.ny - 1);
      for (let j = 0; j < T.nx; j++) {
        const lon = T.lon0 + j * (T.lon1 - T.lon0) / Math.max(1, T.nx - 1);
        csv += `${i},${j},${lat.toFixed(6)},${lon.toFixed(6)},${T.grid[i][j]}\n`;
      }
    }
    downloadText(`${base}_terrain_grid_${stamp}.csv`, csv, 'text/csv');
    downloadText(`${base}_terrain_meta_${stamp}.json`, JSON.stringify({
      ny: T.ny, nx: T.nx, lat0: T.lat0, lat1: T.lat1, lon0: T.lon0, lon1: T.lon1,
      minE: T.minE, maxE: T.maxE, meanE: T.meanE, source: T.source,
    }, null, 2), 'application/json');
    addLog('Exported terrain grid + meta', 'o');
  }

  function exportRoughnessCsv(base, stamp) {
    let zc = 'lu,z0_m,n_pts,centroid_lat,centroid_lon\n';
    (S.roughnessZones || []).forEach((z) => {
      const la = z.pts ? mean(z.pts.map((p) => p.lat)) : '';
      const lo = z.pts ? mean(z.pts.map((p) => p.lon)) : '';
      zc += `${(z.lu || '').replace(/,/g, ';')},${z.z0},${z.pts ? z.pts.length : 0},${la},${lo}\n`;
    });
    downloadText(`${base}_roughness_zones_${stamp}.csv`, zc, 'text/csv');
    const rose = S.autoRoughnessRose || S.roughnessRose || [];
    let rc = 'sector,dir_deg,z0_1,z0_2,z0_3,x1_m,x2_m,x3_m\n';
    rose.forEach((r, i) => {
      const z0 = r.z0 || []; const x = r.x || [];
      const dir = r.dir != null ? r.dir : (i + 0.5) * 360 / rose.length;
      rc += `${i},${dir},${z0[0] ?? ''},${z0[1] ?? ''},${z0[2] ?? ''},${x[0] ?? ''},${x[1] ?? ''},${x[2] ?? ''}\n`;
    });
    downloadText(`${base}_roughness_rose_${stamp}.csv`, rc, 'text/csv');
    addLog(`Exported roughness zones (${(S.roughnessZones || []).length}) + rose (${rose.length})`, 'o');
  }

  function exportWaspTab(base, stamp) {
    // Simple WAsP .tab from sector Weibull (hub height)
    const R = S.results;
    const secs = R?.sectors || [];
    if (!secs.length) { addLog('No sectors for .tab', 'w'); return; }
    const hh = currentPC().hh;
    const c = centerOf(S.turbines.length ? S.turbines : S.boundary);
    let tab = `${S.project} - hub ${hh}m\n`;
    tab += ` ${c.lat.toFixed(3)} ${c.lon.toFixed(3)} height ${hh}m\n`;
    // frequencies in %
    const freqs = secs.map((s) => (s.freq * 100));
    tab += ` ${secs.length} ${hh.toFixed(1)}\n`;
    tab += ' ' + freqs.map((f) => f.toFixed(3)).join(' ') + '\n';
    tab += ' ' + secs.map((s) => s.A.toFixed(3)).join(' ') + '\n';
    tab += ' ' + secs.map((s) => s.k.toFixed(3)).join(' ') + '\n';
    downloadText(`${base}_hub_${stamp}.tab`, tab, 'text/plain');
    addLog('Exported WAsP-style .tab (sector Weibull at hub)', 'o');
  }

  function exportLayoutKml(base, stamp) {
    let kml = `<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>${S.project}</name>\n`;
    if (S.boundary.length >= 3) {
      kml += `<Placemark><name>Boundary</name><Style><LineStyle><color>ffbd8f3d</color><width>2</width></LineStyle><PolyStyle><color>2200ff00</color></PolyStyle></Style><Polygon><outerBoundaryIs><LinearRing><coordinates>`;
      S.boundary.forEach((p) => { kml += `${p.lon},${p.lat},0 `; });
      kml += `</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>\n`;
    }
    S.turbines.forEach((t, i) => {
      const ws = S.speedField && S.speedField[i] != null ? ` WS=${Number(S.speedField[i]).toFixed(2)}` : '';
      kml += `<Placemark><name>${t.name || ('T' + (i + 1))}</name><description>HH=${t.hh || ''}${ws}</description><Point><coordinates>${t.lon},${t.lat},0</coordinates></Point></Placemark>\n`;
    });
    if (S.windPoint || (S.wind && S.wind.lat != null)) {
      const wp = S.windPoint || S.wind;
      kml += `<Placemark><name>WindData_${wp.source || ''}</name><Point><coordinates>${wp.lon},${wp.lat},0</coordinates></Point></Placemark>\n`;
    }
    kml += `</Document></kml>`;
    downloadText(`${base}_layout_${stamp}.kml`, kml, 'application/vnd.google-earth.kml+xml');
    addLog('Exported layout/boundary KML', 'o');
  }

  function exportSummaryJson(base, stamp) {
    const R = S.results;
    downloadText(`${base}_summary_${stamp}.json`, JSON.stringify({
      project: S.project,
      results: R ? { ...R, perTurbine: undefined } : null,
      nTurbines: R?.n, wind: S.wind && {
        source: S.wind.source, height: S.wind.height, meanWS: S.wind.meanWS,
        lat: S.wind.lat, lon: S.wind.lon, n: S.wind.speeds?.length,
      },
      terrain: S.terrain && { minE: S.terrain.minE, maxE: S.terrain.maxE, meanE: S.terrain.meanE, ny: S.terrain.ny, nx: S.terrain.nx, source: S.terrain.source },
      roughnessZones: (S.roughnessZones || []).length,
      verticalMethod: $('vertMethod')?.value,
      gwaMeta: S.gwaMeta || null,
    }, null, 2), 'application/json');
  }

  async function exportMapPng() {
    if (!S.map) { alert('Map not ready'); return; }
    try {
      addLog('Capturing map PNG…', 'i');
      // Use leaflet's map pane + html2canvas-free approach: draw to canvas via dom-to-image alternative
      // Simple approach: leaflet-image style — composite tile + overlay canvases
      const map = S.map;
      const size = map.getSize();
      const canvas = document.createElement('canvas');
      canvas.width = size.x * 2; canvas.height = size.y * 2;
      const ctx = canvas.getContext('2d');
      ctx.scale(2, 2);
      ctx.fillStyle = '#0a101c';
      ctx.fillRect(0, 0, size.x, size.y);

      // try to draw tile images
      const tiles = map.getPane('tilePane')?.querySelectorAll('img') || [];
      const mapBounds = map.getBounds();
      const nw = map.latLngToContainerPoint(mapBounds.getNorthWest());
      // Draw white background for export clarity
      ctx.fillStyle = '#e8eef2';
      ctx.fillRect(0, 0, size.x, size.y);

      const drawImgs = [];
      tiles.forEach((img) => {
        if (!img.complete || !img.naturalWidth) return;
        const style = img.parentElement?.style || img.style;
        // leaflet positions tiles via transform on parent
        const rect = img.getBoundingClientRect();
        const mapRect = map.getContainer().getBoundingClientRect();
        const x = rect.left - mapRect.left;
        const y = rect.top - mapRect.top;
        drawImgs.push({ img, x, y, w: rect.width, h: rect.height });
      });
      for (const d of drawImgs) {
        try { ctx.drawImage(d.img, d.x, d.y, d.w, d.h); } catch (_) {}
      }

      // Draw overlays from SVG paths roughly via overlay pane is hard; instead re-render simple layers
      // Boundary
      if (S.boundary.length >= 2) {
        ctx.beginPath();
        S.boundary.forEach((p, i) => {
          const pt = map.latLngToContainerPoint([p.lat, p.lon]);
          if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(61,139,253,0.12)';
        ctx.strokeStyle = '#3d8bfd';
        ctx.lineWidth = 2;
        ctx.fill(); ctx.stroke();
      }
      // Turbines + speed
      const vals = (S.speedField || []).filter(isFinite);
      const wmin = vals.length ? Math.min(...vals) : 0;
      const wmax = vals.length ? Math.max(...vals) : 1;
      S.turbines.forEach((t, i) => {
        const pt = map.latLngToContainerPoint([t.lat, t.lon]);
        const ws = S.speedField && S.speedField[i];
        ctx.beginPath();
        if (ws != null && isFinite(ws)) {
          const r = 5 + 8 * ((ws - wmin) / Math.max(0.2, wmax - wmin));
          ctx.fillStyle = wsColor(ws, wmin, wmax);
          ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
        } else {
          ctx.fillStyle = '#1b7a4a';
          ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      // Wind point
      const wp = S.windPoint || (S.wind && S.wind.lat != null ? S.wind : null);
      if (wp && wp.lat != null) {
        const pt = map.latLngToContainerPoint([wp.lat, wp.lon]);
        ctx.beginPath();
        ctx.fillStyle = '#f5b942';
        ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = '#0b1220';
        ctx.font = '12px sans-serif';
        ctx.fillText((wp.source || 'WIND') + (wp.meanWS != null ? ` ${Number(wp.meanWS).toFixed(2)} m/s` : ''), pt.x + 10, pt.y - 8);
      }
      // Title
      ctx.fillStyle = 'rgba(11,18,32,0.85)';
      ctx.fillRect(8, 8, 320, 40);
      ctx.fillStyle = '#e8eefc';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(S.project || 'Wind Farm', 16, 28);
      ctx.font = '11px sans-serif';
      ctx.fillStyle = '#9aabcc';
      ctx.fillText(new Date().toISOString().slice(0, 10) + ' · AEP Studio map export', 16, 42);

      canvas.toBlob((blob) => {
        if (!blob) { addLog('Map PNG export failed', 'e'); return; }
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `${(S.project || 'windfarm').replace(/\s+/g, '_')}_map_${new Date().toISOString().slice(0, 10)}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
        addLog('Exported map PNG', 'o');
      }, 'image/png');
    } catch (e) {
      addLog('Map PNG export error: ' + e.message, 'e');
      alert('Map export failed: ' + e.message);
    }
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

  // ─── Turbine Editing ─────────────────────────────────────────────────────
  function deleteTurbine(index) {
    if (index < 0 || index >= S.turbines.length) return;
    const t = S.turbines[index];
    S.turbines.splice(index, 1);
    // Re-index remaining turbines if they have default names
    S.turbines.forEach((wtg, idx) => {
      if (wtg.name.startsWith('T') && isFinite(wtg.name.slice(1))) {
        wtg.name = `T${idx + 1}`;
      }
    });
    addLog(`Deleted WTG: ${t.name || `T${index + 1}`}`, 'w');
    refreshSiteUI();
    redrawMap({ fit: false });
  }

  // ─── Save and Open Project ───────────────────────────────────────────────
  function saveProject() {
    try {
      const data = {
        version: "1.0",
        inputs: {
          projName: $('projName')?.value,
          region: $('region')?.value,
          z0: $('z0')?.value,
          inCrs: $('inCrs')?.value,
          utmZone: $('utmZone')?.value,
          utmHem: $('utmHem')?.value,
          colOrder: $('colOrder')?.value,
          swLon: $('swLon')?.value,
          swLat: $('swLat')?.value,
          neLon: $('neLon')?.value,
          neLat: $('neLat')?.value,
          cLon: $('cLon')?.value,
          cLat: $('cLat')?.value,
          cRad: $('cRad')?.value,
          nWtg: $('nWtg')?.value,
          spD: $('spD')?.value,
          spC: $('spC')?.value,
          preset: $('preset')?.value,
          hh: $('hh')?.value,
          D: $('D')?.value,
          rated: $('rated')?.value,
          calcMethod: $('calcMethod')?.value,
          wakeModel: $('wakeModel')?.value,
          wakeTI: $('wakeTI')?.value,
          wakeK: $('wakeK')?.value,
          lossElec: $('lossElec')?.value,
          lossEnv: $('lossEnv')?.value,
          markovFail: $('markovFail')?.value,
          markovMTTR: $('markovMTTR')?.value,
          markovMTTD: $('markovMTTD')?.value,
          lossDegradation: $('lossDegradation')?.value,
          lossSuboptimal: $('lossSuboptimal')?.value,
          terrR: $('terrR')?.value,
          terrG: $('terrG')?.value,
          windSrc: $('windSrc')?.value,
          y0: $('y0')?.value,
          y1: $('y1')?.value,
          dataH: $('dataH')?.value,
          nSec: $('nSec')?.value,
          vertMethod: $('vertMethod')?.value || 'log',
          shearAlpha: $('shearAlpha')?.value || '0.14',
          lyrBoundary: $('lyrBoundary')?.checked,
          lyrTurbines: $('lyrTurbines')?.checked,
          lyrElev: $('lyrElev')?.checked,
          lyrRough: $('lyrRough')?.checked,
          lyrSpeed: $('lyrSpeed')?.checked,
          lyrWindPt: $('lyrWindPt')?.checked,
          lyrLabels: $('lyrLabels')?.checked
        },
        state: {
          project: S.project,
          boundary: S.boundary,
          turbines: S.turbines,
          pc: S.pc,
          terrain: S.terrain,
          roughnessZones: S.roughnessZones,
          roughnessRose: S.roughnessRose,
          wind: S.wind,
          gwaMeta: S.gwaMeta,
          results: S.results,
          speedField: S.speedField,
          windPoint: S.windPoint,
          masts: S.masts
        }
      };
      const name = (data.inputs.projName || 'project').replace(/[^a-z0-9_-]/gi, '_');
      const jsonText = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonText], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${name}_aep_project.json`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
      addLog(`Project saved: ${name}_aep_project.json`, 'o');
    } catch (err) {
      alert(`Failed to save project: ${err.message}`);
      console.error(err);
    }
  }

  function openProject(file) {
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (!data || !data.inputs || !data.state) {
          throw new Error('Invalid project file format.');
        }

        // Restore inputs
        const inputs = data.inputs;
        for (const [id, val] of Object.entries(inputs)) {
          const el = document.getElementById(id);
          if (el) {
            if (el.type === 'checkbox') {
              el.checked = !!val;
            } else {
              el.value = val;
            }
          }
        }

        // Restore state
        S.project = data.state.project || 'My Wind Farm';
        S.boundary = data.state.boundary || [];
        S.turbines = data.state.turbines || [];
        S.pc = data.state.pc || null;
        S.terrain = data.state.terrain || null;
        S.roughnessZones = data.state.roughnessZones || [];
        S.roughnessRose = data.state.roughnessRose || null;
        S.wind = data.state.wind || null;
        S.masts = data.state.masts || [];
        S.gwaMeta = data.state.gwaMeta || null;
        S.results = data.state.results || null;
        S.speedField = data.state.speedField || null;
        S.windPoint = data.state.windPoint || null;

        // Restore step styling
        setStep('site', S.boundary.length > 0 || S.turbines.length > 0 ? 'done' : null);
        setStep('wtg', S.pc || currentPC() ? 'done' : null);
        setStep('maps', S.terrain || S.roughnessRose ? 'done' : null);
        setStep('wind', S.wind ? 'done' : null);
        setStep('aep', S.results ? 'done' : null);

        refreshSiteUI();
        redrawMap({ fit: true });
        restoreResultsUI();
        updateWindStatusUI();
        refreshMastsUI();

        addLog(`Project loaded successfully: ${S.project}`, 'o');
      } catch (err) {
        alert(`Failed to open project: ${err.message}`);
        console.error(err);
      }
    };
    reader.readAsText(file);
  }

  function restoreResultsUI() {
    const R = S.results;
    if (!R) {
      $('kGross').textContent = '-';
      $('kNet').textContent = '-';
      $('kCF').textContent = '-';
      $('kWake').textContent = '-';
      $('kWS').textContent = '-';
      $('kCap').textContent = '-';
      $('tbl').innerHTML = '<tr><td colspan="6" class="muted text-center">Run AEP to see results</td></tr>';
      $('sectbl').innerHTML = '<tr><td colspan="5" class="muted text-center">Run AEP to see sectors</td></tr>';
      $('btnExport').disabled = true;
      chipBox('resChips', []);
      return;
    }
    
    $('kGross').textContent = R.grossAEP.toFixed(2) + ' GWh/y';
    $('kNet').textContent = R.netAEP.toFixed(2) + ' GWh/y';
    $('kCF').textContent = R.CF.toFixed(1) + ' %';
    $('kWake').textContent = R.wakeLoss.toFixed(1) + ' %';
    $('kWS').textContent = R.hubMean.toFixed(2) + ' m/s';
    $('kCap').textContent = R.capacityMW.toFixed(1) + ' MW';
    
    chipBox('resChips', [
      [`Wind: ${R.windSource || 'Loaded'}`, true],
      [`Engine: ${R.engine || 'n/a'}`, true],
      [`RIX ${(+R.rix || 0).toFixed(1)}%`, true],
      [`${R.n || S.turbines.length} WTGs`, true],
    ]);
    
    $('tbl').innerHTML = R.perTurbine.map((t) =>
      `<tr><td>${t.id}</td><td>${t.freeWS.toFixed(2)}</td><td>${(+t.SU || 1.0).toFixed(3)}</td><td>${t.wakePct.toFixed(1)}</td><td>${t.netGWh.toFixed(2)}</td><td>${t.CF.toFixed(1)}</td></tr>`
    ).join('');
    
    $('sectbl').innerHTML = R.sectors.map((s) =>
      `<tr><td>${s.dir.toFixed(0)}°</td><td>${(100 * s.freq).toFixed(1)}</td><td>${s.A.toFixed(2)}</td><td>${s.k.toFixed(2)}</td><td>${s.WS.toFixed(2)}</td></tr>`
    ).join('');
    
    $('btnExport').disabled = false;
  }

  // ─── WRG Map Generation ──────────────────────────────────────────────────
  function solveWeibullK(E_pf) {
    if (E_pf <= 1.0) return 10.0;
    let low = 1.01, high = 10.0;
    for (let iter = 0; iter < 25; iter++) {
      const mid = (low + high) / 2;
      const ratio = gamma(1 + 3 / mid) / Math.pow(gamma(1 + 1 / mid), 3);
      if (ratio > E_pf) {
        low = mid;
      } else {
        high = mid;
      }
    }
    return (low + high) / 2;
  }

  async function generateWrgMap() {
    try {
      if (!S.terrain) {
        alert('Please download terrain first (TERRAIN tab) — elevation data is required.');
        return;
      }
      if (!S.wind || !S.results || !S.results.sectors) {
        alert('Please run AEP first to establish the site wind climate and speed-ups.');
        return;
      }

      const cell_size = parseFloat($('wrgCellSize').value) || 200;
      const height = parseFloat($('wrgHeight').value) || currentPC().hh || 100;

      const pts = S.boundary.concat(S.turbines);
      if (!pts.length) {
        alert('Please define a boundary or layout first.');
        return;
      }

      const center = centerOf(pts);
      const centerUtm = latLonToUtm(center.lat, center.lon);
      const zone = centerUtm.zone;
      const northern = center.lat >= 0;

      const utmPts = pts.map(p => latLonToUtm(p.lat, p.lon));
      const eastings = utmPts.map(p => p.easting);
      const northings = utmPts.map(p => p.northing);

      let Xmin = Math.min(...eastings);
      let Xmax = Math.max(...eastings);
      let Ymin = Math.min(...northings);
      let Ymax = Math.max(...northings);

      Xmin -= cell_size;
      Xmax += cell_size;
      Ymin -= cell_size;
      Ymax += cell_size;

      Xmin = Math.floor(Xmin / cell_size) * cell_size;
      Xmax = Math.ceil(Xmax / cell_size) * cell_size;
      Ymin = Math.floor(Ymin / cell_size) * cell_size;
      Ymax = Math.ceil(Ymax / cell_size) * cell_size;

      const Nx = Math.round((Xmax - Xmin) / cell_size) + 1;
      const Ny = Math.round((Ymax - Ymin) / cell_size) + 1;
      const totalPoints = Nx * Ny;

      if (totalPoints > 10000) {
        if (!confirm(`The selected cell size of ${cell_size}m results in a grid of ${Nx}x${Ny} = ${totalPoints} points. This might take a few seconds to compute. Do you want to proceed?`)) {
          return;
        }
      }

      addLog(`Generating WRG map: Grid ${Nx}x${Ny} (${totalPoints} points) at resolution ${cell_size}m, height ${height}m`, 'i');

      const gridPoints = [];
      for (let iy = 0; iy < Ny; iy++) {
        for (let ix = 0; ix < Nx; ix++) {
          const x = Xmin + ix * cell_size;
          const y = Ymin + iy * cell_size;
          const latlon = utmToLatLon(x, y, zone, northern);
          gridPoints.push({
            ix, iy, x, y,
            lat: latlon.lat,
            lon: latlon.lon,
            elev: elevAt(latlon.lat, latlon.lon) || 0
          });
        }
      }

      const nSec = S.results.sectors.length;
      let sectorSpeedups = [];
      let mastSectorSU = [];
      let sectorRoughRC = window.BZ.sectorRoughRC || Array(nSec).fill(1);

      if (S.terrain && S.terrain.grid && window.WFP61 && typeof window.WFP61.computeSpectralFlowField === 'function') {
        try {
          const options = {
            hubH: height,
            z0Site: +$('z0').value || 0.03,
            z0Mast: +$('z0').value || 0.03,
            nSect: nSec,
            sectorFreq: S.results.sectors.map(s => s.freq),
            log: () => {}
          };
          const mast = centerOf(S.boundary.length ? S.boundary : S.turbines);
          const flow = window.WFP61.computeSpectralFlowField(S.terrain, gridPoints, mast, options);
          sectorSpeedups = flow.sectorSpeedups;
          mastSectorSU = flow.mastSectorSU;
        } catch (e) {
          addLog(`Spectral BZ failed for WRG grid, falling back to elevation model: ${e.message}`, 'w');
        }
      }

      if (!sectorSpeedups.length) {
        const mast = centerOf(S.boundary.length ? S.boundary : S.turbines);
        const mastE = elevAt(mast.lat, mast.lon) || 0;
        sectorSpeedups = [];
        mastSectorSU = Array(nSec).fill(1);
        for (let s = 0; s < nSec; s++) {
          const row = gridPoints.map((pt) => {
            const dE = pt.elev - mastE;
            return clamp(1 + 0.4 * dE / Math.max(80, height), 0.75, 1.35);
          });
          sectorSpeedups.push(row);
        }
      }

      let wrgText = `  ${Nx}  ${Ny}  ${Xmin.toFixed(1)}  ${Ymin.toFixed(1)}  ${cell_size.toFixed(1)}\r\n`;

      const rho = 1.225;
      const sectors = S.results.sectors;

      for (let pi = 0; pi < gridPoints.length; pi++) {
        const pt = gridPoints[pi];
        let V_total = 0;
        let P_total = 0;
        const localSectors = [];

        for (let s = 0; s < nSec; s++) {
          const f = sectors[s].freq;
          const A_mast = sectors[s].A;
          const k_mast = sectors[s].k;

          const su = (sectorSpeedups[s] && sectorSpeedups[s][pi]) || 1;
          const mSU = mastSectorSU[s] || 1;
          const rc = sectorRoughRC[s] || 1;

          const rel = (su / Math.max(1e-6, mSU)) * rc;
          const A_local = A_mast * rel;
          const k_local = k_mast;

          const V_sec = A_local * gamma(1 + 1 / k_local);
          const P_sec = 0.5 * rho * Math.pow(A_local, 3) * gamma(1 + 3 / k_local);

          V_total += f * V_sec;
          P_total += f * P_sec;

          localSectors.push({ freq: f, A: A_local, k: k_local });
        }

        let k_total = 2.0;
        if (V_total > 0.1) {
          const E_pf = P_total / (0.5 * rho * Math.pow(V_total, 3));
          k_total = solveWeibullK(E_pf);
        }
        let A_total = V_total / gamma(1 + 1 / k_total);

        if (!isFinite(A_total) || isNaN(A_total)) A_total = 0;
        if (!isFinite(k_total) || isNaN(k_total)) k_total = 1.0;
        if (!isFinite(P_total) || isNaN(P_total)) P_total = 0;

        let line = "GridPoint ";
        line += pt.x.toFixed(0).padStart(10);
        line += pt.y.toFixed(0).padStart(10);
        line += pt.elev.toFixed(0).padStart(8);
        line += height.toFixed(0).padStart(5);
        line += A_total.toFixed(1).padStart(5);
        line += k_total.toFixed(2).padStart(6);
        line += P_total.toFixed(0).padStart(15);
        line += nSec.toString().padStart(3);

        for (let s = 0; s < nSec; s++) {
          const sec = localSectors[s];
          const freq10 = Math.round(sec.freq * 1000);
          const A10 = Math.round(sec.A * 10);
          const k100 = Math.round(sec.k * 100);

          line += freq10.toString().padStart(4);
          line += A10.toString().padStart(4);
          line += k100.toString().padStart(5);
        }

        wrgText += line + "\r\n";
      }

      const name = (S.project || 'site').replace(/[^a-z0-9_-]/gi, '_');
      const filename = `${name}_res_${cell_size}m_${height}m.wrg`;
      triggerTextDownload(filename, wrgText);
      addLog(`WRG Map generated and exported: ${filename}`, 'o');
    } catch (err) {
      console.error(err);
      alert(`Failed to generate WRG map: ${err.message}`);
    }
  }

  // ─── Drag/drop helpers ───────────────────────────────────────────────────
  function bindDrop(zoneId, inputId, handler) {
    const z = $(zoneId), inp = $(inputId);
    z.addEventListener('click', () => inp.click());
    z.addEventListener('dragover', (e) => { e.preventDefault(); z.classList.add('drag'); });
    z.addEventListener('dragleave', () => z.classList.remove('drag'));
    z.addEventListener('drop', async (e) => {
      e.preventDefault(); z.classList.remove('drag');
      const fl = e.dataTransfer.files; if (fl && fl.length) handler(fl.length > 1 ? fl : fl[0]);
    });
    inp.addEventListener('change', () => {
      const fl = inp.files; if (fl && fl.length) handler(fl.length > 1 ? fl : fl[0]);
    });
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
    
    if ($('btnSaveProj')) $('btnSaveProj').onclick = saveProject;
    if ($('btnOpenProj')) $('btnOpenProj').onclick = () => $('fileOpenProj').click();
    if ($('fileOpenProj')) {
      $('fileOpenProj').onchange = (e) => {
        const file = e.target.files[0];
        if (file) openProject(file);
      };
    }
    if ($('btnExpWrg')) $('btnExpWrg').onclick = generateWrgMap;

    // Real-time switching of active wind source in state when dropdown selection changes
    if ($('windSrc')) {
      $('windSrc').addEventListener('change', () => {
        const src = $('windSrc').value;
        if (S.windSources[src]) {
          S.wind = S.windSources[src];
          addLog(`Switched active wind dataset to cached ${src}`, 'i');
        } else if (src === 'GWA' && S.windSources['GWA_FILE']) {
          S.wind = S.windSources['GWA_FILE'];
          addLog(`Switched active wind dataset to cached GWA_FILE`, 'i');
        } else {
          S.wind = null;
          addLog(`No cached wind data for ${src}. Will download or require upload on Run AEP.`, 'w');
        }
        updateWindStatusUI();
        refreshSiteUI();
        redrawMap({ fit: false });
      });
    }

    // Initial load of wind status UI on startup
    updateWindStatusUI();
    if ($('btnGwaLib') && $('fileGwaLib')) {
      $('btnGwaLib').onclick = () => $('fileGwaLib').click();
      $('fileGwaLib').addEventListener('change', async () => {
        const f = $('fileGwaLib').files && $('fileGwaLib').files[0];
        if (!f) return;
        try {
          const pts = S.boundary.length ? S.boundary : S.turbines;
          const c = pts.length ? centerOf(pts) : { lat: null, lon: null };
          await ingestGwaLibText(await f.text(), { lat: c.lat, lon: c.lon, source: 'GWA_FILE', fileName: f.name });
        } catch (e) {
          addLog('GWA .lib upload failed: ' + e.message, 'e');
          alert('Could not parse GWA .lib: ' + e.message);
        }
      });
    }
    $('btnRun').onclick = () => runAEP();
    $('btnExport').onclick = exportResults;
    $('btnDemo').onclick = loadDemo;
    if ($('btnExpMap')) $('btnExpMap').onclick = () => exportMapPng();
    if ($('btnExpTerr')) $('btnExpTerr').onclick = () => exportTerrainCsv((S.project||'wf').replace(/\s+/g,'_'), new Date().toISOString().slice(0,10));
    if ($('btnExpRough')) $('btnExpRough').onclick = () => exportRoughnessCsv((S.project||'wf').replace(/\s+/g,'_'), new Date().toISOString().slice(0,10));
    if ($('btnExpWind')) $('btnExpWind').onclick = () => exportWindCsv((S.project||'wf').replace(/\s+/g,'_'), new Date().toISOString().slice(0,10));
    if ($('btnExpTab')) $('btnExpTab').onclick = () => exportWaspTab((S.project||'wf').replace(/\s+/g,'_'), new Date().toISOString().slice(0,10));
    if ($('btnExpKml')) $('btnExpKml').onclick = () => exportLayoutKml((S.project||'wf').replace(/\s+/g,'_'), new Date().toISOString().slice(0,10));
    if ($('btnExpLayout')) $('btnExpLayout').onclick = () => exportLayoutCsv((S.project||'wf').replace(/\s+/g,'_'), new Date().toISOString().slice(0,10));
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
    refreshMastsUI();

    // Real-time Markov Turbine Availability Solver (IEC 61400-15-2)
    function updateMarkovUI() {
      const fail = +$('markovFail').value || 0;
      const mttr = +$('markovMTTR').value || 0;
      const mttd = +$('markovMTTD').value || 0;
      const lambda = fail;
      const mu = 8760 / Math.max(0.1, mttr);
      const gamma_val = 8760 / Math.max(0.1, mttd);
      const avail = 1 / (1 + lambda / mu + lambda / gamma_val);
      $('markovSolved').textContent = (avail * 100).toFixed(2) + '%';
    }

    if ($('markovFail')) $('markovFail').oninput = updateMarkovUI;
    if ($('markovMTTR')) $('markovMTTR').oninput = updateMarkovUI;
    if ($('markovMTTD')) $('markovMTTD').oninput = updateMarkovUI;
    updateMarkovUI();

    // Collapsible Adjustable Side Panes
    let leftCollapsed = false;
    let rightCollapsed = false;
    
    function updatePaneClasses() {
      const layout = document.querySelector('.layout');
      if (!layout) return;
      layout.classList.remove('collapse-left', 'collapse-right', 'collapse-both');
      if (leftCollapsed && rightCollapsed) {
        layout.classList.add('collapse-both');
      } else if (leftCollapsed) {
        layout.classList.add('collapse-left');
      } else if (rightCollapsed) {
        layout.classList.add('collapse-right');
      }
      if (S.map) S.map.invalidateSize();
    }

    if ($('btnToggleLeft')) {
      $('btnToggleLeft').onclick = () => {
        leftCollapsed = !leftCollapsed;
        $('btnToggleLeft').textContent = leftCollapsed ? '▶ Show Inputs' : '◀ Toggle Inputs';
        updatePaneClasses();
      };
    }
    
    if ($('btnToggleRight')) {
      $('btnToggleRight').onclick = () => {
        rightCollapsed = !rightCollapsed;
        $('btnToggleRight').textContent = rightCollapsed ? 'Show Results ◀' : 'Toggle Results ▶';
        updatePaneClasses();
      };
    }

    // Interactive Layout Editor - Add WTG on map click
    let addMode = false;
    if ($('btnAddWtg')) {
      $('btnAddWtg').onclick = () => {
        addMode = !addMode;
        if (addMode) {
          $('btnAddWtg').style.background = '#14301f';
          $('btnAddWtg').style.color = 'var(--ok)';
          $('btnAddWtg').style.borderColor = '#1f5a3a';
          $('btnAddWtg').textContent = '🟢 Click map to add WTGs (Stop)';
          S.map.getContainer().style.cursor = 'crosshair';
          addLog('Map click layout editing mode: active', 'i');
        } else {
          $('btnAddWtg').style.background = '';
          $('btnAddWtg').style.color = '';
          $('btnAddWtg').style.borderColor = '';
          $('btnAddWtg').textContent = '➕ Add WTG by map click';
          S.map.getContainer().style.cursor = '';
          addLog('Map click layout editing mode: stopped', 'i');
        }
      };
    }

    if (S.map) {
      S.map.on('click', (e) => {
        if (addMode) {
          const lat = e.latlng.lat;
          const lon = e.latlng.lng;
          const elev = elevAt(lat, lon) || 0;
          const utm = latLonToUtm(lat, lon);
          const nextId = S.turbines.length + 1;
          S.turbines.push({
            lat, lon, elev,
            hh: +$('hh').value || 140,
            name: `T${nextId}`,
            easting: utm.easting, northing: utm.northing, zone: utm.zone,
            _customHH: false
          });
          addLog(`Added T${nextId} at ${lat.toFixed(5)}, ${lon.toFixed(5)}`, 'o');
          refreshSiteUI();
          redrawMap({ fit: false });
        }
      });
    }

    // Public API for automation / external scripts
    window.AEPStudio = {
      S, downloadTerrain, downloadRoughness, downloadERA5, downloadGWA,
      runAEP, exportResults, applyPreset, generateGrid, currentPC,
      loadDemo, redrawMap, refreshSiteUI, updateLegend,
      drawElevationLayer, drawRoughnessLayer, drawSpeedLayer, drawWindPointLayer,
      ingestGwaLibText, showGwaHelpPanel,
      exportWindCsv, exportTerrainCsv, exportRoughnessCsv, exportMapPng,
      exportLayoutKml, exportWaspTab, readPointsFile, utmToLatLon, latLonToUtm,
      verticalExtrapolate, saveProject, openProject, generateWrgMap, deleteTurbine,
      toggleMast, deleteMast, refreshMastsUI
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
