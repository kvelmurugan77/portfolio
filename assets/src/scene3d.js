/* ==========================================================================
   VK Portfolio — 3D Wind Farm Scene
   Three.js digital-twin backdrop: scroll-driven camera flight over a
   procedurally generated wind farm with spinning WTGs, nacelle-mounted LiDAR
   beams, met mast, ground LiDAR, flow streamlines with simple wake deficit,
   tip vortices, terrain contours and simulated SCADA telemetry.
   Source file — bundled to assets/scene3d.min.js with esbuild.
   ========================================================================== */
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

(function boot() {
  // ---------- opt-out / capability checks ----------
  let pref = null;
  try { pref = localStorage.getItem('vk3d'); } catch (e) {}
  const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const disabled = pref === 'off' || (reducedMotion && pref !== 'on');
  mountToggle(!disabled);
  if (disabled) return;

  const host = document.getElementById('scene3d');
  if (!host) return;

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
  } catch (e) { return; }
  if (!renderer || !renderer.getContext()) return;

  // Signal the legacy 2D canvases to stop and swap styles
  window.__VK3D = true;
  document.body.classList.add('vk3d');

  const isMobile = window.matchMedia('(max-width: 900px)').matches || ('ontouchstart' in window);
  const Q = isMobile
    ? { dpr: Math.min(window.devicePixelRatio || 1, 1.3), streaks: 240, trail: 10, bloom: false, rows: 3, cols: 3, terrainSeg: 150 }
    : { dpr: Math.min(window.devicePixelRatio || 1, 1.75), streaks: 680, trail: 14, bloom: true, rows: 5, cols: 4, terrainSeg: 240 };

  renderer.setPixelRatio(Q.dpr);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const FOG = new THREE.Color('#0f3556');
  scene.fog = new THREE.FogExp2(FOG, 0.0019);

  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.5, 9000);

  // ---------- constants (1 unit = 10 m) ----------
  const HUB_H = 11;           // 110 m hub height
  const R = 7.5;              // 150 m rotor diameter
  const OVERHANG = 1.45;
  const W = new THREE.Vector3(-0.45, 0, -1).normalize(); // flow direction
  const P = new THREE.Vector3(W.z, 0, -W.x);             // cross-wind
  const UP = new THREE.Vector3(0, 1, 0);
  const YAW = Math.atan2(-W.x, -W.z);                    // rotor faces upwind
  const SUN_DIR = new THREE.Vector3(-0.06, 0.04, -1).normalize();

  // ---------- noise / terrain height ----------
  function hash(x, y) { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); }
  function vnoise(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  }
  function fbm(x, y) { let s = 0, a = 0.5, f = 1; for (let i = 0; i < 5; i++) { s += a * vnoise(x * f, y * f); f *= 2.03; a *= 0.5; } return s; }
  const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const HERO = new THREE.Vector3(10, 0, 0);
  function heightAt(x, z) {
    const r = Math.hypot(x, z);
    const amp = 16 + 110 * smooth(650, 1500, r);
    let h = (fbm(x * 0.0042 + 3.1, z * 0.0042 - 7.7) - 0.5) * 2 * amp;
    h += (vnoise(x * 0.02, z * 0.02) - 0.5) * 2.2;
    const dh = Math.hypot(x - HERO.x, z - HERO.z);
    h *= 0.35 + 0.65 * smooth(20, 140, dh); // gentle plateau at the hero WTG
    return h;
  }
  HERO.y = heightAt(HERO.x, HERO.z);

  // ---------- sky ----------
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(6000, 48, 24),
    new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { uSun: { value: SUN_DIR }, uTime: { value: 0 } },
      vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `
        varying vec3 vDir; uniform vec3 uSun;
        void main(){
          float y = vDir.y;
          vec3 zenith = vec3(0.012,0.04,0.11);
          vec3 mid    = vec3(0.03,0.14,0.27);
          vec3 hor    = vec3(0.075,0.24,0.38);
          vec3 col = mix(hor, mid, smoothstep(0.0,0.18,y));
          col = mix(col, zenith, smoothstep(0.15,0.75,y));
          float s = max(dot(normalize(vDir), uSun), 0.0);
          col += vec3(1.0,0.62,0.22) * pow(s, 7.0) * 0.4 * (1.0 - smoothstep(0.0,0.45,y));
          col += vec3(1.0,0.75,0.4) * pow(s, 140.0) * 0.55;
          col += vec3(1.0,0.9,0.7) * smoothstep(0.99955,0.9998,s) * 1.6;
          col = mix(col, vec3(0.06,0.2,0.33), 1.0 - smoothstep(-0.05,0.03,y));
          gl_FragColor = vec4(col,1.0);
        }`
    })
  );
  scene.add(sky);

  // stars
  {
    const n = 1800, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const th = Math.random() * Math.PI * 2, y = 0.12 + Math.random() * 0.88, r = Math.sqrt(1 - y * y);
      pos[i * 3] = Math.cos(th) * r * 5000; pos[i * 3 + 1] = y * 5000; pos[i * 3 + 2] = Math.sin(th) * r * 5000;
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xcfe6ff, size: 1.5, sizeAttenuation: false, transparent: true, opacity: 0.75, fog: false, depthWrite: false })));
  }

  // ---------- lights ----------
  scene.add(new THREE.HemisphereLight(0x8fc6ff, 0x0a1a14, 0.75));
  const sun = new THREE.DirectionalLight(0xffc98a, 1.25); sun.position.copy(SUN_DIR).multiplyScalar(500); scene.add(sun);
  const fill = new THREE.DirectionalLight(0x9fd4ff, 0.85); fill.position.set(120, 160, 300); scene.add(fill);

  // ---------- terrain with contour shader ----------
  const terrainUniforms = { uTime: { value: 0 }, uHero: { value: new THREE.Vector2(HERO.x, HERO.z) } };
  {
    const SIZE = 3600, SEG = Q.terrainSeg;
    const g = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG); g.rotateX(-Math.PI / 2);
    const p = g.attributes.position, cols = new Float32Array(p.count * 3);
    const cLow = new THREE.Color('#10322a'), cMid = new THREE.Color('#1d4a39'), cHi = new THREE.Color('#3b4d4a'), cTmp = new THREE.Color();
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), z = p.getZ(i), h = heightAt(x, z);
      p.setY(i, h);
      const t = smooth(-20, 60, h);
      cTmp.copy(cLow).lerp(cMid, Math.min(1, t * 1.6)).lerp(cHi, smooth(30, 120, h));
      const j = 0.9 + hash(x * 0.37, z * 0.71) * 0.2;
      cols[i * 3] = cTmp.r * j; cols[i * 3 + 1] = cTmp.g * j; cols[i * 3 + 2] = cTmp.b * j;
    }
    g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    g.computeVertexNormals();
    const m = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.95, metalness: 0.0 });
    m.onBeforeCompile = (sh) => {
      sh.uniforms.uTime = terrainUniforms.uTime; sh.uniforms.uHero = terrainUniforms.uHero;
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWp;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvWp = position;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWp; uniform float uTime; uniform vec2 uHero;')
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          float f1 = vWp.y / 2.0; float g1 = abs(fract(f1 - 0.5) - 0.5); float l1 = 1.0 - smoothstep(0.0, fwidth(f1) * 1.3, g1);
          float f2 = vWp.y / 10.0; float g2 = abs(fract(f2 - 0.5) - 0.5); float l2 = 1.0 - smoothstep(0.0, fwidth(f2) * 1.6, g2);
          float dHero = length(vWp.xz - uHero);
          float near = 1.0 - smoothstep(200.0, 1400.0, dHero);
          totalEmissiveRadiance += vec3(0.11,0.72,0.49) * (l1 * 0.07 + l2 * 0.22) * (0.35 + 0.65 * near);
          float ring = mod(uTime * 55.0, 900.0);
          float wave = exp(-pow((dHero - ring) / 5.0, 2.0)) * (1.0 - ring / 900.0);
          totalEmissiveRadiance += vec3(0.18,0.85,1.0) * wave * 0.55;`);
    };
    const mesh = new THREE.Mesh(g, m);
    scene.add(mesh);
  }

  // ---------- shared WTG geometry ----------
  function bladeGeometry() {
    const S = 26, M = 14, pos = [], idx = [];
    for (let i = 0; i <= S; i++) {
      const s = i / S, y = 0.55 + s * (R - 0.55);
      const chord = s < 0.18 ? 0.42 + (s / 0.18) * 0.55 : 0.97 * Math.pow(Math.max(0, 1 - (s - 0.18) / 0.82), 0.85) + 0.07;
      const thick = s < 0.12 ? 0.42 * (1 - s / 0.12) + 0.11 * (s / 0.12) : 0.11 - s * 0.05;
      const twist = (1 - s) * 0.32;
      for (let j = 0; j < M; j++) {
        const a = (j / M) * Math.PI * 2;
        let x = Math.cos(a) * chord * 0.5 + chord * 0.12;
        let z = Math.sin(a) * thick * chord * (Math.cos(a) > 0 ? 0.7 : 1.0);
        if (s < 0.12) { const cr = 0.2 * (1 - s / 0.12); x = x * (1 - cr) + Math.cos(a) * 0.2 * cr; z = z * (1 - cr) + Math.sin(a) * 0.2 * cr; }
        const cx = x * Math.cos(twist) - z * Math.sin(twist), cz = x * Math.sin(twist) + z * Math.cos(twist);
        pos.push(cx, y, cz);
      }
    }
    for (let i = 0; i < S; i++) for (let j = 0; j < M; j++) {
      const a = i * M + j, b = i * M + (j + 1) % M, c = (i + 1) * M + j, d = (i + 1) * M + (j + 1) % M;
      idx.push(a, c, b, b, c, d);
    }
    const tip = pos.length / 3; pos.push(0.05, R + 0.02, 0);
    for (let j = 0; j < M; j++) idx.push(S * M + j, tip, S * M + (j + 1) % M);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
    return g;
  }
  const G = {
    tower: new THREE.CylinderGeometry(0.2, 0.4, HUB_H, 28, 1).translate(0, HUB_H / 2, 0),
    found: new THREE.CylinderGeometry(1.0, 1.15, 0.25, 28).translate(0, 0.05, 0),
    nacelle: new THREE.CapsuleGeometry(0.48, 2.2, 6, 18).rotateX(Math.PI / 2),
    hub: new THREE.SphereGeometry(0.55, 24, 16).scale(1, 1, 1.25),
    spinner: new THREE.ConeGeometry(0.55, 1.0, 24).rotateX(Math.PI / 2).translate(0, 0, 0.95),
    blade: bladeGeometry(),
    cooler: new THREE.BoxGeometry(0.7, 0.35, 0.5)
  };
  const M = {
    white: new THREE.MeshStandardMaterial({ color: 0xc9d3da, roughness: 0.42, metalness: 0.15 }),
    blade: new THREE.MeshStandardMaterial({ color: 0xd4dce2, roughness: 0.35, metalness: 0.05 }),
    concrete: new THREE.MeshStandardMaterial({ color: 0x5c6670, roughness: 1 }),
    dark: new THREE.MeshStandardMaterial({ color: 0x2b3640, roughness: 0.6 })
  };

  function glowTexture() {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const x = c.getContext('2d'), gr = x.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.25, 'rgba(255,255,255,.75)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = gr; x.fillRect(0, 0, 64, 64);
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  const GLOW = glowTexture();
  const beaconMat = new THREE.SpriteMaterial({ map: GLOW, color: new THREE.Color(3.2, 0.15, 0.1), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true });

  const turbines = [];
  function makeTurbine(x, z, opts = {}) {
    const root = new THREE.Group();
    root.position.set(x, heightAt(x, z) - 0.05, z);
    const yaw = new THREE.Group(); yaw.rotation.y = YAW + (opts.yawErr || 0); root.add(yaw);
    root.add(new THREE.Mesh(G.found, M.concrete));
    root.add(new THREE.Mesh(G.tower, M.white));
    const nac = new THREE.Mesh(G.nacelle, M.white); nac.position.set(0, HUB_H + 0.25, -0.35); yaw.add(nac);
    const cool = new THREE.Mesh(G.cooler, M.white); cool.position.set(0, HUB_H + 0.85, -1.6); yaw.add(cool);
    const rotor = new THREE.Group(); rotor.position.set(0, HUB_H + 0.2, OVERHANG); yaw.add(rotor);
    rotor.add(new THREE.Mesh(G.hub, M.white));
    rotor.add(new THREE.Mesh(G.spinner, M.white));
    const tips = [];
    for (let b = 0; b < 3; b++) {
      const arm = new THREE.Group(); arm.rotation.z = b * Math.PI * 2 / 3; rotor.add(arm);
      const bl = new THREE.Mesh(G.blade, M.blade); bl.rotation.x = -0.05; arm.add(bl);
      const tip = new THREE.Object3D(); tip.position.set(0, R, 0); arm.add(tip); tips.push(tip);
    }
    const beacon = new THREE.Sprite(beaconMat); beacon.scale.setScalar(opts.hero ? 1.1 : 2.0); beacon.position.set(0, HUB_H + 1.0, -1.9); yaw.add(beacon);
    scene.add(root);
    const t = { root, yaw, rotor, tips, phase: Math.random() * Math.PI * 2, factor: opts.factor || 1, hub: new THREE.Vector3() };
    rotor.rotation.z = t.phase;
    rotor.getWorldPosition(t.hub);
    root.updateMatrixWorld(true); rotor.getWorldPosition(t.hub);
    turbines.push(t);
    return t;
  }

  // hero WTG + array layout
  const hero = makeTurbine(HERO.x, HERO.z, { hero: true });
  for (let r = -1; r < Q.rows; r++) {
    for (let c = -Q.cols; c <= Q.cols; c++) {
      if (r === 0 && c === 0) continue;
      const off = (r & 1) ? 27 : 0;
      const p = HERO.clone().addScaledVector(W, r * 105 + (hash(r, c) - 0.5) * 14).addScaledVector(P, c * 55 + off + (hash(c, r) - 0.5) * 10);
      if (r === -1 && Math.abs(c) < 2) continue; // keep the upstream sector clear for the mast
      makeTurbine(p.x, p.z, { factor: Math.pow(0.9, Math.max(0, r)) * (0.92 + hash(r * 3, c) * 0.16), yawErr: (hash(c * 7, r) - 0.5) * 0.08 });
    }
  }

  // ---------- nacelle-mounted LiDAR on hero ----------
  const lidar = new THREE.Group();
  {
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.3, 0.5), M.dark); box.position.set(0, HUB_H + 0.88, -0.6); hero.yaw.add(box);
    const beamMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.2, 2.4, 1.2), transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
    const LEN = 2.5 * 15; // 2.5D upstream
    const beams = [];
    [-0.27, 0.27].forEach((ang) => {
      const g = new THREE.CylinderGeometry(0.028, 0.028, LEN, 6, 1, true).translate(0, LEN / 2, 0).rotateX(Math.PI / 2);
      const m = new THREE.Mesh(g, beamMat.clone());
      m.position.set(0, HUB_H + 0.95, -0.6); m.rotation.y = ang; m.rotation.x = 0.012; hero.yaw.add(m);
      beams.push(m);
      for (let k = 1; k <= 5; k++) {
        const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: GLOW, color: new THREE.Color(0.4, 2.6, 1.4), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false }));
        s.scale.setScalar(0.9); s.userData.k = k;
        const d = (k / 5) * LEN;
        s.position.set(Math.sin(ang) * d, HUB_H + 0.95 - 0.012 * d, -0.6 + Math.cos(ang) * d);
        hero.yaw.add(s); beams.push(s);
      }
    });
    lidar.userData.beams = beams;
  }

  // ---------- met mast + ground LiDAR (upstream, 2.5D) ----------
  const ground = { cones: null };
  {
    const base = HERO.clone().addScaledVector(W, -2.5 * 15).addScaledVector(P, -14);
    base.y = heightAt(base.x, base.z);
    const mast = new THREE.Group(); mast.position.copy(base); scene.add(mast);
    const legR = 0.32, H = HUB_H;
    const legMat = new THREE.MeshStandardMaterial({ color: 0xb9c3cc, roughness: 0.5, metalness: 0.6 });
    const pts = [];
    for (let l = 0; l < 3; l++) {
      const a = l * Math.PI * 2 / 3, leg = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, H, 5).translate(0, H / 2, 0), legMat);
      leg.position.set(Math.cos(a) * legR, 0, Math.sin(a) * legR); mast.add(leg);
    }
    for (let y = 0; y < H; y += 0.55) for (let l = 0; l < 3; l++) {
      const a1 = l * Math.PI * 2 / 3, a2 = (l + 1) * Math.PI * 2 / 3;
      pts.push(Math.cos(a1) * legR, y, Math.sin(a1) * legR, Math.cos(a2) * legR, y + 0.55, Math.sin(a2) * legR);
    }
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    mast.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x9fb0bd, transparent: true, opacity: 0.8 })));
    [H, H * 0.78, H * 0.55].forEach((h, i) => {
      const boom = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.04, 0.04), legMat); boom.position.set(0.8, h, 0); boom.rotation.y = -YAW + Math.PI / 2 + i * 0.3; mast.add(boom);
    });
    const top = new THREE.Sprite(beaconMat); top.scale.setScalar(1.2); top.position.set(0, H + 0.3, 0); mast.add(top);
    // ground-based profiling LiDAR with rotating scan cone
    const gl = new THREE.Group(); gl.position.copy(base).addScaledVector(P, 7); gl.position.y = heightAt(gl.position.x, gl.position.z); scene.add(gl);
    gl.add(new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.55, 0.6).translate(0, 0.28, 0), new THREE.MeshStandardMaterial({ color: 0xe9eef1, roughness: 0.6 })));
    const cone = new THREE.Group(); cone.position.y = 0.6; gl.add(cone);
    const cg = new THREE.ConeGeometry(Math.tan(0.49) * 20, 20, 48, 1, true).translate(0, -10, 0).rotateX(Math.PI);
    cone.add(new THREE.Mesh(cg, new THREE.MeshBasicMaterial({ color: new THREE.Color(0.1, 0.9, 1.3), transparent: true, opacity: 0.06, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide })));
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 20, 5, 1, true).translate(0, 10, 0),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.3, 2.0, 2.6), transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false }));
    const pivot = new THREE.Group(); pivot.add(beam); beam.rotation.z = 0.49; cone.add(pivot);
    ground.cones = pivot;
  }

  // ---------- flow streamlines with wake deficit ----------
  const N = Q.streaks, K = Q.trail;
  const head = new Float32Array(N * 3), hoff = new Float32Array(N), life = new Float32Array(N), trail = new Float32Array(N * K * 3), wakeAmt = new Float32Array(N);
  const segPos = new Float32Array(N * (K - 1) * 6), segCol = new Float32Array(N * (K - 1) * 6);
  const streakGeo = new THREE.BufferGeometry();
  streakGeo.setAttribute('position', new THREE.BufferAttribute(segPos, 3).setUsage(THREE.DynamicDrawUsage));
  streakGeo.setAttribute('color', new THREE.BufferAttribute(segCol, 3).setUsage(THREE.DynamicDrawUsage));
  const streaks = new THREE.LineSegments(streakGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  streaks.frustumCulled = false; scene.add(streaks);
  const focus = new THREE.Vector3().copy(HERO);
  const SPAN_L = 230, SPAN_S = 170;
  function spawn(i, anywhere) {
    const a = anywhere ? (Math.random() * 2 - 1) * SPAN_L : -SPAN_L * (0.8 + Math.random() * 0.2);
    const s = (Math.random() * 2 - 1) * SPAN_S;
    const x = focus.x + W.x * a + P.x * s, z = focus.z + W.z * a + P.z * s;
    hoff[i] = 1.5 + Math.pow(Math.random(), 1.4) * 22;
    const y = heightAt(x, z) + hoff[i];
    head[i * 3] = x; head[i * 3 + 1] = y; head[i * 3 + 2] = z;
    for (let k = 0; k < K; k++) { trail[(i * K + k) * 3] = x; trail[(i * K + k) * 3 + 1] = y; trail[(i * K + k) * 3 + 2] = z; }
    life[i] = 0; wakeAmt[i] = 0;
  }
  for (let i = 0; i < N; i++) spawn(i, true);
  const tmpV = new THREE.Vector3();
  function updateStreaks(dt, U) {
    const base = U * 1.45;
    for (let i = 0; i < N; i++) {
      const ix = i * 3;
      let x = head[ix], y = head[ix + 1], z = head[ix + 2];
      // power-law shear profile (alpha 0.14)
      let sp = base * Math.pow(Math.max(hoff[i], 0.5) / HUB_H, 0.14);
      // wake deficit (Jensen-like top-hat with k=0.075)
      let wk = 0;
      for (let t = 0; t < turbines.length; t++) {
        const h = turbines[t].hub;
        const dx = x - h.x, dy = y - h.y, dz = z - h.z;
        const along = dx * W.x + dz * W.z;
        if (along < 0.5 || along > 160) continue;
        const lx = dx - along * W.x, lz = dz - along * W.z;
        const lat2 = lx * lx + lz * lz + dy * dy;
        const rw = R + 0.075 * along;
        if (lat2 < rw * rw) { const def = (1 - Math.sqrt(1 - 0.8)) * (R / rw) * (R / rw); if (def > wk) wk = def; }
      }
      sp *= 1 - wk;
      wakeAmt[i] += (wk * 3.2 - wakeAmt[i]) * Math.min(1, dt * 4);
      x += W.x * sp * dt; z += W.z * sp * dt;
      const ty = heightAt(x, z) + hoff[i];
      y += (ty - y) * Math.min(1, dt * 2.5);
      head[ix] = x; head[ix + 1] = y; head[ix + 2] = z;
      life[i] += dt;
      // shift trail
      const o = i * K * 3;
      trail.copyWithin(o + 3, o, o + (K - 1) * 3);
      trail[o] = x; trail[o + 1] = y; trail[o + 2] = z;
      tmpV.set(x - focus.x, 0, z - focus.z);
      const a = tmpV.dot(W), s = Math.abs(tmpV.x * P.x + tmpV.z * P.z);
      if (a > SPAN_L || s > SPAN_S * 1.15 || life[i] > 40) spawn(i, false);
      // write segments
      const fadeIn = Math.min(1, life[i] * 1.5);
      const wkA = Math.min(1, wakeAmt[i]);
      const cr = 0.25 + 0.85 * wkA, cg = 0.75 - 0.15 * wkA, cb = 1.0 - 0.8 * wkA;
      for (let k = 0; k < K - 1; k++) {
        const so = (i * (K - 1) + k) * 6, ta = o + k * 3, tb = o + (k + 1) * 3;
        segPos[so] = trail[ta]; segPos[so + 1] = trail[ta + 1]; segPos[so + 2] = trail[ta + 2];
        segPos[so + 3] = trail[tb]; segPos[so + 4] = trail[tb + 1]; segPos[so + 5] = trail[tb + 2];
        const a1 = (1 - k / (K - 1)) * fadeIn * 0.55, a2 = (1 - (k + 1) / (K - 1)) * fadeIn * 0.55;
        segCol[so] = cr * a1; segCol[so + 1] = cg * a1; segCol[so + 2] = cb * a1;
        segCol[so + 3] = cr * a2; segCol[so + 4] = cg * a2; segCol[so + 5] = cb * a2;
      }
    }
    streakGeo.attributes.position.needsUpdate = true;
    streakGeo.attributes.color.needsUpdate = true;
  }

  // ---------- tip vortices behind the hero rotor ----------
  const VN = isMobile ? 500 : 1400;
  const vPos = new Float32Array(VN * 3), vCol = new Float32Array(VN * 3), vAge = new Float32Array(VN).fill(99);
  const vGeo = new THREE.BufferGeometry();
  vGeo.setAttribute('position', new THREE.BufferAttribute(vPos, 3).setUsage(THREE.DynamicDrawUsage));
  vGeo.setAttribute('color', new THREE.BufferAttribute(vCol, 3).setUsage(THREE.DynamicDrawUsage));
  const vortex = new THREE.Points(vGeo, new THREE.PointsMaterial({ size: 0.3, map: GLOW, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  vortex.frustumCulled = false; scene.add(vortex);
  let vHead = 0;
  const tipWP = new THREE.Vector3();
  function updateVortex(dt, U) {
    for (let b = 0; b < 3; b++) {
      hero.tips[b].getWorldPosition(tipWP);
      for (let e = 0; e < 2; e++) {
        const i = vHead; vHead = (vHead + 1) % VN;
        vPos[i * 3] = tipWP.x; vPos[i * 3 + 1] = tipWP.y; vPos[i * 3 + 2] = tipWP.z; vAge[i] = e * dt * 0.5;
      }
    }
    const conv = U * 0.75, MAX = 3.2;
    for (let i = 0; i < VN; i++) {
      const a = vAge[i];
      if (a > MAX) { vCol[i * 3] = vCol[i * 3 + 1] = vCol[i * 3 + 2] = 0; continue; }
      vAge[i] = a + dt;
      vPos[i * 3] += W.x * conv * dt; vPos[i * 3 + 2] += W.z * conv * dt;
      const f = (1 - a / MAX); const k = f * f * 0.9;
      vCol[i * 3] = 0.35 * k; vCol[i * 3 + 1] = 0.9 * k; vCol[i * 3 + 2] = 1.0 * k;
    }
    vGeo.attributes.position.needsUpdate = true; vGeo.attributes.color.needsUpdate = true;
  }

  // ---------- post-processing ----------
  let composer = null, bloom = null;
  if (Q.bloom) {
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.6, 0.55, 0.9);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
  }

  // ---------- camera choreography ----------
  const V = (base, a, s, up) => base.clone().addScaledVector(W, a).addScaledVector(P, s).add(new THREE.Vector3(0, up, 0));
  const portrait = () => camera.aspect < 0.95;
  const SCENES = [
    { id: 'hero', label: 'Hero WTG · NML', f: (t) => portrait()
      ? { p: HERO.clone().add(new THREE.Vector3(-4, 7.5, 40)), t: HERO.clone().add(new THREE.Vector3(1.5, 13.5, 0)) }
      : { p: HERO.clone().add(new THREE.Vector3(-13 + Math.sin(t * 0.07) * 1.2, 8, 37)), t: HERO.clone().add(new THREE.Vector3(-10, 12.5, 0)) } },
    { id: 'about', label: 'Wind Farm Overview', f: () => ({ p: V(HERO, -70, -60, 55), t: V(HERO, 160, 30, 0) }) },
    { id: 'experience', label: 'Array Fly-through', f: (t) => ({ p: V(HERO, 52 + Math.sin(t * 0.05) * 6, -230, 17), t: V(HERO, 52, 60, 9) }) },
    { id: 'projects', label: 'Nacelle LiDAR · Met Mast', f: () => ({ p: V(HERO, -78, 48, 26), t: V(HERO, -22, -6, 10) }) },
    { id: 'skills', label: 'Layout Plan View', f: () => ({ p: V(HERO, 120, -40, 360), t: V(HERO, 190, 0, 0) }) },
    { id: 'education', label: 'Horizon', f: () => ({ p: V(HERO, -40, 110, 5), t: V(HERO, 380, -40, 22) }) },
    { id: 'tools', label: 'Orbit · Digital Twin', f: (t) => { const a = t * 0.08; return { p: HERO.clone().add(new THREE.Vector3(Math.cos(a) * 42, 16, Math.sin(a) * 42)), t: HERO.clone().add(new THREE.Vector3(0, 10, 0)) }; } },
    { id: 'contact', label: 'Sunset Departure', f: () => ({ p: V(HERO, -140, 20, 70), t: V(HERO, 520, -60, 0) }) }
  ];
  const secEls = SCENES.map((s) => document.getElementById(s.id));
  let tops = [];
  function measure() { tops = secEls.map((el) => el ? el.getBoundingClientRect().top + window.scrollY : 1e9); }
  measure();
  setInterval(measure, 1500);

  const camPos = new THREE.Vector3(), camTgt = new THREE.Vector3(), desP = new THREE.Vector3(), desT = new THREE.Vector3();
  const ease = (x) => x * x * (3 - 2 * x);
  let activeIdx = 0;
  function desired(time) {
    const y = window.scrollY + window.innerHeight * 0.4;
    let i = 0;
    for (let k = 0; k < tops.length; k++) if (y >= tops[k]) i = k;
    const next = Math.min(i + 1, SCENES.length - 1);
    const h = (tops[next] - tops[i]) || 1;
    const t = next === i ? 0 : Math.min(1, Math.max(0, (y - tops[i]) / h));
    const e = ease(smooth(0.3, 1.0, t));
    const A = SCENES[i].f(time), B = SCENES[next].f(time);
    desP.copy(A.p).lerp(B.p, e); desT.copy(A.t).lerp(B.t, e);
    activeIdx = e > 0.5 ? next : i;
  }

  // mouse parallax
  let mx = 0, my = 0, smx = 0, smy = 0;
  window.addEventListener('pointermove', (e) => { mx = e.clientX / window.innerWidth - 0.5; my = e.clientY / window.innerHeight - 0.5; }, { passive: true });

  // ---------- SCADA HUD (simulated) ----------
  const hud = buildHud();
  const dots = buildDots(SCENES, secEls);

  // ---------- resize ----------
  function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    if (composer) { composer.setSize(window.innerWidth, window.innerHeight); }
    measure();
  }
  window.addEventListener('resize', onResize);
  onResize();

  // initial camera
  desired(0); camPos.copy(desP); camTgt.copy(desT);

  // ---------- loop ----------
  const clock = new THREE.Clock();
  let time = 0, frames = 0, fpsAcc = 0, degraded = false, hudTimer = 0, shown = false;
  const right = new THREE.Vector3();
  function windSpeed(t) { return 9.1 + 1.3 * Math.sin(t * 0.31) + 0.7 * Math.sin(t * 0.73 + 1.3) + 0.35 * Math.sin(t * 1.9 + 0.4); }

  function frame() {
    requestAnimationFrame(frame);
    if (document.hidden) return;
    const dt = Math.min(clock.getDelta(), 0.05);
    time += dt;

    const U = windSpeed(time);
    const omega = Math.min(8.2 * U / 75, 11.6 * 2 * Math.PI / 60); // rad/s (TSR 8.2, rated 11.6 rpm)

    // rotors
    for (let i = 0; i < turbines.length; i++) turbines[i].rotor.rotation.z -= omega * turbines[i].factor * dt;

    // beacons (synchronised, 30 fpm)
    beaconMat.opacity = 0.15 + 0.85 * Math.pow(Math.max(0, Math.sin(time * Math.PI)), 6);

    // LiDAR pulses
    const beams = lidar.userData.beams;
    for (let b = 0; b < beams.length; b++) {
      const o = beams[b];
      if (o.isSprite) { const ph = (time * 1.6 - o.userData.k * 0.18) % 1; o.material.opacity = 0.25 + 0.75 * Math.exp(-ph * 9); }
      else o.material.opacity = 0.35 + 0.2 * Math.sin(time * 6);
    }
    if (ground.cones) ground.cones.rotation.y = time * 1.1;

    terrainUniforms.uTime.value = time;

    // camera
    desired(time);
    smx += (mx - smx) * 0.05; smy += (my - smy) * 0.05;
    const k = 1 - Math.pow(0.0015, dt);
    camPos.lerp(desP, k); camTgt.lerp(desT, k);
    camera.position.copy(camPos);
    right.subVectors(camTgt, camPos).cross(UP).normalize();
    const dist = camPos.distanceTo(camTgt);
    const par = Math.min(4, dist * 0.06);
    camera.position.addScaledVector(right, smx * par).addScaledVector(UP, -smy * par * 0.6);
    const gy = heightAt(camera.position.x, camera.position.z) + 2.5;
    if (camera.position.y < gy) camera.position.y = gy;
    camera.lookAt(camTgt);

    // focus region for flow particles follows view target
    focus.x += (camTgt.x - focus.x) * 0.02; focus.z += (camTgt.z - focus.z) * 0.02;
    updateStreaks(dt, U);
    updateVortex(dt, U);

    if (composer) composer.render(); else renderer.render(scene, camera);

    if (!shown) { shown = true; host.classList.add('ready'); }

    // HUD + scene dots
    hudTimer += dt;
    if (hudTimer > 0.25) { hudTimer = 0; hud.update(time, U, omega); dots.setActive(activeIdx); }

    // adaptive quality
    if (!degraded && time > 1.5 && time < 6) {
      frames++; fpsAcc += dt;
      if (frames > 90) {
        const fps = frames / fpsAcc;
        if (fps < 34) {
          degraded = true;
          renderer.setPixelRatio(1);
          if (composer) { composer.setPixelRatio ? composer.setPixelRatio(1) : null; composer.setSize(window.innerWidth, window.innerHeight); }
          if (fps < 24) composer = null;
        }
        frames = 1e9;
      }
    }
  }
  requestAnimationFrame(frame);

  // ---------- DOM helpers ----------
  function buildHud() {
    const parent = document.getElementById('heroRight');
    const el = document.createElement('div');
    el.className = 'vk-hud';
    el.innerHTML = `
      <div class="vk-hud-h"><span class="vk-live"></span>WTG-01 · Simulated SCADA</div>
      <div class="vk-hud-g">
        <div><b data-k="ws">—</b><i>Hub wind m/s</i></div>
        <div><b data-k="rpm">—</b><i>Rotor rpm</i></div>
        <div><b data-k="p">—</b><i>Active power MW</i></div>
        <div><b data-k="yaw">—</b><i>Yaw error °</i></div>
      </div>
      <div class="vk-hud-bar"><span data-k="bar"></span></div>
      <div class="vk-hud-f">NML · 2 beams · 2.5D upstream · 10-min avg ↺</div>`;
    if (parent) parent.appendChild(el);
    const q = (k) => el.querySelector(`[data-k="${k}"]`);
    const ws = q('ws'), rpm = q('rpm'), p = q('p'), yaw = q('yaw'), bar = q('bar');
    return {
      update(t, U, omega) {
        const pw = Math.min(6.0, 0.5 * 1.225 * Math.PI * 75 * 75 * 0.47 * U * U * U / 1e6);
        ws.textContent = U.toFixed(1);
        rpm.textContent = (omega * 60 / (2 * Math.PI)).toFixed(1);
        p.textContent = pw.toFixed(2);
        yaw.textContent = (1.6 * Math.sin(t * 0.21) + 0.6 * Math.sin(t * 0.9)).toFixed(1);
        bar.style.width = (pw / 6 * 100).toFixed(1) + '%';
      }
    };
  }

  function buildDots(scenes, els) {
    const nav = document.createElement('div');
    nav.className = 'vk-dots'; nav.setAttribute('role', 'navigation'); nav.setAttribute('aria-label', 'Section navigation');
    scenes.forEach((s, i) => {
      if (!els[i]) return;
      const a = document.createElement('a');
      a.href = '#' + s.id; a.innerHTML = `<span class="vk-dot-l">${String(i + 1).padStart(2, '0')} · ${s.label}</span><span class="vk-dot"></span>`;
      nav.appendChild(a);
    });
    document.body.appendChild(nav);
    const links = [...nav.querySelectorAll('a')];
    let last = -1;
    return { setActive(i) { if (i === last) return; last = i; links.forEach((l, j) => l.classList.toggle('on', j === i)); } };
  }
})();

// ---------- 3D toggle (always mounted) ----------
function mountToggle(on) {
  const b = document.createElement('button');
  b.className = 'vk-3d-toggle' + (on ? ' on' : '');
  b.type = 'button';
  b.title = on ? 'Switch to the lightweight 2D site' : 'Switch on the 3D wind-farm experience';
  b.innerHTML = `<span class="vk-3d-ico"></span>3D ${on ? 'ON' : 'OFF'}`;
  b.addEventListener('click', () => { try { localStorage.setItem('vk3d', on ? 'off' : 'on'); } catch (e) {} location.reload(); });
  const add = () => document.body.appendChild(b);
  document.body ? add() : window.addEventListener('DOMContentLoaded', add);
}

// ---------- 3D tilt cards (independent of WebGL) ----------
(function tilt() {
  if (('ontouchstart' in window) || (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) return;
  const els = document.querySelectorAll('.pcard, .sgroup, .acard, .ecard2, .metric, .ecard');
  els.forEach((el) => {
    el.classList.add('tilt3d');
    const max = el.classList.contains('pcard--w') ? 4 : (el.classList.contains('metric') || el.classList.contains('ecard')) ? 5 : 8;
    let raf = 0;
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        el.style.transform = `perspective(1000px) rotateX(${(0.5 - py) * max}deg) rotateY(${(px - 0.5) * max}deg) translateZ(0) translateY(-4px)`;
        el.style.setProperty('--gx', (px * 100).toFixed(1) + '%');
        el.style.setProperty('--gy', (py * 100).toFixed(1) + '%');
      });
    });
    el.addEventListener('pointerleave', () => { cancelAnimationFrame(raf); el.style.transform = ''; });
  });
})();
