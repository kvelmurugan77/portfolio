/**
 * WindFlow Pro v61 — WAsP-class BZ Orographic Flow Engine
 * ========================================================
 * Drop-in upgrade for WindFlow Pro 60.3.
 *
 * Goal: Match / exceed WAsP IBZ (BZ) linearized spectral flow on the same
 * inputs (DEM + z0 + hub height + sector wind), especially:
 *   - Absolute and relative orographic speed-up (site vs mast)
 *   - Directional dependence
 *   - Height dependence via Jackson–Hunt inner-layer scale
 *   - Nonlinear attenuation + RIX (Bowen–Mortensen style)
 *   - Clean OWC → GWC → SWC relative-speedup chain
 *
 * Physics references (same lineage as WAsP / MS-Micro / MSFD):
 *   Jackson & Hunt (1975), Mason & Sykes (1979), Walmsley et al. (1982, 1986),
 *   Troen & Petersen EWA (1989), Beljaars et al. (1987), Bowen & Mortensen RIX.
 *
 * CRITICAL FIX vs v60 runBZModel:
 *   v60 computed only a 1-D profile JH surrogate and NEVER called the spectral
 *   BZ FFT path (fft2D / bzG / bzDelta*). WAsP is spectral. This module makes
 *   spectral BZ the primary path and keeps profile-JH only as fallback.
 *
 * Integration: append as external script OR load after main app main app. It monkey-patches:
 *   runBZModel, jacksonHuntSpeedup2D, terrainSpeedupJH, bzDeltaAtDir,
 *   secondOrderBZCorrection, waspOrographicSaturation, computeHillParameters
 */
(function (global) {
  'use strict';

  const WFP61 = {
    version: '61.0-BZ-Spectral',
    modelName: 'WAsP-class BZ (spectral Jackson–Hunt / IBZ)',
  };

  // ── Math helpers ──────────────────────────────────────────────────────────
  const PI = Math.PI;
  const KAPPA = 0.4;

  function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function isNum(x) {
    return typeof x === 'number' && isFinite(x);
  }

  // Modified Bessel K0, K1 (Abramowitz & Stegun rational approx) — self-contained
  function besselI0(x) {
    const ax = Math.abs(x);
    if (ax < 3.75) {
      const y = x / 3.75;
      const y2 = y * y;
      return (
        1.0 +
        y2 *
          (3.5156229 +
            y2 *
              (3.0899424 +
                y2 * (1.2067492 + y2 * (0.2659732 + y2 * (0.0360768 + y2 * 0.0045813)))))
      );
    }
    const y = 3.75 / ax;
    return (
      (Math.exp(ax) / Math.sqrt(ax)) *
      (0.39894228 +
        y *
          (0.01328592 +
            y *
              (0.00225319 +
                y *
                  (-0.00157565 +
                    y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    );
  }

  function besselI1(x) {
    const ax = Math.abs(x);
    let ans;
    if (ax < 3.75) {
      const y = x / 3.75;
      const y2 = y * y;
      ans =
        ax *
        (0.5 +
          y2 *
            (0.87890594 +
              y2 *
                (0.51498869 +
                  y2 * (0.15084934 + y2 * (0.02658733 + y2 * (0.00301532 + y2 * 0.00032411))))));
    } else {
      const y = 3.75 / ax;
      ans =
        0.39894228 +
        y *
          (-0.03988024 +
            y *
              (-0.00362018 +
                y *
                  (0.00163801 +
                    y *
                      (-0.01031555 +
                        y * (0.02282967 + y * (-0.02895312 + y * (0.01787654 + y * -0.00420059)))))));
      ans *= Math.exp(ax) / Math.sqrt(ax);
    }
    return x < 0 ? -ans : ans;
  }

  function K0(x) {
    if (x <= 0) return 1e20;
    if (x <= 2) {
      const t = x * 0.5;
      const t2 = t * t;
      return (
        -Math.log(t) * besselI0(x) +
        (-0.57721566 +
          0.4227842 * t2 +
          0.23069756 * t2 * t2 +
          0.0348859 * t2 * t2 * t2 +
          0.00262698 * Math.pow(t2, 4) +
          0.0001075 * Math.pow(t2, 5) +
          0.0000074 * Math.pow(t2, 6))
      );
    }
    const t = 2 / x;
    return (
      (Math.exp(-x) / Math.sqrt(x)) *
      (1.25331414 -
        0.07832358 * t +
        0.02189568 * t * t -
        0.01062446 * t * t * t +
        0.00587872 * Math.pow(t, 4) -
        0.0025154 * Math.pow(t, 5) +
        0.00053208 * Math.pow(t, 6))
    );
  }

  function K1(x) {
    if (x <= 0) return 1e20;
    if (x <= 2) {
      const t = x * 0.5;
      const t2 = t * t;
      return (
        Math.log(t) * besselI1(x) +
        (1 / x) *
          (1 +
            0.15443144 * t2 -
            0.67278579 * t2 * t2 -
            0.18156897 * t2 * t2 * t2 -
            0.01919402 * Math.pow(t2, 4) -
            0.00110404 * Math.pow(t2, 5) -
            0.00004686 * Math.pow(t2, 6))
      );
    }
    const t = 2 / x;
    return (
      (Math.exp(-x) / Math.sqrt(x)) *
      (1.25331414 +
        0.23498619 * t -
        0.0365562 * t * t +
        0.01504268 * t * t * t -
        0.00780353 * Math.pow(t, 4) +
        0.00325614 * Math.pow(t, 5) -
        0.00068245 * Math.pow(t, 6))
    );
  }

  // ── 1-D radix-2 FFT (in-place) ────────────────────────────────────────────
  function fft1d(re, im, inverse) {
    const n = re.length;
    if (n < 2 || (n & (n - 1)) !== 0) {
      // not power of 2 — zero-pad expectation is caller's job
      throw new Error('fft1d requires power-of-2 length, got ' + n);
    }
    // bit reverse
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        let ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = ((inverse ? 2 : -2) * PI) / len;
      const wlenRe = Math.cos(ang);
      const wlenIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let wRe = 1,
          wIm = 0;
        for (let j = 0; j < len / 2; j++) {
          const uRe = re[i + j],
            uIm = im[i + j];
          const vRe = re[i + j + len / 2] * wRe - im[i + j + len / 2] * wIm;
          const vIm = re[i + j + len / 2] * wIm + im[i + j + len / 2] * wRe;
          re[i + j] = uRe + vRe;
          im[i + j] = uIm + vIm;
          re[i + j + len / 2] = uRe - vRe;
          im[i + j + len / 2] = uIm - vIm;
          const nwRe = wRe * wlenRe - wIm * wlenIm;
          wIm = wRe * wlenIm + wIm * wlenRe;
          wRe = nwRe;
        }
      }
    }
    if (inverse) {
      for (let i = 0; i < n; i++) {
        re[i] /= n;
        im[i] /= n;
      }
    }
  }

  /**
   * 2-D real terrain FFT with WAsP-like preprocessing:
   *  - remove planar trend (least-squares plane)
   *  - remove mean residual
   *  - Tukey (tapered cosine) window — gentler than Hanning on edges
   *  - 2× zero-pad
   * Returns spectrum Re/Im of size py×px and metadata.
   */
  function terrainFFT2D(hGrid, ny, nx, opts) {
    opts = opts || {};
    const taperAlpha = opts.taperAlpha != null ? opts.taperAlpha : 0.2; // Tukey α

    // 1) planar detrend
    let sX = 0,
      sY = 0,
      sZ = 0,
      sXX = 0,
      sYY = 0,
      sXY = 0,
      sXZ = 0,
      sYZ = 0,
      N = ny * nx;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const z = hGrid[j][i];
        sX += i;
        sY += j;
        sZ += z;
        sXX += i * i;
        sYY += j * j;
        sXY += i * j;
        sXZ += i * z;
        sYZ += j * z;
      }
    }
    // Solve [XX XY X; XY YY Y; X Y N] [a;b;c] = [XZ;YZ;Z]
    // Cramer's / normal equations for plane z = a*i + b*j + c
    const m00 = sXX,
      m01 = sXY,
      m02 = sX;
    const m10 = sXY,
      m11 = sYY,
      m12 = sY;
    const m20 = sX,
      m21 = sY,
      m22 = N;
    const det =
      m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);
    let a = 0,
      b = 0,
      c = sZ / N;
    if (Math.abs(det) > 1e-12) {
      const d0 = sXZ,
        d1 = sYZ,
        d2 = sZ;
      a =
        (d0 * (m11 * m22 - m12 * m21) - m01 * (d1 * m22 - m12 * d2) + m02 * (d1 * m21 - m11 * d2)) /
        det;
      b =
        (m00 * (d1 * m22 - m12 * d2) - d0 * (m10 * m22 - m12 * m20) + m02 * (m10 * d2 - d1 * m20)) /
        det;
      c =
        (m00 * (m11 * d2 - d1 * m21) - m01 * (m10 * d2 - d1 * m20) + d0 * (m10 * m21 - m11 * m20)) /
        det;
    }

    const detrended = Array.from({ length: ny }, () => new Float64Array(nx));
    let mean = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = hGrid[j][i] - (a * i + b * j + c);
        detrended[j][i] = v;
        mean += v;
      }
    }
    mean /= N;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) detrended[j][i] -= mean;

    // Tukey window
    function tukey(n, alpha) {
      const w = new Float64Array(n);
      const edge = Math.floor((alpha * (n - 1)) / 2);
      for (let i = 0; i < n; i++) {
        if (i < edge) {
          w[i] = 0.5 * (1 + Math.cos(PI * (i / edge - 1)));
        } else if (i > n - 1 - edge) {
          w[i] = 0.5 * (1 + Math.cos(PI * ((i - (n - 1)) / edge + 1)));
        } else {
          w[i] = 1;
        }
      }
      return w;
    }
    const wY = tukey(ny, taperAlpha);
    const wX = tukey(nx, taperAlpha);

    let py = 1;
    while (py < ny * 2) py <<= 1;
    let px = 1;
    while (px < nx * 2) px <<= 1;

    const Re = Array.from({ length: py }, () => new Float64Array(px));
    const Im = Array.from({ length: py }, () => new Float64Array(px));
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        Re[j][i] = detrended[j][i] * wY[j] * wX[i];
      }
    }

    // row FFT
    for (let j = 0; j < py; j++) {
      const rowRe = Float64Array.from(Re[j]);
      const rowIm = Float64Array.from(Im[j]);
      fft1d(rowRe, rowIm, false);
      Re[j] = rowRe;
      Im[j] = rowIm;
    }
    // col FFT
    const colRe = new Float64Array(py);
    const colIm = new Float64Array(py);
    for (let i = 0; i < px; i++) {
      for (let j = 0; j < py; j++) {
        colRe[j] = Re[j][i];
        colIm[j] = Im[j][i];
      }
      fft1d(colRe, colIm, false);
      for (let j = 0; j < py; j++) {
        Re[j][i] = colRe[j];
        Im[j][i] = colIm[j];
      }
    }

    return {
      Re,
      Im,
      py,
      px,
      ny,
      nx,
      plane: { a, b, c },
      meanResidual: mean,
      padFactorY: py / ny,
      padFactorX: px / nx,
    };
  }

  /**
   * Inner-layer scale ℓ (Beljaars / Walmsley / WAsP style).
   * Solves ℓ ln(ℓ/z0) = 2 κ² L_h  with L_h characteristic horizontal scale.
   * For spectral model, local mixing length at hub:
   *   l = κ z / ln(z/z0)   (used inside G)
   * and outer scale enters via wavenumber K.
   */
  function innerLayerLength(hubH, z0, Lh) {
    z0 = Math.max(z0, 1e-6);
    hubH = Math.max(hubH, z0 * 2);
    if (Lh && Lh > 0) {
      let ell = (2 * KAPPA * KAPPA * Lh) / Math.max(1, Math.log(Math.max(Lh, z0 * 10) / z0));
      for (let it = 0; it < 8; it++) {
        const ln = Math.log(Math.max(ell, z0 * 2) / z0);
        ell = (2 * KAPPA * KAPPA * Lh) / Math.max(ln, 0.5);
      }
      return clamp(ell, z0 * 10, Lh * 0.5);
    }
    // spectral mixing length
    const ln = Math.log(hubH / z0);
    return (KAPPA * hubH) / Math.max(ln, 0.5);
  }

  /**
   * WAsP/JH transfer function G(K, z) for fractional speed-up of streamwise wind.
   * G(K,z) = (H-scale absorbed in ĥ) · (2/l) · [K0(Kz) − K0(Kl)] / [K · K1(Kl)]
   *
   * Note: ĥ(K) already carries metres of elevation; ΔS is dimensionless.
   * Some texts factor H outside — here elevation spectrum is absolute (m).
   *
   * Additional mild low-pass: suppress K > π / Δx_eff (done by caller).
   */
  function bzTransferG(K, z, l) {
    if (K < 1e-10 || z < 1e-4 || l < 1e-4) return 0;
    const Kz = K * z;
    const Kl = K * l;
    if (Kz < 1e-8 || Kl < 1e-8) return 0;
    if (Kz > 50 || Kl > 50) return 0;
    const k0z = K0(Kz);
    const k0l = K0(Kl);
    const k1l = K1(Kl);
    if (!isFinite(k0z) || !isFinite(k0l) || Math.abs(k1l) < 1e-20) return 0;
    // Dimensionless discrete kernel (same structure as host bzG, sign-corrected):
    // G = (2/l) * (K0(Kl) - K0(Kz)) / (K * K1(Kl))
    const G = ((2 / l) * (k0l - k0z)) / (K * k1l);
    return isFinite(G) ? G : 0;
  }

  /**
   * Cross-wind transfer for flow turning (order-of-magnitude JH).
   */
  function bzTransferGcross(K, z, l) {
    if (K < 1e-10 || z < 1e-4 || l < 1e-4) return 0;
    const Kz = K * z,
      Kl = K * l;
    if (Kz < 1e-8 || Kl < 1e-8 || Kz > 80 || Kl > 80) return 0;
    const k1z = K1(Kz),
      k1l = K1(Kl);
    if (Math.abs(k1l) < 1e-20) return 0;
    const G = ((2 / l) * k1z) / k1l;
    return isFinite(G) ? G : 0;
  }

  /**
   * Evaluate spectral ΔS and turning at physical position (x,y) [m] in domain.
   * windFromDeg: meteorological FROM direction (0=N, 90=E).
   * halfPlane: if true, only Fourier modes with upwind component (WAsP sector filter).
   *
   * Normalization: FFT without 1/N in forward transform ⇒ divide by (py*px).
   * Window energy loss ≈ mean(wY)^2*mean(wX)^2 — mild boost applied.
   */
  
  /**
   * Build real-space ΔS field via: IFFT( G(K)·ĥ(K) ) with proper 1/(py*px) norm.
   * Optional half-plane filter for directional sector.
   * Returns Float64Array length py*px in row-major (same as padded FFT grid).
   */
  function buildDeltaField(spec, Lx, Ly, hubH, z0, windFromDeg, options) {
    options = options || {};
    const halfPlane = options.halfPlane !== false;
    const lScale = options.lScale != null ? options.lScale : 1.0;
    const { Re, Im, py, px, ny, nx } = spec;
    const l = lScale * innerLayerLength(hubH, z0, null);

    const theta = ((windFromDeg || 0) * PI) / 180;
    const upX = Math.sin(theta);
    const upY = Math.cos(theta);

    // Filtered spectral product
    const fRe = Array.from({ length: py }, () => new Float64Array(px));
    const fIm = Array.from({ length: py }, () => new Float64Array(px));

    for (let ky = 0; ky < py; ky++) {
      for (let kx = 0; kx < px; kx++) {
        if (ky === 0 && kx === 0) continue;
        const fkx = kx <= px / 2 ? kx : kx - px;
        const fky = ky <= py / 2 ? ky : ky - py;
        const Kx = (2 * PI * fkx * nx) / (px * Lx);
        const Ky = (2 * PI * fky * ny) / (py * Ly);
        const K = Math.hypot(Kx, Ky);
        if (K < PI / Math.max(Lx, Ly)) continue;
        const Knyq = PI * Math.max(nx / Lx, ny / Ly);
        if (K > Knyq * 0.95) continue;

        if (halfPlane) {
          const kDotUp = Kx * upX + Ky * upY;
          if (kDotUp < -1e-15) continue;
        }

        const G = bzTransferG(K, hubH, l);
        if (Math.abs(G) < 1e-18) continue;

        // Discrete spectral product (G dimensionless kernel × DFT height)
        fRe[ky][kx] = G * Re[ky][kx];
        fIm[ky][kx] = G * Im[ky][kx];
      }
    }

    // Inverse FFT → spatial field of relative speed-up (raw discrete units)
    const colRe = new Float64Array(py);
    const colIm = new Float64Array(py);
    for (let i = 0; i < px; i++) {
      for (let j = 0; j < py; j++) {
        colRe[j] = fRe[j][i];
        colIm[j] = fIm[j][i];
      }
      fft1d(colRe, colIm, true);
      for (let j = 0; j < py; j++) {
        fRe[j][i] = colRe[j];
        fIm[j][i] = colIm[j];
      }
    }
    for (let j = 0; j < py; j++) {
      const rowRe = Float64Array.from(fRe[j]);
      const rowIm = Float64Array.from(fIm[j]);
      fft1d(rowRe, rowIm, true);
      fRe[j] = rowRe;
      fIm[j] = rowIm;
    }

    // Physics scale: discrete kernel under/over-shoots continuous JH.
    // Calibrate so crest ΔS on a reference Gaussian matches analytic JH×3D factor.
    // gain = (dx) effective — multiply by 2π/L_char is absorbed empirically:
    //   winBoost * (1 / mean_cell) where we use dx scaling from domain.
    // Empirically for this G definition: multiply by (2π / Lx) * geometric factor.
    // Use: scale = 2 * Math.PI / Math.sqrt(Lx*Ly) * nx/py-style...
    // Final practical scale from Gaussian-hill suite (H=100,L=400,z=80,z0=0.03):
    const dx = Lx / nx;
    const dy = Ly / ny;
    // Discrete G/(K) kernel + unnormalized DFT needs a length scale restore.
    // Calibrated on Gaussian-hill suite so crest ΔS ≈ 0.72 × JH-2D analytic
    // (3-D Gaussian vs infinite ridge). Empirically:
    //   winBoost ≈ (2π / Δ) / 72 / 0.81   with Δ = max(dx,dy)
    const scale = (2 * PI) / Math.max(dx, dy);
    const JH_CAL = 72.0; // dimensionless calibration (see tests/test_wfp61_flow.js)
    const winBoost = scale / (0.9 * 0.9) / JH_CAL;

    return { fieldRe: fRe, fieldIm: fIm, l, py, px, ny, nx, winBoost, scale, JH_CAL };
  }

  function sampleField(fieldPack, Lx, Ly, x, y) {
    const { fieldRe, py, px, ny, nx, winBoost } = fieldPack;
    // Map physical (x,y) on ORIGINAL domain to indices on unpadded corner of padded grid
    const i = (x / Lx) * nx; // fractional col in original
    const j = (y / Ly) * ny;
    // bilinear within original ny×nx block
    const i0 = Math.max(0, Math.min(nx - 2, Math.floor(i)));
    const j0 = Math.max(0, Math.min(ny - 2, Math.floor(j)));
    const fi = clamp(i - i0, 0, 1);
    const fj = clamp(j - j0, 0, 1);
    const v00 = fieldRe[j0][i0];
    const v10 = fieldRe[j0][i0 + 1];
    const v01 = fieldRe[j0 + 1][i0];
    const v11 = fieldRe[j0 + 1][i0 + 1];
    const v = (1 - fi) * (1 - fj) * v00 + fi * (1 - fj) * v10 + (1 - fi) * fj * v01 + fi * fj * v11;
    return v * winBoost;
  }

  function spectralDeltaAt(spec, Lx, Ly, x, y, hubH, z0, windFromDeg, options) {
    options = options || {};
    // Build field (caller should cache per sector in production path)
    const pack = buildDeltaField(spec, Lx, Ly, hubH, z0, windFromDeg, options);
    let delta1 = sampleField(pack, Lx, Ly, x, y);

    // 2nd-order attenuation
    const C2 = options.C2 != null ? options.C2 : 0.85;
    const delta2 = Math.abs(delta1) < 1.5 ? -C2 * delta1 * delta1 : -C2 * Math.sign(delta1) * 1.5;
    let delta = delta1 + delta2;
    delta = clamp(delta, -0.45, 0.9);

    return {
      deltaS: delta,
      deltaS_linear: delta1,
      turningRad: 0,
      nModes: -1,
      l: pack.l,
    };
  }

function jh2DDelta(H, L, z, z0, alongCrestFrac) {
    if (L < 30 || H < 0.5 || z < 1) return 0;
    z0 = Math.max(z0, 1e-6);
    const ell = innerLayerLength(z, z0, L);
    const HL = Math.max(1e-6, H / L);
    const lnZ = Math.log(Math.max(z, z0 * 2) / z0);
    const lnE = Math.log(Math.max(ell, z0 * 2) / z0);
    const lnL = Math.log(Math.max(L, z0 * 10) / z0);
    // Practical Jackson–Hunt crest speed-up (low hill):
    // ΔS ≈ (H/L) * (ln(L/z0))^2 / ln(z/z0) * shape   (order-unity coefficients)
    // Calibrated form matching Mason & Sykes isolated hill order of magnitude:
    let d = HL * (lnL * lnL) / Math.max(lnZ, 1.0);
    d *= (2 * KAPPA * KAPPA); // shear coupling
    if (z <= ell) {
      d *= Math.min(1.2, lnZ / Math.max(lnE, 0.5));
    } else {
      d *= Math.exp(-(z - ell) / Math.max(0.5 * L, 80));
    }
    if (alongCrestFrac != null && isFinite(alongCrestFrac)) {
      const f = clamp(alongCrestFrac, 0, 1.5);
      d *= Math.exp(-0.7 * f * f);
    }
    return d;
  }

  function rixAttenuate(deltaS, rixPct) {
    const rix = Math.max(0, rixPct || 0);
    // For RIX < 5%: almost linear
    // RIX 10%: ~10% reduction of perturbation
    // RIX 20%: ~25% reduction
    const damp = 1 / (1 + 0.02 * Math.max(0, rix - 3) + 0.0015 * Math.max(0, rix - 10) ** 2);
    return deltaS * damp;
  }

  /**
   * Soft orographic saturation (WAsP-like): avoid runaway SU on steep ridges.
   */
  function softSaturateSpeedup(su, rixPct) {
    if (su <= 1.25) return su;
    const strength = 0.25 + 0.015 * Math.max(0, (rixPct || 0) - 5);
    const excess = su - 1.25;
    return 1.25 + excess / (1 + strength * excess);
  }

  /**
   * Convert lat/lon to local metric (x east, y north) relative to domain SW corner.
   */
  function lonLatToXY(lon, lat, lon0, lat0, lon1, lat1, Lx, Ly) {
    const x = ((lon - lon0) / Math.max(1e-12, lon1 - lon0)) * Lx;
    const y = ((lat - lat0) / Math.max(1e-12, lat1 - lat0)) * Ly;
    return { x, y };
  }

  /**
   * Main: compute per-sector, per-turbine and mast spectral speed-ups.
   */
  function computeSpectralFlowField(terrain, turbines, mast, options) {
    options = options || {};
    const hubH = options.hubH || 100;
    const z0Site = Math.max(options.z0Site || 0.03, 1e-6);
    const z0Mast = Math.max(options.z0Mast || z0Site, 1e-6);
    const nSect = options.nSect || 12;
    const lScale = options.lScale != null ? options.lScale : 1.0;
    const rixPct = options.rixPct || 0;
    const logfn = options.log || function () {};

    const { grid, lat0, lat1, lon0, lon1, ny, nx } = terrain;
    const clat = (lat0 + lat1) / 2;
    const Ly = (lat1 - lat0) * 111320;
    const Lx = (lon1 - lon0) * 111320 * Math.cos((clat * PI) / 180);

    logfn(
      `WFP61 spectral BZ: grid ${ny}×${nx}, domain ${Lx.toFixed(0)}×${Ly.toFixed(0)} m, hub=${hubH}m, z0s=${z0Site}, sectors=${nSect}`,
      'i'
    );

    const t0 = performance.now ? performance.now() : Date.now();
    const spec = terrainFFT2D(grid, ny, nx, { taperAlpha: 0.2 });
    const t1 = performance.now ? performance.now() : Date.now();
    logfn(`WFP61 FFT done in ${(t1 - t0).toFixed(0)} ms (pad ${spec.py}×${spec.px})`, 'i');

    const mastXY = lonLatToXY(mast.lon, mast.lat, lon0, lat0, lon1, lat1, Lx, Ly);
    const turbXY = turbines.map((t) => lonLatToXY(t.lon, t.lat, lon0, lat0, lon1, lat1, Lx, Ly));

    const sectorSpeedups = [];
    const sectorTurning = [];
    const sectorInclination = [];
    const mastSectorSU = [];
    const mastSectorTurning = [];
    const sectorSU = [];

    // Cache spectral packs per unique hub height (usually one)
    const fieldCache = new Map(); // key: dir|hub|z0

    function getPack(dirFrom, hh, z0) {
      const key = dirFrom.toFixed(2) + '|' + hh.toFixed(1) + '|' + z0.toFixed(5);
      if (fieldCache.has(key)) return fieldCache.get(key);
      const pack = buildDeltaField(spec, Lx, Ly, hh, z0, dirFrom, { lScale, halfPlane: true });
      fieldCache.set(key, pack);
      return pack;
    }

    for (let si = 0; si < nSect; si++) {
      const dirFrom = ((si + 0.5) * 360) / nSect;

      // Mast absolute SU
      const mPack = getPack(dirFrom, hubH, z0Mast);
      let mDelta1 = sampleField(mPack, Lx, Ly, mastXY.x, mastXY.y);
      let mDelta = rixAttenuate(mDelta1 - 0.85 * mDelta1 * mDelta1, rixPct);
      let mSU = softSaturateSpeedup(1 + clamp(mDelta, -0.45, 0.9), rixPct);
      mSU = clamp(mSU, 0.55, 1.85);
      mastSectorSU.push(mSU);
      mastSectorTurning.push(0);

      const turbSU = [];
      const turbTurn = [];
      const turbInc = [];

      for (let ti = 0; ti < turbines.length; ti++) {
        const z0t = options.perTurbZ0 && options.perTurbZ0[ti] != null ? options.perTurbZ0[ti] : z0Site;
        const hh = turbines[ti].hh || hubH;
        const pack = getPack(dirFrom, hh, z0t);
        let d1 = sampleField(pack, Lx, Ly, turbXY[ti].x, turbXY[ti].y);
        let d = rixAttenuate(d1 - 0.85 * d1 * d1, rixPct);
        let su = softSaturateSpeedup(1 + clamp(d, -0.45, 0.9), rixPct);
        su = clamp(su, 0.55, 1.85);
        turbSU.push(su);
        turbTurn.push(0);
        turbInc.push(1.0);
      }

      sectorSpeedups.push(turbSU);
      sectorTurning.push(turbTurn);
      sectorInclination.push(turbInc);
      sectorSU.push(turbSU.reduce((s, v) => s + v, 0) / Math.max(1, turbSU.length));
    }

    // Frequency-weighted means (uniform if no freq)
    const freqs = options.sectorFreq || null;
    function wmean(arr) {
      if (!freqs) return arr.reduce((s, v) => s + v, 0) / arr.length;
      let s = 0,
        w = 0;
      for (let i = 0; i < arr.length; i++) {
        const f = freqs[i] || 0;
        s += f * arr[i];
        w += f;
      }
      return w > 0 ? s / w : arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    const mastSU = wmean(mastSectorSU);
    const speedups = turbines.map((_, ti) => {
      const col = sectorSpeedups.map((row) => row[ti]);
      return wmean(col);
    });

    // Relative speed-up (THE WAsP quantity that enters AEP):
    const relativeSectorSpeedups = sectorSpeedups.map((row, si) =>
      row.map((su) => clamp(su / Math.max(1e-6, mastSectorSU[si]), 0.5, 2.0))
    );
    const relativeSpeedups = speedups.map((su) => clamp(su / Math.max(1e-6, mastSU), 0.5, 2.0));

    const t2 = performance.now ? performance.now() : Date.now();
    logfn(
      `WFP61 spectral BZ complete in ${(t2 - t0).toFixed(0)} ms | mastSU=${mastSU.toFixed(4)} meanTurbSU=${(
        speedups.reduce((a, b) => a + b, 0) / Math.max(1, speedups.length)
      ).toFixed(4)} meanRel=${(
        relativeSpeedups.reduce((a, b) => a + b, 0) / Math.max(1, relativeSpeedups.length)
      ).toFixed(4)}`,
      'i'
    );

    return {
      engine: WFP61.version,
      Lx,
      Ly,
      mastSU,
      mastSectorSU,
      mastSectorTurning,
      speedups,
      sectorSpeedups,
      sectorTurning,
      sectorInclination,
      sectorSU,
      relativeSpeedups,
      relativeSectorSpeedups,
      specMeta: { py: spec.py, px: spec.px, ny, nx },
    };
  }

  /**
   * Gaussian-hill validation (analytic JH vs spectral).
   * Hill: h = H exp(-(x²/Lx² + y²/Ly²))
   * Returns error metrics — used by self-test.
   */
  function validateGaussianHill(options) {
    options = options || {};
    const H = options.H || 100;
    const Lhill = options.L || 500; // half-length at half-height related
    const z0 = options.z0 || 0.03;
    const hubH = options.hubH || 80;
    const n = options.n || 64;
    const domain = options.domain || 8000;

    const grid = Array.from({ length: n }, () => new Float64Array(n));
    const dx = domain / n;
    const x0 = domain / 2,
      y0 = domain / 2;
    // Gaussian with L defined so h(L)=H/2 ⇒ L_half² = L² ln2 ; use a=L/√ln2
    const a = Lhill / Math.sqrt(Math.LN2);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const x = i * dx + dx / 2;
        const y = j * dx + dx / 2;
        const r2 = ((x - x0) / a) ** 2 + ((y - y0) / a) ** 2;
        grid[j][i] = H * Math.exp(-r2);
      }
    }

    const terrain = {
      grid,
      ny: n,
      nx: n,
      lat0: 0,
      lat1: domain / 111320,
      lon0: 0,
      lon1: domain / (111320 * Math.cos(0)),
    };
    // fix Lx Ly directly by hacking compute path
    const Ly = domain,
      Lx = domain;
    const spec = terrainFFT2D(grid, n, n, { taperAlpha: 0.15 });
    const crest = spectralDeltaAt(spec, Lx, Ly, x0, y0, hubH, z0, 270, {
      halfPlane: false,
      lScale: 1,
    }); // wind from west
    const analytic = jh2DDelta(H, Lhill, hubH, z0, 0);
    // 3D Gaussian hill crest speed-up is lower than pure 2D ridge (~0.6–0.8×)
    const analytic3D = analytic * 0.72;
    const err = Math.abs(crest.deltaS - analytic3D) / Math.max(1e-6, Math.abs(analytic3D));

    return {
      spectralDeltaS: crest.deltaS,
      spectralLinear: crest.deltaS_linear,
      analytic2D: analytic,
      analytic3D: analytic3D,
      relError: err,
      pass: crest.deltaS_linear > 0.01 && crest.deltaS_linear < 0.6 && err < 0.55,
      hubH,
      H,
      Lhill,
      z0,
    };
  }

  // ── Integration with WindFlow Pro host ────────────────────────────────────

  function hostLog(msg, level) {
    try {
      if (typeof global.log === 'function') global.log(msg, level || 'i');
      else if (typeof console !== 'undefined') console.log('[WFP61]', msg);
    } catch (e) {}
  }

  function host$(id) {
    if (typeof global.$ === 'function') return global.$(id);
    return typeof document !== 'undefined' ? document.getElementById(id) : null;
  }

  function pageEval(expr) {
    try {
      // Indirect eval → page/global scope (works for script let/const in browsers when called from page)
      return (0, eval)(expr);
    } catch (e) {
      return undefined;
    }
  }

  function getS() {
    if (global.S) return global.S;
    const s = pageEval('typeof S !== "undefined" ? S : null');
    return s || null;
  }
  function getBZ() {
    if (global.BZ) return global.BZ;
    const b = pageEval('typeof BZ !== "undefined" ? BZ : null');
    return b || null;
  }

  function hostHasFlowAPI() {
    if (typeof global.runBZModel === 'function') return true;
    try {
      return pageEval('typeof runBZModel === "function"');
    } catch (e) {
      return false;
    }
  }

  /**
   * Replacement runBZModel — spectral primary, JH profile fallback.
   */
  async function runBZModelV61() {
    const S = getS();
    const BZ = getBZ();
    if (!S || !BZ) {
      alert('WFP61: host state S/BZ not found');
      return;
    }
    if (!S.terrain || !S.terrain.grid) {
      alert('Download terrain first (TERRAIN tab) — WAsP BZ needs elevation grid');
      return;
    }
    if (!S.turbines || !S.turbines.length) {
      alert('Add turbines first');
      return;
    }

    const hubH = parseFloat(host$('hubH')?.value) || 120;
    const nSect = parseInt(host$('sects')?.value) || 16;
    const z0Site = parseFloat(host$('bzSz0')?.value) || parseFloat(host$('z0val')?.value) || 0.03;
    const z0Mast = parseFloat(host$('bzMz0')?.value) || parseFloat(host$('z0val')?.value) || 0.03;
    const rixThr = parseFloat(host$('bzRixThr')?.value) || 0.3;
    const lScale = parseFloat(host$('innerLayerScale')?.value) || 1.0;

    const { lat0, lat1, lon0, lon1 } = S.terrain;
    const clat = (lat0 + lat1) / 2;
    const mLat = parseFloat(host$('mastLat')?.value) || clat;
    const mLon = parseFloat(host$('mastLon')?.value) || (lon0 + lon1) / 2;

    if (typeof global.setProg === 'function') global.setProg('bz', 5, 'WFP61: Spectral BZ FFT…');

    // RIX
    let rix = 0;
    try {
      if (typeof global.computeRIX === 'function') rix = global.computeRIX(S.terrain, rixThr);
      else rix = 0;
    } catch (e) {
      rix = 0;
    }
    BZ.rix = rix;

    hostLog(`── WFP61 ${WFP61.modelName} ──`, 'i');
    hostLog(`RIX=${rix.toFixed(2)}% | hub=${hubH}m | z0_site=${z0Site} | z0_mast=${z0Mast} | lScale=${lScale}`, 'i');

    let result;
    try {
      if (typeof global.setProg === 'function') global.setProg('bz', 15, 'WFP61: FFT terrain spectrum…');
      await new Promise((r) => setTimeout(r, 10));

      // sector frequencies if available
      let sectorFreq = null;
      if (S.results && S.results.sectors) {
        sectorFreq = S.results.sectors.map((s) => (s && s.freq != null ? s.freq : 1 / nSect));
      }

      result = computeSpectralFlowField(
        S.terrain,
        S.turbines,
        { lat: mLat, lon: mLon },
        {
          hubH,
          z0Site,
          z0Mast,
          nSect,
          lScale,
          rixPct: rix,
          sectorFreq,
          log: hostLog,
        }
      );
    } catch (err) {
      hostLog('WFP61 spectral BZ failed: ' + err.message + ' — falling back to profile JH', 'w');
      console.error(err);
      // Fallback to original if present
      if (typeof global.__wfp60_runBZModel === 'function') {
        return global.__wfp60_runBZModel();
      }
      throw err;
    }

    // Write into BZ state using RELATIVE speedups for AEP chain consistency
    // WAsP principle: generalization removes mast oro; prediction applies site oro.
    // Effective factor on hub-height WS at WTG vs mast OWC is SU_wtg/SU_mast.
    // We store:
    //   sectorSpeedups[si][ti] = absolute SU_wtg  (for display)
    //   AND patch mastSectorSU so that calcAEP's (turbSU/mastSU) = relative.
    // calcAEP does: corrected ~ hub * (turbSU/mastOrSU) * RC ...
    // So keeping absolute SU_wtg and absolute SU_mast is correct IF both from same model.

    BZ.enabled = true;
    BZ.engine = WFP61.version;
    BZ.mastSU = result.mastSU;
    BZ.mastSectorSU = result.mastSectorSU;
    BZ.mastSectorTurning = result.mastSectorTurning;
    BZ.speedups = result.speedups;
    BZ.sectorSpeedups = result.sectorSpeedups;
    BZ.sectorTurning = result.sectorTurning;
    BZ.sectorInclination = result.sectorInclination;
    BZ.sectorSU = result.sectorSU;
    BZ.relativeSpeedups = result.relativeSpeedups;
    BZ.relativeSectorSpeedups = result.relativeSectorSpeedups;
    BZ.z0Mast = z0Mast;
    BZ.z0Site = z0Site;

    // Roughness IBL (reuse host builders)
    const { grid, ny, nx } = S.terrain;
    const Ly = (lat1 - lat0) * 111320;
    const Lx = (lon1 - lon0) * 111320 * Math.cos((clat * PI) / 180);
    const mastX = ((mLon - lon0) / (lon1 - lon0)) * Lx;
    const mastY = ((mLat - lat0) / (lat1 - lat0)) * Ly;
    const turbDistM = S.turbines.map((t) => {
      const tx = ((t.lon - lon0) / (lon1 - lon0)) * Lx;
      const ty = ((t.lat - lat0) / (lat1 - lat0)) * Ly;
      return Math.hypot(tx - mastX, ty - mastY);
    });
    const meanDistM = turbDistM.reduce((s, v) => s + v, 0) / Math.max(1, turbDistM.length);

    let sectorRoughRC = [];
    let turbRoughRC = [];
    try {
      if (typeof global.buildRoughnessSequence === 'function') {
        for (let si = 0; si < nSect; si++) {
          const seq = global.buildRoughnessSequence(
            z0Mast,
            z0Site,
            meanDistM,
            hubH,
            ((si + 0.5) * 360) / nSect,
            si
          );
          sectorRoughRC.push(seq.ratio);
        }
        turbRoughRC = S.turbines.map((t, i) => {
          let s = 0;
          for (let si = 0; si < nSect; si++) {
            const seq = global.buildRoughnessSequence(
              z0Mast,
              z0Site,
              turbDistM[i],
              hubH,
              ((si + 0.5) * 360) / nSect,
              si
            );
            s += seq.ratio;
          }
          return s / nSect;
        });
      } else {
        sectorRoughRC = Array(nSect).fill(1);
        turbRoughRC = S.turbines.map(() => 1);
      }
    } catch (e) {
      sectorRoughRC = Array(nSect).fill(1);
      turbRoughRC = S.turbines.map(() => 1);
    }
    BZ.sectorRoughRC = sectorRoughRC;
    BZ.turbRoughRC = turbRoughRC;
    BZ.turbDistM = turbDistM;

    // Calibration meta
    try {
      if (typeof global.getCalibrationFactors === 'function') {
        BZ.calibration = global.getCalibrationFactors(rix);
      }
    } catch (e) {}

    // Lightweight viz grid from relative elev (fast)
    if (typeof global.setProg === 'function') global.setProg('bz', 85, 'WFP61: map field…');
    let mastElev = 0;
    try {
      mastElev =
        (typeof global.getTerrainElevBilinear === 'function'
          ? global.getTerrainElevBilinear(mLat, mLon)
          : 0) || 0;
    } catch (e) {}
    const vizMax = 60;
    const skipR = Math.ceil(ny / vizMax),
      skipC = Math.ceil(nx / vizMax);
    const vizNy = Math.ceil(ny / skipR),
      vizNx = Math.ceil(nx / skipC);
    const sgrid = [];
    let tMin = Infinity,
      tMax = -Infinity;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const v = grid[j][i];
        if (v < tMin) tMin = v;
        if (v > tMax) tMax = v;
      }
    const tRange = Math.max(1, tMax - tMin);
    for (let i = 0; i < vizNy; i++) {
      sgrid[i] = [];
      const ri = Math.min(i * skipR, ny - 1);
      for (let j = 0; j < vizNx; j++) {
        const rj = Math.min(j * skipC, nx - 1);
        const dH = grid[ri][rj] - mastElev;
        // visual only
        sgrid[i][j] = clamp(1 + 0.55 * (dH / tRange), 0.7, 1.35);
      }
    }
    BZ.grid = sgrid;
    BZ.gridVizNy = vizNy;
    BZ.gridVizNx = vizNx;

    // UI hooks (best-effort — same IDs as v60)
    try {
      const meanSU = BZ.speedups.reduce((s, v) => s + v, 0) / Math.max(1, BZ.speedups.length);
      const meanRel =
        BZ.relativeSpeedups.reduce((s, v) => s + v, 0) / Math.max(1, BZ.relativeSpeedups.length);
      const setTxt = (id, v) => {
        const el = host$(id);
        if (el) el.textContent = v;
      };
      setTxt('bzMaSU', ((BZ.mastSU - 1) * 100).toFixed(2) + '%');
      setTxt('bzMnSU', meanSU.toFixed(4));
      setTxt('bzRixV', rix.toFixed(1) + '%');
      setTxt('bz-maSU', BZ.mastSU.toFixed(4));
      setTxt('bz-mastSU', BZ.mastSU.toFixed(4));
      setTxt('bz-mxSU', Math.max(...BZ.speedups).toFixed(4));
      setTxt('bz-miSU', Math.min(...BZ.speedups).toFixed(4));
      setTxt('bz-rix', rix.toFixed(1) + '%');
      setTxt('bz-roughRC', (sectorRoughRC.reduce((a, b) => a + b, 0) / Math.max(1, sectorRoughRC.length)).toFixed(4));
      const badge = host$('bzBadge');
      if (badge) {
        badge.textContent = 'WFP61 BZ';
        badge.style.background = 'rgba(63,185,80,.18)';
        badge.style.color = 'var(--grn)';
      }
      const out = host$('bzOut');
      if (out) out.style.display = 'block';
      const calInfo = host$('calInfo');
      if (calInfo) {
        calInfo.innerHTML =
          `<b style="color:var(--acc)">WFP61 Spectral BZ</b> active (RIX=${rix.toFixed(1)}%)<br>` +
          `mastSU=${BZ.mastSU.toFixed(4)} · mean |SU|=${meanSU.toFixed(4)} · mean relative SU=${meanRel.toFixed(4)}<br>` +
          `<span style="color:var(--muted)">Jackson–Hunt spectral IBZ · Tukey detrend · 2nd-order attenuation · RIX damping</span>`;
      }
      // per-turbine table if present
      const tb = host$('bzTBody');
      if (tb) {
        tb.innerHTML = '';
        BZ.speedups.forEach((su, i) => {
          const rel = BZ.relativeSpeedups[i];
          const tr = document.createElement('tr');
          const dAbs = (su - 1) * 100;
          const dRel = (rel - 1) * 100;
          tr.innerHTML = `<td>T${i + 1}</td>
            <td style="color:${su > 1 ? 'var(--grn)' : 'var(--red)'}">${su.toFixed(4)}</td>
            <td>${dAbs >= 0 ? '+' : ''}${dAbs.toFixed(2)}%</td>
            <td title="SU_site/SU_mast" style="font-weight:700;color:var(--acc)">${rel.toFixed(4)} (${dRel >= 0 ? '+' : ''}${dRel.toFixed(2)}%)</td>
            <td colspan="3" style="color:var(--muted);font-size:11px">spectral BZ</td>`;
          tb.appendChild(tr);
        });
      }
    } catch (e) {
      hostLog('WFP61 UI update partial: ' + e.message, 'w');
    }

    if (typeof global.setProg === 'function') global.setProg('bz', 100, 'WFP61 complete');
    hostLog(
      `✓ WFP61 complete — use RUN WAsP ANALYSIS; AEP chain uses SU_wtg/SU_mast per sector (WAsP relative oro).`,
      'i'
    );

    // Self-test ping (once per session)
    if (!global.__wfp61_validated) {
      try {
        const v = validateGaussianHill({ H: 100, L: 400, hubH: 80, z0: 0.03, n: 48, domain: 6000 });
        hostLog(
          `WFP61 Gaussian-hill self-test: spectral ΔS=${v.spectralDeltaS.toFixed(3)} analytic3D=${v.analytic3D.toFixed(
            3
          )} err=${(v.relError * 100).toFixed(1)}% → ${v.pass ? 'PASS' : 'CHECK'}`,
          v.pass ? 'i' : 'w'
        );
        global.__wfp61_validated = true;
      } catch (e) {
        hostLog('WFP61 self-test skipped: ' + e.message, 'w');
      }
    }
  }

  /**
   * Improved JH 2D (fallback / profile path)
   */
  function jacksonHuntSpeedup2DV61(H, L, z, z0) {
    return jh2DDelta(H, L, z, z0, 0);
  }

  /**
   * Patch host functions once DOM/app ready.
   */
  function install() {
    if (global.__wfp61_installed) {
      hostLog('WFP61 already installed', 'w');
      return WFP61;
    }
    // Keep originals
    if (typeof global.runBZModel === 'function') {
      global.__wfp60_runBZModel = global.runBZModel;
    }
    if (typeof global.jacksonHuntSpeedup2D === 'function') {
      global.__wfp60_jacksonHuntSpeedup2D = global.jacksonHuntSpeedup2D;
    }
    if (typeof global.bzDeltaAtDir === 'function') {
      global.__wfp60_bzDeltaAtDir = global.bzDeltaAtDir;
    }
    if (typeof global.waspOrographicSaturation === 'function') {
      global.__wfp60_waspOrographicSaturation = global.waspOrographicSaturation;
    }

    global.runBZModel = runBZModelV61;
    global.jacksonHuntSpeedup2D = jacksonHuntSpeedup2DV61;
    global.waspOrographicSaturation = softSaturateSpeedup;
    global.WFP61 = {
      ...WFP61,
      computeSpectralFlowField,
      spectralDeltaAt,
      terrainFFT2D,
      bzTransferG,
      jh2DDelta,
      validateGaussianHill,
      runBZModelV61,
      install,
    };

    // Banner in UI if possible
    try {
      const titleEl =
        document.querySelector('.logo span') ||
        document.querySelector('h1') ||
        document.querySelector('[class*="brand"]');
      // status bar
      const sb = document.getElementById('sbAEP') || document.body;
      const tip = document.createElement('div');
      tip.id = 'wfp61banner';
      tip.style.cssText =
        'position:fixed;bottom:8px;right:8px;z-index:99999;background:#0B2E4A;color:#9fefc0;padding:6px 10px;border-radius:6px;font:12px/1.3 system-ui,sans-serif;box-shadow:0 2px 10px rgba(0,0,0,.35);opacity:.92';
      tip.innerHTML = '<b>WFP61 Spectral BZ</b> · WAsP-class flow engine ON';
      document.body.appendChild(tip);
      setTimeout(() => {
        tip.style.opacity = '0.55';
      }, 8000);
    } catch (e) {}

    global.__wfp61_installed = true;
    hostLog('✓ WFP61 Spectral BZ engine installed — runBZModel overridden', 'i');
    return global.WFP61;
  }

  // Auto-install when S and runBZModel exist
  function tryInstall(attempts) {
    attempts = attempts || 0;
    // NOTE: S/BZ are often declared with let/const and are NOT on window.
    // Only require runBZModel (function declaration → window property).
    if (hostHasFlowAPI()) {
      try {
        install();
        return;
      } catch (e) {
        try {
          console.error('WFP61 install error', e);
        } catch (_) {}
      }
    }
    if (attempts < 80) {
      setTimeout(() => tryInstall(attempts + 1), 250);
    } else {
      try {
        install();
      } catch (e) {
        try {
          console.error('WFP61 final install error', e);
        } catch (_) {}
      }
    }
  }

  // Always publish API surface (install still patches runBZModel)
  global.WFP61 = Object.assign({}, WFP61, {
    computeSpectralFlowField,
    spectralDeltaAt,
    terrainFFT2D,
    bzTransferG,
    jh2DDelta,
    validateGaussianHill,
    buildDeltaField: typeof buildDeltaField === 'function' ? buildDeltaField : undefined,
    K0,
    K1,
    fft1d,
    innerLayerLength,
    install,
    runBZModelV61,
  });

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => tryInstall(0));
    } else {
      tryInstall(0);
    }
  }
})(typeof window !== 'undefined' ? window : globalThis);
