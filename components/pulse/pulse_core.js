/*
 * pulse_core.js — fingertip camera photoplethysmography (PPG) → heart rate & HRV.
 *
 * Pure functions, no DOM: the same file runs in the browser component and in
 * node for testing (module.exports at the bottom).
 *
 * Pipeline (per recording):
 *   1. frame means   : mean red (and green) of a small central patch, with the
 *                      frame's real timestamp (requestVideoFrameCallback when
 *                      available — camera frame timing jitters by several ms)
 *   2. resample      : linear interpolation onto an even 30 Hz grid
 *   3. band-pass     : 2nd-order Butterworth high-pass 0.6 Hz + low-pass 4 Hz,
 *                      run forward and backward (zero phase, like filtfilt)
 *   4. invert        : more blood under the finger = less light, so each beat
 *                      is a dip in brightness; inverting makes beats peaks
 *   5. spectral HR   : Hann-windowed DFT scan 40–200 bpm → dominant rate
 *                      (robust anchor for the beat detector and a quality check)
 *   6. beats         : local maxima with a refractory distance derived from the
 *                      spectral HR, amplitude threshold, then parabolic
 *                      interpolation for sub-sample (≈ms) peak timing
 *   7. intervals     : inter-beat intervals (IBI); reject physiologically
 *                      impossible ones (<300 ms, >2000 ms) and ectopic/artefact
 *                      ones deviating >20 % from the local median (5 beats)
 *   8. metrics       : HR = 60000 / mean(IBI); RMSSD from successive pairs of
 *                      accepted intervals only; SDNN; lnRMSSD
 *   9. quality       : share of accepted beats, agreement between spectral and
 *                      beat-based HR, spectral peak prominence → good/fair/poor
 */
(function (root) {
  "use strict";

  var FS = 30; // Hz, resampling rate
  var HRV_MIN_SHAPE = 0.97; // calibrated on synthetic recordings — see tests

  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; }
  function std(a) { var m = mean(a), s = 0; for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m); return a.length > 1 ? Math.sqrt(s / (a.length - 1)) : 0; }
  function mad(a) { var m = median(a); return median(a.map(function (v) { return Math.abs(v - m); })) * 1.4826; }
  /** Clip motion spikes so one bump can't dominate the spectrum or thresholds. */
  function winsorize(x, k) { var s = mad(x) || std(x), m = median(x), lo = m - k * s, hi = m + k * s;
    return x.map(function (v) { return v < lo ? lo : (v > hi ? hi : v); }); }
  function median(a) { if (!a.length) return 0; var b = a.slice().sort(function (x, y) { return x - y; }); var n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; }

  /** Is the fingertip covering the lens? (bright red, little green/blue, even) */
  function fingerOn(meanR, meanG, meanB, spatialStd) {
    return meanR > 60 && meanR > meanG * 1.6 && meanR > meanB * 1.6 && (spatialStd === undefined || spatialStd < 40);
  }

  /** Linear resampling of (t ms, v) onto an even grid. */
  function resample(ts, vs, fs) {
    fs = fs || FS;
    if (ts.length < 2) return { t0: ts[0] || 0, v: vs.slice() };
    var dt = 1000 / fs, t0 = ts[0], t1 = ts[ts.length - 1];
    var n = Math.floor((t1 - t0) / dt) + 1, out = new Array(n), j = 0;
    for (var i = 0; i < n; i++) {
      var t = t0 + i * dt;
      while (j < ts.length - 2 && ts[j + 1] < t) j++;
      var ta = ts[j], tb = ts[j + 1], va = vs[j], vb = vs[j + 1];
      out[i] = tb === ta ? va : va + (vb - va) * (t - ta) / (tb - ta);
    }
    return { t0: t0, v: out };
  }

  /** RBJ biquad coefficients (Butterworth Q = 1/sqrt(2)). */
  function biquad(type, fc, fs) {
    var w0 = 2 * Math.PI * fc / fs, cw = Math.cos(w0), sw = Math.sin(w0), Q = Math.SQRT1_2, al = sw / (2 * Q);
    var b0, b1, b2, a0 = 1 + al, a1 = -2 * cw, a2 = 1 - al;
    if (type === "lp") { b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2; }
    else { b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2; }
    return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
  }
  function runBiquad(c, x) {
    var y = new Array(x.length), x1 = x[0], x2 = x[0], y1 = x[0] * (c[0] + c[1] + c[2]) / (1 + c[3] + c[4]), y2 = y1;
    // start in steady state for the first sample to avoid a big transient
    if (!isFinite(y1)) { y1 = 0; y2 = 0; }
    for (var i = 0; i < x.length; i++) {
      var yi = c[0] * x[i] + c[1] * x1 + c[2] * x2 - c[3] * y1 - c[4] * y2;
      x2 = x1; x1 = x[i]; y2 = y1; y1 = yi; y[i] = yi;
    }
    return y;
  }
  function filtfilt(c, x) {
    var y = runBiquad(c, x).reverse();
    return runBiquad(c, y).reverse();
  }
  function bandpass(x, fs, lo, hi) {
    fs = fs || FS; lo = lo || 0.6; hi = hi || 4.0;
    var m = mean(x), d = x.map(function (v) { return v - m; });
    return filtfilt(biquad("lp", hi, fs), filtfilt(biquad("hp", lo, fs), d));
  }

  /** Dominant pulse rate (bpm) by scanning a Hann-windowed DFT from 40 to 200 bpm. */
  function spectralHR(x, fs) {
    fs = fs || FS;
    var n = x.length; if (n < fs * 5) return { bpm: 0, prominence: 0 };
    var w = new Array(n);
    for (var i = 0; i < n; i++) w[i] = x[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
    var best = 0, bestP = 0, total = 0, count = 0, powers = [];
    for (var bpm = 40; bpm <= 200; bpm += 0.5) {
      var f = bpm / 60, re = 0, im = 0, k = 2 * Math.PI * f / fs;
      for (var j = 0; j < n; j++) { re += w[j] * Math.cos(k * j); im -= w[j] * Math.sin(k * j); }
      var p = re * re + im * im; powers.push([bpm, p]); total += p; count++;
      if (p > bestP) { bestP = p; best = bpm; }
    }
    // The dicrotic wave puts energy at 2× the pulse rate; if the half-rate
    // peak is nearly as strong, the true pulse is the half rate.
    if (best / 2 >= 40) {
      var half = 0;
      for (var h = 0; h < powers.length; h++) if (Math.abs(powers[h][0] - best / 2) <= 1.5 && powers[h][1] > half) half = powers[h][1];
      if (half >= 0.35 * bestP) { best = best / 2; bestP = half; }
    }
    // prominence: power within ±3 bpm of the peak vs everything else
    var near = 0;
    for (var q = 0; q < powers.length; q++) if (Math.abs(powers[q][0] - best) <= 3) near += powers[q][1];
    return { bpm: best, prominence: total ? near / total : 0 };
  }

  /** Beat detection with sub-sample refinement. Returns beat times in ms (relative to t0). */
  function detectBeats(x, fs, hrGuess) {
    fs = fs || FS;
    var minDist = Math.max(Math.round(fs * 60 / (hrGuess > 0 ? hrGuess : 90) * 0.6), Math.round(fs * 0.3));
    var sd = mad(x) || std(x), thr = 0.3 * sd, peaks = [];
    for (var i = 1; i < x.length - 1; i++) {
      if (x[i] > x[i - 1] && x[i] >= x[i + 1] && x[i] > thr) {
        if (peaks.length && i - peaks[peaks.length - 1] < minDist) {
          if (x[i] > x[peaks[peaks.length - 1]]) peaks[peaks.length - 1] = i; // keep the taller one
        } else peaks.push(i);
      }
    }
    return peaks.map(function (i) {
      var a = x[i - 1], b = x[i], c = x[i + 1], den = a - 2 * b + c;
      var off = den !== 0 ? 0.5 * (a - c) / den : 0;
      if (off > 0.5 || off < -0.5) off = 0;
      return (i + off) * 1000 / fs;
    });
  }

  /**
   * Refine beat timing by cross-correlating each beat with the average beat
   * shape (template). Uses the whole pulse wave instead of one sample at the
   * top, which cuts timing noise several-fold — this is what makes RMSSD
   * usable at 30 frames per second.
   */
  function refineBeats(x, fs, beatsMs) {
    fs = fs || FS;
    var pre = Math.round(0.20 * fs), post = Math.round(0.40 * fs), L = pre + post + 1, maxLag = Math.round(0.08 * fs);
    var idx = beatsMs.map(function (b) { return Math.round(b * fs / 1000); });
    var segs = [];
    idx.forEach(function (i) { if (i - pre - maxLag >= 0 && i + post + maxLag < x.length) segs.push(x.slice(i - pre, i + post + 1)); });
    if (segs.length < 5) { refineBeats.lastCorr = 0; return beatsMs; }
    var tpl = new Array(L).fill(0);
    segs.forEach(function (sg) { for (var k = 0; k < L; k++) tpl[k] += sg[k] / segs.length; });
    var tm = mean(tpl); tpl = tpl.map(function (v) { return v - tm; });
    var tn = Math.sqrt(tpl.reduce(function (a, v) { return a + v * v; }, 0)), corrs = [];
    var out = beatsMs.map(function (b, n) {
      var i = idx[n];
      if (i - pre - maxLag < 0 || i + post + maxLag >= x.length) return b;
      var cc = [];
      for (var lag = -maxLag; lag <= maxLag; lag++) {
        var seg = x.slice(i - pre + lag, i + post + 1 + lag), sm = mean(seg), c = 0;
        for (var k = 0; k < L; k++) c += (seg[k] - sm) * tpl[k];
        cc.push(c);
      }
      var bi = 0; for (var q = 1; q < cc.length; q++) if (cc[q] > cc[bi]) bi = q;
      var bs = x.slice(i - pre + bi - maxLag, i + post + 1 + bi - maxLag), bm = mean(bs);
      var bn = Math.sqrt(bs.reduce(function (a, v) { return a + (v - bm) * (v - bm); }, 0));
      if (bn > 0 && tn > 0) corrs.push(cc[bi] / (bn * tn));
      var off = 0;
      if (bi > 0 && bi < cc.length - 1) { var den = cc[bi - 1] - 2 * cc[bi] + cc[bi + 1]; off = den !== 0 ? 0.5 * (cc[bi - 1] - cc[bi + 1]) / den : 0; if (Math.abs(off) > 0.5) off = 0; }
      return (i + (bi - maxLag) + off) * 1000 / fs;
    });
    // Beat-shape consistency (0–1): how clean each beat is. Timing precision,
    // and therefore HRV, is only trustworthy when this is high.
    refineBeats.lastCorr = corrs.length ? median(corrs) : 0;
    return out;
  }

  /** Inter-beat intervals with artefact rejection. */
  function cleanIntervals(beatsMs) {
    var ibi = [];
    for (var i = 1; i < beatsMs.length; i++) ibi.push(beatsMs[i] - beatsMs[i - 1]);
    var ok = ibi.map(function (v) { return v >= 300 && v <= 2000; });
    for (var k = 0; k < ibi.length; k++) {
      if (!ok[k]) continue;
      var win = [];
      for (var j = Math.max(0, k - 2); j <= Math.min(ibi.length - 1, k + 2); j++) if (j !== k && ibi[j] >= 300 && ibi[j] <= 2000) win.push(ibi[j]);
      var med = median(win.length ? win : [ibi[k]]);
      if (Math.abs(ibi[k] - med) > 0.2 * med) ok[k] = false;
    }
    return { ibi: ibi, ok: ok };
  }

  function hrvMetrics(ibi, ok) {
    var good = ibi.filter(function (_, i) { return ok[i]; });
    var diffs = [];
    for (var i = 1; i < ibi.length; i++) if (ok[i] && ok[i - 1]) diffs.push(ibi[i] - ibi[i - 1]);
    var rmssd = diffs.length ? Math.sqrt(mean(diffs.map(function (d) { return d * d; }))) : 0;
    return {
      hr: good.length ? 60000 / mean(good) : 0,
      rmssd: rmssd, lnrmssd: rmssd > 0 ? Math.log(rmssd) : 0,
      sdnn: std(good), nBeats: ibi.length + 1, nGood: good.length, nPairs: diffs.length,
    };
  }

  /**
   * Full analysis of one recording.
   * frames: [{t: ms, r: meanRed, g: meanGreen, finger: bool}]
   * opts.windows: optional [[startSec, endSec], ...] → HR per window (for recovery)
   */
  function analyse(frames, opts) {
    opts = opts || {};
    var fr = frames.filter(function (f) { return f.finger !== false; });
    if (fr.length < FS * 8) return { ok: false, reason: "too_short" };
    // Red saturates on some phones with the torch on; fall back to green then.
    var useGreen = mean(fr.map(function (f) { return f.r; })) > 245;
    var ts = fr.map(function (f) { return f.t; }), vs = fr.map(function (f) { return useGreen ? f.g : f.r; });
    var rs = resample(ts, vs, FS);
    var filt = bandpass(rs.v, FS).map(function (v) { return -v; });
    // drop 1 s at each edge (filter settling)
    var edge = FS, core = filt.slice(edge, filt.length - edge), coreT0 = rs.t0 + edge * 1000 / FS;
    core = winsorize(core, 4);
    var spec = spectralHR(core, FS);
    var beatsRel = refineBeats(core, FS, detectBeats(core, FS, spec.bpm));
    var beats = beatsRel.map(function (b) { return b + coreT0; });
    var ci = cleanIntervals(beats);
    var m = hrvMetrics(ci.ibi, ci.ok);
    var goodShare = ci.ibi.length ? m.nGood / ci.ibi.length : 0;
    var agree = spec.bpm > 0 && m.hr > 0 ? Math.abs(spec.bpm - m.hr) : 99;
    var shape = refineBeats.lastCorr || 0;
    var quality = (goodShare >= 0.85 && agree <= 5 && spec.prominence >= 0.25) ? "good"
                : (goodShare >= 0.7 && agree <= 8) ? "fair" : "poor";
    var out = {
      ok: quality !== "poor", quality: quality,
      hr: Math.round(m.hr), hrSpectral: Math.round(spec.bpm), rmssd: Math.round(m.rmssd), sdnn: Math.round(m.sdnn),
      lnrmssd: +m.lnrmssd.toFixed(2), beats: m.nBeats, goodShare: +goodShare.toFixed(2),
      prominence: +spec.prominence.toFixed(2), channel: useGreen ? "green" : "red",
      shape: +shape.toFixed(3),
      // HRV needs millisecond-accurate beats: only report it for clean signals.
      hrvReliable: quality === "good" && shape >= HRV_MIN_SHAPE,
      durationS: +((ts[ts.length - 1] - ts[0]) / 1000).toFixed(1),
      ibi: ci.ibi.map(function (v, i) { return [Math.round(beats[i + 1] - ts[0]), Math.round(v), ci.ok[i] ? 1 : 0]; }),
    };
    if (opts.windows) {
      out.windows = opts.windows.map(function (w) {
        var vals = [];
        for (var i = 0; i < ci.ibi.length; i++) {
          var tt = (beats[i + 1] - ts[0]) / 1000;
          if (ci.ok[i] && tt >= w[0] && tt < w[1]) vals.push(ci.ibi[i]);
        }
        return vals.length >= 3 ? Math.round(60000 / median(vals)) : 0;
      });
    }
    if (quality === "poor") out.reason = "noisy";
    return out;
  }

  var api = { analyse: analyse, fingerOn: fingerOn, resample: resample, bandpass: bandpass,
              spectralHR: spectralHR, detectBeats: detectBeats, cleanIntervals: cleanIntervals,
              hrvMetrics: hrvMetrics, refineBeats: refineBeats, FS: FS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PulseCore = api;
})(typeof window !== "undefined" ? window : this);
