"""
Per-dimension defect injection with multiple variants per dimension.

Two severity levels:
  light — subtle, needs tool-based analysis to detect
  heavy — visible from preview/stats, reasoning-only sufficient

Each dimension has multiple injection methods; one is randomly selected.
Parameters are sampled from severity-dependent ranges (not fixed values).
"""
import math
import numpy as np


# ── Variant definitions ──────────────────────────────────────────────────────
# For each dimension, a list of variants.
# Each variant: {method, light: {param: (lo, hi)}, heavy: {param: (lo, hi)}}
# Tuple (lo, hi) → sampled uniformly; int tuple → randint; scalar → fixed.

DEFECT_VARIANTS = {
    # Calibration: all methods always detectable without tools → only_severity="heavy"
    "missing_value": [
        {"method": "random_scatter", "only_severity": "heavy",
         "heavy": {"ratio": (0.02, 0.08)}},
        {"method": "burst", "only_severity": "heavy",
         "heavy": {"n_bursts": (1, 3), "burst_ratio": (0.02, 0.05)}},
        {"method": "periodic", "only_severity": "heavy",
         "heavy": {"ratio": (0.02, 0.08), "gap_len": (1, 3)}},
    ],
    "noise_level": [
        {"method": "gaussian",
         "light": {"multiplier": (1.2, 2.0)},
         "heavy": {"multiplier": (2.5, 4.0)}},
        {"method": "heteroscedastic",
         "light": {"multiplier": (1.2, 2.0), "affected_ratio": (0.1, 0.2)},
         "heavy": {"multiplier": (2.5, 4.0), "affected_ratio": (0.3, 0.7)}},
        {"method": "impulsive",
         "light": {"multiplier": (1.5, 2.0), "burst_ratio": (0.05, 0.10)},
         "heavy": {"multiplier": (3.0, 5.0), "burst_ratio": (0.15, 0.40)}},
    ],
    "rare_pattern": [
        # Calibration: always heavy (detectable without tools) → only used for severity=heavy
        # Use original light params to keep injection magnitude realistic
        {"method": "point_outlier", "only_severity": "heavy",
         "heavy": {"count": (1, 3), "sigma": (2.5, 3.5)}},
        # Calibration: always light (needs tools) → only used for severity=light
        # Use original heavy params to ensure the anomaly is actually present in data
        {"method": "contextual", "only_severity": "light",
         "light": {"count": (1, 2), "sigma": (4.0, 5.0), "duration": (2, 4)}},
        # Calibration: always heavy → only used for severity=heavy
        # Use original light params
        {"method": "level_shift", "only_severity": "heavy",
         "heavy": {"count": (1, 2), "sigma": (1.5, 2.5), "duration": (5, 10)}},
    ],
    # Calibration: all methods always need tools → only_severity="light"
    "trend": [
        {"method": "flatten", "only_severity": "light",
         "light": {"flatten_ratio": (0.15, 0.40), "noise_boost": (1.2, 3.0)}},
        {"method": "drift", "only_severity": "light",
         "light": {"drift_strength": (0.3, 1.5)}},
        {"method": "reversal", "only_severity": "light",
         "light": {"reverse_ratio": (0.15, 0.40)}},
    ],
    "frequency": [
        # competing: n_competing drives light/heavy (n=1 → light, n=2-3 → heavy)
        {"method": "competing",
         "light": {"n_competing": (1, 1), "competing_amplitude": (0.2, 0.5)},
         "heavy": {"n_competing": (2, 3), "competing_amplitude": (0.7, 0.9)}},
        # jitter: always light across full param range; use original heavy values for diversity
        {"method": "jitter", "only_severity": "light",
         "light": {"jitter_cv": (0.08, 0.22)}},
        # period_shift: always light across full param range; use original heavy values for diversity
        {"method": "period_shift", "only_severity": "light",
         "light": {"shift_ratio": (0.20, 0.50), "period_factor": (0.50, 0.80)}},
    ],
    "amplitude": [
        # random_scale: cv drives light/heavy; heavy requires cv≥0.85 (acc=0.900 at cv=1.0)
        {"method": "random_scale",
         "light": {"cv": (0.20, 0.60)},
         "heavy": {"cv": (0.80, 1.10)}},
        # decay: always light across full param range; use original heavy values for diversity
        {"method": "decay", "only_severity": "light",
         "light": {"decay_factor": (0.20, 0.70)}},
        # clip: always light across full param range; use original heavy values for diversity
        {"method": "clip", "only_severity": "light",
         "light": {"clip_ratio": (0.40, 0.80)}},
    ],
    "pattern_consistency": [
        # variance_switching: always heavy across full param range; use original light values for diversity
        {"method": "variance_switching", "only_severity": "heavy",
         "heavy": {"intensity": (0.20, 0.60)}},
        # structural_break: always light across full param range; use original heavy values for diversity
        {"method": "structural_break", "only_severity": "light",
         "light": {"intensity": (0.30, 0.80)}},
        # flat_spots: always light across full param range; use original heavy values for diversity
        {"method": "flat_spots", "only_severity": "light",
         "light": {"intensity": (0.30, 0.80)}},
        # mean_drift: non-monotonic, no reliable heavy threshold; treat as always light
        {"method": "mean_drift", "only_severity": "light",
         "light": {"drift_std": (0.20, 0.50)}},
    ],
}

ALL_DIMENSIONS = list(DEFECT_VARIANTS.keys())


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sample_params(param_ranges: dict, rng: np.random.Generator) -> dict:
    """Sample concrete values from (min, max) range tuples."""
    params = {}
    for k, v in param_ranges.items():
        if isinstance(v, tuple) and len(v) == 2:
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int):
                params[k] = int(rng.integers(lo, hi + 1))
            else:
                params[k] = float(rng.uniform(lo, hi))
        else:
            params[k] = v
    return params


def _est_noise_std(series: np.ndarray) -> float:
    """Estimate noise std from first-differences (NaN-safe)."""
    diffs = np.diff(series)
    return float(np.nanstd(diffs)) / math.sqrt(2)


def _extract_oscillation(series: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Separate baseline (trend) and oscillation via rolling mean (NaN-safe)."""
    n = len(series)
    filled = np.where(np.isnan(series), np.nanmean(series), series)
    if n > period:
        kernel = np.ones(period) / period
        baseline = np.convolve(filled, kernel, mode="same")
    else:
        baseline = np.full(n, np.nanmean(series))
    return baseline, series - baseline


# ── missing_value ────────────────────────────────────────────────────────────

def _inject_missing_scatter(series: np.ndarray, ratio: float,
                            seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Random NaN positions."""
    rng = np.random.default_rng(seed)
    degraded = series.copy().astype(float)
    n = len(degraded)
    n_missing = max(1, int(n * ratio))
    indices = rng.choice(n, size=n_missing, replace=False).tolist()
    degraded[indices] = np.nan
    return degraded, {"method": "random_scatter", "ratio": ratio,
                      "n_missing": n_missing, "indices": sorted(indices)}


def _inject_missing_burst(series: np.ndarray, n_bursts: int, burst_ratio: float,
                          seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Consecutive NaN blocks (sensor dropout)."""
    rng = np.random.default_rng(seed)
    degraded = series.copy().astype(float)
    n = len(degraded)
    total_missing = max(1, int(n * burst_ratio))
    burst_len = max(2, total_missing // max(1, n_bursts))
    all_indices = set()
    for _ in range(n_bursts):
        start = int(rng.integers(0, max(1, n - burst_len)))
        for j in range(start, min(start + burst_len, n)):
            degraded[j] = np.nan
            all_indices.add(j)
    all_indices = sorted(all_indices)
    return degraded, {"method": "burst", "n_bursts": n_bursts, "burst_ratio": burst_ratio,
                      "n_missing": len(all_indices), "indices": all_indices}


def _inject_missing_periodic(series: np.ndarray, ratio: float, gap_len: int,
                             seed: int | None = None) -> tuple[np.ndarray, dict]:
    """NaN at regular intervals (systematic gaps)."""
    rng = np.random.default_rng(seed)
    degraded = series.copy().astype(float)
    n = len(degraded)
    total_missing = max(1, int(n * ratio))
    n_gaps = max(1, total_missing // max(1, gap_len))
    interval = max(gap_len + 1, n // (n_gaps + 1))
    offset = int(rng.integers(0, max(1, interval // 2)))
    all_indices = set()
    pos = offset
    for _ in range(n_gaps):
        if pos >= n:
            break
        for j in range(pos, min(pos + gap_len, n)):
            degraded[j] = np.nan
            all_indices.add(j)
        jitter = int(rng.integers(-max(1, interval // 4), max(1, interval // 4) + 1))
        pos += interval + jitter
    all_indices = sorted(all_indices)
    return degraded, {"method": "periodic", "ratio": ratio, "gap_len": gap_len,
                      "n_missing": len(all_indices), "indices": all_indices}


# ── noise_level ──────────────────────────────────────────────────────────────

def _inject_noise_gaussian(series: np.ndarray, multiplier: float,
                           seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Additive Gaussian noise across the whole series."""
    rng = np.random.default_rng(seed)
    est_std = _est_noise_std(series)
    extra_std = est_std * (multiplier - 1)
    degraded = series + rng.normal(0, extra_std, len(series))
    return degraded, {"method": "gaussian", "multiplier": multiplier,
                      "est_std": round(est_std, 6), "extra_std": round(extra_std, 6)}


def _inject_noise_heteroscedastic(series: np.ndarray, multiplier: float,
                                  affected_ratio: float,
                                  seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Extra noise concentrated in a contiguous segment."""
    rng = np.random.default_rng(seed)
    n = len(series)
    est_std = _est_noise_std(series)
    extra_std = est_std * (multiplier - 1)
    degraded = series.copy()
    seg_len = max(10, int(n * affected_ratio))
    start = int(rng.integers(0, max(1, n - seg_len)))
    end = min(start + seg_len, n)
    degraded[start:end] += rng.normal(0, extra_std, end - start)
    return degraded, {"method": "heteroscedastic", "multiplier": multiplier,
                      "affected_ratio": affected_ratio, "segment": [int(start), int(end)],
                      "est_std": round(est_std, 6), "extra_std": round(extra_std, 6)}


def _inject_noise_impulsive(series: np.ndarray, multiplier: float,
                            burst_ratio: float,
                            seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Short bursts of high noise scattered across the series."""
    rng = np.random.default_rng(seed)
    n = len(series)
    est_std = _est_noise_std(series)
    extra_std = est_std * (multiplier - 1)
    degraded = series.copy()
    n_bursts = max(1, int(n * burst_ratio / 5))
    mask = np.zeros(n, dtype=bool)
    for _ in range(n_bursts):
        blen = int(rng.integers(3, 8))
        start = int(rng.integers(0, max(1, n - blen)))
        mask[start:min(start + blen, n)] = True
    noise = rng.normal(0, extra_std, n)
    degraded[mask] += noise[mask]
    return degraded, {"method": "impulsive", "multiplier": multiplier,
                      "burst_ratio": burst_ratio, "n_affected": int(mask.sum()),
                      "est_std": round(est_std, 6), "extra_std": round(extra_std, 6)}


# ── rare_pattern ─────────────────────────────────────────────────────────────

def _inject_outlier_point(series: np.ndarray, count: int, sigma: float,
                          seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Cat-1 point outliers (sensor-fault-like spikes)."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    std = float(np.nanstd(series))
    indices = sorted(rng.choice(n, size=min(count, n), replace=False).tolist())
    values = []
    for idx in indices:
        spike = rng.choice([-1, 1]) * sigma * std * rng.uniform(0.8, 1.2)
        degraded[idx] += spike
        values.append(round(float(degraded[idx]), 4))
    return degraded, {"method": "point_outlier", "count": count, "sigma": sigma,
                      "indices": indices, "values": values}


def _inject_outlier_contextual(series: np.ndarray, count: int, sigma: float,
                               duration: int,
                               seed: int | None = None) -> tuple[np.ndarray, dict]:
    """V-shaped excursions (brief contextual anomalies)."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    std = float(np.nanstd(series))
    indices = []
    for _ in range(min(count, max(1, n // (duration + 5)))):
        start = int(rng.integers(duration, max(duration + 1, n - duration)))
        magnitude = rng.choice([-1, 1]) * sigma * std * rng.uniform(0.8, 1.2)
        half = duration // 2
        for j in range(duration):
            idx = start + j
            if idx >= n:
                break
            frac = j / half if j <= half else (duration - j) / max(1, duration - half)
            degraded[idx] += magnitude * frac
        indices.append(int(start))
    return degraded, {"method": "contextual", "count": count, "sigma": sigma,
                      "duration": duration, "indices": sorted(indices)}


def _inject_outlier_level_shift(series: np.ndarray, count: int, sigma: float,
                                duration: int,
                                seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Temporary level shifts (step-then-revert)."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    std = float(np.nanstd(series))
    indices = []
    for _ in range(min(count, max(1, n // (duration + 5)))):
        start = int(rng.integers(5, max(6, n - duration - 5)))
        end = min(start + duration, n)
        magnitude = rng.choice([-1, 1]) * sigma * std * rng.uniform(0.8, 1.2)
        degraded[start:end] += magnitude
        indices.append(int(start))
    return degraded, {"method": "level_shift", "count": count, "sigma": sigma,
                      "duration": duration, "indices": sorted(indices)}


# ── trend ────────────────────────────────────────────────────────────────────

def _degrade_trend_flatten(series: np.ndarray, flatten_ratio: float,
                           noise_boost: float,
                           seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Flatten a segment and add local noise."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    seg_len = max(10, int(n * flatten_ratio))
    start = int(rng.integers(int(n * 0.2), max(int(n * 0.2) + 1, int(n * 0.8) - seg_len)))
    end = start + seg_len
    seg_mean = float(np.nanmean(degraded[start:end]))
    est_std = _est_noise_std(series)
    degraded[start:end] = seg_mean + rng.normal(0, est_std * noise_boost, seg_len)
    return degraded, {"method": "flatten", "flatten_ratio": flatten_ratio,
                      "noise_boost": noise_boost, "segment": [int(start), int(end)]}


def _degrade_trend_drift(series: np.ndarray, drift_strength: float,
                         seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Add gradual slope change to a segment."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    start = int(rng.integers(0, n // 3))
    end = int(rng.integers(2 * n // 3, n))
    series_range = float(np.nanmax(series) - np.nanmin(series))
    est_std = _est_noise_std(series)
    drift_mag = drift_strength * max(series_range, est_std * 10) * rng.choice([-1, 1])
    drift = np.linspace(0, drift_mag, end - start)
    degraded[start:end] += drift
    return degraded, {"method": "drift", "drift_strength": drift_strength,
                      "segment": [int(start), int(end)],
                      "drift_magnitude": round(float(drift_mag), 4)}


def _degrade_trend_reversal(series: np.ndarray, reverse_ratio: float,
                            seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Reverse trend direction in a segment."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    seg_len = max(10, int(n * reverse_ratio))
    start = int(rng.integers(int(n * 0.1), max(int(n * 0.1) + 1, int(n * 0.9) - seg_len)))
    end = start + seg_len
    segment = degraded[start:end].copy()
    reversed_seg = segment[::-1]
    reversed_seg += segment[0] - reversed_seg[0]  # level-align at boundary
    degraded[start:end] = reversed_seg
    return degraded, {"method": "reversal", "reverse_ratio": reverse_ratio,
                      "segment": [int(start), int(end)]}


# ── frequency ────────────────────────────────────────────────────────────────

def _degrade_freq_competing(series: np.ndarray, n_competing: int,
                            competing_amplitude: float,
                            base_period: int | None = None,
                            seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Add competing frequency components."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    if base_period is None:
        base_period = n // 8
    added = []
    for _ in range(n_competing):
        while True:
            p = int(rng.integers(max(5, base_period // 3), base_period * 3))
            ratio = p / base_period
            if abs(ratio - round(ratio)) > 0.15:
                break
        amp = competing_amplitude * float(np.nanstd(series)) * rng.uniform(0.7, 1.3)
        phase = rng.uniform(0, 2 * math.pi)
        t = np.arange(n)
        degraded += amp * np.sin(2 * math.pi * t / p + phase)
        added.append({"period": int(p), "amplitude": round(amp, 4)})
    return degraded, {"method": "competing", "n_competing": n_competing,
                      "competing_amplitude": competing_amplitude, "added_periods": added}


def _degrade_freq_jitter(series: np.ndarray, jitter_cv: float,
                         base_period: int | None = None,
                         seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Add phase noise to make periodicity less precise."""
    rng = np.random.default_rng(seed)
    n = len(series)
    if base_period is None:
        base_period = n // 8
    baseline, oscillation = _extract_oscillation(series, base_period)
    # Cumulative phase noise → time warping
    phase_noise = np.cumsum(rng.normal(0, jitter_cv, n))
    k = max(3, base_period // 3)
    phase_noise = np.convolve(phase_noise, np.ones(k) / k, mode="same")
    t_orig = np.arange(n, dtype=float)
    t_warped = np.clip(t_orig + phase_noise * base_period, 0, n - 1)
    warped_osc = np.interp(t_warped, t_orig, oscillation)
    degraded = baseline + warped_osc
    return degraded, {"method": "jitter", "jitter_cv": jitter_cv,
                      "base_period": base_period}


def _degrade_freq_shift(series: np.ndarray, shift_ratio: float,
                        period_factor: float,
                        base_period: int | None = None,
                        seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Change the dominant period in the latter portion of the series."""
    rng = np.random.default_rng(seed)
    n = len(series)
    if base_period is None:
        base_period = n // 8
    baseline, oscillation = _extract_oscillation(series, base_period)
    shift_start = int(n * (1 - shift_ratio))
    t_orig = np.arange(n, dtype=float)
    t_warped = t_orig.copy()
    tail_len = n - shift_start
    t_warped[shift_start:] = shift_start + np.arange(tail_len, dtype=float) * period_factor
    t_warped = np.clip(t_warped, 0, n - 1)
    warped_osc = np.interp(t_warped, t_orig, oscillation)
    degraded = baseline + warped_osc
    return degraded, {"method": "period_shift", "shift_ratio": shift_ratio,
                      "period_factor": period_factor, "base_period": base_period}


# ── amplitude ────────────────────────────────────────────────────────────────

def _degrade_amp_random(series: np.ndarray, cv: float,
                        period: int | None = None,
                        seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Randomize per-cycle amplitude."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    if period is None:
        period = n // 8
    baseline, oscillation = _extract_oscillation(series, period)
    n_cycles = max(1, n // period)
    scale_factors = rng.lognormal(0, cv, n_cycles).clip(0.2, 5.0)
    for i in range(n_cycles):
        s = i * period
        e = min(s + period, n)
        degraded[s:e] = baseline[s:e] + oscillation[s:e] * scale_factors[i]
    return degraded, {"method": "random_scale", "cv": cv, "period": period,
                      "scale_factors": [round(float(f), 3) for f in scale_factors]}


def _degrade_amp_decay(series: np.ndarray, decay_factor: float,
                       period: int | None = None,
                       seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Gradually decay or grow amplitude over time."""
    rng = np.random.default_rng(seed)
    n = len(series)
    if period is None:
        period = n // 8
    baseline, oscillation = _extract_oscillation(series, period)
    direction = int(rng.choice([-1, 1]))
    if direction == 1:
        envelope = np.linspace(1.0, decay_factor, n)
    else:
        envelope = np.linspace(decay_factor, 1.0, n)
    degraded = baseline + oscillation * envelope
    return degraded, {"method": "decay", "decay_factor": decay_factor, "period": period,
                      "direction": "decay" if direction == 1 else "grow"}


def _degrade_amp_clip(series: np.ndarray, clip_ratio: float,
                      period: int | None = None,
                      seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Clip peaks and troughs to reduce effective amplitude."""
    n = len(series)
    if period is None:
        period = n // 8
    baseline, oscillation = _extract_oscillation(series, period)
    osc_max = float(np.nanmax(np.abs(oscillation)))
    if osc_max > 0:
        clip_level = clip_ratio * osc_max
        clipped = np.clip(oscillation, -clip_level, clip_level)
        degraded = baseline + clipped
    else:
        degraded = series.copy()
    return degraded, {"method": "clip", "clip_ratio": clip_ratio, "period": period}


# ── pattern_consistency ──────────────────────────────────────────────────────

def _degrade_consist_variance(series: np.ndarray, intensity: float,
                              seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Alternate between calm and volatile windows."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    est_std = _est_noise_std(series)
    window_size = max(10, n // 10)
    n_windows = n // window_size
    for i in range(n_windows):
        if i % 2 == 1:
            s = i * window_size
            e = min(s + window_size, n)
            degraded[s:e] += rng.normal(0, est_std * (1 + intensity * 4), e - s)
    return degraded, {"method": "variance_switching", "intensity": intensity}


def _degrade_consist_break(series: np.ndarray, intensity: float,
                           seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Insert abrupt level shifts."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    est_std = _est_noise_std(series)
    n_breaks = max(1, int(intensity * 3))
    candidates = list(range(n // 5, 4 * n // 5))
    if len(candidates) < n_breaks:
        n_breaks = max(1, len(candidates))
    bps = sorted(rng.choice(candidates, size=n_breaks, replace=False))
    shifts = rng.uniform(-est_std * 8 * intensity, est_std * 8 * intensity, n_breaks)
    for bp, shift in zip(bps, shifts):
        degraded[bp:] += shift
    return degraded, {"method": "structural_break", "intensity": intensity,
                      "break_points": [int(b) for b in bps]}


def _degrade_consist_flat(series: np.ndarray, intensity: float,
                          seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Insert flat (constant) segments."""
    rng = np.random.default_rng(seed)
    degraded = series.copy()
    n = len(degraded)
    n_spots = max(1, int(intensity * 3))
    spots = []
    for _ in range(n_spots):
        lo = max(5, int(n * 0.05))
        hi = max(lo + 1, int(n * 0.15))
        spot_len = int(rng.integers(lo, hi))
        start = int(rng.integers(0, max(1, n - spot_len)))
        degraded[start:start + spot_len] = degraded[start]
        spots.append([int(start), int(start + spot_len)])
    return degraded, {"method": "flat_spots", "intensity": intensity, "spots": spots}


def _degrade_consist_mean_drift(series: np.ndarray, drift_std: float,
                                seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Add a slow random-walk drift to the mean level (non-stationary mean).

    Detected by stationarity_test (non-stationary) and change_point_detector.
    """
    rng = np.random.default_rng(seed)
    n = len(series)
    signal_std = max(float(np.nanstd(series)), 1e-6)
    # Random walk with step std = drift_std * signal_std
    step_std = drift_std * signal_std
    drift = np.cumsum(rng.normal(0, step_std, n))
    # Smooth to make it a slow drift (not jumpy)
    k = max(5, n // 15)
    drift = np.convolve(drift, np.ones(k) / k, mode="same")
    degraded = series + drift
    return degraded, {"method": "mean_drift", "drift_std": drift_std,
                      "step_std": round(step_std, 6),
                      "drift_range": [round(float(drift.min()), 4),
                                      round(float(drift.max()), 4)]}


# ── Dispatch ─────────────────────────────────────────────────────────────────

_VARIANT_FUNCTIONS = {
    "missing_value": {
        "random_scatter": _inject_missing_scatter,
        "burst":          _inject_missing_burst,
        "periodic":       _inject_missing_periodic,
    },
    "noise_level": {
        "gaussian":        _inject_noise_gaussian,
        "heteroscedastic": _inject_noise_heteroscedastic,
        "impulsive":       _inject_noise_impulsive,
    },
    "rare_pattern": {
        "point_outlier": _inject_outlier_point,
        "contextual":    _inject_outlier_contextual,
        "level_shift":   _inject_outlier_level_shift,
    },
    "trend": {
        "flatten":   _degrade_trend_flatten,
        "drift":     _degrade_trend_drift,
        "reversal":  _degrade_trend_reversal,
    },
    "frequency": {
        "competing":    _degrade_freq_competing,
        "jitter":       _degrade_freq_jitter,
        "period_shift": _degrade_freq_shift,
    },
    "amplitude": {
        "random_scale": _degrade_amp_random,
        "decay":        _degrade_amp_decay,
        "clip":         _degrade_amp_clip,
    },
    "pattern_consistency": {
        "variance_switching": _degrade_consist_variance,
        "structural_break":   _degrade_consist_break,
        "flat_spots":         _degrade_consist_flat,
        "mean_drift":         _degrade_consist_mean_drift,
    },
}


def inject_defect(
    series: np.ndarray,
    dimension: str,
    severity: str = "heavy",
    seed: int | None = None,
    base_period: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Inject a defect for the given dimension at the given severity level.

    Randomly selects a variant, samples parameters from the severity range,
    and applies the injection.

    Parameters
    ----------
    series : clean base series
    dimension : quality dimension name
    severity : "light" | "heavy"
    seed : random seed
    base_period : period hint (used by frequency/amplitude injectors)

    Returns
    -------
    (degraded_series, defect_metadata)
    """
    rng = np.random.default_rng(seed)

    # Pick a random variant (filter by only_severity if specified)
    all_variants = DEFECT_VARIANTS[dimension]
    variants = [v for v in all_variants
                if v.get("only_severity", severity) == severity]
    if not variants:
        variants = all_variants
    variant = variants[int(rng.integers(len(variants)))]
    method = variant["method"]

    # Sample parameters from the severity range
    params = _sample_params(variant[severity], rng)
    params["seed"] = seed

    # Pass period hint where applicable
    if dimension == "frequency" and base_period is not None:
        params["base_period"] = base_period
    if dimension == "amplitude" and base_period is not None:
        params["period"] = base_period

    fn = _VARIANT_FUNCTIONS[dimension][method]
    return fn(series, **params)
