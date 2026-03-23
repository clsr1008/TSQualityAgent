"""
Pattern-structure detection tools:
  - trend_classifier
  - seasonality_detector
  - spike_detector
  - change_point_detector
  - pattern_consistency_indicators
"""
import numpy as np
from typing import Union

Series = Union[list, np.ndarray]


def _to_array(series: Series) -> np.ndarray:
    return np.array(series, dtype=float)


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation to fill NaNs (needed for decomposition)."""
    if not np.any(np.isnan(arr)):
        return arr
    x = np.arange(len(arr))
    valid = ~np.isnan(arr)
    return np.interp(x, x[valid], arr[valid])


# ── Trend ─────────────────────────────────────────────────────────────────────

def trend_classifier(series: Series, window: int = None) -> dict:
    """
    Classify trend direction and estimate trend strength via linear regression.

    Parameters
    ----------
    window : int | None
        If provided, use only the last `window` points.

    Returns
    -------
    {
        "direction": "increasing" | "decreasing" | "flat",
        "slope": float,
        "trend_strength": float,   # R² of linear fit, 0–1
        "slope_per_step": float,
    }
    """
    arr = _fill_nan(_to_array(series))
    if window:
        arr = arr[-window:]
    n = len(arr)
    if n < 2:
        return {"direction": "flat", "slope": 0.0, "trend_strength": 0.0, "slope_per_step": 0.0}

    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, arr, 1)
    fitted = slope * x + intercept
    ss_res = np.sum((arr - fitted) ** 2)
    ss_tot = np.sum((arr - np.mean(arr)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if abs(slope) < 1e-8 or r2 < 0.05:
        direction = "flat"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return {
        "direction": direction,
        "slope": round(float(slope), 8),
        "trend_strength": round(float(max(r2, 0.0)), 4),
        "slope_per_step": round(float(slope), 8),
    }


# ── Frequency / Seasonality ───────────────────────────────────────────────────

def seasonality_detector(series: Series, max_period: int = None) -> dict:
    """
    Detect dominant seasonal period using autocorrelation.

    Parameters
    ----------
    max_period : int | None
        Maximum lag to search. Defaults to len(series) // 2.

    Returns
    -------
    {
        "dominant_period": int | None,
        "seasonal_strength": float,    # peak autocorrelation at dominant period
        "top_periods": list[int],       # top-3 candidate periods
    }
    """
    arr = _fill_nan(_to_array(series))
    n = len(arr)
    if max_period is None:
        max_period = n // 2
    max_period = min(max_period, n - 1)

    if max_period < 2:
        return {"dominant_period": None, "seasonal_strength": 0.0, "top_periods": []}

    arr_norm = arr - np.mean(arr)
    var = np.var(arr_norm)
    if var == 0:
        return {"dominant_period": None, "seasonal_strength": 0.0, "top_periods": []}

    acf = []
    for lag in range(1, max_period + 1):
        c = float(np.mean(arr_norm[lag:] * arr_norm[:-lag])) / var
        acf.append((lag, c))

    # Find peaks (local maxima in positive territory)
    acf_vals = np.array([c for _, c in acf])
    peaks = []
    for i in range(1, len(acf_vals) - 1):
        if acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1] and acf_vals[i] > 0:
            peaks.append((acf[i][0], acf_vals[i]))

    if not peaks:
        return {"dominant_period": None, "seasonal_strength": 0.0, "top_periods": []}

    peaks.sort(key=lambda x: -x[1])
    dominant_period = peaks[0][0]
    seasonal_strength = round(float(peaks[0][1]), 4)
    top_periods = [p for p, _ in peaks[:3]]

    return {
        "dominant_period": dominant_period,
        "seasonal_strength": seasonal_strength,
        "top_periods": top_periods,
    }


# ── Amplitude / Spikes ────────────────────────────────────────────────────────

def spike_detector(series: Series, threshold: float = 3.0, min_sep: int = 1) -> dict:
    """
    Detect spikes (large amplitude excursions) using Z-score threshold.

    Parameters
    ----------
    threshold  : Z-score threshold for a point to be a spike.
    min_sep    : Minimum separation between detected spikes.

    Returns
    -------
    {
        "spike_count": int,
        "spike_ratio": float,
        "spike_indices": list[int],
        "amplitude_range": float,   # max - min of valid values
        "spike_mean_magnitude": float,
    }
    """
    arr = _to_array(series)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 3:
        return {
            "spike_count": 0, "spike_ratio": 0.0,
            "spike_indices": [], "amplitude_range": float("nan"),
            "spike_mean_magnitude": float("nan"),
        }

    mean, std = np.mean(valid), np.std(valid)
    amp_range = float(np.max(valid) - np.min(valid))

    if std == 0:
        return {
            "spike_count": 0, "spike_ratio": 0.0,
            "spike_indices": [], "amplitude_range": amp_range,
            "spike_mean_magnitude": 0.0,
        }

    z = np.abs((arr - mean) / std)
    candidate_idx = list(np.where((z > threshold) & ~np.isnan(arr))[0])

    # Apply min_sep suppression
    spikes = []
    last = -min_sep - 1
    for idx in candidate_idx:
        if idx - last >= min_sep:
            spikes.append(int(idx))
            last = idx

    magnitudes = [abs(float(arr[i]) - mean) for i in spikes]
    mean_mag = float(np.mean(magnitudes)) if magnitudes else 0.0

    return {
        "spike_count": len(spikes),
        "spike_ratio": round(len(spikes) / len(arr), 4),
        "spike_indices": spikes,
        "amplitude_range": round(amp_range, 6),
        "spike_mean_magnitude": round(mean_mag, 6),
    }


# ── Change Points ─────────────────────────────────────────────────────────────

def change_point_detector(series: Series) -> dict:
    """
    Detect structural change points using CUSUM (cumulative sum) method.

    Returns
    -------
    {
        "change_point_count": int,
        "change_point_indices": list[int],
        "max_cusum": float,
    }
    """
    arr = _fill_nan(_to_array(series))
    n = len(arr)
    if n < 4:
        return {"change_point_count": 0, "change_point_indices": [], "max_cusum": 0.0}

    mean = np.mean(arr)
    cusum = np.cumsum(arr - mean)
    cusum_abs = np.abs(cusum)

    # Detect peaks in |CUSUM|
    threshold = np.std(cusum_abs) * 1.5
    cps = []
    for i in range(1, n - 1):
        if (cusum_abs[i] > cusum_abs[i - 1] and
                cusum_abs[i] > cusum_abs[i + 1] and
                cusum_abs[i] > threshold):
            cps.append(int(i))

    return {
        "change_point_count": len(cps),
        "change_point_indices": cps,
        "max_cusum": round(float(np.max(cusum_abs)), 6),
    }


# ── Pattern Consistency ───────────────────────────────────────────────────────

def pattern_consistency_indicators(series: Series) -> dict:
    """
    Compute consistency indicators:
      - lumpiness      : variance of squared differences in rolling variance
      - flat_spots     : longest run of identical (rounded) values / series length
      - crossing_points: number of times series crosses its mean / series length

    Returns
    -------
    {
        "lumpiness": float,
        "flat_spots": float,
        "crossing_points": int,
        "crossing_rate": float,
    }
    """
    arr = _fill_nan(_to_array(series))
    n = len(arr)

    # Lumpiness: variance of variances across non-overlapping windows
    w = max(2, n // 10)
    windows = [arr[i:i + w] for i in range(0, n - w + 1, w)]
    var_list = [np.var(seg) for seg in windows if len(seg) == w]
    lumpiness = float(np.var(var_list)) if len(var_list) > 1 else 0.0

    # Flat spots: longest run of values that round to the same integer
    rounded = np.round(arr).astype(int)
    max_run = 1
    cur_run = 1
    for i in range(1, n):
        if rounded[i] == rounded[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    flat_spots = round(max_run / n, 4)

    # Crossing points
    mean = np.mean(arr)
    crossings = int(np.sum(np.diff(np.sign(arr - mean)) != 0))
    crossing_rate = round(crossings / (n - 1), 4) if n > 1 else 0.0

    return {
        "lumpiness": round(lumpiness, 6),
        "flat_spots": flat_spots,
        "crossing_points": crossings,
        "crossing_rate": crossing_rate,
    }
