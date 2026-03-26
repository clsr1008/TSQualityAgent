"""
Pattern-structure detection tools:
  - trend_classifier
  - seasonality_detector
  - change_point_detector
  - pattern_consistency_indicators
  - stationarity_test
  - autocorr
  - rolling_amplitude
  - cycle_amplitude
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

    # Per-segment trend clarity: split at change points, measure R² within each segment.
    # A series with clear trends in every segment is high quality,
    # regardless of whether the direction changes between segments.
    cp_result = change_point_detector(arr)
    breakpoints = [0] + cp_result["change_point_indices"] + [n]
    seg_r2s = []
    for i in range(len(breakpoints) - 1):
        seg = arr[breakpoints[i]:breakpoints[i + 1]]
        if len(seg) < 5:
            continue
        sx = np.arange(len(seg), dtype=float)
        s_slope, s_intercept = np.polyfit(sx, seg, 1)
        s_fitted = s_slope * sx + s_intercept
        ss_res = np.sum((seg - s_fitted) ** 2)
        ss_tot = np.sum((seg - np.mean(seg)) ** 2)
        seg_r2s.append(max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)

    if not seg_r2s:
        segment_clarity = "unclear"
    else:
        mean_seg_r2 = float(np.mean(seg_r2s))
        if mean_seg_r2 >= 0.6:
            segment_clarity = "clear"      # each segment has a well-defined trend
        elif mean_seg_r2 >= 0.25:
            segment_clarity = "moderate"   # partial trend structure within segments
        else:
            segment_clarity = "unclear"    # segments are mostly flat or noisy

    return {
        "direction": direction,
        "slope": round(float(slope), 8),
        "trend_strength": round(float(max(r2, 0.0)), 4),
        "segment_clarity": segment_clarity,   # "clear" | "moderate" | "unclear"
        "segment_count": len(seg_r2s),
    }


# ── Frequency / Seasonality ───────────────────────────────────────────────────

def seasonality_detector(series: Series, max_period: int = None) -> dict:
    """
    Detect dominant seasonal period using autocorrelation with prominence-based
    peak filtering and harmonic suppression.

    Parameters
    ----------
    max_period : int | None
        Maximum lag to search. Defaults to len(series) // 2.

    Returns
    -------
    {
        "dominant_period": int | None,
        "seasonal_strength": float,   # ACF value at dominant period (0–1)
        "top_periods": list[int],     # top-3 non-harmonic candidate periods
        "dominance_ratio": float,     # peak1_strength / peak2_strength; high = one freq dominates
        "peak_count": int,            # number of significant non-harmonic peaks found
    }
    """
    from scipy.signal import find_peaks

    arr = _fill_nan(_to_array(series))
    n = len(arr)
    if max_period is None:
        max_period = n // 2
    max_period = min(max_period, n - 1)

    _empty = {"dominant_period": None, "seasonal_strength": 0.0,
              "top_periods": [], "dominance_ratio": float("nan"), "peak_count": 0}

    if max_period < 2:
        return _empty

    arr_norm = arr - np.mean(arr)
    var = float(np.var(arr_norm))
    if var == 0:
        return _empty

    # Compute ACF for all lags
    acf_vals = np.array([
        float(np.mean(arr_norm[lag:] * arr_norm[:-lag])) / var
        for lag in range(1, max_period + 1)
    ])

    # Prominence-based peak detection: filters out shallow noise peaks
    peak_indices, props = find_peaks(acf_vals, height=0.05, prominence=0.05)
    if len(peak_indices) == 0:
        return _empty

    # Sort by ACF value descending
    order = np.argsort(-acf_vals[peak_indices])
    sorted_lags = [(int(peak_indices[i]) + 1, float(acf_vals[peak_indices[i]])) for i in order]

    # Harmonic suppression: keep a period only if it is not a near-integer multiple
    # of an already-accepted stronger period (±10% tolerance)
    accepted = []
    for lag, strength in sorted_lags:
        is_harmonic = any(
            abs(lag / base - round(lag / base)) < 0.10 and round(lag / base) >= 2
            for base, _ in accepted
        )
        if not is_harmonic:
            accepted.append((lag, strength))

    if not accepted:
        return _empty

    dominant_period = accepted[0][0]
    seasonal_strength = round(accepted[0][1], 4)
    top_periods = [p for p, _ in accepted[:3]]

    # Dominance ratio: how much stronger is the top peak vs the second
    if len(accepted) >= 2 and accepted[1][1] > 0:
        dominance_ratio = round(accepted[0][1] / accepted[1][1], 3)
    else:
        dominance_ratio = float("inf") if len(accepted) == 1 else float("nan")

    return {
        "dominant_period": dominant_period,
        "seasonal_strength": seasonal_strength,
        "top_periods": top_periods,
        "dominance_ratio": dominance_ratio,
        "peak_count": len(accepted),
    }


# ── Amplitude / Spikes ────────────────────────────────────────────────────────


# ── Change Points ─────────────────────────────────────────────────────────────

def change_point_detector(series: Series, penalty: float = None, n_cp: int = None) -> dict:
    """
    Detect structural change points using ruptures PELT algorithm (l2 model).
    Falls back to CUSUM if ruptures is not installed.

    Parameters
    ----------
    penalty : float | None
        Penalty value for PELT (controls sensitivity). Default: 3 * log(n).
    n_cp : int | None
        If provided, use Binseg to find exactly this many change points.

    Returns
    -------
    {
        "change_point_count": int,
        "change_point_indices": list[int],
        "method": str,
    }
    """
    arr = _fill_nan(_to_array(series))
    n = len(arr)
    if n < 4:
        return {"change_point_count": 0, "change_point_indices": [], "method": "none"}

    try:
        import ruptures as rpt
        signal = arr.reshape(-1, 1)
        if n_cp is not None:
            algo = rpt.Binseg(model="l2").fit(signal)
            breakpoints = algo.predict(n_bkps=n_cp)
        else:
            pen = penalty if penalty is not None else 3 * np.log(n)
            algo = rpt.Pelt(model="l2").fit(signal)
            breakpoints = algo.predict(pen=pen)
        # ruptures returns indices of the END of each segment; last entry == n
        cps = sorted([int(b) for b in breakpoints if b < n])
        return {
            "change_point_count": len(cps),
            "change_point_indices": cps,
            "method": "ruptures_pelt" if n_cp is None else "ruptures_binseg",
        }
    except ImportError:
        # Fallback: CUSUM
        mean = np.mean(arr)
        cusum = np.cumsum(arr - mean)
        cusum_abs = np.abs(cusum)
        threshold = np.std(cusum_abs) * 1.5
        cps = [
            int(i) for i in range(1, n - 1)
            if cusum_abs[i] > cusum_abs[i - 1]
            and cusum_abs[i] > cusum_abs[i + 1]
            and cusum_abs[i] > threshold
        ]
        return {"change_point_count": len(cps), "change_point_indices": cps, "method": "cusum_fallback"}


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
        "lumpiness":          float,  # variance of per-window variances — higher = choppier variance
        "flat_ratio":         float,  # fraction of steps with negligible change (< 0.1*std)
        "longest_flat_ratio": float,  # longest consecutive flat run / n
        "crossing_points":    int,
        "crossing_rate":      float,
        "roughness":          float,
    }
    """
    arr = _fill_nan(_to_array(series))
    n = len(arr)

    # Lumpiness: variance of variances across non-overlapping windows
    w = max(2, n // 10)
    windows = [arr[i:i + w] for i in range(0, n - w + 1, w)]
    var_list = [np.var(seg) for seg in windows if len(seg) == w]
    lumpiness = float(np.var(var_list)) if len(var_list) > 1 else 0.0

    # Flat spots: steps where |x[i+1] - x[i]| < 0.1 * std are "flat"
    # flat_ratio: fraction of all steps that are flat
    # longest_flat_ratio: longest consecutive flat run / n
    arr_std = float(np.std(arr))
    flat_threshold = 0.1 * arr_std if arr_std > 0 else 0.0
    steps = np.abs(np.diff(arr))
    is_flat = steps < flat_threshold
    flat_ratio = round(float(np.mean(is_flat)), 4) if len(is_flat) > 0 else 0.0

    max_run = 0
    cur_run = 0
    for f in is_flat:
        if f:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    longest_flat_ratio = round(max_run / n, 4)

    # Crossing points
    mean = np.mean(arr)
    crossings = int(np.sum(np.diff(np.sign(arr - mean)) != 0))
    crossing_rate = round(crossings / (n - 1), 4) if n > 1 else 0.0

    # Roughness: normalised total variation (mean absolute step size)
    roughness = round(float(np.mean(np.abs(np.diff(arr)))), 6) if n > 1 else 0.0

    return {
        "lumpiness":          round(lumpiness, 6),
        "flat_ratio":         flat_ratio,
        "longest_flat_ratio": longest_flat_ratio,
        "crossing_points":    crossings,
        "crossing_rate":      crossing_rate,
        "roughness":          roughness,
    }


# ── Stationarity ──────────────────────────────────────────────────────────────

def stationarity_test(series: Series, test: str = "adf") -> dict:
    """
    Test whether the series is stationary using ADF or KPSS.

    Parameters
    ----------
    test : str   "adf" (Augmented Dickey-Fuller) or "kpss".

    Returns
    -------
    {
        "test": str,
        "statistic": float,
        "p_value": float,
        "is_stationary": bool,
    }
    """
    arr = _fill_nan(_to_array(series))
    valid = arr[~np.isnan(arr)]
    if len(valid) < 10:
        return {"test": test, "statistic": float("nan"), "p_value": float("nan"), "is_stationary": None}

    try:
        if test == "kpss":
            from statsmodels.tsa.stattools import kpss
            stat, p_value, _, _ = kpss(valid, regression="c", nlags="auto")
            is_stationary = bool(p_value > 0.05)
        else:
            from statsmodels.tsa.stattools import adfuller
            stat, p_value, _, _, _, _ = adfuller(valid, autolag="AIC")
            is_stationary = bool(p_value < 0.05)
        return {
            "test": test,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "is_stationary": is_stationary,
        }
    except Exception as e:
        return {"test": test, "statistic": float("nan"), "p_value": float("nan"), "is_stationary": None, "error": str(e)}


# ── Autocorrelation ───────────────────────────────────────────────────────────

def autocorr(series: Series, lag: int = 1) -> dict:
    """
    Compute autocorrelation at a specific lag.

    Parameters
    ----------
    lag : int   Lag value (must be > 0 and < len(series)).

    Returns
    -------
    {
        "lag": int,
        "autocorrelation": float,   # -1 to 1
    }
    """
    arr = _fill_nan(_to_array(series))
    n = len(arr)
    if lag <= 0 or lag >= n:
        return {"lag": lag, "autocorrelation": float("nan"), "error": f"lag must be between 1 and {n - 1}"}

    corr = float(np.corrcoef(arr[:-lag], arr[lag:])[0, 1])
    return {
        "lag": lag,
        "autocorrelation": round(corr, 6),
    }


# ── Rolling Amplitude ─────────────────────────────────────────────────────────

def rolling_amplitude(series: Series, window: int = 20) -> dict:
    """
    General-purpose amplitude measure for any series (periodic or not).
    Computes the local range (max - min) within a sliding window, producing
    an "instantaneous amplitude" curve, then summarises it.

    Works on non-periodic series where cycle_amplitude is unreliable.
    For periodic series, use window ≈ half the cycle period for best results.

    Returns
    -------
    {
        "window":           int,
        "mean_local_range": float,  # average local swing — larger = more active amplitude
        "cv_local_range":   float,  # std/mean of local ranges — lower = more consistent amplitude
        "max_local_range":  float,  # peak local swing (captures bursts)
        "min_local_range":  float,  # quietest local window
    }
    """
    arr = _fill_nan(_to_array(series))
    valid = arr[~np.isnan(arr)]
    if len(valid) < window:
        return {"window": window, "mean_local_range": float("nan"),
                "cv_local_range": float("nan"), "max_local_range": float("nan"),
                "min_local_range": float("nan")}

    local_ranges = np.array([
        float(np.max(valid[i:i + window]) - np.min(valid[i:i + window]))
        for i in range(len(valid) - window + 1)
    ])

    mean_r = float(np.mean(local_ranges))
    std_r  = float(np.std(local_ranges))
    cv     = std_r / mean_r if mean_r > 0 else float("nan")

    return {
        "window":           window,
        "mean_local_range": round(mean_r, 6),
        "cv_local_range":   round(cv, 4),
        "max_local_range":  round(float(np.max(local_ranges)), 6),
        "min_local_range":  round(float(np.min(local_ranges)), 6),
    }


# ── Cycle Amplitude ───────────────────────────────────────────────────────────

def cycle_amplitude(series: Series) -> dict:
    """
    Measure oscillation amplitude consistency by analysing peak-trough pairs.

    Finds local maxima (peaks) and minima (troughs), then computes the
    magnitude of each adjacent peak-trough pair.  A high-quality amplitude
    signal has large and consistent cycle magnitudes (low coefficient of
    variation).

    Gate: returns oscillatory=False with NaN metrics if the series does not have
    at least 2 significant peaks and 2 significant troughs (prominence > 0.3*std).
    This prevents noise bumps or trend series from producing spurious results.

    Returns
    -------
    {
        "oscillatory":    bool,   # False if series lacks clear oscillation — other fields unreliable
        "cycle_count":    int,    # number of complete peak-trough pairs found
        "mean_amplitude": float,  # average peak-to-trough magnitude
        "amplitude_cv":   float,  # std/mean of cycle magnitudes (lower = more consistent)
        "amplitude_trend": str,   # "growing" | "shrinking" | "stable" (amplitude modulation)
        "peak_count":     int,
        "trough_count":   int,
    }
    """
    from scipy.signal import find_peaks

    arr = _fill_nan(_to_array(series))
    n = len(arr)

    _not_oscillatory = {
        "oscillatory": False,
        "cycle_count": 0, "mean_amplitude": float("nan"),
        "amplitude_cv": float("nan"), "amplitude_trend": "unknown",
        "peak_count": 0, "trough_count": 0,
    }

    if n < 6:
        return _not_oscillatory

    # Gate: require at least 2 significant peaks and 2 significant troughs
    # (prominence > 0.3 * std filters out noise bumps)
    sig_threshold = 0.3 * float(np.std(arr))
    sig_peaks,   _ = find_peaks( arr, prominence=sig_threshold)
    sig_troughs, _ = find_peaks(-arr, prominence=sig_threshold)
    if len(sig_peaks) < 2 or len(sig_troughs) < 2:
        return _not_oscillatory

    # Full peak/trough detection for pairing (relaxed, no prominence filter)
    peaks, _   = find_peaks(arr,  distance=2)
    troughs, _ = find_peaks(-arr, distance=2)

    if len(peaks) == 0 or len(troughs) == 0:
        return {"cycle_count": 0, "mean_amplitude": float(np.max(arr) - np.min(arr)),
                "amplitude_cv": float("nan"), "amplitude_trend": "stable",
                "peak_count": int(len(peaks)), "trough_count": int(len(troughs))}

    # Pair each peak with the nearest subsequent trough (or preceding trough)
    magnitudes = []
    all_extrema = sorted(
        [(int(i), float(arr[i]), "peak") for i in peaks] +
        [(int(i), float(arr[i]), "trough") for i in troughs],
        key=lambda x: x[0]
    )
    for k in range(len(all_extrema) - 1):
        a, b = all_extrema[k], all_extrema[k + 1]
        if a[2] != b[2]:   # adjacent extrema of different types
            magnitudes.append(abs(a[1] - b[1]))

    if not magnitudes:
        return {"cycle_count": 0, "mean_amplitude": float("nan"),
                "amplitude_cv": float("nan"), "amplitude_trend": "stable",
                "peak_count": int(len(peaks)), "trough_count": int(len(troughs))}

    mean_amp = float(np.mean(magnitudes))
    std_amp  = float(np.std(magnitudes))
    cv       = std_amp / mean_amp if mean_amp > 0 else float("nan")

    # Amplitude modulation: compare first half vs second half of magnitudes
    half = len(magnitudes) // 2
    if half >= 2:
        first_half_mean  = float(np.mean(magnitudes[:half]))
        second_half_mean = float(np.mean(magnitudes[half:]))
        ratio = second_half_mean / first_half_mean if first_half_mean > 0 else 1.0
        amp_trend = "growing" if ratio > 1.2 else ("shrinking" if ratio < 0.8 else "stable")
    else:
        amp_trend = "stable"

    return {
        "oscillatory":    True,
        "cycle_count":    len(magnitudes),
        "mean_amplitude": round(mean_amp, 6),
        "amplitude_cv":   round(cv, 4),
        "amplitude_trend": amp_trend,
        "peak_count":     int(len(peaks)),
        "trough_count":   int(len(troughs)),
    }


