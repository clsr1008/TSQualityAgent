"""
Bad-quality detection tools:
  - missing_ratio
  - noise_profile
  - volatility
  - range_stats
"""
import numpy as np
from typing import Union

Series = Union[list, np.ndarray]


def _to_array(series: Series) -> np.ndarray:
    return np.array(series, dtype=float)


def missing_ratio(series: Series) -> dict:
    """
    Compute the fraction of missing (NaN) values in the series.

    Returns
    -------
    {"missing_ratio": float}   # 0–1
    """
    arr = _to_array(series)
    total = len(arr)
    missing = int(np.sum(np.isnan(arr)))
    ratio = missing / total if total > 0 else 0.0
    return {"missing_ratio": round(ratio, 4)}


def noise_profile(series: Series, window: int = None) -> dict:
    """
    Estimate noise level using a rolling-window residual approach.
    Noise = std of (series - rolling_mean).

    Parameters
    ----------
    window : int | None
        Rolling window size for smoothing.
        If None, auto-adapts: max(5, n // 50), capped at 50.

    Returns
    -------
    {
        "noise_std": float,
        "signal_std": float,
        "noise_ratio": float,    # noise_std / signal_std (lower = better)
    }
    """
    arr = _to_array(series)
    valid = arr[~np.isnan(arr)]
    if window is None:
        window = max(5, min(len(valid) // 50, 50))
    if len(valid) < window:
        return {"noise_std": float("nan"), "signal_std": float("nan"), "noise_ratio": float("nan")}

    # Rolling mean via convolution
    kernel = np.ones(window) / window
    smoothed = np.convolve(valid, kernel, mode="valid")
    # Align residuals: compare valid[window-1:] with smoothed
    residuals = valid[window - 1:] - smoothed
    noise_std = float(np.std(residuals))
    signal_std = float(np.std(valid))
    noise_ratio = noise_std / signal_std if signal_std > 0 else float("nan")

    # Noise type: red noise has significant lag-1 autocorrelation (|acf1| > 0.3)
    if len(valid) >= 3:
        acf1 = float(np.corrcoef(valid[:-1], valid[1:])[0, 1])
        noise_type = "red" if abs(acf1) > 0.3 else "white"
    else:
        noise_type = "unknown"

    return {
        "noise_std": round(noise_std, 6),
        "signal_std": round(signal_std, 6),
        "noise_ratio": round(noise_ratio, 4),
        "noise_type": noise_type,
        "window_used": window,
    }



def volatility(series: Series, window: int = None) -> dict:
    """
    Compute rolling volatility as the std of first differences within each window.
    Measures local instability / amplitude fluctuation.

    Parameters
    ----------
    window : int | None
        Rolling window size. If None, auto-adapts: max(5, n // 50), capped at 50.

    Returns
    -------
    {
        "window": int,
        "mean_volatility": float,   # average rolling volatility
        "max_volatility": float,    # peak rolling volatility
    }
    """
    arr = _to_array(series)
    valid = arr[~np.isnan(arr)]
    if window is None:
        window = max(5, min(len(valid) // 50, 50))
    if len(valid) < window + 1:
        return {"window": window, "mean_volatility": float("nan"), "max_volatility": float("nan")}

    diffs = np.diff(valid)
    rolling_vol = [
        float(np.std(diffs[i:i + window]))
        for i in range(len(diffs) - window + 1)
    ]
    return {
        "window_used": window,
        "mean_volatility": round(float(np.mean(rolling_vol)), 6),
        "max_volatility": round(float(np.max(rolling_vol)), 6),
    }


def range_stats(series: Series, start: int, end: int, stat: str = "mean") -> dict:
    """
    Compute a statistic over a specific index range [start, end) of the series.
    Useful for analysing a local segment rather than the whole series.

    Parameters
    ----------
    start : int   Start index (inclusive).
    end   : int   End index (exclusive).
    stat  : str   One of: mean, std, max, min, sum.

    Returns
    -------
    {
        "start": int,
        "end": int,
        "stat": str,
        "value": float,
        "segment_length": int,
    }
    """
    arr = _to_array(series)
    n = len(arr)
    start = max(0, start)
    end = min(n, end)
    if start >= end:
        return {"start": start, "end": end, "stat": stat, "value": float("nan"), "segment_length": 0}

    segment = arr[start:end]
    valid = segment[~np.isnan(segment)]
    if len(valid) == 0:
        return {"start": start, "end": end, "stat": stat, "value": float("nan"), "segment_length": int(end - start)}

    fn = {"mean": np.mean, "std": np.std, "max": np.max, "min": np.min, "sum": np.sum}.get(stat, np.mean)
    return {
        "start": start,
        "end": end,
        "stat": stat,
        "value": round(float(fn(valid)), 6),
        "segment_length": int(end - start),
    }
