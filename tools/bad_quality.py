"""
Bad-quality detection tools:
  - missing_ratio
  - noise_profile
  - signal_to_noise_ratio
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
    {
        "missing_ratio": float,      # 0–1
        "missing_count": int,
        "total_count": int,
    }
    """
    arr = _to_array(series)
    total = len(arr)
    missing = int(np.sum(np.isnan(arr)))
    ratio = missing / total if total > 0 else 0.0
    return {
        "missing_ratio": round(ratio, 4),
        "missing_count": missing,
        "total_count": total,
    }


def noise_profile(series: Series, window: int = 5) -> dict:
    """
    Estimate noise level using a rolling-window residual approach.
    Noise = std of (series - rolling_mean).

    Parameters
    ----------
    window : int
        Rolling window size for smoothing.

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
    return {
        "noise_std": round(noise_std, 6),
        "signal_std": round(signal_std, 6),
        "noise_ratio": round(noise_ratio, 4),
    }


def signal_to_noise_ratio(series: Series) -> dict:
    """
    SNR = mean / std (a simple proxy for SNR in 1-D time series).

    Returns
    -------
    {
        "snr": float,       # higher = cleaner signal
        "mean": float,
        "std": float,
    }
    """
    arr = _to_array(series)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"snr": float("nan"), "mean": float("nan"), "std": float("nan")}
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    snr = abs(mean) / std if std > 0 else float("inf")
    return {
        "snr": round(snr, 4),
        "mean": round(mean, 6),
        "std": round(std, 6),
    }
