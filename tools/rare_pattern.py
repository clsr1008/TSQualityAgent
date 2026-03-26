"""
Rare-pattern detection tools:
  - zscore_anomaly
  - outlier_density
  - mad_residual_anomaly
  - contextual_anomaly
"""
import numpy as np
from typing import Union

Series = Union[list, np.ndarray]


def _to_array(series: Series) -> np.ndarray:
    return np.array(series, dtype=float)


def zscore_anomaly(series: Series, anomaly_threshold: float = 3.0) -> dict:
    """
    Detect point anomalies using Z-score thresholding.

    Parameters
    ----------
    anomaly_threshold : float
        Number of standard deviations beyond which a point is flagged.

    Returns
    -------
    {
        "anomaly_count": int,
        "anomaly_ratio": float,
        "anomaly_indices": list[int],
        "anomaly_values": list[float],
        "threshold_used": float,
    }
    """
    arr = _to_array(series)
    valid_mask = ~np.isnan(arr)
    valid = arr[valid_mask]
    if len(valid) < 3:
        return {
            "anomaly_count": 0,
            "anomaly_ratio": 0.0,
            "anomaly_indices": [],
            "anomaly_values": [],
            "threshold_used": anomaly_threshold,
        }

    mean = np.mean(valid)
    std = np.std(valid)
    if std == 0:
        return {
            "anomaly_count": 0,
            "anomaly_ratio": 0.0,
            "anomaly_indices": [],
            "anomaly_values": [],
            "threshold_used": anomaly_threshold,
        }

    z_scores = np.abs((arr - mean) / std)
    anomaly_mask = (z_scores > anomaly_threshold) & valid_mask
    indices = list(np.where(anomaly_mask)[0])
    values = [round(float(arr[i]), 6) for i in indices]

    return {
        "anomaly_count": len(indices),
        "anomaly_ratio": round(len(indices) / len(arr), 4),
        "anomaly_indices": indices,
        "anomaly_values": values,
        "threshold_used": anomaly_threshold,
    }


def outlier_density(series: Series) -> dict:
    """
    Estimate the density of outliers using IQR-based method.

    Returns
    -------
    {
        "outlier_count": int,
        "outlier_ratio": float,
        "iqr": float,
        "lower_fence": float,
        "upper_fence": float,
    }
    """
    arr = _to_array(series)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 4:
        return {
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "iqr": float("nan"),
            "lower_fence": float("nan"),
            "upper_fence": float("nan"),
        }

    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (arr < lower) | (arr > upper)
    outlier_count = int(np.sum(outlier_mask & ~np.isnan(arr)))

    return {
        "outlier_count": outlier_count,
        "outlier_ratio": round(outlier_count / len(arr), 4),
        "iqr": round(float(iqr), 6),
        "lower_fence": round(float(lower), 6),
        "upper_fence": round(float(upper), 6),
    }


def mad_residual_anomaly(series: Series, window: int = 15, threshold: float = 3.5) -> dict:
    """
    Robust anomaly detection: detrend via rolling mean, then apply MAD-based scoring
    on residuals. Addresses two Z-score limitations:
      (a) global mean/std is distorted by trend → rolling mean removes local trend first
      (b) outliers inflate std → MAD (median absolute deviation) is outlier-resistant

    Parameters
    ----------
    window    : rolling window size for detrending (default 15)
    threshold : modified Z-score cutoff (default 3.5; Iglewicz & Hoaglin recommend 3.5)

    Returns
    -------
    {
        "anomaly_count": int,
        "anomaly_ratio": float,
        "anomaly_indices": list[int],
        "anomaly_values": list[float],
        "mad": float,               # MAD of residuals — measures typical residual spread
        "threshold_used": float,
    }
    """
    arr = _to_array(series)
    valid_mask = ~np.isnan(arr)
    valid = arr[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    if len(valid) < window + 3:
        return {"anomaly_count": 0, "anomaly_ratio": 0.0,
                "anomaly_indices": [], "anomaly_values": [],
                "mad": float("nan"), "threshold_used": threshold}

    # Rolling mean detrend (causal: window ends at current point)
    kernel = np.ones(window) / window
    smoothed = np.convolve(valid, kernel, mode="valid")   # length: len(valid) - window + 1
    residuals = valid[window - 1:] - smoothed             # same length
    orig_indices = valid_indices[window - 1:]

    # MAD scoring on residuals
    med = np.median(residuals)
    mad = float(np.median(np.abs(residuals - med)))
    if mad == 0:
        mad = float(np.std(residuals)) * 0.6745 + 1e-10

    modified_z = 0.6745 * np.abs(residuals - med) / mad
    anomaly_mask = modified_z > threshold

    indices = [int(orig_indices[i]) for i, flag in enumerate(anomaly_mask) if flag]
    values = [round(float(arr[idx]), 6) for idx in indices]

    return {
        "anomaly_count": len(indices),
        "anomaly_ratio": round(len(indices) / len(arr), 4),
        "anomaly_indices": indices,
        "anomaly_values": values,
        "mad": round(mad, 6),
        "threshold_used": threshold,
    }


def contextual_anomaly(series: Series, context_window: int = 10, threshold: float = 3.0) -> dict:
    """
    Contextual anomaly detection: for each point, fit a linear trend on the preceding
    context_window points and compute the prediction error. Points with unusually large
    errors (scored via MAD normalisation) are flagged as contextual anomalies.

    Unlike global Z-score, this detects points that deviate from their *local* context —
    catching sudden breaks, V-shaped events, or step changes that look normal globally.

    Parameters
    ----------
    context_window : number of preceding points used to build local expectation (default 10)
    threshold      : MAD-normalised error cutoff (default 3.0)

    Returns
    -------
    {
        "anomaly_count": int,
        "anomaly_ratio": float,
        "anomaly_indices": list[int],
        "anomaly_values": list[float],
        "threshold_used": float,
    }
    """
    arr = _to_array(series)
    valid_mask = ~np.isnan(arr)
    valid = arr[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    if len(valid) < context_window + 3:
        return {"anomaly_count": 0, "anomaly_ratio": 0.0,
                "anomaly_indices": [], "anomaly_values": [],
                "threshold_used": threshold}

    # Build prediction errors via linear extrapolation from context window
    errors = []
    check_pos = []
    x_ctx = np.arange(context_window, dtype=float)

    for i in range(context_window, len(valid)):
        ctx = valid[i - context_window: i]
        slope, intercept = np.polyfit(x_ctx, ctx, 1)
        predicted = slope * context_window + intercept
        errors.append(abs(float(valid[i]) - predicted))
        check_pos.append(i)

    errors = np.array(errors)

    # MAD-normalised scoring (errors are non-negative; flag unusually large ones)
    mad_err = float(np.median(np.abs(errors - np.median(errors))))
    if mad_err == 0:
        mad_err = float(np.std(errors)) * 0.6745 + 1e-10

    scores = 0.6745 * errors / mad_err
    anomaly_mask = scores > threshold

    indices = [int(valid_indices[pos]) for pos, flag in zip(check_pos, anomaly_mask) if flag]
    values = [round(float(arr[idx]), 6) for idx in indices]

    return {
        "anomaly_count": len(indices),
        "anomaly_ratio": round(len(indices) / len(arr), 4),
        "anomaly_indices": indices,
        "anomaly_values": values,
        "threshold_used": threshold,
    }
