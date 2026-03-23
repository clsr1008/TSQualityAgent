"""
Rare-pattern detection tools:
  - anomaly_detection
  - outlier_density
"""
import numpy as np
from typing import Union

Series = Union[list, np.ndarray]


def _to_array(series: Series) -> np.ndarray:
    return np.array(series, dtype=float)


def anomaly_detection(series: Series, anomaly_threshold: float = 3.0) -> dict:
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
