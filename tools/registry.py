"""
Tool registry for the Inspector agent.

TOOL_REGISTRY : maps function name → callable
TOOL_SCHEMAS  : OpenAI function-calling schemas for all registered tools
"""

from tools.bad_quality import missing_ratio, noise_profile, signal_to_noise_ratio, volatility, range_stats
from tools.rare_pattern import anomaly_detection, outlier_density
from tools.pattern_structure import (
    trend_classifier,
    seasonality_detector,
    spike_detector,
    change_point_detector,
    pattern_consistency_indicators,
    stationarity_test,
    autocorr,
)

TOOL_REGISTRY = {
    "missing_ratio": missing_ratio,
    "noise_profile": noise_profile,
    "signal_to_noise_ratio": signal_to_noise_ratio,
    "volatility": volatility,
    "range_stats": range_stats,
    "anomaly_detection": anomaly_detection,
    "outlier_density": outlier_density,
    "trend_classifier": trend_classifier,
    "seasonality_detector": seasonality_detector,
    "spike_detector": spike_detector,
    "change_point_detector": change_point_detector,
    "pattern_consistency_indicators": pattern_consistency_indicators,
    "stationarity_test": stationarity_test,
    "autocorr": autocorr,
}

# OpenAI-style function schemas for tool calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "missing_ratio",
            "description": "Compute fraction of missing (NaN) values in a series.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "noise_profile",
            "description": "Estimate noise level using rolling-window residuals. Also classifies noise as white (random) or red (autocorrelated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "window": {"type": "integer", "default": 5},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "signal_to_noise_ratio",
            "description": "Compute signal-to-noise ratio (|mean|/std). Higher = cleaner signal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "volatility",
            "description": "Rolling volatility: std of first-differences within a sliding window. Measures local instability — useful for noise_level and amplitude dimensions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "window": {"type": "integer", "default": 5},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "range_stats",
            "description": "Compute a statistic (mean/std/max/min/sum) over a specific index range [start, end) of the series. Useful for analysing local segments or suspected change-point regions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "stat": {"type": "string", "enum": ["mean", "std", "max", "min", "sum"], "default": "mean"},
                },
                "required": ["series_name", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "anomaly_detection",
            "description": "Detect rare point anomalies / outliers using Z-score threshold. Used for the rare_pattern dimension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "anomaly_threshold": {"type": "number", "default": 3.0},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "outlier_density",
            "description": "Estimate outlier density using IQR fences. Robust alternative to Z-score for non-normal distributions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trend_classifier",
            "description": "Classify trend direction (increasing/decreasing/flat) and strength via linear regression (R²).",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "window": {"type": "integer"},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "seasonality_detector",
            "description": "Detect dominant seasonal period and strength via autocorrelation peak analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "max_period": {"type": "integer"},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "autocorr",
            "description": "Compute autocorrelation at a specific lag. Useful for frequency dimension: strong autocorrelation at lag k confirms periodicity of period k.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "lag": {"type": "integer"},
                },
                "required": ["series_name", "lag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spike_detector",
            "description": "Detect spikes (large amplitude excursions) by Z-score. Also returns overall amplitude range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "threshold": {"type": "number", "default": 3.0},
                    "min_sep": {"type": "integer", "default": 1},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_point_detector",
            "description": "Detect structural change points using ruptures PELT (or CUSUM fallback). Fewer change points = more stable structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "penalty": {"type": "number"},
                    "n_cp": {"type": "integer"},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pattern_consistency_indicators",
            "description": "Compute lumpiness, flat_spots, and crossing_rate to assess overall structural coherence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stationarity_test",
            "description": "Test whether the series is stationary (ADF or KPSS). Useful for pattern_consistency: a stationary series has stable statistical properties over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "test": {"type": "string", "enum": ["adf", "kpss"], "default": "adf"},
                },
                "required": ["series_name"],
            },
        },
    },
]