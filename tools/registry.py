"""
Tool registry for the Inspector agent.

TOOL_REGISTRY : maps function name → callable
TOOL_SCHEMAS  : OpenAI function-calling schemas for all registered tools
"""

from tools.bad_quality import missing_ratio, noise_profile, volatility, range_stats
from tools.rare_pattern import zscore_outlier, outlier_density, mad_residual_outlier, contextual_rare_pattern
from tools.pattern_structure import (
    trend_classifier,
    seasonality_detector,
    change_point_detector,
    pattern_consistency_indicators,
    stationarity_test,
    autocorr,
    rolling_amplitude,
    cycle_amplitude,
)

TOOL_REGISTRY = {
    "missing_ratio": missing_ratio,
    "noise_profile": noise_profile,
    "volatility": volatility,
    "range_stats": range_stats,
    "zscore_outlier": zscore_outlier,
    "outlier_density": outlier_density,
    "mad_residual_outlier": mad_residual_outlier,
    "contextual_rare_pattern": contextual_rare_pattern,
    "trend_classifier": trend_classifier,
    "seasonality_detector": seasonality_detector,
    "change_point_detector": change_point_detector,
    "pattern_consistency_indicators": pattern_consistency_indicators,
    "stationarity_test": stationarity_test,
    "autocorr": autocorr,
    "rolling_amplitude": rolling_amplitude,
    "cycle_amplitude": cycle_amplitude,
}

# OpenAI-style function schemas for tool calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "missing_ratio",
            "description": "Compute fraction of missing (NaN) values in a series. Returns {missing_ratio: 0–1}.",
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
            "description": "Estimate noise level using rolling-window residuals. Also classifies noise as white (random) or red (autocorrelated). Window auto-adapts to series length if omitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "window": {"type": "integer", "description": "Rolling window size. Auto-adapts if omitted: max(5, n//50). Use larger window for long/smooth series, smaller for short/volatile."},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "volatility",
            "description": "Rolling volatility: std of first-differences within a sliding window. Measures local instability — useful for noise_level and amplitude dimensions. Window auto-adapts if omitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "window": {"type": "integer", "description": "Rolling window size. Auto-adapts if omitted: max(5, n//50). Use larger for long series, smaller for short."},
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
            "name": "zscore_outlier",
            "description": "Detect point outliers (Category 1 — data defects) using Z-score threshold. Flags points deviating beyond threshold standard deviations from the global mean.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "anomaly_threshold": {"type": "number", "default": 3.0, "description": "Z-score threshold (default 3.0). Lower (e.g. 2.5) for stricter detection if preview shows subtle spikes; higher (e.g. 4.0) if series has heavy tails."},
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
            "name": "mad_residual_outlier",
            "description": (
                "Robust outlier detection (Category 1 — data defects): detrend via rolling mean, "
                "then apply MAD-based scoring on residuals. Fixes two Z-score weaknesses: "
                "(a) rolling mean removes local trend before scoring, "
                "(b) MAD is outlier-resistant unlike std. "
                "Better than zscore_outlier for series with trends or seasonal drift."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "window": {"type": "integer", "description": "Rolling detrend window. Auto-adapts if omitted: max(7, n//30). Use larger for smooth/long series, smaller for short/volatile."},
                    "threshold": {"type": "number", "default": 3.5, "description": "MAD-based cutoff (default 3.5). Lower for stricter detection."},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contextual_rare_pattern",
            "description": (
                "Contextual rare pattern detection (Category 2 — potentially meaningful events): "
                "for each point, fit a linear trend on the preceding context_window points and flag "
                "points with unusually large prediction errors (MAD-normalised). "
                "Detects sudden breaks, V-shaped events, or step changes that look normal globally "
                "but deviate sharply from local context. Use to identify real-world events that "
                "should be labelled but NOT penalised in scoring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "context_window": {"type": "integer", "description": "Local context length for trend fit. Auto-adapts if omitted: max(5, n//30). Use larger for slow-changing series, smaller for fast-changing."},
                    "threshold": {"type": "number", "default": 3.0, "description": "MAD-normalised error cutoff (default 3.0)."},
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
            "description": (
                "Detect dominant seasonal period via autocorrelation with prominence-based peak filtering "
                "and harmonic suppression. Returns dominant_period, seasonal_strength, top_periods (non-harmonic), "
                "dominance_ratio (peak1/peak2 strength — high means one frequency dominates, low means multiple "
                "frequencies compete), and peak_count (number of significant independent peaks found)."
            ),
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
            "description": (
                "Compute structural coherence indicators: lumpiness (variance of per-window variances — "
                "high = uneven volatility), flat_ratio (fraction of steps with negligible change), "
                "longest_flat_ratio (longest stagnant plateau / n), crossing_rate (mean-crossing frequency), "
                "and roughness (mean absolute step size — lower = smoother)."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "rolling_amplitude",
            "description": (
                "General-purpose amplitude measure for any series (periodic or not). "
                "Slides a window over the series and computes local range (max - min) at each position. "
                "Returns mean_local_range (average swing size), cv_local_range (consistency — lower = more stable amplitude), "
                "max/min_local_range (burst vs quiet windows). "
                "Use when cycle_amplitude returns oscillatory=False, or as a complement for non-periodic series."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "window": {"type": "integer", "description": "Sliding window size. Auto-adapts if omitted: max(10, n//20). For periodic series, set to ~half the dominant period for best results."},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cycle_amplitude",
            "description": (
                "Measure oscillation amplitude consistency via peak-trough analysis. "
                "Returns oscillatory (bool gate — False means the series lacks clear oscillation and other fields are unreliable), "
                "mean_amplitude, amplitude_cv (coefficient of variation — lower = more consistent cycles), "
                "and amplitude_trend (growing/shrinking/stable). "
                "Use for the amplitude dimension; check oscillatory=True before interpreting results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
]