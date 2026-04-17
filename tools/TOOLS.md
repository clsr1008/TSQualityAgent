# TSqualityAgent — Tool Reference

All tools are called by the Inspector agent during ReAct loops. Each tool takes `series_name` ("A" or "B") as the primary parameter, plus optional tuning parameters. Results are returned as dicts and fed back as observations.

---

## Category 1: Bad Quality

Tools for detecting data completeness and noise issues. Primarily used for the `missing_value` and `noise_level` dimensions.

---

### `missing_ratio`

**What it computes:** Counts NaN values and divides by total length.

**Parameters:** none

**Returns:**
| Field | Meaning |
|-------|---------|
| `missing_ratio` | Fraction of missing values (0–1). 0 = complete, 1 = all missing |

**Use for:** `missing_value` dimension. The single most direct indicator — compare A vs B directly.

---

### `noise_profile`

**What it computes:** Estimates noise by computing the residual after subtracting a rolling mean (convolution with a uniform kernel of size `window`). The residual represents the high-frequency component the smoothing cannot explain. Also classifies noise type via lag-1 autocorrelation.

**Parameters:**
- `window` (default 5): rolling window size for smoothing

**Returns:**
| Field | Meaning |
|-------|---------|
| `noise_std` | Standard deviation of residuals — absolute noise level |
| `signal_std` | Overall standard deviation of the series |
| `noise_ratio` | `noise_std / signal_std` — fraction of total variation that is noise; lower = cleaner |
| `noise_type` | `"white"` (random, uncorrelated) or `"red"` (lag-1 autocorr > 0.3, structured/correlated noise) |

**Limitation:** Rolling mean cannot fully remove periodic components. For strongly periodic series with period < `window`, `noise_std` will be inflated. Best used alongside `volatility`.

**Use for:** `noise_level` dimension as the primary noise estimator.

---

### `volatility`

**What it computes:** Computes first differences (`diff`), then slides a window over the differences and takes the standard deviation within each window. Measures how erratically the series jumps step-to-step.

**Parameters:**
- `window` (default 5): window size over the difference series

**Returns:**
| Field | Meaning |
|-------|---------|
| `mean_volatility` | Average local instability across all windows |
| `max_volatility` | Worst single window — detects bursts of instability |

**Distinction from `noise_profile`:** `noise_profile` measures deviation from a smooth trend; `volatility` measures the speed and irregularity of point-to-point jumps. A slowly drifting series can have low volatility but high noise_ratio if the drift is not linear.

**Use for:** `noise_level` dimension as a complementary measure, especially when the series lacks a clear trend or periodicity.

---

### `range_stats`

**What it computes:** Extracts a segment `[start, end)` from the series and computes a single statistic over it.

**Parameters:**
- `start`: inclusive start index
- `end`: exclusive end index
- `stat` (default `"mean"`): one of `mean`, `std`, `max`, `min`, `sum`

**Returns:**
| Field | Meaning |
|-------|---------|
| `value` | The computed statistic for the segment |
| `segment_length` | Number of points in the segment |

**Use for:** Any dimension where local segment analysis is needed — e.g., comparing noise levels before vs after a change point, or verifying a trend in a specific window identified by `change_point_detector`.

---

## Category 2: Rare Pattern

Tools for detecting unusual values or events. Used for the `rare_pattern` dimension, where context determines whether anomalies are defects or meaningful signals.

---

### `zscore_outlier`

**What it computes:** Computes global mean and std, then flags points where `|value - mean| / std > threshold`.

**Parameters:**
- `anomaly_threshold` (default 3.0): Z-score cutoff

**Returns:**
| Field | Meaning |
|-------|---------|
| `anomaly_count` | Number of flagged points |
| `anomaly_ratio` | Fraction of total series length |
| `anomaly_indices` | Time step positions of anomalies |
| `anomaly_values` | Raw values at those positions |
| `threshold_used` | The threshold applied |

**Limitation:** Uses global mean/std, which can be distorted by trends — a point near the end of an upward trend may appear anomalous simply because the global mean is far from the local context. For trended series, prefer `mad_residual_outlier`.

**Use for:** `rare_pattern` dimension as a quick first-pass, especially for flat or stationary series.

---

### `outlier_density`

**What it computes:** IQR-based fencing. Computes Q1 and Q3, then flags values outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`.

**Parameters:** none

**Returns:**
| Field | Meaning |
|-------|---------|
| `outlier_count` | Number of flagged points |
| `outlier_ratio` | Fraction of total length |
| `iqr` | Interquartile range — spread of the middle 50% |
| `lower_fence` / `upper_fence` | Boundary values beyond which points are flagged |

**Distinction from `zscore_outlier`:** Uses median-derived fences rather than mean/std, making it more robust when outliers inflate the standard deviation. However, it still uses global statistics and has similar limitations for trended series.

**Use for:** `rare_pattern` dimension as a robust complement to `zscore_outlier`, or when the series may have heavy tails.

---

### `mad_residual_outlier`

**What it computes:** Two-stage robust detection:
1. Detrend via rolling mean (causal window of size `window`) → residuals
2. Score each residual using the modified Z-score: `0.6745 × |residual − median| / MAD`
3. Flag points where this score exceeds `threshold`

**Parameters:**
- `window` (default 15): rolling mean window for detrending
- `threshold` (default 3.5): modified Z-score cutoff (Iglewicz & Hoaglin recommend 3.5)

**Returns:**
| Field | Meaning |
|-------|---------|
| `anomaly_count` | Number of flagged points |
| `anomaly_ratio` | Fraction of total length |
| `anomaly_indices` | Positions of detected anomalies |
| `anomaly_values` | Raw values at those positions |
| `mad` | MAD of residuals — measures typical residual spread; small = tight fit |
| `threshold_used` | The threshold applied |

**Why better than `zscore_outlier` for trended series:**
- Rolling mean removes local trend before scoring → trend does not inflate scores
- MAD is outlier-resistant → a single large spike does not inflate the scoring scale

**Use for:** `rare_pattern` dimension when the series has a visible trend or seasonal drift.

---

### `contextual_rare_pattern`

**What it computes:** For each point at position `i`, fits a linear regression over the preceding `context_window` points and extrapolates a prediction. The prediction error `|actual − predicted|` is then scored using MAD normalisation across all errors. Points with unusually large prediction errors are flagged.

**Parameters:**
- `context_window` (default 10): number of preceding points used to build local expectation
- `threshold` (default 3.0): MAD-normalised error cutoff

**Returns:**
| Field | Meaning |
|-------|---------|
| `anomaly_count` | Number of flagged points |
| `anomaly_ratio` | Fraction of total length |
| `anomaly_indices` | Positions of contextual anomalies |
| `anomaly_values` | Raw values at those positions |
| `threshold_used` | The threshold applied |

**Key distinction:** A point can look normal globally but be highly anomalous contextually (e.g., a sudden V-shaped crash in an otherwise smooth uptrend). This is the right tool for `rare_contextual` cases where an external event caused a localised deviation that is not a global outlier.

**Use for:** `rare_pattern` dimension when context suggests a real event occurred, or when the anomaly is a structural break rather than a point spike.

---

## Category 3: Pattern Structure

Tools for assessing structural richness of the series. Used for the `trend`, `frequency`, `amplitude`, and `pattern_consistency` dimensions.

---

### `trend_classifier`

**What it computes:** Two-layer analysis:
1. **Global linear regression** over the full series → slope, R² (`trend_strength`), direction
2. **Per-segment clarity** using `change_point_detector` to find natural breakpoints, then fitting a separate regression within each segment and averaging their R² values → `segment_clarity`

**Parameters:**
- `window` (optional): if set, only analyses the last `window` points

**Returns:**
| Field | Meaning |
|-------|---------|
| `direction` | `"increasing"` / `"decreasing"` / `"flat"` based on global slope and R² |
| `slope` | Average change per time step |
| `trend_strength` | Global R² (0–1); how well the full series fits a single line |
| `segment_clarity` | `"clear"` (mean seg R² ≥ 0.6) / `"moderate"` (0.25–0.6) / `"unclear"` (<0.25) — whether each natural segment has a well-defined internal trend |
| `segment_count` | Number of segments used for the clarity calculation |

**Key insight:** `trend_strength` measures global linearity; `segment_clarity` measures whether the series has a clear direction *within each period*, even if the overall direction is not constant. High quality = high `segment_clarity`, regardless of `direction`.

**Use for:** `trend` dimension.

---

### `seasonality_detector`

**What it computes:**
1. Computes ACF (autocorrelation function) for lags 1 to `max_period`
2. Finds significant peaks using `scipy.signal.find_peaks` with `height=0.05` and `prominence=0.05` (filters out noise bumps)
3. Suppresses harmonics: if a candidate period is a near-integer multiple (±10%) of an already-accepted period, it is excluded
4. Returns top-3 accepted periods sorted by ACF strength

**Parameters:**
- `max_period` (default `n // 2`): maximum lag to search

**Returns:**
| Field | Meaning |
|-------|---------|
| `dominant_period` | Period with the strongest ACF peak |
| `seasonal_strength` | ACF value at the dominant period (0–1); higher = more consistent periodicity |
| `top_periods` | Up to 3 significant non-harmonic candidate periods |
| `dominance_ratio` | `peak1_strength / peak2_strength`; high (>3) = one frequency dominates; near 1 = multiple frequencies compete |
| `peak_count` | Number of significant independent peaks found |

**Use for:** `frequency` dimension. `seasonal_strength` measures periodicity quality; `dominance_ratio` + `peak_count` distinguish single-frequency (high quality) from multi-frequency (degraded) signals.

---

### `change_point_detector`

**What it computes:** Detects structural breakpoints where the statistical properties of the series shift. Uses `ruptures` PELT algorithm (L2 cost) by default, with CUSUM as a fallback if `ruptures` is not installed.

**Parameters:**
- `penalty` (default `3 × log(n)`): PELT sensitivity — lower penalty finds more change points
- `n_cp`: if set, uses Binseg to find exactly this many change points

**Returns:**
| Field | Meaning |
|-------|---------|
| `change_point_count` | Number of detected breakpoints |
| `change_point_indices` | Time step positions of breakpoints |
| `method` | `"ruptures_pelt"` / `"ruptures_binseg"` / `"cusum_fallback"` |

**Use for:** All structural dimensions as a supporting tool. `trend_classifier` calls it internally for segment clarity. Also useful standalone to identify where a series changes character before applying local analysis with `range_stats`.

---

### `pattern_consistency_indicators`

**What it computes:** Five indicators of structural coherence:

1. **Lumpiness**: divides series into `n//10`-sized windows, computes variance per window, then takes the variance of those variances
2. **Flat ratio**: fraction of steps where `|x[i+1] − x[i]| < 0.1 × std` (negligible change)
3. **Longest flat ratio**: longest consecutive run of flat steps divided by `n`
4. **Crossing rate**: number of times the series crosses its mean, divided by `n−1`
5. **Roughness**: mean absolute step size `mean(|x[i+1] − x[i]|)`

**Parameters:** none

**Returns:**
| Field | Meaning |
|-------|---------|
| `lumpiness` | Variance of per-window variances — high = uneven/bursty volatility |
| `flat_ratio` | Proportion of near-zero steps — high = series frequently stagnates |
| `longest_flat_ratio` | Longest plateau relative to series length — captures the worst stagnant stretch |
| `crossing_rate` | Mean-crossing frequency — low = long runs on one side (trend/drift); high = rapid oscillation |
| `roughness` | Mean absolute step — lower = smoother, higher = more jagged |

**Use for:** `pattern_consistency` dimension. `lumpiness` is the most discriminative for uneven variance (like `case_pattern`); `roughness` for smoothness; `flat_ratio`/`longest_flat_ratio` for stagnation (like `case_trend` B mid-segment).

---

### `stationarity_test`

**What it computes:** Statistical hypothesis test for stationarity via `statsmodels`.

- **ADF** (default): H₀ = unit root (non-stationary). `p < 0.05` → stationary.
- **KPSS**: H₀ = stationary. `p > 0.05` → stationary. (Opposite direction from ADF.)

**Parameters:**
- `test` (default `"adf"`): `"adf"` or `"kpss"`

**Returns:**
| Field | Meaning |
|-------|---------|
| `statistic` | Raw test statistic |
| `p_value` | Statistical significance |
| `is_stationary` | Boolean conclusion at α=0.05 |

**Use for:** `pattern_consistency` and `trend` dimensions. A non-stationary series (has trend or unit root) fundamentally differs in structural quality from a stationary one. Best used as a supporting check rather than primary evidence — ADF and KPSS can give conflicting results, especially for borderline series.

---

### `autocorr`

**What it computes:** Pearson correlation between the series and its lag-`lag` shifted copy: `corr(arr[:-lag], arr[lag:])`.

**Parameters:**
- `lag` (default 1): the lag value to evaluate

**Returns:**
| Field | Meaning |
|-------|---------|
| `lag` | The lag used |
| `autocorrelation` | Correlation coefficient (−1 to 1) |

**Interpretation:**
- `lag=1`: smoothness of consecutive steps — high positive = smooth; near 0 = white noise; high negative = zigzag
- `lag=P` (known period): confirms whether `seasonality_detector`'s dominant period is genuine

**Use for:** Spot verification of a specific lag hypothesis. Not for exploratory period search — `seasonality_detector` already computes the full ACF curve internally.

---

### `rolling_amplitude`

**What it computes:** Slides a window of size `window` over the series and computes `max − min` (local range) within each window, producing an instantaneous amplitude curve. Returns summary statistics of that curve.

**Parameters:**
- `window` (default 20): window size; for periodic series, set to approximately half the cycle period

**Returns:**
| Field | Meaning |
|-------|---------|
| `mean_local_range` | Average local swing — larger = more active amplitude |
| `cv_local_range` | Coefficient of variation of local ranges — lower = more consistent amplitude |
| `max_local_range` | Peak local swing — detects amplitude bursts |
| `min_local_range` | Quietest window |

**Use for:** `amplitude` dimension when `cycle_amplitude` returns `oscillatory=False` (non-periodic series). Also useful as a complement to `cycle_amplitude` for periodic series to verify amplitude stability across different window sizes.

---

### `cycle_amplitude`

**What it computes:**
1. **Gate check**: uses `find_peaks` with `prominence > 0.3 × std` to detect significant peaks and troughs. If fewer than 2 significant peaks or 2 troughs are found, returns `oscillatory=False` and NaN metrics.
2. **Pair analysis**: finds all local extrema (relaxed detection), sorts them chronologically, and pairs adjacent peak-trough pairs. Computes the magnitude of each pair.
3. **Amplitude modulation**: compares mean magnitude of the first half vs second half of pairs to detect growing/shrinking amplitude.

**Parameters:** none

**Returns:**
| Field | Meaning |
|-------|---------|
| `oscillatory` | `False` if the series lacks clear oscillation — **check this first before using other fields** |
| `cycle_count` | Number of complete peak-trough pairs found |
| `mean_amplitude` | Average peak-to-trough magnitude across all cycles |
| `amplitude_cv` | Std / mean of cycle magnitudes — lower = more consistent cycle-to-cycle amplitude |
| `amplitude_trend` | `"growing"` / `"shrinking"` / `"stable"` — whether amplitude is modulating over time |
| `peak_count` | Total peaks detected |
| `trough_count` | Total troughs detected |

**Use for:** `amplitude` dimension when the series has clear oscillations. `amplitude_cv` is the primary quality indicator — low CV + large `mean_amplitude` = high quality. Always check `oscillatory=True` before interpreting results; fall back to `rolling_amplitude` if `oscillatory=False`.

---

## Tool Selection Guide

| Dimension | Primary Tools | Supporting Tools |
|-----------|--------------|-----------------|
| `missing_value` | `missing_ratio` | — |
| `noise_level` | `noise_profile`, `volatility` | `range_stats` |
| `rare_pattern` | Cat-1 scoring: `mad_residual_outlier`, `zscore_outlier`, `outlier_density`; Cat-2 labelling: `contextual_rare_pattern` | — |
| `trend` | `trend_classifier` | `change_point_detector`, `range_stats`, `stationarity_test` |
| `frequency` | `seasonality_detector` | `autocorr` |
| `amplitude` | `cycle_amplitude` (if oscillatory), `rolling_amplitude` (fallback) | `change_point_detector` |
| `pattern_consistency` | `pattern_consistency_indicators` | `stationarity_test`, `change_point_detector` |