"""
Shared quality dimension definitions for all agents.

DIMENSION_GUIDE is a single source of truth describing what each dimension means
and how it relates to quality. Import and embed it in every agent's SYSTEM_PROMPT.
"""

DIMENSION_GUIDE = """
## Quality Dimensions

### Data Quality  (fewer defects = better)
- **missing_value**
  Fraction of NaN / missing values. More missing → worse quality.

- **noise_level**
  Signal noise and SNR. Higher noise or lower SNR → worse quality.
  A clean, smooth signal is preferred over a noisy, erratic one.

### Rare Pattern  (context-dependent — do NOT assume rare = bad)
- **rare_pattern**
  Statistical outliers, point anomalies, or unusual local patterns.
  Interpretation depends on domain context:
    (a) Sensor / collection artifact (hardware glitch, data corruption)
        → fewer rare patterns = better quality.
    (b) Meaningful real-world event (financial shock, fault signal being monitored)
        → faithfully capturing these events may indicate higher quality, not lower.
  Always justify which interpretation applies based on the dataset description and
  external variables. When context is ambiguous, acknowledge it explicitly.

### Pattern Structure  (structural richness — clearer, more consistent = better)
- **trend**
  A clear, sustained directional movement (upward or downward) over time with minimal
  noise or random fluctuations. A flat or erratic series lacks trend quality.

- **frequency**
  Regular oscillations, periodic behavior, or repetitive cycles that are consistent
  across the series with minimal noise. Irregular or absent periodicity = low quality.

- **amplitude**
  Consistent and well-defined oscillation amplitude — significant and stable variations
  in value range reflecting strong signal intensity with minimal noise. Small or
  irregular variations = low amplitude quality.

- **pattern_consistency**
  Overall structural coherence: the series exhibits clear, repeatable patterns
  (trend, seasonality, cycles, or stable mean) while avoiding excessive noise, random
  fluctuations, lumpiness, flat spots, or abrupt structural breaks.
"""