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

### Rare Pattern  (two categories — only Category 1 affects scoring)
- **rare_pattern**
  Detect and classify unusual values or events into two categories:

  **Category 1 — Outliers (negative quality indicator, used for scoring)**
    Sensor glitches, data corruption, hardware artifacts, recording errors,
    or any anomaly that does NOT correspond to a real-world event.
    Fewer outliers = better quality. Compare A vs B on outlier count/severity.

  **Category 2 — Meaningful rare patterns (informational, NOT scored)**
    Real-world events faithfully captured by the sensor (financial shocks,
    fault signals, environmental anomalies, etc.).
    These are NOT quality defects. Label them per series for reporting only.
    Do NOT penalise a series for having more meaningful rare patterns.
    This category is context-aware: use the dataset description and external
    variables to judge whether a detected pattern is a real-world event or noise.


### Pattern Structure  (structural richness — clearer, more consistent = better)
- **trend**
  Clear, sustained directional movement over time with minimal noise or random
  fluctuations. A series has high trend quality as long as it maintains a discernible
  direction during meaningful periods — the overall direction need not be constant.
  What reduces quality is an absence of any clear direction: flat, erratic, or
  noise-dominated behaviour throughout.

- **frequency**
  Regular oscillations, periodic behavior, or repetitive cycles that are consistent
  across the series with minimal noise. Irregular or absent periodicity = low quality.
  A high-quality periodic signal has energy concentrated in a few dominant frequencies
  (low spectral entropy); a noisy pseudo-periodic signal spreads energy across many
  frequencies (high spectral entropy).

- **amplitude**
  Consistent and well-defined oscillation amplitude across cycles — the peak-to-trough
  magnitude should be significant and stable from one cycle to the next (low coefficient
  of variation). A high-quality amplitude signal is NOT merely "absence of spikes"; it
  requires that each oscillation cycle has a comparable, well-defined swing.
  Amplitude modulation (cycles growing or shrinking over time) reduces quality.

- **pattern_consistency**
  Overall structural coherence: the series exhibits clear, repeatable patterns
  (trend, seasonality, cycles, or stable mean) while avoiding excessive noise, random
  fluctuations, lumpiness, flat spots, or abrupt structural breaks. A high-quality series
  transitions gradually between values rather than jumping erratically.
"""