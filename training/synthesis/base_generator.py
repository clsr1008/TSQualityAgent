"""
Attribute-driven base series generator.

Generates diverse, realistic time series via configurable composition:
    - Additive:       y = trend + seasonal + noise
    - Multiplicative: y = trend_pos * (1 + a * seasonal_norm) + noise
    - Sequential:     y = [seg_1 | seg_2 | ...] + noise   (regime changes)

Diversity sources:
  1. Composition function — additive, multiplicative, sequential
  2. Continuous parameter ranges — amplitude, period, slope from distributions
  3. Combinatorial attribute mixing — any trend x any seasonal x any noise
  4. Intra-component variation — multi-segment trends, amplitude modulation
  5. Noise structure — white, AR(1), heteroscedastic, random walk

Noise is kept intentionally low (baseline floor) since defect_injector will
separately inject dimension-specific degradations on top.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator


# ── Attribute space with weighted probabilities ──────────────────────────────

TREND_TYPES = {
    "flat":        0.20,
    "linear_up":   0.25,
    "linear_down": 0.20,
    "piecewise":   0.20,
    "exponential": 0.10,
    "log":         0.05,
}

SEASONAL_TYPES = {
    "none":     0.15,
    "sine":     0.30,
    "square":   0.08,
    "triangle": 0.12,
    "sawtooth": 0.12,
    "mixed":    0.23,   # sum of 2-3 sine harmonics
}

NOISE_TYPES = {
    "white":           0.45,   # Gaussian iid
    "ar1":             0.30,   # AR(1) correlated
    "heteroscedastic": 0.15,   # time-varying variance
    "random_walk":     0.10,   # non-stationary drift (detrended)
}

COMPOSITION_TYPES = {
    "additive":       0.55,   # y = trend + seasonal + noise
    "multiplicative": 0.30,   # y = trend_pos * (1 + α·seasonal_norm) + noise
    "sequential":     0.15,   # y = [seg₁ | seg₂ | ...] + noise
}


# Scenario descriptions for realistic context (Perceiver uses these)
SCENARIO_POOL = [
    "Daily sensor readings from an industrial monitoring system.",
    "Hourly temperature measurements from a weather station.",
    "Minute-level CPU utilization from a cloud server.",
    "Monthly revenue figures for a consumer product line.",
    "Daily closing prices of a stock index.",
    "Hourly electricity demand from a regional substation.",
    "Vibration amplitude readings from a rotating pump shaft.",
    "Pressure sensor readings from a hydraulic actuator.",
    "Heart rate measurements from a wearable fitness device.",
    "Weekly foot traffic count at a retail store entrance.",
    "Daily PM2.5 concentration from an air quality monitor.",
    "Hourly water flow rate in a municipal pipeline.",
    "Engine RPM readings during a vehicle road test.",
    "Quarterly GDP growth rate estimates.",
    "Daily soil moisture levels from an agricultural sensor.",
    "Minute-level network throughput from a backbone router.",
    "Hourly wind speed measurements at a coastal weather station.",
    "Daily discharge volume from a river gauge station.",
    "Monthly average rental price index for a metropolitan area.",
    "Hourly solar irradiance from a photovoltaic panel sensor.",
    "Accelerometer readings from a structural health monitoring system.",
    "Blood glucose level readings from a continuous glucose monitor.",
    "Daily count of customer support tickets.",
    "Hourly noise level measurements from an urban sound monitor.",
    "Weekly inventory turnover rates for a warehouse.",
    "Tide height measurements from a coastal buoy.",
    "Daily new user signups for a mobile application.",
    "Minute-level CO2 concentration in a conference room.",
    "Monthly shipping container throughput at a port terminal.",
    "Hourly gas consumption readings from a smart meter.",
]


# ── Primitive component generators ───────────────────────────────────────────

def _make_trend(n: int, trend_type: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Generate trend component with random parameters."""
    meta = {"type": trend_type}

    if trend_type == "flat":
        bias = rng.uniform(-2, 2)
        meta["bias"] = round(bias, 4)
        return np.full(n, bias), meta

    elif trend_type in ("linear_up", "linear_down"):
        slope = rng.uniform(0.005, 0.03)
        if trend_type == "linear_down":
            slope = -slope
        start = rng.uniform(-3, 3)
        meta["slope"] = round(slope, 6)
        meta["start"] = round(start, 4)
        return np.linspace(start, start + slope * n, n), meta

    elif trend_type == "piecewise":
        n_turning = rng.integers(1, 5)  # 1-4 turning points
        min_gap = max(10, n // 8)

        # Fall back to linear if segment too short for piecewise
        if n <= 2 * min_gap:
            slope = rng.uniform(-2, 2)
            start = rng.uniform(-3, 3)
            meta["start"] = round(start, 4)
            meta["slope"] = round(slope, 4)
            return np.linspace(start, start + slope * n, n), meta

        # Generate keypoints with minimum spacing (ChatTS approach)
        inner_x = []
        for _ in range(n_turning):
            for _retry in range(100):
                x = rng.integers(min_gap, n - min_gap)
                if all(abs(x - px) >= min_gap for px in inner_x):
                    inner_x.append(int(x))
                    break
        inner_x.sort()
        key_x = np.array([0] + inner_x + [n - 1])
        key_y = rng.uniform(-3, 3, size=len(key_x))

        # PCHIP interpolation for smooth curve
        curve = PchipInterpolator(key_x, key_y)(np.arange(n))
        meta["n_turning"] = len(inner_x)
        meta["keypoints_x"] = key_x.tolist()
        meta["keypoints_y"] = [round(float(y), 4) for y in key_y]
        return curve, meta

    elif trend_type == "exponential":
        rate = rng.uniform(0.005, 0.02)
        direction = rng.choice([-1, 1])
        rate *= direction
        start = rng.uniform(-1, 1)
        curve = start + np.exp(rate * np.arange(n)) - 1
        meta["rate"] = round(rate, 6)
        meta["start"] = round(start, 4)
        return curve, meta

    elif trend_type == "log":
        scale = rng.uniform(0.5, 3.0)
        direction = rng.choice([-1, 1])
        curve = direction * scale * np.log1p(np.arange(n) / (n / 10))
        meta["scale"] = round(scale, 4)
        meta["direction"] = int(direction)
        return curve, meta

    return np.zeros(n), meta


def _make_seasonal(n: int, seasonal_type: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Generate seasonal/periodic component with random parameters."""
    meta = {"type": seasonal_type}

    if seasonal_type == "none":
        return np.zeros(n), meta

    # Random period: short (n/16 ~ n/8) or long (n/8 ~ n/3)
    if rng.random() < 0.5:
        period = rng.integers(max(6, n // 16), max(8, n // 8) + 1)
    else:
        period = rng.integers(max(8, n // 8), max(12, n // 3) + 1)
    period = int(period)
    meta["period"] = period

    # Random amplitude
    amplitude = rng.uniform(0.3, 3.0)
    meta["amplitude"] = round(amplitude, 4)

    t = np.arange(n, dtype=float)
    phase = rng.uniform(0, 2 * math.pi)
    meta["phase"] = round(phase, 4)

    if seasonal_type == "sine":
        wave = amplitude * np.sin(2 * math.pi * t / period + phase)

    elif seasonal_type == "square":
        wave = amplitude * np.sign(np.sin(2 * math.pi * t / period + phase))

    elif seasonal_type == "triangle":
        shifted = (t / period + phase / (2 * math.pi))
        wave = amplitude * (2 * np.abs(2 * (shifted - np.floor(shifted + 0.5))) - 1)

    elif seasonal_type == "sawtooth":
        shifted = (t / period + phase / (2 * math.pi))
        wave = amplitude * (2 * (shifted - np.floor(shifted)) - 1)

    elif seasonal_type == "mixed":
        # 2-3 harmonics with random amplitudes (ChatTS style)
        n_harmonics = rng.integers(2, 4)
        wave = np.zeros(n)
        harmonic_info = []
        for h in range(1, n_harmonics + 1):
            h_amp = amplitude / h * rng.uniform(0.5, 1.5)
            h_phase = rng.uniform(0, 2 * math.pi)
            wave += h_amp * np.sin(2 * math.pi * h * t / period + h_phase)
            harmonic_info.append({"harmonic": h, "amplitude": round(h_amp, 4)})
        meta["harmonics"] = harmonic_info
    else:
        wave = np.zeros(n)

    # Optional amplitude modulation across segments (ChatTS multi-segment)
    if rng.random() < 0.3 and n > 60:
        n_segs = rng.integers(2, 4)
        splits = sorted(rng.choice(range(n // 5, 4 * n // 5), size=n_segs - 1, replace=False))
        boundaries = [0] + [int(s) for s in splits] + [n]
        modulation = np.ones(n)
        seg_scales = []
        for i in range(len(boundaries) - 1):
            scale = rng.uniform(0.5, 2.0)
            modulation[boundaries[i]:boundaries[i + 1]] = scale
            seg_scales.append(round(scale, 3))
        # Smooth the transitions
        kernel_size = min(5, n // 10)
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            modulation = np.convolve(modulation, kernel, mode="same")
        wave *= modulation
        meta["amplitude_modulation"] = seg_scales

    return wave, meta


def _make_base_noise(n: int, noise_type: str, rng: np.random.Generator,
                     signal_std: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Generate low-level baseline noise with configurable temporal structure.

    Overall magnitude is 1-5% of signal std (baseline floor).
    The noise_type controls temporal correlation, not amplitude.
    """
    base_ratio = rng.uniform(0.03, 0.08)
    std = base_ratio * max(signal_std, 0.1)
    meta = {"type": noise_type, "std": round(std, 6)}

    if noise_type == "white":
        noise = rng.normal(0, std, n)

    elif noise_type == "ar1":
        phi = rng.uniform(0.3, 0.9)
        innovation_std = std * math.sqrt(max(1 - phi ** 2, 0.01))
        noise = np.zeros(n)
        noise[0] = rng.normal(0, std)
        for t in range(1, n):
            noise[t] = phi * noise[t - 1] + rng.normal(0, innovation_std)
        meta["phi"] = round(phi, 4)

    elif noise_type == "heteroscedastic":
        # Smooth time-varying variance envelope
        n_knots = max(3, n // 30)
        knot_vals = rng.uniform(0.3, 1.7, size=n_knots)
        envelope = np.interp(np.arange(n), np.linspace(0, n - 1, n_knots), knot_vals)
        noise = rng.normal(0, std, n) * envelope

    elif noise_type == "random_walk":
        increments = rng.normal(0, std * 0.3, n)
        noise = np.cumsum(increments)
        # Remove linear trend so it stays a fluctuation around zero
        noise -= np.linspace(noise[0], noise[-1], n)
        # Rescale to target std
        actual_std = max(float(np.std(noise)), 1e-10)
        noise = noise * (std / actual_std)

    else:
        noise = rng.normal(0, std, n)

    return noise, meta


# ── Composition helpers ──────────────────────────────────────────────────────

def _compose_sequential(n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Sequential composition: 2-3 segments each with independent trend + seasonal.

    Simulates regime changes, phase shifts, or operating-mode transitions.
    Segments are level-aligned at boundaries for continuity.
    """
    n_segments = int(rng.choice([2, 3], p=[0.6, 0.4]))

    # Split points with minimum segment length
    min_seg = max(20, n // 6)
    if n_segments == 2:
        split = int(rng.integers(min_seg, n - min_seg))
        boundaries = [0, split, n]
    else:
        s1 = int(rng.integers(min_seg, n - 2 * min_seg))
        s2 = int(rng.integers(s1 + min_seg, n - min_seg))
        boundaries = [0, s1, s2, n]

    signal = np.zeros(n)
    segments_meta = []
    prev_end_val = 0.0

    for i in range(n_segments):
        s, e = boundaries[i], boundaries[i + 1]
        seg_len = e - s

        seg_trend_type = _weighted_choice(TREND_TYPES, rng)
        seg_seasonal_type = _weighted_choice(SEASONAL_TYPES, rng)
        if seg_len < 30:
            seg_seasonal_type = "none"
        elif seg_len < 60 and seg_seasonal_type == "mixed":
            seg_seasonal_type = "sine"

        seg_trend, seg_t_meta = _make_trend(seg_len, seg_trend_type, rng)
        seg_seasonal, seg_s_meta = _make_seasonal(seg_len, seg_seasonal_type, rng)
        seg = seg_trend + seg_seasonal

        # Level-align with previous segment end
        if i > 0:
            seg += prev_end_val - seg[0]

        signal[s:e] = seg
        prev_end_val = float(seg[-1])
        segments_meta.append({
            "range": [s, e],
            "trend": seg_t_meta,
            "seasonal": seg_s_meta,
        })

    meta = {
        "type": "sequential",
        "n_segments": n_segments,
        "boundaries": boundaries,
        "segments": segments_meta,
    }
    return signal, meta


# ── Weighted random choice helper ────────────────────────────────────────────

def _weighted_choice(options: dict, rng: np.random.Generator) -> str:
    """Pick a key from {name: probability} dict."""
    names = list(options.keys())
    probs = np.array(list(options.values()), dtype=float)
    probs /= probs.sum()
    return str(rng.choice(names, p=probs))




# ── Core random generator ────────────────────────────────────────────────────

def generate_random_base(
    n: int = 150,
    seed: int | None = None,
) -> tuple[np.ndarray, dict, str]:
    """
    Generate a random base series by sampling from the attribute space.

    Every call with a different seed produces a unique series.

    Returns
    -------
    (series, attribute_pool, dataset_description)
    """
    rng = np.random.default_rng(seed)

    # Sample component types
    composition = _weighted_choice(COMPOSITION_TYPES, rng)
    noise_type = _weighted_choice(NOISE_TYPES, rng)

    if composition == "sequential":
        signal, comp_meta = _compose_sequential(n, rng)
    else:
        trend_type = _weighted_choice(TREND_TYPES, rng)
        seasonal_type = _weighted_choice(SEASONAL_TYPES, rng)

        # Length-adaptive constraints
        if n < 30:
            seasonal_type = "none"
        elif n < 60 and seasonal_type == "mixed":
            seasonal_type = "sine"

        trend_component, trend_meta = _make_trend(n, trend_type, rng)
        seasonal_component, seasonal_meta = _make_seasonal(n, seasonal_type, rng)

        if composition == "additive":
            signal = trend_component + seasonal_component
            comp_meta = {"type": "additive"}

        else:  # multiplicative
            # Shift trend to strictly positive, then seasonal modulates its level
            t_min = float(trend_component.min())
            trend_pos = trend_component - t_min + 1.0
            mod_strength = rng.uniform(0.2, 0.6)
            s_max = max(float(np.abs(seasonal_component).max()), 1e-8)
            seasonal_norm = seasonal_component / s_max
            signal = trend_pos * (1 + mod_strength * seasonal_norm)
            comp_meta = {"type": "multiplicative", "mod_strength": round(mod_strength, 4)}

    # Noise (low-level floor)
    signal_std = float(np.std(signal)) if np.std(signal) > 0 else 1.0
    noise_component, noise_meta = _make_base_noise(n, noise_type, rng, signal_std)

    series = signal + noise_component

    # Random overall scale and bias (ChatTS: exponential scale)
    overall_scale = float(10 ** rng.uniform(-1, 2))  # 0.1 ~ 100
    overall_bias = rng.uniform(-overall_scale * 2, overall_scale * 2)
    series = series * overall_scale + overall_bias

    # Pick a scenario description
    desc = str(rng.choice(SCENARIO_POOL))

    # Build attribute pool
    attribute_pool = {
        "n": n,
        "seed": seed,
        "overall_scale": round(overall_scale, 4),
        "overall_bias": round(overall_bias, 4),
        "composition": comp_meta,
        "noise": noise_meta,
    }

    # For non-sequential, include trend/seasonal at top level
    # (sample_generator.py reads attr_pool["seasonal"]["period"] for defect injection)
    if composition != "sequential":
        attribute_pool["trend"] = trend_meta
        attribute_pool["seasonal"] = seasonal_meta
    else:
        attribute_pool["segments"] = comp_meta["segments"]

    return series, attribute_pool, desc




# ── Deterministic generator (for explicit control) ───────────────────────────

def generate_base_series(
    n: int = 150,
    trend: str = "linear_up",
    seasonal: str = "sine",
    period: int = 25,
    amplitude: float = 1.0,
    noise_std: float = 0.1,
    slope: float = 0.01,
    n_segments: int = 3,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Generate a base series with explicit parameters (deterministic).

    Use generate_random_base() for diverse random sampling.
    """
    rng = np.random.default_rng(seed)

    # Trend
    if trend == "flat":
        trend_component = np.zeros(n)
    elif trend == "linear_up":
        trend_component = np.linspace(0, slope * n, n)
    elif trend == "linear_down":
        trend_component = np.linspace(0, -slope * n, n)
    elif trend == "piecewise":
        n_pts = n_segments + 1
        inner_x = sorted(rng.choice(range(1, n - 1), size=max(0, n_pts - 2), replace=False))
        key_x = np.array([0] + list(inner_x) + [n - 1])
        key_y = rng.uniform(-slope * n * 0.5, slope * n * 0.5, size=len(key_x))
        trend_component = PchipInterpolator(key_x, key_y)(np.arange(n))
    else:
        trend_component = np.zeros(n)

    # Seasonal
    t = np.arange(n)
    if seasonal == "none":
        seasonal_component = np.zeros(n)
    elif seasonal == "sine":
        seasonal_component = amplitude * np.sin(2 * math.pi * t / period)
    elif seasonal == "square":
        seasonal_component = amplitude * np.sign(np.sin(2 * math.pi * t / period))
    elif seasonal == "triangle":
        seasonal_component = amplitude * (2 * np.abs(2 * (t / period - np.floor(t / period + 0.5))) - 1)
    else:
        seasonal_component = np.zeros(n)

    # Noise
    noise_component = rng.normal(0, noise_std, n)

    series = trend_component + seasonal_component + noise_component

    attribute_pool = {
        "n": n, "trend": trend, "seasonal": seasonal, "period": period,
        "amplitude": amplitude, "noise_std": noise_std, "slope": slope,
        "n_segments": n_segments, "seed": seed,
    }
    return series, attribute_pool