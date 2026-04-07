"""
Synthetic pairwise test cases for TSqualityAgent.

Design principles:
  - Both series share the same base signal; only one targeted aspect is degraded in B
  - Differences are intentionally subtle — not immediately obvious at a glance
  - Dataset descriptions reflect realistic scenarios to support contextual reasoning

Case taxonomy
─────────────
missing         : bad_quality — B has ~10% missing values, same noise as A
noise           : bad_quality — B has elevated noise, no missing values
rare_point      : rare_pattern — point anomalies that are clearly sensor artifacts
rare_contextual : rare_pattern — contextual anomaly that reflects a real external event
trend           : pattern_structure — B's trend clarity weakens in a middle segment
frequency       : pattern_structure — B has phase jitter, making periodicity less pure
amplitude       : pattern_structure — B shows amplitude modulation (growing oscillation size)
pattern         : pattern_structure — B has higher local variance variability (lumpier)

Usage
─────
    from synthetic_cases import get_cases
    cases = get_cases()                 # all 8 cases
    cases = get_cases("rare_point")     # single case by name

    python synthetic_cases.py                   # plot all
    python synthetic_cases.py --case trend      # plot one case by name
"""

import math
import numpy as np


# ── Primitive signal generators ────────────────────────────────────────────────

def _trend(n: int, start: float, end: float) -> np.ndarray:
    return np.linspace(start, end, n)


def _sine(n: int, period: float, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n)
    return amplitude * np.sin(2 * math.pi * t / period + phase)


def _noise(n: int, std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, std, n)


def _inject_missing(series: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = series.copy().astype(float)
    idx = rng.choice(len(out), size=int(len(out) * ratio), replace=False)
    out[idx] = np.nan
    return out


def _to_list(arr: np.ndarray) -> list:
    return [None if np.isnan(v) else float(round(v, 4)) for v in arr]


# ── Bad quality ────────────────────────────────────────────────────────────────

def case_missing(n: int = 150, seed: int = 1) -> tuple:
    """
    A and B come from the same underlying source (sinusoidal + gentle trend).
    B has ~6% missing values but the same noise level as A.
    The degradation is purely in data completeness, not in measurement quality.
    """
    shared = _sine(n, period=30, amplitude=2.0) + _trend(n, 0, 1.5)

    # A and B get independent noise realisations on the same underlying signal
    series_a = shared + _noise(n, std=0.2, seed=seed)
    series_b = shared + _noise(n, std=0.2, seed=seed + 5)
    series_b = _inject_missing(series_b, ratio=0.06, seed=seed + 20)

    return (
        "bad_quality | missing — B has ~6% missing, same noise as A",
        {
            "dataset_description": (
                "Temperature readings from two co-located sensors over the same period. "
                "Both sensors monitor the same physical process (daily heating cycle with "
                "a slow seasonal drift)."
            ),
            "series_A": _to_list(series_a),
            "series_B": _to_list(series_b),
            "external_variables": {},
        },
    )


def case_noise(n: int = 150, seed: int = 11) -> tuple:
    """
    A and B come from the same underlying source (sinusoidal + gentle trend).
    A has significantly elevated noise but no missing values.
    The degradation is purely in measurement precision, not in completeness.
    """
    base = _sine(n, period=30, amplitude=2.0) + _trend(n, 0, 1.5) + _noise(n, std=0.2, seed=seed)

    series_b = base.copy()
    # A: same base but noise std roughly doubled
    series_a = base + _noise(n, std=0.55, seed=seed + 10)

    return (
        "bad_quality | noise — A has elevated noise, no missing values",
        {
            "dataset_description": (
                "Temperature readings from two co-located sensors over the same period. "
                "Both sensors monitor the same physical process (daily heating cycle with "
                "a slow seasonal drift)."
            ),
            "series_A": _to_list(series_a),
            "series_B": _to_list(series_b),
            "external_variables": {},
        },
    )


# ── Rare pattern ───────────────────────────────────────────────────────────────

def case_rare_point(n: int = 150, seed: int = 2) -> tuple:
    """
    Both series follow a smooth periodic signal.
    B has 4 sudden point spikes at random positions (magnitude ~5–6 std),
    occurring at times with no known external cause — clearly sensor artifacts.
    """
    base = _sine(n, period=25, amplitude=1.5) + _noise(n, std=0.25, seed=seed)

    series_a = base.copy()
    series_b = base.copy()

    rng = np.random.default_rng(seed + 5)
    spike_idx = rng.choice(np.arange(10, n - 10), size=4, replace=False)
    signs = rng.choice([-1, 1], size=4)
    magnitudes = rng.uniform(5.0, 6.5, size=4)   # ~5–6σ
    for i, s, m in zip(spike_idx, signs, magnitudes):
        series_a[i] += s * m

    return (
        "rare_pattern | point anomaly — A has sensor-fault spikes",
        {
            "dataset_description": (
                "Vibration sensor readings from two identical pumps running in parallel "
                "under the same load conditions. Both pumps follow a stable periodic cycle."
            ),
            "series_A": _to_list(series_a),
            "series_B": _to_list(series_b),
            "external_variables": {
                "operational_events": "none",
                "sensor_flag": "Pump A sensor: calibration warning issued",
            },
        },
    )


def case_rare_contextual(n: int = 150, seed: int = 3) -> tuple:
    """
    Both series are financial-like (upward drift + volatility).
    Around day 70–85, a real market event causes a sharp V-shaped drawdown in B
    while A shows no response at all (e.g. a hedged instrument or heavily filtered feed).
    The event is confirmed by external context.

    Key tension: B's anomaly looks statistically unusual, but faithfully captures the
    real-world event; A's flat response may actually reflect lower data fidelity.
    """
    rng = np.random.default_rng(seed)

    # Shared base: gentle upward drift + correlated random walk
    drift = _trend(n, 10.0, 14.0)
    walk = np.cumsum(rng.normal(0, 0.15, n))
    base = drift + walk

    # A: no response to the event — completely flat through the window
    series_a = base.copy()
    event_start, event_end = 70, 85

    # B: sharp V-shaped crash — more faithful to real event, A shows nothing
    series_b = base.copy()
    dip_b = np.zeros(n)
    crash_bottom = 73
    for i in range(event_start, crash_bottom):
        t = (i - event_start) / (crash_bottom - event_start)
        dip_b[i] = -2.5 * t                        # steep drop
    for i in range(crash_bottom, event_end):
        t = (i - crash_bottom) / (event_end - crash_bottom)
        dip_b[i] = -2.5 * (1 - t)                  # rapid recovery
    series_b += dip_b

    # Add individual noise
    series_a += _noise(n, std=0.12, seed=seed + 1)
    series_b += _noise(n, std=0.18, seed=seed + 2)

    return (
        "rare_pattern | contextual anomaly — B captures real market crash, A shows no response",
        {
            "dataset_description": (
                "Daily closing prices of two technology sector indices over a 150-day window. "
                "Both instruments are exposed to the same macro environment."
            ),
            "series_A": _to_list(series_a),
            "series_B": _to_list(series_b),
            "external_variables": {
                "market_event": (
                    "A major regulatory announcement on day 70 triggered a sector-wide selloff. "
                    "The index recovered within two weeks. Event is well-documented in filings."
                ),
            },
        },
    )


# ── Pattern structure ──────────────────────────────────────────────────────────

def case_trend(n: int = 150, seed: int = 4) -> tuple:
    """
    Both series have an overall upward trend.
    B's trend becomes noisy and flat in the middle third, then resumes —
    reducing per-segment trend clarity without removing the overall direction.
    """
    trend = _trend(n, 0, 6.0)
    noise_a = _noise(n, std=0.3, seed=seed)
    series_a = trend + noise_a

    series_b = trend + _noise(n, std=0.3, seed=seed + 1)
    # Middle third: replace with flat noisy plateau (weakened trend clarity)
    mid_lo, mid_hi = n // 3, 2 * n // 3
    plateau_val = float(series_b[mid_lo])
    series_b[mid_lo:mid_hi] = plateau_val + _noise(mid_hi - mid_lo, std=0.9, seed=seed + 2)

    return (
        "pattern_structure | trend — B loses trend clarity mid-series",
        {
            "dataset_description": (
                "Daily energy output (MWh) from two wind farms of similar capacity "
                "over a 150-day period with a seasonal upward trend (increasing wind season)."
            ),
            "series_A": _to_list(series_a),
            "series_B": _to_list(series_b),
            "external_variables": {},
        },
    )


def case_frequency(n: int = 150, seed: int = 5) -> tuple:
    """
    Both series contain a dominant period-24 component.
    A has one clear primary frequency with a weak harmonic (low spectral entropy).
    B is a superposition of three frequencies of roughly equal strength —
    no single dominant frequency, so spectral entropy is high.
    """
    t = np.arange(n)

    # A: strong primary (period 24) + weak harmonic (period 8) — one dominant frequency
    series_a = (2.0 * np.sin(2 * math.pi * t / 24.0)
                + 0.3 * np.sin(2 * math.pi * t / 8.0)
                + _noise(n, std=0.2, seed=seed))

    # B: three components of comparable strength — no dominant frequency
    series_b = (1.0 * np.sin(2 * math.pi * t / 24.0)
                + 0.9 * np.sin(2 * math.pi * t / 15.0)
                + 0.8 * np.sin(2 * math.pi * t / 9.0)
                + _noise(n, std=0.2, seed=seed + 1))

    return (
        "pattern_structure | frequency — A has one dominant freq, B has multiple competing freqs",
        {
            "dataset_description": (
                "Hourly electricity demand (normalised) from two substations supplying "
                "similar residential areas. Both follow a daily consumption cycle (period ≈ 24h)."
            ),
            "series_A": _to_list(series_a),
            "series_B": _to_list(series_b),
            "external_variables": {},
        },
    )


def case_amplitude(n: int = 150, seed: int = 6) -> tuple:
    """
    Both series oscillate with period 20.
    A has consistent peak and trough values across all cycles (low amplitude_cv).
    B has the same period but each cycle's amplitude is drawn independently from a
    wide range (0.4–2.8), so peaks and troughs vary greatly from cycle to cycle.
    """
    period = 20.0
    rng = np.random.default_rng(seed + 3)

    # A: mostly consistent amplitude ~2.0 with slight per-cycle variation
    rng_a = np.random.default_rng(seed + 2)
    series_a = np.zeros(n)
    pos = 0
    while pos < n:
        amp = rng_a.uniform(1.7, 2.3)
        cycle_len = int(period)
        for i in range(cycle_len):
            if pos + i < n:
                series_a[pos + i] = amp * math.sin(2 * math.pi * i / period)
        pos += cycle_len
    series_a += _noise(n, std=0.15, seed=seed)

    # B: wider per-cycle amplitude variation — peaks and troughs noticeably inconsistent
    series_b = np.zeros(n)
    pos = 0
    while pos < n:
        amp = rng.uniform(0.8, 2.6)
        cycle_len = int(period)
        for i in range(cycle_len):
            if pos + i < n:
                series_b[pos + i] = amp * math.sin(2 * math.pi * i / period)
        pos += cycle_len
    series_b += _noise(n, std=0.15, seed=seed + 1)

    return (
        "pattern_structure | amplitude — B has consistent peaks/troughs, A varies widely per cycle",
        {
            "dataset_description": (
                "Pressure readings (bar) from two hydraulic pistons executing identical "
                "reciprocating cycles (period ≈ 20 steps)."
            ),
            "series_A": _to_list(series_b),
            "series_B": _to_list(series_a),
            "external_variables": {},
        },
    )


def case_pattern(n: int = 150, seed: int = 7) -> tuple:
    """
    Both series are stationary with similar mean and overall variance.
    A is smooth with consistent local variance.
    B has alternating low- and high-variance windows (lumpier),
    making it structurally less coherent — a subtler degradation.
    """
    rng = np.random.default_rng(seed)

    # A: AR(1) with moderate, uniform noise — some variability but structurally consistent
    series_a = np.zeros(n)
    series_a[0] = rng.normal(2.0, 0.3)
    for i in range(1, n):
        series_a[i] = 0.7 * series_a[i - 1] + rng.normal(0.6, 0.40)  # AR(1), mean≈2

    # B: same AR(1) base but variance alternates every ~20 steps (calm vs volatile)
    series_b = np.zeros(n)
    series_b[0] = rng.normal(2.0, 0.3)
    for i in range(1, n):
        window_idx = i // 20
        local_std = 0.20 if window_idx % 2 == 0 else 0.85   # alternating calm/volatile
        series_b[i] = 0.7 * series_b[i - 1] + rng.normal(0.6, local_std)

    return (
        "pattern_structure | pattern — A has alternating variance (lumpier)",
        {
            "dataset_description": (
                "Air quality index (PM2.5, normalised) measured at two monitoring stations "
                "in the same city district. Both stations record the same baseline air quality "
                "with a stable long-run mean."
            ),
            "series_A": _to_list(series_b),
            "series_B": _to_list(series_a),
            "external_variables": {
                "station_A_location": "roadside intersection",
                "station_B_location": "park (away from traffic)",
            },
        },
    )


# ── Registry ───────────────────────────────────────────────────────────────────

_ALL_CASES = [
    ("missing",         case_missing),
    ("noise",           case_noise),
    ("rare_point",      case_rare_point),
    ("rare_contextual", case_rare_contextual),
    ("trend",           case_trend),
    ("frequency",       case_frequency),
    ("amplitude",       case_amplitude),
    ("pattern",         case_pattern),
]

CASE_NAMES = [name for name, _ in _ALL_CASES]


def get_cases(cases=None) -> list[tuple[str, dict]]:
    """
    Return a list of (case_name, input_dict) tuples.

    Parameters
    ----------
    cases : str, list[str], or None.
            None returns all cases; a string or list filters by name(s).
    """
    if cases is None:
        return [fn() for _, fn in _ALL_CASES]
    if isinstance(cases, str):
        cases = [cases]
    selected = set(cases)
    return [fn() for name, fn in _ALL_CASES if name in selected]


# ── Visualisation interface ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Preview synthetic test cases")
    parser.add_argument(
        "--case", default=None,
        choices=CASE_NAMES,
        help="Show only this case by name (default: all). "
             f"Choices: {', '.join(CASE_NAMES)}",
    )
    parser.add_argument(
        "--out", default="plots/synthetic_cases",
        help="Directory to save plot images (default: plots/synthetic_cases)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Also display plots interactively (default: save only)",
    )
    args = parser.parse_args()

    cases = get_cases(args.case)
    os.makedirs(args.out, exist_ok=True)

    # Determine which case names map to the filtered results
    filtered_names = [
        name for name, fn in _ALL_CASES
        if args.case is None or name == args.case
    ]

    for case_name, (title, inp) in zip(filtered_names, cases):
        fig, ax = plt.subplots(figsize=(13, 3.2))

        a = [v if v is not None else float("nan") for v in inp["series_A"]]
        b = [v if v is not None else float("nan") for v in inp["series_B"]]
        x = list(range(len(a)))

        ax.plot(x, a, label="Series A", color="steelblue", linewidth=1.4)
        ax.plot(x, b, label="Series B", color="tomato", alpha=0.85, linewidth=1.2)
        ax.set_title(title, fontsize=9, pad=4)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("time step", fontsize=8)

        # Annotate missing values in B
        missing_b = [i for i, v in enumerate(b) if math.isnan(v)]
        if missing_b:
            ax.scatter(missing_b, [ax.get_ylim()[0]] * len(missing_b),
                       marker="|", color="tomato", s=30, label="_nolegend_")

        plt.tight_layout(pad=1.5)
        out_path = os.path.join(args.out, f"{case_name}.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        if args.show:
            plt.show()
        plt.close(fig)