"""
Synthetic pairwise test cases for TSqualityAgent.

Each case isolates one quality aspect so the agent's behaviour
can be validated dimension by dimension.
In every case series A is the "good" reference; series B degrades
only the targeted aspect while keeping everything else identical.

Usage:
    from synthetic_cases import get_cases
    cases = get_cases()          # list of (name, input_dict)
    cases = get_cases("bad")     # filter by aspect tag
"""
import math
import random


# ── Reproducible base signal generators ───────────────────────────────────────

def _linspace(start: float, stop: float, n: int) -> list[float]:
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _sine(n: int, cycles: float = 2.0, amplitude: float = 1.0) -> list[float]:
    return [amplitude * math.sin(2 * math.pi * cycles * i / n) for i in range(n)]


def _noise(n: int, std: float, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, std) for _ in range(n)]


def _add(a: list, b: list) -> list:
    return [x + y for x, y in zip(a, b)]


# ── Bad quality cases ──────────────────────────────────────────────────────────

def case_missing_value(n: int = 100, missing_ratio: float = 0.25, seed: int = 1) -> tuple:
    """
    B has ~25% missing values; A is clean.
    Only dimension affected: missing_value.
    """
    base = _add(_linspace(0, 5, n), _noise(n, std=0.2, seed=seed))

    series_a = base[:]

    rng = random.Random(seed)
    missing_idx = set(rng.sample(range(n), k=int(n * missing_ratio)))
    series_b = [float("nan") if i in missing_idx else v for i, v in enumerate(base)]

    return (
        "bad_quality | missing_value (B has 25% NaN)",
        {
            "dataset_description": f"Synthetic linear trend, n={n}, mild noise.",
            "series_A": series_a,
            "series_B": series_b,
            "external_variables": {},
        },
    )


def case_noise_level(n: int = 100, seed: int = 2) -> tuple:
    """
    B has 8× higher noise than A; trend is identical.
    Only dimension affected: noise_level.
    """
    trend = _linspace(0, 5, n)
    series_a = _add(trend, _noise(n, std=0.2, seed=seed))
    series_b = _add(trend, _noise(n, std=1.6, seed=seed + 1))

    return (
        "bad_quality | noise_level (B has 8× noise)",
        {
            "dataset_description": f"Synthetic linear trend, n={n}, varying noise.",
            "series_A": series_a,
            "series_B": series_b,
            "external_variables": {},
        },
    )


# ── Rare pattern cases ─────────────────────────────────────────────────────────

def case_anomaly(n: int = 100, n_anomalies: int = 5, magnitude: float = 8.0, seed: int = 3) -> tuple:
    """
    B has point anomalies injected at fixed positions; A is clean.
    Only dimension affected: rare_pattern.
    """
    base = _add(_linspace(0, 3, n), _noise(n, std=0.3, seed=seed))
    series_a = base[:]

    rng = random.Random(seed)
    anomaly_idx = rng.sample(range(10, n - 10), k=n_anomalies)
    series_b = base[:]
    for i in anomaly_idx:
        series_b[i] += magnitude * (1 if i % 2 == 0 else -1)

    return (
        f"rare_pattern | anomaly (B has {n_anomalies} point anomalies, magnitude={magnitude})",
        {
            "dataset_description": f"Synthetic flat-trend signal, n={n}.",
            "series_A": series_a,
            "series_B": series_b,
            "external_variables": {},
        },
    )


def case_outlier_density(n: int = 100, seed: int = 4) -> tuple:
    """
    B is drawn from a heavy-tailed distribution (high outlier density);
    A is near-Gaussian. Mean and trend identical.
    Only dimension affected: outlier density.
    """
    rng = random.Random(seed)
    trend = _linspace(0, 3, n)

    # A: Gaussian noise
    series_a = [t + rng.gauss(0, 0.4) for t in trend]

    # B: mix of Gaussian and occasional large jumps (Laplace-like)
    series_b = []
    for t in trend:
        if rng.random() < 0.12:                         # 12% chance of large excursion
            series_b.append(t + rng.uniform(3, 6) * rng.choice([-1, 1]))
        else:
            series_b.append(t + rng.gauss(0, 0.4))

    return (
        "rare_pattern | outlier_density (B has heavy-tailed distribution)",
        {
            "dataset_description": f"Synthetic linear trend, n={n}.",
            "series_A": series_a,
            "series_B": series_b,
            "external_variables": {},
        },
    )


# ── Pattern structure cases ────────────────────────────────────────────────────

def case_trend(n: int = 100, seed: int = 5) -> tuple:
    """
    A has a clear upward trend; B is mean-stationary (no trend).
    Only dimension affected: trend.
    """
    noise = _noise(n, std=0.3, seed=seed)
    series_a = _add(_linspace(0, 8, n), noise)
    series_b = _add([4.0] * n, noise)          # same noise, no trend

    return (
        "pattern_structure | trend (A has strong trend, B is flat)",
        {
            "dataset_description": f"Synthetic signal, n={n}.",
            "series_A": series_a,
            "series_B": series_b,
            "external_variables": {},
        },
    )


def case_seasonality(n: int = 120, seed: int = 6) -> tuple:
    """
    A has a clear periodic pattern; B is pure noise with the same variance.
    Only dimension affected: frequency / seasonality.
    """
    noise = _noise(n, std=0.25, seed=seed)
    series_a = _add(_sine(n, cycles=4, amplitude=2.0), noise)
    series_b = _noise(n, std=2.0, seed=seed + 1)   # same std, no structure

    return (
        "pattern_structure | frequency (A has clear seasonality, B is noise)",
        {
            "dataset_description": f"Synthetic periodic vs. random signal, n={n}.",
            "series_A": series_a,
            "series_B": series_b,
            "external_variables": {},
        },
    )


def case_amplitude_spikes(n: int = 100, seed: int = 7) -> tuple:
    """
    B has frequent large spikes; A has smooth amplitude variation.
    Only dimension affected: amplitude / spike.
    """
    base = _sine(n, cycles=2, amplitude=1.0)
    noise = _noise(n, std=0.15, seed=seed)
    series_a = _add(base, noise)

    rng = random.Random(seed)
    series_b = _add(base, noise)[:]
    spike_positions = rng.sample(range(5, n - 5), k=8)
    for i in spike_positions:
        series_b[i] += rng.uniform(4, 7) * rng.choice([-1, 1])

    return (
        "pattern_structure | amplitude (B has frequent spikes)",
        {
            "dataset_description": f"Synthetic sinusoidal signal, n={n}.",
            "series_A": series_a,
            "series_B": series_b,
            "external_variables": {},
        },
    )


def case_pattern_consistency(n: int = 120, seed: int = 8) -> tuple:
    """
    A is stationary with consistent variance; B has two distinct regimes
    (a structural change point at the midpoint).
    Only dimension affected: pattern_consistency / change_point.
    """
    rng = random.Random(seed)
    mid = n // 2

    series_a = [rng.gauss(2.0, 0.4) for _ in range(n)]

    # B: first half ~ N(2, 0.4), second half ~ N(6, 1.5)
    series_b = (
        [rng.gauss(2.0, 0.4) for _ in range(mid)]
        + [rng.gauss(6.0, 1.5) for _ in range(n - mid)]
    )

    return (
        "pattern_structure | consistency (B has a structural change point at midpoint)",
        {
            "dataset_description": f"Synthetic stationary vs. regime-shift signal, n={n}.",
            "series_A": series_a,
            "series_B": series_b,
            "external_variables": {},
        },
    )


# ── Registry ───────────────────────────────────────────────────────────────────

_ALL_CASES = [
    ("bad",      case_missing_value),
    ("bad",      case_noise_level),
    ("rare",     case_anomaly),
    ("rare",     case_outlier_density),
    ("pattern",  case_trend),
    ("pattern",  case_seasonality),
    ("pattern",  case_amplitude_spikes),
    ("pattern",  case_pattern_consistency),
]


def get_cases(aspect: str = None) -> list[tuple[str, dict]]:
    """
    Return a list of (case_name, input_dict) tuples.

    Parameters
    ----------
    aspect : "bad" | "rare" | "pattern" | None
        Filter by quality aspect. None returns all cases.
    """
    results = []
    for tag, fn in _ALL_CASES:
        if aspect is None or tag == aspect:
            results.append(fn())
    return results