"""
Training sample generator for Perceiver.

Generates (input, label) pairs where:
  - input  = series previews + stats + description (same as Perceiver sees at inference)
  - labels = target_dimensions + tool_required
"""
import numpy as np

from training.synthesis.base_generator import generate_random_base
from training.synthesis.defect_injector import inject_defect, ALL_DIMENSIONS
from training.synthesis.label_schema import (
    SEVERITIES, N_DIM_WEIGHTS, needs_tool,
)


# ── Input feature helpers (mirrored from agents/perceiver.py) ────────────────

def _basic_stats(series: list) -> dict:
    """Compute basic statistics — same as Perceiver sees at inference."""
    arr = np.array(series, dtype=float)
    valid = arr[~np.isnan(arr)]

    if len(valid) >= 2:
        x = np.where(~np.isnan(arr))[0].astype(float)
        slope = float(np.polyfit(x, valid, 1)[0])
    else:
        slope = None

    return {
        "length": len(arr),
        "missing_ratio": round(1 - len(valid) / len(arr), 4) if len(arr) > 0 else None,
        "mean": round(float(np.mean(valid)), 4) if len(valid) else None,
        "std": round(float(np.std(valid)), 4) if len(valid) else None,
        "min": round(float(np.min(valid)), 4) if len(valid) else None,
        "max": round(float(np.max(valid)), 4) if len(valid) else None,
        "p25": round(float(np.percentile(valid, 25)), 4) if len(valid) else None,
        "p75": round(float(np.percentile(valid, 75)), 4) if len(valid) else None,
        "slope": round(slope, 6) if slope is not None else None,
    }


def _series_preview(series: list, max_full: int = 200, sample_size: int = 60) -> list:
    """Return full series if short; otherwise head + middle sample + tail."""
    arr = [None if (v is None or (isinstance(v, float) and v != v)) else round(v, 4)
           for v in series]
    if len(arr) <= max_full:
        return arr
    head = arr[:20]
    tail = arr[-20:]
    middle_n = sample_size - 40
    step = max(1, (len(arr) - 40) // middle_n)
    middle = arr[20:-20:step][:middle_n]
    return head + middle + tail


# ── Series conversion ────────────────────────────────────────────────────────

def _round_series(arr: np.ndarray) -> list[float | None]:
    """Convert numpy array to JSON-safe list, NaN → None."""
    out = []
    for v in arr:
        if np.isnan(v):
            out.append(None)
        else:
            out.append(round(float(v), 4))
    return out


# ── Sample generation ────────────────────────────────────────────────────────

def generate_sample(
    seed: int,
    n_min: int = 100,
    n_max: int = 150,
) -> dict:
    """
    Generate a single Perceiver training sample.

    Series length is sampled uniformly from [n_min, n_max] so both A and B
    share the same randomly chosen length within each sample.

    Returns a dict with keys: sample_id, input, labels, meta.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(n_min, n_max + 1))

    # 1. Generate base series
    base, attr_pool, desc = generate_random_base(n=n, seed=seed)

    # 2. Sample number of dimensions to inject
    n_dims_options = list(N_DIM_WEIGHTS.keys())
    n_dims_probs = np.array(list(N_DIM_WEIGHTS.values()), dtype=float)
    n_dims_probs /= n_dims_probs.sum()
    n_dims = int(rng.choice(n_dims_options, p=n_dims_probs))
    n_dims = min(n_dims, len(ALL_DIMENSIONS))

    # 3. Pick which dimensions
    chosen_dims = rng.choice(ALL_DIMENSIONS, size=n_dims, replace=False).tolist() if n_dims > 0 else []

    # 4. Inject defects
    series_A = base.copy()
    series_B = base.copy()
    defect_details = []
    target_dimensions = []
    tool_required = []

    for j, dim in enumerate(chosen_dims):
        severity = str(rng.choice(SEVERITIES))
        side = str(rng.choice(["A", "B"]))

        if side == "B":
            series_B, meta = inject_defect(
                series_B, dim, severity,
                seed=seed + j + 100,
                base_period=attr_pool.get("seasonal", {}).get("period"),
            )
        else:
            series_A, meta = inject_defect(
                series_A, dim, severity,
                seed=seed + j + 100,
                base_period=attr_pool.get("seasonal", {}).get("period"),
            )

        target_dimensions.append(dim)
        if needs_tool(severity):
            tool_required.append(dim)

        defect_details.append({
            "dimension": dim,
            "severity": severity,
            "side": side,
            "metadata": meta,
        })

    # Add small independent noise to both sides
    base_noise_std = attr_pool.get("noise", {}).get("std", 0.01)
    series_A = series_A + rng.normal(0, base_noise_std * rng.uniform(0.4, 0.6), n)
    series_B = series_B + rng.normal(0, base_noise_std * rng.uniform(0.4, 0.6), n)

    # 5. Build input features (same as Perceiver sees)
    series_A_list = _round_series(series_A)
    series_B_list = _round_series(series_B)
    stats_A = _basic_stats(series_A_list)
    stats_B = _basic_stats(series_B_list)
    preview_A = _series_preview(series_A_list)
    preview_B = _series_preview(series_B_list)

    # 6. Build sample ID
    dim_str = "_".join(sorted(chosen_dims)) if chosen_dims else "tie"
    sample_id = f"seed{seed}_{dim_str}"

    return {
        "sample_id": sample_id,
        "input": {
            "dataset_description": desc,
            "preview_A": preview_A,
            "preview_B": preview_B,
            "stats_A": stats_A,
            "stats_B": stats_B,
        },
        "labels": {
            "target_dimensions": target_dimensions,
            "tool_required": tool_required,
        },
        "meta": {
            "defect_details": defect_details,
            "base_attributes": attr_pool,
        },
    }


def generate_batch(
    n_samples: int,
    seed_offset: int = 0,
) -> list[dict]:
    """Generate a batch of training samples."""
    samples = []
    for i in range(n_samples):
        seed = seed_offset + i
        sample = generate_sample(seed=seed)
        samples.append(sample)
    return samples