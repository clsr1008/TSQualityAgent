"""
Data loader for Perceiver GRPO training.

Loads JSONL training samples and converts them to HuggingFace Datasets with
the exact prompt format the Perceiver sees at inference time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from agents.perceiver import SYSTEM_PROMPT


# ── Dimension-specific diagnostic indicators ─────────────────────────────────

def _clean(vals: list) -> np.ndarray:
    return np.array(
        [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))],
        dtype=float,
    )


def _compute_hint_stats(preview_a: list, preview_b: list,
                        stats_a: dict, stats_b: dict) -> dict | None:
    """
    Compute all dimension-specific diagnostic statistics comparing A vs B.

    Returns a dict of raw values, or None if series are too short to compute.

    Design rationale per dimension:
      missing_value      — missing_ratio: direct, unambiguous
      rare_pattern       — max_local_zscore: localized anomaly in sliding window
      noise_level        — rolling_std_mean + ratio
      amplitude          — std + ratio (catches both scaling-up and clipping)
      frequency          — spectral_corr: FFT-magnitude correlation between A and B
      trend              — slope + trend_strength = |slope|*n/std (dimensionless)
      pattern_consistency— autocorr_lag1
    """
    a, b = _clean(preview_a), _clean(preview_b)
    na, nb = len(a), len(b)
    if na < 4 or nb < 4:
        return None

    # ── missing_value ────────────────────────────────────────────────────────
    miss_a = stats_a.get("missing_ratio", 0.0)
    miss_b = stats_b.get("missing_ratio", 0.0)

    # ── rare_pattern: max local z-score ──────────────────────────────────────
    def max_local_zscore(arr):
        n = len(arr)
        win = max(10, n // 5)
        max_z = 0.0
        for i in range(0, n - win + 1, max(1, win // 2)):
            seg = arr[i:i + win]
            s = seg.std()
            if s > 0:
                max_z = max(max_z, float(np.max(np.abs((seg - seg.mean()) / s))))
        return round(max_z, 2)

    mlz_a = max_local_zscore(a)
    mlz_b = max_local_zscore(b)
    mlz_ratio = max(mlz_a, mlz_b) / min(mlz_a, mlz_b) if min(mlz_a, mlz_b) > 1e-9 else 1.0

    # ── noise_level: rolling std mean + ratio ────────────────────────────────
    def rolling_std_mean(arr):
        n = len(arr)
        win = max(5, n // 10)
        return float(np.array([arr[i:i + win].std() for i in range(n - win + 1)]).mean())

    rs_a = rolling_std_mean(a)
    rs_b = rolling_std_mean(b)
    rs_ratio = max(rs_a, rs_b) / min(rs_a, rs_b) if min(rs_a, rs_b) > 1e-9 else 1.0

    # ── amplitude: std + ratio ────────────────────────────────────────────────
    std_a, std_b = float(a.std()), float(b.std())
    amp_ratio = max(std_a, std_b) / min(std_a, std_b) if min(std_a, std_b) > 1e-9 else 1.0

    # ── frequency: spectral correlation ──────────────────────────────────────
    fa = np.abs(np.fft.rfft(a - a.mean()))
    fb = np.abs(np.fft.rfft(b - b.mean()))
    min_len = min(len(fa), len(fb))
    fa_, fb_ = fa[:min_len].copy(), fb[:min_len].copy()
    fa_[0] = fb_[0] = 0  # suppress DC
    if fa_.std() > 1e-9 and fb_.std() > 1e-9:
        spec_corr = round(float(np.corrcoef(fa_, fb_)[0, 1]), 4)
    else:
        spec_corr = 1.0

    # ── trend: slope + trend strength ────────────────────────────────────────
    slope_a = stats_a.get("slope", 0.0) or 0.0
    slope_b = stats_b.get("slope", 0.0) or 0.0
    ts_a = round(abs(slope_a) * na / std_a, 2) if std_a > 1e-9 else 0.0
    ts_b = round(abs(slope_b) * nb / std_b, 2) if std_b > 1e-9 else 0.0

    # ── pattern_consistency: lag-1 autocorrelation ───────────────────────────
    ac_a = round(float(np.corrcoef(a[:-1], a[1:])[0, 1]), 4) if na > 2 else 0.0
    ac_b = round(float(np.corrcoef(b[:-1], b[1:])[0, 1]), 4) if nb > 2 else 0.0
    ac_diff = round(abs(ac_a - ac_b), 4)

    return dict(
        miss_a=miss_a, miss_b=miss_b,
        mlz_a=mlz_a, mlz_b=mlz_b, mlz_ratio=mlz_ratio,
        rs_a=rs_a, rs_b=rs_b, rs_ratio=rs_ratio,
        std_a=std_a, std_b=std_b, amp_ratio=amp_ratio,
        spec_corr=spec_corr,
        slope_a=slope_a, slope_b=slope_b, ts_a=ts_a, ts_b=ts_b,
        ac_a=ac_a, ac_b=ac_b, ac_diff=ac_diff,
    )


# ── Hint-label consistency thresholds ────────────────────────────────────────
#
# Two-tier design:
#   REMOVE threshold (looser, ~1.3x): remove a GT label when its hint stat is
#     below this — the injected defect left no detectable signal, producing a
#     contradictory "hint says don't select, GT says select" training signal.
#
#   ADD threshold (stricter, ~1.8x / tighter on other metrics): add a missing
#     GT label when the hint stat exceeds this — a cross-contamination effect
#     or incidental side-effect of another injection created a real, detectable
#     difference that the model would be penalised for selecting without this.
#
# Using asymmetric thresholds prevents noise: the bar to ADD a label is higher
# than the bar to KEEP one.

_HINT_REMOVE: dict[str, object] = {
    "missing_value":       lambda s: max(s["miss_a"], s["miss_b"]) > 0.01,
    "noise_level":         lambda s: s["rs_ratio"] > 1.3,
    "amplitude":           lambda s: s["amp_ratio"] > 1.3,
    "rare_pattern":        lambda s: s["mlz_ratio"] > 1.3,
    "frequency":           lambda s: s["spec_corr"] < 0.90,
    "trend":               lambda s: max(s["ts_a"], s["ts_b"]) > 1.0,
    "pattern_consistency": lambda s: s["ac_diff"] > 0.10,
}

_HINT_ADD: dict[str, object] = {
    "missing_value":       lambda s: max(s["miss_a"], s["miss_b"]) > 0.03,
    "noise_level":         lambda s: s["rs_ratio"] > 1.8,
    "amplitude":           lambda s: s["amp_ratio"] > 1.8,
    "rare_pattern":        lambda s: s["mlz_ratio"] > 1.8,
    "frequency":           lambda s: s["spec_corr"] < 0.80,
    "trend":               lambda s: max(s["ts_a"], s["ts_b"]) > 2.0,
    "pattern_consistency": lambda s: s["ac_diff"] > 0.20,
}

_ALL_DIMENSIONS = [
    "missing_value", "noise_level", "rare_pattern",
    "trend", "frequency", "amplitude", "pattern_consistency",
]


def verify_hint_label_consistency(
    preview_a: list, preview_b: list,
    stats_a: dict, stats_b: dict,
    target_dims: list[str],
) -> list[str]:
    """
    Return a bidirectionally corrected label list aligned with hint statistics.

    - REMOVE: GT label whose hint stat is below the remove-threshold
      (contradictory signal: hint says "don't select" but GT says "select").
    - ADD: dimension not in GT whose hint stat exceeds the add-threshold
      (cross-contamination: another injection incidentally created a detectable
      difference that the model would be penalised for selecting).

    Returns *target_dims* unchanged if stats cannot be computed (too few points).
    """
    hint_stats = _compute_hint_stats(preview_a, preview_b, stats_a, stats_b)
    if hint_stats is None:
        return target_dims

    gt_set = set(target_dims)

    # Keep dims that pass the (looser) remove threshold
    verified = [d for d in target_dims
                if (chk := _HINT_REMOVE.get(d)) is None or chk(hint_stats)]

    # Add dims not in GT that pass the (stricter) add threshold
    verified_set = set(verified)
    for dim in _ALL_DIMENSIONS:
        if dim not in gt_set and dim not in verified_set:
            chk = _HINT_ADD.get(dim)
            if chk is not None and chk(hint_stats):
                verified.append(dim)

    return verified


def _build_hint_section(preview_a: list, preview_b: list,
                        stats_a: dict, stats_b: dict) -> str:
    """Format dimension-specific diagnostic indicators as a prompt string."""
    s = _compute_hint_stats(preview_a, preview_b, stats_a, stats_b)
    if s is None:
        return ""

    def _pct(v): return f"{v:.1%}"
    def _f4(v):  return f"{v:.4f}"
    def _f2(v):  return f"{v:.2f}"

    lines = [
        "Dimension-specific indicators (A  →  B, larger gap = more likely relevant):",
        f"  missing_ratio    : {_pct(s['miss_a']):>7}  →  {_pct(s['miss_b']):<7}   [missing_value]",
        f"  max_local_zscore : {_f2(s['mlz_a']):>7}  →  {_f2(s['mlz_b']):<7}   [rare_pattern]   (max/min ratio={s['mlz_ratio']:.2f}x, select if ratio>1.5)",
        f"  rolling_std_mean : {_f4(s['rs_a']):>7}  →  {_f4(s['rs_b']):<7}   [noise_level]    (ratio={s['rs_ratio']:.2f}x, select if ratio>1.5)",
        f"  std              : {_f4(s['std_a']):>7}  →  {_f4(s['std_b']):<7}   [amplitude]      (ratio={s['amp_ratio']:.2f}x, select if ratio>1.5)",
        f"  spectral_corr    : {s['spec_corr']:.4f} (A\u2194B)                  [frequency]      (lower = more different, select if <0.85)",
        f"  slope            : {_f4(s['slope_a']):>7}  →  {_f4(s['slope_b']):<7}   [trend]          (strength A={s['ts_a']} B={s['ts_b']})",
        f"  autocorr_lag1    : {_f4(s['ac_a']):>7}  →  {_f4(s['ac_b']):<7}   [pattern_consistency] (diff={s['ac_diff']})",
    ]
    return "\n".join(lines)


# ── Prompt construction (mirrors agents/perceiver.py:112-156) ────────────────

def _build_user_message(sample_input: dict, n_dims: int | None = None) -> str:
    """Reconstruct the user message exactly as perceiver.py does.

    Parameters
    ----------
    sample_input : dict
        Raw sample input from the JSONL file.
    n_dims : int | None
        If provided, append a line telling the model exactly how many dimensions
        to select.  Used in diagnostic evals to separate the "how many" sub-task
        from the "which ones" sub-task.
    """
    inp = sample_input
    preview_A = inp["preview_A"]
    preview_B = inp["preview_B"]
    stats_A = inp["stats_A"]
    stats_B = inp["stats_B"]
    desc = inp.get("dataset_description", "Not provided.")

    len_A = stats_A.get("length", len(preview_A))
    len_B = stats_B.get("length", len(preview_B))
    sampled_A = ", sampled" if len_A > 200 else ""
    sampled_B = ", sampled" if len_B > 200 else ""

    hint_section = _build_hint_section(preview_A, preview_B, stats_A, stats_B)

    n_dims_line = (
        f"\nNote: exactly {n_dims} quality dimension(s) should be selected.\n"
        if n_dims is not None else ""
    )

    return f"""Dataset description: {desc}

Series A — values ({len_A} total{sampled_A}):
{json.dumps(preview_A)}

Series A statistics:
{json.dumps(stats_A, indent=2)}

Series B — values ({len_B} total{sampled_B}):
{json.dumps(preview_B)}

Series B statistics:
{json.dumps(stats_B, indent=2)}

{hint_section}

External variables: {{}}
{n_dims_line}
Based on the above, decide which quality dimensions to assess.
Remember: output ONLY valid JSON as specified."""


def build_prompt_messages(sample: dict, n_dims: int | None = None) -> list[dict]:
    """Build chat messages list for a single sample."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(sample["input"], n_dims=n_dims)},
    ]


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(jsonl_path: str, n_dims_hint: bool = False) -> Dataset:
    """
    Load JSONL and return a HuggingFace Dataset.

    Columns:
      - prompt: list[dict]  (chat messages for tokenizer.apply_chat_template)
      - target_dimensions: list[str]
      - sample_id: str

    Parameters
    ----------
    n_dims_hint : bool
        If True, inject the GT dimension count into the user message so the
        model only needs to decide *which* dimensions to select (not *how many*).
        Useful for diagnostic evals.
    """
    records = []
    path = Path(jsonl_path)
    raw = path.read_bytes()

    # Skip BOM or other leading non-JSON bytes
    start_idx = raw.find(b"{")
    if start_idx < 0:
        raise ValueError(f"No JSON found in {jsonl_path}")
    text = raw[start_idx:].decode("utf-8")

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        target_dims = sample["labels"]["target_dimensions"]
        n_dims = len(target_dims) if n_dims_hint else None
        messages = build_prompt_messages(sample, n_dims=n_dims)
        records.append({
            "prompt": messages,
            "target_dimensions": target_dims,
            "sample_id": sample["sample_id"],
        })

    return Dataset.from_list(records)
