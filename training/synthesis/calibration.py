"""
Severity calibration experiment.

For each dimension × method, sweep the key parameter across a range and ask
the LLM (preview + stats only, NO tools) to identify which side is degraded.
Accuracy across N pairs per parameter value reveals where the light/heavy
boundary should sit.

Usage
-----
# cloud (default)
python -m training.synthesis.calibration --model gpt-4o-mini

# local vLLM
python -m training.synthesis.calibration \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY

# single dimension
python -m training.synthesis.calibration --dim noise_level

# quick test (fewer pairs)
python -m training.synthesis.calibration --n_pairs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Allow running as `python -m training.synthesis.calibration` from repo root ─────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.synthesis.base_generator import generate_random_base
from training.synthesis.defect_injector import inject_defect, _VARIANT_FUNCTIONS, DEFECT_VARIANTS
from models.llm import OpenAICompatibleLLM, CHATANYWHERE_BASE_URL


# ─────────────────────────────────────────────────────────────────────────────
# Sweep configurations — one entry per (dimension, method, parameter)
# Each multi-parameter method gets one entry per parameter; other params are
# held at a representative mid-severity fixed value so only the swept param
# varies.
# key_param : the parameter being swept
# values    : ordered low → high (increasing severity, except decay/clip)
# fixed     : other required params held constant at mid-severity values
# NOTE: for decay/clip, "higher severity" means lower numeric value
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_CONFIGS: dict[str, list[dict]] = {
    "missing_value": [
        # random_scatter — 1 param
        {
            "method": "random_scatter",
            "key_param": "ratio",
            "values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20],
            "fixed": {},
        },
        # burst — 2 params: burst_ratio + n_bursts
        {
            "method": "burst",
            "key_param": "burst_ratio",
            "values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16],
            "fixed": {"n_bursts": 2},
        },
        {
            "method": "burst",
            "key_param": "n_bursts",
            "values": [1, 2, 3, 5, 8, 12],
            "fixed": {"burst_ratio": 0.05},
        },
        # periodic — 2 params: ratio + gap_len
        {
            "method": "periodic",
            "key_param": "ratio",
            "values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16],
            "fixed": {"gap_len": 2},
        },
        {
            "method": "periodic",
            "key_param": "gap_len",
            "values": [1, 2, 3, 5, 8, 12],
            "fixed": {"ratio": 0.06},
        },
    ],

    "noise_level": [
        # gaussian — 1 param
        {
            "method": "gaussian",
            "key_param": "multiplier",
            "values": [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            "fixed": {},
        },
        # heteroscedastic — 2 params: multiplier + affected_ratio
        {
            "method": "heteroscedastic",
            "key_param": "multiplier",
            "values": [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            "fixed": {"affected_ratio": 0.5},
        },
        {
            "method": "heteroscedastic",
            "key_param": "affected_ratio",
            "values": [0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
            "fixed": {"multiplier": 2.5},
        },
        # impulsive — 2 params: multiplier + burst_ratio
        {
            "method": "impulsive",
            "key_param": "multiplier",
            "values": [1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
            "fixed": {"burst_ratio": 0.20},
        },
        {
            "method": "impulsive",
            "key_param": "burst_ratio",
            "values": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
            "fixed": {"multiplier": 3.0},
        },
    ],

    "rare_pattern": [
        # point_outlier — 2 params: sigma + count
        {
            "method": "point_outlier",
            "key_param": "sigma",
            "values": [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
            "fixed": {"count": 3},
        },
        {
            "method": "point_outlier",
            "key_param": "count",
            "values": [1, 2, 3, 5, 7, 10],
            "fixed": {"sigma": 4.0},
        },
        # contextual — 3 params: sigma + count + duration
        {
            "method": "contextual",
            "key_param": "sigma",
            "values": [2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
            "fixed": {"count": 2, "duration": 5},
        },
        {
            "method": "contextual",
            "key_param": "count",
            "values": [1, 2, 3, 4, 5],
            "fixed": {"sigma": 3.5, "duration": 5},
        },
        {
            "method": "contextual",
            "key_param": "duration",
            "values": [2, 3, 5, 7, 10, 14],
            "fixed": {"sigma": 3.5, "count": 2},
        },
        # level_shift — 3 params: sigma + count + duration
        {
            "method": "level_shift",
            "key_param": "sigma",
            "values": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            "fixed": {"count": 2, "duration": 10},
        },
        {
            "method": "level_shift",
            "key_param": "count",
            "values": [1, 2, 3, 4],
            "fixed": {"sigma": 3.0, "duration": 10},
        },
        {
            "method": "level_shift",
            "key_param": "duration",
            "values": [3, 5, 8, 10, 12, 15],
            "fixed": {"sigma": 3.0, "count": 2},
        },
    ],

    "trend": [
        # flatten — 2 params: flatten_ratio + noise_boost
        {
            "method": "flatten",
            "key_param": "flatten_ratio",
            "values": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
            "fixed": {"noise_boost": 1.5},
        },
        {
            "method": "flatten",
            "key_param": "noise_boost",
            "values": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
            "fixed": {"flatten_ratio": 0.20},
        },
        # drift — 1 param
        {
            "method": "drift",
            "key_param": "drift_strength",
            "values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
            "fixed": {},
        },
        # reversal — 1 param
        {
            "method": "reversal",
            "key_param": "reverse_ratio",
            "values": [0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
            "fixed": {},
        },
    ],

    "frequency": [
        # competing — 2 params: competing_amplitude + n_competing
        {
            "method": "competing",
            "key_param": "competing_amplitude",
            "values": [0.10, 0.20, 0.35, 0.50, 0.70, 0.90],
            "fixed": {"n_competing": 1},
        },
        {
            "method": "competing",
            "key_param": "n_competing",
            "values": [1, 2, 3, 4],
            "fixed": {"competing_amplitude": 0.40},
        },
        # jitter — 1 param
        {
            "method": "jitter",
            "key_param": "jitter_cv",
            "values": [0.02, 0.05, 0.08, 0.12, 0.18, 0.25],
            "fixed": {},
        },
        # period_shift — 2 params: shift_ratio + period_factor
        {
            "method": "period_shift",
            "key_param": "shift_ratio",
            "values": [0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
            "fixed": {"period_factor": 0.70},
        },
        {
            "method": "period_shift",
            "key_param": "period_factor",
            # lower value = larger period change = more severe
            "values": [0.95, 0.85, 0.75, 0.65, 0.55, 0.45],
            "fixed": {"shift_ratio": 0.30},
            "note": "lower period_factor = more severe",
        },
    ],

    "amplitude": [
        # random_scale — 1 param
        {
            "method": "random_scale",
            "key_param": "cv",
            "values": [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00],
            "fixed": {},
        },
        # decay — 1 param (lower = more severe)
        {
            "method": "decay",
            "key_param": "decay_factor",
            "values": [0.80, 0.70, 0.60, 0.50, 0.35, 0.20],
            "fixed": {},
            "note": "lower decay_factor = more severe",
        },
        # clip — 1 param (lower = more severe)
        {
            "method": "clip",
            "key_param": "clip_ratio",
            "values": [0.90, 0.80, 0.70, 0.60, 0.50, 0.40],
            "fixed": {},
            "note": "lower clip_ratio = more severe",
        },
    ],

    "pattern_consistency": [
        {
            "method": "variance_switching",
            "key_param": "intensity",
            "values": [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00],
            "fixed": {},
        },
        {
            "method": "structural_break",
            "key_param": "intensity",
            "values": [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00],
            "fixed": {},
        },
        {
            "method": "flat_spots",
            "key_param": "intensity",
            "values": [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00],
            "fixed": {},
        },
        {
            "method": "mean_drift",
            "key_param": "drift_std",
            "values": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
            "fixed": {},
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers (same as sample_generator.py)
# ─────────────────────────────────────────────────────────────────────────────

def _basic_stats(series: list) -> dict:
    arr = np.array(series, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) >= 2:
        x = np.where(~np.isnan(arr))[0].astype(float)
        slope = float(np.polyfit(x, valid, 1)[0])
    else:
        slope = None
    return {
        "length": len(arr),
        "mean": round(float(np.mean(valid)), 4) if len(valid) else None,
        "std": round(float(np.std(valid)), 4) if len(valid) else None,
        "min": round(float(np.min(valid)), 4) if len(valid) else None,
        "max": round(float(np.max(valid)), 4) if len(valid) else None,
        "p25": round(float(np.percentile(valid, 25)), 4) if len(valid) else None,
        "p75": round(float(np.percentile(valid, 75)), 4) if len(valid) else None,
        "nan_count": int(np.isnan(arr).sum()),
        "slope": round(slope, 6) if slope is not None else None,
        "head": [None if np.isnan(v) else round(v, 4) for v in arr[:8].tolist()],
        "tail": [None if np.isnan(v) else round(v, 4) for v in arr[-8:].tolist()],
    }


def _preview(arr: np.ndarray, max_pts: int = 60) -> list:
    lst = [None if np.isnan(v) else round(float(v), 4) for v in arr]
    if len(lst) <= max_pts:
        return lst
    head = lst[:15]
    tail = lst[-15:]
    mid_n = max_pts - 30
    step = max(1, (len(lst) - 30) // mid_n)
    mid = lst[15:-15:step][:mid_n]
    return head + mid + tail


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(dim: str, method: str, key_param: str, param_val: float,
                  arr_a: np.ndarray, arr_b: np.ndarray, desc: str) -> str:
    stats_a = _basic_stats(arr_a.tolist())
    stats_b = _basic_stats(arr_b.tolist())
    prev_a = _preview(arr_a)
    prev_b = _preview(arr_b)

    return f"""You are a time-series quality expert.

Dataset context: {desc}

Below are two time series (A and B). One of them has been degraded in the
"{dim}" dimension using the "{method}" injection method. The other is clean.

Series A preview (up to 60 points): {json.dumps(prev_a)}
Series A stats: {json.dumps(stats_a)}

Series B preview (up to 60 points): {json.dumps(prev_b)}
Series B stats: {json.dumps(stats_b)}

Task: Based ONLY on the preview values and stats above (no tools available),
decide which series has the lower quality with respect to the "{dim}" dimension.

Respond with a JSON object on a single line:
{{"answer": "A" or "B", "confidence": 0.0-1.0, "reasoning": "brief reason"}}

Do not output anything else."""


# ─────────────────────────────────────────────────────────────────────────────
# Single pair evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(llm: OpenAICompatibleLLM, prompt: str) -> dict:
    """Call LLM and parse JSON response. Returns dict with answer/confidence."""
    resp = llm.chat([{"role": "user", "content": prompt}])
    raw = resp.content.strip()
    # Try to find JSON object in response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    # Fallback: look for A or B
    upper = raw.upper()
    if '"A"' in upper or "'A'" in upper or upper.strip().startswith("A"):
        return {"answer": "A", "confidence": 0.5, "reasoning": raw}
    if '"B"' in upper or "'B'" in upper or upper.strip().startswith("B"):
        return {"answer": "B", "confidence": 0.5, "reasoning": raw}
    return {"answer": "?", "confidence": 0.0, "reasoning": raw}


def _evaluate_pair(
    llm: OpenAICompatibleLLM,
    dim: str,
    method: str,
    key_param: str,
    param_val: float,
    fixed_params: dict,
    seed: int,
) -> dict:
    """Generate one clean/degraded pair, ask LLM, return result dict."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(100, 151))
    base_arr, attr_pool, desc = generate_random_base(n=n, seed=seed)

    # Build kwargs for injection function
    fn = _VARIANT_FUNCTIONS[dim][method]
    kwargs = {key_param: param_val, **fixed_params, "seed": seed + 1}
    # Add period hints for frequency/amplitude methods
    if dim == "frequency":
        kwargs.setdefault("base_period", n // 8)
    if dim == "amplitude":
        kwargs.setdefault("period", n // 8)

    degraded_arr, meta = fn(base_arr, **kwargs)

    # Randomly assign degraded side to A or B
    degraded_side = "A" if rng.integers(2) == 0 else "B"
    if degraded_side == "A":
        arr_a, arr_b = degraded_arr, base_arr
    else:
        arr_a, arr_b = base_arr, degraded_arr

    prompt = _build_prompt(dim, method, key_param, param_val, arr_a, arr_b, desc)
    parsed = _call_llm(llm, prompt)

    answer = parsed.get("answer", "?").upper().strip()
    correct = (answer == degraded_side)
    return {
        "seed": seed,
        "degraded_side": degraded_side,
        "llm_answer": answer,
        "correct": correct,
        "confidence": parsed.get("confidence", 0.0),
        "reasoning": parsed.get("reasoning", ""),
        "meta": meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    llm: OpenAICompatibleLLM,
    dims: list[str],
    n_pairs: int,
    base_seed: int = 42,
) -> dict:
    """
    Run the full calibration sweep.

    Returns a nested dict:
    results[dim]["method:key_param"][param_val_str] = {
        "param_val": float,
        "accuracy": float,
        "ci95_lo": float,
        "ci95_hi": float,
        "n_correct": int,
        "n_total": int,
        "pairs": [...]   # stripped before saving to disk
    }
    """
    results: dict = {}

    total_configs = sum(
        len(methods)
        for dim, methods in SWEEP_CONFIGS.items()
        if dim in dims
    )
    total_evals = sum(
        len(cfg["values"]) * n_pairs
        for dim, methods in SWEEP_CONFIGS.items()
        if dim in dims
        for cfg in methods
    )
    print(f"\nCalibration sweep: {len(dims)} dim(s), "
          f"{total_configs} method configs, {total_evals} total LLM calls\n")

    seed_counter = base_seed

    for dim in dims:
        if dim not in SWEEP_CONFIGS:
            print(f"  [WARN] No sweep config for dim '{dim}', skipping.")
            continue

        results[dim] = {}
        print(f"{'─'*60}")
        print(f"Dimension: {dim}")

        for cfg in SWEEP_CONFIGS[dim]:
            method = cfg["method"]
            key_param = cfg["key_param"]
            fixed = {k: v for k, v in cfg["fixed"].items()
                     if k not in ("note",)}
            note = cfg.get("note", "")

            # Use "method:key_param" as key so multiple params per method
            # don't overwrite each other
            cfg_key = f"{method}:{key_param}"
            results[dim][cfg_key] = {}
            print(f"\n  Method: {method}  (sweep: {key_param})"
                  + (f"  [{note}]" if note else ""))
            print(f"  {'param_val':>10}  {'acc':>6}  {'correct/total':>13}  {'95% CI':>15}")
            print(f"  {'─'*55}")

            for val in cfg["values"]:
                pairs = []
                for i in range(n_pairs):
                    try:
                        res = _evaluate_pair(
                            llm, dim, method, key_param, val, fixed,
                            seed=seed_counter + i,
                        )
                        pairs.append(res)
                    except Exception as exc:
                        print(f"    [ERROR] seed={seed_counter+i}: {exc}")
                        pairs.append({"correct": False, "llm_answer": "?",
                                      "error": str(exc)})

                seed_counter += n_pairs
                n = len(pairs)
                n_correct = sum(1 for p in pairs if p.get("correct"))
                acc = n_correct / n if n else 0.0

                # Wilson score 95% CI (more accurate than normal approx near 0/1)
                z = 1.96
                denom = 1 + z * z / n
                centre = (acc + z * z / (2 * n)) / denom
                margin = (z / denom) * (acc * (1 - acc) / n + z * z / (4 * n * n)) ** 0.5
                ci_lo = max(0.0, centre - margin)
                ci_hi = min(1.0, centre + margin)

                # Mark inconclusive if CI straddles the threshold
                straddles = ci_lo < HEAVY_THRESHOLD < ci_hi
                flag = " ?" if straddles else ("  heavy" if ci_lo >= HEAVY_THRESHOLD else "")

                val_str = f"{val:.4g}"
                results[dim][cfg_key][val_str] = {
                    "param_val": val,
                    "accuracy": round(acc, 3),
                    "ci95_lo": round(ci_lo, 3),
                    "ci95_hi": round(ci_hi, 3),
                    "n_correct": n_correct,
                    "n_total": n,
                    "pairs": pairs,
                }
                bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
                ci_str = f"[{ci_lo:.0%}, {ci_hi:.0%}]"
                print(f"  {val_str:>10}  {acc:>5.0%}  {n_correct:>5}/{n:<5}  "
                      f"{bar}  {ci_str}{flag}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Boundary suggestion
# ─────────────────────────────────────────────────────────────────────────────

HEAVY_THRESHOLD = 0.85   # acc ≥ 85% → heavy (detectable without tools)
                         # acc <  85% → light (needs tool-based analysis)


def suggest_boundaries(results: dict) -> dict:
    """
    For each (dim, method, key_param), find the first param value where
    accuracy first reaches HEAVY_THRESHOLD — that is the heavy boundary.
    Everything below is considered light (tool required).
    """
    suggestions: dict = {}
    print(f"\n{'='*60}")
    print(f"Boundary Suggestions  (heavy ≥ {HEAVY_THRESHOLD:.0%} accuracy)")
    print(f"  acc < {HEAVY_THRESHOLD:.0%} → light (tool required)")
    print(f"  acc ≥ {HEAVY_THRESHOLD:.0%} → heavy (reasoning only)")
    print(f"{'='*60}")

    for dim, method_params in results.items():
        suggestions[dim] = {}
        for cfg_key, vals_dict in method_params.items():
            entries = sorted(vals_dict.values(), key=lambda x: x["param_val"])
            heavy_val = None
            for e in entries:
                if e["accuracy"] >= HEAVY_THRESHOLD and heavy_val is None:
                    heavy_val = e["param_val"]

            suggestions[dim][cfg_key] = {"heavy_boundary": heavy_val}
            hv = f"{heavy_val:.4g}" if heavy_val is not None else "never reached"
            print(f"  {dim:<22} {cfg_key:<30}  heavy≥ {hv}")

    return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

_HTML_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 960px; margin: 0 auto; padding: 28px 24px; color: #222; font-size: 13px; }
h1   { font-size: 1.1em; color: #111; margin-bottom: 4px; }
.meta { color: #888; font-size: 0.80em; margin-bottom: 24px; }
h2   { font-size: 0.95em; color: #334155; margin: 28px 0 2px;
       border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
h3   { font-size: 0.82em; color: #64748b; margin: 14px 0 6px; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin-bottom: 6px; }
th   { text-align: left; padding: 4px 10px; font-size: 0.75em; color: #94a3b8;
       text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid #e2e8f0; }
td   { padding: 4px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
tr:hover td { background: #f8fafc; }
.val   { font-variant-numeric: tabular-nums; color: #334155; }
.acc   { font-weight: 600; }
.acc-heavy { color: #16a34a; }
.acc-light { color: #64748b; }
.acc-maybe { color: #d97706; }
.bar-wrap { width: 100px; height: 12px; background: #e2e8f0; border-radius: 3px;
            overflow: hidden; display: inline-block; vertical-align: middle; }
.bar-fill { height: 100%; border-radius: 3px; }
.bar-heavy { background: #16a34a; }
.bar-light { background: #94a3b8; }
.bar-maybe { background: #f59e0b; }
.ci    { color: #94a3b8; font-size: 0.85em; }
.tag   { font-size: 0.75em; font-weight: 600; padding: 1px 6px; border-radius: 4px;
         display: inline-block; }
.tag-heavy { background: #dcfce7; color: #16a34a; }
.tag-light { background: #f1f5f9; color: #64748b; }
.tag-maybe { background: #fef3c7; color: #d97706; }
.boundary  { font-size: 0.78em; color: #64748b; margin-top: 2px; margin-bottom: 10px; }
.boundary b { color: #334155; }
"""


def _save_html(path: Path, payload: dict) -> None:
    """Write a self-contained HTML calibration report."""
    meta = payload["meta"]
    results = payload["results"]
    suggestions = payload["suggestions"]

    rows_html = []
    for dim, method_params in results.items():
        rows_html.append(f"<h2>{dim}</h2>")
        for cfg_key, vals_dict in method_params.items():
            boundary = suggestions.get(dim, {}).get(cfg_key, {}).get("heavy_boundary")
            bnd_str = (f"<b>{boundary:.4g}</b>" if boundary is not None
                       else "<b>never reached</b>")
            rows_html.append(
                f"<h3>{cfg_key}</h3>"
                f"<div class='boundary'>heavy boundary: {bnd_str} "
                f"&nbsp;·&nbsp; threshold ≥ {HEAVY_THRESHOLD:.0%}</div>"
            )

            rows_html.append(
                "<table><thead><tr>"
                "<th>param_val</th><th>acc</th><th>correct/total</th>"
                "<th>bar</th><th>95% CI</th><th></th>"
                "</tr></thead><tbody>"
            )

            entries = sorted(vals_dict.values(), key=lambda x: x["param_val"])
            for e in entries:
                acc = e["accuracy"]
                ci_lo = e["ci95_lo"]
                ci_hi = e["ci95_hi"]
                n_correct = e["n_correct"]
                n_total = e["n_total"]
                pv = e["param_val"]

                straddles = ci_lo < HEAVY_THRESHOLD < ci_hi
                is_heavy = ci_lo >= HEAVY_THRESHOLD

                if is_heavy:
                    cls, bar_cls, tag = "acc-heavy", "bar-heavy", "<span class='tag tag-heavy'>heavy</span>"
                elif straddles:
                    cls, bar_cls, tag = "acc-maybe", "bar-maybe", "<span class='tag tag-maybe'>?</span>"
                else:
                    cls, bar_cls, tag = "acc-light", "bar-light", "<span class='tag tag-light'>light</span>"

                bar_w = int(acc * 100)
                rows_html.append(
                    f"<tr>"
                    f"<td class='val'>{pv:.4g}</td>"
                    f"<td class='acc {cls}'>{acc:.0%}</td>"
                    f"<td class='val'>{n_correct}/{n_total}</td>"
                    f"<td><span class='bar-wrap'>"
                    f"<span class='bar-fill {bar_cls}' style='width:{bar_w}%'></span>"
                    f"</span></td>"
                    f"<td class='ci'>[{ci_lo:.0%}, {ci_hi:.0%}]</td>"
                    f"<td>{tag}</td>"
                    f"</tr>"
                )
            rows_html.append("</tbody></table>")

    ts = meta.get("timestamp", "")[:19]
    title = f"Calibration — {meta.get('model', '')}  |  n_pairs={meta.get('n_pairs')}  |  {ts}"

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Calibration Report</title>"
        f"<style>{_HTML_CSS}</style></head><body>"
        f"<h1>Calibration Report</h1>"
        f"<div class='meta'>{title}</div>"
        + "\n".join(rows_html)
        + "</body></html>"
    )
    path.write_text(html, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

ALL_DIMS = list(SWEEP_CONFIGS.keys())


def main():
    parser = argparse.ArgumentParser(description="Severity calibration experiment")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base_url", default=CHATANYWHERE_BASE_URL)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--dim", nargs="+", default=ALL_DIMS,
                        help=f"Dimension(s) to sweep. One or more of: {' '.join(ALL_DIMS)}")
    parser.add_argument("--n_pairs", type=int, default=30,
                        help="Number of pairs per parameter value (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Save full results JSON to this path "
                             "(default: logs/calibration_<timestamp>.json)")
    parser.add_argument("--no_save", action="store_true", default=False,
                        help="Do not save results to disk (print only)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    llm = OpenAICompatibleLLM(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        enable_thinking=args.enable_thinking,
    )

    dims = args.dim
    # validate
    bad = [d for d in dims if d not in ALL_DIMS]
    if bad:
        parser.error(f"Unknown dimension(s): {bad}. Choose from: {ALL_DIMS}")

    results = run_sweep(llm, dims, n_pairs=args.n_pairs, base_seed=args.seed)
    suggestions = suggest_boundaries(results)

    if not args.no_save:
        if args.out:
            out_path = Path(args.out)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dim_tag = "_".join(dims) if len(dims) <= 3 else f"{len(dims)}dims"
            out_path = Path("logs") / f"calibration_{dim_tag}_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip per-pair details — only keep the summary stats per param value
        summary_results = {
            dim: {
                cfg_key: {
                    val_str: {k: v for k, v in entry.items() if k != "pairs"}
                    for val_str, entry in vals_dict.items()
                }
                for cfg_key, vals_dict in method_params.items()
            }
            for dim, method_params in results.items()
        }
        payload = {
            "meta": {
                "model": args.model,
                "dims": dims,
                "n_pairs": args.n_pairs,
                "seed": args.seed,
                "timestamp": datetime.now().isoformat(),
            },
            "results": summary_results,
            "suggestions": suggestions,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        html_path = out_path.with_suffix(".html")
        _save_html(html_path, payload)
        print(f"\nResults saved to : {out_path}")
        print(f"HTML report      : {html_path}")


if __name__ == "__main__":
    main()
