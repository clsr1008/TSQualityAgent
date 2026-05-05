"""
Quality comparison evaluation (Benchmark Sub-experiment B).

Measures how accurately an LLM can judge which of two time series is higher
quality on a *given* dimension, in two conditions:

  --no_tools   : model reasons from statistics + preview only (no tool calls).
  --with_tools : model may call statistical tools via a ReAct loop (same as
                 the live Inspector agent).

Expected result pattern (Gap 3 evidence):
  - heavy severity: both conditions perform well (difference obvious from preview).
  - light severity: --no_tools ≈ random (50%); --with_tools clearly better.

Input format: the standard JSONL produced by build_dataset.py.
Each sample may contain multiple injected dimensions; one dimension is randomly
selected per sample for evaluation (other injected dimensions remain as
"interference noise", testing whether the model can focus on the target).

Ground-truth winner: the side that did NOT receive the injected defect.
  defect_side == "B"  →  winner == "A"
  defect_side == "A"  →  winner == "B"

Series data: preview_A / preview_B from the existing format serve as the full
series (series length 100–150 ≤ max_full=200, so preview IS the full array).

Generate comparison validation set (500 samples, 50/50 light/heavy):
    python -m training.synthesis.build_dataset \
        --n_samples 500 \
        --heavy_prob 0.5 \
        --seed_offset 2000000 \
        --output training/data/comparison_val.jsonl \
        --stats

Usage
-----
# No-tools (text reasoning only)
python -m training.eval_comparison \
    --data training/data/comparison_val.jsonl \
    --base_url http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B \
    --no_tools

# With tools (ReAct loop)
python -m training.eval_comparison \
    --data training/data/comparison_val.jsonl \
    --base_url http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B \
    --with_tools

# Both conditions in one run (prints a delta table at the end)
python -m training.eval_comparison \
    --data training/data/comparison_val.jsonl \
    --base_url http://localhost:8001/v1 \
    --model Qwen/Qwen3-4B \
    --no_tools --with_tools

# Debug: print per-sample detail for first N samples
python -m training.eval_comparison ... --no_tools --debug_n 3

vLLM deployment (tool support required for --with_tools):
    CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-4B \
        --port 8001 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --max-model-len 32768

    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
        --port 8002 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --max-model-len 32768
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from agents.dimensions import DIMENSION_GUIDE
from tools import NumpyEncoder
from tools.registry import TOOL_REGISTRY, TOOL_SCHEMAS


ALL_DIMENSIONS = [
    "missing_value", "noise_level", "rare_pattern",
    "trend", "frequency", "amplitude", "pattern_consistency",
]
SEVERITIES = ["light", "heavy"]


# ── Prompts ────────────────────────────────────────────────────────────────────

_NO_TOOLS_SYSTEM = f"""You are a time series quality assessment expert.

You will be given two time series segments (A and B) with their basic statistics
and value previews, along with one specific quality dimension to evaluate.

Your task: judge which series has better quality **on that dimension only**.
Base your judgment entirely on the statistics and preview — do NOT call any tools.
{DIMENSION_GUIDE}
Respond with ONLY valid JSON (no extra text):
{{
  "winner": "A" | "B" | "tie",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one or two sentences>"
}}

confidence: 1.0 = overwhelming difference clearly visible, 0.5 = moderate,
0.2 = barely distinguishable, 0.0 = cannot tell."""

_WITH_TOOLS_SYSTEM = f"""You are the Inspector agent in a time series quality assessment pipeline.

You will assess exactly ONE quality dimension for two time series (A and B).
Use a ReAct loop: Thought → (Tool Call → Observation)? → DIMENSION_COMPLETE
{DIMENSION_GUIDE}
For rare_pattern: score based ONLY on Category 1 outliers (data defects).

## Workflow

**Step 1 — Decide: reasoning-only or use tools?**
Ask: "Can I determine the winner for this dimension confidently from preview/stats alone?"
- Reasoning only: difference is visually obvious.
- Use tools: difference is subtle and requires precise measurement.

**Step 2 — Call tools if needed** (minimum tools required for this dimension).

**Step 3 — Output DIMENSION_COMPLETE**

DIMENSION_COMPLETE
{{
  "dimension": "<name>",
  "winner": "A" | "B" | "tie",
  "confidence": <0.0 to 1.0>,
  "evidence": {{
    "A": {{ <decisive metrics> }},
    "B": {{ <decisive metrics> }}
  }},
  "conclusion": "<one sentence>"
}}
END_DIMENSION

confidence: 1.0 = overwhelming advantage, 0.5 = moderate, ~0.0 = negligible gap."""


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_comparison_tasks(path: str, seed: int = 0) -> list[dict]:
    """Load JSONL and expand to one comparison task per sample.

    Each task contains a single (dimension, severity, ground_truth_winner) drawn
    randomly from the sample's injected defects. The full preview and stats are
    carried over unchanged.
    """
    rng = random.Random(seed)
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            details = record.get("meta", {}).get("defect_details", [])
            if not details:
                continue
            # Pick one dimension randomly
            detail = rng.choice(details)
            tasks.append({
                "sample_id": record["sample_id"],
                "dimension": detail["dimension"],
                "severity": detail["severity"],
                "defect_side": detail["side"],
                "ground_truth_winner": "A" if detail["side"] == "B" else "B",
                # preview IS the full series for length ≤ 200
                "series_A": record["input"]["preview_A"],
                "series_B": record["input"]["preview_B"],
                "stats_A": record["input"]["stats_A"],
                "stats_B": record["input"]["stats_B"],
            })
    return tasks


# ── User message builders ──────────────────────────────────────────────────────

def _build_user_message(task: dict, mode: str) -> str:
    dim = task["dimension"]
    n_A = len(task["series_A"])
    n_B = len(task["series_B"])

    body = f"""Dimension to compare: **{dim}**

Series A — values ({n_A} points):
{json.dumps(task["series_A"])}

Series A statistics:
{json.dumps(task["stats_A"], indent=2)}

Series B — values ({n_B} points):
{json.dumps(task["series_B"])}

Series B statistics:
{json.dumps(task["stats_B"], indent=2)}"""

    if mode == "no_tools":
        body += f"\n\nWhich series has better quality on the '{dim}' dimension?"
    else:
        body += f"\n\nAssess the '{dim}' dimension for A vs B. Use tools if needed, then output DIMENSION_COMPLETE."
    return body


# ── Output parsers ─────────────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_no_tools(text: str) -> Optional[tuple[str, float]]:
    text = _strip_think(text)
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        winner = str(parsed.get("winner", "")).strip().upper()
        if winner == "TIE":
            winner = "tie"
        if winner not in ("A", "B", "tie"):
            return None
        return winner, float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError):
        return None


def _parse_dimension_complete(text: str) -> Optional[tuple[str, float]]:
    match = re.search(r"DIMENSION_COMPLETE\s*(\{.*?\})\s*END_DIMENSION", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: any JSON with a 'winner' key
        m2 = re.search(r"\{[^{}]*\"winner\"[^{}]*\}", text, re.DOTALL)
        if not m2:
            return None
        json_str = m2.group()
    try:
        parsed = json.loads(json_str)
        winner = str(parsed.get("winner", "")).strip()
        if winner not in ("A", "B", "tie"):
            return None
        return winner, float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError):
        return None


# ── Tool execution ─────────────────────────────────────────────────────────────

class _ToolCache:
    def __init__(self):
        self._store: dict = {}

    def _key(self, name, sname, args):
        return (name, sname, tuple(sorted(args.items())))

    def get(self, name, sname, args):
        return self._store.get(self._key(name, sname, args))

    def put(self, name, sname, args, result):
        self._store[self._key(name, sname, args)] = result


def _execute_tool(name: str, raw_args: dict,
                  series_A: list, series_B: list,
                  cache: _ToolCache) -> dict:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    args = dict(raw_args)
    sname = args.pop("series_name", "A")
    series = series_A if sname == "A" else series_B
    cached = cache.get(name, sname, args)
    if cached is not None:
        return {**cached, "_cached": True}
    try:
        result = fn(series, **args)
        result["series"] = sname
        cache.put(name, sname, args, result)
        return result
    except Exception as e:
        return {"error": str(e), "series": sname}


# ── Inference ──────────────────────────────────────────────────────────────────

def run_no_tools(client: OpenAI, model: str, task: dict,
                 max_tokens: int, temperature: float) -> Optional[tuple[str, float]]:
    messages = [
        {"role": "system", "content": _NO_TOOLS_SYSTEM},
        {"role": "user",   "content": _build_user_message(task, "no_tools")},
    ]
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _parse_no_tools(_strip_think(resp.choices[0].message.content or ""))
    except Exception as e:
        print(f"    [API error] {e}")
        return None


def run_with_tools(client: OpenAI, model: str, task: dict,
                   max_tokens: int, temperature: float,
                   max_steps: int = 10, debug: bool = False) -> Optional[tuple[str, float]]:
    series_A = task["series_A"]
    series_B = task["series_B"]
    cache = _ToolCache()
    messages = [
        {"role": "system", "content": _WITH_TOOLS_SYSTEM},
        {"role": "user",   "content": _build_user_message(task, "with_tools")},
    ]
    for step in range(max_steps):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                tools=TOOL_SCHEMAS, tool_choice="auto",
                max_tokens=max_tokens, temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:
            print(f"    [API error step {step}] {e}")
            return None

        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                try:
                    raw_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    raw_args = {}
                result = _execute_tool(tc.function.name, raw_args, series_A, series_B, cache)
                if debug:
                    print(f"      [tool] {tc.function.name}({raw_args}) → {result}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                  "content": json.dumps(result, cls=NumpyEncoder)})
        else:
            content = _strip_think(msg.content or "")
            messages.append({"role": "assistant", "content": content})
            result = _parse_dimension_complete(content)
            if result is not None:
                return result
            if resp.choices[0].finish_reason == "stop":
                return _parse_no_tools(content)
    return None


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_table(title: str, stats: dict) -> None:
    print(f"\n  {title}")
    hdr = (f"  {'Dimension':<22}  {'light acc':>9}  {'(n)':>4}"
           f"  {'heavy acc':>9}  {'(n)':>4}  {'overall':>7}")
    print(hdr)
    print(f"  {'-'*22}  {'-'*9}  {'-'*4}  {'-'*9}  {'-'*4}  {'-'*7}")

    total_c = total_t = 0
    light_c = light_t = 0
    heavy_c = heavy_t = 0
    for dim in ALL_DIMENSIONS:
        lc, lt = stats[dim]["light"]
        hc, ht = stats[dim]["heavy"]
        oc, ot = lc + hc, lt + ht
        total_c += oc; total_t += ot
        light_c += lc; light_t += lt
        heavy_c += hc; heavy_t += ht
        la = f"{lc/lt:.1%}" if lt else "    n/a"
        ha = f"{hc/ht:.1%}" if ht else "    n/a"
        oa = f"{oc/ot:.1%}" if ot else "  n/a"
        print(f"  {dim:<22}  {la:>9}  {lt:>4}  {ha:>9}  {ht:>4}  {oa:>7}")

    print(f"  {'-'*22}  {'-'*9}  {'-'*4}  {'-'*9}  {'-'*4}  {'-'*7}")
    la_ov = f"{light_c/light_t:.1%}" if light_t else "    n/a"
    ha_ov = f"{heavy_c/heavy_t:.1%}" if heavy_t else "    n/a"
    ov    = f"{total_c/total_t:.1%}" if total_t else "  n/a"
    print(f"  {'[overall]':<22}  {la_ov:>9}  {light_t:>4}  {ha_ov:>9}  {heavy_t:>4}  {ov:>7}")


# ── Tee ────────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text):
        self._stdout.write(text); self._file.write(text)

    def flush(self):
        self._stdout.flush(); self._file.flush()

    def close(self):
        self._file.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    modes = [m for m in ("no_tools", "with_tools") if getattr(args, m)]
    if not modes:
        print("Error: specify --no_tools and/or --with_tools"); sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        slug = args.model.replace("/", "_").replace("\\", "_")
        tag = "+".join(modes)
        out_path = Path(f"logs/comparison_val/eval_{slug}_{tag}_{datetime.now().strftime('%Y%m%d')}.txt")
    tee = _Tee(out_path)
    sys.stdout = tee

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    tasks = load_comparison_tasks(args.data, seed=args.seed)

    if args.n_samples and args.n_samples < len(tasks):
        tasks = tasks[: args.n_samples]

    n_total = len(tasks)
    light_n = sum(1 for t in tasks if t["severity"] == "light")
    heavy_n = n_total - light_n
    print(f"Evaluating {n_total} tasks from {args.data}")
    print(f"  light={light_n}  heavy={heavy_n}")
    print(f"Model: {args.model}  |  modes: {modes}  |  temperature={args.temperature}\n")

    # stats[mode][dim][sev] = [correct, total]
    all_stats = {
        mode: {dim: {sev: [0, 0] for sev in SEVERITIES} for dim in ALL_DIMENSIONS}
        for mode in modes
    }
    n_fail = {mode: 0 for mode in modes}

    for i, task in enumerate(tasks):
        dim = task["dimension"]
        sev = task["severity"] if task["severity"] in SEVERITIES else "heavy"
        gt  = task["ground_truth_winner"]

        for mode in modes:
            if mode == "no_tools":
                result = run_no_tools(client, args.model, task, args.max_tokens, args.temperature)
            else:
                result = run_with_tools(
                    client, args.model, task, args.max_tokens, args.temperature,
                    max_steps=args.max_steps,
                    debug=(args.debug_n > 0 and i < args.debug_n),
                )

            if result is None:
                n_fail[mode] += 1
                if args.debug_n == 0 or i >= args.debug_n:
                    print(f"  [{i+1}/{n_total}] {task['sample_id']} [{mode}] PARSE FAIL")
                continue

            pred, conf = result
            correct = int(pred == gt)
            all_stats[mode][dim][sev][0] += correct
            all_stats[mode][dim][sev][1] += 1

            if args.debug_n > 0 and i < args.debug_n:
                mark = "✓" if correct else "✗"
                print(f"\n  [debug {i+1}] {task['sample_id']}  [{mode}]")
                print(f"    dim={dim}  sev={sev}  gt={gt}  pred={pred}  conf={conf:.2f}  {mark}")

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_total}] done")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  Samples: {n_total}")
    for mode in modes:
        print(f"  Parse failures [{mode}]: {n_fail[mode]}")
    print(f"{'='*68}")

    for mode in modes:
        _print_table(f"Accuracy — {mode}", all_stats[mode])

    if len(modes) == 2:
        print(f"\n  Delta (with_tools − no_tools):")
        print(f"  {'Dimension':<22}  {'Δ light':>7}  {'Δ heavy':>7}")
        print(f"  {'-'*22}  {'-'*7}  {'-'*7}")
        for dim in ALL_DIMENSIONS:
            row = []
            for sev in SEVERITIES:
                wc, wt = all_stats["with_tools"][dim][sev]
                nc, nt = all_stats["no_tools"][dim][sev]
                if wt > 0 and nt > 0:
                    row.append(f"{wc/wt - nc/nt:+.1%}")
                else:
                    row.append("    n/a")
            print(f"  {dim:<22}  {row[0]:>7}  {row[1]:>7}")

    sys.stdout = tee._stdout
    tee.close()
    print(f"\n  Report saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate quality comparison accuracy with and without tools"
    )
    parser.add_argument("--data",        required=True,
                        help="Path to comparison_val.jsonl (standard build_dataset.py output)")
    parser.add_argument("--base_url",    default="http://localhost:8000/v1")
    parser.add_argument("--api_key",     default="EMPTY")
    parser.add_argument("--model",       default="Qwen/Qwen3-4B")
    parser.add_argument("--no_tools",    action="store_true",
                        help="Run text-reasoning-only condition")
    parser.add_argument("--with_tools",  action="store_true",
                        help="Run ReAct tool-calling condition")
    parser.add_argument("--max_tokens",  type=int,   default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_steps",   type=int,   default=10,
                        help="Max ReAct steps per sample in --with_tools mode")
    parser.add_argument("--n_samples",   type=int,   default=None,
                        help="Evaluate only first N samples (default: all)")
    parser.add_argument("--seed",        type=int,   default=0,
                        help="Random seed for dimension selection per sample")
    parser.add_argument("--output",      default=None,
                        help="Output .txt path (default: logs/comparison_val/...)")
    parser.add_argument("--debug_n",     type=int,   default=0,
                        help="Print per-sample details for first N samples")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()