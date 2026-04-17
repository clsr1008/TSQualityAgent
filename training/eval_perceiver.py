"""
Perceiver evaluation — dimension selection accuracy.

Works with both the base model and a LoRA-finetuned checkpoint served via vLLM.
Reports:
  - Per-dimension selection accuracy (Precision / Recall / F1 / TP / FP / FN)
  - Average number of predicted dimensions per sample
  - Parse failure rate

Usage:
    # Base model (no LoRA)
    python -m training.eval_perceiver \
        --data training/data/perceiver_val.jsonl \
        --base_url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B

    CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-4B \
        --port 8001 \
        --max-model-len 32768

    # start LoRA checkpoint
    CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-4B \
    --enable-lora \
    --lora-modules perceiver-grpo-v1=training/checkpoints/perceiver-grpo-v1 \
    --port 8000

    # LoRA checkpoint (vLLM must be started with --enable-lora --lora-modules perceiver=<path>)
    python -m training.eval_perceiver \
        --data training/data/perceiver_val.jsonl \
        --base_url http://localhost:8000/v1 \
        --model perceiver-grpo-v2 \
        --n_dims_hint

    python -m training.eval_perceiver \
        --data training/data/perceiver_val.jsonl \
        --base_url http://localhost:8001/v1 \
        --model Qwen/Qwen3-4B \
        --n_dims_hint
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from training.rl.data_loader import load_dataset
from training.rl.reward import parse_perceiver_output

ALL_DIMENSIONS = [
    "missing_value", "noise_level", "rare_pattern",
    "trend", "frequency", "amplitude", "pattern_consistency",
]


def call_model(client: OpenAI, model: str, messages: list[dict],
               max_tokens: int, temperature: float) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  [API error] {e}")
        return None


class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text: str):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def evaluate(args: argparse.Namespace) -> None:
    if args.output:
        out_path = Path(args.output)
    else:
        model_slug = args.model.replace("/", "_").replace("\\", "_")
        out_path = Path(f"logs/perceiver_val/eval_{model_slug}_{datetime.now().strftime('%Y%m%d')}.txt")
    tee = _Tee(out_path)
    sys.stdout = tee

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    dataset = load_dataset(args.data, n_dims_hint=args.n_dims_hint)

    if args.n_samples and args.n_samples < len(dataset):
        dataset = dataset.select(range(args.n_samples))

    n_total = len(dataset)
    print(f"Evaluating {n_total} samples from {args.data} ...")
    print(f"Model: {args.model}  |  temperature={args.temperature}  |  max_tokens={args.max_tokens}")
    print(f"n_dims_hint: {'ON (GT count injected into prompt)' if args.n_dims_hint else 'OFF'}\n")

    # Counters
    n_parse_fail = 0
    pred_dim_counts = []  # number of predicted dims per sample
    gt_dim_counts = []    # number of GT dims per sample

    # Per-dimension selection accuracy (TP / FP / FN)
    tp_count = defaultdict(int)  # predicted d AND d in GT
    fp_count = defaultdict(int)  # predicted d but d NOT in GT
    fn_count = defaultdict(int)  # d in GT but NOT predicted

    for i, row in enumerate(dataset):
        sample_id = row["sample_id"]
        target_dims: list[str] = row["target_dimensions"]
        messages: list[dict] = row["prompt"]

        output = call_model(client, args.model, messages, args.max_tokens, args.temperature)
        if output is None:
            n_parse_fail += 1
            continue

        parsed = parse_perceiver_output(output)
        if parsed is None:
            n_parse_fail += 1
            preview = output[:120].replace("\n", " ") if output else "(empty)"
            print(f"  [{i+1}/{n_total}] {sample_id}  PARSE FAIL | {preview}")
            continue

        pred_dims = [d for d in parsed.get("planned_dimensions", []) if d in ALL_DIMENSIONS]
        pred_set  = set(pred_dims)
        gt_set    = set(target_dims)

        # ── Debug print ───────────────────────────────────────────────────────
        if args.debug_n > 0 and i < args.debug_n:
            tp  = pred_set & gt_set
            fp  = pred_set - gt_set
            fn  = gt_set - pred_set

            # Extract hint table from user message (added by _build_user_message)
            user_content = messages[-1]["content"] if messages else ""
            h_start = user_content.find("Dimension-specific indicators")
            h_end   = user_content.find("\nExternal variables", h_start) if h_start >= 0 else -1
            hint_block = user_content[h_start:h_end].strip() if h_start >= 0 else "(no hints)"

            # Annotate each hint line with ✓ (TP), ✗ (FP), ? (FN), · (not relevant)
            dim_tag = {
                "missing_value":      "missing_ratio",
                "rare_pattern":       "max_local_zscore",
                "noise_level":        "rolling_std_mean",
                "amplitude":          "  std  ",        # avoid matching "rolling_std"
                "frequency":          "spectral_corr",
                "trend":              "slope",
                "pattern_consistency":"autocorr_lag1",
            }
            tag_to_dim = {v: k for k, v in dim_tag.items()}
            annotated_lines = []
            for ln in hint_block.splitlines():
                marker = "  "
                for key, dim in tag_to_dim.items():
                    if key in ln:
                        if dim in tp:   marker = "✓ "
                        elif dim in fp: marker = "✗ "
                        elif dim in fn: marker = "? "
                        break
                annotated_lines.append(f"    {marker}{ln}")

            raw_preview = output.replace("\n", "\\n")[:200]
            print(f"\n  [debug {i+1}] {sample_id}")
            print(f"    GT   : {sorted(gt_set)}")
            print(f"    Pred : {sorted(pred_set)}")
            print(f"    TP={sorted(tp)}  FP={sorted(fp)}  FN={sorted(fn)}")
            print("\n".join(annotated_lines))
            print(f"    raw  : {raw_preview}")

        pred_dim_counts.append(len(pred_set))
        gt_dim_counts.append(len(gt_set))

        # Dim selection TP / FP / FN
        for d in pred_set:
            if d in gt_set:
                tp_count[d] += 1
            else:
                fp_count[d] += 1
        for d in gt_set:
            if d not in pred_set:
                fn_count[d] += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_total}] done")

    # ── Report ────────────────────────────────────────────────────────────────
    n_evaluated = n_total - n_parse_fail
    print(f"\n{'='*60}")
    print(f"  Samples evaluated : {n_evaluated} / {n_total}  (parse fail: {n_parse_fail})")
    print(f"{'='*60}\n")

    # Dimension count stats
    if pred_dim_counts:
        avg_pred = sum(pred_dim_counts) / len(pred_dim_counts)
        avg_gt   = sum(gt_dim_counts) / len(gt_dim_counts)
        print(f"  Avg dims per sample (GT)   : {avg_gt:.2f}")
        print(f"  Avg dims per sample (pred) : {avg_pred:.2f}")
        print(f"  Bias                       : {avg_pred - avg_gt:+.2f}  ({'over-selecting' if avg_pred > avg_gt else 'under-selecting'})\n")

    # ── Dimension selection accuracy ──────────────────────────────────────────
    print(f"  {'Dimension':<22}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*6}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*4}")

    total_tp = total_fp = total_fn = 0
    for dim in ALL_DIMENSIONS:
        tp = tp_count[dim]
        fp = fp_count[dim]
        fn = fn_count[dim]
        total_tp += tp; total_fp += fp; total_fn += fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec  = tp / (tp + fn) if (tp + fn) > 0 else None
        f1   = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec > 0) else None
        p_str = f"{prec:.1%}" if prec is not None else "  n/a"
        r_str = f"{rec:.1%}"  if rec  is not None else "  n/a"
        f_str = f"{f1:.1%}"   if f1   is not None else "  n/a"
        print(f"  {dim:<22}  {p_str:>9}  {r_str:>6}  {f_str:>6}  {tp:>4}  {fp:>4}  {fn:>4}")

    # Overall micro averages
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    print(f"  {'[micro avg]':<22}  {micro_prec:>8.1%}  {micro_rec:>5.1%}  {micro_f1:>5.1%}  {total_tp:>4}  {total_fp:>4}  {total_fn:>4}")
    print()

    # Suggestion
    if pred_dim_counts and avg_pred > avg_gt + 1.0:
        print(f"  Suggestion: model over-selects dimensions (avg {avg_pred:.1f} vs GT {avg_gt:.1f}).")
        print(f"              Focus training on improving Precision.")
    elif pred_dim_counts and avg_pred < avg_gt - 0.5:
        print(f"  Suggestion: model under-selects dimensions (avg {avg_pred:.1f} vs GT {avg_gt:.1f}).")
        print(f"              Focus training on improving Recall.")
    else:
        print(f"  Suggestion: dimension selection looks balanced.")

    sys.stdout = tee._stdout
    tee.close()
    print(f"\n  Report saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True,  help="Path to validation JSONL")
    parser.add_argument("--base_url",    default="http://localhost:8000/v1")
    parser.add_argument("--model",       default="Qwen/Qwen3-4B")
    parser.add_argument("--max_tokens",  type=int,   default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_samples",   type=int,   default=None,
                        help="Evaluate only first N samples (default: all)")
    parser.add_argument("--output",      default=None,
                        help="Output .txt path (default: logs/perceiver_val/eval_<timestamp>.txt)")
    parser.add_argument("--debug_n",     type=int, default=0,
                        help="Print raw output + parsed dims + GT for first N samples (0 disables).")
    parser.add_argument("--n_dims_hint", action="store_true",
                        help="Inject the GT dimension count into each prompt. "
                             "Isolates the 'which dimensions' sub-task from 'how many'.")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()