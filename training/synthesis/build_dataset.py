"""
CLI entry point for generating Perceiver training datasets.

Usage:
    python -m training.synthesis.build_dataset --n_samples 5000 --output training/data/train.jsonl
    python -m training.synthesis.build_dataset --n_samples 500 --seed_offset 1000000 --output training/data/val.jsonl
    python -m training.synthesis.build_dataset --n_samples 1000 --output training/data/check.jsonl --stats
    python -m training.synthesis.build_dataset --n_samples 10 --output training/data/test.jsonl --visualize

Filtered datasets (hint-label consistent):
    # 训练集：多生成以补偿过滤损耗（6000 → 约 4000~4500 有效）
    python -m training.synthesis.build_dataset \
        --n_samples 4000 \
        --output training/data/perceiver_train_filtered.jsonl \
        --filter_by_hints \
        --stats

    python -m training.synthesis.build_dataset \
        --n_samples 2000 \
        --heavy_prob 0.5 \
        --output training/data/inspector_train.jsonl \
        --stats

    # 验证集：seed_offset 不变，保持和原 val 不重叠
    python -m training.synthesis.build_dataset \
        --n_samples 500 \
        --seed_offset 1000000 \
        --output training/data/perceiver_val.jsonl \
        --stats

    python -m training.synthesis.build_dataset \
        --n_samples 300 \
        --heavy_prob 0.5 \
        --seed_offset 1000000 \
        --output training/data/inspector_val.jsonl \
        --stats
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from training.synthesis.sample_generator import generate_sample
from training.synthesis.visualize import render_dataset_html
from training.rl.data_loader import verify_hint_label_consistency


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description="Generate Perceiver training data")
    parser.add_argument("--n_samples", type=int, default=4000, help="Number of samples")
    parser.add_argument("--seed_offset", type=int, default=0, help="Starting seed")
    parser.add_argument("--output", type=str, default="training/data/perceiver_train.jsonl")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--visualize", action="store_true", help="Render all samples as HTML pages")
    parser.add_argument("--heavy_prob", type=float, default=0.9,
                        help="Prior probability of heavy severity (default 0.9). "
                             "Increase to make differences more visible from stats/preview.")
    parser.add_argument("--filter_by_hints", action="store_true",
                        help="Remove GT labels whose injected defect leaves no detectable "
                             "signal in hint statistics. Eliminates contradictory training "
                             "signal (hint says 'don't select' but label says 'select'). "
                             "Samples whose entire label set is removed are skipped.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Counters for stats
    dim_counter = Counter()
    n_dims_counter = Counter()
    severity_counter = Counter()
    tool_counter = Counter()
    n_labels_removed = 0
    n_labels_added = 0
    n_samples_skipped = 0

    generated_samples = []
    written = 0
    with open(args.output, "w") as f:
        for i in range(args.n_samples):
            seed = args.seed_offset + i
            sample = generate_sample(seed=seed, heavy_prob=args.heavy_prob)

            # ── hint-label consistency filtering ─────────────────────────────
            if args.filter_by_hints:
                inp = sample["input"]
                orig_dims = sample["labels"]["target_dimensions"]
                verified_dims = verify_hint_label_consistency(
                    inp["preview_A"], inp["preview_B"],
                    inp["stats_A"], inp["stats_B"],
                    orig_dims,
                )
                orig_set = set(orig_dims)
                verified_set = set(verified_dims)
                n_labels_removed += len(orig_set - verified_set)
                n_labels_added   += len(verified_set - orig_set)
                if not verified_dims:
                    n_samples_skipped += 1
                    continue
                sample["labels"]["target_dimensions"] = verified_dims
                # Keep tool_required only for dimensions that survived filtering;
                # newly added dims are not tool_required (hint-detectable = heavy)
                sample["labels"]["tool_required"] = [
                    d for d in sample["labels"]["tool_required"]
                    if d in verified_set
                ]

            f.write(json.dumps(sample, cls=_NumpyEncoder, ensure_ascii=False) + "\n")
            written += 1
            if args.visualize:
                generated_samples.append(sample)

            # Collect stats
            labels = sample["labels"]
            n_dims = len(labels["target_dimensions"])
            n_dims_counter[n_dims] += 1
            for d in labels["target_dimensions"]:
                dim_counter[d] += 1
            for detail in sample["meta"]["defect_details"]:
                severity_counter[detail["severity"]] += 1
            tool_counter["tool"] += len(labels["tool_required"])
            tool_counter["reasoning"] += n_dims - len(labels["tool_required"])

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{args.n_samples}")

    print(f"Generated {args.n_samples} samples → {written} written to {args.output}")
    if args.filter_by_hints:
        print(f"  hint filtering: {n_labels_removed} labels removed, "
              f"{n_labels_added} labels added, "
              f"{n_samples_skipped} samples skipped (empty after filtering)")

    if args.visualize:
        vis_path = args.output.replace(".jsonl", "_vis.html")
        print(f"Rendering {args.n_samples} samples → {vis_path}")
        html = render_dataset_html(generated_samples, title=os.path.basename(args.output))
        with open(vis_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Done.")

    if args.stats:
        total_dims = tool_counter["tool"] + tool_counter["reasoning"]
        stats_data = {
            "n_samples": args.n_samples,
            "dimension_count_distribution": {str(k): n_dims_counter[k] for k in sorted(n_dims_counter.keys())},
            "per_dimension_frequency": dict(dim_counter.most_common()),
            "severity_distribution": dict(severity_counter.most_common()),
            "tool_requirement": {
                "tool": tool_counter["tool"],
                "tool_pct": round(tool_counter["tool"] / total_dims * 100, 1) if total_dims else 0,
                "reasoning": tool_counter["reasoning"],
                "reasoning_pct": round(tool_counter["reasoning"] / total_dims * 100, 1) if total_dims else 0,
            },
        }

        stats_path = args.output.replace(".jsonl", ".stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

        print(f"\n--- Dataset Statistics ---")
        print(f"\nDimension count distribution:")
        for k in sorted(n_dims_counter.keys()):
            pct = n_dims_counter[k] / args.n_samples * 100
            print(f"  N={k}: {n_dims_counter[k]:>5} ({pct:.1f}%)")

        print(f"\nPer-dimension frequency:")
        for dim, cnt in dim_counter.most_common():
            print(f"  {dim:<24} {cnt:>5}")

        print(f"\nSeverity distribution:")
        for sev, cnt in severity_counter.most_common():
            print(f"  {sev:<10} {cnt:>5}")

        if total_dims:
            print(f"\nTool requirement:")
            print(f"  tool:      {tool_counter['tool']:>5} ({tool_counter['tool']/total_dims*100:.1f}%)")
            print(f"  reasoning: {tool_counter['reasoning']:>5} ({tool_counter['reasoning']/total_dims*100:.1f}%)")

        print(f"\nStats saved → {stats_path}")


if __name__ == "__main__":
    main()
