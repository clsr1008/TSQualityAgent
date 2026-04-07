"""
CLI entry point for generating Perceiver training datasets.

Usage:
    python -m training.synthesis.build_dataset --n_samples 5000 --output training/data/train.jsonl
    python -m training.synthesis.build_dataset --n_samples 500 --seed_offset 1000000 --output training/data/val.jsonl
    python -m training.synthesis.build_dataset --n_samples 1000 --output training/data/check.jsonl --stats
    python -m training.synthesis.build_dataset --n_samples 10 --output training/data/test.jsonl --visualize
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
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seed_offset", type=int, default=0, help="Starting seed")
    parser.add_argument("--output", type=str, default="training/data/perceiver_train.jsonl")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--visualize", action="store_true", help="Render all samples as HTML pages")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Counters for stats
    dim_counter = Counter()
    n_dims_counter = Counter()
    severity_counter = Counter()
    tool_counter = Counter()

    generated_samples = []
    with open(args.output, "w") as f:
        for i in range(args.n_samples):
            seed = args.seed_offset + i
            sample = generate_sample(seed=seed)
            f.write(json.dumps(sample, cls=_NumpyEncoder, ensure_ascii=False) + "\n")
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

    print(f"Generated {args.n_samples} samples → {args.output}")

    if args.visualize:
        vis_path = args.output.replace(".jsonl", "_vis.html")
        print(f"Rendering {args.n_samples} samples → {vis_path}")
        html = render_dataset_html(generated_samples, title=os.path.basename(args.output))
        with open(vis_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Done.")

    if args.stats:
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

        total_dims = tool_counter["tool"] + tool_counter["reasoning"]
        if total_dims:
            print(f"\nTool requirement:")
            print(f"  tool:      {tool_counter['tool']:>5} ({tool_counter['tool']/total_dims*100:.1f}%)")
            print(f"  reasoning: {tool_counter['reasoning']:>5} ({tool_counter['reasoning']/total_dims*100:.1f}%)")


if __name__ == "__main__":
    main()
