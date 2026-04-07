"""
Data loader for Perceiver GRPO training.

Loads JSONL training samples and converts them to HuggingFace Datasets with
the exact prompt format the Perceiver sees at inference time.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from agents.perceiver import SYSTEM_PROMPT


# ── Prompt construction (mirrors agents/perceiver.py:112-156) ────────────────

def _build_user_message(sample_input: dict) -> str:
    """Reconstruct the user message exactly as perceiver.py does."""
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

    return f"""Dataset description: {desc}

Series A — values ({len_A} total{sampled_A}):
{json.dumps(preview_A)}

Series A statistics:
{json.dumps(stats_A, indent=2)}

Series B — values ({len_B} total{sampled_B}):
{json.dumps(preview_B)}

Series B statistics:
{json.dumps(stats_B, indent=2)}

External variables: {{}}

Based on the above, decide which quality dimensions to assess.
Remember: output ONLY valid JSON as specified."""


def build_prompt_messages(sample: dict) -> list[dict]:
    """Build chat messages list for a single sample."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(sample["input"])},
    ]


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(jsonl_path: str) -> Dataset:
    """
    Load JSONL and return a HuggingFace Dataset.

    Columns:
      - prompt: list[dict]  (chat messages for tokenizer.apply_chat_template)
      - target_dimensions: list[str]
      - tool_required: list[str]
      - sample_id: str
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
        messages = build_prompt_messages(sample)
        records.append({
            "prompt": messages,
            "target_dimensions": sample["labels"]["target_dimensions"],
            "tool_required": sample["labels"]["tool_required"],
            "sample_id": sample["sample_id"],
        })

    return Dataset.from_list(records)
