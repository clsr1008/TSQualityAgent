"""
TSqualityAgent – entry point
Demonstrates end-to-end pairwise time series quality assessment.
"""
import numpy as np
from config import Config, build_llm
from workflow import run_pipeline


def make_test_cases():
    rng = np.random.default_rng(42)
    n = 120

    # ── Test case 1: A is clearly better than B ───────────────────────────────
    # A: clean upward trend with mild noise
    a1 = (np.linspace(0, 10, n) + rng.normal(0, 0.3, n)).tolist()
    # B: heavy noise + 20% missing + 3 spike anomalies
    b1 = (np.linspace(0, 10, n) + rng.normal(0, 2.5, n)).tolist()
    spike_idx = [20, 55, 90]
    for i in spike_idx:
        b1[i] += 15.0
    missing_idx = rng.choice(n, size=24, replace=False)
    for i in missing_idx:
        b1[i] = float("nan")

    case1 = {
        "task_prompt": "Compare the quality of these two sensor readings.",
        "dataset_description": "Industrial temperature sensor, 1-minute intervals, 120 steps.",
        "series_A": a1,
        "series_B": b1,
    }

    # ── Test case 2: similar quality ──────────────────────────────────────────
    a2 = (np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 0.2, n)).tolist()
    b2 = (np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 0.25, n)).tolist()

    case2 = {
        "task_prompt": "Which series is of higher quality?",
        "dataset_description": "Simulated sinusoidal signal, 2 cycles, similar noise.",
        "series_A": a2,
        "series_B": b2,
    }

    return [("Case 1 (A clearly better)", case1), ("Case 2 (similar quality)", case2)]


def hello_world():
    print('Hello, World!')


if __name__ == "__main__":
    hello_world()

    # ── Configure model ────────────────────────────────────────────────────────
    # Available models on chatanywhere: "gpt-4o-mini", "gpt-4o",
    #   "claude-haiku-20240307", "gemini-2.5-flash", etc.
    cfg = Config(model="gpt-4o-mini")
    llm = build_llm(cfg)

    test_cases = make_test_cases()

    for name, input_data in test_cases:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print("=" * 60)

        result = run_pipeline(input_data, llm, cfg)

        print(f"  Winner     : {result.get('winner', 'N/A').upper()}")
        print(f"  Confidence : {result.get('confidence', 0):.0%}")
        print(f"  Explanation:")
        explanation = result.get("explanation", "")
        for line in explanation.split(". "):
            if line.strip():
                print(f"    • {line.strip().rstrip('.')}.")
