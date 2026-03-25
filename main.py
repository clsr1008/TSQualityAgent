"""
TSqualityAgent – entry point
"""
import argparse
from dataclasses import asdict
from config import Config, build_llm
from workflow import run_pipeline
from synthetic_cases import get_cases
from run_logger import save_run


def hello_world():
    print('Hello, World!')


if __name__ == "__main__":
    hello_world()

    parser = argparse.ArgumentParser(description="TSqualityAgent – pairwise time series quality assessment")

    # ── LLM ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name on chatanywhere (e.g. gpt-4o-mini, gpt-4o, claude-haiku-20240307)",
    )

    # ── Test case selection ────────────────────────────────────────────────────
    parser.add_argument(
        "--aspect",
        type=str,
        default="bad",
        choices=["all", "bad", "rare", "pattern"],
        help="Quality aspect to test: all | bad | rare | pattern",
    )

    # ── Inspector ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--max_steps",
        type=int,
        default=6,
        help="Max ReAct steps per quality dimension in Inspector",
    )

    # ── Adjudicator reflection limits ─────────────────────────────────────────
    parser.add_argument(
        "--max_recheck",
        type=int,
        default=2,
        help="Max times Adjudicator sends Inspector back for recheck",
    )
    parser.add_argument(
        "--max_replan",
        type=int,
        default=1,
        help="Max times Adjudicator sends Perceiver back for replanning",
    )

    args = parser.parse_args()

    cfg = Config.from_args(args)
    llm = build_llm(cfg)

    aspect = None if args.aspect == "all" else args.aspect
    test_cases = get_cases(aspect)

    print(f"\nModel: {args.model}  |  Aspect: {args.aspect}  |  Cases: {len(test_cases)}")

    for name, input_data in test_cases:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print("=" * 60)

        state = run_pipeline(input_data, llm, cfg)
        result = state.get("final_result", {})

        print(f"  Winner     : {result.get('winner', 'N/A').upper()}")
        print(f"  Confidence : {result.get('confidence', 0):.0%}")
        print(f"  Explanation:")
        explanation = result.get("explanation", "")
        for line in explanation.split(". "):
            if line.strip():
                print(f"    • {line.strip().rstrip('.')}.")

        log_path = save_run(state, name, asdict(cfg))
        print(f"  Log saved  : {log_path}")