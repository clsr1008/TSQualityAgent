"""
TSqualityAgent – entry point
"""
import argparse
from dataclasses import asdict
from config import Config, build_llm
from models.llm import CHATANYWHERE_BASE_URL
from workflow import run_pipeline
from synthetic_cases import get_cases, CASE_NAMES
from run_logger import save_run


def hello_world():
    print('Hello, TSQualityAgent!')


if __name__ == "__main__":
    hello_world()

    parser = argparse.ArgumentParser(description="TSqualityAgent – pairwise time series quality assessment")

    # ── LLM ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
        help="Model name, e.g. gpt-5.4-mini (cloud) or Qwen/Qwen3-4B (local vLLM)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=CHATANYWHERE_BASE_URL,
        help="OpenAI-compatible API base URL. Use http://localhost:8000/v1 for local vLLM.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key. Leave empty to use OPENAI_API_KEY env var. Use 'EMPTY' for local vLLM.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 thinking mode (disabled by default for speed).",
    )

    # ── Test case selection ────────────────────────────────────────────────────
    parser.add_argument(
        "--case",
        type=str,
        nargs="+",
        default=None,
        help=f"Test case(s) to run: all | one or more of: {' '.join(CASE_NAMES)} (default: all)",
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

    cases = None if (args.case is None or "all" in args.case) else args.case
    test_cases = get_cases(cases)

    case_label = " ".join(args.case) if args.case else "all"
    print(f"\nModel: {args.model}  |  Base URL: {args.base_url}  |  Case: {case_label}  |  Cases: {len(test_cases)}")

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