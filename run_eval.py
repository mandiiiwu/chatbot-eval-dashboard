#!/usr/bin/env python3
"""Run the chatbot evaluation harness end to end.

Usage:
    python run_eval.py
    python run_eval.py --questions questions/sample_questions.json
    python run_eval.py --target-model some-other-model   # V2-C: compare models
    without hand-editing .env between runs -- see the dashboard's leaderboard
    section for the resulting side-by-side comparison.
"""

import argparse
import json

from harness import config
from harness.evaluator import run_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default=config.QUESTIONS_FILE,
        help="Path to a questions JSON file (default: questions/sample_questions.json)",
    )
    parser.add_argument(
        "--skip-coverage-check",
        action="store_true",
        help=(
            "Skip the corpus-question correspondence check (V2-H). Off by default: "
            "the check hard-blocks a run when a question doesn't have relevant "
            "material in corpus/, since that usually means mismatched "
            "questions/corpus, not a real eval worth running."
        ),
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help=(
            "Override TARGET_MODEL for this run only, without touching .env "
            "(V2-C: run the same corpus+questions against several models to "
            "compare them on the dashboard's leaderboard)."
        ),
    )
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)

    results = run_and_save(
        questions,
        skip_coverage_check=args.skip_coverage_check,
        target_model=args.target_model,
    )

    print()
    print(f"target model: {results['target_model']}")
    print(f"concern percentage: {results['concern_percentage']}%  "
          f"({results['flagged_count']}/{results['num_questions']} flagged)")
    print(f"avg truthfulness score: {results['avg_truthfulness_score']}")
    print(f"avg tone consistency score: {results['avg_tone_consistency_score']}")
    print()
    print(f"report: {config.RESULTS_DIR}/latest.html")


if __name__ == "__main__":
    main()
