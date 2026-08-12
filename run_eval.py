#!/usr/bin/env python3
"""Run the chatbot evaluation harness end to end.

Usage:
    python run_eval.py
    python run_eval.py --questions questions/sample_questions.json
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
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)

    results = run_and_save(questions, skip_coverage_check=args.skip_coverage_check)

    print()
    print(f"concern percentage: {results['concern_percentage']}%  "
          f"({results['flagged_count']}/{results['num_questions']} flagged)")
    print(f"avg truthfulness score: {results['avg_truthfulness_score']}")
    print(f"avg tone consistency score: {results['avg_tone_consistency_score']}")
    print()
    print(f"report: {config.RESULTS_DIR}/latest.html")


if __name__ == "__main__":
    main()
