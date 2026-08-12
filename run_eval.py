#!/usr/bin/env python3
"""Run the chatbot evaluation harness end to end.

Usage:
    python run_eval.py
    python run_eval.py --questions questions/sample_questions.json
"""

import argparse
import json
import os
from datetime import datetime, timezone

from harness import config, coverage_check, report
from harness.evaluator import run_evaluation


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

    if not args.skip_coverage_check:
        coverage_check.require_coverage(questions)

    results = run_evaluation(questions)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(config.RESULTS_DIR, f"{timestamp}.json")
    latest_json = os.path.join(config.RESULTS_DIR, "latest.json")
    latest_html = os.path.join(config.RESULTS_DIR, "latest.html")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(latest_json, "w") as f:
        json.dump(results, f, indent=2)
    report.write_report(results, latest_html)

    print()
    print(f"concern percentage: {results['concern_percentage']}%  "
          f"({results['flagged_count']}/{results['num_questions']} flagged)")
    print(f"avg truthfulness score: {results['avg_truthfulness_score']}")
    print(f"avg tone consistency score: {results['avg_tone_consistency_score']}")
    print()
    print(f"report: {latest_html}")
    print(f"raw json: {json_path}")


if __name__ == "__main__":
    main()
