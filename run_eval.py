#!/usr/bin/env python3
"""Run the chatbot evaluation harness end to end.

Usage:
    python run_eval.py
    python run_eval.py --questions questions/sample_questions.json
    python run_eval.py --target-model some-other-model   # V2-C: compare models
    without hand-editing .env between runs.

If no questions file is configured (--questions / .env's QUESTIONS_FILE),
one gets generated automatically from corpus/ before the run -- same
behavior as the dashboard's RUN_EVAL button, so a corpus+model set up
through this CLI, dropped directly into corpus/ on disk, or configured
through the dashboard UI all converge on the exact same process. Pass
--no-auto-generate-questions for the old strict-fail behavior instead.
"""

import argparse

from harness import config
from harness.evaluator import run_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default=None,
        help="Path to a questions JSON file (default: $QUESTIONS_FILE from .env)",
    )
    parser.add_argument(
        "--no-auto-generate-questions",
        action="store_true",
        help=(
            "If no questions file resolves (--questions / .env's QUESTIONS_FILE), "
            "fail with a clear error instead of generating one from corpus/ "
            "automatically (the default -- same pipeline as generate_questions.py, "
            "run implicitly). Use this for scripted/CI usage where an unexpected "
            "MicroDC spend from auto-generation is worse than a clear failure."
        ),
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
            "(V2-C: run the same corpus+questions against several models "
            "without hand-editing .env between runs)."
        ),
    )
    args = parser.parse_args()

    # No silent default here: run_and_save() -> config.require_questions_file()
    # raises a clear SystemExit (not a bare FileNotFoundError) if neither
    # --questions nor .env's QUESTIONS_FILE resolves to a real file and
    # --no-auto-generate-questions was passed -- see PLAN.md's 2026-08-16
    # audit for why a silent default was the problem. Without that flag,
    # a missing questions file now generates one from corpus/ instead,
    # same as the dashboard's RUN_EVAL button.
    results = run_and_save(
        questions_file=args.questions,
        auto_generate_questions=not args.no_auto_generate_questions,
        skip_coverage_check=args.skip_coverage_check,
        target_model=args.target_model,
    )

    print()
    print(f"target model: {results['target_model']}")
    print(f"questions file: {results.get('questions_file', 'n/a')}")
    print(f"concern percentage: {results['concern_percentage']}%  "
          f"({results['flagged_count']}/{results['num_questions']} flagged)")
    print(f"avg truthfulness score: {results['avg_truthfulness_score']}")
    print(f"avg tone consistency score: {results['avg_tone_consistency_score']}")
    minutes, seconds = divmod(results.get("duration_seconds", 0), 60)
    print(f"duration: {int(minutes)}m {seconds:.0f}s")
    print(f"microdc cost: ${results.get('microdc_cost_usd', 0):.4f}")
    print()
    print(f"report: {config.RESULTS_DIR}/latest.html")


if __name__ == "__main__":
    main()
