#!/usr/bin/env python3
"""V2-E: export flagged/minor answers from a results file as JSONL, for
manual review or later use as retraining data. Export only -- this doesn't
feed any fine-tuning pipeline (a deliberate future decision, see PLAN.md).

Usage:
    python export_training_data.py
    python export_training_data.py --results results/20260813T014434Z_rescored.json
    python export_training_data.py --output results/my_export.jsonl

Each line is a JSON record: {id, question, severity, reason, rejected
(the verified-wrong ungrounded answer), reference (the real corpus text --
verified ground truth, not a model output), grounded_answer,
grounded_answer_reliable (False if it trips the same vague-hedge detector
the dashboard uses -- don't trust it blindly), target_model, run_timestamp}.
"""

import argparse
import json

from harness import config
from harness.export import export_records, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=None,
        help="Path to a results JSON file (default: results/latest.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: results/training_export_<timestamp>.jsonl)",
    )
    args = parser.parse_args()

    results_path = args.results or f"{config.RESULTS_DIR}/latest.json"
    with open(results_path) as f:
        results = json.load(f)

    records = export_records(results)

    output_path = args.output or f"{config.RESULTS_DIR}/training_export_{results.get('timestamp', 'unknown')[:19].replace(':', '')}.jsonl"
    write_jsonl(records, output_path)

    unreliable = sum(1 for r in records if not r["grounded_answer_reliable"])
    print(f"{len(records)} record(s) exported (severity: flag/minor) from {results_path}")
    print(f"  {unreliable} have an unreliable grounded_answer (vague-hedge detected) -- check before trusting")
    print(f"written to {output_path}")


if __name__ == "__main__":
    main()
