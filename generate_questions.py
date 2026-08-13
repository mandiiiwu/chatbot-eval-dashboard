#!/usr/bin/env python3
"""V2-I: auto-generate an eval question set from corpus/.

Usage:
    python generate_questions.py
    python generate_questions.py --questions-per-topic 5 --variants-per-question 5
    python generate_questions.py --output questions/generated_questions.json

Writes a questions/*.json file in the same schema as sample_questions.json
(id/group_id/question) -- run it against your target model the same way:
    python run_eval.py --questions questions/generated_questions.json

Doesn't touch questions/sample_questions.json -- generated questions are
opt-in via --questions, not a replacement for hand-curated ones.
"""

import argparse
import json

from harness.question_gen import (
    QUESTIONS_PER_TOPIC_DEFAULT,
    VARIANTS_PER_QUESTION_DEFAULT,
    generate_questions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-per-topic",
        type=int,
        default=QUESTIONS_PER_TOPIC_DEFAULT,
        help=f"Ceiling, not a guarantee -- generates fewer if the corpus doesn't "
             f"honestly support the count (default: {QUESTIONS_PER_TOPIC_DEFAULT})",
    )
    parser.add_argument(
        "--variants-per-question",
        type=int,
        default=VARIANTS_PER_QUESTION_DEFAULT,
        help=f"Phrasing variants per question, including the canonical one "
             f"(default: {VARIANTS_PER_QUESTION_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        default="questions/generated_questions.json",
        help="Output path (default: questions/generated_questions.json)",
    )
    args = parser.parse_args()

    questions, stats = generate_questions(
        questions_per_topic=args.questions_per_topic,
        variants_per_question=args.variants_per_question,
    )

    with open(args.output, "w") as f:
        json.dump(questions, f, indent=2)

    print()
    print("per-topic results (requested ceiling vs. what the corpus actually supported):")
    for s in stats:
        print(
            f"  [{s['topic']}] {s['chunks_available']} content chunk(s) available -- "
            f"{s['questions_accepted']}/{s['questions_requested']} questions, "
            f"{s['variants_accepted']}/{s['variants_requested']} total variants"
        )
    print()
    print(f"{len(questions)} question(s) written to {args.output}")
    print(f"run them with: python run_eval.py --questions {args.output}")


if __name__ == "__main__":
    main()
