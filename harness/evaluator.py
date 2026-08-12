"""Core eval loop.

For each question:
  1. Ask the target model directly (no context) -- "ungrounded" answer.
  2. Ask the target model again, this time with retrieved corpus context
     injected -- "grounded" answer. This is the RAG-baseline comparison
     described in the project notes.
  3. Fact-check the ungrounded answer against the retrieved reference
     material -- locally, via harness/fact_check.py's rule-based numeric
     comparison + small NLI classifier. NOT a generative LLM judge call --
     see PLAN.md's Phase 4 and memory feedback_no_llm_judge.md for why.

Separately, questions sharing a group_id are treated as tone/phrasing
variants of the same underlying question. We measure how much the target
model's (ungrounded) answers drift across phrasings of the same question --
a consistency check that's independent of ground truth.
"""

import json
import os
from datetime import datetime, timezone

from . import config, coverage_check, embeddings, fact_check, report, retrieval
from .ollama_client import chat as target_chat


def _tone_consistency(answers: list[str]) -> float:
    """0-100 score for how similar a set of answers are to each other, via
    pairwise cosine similarity in embedding space (MicroDC's
    qwen3-embedding:8b) -- catches paraphrases/synonyms the original v1
    Jaccard-token-overlap approach missed (see PLAN.md's Phase 5). Only
    meaningful for 2+ answers. One batch embedding call for the whole group,
    not one per pair."""
    if len(answers) < 2:
        return 100.0
    vectors = embeddings.embed_texts(answers)
    scores = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            scores.append(retrieval._cosine(vectors[i], vectors[j]))
    return round(100 * sum(scores) / len(scores), 1)


def load_questions() -> list[dict]:
    with open(config.QUESTIONS_FILE) as f:
        return json.load(f)


def run_evaluation(questions: list[dict] | None = None, verbose: bool = True) -> dict:
    questions = questions if questions is not None else load_questions()
    per_question = []
    groups: dict[str, list[str]] = {}

    for q in questions:
        if verbose:
            print(f"[{q['id']}] asking target model...")
        ungrounded = target_chat(config.TARGET_MODEL, [{"role": "user", "content": q["question"]}])

        context = retrieval.retrieve_context(q["question"])
        grounded = target_chat(
            config.TARGET_MODEL,
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using ONLY the following reference material. "
                        "If it doesn't cover the question, say so explicitly.\n\n" + context
                    ),
                },
                {"role": "user", "content": q["question"]},
            ],
        )

        if verbose:
            print(f"[{q['id']}] fact-checking (local, no LLM judge)...")
        verdict = fact_check.check_answer(context, ungrounded)

        per_question.append(
            {
                "id": q["id"],
                "group_id": q.get("group_id", q["id"]),
                "question": q["question"],
                "ungrounded_answer": ungrounded,
                "grounded_answer": grounded,
                "reference_context": context,
                "truthfulness_score": verdict["truthfulness_score"],
                "severity": verdict["severity"],
                "concern": bool(verdict["concern"]),
                "reason": verdict["reason"],
                "evidence": verdict["evidence"],
            }
        )
        groups.setdefault(q.get("group_id", q["id"]), []).append(ungrounded)

    tone_scores = {gid: _tone_consistency(answers) for gid, answers in groups.items()}
    for row in per_question:
        row["tone_consistency_score"] = tone_scores[row["group_id"]]

    n = len(per_question)
    flagged = sum(1 for r in per_question if r["severity"] == "flag")
    minor = sum(1 for r in per_question if r["severity"] == "minor")
    avg_truthfulness = round(sum(r["truthfulness_score"] for r in per_question) / n, 1) if n else 0
    avg_tone = round(sum(tone_scores.values()) / len(tone_scores), 1) if tone_scores else 100.0

    return {
        "schema_version": config.SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_model": config.TARGET_MODEL,
        "judge_model": f"local: rules + {fact_check.NLI_MODEL_NAME} (no generative LLM judge)",
        "num_questions": n,
        "flagged_count": flagged,
        "minor_count": minor,
        "concern_percentage": round(100 * flagged / n, 1) if n else 0,
        "avg_truthfulness_score": avg_truthfulness,
        "avg_tone_consistency_score": avg_tone,
        "questions": per_question,
    }


def run_and_save(
    questions: list[dict] | None = None,
    skip_coverage_check: bool = False,
    verbose: bool = True,
) -> dict:
    """Load questions (if not given), run the coverage check (V2-H) unless
    skipped, run the evaluation, and persist results to results/. The single
    entry point both the CLI (run_eval.py) and the live dashboard's RUN_EVAL
    button (harness/server.py) call, so the two can't drift out of sync."""
    questions = questions if questions is not None else load_questions()
    if not skip_coverage_check:
        coverage_check.require_coverage(questions)

    results = run_evaluation(questions, verbose=verbose)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(config.RESULTS_DIR, f"{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(config.RESULTS_DIR, "latest.json"), "w") as f:
        json.dump(results, f, indent=2)
    report.write_report(results, os.path.join(config.RESULTS_DIR, "latest.html"))

    return results
