"""V2-H: corpus-question correspondence check.

Validates that every question in a question set actually has relevant
material in the attached corpus, before spending any Ollama/embedding
budget on a doomed run. This is the automated version of the exact mistake
that kicked off this whole project's planning -- trivia questions run
against a model with a real medical corpus attached, caught only because a
human happened to read the answers (see PLAN.md's "Reframed 2026-08-11"
section and memory feedback_no_ai_mediated_ground_truth.md).

Threshold calibrated empirically (2026-08-12), not guessed: real on-topic
questions against this project's own corpus scored 0.78-0.86 cosine
similarity; deliberately mismatched questions (leftover trivia) scored
0.20-0.27; medically-adjacent but genuinely uncovered questions (asthma,
chest pain, blood glucose -- topics this corpus doesn't discuss) landed at
0.41-0.51, in the gap, and correctly *should* fail this check since the
corpus doesn't actually cover them. 0.55 sits with margin on both sides.
"""

from . import retrieval

COVERAGE_THRESHOLD = 0.55


def check_coverage(questions: list[dict]) -> list[dict]:
    """Returns one result dict per question: {id, question, score, covered}.
    Does not raise -- callers decide what to do with poor coverage (see
    require_coverage() for the hard-block default)."""
    results = []
    for q in questions:
        scored = retrieval.retrieve_scored(q["question"])
        top_score = scored[0][0] if scored else 0.0
        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "score": round(top_score, 4),
                "covered": top_score >= COVERAGE_THRESHOLD,
            }
        )
    return results


def require_coverage(questions: list[dict]) -> None:
    """Hard-blocks a run if any question doesn't have relevant corpus
    material. Default behavior, not a soft warning -- per PLAN.md's V2-H
    scope, this is a correctness guarantee the tool makes on the user's
    behalf, not an optional lint. Raises SystemExit with the specific
    failing questions and scores, same pattern as config.py's
    require_api_key()/require_target_model()."""
    results = check_coverage(questions)
    failures = [r for r in results if not r["covered"]]
    if not failures:
        return
    lines = [
        f"  [{r['id']}] score={r['score']} (need >= {COVERAGE_THRESHOLD}): {r['question']}"
        for r in failures
    ]
    raise SystemExit(
        f"{len(failures)}/{len(results)} question(s) don't have relevant material in "
        f"corpus/ -- refusing to run (this usually means the questions don't match the "
        f"attached corpus, e.g. leftover placeholder questions after swapping corpus):\n"
        + "\n".join(lines)
        + "\n\nEither fix the mismatched questions, add corpus material that covers them, "
        "or remove them from the question set."
    )
