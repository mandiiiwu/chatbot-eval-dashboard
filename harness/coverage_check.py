"""V2-H: corpus-question correspondence check.

Validates that every question in a question set actually has relevant
material in the attached corpus, before spending any Ollama/embedding
budget on a doomed run. This is the automated version of the exact mistake
that kicked off this whole project's planning—trivia questions run
against a model with a real medical corpus attached, caught only because a
human happened to read the answers (see PLAN.md's "Reframed 2026-08-11"
section and memory feedback_no_ai_mediated_ground_truth.md).

Threshold calibrated empirically (2026-08-12), not guessed: real on-topic
questions against this project's own corpus scored 0.78-0.86 cosine
similarity; deliberately mismatched questions (leftover trivia) scored
0.20-0.27; medically-adjacent but genuinely uncovered questions (asthma,
chest pain, blood glucose—topics this corpus doesn't discuss) landed at
0.41-0.51, in the gap, and correctly *should* fail this check since the
corpus doesn't actually cover them. 0.55 sits with margin on both sides.
"""

from . import retrieval

COVERAGE_THRESHOLD = 0.55

# Free pre-check (harness/retrieval.py's retrieve_scored_keyword(), zero
# embedding calls) run before the expensive corpus-wide embedding pass
# above -- added 2026-08-16 after a real ~$3 run against the 116,616-chunk
# CUAD corpus completed a multi-hour embed only to fail this exact check
# afterward (see PLAN.md's full generalization audit). Calibrated on real
# data, not guessed, but deliberately coarser than COVERAGE_THRESHOLD and
# NOT as clean a gap: against the actual CUAD corpus on disk, the 41 real
# medical questions in this repo's own question files (a genuine mismatch)
# scored 0.167-0.667 keyword overlap, while 5 hand-written on-topic legal
# questions scored 0.778-1.0. The two outliers on the mismatch side (0.667,
# 0.556) were real coincidental vocabulary collisions -- one CUAD contract
# is literally a sponsorship agreement with the "American Diabetes
# Association", another is a medical-device distribution contract full of
# "diagnostic"/"clinical" language -- exactly the kind of false-negative
# risk plain keyword overlap can't avoid without semantic understanding,
# which is why this stays a supplementary pre-check, not a replacement for
# the embedding-based one below. 0.7 sits in the gap between the highest
# observed mismatch score and the lowest observed match score, with real
# margin on both sides, but on a small sample (one corpus, 46 total
# questions) -- don't treat this as calibrated as tightly as
# COVERAGE_THRESHOLD was.
KEYWORD_PRECHECK_THRESHOLD = 0.7


def check_coverage(questions: list[dict]) -> list[dict]:
    """Returns one result dict per question: {id, question, score, covered}.
    Does not raise; callers decide what to do with poor coverage (see
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


def check_coverage_keyword(questions: list[dict]) -> list[dict]:
    """Free keyword-overlap version of check_coverage() -- zero embedding
    calls, zero MicroDC cost. Same shape as check_coverage()'s return value
    so callers/tests can treat them uniformly."""
    results = []
    for q in questions:
        scored = retrieval.retrieve_scored_keyword(q["question"])
        top_score = scored[0][0] if scored else 0.0
        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "score": round(top_score, 4),
                "covered": top_score >= KEYWORD_PRECHECK_THRESHOLD,
            }
        )
    return results


def require_coverage_keyword(questions: list[dict]) -> None:
    """Cheap pre-flight before require_coverage()'s expensive embedding
    pass -- zero MicroDC cost, catches an obviously wrong question/corpus
    pairing (e.g. leftover questions from an entirely different domain)
    before the corpus gets embedded at all. Deliberately coarser than the
    real check: passing here doesn't guarantee require_coverage() will also
    pass, only that this isn't an obvious mismatch -- a genuine near-miss
    (paraphrased vocabulary, synonyms) is exactly what the embedding check
    exists to catch and this would false-negative on, so this can never be
    the only check that runs."""
    results = check_coverage_keyword(questions)
    failures = [r for r in results if not r["covered"]]
    if not failures:
        return
    lines = [
        f"  [{r['id']}] keyword score={r['score']} (need >= {KEYWORD_PRECHECK_THRESHOLD}): {r['question']}"
        for r in failures
    ]
    raise SystemExit(
        f"{len(failures)}/{len(results)} question(s) share almost no vocabulary with "
        f"anything in corpus/ -- refusing to run before spending anything on the full "
        f"embedding-based check (this usually means the questions and corpus are from "
        f"two different domains entirely, e.g. leftover questions after swapping "
        f"corpus):\n"
        + "\n".join(lines)
        + "\n\nEither fix the mismatched questions, attach a matching corpus, or pass "
        "--skip-coverage-check to bypass both checks."
    )


def require_coverage(questions: list[dict]) -> None:
    """Hard-blocks a run if any question doesn't have relevant corpus
    material. Default behavior, not a soft warning; per PLAN.md's V2-H
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
        f"corpus/—refusing to run (this usually means the questions don't match the "
        f"attached corpus, e.g. leftover placeholder questions after swapping corpus):\n"
        + "\n".join(lines)
        + "\n\nEither fix the mismatched questions, add corpus material that covers them, "
        "or remove them from the question set."
    )
