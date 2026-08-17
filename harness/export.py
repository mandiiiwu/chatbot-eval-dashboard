"""V2-E: export flagged/minor answers as training data—"closing the loop"
from eval output back toward retraining, without actually building a
fine-tuning pipeline (that's a deliberate future decision, not something to
slip in as a side effect—see PLAN.md's V2-E and the long-term-plan note
next to it).

Deliberately does NOT use the original plan's "ungrounded (wrong) vs.
grounded (correct)" framing; grounded_answer is never scored by
fact_check.py, so it isn't verified to actually be correct. Caught a real
case of this: diabetes_q1-2's grounded_answer was an evasive non-answer
("The question is not clear..."), not something you'd want as a positive
training example. Uses the real corpus text (reference_context) as the
verified-correct side instead, and flags grounded_answer's reliability
separately (reusing fact_check.detect_vague_hedge—already validated, not
new unverified machinery) rather than silently trusting it.
"""

import json

from . import fact_check, retrieval


def _clean_reference(reference_context: str) -> str:
    """Strips markdown headers/citation boilerplate, keeping just the real
    content—reuses the same filter fact_check.py and question_gen.py
    already rely on, rather than a third copy of this logic. A training-data
    export shouldn't include "**Citation:** ..." lines as if they were part
    of the reference content."""
    chunks = [c.strip() for c in reference_context.split("\n\n") if retrieval.is_content_chunk(c)]
    return "\n\n".join(chunks) if chunks else reference_context


def export_records(results: dict, severities: tuple[str, ...] = ("flag", "minor")) -> list[dict]:
    """One record per question whose severity is in `severities`. Returns
    the records; callers decide how to persist them (see write_jsonl)."""
    records = []
    for q in results["questions"]:
        if q.get("severity") not in severities:
            continue
        grounded = q.get("grounded_answer", "")
        records.append(
            {
                "id": q["id"],
                "question": q["question"],
                "severity": q["severity"],
                "reason": q["reason"],
                "rejected": q["ungrounded_answer"],
                "reference": _clean_reference(q["reference_context"]),
                "grounded_answer": grounded,
                "grounded_answer_reliable": fact_check.detect_vague_hedge(grounded) is None,
                "target_model": results.get("target_model"),
                "run_timestamp": results.get("timestamp"),
            }
        )
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
