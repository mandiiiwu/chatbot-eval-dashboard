"""Load past eval runs from results/*.json for the trend view.

Only returns runs whose schema_version matches the current one; earlier
architectures (different target model, different judge mechanism, different
retrieval mechanism) produce numbers that aren't comparable to current runs
even when they look superficially similar (e.g. Phase 5's embeddings swap
changed what tone_consistency_score even means). Runs from before
schema_version existed have no such field at all and are always excluded.
See harness/config.py's SCHEMA_VERSION docstring and PLAN.md's Phase 6."""

import glob
import json
import os

from . import config


def load_comparable_runs(
    target_model: str | None = None,
    corpus_fingerprint: str | None = None,
    questions_fingerprint: str | None = None,
) -> list[dict]:
    """All past results/*.json runs matching the current schema_version,
    sorted oldest to newest. Excludes latest.json (a duplicate of the most
    recent timestamped file, not a distinct run).

    target_model, if given, additionally filters to runs of that specific
    model—fixes a real bug where the trend chart used to mix different
    models' scores onto the same line, which is misleading, not just
    incomplete.

    corpus_fingerprint, if given, additionally filters to runs against that
    exact corpus content—same principle, for corpus swaps instead of
    model swaps (e.g. this project's medical corpus vs. a future legal/HR
    one). Runs from before this field existed have none at all (None) and
    are correctly excluded when a fingerprint filter is active, same as
    missing schema_version already is.

    questions_fingerprint, if given, additionally filters to runs that used
    the exact same question set content—same principle again, but for
    questions instead of corpus/model. Added 2026-08-16 alongside making
    auto_generate_questions the default everywhere: an auto-generated
    question set always saves to the same filename
    (questions/generated_questions.json) regardless of what MicroDC
    actually generated that time (temperature 0.7-0.8, so two generation
    runs against the identical corpus can produce different real question
    text), so `results['questions_file']`'s basename alone can no longer
    be trusted to mean "the same questions were asked"—only the content
    fingerprint can. Without this filter, two runs that happened to ask
    genuinely different questions (because the file got regenerated
    between them) could land on the same trend line looking like model
    drift that's actually just different questions."""
    runs = []
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "*.json"))):
        if os.path.basename(path) == "latest.json":
            continue
        with open(path) as f:
            data = json.load(f)
        if data.get("schema_version") != config.SCHEMA_VERSION:
            continue
        if target_model is not None and data.get("target_model") != target_model:
            continue
        if corpus_fingerprint is not None and data.get("corpus_fingerprint") != corpus_fingerprint:
            continue
        if questions_fingerprint is not None and data.get("questions_fingerprint") != questions_fingerprint:
            continue
        runs.append(data)
    runs.sort(key=lambda r: r.get("timestamp", ""))
    return runs
