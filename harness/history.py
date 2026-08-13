"""Load past eval runs from results/*.json for the trend view.

Only returns runs whose schema_version matches the current one -- earlier
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


def load_comparable_runs(target_model: str | None = None) -> list[dict]:
    """All past results/*.json runs matching the current schema_version,
    sorted oldest to newest. Excludes latest.json (a duplicate of the most
    recent timestamped file, not a distinct run).

    target_model, if given, additionally filters to runs of that specific
    model -- fixes a real bug where the trend chart used to mix different
    models' scores onto the same line, which is misleading, not just
    incomplete."""
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
        runs.append(data)
    runs.sort(key=lambda r: r.get("timestamp", ""))
    return runs
