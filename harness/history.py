"""Load past eval runs from results/*.json for the trend view and V2-C's
model-comparison leaderboard.

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
    model -- the trend chart wants this (V2-C fix: it used to mix different
    models' scores onto the same line, which is misleading, not just
    incomplete), while the leaderboard wants every model, so it's optional
    rather than baked in."""
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


def leaderboard() -> list[dict]:
    """V2-C: one row per distinct target_model with schema_version-matching
    runs on disk, aggregated across all of that model's runs, sorted
    best-to-worst by avg truthfulness. Populated by running run_eval.py
    --target-model against each model you want to compare -- no new
    orchestration needed, each run is a normal run that happens to record
    a different target_model."""
    runs = load_comparable_runs()
    by_model: dict[str, list[dict]] = {}
    for r in runs:
        by_model.setdefault(r.get("target_model", "?"), []).append(r)

    rows = []
    for model, model_runs in by_model.items():
        n = len(model_runs)
        rows.append(
            {
                "target_model": model,
                "runs": n,
                "avg_truthfulness_score": round(sum(r["avg_truthfulness_score"] for r in model_runs) / n, 1),
                "avg_tone_consistency_score": round(sum(r["avg_tone_consistency_score"] for r in model_runs) / n, 1),
                "avg_concern_percentage": round(sum(r["concern_percentage"] for r in model_runs) / n, 1),
            }
        )
    rows.sort(key=lambda r: r["avg_truthfulness_score"], reverse=True)
    return rows
