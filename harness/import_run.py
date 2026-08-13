"""Validates and persists an externally-produced results.json (the
dashboard's [IMPORT] button) -- lets a run computed elsewhere (another
machine, a colleague's export, another tool's output) join this machine's
results/ the same way a real RUN_EVAL run would, so it shows up in the
trend chart and severity groups like any other run.

Strict schema validation, not a permissive merge: a malformed or
wrong-shaped file is rejected outright with a specific reason, never
silently coerced or partially accepted. Silently accepting a shape
mismatch would corrupt the trend chart's comparability guarantees the
same way schema_version/target_model/corpus_fingerprint already protect
against mixing incompatible runs (see harness/history.py) -- an import is
just another way an incompatible run could sneak in."""

import json
import os
from datetime import datetime, timezone

from . import config, report


class ImportValidationError(ValueError):
    pass


_REQUIRED_TOP_LEVEL = {
    "schema_version": int,
    "timestamp": str,
    "target_model": str,
    "judge_model": str,
    "num_questions": int,
    "flagged_count": int,
    "minor_count": int,
    "concern_percentage": (int, float),
    "avg_truthfulness_score": (int, float),
    "avg_tone_consistency_score": (int, float),
    "questions": list,
}

_REQUIRED_QUESTION_FIELDS = {
    "id": str,
    "group_id": str,
    "question": str,
    "ungrounded_answer": str,
    "grounded_answer": str,
    "reference_context": str,
    "truthfulness_score": (int, float),
    "severity": str,
    "concern": bool,
    "reason": str,
    "evidence": dict,
    "tone_consistency_score": (int, float),
}

_VALID_SEVERITIES = {"flag", "minor", "none"}  # "none" here is the raw severity value fact_check.py assigns; the dashboard displays it as "OK"


def _type_ok(value, expected_type) -> bool:
    """isinstance() but bool never silently matches int/float -- Python's
    bool is a subclass of int, so isinstance(True, int) is True by
    default, which would let a JSON boolean sneak past as a numeric
    field (e.g. num_questions: true)."""
    if isinstance(value, bool):
        return expected_type is bool or (isinstance(expected_type, tuple) and bool in expected_type)
    return isinstance(value, expected_type)


def validate(data) -> None:
    """Raises ImportValidationError with a specific, actionable reason on
    the first mismatch found; returns silently if the shape is valid.
    Unknown extra fields are allowed through untouched (e.g.
    corpus_fingerprint, additive since schema_version 3) -- this checks
    that the required shape is present and correctly typed, not that
    nothing else exists."""
    if not isinstance(data, dict):
        raise ImportValidationError("top level must be a JSON object")

    for field, expected_type in _REQUIRED_TOP_LEVEL.items():
        if field not in data:
            raise ImportValidationError(f"missing required field: {field!r}")
        if not _type_ok(data[field], expected_type):
            raise ImportValidationError(f"field {field!r} has the wrong type (expected {expected_type})")

    if data["schema_version"] != config.SCHEMA_VERSION:
        raise ImportValidationError(
            f"schema_version {data['schema_version']} does not match this harness's "
            f"current schema_version {config.SCHEMA_VERSION} -- results from a different "
            f"harness architecture aren't comparable (see PLAN.md's SCHEMA_VERSION history)"
        )

    try:
        dt = datetime.fromisoformat(data["timestamp"])
    except ValueError:
        raise ImportValidationError(f"timestamp {data['timestamp']!r} is not a valid ISO 8601 datetime")

    if not data["questions"]:
        raise ImportValidationError("questions list is empty")

    for i, q in enumerate(data["questions"]):
        if not isinstance(q, dict):
            raise ImportValidationError(f"questions[{i}] is not an object")
        for field, expected_type in _REQUIRED_QUESTION_FIELDS.items():
            if field not in q:
                raise ImportValidationError(f"questions[{i}] missing required field: {field!r}")
            if not _type_ok(q[field], expected_type):
                raise ImportValidationError(
                    f"questions[{i}].{field} has the wrong type (expected {expected_type})"
                )
        if q["severity"] not in _VALID_SEVERITIES:
            raise ImportValidationError(
                f"questions[{i}].severity {q['severity']!r} must be one of {sorted(_VALID_SEVERITIES)}"
            )

    if len(data["questions"]) != data["num_questions"]:
        raise ImportValidationError(
            f"num_questions ({data['num_questions']}) does not match "
            f"len(questions) ({len(data['questions'])})"
        )

    return dt


def save(data: dict) -> str:
    """Validates, then persists exactly like evaluator.run_and_save() does
    -- a timestamped file plus latest.json/latest.html -- so an imported
    run behaves identically to one this machine produced itself. The
    filename is derived from the run's own `timestamp` field (not import
    time), so re-importing the same export is idempotent rather than
    piling up duplicate files. Returns the timestamped filename written."""
    dt = validate(data)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    timestamp = dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(config.RESULTS_DIR, f"{timestamp}_imported.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    with open(os.path.join(config.RESULTS_DIR, "latest.json"), "w") as f:
        json.dump(data, f, indent=2)
    report.write_report(data, os.path.join(config.RESULTS_DIR, "latest.html"))

    return os.path.basename(json_path)
