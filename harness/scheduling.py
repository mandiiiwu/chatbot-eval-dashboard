"""Keeps the daily scheduled eval job (Phase 6,
com.chatbot-eval-dashboard.daily.plist) in sync with whatever target_model
was last run via the dashboard's RUN_EVAL button -- a one-off UI test also
becomes the ongoing daily monitoring target, not just a single run. If the
job doesn't exist yet on this machine, creates it (portable: derives every
path from this repo's own location and config.py, not hardcoded to one
user's home directory the way the original hand-written Phase 6 plist was).

Uses plistlib to read/modify/write the plist as a parsed structure, never
as raw XML text -- same "substitute into parsed data, not a raw
string/markup blob" principle already applied twice in this project
(alerting.py's AppleScript quoting, custom_client.py's JSON templating),
for the same reason: a target_model string containing a stray character
shouldn't be able to corrupt the plist.

Deliberately does NOT persist endpoint_config (provider/URL/headers) into
the plist -- endpoint_headers can carry an auth secret, and a plist file on
disk has no special protection. The daily job keeps reading
TARGET_PROVIDER/CUSTOM_* from .env like it always has, same reasoning
already applied to keeping secrets out of results/*.json."""

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from . import config

_LABEL = "com.chatbot-eval-dashboard.daily"
_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{_LABEL}.plist"


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _venv_python(root: str) -> str:
    venv_python = os.path.join(root, ".venv", "bin", "python3")
    return venv_python if os.path.exists(venv_python) else sys.executable


# 2 AM, not the original Phase 6 default of 9 AM -- the user's explicit
# intent is that this is a batch of computations meant to run overnight
# unattended, not compete for the same machine during the day. Changed
# 2026-08-15 alongside the corpus-size incident that first made "how long
# does this actually take" worth measuring.
_DEFAULT_HOUR = 2


def _default_plist(target_model: str | None, questions_file: str | None = None) -> dict:
    root = _repo_root()
    args = [_venv_python(root), os.path.join(root, "run_eval.py")]
    resolved_questions = questions_file or config.QUESTIONS_FILE
    if resolved_questions:
        args += ["--questions", resolved_questions]
    if target_model:
        args += ["--target-model", target_model]
    return {
        "Label": _LABEL,
        "ProgramArguments": args,
        "WorkingDirectory": root,
        "StartCalendarInterval": {"Hour": _DEFAULT_HOUR, "Minute": 0},
        "StandardOutPath": os.path.join(config.RESULTS_DIR, "launchd.log"),
        "StandardErrorPath": os.path.join(config.RESULTS_DIR, "launchd.error.log"),
        "RunAtLoad": False,
    }


def _with_target_model(args: list[str], target_model: str | None) -> list[str]:
    """Returns args with any existing --target-model <value> pair removed,
    then the new one appended if given -- so re-running this repeatedly
    updates the flag in place instead of accumulating duplicates."""
    cleaned = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--target-model":
            skip_next = True
            continue
        cleaned.append(arg)
    if target_model:
        cleaned += ["--target-model", target_model]
    return cleaned


def _with_questions_file(args: list[str], questions_file: str | None) -> list[str]:
    """Same in-place-update pattern as _with_target_model(), for
    --questions -- keeps the daily job pointed at whatever question set was
    last actually used from the dashboard's RUN_EVAL button, the same way
    it already tracks target_model. Added alongside the questions-file
    selector (PLAN.md's 2026-08-16 audit fix #1): a one-off UI test against
    a newly-picked question set should also become the ongoing nightly
    target, not silently leave the scheduled job pointed at whatever file
    was configured when the plist was first created."""
    cleaned = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--questions":
            skip_next = True
            continue
        cleaned.append(arg)
    if questions_file:
        cleaned += ["--questions", questions_file]
    return cleaned


def set_schedule(hour: int, minute: int = 0) -> bool:
    """One-off schedule change (not called from the per-run RUN_EVAL sync
    path -- that's about target_model, a different concern). Creates the
    job with this schedule if it doesn't exist yet, otherwise updates the
    existing job's StartCalendarInterval and reloads it. Returns True if
    anything changed."""
    if _PLIST_PATH.exists():
        with open(_PLIST_PATH, "rb") as f:
            plist = plistlib.load(f)
    else:
        plist = _default_plist(None)
    new_interval = {"Hour": hour, "Minute": minute}
    if plist.get("StartCalendarInterval") == new_interval:
        return False
    plist["StartCalendarInterval"] = new_interval

    _PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)

    subprocess.run(["launchctl", "unload", str(_PLIST_PATH)], capture_output=True)
    subprocess.run(["launchctl", "load", str(_PLIST_PATH)], capture_output=True)
    return True


def ensure_daily_job(target_model: str | None, questions_file: str | None = None) -> bool:
    """Creates the daily job if it doesn't exist, or updates its
    --target-model/--questions and reloads it if the job exists but was
    pointed at a different model or question set. Returns True if anything
    changed (and was reloaded via launchctl), False if the job already
    matched and nothing needed to happen.

    questions_file should be a resolved path (as returned by
    config.require_questions_file()/what run_and_save() used), not a bare
    filename -- it's written directly into the plist's --questions arg."""
    if _PLIST_PATH.exists():
        with open(_PLIST_PATH, "rb") as f:
            plist = plistlib.load(f)
        new_args = _with_target_model(plist.get("ProgramArguments", []), target_model)
        new_args = _with_questions_file(new_args, questions_file)
        if new_args == plist.get("ProgramArguments", []):
            return False
        plist["ProgramArguments"] = new_args
    else:
        plist = _default_plist(target_model, questions_file)

    _PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)

    subprocess.run(["launchctl", "unload", str(_PLIST_PATH)], capture_output=True)
    subprocess.run(["launchctl", "load", str(_PLIST_PATH)], capture_output=True)
    return True
