"""V2-D: threshold-based alerting when concern_percentage crosses a line.

Channel and threshold both decided directly with the user (2026-08-13), not
guessed; the original plan explicitly left the channel "TBD with user."
Local macOS notification (osascript) chosen over email/Slack: zero external
services or credentials to manage, and the daily launchd job (Phase 6)
already runs on this machine, so a native notification reaches the user the
same way the job itself does.

concern_percentage isn't shown on the dashboard anymore (dropped in the
Claude Design redesign), but it's still computed and stored in every run's
JSON; this reads that existing field, doesn't need a new one.

Deliberately does NOT decide what happens when the alert fires beyond
notifying; per the original plan, that's "a human decision, not something
to automate away."
"""

import subprocess

# 25% chosen directly by the user (2026-08-13), not guessed: their 3 real
# runs of the current 35-question set landed at 20.0%, 28.6%, and 22.9%
# concern; 25% sits inside that range (would have fired on 1 of 3), a
# deliberately more sensitive choice than the alternative (30%, which
# would have stayed quiet on all 3). If the corpus/questions change
# significantly, this may need recalibrating the same way other thresholds
# in this project have been (see coverage_check.py, question_gen.py).
ALERT_THRESHOLD_PERCENT = 25.0


def maybe_alert(results: dict) -> bool:
    """Fires a local macOS notification if results['concern_percentage']
    exceeds ALERT_THRESHOLD_PERCENT. Returns whether it fired. Never raises;
    a notification failure (e.g. not running on macOS) shouldn't take down
    an otherwise-successful eval run."""
    concern = results.get("concern_percentage", 0)
    if concern <= ALERT_THRESHOLD_PERCENT:
        return False

    title = "Chatbot Eval Dashboard—concern threshold exceeded"
    message = (
        f"{results.get('target_model', '?')}: {concern}% concern "
        f"({results.get('flagged_count', '?')}/{results.get('num_questions', '?')} flagged)"
        f"—exceeds {ALERT_THRESHOLD_PERCENT}% threshold"
    )
    # osascript -e runs the string as AppleScript; an unescaped " or \ in
    # message/title (both can contain user-controlled text, e.g. a
    # --target-model value) would break out of the quoted string literal,
    # not just corrupt the notification but let arbitrary AppleScript
    # follow. Escape both before interpolating.
    def _escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    script = f'display notification "{_escape(message)}" with title "{_escape(title)}"'
    try:
        subprocess.run(["osascript", "-e", script], check=True, timeout=10)
        return True
    except Exception:
        # Local notification is a nice-to-have on top of a real eval run;
        # never let it fail the run itself.
        return False
