"""Loads settings from .env / environment. Nothing here should require editing
the code to swap models—change .env instead."""

import os

from dotenv import load_dotenv

load_dotenv()

MDC_BASE_URL = "https://api.microdc.ai/v1"
MDC_API_KEY = os.environ.get("MDC_API_KEY", "")

# The target model ("the chatbot under test") runs locally via Ollama
# (harness/ollama_client.py); it's not on MicroDC's catalog. Full MicroDC
# catalog + live pricing: https://api.microdc.ai/api/public/models
#
# TARGET_MODEL has no default: this harness is meant to evaluate whatever
# specific-purpose chatbot you point it at (medical, legal, support, etc),
# so silently falling back to one particular model would be misleading.
# Every run must say explicitly what it's testing.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
TARGET_MODEL = os.environ.get("TARGET_MODEL", "")

# V2-G: lets the harness point at essentially any REST/JSON chatbot API, not
# just OpenAI-compatible ones. Deliberately a generic templated HTTP client
# (harness/custom_client.py) rather than a provider registry (HF/AWS/Azure/
# etc. special-casing); a homegrown one-off API won't match any specific
# vendor's shape anyway, and that's the realistic majority case for "a
# specific-purpose chatbot that isn't already OpenAI-compatible." gRPC
# explicitly out of scope, no evidence of a real need (see PLAN.md's V2-G).
#
# TARGET_PROVIDER: "ollama" (default) or "custom". When "custom":
#   CUSTOM_ENDPOINT_URL is the API to POST to.
#   CUSTOM_ENDPOINT_HEADERS is a JSON object string, e.g. for auth.
#   CUSTOM_REQUEST_TEMPLATE is a JSON string; {{model}}/{{system}}/{{message}}
#     get substituted in before sending.
#   CUSTOM_RESPONSE_PATH is a dot-notation path to the answer text in the
#     JSON response, e.g. "choices.0.message.content".
# See .env.example for a worked example.
TARGET_PROVIDER = os.environ.get("TARGET_PROVIDER", "ollama")
CUSTOM_ENDPOINT_URL = os.environ.get("CUSTOM_ENDPOINT_URL", "")
CUSTOM_ENDPOINT_HEADERS = os.environ.get("CUSTOM_ENDPOINT_HEADERS", "{}")
CUSTOM_REQUEST_TEMPLATE = os.environ.get("CUSTOM_REQUEST_TEMPLATE", '{"prompt": "{{message}}"}')
CUSTOM_RESPONSE_PATH = os.environ.get("CUSTOM_RESPONSE_PATH", "response")

# V2-I: generates candidate questions/paraphrase-variants from the corpus
# (harness/question_gen.py). Stays on MicroDC, deliberately a different
# model family than the target—the same "don't let a model grade/design
# its own exam" principle this project already applied to judging. This
# variable used to be named JUDGE_MODEL/MDC_JUDGE_MODEL, back when MicroDC
# hosted an LLM judge; renamed when that was replaced by the local rules+NLI
# judge (see feedback_no_llm_judge memory)—kept the same default model
# and env var *value*, since gpt-oss:120b already satisfied the
# different-model-family requirement, just gained a new job.
GENERATION_MODEL = os.environ.get("MDC_GENERATION_MODEL", "gpt-oss:120b")


def require_target_model() -> str:
    if not TARGET_MODEL:
        raise SystemExit(
            "TARGET_MODEL is not set. Set it in .env to the Ollama tag of the "
            "chatbot you want to evaluate (e.g. `ollama list` to see what's "
            "available locally)."
        )
    return TARGET_MODEL

# Bumped whenever a structural change makes results incomparable to earlier
# runs (target model swap, judge mechanism change, retrieval mechanism
# change, results schema change). Embedded in every run's output
# (evaluator.run_evaluation) so history/trend code can tell which past runs
# are actually comparable to the current setup rather than guessing from
# target_model/judge_model strings—e.g. Phase 5's embeddings swap made
# tone_consistency_score numbers incomparable to pre-Phase-5 runs even
# though the target model didn't change. Runs from before this field
# existed have no schema_version at all (None) and should never be treated
# as current.
# v2 (2026-08-11): renamed the per-question/aggregate field from
# consistency_score/avg_consistency_score to truthfulness_score/
# avg_truthfulness_score—disambiguates from tone_consistency_score, which
# measures something unrelated (paraphrase stability, not fact-check
# agreement) but shared the word "consistency" and was genuinely confusing.
# v3 (2026-08-13): recalibrated the none/minor severity boundary in
# fact_check.py; "ok" no longer requires every sentence to be
# argmax-entailment (near-unachievable in practice, saw 74% minor / 2% ok
# across 3 real runs), now requires at least one entailed sentence and zero
# contradiction. Same underlying evidence/scores, different severity labels
# on top of them; v2 runs would be silently mislabeled if compared
# directly against v3 ones. Pre-recalibration data archived, not deleted:
# results/archive_pre_severity_recalibration_2026-08-13/.
# v4 (2026-08-14): generalized fact_check.py's numeric-claim regex past its
# original medical-only unit whitelist (added currency, time/duration, comma
# thousands-separators; see PLAN.md) -- a genuinely new detection capability,
# not a redefinition of existing evidence like v3 was, so no archiving needed;
# old runs just stop showing in the trend chart under the new schema_version,
# same as any other judge-mechanism change. Verified as a strict superset
# against this project's own real medical run first (35/35 questions produced
# identical severity + score under the new rules) before generalizing further.
SCHEMA_VERSION = 4

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "questions")

# No hardcoded default filename here on purpose (used to be
# questions/sample_questions.json unconditionally). That silent default is
# exactly what let a real run fire against a stale medical question set
# after corpus/ had already been swapped to something else entirely (see
# PLAN.md's 2026-08-16 audit) -- a doomed-from-the-start run that still cost
# real MicroDC money. require_questions_file() below still enforces this
# same "no silent default" guarantee at THIS layer -- it never guesses a
# file. What changed later the same day: evaluator.run_and_save() (one
# layer up) now defaults to auto-generating a fresh question set from the
# corpus when this raises, rather than propagating the error -- a
# deliberate, visible fallback, not a silent one (the generation step is
# announced, progress-tracked, and cost-estimated same as everything else).
# require_questions_file() itself is unchanged: it's what actually fires
# when that fallback is explicitly turned off (--no-auto-generate-questions
# on the CLI), or when a specific file was named but doesn't exist on disk.
QUESTIONS_FILE = os.environ.get("QUESTIONS_FILE", "")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def require_questions_file(path: str | None = None) -> str:
    """Resolves and validates a questions file path, same "no silent
    default" pattern as require_target_model()/require_api_key(). path, if
    given, overrides QUESTIONS_FILE for this call only (the CLI's
    --questions flag, or the dashboard's per-run questions-file selector)
    without touching .env -- same override convention as target_model
    elsewhere in this harness. Relative paths are resolved against the repo
    root (so "questions/foo.json", the existing --questions convention,
    keeps working)."""
    path = path if path is not None else QUESTIONS_FILE
    if not path:
        raise SystemExit(
            "No questions file configured. Set QUESTIONS_FILE in .env to a "
            "file under questions/, pass --questions on the CLI, or pick "
            "one from the dashboard's [CONFIG] rail. Don't have one yet? "
            "`python generate_questions.py` creates one from your corpus."
        )
    resolved = path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)
    if not os.path.isfile(resolved):
        raise SystemExit(f"Questions file not found: {path}")
    return resolved


def require_api_key() -> str:
    if not MDC_API_KEY:
        raise SystemExit(
            "MDC_API_KEY is not set. Copy .env.example to .env and fill in "
            "your MicroDC.ai API key (console.microdc.ai -> API Keys)."
        )
    return MDC_API_KEY


def reload() -> None:
    """Re-reads .env and refreshes this module's settings in place. The
    values below were all read once at import time; that's fine for a
    short-lived CLI run, but the dashboard server now stays up for days
    (Phase 6's persistent-server launchd agent), so an .env edit made
    while it's running would otherwise sit ignored until a manual
    restart -- unlike a UI-based config change, which takes effect on the
    next action with no restart needed. server.py calls this at the start
    of every request so both paths behave the same way. override=True is
    required: plain load_dotenv() leaves an already-set os.environ value
    alone, so without it a changed .env value would never actually
    take effect on a second call."""
    load_dotenv(override=True)
    global MDC_API_KEY, OLLAMA_BASE_URL, TARGET_MODEL, TARGET_PROVIDER
    global CUSTOM_ENDPOINT_URL, CUSTOM_ENDPOINT_HEADERS, CUSTOM_REQUEST_TEMPLATE, CUSTOM_RESPONSE_PATH
    global GENERATION_MODEL, QUESTIONS_FILE
    MDC_API_KEY = os.environ.get("MDC_API_KEY", "")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    TARGET_MODEL = os.environ.get("TARGET_MODEL", "")
    TARGET_PROVIDER = os.environ.get("TARGET_PROVIDER", "ollama")
    CUSTOM_ENDPOINT_URL = os.environ.get("CUSTOM_ENDPOINT_URL", "")
    CUSTOM_ENDPOINT_HEADERS = os.environ.get("CUSTOM_ENDPOINT_HEADERS", "{}")
    CUSTOM_REQUEST_TEMPLATE = os.environ.get("CUSTOM_REQUEST_TEMPLATE", '{"prompt": "{{message}}"}')
    CUSTOM_RESPONSE_PATH = os.environ.get("CUSTOM_RESPONSE_PATH", "response")
    GENERATION_MODEL = os.environ.get("MDC_GENERATION_MODEL", "gpt-oss:120b")
    QUESTIONS_FILE = os.environ.get("QUESTIONS_FILE", "")
