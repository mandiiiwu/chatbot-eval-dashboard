"""Loads settings from .env / environment. Nothing here should require editing
the code to swap models -- change .env instead."""

import os

from dotenv import load_dotenv

load_dotenv()

MDC_BASE_URL = "https://api.microdc.ai/v1"
MDC_API_KEY = os.environ.get("MDC_API_KEY", "")

# The target model ("the chatbot under test") runs locally via Ollama
# (harness/ollama_client.py) -- it's not on MicroDC's catalog. Full MicroDC
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
# etc. special-casing) -- a homegrown one-off API won't match any specific
# vendor's shape anyway, and that's the realistic majority case for "a
# specific-purpose chatbot that isn't already OpenAI-compatible." gRPC
# explicitly out of scope, no evidence of a real need (see PLAN.md's V2-G).
#
# TARGET_PROVIDER: "ollama" (default) or "custom". When "custom":
#   CUSTOM_ENDPOINT_URL      -- the API to POST to
#   CUSTOM_ENDPOINT_HEADERS  -- JSON object string, e.g. for auth
#   CUSTOM_REQUEST_TEMPLATE  -- JSON string; {{model}}/{{system}}/{{message}}
#                               get substituted in before sending
#   CUSTOM_RESPONSE_PATH     -- dot-notation path to the answer text in the
#                               JSON response, e.g. "choices.0.message.content"
# See .env.example for a worked example.
TARGET_PROVIDER = os.environ.get("TARGET_PROVIDER", "ollama")
CUSTOM_ENDPOINT_URL = os.environ.get("CUSTOM_ENDPOINT_URL", "")
CUSTOM_ENDPOINT_HEADERS = os.environ.get("CUSTOM_ENDPOINT_HEADERS", "{}")
CUSTOM_REQUEST_TEMPLATE = os.environ.get("CUSTOM_REQUEST_TEMPLATE", '{"prompt": "{{message}}"}')
CUSTOM_RESPONSE_PATH = os.environ.get("CUSTOM_RESPONSE_PATH", "response")


def require_custom_endpoint() -> str:
    if not CUSTOM_ENDPOINT_URL:
        raise SystemExit(
            "TARGET_PROVIDER=custom but CUSTOM_ENDPOINT_URL is not set. See "
            ".env.example for the custom-endpoint config format (URL, headers, "
            "request template, response path)."
        )
    return CUSTOM_ENDPOINT_URL

# V2-I: generates candidate questions/paraphrase-variants from the corpus
# (harness/question_gen.py). Stays on MicroDC, deliberately a different
# model family than the target -- the same "don't let a model grade/design
# its own exam" principle this project already applied to judging. This
# variable used to be named JUDGE_MODEL/MDC_JUDGE_MODEL, back when MicroDC
# hosted an LLM judge; renamed when that was replaced by the local rules+NLI
# judge (see feedback_no_llm_judge memory) -- kept the same default model
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
# target_model/judge_model strings -- e.g. Phase 5's embeddings swap made
# tone_consistency_score numbers incomparable to pre-Phase-5 runs even
# though the target model didn't change. Runs from before this field
# existed have no schema_version at all (None) and should never be treated
# as current.
# v2 (2026-08-11): renamed the per-question/aggregate field from
# consistency_score/avg_consistency_score to truthfulness_score/
# avg_truthfulness_score -- disambiguates from tone_consistency_score, which
# measures something unrelated (paraphrase stability, not fact-check
# agreement) but shared the word "consistency" and was genuinely confusing.
# v3 (2026-08-13): recalibrated the none/minor severity boundary in
# fact_check.py -- "ok" no longer requires every sentence to be
# argmax-entailment (near-unachievable in practice, saw 74% minor / 2% ok
# across 3 real runs), now requires at least one entailed sentence and zero
# contradiction. Same underlying evidence/scores, different severity labels
# on top of them -- v2 runs would be silently mislabeled if compared
# directly against v3 ones. Pre-recalibration data archived, not deleted:
# results/archive_pre_severity_recalibration_2026-08-13/.
SCHEMA_VERSION = 3

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")
QUESTIONS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "questions", "sample_questions.json"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def require_api_key() -> str:
    if not MDC_API_KEY:
        raise SystemExit(
            "MDC_API_KEY is not set. Copy .env.example to .env and fill in "
            "your MicroDC.ai API key (console.microdc.ai -> API Keys)."
        )
    return MDC_API_KEY
