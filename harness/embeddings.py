"""Thin wrapper around MicroDC's embedding job queue (qwen3-embedding:8b),
via the official microdc-client library (https://gitlab.com/microdc/python-client).

Embeddings route through MicroDC's async job queue, not the synchronous chat
endpoint the rest of the harness uses -- see PLAN.md's Phase 5. Funded by
MicroDC's account credit balance rather than local compute, unlike the
target model (Ollama) and the fact-checking classifier (local transformers)."""

import os
import time

from dotenv import load_dotenv
from microdc import Client, LLMEmbed
from microdc.exceptions.errors import APIError

load_dotenv()

EMBED_MODEL = os.environ.get("MDC_EMBED_MODEL", "qwen3-embedding:8b")

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        api_key = os.environ.get("MDC_API_KEY", "")
        if not api_key:
            raise RuntimeError("MDC_API_KEY is not set -- required for embeddings (see .env)")
        _client = Client(api_key=api_key)
    return _client


def embed_texts(texts: list[str], retries: int = 3) -> list[list[float]]:
    """Embed a batch of texts in a single job submission (cheaper and faster
    than one job per text -- the async job queue has real round-trip latency
    per submission). Retries on transient job-queue errors (e.g. a 502 while
    polling) -- confirmed necessary during testing, mirrors the retry
    behavior harness/microdc_client.py already has for the chat endpoint."""
    if not texts:
        return []
    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            job = LLMEmbed(model=EMBED_MODEL)
            job.add_texts(texts)
            job_id = client.send_job(job)
            client.wait_for_all()
            details = client.get_job_details(job_id)
            return details.result["embeddings"]
        except APIError as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Embedding job failed after {retries} attempts: {last_error}")
