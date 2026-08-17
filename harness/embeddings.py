"""Thin wrapper around MicroDC's embedding job queue (qwen3-embedding:8b),
via the official microdc-client library (https://gitlab.com/microdc/python-client).

Embeddings route through MicroDC's async job queue, not the synchronous chat
endpoint the rest of the harness uses—see PLAN.md's Phase 5. Funded by
MicroDC's account credit balance rather than local compute, unlike the
target model (Ollama) and the fact-checking classifier (local transformers)."""

import os
import threading
import time
from typing import Callable

from dotenv import load_dotenv
from microdc import Client, LLMEmbed
from microdc.exceptions.errors import APIError

load_dotenv()

EMBED_MODEL = os.environ.get("MDC_EMBED_MODEL", "qwen3-embedding:8b")

_client: Client | None = None

# MicroDC's own job-details response includes actual_cost directly (real
# ground truth from their billing, confirmed via a real test job -- not an
# estimate reverse-engineered from a pricing table, which would risk being
# systematically wrong). Accumulated here across every embed_texts() call
# so evaluator.py can report total MicroDC spend for a whole eval run, not
# just per-call. Lock guards it since the dashboard server is now threaded.
_cost_lock = threading.Lock()
_total_cost_usd = 0.0


# Empirically observed: a 32-character test job and a real 29,120-character
# (50-chunk) job both cost exactly this -- strongly suggests a per-job
# minimum fee dominates at small-to-medium batch sizes, not a meaningful
# per-character rate (verified before relying on it: a linear
# extrapolation from just the 32-character sample alone would have been
# wrong by ~900x). Used only for a rough pre-run cost estimate shown to
# the user before they click RUN_EVAL, never for real billing math --
# MicroDC's own actual_cost (see _record_cost) is always the real number
# actually charged.
OBSERVED_MIN_JOB_COST_USD = 0.0015


def reset_cost_tracker() -> None:
    global _total_cost_usd
    with _cost_lock:
        _total_cost_usd = 0.0


def get_total_cost() -> float:
    with _cost_lock:
        return _total_cost_usd


def _record_cost(details) -> None:
    global _total_cost_usd
    cost = getattr(details, "actual_cost", None)
    if cost is None:
        cost = getattr(details, "estimated_cost", None)
    if cost is not None:
        with _cost_lock:
            _total_cost_usd += cost


def _get_client() -> Client:
    global _client
    if _client is None:
        api_key = os.environ.get("MDC_API_KEY", "")
        if not api_key:
            raise RuntimeError("MDC_API_KEY is not set—required for embeddings (see .env)")
        _client = Client(api_key=api_key)
    return _client


class EmbeddingJobError(RuntimeError):
    pass


# Real failure, not theoretical: a 116,616-chunk corpus (a full raw dataset
# dump, not curated excerpts) sent as one job crashed with a bare
# "NoneType is not subscriptable"-- MicroDC rejected the job as too large,
# and embed_texts() didn't check for that before reading the result. 200
# is a conservative choice, not a value confirmed as MicroDC's actual
# limit (that limit was never found, on purpose -- probing for the exact
# threshold would mean deliberately sending oversized jobs and eating more
# failed-job cost just to find a number). Validated with real batches up
# to 500 chunks (196,913 characters) succeeding fine, at the same
# per-job minimum cost as much smaller batches -- 200 sits comfortably
# below the largest size actually confirmed to work.
MAX_BATCH_SIZE = 200

# Real, measured, not guessed (2026-08-16): a full CUAD-corpus run (584
# MAX_BATCH_SIZE-sized jobs) took over 8 hours because embed_texts() used to
# submit one job, then block on client.wait_for_all() until *that job
# alone* completed, before even submitting the next -- fully serial, zero
# concurrency, despite the MicroDC client already supporting "submit many,
# then wait once" (client.send_job() just tracks the job; wait_for_all()
# waits for every currently-pending job together, submission itself is a
# fast HTTP POST -- the slow part is server-side processing, which happens
# after submission regardless of whether you're waiting for it yet).
# Confirmed empirically before committing to this: 3 real 50-chunk jobs
# submitted serially (submit-wait-fetch, repeat) took 44.1s total; the same
# 3 jobs submitted concurrently (submit all 3, wait once) took 19.2s;
# 10 concurrent real jobs completed cleanly in 16.7s with zero errors and
# perfectly linear cost ($0.0015 x 10 = $0.015, no surprise pricing from
# concurrency itself). 10 sits at the largest concurrency actually tested,
# not probed beyond that on purpose -- same "don't deliberately send
# oversized/excessive jobs just to find a limit" philosophy already applied
# to MAX_BATCH_SIZE above. Windowed (not "submit literally everything at
# once") so a 584-job corpus run doesn't try to open hundreds of
# simultaneous jobs against an untested ceiling.
MAX_CONCURRENT_JOBS = 10

# Real, observed (2026-08-17): a full CUAD-corpus embed against the actual
# production corpus froze at 279/584 batches for 3+ minutes straight, zero
# progress, process alive, one live connection held open to MicroDC. Root
# cause found by reading the installed microdc-client source directly:
# every client.wait_for_all() call below passed no timeout, and that
# library's own default is timeout=None -- wait forever. MicroDC runs jobs
# on a peer-to-peer GPU marketplace (see microdc_client.py's chat()
# docstring for the same point made independently, well before this
# incident) -- an individual worker stalling is a known, expected failure
# mode there, not a hypothetical. 120s matches chat()'s own established
# timeout for the same reason: generous enough not to falsely time out a
# real slow job, short enough that one stuck worker can't freeze an entire
# run indefinitely -- the retry loops already in place here (in both
# _embed_batch and _embed_batches_concurrent) take it from there.
WAIT_FOR_ALL_TIMEOUT = 120


def embed_texts(
    texts: list[str],
    retries: int = 3,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Embeds arbitrarily many texts by splitting into MAX_BATCH_SIZE-sized
    jobs and concatenating the results in order. Batches are submitted in
    windows of up to MAX_CONCURRENT_JOBS at a time -- see that constant's
    docstring for the real measurements behind why this isn't still fully
    serial. A single oversized job (no batching at all) is what originally
    caused a bare "NoneType is not subscriptable" crash; serial batching
    fixed correctness but left an 8-hour real-world runtime on the table,
    which concurrency is what actually addresses.

    progress_callback(done, total), if given, fires once immediately with
    (0, total_batches) before any batch completes -- so a caller watching
    for live feedback (see retrieval.py's corpus-wide embed, the one call
    site that actually passes this) has something to show right away
    instead of nothing until the first (possibly tens-of-seconds-away)
    batch finishes -- then again after every batch completes."""
    if not texts:
        return []
    batches = [texts[i : i + MAX_BATCH_SIZE] for i in range(0, len(texts), MAX_BATCH_SIZE)]
    if progress_callback:
        progress_callback(0, len(batches))
    if len(batches) == 1:
        result = _embed_batch(batches[0], retries=retries)
        if progress_callback:
            progress_callback(1, 1)
        return result
    return _embed_batches_concurrent(batches, retries=retries, progress_callback=progress_callback)


def _submit_batch(client: Client, texts: list[str]) -> str:
    job = LLMEmbed(model=EMBED_MODEL)
    job.add_texts(texts)
    return client.send_job(job)


def _fetch_batch_result(client: Client, job_id: str, batch_size: int) -> list[list[float]]:
    """Explicitly checks is_successful before touching .result: a job that
    completes but fails still returns a details object with no APIError
    raised, just result=None -- blindly doing details.result["embeddings"]
    in that case crashes with a bare "NoneType is not subscriptable" that
    gives no hint what actually went wrong."""
    details = client.get_job_details(job_id)
    _record_cost(details)
    if not getattr(details, "is_successful", True):
        raise EmbeddingJobError(
            f"MicroDC embedding job failed (status={getattr(details, 'status', '?')}, "
            f"{batch_size} texts submitted): "
            f"{getattr(details, 'error_message', None) or 'no error message given'}"
        )
    if details.result is None or "embeddings" not in details.result:
        raise EmbeddingJobError(
            f"MicroDC embedding job returned no result ({batch_size} texts submitted, "
            f"status={getattr(details, 'status', '?')}) -- the batch may be too large "
            f"for a single job; consider lowering MAX_BATCH_SIZE"
        )
    return details.result["embeddings"]


def _embed_batch(texts: list[str], retries: int = 3) -> list[list[float]]:
    """One MicroDC job for up to MAX_BATCH_SIZE texts, submitted and waited
    on alone -- used for the single-batch case (embed_texts() skips the
    concurrent path entirely when there's only one batch) and as the
    per-batch retry unit inside _embed_batches_concurrent(). Retries on
    transient job-queue errors (e.g. a 502 while polling) and on a job that
    never completes (WAIT_FOR_ALL_TIMEOUT)—confirmed necessary during
    testing, mirrors the retry behavior harness/microdc_client.py already
    has for the chat endpoint."""
    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            job_id = _submit_batch(client, texts)
            client.wait_for_all(timeout=WAIT_FOR_ALL_TIMEOUT)
            return _fetch_batch_result(client, job_id, len(texts))
        except (APIError, TimeoutError) as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Embedding job failed after {retries} attempts: {last_error}")


def _embed_batches_concurrent(
    batches: list[list[str]],
    retries: int = 3,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Submits up to MAX_CONCURRENT_JOBS batches at once, waits for that
    whole window together (client.wait_for_all() waits for every job
    currently tracked, not just one), then fetches each job's own result --
    turns N serial submit-wait-fetch cycles into ceil(N/MAX_CONCURRENT_JOBS)
    windows. Order is preserved via explicit indices, not submission/
    completion order (jobs in a window don't necessarily finish in the
    order they were submitted). Failures are retried per-batch, not by
    redoing the whole window -- a single flaky job shouldn't cost redoing
    everything else that already succeeded.

    progress_callback(done, total) fires after each individual batch is
    successfully fetched (done counts real completions, not submissions --
    a batch that fails and gets retried doesn't count until it actually
    succeeds), so a caller watching a long corpus-wide embed sees steady
    progress roughly every few seconds rather than one jump per window."""
    client = _get_client()
    results: list[list[list[float]] | None] = [None] * len(batches)
    pending = list(range(len(batches)))
    last_error: Exception | None = None
    done_count = 0

    for attempt in range(retries):
        if not pending:
            break
        still_pending: list[int] = []
        for window_start in range(0, len(pending), MAX_CONCURRENT_JOBS):
            window = pending[window_start : window_start + MAX_CONCURRENT_JOBS]
            job_ids: dict[int, str] = {}
            for idx in window:
                try:
                    job_ids[idx] = _submit_batch(client, batches[idx])
                except APIError as e:
                    last_error = e
                    still_pending.append(idx)
            if job_ids:
                try:
                    client.wait_for_all(timeout=WAIT_FOR_ALL_TIMEOUT)
                except Exception:
                    pass  # individual fetches below surface the real per-job errors -- including
                    # a job still pending after WAIT_FOR_ALL_TIMEOUT, which _fetch_batch_result()
                    # below reports as EmbeddingJobError and the outer retry loop picks back up
            for idx in window:
                if idx not in job_ids:
                    continue
                try:
                    results[idx] = _fetch_batch_result(client, job_ids[idx], len(batches[idx]))
                    done_count += 1
                    if progress_callback:
                        progress_callback(done_count, len(batches))
                except (APIError, EmbeddingJobError) as e:
                    last_error = e
                    still_pending.append(idx)
        pending = still_pending
        if pending:
            time.sleep(2 * (attempt + 1))

    if pending:
        raise RuntimeError(
            f"Embedding failed for {len(pending)}/{len(batches)} batch(es) after "
            f"{retries} attempts: {last_error}"
        )
    flat: list[list[float]] = []
    for batch_result in results:
        flat.extend(batch_result)  # type: ignore[arg-type]
    return flat
