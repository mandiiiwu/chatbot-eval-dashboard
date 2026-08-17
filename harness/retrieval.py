"""Retrieval over the corpus/ directory.

retrieve_context() now uses real semantic embeddings (MicroDC's
qwen3-embedding:8b, via harness/embeddings.py) instead of the original v1
keyword-overlap approach—see PLAN.md's Phase 5. The old keyword-overlap
implementation is kept as _retrieve_context_keyword(), both for the
before/after comparison Phase 5 called for and as a fallback if embeddings
are ever unavailable (no MDC_API_KEY, network issues, etc).
"""

import functools
import hashlib
import os
import re
from typing import Callable

from . import config, embeddings


def corpus_fingerprint() -> str:
    """Short hash of the current corpus/ directory's content—changes
    automatically whenever a corpus file is added/removed/edited. Stored on
    every run and used (alongside target_model, schema_version) to filter
    the trend chart, so swapping in an entirely different corpus (e.g. this
    project's medical one for a future legal/HR one) can't silently mix
    incompatible results on the same trend line—same principle already
    applied to target_model (V2-C) and schema_version, just for corpus
    content. No manual "project" setup needed; this is fully automatic."""
    if not os.path.isdir(config.CORPUS_DIR):
        return "empty"
    parts = []
    for fname in sorted(os.listdir(config.CORPUS_DIR)):
        if not fname.endswith((".md", ".txt")):
            continue
        with open(os.path.join(config.CORPUS_DIR, fname)) as f:
            parts.append(fname + "\x00" + f.read())
    if not parts:
        return "empty"
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "at", "to", "for", "and", "or", "but", "with", "as", "by", "it",
    "that", "this", "what", "when", "where", "who", "how", "why", "do",
    "does", "did", "can", "could", "should", "would", "will", "which",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


# A chunk is "content" (real reference material) vs. metadata (headers,
# citation/license lines, this project's own corpus-provenance disclaimer)—
# shared by fact_check.py (don't NLI-compare an answer against a
# citation line) and question_gen.py (don't generate a question from a
# license blurb). One predicate, not two copies of the same filter.
_DISCLAIMER_MARKER = "not written or paraphrased by an ai"


def is_content_chunk(chunk: str) -> bool:
    c = chunk.strip()
    return bool(c) and not c.startswith("#") and not c.startswith("**") and _DISCLAIMER_MARKER not in c.lower()


def load_chunks_by_file() -> dict[str, list[str]]:
    """Content-only corpus chunks (see is_content_chunk), grouped by source
    filename—what question_gen.py seeds candidate questions from, one
    topic per file. Separate from _load_chunks()/_chunk_embeddings() (which
    stay unfiltered, flat, and untouched here) so this doesn't change what
    retrieval/coverage-check see or risk invalidating V2-H's calibrated
    threshold."""
    result: dict[str, list[str]] = {}
    if not os.path.isdir(config.CORPUS_DIR):
        return result
    for fname in sorted(os.listdir(config.CORPUS_DIR)):
        if not fname.endswith((".md", ".txt")):
            continue
        with open(os.path.join(config.CORPUS_DIR, fname)) as f:
            text = f.read()
        chunks = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 30]
        result[fname] = [c for c in chunks if is_content_chunk(c)]
    return result


def _load_chunks() -> list[str]:
    chunks = []
    if not os.path.isdir(config.CORPUS_DIR):
        return chunks
    for fname in sorted(os.listdir(config.CORPUS_DIR)):
        if not fname.endswith((".md", ".txt")):
            continue
        with open(os.path.join(config.CORPUS_DIR, fname)) as f:
            text = f.read()
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) > 30:
                chunks.append(para)
    return chunks


@functools.lru_cache(maxsize=1)
def _chunk_tokens() -> tuple[tuple[str, ...], tuple[frozenset, ...]]:
    """Corpus chunks + their tokenized token sets, cached once per process --
    same rationale as _chunk_embeddings(): corpus/ doesn't change mid-run,
    and retokenizing every chunk (this project's own CUAD stress-test
    corpus produces 100k+ of them, see PLAN.md) on every single call would
    defeat the point of this backing the *cheap* pre-check in
    coverage_check.py. Cleared alongside _chunk_embeddings via
    clear_caches() whenever corpus/ actually changes."""
    chunks = tuple(_load_chunks())
    return chunks, tuple(frozenset(tokenize(c)) for c in chunks)


def retrieve_scored_keyword(query: str) -> list[tuple[float, str]]:
    """Free keyword-overlap version of retrieve_scored() -- zero embedding
    calls, zero MicroDC cost. Score = shared non-stopword tokens / query
    tokens (so it's comparable across questions of different lengths, the
    same way retrieve_scored()'s cosine similarity already is), sorted best
    first. Coarser than the embedding version (blind to synonyms/
    paraphrases, and a query can coincidentally share generic vocabulary
    with an unrelated chunk -- see coverage_check.py's
    KEYWORD_PRECHECK_THRESHOLD for real examples of both). Good enough to
    catch an *obviously* wrong corpus/question pairing for free before
    spending anything on the real check, not meant to replace it."""
    chunks, chunk_tokens = _chunk_tokens()
    if not chunks:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return [(0.0, c) for c in chunks]
    scored = [(len(query_tokens & ct) / len(query_tokens), c) for c, ct in zip(chunks, chunk_tokens)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def clear_caches() -> None:
    """Unconditionally clears both corpus caches (embeddings and keyword
    tokens) -- call this right after a KNOWN corpus mutation (the
    dashboard's own upload/delete endpoints), where the caller already
    knows for a fact something changed and there's no reason to check.
    For "might corpus/ have changed since last time, or not" -- the common
    case at the start of every eval run -- use ensure_fresh_caches()
    instead, which skips the (real, not free -- see its own docstring)
    re-embedding cost when nothing actually changed. One function so a
    future new corpus-derived cache can't be added without also being
    wired into every existing clear-cache call site."""
    _chunk_embeddings.cache_clear()
    _chunk_tokens.cache_clear()


_last_embedded_fingerprint: str | None = None


def ensure_fresh_caches() -> None:
    """Clears the corpus caches only if corpus/ has actually changed since
    they were last computed (compared via corpus_fingerprint(), which reads
    every corpus file fresh from disk regardless of caching -- this
    function's only job is deciding whether the *embedding* work downstream
    of that is worth redoing).

    Added 2026-08-16 (see PLAN.md's corpus-scaling-plan lever #1): the
    previous behavior -- run_and_save() calling clear_caches()
    unconditionally on every single run -- was a real staleness-bug fix
    (a corpus file edited directly on disk while the persistent server was
    already running needed to be picked up), but it was too blunt: it threw
    away a fully-valid, still-current embedding cache and forced a full
    corpus-wide re-embed on every run regardless of whether anything
    changed, including the common case of a scheduled nightly run against
    an unchanged corpus. For the CUAD stress-test corpus (116,616 chunks),
    that's the difference between "skip embedding entirely, near-instant"
    and "re-embed the whole thing again," a genuinely large chunk of the
    8-hour runtime the previous full run took. Detecting "unchanged" still
    correctly covers the original staleness scenario, since
    corpus_fingerprint() reads real file content fresh every call -- a
    direct on-disk edit changes the fingerprint just as reliably as it did
    before, this only skips the redundant work when nothing actually
    changed."""
    global _last_embedded_fingerprint
    current = corpus_fingerprint()
    if current != _last_embedded_fingerprint:
        _chunk_embeddings.cache_clear()
        _chunk_tokens.cache_clear()
        _last_embedded_fingerprint = current


# Set by evaluator.run_and_save() for the duration of a run so a caller
# watching for live progress (the dashboard's RUN_EVAL button) can see the
# corpus-wide embed actually happening, not just sit blank -- see
# _chunk_embeddings()'s use of this below. Module-level, not a parameter on
# _chunk_embeddings() itself: that function is @lru_cache(maxsize=1), and a
# callback argument would either bust the cache on every call (defeating
# the whole point of caching) or need to be excluded from the cache key,
# which functools.lru_cache doesn't support -- reading module state at call
# time avoids both problems and doesn't affect what gets cached. Only ever
# threaded into _chunk_embeddings()'s own embed_texts() call (the one
# potentially-slow, many-batch corpus-wide embed) -- the many other, much
# smaller embed_texts() calls elsewhere (a single query in retrieve_scored()
# below, tone-consistency answers in evaluator.py) never read this and so
# never fire spurious progress updates for calls that were never slow to
# begin with.
_embedding_progress_callback: Callable[[int, int], None] | None = None


def set_embedding_progress_callback(callback: Callable[[int, int], None] | None) -> None:
    global _embedding_progress_callback
    _embedding_progress_callback = callback


def _retrieve_context_keyword(query: str, k: int = 3) -> str:
    """Original v1 approach: relevance = number of shared non-stopword
    tokens. Simple, transparent, zero-dependency, but blind to synonyms/
    abbreviations (e.g. "BP" vs "blood pressure")—see the retrieve_context
    docstring and PLAN.md's Phase 5 for why this got replaced as the default."""
    chunks = _load_chunks()
    if not chunks:
        return ""
    query_tokens = tokenize(query)
    scored = []
    for chunk in chunks:
        overlap = len(query_tokens & tokenize(chunk))
        if overlap > 0:
            scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [chunk for _, chunk in scored[:k]]
    return "\n\n".join(top)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


@functools.lru_cache(maxsize=1)
def _chunk_embeddings() -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Corpus chunk embeddings, computed once per process and cached;
    corpus/ doesn't change mid-run, no reason to re-embed it for every
    question. One batch job for the whole corpus rather than one job per
    chunk (the async job queue has real per-submission latency)."""
    chunks = tuple(_load_chunks())
    if not chunks:
        return (), ()
    vectors = embeddings.embed_texts(list(chunks), progress_callback=_embedding_progress_callback)
    return chunks, tuple(tuple(v) for v in vectors)


def retrieve_scored(query: str) -> list[tuple[float, str]]:
    """All corpus chunks with their cosine similarity to query, sorted best
    first. The scores are what V2-H's coverage check needs and
    retrieve_context() discarded; both now share this instead of
    duplicating the embed-and-rank logic."""
    chunks, chunk_vectors = _chunk_embeddings()
    if not chunks:
        return []
    query_vector = embeddings.embed_texts([query])[0]
    scored = [(_cosine(query_vector, list(cv)), chunk) for chunk, cv in zip(chunks, chunk_vectors)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def retrieve_context(query: str, k: int = 3) -> str:
    """Return the top-k corpus chunks most relevant to query, joined
    together. Relevance = cosine similarity in embedding space (MicroDC's
    qwen3-embedding:8b)—catches semantic/synonym matches the old
    keyword-overlap approach missed (see _retrieve_context_keyword)."""
    top = [chunk for _, chunk in retrieve_scored(query)[:k]]
    return "\n\n".join(top)
