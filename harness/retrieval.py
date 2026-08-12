"""Retrieval over the corpus/ directory.

retrieve_context() now uses real semantic embeddings (MicroDC's
qwen3-embedding:8b, via harness/embeddings.py) instead of the original v1
keyword-overlap approach -- see PLAN.md's Phase 5. The old keyword-overlap
implementation is kept as _retrieve_context_keyword(), both for the
before/after comparison Phase 5 called for and as a fallback if embeddings
are ever unavailable (no MDC_API_KEY, network issues, etc).
"""

import functools
import os
import re

from . import config, embeddings

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "at", "to", "for", "and", "or", "but", "with", "as", "by", "it",
    "that", "this", "what", "when", "where", "who", "how", "why", "do",
    "does", "did", "can", "could", "should", "would", "will", "which",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


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


def _retrieve_context_keyword(query: str, k: int = 3) -> str:
    """Original v1 approach: relevance = number of shared non-stopword
    tokens. Simple, transparent, zero-dependency, but blind to synonyms/
    abbreviations (e.g. "BP" vs "blood pressure") -- see the retrieve_context
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
    """Corpus chunk embeddings, computed once per process and cached --
    corpus/ doesn't change mid-run, no reason to re-embed it for every
    question. One batch job for the whole corpus rather than one job per
    chunk (the async job queue has real per-submission latency)."""
    chunks = tuple(_load_chunks())
    if not chunks:
        return (), ()
    vectors = embeddings.embed_texts(list(chunks))
    return chunks, tuple(tuple(v) for v in vectors)


def retrieve_scored(query: str) -> list[tuple[float, str]]:
    """All corpus chunks with their cosine similarity to query, sorted best
    first. The scores are what V2-H's coverage check needs and
    retrieve_context() discarded -- both now share this instead of
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
    qwen3-embedding:8b) -- catches semantic/synonym matches the old
    keyword-overlap approach missed (see _retrieve_context_keyword)."""
    top = [chunk for _, chunk in retrieve_scored(query)[:k]]
    return "\n\n".join(top)
