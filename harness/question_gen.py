"""V2-I: auto-generate an eval question set from the attached corpus/.

Method (see PLAN.md for the full design discussion/approval): for each
corpus file (one "topic"), generate candidate questions from its real
content chunks via a MicroDC model (harness/microdc_client.py) --
deliberately a different model family than TARGET_MODEL, the same
don't-let-a-model-write-its-own-exam principle already applied to judging.
For each accepted candidate, generate a few phrasing/tone variants sharing
a group_id, to drive the tone-consistency check.

Two guardrails, both reusing existing infrastructure rather than trusting
generated text on its own:
  1. Every candidate question (and every variant) must pass the SAME
     coverage check (harness/coverage_check.py, V2-H) a human-written
     question would -- cosine similarity >= COVERAGE_THRESHOLD against the
     corpus. Generated text earns its way in; it isn't assumed grounded.
  2. Every variant must also score >= PARAPHRASE_THRESHOLD cosine
     similarity against its own group's canonical question, catching
     clear drift (a variant that became a different question, not just a
     different phrasing) before it corrupts the tone-consistency check,
     which assumes all variants in a group ask the same thing. This isn't
     as clean a filter as guardrail 1 -- see PARAPHRASE_THRESHOLD's own
     comment for why question-vs-question similarity can't fully separate
     "reworded" from "same-topic-but-different." The variant-generation
     prompt's explicit intent-preservation instruction is the primary
     defense; this is a secondary catch, not a guarantee.

Per the user's explicit choice (2026-08-13): questions_per_topic/
variants_per_question are a CEILING, not a guarantee -- if a topic's real
corpus content can't honestly support the requested count, this generates
fewer and reports actual-vs-requested rather than forcing weak questions
through or silently coming up short.
"""

import os
import re

from . import config, coverage_check, microdc_client, retrieval
from .config import GENERATION_MODEL

QUESTIONS_PER_TOPIC_DEFAULT = 4
VARIANTS_PER_QUESTION_DEFAULT = 4

# Calibrated empirically 2026-08-13 (see PLAN.md), and NOT as clean a
# separation as V2-H's coverage check: real MicroDC-generated variants of
# one real candidate question scored 0.72-0.96 cosine similarity to their
# canonical question, but deliberately-different questions on the SAME
# topic (e.g. "what causes CAP" vs. "how severe is CAP") scored 0.53-0.78 --
# overlapping the real range. Unlike question-vs-corpus-chunk similarity
# (V2-H), question-vs-question similarity on short text seems to track
# topical closeness more than precise intent, so no threshold here fully
# separates "reworded" from "same-topic-but-different". 0.68 clears the
# clearest drift case (0.53) with margin and keeps every real variant
# observed (min 0.72) with a little headroom -- but a variant that drifts
# to a *closely related* same-topic question could still slip through.
# The variant-generation prompt's explicit "same intent, same expected
# answer" instruction is the primary defense; this threshold is a
# secondary catch for clear failures, not a fully reliable filter -- said
# so directly rather than overclaiming precision this metric doesn't have.
PARAPHRASE_THRESHOLD = 0.68

# Ordered so the first (variants_per_question - 1) get used by default --
# real-world-usage variety first (typo'd/rushed, roundabout), cleaner
# registers after. Updated 2026-08-13 per user feedback: the original set
# (casual/formal/brief/worried) was too uniformly "clean" -- real chatbot
# users type fast with minor errors, and often hedge/backstory their way to
# the actual question instead of asking directly.
_TONE_HINTS = [
    "very casual, typed quickly on a phone -- include a couple of small, "
    "easily-readable typos or grammar slips (a dropped word, missing "
    "capitalization, a common misspelling), but the question must still be "
    "clearly understandable, not garbled",
    "roundabout and indirect -- give a little unnecessary backstory or "
    "context before getting to the actual question, the way real people "
    "often over-explain to a chatbot instead of just asking directly",
    "casual and conversational, plain everyday language, like texting a friend",
    "formal and precise, using correct clinical terminology",
    "brief and direct, as few words as possible while staying a complete question",
    "a worried patient hedging and unsure, asking in their own words, not medical jargon",
]

# Generic descriptor words stripped when deriving a short topic name from a
# corpus file's title -- not meaningful on their own (every file in this
# genre has an "overview" or "guidelines"), so they'd make every topic name
# look the same instead of naming what's actually distinctive about it.
_GENERIC_TITLE_WORDS = {
    "overview", "management", "strategies", "guidelines", "guideline",
    "diagnosis", "treatment", "nutritional", "causes", "care", "approach",
    "review", "and", "for", "of", "the", "a", "an", "with", "in", "on",
}


def _topic_name(filename: str, title: str | None = None) -> str:
    """A short (<=2 word, 1 word if that's all that's left after filtering
    out generic descriptors) topic name for question/group IDs -- derived
    from the corpus file's own `# Title` line, not the filename (which
    tends to be more verbose, e.g. "..._overview.md"). Falls back to the
    filename if no title is available, for corpora that don't follow this
    project's own `# Title` convention.

    Not a fully general solution -- the descriptor stoplist is tuned to
    this kind of medical-writeup title, not guaranteed to pick the right
    word for an arbitrary future corpus's title style. Accepted tradeoff,
    per the user's explicit call: this is a cosmetic ID/label concern, not
    a correctness one."""
    source = title or re.sub(r"[_\-]+", " ", re.sub(r"\.(md|txt)$", "", filename))
    words = [w for w in source.split() if w.lower() not in _GENERIC_TITLE_WORDS]
    if not words:
        words = source.split()
    if len(words) > 2:
        words = words[-1:]
    return "_".join(w.lower() for w in words)


def _file_title(text: str) -> str | None:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _generate_candidate(chunk: str) -> str:
    """One MicroDC call: a natural question a real person might ask that
    this specific real passage answers. Explicitly instructed not to
    introduce facts beyond the passage -- generation is grounded, not
    free invention; the coverage check afterward is the actual guarantee,
    this is just trying to make that check's job easy."""
    reply = microdc_client.chat(
        GENERATION_MODEL,
        [
            {
                "role": "system",
                "content": (
                    "You write eval questions for testing chatbots. Given a passage of "
                    "real reference text, write ONE natural question that a real person "
                    "might genuinely ask, which this passage directly and fully answers. "
                    "Do not introduce any fact, number, or claim not present in the "
                    "passage. Reply with ONLY the question text -- no preamble, no "
                    "quotes, no numbering."
                ),
            },
            {"role": "user", "content": chunk},
        ],
        temperature=0.7,
    )
    return reply.strip().strip('"')


def _generate_variant(canonical_question: str, tone_hint: str) -> str:
    """One MicroDC call: reword canonical_question in a different
    tone/phrasing, preserving the exact same underlying question and
    intent -- a paraphrase, not a new question."""
    reply = microdc_client.chat(
        GENERATION_MODEL,
        [
            {
                "role": "system",
                "content": (
                    "You rewrite questions for an eval question set. Given a question, "
                    "rewrite it so it asks the EXACT same underlying thing -- same "
                    "intent, same expected answer -- but in a different tone/phrasing: "
                    f"{tone_hint}. Do not change what's being asked, only how it's "
                    "asked. Reply with ONLY the rewritten question -- no preamble, no "
                    "quotes, no numbering."
                ),
            },
            {"role": "user", "content": canonical_question},
        ],
        temperature=0.8,
    )
    return reply.strip().strip('"')


def _passes_coverage(question: str) -> bool:
    return coverage_check.check_coverage([{"id": "_probe", "question": question}])[0]["covered"]


def _passes_paraphrase(candidate: str, canonical: str) -> bool:
    from . import embeddings

    vecs = embeddings.embed_texts([canonical, candidate])
    return retrieval._cosine(vecs[0], vecs[1]) >= PARAPHRASE_THRESHOLD


def generate_questions(
    questions_per_topic: int = QUESTIONS_PER_TOPIC_DEFAULT,
    variants_per_question: int = VARIANTS_PER_QUESTION_DEFAULT,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Returns (questions, stats). questions is the same schema as
    questions/*.json (id/group_id/question). stats is one dict per topic:
    {topic, chunks_available, questions_requested, questions_accepted,
    variants_requested, variants_accepted} -- for transparent
    actual-vs-requested reporting, not silently generating fewer than asked."""
    chunks_by_file = retrieval.load_chunks_by_file()
    questions: list[dict] = []
    stats: list[dict] = []

    for fname, chunks in chunks_by_file.items():
        with open(os.path.join(config.CORPUS_DIR, fname)) as f:
            title = _file_title(f.read())
        topic = _topic_name(fname, title)
        # Richer (longer) passages make better question seeds than
        # throwaway one-line sentences -- prioritize them when a topic has
        # more chunks than the requested ceiling.
        candidate_chunks = sorted(chunks, key=len, reverse=True)[:questions_per_topic]
        topic_stats = {
            "topic": topic,
            "chunks_available": len(chunks),
            "questions_requested": questions_per_topic,
            "questions_accepted": 0,
            "variants_requested": 0,
            "variants_accepted": 0,
        }

        qidx = 0
        for chunk in candidate_chunks:
            if verbose:
                print(f"[{topic}] generating candidate question...")
            candidate = _generate_candidate(chunk)
            if not candidate or not _passes_coverage(candidate):
                if verbose:
                    print(f"[{topic}] candidate discarded (failed coverage check): {candidate!r}")
                continue

            qidx += 1
            group_id = f"{topic}_q{qidx}"
            topic_stats["questions_accepted"] += 1

            variants = [candidate]
            topic_stats["variants_requested"] += variants_per_question
            topic_stats["variants_accepted"] += 1  # the canonical counts as variant 1
            for tone_hint in _TONE_HINTS[: variants_per_question - 1]:
                if verbose:
                    print(f"[{topic}] generating variant ({tone_hint[:30]}...)...")
                variant = _generate_variant(candidate, tone_hint)
                if not variant:
                    continue
                if not _passes_coverage(variant):
                    if verbose:
                        print(f"[{topic}] variant discarded (failed coverage check): {variant!r}")
                    continue
                if not _passes_paraphrase(variant, candidate):
                    if verbose:
                        print(f"[{topic}] variant discarded (drifted from canonical question): {variant!r}")
                    continue
                variants.append(variant)
                topic_stats["variants_accepted"] += 1

            for vidx, v in enumerate(variants, start=1):
                questions.append({"id": f"{group_id}-{vidx}", "group_id": group_id, "question": v})

        stats.append(topic_stats)

    return questions, stats
