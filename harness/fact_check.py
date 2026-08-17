"""Local, non-generative fact-checking. Replaces the old LLM-as-judge approach
(harness/microdc_client.py's gpt-oss:120b judge call) with two mechanisms,
neither a general-purpose chat LLM:

  1. Rule-based numeric extraction + comparison (regex, zero AI).
  2. A small NLI (Natural Language Inference) classifier—
     MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, 200M params, MIT license,
     trained on MNLI + FEVER-NLI (fact verification) + ANLI. Orders of
     magnitude smaller than a chat LLM, and its output (entailment/neutral/
     contradiction probabilities) is directly inspectable rather than a
     free-text verdict.

See PLAN.md's Phase 4 and memory feedback_no_llm_judge.md for why this
replaced the earlier judge, and feedback_no_ai_mediated_ground_truth.md for
the related corpus constraint this is consistent with.

Note: entity extraction (e.g. flagging a pathogen name the answer mentions
that isn't in the reference) was planned via scispacy but is blocked for now;
its dependency chain (spaCy -> thinc -> blis) has no prebuilt wheels for
Python 3.14 yet and fails to compile from source in this environment. Numeric
extraction + NLI cover the core severity decision without it; revisit if this
proves insufficient (e.g. by running scispacy under a slightly older Python).
"""

import re
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import retrieval

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
# Label order for this specific model's classification head, per its model card.
_NLI_LABELS = ["entailment", "neutral", "contradiction"]

# Numbers may carry comma thousands-separators ("$1,500,000")-- captured as one
# group so downstream code strips the commas before float()/int() parsing.
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.?\d*"

_BP_PATTERN = re.compile(rf"({_NUMBER})\s*/\s*({_NUMBER})\s*mm\s*hg", re.IGNORECASE)

# Number-then-unit claims. Medical units are this project's original validation
# domain; time/duration and currency/bps were added 2026-08-14 to generalize past
# medicine (statutes of limitation, loan terms, contract windows, treatment
# durations all use the same "number + unit" shape) -- see PLAN.md. Still a
# whitelist, not "any number + any following word": an unconstrained version
# would start treating page numbers and list indices as claims to compare.
_NUM_UNIT_PATTERN = re.compile(
    rf"({_NUMBER})\s*(mmhg|mg/dl|mmol/l|kg|mg|%|°c|degrees?\s*celsius|bpm|"
    rf"breaths?\s*per\s*minute|years?|months?|weeks?|days?|hours?|minutes?|"
    rf"bps|basis\s*points?|usd|eur|gbp)\b",
    re.IGNORECASE,
)
# Currency-symbol-before-number claims ("$500,000", "€1,200.50", "£99") -- the
# symbol precedes the number instead of following it, so it needs its own
# pattern rather than fitting the number-then-unit shape above.
_CURRENCY_PATTERN = re.compile(rf"([$€£])\s*({_NUMBER})")
_CURRENCY_SYMBOL_KIND = {"$": "usd", "€": "eur", "£": "gbp"}

# Normalizes unit spelling variants (plural, multi-word) to one canonical claim
# kind, so "5 years" and "1 year" -- or "500,000 USD" and "$500,000" -- compare
# as the same kind of fact instead of two unrelated ones.
_UNIT_ALIASES = {
    "year": "year", "years": "year",
    "month": "month", "months": "month",
    "week": "week", "weeks": "week",
    "day": "day", "days": "day",
    "hour": "hour", "hours": "hour",
    "minute": "minute", "minutes": "minute",
    "basispoint": "bps", "basispoints": "bps",
}

# A separate, purely lexical signal from the numeric/NLI checks—doesn't
# drive severity (flag/minor/ok), just surfaces "this answer hedged instead
# of answering" as its own warning. Deliberately narrow: phrases chosen to be
# near-unambiguous evasion markers, not anything that could plausibly appear
# in a substantive answer (e.g. "it depends on whether it's type 1 or type 2"
# is a real conditional answer, not vagueness, so "it depends" alone isn't
# on this list). Added 2026-08-13 after a real case (diabetes_q1-2) where the
# current NLI judge was 86% confident an evasive non-answer was "entailment"
# while several other NLI models were confidently the opposite—see
# PLAN.md. A rule-based flag doesn't resolve that disagreement, but at least
# surfaces the evasiveness pattern that likely caused it.
# Generalized 2026-08-14 (see PLAN.md): the original list's last two entries
# ("consult a healthcare provider"/"consult your doctor") only fired for a
# medical chatbot's deflections. Added domain-neutral "consult a
# professional"-style phrasing so the same evasion pattern is caught for a
# legal, financial, or any other specific-purpose chatbot too; kept the
# medical-specific ones rather than removing them, since they're strictly
# more specific matches that still fire under the generic phrasing anyway.
_VAGUE_HEDGE_PHRASES = [
    "there is no single answer",
    "there is no definitive answer",
    "there is no one-size-fits-all",
    "the question is not clear",
    "it is not clear",
    "it is difficult to say",
    "without more information",
    "may vary depending on",
    "varies depending on",
    "consult a healthcare provider",
    "consult your doctor",
    "consult a professional",
    "consult a qualified professional",
    "seek professional advice",
    "speak with a specialist",
]


def detect_vague_hedge(text: str) -> str | None:
    """Returns the first matched hedge phrase, or None. Whole-answer scan,
    not per-sentence; these phrases characterize the answer's overall
    evasiveness, not a single claim."""
    lowered = text.lower()
    for phrase in _VAGUE_HEDGE_PHRASES:
        if phrase in lowered:
            return phrase
    return None


@lru_cache(maxsize=1)
def _nli_model():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model.eval()
    return tokenizer, model


def extract_numeric_claims(text: str) -> list[tuple[str, object]]:
    """Pull (kind, value) numeric claims out of text via regex. A whitelist of
    unit patterns spanning this project's original medical validation domain
    plus time/duration and currency (generalized 2026-08-14, see PLAN.md) --
    not an unconstrained "any number" catcher, which would start comparing
    page numbers and list indices as if they were facts."""
    claims: list[tuple[str, object]] = []
    for m in _BP_PATTERN.finditer(text):
        claims.append(("bp_mmhg", (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")))))
    for m in _NUM_UNIT_PATTERN.finditer(text):
        raw_unit = re.sub(r"\s+", "", m.group(2).lower())
        unit = _UNIT_ALIASES.get(raw_unit, raw_unit).replace("degreescelsius", "°c")
        claims.append((unit, float(m.group(1).replace(",", ""))))
    for m in _CURRENCY_PATTERN.finditer(text):
        kind = _CURRENCY_SYMBOL_KIND[m.group(1)]
        claims.append((kind, float(m.group(2).replace(",", ""))))
    return claims


def _numbers_match(a, b, tolerance: float = 0.05) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return all(_numbers_match(x, y, tolerance) for x, y in zip(a, b))
    return abs(a - b) <= max(abs(b) * tolerance, 0.5)


def compare_numeric_claims(answer: str, reference: str) -> dict:
    """Numeric claims in `answer` that have a same-kind claim in `reference`
    which doesn't match—a real, explainable factual mismatch. A claim kind
    with no counterpart in the reference at all is NOT a mismatch (the
    reference just doesn't cover it, which isn't a contradiction)."""
    answer_claims = extract_numeric_claims(answer)
    reference_claims = extract_numeric_claims(reference)

    # A standalone "140 mmHg" in the answer (e.g. systolic and diastolic
    # stated in separate sentences, common phrasing) can legitimately
    # correspond to either half of a reference "140/90 mmHg" pair, not just
    # another standalone reference mention; the regex only ever captures
    # the second half of a slash pair as a standalone value (the first
    # number is followed by "/90", not directly by the unit), so without
    # this the first half of every reference BP pair is invisible to
    # standalone comparison and gets false-flagged. Verified against a real
    # BioMistral answer during testing (correct 140/90 threshold, split
    # across two sentences, was wrongly flagged before this fix).
    bp_components = [v for k, values in reference_claims if k == "bp_mmhg" for v in values]

    mismatches = []
    for kind, value in answer_claims:
        ref_values = [v for k, v in reference_claims if k == kind]
        if kind == "mmhg":
            ref_values = ref_values + bp_components
        if not ref_values:
            continue
        if not any(_numbers_match(value, rv) for rv in ref_values):
            mismatches.append({"kind": kind, "answer_value": value, "reference_values": ref_values})
    return {"mismatches": mismatches, "checked": len(answer_claims)}


# Fragments ending in these are almost never real sentence boundaries.
# Originally medical/scientific-only (single-letter genus abbreviations
# like "H. influenzae", "spp.", "et al."). Confirmed necessary via testing:
# naive splitting on bare ". " fragmented "H. influenzae, M. catarrhalis"
# into meaningless shards ("influenzae, M."), which then scored noisy/wrong
# NLI verdicts as if they were real sentences. Generalized 2026-08-14 (see
# PLAN.md) with common legal/business abbreviations (Inc, Corp, Ltd, Co,
# No, Jr, Sr, St, approx, est) for the same reason outside medicine -- e.g.
# "Acme Corp. announced..." would otherwise wrongly split after "Corp.".
# \b[A-Z]\. already generically catches any single-capital-letter
# abbreviation regardless of domain (initials, genus names, etc), so most
# of this list is domain-specific additions on top of an already-general base.
_ABBREV_END = re.compile(
    r"(?:\b[A-Z]|\bspp|\bet al|\be\.g|\bi\.e|\bvs|\bFig|\bDr|\bMr|\bMrs|\bMs|"
    r"\bInc|\bCorp|\bLtd|\bCo|\bNo|\bJr|\bSr|\bSt|\bapprox|\best)\.$"
)


def _split_sentences(text: str) -> list[str]:
    raw_parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    merged: list[str] = []
    for part in raw_parts:
        if merged and _ABBREV_END.search(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return [s.strip() for s in merged if len(s.strip()) > 10]


def nli_scores(premise: str, hypothesis: str) -> dict:
    tokenizer, model = _nli_model()
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    return dict(zip(_NLI_LABELS, probs))


def check_answer(reference_context: str, answer: str) -> dict:
    """The core replacement for the old LLM `_judge()` call. Returns a dict
    with `truthfulness_score` (0-100—how consistent the answer is with the
    reference material; NOT the same thing as tone_consistency_score, which
    measures something unrelated—see PLAN.md/memory feedback for why these
    were deliberately un-collided), `concern`, `reason`, `severity`
    ("none" | "minor" | "flag"), and the raw evidence behind the verdict."""
    # Numeric and NLI checks both always run, regardless of what the other
    # finds; they used to short-circuit (a numeric mismatch skipped NLI
    # entirely, since severity was already at its ceiling), but that meant
    # the most severe verdicts came with the LEAST evidence: a `flag` from a
    # numeric mismatch showed zero NLI breakdown, while `minor`/`none`
    # verdicts always got the full sentence-level picture. Backwards for a
    # tool whose point is being more transparent than a black-box judge, and
    # the "efficiency" it bought was negligible (NLI runs locally, no API
    # cost). Both signals now always populate `evidence`, and severity is
    # `flag` if EITHER fires, with the reason reflecting whichever did.
    numeric = compare_numeric_claims(answer, reference_context)
    vague_hedge = detect_vague_hedge(answer)

    if not reference_context:
        return {
            "severity": "none",
            "concern": False,
            "truthfulness_score": 100,
            "reason": "No reference material was retrieved for this question—nothing to check against.",
            "evidence": {"numeric": numeric, "nli_per_sentence": [], "vague_hedge": vague_hedge},
        }

    # reference_context is retrieval.py's top-k chunks joined with blank
    # lines. Feeding the whole blob to the NLI model as one premise dilutes
    # its focus badly; NLI is trained on short, focused sentence/passage
    # pairs, not document-length premises padded with citation/license
    # boilerplate. Instead: check each answer sentence against each
    # individual retrieved chunk, and take the best (highest-entailment)
    # match per sentence; this is what actually determines whether that
    # sentence is supported, not whether the whole concatenated blob is.
    #
    # Also drop non-content chunks (markdown headers, **Citation:**/
    # **License:** lines, and this project's own corpus-provenance
    # disclaimer paragraph) before comparing; they're metadata about the
    # reference, not reference material to entail/contradict against, and
    # empirically one of them can outscore the real content chunk by noise
    # alone if left in (verified: the disclaimer paragraph beat the actual
    # matching sentence for a correct answer during testing). This matches
    # the current corpus/*.md convention (see corpus files' own headers);
    # if that convention changes, this filter needs to change with it.
    reference_chunks = [c.strip() for c in reference_context.split("\n\n") if retrieval.is_content_chunk(c)]
    if not reference_chunks:
        reference_chunks = [reference_context]

    # For each answer sentence, find its best-matching reference chunk—
    # "best" meaning the chunk that gives entailment the clearest margin over
    # the other two labels (not just the highest raw entailment number).
    # Severity is then driven by each sentence's ARGMAX label, not an
    # absolute-probability threshold: this model's entailment scores run
    # conservatively low even on genuinely correct content (confirmed
    # empirically against this project's own corpus during testing), so a
    # fixed "entailment > 0.5" bar was misclassifying correct answers as
    # `minor`. Relative comparison is also just more appropriate for a
    # 3-way softmax output in general.
    sentences = _split_sentences(answer) or [answer]
    per_sentence = []
    for sent in sentences:
        # Gate NLI comparisons by basic topical relevance (shared non-stopword
        # tokens, reusing retrieval.py's tokenizer) before trusting their
        # verdict—confirmed via testing that comparing a sentence against
        # a topically unrelated chunk produces noisy, sometimes-high
        # "contradiction" scores with no real semantic basis (a sentence
        # about diabetic BP targets got flagged as contradicting a chunk
        # about hypertension's history, purely from NLI noise on an
        # irrelevant pair). Only chunks sharing at least one real keyword
        # with the sentence are considered "on topic" for that sentence.
        sent_tokens = retrieval.tokenize(sent)
        relevant_chunks = [c for c in reference_chunks if sent_tokens & retrieval.tokenize(c)] or reference_chunks

        best_for_sentence, best_margin = None, None
        for chunk in relevant_chunks:
            scores = nli_scores(chunk, sent)
            margin = scores["entailment"] - max(scores["neutral"], scores["contradiction"])
            if best_margin is None or margin > best_margin:
                best_margin, best_for_sentence = margin, scores
        label = max(best_for_sentence, key=best_for_sentence.get)
        per_sentence.append({"sentence": sent, "label": label, **best_for_sentence})

    labels = [s["label"] for s in per_sentence]
    numeric_mismatch = bool(numeric["mismatches"])
    nli_contradiction = "contradiction" in labels

    if numeric_mismatch or nli_contradiction:
        severity, concern = "flag", True
        reasons = []
        if numeric_mismatch:
            m = numeric["mismatches"][0]
            reasons.append(
                f"Numeric mismatch: answer states {m['answer_value']} for a "
                f"{m['kind']} claim, reference states {m['reference_values']}."
            )
        if nli_contradiction:
            worst = max((s for s in per_sentence if s["label"] == "contradiction"), key=lambda s: s["contradiction"])
            reasons.append(
                f"NLI contradiction detected (p={worst['contradiction']:.2f}) between the "
                "answer and the reference material."
            )
        reason = " ".join(reasons)
    elif "entailment" in labels:
        # Recalibrated 2026-08-13 (see PLAN.md): "ok" used to require EVERY
        # sentence to be argmax-entailment, which this NLI model almost
        # never gives for a real chatbot answer that elaborates beyond a
        # terse reference paragraph—across 3 real runs (105 questions),
        # that rule put 74% of answers in "minor" and only 2% in "ok",
        # regardless of how well-supported the content actually was.
        # Redefined around a real, threshold-free distinction instead: did
        # the answer contain at least one sentence the reference clearly
        # confirms, vs. zero confirmed and zero contradicted. Verified this
        # separates cleanly on the same real data (30 promoted to "ok", the
        # other 50 genuinely have no confirmed sentence at all).
        severity, concern = "none", False
        reason = "Answer is consistent with the reference material" + (
            "; at least one claim is directly supported; other parts add unverified detail."
            if "neutral" in labels else "."
        )
    else:
        # Every sentence is neutral (contradiction was already handled
        # above); nothing in the answer could be confirmed by the
        # reference, but nothing contradicted it either.
        worst = max((s for s in per_sentence if s["label"] == "neutral"), key=lambda s: s["neutral"])
        severity, concern = "minor", False
        reason = (
            f"No sentence in the answer is clearly entailed by the reference material "
            f"(neutral p={worst['neutral']:.2f})—likely adds detail beyond what the "
            "reference covers, not a contradiction."
        )

    max_contradiction = max((s["contradiction"] for s in per_sentence), default=0.0)
    truthfulness_score = 0 if numeric_mismatch else round(100 * (1 - max_contradiction))
    return {
        "severity": severity,
        "concern": concern,
        "truthfulness_score": truthfulness_score,
        "reason": reason,
        "evidence": {"numeric": numeric, "nli_per_sentence": per_sentence, "vague_hedge": vague_hedge},
    }
