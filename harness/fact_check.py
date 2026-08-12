"""Local, non-generative fact-checking. Replaces the old LLM-as-judge approach
(harness/microdc_client.py's gpt-oss:120b judge call) with two mechanisms,
neither a general-purpose chat LLM:

  1. Rule-based numeric extraction + comparison (regex, zero AI).
  2. A small NLI (Natural Language Inference) classifier --
     MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, 200M params, MIT license,
     trained on MNLI + FEVER-NLI (fact verification) + ANLI. Orders of
     magnitude smaller than a chat LLM, and its output (entailment/neutral/
     contradiction probabilities) is directly inspectable rather than a
     free-text verdict.

See PLAN.md's Phase 4 and memory feedback_no_llm_judge.md for why this
replaced the earlier judge, and feedback_no_ai_mediated_ground_truth.md for
the related corpus constraint this is consistent with.

Note: entity extraction (e.g. flagging a pathogen name the answer mentions
that isn't in the reference) was planned via scispacy but is blocked for now
-- its dependency chain (spaCy -> thinc -> blis) has no prebuilt wheels for
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

_BP_PATTERN = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s*hg", re.IGNORECASE)
_NUM_UNIT_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(mmhg|mg/dl|mmol/l|kg|mg|%|°c|degrees?\s*celsius|bpm|breaths?\s*per\s*minute)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _nli_model():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model.eval()
    return tokenizer, model


def extract_numeric_claims(text: str) -> list[tuple[str, object]]:
    """Pull (kind, value) numeric claims out of text via regex. Deliberately a
    small, hand-coded pattern set matching this project's current corpus --
    extend as the corpus grows rather than trying to cover all of medicine
    up front."""
    claims: list[tuple[str, object]] = []
    for m in _BP_PATTERN.finditer(text):
        claims.append(("bp_mmhg", (int(m.group(1)), int(m.group(2)))))
    for m in _NUM_UNIT_PATTERN.finditer(text):
        unit = re.sub(r"\s+", "", m.group(2).lower()).replace("degreescelsius", "°c")
        claims.append((unit, float(m.group(1))))
    return claims


def _numbers_match(a, b, tolerance: float = 0.05) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return all(_numbers_match(x, y, tolerance) for x, y in zip(a, b))
    return abs(a - b) <= max(abs(b) * tolerance, 0.5)


def compare_numeric_claims(answer: str, reference: str) -> dict:
    """Numeric claims in `answer` that have a same-kind claim in `reference`
    which doesn't match -- a real, explainable factual mismatch. A claim kind
    with no counterpart in the reference at all is NOT a mismatch (the
    reference just doesn't cover it, which isn't a contradiction)."""
    answer_claims = extract_numeric_claims(answer)
    reference_claims = extract_numeric_claims(reference)

    # A standalone "140 mmHg" in the answer (e.g. systolic and diastolic
    # stated in separate sentences, common phrasing) can legitimately
    # correspond to either half of a reference "140/90 mmHg" pair, not just
    # another standalone reference mention -- the regex only ever captures
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


# Fragments ending in these are almost never real sentence boundaries in
# medical/scientific text -- single-letter genus abbreviations ("H.
# influenzae", "S. pneumoniae"), "spp.", "et al.", etc. Confirmed necessary
# via testing: naive splitting on bare ". " fragmented "H. influenzae, M.
# catarrhalis" into meaningless shards ("influenzae, M."), which then scored
# noisy/wrong NLI verdicts as if they were real sentences.
_ABBREV_END = re.compile(r"(?:\b[A-Z]|\bspp|\bet al|\be\.g|\bi\.e|\bvs|\bFig|\bDr|\bMr|\bMrs|\bMs)\.$")


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
    with `truthfulness_score` (0-100 -- how consistent the answer is with the
    reference material; NOT the same thing as tone_consistency_score, which
    measures something unrelated -- see PLAN.md/memory feedback for why these
    were deliberately un-collided), `concern`, `reason`, `severity`
    ("none" | "minor" | "flag"), and the raw evidence behind the verdict."""
    # Numeric and NLI checks both always run, regardless of what the other
    # finds -- they used to short-circuit (a numeric mismatch skipped NLI
    # entirely, since severity was already at its ceiling), but that meant
    # the most severe verdicts came with the LEAST evidence: a `flag` from a
    # numeric mismatch showed zero NLI breakdown, while `minor`/`none`
    # verdicts always got the full sentence-level picture. Backwards for a
    # tool whose point is being more transparent than a black-box judge, and
    # the "efficiency" it bought was negligible (NLI runs locally, no API
    # cost). Both signals now always populate `evidence`, and severity is
    # `flag` if EITHER fires, with the reason reflecting whichever did.
    numeric = compare_numeric_claims(answer, reference_context)

    if not reference_context:
        return {
            "severity": "none",
            "concern": False,
            "truthfulness_score": 100,
            "reason": "No reference material was retrieved for this question -- nothing to check against.",
            "evidence": {"numeric": numeric, "nli_per_sentence": []},
        }

    # reference_context is retrieval.py's top-k chunks joined with blank
    # lines. Feeding the whole blob to the NLI model as one premise dilutes
    # its focus badly -- NLI is trained on short, focused sentence/passage
    # pairs, not document-length premises padded with citation/license
    # boilerplate. Instead: check each answer sentence against each
    # individual retrieved chunk, and take the best (highest-entailment)
    # match per sentence -- this is what actually determines whether that
    # sentence is supported, not whether the whole concatenated blob is.
    #
    # Also drop non-content chunks (markdown headers, **Citation:**/
    # **License:** lines, and this project's own corpus-provenance
    # disclaimer paragraph) before comparing -- they're metadata about the
    # reference, not reference material to entail/contradict against, and
    # empirically one of them can outscore the real content chunk by noise
    # alone if left in (verified: the disclaimer paragraph beat the actual
    # matching sentence for a correct answer during testing). This matches
    # the current corpus/*.md convention (see corpus files' own headers) --
    # if that convention changes, this filter needs to change with it.
    reference_chunks = [
        c.strip()
        for c in reference_context.split("\n\n")
        if c.strip()
        and not c.strip().startswith("#")
        and not c.strip().startswith("**")
        and "not written or paraphrased by an ai" not in c.lower()
    ]
    if not reference_chunks:
        reference_chunks = [reference_context]

    # For each answer sentence, find its best-matching reference chunk --
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
        # verdict -- confirmed via testing that comparing a sentence against
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
    elif "neutral" in labels:
        worst = max((s for s in per_sentence if s["label"] == "neutral"), key=lambda s: s["neutral"])
        severity, concern = "minor", False
        reason = (
            f"Answer isn't clearly entailed by the reference material (neutral "
            f"p={worst['neutral']:.2f}) -- likely adds detail beyond what the reference "
            "covers, not a contradiction."
        )
    else:
        severity, concern = "none", False
        reason = "Answer is consistent with the reference material."

    max_contradiction = max((s["contradiction"] for s in per_sentence), default=0.0)
    truthfulness_score = 0 if numeric_mismatch else round(100 * (1 - max_contradiction))
    return {
        "severity": severity,
        "concern": concern,
        "truthfulness_score": truthfulness_score,
        "reason": reason,
        "evidence": {"numeric": numeric, "nli_per_sentence": per_sentence},
    }
