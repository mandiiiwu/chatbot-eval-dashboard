"""Core eval loop.

For each question:
  1. Ask the target model directly (no context)—"ungrounded" answer.
  2. Ask the target model again, this time with retrieved corpus context
     injected—"grounded" answer. This is the RAG-baseline comparison
     described in the project notes.
  3. Fact-check the ungrounded answer against the retrieved reference
     material—locally, via harness/fact_check.py's rule-based numeric
     comparison + small NLI classifier. NOT a generative LLM judge call—
     see PLAN.md's Phase 4 and memory feedback_no_llm_judge.md for why.

Separately, questions sharing a group_id are treated as tone/phrasing
variants of the same underlying question. We measure how much the target
model's (ungrounded) answers drift across phrasings of the same question—
a consistency check that's independent of ground truth.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Callable

from . import alerting, config, coverage_check, embeddings, fact_check, report, retrieval
from .target_client import chat as target_chat


def _questions_fingerprint(questions: list[dict]) -> str:
    """Short hash of the exact question set a run actually used -- mirrors
    retrieval.corpus_fingerprint()'s same principle, applied to the other
    domain-specific input that can now vary run to run even when its
    *filename* doesn't. Added 2026-08-16 alongside making
    auto_generate_questions the default everywhere: an auto-generated set
    always saves to the same path (questions/generated_questions.json),
    but MicroDC's generation calls run at temperature 0.7-0.8, so two
    separate generation runs against the identical corpus can produce
    genuinely different question text under that same filename. Without
    this, history.load_comparable_runs()'s existing target_model/
    corpus_fingerprint filters would have no way to tell "same file,
    coincidentally same content" from "same file, actually different
    questions" apart, and could silently mix runs that asked different
    questions onto the same trend line -- exactly the kind of mixing
    target_model/corpus_fingerprint filtering already exists to prevent
    for the other two axes."""
    encoded = json.dumps(
        sorted((q["id"], q["question"]) for q in questions),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()[:12]


def _tone_consistency(answers: list[str]) -> float:
    """0-100 score for how similar a set of answers are to each other, via
    pairwise cosine similarity in embedding space (MicroDC's
    qwen3-embedding:8b)—catches paraphrases/synonyms the original v1
    Jaccard-token-overlap approach missed (see PLAN.md's Phase 5). Only
    meaningful for 2+ answers. One batch embedding call for the whole group,
    not one per pair."""
    if len(answers) < 2:
        return 100.0
    vectors = embeddings.embed_texts(answers)
    scores = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            scores.append(retrieval._cosine(vectors[i], vectors[j]))
    return round(100 * sum(scores) / len(scores), 1)


def load_questions(questions_file: str | None = None) -> list[dict]:
    """questions_file overrides config.QUESTIONS_FILE for this call only
    (CLI --questions / the dashboard's per-run questions-file selector).
    Hard-blocks via config.require_questions_file() if neither resolves to
    a real file -- no silent default, see that function's docstring for
    why."""
    path = config.require_questions_file(questions_file)
    with open(path) as f:
        return json.load(f)


def estimate_cost(
    questions_file: str | None = None, auto_generate_questions: bool = True
) -> tuple[float, float, bool]:
    """Rough (low, high) USD range for what clicking RUN_EVAL would cost in
    MicroDC credits -- shown to the user before a real run starts, never
    used for the actual charged amount (that always comes from MicroDC's
    own billing via embeddings._record_cost). Estimated from the number of
    embedding jobs a real run would make, not corpus size directly: corpus
    chunks batched at embeddings.MAX_BATCH_SIZE, one query embedding per
    question in the coverage check, one more per question in the eval
    loop's own retrieval, and one tone-consistency embed per group that
    actually has 2+ phrasing variants (a group of 1 skips that call
    entirely -- see _tone_consistency). Job count times the one real
    per-job cost data point observed so far (see
    embeddings.OBSERVED_MIN_JOB_COST_USD); "high" just triples that as a
    hedge against real uncertainty about how cost might scale at batch
    sizes larger than what's actually been tested, not a second,
    better-grounded figure.

    auto_generate_questions mirrors run_and_save()'s own flag and its
    default (see that function's docstring for why True is the default
    everywhere now, CLI included): if no questions file resolves and this
    is True, the estimate is projected from question_gen's own per-topic/
    per-variant ceilings (one topic per corpus file) instead of raising --
    can overestimate, since real generation often accepts fewer candidates
    than the ceiling (the same "ceiling, not a guarantee" framing
    generate_questions() itself documents), but never silently hides that
    a generation step is about to run. The third return value flags this;
    the generation step's own MicroDC cost (chat completions, not
    embedding jobs -- a materially different cost model with no calibrated
    figure yet) is deliberately NOT folded into low/high, rather than
    showing a falsely precise number for a cost model nobody's actually
    measured."""
    will_generate_questions = False
    try:
        questions = load_questions(questions_file)
        n = len(questions)
        group_sizes: dict[str, int] = {}
        for q in questions:
            gid = q.get("group_id", q["id"])
            group_sizes[gid] = group_sizes.get(gid, 0) + 1
        tone_jobs = sum(1 for count in group_sizes.values() if count >= 2)
    except SystemExit:
        if not auto_generate_questions:
            raise
        will_generate_questions = True
        from . import question_gen

        num_topics = len(retrieval.load_chunks_by_file())
        n = num_topics * question_gen.QUESTIONS_PER_TOPIC_DEFAULT * question_gen.VARIANTS_PER_QUESTION_DEFAULT
        tone_jobs = num_topics * question_gen.QUESTIONS_PER_TOPIC_DEFAULT

    chunk_count = len(retrieval._load_chunks())
    corpus_jobs = -(-chunk_count // embeddings.MAX_BATCH_SIZE) if chunk_count else 0  # ceil division

    total_jobs = corpus_jobs + (2 * n) + tone_jobs
    low = round(total_jobs * embeddings.OBSERVED_MIN_JOB_COST_USD, 2)
    high = round(low * 3, 2)
    return low, high, will_generate_questions


def run_evaluation(
    questions: list[dict] | None = None,
    verbose: bool = True,
    target_model: str | None = None,
    endpoint_config: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """target_model overrides config.TARGET_MODEL for this run only—V2-C's
    --target-model (run_eval.py), so comparing several models doesn't
    require hand-editing .env between runs.

    endpoint_config (V2-G dashboard UI) similarly overrides
    config.TARGET_PROVIDER/CUSTOM_* for this run only—keys: provider,
    endpoint_url, endpoint_headers, request_template, response_path. None
    or a missing key means "use the .env value". Never stored in the
    results dict (unlike target_model); endpoint_headers can carry an
    auth secret, and results/*.json files are meant to be shareable.

    progress_callback(done, total), if given, fires after each question
    finishes -- the dashboard's RUN_EVAL button uses this to show
    "RUNNING... N/M" instead of sitting blank for however long a full run
    takes (each question is 2 model calls, so this can be several minutes
    for a real model; a bare "RUNNING..." with no feedback for that long
    reads as broken/hung even when it's working fine)."""
    questions = questions if questions is not None else load_questions()  # config.QUESTIONS_FILE
    target_model = target_model or config.TARGET_MODEL
    endpoint_config = endpoint_config or {}
    total = len(questions)

    def _chat(messages: list[dict]) -> str:
        return target_chat(target_model, messages, **endpoint_config)

    per_question = []
    groups: dict[str, list[str]] = {}

    for i, q in enumerate(questions):
        if verbose:
            print(f"[{q['id']}] asking target model...")
        ungrounded = _chat([{"role": "user", "content": q["question"]}])

        context = retrieval.retrieve_context(q["question"])
        grounded = _chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using ONLY the following reference material. "
                        "If it doesn't cover the question, say so explicitly.\n\n" + context
                    ),
                },
                {"role": "user", "content": q["question"]},
            ],
        )

        if verbose:
            print(f"[{q['id']}] fact-checking (local, no LLM judge)...")
        verdict = fact_check.check_answer(context, ungrounded)

        per_question.append(
            {
                "id": q["id"],
                "group_id": q.get("group_id", q["id"]),
                "question": q["question"],
                "ungrounded_answer": ungrounded,
                "grounded_answer": grounded,
                "reference_context": context,
                "truthfulness_score": verdict["truthfulness_score"],
                "severity": verdict["severity"],
                "concern": bool(verdict["concern"]),
                "reason": verdict["reason"],
                "evidence": verdict["evidence"],
            }
        )
        groups.setdefault(q.get("group_id", q["id"]), []).append(ungrounded)
        if progress_callback:
            progress_callback(i + 1, total)

    tone_scores = {gid: _tone_consistency(answers) for gid, answers in groups.items()}
    for row in per_question:
        row["tone_consistency_score"] = tone_scores[row["group_id"]]

    n = len(per_question)
    flagged = sum(1 for r in per_question if r["severity"] == "flag")
    minor = sum(1 for r in per_question if r["severity"] == "minor")
    avg_truthfulness = round(sum(r["truthfulness_score"] for r in per_question) / n, 1) if n else 0
    avg_tone = round(sum(tone_scores.values()) / len(tone_scores), 1) if tone_scores else 100.0

    return {
        "schema_version": config.SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_model": target_model,
        "corpus_fingerprint": retrieval.corpus_fingerprint(),
        "questions_fingerprint": _questions_fingerprint(questions),
        "judge_model": f"local: rules + {fact_check.NLI_MODEL_NAME} (no generative LLM judge)",
        "num_questions": n,
        "flagged_count": flagged,
        "minor_count": minor,
        "concern_percentage": round(100 * flagged / n, 1) if n else 0,
        "avg_truthfulness_score": avg_truthfulness,
        "avg_tone_consistency_score": avg_tone,
        "questions": per_question,
    }


def run_and_save(
    questions: list[dict] | None = None,
    questions_file: str | None = None,
    auto_generate_questions: bool = True,
    skip_coverage_check: bool = False,
    verbose: bool = True,
    target_model: str | None = None,
    endpoint_config: dict | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Load questions (if not given directly), run the coverage checks
    (V2-H) unless skipped, run the evaluation, and persist results to
    results/. The single entry point both the CLI (run_eval.py) and the
    live dashboard's RUN_EVAL button (harness/server.py) call, so the two
    can't drift out of sync.

    questions_file, when `questions` isn't given directly, overrides
    config.QUESTIONS_FILE for this call only (same override convention as
    target_model) -- hard-blocks via config.require_questions_file() if
    neither resolves to a real file AND auto_generate_questions is False.

    auto_generate_questions defaults to True everywhere (added 2026-08-16;
    flipped from an opt-in default 2026-08-16 same day, per explicit user
    request: uploading a corpus+model should converge on the exact same
    behavior whether it happens through the dashboard UI, the CLI, or
    dropping files directly into corpus/ and running run_eval.py -- no
    "the UI is friendlier than the CLI" split). If no questions file
    resolves, runs question_gen.generate_questions() against the attached
    corpus instead of hard-blocking, saves the result to
    questions/generated_questions.json, and uses it for this run -- the
    same generation pipeline generate_questions.py already exposes as a
    separate manual CLI step, just triggered automatically when nothing
    else is configured. run_eval.py exposes `--no-auto-generate-questions`
    for anyone who explicitly wants the old strict-fail behavior (e.g.
    scripted/CI usage where an unexpected MicroDC spend is worse than a
    clear error). Still hard-blocks (via a clear SystemExit) if generation
    itself produces zero usable questions -- an auto-generation attempt
    that silently produced nothing wouldn't actually satisfy "one click
    does it all," it would just fail one step later with a more confusing
    error.

    If `questions` IS given directly (run_eval.py already loaded it from
    disk itself), questions_file is still recorded on the results dict
    when passed, purely for diagnostics/display -- it isn't re-opened, and
    auto_generate_questions has no effect.

    Either way, the resolved file's basename ends up on
    results['questions_file'] so the dashboard/daily-job-sync can see
    exactly which question set produced a given run -- added after a real
    run fired against a stale, mismatched question set with no record
    anywhere of which file was actually used (see PLAN.md's 2026-08-16
    audit).

    progress_callback(phase, done, total), if given, fires during both the
    corpus-embedding phase (phase="embedding", see
    retrieval.set_embedding_progress_callback() below -- only actually
    fires real work when a fresh/changed corpus needs re-embedding, a
    no-op cache hit fires nothing) and the per-question evaluation phase
    (phase="questions", run_evaluation()'s own (done, total) callback
    wrapped here). Added 2026-08-16 after a real run against the
    116,616-chunk CUAD corpus sat at "starting..." in the dashboard for
    hours with zero visible progress, indistinguishable from a hang --
    the corpus-wide embed happens entirely before question 1, and the
    question-loop-only progress callback that already existed had no
    visibility into that phase at all.

    Times and cost-tracks the whole call, not just run_evaluation()'s own
    loop -- the coverage check above also embeds every question against the
    corpus, which is real wall-clock time and real MicroDC spend that
    "from when you click RUN_EVAL" should include. Requested directly after
    an intentionally oversized corpus (a full raw dataset dump, not curated
    excerpts) made a run slow and expensive enough to be worth measuring
    rather than guessing at."""
    start_time = time.monotonic()
    embeddings.reset_cost_tracker()

    def _embedding_progress(done: int, total: int) -> None:
        if progress_callback:
            progress_callback("embedding", done, total)

    # Scoped to just this call via try/finally: retrieval.py's hook is
    # module-level state (see its own docstring for why -- _chunk_embeddings
    # is an lru_cache'd zero-arg function, so a callback can't be passed as
    # a normal argument without either busting the cache every call or
    # needing a cache-key exclusion functools.lru_cache doesn't support).
    # Clearing it afterward stops a later call in the same long-lived
    # server process (a CLI run, a cost estimate) from firing into a
    # callback that belongs to a request that already finished.
    retrieval.set_embedding_progress_callback(_embedding_progress if progress_callback else None)
    try:
        # A corpus file edited directly on disk (or dropped in some other
        # way) while this process is alive needs to be picked up before
        # this run uses it -- ensure_fresh_caches() guarantees that (via
        # corpus_fingerprint(), which always reads real file content fresh)
        # without forcing a full, real re-embed on every single run when
        # corpus/ hasn't actually changed since the last one. That
        # distinction matters a lot in practice: for the CUAD stress-test
        # corpus (116,616 chunks), an unconditional clear here was
        # responsible for a large chunk of an 8-hour real run, since it
        # silently discarded a still-valid cache and re-embedded the whole
        # corpus from scratch every time, including the common case of a
        # scheduled nightly run against an unchanged corpus (see PLAN.md's
        # corpus-scaling-plan lever #1).
        retrieval.ensure_fresh_caches()
        just_generated = False
        if questions is None:
            try:
                questions_file = config.require_questions_file(questions_file)
            except SystemExit:
                if not auto_generate_questions:
                    raise
                questions_file = None  # not resolved to a real file yet -- generated below instead

            if questions_file:
                with open(questions_file) as f:
                    questions = json.load(f)
            else:

                def _generation_progress(done: int, total: int) -> None:
                    if progress_callback:
                        progress_callback("generating_questions", done, total)

                if verbose:
                    print("No questions file configured -- auto-generating one from corpus/...")
                from . import question_gen

                questions, _stats = question_gen.generate_questions(
                    verbose=verbose, progress_callback=_generation_progress
                )
                if not questions:
                    raise SystemExit(
                        "Auto-generation produced zero usable questions from the attached "
                        "corpus -- it may be too sparse, off-topic, or lack a `# Title` per "
                        "file for topic naming. Provide a questions/*.json file manually "
                        "instead (see README/PLAN.md)."
                    )
                just_generated = True
                os.makedirs(config.QUESTIONS_DIR, exist_ok=True)
                questions_file = os.path.join(config.QUESTIONS_DIR, "generated_questions.json")
                with open(questions_file, "w") as f:
                    json.dump(questions, f, indent=2)
        if not skip_coverage_check and not just_generated:
            # Free, local, zero MicroDC cost -- catches an obviously wrong
            # question/corpus pairing before the expensive corpus-wide
            # embedding pass below even starts (see coverage_check.py's
            # KEYWORD_PRECHECK_THRESHOLD). Deliberately coarser than the
            # real check that follows it, not a replacement for it.
            #
            # Skipped entirely for questions this same call just generated
            # (just_generated): question_gen.generate_questions() already
            # ran the real embedding-based coverage check
            # (coverage_check.check_coverage(), the same one require_coverage()
            # calls) on every individual candidate AND every individual
            # variant during generation -- more granular than this
            # aggregate check, not less. Re-running the keyword precheck on
            # top of that is worse than redundant: it's a real, confirmed
            # false-positive source, since generated tone variants are
            # deliberately casual/typo'd/roundabout paraphrases (see
            # question_gen.py's _TONE_HINTS) that can score low on raw
            # keyword overlap even though they passed the rigorous
            # embedding check moments earlier. Caught live (2026-08-16): a
            # real generation run against a small test corpus produced 12
            # individually-verified questions, then immediately had 8 of
            # them rejected by the keyword precheck for exactly this
            # reason, hard-blocking a run that should have succeeded.
            coverage_check.require_coverage_keyword(questions)
            coverage_check.require_coverage(questions)  # corpus-wide embed (if needed) happens inside here

        def _question_progress(done: int, total: int) -> None:
            if progress_callback:
                progress_callback("questions", done, total)

        results = run_evaluation(
            questions,
            verbose=verbose,
            target_model=target_model,
            endpoint_config=endpoint_config,
            progress_callback=_question_progress,
        )
    finally:
        retrieval.set_embedding_progress_callback(None)
    if questions_file:
        results["questions_file"] = os.path.basename(questions_file)
    results["duration_seconds"] = round(time.monotonic() - start_time, 1)
    results["microdc_cost_usd"] = round(embeddings.get_total_cost(), 4)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(config.RESULTS_DIR, f"{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(config.RESULTS_DIR, "latest.json"), "w") as f:
        json.dump(results, f, indent=2)
    report.write_report(results, os.path.join(config.RESULTS_DIR, "latest.html"))
    alerting.maybe_alert(results)  # V2-D: local notification if concern_percentage crosses the line

    return results
