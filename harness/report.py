"""Renders a results dict (see evaluator.run_evaluation) to a single
self-contained HTML dashboard. No JS framework, no build step -- just open it
in a browser, or serve it live via harness/server.py.

v2 layout (2026-08-13), redesigned from a Claude Design handoff package,
superseding the earlier hero-concern-percentage version: a left rail (run
config + metadata), Truthfulness/Tone Consistency score panels, a 2-series
trend chart where clicking any point swaps the ENTIRE dashboard to that run's
data (client-side -- every comparable run's full per-question data is
embedded in the page), and questions triaged into Flagged/Minor/Ok groups
instead of one flat list. The concern-percentage hero figure is intentionally
dropped per the user's design decision, not an oversight.

Score panels, run-date labels, and the flagged/minor/ok groups are rendered
entirely client-side from window.__RUNS__ (one shared renderRun() function
handles both the initial page load and every run-selection click), so there's
no duplicate rendering logic between Python and JS for that reactive part.
Only the trend chart's static SVG geometry and the left rail's config/
metadata (which don't change per selected run -- they describe the current
setup, not the run being viewed) are server-rendered.
"""

import glob
import html
import json
import os

from . import config, history

_SEVERITY_LABEL = {"flag": "FLAGGED", "minor": "MINOR", "none": "OK"}
_SEVERITY_CLASS = {"flag": "critical", "minor": "warning", "none": "good"}

# Short, accurate bullets for the score panels' "what does this mean?" hover
# popups and the static Key panel. Corrected 2026-08-13: an earlier design
# draft's copy claimed NLI only runs when there's no numeric claim (a
# short-circuit) -- that's not how fact_check.py actually works. Both the
# numeric/claim check and NLI entailment always run, independently, and
# severity escalates to FLAGGED if either one finds a problem (see
# fact_check.check_answer's docstring for why the short-circuit was removed).
_TRUTH_BULLETS = [
    "numeric/claim check and NLI entailment both always run, independently",
    "flags if either finds a problem: a numeric mismatch, or any sentence NLI-contradicts the reference",
    "can misread hedging, added detail, or emphasis as neutral/contradiction",
]
_TONE_BULLETS = [
    "embedding cosine similarity across paraphrased rephrasings of the same question",
    "no reference corpus involved -- compares the model's own answers to each other",
    "penalizes legitimate specificity/detail variance, not just tone drift",
]
_KEY_ITEMS = [
    ("flagged", "critical", [
        "numeric value contradicts the reference",
        "or any answer sentence NLI-contradicts it",
    ]),
    ("minor", "warning", [
        "no contradiction, but no sentence is clearly entailed (all NLI-neutral)",
        "often just detail the reference doesn't cover, not an error",
    ]),
    ("ok", "good", [
        "at least one sentence is clearly entailed by the reference",
        "and no numeric mismatch or contradiction",
    ]),
]
_KEY_CAVEATS = [
    "not 100% reliable -- NLI can misread hedging as contradiction",
    "doesn't penalize incomplete answers -- only checks whether what's present "
    "contradicts the reference, not whether the answer covers everything asked",
]


def _corpus_files() -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(config.CORPUS_DIR, "*.md")))


def _trend_chart_svg(runs: list[dict]) -> str:
    """Static SVG geometry for all comparable runs -- 2 series (truthfulness,
    tone consistency). Points carry data-run attributes; JS toggles a
    'selected' class and moves the guide line on click, it doesn't redraw
    the chart."""
    if len(runs) < 2:
        return (
            f'<p class="trend-note">Only {len(runs)} run{"s" if len(runs) != 1 else ""} so far on the '
            "current architecture -- need at least 2 to show a trend. Runs from earlier "
            "architectures (different target model, judge, or retrieval mechanism) are "
            "intentionally excluded since their numbers aren't directly comparable.</p>"
        )

    w, h, pad_l, pad_r, pad_t, pad_b = 900, 320, 30, 20, 14, 26
    n = len(runs)

    def x_of(i: int) -> float:
        return pad_l + (w - pad_l - pad_r) * i / (n - 1)

    def y_of(value: float) -> float:
        return h - pad_b - (h - pad_t - pad_b) * (value / 100)

    gridlines = "".join(
        f'<line x1="{pad_l}" y1="{y_of(v):.1f}" x2="{w - pad_r}" y2="{y_of(v):.1f}" class="gridline" />'
        f'<text x="{pad_l - 6}" y="{y_of(v) + 3:.1f}" class="axis-label" text-anchor="end">{v}</text>'
        for v in (0, 25, 50, 75, 100)
    )

    series = [("avg_truthfulness_score", "trend-truth"), ("avg_tone_consistency_score", "trend-tone")]
    series_svg = []
    for key, css_class in series:
        pts = [(x_of(i), y_of(r[key])) for i, r in enumerate(runs)]
        points_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dots = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="trend-dot {css_class}" '
            f'data-run="{i}" onclick="selectRun({i})" />'
            for i, (x, y) in enumerate(pts)
        )
        series_svg.append(f'<polyline points="{points_attr}" class="trend-line {css_class}" />{dots}')

    hit_columns = "".join(
        f'<rect x="{x_of(i) - (w / n) / 2:.1f}" y="{pad_t}" width="{(w - pad_l - pad_r) / max(n - 1, 1):.1f}" '
        f'height="{h - pad_t - pad_b}" class="hit-col" data-run="{i}" onclick="selectRun({i})" />'
        for i in range(n)
    )

    guide_x = [x_of(i) for i in range(n)]
    guide_x_json = json.dumps(guide_x)

    return f"""
    <svg viewBox="0 0 {w} {h}" class="trend-svg" id="trend-svg" data-guide-x='{guide_x_json}'>
      {gridlines}
      <line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" class="axis-line" />
      {''.join(series_svg)}
      <line id="trend-guide" x1="0" y1="{pad_t}" x2="0" y2="{h - pad_b}" class="trend-guide" />
      {hit_columns}
    </svg>
    <div class="legend">
      <span class="legend-item"><span class="legend-swatch trend-truth"></span>truthfulness</span>
      <span class="legend-item"><span class="legend-swatch trend-tone"></span>tone consistency</span>
    </div>
    <details class="table-toggle">
      <summary>View as table</summary>
      <table class="trend-table">
        <thead><tr><th>timestamp (UTC)</th><th>truthfulness</th><th>tone consistency</th><th>flagged</th></tr></thead>
        <tbody>{"".join(
            f"<tr><td>{html.escape(r.get('timestamp', '?'))}</td><td>{r['avg_truthfulness_score']}</td>"
            f"<td>{r['avg_tone_consistency_score']}</td><td>{r['flagged_count']}/{r['num_questions']}</td></tr>"
            for r in runs
        )}</tbody>
      </table>
    </details>
    """


def _key_panel_html() -> str:
    items = "".join(
        f"""
        <div class="key-item">
          <div class="key-label status-{cls}">{label}</div>
          <ul>{"".join(f"<li>{html.escape(b)}</li>" for b in bullets)}</ul>
        </div>
        """
        for label, cls, bullets in _KEY_ITEMS
    )
    return f"""
    <div class="key-panel">
      <span class="section-label">[KEY]</span>
      {items}
      {"".join(f'<p class="key-caveat">{html.escape(c)}</p>' for c in _KEY_CAVEATS)}
    </div>
    """


def render_html(results: dict) -> str:
    # Filtered to the current run's target_model -- fixes a real bug where
    # this used to mix different models' scores onto the same trend line,
    # which is misleading, not just incomplete. Kept independent of the
    # leaderboard UI (removed 2026-08-13, not a wanted feature) since the
    # bug it fixes exists regardless of whether there's a comparison view.
    runs = history.load_comparable_runs(target_model=results["target_model"])
    if not runs or runs[-1].get("timestamp") != results.get("timestamp"):
        runs = runs + [results]

    runs_json = json.dumps(runs)
    corpus_files = _corpus_files()
    corpus_summary = f"{len(corpus_files)} file{'s' if len(corpus_files) != 1 else ''}"
    truth_bullets_html = "".join(f"<span>{html.escape(b)}</span>" for b in _TRUTH_BULLETS)
    tone_bullets_html = "".join(f"<span>{html.escape(b)}</span>" for b in _TONE_BULLETS)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chatbot Eval Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --good: #1f8a4c; --warning: #c8860d; --bad: #c94a3f;
    --good-bg: rgba(31,138,76,0.12); --warning-bg: rgba(200,134,13,0.12); --bad-bg: rgba(201,74,63,0.12);
  }}
  .viz-root {{
    color-scheme: dark;
    --bg:     #151715;
    --border: #2c302d;
    --track:  #20221f;
    --text:   #c9d1c9;
    --muted:  #7c877c;
    --faint:  #565c56;
    --trend-truth: #c9d1c9;
    --trend-tone:  #4a5c4a;
  }}
  :root[data-theme="light"] .viz-root {{
    color-scheme: light;
    --bg:     #e9e4d8;
    --border: #cdc7b8;
    --track:  #ddd7c8;
    --text:   #26292b;
    --muted:  #63665f;
    --faint:  #8b8d85;
    --trend-truth: #26292b;
    --trend-tone:  #9a998f;
  }}

  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; }}
  .viz-root {{
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    background: var(--bg); color: var(--text);
    min-height: 100vh;
  }}
  .layout {{ display: flex; align-items: stretch; min-height: 100vh; }}

  /* --- left rail --- */
  .rail {{ width: 240px; flex-shrink: 0; border-right: 1px solid var(--border);
          padding: 18px 16px; display: flex; flex-direction: column; gap: 14px; }}
  .live-line {{ display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); }}
  .live-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--good); flex-shrink: 0; }}
  .section-label {{ font-size: 12px; color: var(--muted); font-weight: 600; }}
  .rail-field {{ display: flex; flex-direction: column; gap: 4px; }}
  .rail-field-label {{ font-size: 10px; color: var(--muted); }}
  .field-help {{ position: relative; display: inline-flex; align-items: center; justify-content: center;
                margin-left: 4px; cursor: help; color: var(--muted); border: 1px solid var(--border);
                border-radius: 50%; width: 12px; height: 12px; font-size: 9px; line-height: 1; }}
  .field-help-popup {{ display: none; position: absolute; top: 100%; left: 0; margin-top: 6px; width: 190px;
                       background: var(--bg); border: 1px solid var(--border); box-shadow: 0 4px 16px rgba(0,0,0,0.3);
                       padding: 8px 10px; font-size: 10px; color: var(--muted); line-height: 1.5; z-index: 10;
                       text-transform: none; }}
  .field-help:hover .field-help-popup {{ display: block; }}
  .rail-box {{ border: 1px solid var(--border); padding: 8px 10px; font-size: 11px; word-break: break-word; }}
  .rail-box.dashed {{ border-style: dashed; color: var(--muted); }}
  .rail-input {{ border: 1px solid var(--border); background: var(--bg); color: var(--text);
                padding: 8px 10px; font: 11px 'IBM Plex Mono'; width: 100%; }}
  .corpus-list {{ display: flex; flex-direction: column; gap: 4px; }}
  .corpus-item {{ display: flex; align-items: center; justify-content: space-between; gap: 6px;
                  border: 1px solid var(--border); padding: 5px 8px; font-size: 10px; }}
  .corpus-fname {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .corpus-delete {{ border: none; background: transparent; color: var(--bad); cursor: pointer;
                    font-size: 13px; line-height: 1; flex-shrink: 0; padding: 0 2px; }}
  .corpus-upload {{ display: flex; gap: 6px; align-items: center; margin-top: 2px; }}
  .corpus-upload input[type="file"] {{ flex: 1; min-width: 0; font-size: 9px; color: var(--muted); }}
  .corpus-upload button {{ border: 1px solid var(--border); background: transparent; color: var(--text);
                           font: 10px 'IBM Plex Mono'; padding: 5px 8px; cursor: pointer; flex-shrink: 0; }}
  .run-eval-btn {{ margin-top: 2px; padding: 10px; border: 1px solid var(--text); background: var(--text);
                   color: var(--bg); font: 600 11px 'IBM Plex Mono'; cursor: pointer; }}
  .run-eval-btn:disabled {{ opacity: 0.5; cursor: default; }}
  .run-eval-error {{ font-size: 10px; color: var(--bad); display: none; }}
  .rail-divider {{ border: none; border-top: 1px solid var(--border); margin: 0; }}
  .meta-row {{ display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); margin-top: 6px; }}
  .meta-value {{ color: var(--text); }}

  /* --- main content --- */
  .main {{ flex: 1; min-width: 0; padding: 16px 24px 40px; }}
  .header-strip {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }}
  #run-label-header {{ font-size: 15px; color: var(--muted); }}
  #theme-toggle {{ display: flex; gap: 6px; border: 1px solid var(--border); background: transparent;
                   color: var(--text); cursor: pointer; font: 11px 'IBM Plex Mono'; padding: 5px 10px; }}
  #theme-toggle .dim {{ opacity: 0.35; }}

  .panels-row {{ display: flex; gap: 16px; margin-bottom: 18px; }}
  .score-panel {{ flex: 1; border: 1px solid var(--border); padding: 32px; display: flex;
                 flex-direction: column; gap: 16px; position: relative; }}
  .score-figure {{ display: flex; align-items: baseline; gap: 8px; }}
  .score-figure .num {{ font-size: 38px; font-weight: 600; }}
  .score-figure .of100 {{ font-size: 12px; color: var(--muted); }}
  .score-bar-track {{ height: 6px; background: var(--track); overflow: hidden; }}
  .score-bar-fill {{ height: 100%; }}
  .meaning-trigger {{ font-size: 10px; color: var(--muted); text-decoration: underline dotted; cursor: help;
                      width: fit-content; }}
  .meaning-popup {{ display: none; position: absolute; left: 0; right: 0; top: 100%; margin-top: 6px; background: var(--bg);
                    border: 1px solid var(--border); box-shadow: 0 4px 16px rgba(0,0,0,0.3); padding: 12px 14px;
                    font-size: 10px; color: var(--muted); flex-direction: column; gap: 4px; z-index: 5; }}
  .meaning-popup span:before {{ content: "\\2022  "; color: var(--muted); }}
  .score-panel:hover .meaning-popup {{ display: flex; }}

  section {{ margin-bottom: 20px; }}
  .trend-box {{ border: 1px solid var(--border); padding: 32px; }}
  .trend-note {{ color: var(--muted); font-size: 0.85rem; margin: 0; }}
  .trend-svg {{ width: 100%; height: auto; overflow: visible; cursor: pointer; }}
  .gridline {{ stroke: var(--border); stroke-width: 1; }}
  .axis-line {{ stroke: var(--border); stroke-width: 1; }}
  .axis-label {{ font-size: 9px; fill: var(--muted); font-family: 'IBM Plex Mono'; }}
  .trend-line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .trend-line.trend-truth {{ stroke: var(--trend-truth); }}
  .trend-line.trend-tone {{ stroke: var(--trend-tone); }}
  .trend-dot {{ stroke: var(--bg); stroke-width: 2; cursor: pointer; }}
  .trend-dot.trend-truth {{ fill: var(--trend-truth); }}
  .trend-dot.trend-tone {{ fill: var(--trend-tone); }}
  .trend-dot.selected {{ r: 6; }}
  .hit-col {{ fill: transparent; cursor: pointer; }}
  .trend-guide {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; pointer-events: none; }}
  .legend {{ display: flex; gap: 16px; font-size: 0.75rem; color: var(--muted); margin-top: 10px; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-swatch {{ width: 10px; height: 2px; display: inline-block; }}
  .legend-swatch.trend-truth {{ background: var(--trend-truth); }}
  .legend-swatch.trend-tone {{ background: var(--trend-tone); }}
  .table-toggle {{ margin-top: 10px; font-size: 0.78rem; }}
  .table-toggle summary {{ cursor: pointer; color: var(--muted); }}
  .trend-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.76rem; }}
  .trend-table th, .trend-table td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--border); }}
  .trend-table th {{ color: var(--muted); font-weight: 500; }}

  .groups-row {{ display: flex; gap: 16px; align-items: flex-start; }}
  .groups-col {{ flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 14px; }}
  .sev-group summary {{ cursor: pointer; font-size: 24px; font-weight: 700; list-style: none; margin-bottom: 4px; }}
  .sev-group summary::-webkit-details-marker {{ display: none; }}
  .sev-group summary:before {{ content: "\\25b8  "; color: var(--muted); }}
  .sev-group[open] summary:before {{ content: "\\25be  "; }}
  .sev-group.status-critical summary {{ color: var(--bad); }}
  .sev-group.status-warning summary {{ color: var(--warning); }}
  .sev-group.status-good summary {{ color: var(--good); }}
  .sev-group-body {{ margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }}
  .sev-empty {{ font-size: 11px; color: var(--muted); }}

  .q-row {{ border: 1px solid var(--border); padding: 10px 12px; font-size: 13px; }}
  .q-row.status-critical {{ border-color: var(--bad); }}
  .q-row.status-warning {{ border-color: var(--warning); }}
  .q-row.status-good {{ border-color: var(--good); }}
  .q-row summary {{ cursor: pointer; display: flex; justify-content: space-between; gap: 10px;
                    list-style: none; font-size: 13px; font-weight: 400; }}
  .q-row summary::-webkit-details-marker {{ display: none; }}
  .q-row .qtext {{ flex: 1; }}
  .q-row .qscores {{ color: var(--muted); flex-shrink: 0; }}
  .q-detail {{ margin-top: 8px; display: flex; flex-direction: column; gap: 8px; }}
  .q-question {{ color: var(--text); font-weight: 600; }}
  .q-verdict {{ color: var(--muted); }}
  .vague-warning {{ color: var(--warning); border: 1px dashed var(--warning); padding: 6px 8px; font-size: 12px; }}
  .hedge-marker {{ color: var(--warning); font-size: 11px; font-weight: 600; }}
  .numeric-readout {{ display: flex; gap: 8px; }}
  .numeric-box {{ flex: 1; border: 1px solid var(--border); padding: 6px 8px; font-size: 12px; }}
  .numeric-box .k {{ color: var(--muted); }}
  .numeric-box.answer {{ border-color: var(--bad); color: var(--bad); }}
  .numeric-box.reference {{ color: var(--muted); }}
  .nli-block {{ display: flex; flex-direction: column; gap: 4px; }}
  .nli-row {{ display: flex; align-items: center; gap: 8px; }}
  .nli-sentence {{ flex: 1; font-size: 12px; color: var(--muted); }}
  .nli-bar {{ display: flex; width: 80px; height: 6px; flex-shrink: 0; }}
  .nli-seg.entailment {{ background: var(--good); }}
  .nli-seg.neutral {{ background: var(--warning); }}
  .nli-seg.contradiction {{ background: var(--bad); }}
  .nli-label {{ font-size: 12px; min-width: 90px; text-align: right; flex-shrink: 0; }}
  .nli-label.status-good {{ color: var(--good); }}
  .nli-label.status-warning {{ color: var(--warning); }}
  .nli-label.status-critical {{ color: var(--bad); }}
  .full-answers-toggle summary {{ cursor: pointer; font-size: 10px; color: var(--muted); text-decoration: underline; }}
  .full-answers-toggle pre {{ background: var(--track); padding: 8px; white-space: pre-wrap; font-size: 10px;
                              margin: 6px 0 0; }}
  .full-answers-toggle p {{ margin: 8px 0 2px; font-size: 10px; color: var(--muted); }}

  .key-panel {{ width: 230px; flex-shrink: 0; border: 1px solid var(--border); padding: 14px;
               display: flex; flex-direction: column; gap: 10px; }}
  .key-item {{ display: flex; flex-direction: column; gap: 2px; }}
  .key-label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; }}
  .key-label.status-critical {{ color: var(--bad); }}
  .key-label.status-warning {{ color: var(--warning); }}
  .key-label.status-good {{ color: var(--good); }}
  .key-item ul {{ margin: 2px 0 0; padding-left: 14px; font-size: 10px; color: var(--muted); }}
  .key-caveat {{ font-size: 9px; color: var(--faint); margin: 4px 0 0; }}

  @media (max-width: 720px) {{
    .layout {{ flex-direction: column; }}
    .rail {{ width: auto; border-right: none; border-bottom: 1px solid var(--border); }}
    .panels-row, .groups-row {{ flex-direction: column; }}
    .key-panel {{ width: auto; }}
  }}
</style>
</head>
<body>
<div class="viz-root">
<div class="layout">
  <div class="rail">
    <div class="live-line"><span class="live-dot"></span><span id="run-label-rail">live</span></div>

    <div>
      <span class="section-label">[CONFIG]</span>
      <div style="display:flex;flex-direction:column;gap:10px;margin-top:10px;">
        <div class="rail-field">
          <span class="rail-field-label">model_endpoint<span class="field-help">?<span class="field-help-popup">Must match an Ollama-pulled model tag exactly &mdash; run <code>ollama list</code> in your terminal to see what's available locally. Needs to be reachable via the OpenAI-compatible endpoint at OLLAMA_BASE_URL. Changing this only affects your next run, it doesn't edit .env.</span></span></span>
          <input id="model-endpoint-input" class="rail-input" type="text" value="{html.escape(results['target_model'])}" spellcheck="false">
        </div>
        <div class="rail-field">
          <span class="rail-field-label">corpus ({corpus_summary})<span class="field-help">?<span class="field-help-popup">Upload real, verbatim reference text (.md or .txt) &mdash; not AI-generated or paraphrased content. Include citation/license info if it isn't your own writing. One coherent topic per file works best for retrieval.</span></span></span>
          <div id="corpus-list" class="corpus-list">
            {"".join(
                f'<div class="corpus-item"><span class="corpus-fname" title="{html.escape(f)}">{html.escape(f)}</span>'
                f'<button type="button" class="corpus-delete" data-filename="{html.escape(f)}" title="delete">&times;</button></div>'
                for f in corpus_files
            )}
          </div>
          <div class="corpus-upload">
            <input id="corpus-file-input" type="file" accept=".md,.txt">
            <button id="corpus-upload-btn" type="button">add file</button>
          </div>
          <div id="corpus-error" class="run-eval-error"></div>
        </div>
        <div class="rail-field">
          <span class="rail-field-label">questions</span>
          <div class="rail-box dashed">{results['num_questions']} configured &middot; edit questions/*.json</div>
        </div>
        <button id="run-eval-btn" class="run-eval-btn" type="button">RUN_EVAL</button>
        <div id="run-eval-error" class="run-eval-error"></div>
      </div>
    </div>

    <hr class="rail-divider">

    <div>
      <span class="section-label">[METADATA]</span>
      <div class="meta-row"><span>total runs</span><span class="meta-value">{len(runs)}</span></div>
      <div class="meta-row"><span>questions/run</span><span class="meta-value">{results['num_questions']}</span></div>
      <div class="meta-row"><span>judge</span></div>
      <div style="font-size:9px;color:var(--faint);margin-top:2px;">{html.escape(results['judge_model'])}</div>
    </div>

    <hr class="rail-divider">

    <div>
      <span class="section-label">[EXPORT]</span>
      <button id="save-json-btn" class="run-eval-btn" type="button" style="margin-top:10px;">SAVE_DATA_AS_JSON</button>
      <div style="font-size:9px;color:var(--faint);margin-top:6px;">downloads every comparable run currently loaded on this page &mdash; for feeding to another LLM or tool</div>
    </div>
  </div>

  <div class="main">
    <div class="header-strip">
      <span id="run-label-header"></span>
      <button id="theme-toggle" type="button"><span class="dim" id="theme-dark-label">[DARK]</span><span id="theme-light-label">[LIGHT]</span></button>
    </div>

    <div class="panels-row">
      <div class="score-panel">
        <span class="section-label">[TRUTHFULNESS] (T)</span>
        <div class="score-figure"><span class="num" id="truth-num"></span><span class="of100">/ 100</span></div>
        <div class="score-bar-track"><div class="score-bar-fill" id="truth-bar"></div></div>
        <span class="meaning-trigger">what does this mean?</span>
        <div class="meaning-popup">{truth_bullets_html}</div>
      </div>
      <div class="score-panel">
        <span class="section-label">[TONE / PHRASING CONSISTENCY] (C)</span>
        <div class="score-figure"><span class="num" id="tone-num"></span><span class="of100">/ 100</span></div>
        <div class="score-bar-track"><div class="score-bar-fill" id="tone-bar"></div></div>
        <span class="meaning-trigger">what does this mean?</span>
        <div class="meaning-popup">{tone_bullets_html}</div>
      </div>
    </div>

    <section class="trend-box">
      <span class="section-label">[TREND OVER TIME] click a run to see more detail</span>
      {_trend_chart_svg(runs)}
    </section>

    <section class="groups-row">
      <div class="groups-col" id="groups-col"></div>
      {_key_panel_html()}
    </section>
  </div>
</div>

<script>
  window.__RUNS__ = {runs_json};
  window.__SEVERITY_LABEL__ = {json.dumps(_SEVERITY_LABEL)};
  window.__SEVERITY_CLASS__ = {json.dumps(_SEVERITY_CLASS)};

  (function() {{
    document.getElementById('save-json-btn').addEventListener('click', function() {{
      const blob = new Blob([JSON.stringify(window.__RUNS__, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'chatbot_eval_runs_' + new Date().toISOString().slice(0, 10) + '.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }});
  }})();

  (function() {{
    function escapeHtml(s) {{
      const d = document.createElement('div');
      d.textContent = s == null ? '' : String(s);
      return d.innerHTML;
    }}
    function statusFor(score) {{
      return score >= 85 ? 'good' : (score >= 70 ? 'warning' : 'critical');
    }}
    function statusVar(status) {{
      return status === 'good' ? 'var(--good)' : (status === 'warning' ? 'var(--warning)' : 'var(--bad)');
    }}
    function categoryLabel(id) {{
      const m = /^(.*)-(\\d+)$/.exec(id || '');
      return m ? (m[1] + ' v' + m[2]) : (id || '');
    }}

    let selectedRun = window.__RUNS__.length - 1;

    function renderNliRow(s) {{
      const label = s.label;
      const status = label === 'entailment' ? 'good' : (label === 'contradiction' ? 'critical' : 'warning');
      const segs = ['entailment', 'neutral', 'contradiction'].map(function(k) {{
        return '<div class="nli-seg ' + k + '" style="width:' + ((s[k] || 0) * 100).toFixed(0) + '%"></div>';
      }}).join('');
      return '<div class="nli-row">' +
        '<div class="nli-sentence">' + escapeHtml(s.sentence) + '</div>' +
        '<div class="nli-bar">' + segs + '</div>' +
        '<div class="nli-label status-' + status + '">' + label + ' ' + ((s[label] || 0) * 100).toFixed(0) + '%</div>' +
        '</div>';
    }}

    function renderQuestionDetail(q) {{
      const ev = q.evidence || {{}};
      const numeric = ev.numeric || {{}};
      const perSentence = ev.nli_per_sentence || [];
      let html = '<div class="q-detail">';
      html += '<div class="q-question">' + escapeHtml(q.question) + '</div>';
      html += '<div class="q-verdict">' + escapeHtml(q.reason) + '</div>';

      if (ev.vague_hedge) {{
        html += '<div class="vague-warning">&#9888; possible non-answer &mdash; contains hedging language ("' +
          escapeHtml(ev.vague_hedge) + '"). Severity above is unaffected; this is a separate signal worth a human look.</div>';
      }}

      if (numeric.mismatches && numeric.mismatches.length) {{
        numeric.mismatches.forEach(function(m) {{
          html += '<div class="numeric-readout">' +
            '<div class="numeric-box answer"><span class="k">answer:</span> ' + escapeHtml(m.answer_value) + ' (' + escapeHtml(m.kind) + ')</div>' +
            '<div class="numeric-box reference"><span class="k">reference:</span> ' + escapeHtml(JSON.stringify(m.reference_values)) + '</div>' +
            '</div>';
        }});
      }}

      if (perSentence.length) {{
        html += '<div class="nli-block">' + perSentence.map(renderNliRow).join('') + '</div>';
      }}

      html += '<details class="full-answers-toggle"><summary>show full answers</summary>' +
        '<p>ungrounded answer (no reference given):</p><pre>' + escapeHtml(q.ungrounded_answer) + '</pre>' +
        '<p>grounded answer (reference injected):</p><pre>' + escapeHtml(q.grounded_answer) + '</pre>' +
        '<p>reference context used:</p><pre>' + escapeHtml(q.reference_context || '(none found)') + '</pre>' +
        '</details>';

      html += '</div>';
      return html;
    }}

    function renderGroups(questions) {{
      const groups = {{ flag: [], minor: [], none: [] }};
      questions.forEach(function(q) {{
        const sev = q.severity || (q.concern ? 'flag' : 'none');
        (groups[sev] || groups.flag).push(q);
      }});

      const order = ['flag', 'minor', 'none'];
      const defaultOpen = {{ flag: false, minor: false, none: false }};
      let out = '';
      order.forEach(function(sev) {{
        const label = window.__SEVERITY_LABEL__[sev];
        const cls = window.__SEVERITY_CLASS__[sev];
        const qs = groups[sev];
        out += '<details class="sev-group status-' + cls + '"' + (defaultOpen[sev] ? ' open' : '') + '>';
        out += '<summary>' + label + ' (' + qs.length + ')</summary>';
        out += '<div class="sev-group-body">';
        if (!qs.length) {{
          out += '<div class="sev-empty">none this run</div>';
        }} else {{
          qs.forEach(function(q) {{
            const hasHedge = q.evidence && q.evidence.vague_hedge;
            out += '<details class="q-row status-' + cls + '">';
            out += '<summary><span class="qtext">' + escapeHtml(categoryLabel(q.id)) +
              (hasHedge ? ' <span class="hedge-marker" title="possible non-answer">&#9888; vague</span>' : '') + '</span>' +
              '<span class="qscores">T ' + q.truthfulness_score + ' &middot; C ' + q.tone_consistency_score + '</span></summary>';
            out += renderQuestionDetail(q);
            out += '</details>';
          }});
        }}
        out += '</div></details>';
      }});
      return out;
    }}

    window.renderRun = function(idx) {{
      selectedRun = idx;
      const run = window.__RUNS__[idx];

      document.getElementById('run-label-header').textContent =
        (run.target_model || '') + '.eval \\u2014 ' + (run.timestamp || '').slice(0, 10);
      document.getElementById('run-label-rail').textContent = 'live \\u00b7 ' + (run.timestamp || '').slice(0, 10);

      const truthStatus = statusFor(run.avg_truthfulness_score);
      document.getElementById('truth-num').textContent = run.avg_truthfulness_score;
      document.getElementById('truth-num').style.color = statusVar(truthStatus);
      document.getElementById('truth-bar').style.width = run.avg_truthfulness_score + '%';
      document.getElementById('truth-bar').style.background = statusVar(truthStatus);

      const toneStatus = statusFor(run.avg_tone_consistency_score);
      document.getElementById('tone-num').textContent = run.avg_tone_consistency_score;
      document.getElementById('tone-num').style.color = statusVar(toneStatus);
      document.getElementById('tone-bar').style.width = run.avg_tone_consistency_score + '%';
      document.getElementById('tone-bar').style.background = statusVar(toneStatus);

      document.getElementById('groups-col').innerHTML = renderGroups(run.questions);

      const svg = document.getElementById('trend-svg');
      if (svg) {{
        svg.querySelectorAll('.trend-dot').forEach(function(d) {{
          d.classList.toggle('selected', parseInt(d.getAttribute('data-run'), 10) === idx);
        }});
        const guideX = JSON.parse(svg.getAttribute('data-guide-x'));
        const guide = document.getElementById('trend-guide');
        if (guide && guideX[idx] !== undefined) {{
          guide.setAttribute('x1', guideX[idx]);
          guide.setAttribute('x2', guideX[idx]);
        }}
      }}
    }};

    window.selectRun = function(idx) {{ window.renderRun(idx); }};

    renderRun(selectedRun);
  }})();

  (function() {{
    const btn = document.getElementById('theme-toggle');
    const root = document.documentElement;
    const darkLabel = document.getElementById('theme-dark-label');
    const lightLabel = document.getElementById('theme-light-label');
    function apply(theme) {{
      root.setAttribute('data-theme', theme);
      darkLabel.classList.toggle('dim', theme !== 'dark');
      lightLabel.classList.toggle('dim', theme !== 'light');
    }}
    const saved = localStorage.getItem('theme');
    const osPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    apply(saved || (osPrefersDark ? 'dark' : 'light'));
    btn.addEventListener('click', function() {{
      const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      apply(next);
      localStorage.setItem('theme', next);
    }});
  }})();

  (function() {{
    const btn = document.getElementById('run-eval-btn');
    const errEl = document.getElementById('run-eval-error');
    const modelInput = document.getElementById('model-endpoint-input');
    btn.addEventListener('click', function() {{
      btn.disabled = true;
      btn.textContent = 'RUNNING...';
      errEl.style.display = 'none';
      fetch('/run', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ target_model: modelInput.value.trim() }}),
      }})
        .then(function(r) {{ return r.json().then(function(data) {{ return {{ ok: r.ok, data: data }}; }}); }})
        .then(function(res) {{
          if (res.ok && res.data.ok) {{
            window.location.reload();
          }} else {{
            throw new Error(res.data.error || 'run failed');
          }}
        }})
        .catch(function(e) {{
          btn.disabled = false;
          btn.textContent = 'RUN_EVAL';
          errEl.textContent = String(e.message || e);
          errEl.style.display = 'block';
        }});
    }});
  }})();

  (function() {{
    const errEl = document.getElementById('corpus-error');
    function showError(e) {{
      errEl.textContent = String((e && e.message) || e);
      errEl.style.display = 'block';
    }}
    function postJson(url, payload) {{
      return fetch(url, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload),
      }}).then(function(r) {{
        return r.json().then(function(data) {{
          if (!r.ok || !data.ok) throw new Error(data.error || 'request failed');
          return data;
        }});
      }});
    }}

    document.querySelectorAll('.corpus-delete').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        const filename = btn.getAttribute('data-filename');
        if (!window.confirm('Delete ' + filename + ' from the corpus?')) return;
        errEl.style.display = 'none';
        postJson('/corpus/delete', {{ filename: filename }})
          .then(function() {{ window.location.reload(); }})
          .catch(showError);
      }});
    }});

    const uploadBtn = document.getElementById('corpus-upload-btn');
    const fileInput = document.getElementById('corpus-file-input');
    uploadBtn.addEventListener('click', function() {{
      const file = fileInput.files[0];
      if (!file) {{ showError('choose a .md or .txt file first'); return; }}
      errEl.style.display = 'none';
      uploadBtn.disabled = true;
      uploadBtn.textContent = 'uploading...';
      const reader = new FileReader();
      reader.onload = function() {{
        postJson('/corpus/upload', {{ filename: file.name, content: reader.result }})
          .then(function() {{ window.location.reload(); }})
          .catch(function(e) {{
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'add file';
            showError(e);
          }});
      }};
      reader.onerror = function() {{
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'add file';
        showError('could not read file');
      }};
      reader.readAsText(file);
    }});
  }})();
</script>
</div>
</body>
</html>
"""


def write_report(results: dict, path: str) -> None:
    with open(path, "w") as f:
        f.write(render_html(results))
