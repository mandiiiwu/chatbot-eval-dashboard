"""Renders a results dict (see evaluator.run_evaluation) to a single
self-contained HTML dashboard. No JS framework, no build step—just open it
in a browser, or serve it live via harness/server.py.

v2 layout (2026-08-13), redesigned from a Claude Design handoff package,
superseding the earlier hero-concern-percentage version: a left rail (run
config + metadata), Truthfulness/Tone Consistency score panels, a 2-series
trend chart where clicking any point swaps the ENTIRE dashboard to that run's
data (client-side; every comparable run's full per-question data is
embedded in the page), and questions triaged into Flagged/Minor/Ok groups
instead of one flat list. The concern-percentage hero figure is intentionally
dropped per the user's design decision, not an oversight.

Score panels, run-date labels, and the flagged/minor/ok groups are rendered
entirely client-side from window.__RUNS__ (one shared renderRun() function
handles both the initial page load and every run-selection click), so there's
no duplicate rendering logic between Python and JS for that reactive part.
Only the trend chart's static SVG geometry and the left rail's config/
metadata (which don't change per selected run; they describe the current
setup, not the run being viewed) are server-rendered.

Added 2026-08-16: render_onboarding_html(), a second entry point shown by
harness/server.py before any run has ever completed (no results/latest.json
yet). Previously that state was a bare "No eval runs yet" message with no
interactive controls at all -- a real gap, since the dashboard's own
corpus-upload/model-endpoint/RUN_EVAL controls (the ones a fresh user would
naturally reach for) were unreachable until a run had already happened some
other way (the CLI, per README_example.md's documented setup flow). The
[CONFIG] rail, its CSS, and the JS that drives it are now shared between
render_html() and render_onboarding_html() via the _config_rail_html()/
_STYLE_BLOCK/_CONFIG_RAIL_JS module-level pieces below, so a user can
genuinely choose either path -- configure everything through the UI, or
drop files into corpus/questions/ on disk and run python run_eval.py
directly -- without the two ever drifting out of sync with each other.
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
# short-circuit); that's not how fact_check.py actually works. Both the
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
    "no reference corpus involved—compares the model's own answers to each other",
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
    "not 100% reliable; NLI can misread hedging as contradiction",
    "doesn't penalize incomplete answers—only checks whether what's present "
    "contradicts the reference, not whether the answer covers everything asked",
]


def _corpus_files() -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(config.CORPUS_DIR, "*.md")))


def _questions_files() -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(config.QUESTIONS_DIR, "*.json")))


def _trend_chart_svg(runs: list[dict]) -> str:
    """Static SVG geometry for all comparable runs—2 series (truthfulness,
    tone consistency). Points carry data-run attributes; JS toggles a
    'selected' class and moves the guide line on click, it doesn't redraw
    the chart."""
    if len(runs) < 2:
        return (
            f'<p class="trend-note">Only {len(runs)} run{"s" if len(runs) != 1 else ""} so far on the '
            "current architecture&mdash;need at least 2 to show a trend. Runs from earlier "
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

    def _col_bounds(i: int) -> tuple[float, float]:
        # Each point's click target spans the midpoint to its neighbors
        # (Voronoi-style), clamped to the plot edges at the ends -- gapless
        # and non-overlapping for any n. The previous version approximated
        # this with a fixed per-column width derived from the *inter-point*
        # spacing (w - pad_l - pad_r) / (n - 1), while separately centering
        # it using a *different* divisor, (w / n) / 2 -- fine-ish for many
        # points, but at n=2 there's only 1 gap, so width blew up to the
        # entire plot width (850 of 900 units) and the offset placed its
        # left edge at x=-195, well past the SVG's own left edge. Since the
        # SVG doesn't clip overflow, that oversized rect physically covered
        # part of the RUN_EVAL button in the sidebar next to it, silently
        # stealing clicks meant for the button (reported 2026-08-17).
        left = pad_l if i == 0 else (x_of(i - 1) + x_of(i)) / 2
        right = (w - pad_r) if i == n - 1 else (x_of(i) + x_of(i + 1)) / 2
        return left, right

    hit_columns = "".join(
        f'<rect x="{_col_bounds(i)[0]:.1f}" y="{pad_t}" width="{_col_bounds(i)[1] - _col_bounds(i)[0]:.1f}" '
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


# --- Shared, results-independent page pieces -------------------------------
# Everything below is used by BOTH render_html() (a real run is loaded) and
# render_onboarding_html() (no run has ever completed yet) -- extracted so
# the [CONFIG] rail's markup, styling, and behavior can never drift between
# the two pages. None of this interpolates a real `results` dict.

_STYLE_BLOCK = """<style>
  /* Self-hosted (see harness/static/fonts/, served by server.py's
     /static/fonts/ route) instead of Google Fonts. The RUN_EVAL "dead
     button" saga (see PLAN.md) traced all the way back to this font:
     fonts.googleapis.com/fonts.gstatic.com were being silently blackholed
     by something on the user's network (router/DNS-level filtering, not a
     clean failure), which stalled the page even with the earlier
     media="print" non-blocking-load trick in place. Self-hosting removes
     the external network dependency entirely -- confirmed fixed via a
     controlled /etc/hosts test blocking those two domains while the rest
     of the network stayed up. font-display: swap kept so a slow/failed
     load (now impossible for this specific cause, but cheap insurance)
     still only ever costs typography, never functionality. */
  @font-face {
    font-family: 'IBM Plex Mono';
    font-style: normal;
    font-weight: 400;
    font-display: swap;
    src: url('/static/fonts/ibm-plex-mono-400.woff2') format('woff2');
  }
  @font-face {
    font-family: 'IBM Plex Mono';
    font-style: normal;
    font-weight: 500;
    font-display: swap;
    src: url('/static/fonts/ibm-plex-mono-500.woff2') format('woff2');
  }
  @font-face {
    font-family: 'IBM Plex Mono';
    font-style: normal;
    font-weight: 600;
    font-display: swap;
    src: url('/static/fonts/ibm-plex-mono-600.woff2') format('woff2');
  }
  @font-face {
    font-family: 'IBM Plex Mono';
    font-style: normal;
    font-weight: 700;
    font-display: swap;
    src: url('/static/fonts/ibm-plex-mono-700.woff2') format('woff2');
  }
  :root {
    --good: #1f8a4c; --warning: #c8860d; --bad: #c94a3f;
    --good-bg: rgba(31,138,76,0.12); --warning-bg: rgba(200,134,13,0.12); --bad-bg: rgba(201,74,63,0.12);
  }
  .viz-root {
    color-scheme: dark;
    --bg:     #151715;
    --border: #2c302d;
    --track:  #20221f;
    --text:   #c9d1c9;
    --muted:  #7c877c;
    --faint:  #565c56;
    --trend-truth: #c9d1c9;
    --trend-tone:  #4a5c4a;
  }
  :root[data-theme="light"] .viz-root {
    color-scheme: light;
    --bg:     #e9e4d8;
    --border: #cdc7b8;
    --track:  #ddd7c8;
    --text:   #26292b;
    --muted:  #63665f;
    --faint:  #8b8d85;
    --trend-truth: #26292b;
    --trend-tone:  #9a998f;
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; }
  .viz-root {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    background: var(--bg); color: var(--text);
    min-height: 100vh;
  }
  .layout { display: flex; align-items: stretch; min-height: 100vh; }

  /* --- left rail --- */
  .rail { width: 300px; flex-shrink: 0; border-right: 1px solid var(--border);
          padding: 18px 16px; display: flex; flex-direction: column; gap: 14px; }
  .live-line { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); }
  .live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--good); flex-shrink: 0; }
  .section-label { font-size: 12px; color: var(--muted); font-weight: 600; }
  .rail-field { display: flex; flex-direction: column; gap: 4px; }
  .rail-field-label { font-size: 10px; color: var(--muted); }
  .field-help { position: relative; display: inline-flex; align-items: center; justify-content: center;
                margin-left: 4px; cursor: help; color: var(--muted); border: 1px solid var(--border);
                border-radius: 50%; width: 12px; height: 12px; font-size: 9px; line-height: 1; }
  .field-help-popup { display: none; position: absolute; top: 100%; left: 0; margin-top: 6px; width: 190px;
                       background: var(--bg); border: 1px solid var(--border); box-shadow: 0 4px 16px rgba(0,0,0,0.3);
                       padding: 8px 10px; font-size: 10px; color: var(--muted); line-height: 1.5; z-index: 10;
                       text-transform: none; }
  .field-help-popup.popup-above { top: auto; bottom: 100%; margin-top: 0; margin-bottom: 6px; }
  .field-help-popup.popup-wide { width: 250px; }
  .field-help-popup ul { margin: 0; padding-left: 14px; }
  .field-help-popup li { margin-top: 5px; }
  .field-help-popup li:first-child { margin-top: 0; }
  .field-help:hover .field-help-popup { display: block; }
  .rail-box { border: 1px solid var(--border); padding: 8px 10px; font-size: 11px; word-break: break-word; }
  .rail-box.dashed { border-style: dashed; color: var(--muted); }
  .rail-input { border: 1px solid var(--border); background: var(--bg); color: var(--text);
                padding: 8px 10px; font: 11px 'IBM Plex Mono', ui-monospace, monospace; width: 100%; }
  .corpus-list { display: flex; flex-direction: column; gap: 4px; }
  .corpus-item { display: flex; align-items: center; justify-content: space-between; gap: 6px;
                  border: 1px solid var(--border); padding: 5px 8px; font-size: 10px; }
  .corpus-fname { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .corpus-delete { border: none; background: transparent; color: var(--bad); cursor: pointer;
                    font-size: 13px; line-height: 1; flex-shrink: 0; padding: 0 2px; }
  .corpus-upload { display: flex; gap: 6px; align-items: center; margin-top: 2px; }
  .corpus-upload input[type="file"] { flex: 1; min-width: 0; font-size: 9px; color: var(--muted); }
  .corpus-upload input[type="text"] { flex: 1; min-width: 0; border: 1px solid var(--border);
                background: var(--bg); color: var(--text); padding: 5px 8px; font: 10px 'IBM Plex Mono', ui-monospace, monospace; }
  .corpus-upload button { border: 1px solid var(--border); background: transparent; color: var(--text);
                           font: 10px 'IBM Plex Mono', ui-monospace, monospace; padding: 5px 8px; cursor: pointer; flex-shrink: 0; }
  .run-eval-btn { margin-top: 2px; padding: 5px 8px; border: 1px solid var(--text); background: var(--text);
                   color: var(--bg); font: 600 11px 'IBM Plex Mono', ui-monospace, monospace; cursor: pointer; }
  .run-eval-btn:disabled { opacity: 0.5; cursor: default; }
  .run-eval-progress-track { height: 6px; background: var(--track); overflow: hidden; margin-top: 4px; }
  .run-eval-progress-fill { height: 100%; background: var(--text); width: 0%; transition: width 0.3s ease; }
  .run-eval-progress-label { font-size: 9px; color: var(--muted); margin-top: 2px; text-align: center; }
  .run-eval-info { display: none; align-items: center; justify-content: space-between; gap: 6px;
                    font-size: 10px; color: var(--muted); border: 1px dashed var(--border);
                    padding: 6px 8px; margin-top: 4px; }
  .run-eval-info-dismiss { border: none; background: transparent; color: var(--muted); cursor: pointer;
                            font-size: 12px; line-height: 1; flex-shrink: 0; padding: 0 2px; }
  .run-eval-error { font-size: 10px; color: var(--bad); display: none; }
  .cost-confirm-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
                           align-items: center; justify-content: center; z-index: 100; }
  .cost-confirm-box { background: var(--bg); border: 1px solid var(--border); padding: 18px;
                       width: 260px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
  .cost-confirm-message { font-size: 12px; color: var(--text); line-height: 1.6; margin-top: 10px; }
  .cost-confirm-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }
  .cost-confirm-btn { border: 1px solid var(--border); background: transparent; color: var(--text);
                       font: 11px 'IBM Plex Mono', ui-monospace, monospace; padding: 8px 14px; cursor: pointer; }
  .cost-confirm-btn-primary { background: var(--text); color: var(--bg); border-color: var(--text); }
  .rail-divider { border: none; border-top: 1px solid var(--border); margin: 0; }
  .meta-row { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); margin-top: 6px; }
  .meta-value { color: var(--text); }

  /* --- main content --- */
  .main { flex: 1; min-width: 0; padding: 16px 24px 40px; }
  .header-strip { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
  #run-label-header { font-size: 15px; color: var(--muted); }
  #theme-toggle { display: flex; gap: 6px; border: 1px solid var(--border); background: transparent;
                   color: var(--text); cursor: pointer; font: 11px 'IBM Plex Mono', ui-monospace, monospace; padding: 5px 10px; }
  #theme-toggle .dim { opacity: 0.35; }

  .panels-row { display: flex; gap: 16px; margin-bottom: 18px; }
  .score-panel { flex: 1; border: 1px solid var(--border); padding: 32px; display: flex;
                 flex-direction: column; gap: 16px; position: relative; }
  .score-figure { display: flex; align-items: baseline; gap: 8px; }
  .score-figure .num { font-size: 38px; font-weight: 600; }
  .score-figure .of100 { font-size: 12px; color: var(--muted); }
  .score-bar-track { height: 6px; background: var(--track); overflow: hidden; }
  .score-bar-fill { height: 100%; }
  .meaning-trigger { font-size: 10px; color: var(--muted); text-decoration: underline dotted; cursor: help;
                      width: fit-content; }
  .meaning-popup { display: none; position: absolute; left: 0; right: 0; top: 100%; margin-top: 6px; background: var(--bg);
                    border: 1px solid var(--border); box-shadow: 0 4px 16px rgba(0,0,0,0.3); padding: 12px 14px;
                    font-size: 10px; color: var(--muted); flex-direction: column; gap: 4px; z-index: 5; }
  .meaning-popup span:before { content: "\\2022  "; color: var(--muted); }
  .score-panel:hover .meaning-popup { display: flex; }

  section { margin-bottom: 20px; }
  .trend-box { border: 1px solid var(--border); padding: 32px; }
  .trend-note { color: var(--muted); font-size: 0.85rem; margin: 0; }
  .trend-svg { width: 100%; height: auto; overflow: visible; cursor: pointer; }
  .gridline { stroke: var(--border); stroke-width: 1; }
  .axis-line { stroke: var(--border); stroke-width: 1; }
  .axis-label { font-size: 9px; fill: var(--muted); font-family: 'IBM Plex Mono', ui-monospace, monospace; }
  .trend-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .trend-line.trend-truth { stroke: var(--trend-truth); }
  .trend-line.trend-tone { stroke: var(--trend-tone); }
  .trend-dot { stroke: var(--bg); stroke-width: 2; cursor: pointer; }
  .trend-dot.trend-truth { fill: var(--trend-truth); }
  .trend-dot.trend-tone { fill: var(--trend-tone); }
  .trend-dot.selected { r: 6; }
  .hit-col { fill: transparent; cursor: pointer; }
  .trend-guide { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; pointer-events: none; }
  .legend { display: flex; gap: 16px; font-size: 0.75rem; color: var(--muted); margin-top: 10px; }
  .legend-item { display: flex; align-items: center; gap: 5px; }
  .legend-swatch { width: 10px; height: 2px; display: inline-block; }
  .legend-swatch.trend-truth { background: var(--trend-truth); }
  .legend-swatch.trend-tone { background: var(--trend-tone); }
  .table-toggle { margin-top: 10px; font-size: 0.78rem; }
  .table-toggle summary { cursor: pointer; color: var(--muted); }
  .trend-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.76rem; }
  .trend-table th, .trend-table td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--border); }
  .trend-table th { color: var(--muted); font-weight: 500; }

  .groups-row { display: flex; gap: 16px; align-items: flex-start; }
  .groups-col { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 14px; }
  .sev-group summary { cursor: pointer; font-size: 24px; font-weight: 700; list-style: none; margin-bottom: 4px; }
  .sev-group summary::-webkit-details-marker { display: none; }
  .sev-group summary:before { content: "\\25b8  "; color: var(--muted); }
  .sev-group[open] summary:before { content: "\\25be  "; }
  .sev-group.status-critical summary { color: var(--bad); }
  .sev-group.status-warning summary { color: var(--warning); }
  .sev-group.status-good summary { color: var(--good); }
  .sev-group-body { margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }
  .sev-empty { font-size: 11px; color: var(--muted); }

  .q-row { border: 1px solid var(--border); padding: 10px 12px; font-size: 13px; }
  .q-row.status-critical { border-color: var(--bad); }
  .q-row.status-warning { border-color: var(--warning); }
  .q-row.status-good { border-color: var(--good); }
  .q-row summary { cursor: pointer; display: flex; justify-content: space-between; gap: 10px;
                    list-style: none; font-size: 13px; font-weight: 400; }
  .q-row summary::-webkit-details-marker { display: none; }
  .q-row .qtext { flex: 1; }
  .q-row .qscores { color: var(--muted); flex-shrink: 0; }
  .q-detail { margin-top: 8px; display: flex; flex-direction: column; gap: 8px; }
  .q-question { color: var(--text); font-weight: 600; }
  .q-verdict { color: var(--muted); }
  .vague-warning { color: var(--warning); border: 1px dashed var(--warning); padding: 6px 8px; font-size: 12px; }
  .hedge-marker { color: var(--warning); font-size: 11px; font-weight: 600; }
  .numeric-readout { display: flex; gap: 8px; }
  .numeric-box { flex: 1; border: 1px solid var(--border); padding: 6px 8px; font-size: 12px; }
  .numeric-box .k { color: var(--muted); }
  .numeric-box.answer { border-color: var(--bad); color: var(--bad); }
  .numeric-box.reference { color: var(--muted); }
  .nli-block { display: flex; flex-direction: column; gap: 4px; }
  .nli-row { display: flex; align-items: center; gap: 8px; }
  .nli-sentence { flex: 1; font-size: 12px; color: var(--muted); }
  .nli-bar { display: flex; width: 80px; height: 6px; flex-shrink: 0; }
  .nli-seg.entailment { background: var(--good); }
  .nli-seg.neutral { background: var(--warning); }
  .nli-seg.contradiction { background: var(--bad); }
  .nli-label { font-size: 12px; min-width: 90px; text-align: right; flex-shrink: 0; }
  .nli-label.status-good { color: var(--good); }
  .nli-label.status-warning { color: var(--warning); }
  .nli-label.status-critical { color: var(--bad); }
  .full-answers-toggle summary { cursor: pointer; font-size: 10px; color: var(--muted); text-decoration: underline; }
  .full-answers-toggle pre { background: var(--track); padding: 8px; white-space: pre-wrap; font-size: 10px;
                              margin: 6px 0 0; }
  .full-answers-toggle p { margin: 8px 0 2px; font-size: 10px; color: var(--muted); }

  .key-panel { width: 230px; flex-shrink: 0; border: 1px solid var(--border); padding: 14px;
               display: flex; flex-direction: column; gap: 10px; }
  .key-item { display: flex; flex-direction: column; gap: 2px; }
  .key-label { font-size: 11px; font-weight: 600; text-transform: uppercase; }
  .key-label.status-critical { color: var(--bad); }
  .key-label.status-warning { color: var(--warning); }
  .key-label.status-good { color: var(--good); }
  .key-item ul { margin: 2px 0 0; padding-left: 14px; font-size: 10px; color: var(--muted); }
  .key-caveat { font-size: 9px; color: var(--faint); margin: 4px 0 0; }

  .onboarding-note { color: var(--muted); font-size: 13px; line-height: 1.8; max-width: 620px; }
  .onboarding-note p { margin: 0 0 14px; }
  .onboarding-note code { background: var(--track); color: var(--text); padding: 1px 5px; }
  .onboarding-note ul { margin: 0 0 14px; padding-left: 18px; }
  .onboarding-note li { margin-top: 6px; }

  @media (max-width: 720px) {
    .layout { flex-direction: column; }
    .rail { width: auto; border-right: none; border-bottom: 1px solid var(--border); }
    .panels-row, .groups-row { flex-direction: column; }
    .key-panel { width: auto; }
  }
</style>"""

_COST_CONFIRM_MODAL_HTML = """
    <div id="cost-confirm-overlay" class="cost-confirm-overlay">
      <div class="cost-confirm-box">
        <span class="section-label">[COST ESTIMATE]</span>
        <div id="cost-confirm-message" class="cost-confirm-message"></div>
        <div class="cost-confirm-actions">
          <button type="button" id="cost-confirm-cancel" class="cost-confirm-btn">CANCEL</button>
          <button type="button" id="cost-confirm-ok" class="cost-confirm-btn cost-confirm-btn-primary">OK, RUN</button>
        </div>
      </div>
    </div>
"""

_THEME_TOGGLE_BTN_HTML = (
    '<button id="theme-toggle" type="button"><span class="dim" id="theme-dark-label">[DARK]</span>'
    '<span id="theme-light-label">[LIGHT]</span></button>'
)

_IMPORT_SECTION_HTML = """
    <hr class="rail-divider">

    <div>
      <span class="section-label">[IMPORT]<span class="field-help">?<span class="field-help-popup popup-above">Loads a results.json produced elsewhere (another machine, a colleague, another tool) into this machine's results/&mdash;strictly validated against the current schema version first. Rejected outright if the shape doesn't match; nothing is coerced or guessed.</span></span></span>
      <div class="corpus-upload" style="margin-top:10px;">
        <input id="import-json-file-input" type="file" accept=".json">
        <button id="import-json-btn" type="button">import</button>
      </div>
      <div id="import-json-error" class="run-eval-error"></div>
    </div>
"""


def _config_rail_html(
    target_model_value: str,
    corpus_files: list[str],
    corpus_summary: str,
    questions_files: list[str],
    current_questions_file: str | None,
) -> str:
    """The [CONFIG] rail's body -- target provider, model endpoint, corpus
    list/upload, questions selector, RUN_EVAL button + progress bar. Shared
    between render_html() (target_model_value/current_questions_file come
    from the currently-loaded run) and render_onboarding_html()
    (target_model_value falls back to config.TARGET_MODEL,
    current_questions_file is always None, since there's no run yet) --
    see this module's docstring for why sharing this matters."""
    corpus_list_html = "".join(
        f'<div class="corpus-item"><span class="corpus-fname" title="{html.escape(f)}">{html.escape(f)}</span>'
        f'<button type="button" class="corpus-delete" data-filename="{html.escape(f)}" title="delete">&times;</button></div>'
        for f in corpus_files
    )
    questions_options_html = (
        "".join(
            f'<option value="{html.escape(f)}"{" selected" if f == current_questions_file else ""}>'
            f'{html.escape(f)}</option>'
            for f in questions_files
        )
        if questions_files
        else '<option value="" disabled selected>no questions/*.json found</option>'
    )
    return f"""
        <div class="rail-field">
          <span class="rail-field-label">target provider<span class="field-help">?<span class="field-help-popup">"ollama" talks to a local Ollama model; "custom" lets you point at any REST/JSON chatbot API by describing its request/response shape. Choose one to see the rest of the setup. Changing this only affects your next run; it doesn't edit .env.</span></span></span>
          <select id="target-provider-select" class="rail-input">
            <option value="" disabled selected>choose a provider</option>
            <option value="ollama">ollama (local)</option>
            <option value="custom">custom HTTP endpoint</option>
          </select>
        </div>
        <div id="config-fields-below" style="display:none;flex-direction:column;gap:10px;">
          <div class="rail-field">
            <span class="rail-field-label">model endpoint<span class="field-help">?<span class="field-help-popup">Must match an Ollama-pulled model tag exactly; run <code>ollama list</code> in your terminal to see what's available locally. Needs to be reachable via the OpenAI-compatible endpoint at OLLAMA_BASE_URL. Changing this only affects your next run; it doesn't edit .env.</span></span></span>
            <input id="model-endpoint-input" class="rail-input" type="text" value="{html.escape(target_model_value)}" spellcheck="false">
          </div>
          <div id="custom-endpoint-fields" class="rail-field" style="display:none;flex-direction:column;gap:10px;">
            <div class="rail-field">
              <span class="rail-field-label">endpoint url</span>
              <input id="endpoint-url-input" class="rail-input" type="text" value="{html.escape(config.CUSTOM_ENDPOINT_URL)}" placeholder="https://..." spellcheck="false">
            </div>
            <div class="rail-field">
              <span class="rail-field-label">endpoint headers<span class="field-help">?<span class="field-help-popup">JSON object, e.g. {{"Authorization": "Bearer ..."}}. Never pre-filled or saved to results/; leave blank to use the value already in .env.</span></span></span>
              <input id="endpoint-headers-input" class="rail-input" type="password" placeholder="leave blank for .env value" spellcheck="false" autocomplete="off">
            </div>
            <div class="rail-field">
              <span class="rail-field-label">request template<span class="field-help">?<span class="field-help-popup">JSON string describing the request body your endpoint expects. {{{{model}}}}/{{{{system}}}}/{{{{message}}}} get substituted in before sending.</span></span></span>
              <textarea id="request-template-input" class="rail-input" rows="3" spellcheck="false">{html.escape(config.CUSTOM_REQUEST_TEMPLATE)}</textarea>
            </div>
            <div class="rail-field">
              <span class="rail-field-label">response path<span class="field-help">?<span class="field-help-popup">Dot-notation path to the answer text in the JSON response, e.g. choices.0.message.content</span></span></span>
              <input id="response-path-input" class="rail-input" type="text" value="{html.escape(config.CUSTOM_RESPONSE_PATH)}" spellcheck="false">
            </div>
          </div>
          <hr class="rail-divider">
          <div class="rail-field">
            <span class="rail-field-label">corpus <span id="corpus-count">({corpus_summary})</span><span class="field-help">?<span class="field-help-popup popup-wide"><ul>
              <li>works best on real, verbatim reference text; not AI-generated and paraphrased references</li>
              <li>Direct upload: .md, .txt, .pdf (text auto-extracted; scanned-image PDFs aren't supported)</li>
              <li>.json works too, but only a few shapes: a flat list of strings, a list of {{"text": ...}} objects, or SQuAD-style (data/paragraphs/context, like CUAD). Anything else gets rejected instead of guessed at</li>
              <li>.zip: any mix of the above; other files inside are skipped</li>
              <li>URL fetch also works below&mdash;http/https only, 20MB max, .zip URLs included</li>
              <li>Or skip the UI: drop files into ~/corpus/ on disk, then rerun</li>
              <li>Include citation/license info if it isn't your own writing</li>
              <li>One coherent topic per file works best for retrieval</li>
            </ul></span></span></span>
            <div id="corpus-list" class="corpus-list">
              {corpus_list_html}
            </div>
            <div class="corpus-upload">
              <input id="corpus-file-input" type="file" accept=".md,.txt,.pdf,.zip">
              <button id="corpus-upload-btn" type="button">add file</button>
            </div>
            <div class="corpus-upload" style="margin-top:6px;">
              <input id="corpus-url-input" type="text" placeholder="or paste a direct file URL..." spellcheck="false">
              <button id="corpus-url-btn" type="button">fetch</button>
            </div>
            <div id="corpus-error" class="run-eval-error"></div>
          </div>
          <div class="rail-field">
            <span class="rail-field-label">questions<span class="field-help">?<span class="field-help-popup">Which questions/*.json file to run against. Checked automatically before every run&mdash;a free keyword pre-check, then a full embedding-based check&mdash;against whatever corpus is currently attached; a mismatched pairing (e.g. leftover questions after swapping corpus) is refused rather than run. Nothing listed here yet? Just click RUN EVAL&mdash;it generates a question set from your attached corpus automatically before running (same as <code>python generate_questions.py</code>, just triggered for you). Changing this only affects your next run; it doesn't edit .env.</span></span></span>
            <select id="questions-file-select" class="rail-input">
              {questions_options_html}
            </select>
          </div>
          <button id="run-eval-btn" class="run-eval-btn" type="button">RUN EVAL</button>
          <div id="run-eval-progress" class="run-eval-progress-track" style="display:none;">
            <div id="run-eval-progress-fill" class="run-eval-progress-fill"></div>
          </div>
          <div id="run-eval-progress-label" class="run-eval-progress-label" style="display:none;"></div>
          <div id="run-eval-info" class="run-eval-info">
            <span>This runs in the background&mdash;feel free to close this tab or step away, no need to watch it.</span>
            <button type="button" id="run-eval-info-dismiss" class="run-eval-info-dismiss" title="dismiss">&times;</button>
          </div>
          <div id="run-eval-error" class="run-eval-error"></div>
        </div>
    """


# JS driving [CONFIG]'s provider-select gating, RUN_EVAL (+ cost-estimate
# confirmation + live progress polling), and corpus upload/delete. Entirely
# independent of any loaded run (no window.__RUNS__/results reference
# anywhere in here), so it's shared verbatim between render_html() and
# render_onboarding_html() -- one script, can't drift between the two pages.
_CONFIG_RAIL_JS = """
  (function() {
    const providerSelect = document.getElementById('target-provider-select');
    const belowFields = document.getElementById('config-fields-below');
    const customFields = document.getElementById('custom-endpoint-fields');
    providerSelect.addEventListener('change', function() {
      const chosen = providerSelect.value;
      belowFields.style.display = chosen ? 'flex' : 'none';
      customFields.style.display = chosen === 'custom' ? 'flex' : 'none';
    });
  })();

  (function() {
    const btn = document.getElementById('run-eval-btn');
    const errEl = document.getElementById('run-eval-error');
    const modelInput = document.getElementById('model-endpoint-input');
    const providerSelect = document.getElementById('target-provider-select');
    const questionsSelect = document.getElementById('questions-file-select');
    const progressTrack = document.getElementById('run-eval-progress');
    const progressFill = document.getElementById('run-eval-progress-fill');
    const progressLabel = document.getElementById('run-eval-progress-label');
    const infoBox = document.getElementById('run-eval-info');
    document.getElementById('run-eval-info-dismiss').addEventListener('click', function() {
      infoBox.style.display = 'none';
    });

    // Polls /run/progress while /run is in flight -- without this, the
    // button just says "RUNNING..." for however long a real model takes
    // across every question (each one is 2 model calls), which is
    // indistinguishable from "hung" with no way to tell from the UI alone.
    // The eval itself runs server-side regardless of this tab, so the info
    // box's claim ("close the tab, it keeps going") is literally true --
    // the request handler runs to completion and saves results even if no
    // client is left listening for the response.
    let pollTimer = null;
    function startPolling() {
      progressTrack.style.display = 'block';
      progressLabel.style.display = 'block';
      infoBox.style.display = 'flex';
      progressFill.style.width = '0%';
      progressLabel.textContent = 'starting...';
      pollTimer = setInterval(function() {
        fetch('/run/progress')
          .then(function(r) { return r.json(); })
          .then(function(p) {
            // "embedding" fires while a fresh/changed corpus is being
            // embedded (can be the slowest part of a run for a large
            // corpus, and used to be entirely invisible -- the button just
            // said "RUNNING..." with zero feedback for however long that
            // took, indistinguishable from a hang). "generating_questions"
            // fires when RUN_EVAL had no questions file and is building one
            // from the corpus automatically (see auto_generate_questions) --
            // this makes real MicroDC round-trips per candidate question, so
            // it needs the same visibility. "questions" is the original
            // per-question eval-loop progress. Anything else (empty phase,
            // or total still 0 because the first unit of work hasn't
            // completed yet) falls back to "starting...".
            if (p.phase === 'embedding' && p.total > 0) {
              const pct = Math.round(100 * p.current / p.total);
              progressFill.style.width = pct + '%';
              progressLabel.textContent = 'embedding corpus... ' + p.current + ' / ' + p.total + ' batches';
              btn.textContent = 'EMBEDDING CORPUS... ' + p.current + '/' + p.total;
            } else if (p.phase === 'generating_questions' && p.total > 0) {
              const pct = Math.round(100 * p.current / p.total);
              progressFill.style.width = pct + '%';
              progressLabel.textContent = 'generating questions... ' + p.current + ' / ' + p.total;
              btn.textContent = 'GENERATING QUESTIONS... ' + p.current + '/' + p.total;
            } else if (p.phase === 'questions' && p.total > 0) {
              const pct = Math.round(100 * p.current / p.total);
              progressFill.style.width = pct + '%';
              progressLabel.textContent = p.current + ' / ' + p.total + ' questions';
              btn.textContent = 'RUNNING... ' + p.current + '/' + p.total;
            } else {
              progressLabel.textContent = 'starting...';
            }
          })
          .catch(function() {});  // a missed poll tick isn't worth surfacing as an error
      }, 1000);
    }
    function stopPolling() {
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      progressTrack.style.display = 'none';
      progressLabel.style.display = 'none';
      infoBox.style.display = 'none';
    }

    function startRun() {
      btn.disabled = true;
      btn.textContent = 'RUNNING...';
      errEl.style.display = 'none';
      startPolling();
      const requestBody = { target_model: modelInput.value.trim() };
      if (questionsSelect && questionsSelect.value) {
        requestBody.questions_file = questionsSelect.value;
      }
      if (providerSelect.value === 'custom') {
        const headersVal = document.getElementById('endpoint-headers-input').value.trim();
        requestBody.endpoint_config = {
          provider: 'custom',
          endpoint_url: document.getElementById('endpoint-url-input').value.trim(),
          endpoint_headers: headersVal || null,
          request_template: document.getElementById('request-template-input').value.trim(),
          response_path: document.getElementById('response-path-input').value.trim(),
        };
      } else {
        requestBody.endpoint_config = { provider: 'ollama' };
      }
      fetch('/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })
        .then(function(r) { return r.json().then(function(data) { return { ok: r.ok, data: data }; }); })
        .then(function(res) {
          if (res.ok && res.data.ok) {
            stopPolling();
            window.location.reload();
          } else {
            throw new Error(res.data.error || 'run failed');
          }
        })
        .catch(function(e) {
          stopPolling();
          btn.disabled = false;
          btn.textContent = 'RUN EVAL';
          errEl.textContent = String(e.message || e);
          errEl.style.display = 'block';
        });
    }

    // In-page modal instead of window.confirm() -- a native browser dialog
    // reads as separate from the dashboard itself; this one matches the
    // rest of the UI (same rail styling, dark/light theme included, since
    // it's just var(--bg)/var(--text) like everything else here).
    const confirmOverlay = document.getElementById('cost-confirm-overlay');
    const confirmMessage = document.getElementById('cost-confirm-message');
    const confirmOkBtn = document.getElementById('cost-confirm-ok');
    const confirmCancelBtn = document.getElementById('cost-confirm-cancel');
    function showCostConfirm(message) {
      confirmMessage.textContent = message;
      confirmOverlay.style.display = 'flex';
    }
    function hideCostConfirm() {
      confirmOverlay.style.display = 'none';
    }
    // Cancelling (either button) reverts the button back to its normal
    // clickable state -- OK doesn't need this, startRun() immediately
    // overwrites it to "RUNNING..." anyway.
    function cancelCostConfirm() {
      hideCostConfirm();
      btn.disabled = false;
      btn.textContent = 'RUN EVAL';
    }
    confirmCancelBtn.addEventListener('click', cancelCostConfirm);
    confirmOverlay.addEventListener('click', function(ev) {
      if (ev.target === confirmOverlay) cancelCostConfirm();  // click on the dimmed backdrop also cancels
    });
    confirmOkBtn.addEventListener('click', function() {
      hideCostConfirm();
      startRun();
    });

    // Cost estimate is fetched fresh on every click rather than once on
    // page load -- corpus/questions can change between page loads (or
    // even between an earlier cancelled attempt and now), and the estimate
    // needs to reflect what would actually run this time, not a stale
    // snapshot from whenever the page first rendered. The button shows
    // "processing..." for this brief gap so the click visibly registered
    // before the modal appears, rather than looking unresponsive.
    btn.addEventListener('click', function() {
      errEl.style.display = 'none';
      btn.disabled = true;
      btn.textContent = 'processing...';
      const questionsQs = (questionsSelect && questionsSelect.value)
        ? ('?questions_file=' + encodeURIComponent(questionsSelect.value))
        : '';
      // Bounded with an explicit timeout: this endpoint only ever reads
      // local corpus/question counts (no MicroDC calls), so it should
      // always be fast -- a hang here has no legitimate reason to happen,
      // whatever its actual cause turns out to be in a given browser/
      // network environment. Without this, a stall at the network layer
      // left the button stuck at "processing..." with zero feedback and
      // no way to recover short of reloading the page. Now it fails loud
      // and recoverable instead of silently.
      const controller = new AbortController();
      const timeoutId = setTimeout(function() { controller.abort(); }, 15000);
      fetch('/run/cost_estimate' + questionsQs, { signal: controller.signal })
        .then(function(r) { return r.json(); })
        .then(function(est) {
          clearTimeout(timeoutId);
          const range = est.ok ? ('~$' + est.low.toFixed(2) + '-$' + est.high.toFixed(2)) : 'an unknown amount';
          // No questions file exists yet -- RUN_EVAL will generate one from
          // the attached corpus first (see evaluator.run_and_save()'s
          // auto_generate_questions), which makes its own MicroDC chat
          // calls with no calibrated cost figure to fold into the range
          // above, so this is flagged as a separate, unestimated addition
          // rather than silently baked into (or silently left out of) the
          // number shown.
          const genNote = est.ok && est.will_generate_questions
            ? ' This also includes auto-generating a question set from your corpus first (separate MicroDC cost, not included above, typically small).'
            : '';
          showCostConfirm('This run will cost about ' + range + ' in MicroDC.ai credits.' + genNote + ' OK to proceed?');
        })
        .catch(function() {
          clearTimeout(timeoutId);
          showCostConfirm('Could not estimate cost ahead of time (request timed out or failed). Run anyway?');
        });
    });
  })();

  (function() {
    const errEl = document.getElementById('corpus-error');
    const listEl = document.getElementById('corpus-list');
    const countEl = document.getElementById('corpus-count');
    function showError(e) {
      errEl.textContent = String((e && e.message) || e);
      errEl.style.display = 'block';
    }
    function postJson(url, payload) {
      return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).then(function(r) {
        return r.json().then(function(data) {
          if (!r.ok || !data.ok) throw new Error(data.error || 'request failed');
          return data;
        });
      });
    }
    function updateCount() {
      const n = listEl.children.length;
      countEl.textContent = '(' + n + ' file' + (n === 1 ? '' : 's') + ')';
    }
    // Rebuilding the corpus list in place (instead of window.location.reload())
    // after every upload/delete keeps the rest of [CONFIG] intact -- provider
    // selection, model endpoint, custom endpoint fields -- rather than
    // resetting the whole form back to its default gated state every time.
    function addCorpusItem(filename) {
      if (listEl.querySelector('[data-filename="' + CSS.escape(filename) + '"]')) return;
      const item = document.createElement('div');
      item.className = 'corpus-item';
      const nameEl = document.createElement('span');
      nameEl.className = 'corpus-fname';
      nameEl.title = filename;
      nameEl.textContent = filename;
      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'corpus-delete';
      delBtn.title = 'delete';
      delBtn.setAttribute('data-filename', filename);
      delBtn.textContent = '\\u00d7';
      item.appendChild(nameEl);
      item.appendChild(delBtn);
      listEl.appendChild(item);
    }

    // Event delegation on the list container, not per-button listeners --
    // items added dynamically after the initial render (via addCorpusItem)
    // need no separate re-binding step this way.
    listEl.addEventListener('click', function(ev) {
      const btn = ev.target.closest('.corpus-delete');
      if (!btn) return;
      const filename = btn.getAttribute('data-filename');
      if (!window.confirm('Delete ' + filename + ' from the corpus?')) return;
      errEl.style.display = 'none';
      postJson('/corpus/delete', { filename: filename })
        .then(function() {
          btn.closest('.corpus-item').remove();
          updateCount();
        })
        .catch(showError);
    });

    function arrayBufferToBase64(buffer) {
      let binary = '';
      const bytes = new Uint8Array(buffer);
      const chunkSize = 0x8000;  // avoid a call-stack blowup from spreading a huge array at once
      for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
      }
      return btoa(binary);
    }

    const uploadBtn = document.getElementById('corpus-upload-btn');
    const fileInput = document.getElementById('corpus-file-input');
    uploadBtn.addEventListener('click', function() {
      const file = fileInput.files[0];
      if (!file) { showError('choose a file first'); return; }
      errEl.style.display = 'none';
      uploadBtn.disabled = true;
      uploadBtn.textContent = 'uploading...';
      const isBinary = /\\.(pdf|zip)$/i.test(file.name);
      const reader = new FileReader();
      reader.onload = function() {
        const req = isBinary
          ? postJson('/corpus/upload_binary', { filename: file.name, content_base64: arrayBufferToBase64(reader.result) })
          : postJson('/corpus/upload', { filename: file.name, content: reader.result });
        req
          .then(function(data) {
            (data.filenames || [file.name]).forEach(addCorpusItem);
            updateCount();
            fileInput.value = '';
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'add file';
          })
          .catch(function(e) {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'add file';
            showError(e);
          });
      };
      reader.onerror = function() {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'add file';
        showError('could not read file');
      };
      if (isBinary) {
        reader.readAsArrayBuffer(file);
      } else {
        reader.readAsText(file);
      }
    });

    const urlBtn = document.getElementById('corpus-url-btn');
    const urlInput = document.getElementById('corpus-url-input');
    urlBtn.addEventListener('click', function() {
      const url = urlInput.value.trim();
      if (!url) { showError('paste a URL first'); return; }
      errEl.style.display = 'none';
      urlBtn.disabled = true;
      urlBtn.textContent = 'fetching...';
      postJson('/corpus/upload_url', { url: url })
        .then(function(data) {
          (data.filenames || []).forEach(addCorpusItem);
          updateCount();
          urlInput.value = '';
          urlBtn.disabled = false;
          urlBtn.textContent = 'fetch';
        })
        .catch(function(e) {
          urlBtn.disabled = false;
          urlBtn.textContent = 'fetch';
          showError(e);
        });
    });
  })();
"""

_THEME_TOGGLE_JS = """
  (function() {
    const btn = document.getElementById('theme-toggle');
    const root = document.documentElement;
    const darkLabel = document.getElementById('theme-dark-label');
    const lightLabel = document.getElementById('theme-light-label');
    function apply(theme) {
      root.setAttribute('data-theme', theme);
      darkLabel.classList.toggle('dim', theme !== 'dark');
      lightLabel.classList.toggle('dim', theme !== 'light');
    }
    const saved = localStorage.getItem('theme');
    const osPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    apply(saved || (osPrefersDark ? 'dark' : 'light'));
    btn.addEventListener('click', function() {
      const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      apply(next);
      localStorage.setItem('theme', next);
    });
  })();
"""

_IMPORT_JS = """
  (function() {
    const btn = document.getElementById('import-json-btn');
    const fileInput = document.getElementById('import-json-file-input');
    const errEl = document.getElementById('import-json-error');
    function showError(msg) {
      errEl.textContent = msg;
      errEl.style.display = 'block';
    }
    btn.addEventListener('click', function() {
      const file = fileInput.files[0];
      if (!file) { showError('choose a .json file first'); return; }
      errEl.style.display = 'none';
      btn.disabled = true;
      btn.textContent = 'importing...';
      const reader = new FileReader();
      reader.onload = function() {
        let parsed;
        try {
          parsed = JSON.parse(reader.result);
        } catch (e) {
          btn.disabled = false;
          btn.textContent = 'import';
          showError('not valid JSON: ' + e.message);
          return;
        }
        fetch('/results/import', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(parsed),
        })
          .then(function(r) { return r.json().then(function(data) { return { ok: r.ok, data: data }; }); })
          .then(function(res) {
            if (res.ok && res.data.ok) {
              window.location.reload();
            } else {
              throw new Error(res.data.error || 'import failed');
            }
          })
          .catch(function(e) {
            btn.disabled = false;
            btn.textContent = 'import';
            showError(String(e.message || e));
          });
      };
      reader.onerror = function() {
        btn.disabled = false;
        btn.textContent = 'import';
        showError('could not read file');
      };
      reader.readAsText(file);
    });
  })();
"""


def render_html(results: dict) -> str:
    # Filtered to the current run's target_model, corpus_fingerprint, AND
    # (2026-08-16) questions_fingerprint—fixes real bugs where this used to
    # mix different models', different corpora's, or (now that
    # auto-generated question sets always share one filename regardless of
    # their actual content) different *questions'* scores onto the same
    # trend line, which is misleading, not just incomplete. Kept
    # independent of the leaderboard UI (removed 2026-08-13, not a wanted
    # feature) since these bugs exist regardless of whether there's a
    # comparison view.
    runs = history.load_comparable_runs(
        target_model=results["target_model"],
        corpus_fingerprint=results.get("corpus_fingerprint"),
        questions_fingerprint=results.get("questions_fingerprint"),
    )
    if not runs or runs[-1].get("timestamp") != results.get("timestamp"):
        runs = runs + [results]

    runs_json = json.dumps(runs)
    corpus_files = _corpus_files()
    corpus_summary = f"{len(corpus_files)} file{'s' if len(corpus_files) != 1 else ''}"
    questions_files = _questions_files()
    current_questions_file = results.get("questions_file")
    # Averaged across every comparable run, not just the one currently
    # displayed -- a single run's duration/cost can be a fluke (a slow
    # first-load, a transient MicroDC retry); the average is the more
    # honest answer to "how long/much does this actually take". Runs from
    # before this was tracked are skipped rather than counted as zero,
    # which would silently drag the average down.
    _durations = [r["duration_seconds"] for r in runs if "duration_seconds" in r]
    if _durations:
        _avg_minutes, _avg_seconds = divmod(sum(_durations) / len(_durations), 60)
        duration_summary = f"{int(_avg_minutes)}m {_avg_seconds:.0f}s"
    else:
        duration_summary = "n/a"
    _costs = [r["microdc_cost_usd"] for r in runs if "microdc_cost_usd" in r]
    cost_summary = f"${sum(_costs) / len(_costs):.4f}" if _costs else "n/a"
    truth_bullets_html = "".join(f"<span>{html.escape(b)}</span>" for b in _TRUTH_BULLETS)
    tone_bullets_html = "".join(f"<span>{html.escape(b)}</span>" for b in _TONE_BULLETS)
    config_rail_html = _config_rail_html(
        results["target_model"], corpus_files, corpus_summary, questions_files, current_questions_file
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chatbot Eval Dashboard</title>
{_STYLE_BLOCK}
</head>
<body>
<div class="viz-root">
<div class="layout">
  <div class="rail">
    <div class="live-line"><span class="live-dot"></span><span id="run-label-rail">live</span></div>

    <div>
      <span class="section-label">[CONFIG]</span>
      <div style="display:flex;flex-direction:column;gap:10px;margin-top:10px;">
        {config_rail_html}
      </div>
    </div>

    {_COST_CONFIRM_MODAL_HTML}

    <hr class="rail-divider">

    <div>
      <span class="section-label">[METADATA]</span>
      <div class="meta-row"><span>total runs</span><span class="meta-value">{len(runs)}</span></div>
      <div class="meta-row"><span>questions/run</span><span class="meta-value">{results['num_questions']}{f" ({html.escape(current_questions_file)})" if current_questions_file else ""}</span></div>
      <div class="meta-row"><span>corpus fingerprint<span class="field-help">?<span class="field-help-popup">Changes automatically whenever a corpus/ file is added, removed, or edited&mdash;even a minor text fix. The trend chart only compares runs against this exact corpus content, so editing a corpus file starts a fresh baseline; older runs won't disappear, they just stop showing up in the trend line until the corpus matches again.</span></span></span><span class="meta-value">{html.escape(str(results.get('corpus_fingerprint', 'n/a')))}</span></div>
      <div class="meta-row"><span>questions fingerprint<span class="field-help">?<span class="field-help-popup">Same idea as corpus fingerprint, but for the actual question content asked&mdash;matters especially for auto-generated question sets, which always save to the same filename (questions/generated_questions.json) even though regenerating can produce genuinely different questions each time. The trend chart only compares runs that asked the exact same questions, not just runs that happened to load a file with the same name.</span></span></span><span class="meta-value">{html.escape(str(results.get('questions_fingerprint', 'n/a')))}</span></div>
      <div class="meta-row"><span>avg time<span class="field-help">?<span class="field-help-popup">Average wall-clock time per run, across every comparable run shown in the trend chart&mdash;coverage check, every question, saving the results, not just model latency. A single run can be a fluke (a slow first model load); the average is the more honest answer. Corpus size affects this too: more corpus text means more embedding calls before any question even runs.</span></span></span><span class="meta-value">{duration_summary}</span></div>
      <div class="meta-row"><span>avg cost<span class="field-help">?<span class="field-help-popup">Average MicroDC spend per run, across every comparable run&mdash;real cost from MicroDC's own billing, not an estimate. Scales with corpus size: more corpus text means more chunks to embed on every single run, not just the first one.</span></span></span><span class="meta-value">{cost_summary}</span></div>
      <div class="meta-row"><span>judge</span></div>
      <div style="font-size:9px;color:var(--faint);margin-top:2px;">{html.escape(results['judge_model'])}</div>
    </div>

    <hr class="rail-divider">

    <div>
      <span class="section-label">[EXPORT]</span>
      <button id="save-json-btn" class="run-eval-btn" type="button" style="margin-top:10px;">SAVE DATA AS JSON</button>
      <div style="font-size:9px;color:var(--faint);margin-top:6px;">downloads every comparable run currently loaded on this page&mdash;for feeding to another LLM or tool</div>
    </div>
    {_IMPORT_SECTION_HTML}
  </div>

  <div class="main">
    <div class="header-strip">
      <span id="run-label-header"></span>
      {_THEME_TOGGLE_BTN_HTML}
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
        html += '<div class="vague-warning">&#9888; possible non-answer&mdash;contains hedging language ("' +
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
        (run.target_model || '') + '.eval\\u2014' + (run.timestamp || '').slice(0, 10);
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
{_THEME_TOGGLE_JS}
{_CONFIG_RAIL_JS}
{_IMPORT_JS}
</script>
</div>
</body>
</html>
"""


def render_onboarding_html() -> str:
    """Shown by harness/server.py's `/` handler before any eval run has
    ever completed (no results/latest.json yet) -- previously a bare
    "No eval runs yet" message with zero interactive controls, which meant
    the dashboard's own corpus-upload/model-endpoint/RUN_EVAL UI (the
    thing a fresh user would naturally reach for first) was completely
    unreachable until a run had already happened some other way. Reuses
    the exact same [CONFIG] rail (_config_rail_html/_CONFIG_RAIL_JS) as the
    full dashboard, so both onboarding paths this page explicitly presents
    -- configure everything here and click RUN EVAL, or skip the UI
    entirely and drop files into corpus/questions/ on disk then run
    `python run_eval.py` from a terminal -- are real, equally-supported
    choices, not one blessed path with the other left undiscoverable.

    model endpoint defaults to config.TARGET_MODEL (whatever .env already
    has, possibly blank) rather than a prior run's target_model, since
    there isn't one yet. No trend chart, score panels, severity groups, or
    [EXPORT] section -- there's no run data for any of them to show yet;
    [IMPORT] stays, since loading a results.json produced elsewhere is a
    valid way to get this dashboard's first real data without running
    anything at all."""
    corpus_files = _corpus_files()
    corpus_summary = f"{len(corpus_files)} file{'s' if len(corpus_files) != 1 else ''}"
    questions_files = _questions_files()
    config_rail_html = _config_rail_html(
        config.TARGET_MODEL or "", corpus_files, corpus_summary, questions_files, None
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chatbot Eval Dashboard</title>
{_STYLE_BLOCK}
</head>
<body>
<div class="viz-root">
<div class="layout">
  <div class="rail">
    <div class="live-line"><span class="live-dot"></span><span>no runs yet</span></div>

    <div>
      <span class="section-label">[CONFIG]</span>
      <div style="display:flex;flex-direction:column;gap:10px;margin-top:10px;">
        {config_rail_html}
      </div>
    </div>

    {_COST_CONFIRM_MODAL_HTML}

    <hr class="rail-divider">

    <div>
      <span class="section-label">[METADATA]</span>
      <div class="meta-row"><span>total runs</span><span class="meta-value">0</span></div>
      <div class="meta-row"><span>corpus files</span><span class="meta-value">{len(corpus_files)}</span></div>
      <div class="meta-row"><span>questions files</span><span class="meta-value">{len(questions_files)}</span></div>
    </div>
    {_IMPORT_SECTION_HTML}
  </div>

  <div class="main">
    <div class="header-strip">
      <span id="run-label-header">No eval runs yet</span>
      {_THEME_TOGGLE_BTN_HTML}
    </div>

    <div class="onboarding-note">
      <p>This dashboard needs a target model, a reference corpus, and a matching
      question set before it has anything to show. Pick whichever setup path you
      prefer&mdash;both are fully supported:</p>
      <ul>
        <li><strong>Through this UI</strong>: choose a target provider on the left,
        set the model endpoint, add corpus files (upload, paste a URL, or both),
        then click RUN EVAL&mdash;if you haven't picked a questions file, one gets
        generated from your corpus automatically before the run starts. Once it
        finishes this page reloads into the full dashboard automatically.</li>
        <li><strong>Directly on disk</strong>: drop files into <code>corpus/</code>,
        set <code>TARGET_MODEL</code>/<code>QUESTIONS_FILE</code> in <code>.env</code>
        (optional&mdash;<code>run_eval.py</code> generates a question set from your
        corpus automatically if <code>QUESTIONS_FILE</code> isn't set), then run
        <code>python run_eval.py</code> from a terminal. Refresh this page afterward
        to see it.</li>
      </ul>
      <p>Either path writes to the same <code>corpus/</code>/<code>questions/</code>/
      <code>.env</code> the other one reads from, so they're interchangeable run to
      run&mdash;upload a file here today, drop one on disk tomorrow, no difference.
      You can also skip running anything yourself and use [IMPORT] on the left to
      load a results.json produced elsewhere.</p>
    </div>
  </div>
</div>

<script>
{_THEME_TOGGLE_JS}
{_CONFIG_RAIL_JS}
{_IMPORT_JS}
</script>
</div>
</body>
</html>
"""


def write_report(results: dict, path: str) -> None:
    with open(path, "w") as f:
        f.write(render_html(results))
