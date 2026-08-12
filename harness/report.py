"""Renders a results dict (see evaluator.run_evaluation) to a single
self-contained HTML file. No JS framework, no build step -- just open it in a
browser, or serve it live via harness/server.py (Phase 7)."""

import html
import json

from . import history

# Palette: dataviz skill's validated default (references/palette.md),
# categorical slots 1-3 (blue/orange/aqua) for the trend chart's 3 series --
# validated CVD-safe all-pairs in both light and dark (see PLAN.md Phase 7).
# Status palette maps directly onto severity: none=good, minor=warning,
# flag=critical.

_SEVERITY_BADGE = {
    "flag": ("FLAGGED", "critical"),
    "minor": ("minor", "warning"),
    "none": ("ok", "good"),
}

_TREND_SERIES = [
    ("concern_percentage", "series-1", "concern %"),
    ("avg_truthfulness_score", "series-2", "avg truthfulness"),
    ("avg_tone_consistency_score", "series-3", "avg tone consistency"),
]


_NLI_LABEL_STATUS = {"entailment": "good", "neutral": "warning", "contradiction": "critical"}


def _evidence_html(evidence: dict | None) -> str:
    """Claim-level breakdown: which numeric values were checked and whether
    they matched, plus the per-sentence NLI entailment/neutral/contradiction
    confidence behind the verdict -- this data was always computed by
    fact_check.py but never surfaced in the UI until now."""
    if not evidence:
        return ""
    numeric = evidence.get("numeric", {})
    mismatches = numeric.get("mismatches", [])
    per_sentence = evidence.get("nli_per_sentence", [])
    parts = []

    if mismatches:
        items = "".join(
            f"<li><b>{html.escape(str(m['kind']))}</b> &mdash; answer: <code>{html.escape(str(m['answer_value']))}</code>, "
            f"reference: <code>{html.escape(str(m['reference_values']))}</code></li>"
            for m in mismatches
        )
        parts.append(f'<div class="evidence-block status-critical"><b>Numeric check &mdash; mismatch</b><ul>{items}</ul></div>')
    elif numeric.get("checked"):
        parts.append(
            f'<div class="evidence-block status-good"><b>Numeric check</b> &mdash; '
            f'{numeric["checked"]} value(s) extracted from the answer, all consistent with the reference.</div>'
        )

    if per_sentence:
        rows = []
        for s in per_sentence:
            label = s.get("label", "neutral")
            status = _NLI_LABEL_STATUS.get(label, "warning")
            bar = "".join(
                f'<div class="conf-seg conf-{lbl}" style="width:{s.get(lbl, 0) * 100:.1f}%"></div>'
                for lbl in ("entailment", "neutral", "contradiction")
            )
            rows.append(f"""
            <div class="evidence-row">
              <div class="evidence-sentence">{html.escape(s['sentence'])}</div>
              <div class="conf-bar" title="entailment {s.get('entailment', 0) * 100:.0f}% / neutral {s.get('neutral', 0) * 100:.0f}% / contradiction {s.get('contradiction', 0) * 100:.0f}%">{bar}</div>
              <div class="evidence-verdict status-{status}">{label} {s.get(label, 0) * 100:.0f}%</div>
            </div>
            """)
        parts.append(f'<div class="evidence-block"><b>Sentence-level NLI breakdown</b>{"".join(rows)}</div>')

    if not parts:
        return ""
    return f'<div class="evidence-section">{"".join(parts)}</div>'


def _row_html(q: dict) -> str:
    severity = q.get("severity", "flag" if q["concern"] else "none")
    label, status = _SEVERITY_BADGE.get(severity, _SEVERITY_BADGE["flag"])
    return f"""
    <details class="question-row status-{status}" id="q-{html.escape(q['id'])}">
      <summary>
        <span class="qid">{html.escape(q['id'])}</span>
        <span class="qtext">{html.escape(q['question'])}</span>
        <span class="badge status-{status}"><span class="badge-dot"></span>{label}</span>
        <span class="score">truthfulness {q['truthfulness_score']}</span>
        <span class="score">tone {q['tone_consistency_score']}</span>
      </summary>
      <div class="detail">
        <p><b>Fact-check verdict:</b> {html.escape(q['reason'])}</p>
        {_evidence_html(q.get('evidence'))}
        <p><b>Ungrounded answer</b> (target model, no reference):</p>
        <pre>{html.escape(q['ungrounded_answer'])}</pre>
        <p><b>Grounded answer</b> (target model + retrieved context):</p>
        <pre>{html.escape(q['grounded_answer'])}</pre>
        <p><b>Reference context used:</b></p>
        <pre>{html.escape(q['reference_context'] or '(none found)')}</pre>
      </div>
    </details>
    """


# Each tooltip is a short list of one-line reasons, not a paragraph --
# placeholder-quality copy is fine here, this earns a real pass in the final
# product per the user's note (2026-08-12) that this doesn't need polishing now.
_TOOLTIPS: dict[str, list[str]] = {
    "concern %": ["share of questions flagged this run"],
    "truthfulness": ["numeric check + small NLI model vs. reference", "can misread emphasis as contradiction"],
    "tone consistency": ["does rephrasing change the answer?", "unrelated to correctness"],
    "FLAGGED": ["real error", "overstated certainty", "reference too narrow"],
    "minor": ["not clearly supported, not contradicted", "often just correct extra detail"],
    "ok": ["consistent with the reference"],
}


def _hoverable(label: str, extra_class: str = "") -> str:
    """A short label with its fuller interpretation revealed on hover, not
    shown by default."""
    bullets = "".join(f"<span>{html.escape(line)}</span>" for line in _TOOLTIPS.get(label, []))
    return f'<span class="info-hover {extra_class}">{label}<span class="tooltip-content">{bullets}</span></span>'


def _key_html() -> str:
    """Compact legend, hover for the short version of "why" -- not a wall of
    text by default. See _TOOLTIPS for content."""
    return f"""
    <div class="key-strip">
      {_hoverable("concern %")}
      {_hoverable("truthfulness")}
      {_hoverable("tone consistency")}
      {_hoverable("FLAGGED", "badge status-critical")}
      {_hoverable("minor", "badge status-warning")}
      {_hoverable("ok", "badge status-good")}
    </div>
    """


def _attention_feed_html(questions: list[dict]) -> str:
    """Flagged/minor questions pulled out into their own compact,
    triage-first list -- a real dashboard's job is "what needs my attention,"
    not making the reader scan every row to find the two that matter."""
    flagged = [q for q in questions if q.get("severity") == "flag"]
    minor = [q for q in questions if q.get("severity") == "minor"]
    if not flagged and not minor:
        return '<p class="attention-empty">Nothing flagged or minor this run &mdash; every answer checked out clean.</p>'

    items = []
    for q in flagged + minor:
        label, status = _SEVERITY_BADGE.get(q.get("severity"), _SEVERITY_BADGE["flag"])
        items.append(f"""
        <a class="attention-item status-{status}" href="#q-{html.escape(q['id'])}">
          <span class="badge status-{status}"><span class="badge-dot"></span>{label}</span>
          <span class="attention-qid">{html.escape(q['id'])}</span>
          <span class="attention-reason">{html.escape(q['reason'])}</span>
        </a>
        """)
    return f'<div class="attention-feed">{"".join(items)}</div>'


def _stat_tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="stat-sub">{html.escape(sub)}</div>' if sub else ""
    return f"""
    <div class="stat-tile">
      <div class="stat-label">{html.escape(label)}</div>
      <div class="stat-value">{html.escape(str(value))}</div>
      {sub_html}
    </div>
    """


def _trend_chart_html(runs: list[dict]) -> str:
    """Line chart, 3 series, with a JS crosshair + tooltip (per the dataviz
    skill's interaction spec) and a table-view twin for accessibility."""
    if len(runs) < 2:
        return f"""
        <p class="trend-note">Only {len(runs)} run{'s' if len(runs) != 1 else ''} so far on the
        current architecture -- need at least 2 to show a trend. Runs from earlier
        architectures (different target model, judge, or retrieval mechanism) are
        intentionally excluded since their numbers aren't directly comparable.</p>
        """

    w, h, pad_l, pad_r, pad_t, pad_b = 640, 220, 36, 16, 16, 28
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

    series_svg = []
    legend_items = []
    end_labels = []
    for key, css_class, label in _TREND_SERIES:
        pts = [(x_of(i), y_of(r[key])) for i, r in enumerate(runs)]
        points_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dots = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="dot {css_class}" '
            f'data-run="{i}" data-series="{html.escape(label)}" data-value="{runs[i][key]}" />'
            for i, (x, y) in enumerate(pts)
        )
        series_svg.append(f'<polyline points="{points_attr}" class="trend-line {css_class}" />{dots}')
        legend_items.append(f'<span class="legend-item"><span class="legend-swatch {css_class}"></span>{html.escape(label)}</span>')
        last_x, last_y = pts[-1]
        end_labels.append(
            f'<text x="{last_x + 6:.1f}" y="{last_y + 4:.1f}" class="end-label {css_class}">{runs[-1][key]}</text>'
        )

    # invisible per-run hit columns drive the crosshair via mousemove
    hit_columns = "".join(
        f'<rect x="{x_of(i) - (w / n) / 2:.1f}" y="{pad_t}" width="{(w - pad_l - pad_r) / max(n - 1, 1):.1f}" '
        f'height="{h - pad_t - pad_b}" class="hit-col" data-run="{i}" />'
        for i in range(n)
    )

    table_rows = "".join(
        f"<tr><td>{html.escape(r.get('timestamp', '?'))}</td>"
        f"<td>{r['concern_percentage']}%</td><td>{r['avg_truthfulness_score']}</td>"
        f"<td>{r['avg_tone_consistency_score']}</td><td>{r['flagged_count']}/{r['num_questions']}</td></tr>"
        for r in runs
    )
    timestamps_json = json.dumps([r.get("timestamp", "?") for r in runs])

    return f"""
    <div class="chart-container">
      <svg viewBox="0 0 {w} {h}" class="trend-svg" id="trend-svg">
        {gridlines}
        <line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" class="axis-line" />
        {''.join(series_svg)}
        {''.join(end_labels)}
        <line id="crosshair" x1="0" y1="{pad_t}" x2="0" y2="{h - pad_b}" class="crosshair" style="display:none" />
        {hit_columns}
      </svg>
      <div id="trend-tooltip" class="trend-tooltip" style="display:none"></div>
    </div>
    <div class="legend">{''.join(legend_items)}</div>
    <details class="table-toggle">
      <summary>View as table</summary>
      <table class="trend-table">
        <thead><tr><th>timestamp (UTC)</th><th>concern %</th><th>avg truthfulness</th>
        <th>avg tone</th><th>flagged</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </details>
    <script>
      (function() {{
        const timestamps = {timestamps_json};
        const svg = document.getElementById('trend-svg');
        const crosshair = document.getElementById('crosshair');
        const tooltip = document.getElementById('trend-tooltip');
        if (!svg) return;
        svg.querySelectorAll('.hit-col').forEach(function(col) {{
          col.addEventListener('pointerenter', show);
          col.addEventListener('pointermove', show);
          col.addEventListener('pointerleave', hide);
        }});
        function show(e) {{
          const run = e.target.getAttribute('data-run');
          const x = e.target.getAttribute('x');
          const w = e.target.getAttribute('width');
          const cx = parseFloat(x) + parseFloat(w) / 2;
          crosshair.setAttribute('x1', cx);
          crosshair.setAttribute('x2', cx);
          crosshair.style.display = 'block';
          const dots = svg.querySelectorAll('.dot[data-run="' + run + '"]');
          let rowsHtml = '';
          dots.forEach(function(dot) {{
            const seriesEl = document.createElement('span');
            seriesEl.textContent = dot.getAttribute('data-series');
            const cls = Array.from(dot.classList).find(function(c) {{ return c.startsWith('series-'); }});
            rowsHtml += '<div class="tooltip-row"><span class="tooltip-key ' + cls + '"></span>' +
              '<span class="tooltip-value">' + dot.getAttribute('data-value') + '</span> ' +
              '<span class="tooltip-series">' + seriesEl.textContent + '</span></div>';
          }});
          const tsEl = document.createElement('div');
          tsEl.className = 'tooltip-ts';
          tsEl.textContent = timestamps[run] || '';
          tooltip.innerHTML = '';
          tooltip.appendChild(tsEl);
          const body = document.createElement('div');
          body.innerHTML = rowsHtml;
          tooltip.appendChild(body);
          tooltip.style.display = 'block';
          const rect = svg.getBoundingClientRect();
          const scale = rect.width / {w};
          tooltip.style.left = Math.min(cx * scale - 60, rect.width - 160) + 'px';
          tooltip.style.top = '4px';
        }}
        function hide() {{
          crosshair.style.display = 'none';
          tooltip.style.display = 'none';
        }}
      }})();
    </script>
    """


def render_html(results: dict) -> str:
    pct = results["concern_percentage"]
    pct_status = "good" if pct < 15 else ("warning" if pct < 40 else "critical")
    rows = "\n".join(_row_html(q) for q in results["questions"])
    runs = history.load_comparable_runs()

    stat_tiles = "".join([
        _stat_tile("flagged", results["flagged_count"], f"of {results['num_questions']} questions"),
        _stat_tile("minor", results.get("minor_count", 0)),
        _stat_tile("avg truthfulness", results["avg_truthfulness_score"]),
        _stat_tile("avg tone consistency", results["avg_tone_consistency_score"]),
    ])

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chatbot Eval Dashboard</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:      #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --axis:           #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
    --status-good: #0ca30c; --status-warning: #fab219; --status-critical: #d03b3b;
    --status-good-bg: #eaf7ea; --status-warning-bg: #fff6e0; --status-critical-bg: #fbeaea;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:      #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --axis:           #383835;
      --border:         rgba(255,255,255,0.10);
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
      --status-good: #0ca30c; --status-warning: #fab219; --status-critical: #e66767;
      --status-good-bg: #10240f; --status-warning-bg: #2a220a; --status-critical-bg: #2a1414;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:      #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --axis:           #383835;
    --border:         rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --status-good: #0ca30c; --status-warning: #fab219; --status-critical: #e66767;
    --status-good-bg: #10240f; --status-warning-bg: #2a220a; --status-critical-bg: #2a1414;
  }}

  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; }}
  .viz-root {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
              background: var(--page-plane); color: var(--text-primary);
              min-height: 100vh; margin: 0 auto; }}
  .viz-inner {{ max-width: 880px; margin: 0 auto; padding: 32px 20px 64px; }}
  header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }}
  h1 {{ font-size: 1.25rem; font-weight: 600; margin: 0; }}
  h2 {{ font-size: 1rem; font-weight: 600; margin: 0 0 12px; color: var(--text-primary); }}
  #theme-toggle {{ font: inherit; font-size: 0.8rem; padding: 6px 12px; border-radius: 6px;
                   border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary);
                   cursor: pointer; }}

  .hero-row {{ display: flex; gap: 28px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 32px; }}
  .hero-figure {{ font-size: 3.2rem; font-weight: 600; line-height: 1;
                  color: var(--status-{pct_status}); }}
  .hero-sub {{ font-size: 0.8rem; color: var(--text-secondary); margin-top: 6px; }}
  .kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; flex: 1; }}
  .stat-tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
               padding: 10px 14px; min-width: 110px; }}
  .stat-label {{ font-size: 0.72rem; color: var(--text-muted); text-transform: lowercase; }}
  .stat-value {{ font-size: 1.4rem; font-weight: 600; margin-top: 2px; }}
  .stat-sub {{ font-size: 0.7rem; color: var(--text-muted); margin-top: 2px; }}

  section {{ margin-bottom: 36px; }}
  .trend-note {{ color: var(--text-secondary); font-size: 0.85rem; }}
  .chart-container {{ position: relative; }}
  .trend-svg {{ width: 100%; height: auto; overflow: visible; }}
  .gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .axis-line {{ stroke: var(--axis); stroke-width: 1; }}
  .axis-label {{ font-size: 9px; fill: var(--text-muted); font-family: system-ui, sans-serif; }}
  .trend-line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .trend-line.series-1 {{ stroke: var(--series-1); }}
  .trend-line.series-2 {{ stroke: var(--series-2); }}
  .trend-line.series-3 {{ stroke: var(--series-3); }}
  .dot {{ stroke: var(--surface-1); stroke-width: 2; cursor: pointer; }}
  .dot.series-1 {{ fill: var(--series-1); }}
  .dot.series-2 {{ fill: var(--series-2); }}
  .dot.series-3 {{ fill: var(--series-3); }}
  .end-label {{ font-size: 10px; font-weight: 600; font-family: system-ui, sans-serif; }}
  .end-label.series-1 {{ fill: var(--series-1); }}
  .end-label.series-2 {{ fill: var(--series-2); }}
  .end-label.series-3 {{ fill: var(--series-3); }}
  .hit-col {{ fill: transparent; }}
  .crosshair {{ stroke: var(--axis); stroke-width: 1; pointer-events: none; }}
  .trend-tooltip {{ position: absolute; background: var(--surface-1); border: 1px solid var(--border);
                    border-radius: 6px; padding: 8px 10px; font-size: 0.75rem; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
                    pointer-events: none; min-width: 150px; }}
  .tooltip-ts {{ color: var(--text-muted); font-size: 0.68rem; margin-bottom: 4px; }}
  .tooltip-row {{ display: flex; align-items: center; gap: 6px; margin-top: 2px; }}
  .tooltip-key {{ width: 12px; height: 2px; border-radius: 1px; display: inline-block; }}
  .tooltip-key.series-1 {{ background: var(--series-1); }}
  .tooltip-key.series-2 {{ background: var(--series-2); }}
  .tooltip-key.series-3 {{ background: var(--series-3); }}
  .tooltip-value {{ font-weight: 600; color: var(--text-primary); }}
  .tooltip-series {{ color: var(--text-secondary); }}
  .legend {{ display: flex; gap: 16px; font-size: 0.78rem; color: var(--text-secondary); margin-top: 10px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .legend-swatch.series-1 {{ background: var(--series-1); }}
  .legend-swatch.series-2 {{ background: var(--series-2); }}
  .legend-swatch.series-3 {{ background: var(--series-3); }}
  .table-toggle {{ margin-top: 10px; font-size: 0.8rem; }}
  .table-toggle summary {{ cursor: pointer; color: var(--text-secondary); }}
  .trend-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.78rem;
                  font-variant-numeric: tabular-nums; }}
  .trend-table th, .trend-table td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--gridline); }}
  .trend-table th {{ color: var(--text-muted); font-weight: 500; }}

  .question-row {{ border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px;
                   padding: 10px 14px; background: var(--surface-1); }}
  .question-row.status-critical {{ border-color: var(--status-critical); }}
  .question-row.status-warning {{ border-color: var(--status-warning); }}
  summary {{ cursor: pointer; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .qid {{ font-family: ui-monospace, monospace; color: var(--text-muted); font-size: 0.8rem; }}
  .qtext {{ flex: 1; min-width: 200px; }}
  .badge {{ font-size: 0.72rem; font-weight: 600; padding: 3px 9px 3px 7px; border-radius: 20px;
           display: inline-flex; align-items: center; gap: 5px; }}
  .badge-dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
  .badge.status-good {{ background: var(--status-good-bg); color: var(--status-good); }}
  .badge.status-good .badge-dot {{ background: var(--status-good); }}
  .badge.status-warning {{ background: var(--status-warning-bg); color: #9a6a00; }}
  .badge.status-warning .badge-dot {{ background: var(--status-warning); }}
  .badge.status-critical {{ background: var(--status-critical-bg); color: var(--status-critical); }}
  .badge.status-critical .badge-dot {{ background: var(--status-critical); }}
  .score {{ font-size: 0.72rem; color: var(--text-secondary); background: var(--page-plane);
           padding: 2px 8px; border-radius: 4px; }}
  .detail pre {{ background: var(--page-plane); padding: 10px; border-radius: 6px; white-space: pre-wrap;
                font-size: 0.82rem; color: var(--text-primary); }}
  .detail p {{ font-size: 0.85rem; }}

  .evidence-section {{ margin: 10px 0 16px; }}
  .evidence-block {{ background: var(--page-plane); border-radius: 6px; padding: 10px 12px; margin-bottom: 8px;
                     font-size: 0.82rem; border-left: 3px solid var(--border); }}
  .evidence-block.status-critical {{ border-left-color: var(--status-critical); }}
  .evidence-block.status-good {{ border-left-color: var(--status-good); }}
  .evidence-block ul {{ margin: 6px 0 0; padding-left: 18px; }}
  .evidence-block code {{ font-family: ui-monospace, monospace; font-size: 0.78rem; }}
  .evidence-row {{ display: flex; align-items: center; gap: 10px; margin-top: 8px; flex-wrap: wrap; }}
  .evidence-sentence {{ flex: 1; min-width: 200px; font-size: 0.8rem; color: var(--text-secondary); }}
  .conf-bar {{ display: flex; width: 100px; height: 8px; border-radius: 4px; overflow: hidden;
              background: var(--gridline); flex-shrink: 0; }}
  .conf-seg {{ height: 100%; }}
  .conf-seg.conf-entailment {{ background: var(--status-good); }}
  .conf-seg.conf-neutral {{ background: var(--status-warning); }}
  .conf-seg.conf-contradiction {{ background: var(--status-critical); }}
  .evidence-verdict {{ font-size: 0.75rem; font-weight: 600; min-width: 90px; text-align: right; }}
  .evidence-verdict.status-good {{ color: var(--status-good); }}
  .evidence-verdict.status-warning {{ color: #9a6a00; }}
  .evidence-verdict.status-critical {{ color: var(--status-critical); }}

  .key-strip {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.76rem;
               color: var(--text-secondary); margin-bottom: 20px; align-items: center; }}
  .key-strip .badge {{ transform: scale(0.85); }}

  .info-hover {{ position: relative; cursor: help; border-bottom: 1px dotted var(--text-muted); }}
  .info-hover.badge {{ border-bottom: none; }}
  .info-hover .tooltip-content {{
    visibility: hidden; opacity: 0; position: absolute; bottom: 130%; left: 0;
    background: var(--text-primary); color: var(--surface-1); padding: 8px 10px;
    border-radius: 6px; font-size: 0.7rem; width: max-content; max-width: 200px;
    line-height: 1.5; transition: opacity 0.12s ease; z-index: 20;
    display: flex; flex-direction: column; gap: 2px; font-weight: 400;
    text-transform: none; pointer-events: none;
  }}
  .info-hover:hover .tooltip-content {{ visibility: visible; opacity: 1; }}
  .tooltip-content span:before {{ content: "\\2022  "; color: var(--text-muted); }}

  .attention-empty {{ color: var(--text-secondary); font-size: 0.85rem; }}
  .attention-feed {{ display: flex; flex-direction: column; gap: 6px; }}
  .attention-item {{ display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 6px;
                     background: var(--surface-1); border: 1px solid var(--border); text-decoration: none;
                     color: var(--text-primary); font-size: 0.82rem; }}
  .attention-item.status-critical {{ border-left: 3px solid var(--status-critical); }}
  .attention-item.status-warning {{ border-left: 3px solid var(--status-warning); }}
  .attention-item:hover {{ background: var(--page-plane); }}
  .attention-qid {{ font-family: ui-monospace, monospace; color: var(--text-muted); font-size: 0.75rem; }}
  .attention-reason {{ color: var(--text-secondary); flex: 1; }}
</style>
</head>
<body>
<div class="viz-root">
<div class="viz-inner">
  <header>
    <h1>AI Chatbot Evaluation Dashboard</h1>
    <button id="theme-toggle" type="button">Toggle theme</button>
  </header>

  <div class="hero-row">
    <div>
      <div class="hero-figure">{pct}%</div>
      <div class="hero-sub">concern &middot; {results['flagged_count']}/{results['num_questions']} flagged</div>
    </div>
    <div class="kpi-row">{stat_tiles}</div>
  </div>

  <div class="hero-sub" style="margin: -20px 0 24px;">
    target model: <b>{html.escape(results['target_model'])}</b> &middot;
    judged by: {html.escape(results['judge_model'])} &middot;
    run: {html.escape(results.get('timestamp', 'unknown'))}
  </div>

  {_key_html()}

  <section class="attention-section">
    <h2>Needs attention</h2>
    {_attention_feed_html(results['questions'])}
  </section>

  <section class="trend-section">
    <h2>Trend{' (' + str(len(runs)) + ' comparable runs)' if len(runs) >= 2 else ''}</h2>
    {_trend_chart_html(runs)}
  </section>

  <section class="questions-section">
    <h2>Questions</h2>
    {rows}
  </section>

  <script>
    // raw results, for programmatic access / future charting
    window.__EVAL_RESULTS__ = {json.dumps(results)};
    (function() {{
      const btn = document.getElementById('theme-toggle');
      const root = document.documentElement;
      const osPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      btn.addEventListener('click', function() {{
        const explicit = root.getAttribute('data-theme');
        // effective theme = explicit override, or the OS preference if none set yet
        const effectiveIsDark = explicit ? explicit === 'dark' : osPrefersDark;
        root.setAttribute('data-theme', effectiveIsDark ? 'light' : 'dark');
      }});
    }})();
    (function() {{
      // "Needs attention" links jump to a <details> row -- native anchor
      // navigation scrolls to it but won't open a closed <details>, so open
      // it manually and scroll smoothly instead of an instant jump.
      function openTarget() {{
        const id = decodeURIComponent(location.hash.slice(1));
        if (!id) return;
        const el = document.getElementById(id);
        if (!el) return;
        el.open = true;
        el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
      }}
      window.addEventListener('hashchange', openTarget);
      if (location.hash) openTarget();
    }})();
  </script>
</div>
</div>
</body>
</html>
"""


def write_report(results: dict, path: str) -> None:
    with open(path, "w") as f:
        f.write(render_html(results))
