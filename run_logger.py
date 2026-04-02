"""
Run logger — saves a completed pipeline state as a self-contained HTML file
under logs/. Each file embeds the series A/B comparison plot (base64 PNG)
and a structured, human-readable record of every agent's reasoning.
"""
import base64
import html as _html
import io
import json
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools import NumpyEncoder


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compact_arrays(text: str) -> str:
    def collapse(m):
        return re.sub(r'\s+', ' ', m.group(0))
    return re.sub(r'\[[^\[\]{}]*\]', collapse, text, flags=re.DOTALL)


def _e(s) -> str:
    """HTML-escape a value for safe embedding."""
    return _html.escape(str(s))


def _plot_series_base64(series_a: list, series_b: list, title: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series_a, label="Series A", color="steelblue", linewidth=1.2)
    ax.plot(series_b, label="Series B", color="tomato", alpha=0.85, linewidth=1.2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Time step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Structured record renderer ────────────────────────────────────────────────

def _render_messages(messages: list, collapse_user: bool = False) -> str:
    """Render a message list inside a collapsible <details> block."""
    rows = []
    for m in messages:
        role = m.get("role", "")
        # Use react_role for display label when available (inspector messages only)
        label = m.get("react_role") or role
        content = m.get("content") or ""

        if role == "system":
            rows.append(
                f'<div class="msg msg-system">'
                f'<span class="role">{_e(label)}</span>'
                f'<span class="content-sys">{_e(content)}</span></div>'
            )
        elif role == "user":
            if collapse_user:
                rows.append(
                    f'<div class="msg msg-user">'
                    f'<span class="role">{_e(label)}</span>'
                    f'<details><summary class="evidence-toggle">'
                    f'(dimension results — see ② Inspector message chains)</summary>'
                    f'<pre class="content-user">{_e(content)}</pre></details></div>'
                )
            else:
                rows.append(
                    f'<div class="msg msg-user">'
                    f'<span class="role">{_e(label)}</span>'
                    f'<pre class="content-user">{_e(content)}</pre></div>'
                )
        elif role == "assistant":
            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                tc_html = "".join(
                    f'<div class="tool-call">▶ <b>{_e(tc["function"]["name"])}</b>'
                    f'({_e(tc["function"]["arguments"])})</div>'
                    for tc in tool_calls
                )
                content_html = ""
                if content and content.strip():
                    content_html = f'<pre class="content-assistant">{_e(content)}</pre>'
                rows.append(
                    f'<div class="msg msg-assistant">'
                    f'<span class="role">{_e(label)}</span>{content_html}{tc_html}</div>'
                )
            else:
                rows.append(
                    f'<div class="msg msg-assistant">'
                    f'<span class="role">{_e(label)}</span>'
                    f'<pre class="content-assistant">{_e(content)}</pre></div>'
                )
        elif role == "tool":
            rows.append(
                f'<div class="msg msg-tool">'
                f'<span class="role">{_e(label)}</span>'
                f'<pre class="content-tool">{_e(content)}</pre></div>'
            )

    inner = "\n".join(rows)
    return (
        f'<details><summary class="msg-toggle">'
        f'Show / hide message chain ({len(messages)} messages)</summary>'
        f'<div class="msg-list">{inner}</div></details>'
    )


def _render_messages_with_dim_markers(messages: list) -> str:
    """Render the full Inspector message chain with dimension dividers.

    Messages containing DIMENSION_COMPLETE are labelled as 'thought'.
    A divider is inserted AFTER the conclusion to mark the dimension boundary.
    """
    rows = []
    dim_pattern = re.compile(r'DIMENSION_COMPLETE')

    for m in messages:
        role = m.get("role", "")
        label = m.get("react_role") or role
        content = m.get("content") or ""

        has_dim_complete = role == "assistant" and dim_pattern.search(content)

        # Render the message
        if role == "system":
            rows.append(
                f'<div class="msg msg-system">'
                f'<span class="role">{_e(label)}</span>'
                f'<span class="content-sys">{_e(content)}</span></div>'
            )
        elif role == "user":
            rows.append(
                f'<div class="msg msg-user">'
                f'<span class="role">{_e(label)}</span>'
                f'<pre class="content-user">{_e(content)}</pre></div>'
            )
        elif role == "assistant":
            tool_calls = m.get("tool_calls", [])

            if has_dim_complete and tool_calls:
                # Split: render conclusion as thought, then divider, then tool calls as action
                # ① Conclusion
                if content and content.strip():
                    rows.append(
                        f'<div class="msg msg-assistant">'
                        f'<span class="role">thought</span>'
                        f'<pre class="content-assistant">{_e(content)}</pre></div>'
                    )
                # ② Divider
                dim_name = ""
                dim_match = re.search(r'"dimension"\s*:\s*"([^"]+)"', content)
                if dim_match:
                    dim_name = dim_match.group(1)
                divider_label = f"Dimension: {dim_name}" if dim_name else "Dimension conclusion"
                rows.append(
                    f'<div style="border-bottom:2px solid #4a90d9; margin:8px 0 14px; padding-bottom:6px;">'
                    f'<b style="color:#4a90d9; font-size:0.85em;">▸ {_e(divider_label)} ✓</b></div>'
                )
                # ③ Tool calls for next dimension
                tc_html = "".join(
                    f'<div class="tool-call">▶ <b>{_e(tc["function"]["name"])}</b>'
                    f'({_e(tc["function"]["arguments"])})</div>'
                    for tc in tool_calls
                )
                rows.append(
                    f'<div class="msg msg-assistant">'
                    f'<span class="role">action</span>{tc_html}</div>'
                )
                has_dim_complete = False  # divider already inserted
            elif tool_calls:
                tc_html = "".join(
                    f'<div class="tool-call">▶ <b>{_e(tc["function"]["name"])}</b>'
                    f'({_e(tc["function"]["arguments"])})</div>'
                    for tc in tool_calls
                )
                content_html = ""
                if content and content.strip():
                    content_html = f'<pre class="content-assistant">{_e(content)}</pre>'
                rows.append(
                    f'<div class="msg msg-assistant">'
                    f'<span class="role">{_e(label)}</span>{content_html}{tc_html}</div>'
                )
            else:
                cur_label = "thought" if has_dim_complete else label
                rows.append(
                    f'<div class="msg msg-assistant">'
                    f'<span class="role">{_e(cur_label)}</span>'
                    f'<pre class="content-assistant">{_e(content)}</pre></div>'
                )
        elif role == "tool":
            rows.append(
                f'<div class="msg msg-tool">'
                f'<span class="role">{_e(label)}</span>'
                f'<pre class="content-tool">{_e(content)}</pre></div>'
            )

        # Insert dimension divider AFTER pure text conclusions (no tool_calls in same message)
        if has_dim_complete:
            dim_name = ""
            dim_match = re.search(r'"dimension"\s*:\s*"([^"]+)"', content)
            if dim_match:
                dim_name = dim_match.group(1)
            divider_label = f"Dimension: {dim_name}" if dim_name else "Dimension conclusion"
            rows.append(
                f'<div style="border-bottom:2px solid #4a90d9; margin:8px 0 14px; padding-bottom:6px;">'
                f'<b style="color:#4a90d9; font-size:0.85em;">▸ {_e(divider_label)} ✓</b></div>'
            )

    inner = "\n".join(rows)
    return (
        f'<details><summary class="msg-toggle">'
        f'<b>ReAct chain</b> ({len(messages)} messages)</summary>'
        f'<div class="msg-list">{inner}</div></details>'
    )


def _winner_badge(winner: str, confidence: float) -> str:
    color = {"A": "#1a6fa8", "B": "#c0392b", "tie": "#7f8c8d"}.get(winner.upper(), "#555")
    return (
        f'<span class="badge" style="background:{color}20;color:{color}">'
        f'{_e(winner.upper())}</span> '
        f'<span class="conf">{confidence:.0%}</span>'
    )


def _render_record(record: dict) -> str:
    parts = []

    # ── Perceiver ─────────────────────────────────────────────────────────────
    perceiver = record.get("perceiver", {})
    planned = perceiver.get("planned_dimensions", [])
    tool_req = set(perceiver.get("tool_required", planned))
    dim_badges = []
    for d in planned:
        if d in tool_req:
            dim_badges.append(f'<code>{_e(d)}</code><span class="dim-tag dim-tag-tool">tool</span>')
        else:
            dim_badges.append(f'<code>{_e(d)}</code><span class="dim-tag dim-tag-reason">reasoning</span>')
    dims_html = " &nbsp; ".join(dim_badges) or "—"
    parts.append(f"""
<section>
  <h2>① Perceiver</h2>
  <p><b>Planned dimensions:</b> {dims_html}</p>
  <p><b>Perception summary:</b> {_e(perceiver.get("perception_summary", ""))}</p>
  {_render_messages(perceiver.get("messages", []))}
</section>""")

    # ── Inspector (per dimension) ─────────────────────────────────────────────
    inspector_items = record.get("inspector", [])

    # Dimension summary cards (no message chain per card)
    dim_cards = []
    for r in inspector_items:
        evidence_json = _compact_arrays(
            json.dumps(r.get("evidence", {}), indent=2, cls=NumpyEncoder, ensure_ascii=False)
        )
        dim_name = r["dimension"]
        if dim_name in tool_req:
            mode_tag = '<span class="dim-tag dim-tag-tool">tool</span>'
        else:
            mode_tag = '<span class="dim-tag dim-tag-reason">reasoning</span>'
        dim_cards.append(f"""
  <div class="dim-card">
    <div class="dim-header">
      <b>{_e(dim_name)}</b> {mode_tag}
      &nbsp; {_winner_badge(r.get("winner","tie"), r.get("confidence", 0))}
    </div>
    <p class="conclusion">{_e(r.get("conclusion",""))}</p>
    <details><summary class="evidence-toggle">Evidence</summary>
      <pre class="evidence">{_e(evidence_json)}</pre>
    </details>
  </div>""")

    # Render one unified ReAct chain (take the longest message list)
    all_msg_lists = [r.get("messages", []) for r in inspector_items]
    longest_chain = max(all_msg_lists, key=len) if all_msg_lists else []
    chain_html = _render_messages_with_dim_markers(longest_chain) if longest_chain else ""

    parts.append(f"""
<section>
  <h2>② Inspector</h2>
  {"".join(dim_cards)}
  {chain_html}
</section>""")

    # ── Adjudicator ───────────────────────────────────────────────────────────
    adj = record.get("adjudicator", {})
    parts.append(f"""
<section>
  <h2>③ Adjudicator</h2>
  <p><b>Recheck count:</b> {adj.get("recheck_count", 0)} &nbsp;
     <b>Replan count:</b> {adj.get("replan_count", 0)}</p>
  {_render_messages(adj.get("messages", []), collapse_user=True)}
</section>""")

    # ── Final result ──────────────────────────────────────────────────────────
    final = record.get("final_result", {})
    parts.append(f"""
<section>
  <h2>④ Final Result</h2>
  <p><b>Winner:</b> {_winner_badge(final.get("winner","tie"), final.get("confidence",0))}</p>
  <p><b>Explanation:</b></p>
  <blockquote>{_e(final.get("explanation",""))}</blockquote>
</section>""")

    return "\n".join(parts)


# ── HTML template ─────────────────────────────────────────────────────────────

CSS = """
body    { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1100px; margin: 0 auto; padding: 24px; color: #222; font-size: 14px; }
h1      { font-size: 1.15em; margin-bottom: 4px; }
h2      { font-size: 0.95em; font-weight: 700; color: #1a1a1a;
          border-left: 3px solid #4a90d9; padding-left: 10px;
          margin-top: 32px; margin-bottom: 12px; }
.meta   { color: #666; font-size: 0.85em; margin-bottom: 20px; }
.badge  { display: inline-block; padding: 1px 9px; border-radius: 10px; font-weight: 700; font-size: 0.9em; }
.conf   { color: #555; font-size: 0.88em; }
section { margin-bottom: 12px; }
img     { width: 100%; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 8px; }

/* dimension cards */
.dim-card   { border: 1px solid #e0e0e0; border-radius: 6px;
              padding: 12px 16px; margin-bottom: 10px; background: #fafafa; }
.dim-header { font-size: 1em; margin-bottom: 6px; }
.conclusion { margin: 4px 0 8px; color: #333; }
.dim-tag    { display: inline-block; font-size: 0.72em; font-weight: 600;
              padding: 1px 7px; border-radius: 8px; vertical-align: middle; margin-left: 4px; }
.dim-tag-tool   { background: #e8f0fe; color: #1a56c7; }
.dim-tag-reason { background: #e8f5e9; color: #2e7d32; }

/* evidence */
.evidence-toggle { cursor: pointer; font-size: 0.82em; color: #888; user-select: none; }
pre.evidence { background: #f4f4f4; border: 1px solid #e4e4e4; border-radius: 4px;
               padding: 10px; font-size: 12px; white-space: pre-wrap; word-break: break-all;
               margin-top: 6px; }

/* messages */
.msg-toggle  { cursor: pointer; font-size: 0.82em; color: #888; user-select: none; margin-top: 6px; }
.msg-list    { margin-top: 6px; border-left: 2px solid #e8e8e8; padding-left: 10px; }
.msg         { margin: 5px 0; }
.role        { display: inline-block; min-width: 72px; font-size: 0.78em;
               font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }

/* role colours */
.msg-system .role  { color: #bbb; }
.content-sys       { color: #bbb; font-size: 0.78em; font-style: italic; }
.msg-user .role    { color: #888; }
pre.content-user   { display: inline; background: none; border: none; padding: 0;
                     font-size: 0.88em; color: #444; white-space: pre-wrap; word-break: break-all; }
.msg-assistant .role { color: #1a6fa8; }
pre.content-assistant { display: inline; background: none; border: none; padding: 0;
                        font-size: 0.88em; color: #1a1a1a; white-space: pre-wrap; word-break: break-all; }
.tool-call { font-size: 0.85em; color: #2c7a4b; margin-left: 4px; }
.msg-tool .role { color: #2c7a4b; }
pre.content-tool { display: inline; background: none; border: none; padding: 0;
                   font-size: 0.82em; color: #555; white-space: pre-wrap; word-break: break-all; }

blockquote { border-left: 3px solid #e0e0e0; margin: 8px 0; padding: 6px 14px; color: #333; }
code       { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
"""


def _build_html(case_name: str, b64_img: str, record: dict) -> str:
    final = record.get("final_result", {})
    winner = final.get("winner", "N/A")
    confidence = final.get("confidence", 0)
    model = record.get("config", {}).get("model", "")
    timestamp = record.get("timestamp", "")

    record_html = _render_record(record)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{_e(case_name)}</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>{_e(case_name)}</h1>
  <p class="meta">
    {_e(timestamp)} &nbsp;|&nbsp; model: {_e(model)}
    &nbsp;|&nbsp; winner: {_winner_badge(winner, confidence)}
  </p>

  <h2 style="border-left-color:#4a4a4a">Series A vs B</h2>
  <img src="data:image/png;base64,{b64_img}" alt="series plot">

  {record_html}
</body>
</html>"""


# ── Public API ────────────────────────────────────────────────────────────────

def save_run(
    state: dict,
    case_name: str,
    config_dict: dict,
    log_dir: str = "logs",
) -> str:
    now = datetime.now()
    date_dir = os.path.join(log_dir, now.strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)

    timestamp = now.strftime("%H%M%S")
    safe_name = (
        case_name.replace(" ", "_").replace("|", "").replace("/", "-").replace("%", "pct")
    )[:60]
    filename = f"{timestamp}_{safe_name}.html"
    filepath = os.path.join(date_dir, filename)

    inp = state.get("input", {})
    series_a = inp.get("series_A", [])
    series_b = inp.get("series_B", [])

    record = {
        "run_id": f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_name}",
        "timestamp": now.isoformat(),
        "case_name": case_name,
        "config": config_dict,
        "input": {
            "dataset_description": inp.get("dataset_description", ""),
            "series_A": series_a,
            "series_B": series_b,
            "external_variables": inp.get("external_variables", {}),
        },
        "perceiver": {
            "perception_summary": state.get("perception_summary", ""),
            "planned_dimensions": state.get("planned_dimensions", []),
            "tool_required": state.get("tool_required", []),
            "messages": state.get("perceiver_messages", []),
        },
        "inspector": [
            {
                "dimension": r["dimension"],
                "winner": r["winner"],
                "confidence": r["confidence"],
                "evidence": r["evidence"],
                "conclusion": r["conclusion"],
                "messages": r.get("messages", []),
            }
            for r in state.get("dimension_results", [])
        ],
        "adjudicator": {
            "recheck_count": state.get("recheck_count", 0),
            "replan_count": state.get("replan_count", 0),
            "messages": state.get("adjudicator_messages", []),
        },
        "final_result": state.get("final_result", {}),
    }

    b64_img = _plot_series_base64(series_a, series_b, case_name)
    html = _build_html(case_name, b64_img, record)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath