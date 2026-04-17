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


def _split_at_dim_complete(content: str) -> tuple[str, str, str]:
    """Split assistant content at the DIMENSION_COMPLETE / END_DIMENSION boundary.

    Returns (pre, conclusion, post):
      pre        — text before DIMENSION_COMPLETE (may be empty)
      conclusion — from DIMENSION_COMPLETE up to and including END_DIMENSION
      post       — text after END_DIMENSION (next dimension's thought, may be empty)
    """
    dc_match = re.search(r'DIMENSION_COMPLETE', content)
    if not dc_match:
        return content, "", ""

    pre = content[:dc_match.start()].strip()
    rest = content[dc_match.start():]
    ed_idx = rest.find("END_DIMENSION")
    if ed_idx >= 0:
        conclusion = rest[:ed_idx + len("END_DIMENSION")]
        post = rest[ed_idx + len("END_DIMENSION"):].strip()
    else:
        conclusion = rest
        post = ""
    return pre, conclusion, post


def _dim_divider(content: str) -> str:
    """Blue divider bar labelled with the dimension name extracted from content."""
    dim_match = re.search(r'"dimension"\s*:\s*"([^"]+)"', content)
    dim_name = dim_match.group(1) if dim_match else ""
    label = f"Dimension: {dim_name}" if dim_name else "Dimension conclusion"
    return (
        f'<div style="border-bottom:2px solid #4a90d9; margin:8px 0 14px; padding-bottom:6px;">'
        f'<b style="color:#4a90d9; font-size:0.85em;">▸ {_e(label)} ✓</b></div>'
    )


def _render_messages_with_dim_markers(messages: list) -> str:
    """Render the full Inspector message chain with dimension dividers.

    Labels used:
      thought     — reasoning text (blue)
      conclusion  — DIMENSION_COMPLETE JSON block (purple), followed immediately by divider
      action      — tool calls (green)
      observation — tool results (green)

    Content is split at the text level so dividers appear right after END_DIMENSION,
    even when the model writes a Thought for the next dimension in the same response.
    """
    rows = []

    def _asst_block(label: str, text: str = "", tc_html: str = "") -> str:
        if label == "conclusion":
            css, cls = "content-conclusion", "msg-conclusion"
        elif label == "thought":
            css, cls = "content-assistant", "msg-thought"
        else:  # action and fallback
            css, cls = "content-assistant", "msg-assistant"
        content_html = f'<pre class="{css}">{_e(text)}</pre>' if text else ""
        return (
            f'<div class="msg {cls}">'
            f'<span class="role">{_e(label)}</span>{content_html}{tc_html}</div>'
        )

    def _render_content(text: str, tc_html: str = "") -> None:
        """Recursively render assistant text that may contain multiple DIMENSION_COMPLETE blocks."""
        if "DIMENSION_COMPLETE" not in text:
            # Strip ALL_DIMENSIONS_COMPLETE protocol marker; render the rest as thought or action
            display = text.replace("ALL_DIMENSIONS_COMPLETE", "").strip()
            if display:
                rows.append(_asst_block("thought", display))
            elif text.strip() == "ALL_DIMENSIONS_COMPLETE":
                rows.append(
                    f'<div class="msg" style="color:#bbb;font-size:0.8em;font-style:italic;">'
                    f'ALL_DIMENSIONS_COMPLETE</div>'
                )
            if tc_html:
                rows.append(_asst_block("action", tc_html=tc_html))
            return

        pre, conclusion, post = _split_at_dim_complete(text)
        if pre:
            rows.append(_asst_block("thought", pre))
        rows.append(_asst_block("conclusion", conclusion))
        rows.append(_dim_divider(conclusion))
        # Recurse: post may contain another DIMENSION_COMPLETE (model batched conclusions)
        _render_content(post, tc_html)

    for m in messages:
        role = m.get("role", "")
        label = m.get("react_role") or role
        content = m.get("content") or ""

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
            tc_html = "".join(
                f'<div class="tool-call">▶ <b>{_e(tc["function"]["name"])}</b>'
                f'({_e(tc["function"]["arguments"])})</div>'
                for tc in tool_calls
            ) if tool_calls else ""

            if "DIMENSION_COMPLETE" in content or tc_html:
                _render_content(content, tc_html)
            else:
                # Plain thought or ALL_DIMENSIONS_COMPLETE
                _render_content(content)

        elif role == "tool":
            rows.append(
                f'<div class="msg msg-tool">'
                f'<span class="role">{_e(label)}</span>'
                f'<pre class="content-tool">{_e(content)}</pre></div>'
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
    # Inspector self-determines tool usage per dimension; build badge list from planned only.
    # Tags are filled in after Inspector results are read below.
    dims_html_placeholder = " &nbsp; ".join(f'<code>{_e(d)}</code>' for d in planned) or "—"
    parts.append(f"""
<section>
  <h2>① Perceiver</h2>
  <p><b>Planned dimensions:</b> {dims_html_placeholder}</p>
  <p><b>Perception summary:</b> {_e(perceiver.get("perception_summary", ""))}</p>
  {_render_messages(perceiver.get("messages", []))}
</section>""")

    # ── Inspector (per dimension) ─────────────────────────────────────────────
    inspector_items = record.get("inspector", [])

    # Infer per-dimension tool usage from message chain snapshots.
    # Each dimension's snapshot grows cumulatively; the window for dim N is
    # messages[prev_len:cur_len]. If any tool message appears there, tools were used.
    def _infer_used_tools(items: list) -> list[bool]:
        result = []
        prev_len = 0
        for item in items:
            msgs = item.get("messages", [])
            window = msgs[prev_len:]
            used = any(m.get("role") == "tool" for m in window)
            result.append(used)
            prev_len = len(msgs)
        return result

    used_tools_flags = _infer_used_tools(inspector_items)

    # Dimension summary cards (no message chain per card)
    dim_cards = []
    for r, used_tools in zip(inspector_items, used_tools_flags):
        evidence_json = _compact_arrays(
            json.dumps(r.get("evidence", {}), indent=2, cls=NumpyEncoder, ensure_ascii=False)
        )
        dim_name = r["dimension"]
        if used_tools:
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
.msg-thought .role   { color: #92400e; }
.msg-assistant .role { color: #1a6fa8; }
pre.content-assistant { display: inline; background: none; border: none; padding: 0;
                        font-size: 0.88em; color: #1a1a1a; white-space: pre-wrap; word-break: break-all; }
.msg-conclusion .role { color: #6a1fb5; font-weight: 700; }
pre.content-conclusion { display: block; background: none; border: none;
                         padding: 0; margin-top: 4px; margin-left: 0;
                         font-size: 0.88em; color: #1a1a1a; white-space: pre-wrap; word-break: break-all; }
.tool-call { font-size: 0.85em; color: #1a6fa8; margin-left: 4px; }
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