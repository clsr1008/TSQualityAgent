"""
Visualize Perceiver training samples as a single collapsible HTML page.

Usage:
    python -m training.visualize --input training/data/train.jsonl --n 10
    python -m training.visualize --input training/data/train.jsonl --n 20 --output training/inspect/train_vis.html
"""
import argparse
import base64
import html as _html
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _e(s) -> str:
    return _html.escape(str(s))


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1020px; margin: 0 auto; padding: 24px; color: #222; font-size: 14px; }
h1   { font-size: 1.15em; margin-bottom: 4px; color: #111; }
.page-meta { color: #999; font-size: 0.82em; margin-bottom: 20px; }

/* collapsible sample */
details { border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 8px; overflow: hidden; }
details[open] { border-color: #94b4d1; }
summary {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 14px; cursor: pointer; list-style: none;
  background: #f8fafc; user-select: none;
}
summary::-webkit-details-marker { display: none; }
summary:hover { background: #f0f5fa; }
.toggle-icon { font-size: 0.75em; color: #94a3b8; transition: transform 0.15s; flex-shrink: 0; }
details[open] .toggle-icon { transform: rotate(90deg); }
.summary-id { font-weight: 600; font-size: 0.85em; color: #334155; flex: 1;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.summary-badges { display: flex; gap: 5px; flex-shrink: 0; }
.summary-desc { font-size: 0.76em; color: #94a3b8; max-width: 300px;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 1; }

/* sample body */
.sample-body { padding: 14px 16px 16px; }
.section-title { font-size: 0.72em; font-weight: 700; text-transform: uppercase;
                 letter-spacing: .06em; color: #94a3b8; margin: 14px 0 8px; }
.section-title:first-child { margin-top: 0; }
img { width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; margin-bottom: 2px; }

/* label chips — three distinct palettes */
.label-row { display: flex; gap: 10px; flex-wrap: wrap; }
.label-box { border: 1px solid #e8edf2; border-radius: 6px; padding: 8px 12px;
             background: #fafbfc; flex: 1; min-width: 150px; }
.label-box .lbl { font-size: 0.70em; font-weight: 700; text-transform: uppercase;
                  letter-spacing: .05em; color: #b0b8c4; margin-bottom: 6px; }
.chip { display: inline-block; margin: 2px 3px; padding: 2px 9px;
        border-radius: 10px; font-size: 0.79em; font-weight: 600; }
/* target dims: slate/neutral */
.chip-target { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }
/* needs tool: amber */
.chip-tool   { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
/* reasoning only: violet */
.chip-reason { background: #f5f3ff; color: #5b21b6; border: 1px solid #ddd6fe; }

/* defect table */
table { width: 100%; border-collapse: collapse; font-size: 0.83em; margin-top: 4px; }
th { background: #f4f6f8; text-align: left; padding: 6px 10px; font-weight: 600;
     color: #64748b; border-bottom: 2px solid #e2e8f0; }
td { padding: 5px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
/* severity: heavy = amber (obvious/strong), light = teal (subtle/mild) — distinct from side red/blue */
.sev-heavy { color: #d97706; }
.sev-light { color: #0d9488; }
/* side: match plot line colors */
.side-A { color: #1d6fa5; font-weight: 600; }
.side-B { color: #b83c28; font-weight: 600; }
.params { color: #94a3b8; font-size: 0.82em; }
.tie-note { color: #b0b8c4; font-size: 0.84em; padding: 6px 0; }
"""


def _plot_b64(preview_A, preview_B) -> str:
    a_vals = [v if v is not None else float("nan") for v in preview_A]
    b_vals = [v if v is not None else float("nan") for v in preview_B]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(a_vals, label="A", color="#3a80b8", linewidth=1.2)
    ax.plot(b_vals, label="B", color="#b83c28", alpha=0.85, linewidth=1.2)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _render_sample_block(sample: dict, index: int) -> str:
    sid = sample["sample_id"]
    inp = sample["input"]
    labels = sample["labels"]
    meta = sample["meta"]

    target_dims = labels["target_dimensions"]
    tool_req = set(labels["tool_required"])
    defects = meta["defect_details"]
    desc = inp.get("dataset_description", "")

    b64 = _plot_b64(inp["preview_A"], inp["preview_B"])

    # Label chips
    target_chips = " ".join(
        f'<span class="chip chip-target">{_e(d)}</span>' for d in target_dims
    ) or '<span style="color:#c0c8d4">tie — no defects</span>'

    tool_chips = " ".join(
        f'<span class="chip chip-tool">{_e(d)}</span>'
        for d in target_dims if d in tool_req
    ) or '<span style="color:#c0c8d4">none</span>'

    reason_chips = " ".join(
        f'<span class="chip chip-reason">{_e(d)}</span>'
        for d in target_dims if d not in tool_req
    ) or '<span style="color:#c0c8d4">none</span>'

    # Summary-line mini badges
    n_dims = len(target_dims)
    n_tool = len([d for d in target_dims if d in tool_req])
    n_reason = n_dims - n_tool
    mini = lambda cls, txt: f'<span class="chip {cls}" style="font-size:0.72em;padding:1px 7px">{txt}</span>'
    badges = mini("chip-target", f"{n_dims}d")
    if n_tool:
        badges += mini("chip-tool", f"{n_tool}T")
    if n_reason:
        badges += mini("chip-reason", f"{n_reason}R")

    # Defect table
    if defects:
        rows = ""
        for d in defects:
            sev_cls = "sev-heavy" if d["severity"] == "heavy" else "sev-light"
            side_cls = f"side-{d['side']}"
            meta_str = ", ".join(
                f"{k}={v}" for k, v in d["metadata"].items()
                if k not in ("indices", "values", "keypoints_x", "keypoints_y", "scale_factors")
            )
            rows += (
                f'<tr><td><b>{_e(d["dimension"])}</b></td>'
                f'<td class="{sev_cls}">{_e(d["severity"])}</td>'
                f'<td class="{side_cls}">{_e(d["side"])} degraded</td>'
                f'<td class="params">{_e(meta_str)}</td></tr>'
            )
        defect_section = f"""<div class="section-title">Injected Defects</div>
<table>
  <tr><th>Dimension</th><th>Severity</th><th>Side</th><th>Parameters</th></tr>
  {rows}
</table>"""
    else:
        defect_section = '<div class="tie-note">No defects injected — tie case.</div>'

    return f"""<details>
  <summary>
    <span class="toggle-icon">▶</span>
    <span class="summary-id">#{index:03d} &nbsp; {_e(sid)}</span>
    <span class="summary-badges">{badges}</span>
    <span class="summary-desc">{_e(desc)}</span>
  </summary>
  <div class="sample-body">
    <div class="section-title">Series Preview</div>
    <img src="data:image/png;base64,{b64}" alt="series plot">

    <div class="section-title">Labels</div>
    <div class="label-row">
      <div class="label-box">
        <div class="lbl">Target Dimensions</div>
        {target_chips}
      </div>
      <div class="label-box">
        <div class="lbl">Needs Tool</div>
        {tool_chips}
      </div>
      <div class="label-box">
        <div class="lbl">Reasoning Only</div>
        {reason_chips}
      </div>
    </div>

    {defect_section}
  </div>
</details>"""


def render_dataset_html(samples: list, title: str = "Perceiver Training Samples") -> str:
    """Render a list of samples into a single collapsible HTML page."""
    blocks = "\n".join(_render_sample_block(s, i) for i, s in enumerate(samples))
    n = len(samples)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{_e(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>{_e(title)}</h1>
  <p class="page-meta">{n} sample{"s" if n != 1 else ""} &mdash; click to expand</p>
  {blocks}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize Perceiver training samples as HTML")
    parser.add_argument("--input", type=str, required=True, help="JSONL file to read from")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--output", type=str, default=None, help="Output HTML file (default: <input>_vis.html)")
    args = parser.parse_args()

    with open(args.input) as f:
        lines = [l.strip() for l in f if l.strip()]

    n_show = min(args.n, len(lines))
    samples = [json.loads(line) for line in lines[:n_show]]

    out_path = args.output or os.path.splitext(args.input)[0] + "_vis.html"
    title = os.path.basename(args.input)

    print(f"Rendering {n_show} samples → {out_path}")
    html = render_dataset_html(samples, title=title)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Done.")


if __name__ == "__main__":
    main()
