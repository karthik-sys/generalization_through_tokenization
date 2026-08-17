"""Regenerate the MoT dashboard from the results store + live pod status.

The dashboard is a static artifact (a sandboxed published page can't poll the pods), so
"live" means: this generator is cheap to re-run, and re-running it + republishing reflects
current state. It reads ONLY structured data -
  - results/metrics.json (via src.eval.metrics) : every evaluated arm's numbers
  - results/live_status.json                    : in-flight arm steps (refresh by pulling the
                                                   pods; kept separate so training status can
                                                   update without re-touching eval numbers)
- so there is never any hand-transcription of numbers into HTML (which is exactly where the
earlier silent dashboard-corruption bugs came from). Run: `python3 scripts/gen_dashboard.py`
then publish the printed path with the Artifact tool. Wrap it in a loop for auto-refresh.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.eval.metrics import latest_per_arm, load_results  # noqa: E402

OUT = Path("/private/tmp/claude-501/-Users-vk55-karthik-domain/"
           "1771c72c-fa4b-4f95-a0e0-28be70e38bd8/scratchpad/mot_dashboard.html")
LIVE = REPO / "results" / "live_status.json"
LOSS_CURVES = REPO / "results" / "loss_curves.json"

ARM_META = {  # display order-ish + colors, single source
    "mot": ("MoT", "disjoint tables", "var(--mot)"),
    "routed": ("Routed", "mid-seq switching", "var(--routed)"),
    "baseline": ("Baseline", "unified 48k BPE", "var(--baseline)"),
    "sota": ("SOTA", "cl100k, 100k vocab", "var(--sota)"),
    "pooled": ("Pooled", "PMA/DANN + fitting loss", "var(--pooled)"),
    "hybrid": ("Hybrid", "GradNorm switch loss + blend", "var(--routed)"),
    "routed2": ("Routed2", "GradNorm loss only", "var(--routed)"),
    "routed3": ("Routed3", "GradNorm + denser data", "var(--routed)"),
    "pooled2": ("Pooled2", "GradNorm + sparse routing", "var(--pooled)"),
    "routed-large": ("Routed-large", "190M scale test", "var(--routed)"),
    "baseline-large": ("Baseline-large", "160M scale control", "var(--baseline)"),
    "mot-large": ("MoT-large", "190M scale test", "var(--mot)"),
}


def _fmt(v, digits=3, dash="—"):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else dash


def _fmt_ppl(v):
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:,.1f}" if v >= 100 else f"{v:.2f}"


def scoreboard_rows() -> str:
    res = latest_per_arm()
    # rank: arms WITH a LAMBADA number first (by ppl asc - the headline generalization metric),
    # then arms without it by single-domain BPB. Never mix the two scales in one sort key -
    # a ppl of 48 and a BPB of 2.4 are not comparable numbers.
    def key(r):
        lam = r.get("lambada_ppl")
        return (0, lam) if isinstance(lam, (int, float)) else (1, r.get("bpb_single") or 9e9)
    order = sorted(res.values(), key=key)
    rows = []
    for r in order:
        name, tag, color = ARM_META.get(r["arm"] + ("-large" if r.get("scale") == "large" else ""),
                                        (r["arm"], "", "var(--text2)"))
        best_lam = isinstance(r.get("lambada_ppl"), (int, float)) and r["lambada_ppl"] == min(
            (x.get("lambada_ppl") for x in res.values() if isinstance(x.get("lambada_ppl"), (int, float))), default=9e9)
        lam_style = ' style="color:var(--good)"' if best_lam else ""
        rows.append(
            f'<tr><td><span class="dot" style="background:{color}"></span>{html.escape(name)} '
            f'<small class="mono" style="color:var(--text3)">{html.escape(tag)}</small></td>'
            f'<td class="num">{(r.get("params") or 0)/1e6:.1f}M</td>'
            f'<td class="num">{r["step"]//1000}k</td>'
            f'<td class="num">{_fmt(r.get("bpb_single"))}</td>'
            f'<td class="num">{_fmt(r.get("bpb_cross"))}</td>'
            f'<td class="num"{lam_style}><b>{_fmt_ppl(r.get("lambada_ppl"))}</b></td>'
            f'<td class="num">{(str(round(r["switch_accuracy"]*100))+"%") if isinstance(r.get("switch_accuracy"),(int,float)) else "—"}</td></tr>'
        )
    return "\n".join(rows)


def inflight_rows() -> str:
    live = json.loads(LIVE.read_text()) if LIVE.exists() else {"in_flight": []}
    rows = []
    for a in live.get("in_flight", []):
        pct = 100 * a["step"] / a["total"]
        state = a.get("state", "running")
        pill_cls = {"running": "pill live", "done": "pill done", "stopped": "pill warn"}.get(state, "pill live")
        suffix = {"done": " · DONE", "stopped": " · STOPPED"}.get(state, "")
        pill_txt = f'{a["step"]//1000}k / {a["total"]//1000}k · {pct:.0f}%{suffix}'
        rows.append(
            f'<tr><td><b>{html.escape(a["arm"])}</b> <small class="mono" style="color:var(--text3)">'
            f'{html.escape(a.get("base",""))} variant</small></td>'
            f'<td><span class="{pill_cls}">{pill_txt}</span></td>'
            f'<td>{html.escape(a.get("tests",""))}</td></tr>'
        )
    return "\n".join(rows), live.get("updated", "")


def live_loss_svg() -> tuple[str, str]:
    """SVG + legend for the in-flight arms' loss trend, pulled straight from each pod's log by
    refresh_dashboard.sh (results/loss_curves.json). This is a SEPARATE chart from the archived
    5-arch one above, not merged into it - the GradNorm variants (hybrid/pooled2/routed2/routed3)
    sum multiple weighted loss terms, so their raw loss values live on a different scale than the
    single-term arms and aren't comparable even to each other in absolute terms. Each series is
    instead normalized to its OWN early value (avg of its first up to 3 points = 1.0), so what's
    plotted is relative progress, not an absolute number - a shape comparison, not a scoreboard.
    """
    if not LOSS_CURVES.exists():
        return "", ""
    curves = json.loads(LOSS_CURVES.read_text())
    live = json.loads(LIVE.read_text()) if LIVE.exists() else {"in_flight": []}
    in_flight_arms = [a["arm"] for a in live.get("in_flight", [])]
    # only plot arms currently in the in-flight table, in that order, so the chart never shows
    # a stale arm the table itself has already dropped
    arms = [a for a in in_flight_arms if a in curves and curves[a]]

    x0, x1, y0, y1 = 44, 628, 12, 140  # plot area in the 640x160 viewBox
    max_step = 150000

    def norm(series):
        base = sum(l for _, l in series[:3]) / min(3, len(series))
        return [(s, l / base if base else 1.0) for s, l in series]

    lines = []
    legend = []
    for arm in arms:
        _, _, color = ARM_META.get(arm, (arm, "", "var(--text2)"))
        pts = norm(curves[arm])
        top = 1.35  # normalized-value at the top of the plot; clip noisy early spikes above it
        coords = []
        for step, val in pts:
            x = x0 + (x1 - x0) * min(step, max_step) / max_step
            yv = min(val, top)
            y = y1 - (y1 - y0) * yv / top
            coords.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" '
                      f'points="{" ".join(coords)}"/>')
        legend.append(f'<span><i style="background:{color}"></i>{html.escape(arm)}</span>')

    gridline_1 = y1 - (y1 - y0) * 1.0 / 1.35
    svg = (
        f'<svg viewBox="0 0 640 160" width="100%" style="max-width:660px;display:block;margin-top:6px" '
        f'role="img" aria-label="normalized loss trend for in-flight arms">'
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="var(--border)" stroke-width="0.5"/>'
        f'<line x1="{x0}" y1="{gridline_1:.1f}" x2="{x1}" y2="{gridline_1:.1f}" stroke="var(--border2)" stroke-width="0.5" stroke-dasharray="2,2"/>'
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="var(--border2)" stroke-width="0.5"/>'
        f'<text x="{x0-6}" y="{gridline_1+3:.1f}" text-anchor="end" font-family="var(--mono)" font-size="9" fill="var(--text3)">1.0x</text>'
        f'<text x="{x0-6}" y="{y1+3}" text-anchor="end" font-family="var(--mono)" font-size="9" fill="var(--text3)">0</text>'
        f'<text x="{x0}" y="{y1+14}" text-anchor="middle" font-family="var(--mono)" font-size="9" fill="var(--text3)">0</text>'
        f'<text x="{(x0+x1)/2:.0f}" y="{y1+14}" text-anchor="middle" font-family="var(--mono)" font-size="9" fill="var(--text3)">75k</text>'
        f'<text x="{x1}" y="{y1+14}" text-anchor="middle" font-family="var(--mono)" font-size="9" fill="var(--text3)">150k steps</text>'
        + "".join(lines) +
        '</svg>'
    )
    return svg, "\n".join(legend)


def build() -> str:
    sb = scoreboard_rows()
    infl, updated = inflight_rows()
    live_svg, live_legend = live_loss_svg()
    tmpl = (REPO / "scripts" / "dashboard_template.html").read_text()
    return (tmpl
            .replace("{{SCOREBOARD_ROWS}}", sb)
            .replace("{{INFLIGHT_ROWS}}", infl)
            .replace("{{UPDATED}}", html.escape(updated))
            .replace("{{LIVE_LOSS_SVG}}", live_svg)
            .replace("{{LIVE_LOSS_LEGEND}}", live_legend))


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT}")
    print(f"evaluated arms in store: {len(latest_per_arm())}   total result rows: {len(load_results())}")
    print("publish with the Artifact tool at the path above (same URL).")
