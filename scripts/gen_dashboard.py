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
    # names describe WHAT was tried, not the trial number it happened to launch as - "routed7"
    # tells you nothing about the run; "GPT-2-Corpus" does. Internal --arm codes (routed2..10)
    # stay as-is (checkpoint filenames + live pod flags depend on them), this is display-only.
    "mot": ("MoT", "disjoint tables", "var(--mot)"),
    "routed": ("Routed", "mid-seq switching (champion)", "var(--routed)"),
    "baseline": ("Baseline", "unified 48k BPE", "var(--baseline)"),
    "sota": ("SOTA", "cl100k, 100k vocab", "var(--sota)"),
    "pooled": ("Pooled", "PMA/DANN + fitting loss", "var(--pooled)"),
    "hybrid": ("Hybrid", "GradNorm switch loss + 60% natural-data blend", "var(--routed)"),
    "routed2": ("GradNorm-only", "GradNorm switch loss, data unchanged", "var(--routed)"),
    "routed3": ("GradNorm+Dense", "GradNorm + max cross-domain density", "var(--routed)"),
    "pooled2": ("Pooled2", "GradNorm + sparse top-2/16 routing", "var(--pooled)"),
    "routed4": ("Decoupled+LearnedWeight", "gradient-decoupled head + Kendall uncertainty weight", "var(--routed)"),
    "routed5": ("Decoupled-Head", "gradient-decoupled switch head only (reserved ablation)", "var(--routed)"),
    "routed6": ("Long-Context", "2x context (2048), otherwise unchanged (reserved ablation)", "var(--routed)"),
    "routed7-large": ("GPT-2-Corpus", "nlp sourced from OpenWebText, large scale", "var(--routed)"),
    "routed8": ("4x-Data-Volume", "nlp OWT-sourced, 4x steps (600k) - matches llm.c's 10B tok/domain budget", "var(--routed)"),
    "routed9": ("Books+Warmstart-89M", "PG-19 books, ~55% nlp mixture, warm-started from champion", "var(--routed)"),
    "routed10-large": ("Books+Warmstart-190M", "PG-19 books, ~55% nlp mixture, warm-started from routed7", "var(--routed)"),
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
        pill_cls = {"running": "pill live", "done": "pill done", "stopped": "pill warn", "pending": "pill"}.get(state, "pill live")
        suffix = {"done": " · DONE", "stopped": " · STOPPED", "pending": " · PENDING"}.get(state, "")
        pill_txt = f'{a["step"]//1000}k / {a["total"]//1000}k · {pct:.0f}%{suffix}'
        name, _, _ = ARM_META.get(a["arm"], (a["arm"], "", None))
        rows.append(
            f'<tr><td><b>{html.escape(name)}</b> <small class="mono" style="color:var(--text3)">'
            f'{html.escape(a.get("base",""))} variant</small></td>'
            f'<td><span class="{pill_cls}">{pill_txt}</span></td>'
            f'<td>{html.escape(a.get("tests",""))}</td></tr>'
        )
    return "\n".join(rows), live.get("updated", "")


def live_loss_cards() -> str:
    """One small, interactive chart PER in-flight arm, pulled from each pod's log by
    refresh_dashboard.sh (results/loss_curves.json).

    Each card is scaled to that arm's OWN step and loss range, not a shared 0-150k axis. That
    matters because several arms only have a short recent window of history: their train.log
    gets overwritten (not appended) on every manual CLI restart, so a resumed arm's log only
    covers the steps since its last resume - not the full run. On a shared axis those short
    segments collapse into an illegible dot; scaled to their own range they're a normal-looking
    curve. Raw loss (not normalized) is plotted since these are single-arm views now - the
    earlier normalized/overlaid design made cross-arm comparison possible but sacrificed
    per-arm legibility, and the note below already says raw values aren't comparable across
    arms anyway (GradNorm variants sum different weighted loss terms).

    Hover/tap a point (via the shared <script> at the bottom of the page) to see its exact
    step and loss in a tooltip - plain SVG + vanilla JS, no chart library.
    """
    if not LOSS_CURVES.exists():
        return ""
    curves = json.loads(LOSS_CURVES.read_text())
    live = json.loads(LIVE.read_text()) if LIVE.exists() else {"in_flight": []}
    in_flight_arms = [a["arm"] for a in live.get("in_flight", [])]
    arms = [a for a in in_flight_arms if a in curves and curves[a]]

    w, h = 280, 130
    x0, x1, y0, y1 = 34, w - 10, 10, h - 20

    cards = []
    for arm in arms:
        name, tag, color = ARM_META.get(arm, (arm, "", "var(--text2)"))
        pts = curves[arm]
        steps = [s for s, _ in pts]
        losses = [l for _, l in pts]
        smin, smax = min(steps), max(steps)
        lmin, lmax = min(losses), max(losses)
        pad = (lmax - lmin) * 0.1 or lmax * 0.05 or 1
        lo, hi = lmin - pad, lmax + pad
        srange = (smax - smin) or 1

        def sx(s):
            return x0 + (x1 - x0) * (s - smin) / srange

        def sy(l):
            return y1 - (y1 - y0) * (l - lo) / (hi - lo)

        coords = [f"{sx(s):.1f},{sy(l):.1f}" for s, l in pts]
        circles = "".join(
            f'<circle class="pt" cx="{sx(s):.1f}" cy="{sy(l):.1f}" r="7" fill="transparent" '
            f'data-step="{s}" data-loss="{l:.4f}"/>'
            for s, l in pts
        )
        n_steps_covered = smax - smin
        truncated_note = (
            f'<div class="note" style="margin-top:4px;font-size:10.5px">'
            f'log only covers its last {n_steps_covered:,} steps (restarted since — earlier '
            f'history was overwritten)</div>'
            if n_steps_covered < 5000 else ""
        )
        cards.append(f'''
<div class="loss-card">
  <div class="loss-card-title"><span class="dot" style="background:{color}"></span>{html.escape(name)}
    <small class="mono" style="color:var(--text3)">step {smax:,} · loss {losses[-1]:.3f}</small></div>
  <svg viewBox="0 0 {w} {h}" width="100%" style="display:block" data-arm="{html.escape(arm)}"
       role="img" aria-label="loss trend for {html.escape(arm)}, step {smin} to {smax}">
    <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="var(--border)" stroke-width="0.5"/>
    <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="var(--border)" stroke-width="0.5"/>
    <text x="{x0-4}" y="{y0+7}" text-anchor="end" font-family="var(--mono)" font-size="8.5" fill="var(--text3)">{hi:.2f}</text>
    <text x="{x0-4}" y="{y1+3}" text-anchor="end" font-family="var(--mono)" font-size="8.5" fill="var(--text3)">{lo:.2f}</text>
    <text x="{x0}" y="{y1+13}" text-anchor="start" font-family="var(--mono)" font-size="8.5" fill="var(--text3)">{smin:,}</text>
    <text x="{x1}" y="{y1+13}" text-anchor="end" font-family="var(--mono)" font-size="8.5" fill="var(--text3)">{smax:,}</text>
    <polyline fill="none" stroke="{color}" stroke-width="1.6" points="{" ".join(coords)}"/>
    {circles}
    <circle class="hover-dot" r="3" fill="{color}" style="display:none"/>
  </svg>
  {truncated_note}
</div>''')

    return (
        '<div class="loss-tooltip" id="lossTooltip" style="display:none"></div>\n'
        '<div class="loss-grid">' + "".join(cards) + '</div>'
    )


def build() -> str:
    sb = scoreboard_rows()
    infl, updated = inflight_rows()
    loss_cards = live_loss_cards()
    tmpl = (REPO / "scripts" / "dashboard_template.html").read_text()
    return (tmpl
            .replace("{{SCOREBOARD_ROWS}}", sb)
            .replace("{{INFLIGHT_ROWS}}", infl)
            .replace("{{UPDATED}}", html.escape(updated))
            .replace("{{LIVE_LOSS_CARDS}}", loss_cards))


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT}")
    print(f"evaluated arms in store: {len(latest_per_arm())}   total result rows: {len(load_results())}")
    print("publish with the Artifact tool at the path above (same URL).")
