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

ARM_META = {  # display order-ish + colors, single source
    "mot": ("MoT", "disjoint tables", "var(--mot)"),
    "routed": ("Routed", "mid-seq switching", "var(--routed)"),
    "baseline": ("Baseline", "unified 48k BPE", "var(--baseline)"),
    "sota": ("SOTA", "cl100k, 100k vocab", "var(--sota)"),
    "pooled": ("Pooled", "PMA/DANN + fitting loss", "var(--pooled)"),
    "hybrid": ("Hybrid", "GradNorm switch loss + blend", "var(--routed)"),
    "routed-large": ("Routed-large", "190M scale test", "var(--routed)"),
    "baseline-large": ("Baseline-large", "160M scale control", "var(--baseline)"),
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


def build() -> str:
    sb = scoreboard_rows()
    infl, updated = inflight_rows()
    tmpl = (REPO / "scripts" / "dashboard_template.html").read_text()
    return (tmpl
            .replace("{{SCOREBOARD_ROWS}}", sb)
            .replace("{{INFLIGHT_ROWS}}", infl)
            .replace("{{UPDATED}}", html.escape(updated)))


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT}")
    print(f"evaluated arms in store: {len(latest_per_arm())}   total result rows: {len(load_results())}")
    print("publish with the Artifact tool at the path above (same URL).")
