"""Regenerate the MoT dashboard's STAGE2_CURVES block from real training data and rewrite it
in place, safely.

Why this exists: an earlier ad-hoc `for line in file: if line matches "mot:" or "baseline:":
replace it` patch silently corrupted two OTHER objects in the dashboard (`models` and
`stage2Models`) that happen to share those same key names for an unrelated purpose (per-arm
parameter breakdowns, not loss curves) - a blind per-line regex has no way to know which
object it's inside. That bug produced no visible error until a JS syntax check caught it; the
whole inline <script> would otherwise have silently failed to run.

This script never does line-by-line regex over the whole file. It locates the *specific*
`const STAGE2_CURVES = { ... };` statement by brace-matching from its declaration (see
_replace_named_const), and replaces only that span. Structurally impossible to touch
`models` or `stage2Models`, whatever their internal shape happens to be.

Data source: checkpoint `history` fields on the Modal volume `mot-stage2-data` (survives
relaunches/log loss, unlike local log files - see dump_history.py, the tool this factors out
of). Pass --arms to control which arms get refreshed; arms not listed keep their existing
curve in the dashboard untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DASHBOARD_PATH = Path(
    "/private/tmp/claude-501/-Users-vk55-karthik-domain/"
    "1771c72c-fa4b-4f95-a0e0-28be70e38bd8/scratchpad/mot_dashboard.html"
)
COLORS = {
    "mot": "var(--math)", "baseline": "var(--science)", "sota": "var(--text-dim)",
    "routed": "var(--code)", "pooled": "var(--nlp)",
}
DUMP_SCRIPT = Path(__file__).parent / "_dump_checkpoint_history.py"


def fetch_checkpoint_histories(arms: list[str], stride: int = 4000) -> dict[str, list[tuple[int, float]]]:
    """Reconstructs continuous per-arm (step, loss) series from checkpoint `history` fields
    on the Modal volume. Requires the `modal` CLI to be authed locally."""
    cmd = ["python3", "-m", "modal", "run", str(DUMP_SCRIPT), "--arms", ",".join(arms), "--stride", str(stride)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=DUMP_SCRIPT.parent)
    m = re.search(r"DUMP_JSON_START\n(.*)\nDUMP_JSON_END", out.stdout, re.S)
    if not m:
        print(out.stdout[-2000:], file=sys.stderr)
        print(out.stderr[-2000:], file=sys.stderr)
        raise RuntimeError("dump job did not produce DUMP_JSON_START/END - see output above")
    return json.loads(m.group(1))


def smooth(vals: list[float], w: int = 8) -> list[float]:
    return [sum(vals[max(0, i - w):min(len(vals), i + w + 1)]) / (min(len(vals), i + w + 1) - max(0, i - w))
            for i in range(len(vals))]


def downsample(steps: list[int], losses: list[float], k: int = 70) -> tuple[list[int], list[float]]:
    if len(steps) <= k:
        return steps, losses
    idx = sorted(set(round(i * (len(steps) - 1) / (k - 1)) for i in range(k)))
    return [steps[i] for i in idx], [losses[i] for i in idx]


def build_curve_line(arm: str, points: list[list[float]]) -> str:
    steps = [round(p[0]) for p in points]
    losses = smooth([p[1] for p in points])
    steps, losses = downsample(steps, losses)
    losses = [round(l, 3) for l in losses]
    return f"  {arm}: {{ color: '{COLORS[arm]}', steps: {steps}, losses: {losses} }},"


def _replace_named_const(html: str, const_name: str, new_body_lines: dict[str, str]) -> str:
    """Replace only the named per-arm entries inside `const {const_name} = { ... };`, located
    by brace-matching from the declaration - never a blind whole-file line scan. Entries not
    in new_body_lines are left exactly as they are."""
    decl = f"const {const_name} = "
    start = html.index(decl)
    body_start = start + len(decl)
    depth, end = 0, -1
    for i, ch in enumerate(html[body_start:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = body_start + i + 1
                break
    if end == -1:
        raise RuntimeError(f"could not find matching close brace for `{decl}`")
    block = html[body_start:end]

    for arm, new_line in new_body_lines.items():
        pattern = re.compile(rf"^\s*{re.escape(arm)}:\s*\{{.*?\}},?\s*$", re.M)
        if not pattern.search(block):
            raise RuntimeError(f"arm `{arm}` not found as a top-level entry in {const_name} - refusing to guess")
        block = pattern.sub(lambda m, nl=new_line: nl, block, count=1)

    return html[:body_start] + block + html[end:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="mot,baseline,pooled", help="comma-separated arms to refresh")
    ap.add_argument("--stride", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true", help="print the new lines, don't write the file")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",")]

    print(f"fetching checkpoint histories for {arms} ...")
    data = fetch_checkpoint_histories(arms, args.stride)
    new_lines = {arm: build_curve_line(arm, pts) for arm, pts in data.items() if pts}

    for arm, line in new_lines.items():
        n_pts = len(re.search(r"steps: \[(.*?)\]", line).group(1).split(","))
        print(f"  {arm}: {n_pts} points -> {line[:100]}...")

    if args.dry_run:
        return

    html = DASHBOARD_PATH.read_text()
    html = _replace_named_const(html, "STAGE2_CURVES", new_lines)
    DASHBOARD_PATH.write_text(html)
    print(f"wrote {DASHBOARD_PATH}")
    print("NOTE: this only updates the file on disk - publish it with the Artifact tool "
          "(same file_path) to push the update live, and validate JS syntax first.")


if __name__ == "__main__":
    main()
