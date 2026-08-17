#!/usr/bin/env bash
# Refresh live_status.json from the pods, then regenerate the dashboard HTML.
#
# Each cycle the SSH host:port of a pod can change (they change on restart), so this script
# does NOT hardcode them - the caller (an agent that can call the RunPod list-pods tool)
# passes the current direct-SSH endpoints of the RUNNING pods as args:
#
#   scripts/refresh_dashboard.sh HOST:PORT [HOST:PORT ...]
#
# For each endpoint it SSHes in, discovers which arm is training from the live process's
# --arm/--scale flags (NOT the pod name, which goes stale after repurposing), reads the last
# logged step, and notes whether the training process is still alive. It then writes
# results/live_status.json and runs gen_dashboard.py. Publishing the HTML is the caller's job
# (the Artifact tool isn't callable from bash).
#
# Arms with no live --arm process are reported state=stopped (crashed / finished / killed).
set -euo pipefail
cd "$(dirname "$0")/.."

SSHOPT=(-o StrictHostKeyChecking=no -o ConnectTimeout=12
        -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o BatchMode=yes)

# static per-arm metadata (base architecture + what the variant tests) so cold refreshes keep
# the descriptions. Keyed by the discovered arm id (mot/baseline/routed/routed2/routed3/pooled2/
# hybrid, plus "<arm>-large" for scale runs).
python3 - "$@" <<'PYEOF'
import json, subprocess, sys, datetime, re

SSHOPT = ["-o","StrictHostKeyChecking=no","-o","ConnectTimeout=12",
          "-o","ServerAliveInterval=5","-o","ServerAliveCountMax=3","-o","BatchMode=yes"]

META = {  # arm -> (base architecture, tests-description)
    "routed":  ("Routed", "mid-sequence domain switching (the 150k champion arm)"),
    "routed2": ("Routed", "+ GradNorm loss ONLY (isolates the loss fix)"),
    "routed3": ("Routed", "+ GradNorm + 2.4x denser cross-domain data"),
    "hybrid":  ("Routed", "+ GradNorm switch-loss + 60% natural-data blend"),
    "pooled2": ("Pooled", "+ GradNorm + sparse top-2/16 routing, no confidence head"),
    "mot":     ("MoT",    "disjoint per-domain tables"),
    "routed-large":   ("Routed",   "SCALE TEST - 190M (2.1x): does the champion survive scale"),
    "baseline-large": ("Baseline", "SCALE TEST - 160M unified-BPE control"),
    "mot-large":      ("MoT",      "SCALE TEST - 190M disjoint-table scale run"),
}
TOTAL = 150000

REMOTE = (
    'cd /workspace/repo 2>/dev/null || exit 9; '
    'proc=$(ps aux | grep train_stage2 | grep -v grep); '
    'arm=$(echo "$proc" | grep -oE -- "--arm [a-z0-9]+" | head -1 | awk "{print \\$2}"); '
    'scale=$(echo "$proc" | grep -oE -- "--scale [a-z]+" | head -1 | awk "{print \\$2}"); '
    'if [ -n "$arm" ]; then '
    # live process: read step from the TAIL of its log only. A full-file grep is unsafe here -
    # a stray non-UTF8/NUL byte earlier in the file (write-flush races are common on these pods)
    # makes grep treat it as binary and silently stop scanning partway through, returning a
    # stale step instead of the true last one. Reading only the last 4KB sidesteps that: the
    # freshest write is always at the end, so we never touch whatever is corrupt upstream.
    '  alive=1; '
    '  log=$([ "$scale" = large ] && echo train_large.log || echo train.log); '
    '  step=$(tail -c 4096 "$log" 2>/dev/null | tr "\\r" "\\n" | grep -aoE "step [0-9]+" | tail -1 | grep -oE "[0-9]+"); '
    'else '
    # no live process: recover arm+step from the NEWEST checkpoint (large_<arm>_stepN.pt or <arm>_stepN.pt)
    '  alive=0; '
    '  ck=$(ls -t checkpoints/*.pt 2>/dev/null | head -1); '
    '  base=$(basename "$ck" .pt); '            # e.g. large_routed_step150000  or  pooled2_step124000
    '  step=$(echo "$base" | grep -oE "[0-9]+$"); '
    '  rest=${base%_step*}; '                   # large_routed  or  pooled2
    '  case "$rest" in large_*) scale=large; arm=${rest#large_};; *) arm=$rest;; esac; '
    '  log=$([ "$scale" = large ] && echo train_large.log || echo train.log); '
    'fi; '
    'echo "ARM=$arm SCALE=$scale ALIVE=$alive STEP=$step"; '
    # loss history for the chart: full-log scan is safe now that -a forces text mode (the
    # earlier corruption bug was grep's binary heuristic silently truncating a scan WITHOUT
    # -a, not the full-file read itself) - so pull the whole step/loss series, not just the
    # tail, and let the caller downsample it.
    'echo "===LOSS_START==="; grep -aoE "step [0-9]+/[0-9]+  loss=[0-9.]+" "$log" 2>/dev/null; echo "===LOSS_END==="'
)

def downsample(points, n=50):
    """Evenly-spaced subset of (step, loss) pairs, always keeping the first and last."""
    if len(points) <= n:
        return points
    idx = [round(i * (len(points) - 1) / (n - 1)) for i in range(n)]
    seen_i, out = set(), []
    for i in idx:
        if i not in seen_i:
            seen_i.add(i); out.append(points[i])
    return out

try:
    prev_rows = json.load(open("results/live_status.json")).get("in_flight", [])
except FileNotFoundError:
    prev_rows = []
prev = {a["arm"]: a["step"] for a in prev_rows}
prev_full = {a["arm"]: a for a in prev_rows}

try:
    loss_curves = json.load(open("results/loss_curves.json"))
except FileNotFoundError:
    loss_curves = {}

seen = {}
for ep in sys.argv[1:]:
    host, port = ep.rsplit(":", 1)
    try:
        out = subprocess.run(["ssh", *SSHOPT, "-p", port, f"root@{host}", REMOTE],
                             capture_output=True, text=True, timeout=40).stdout
    except Exception as e:
        print(f"  ! {ep}: {e}", file=sys.stderr); continue
    m = dict(re.findall(r"(ARM|SCALE|ALIVE|STEP)=(\S*)", out))
    arm = m.get("ARM", "")
    if not arm:
        print(f"  ? {ep}: no --arm process (idle/finished pod), skipping", file=sys.stderr); continue
    if m.get("SCALE") == "large":
        arm = f"{arm}-large"
    step = int(m["STEP"]) if m.get("STEP") else 0
    alive = m.get("ALIVE") == "1"

    loss_block = out.split("===LOSS_START===\n", 1)
    if len(loss_block) == 2:
        body = loss_block[1].split("===LOSS_END===", 1)[0]
        pts = [(int(s), float(l)) for s, l in
               re.findall(r"step (\d+)/\d+\s+loss=([0-9.]+)", body)]
        if pts:
            loss_curves[arm] = downsample(pts)
    # A running arm's step should only go up. A lower reading than last time means the read was
    # corrupted (see the tail -c comment above) rather than the run rewinding - keep the last
    # known-good step in that case instead of publishing a regression.
    if alive and arm in prev and step < prev[arm]:
        print(f"  ! {ep}: {arm} read step {step} < previous {prev[arm]}, keeping previous (stale/corrupt read)", file=sys.stderr)
        step = prev[arm]
    base, tests = META.get(arm, ("?", ""))
    # keep the furthest-along sighting if an arm appears on two endpoints
    if arm in seen and seen[arm]["step"] >= step:
        continue
    seen[arm] = {"arm": arm, "base": base, "step": step, "total": TOTAL,
                 "state": "running" if alive else "stopped",
                 "tests": tests + ("" if alive else " - process stopped, eval-ready or restart")}

# Carry forward arms this cycle's endpoints didn't cover at all (e.g. a pod that fully
# terminated, so there's no SSH endpoint left to sweep) - a pod going away entirely is not
# the same as the arm no longer existing. Only "done"/eval-recorded rows are worth keeping
# indefinitely; a "stopped" row not re-sighted just means we lost reach to a still-dead pod,
# which is also worth keeping rather than silently deleting.
for name, row in prev_full.items():
    if name not in seen:
        seen[name] = row
        print(f"  = {name}: not covered this cycle (pod likely terminated), carrying forward "
              f"last known state (step {row['step']}, {row['state']})", file=sys.stderr)

# order: large scale runs first, then by step desc
order = sorted(seen.values(),
               key=lambda a: (0 if a["arm"].endswith("-large") else 1, -a["step"]))
live = {"updated": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "in_flight": order}
open("results/live_status.json", "w").write(json.dumps(live, indent=2))
open("results/loss_curves.json", "w").write(json.dumps(loss_curves, indent=2))
print(f"live_status.json: {len(order)} arms -> " +
      ", ".join(f'{a["arm"]}@{a["step"]//1000}k{"" if a["state"]=="running" else "(stopped)"}'
                for a in order))
print(f"loss_curves.json: {len(loss_curves)} arms with loss history")
PYEOF

python3 scripts/gen_dashboard.py | tail -1
