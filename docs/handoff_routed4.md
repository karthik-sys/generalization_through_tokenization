# Launch: routed4 (decoupled head + learned weight + 2x context, stacked)

Code is pushed to `main` (commit `ff36ed1`). This doc is everything needed to launch it on
a pod — no context beyond this file required.

## What it is

Three separate fixes for the same problem, run together as ONE arm instead of three
ablations (cheaper: if it wins, pull the pieces apart later; if it doesn't, one run found
that out instead of three). The problem: every GradNorm-based attempt to improve routed's
switch-prediction accuracy (routed2, routed3, hybrid) regressed BPB/LAMBADA hard despite the
switch-accuracy gain — see `results/metrics.json`, e.g. routed2's LAMBADA ppl 1598 vs plain
routed's 47.76. routed4 targets the likely mechanism (switch loss competing with content
loss for backbone gradient) from three angles at once:

1. **Gradient-decoupled switch head** — switch prediction gets its own small per-domain
   classifier, fed the backbone's hidden state with `.detach()`. The shared backbone only
   ever receives gradient from content loss now. Verified with a unit test: an all-switch
   batch produces exactly zero backbone gradient.
2. **Learned switch weight** (Kendall et al. 2018 homoscedastic-uncertainty form) —
   `switch_loss/(2*sigma^2) + log(sigma)`, sigma initialized so the effective weight starts
   at exactly 50 (matching plain routed's fixed constant), but free to move. The `log(sigma)`
   term is a real regularizer against the weight collapsing to zero, verified numerically.
3. **2x context** (1024 → 2048) — more room between switches to re-establish domain state.

Standard cross-entropy throughout — no GradNorm balancer anywhere in this arm.

Model: `src/model/mot_routed_combined_model.py` (`MoTRoutedCombinedModel`). Config:
`LONGCTX_MODEL_CFG` in `src/model/stage2_config.py`. Fully wired into
`scripts/train_stage2_pod.py` (`_build_model`, both loop dispatches, argparse choices) and
`stage2_modal.py`'s `evaluate`/`evaluate_lambada` (accepts the same `--scale` mechanism used
for the large-scale runs, though routed4 itself uses `scale=base` — its size difference from
plain routed is context length, not param count).

## Launch

Any idle pod works — check `runpodctl get pod` or the RunPod dashboard for one not already
running `baseline --scale large` or `routed --scale large` (those two must keep running
uninterrupted overnight).

```bash
cd /workspace/repo && git pull origin main
```

Optional sanity check first (~30s, confirms the model builds and trains a few steps without
error on this GPU):

```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed4 --steps 30
```

Should print `routed4 (base) params: <some number>` then step timing with no traceback.

Then launch the real run, backgrounded so it survives an SSH disconnect:

```bash
nohup python3 scripts/train_stage2_pod.py train --arm routed4 --steps 150000 > train.log 2>&1 &
```

Checkpoints save every 1000 steps to `checkpoints/routed4_step*.pt` (only the newest is
kept). Resumes automatically from the latest checkpoint if the process restarts — no
special flag needed, just re-run the same `train` command.

## What to expect

- `adaptive controller ON` should print at start (routed4 is in the controller-enabled list
  — its learned switch-weight is a dynamic parameter, same destabilization risk profile as
  GradNorm's reweighting, so the spike-guard/plateau-rescue safety net applies).
- Held-out val CE logs every 10k steps (`EVAL_EVERY`).
- Roughly similar step time to plain routed's base-config runs, maybe somewhat slower from
  the 2x context (longer attention) - if it's dramatically slower, something's off, check
  `peak GPU mem after first step` in the calibrate output first.

## When it's done (or check in on progress)

Eval exactly like every other arm this session — pull the checkpoint, verify MD5 against
the pod's, push to the Modal volume, then:

```bash
python3 -m modal run stage2_modal.py --step evaluate --arm routed4 --steps 150000
python3 -m modal run stage2_modal.py --step evaluate-lambada --arm routed4 --steps 150000
```

Record into `results/metrics.json` via `src.eval.metrics.record()` (see any recent commit
touching that file for the exact call shape), then regenerate + republish the dashboard
(`python3 scripts/gen_dashboard.py`).

## Reserved, not launched

`routed5` (decoupled-head-only) and `routed6` (long-context-only) are fully wired and ready
to launch the same way if routed4's combined result is worth pulling apart into its
individual contributions - not run tonight, deliberately.
