# Launch: round 2 (routed15-18) — informed by routed11-14's post-mortem

Code is pushed to `main`. This doc is everything needed to launch all four — no other context
required.

## What it is, and why

routed11/14 (the copy-gate bet, at 89M and 190M) both regressed sharply against their parents,
despite a diagnostic that strongly favored the lever (82.8% of routed8's LAMBADA errors were
copy-failures). A real forensic pass (loading the trained checkpoints directly, running actual
LAMBADA examples through the model, and comparing the copy mechanism's own predictions against
the plain vocab head's) found something specific: **the copy mechanism itself never hurt** - 0
out of 60 real examples where blending in the copy distribution flipped a correct answer to
wrong, and it actively helped in 15-20% of cases. What actually broke was the **underlying
vocab head**, which collapsed to near-zero first-token accuracy during the 100k-step joint
training - most likely because the copy-attention computation reads hidden states directly from
the shared backbone, so its gradient flows straight back through the same representations the
vocab head depends on, even at a throttled (0.3x) LR.

routed12/13 (deep-experts, precision-head) both showed small real gains, landing at nearly
identical numbers (18.99 vs 18.94 ppl) despite being mechanistically unrelated - suspicious
enough that the gain might be "100k more steps of stable training" rather than either
mechanism specifically.

Four recipes for round 2, **deliberately not another blind repeat of copy-gate** - each tests a
different lever:

1. **routed15 (control)** - isolates whether routed12/13's gains were real.
2. **routed16 (copy-gate v2)** - the actual fix for routed11/14's failure mode, not a retry.
3. **routed17 (diet phase 2)** - the data-distribution lever, confirmed to work twice already
   (source-matching, volume), applied a third way (mixture share) without a reinit confound.
4. **routed18 (copy-structure mining)** - attacks the same error class as Bet 1, but via data
   instead of architecture, avoiding the backbone-contamination risk entirely.

All four are continued training from **routed8@600000** (the finished, final checkpoint - not
575000). Each was verified locally before being handed off: model dispatch, warm-start key
matching, and (for routed15/16 specifically) the actual optimizer/freezing behavior - confirmed
via a real forward/backward pass that routed16 has exactly 6 trainable tensors with the
backbone genuinely receiving zero gradient, and routed15's optimizer covers 100% of the model
in one flat param group.

## Prerequisite: routed8@600000 checkpoint

All four warm-start from it directly (unlike round 1, none of these need routed7's checkpoint).

```bash
ls checkpoints/routed8_step*.pt
```

If missing, pull it from wherever it's checkpointed or from the Modal volume
(`modal volume get mot-stage2-data checkpoints/routed8_step600000.pt checkpoints/routed8_step600000.pt`).

## Launch (all four)

Same pattern: `git pull` → `calibrate --steps 30` (sanity check) → `train`. All target
**100,000 steps** (`ROUND2_STEPS` in `stage2_config.py`).

```bash
cd /workspace/repo && git pull origin main
```

**Calibrate never exercises warm-start** (same as every prior warm-started arm) - it builds a
fresh random-init model purely to time the loop. The warm-start/freeze checks below happen on
the **first `train` launch**, watch that log, not calibrate's.

**routed15 (control - no new module):**
```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed15 --steps 30
nohup python3 scripts/train_stage2_pod.py train --arm routed15 --steps 100000 > train_routed15.log 2>&1 &
```
On first launch, check for a warm-start line referencing `checkpoints/routed8_step600000.pt`
with **97 tensors loaded, 0 skipped** (identical architecture to routed8, nothing new). This is
the single most valuable result of the round - if its final LAMBADA lands near 18.9-19.0 ppl
(matching routed12/13), their "wins" were never about the mechanism.

**routed16 (copy-gate v2, frozen backbone):**
```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed16 --steps 30
nohup python3 scripts/train_stage2_pod.py train --arm routed16 --steps 100000 > train_routed16.log 2>&1 &
```
On first launch, check for the warm-start line (same as routed11's - 97 loaded, 0 skipped,
same architecture). **Then check that training is actually fast and stable** - with only 6
trainable tensors (copy_q/copy_k/copy_gate's weight+bias), this should run noticeably faster
per step than routed11 did (far fewer params getting gradients), and loss should NOT show the
kind of divergence that would suggest a bug. If loss looks flat-lined from step 1 (no movement
at all), double check the optimizer actually has non-empty params - `trainable` should be a
list of exactly 6 tensors, logged nowhere currently, so if in doubt add a quick print of
`sum(p.numel() for p in trainable)` before trusting a silent run.

**routed17 (diet phase 2 - nlp upweighted, no reinit):**
```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed17 --steps 30
nohup python3 scripts/train_stage2_pod.py train --arm routed17 --steps 100000 > train_routed17.log 2>&1 &
```
Uses the SAME nlp-vs-rest differential LR path routed9/10 used (nlp tables full LR, everything
else at `COOLDOWN_BACKBONE_LR_SCALE`) - no new code, just no reinit this time since it's the
same OWT tokenizer routed8 already uses. Expect ~70% of tokens to come from the nlp domain
(vs the standard ~25%) - code/math/science will still appear, just less often.

**routed18 (copy-structure mining):**
```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed18 --steps 30
nohup python3 scripts/train_stage2_pod.py train --arm routed18 --steps 100000 > train_routed18.log 2>&1 &
```
Same as routed17, plus every nlp document has to pass `_is_copy_structured` (a real content
word recurs with a real gap) before it's used at all. **This can be slow to warm up** - OpenWebText
pages are often short, and the filter is real (not every document qualifies), so the first
`next()` calls on the nlp stream may take noticeably longer than routed17's before the first
training step actually starts. If `calibrate` seems to hang for more than ~2-3 minutes with no
output at all, that's likely just the filter working through unqualifying documents, not a bug
- give it a few minutes before assuming something's wrong.

## Eval (all four)

Same flow as every checkpoint tonight - pull, MD5-verify if paranoid, push to the Modal volume,
then:

```bash
for arm in routed15 routed16 routed17 routed18; do
  python3 -m modal run stage2_modal.py --step evaluate --arm $arm --steps 100000
  python3 -m modal run stage2_modal.py --step evaluate-lambada --arm $arm --steps 100000
done
```

All four are wired into `stage2_modal.py`'s `evaluate`/`evaluate-lambada` the same way as
round 1 - base scale, OWT tokenizer dir, correct model class (routed15/17/18 → plain
`MoTRoutedModel`, routed16 → `MoTRoutedCopyGateModel`, same as routed11/14 - the gate's init
value doesn't matter for eval since `load_state_dict` overwrites it with the real trained
weights regardless of what it was constructed with).

Record via `src.eval.metrics.record()` with the matching `arm=`, then regenerate + republish
the dashboard. `ARM_META` in `gen_dashboard.py` doesn't have display names for these four yet -
cosmetic gap, not blocking.

## Cost / time, roughly

100,000 steps each. routed15/17/18 should run at roughly routed11-13's observed rate
(~0.1s/step, ~2.5-3 GPU-hours total based on round 1's actual wall-clock). routed16 should be
**faster** than routed11 was, not slower - only 6 tensors need gradients instead of the full
model, so the backward pass is cheaper even though the forward pass is unchanged. All four in
parallel: ~3 hours wall-clock for the slowest one.

## Honest simplifications / not built here

- **routed16's "frozen" is `requires_grad=False`, not a separate held-out copy of the
  weights** - the backbone/vocab head parameters are literally the same tensors as routed8's
  checkpoint, just excluded from the optimizer. This is correct and standard practice, just
  worth knowing precisely what "frozen" means here.
- **routed18's copy-structure filter is a cheap heuristic**, not a linguistic parse -
  word-level, case/punctuation-insensitive, no attempt to distinguish "the same word means the
  same thing" from coincidental repetition. It's a proxy, described as one in the code.
- **No base-rate control was run on the filter or the original 82.8% diagnostic** - i.e., we
  don't yet know whether 82.8% copy-failure (or the filter's admit rate) is LAMBADA-specific or
  just a property of how often words recur in ordinary web text. Worth doing before trusting
  routed18's result too strongly either way, not built into this pass.
- **routed15 and routed16 are the higher-confidence recipes; routed17/18 are the lower-risk,
  higher-uncertainty ones** - if only two of four can run at once, prioritize 15 and 16 first,
  since they directly resolve open questions from round 1 rather than opening new ones.
