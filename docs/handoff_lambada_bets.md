# Launch: 4 LAMBADA-targeted bets (routed11-14), continued from routed8

Code is pushed to `main`. This doc is everything needed to launch all four — no other
context required.

## What it is, and why

routed8 (89M, nlp sourced from OpenWebText, 600k-step target) plateaued: LAMBADA exact-match
held at exactly **6.20%** from step 496k through 575k (96% done), and perplexity barely moved
(19.25 -> 19.13). That's the best result in the whole project by a wide margin (previous best
was routed7/routed-large at ~4.2%), but the volume lever on this exact recipe is exhausted.

An outside review (external LLM analysis) proposed 3 bets targeting *why* exact-match
specifically lags perplexity - NLL is happy spreading probability mass across plausible
synonyms ("glass/cup/mug"), which keeps perplexity low without ever putting the true word in
first place, which is all EM measures. Two refinements were made to the original proposal
before building (see conversation for the full reasoning):

- **Bet 2 was changed from a from-scratch run to continued training** (LoRA adapters,
  zero-init so it's a no-op at warm-start) - no reason to pay for a second full run when the
  question doesn't require training from zero to answer.
- **Bet 3's "periodic competitor mining" was replaced with in-batch hard negatives** - the
  vocab logits are already computed every step; reusing them as the competitor pool is
  simpler than a separate mining loop and never stale.

A 4th recipe was added afterward: routed14, a scaled-up (190.5M) version of routed8 with
Bet 1's mechanism folded in, warm-started from routed7 (not routed8 - wrong param count) to
test whether scale and the data/volume lever compound now that both independently helped.

All four are **continued training from an already-trained checkpoint**, not from-scratch runs
- cheap relative to routed8's own 600k-step run, and each was verified locally (forward/
backward pass + warm-start key-matching, against real model dimensions) before being handed
off, the same verification bar as every other arm tonight.

## Step 0 (recommended, not required): run the error-taxonomy diagnostic first

Before spending GPU-hours on 4 bets ordered by theory, this tells you what routed8 is
*actually* getting wrong, so you can weight the results once they land instead of guessing.
Cheap (one eval pass + classification, no training):

```bash
python3 -m modal run stage2_modal.py --step diagnose-lambada --arm routed8 --steps 575000
```

Reports, over ~300 LAMBADA examples:
- **single_token_coverage**: fraction of targets that are exactly 1 nlp-tokenizer token.
  Bounds Bet 1/3's ceiling - a low number means even a perfect single-token fix can't move
  EM much, since exact-match needs every piece of a multi-token target right.
- **copy_failure** rate (target word appeared earlier in context, model didn't retrieve it)
  -> weight toward Bet 1 (routed11/14).
- **near_miss** rate (true token was in the model's own top-5, just not top-1) -> weight
  toward Bet 3 (routed13).
- **other** (neither) -> Bet 2's territory (routed12) or genuinely rare/unseen words, which
  none of these four levers fix.

This doesn't gate the other launches - they can go out in parallel - but read its output
before deciding which bet's result to trust most, and before deciding whether to extend any
of them past their default step budget.

## Prerequisite: routed8@575000 checkpoint

All four warm-start from it (routed11/12/13 directly, routed14 from routed7 which is itself
downstream of the same lineage). Confirm it's on the launching pod before starting:

```bash
ls checkpoints/routed8_step*.pt       # routed11/12/13's parent
ls checkpoints/large_routed7_step*.pt  # routed14's parent
```

If missing, pull from wherever it's checkpointed or from the Modal volume
(`modal volume get mot-stage2-data checkpoints/<file> checkpoints/<file>`). If no parent
checkpoint is found, `_warm_start_from_parent`/`_warm_start_deep_expert` log a clear warning
and fall back to random init rather than failing silently - but that defeats the point of
these being *continued*-training bets, so verify first.

## Launch (all four)

Same pattern for each: `git pull` -> `calibrate --steps 30` (sanity check, new code paths on
real data for the first time) -> `train`. All target **100,000 steps** by default
(`BET_STEPS` in `stage2_config.py`) - meaningfully more than a quick check, cheap relative to
a full run. Since you're on unrestricted usage now, if a bet is still visibly improving at
100k (check the LAMBADA slope, not just training loss), extending is fully supported - just
relaunch `train` with a higher `--steps`; it resumes from the latest checkpoint automatically,
same as every other arm.

```bash
cd /workspace/repo && git pull origin main
```

**Calibrate only exercises the model/data plumbing, not the warm start** - like every other
warm-started arm tonight (routed9/10), `calibrate()` builds a fresh, randomly-initialized
model purely to time the forward/backward loop; it never touches checkpoints. The warm-start
check happens on the **first `train` launch** instead - watch its opening log lines, not
calibrate's.

**routed11 (Bet 1 - copy gate):**
```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed11 --steps 30
# check calibrate for: "routed11 (base) params: 89,...-ish" and no traceback - confirms the
# new loader/model path works before committing to a real launch.
nohup python3 scripts/train_stage2_pod.py train --arm routed11 --steps 100000 > train_routed11.log 2>&1 &
# on the FIRST launch, check train_routed11.log for:
#   "warm-started from checkpoints/routed8_step*.pt (parent step 575000): 97 tensors loaded,
#    0 tensors left at fresh init" - wait, that's WRONG if it says 0 fresh: copy_q/copy_k/
#   copy_gate are new and should stay at random init. It should read closer to "97 loaded"
#   with the new copy-gate params simply absent from the loaded set (they're not printed as
#   a separate "skipped" count here since skip_prefixes=() - verify via the model's own
#   printed param count instead if in doubt: routed11 (base) params should be ~89.1M + ~525K
#   for the copy-gate module, i.e. slightly ABOVE routed8's params printed above).
```
Watch: `nlp_copy_gate_mean` in the training log (printed via the `parts` dict every LOG_EVERY
steps). If it's rising and LAMBADA improves while ppl barely moves, this is the lever. If it
collapses toward 0, retrieval wasn't the limiting error class - stop.

**routed12 (Bet 2 - deep experts):**
```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed12 --steps 30
nohup python3 scripts/train_stage2_pod.py train --arm routed12 --steps 100000 > train_routed12.log 2>&1 &
# on the FIRST launch, check train_routed12.log for:
#   "warm-started (deep-expert remap) from checkpoints/routed8_step*.pt (parent step 575000):
#    ... 16 total child tensors left at fresh init (new LoRA adapters)" - exactly 16 (2 layers
#   x 4 domains x {lora_a,lora_b}), everything else loaded. A different number means the key
#   remap broke against the REAL checkpoint (this was verified pre-launch against a synthetic
#   one with matching shapes/names, but verify again here) - stop and check before letting it
#   run further.
```
Watch: compare LAMBADA at matched steps against routed8's own historical trajectory
(`results/metrics.json` has routed8@198000/496000/575000). If routed12 tracks that curve
within noise, top-layer contention wasn't the limiter - a legitimate negative result.

**routed13 (Bet 3, exploratory - precision head):**
```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed13 --steps 30
nohup python3 scripts/train_stage2_pod.py train --arm routed13 --steps 100000 > train_routed13.log 2>&1 &
```
Watch: `nlp_precision_gate_mean` and `nlp_margin_loss` in the log. Gate collapsing to ~0, or
EM not beating routed11 at matched steps, means the NLL head's ranking was already the
binding constraint - this pathway isn't adding anything precision-specific.

**routed14 (scaled-up routed8 + copy gate):**
```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed14 --steps 30
# check calibrate for: "routed14 (large) params: 190,..." - should match routed7/routed10's
# param count almost exactly (same architecture + copy gate's ~525K params).
nohup python3 scripts/train_stage2_pod.py train --arm routed14 --steps 100000 > train_routed14.log 2>&1 &
# on the FIRST launch, check train_routed14.log for a warm-start line referencing
# checkpoints/large_routed7_step*.pt (NOT routed8 - wrong param count) with ~169 tensors
# loaded (matches the 12-layer large-scale architecture's module count) and 0 skipped.
```
Checkpoints: `checkpoints/large_routed14_step*.pt` (large-scale prefix, automatic).

## Eval (all four)

All wired into `stage2_modal.py` the same way as every other arm - `evaluate` and
`evaluate-lambada` already know their tokenizer dir (OWT, same as routed8), scale
(routed14 forces large, the others stay base), and model class:

```bash
for arm in routed11 routed12 routed13 routed14; do
  python3 -m modal run stage2_modal.py --step evaluate --arm $arm --steps 100000
  python3 -m modal run stage2_modal.py --step evaluate-lambada --arm $arm --steps 100000
done
```

(Adjust `--steps` to whatever checkpoint you actually pulled - same pull/push/evaluate flow
as every checkpoint eval tonight: `scp` off the pod, MD5-verify if paranoid, `modal volume
put` to `checkpoints/<prefix>_step<N>.pt`, then run both.)

Record via `src.eval.metrics.record()` with the matching `arm=` and `scale=` (routed14 is
`scale="large"`), then regenerate + republish the dashboard
(`python3 scripts/gen_dashboard.py`) - `ARM_META` in `gen_dashboard.py` doesn't have display
names for these four yet, so they'll show their raw arm code until that's added; not
blocking, just cosmetic.

## Cost / time, roughly

100,000 steps at routed8's observed ~0.2-0.25s/step (base scale) is **~6-7 GPU-hours** for
routed11/12/13 each. routed14 (large scale) will run somewhat slower per step, closer to
routed7's rate - budget **~8-10 GPU-hours**. All four in parallel on separate pods: ~10 hours
wall-clock for the slowest one. Sequentially: ~28-30 GPU-hours combined.

## Honest simplifications / not built here

- **Bet 3's word list** is the nlp tokenizer's first 16,000 surface-tier ids by frequency,
  not an externally curated wordlist - see `mot_routed_precision_model.py`'s docstring for
  why this is a reasonable proxy, not just a shortcut.
- **The diagnostic (Step 0) only classifies ~300 examples into 3 buckets** (copy-failure /
  near-miss / other) - enough to weight the four bets against each other, not a rigorous
  linguistic error analysis.
- **routed14 only combines scale + Bet 1**, not scale + all three bets simultaneously -
  stacking every lever at once would make it impossible to tell which one is doing the work
  if it succeeds. If routed14 shows a real gain, the natural follow-up is repeating routed12/
  13's treatment at large scale too, once it's clear Bet 1 itself survives scale.
- **No differential LR schedule beyond the flat 2-group split** (new module at full LR,
  everything else at a fixed scale) - same simplification as routed9/10's cooldown, not
  something this pass changed.
