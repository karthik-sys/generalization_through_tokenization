# Launch hybrid + pooled2 on the 2 free pods

Two new arms are built, wired, calibrated on Modal (no crashes, clean loss parts, correct
param deltas), and pushed to `main`. This doc is everything needed to launch both on the
2 currently-idle pods (`mot-stage2-a40` / `ro29h1mrn5uduf` and `mot-stage2-baseline` /
`fjgwbnh2menybv` — both EXITED, their original jobs finished at 150k).

## 1. Pull latest on both pods

```bash
cd /workspace/repo && git pull origin main
```

Confirm `src/model/mot_hybrid_model.py`, `src/model/mot_pooled2_model.py`, and
`src/model/gradnorm.py` are present after the pull.

## 2. What each arm is, briefly

**hybrid** — routed's mid-sequence switching mechanism (control_embedding + widened heads,
unchanged), retrained to remove two taxes that made plain `routed` (2.057 bpb) underperform
vanilla `mot` (1.909 bpb) on the same held-out data:
- `switch_weight=50.0`'s flat multiplier → replaced with GradNorm-lite (EMA-normalized
  content/switch loss combination, no hand-picked constant).
- 100% synthetic multi-domain training data (never a natural continuous passage) → now
  blends in natural single-domain batches 60% of the time (`HYBRID_NATURAL_DATA_FRACTION`).
- Gets the adaptive controller (spike-guard, plateau-rescue, online LR) too.

**pooled2** — same PMA+DANN pooling as `pooled`, deliberately kept intact (cross-domain BPB
beat single-domain for both routed and pooled on real data - the pooling looks like it's
doing real work, so this isn't a strip-down). Only compute/loss-balance changes:
- GradNorm-lite across (content, load_balance, adversarial), replacing fixed weights.
- Sparse top-2-of-16 expert routing instead of dense-all-16 (real compute cut, same
  capability - the pooled representation still drives routing, just cheaper to compute).
- PMA chunk_size 128 → 256 (fewer pooling calls).
- Confidence head dropped (calibration side-signal, unrelated to what's actually working).

## 3. Launch (one arm per pod)

Same CLI shape as the other arms - `train_stage2_pod.py` now accepts `hybrid` and `pooled2`
as `--arm` choices:

```bash
# on mot-stage2-a40 (or whichever pod you pick)
cd /workspace/repo
nohup python3 scripts/train_stage2_pod.py train --arm hybrid --steps 150000 > train.log 2>&1 &

# on mot-stage2-baseline (or the other pod)
cd /workspace/repo
nohup python3 scripts/train_stage2_pod.py train --arm pooled2 --steps 150000 > train.log 2>&1 &
```

Per the earlier decision: **this run stays at 150k** (matched with the rest of today's
batch); 100k is the default for anything scaled up *after* this.

## 4. Sanity numbers from Modal calibration (already run, so you don't have to re-check)

| arm | params | notes |
|---|---|---|
| hybrid | 89,114,364 | identical to routed's param count (GradNorm adds no weights) - confirmed |
| pooled2 | 98,720,272 | exactly 513 less than pooled's 98,720,785 = the dropped confidence head's 512+1 bias, exactly - confirmed |

Both ran 40 calibration steps on a T4 with no crashes. Rough cost estimate from that
calibration (T4 pricing - your A40 will differ, just a sanity bound): hybrid ~$3.56/20k
steps, pooled2 ~$2.54/20k steps.

**One thing to know about hybrid's printed training loss**: because GradNorm normalizes each
loss term by its own running EMA before summing, the number in the log (`loss=...`) is a
*normalized* value, not raw nats - don't expect it to look like other arms' loss curves or
be numerically comparable to them at a glance. This doesn't affect correctness; the real
comparison always goes through a separate BPB eval anyway (plain, unnormalized CE,
independent of whatever the training loss balancer does).

## 5. Checkpoints

Pod behavior confirmed: only the *newest* checkpoint per arm is kept (older ones pruned on
each save) - same as the other arms already running there. `checkpoints/hybrid_step*.pt`
and `checkpoints/pooled2_step*.pt` will appear as training progresses.

## 6. Eval

No `evaluate()` on the pod side - held-out BPB (single-domain + cross-domain + switch
accuracy, same as routed/pooled get) happens via `stage2_modal.py` against a checkpoint
pulled back to the Modal volume, same workflow as every other arm today. Ping the other
session (or pull + push the checkpoint yourself) once there's something worth evaluating.
