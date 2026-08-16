# Launch routed2 + routed3 (routed's pod just freed up)

Two new arms, both isolating exactly one thing hybrid bundles together. Built, calibrated
clean on Modal (89,114,364 params each - matches routed/hybrid exactly), and pushed to
`main`. **75k steps, not 150k** - per the new default for exploratory runs going forward.

## 1. Pull latest

```bash
cd /workspace/repo && git pull origin main
```

Confirm `src/model/mot_hybrid_model.py` is present (reused, not new) and
`src/model/stage2_config.py` has `ROUTED3_MIN_DOMAINS` / `ROUTED3_MAX_DOMAINS` /
`ROUTED3_SNIPPET_WORDS`.

## 2. What each arm answers

hybrid (already running) bundles two independent changes: GradNorm switch-loss balancing
+ blending 60% natural single-domain data back in. Whatever hybrid's result turns out to
be, there's no way to know which change earned it. These two isolate them:

**routed2** — GradNorm fix only, data **unchanged** (100% synthetic multi-domain, same as
plain routed always trained on). Answers: does fixing the switch-loss tax alone move an
architecture that's already winning on cross-domain BPB (1.589) and LAMBADA (54.9 ppl,
best of all 5 arms)?

**routed3** — GradNorm fix + data pushed the **opposite** direction from hybrid: more
cross-domain density, not less. Always 4 domains per doc (routed/routed2 use 2-4) and
100-word snippets (routed/routed2 use 250) - verified for real, not just a config flag:
6.8 switch tokens per 1024-token window vs 2.8 for standard data, ~2.4x denser. Answers:
does pushing harder in the direction that's already winning help more than the loss fix
alone?

Both reuse `MoTHybridModel` unchanged - no new model code, only different data feeds
`train_stage2_pod.py` already knows how to route by arm name.

## 3. Launch (one per pod)

```bash
# pod A
cd /workspace/repo
nohup python3 scripts/train_stage2_pod.py train --arm routed2 --steps 75000 > train.log 2>&1 &

# pod B
cd /workspace/repo
nohup python3 scripts/train_stage2_pod.py train --arm routed3 --steps 75000 > train.log 2>&1 &
```

## 4. Sanity numbers from Modal calibration

| arm | params | step-1 loss parts |
|---|---|---|
| routed2 | 89,114,364 | content=10.34, switch=10.59 |
| routed3 | 89,114,364 | content=10.36, switch=10.24 |

Both loss parts non-zero at step 1 for both arms (unlike hybrid, which sometimes draws a
natural-only batch with switch=0.0000) - routed2/routed3 always train on switching data,
so every step has both a content and a switch component.

## 5. What to compare once these + hybrid have real checkpoints

Same BPB (single-domain + cross-domain + switch-accuracy) and LAMBADA eval as every other
arm, via `stage2_modal.py` against a checkpoint pulled to the Modal volume - same workflow
as today. The three-way comparison (hybrid vs routed2 vs routed3) is what tells us whether
the win is the loss fix, the data direction, or both.
