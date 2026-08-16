# Queue: the scale test — mot-large + baseline-large

The single highest-value open question: **does MoT's advantage over unified-BPE survive a
real parameter increase, or is it a small-scale artifact?** Every result so far is at
68-122M params, exactly where tokenizer-specialization effects show strongest. This is the
clean, decisive test. Built, calibrated, pushed to `main`. **Queue after the current 150k
runs free up 2 pods.**

## What it is

A matched pair at ~2.1-2.3× the params, everything else identical (same tokenizers, same
data pipeline, same 1024 context, same steps). Only a *matched* pair is informative — scaling
just the winner tells you nothing, since you'd have no control to compare against.

| arm | base params | large params | calibrated (A10G) |
|---|---|---|---|
| mot | 89.1M | **190.5M** (2.14×) | 0.77 s/step → ~32 GPU-hr / 150k |
| baseline | 68.6M | **159.6M** (2.33×) | 0.54 s/step → ~22 GPU-hr / 150k |

Config (`LARGE_MODEL_CFG` in `src/model/stage2_config.py`): d_model 512→768, layers 6→12,
heads 8→12, ffn 2048→3072, emb_dim 128→192, context held at 1024. Peak GPU mem 9.34 GB on the
larger baseline — fits the A40 (48GB) with enormous headroom, no OOM risk. ~$24 total for the
pair at 150k on A40s.

## Launch (one arm per pod)

The `--scale large` flag is the whole mechanism — it swaps the config and namespaces the
checkpoints as `large_mot_step*.pt` / `large_baseline_step*.pt` so they never collide with
the base-size runs. Everything else about the training path is byte-identical to the normal
mot/baseline runs.

```bash
cd /workspace/repo && git pull origin main

# pod A
nohup python3 scripts/train_stage2_pod.py train --arm mot --steps 150000 --scale large > train.log 2>&1 &

# pod B
nohup python3 scripts/train_stage2_pod.py train --arm baseline --steps 150000 --scale large > train.log 2>&1 &
```

(Optional sanity first: `python3 scripts/train_stage2_pod.py calibrate --arm mot --steps 30 --scale large` — should print `mot (large) params: 190,481,644` and a per-step time.)

## The decision this answers

- **If mot-large still beats baseline-large on BPB + LAMBADA** → the architectural advantage
  is scale-robust. That's the publishable result and the green light to go bigger / transplant.
- **If the gap closes** → it was a small-model capacity-pressure artifact (a bigger shared
  vocab has room to absorb all domains). We learn that cheaply, for ~$24, before spending on
  anything larger.

## Eval when done

Pull `large_mot_step150000.pt` / `large_baseline_step150000.pt` back to the Modal volume and
run the usual `stage2_modal.py --step evaluate` (BPB) and `--step evaluate-lambada`. Note:
the eval side of `stage2_modal.py` currently builds eval models with the base `MODEL_CFG`, so
before evaluating a large checkpoint, `evaluate()` / `evaluate_lambada()` need the same
`--scale`-style branch the training path got. Flag this to whoever runs the eval — it's a
small addition, not yet wired, so the large checkpoints can't be evaluated until it's added.
