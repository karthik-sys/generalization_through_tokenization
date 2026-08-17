# Launch: data-equivalent runs (routed8 + routed7 extended to 600k)

Code is pushed to `main`. This doc is everything needed to launch both — no other context
required.

## What it is, and why

Tonight's routed7 run (large scale, nlp sourced from OpenWebTextCorpus instead of FineWeb)
closed the data-*source* gap vs GPT-2 for the domain LAMBADA routes through. It didn't close
the data-*volume* gap.

The math: at `BATCH_SIZE(4) * GRAD_ACCUM_STEPS(16) * max_seq_len(1024) = 65,536` tokens/step,
150,000 steps is `9.83B` tokens **total**, split across 4 domains. The document-composition
sampler (`PackedRoutedStream`, 2-4 domains/doc, uniform) treats all 4 domains symmetrically, so
each domain gets roughly an equal share: `~2.46B tokens/domain`. Karpathy's `llm.c` reproduction
of GPT-2-124M (github.com/karpathy/llm.c/discussions/481) needed `~10B tokens` **dedicated to
one domain** to reach GPT-2's real reported performance. `2.46B * 4 ≈ 9.83B` — running 4x the
steps (`600,000`) brings *every* domain to roughly that 10B-token benchmark simultaneously, not
just nlp.

Two runs test this at both scales already in play tonight:

- **Run 1 (new arm `routed8`)**: champion/base scale (89.1M params, same architecture as the
  original `routed` champion), nlp sourced from OpenWebText (same mechanism as routed7),
  600,000 steps. Isolates "does data-volume-equivalence help independent of scale."
- **Run 2 (routed7, extended)**: no new arm — routed7 already has everything this needs
  (large scale, OpenWebText nlp source). Just re-launch it with a higher step target; it
  resumes from its existing checkpoint rather than restarting, so none of tonight's 150k
  steps are wasted.

## Run 1: routed8

Architecture is `MoTRoutedModel` + base `MODEL_CFG` — identical to plain `routed`, except the
nlp domain sources from OpenWebText instead of FineWeb (same `_apply_openwebtext_nlp_source()`
mechanism as routed7, fires automatically for `arm=routed8`, no flag needed). code/math/science
untouched, same rationale as routed7 (OpenWebText is genuinely too sparse in those three to
swap them too — see `docs/handoff_routed7.md`).

**Tokenizer**: reuses the exact same retrained OWT nlp tokenizer routed7 needs
(`tokenizers_stage2_owt/nlp/` — vocab size doesn't depend on model scale, only `d_model`/
`emb_dim` differ between base and large, so no separate retraining is needed).

- If routed8 runs on the **same pod** as routed7: the directory already exists locally, skip
  straight to calibrate.
- If routed8 runs on a **different pod**: either rerun the retrain script (a few minutes,
  CPU-only — a different OpenWebText sample than routed7's pod used, but that's fine, both are
  fit on the same corpus/vocab sizes):
  ```bash
  cd /workspace/repo && python3 scripts/retrain_nlp_tokenizer_openwebtext.py
  ```
  or pull the copy already pushed to the Modal volume during routed7 setup (faster):
  ```bash
  python3 -m modal volume get mot-stage2-data tokenizers_stage2_owt/nlp tokenizers_stage2_owt/nlp
  ```

**Launch**:

```bash
cd /workspace/repo && git pull origin main
python3 scripts/train_stage2_pod.py calibrate --arm routed8 --steps 30
```

Check the calibrate output for:
1. `[routed7] nlp domain source overridden: ...` line — the same log line fires for routed8
   too (it's printed by the shared override function, message text unchanged).
2. `routed8 (base) params: ...` — should be very close to plain `routed`'s param count (same
   architecture, `scale` stays `base` — routed8 does NOT force large the way routed7 does).
3. No traceback.

If clean:

```bash
nohup python3 scripts/train_stage2_pod.py train --arm routed8 --steps 600000 > train_routed8.log 2>&1 &
```

Checkpoints save to `checkpoints/routed8_step*.pt` (no `large_` prefix — base scale). Resumes
automatically on restart, same as every other arm.

## Run 2: routed7, extended to 600k

Once routed7 finishes (or even before — resuming mid-run works the same way), relaunch with
the higher step target on whichever pod has its checkpoints:

```bash
cd /workspace/repo && git pull origin main
nohup python3 scripts/train_stage2_pod.py train --arm routed7 --steps 600000 > train_routed7_ext.log 2>&1 &
```

It auto-resumes from the latest `checkpoints/large_routed7_step*.pt` and continues — no flag
needed to signal "extend," `train()` always resumes from the newest checkpoint for the arm.

**One real caveat, not a bug**: the LR schedule (`lr_at()`) is a cosine decay computed against
whatever `total_steps` is passed on THIS invocation. Under the original `--steps 150000` run,
LR was decaying toward ~0 by step 150,000. Re-launching with `--steps 600000` recomputes the
cosine curve fresh against the new 600k target — so LR will jump back up from wherever it had
decayed to, then decay again toward the new, later endpoint. This is a standard, acceptable way
to extend a cosine-scheduled run (better than restarting from scratch), but it means the loss
curve around the step-150k transition will show a real, expected kink — don't mistake it for
instability.

## Eval (both runs)

Both `evaluate()` and `evaluate_lambada()` in `stage2_modal.py` already know about `routed8`
(base scale, OpenWebText nlp tokenizer dir, `STREAM_SOURCES["nlp"]` override applied in
`evaluate()` the same way as routed7 — `evaluate_lambada()` doesn't need it, LAMBADA always
reads the fixed external benchmark). routed7 continues to work identically at whatever step
checkpoint you evaluate — just pass the new step number:

```bash
# routed8 (after retrieving + pushing its checkpoint to the Modal volume, same flow as always)
python3 -m modal run stage2_modal.py --step evaluate --arm routed8 --steps 600000
python3 -m modal run stage2_modal.py --step evaluate-lambada --arm routed8 --steps 600000

# routed7 at its new, extended checkpoint
python3 -m modal run stage2_modal.py --step evaluate --arm routed7 --steps 600000
python3 -m modal run stage2_modal.py --step evaluate-lambada --arm routed7 --steps 600000
```

Record via `src.eval.metrics.record()` with `arm="routed8"` / `arm="routed7"` respectively
(the 600k routed7 entry sits alongside tonight's 150k entry in `results/metrics.json` — same
arm, different step, both stay on the dashboard), then regenerate + republish
(`python3 scripts/gen_dashboard.py`).

## Cost / time, roughly

Both target 600,000 steps — 4x tonight's 150,000-step runs. routed-large/routed7 ran at
~0.2-0.25s/step tonight; at that rate 600k steps is **~35-40 GPU-hours per run**, not
per-step-of-both. routed8 (smaller, base-scale model) should calibrate a bit faster than that
per-step estimate — `calibrate` will report the real number before you commit to the full
`train` launch. If both run on separate pods concurrently, wall-clock is ~35-40 hours either
way; sequentially on one pod, closer to 70-80 hours combined. Worth calibrating both before
deciding whether to run them concurrently or one after another, given the RunPod hourly billing
already in play tonight.

## Not part of this scope

- code/math/science's own sources are unchanged in both runs — this fix only addresses
  *volume* (4x steps), not *source quality*, for those three domains. They already had their
  own rich sources (codeparrot, open-web-math, arxiv-abstracts); nlp is the only domain that
  additionally got a *source* swap (routed7/8 vs plain routed).
- The N-domain sub-classifier follow-up remains deferred (see `docs/handoff_routed7.md`'s
  "Not part of tonight's scope" section — unchanged).
