# Infra / system-design changelog

Tracks infrastructure, eval-pipeline, and training-throughput changes — distinct from model-recipe additions (routed24–33), which live in `results/audit_2026-08-20.md`. The Arm Explorer dashboard (`scripts/gen_arm_explorer.py`) covers per-arm metrics; this doc covers *how things run*.

## 2026-08-20

### Training throughput — bf16 + TF32 + fused AdamW
**Commit:** `13ee1eb`

`scripts/train_stage2_pod.py` switched from fp16 autocast + `GradScaler` to bf16 autocast (no loss scaling needed — bf16's exponent range covers fp32's, unlike fp16), enabled TF32 for the remaining fp32 ops, and added `fused=True` to every `AdamW` construction site. Effective batch size and LR schedule untouched, so results stay comparable to every prior arm.

**Validated live** via `calibrate --arm routed25 --steps 150` on a spare pod: **0.526 → 0.120 sec/step, a 4.4x speedup**, before touching batch size at all.

**Not yet wired in — the biggest remaining lever:** micro-batch 4→32 / grad-accum 16→2 (same effective batch 64, ~2x more on top of the 4.4x already landed). This exact setting is already proven safe in this codebase — `routed19` has used `ROUTED19_BATCH_SIZE=32` / `ROUTED19_GRAD_ACCUM_STEPS=2` successfully for a full 300k-step run (calibrated live: 22GB/46GB peak memory). Applying it to a new arm means extending the existing `if arm == "routed19"` dispatch (6 call sites across `calibrate()`/`train()`, all touching the same `routed19_batch_size` variable) to include the new arm — a one-line-per-site change, deliberately deferred until a concrete new arm (routed34+) exists to wire it to, rather than speculatively generalizing dispatch logic for an arm that doesn't exist yet.

**Does not affect any currently-running arm** — these are process-level settings baked in at next launch, not retroactive. C/D/routed33 keep training on their original (pre-fix) settings for the rest of their run, by design — changing settings mid-flight would make their own before/after checkpoints non-comparable.

### Eval-pipeline bug fixes
**Commits:** `40d3ab0`, `ad0b525`

`stage2_modal.py`'s `evaluate()`/`evaluate_lambada()` had never been extended for the routed29–33 architecture family (tied heads, modern backbone, generalist domain) — every eval call for those arms crashed on a state-dict shape mismatch. Fixed by adding the missing model-dispatch branches. Separately, `evaluate()`'s single-domain BPB loop KeyError'd on routed33's "generalist" domain, which has no real external `STREAM_SOURCES` entry (it's a synthetic pool of the other 4 domains, not a genuinely held-out stream) — fixed by excluding it from that specific loop rather than faking a source.

### Eval speed diagnosis (not yet fixed — see below)
Read through the eval code end to end: the ~40-minute checkpoint-eval wall time is **not GPU-bound**. `BATCH_SIZE=4` with no `num_workers` set on the eval DataLoaders means every batch synchronously blocks on live HuggingFace streaming — the same class of stall that caused tonight's rate-limit fights on fresh pods. `bpt_domain()` also re-streams 150 fresh documents per domain on every single eval call just to compute a static bytes-per-token constant that barely changes between checkpoints. None of the identified fixes (worker parallelism, local eval-corpus caching, caching the bpt constant per arm, bigger batches at the same total sample count, running single/cross-domain passes in parallel) touch what's measured — only `eval_batches` itself (already used tonight as a deliberate, documented precision/speed tradeoff on C/D) changes the numbers. Not yet implemented — flagged as a fast follow.

### Pod-migration reliability lessons (operational, no code)
Two RunPod-specific gotchas discovered and worked around live tonight, worth remembering for any future pod restart:
- **Fresh/migrated pods lose pip packages** (base-image reset) and need `pip install --break-system-packages -r requirements.txt` — the plain image has PEP 668's externally-managed-environment guard on, unlike the pods' original first-boot state.
- **`NUM_WORKERS=4` (a real, deliberate, working optimization for properly-sharded streams) redundantly re-fetches HF dataset metadata once per worker on datasets with only 1 shard** (e.g. `gfissore/arxiv-abstracts-2021`), which compounds badly under repeated crash-restarts and can exhaust HuggingFace's anonymous per-dataset API quota. Patched down to `NUM_WORKERS=1` on C's specific pod only (not the shared repo) to unblock its resume — this is a target-specific workaround, not a permanent fix; a real fix would special-case single-shard datasets rather than lowering worker count globally.
- **`nohup ... & disown` alone did not reliably survive an SSH session closing** in every case observed tonight (inconsistent — worked on some pods, silently died on others); `setsid nohup ... < /dev/null > /dev/null 2>&1 & disown` was the version that reliably detached in every case tested afterward.
- Every resumed arm now runs under a **crash-restart watchdog** (`until <train_cmd>; do sleep N; done`), added after discovering that a plain relaunch left 4 pods idling at 0% GPU for ~2 hours post-migration with no automatic recovery.

### Results provenance backfill
**Commit:** `be1ff1f`

`results/metrics.json` was missing 10 real, already-computed results (routed24–28, routed31–33) that had only ever been reported in chat during a prior session, never persisted. Backfilled from the session transcript's raw eval output (not re-derived/estimated) and cross-checked against the actual printed BPB/LAMBADA lines.

## Known gaps / not yet done
- Eval-speed fixes (above) — diagnosed, not implemented.
- Micro-batch/grad-accum fix for new arms — validated on routed19's existing precedent, not yet wired to a concrete new arm.
- Arm Explorer dashboard (`scripts/gen_arm_explorer.py`) — its `FAMILIES` list stops at `routed19`; routed20–33 are entirely absent from the generated dashboard. Needs a regeneration pass, separate from this changelog.
