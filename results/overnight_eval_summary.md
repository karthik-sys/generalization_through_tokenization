# Overnight architecture ablation — eval summary

Checkpoints pulled from RunPod after an account-balance-triggered pod stop and migration; final numbers as of this run. Raw checkpoint files live in `checkpoints/` (gitignored — too large for git, kept locally for inspection).

## routed28 — flagship recipe (copy-gate + diet) at 190M, full 300k run

| Checkpoint | LAMBADA EM | ppl | Single BPB | Cross BPB | Math ppl | Science ppl | NLP ppl |
|---|---|---|---|---|---|---|---|
| 140k | 15.80% | 7.32 | 1.58 | 1.11 | 298 | 648 | 24 |
| 223k | 16.80% | 6.68 | 1.73 | 1.05 | 668 | 1380 | 21 |
| 300k (final) | 16.60% | 6.54 | 1.80 | 1.04 | 919 | 2260 | 20.7 |

**Read**: ppl/nlp/cross-domain metrics improve monotonically but with sharply diminishing returns after ~140k. Math and science degrade monotonically and dramatically over the same steps — real, ongoing cost, not a one-time trade. Single-domain BPB (the most honest "overall quality" number) gets *worse* every checkpoint. The run should probably have stopped earlier, not been extended to 300k.

## Architecture ablation arms (tied heads / modern backbone / generalist domain)

| Arm | Final step (of 300k) | LAMBADA EM | ppl | Single BPB | Cross BPB | Switch acc |
|---|---|---|---|---|---|---|
| **C (routed31)** — tied + RoPE + RMSNorm + SwiGLU + QK-norm | 94,000 (31%) | 17.60% | 8.05 | ~1.76 (half-sample, see note) | *unavailable* | *unavailable* |
| **D (routed32)** — tied + RoPE + RMSNorm only ("safe improver") | 222,000 (74%) | 16.60% | 7.89 | *unavailable* — see note | *unavailable* | *unavailable* |
| **routed33** — 5-domain generalist (untied, from scratch, aggressive nlp filter) | 210,000 (70%) | **18.00%** | **6.57** | 1.72 | 1.15 | 19.49% |

**Note on missing C/D BPB numbers**: the final-checkpoint BPB eval for both C and D failed repeatedly (8 attempts for C, 5 for D) with a Modal-side `"Received a cancellation signal while processing input"` error — no Python traceback, so not a code bug in the usual sense. Ruled out as causes: local-client disconnect (fixed with `modal run --detach`, didn't help), log-streaming interference (removed, didn't help), launch concurrency (ran solo, still failed), and eval runtime length (cut `eval_batches` 200→80, still failed). routed33's BPB eval — architecturally simpler, no RoPE/SwiGLU/QK-norm — succeeded reliably twice on the same launch pattern, so the failure looks specific to the `MoTRoutedTiedModel` + `ModernBackbone` construction path, possibly an intermittent CUDA-level hang that Modal's worker health-check kills silently. C's single-domain BPB above is the running average from its best partial attempt (100/200 batches, i.e. real data but a half sample, ~±0.05 noisier than a full 200-batch read) — not a clean final number. D never got a partial reading; its most recent confirmed clean BPB remains the 138k checkpoint below. Worth a fresh look with eyes on `ModernBackbone`'s RoPE cache/memory footprint rather than more blind retries.

Earlier checkpoints, for trend:

| Arm | Step | LAMBADA EM | ppl | Single BPB | Cross BPB |
|---|---|---|---|---|---|
| C (routed31) | 62,000 | 16.40% | 8.72 | 1.89 | 1.54 |
| D (routed32) | 138,000 | 17.60% | 7.83 | 1.71 | 1.40 |
| routed33 | 108,000 | 15.60% | 7.34 | 1.67 | 1.28 |

**Read (LAMBADA complete for all 3, BPB pending)**: routed33 is the standout — EM 18.00% AND ppl 6.57 simultaneously, the best of any arm tonight on both the noisy and the reliable metric at once (not just a lucky EM draw). C improved from 62k→94k (EM 16.40%→17.60%) and is currently ahead of D's final EM (16.60%), but D's own EM moved *down* 1pp between 138k and 222k with more training — a reminder that a single EM reading at n=500 carries ~±1.7pp noise, so C's lead over D isn't confirmed until BPB (or more LAMBADA samples) back it up. D's ppl (7.89) is still marginally better than C's (8.05), so the two metrics disagree on ranking C vs D right now. routed33's earlier BPB numbers (before this final pull) were already the best-compressing of the three, especially on the starved math/science domains — consistent with the generalist domain doing exactly what it was built to test.

## Other arms this session

- **B (routed30)** — direct-tied (wide emb_dim=512, no bridge) + shrunk vocab: diverged to NaN around step ~80k, unrecoverable, killed.
- **A (routed29)** — bridge-tied, 23 layers, from scratch: hit persistent host-level network storage issues most of the night, stopped early (~step 4-5k) after repeated stalls; not evaluated.
- Bonus checkpoints recovered from a reused pod: `routed17_step100000.pt`, `routed23_step100000.pt`, `routed26_step300000.pt` — earlier-session arms, not yet re-evaluated against current understanding.

## Gradient-conflict diagnostic (routed25, 89M)

Cosine similarity between the main LM loss gradient and the switch-prediction loss gradient, on shared backbone params, 30 held-out batches: **mean -0.117, negative in 96.7% of batches** — real, consistent conflict. The switch-prediction auxiliary task is measurably fighting the main language-modeling objective for backbone capacity. Gradient surgery (dropping the conflicting component) is a real, untried next step.

---
*Generated during the overnight session. C and D's final-checkpoint BPB evals remain incomplete (see note above) — revisit if the Modal cancellation issue gets root-caused.*
