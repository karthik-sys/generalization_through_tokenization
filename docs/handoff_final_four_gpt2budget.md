# The final four — one batch, all at GPT-2-class budgets

**Status: saved as specified, NOT launch-ready. Two verification issues below must be resolved first.**

Supersedes the pre-infra-win routed34–37 assignments (generalist-merge / diet-anneal / PCGrad move to the backlog — see `docs/handoff_final_four_gpt2budget.md`'s history in `results/audit_2026-08-20.md` for what those were). At 4.4x infra speed the calculus changed — data budget beats mechanism tweaks, per the project's own "data > steps > scale" law. All four run at the new settings (bf16/TF32/fused, micro-batch 32 / grad-accum 2), where 300k steps = 9.8B tokens = llm.c's GPT-2-matching budget.

## Verification notes (checked against the actual codebase before saving)

1. **routed35's "NO gate" premise is not currently buildable.** `src/model/mot_routed_tied_model.py`'s `MoTRoutedTiedModel.__init__` constructs `copy_q`/`copy_k`/`copy_gate` unconditionally — no flag, no `if`, no way to opt out — and `head_loss` unconditionally routes every nlp-domain position's loss through the copy-gate blend (`_nlp_copy_gate_pmix`/`_nlp_copy_gate_losses`). D (routed32) already trains with copy-gate active, whether the recipe intended it or not — this was confirmed independently earlier the same session while investigating why C's recipe ("the stack") turned out to already be C's actual running config. **As specced, routed35 and routed34 would differ only in data mixture (FWE/edu + long-doc), not in the gate** — the batch's stated "#1 vs #3 isolates the gate at 10B tokens" purpose doesn't hold until this is fixed. Fix options, cheapest first: (a) add a `use_copy_gate: bool = True` constructor flag to `MoTRoutedTiedModel` that skips constructing/using the mechanism when `False` — small, scoped, additive change; (b) accept routed35 as "D's recipe, gate included" and drop the "isolates the gate" framing, redesigning what the control pair actually tests. Not yet decided — needs a call before launch.
2. **`handoff_routed34_gpt2budget.md`, referenced throughout for shared logistics (cache pre-build, EM-ceiling audit, 300-step calibration prerequisites), does not exist anywhere in this repo** — not in the working tree, not in git history, not on any branch. Same situation as the earlier `routed38` naming collision: it's presumably sitting in a separate claude.ai/Fable conversation that was never pushed here. Those prerequisites are described as required for all four arms and aren't available to plan against yet.
3. Confirmed accurate: `WARMUP_STEPS = 500` in `src/model/stage2_config.py` (the doc's "raise to 2000" recommendation is a real, correctly-scoped one-line change). The `AdamW(...)` calls in `train_stage2_pod.py` don't pass `weight_decay`, so they run on PyTorch's default (0.01) — the doc's claim about the current value is accurate; adopting llm.c's 0.1 is a genuine, undecided choice, not a bug to fix.

## The batch

| # | Arm | Role | Recipe | Params | Tokens | Est. cost |
|---|---|---|---|---|---|---|
| 1 | routed35 | SAFE | D's exact recipe (tied + RoPE+RMSNorm + diet, NO gate*) + FWE/edu data + long-doc fraction | ~91M | 9.8B | ~$11–17 |
| 2 | routed36 | EXPLORATORY | Max stack: C's full modern backbone (SwiGLU+QK-norm) + copy-gate + generalist 5th domain + FWE/edu data + long-doc fraction | ~95M | 9.8B | ~$12–18 |
| 3 | routed34 | FLAGSHIP | D chassis + copy-gate + FWE/edu data + long-doc fraction (as specced in the missing gpt2budget handoff) | ~91M | 9.8B | ~$14–20 (4×A40, ~7–11h) |
| 4 | routed37 | SCALED | routed34's exact recipe at LARGE scale (~190M) — the arm most likely to actually touch 35% | ~190M | 12–15B (375–450k steps) | ~$35–50 (4×A40, ~15–20h) |

\* see verification note 1 — not currently buildable as a genuine gate-free variant.

Batch total ~$75–105. Funding: the $61 reserve + the retired 35–37 line items + replicate savings. If it doesn't all fit, cut #2 first (exploratory), never #1 (the control that makes #3 and #4 interpretable — modulo verification note 1).

**Why this shape**: #1 vs #3 isolates the gate at 10B tokens (D was only ever proven at 1.2B — mechanisms that help data-starved models sometimes wash out data-rich ones, and knowing that is worth $15) — *pending verification note 1's fix*. #2 is the high-variance draw — if the max stack composes, it leapfrogs; if it erodes or diverges, its levers were already individually banked so nothing is lost but $15. #3 is the main bet. #4 is #3's recipe with the two knobs that most reliably buy EM — params and tokens — turned up together; its extra tokens (12–15B) keep it past Chinchilla-optimal for 190M with headroom.

**Launch order**: #1+#3 first (they're the control pair — same pods DDP or two pods), #4 next, #2 last.

## Shared eval + gates (eval v2)

Full-set LAMBADA + CI + per-domain BPB every 25k steps; externals every 100k; checkpoint selection is the deliverable. Progress gate on every arm: ≥25% full-set EM by 5B tokens (#4: by 7B), else pause and diagnose. Kill: NaN, or >2pp behind routed35 (the safe arm) at matched tokens. Champion rule unchanged: guardrails first, CI-overlap = tie, tie → simpler recipe. Seed twin: at these prices, twin the WINNER after the batch reads out (~$15) rather than pre-committing twins to all four.

## The GPT-2-parity checklist — what else has to scale (and what already matches)

Token count is the big one, but "GPT-2-equivalent" has a few more dials.

| Factor | GPT-2 / llm.c | You (after this batch) | Verdict |
|---|---|---|---|
| Train tokens | ~100B / 10B (FWE) | 9.8–15B, FWE-heavy | matched by this batch |
| Data quality/dedup | WebText / FineWeb-Edu | FineWeb-Edu primary | matched |
| Epochs (data freshness) | ~1 epoch | verified <1 epoch every source (FWE 1.3T, OWT 9B vs 2.8B drawn, OWM 14.7B vs 1B, Cosmopedia 25B vs ~1B) | matched |
| Context length | 1024 | 1024 | already equal |
| Params | 124M / 124M | 91M (#1–3), 190M (#4) | #3 fights ~25% under GPT-2's weight; #4 covers it |
| Effective batch | ~0.5M tokens/update | 65k tokens/update (64×1024) | deliberate divergence — kept because the LR schedule is validated at 64; more/smaller updates at fixed data is quality-neutral-to-positive, just less parallel. Do NOT raise batch and keep LR fixed. |
| Warmup | ~350M tokens | 500 steps × 32k = 16M tokens | short at the new batch size — raise `WARMUP_STEPS` to 2000 (~65M tokens) for these four arms; one constant, low risk, confirmed feasible |
| Weight decay | 0.1 on matmul weights | AdamW default 0.01 (confirmed — no `weight_decay=` passed anywhere) | optional — llm.c uses 0.1; worth adopting for these arms, note it in the run config either way |
| Dropout | 0 (llm.c, 1-epoch regime) | 0 | matched |
| LR decay floor | cosine to ~0.1×max (llm.c) | cosine to 0 | minor — fine as-is |
| Eval scoring | lm-eval-harness | eval v2 + calibration harness | matched once calibration passes — REQUIRED before any "GPT-2 parity" claim |

**Net**: after this batch the only structural deficits left vs GPT-2-124M are ~25% fewer params on the 91M arms (covered by #4) and the smaller effective batch (a documented, deliberate choice). Everything else is matched or better. Set `WARMUP_STEPS=2000` and decide the weight-decay flag before launch; both are one-line config constants.

## Before this can launch

1. Resolve verification note 1 (build a real no-gate `MoTRoutedTiedModel` option, or redesign what the control pair tests).
2. Get the missing `handoff_routed34_gpt2budget.md` content (cache pre-build steps, EM-ceiling audit, 300-step calibration protocol) into this repo — paste it or push the file.
3. Decide the weight-decay flag (0.01 vs 0.1) and confirm `WARMUP_STEPS=2000` for these four arms specifically (not a global change — don't touch in-flight or already-launched arms' settings).
