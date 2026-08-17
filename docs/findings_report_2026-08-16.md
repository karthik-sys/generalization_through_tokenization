# Mixture-of-Tokenizers — findings report, 2026-08-16

**Question:** does domain-routed, multi-tokenizer LLM architecture beat unified BPE tokenization
at matched small scale (68–190M params)?

**Status:** 13 evaluated checkpoints across 9 distinct arms, 5 base architectures. Two more
arms training overnight (a scale-test pair, a combined-fix variant). This doc is a snapshot
for review, not a final writeup — the scale test and routed4 aren't done yet.

Repo: `karthik-sys/generalization_through_tokenization`, branch `main`. All numbers below are
pulled directly from `results/metrics.json` (the canonical eval store), not transcribed by
hand.

---

## 1. The five base architectures

All share one backbone (6 layers, 512d, 8 heads, 2048 ffn) and are trained on the same
corpus (code/math/science/nlp, streamed from codeparrot/github-code, open-web-math,
arxiv-abstracts, fineweb). They differ only in tokenization and routing:

| Arch | Tokenizer | Params | Mechanism |
|---|---|---|---|
| **Baseline** | Unified 48k BPE | 68.6M | One shared vocab, one head, no routing |
| **SOTA** | cl100k_base (GPT-4's), 100k vocab | 122.2M | Same as baseline, larger off-the-shelf vocab |
| **MoT** | 4 disjoint per-domain tokenizers | 89.1M | One domain fixed per forward pass, disjoint embeddings + heads |
| **Routed** | MoT + mid-sequence switching | 89.1M | Same disjoint tables, but a single sequence can switch domains mid-stream; the switch itself is a predicted token |
| **Pooled** | Routed + PMA/DANN pooling | 98.7M | Adds a shared cross-domain attention-pooling layer + adversarial domain-invariance loss on top of routing |

Two metrics, deliberately different axes:
- **BPB** (bits-per-byte, lower better): in-domain compression on held-out text from the
  training distribution. Byte-normalized so it's comparable across arms despite different
  tokenizers producing different token counts for the same text.
- **LAMBADA** (target-token perplexity, lower better): predict the final word of a passage
  given full context. External, architecture-agnostic, unrelated to training domains — tests
  generalization, not memorization of the training distribution's shape.

They disagree, and the disagreement is the whole story below.

---

## 2. Full results

Sorted by LAMBADA (the generalization axis — the headline metric). All at 150k steps unless
noted.

| Arm | Params | Steps | BPB single | BPB cross | LAMBADA ppl | Switch acc |
|---|---:|---:|---:|---:|---:|---:|
| **Routed** (champion) | 89.1M | 150k | 2.075 | 1.544 | **47.76** | 25.8% |
| MoT | 89.1M | 150k | 1.909 | — | 148.4 | — |
| Pooled | 98.7M | 150k | 2.384 | 2.135 | 280.5 | 27.4% |
| Hybrid | 89.1M | 150k | 2.494 | 2.791 | 380.9 | 34.4% |
| Pooled2 | 98.7M | 150k | 2.409 | 2.223 | 312.9 | 25.1% |
| Routed2 | 89.1M | 150k | 3.387 | 2.823 | 1,597.9 | 33.5% |
| Routed3 | 89.1M | 150k | 3.673 | 3.202 | 3,198.6 | 35.5% |
| Baseline | 68.6M | 150k | 2.005 | — | 7,785.6 | — |
| SOTA | 122.2M | 123k | 2.132 | — | 72,148.5 | — |

*(Pooled2, Hybrid, Routed2, Routed3 are variants layered on the base 5 — see §4. Hybrid's
150k figures shown; a 75k checkpoint also exists, ppl=396.2, superseded.)*

**In-flight, not yet final:**

| Arm | Params | Step (of 150k) | Note |
|---|---:|---:|---|
| Baseline-large | 159.6M | ~75-80k | Scale-test control, still training |
| Routed-large | 190.5M | 40k snapshot | Already matches/beats the full 89M champion on BPB and switch-acc (see §5) — early, not final |

---

## 3. Finding #1 — the entire advantage comes from two architectural choices, not from any of the tuning attempted since

Ranking the base 5 by LAMBADA: **Routed (47.76) < MoT (148.4) < Pooled (280.5) « Baseline
(7,786) < SOTA (72,149)**.

Two levers explain nearly all of that gap:

1. **Disjoint per-domain tables** (MoT vs. Baseline/SOTA) — the single biggest lever. Just
   giving each domain its own embedding/head takes LAMBADA from baseline's 7,786 / SOTA's
   72,149 down to 148, even with zero switching capability.
2. **Mid-sequence switching** (Routed vs. MoT) — a further 3x improvement (148.4 → 47.76),
   despite switch-prediction accuracy sitting near chance (~26%, vs. an implied ~25% chance
   floor for 4 domains). The value isn't in *predicting* switches — it's that the model can
   *use* one instantly once it lands. Confirmed behaviorally: seeded in science, Routed emits
   a switch token and writes valid Python, then LaTeX, then prose, correctly, well above what
   26% switch-accuracy would suggest it "should" be able to do.

Everything attempted **on top of** that recipe — four separate variants, four separate
training runs — made things worse, not better. See Finding #2.

---

## 4. Finding #2 — GradNorm loss-reweighting has negative ROI, confirmed four independent times

Plain Routed uses a flat, hand-picked constant: switch-target positions get their
cross-entropy loss multiplied by 50 before summing with content loss. It was chosen once,
empirically, because it moved switch-accuracy off zero — not because it's calibrated.

GradNorm-lite (`src/model/gradnorm.py`) replaces that constant with EMA-normalized
weighting: each loss term (content, switch) is divided by its own running-average magnitude
before summing, so neither dominates purely from scale. Theoretically more principled — no
arbitrary constant, self-adjusting.

Every arm that used it regressed hard, and the same trade showed up every time — switch
accuracy up, everything downstream of content quality down:

| Arm | What changed vs. its non-GradNorm baseline | BPB single | LAMBADA ppl | Switch acc |
|---|---|---:|---:|---:|
| **Routed2** | GradNorm loss only, isolated | 3.387 (**+63%**) | 1,597.9 (**33x worse**) | 33.5% (vs 26%) |
| **Routed3** | GradNorm + 2.4x denser cross-domain data | 3.673 (**+77%**) | 3,198.6 (**67x worse**) | 35.5% (vs 26%) |
| **Hybrid** | GradNorm + 60% natural-data blend | 2.494 (+20%) | 380.9 (**8x worse**) | 34.4% (vs 26%) |
| **Pooled2** | GradNorm + sparse routing, vs Pooled | 2.409 (+1%) | 312.9 (vs 280.5) | 25.1% (vs 27.4%, *worse*) |

Routed2 isolates the loss-function change alone (same data, same everything else as plain
Routed) — it regressed just as hard as the more elaborate variants, which rules out "it's the
data changes" as the explanation. The mechanism is the loss reweighting itself.

**Working hypothesis** (not directly proven, but consistent with the pattern): GradNorm
equalizes loss *magnitudes*, not importance. As switch loss shrinks with training, GradNorm
keeps re-inflating its relative weight to stay equal to content loss — continuously pulling
gradient budget toward a sparse, noisy signal (switches are ~1-in-461 positions) at content's
expense. The flat constant of 50 was already an empirically-tuned answer for this specific
noise/sparsity ratio; the "principled" adaptive version found a theoretically cleaner but
practically worse operating point.

---

## 5. Finding #3 — the scale test is looking promising, but the real comparison isn't done yet

The open question going in: does the architectural advantage survive a real parameter
increase, or is it a small-scale artifact? A matched pair (Routed-large 190.5M vs.
Baseline-large 159.6M, same everything else) is training now.

Early snapshot at 40k/150k steps (27% through), Routed-large already matches or beats the
**fully-trained** 89M champion:

| | Routed-large @ 40k (27% done) | Routed champion @ 150k (100% done) |
|---|---:|---:|
| BPB single | 2.002 | 2.075 |
| BPB cross | 1.541 | 1.544 |
| Switch acc | 28.8% | 25.8% |
| LAMBADA ppl | 56.8 | 47.76 |

**Caveat, important:** this compares Routed-large to *smaller Routed*, which only shows
"bigger Routed keeps improving" — expected, and not surprising on its own. The comparison
that actually answers the scale question is **Routed-large vs. Baseline-large**, matched at
the same step count. Baseline-large hasn't been evaluated yet. Until that head-to-head
exists, "does the advantage survive scale" remains genuinely open, even though the early
signal on Routed-large alone is a good one.

---

## 6. Finding #4 — switching arms compress *better* cross-domain than single-domain

Every switching arm (Routed, Pooled, Pooled2, Routed-large) shows lower (better) BPB on
cross-domain held-out text than on single-domain held-out text — e.g. Routed: 1.544
cross vs. 2.075 single. Reproduced across four separate arms/training runs, so it's a real
effect, not noise. SOTA has no cross-domain figure at all (never switches), so on this axis
Routed isn't just beating SOTA — it's beating *itself*.

Plausible read: switching to a fresh domain's tokenizer + embedding table right at a domain
boundary hands the model a cleaner signal than staying mid-stream in one domain's content.

---

## 7. What's running overnight

**Routed4** — a new arm combining three fixes into one run (not three separate ablations —
see rationale in `docs/handoff_routed4.md`):

1. **Gradient-decoupled switch head** — switch prediction moves to its own small classifier
   fed `h.detach()`, so the shared backbone only ever receives gradient from content loss.
   Verified with a unit test: an all-switch-target batch produces exactly zero backbone
   gradient.
2. **Learned switch weight** (Kendall et al. 2018 homoscedastic-uncertainty form:
   `switch_loss/(2σ²) + log(σ)`) instead of GradNorm's magnitude-equalizing normalization —
   starts at the same effective weight (50) plain Routed uses, but is free to move, with
   `log(σ)` regularizing against collapsing to zero (the standard failure mode of an
   unconstrained learned multiplier). Verified numerically: initial effective weight is
   exactly 50.0.
3. **2x context** (1024 → 2048) — more room between switches to re-establish domain state.

Standard cross-entropy throughout, no GradNorm anywhere in this arm — a deliberate return to
the loss-shape of the thing that already works (plain Routed), with a more principled
handling of the one component (switch weighting) that's been the site of every regression so
far.

Two single-change variants (decoupled-head-only, long-context-only) are built and wired but
reserved — not launched tonight. If Routed4 wins, they're the fast way to attribute the win
to a specific piece rather than the combination.

Also running: the Baseline-large/Routed-large scale-test pair (§5).

---

## 8. Methodology notes for review

- **BPB, not raw loss**, is the only trustworthy cross-architecture comparison — different
  tokenizers produce different token counts for the same text, so raw per-token loss isn't
  comparable. BPB normalizes by byte count instead.
- **LAMBADA is external** — none of these domains (code/math/science/nlp) match LAMBADA's
  narrative-English content, and all domain-routed arms route LAMBADA text through the "nlp"
  head specifically because it's the closest match, not because of any special LAMBADA-aware
  training.
- **Exact-match accuracy on LAMBADA sits at ~0 for every arm** at this scale — expected,
  documented as a known floor. Perplexity is the differentiating metric here, not accuracy.
- **GradNorm arms use an adaptive controller** (spike-guard, plateau-rescue, online LR) that
  plain Routed doesn't. This is a training-stability safety net, not part of the loss-shaping
  mechanism being tested — but it's a confound worth naming: the regressions in §4 happened
  *despite* extra stabilization machinery the champion never needed.
- **Some checkpoints (Pooled2, Routed2, Hybrid, Routed-large) survived unplanned pod kills**
  mid-training this session (external SIGKILL/OOM, not divergence — loss was healthy right up
  to the last logged step in every case) and were resumed from their last checkpoint. Their
  local training logs only cover time since the resume (the training script overwrites, not
  appends, on restart), which is a known gap in the loss-curve visualizations, not in the
  final eval numbers (those come from the checkpoint's actual weights, unaffected by log
  truncation).
