# Launch: optimized 89M/190M (routed9 + routed10) — targeted LAMBADA levers

Code is pushed to `main`. This doc is everything needed to launch both — no other context
required.

## What changed from the routed8/routed7-extended plan

An outside review (external LLM analysis, shared by the user) argued that pure token-*volume*
matching (`docs/handoff_data_equivalent_runs.md` - routed8, routed7 extended to 600k) undersells
two sharper, cheaper levers:

1. **Book-derived data, not just web text.** LAMBADA's passages are themselves book-derived,
   and the skill it tests - resolving a target word from a long-range antecedent - is one
   short web pages (OpenWebText included) rarely train. routed7's OWT swap barely moved EM;
   this may be why.
2. **Mixture share, not just total volume.** nlp gets the same ~25% share of tokens in every
   arm so far (symmetric domain sampling). Upweighting it directly, independent of total step
   count, targets the domain LAMBADA actually cares about instead of diluting it across 4x
   more code/math/science too.

Both are also cheaper than a fresh 600k-step run: **warm-starting** from an already-trained
checkpoint (rather than training from scratch) means only the nlp-specific parameters need to
adapt, not the whole model.

**routed9 (89M) and routed10 (190M) replace routed8/routed7-extended as the recommended next
runs.** routed8's code and routed7's extension plan are still valid and still wired up (kept
as a "pure volume, matched recipe" control if you want that comparison too), but routed9/10 are
the more targeted bet.

## Mechanism

**Data source**: nlp domain re-pointed to `deepmind/pg19` (Project Gutenberg books - long-form
narrative, ungated, confirmed same `"text"` row schema as FineWeb/OpenWebText, no extractor
changes needed). `_apply_books_nlp_source()` in `scripts/train_stage2_pod.py`, same mutation
pattern as routed7/8's OpenWebText override.

**Mixture upweight**: `PackedRoutedStream` gained a `force_domain`/`force_domain_snippet_words`
option (`src/data/stage2_routed_stream.py`). nlp now appears in **every** synthetic doc
(guaranteed, not the usual ~75% chance from random sampling) at a 600-word snippet vs the other
domains' default 250. Expected share: `600 / (600 + 2*250) ≈ 55%` of tokens (2 = expected count
of *other* domains per doc, since `k~Uniform{2,3,4}` and nlp no longer competes for its slot).
code/math/science keep their existing sources and per-appearance snippet length untouched.

**Warm start**: `_warm_start_from_parent()` loads the parent checkpoint's full state dict
*except* `embeddings.nlp.*` / `type_embeddings.nlp.*` / `projections.nlp.*` / `heads.nlp.*`
(verified locally against a real `MoTRoutedModel` instance - exactly 6 tensors match those
prefixes, all others load correctly). Those are reinitialized fresh: the new PG-19-fit
tokenizer has the same vocab *size* as before (same `NLP_*_VOCAB` constants → same tensor
shapes) but a different vocab *content* - token id 500 means something different now, so the
parent's learned weights for those 6 tensors would be confidently wrong, not just stale.
Backbone + code/math/science tables transfer directly, unaffected by the nlp tokenizer swap.

- **routed9** (89M) warm-starts from the plain `routed` champion's checkpoint.
- **routed10** (190M) warm-starts from **routed7's** checkpoint (not routed-large) - it
  inherits routed7's already-measured OpenWebText-source improvement on top of scale, then
  gets the books/upweight treatment layered on top of that.

**Differential LR ("cooldown")**: the freshly-reinitialized nlp branch trains at the normal
LR; everything warm-started (backbone + code/math/science) trains at `0.1x` that
(`COOLDOWN_BACKBONE_LR_SCALE` in `stage2_config.py`) via separate optimizer param groups, so
the cooldown phase doesn't undo what the parent checkpoint already learned while nlp catches
up. `COOLDOWN_STEPS = 50000` (~1/3 of a full run) - cheap, and the backbone doesn't need a full
run's worth of steps to adapt when it's already trained.

## The nlp tokenizer needs retraining on books first — don't skip this

Same reasoning as routed7's OpenWebText tokenizer: training on PG-19 text with a tokenizer
fit on FineWeb/OpenWebText is out of its fitting distribution.

```bash
cd /workspace/repo && python3 scripts/retrain_nlp_tokenizer_books.py
```

Saves to `tokenizers_stage2_books/nlp/`, separate from `tokenizers_stage2/` and
`tokenizers_stage2_owt/`. **Default sample size is 3000 books, not 50000** - PG-19 rows are
whole books (tens to hundreds of thousands of words each), so 3000 is already a much larger
corpus by word count than 50000 OpenWebText pages; 50000 books would be impractically slow to
download and fit. (While building this, found and fixed a real bug: `sample_domain()` in
`src/data/stage2_sample_for_tokenizers.py` silently ignored its caller's requested sample size
and always pulled a hardcoded 50000 rows - harmless for OpenWebText/routed7 since that
happened to match the default anyway, but would have been a real problem here. Fixed to
actually respect the requested count; both retrain scripts now pass it through correctly.)

CPU-only, run on the pod before training:

```
tokenizers_stage2_books/nlp/surface.model
tokenizers_stage2_books/nlp/surface.vocab
tokenizers_stage2_books/nlp/syllable_vocab.json
tokenizers_stage2_books/nlp/morpheme_vocab.json
```

## Launch

**Prerequisite**: the parent checkpoint must already exist on this pod's local disk before
warm-start can find it - `checkpoints/routed_step*.pt` for routed9, `checkpoints/large_routed7_step*.pt`
for routed10. If it's not there, `scp` it over from wherever it's checkpointed, or pull it from
the Modal volume (`modal volume get mot-stage2-data checkpoints/<file> checkpoints/<file>`).
If no parent checkpoint is found, `_warm_start_from_parent` logs a clear warning and falls back
to random init rather than failing silently or crashing - but that defeats the point, so verify
it's there first:

```bash
ls checkpoints/routed_step*.pt        # for routed9
ls checkpoints/large_routed7_step*.pt  # for routed10
```

**Calibrate first**, same as every new arm - this exercises the new loader path
(`force_domain`), the books tokenizer, and (in train, not calibrate) the warm-start/differential-LR
machinery for the first time on real data:

```bash
cd /workspace/repo && git pull origin main
python3 scripts/train_stage2_pod.py calibrate --arm routed9 --steps 30
python3 scripts/train_stage2_pod.py calibrate --arm routed10 --steps 30
```

Check for:
1. `[routed9/10] nlp domain source overridden: FineWeb -> deepmind/pg19`
2. `routed9 (base) params: ...` / `routed10 (large) params: ...` - should match the champion's
   / routed7's param counts almost exactly (same architecture).
3. No traceback. (Calibrate uses a fresh random-init model and a flat LR - it does not
   exercise warm-start itself, only the data/model plumbing. Warm-start fires the first time
   you launch `train`.)

If clean, launch training (50,000 steps, the cooldown target):

```bash
nohup python3 scripts/train_stage2_pod.py train --arm routed9 --steps 50000 > train_routed9.log 2>&1 &
nohup python3 scripts/train_stage2_pod.py train --arm routed10 --steps 50000 > train_routed10.log 2>&1 &
```

Watch the very first log lines for:
```
warm-started from checkpoints/routed_step150000.pt (parent step 150000): 31 tensors loaded, 6 nlp-domain tensors left at fresh init
```
(or the routed10/routed7 equivalent) - confirms the warm start actually fired rather than
silently falling back to random init.

**Expect an early loss spike/adjustment period.** nlp's embedding and head start from scratch
while the rest of the model is already fully trained - this is the expected cost of the
warm-start design, not a crash. **routed9/routed10 are NOT in the adaptive-controller list**
(unlike pooled/hybrid/routed2-4) - there's no automatic spike-guard here. Watch the first few
thousand steps on the log; if loss is genuinely diverging rather than just elevated, that's
worth flagging before letting it run unattended overnight.

Checkpoints: `checkpoints/routed9_step*.pt` (base scale, no prefix) and
`checkpoints/large_routed10_step*.pt` (large scale, per the usual naming rule). Resumes
normally from its own checkpoint on restart - the warm-start path only fires once, the first
time, when no `routed9_step*`/`large_routed10_step*` file exists yet.

## Eval

Both `evaluate()` and `evaluate_lambada()` already know about `routed9`/`routed10` (books
tokenizer dir, `STREAM_SOURCES["nlp"]` → pg19 override in `evaluate()`, large-scale forcing for
routed10 only):

```bash
python3 -m modal run stage2_modal.py --step evaluate --arm routed9 --steps 50000
python3 -m modal run stage2_modal.py --step evaluate-lambada --arm routed9 --steps 50000
python3 -m modal run stage2_modal.py --step evaluate --arm routed10 --steps 50000
python3 -m modal run stage2_modal.py --step evaluate-lambada --arm routed10 --steps 50000
```

(One-time: push the books tokenizer to the Modal volume first, same as routed7's nlp tokenizer -
`python3 -m modal volume put mot-stage2-data tokenizers_stage2_books/nlp tokenizers_stage2_books/nlp`.)

Record via `src.eval.metrics.record()` with `arm="routed9"` / `arm="routed10"`, then
regenerate + republish the dashboard.

## Honest simplifications / not built here

- **Differential LR is a single flat 0.1x scale**, not per-layer or annealed separately from
  the main cosine schedule - a simplification of "lower backbone LR" from the outside review,
  not the full recipe (e.g. no separate warmup/decay curve per param group).
- **The near-free diagnostics the outside review proposed running first** (single-token
  coverage ceiling for LAMBADA targets under the nlp tokenizer, an error taxonomy on
  synonym-confusion vs copy-failure vs rare-word misses, an EM-definition sanity check) are
  **not automated in this pass** - genuinely cheap and worth doing before spending GPU-hours,
  but need their own script against saved eval logs/tokenizer vocab. Flagging as a good next
  step, not silently skipping it.
- **Later-priority levers from the same review** (margin/contrastive cloze tuning for
  synonym-confusion errors, a second-stage nlp-expert finetune with the backbone frozen,
  2048-context for a books phase) are real, reasonable follow-ups but out of scope for
  tonight - each needs its own design pass once routed9/10's results are in.
