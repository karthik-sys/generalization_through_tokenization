# Launch: routed7 (routed-large, nlp domain sourced from GPT-2's real corpus)

Code is pushed to `main` (commit `70a1b83`). This doc is everything needed to launch it —
no other context required.

## What it is, and why

The routed-vs-GPT-2 comparison earlier tonight had two real, confirmed gaps: different eval
methodology (GPT-2's published LAMBADA numbers use a stop-word filter — fixed separately,
see `accuracy_stopword_filtered` in the eval output, made no difference to our numbers) and
different training data (routed trains on this project's own 4-domain corpus, GPT-2 trained
on WebText). This run closes the data gap for the domain that actually matters.

WebText itself was never released by OpenAI. **OpenWebTextCorpus** (Skylion007, the standard
open reconstruction) is what the GPT-2-reproduction community treats as the real substitute
— verified it loads cleanly on HF and its row schema (`"text"` field) matches FineWeb's
exactly, so no new extractor code was needed.

**Only the `nlp` domain's source changes** — code/math/science stay on their existing rich
sources (codeparrot, open-web-math, arxiv-abstracts), unchanged. This was a measured
decision, not a shortcut: a real 5000-document sample of OpenWebText showed code at
~1-in-185 docs, math and science at ~1-in-1250 each — genuinely rare in general web text,
not a classifier-quality problem. Forcing all 4 domains out of OpenWebText would have
starved three of them. `nlp` is also the *only* domain LAMBADA is routed through, so
swapping just that one domain directly closes the comparison gap that matters.

Architecture is otherwise identical to `routed --scale large` (the existing scale-test arm)
— `MoTRoutedModel`, `LARGE_MODEL_CFG` (190.5M params), standard switch_weight=50, no
GradNorm, no other changes. This isolates "same architecture, same scale, only the nlp
corpus changed" as cleanly as possible.

## Mechanism (for context, not action needed)

`_apply_openwebtext_nlp_source()` in `scripts/train_stage2_pod.py` mutates the shared
`STREAM_SOURCES["nlp"]` config in-process, before any data stream is built. This is safe —
each arm runs in its own process on its own pod, so it can't affect any other concurrently
running arm — and it's verified correct (tested directly: the mutation propagates through
Python's shared-object import semantics to every module that already imported
`STREAM_SOURCES` by reference, confirmed with a real interpreter test, not just reasoning
about it). Fires automatically for `arm=routed7` — you don't need to pass anything special,
`--scale` is forced to `large` internally regardless of what you pass.

Verified end-to-end short of tokenization (no local tokenizer files to test against): pulled
a real synthetic multi-domain document after the mutation — `nlp` came from OpenWebText
(confirmed: a real CNN Haiti-earthquake article), `code`/`math`/`science` came from their
normal sources unchanged (real Django source, a real Bayes'-theorem blog post, a real
particle-physics abstract).

## Launch

Any idle pod. Pull the code first:

```bash
cd /workspace/repo && git pull origin main
```

**Calibrate first — don't skip this one.** This run touches new code on real tokenizers
for the first time (everything above was tested short of that step):

```bash
python3 scripts/train_stage2_pod.py calibrate --arm routed7 --steps 30
```

Check the output for, in order:
1. `[routed7] nlp domain source overridden: FineWeb -> Skylion007/openwebtext` — confirms
   the mutation fired.
2. `routed7 (large) params: 190,...` — confirms the right config/param count (should match
   `routed --scale large`'s params almost exactly, since architecture is identical).
3. No traceback, step timing prints normally.

If all three check out, launch the real run, backgrounded so it survives a disconnect:

```bash
nohup python3 scripts/train_stage2_pod.py train --arm routed7 --steps 150000 > train.log 2>&1 &
```

Checkpoints save every 1000 steps to `checkpoints/large_routed7_step*.pt` (the `large_`
prefix is automatic — routed7 always runs at large scale). Resumes automatically from the
latest checkpoint on restart, same as every other arm — just re-run the same `train`
command.

## What to expect

- No adaptive controller (not in that arm list — standard switch_weight=50, no dynamic loss
  component, same as plain routed).
- Held-out val CE logs every 10k steps, reads from the same OpenWebText-sourced nlp domain
  (so it's genuinely held-out from the same distribution, not a mismatched split).
- Step time should be close to `routed --scale large`'s (~0.2-0.25s/step observed tonight)
  — same architecture, same context length (1024, unaffected by this change).

## When it's done (or checking in on progress)

Same eval flow as every other arm this session — pull the checkpoint, MD5-verify against
the pod's copy, push to the Modal volume as `large_routed7_step<N>.pt`, then:

```bash
python3 -m modal run stage2_modal.py --step evaluate --arm routed7 --steps 150000
python3 -m modal run stage2_modal.py --step evaluate-lambada --arm routed7 --steps 150000
```

Both eval functions already know about routed7 (forces `LARGE_MODEL_CFG`, and `evaluate()`
also applies the same nlp-source override so the held-out BPB pass scores against the
distribution it actually trained on — `evaluate-lambada` doesn't need that, LAMBADA always
reads the fixed external benchmark regardless of training source).

Record via `src.eval.metrics.record()` (see any recent commit touching `results/metrics.json`
for the exact call shape — use `arm="routed7"`), then regenerate + republish the dashboard
(`python3 scripts/gen_dashboard.py`).

## Not part of tonight's scope

An "N-domain" follow-up (splitting `nlp` further into register-based sub-buckets, so routing
has more than 4 tables to exercise even on this one corpus) was designed and partially built
(`src/data/domain_classifier.py`, `src/data/webtext_domain_stream.py`) but deliberately not
wired into an arm or launched — on inspection, the heuristic sub-classifier's labels were too
noisy to trust (a campaign-rally news report got labeled "narrative", a local-news piece got
labeled "opinion" — real, checked failures, not a hypothetical concern). That needs a better
classifier before it's worth a GPU-night. The code is there, reusable, just not ready.
