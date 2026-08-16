# Mixture-of-Tokenizers (MoT) LLM — Architecture Spec

## 1. Core thesis

Standard SOTA models force code, math, and prose into one unified BPE vocabulary. This is a lowest-common-denominator compromise: code gets over-tokenized (long sequences), NLP loses morphological/syllabic structure. Small (~100M param) models trained this way aren't "broken," but they burn capacity re-deriving structure a better tokenizer would encode directly — worse sample efficiency and compression, not zero learning.

Goal: prove a domain-routed, multi-tokenizer architecture beats unified BPE at matched small scale (start ~1-10M params for architecture sanity, then ~50-100M on free GPU tiers for the actual ablation), before considering any paid/scaled compute.

## 2. Domain routing

### 2.1 Training-time (solved, cheap)

Domain labels come from data provenance, not runtime detection:

- GitHub repos → `<domain:code>`
- arXiv / scientific corpora → `<domain:science>`
- Common Crawl / books / prose → `<domain:nlp>`
- (extend per dataset you bring in)

Each document is prefixed with its literal domain tag as a token. No classifier needed — this is just correct data labeling during corpus curation.

### 2.2 Inference-time (no separate router module)

Because domain tags are literal tokens seen during training, the model learns to emit them itself as part of normal autoregressive generation — same mechanism as models learning to emit `<tool_call>` tokens. No external classifier, deterministic heuristic, or learned gating network needed. This also solves domain-boundary handling: a mid-sequence domain switch is just the model predicting a `<domain:X>` control token, which is then used to constrain/route the next span of decoding (logit masking into that domain's ID range) — the same pattern as constrained grammar decoding.

Decision: skip deterministic runtime heuristics (fenced-code detection, LaTeX delimiter regex) entirely for v1. They were a reasonable fallback for unlabeled data, but self-emitted domain tokens are strictly better once training data is domain-tagged at the source.

## 3. Embedding architecture — disjoint tables, not collision-prone shared IDs

Problem avoided: if Code-tokenizer ID 402 and NLP-tokenizer ID 402 shared one embedding table, they'd collide (two unrelated meanings competing for one row).

Solution: N separate embedding tables, one per domain tokenizer (Code, Math, NLP, ...).

- ID 402 in the Code table and ID 402 in the NLP table are different parameters that happen to share an integer index for bookkeeping — not a real collision, nothing forces the model to relate them.
- Domain token (from §2) tells you which table to use before the embedding lookup — so at inference you only run one table per step, not all N. This is standard hard-routed MoE logic applied at the embedding layer instead of the FFN layer.
- Projection layer maps each domain table into a shared `d_model` residual stream dimension if table sizes/dims differ.
- Backbone (attention + FFN transformer layers) is fully shared across domains — this is where cross-domain generalization actually happens, via ordinary attention over mixed-domain context. Nothing special needs to be built here; it's a consequence of one shared backbone processing projected vectors from any domain table.
- Separate per-domain output/unembedding heads, same logic as input embeddings.

## 4. Intersection vocabulary (cross-domain shared tokens)

Motivation: tokens like "gradient" (math vs. ML/code) or "kernel" (CS vs. OS/systems) may share meaning across domains, or may be false-positive lookalikes with unrelated meanings. Fully disjoint tables force the model to re-derive any real connection from scratch via co-occurrence every time. A shared intersection table gives the model a head start.

Decision: do NOT pre-label intersection vocabulary by hand. Human-asserted cross-domain pairs risk bias/skew and are exactly the kind of interpretive judgment that should come from the model's own learned geometry, not curator intuition. Instead, use a bootstrapped, iterative mining pipeline:

1. Train v0 with fully disjoint per-domain tables (no intersection vocab). This is the clean baseline.
2. Mine candidates from v0's learned embedding space: compute similarity (cosine distance after projection into shared `d_model` space) between tokens from different domain tables. Cluster tokens that land close together despite coming from different tables. Fully unsupervised — reads off what the model already inferred from co-occurrence.
3. Threshold + validate: high-similarity cross-domain pairs become intersection-vocab candidates. This step is also the natural filter for false positives (e.g. CS-"kernel" vs. OS-"kernel"): if their contexts don't actually overlap in usage, similarity should be lower than genuine polysemy pairs.
4. Retrain/continue-train v1 with mined intersection vocab spliced in, replacing redundant per-domain rows for merged tokens.
5. Repeat periodically as embeddings sharpen — intersection vocab is a living artifact, not fixed at v1.

Known cost: this loop requires a full v0 training run just to get the mining signal — sequenced, not parallel, and more expensive than either pure-disjoint-forever or hand-labeled-intersection. Justified only if the hypothesis (shared vocab improves cross-domain transfer) holds up in v0-vs-v1 comparison.

## 5. Backtracking / fallback — explicitly deferred

Discussed: if a domain table "doesn't suffice" mid-generation, the model could backtrack and retry with a different domain table. Deferred from v1. Triggering this needs a signal (high output entropy, low top-1 probability, or an explicit `<domain:uncertain>` emission), and acting on it means invalidating/recomputing part of the KV cache — expensive, and an open problem similar to speculative decoding.

v1 policy: hard commit. Model emits a domain token, you route to that table, no takebacks. Measure empirically how often this is wrong before building any repair mechanism for a failure mode that hasn't been confirmed as common.

## 6. Tokenizer construction (per domain)

- Byte-safe preprocessing: Unicode normalization, raw-byte fallback preserved for OOV robustness in every domain.
- Per-domain tokenizer training:
  - Code: standard BPE/SentencePiece over code corpora (syntax-aware merging where possible).
  - Math: symbolic/LaTeX-aware tokenization.
  - NLP: candidate for syllable/morpheme-based tokenization (see §7) rather than plain BPE.
- Each token carries a type id (domain + subtype: surface/syllable/morpheme/byte) so the model can condition on granularity, not just identity.

## 7. NLP tokenizer — syllable/morpheme hybrid (optional enhancement, separate axis from routing)

This was the original starting idea and remains a valid NLP-branch-specific enhancement, orthogonal to domain routing:

- Hybrid unit types: surface subwords, syllables, morphemes, phonemes/phonetic pieces, semantic tags.
- Encoding backoff: try surface subword match first → fall back to syllable segmentation → fall back to morpheme → final fallback to byte-level. **(v1 default — see decision log; parallel multi-stream fusion considered and deferred.)**
- Tools: SentencePiece (unigram + BPE), Morfessor (unsupervised morphemes), Pyphen/rule-based or neural syllabifiers, Epitran/phonemizer if phonetic representation is needed.
- Explicitly rejected approach: do not translate the entire corpus into one "maximum-meaning-per-syllable" language. Syllabification/morphology is language-dependent and this is brittle and expensive. If a canonical interlingua is wanted, learn it jointly via training objectives (contrastive cross-lingual alignment), not hard translation.
- This composes with the domain-routing architecture as: does the syllable-based NLP tokenizer (vs. standard BPE) improve the NLP branch specifically, independent of whether domain routing itself helps. Run as a 2x2: {unified BPE, MoT-standard-NLP, MoT-syllable-NLP} × baseline, to see if domain-routing gains and syllable-tokenization gains are additive or redundant.

## 8. Staged compute plan (no GPU available)

1. CPU-only (architecture sanity check): ~1-10M params, small corpus (few MB/domain), short sequences. Goal: verify the pipeline runs correctly — router emits domain tokens, disjoint tables load and route correctly, loss decreases at all. Not expected to show the actual hybrid-vs-BPE hypothesis signal — too small/noisy.
2. Free-tier GPU (real ablation): Google Colab (free T4, limited daily hours), Kaggle (~30hrs/week free GPU quota), or Lightning AI free tier. Target ~50-100M params — enough to get a real comparable perplexity/downstream signal vs. a BPE baseline at matched compute. This is the stage that actually proves or kills the hypothesis.
3. Paid/scaled compute: only pursued once stage 2 shows a real signal worth scaling.

## 9. Baselines to train (matched compute/params at each stage)

- Baseline A: unified byte-level BPE (current SOTA-style, single vocabulary).
- Baseline B: unified SentencePiece (surface subwords, no domain routing).
- Treatment: MoT with domain-tagged data, disjoint per-domain tables, hard-commit routing (v0 — no intersection vocab).
- Treatment+: MoT v1 with mined intersection vocabulary (after bootstrap loop, §4).
- Optional ablations: syllable-only NLP branch, morpheme-only NLP branch, remove type embeddings, remove intersection vocab (isolate where gains come from).

**Fairness rule (decided post-spec):** `<domain:X>` tags appear as literal text in both baseline and treatment corpora. BPE tokenizes the tag as an ordinary token with no special routing behavior; MoT uses it to route. This isolates tokenization architecture as the only variable — see [dataset_methodology.md](dataset_methodology.md).

## 10. Evaluation metrics

- Intrinsic: perplexity per domain, tokens-per-word / tokens-per-character (compression), OOV/coverage rate.
- Boundary-adjacent perplexity: perplexity specifically on the 5-10 tokens immediately after a domain switch. Diagnostic for whether domain embedding spaces are actually compatible in the shared backbone, or whether the model has a jump-discontinuity at switches. Also the key signal for whether intersection vocab (§4) would help.
- Robustness: performance on noisy text, code-switching, misspellings, unseen morphology.
- Downstream tasks: HumanEval (code), GSM8K (math), MMLU (general knowledge/NLP), plus NER/morphological inflection if syllable/morpheme NLP branch is included.
- Cross-domain transfer: train with some domains/languages present, test on related unseen ones.
- Compute: tokens/sec, latency, FLOPs per token (routing/projection overhead vs. unified BPE baseline).

## 11. Open questions to resolve during implementation

- Exact size of each per-domain vocabulary (Code / Math / NLP tables) — needs sizing pass once real corpora are in hand.
- Where projection layers reconcile dimension mismatches between domain tables and shared `d_model` (single linear layer vs. small MLP per domain).
- Threshold value for intersection-vocab similarity clustering in step 3 of §4 — needs empirical tuning against known polysemy/false-positive pairs.
- ~~Domain tag set beyond {code, math, nlp}~~ — resolved: v1 domain set is {code, math, science, nlp}. See [dataset_methodology.md](dataset_methodology.md).

## 12. Datasets

See [dataset_methodology.md](dataset_methodology.md) for the finalized v1 domain set, dataset picks per domain/stage, and the textbook/web mix ratio.

## Decision log (post-spec)

- v1 domain set resolved to {code, math, science, nlp} (spec originally had an inconsistency between §2.1 and §3/§9).
- "Logic" dropped as a training domain — treated as downstream eval only (GSM8K, HotpotQA-style), not a tokenizer table.
- Music/Accounting/Legal/Medical deferred past stage 2.
- Data mix: 50/50 textbook-quality vs. real-world, applied per domain.
- Domain tags shown as literal text to both baseline and treatment arms (apples-to-apples).
- NLP tokenizer: encoding-backoff (single stream) is the v1 default over parallel multi-stream fusion; fusion may be revisited as a later ablation if backoff underperforms.
- Resolves §11's open question on per-domain vocab sizing (first pass): code/math/science BPE tokenizers all trained at vocab_size=4000. Math was initially trained at 800 (assumed data-scarcity limit for its small corpus) - this was wrong and created a real distortion: at 800 vs baseline's 8000, math's tokens-per-word ratio came out to 1.274 (apparently *worse* than unified BPE) purely from vocab starvation. Retrained at 4000 (which the corpus does support, some merges land near freq=9 but it completes), the ratio flips to 0.873 (better, consistent with science/nlp). Keep domain vocab sizes matched unless a specific domain has a real reason to diverge - unequal vocab budgets are a confound, not a finding.

## Finding: §4's mining premise fails at v0 (measured 2026-08-15)

§4 assumes a v0 model trained with fully disjoint tables will, through co-occurrence,
place genuinely-shared tokens ("gradient" in math vs. code) near each other once
projected into the shared `d_model` space - and that mining that geometry recovers
intersection-vocab candidates. Measured against a real 4-epoch stage-1 checkpoint, it
does not. Cross-domain cosine similarity is statistically indistinguishable from random:

| measurement                                   | top-per-token mean cosine |
|-----------------------------------------------|---------------------------|
| random control (independent 128-d unit vectors) | 0.3276                  |
| real cross-domain (code<->math)                 | 0.3265                    |
| real cross-domain (math<->nlp)                  | 0.3425                    |
| real within-domain (math)                       | 0.5117                    |
| real within-domain (nlp)                        | 0.5140                    |

Within-domain structure is clearly learned (0.51 vs 0.33 random). Cross-domain is not.
Two structural reasons, both consequences of the v1 design rather than bugs:

1. **Every training document is single-domain.** The shared backbone never processed
   tokens from two domains in one sequence, so the co-occurrence mechanism §4 depends on
   never operated across domains at all.
2. **Per-domain projection layers permit free rotation.** Each domain owns its
   `Linear(emb_dim, d_model)`. Two domains' projected spaces can be arbitrarily rotated
   relative to one another at zero loss cost, because no term in the LM objective ever
   compares them.

Consequence for sequencing: §4's mining step is gated not merely on "a trained
checkpoint exists" but on "a checkpoint trained with cross-domain context exists". The
§2.2 mid-sequence routing work (multi-domain documents, control-vocab switching) is
therefore a prerequisite for §4, not an independent feature - it supplies the only
mechanism that could create cross-domain alignment pressure. Mining should be re-run
against a routed checkpoint before concluding anything about intersection vocabulary.
