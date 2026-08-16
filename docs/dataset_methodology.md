# Dataset methodology

Decisions locked for the MoT project. See [architecture_spec.md](architecture_spec.md) for the base architecture.

## v1 domain set: code, math, science, nlp

Resolves an ambiguity in the original spec (§2.1 said {code, science, nlp}, §3/§9 said {Code, Math, NLP}) — math and science are kept as separate domains since math has a distinct token distribution (heavy LaTeX/symbol density) from general science prose.

"Logic" is explicitly **not** a training domain — reasoning sets like MetaMathQA/MathInstruct/HotpotQA don't get their own embedding table. Logic/reasoning is expected to emerge from math+code+science exposure and is *measured* via downstream evals (§10 of the spec), not tokenized separately.

Music/Accounting/Legal/Medical are deferred past stage 2 — each additional domain is a full extra embedding+unembedding table and tokenizer-training pass, not worth the cost before the core domain-routing hypothesis has signal.

## Data mix: 50/50 textbook-quality vs. real-world, per domain

Not a single global blend — each domain mixes its own textbook and real-world sources at roughly 50/50:

| Domain | Textbook-quality (curated) | Real-world (web/repo) |
|---|---|---|
| Math | `common-pile/libretexts_filtered` (filter to math.libretexts.org rows), `HuggingFaceTB/openstax_paragraphs` (filter by book_title) | `open-web-math/open-web-math` |
| Science | `common-pile/libretexts_filtered` (filter to science subdomains), `HuggingFaceTB/openstax_paragraphs` | `EleutherAI/proof-pile-2` (arXiv subset) |
| Code | `nampdn-ai/tiny-code-textbooks` (gated — same accept-terms+token step as the-stack-smol) | `bigcode/the-stack-v2-dedup` (gated — accept license terms first) |
| NLP | `roneneldan/TinyStories` (stage 1), curated web (stage 2) | `HuggingFaceFW/fineweb`, `sample-10BT` split |

**Verified, not assumed:** `common-pile/libretexts_filtered` rows carry explicit per-row license metadata (confirmed CC-BY on a sampled row) and a source URL back to the actual LibreTexts page — real textbook content, not a synthetic stand-in. `openstax_paragraphs` is structured as full books (chapters → sections → paragraphs) covering many subjects, not just math/science, so it needs a `book_title` filter before use.

**Why:** Precedent from Microsoft's "Textbooks Are All You Need" (Phi models) — at small scale, dense curated data beats raw web volume for benchmark-style evals (HumanEval/GSM8K/MMLU), but the spec's own §10 robustness metrics (noisy text, misspellings, code-switching) require real "messy" data too, which pure-textbook training can't provide.

## Stage-1 (CPU sanity check) picks — small, clean, few MB per domain

| Domain | Dataset | Notes |
|---|---|---|
| Code | `bigcode/the-stack-smol`, `python` config | ~87MB. Full multi-language set is ~2.9GB — too big for stage 1, so restricted to one language |
| Math | `HuggingFaceH4/MATH-500` | ~200KB, 500 rows. **Swapped from `hendrycks/competition_math`**, which returned HTTP 401 (access-restricted as of 2026-08-15 despite `gated: False` in its metadata) |
| Science | `gfissore/arxiv-abstracts-2021`, first 5,000 rows | Full dataset is ~940MB/2M rows; sliced down for stage 1 |
| NLP | `roneneldan/TinyStories`, first 5,000 rows | Full dataset is ~1GB/2.1M rows; sliced down for stage 1 |

## Stage-2 (free-GPU ablation) picks — streamed, larger

| Domain | Dataset | Notes |
|---|---|---|
| Code | `bigcode/the-stack-v2-dedup` | Gated — requires accepting HF license terms; filter to 1-2 languages |
| Math | `open-web-math/open-web-math` | Streamed; preserves LaTeX structure |
| Science | `EleutherAI/proof-pile-2` | Streamed; arXiv subset |
| NLP | `HuggingFaceFW/fineweb`, `sample-10BT` | Streamed |

## Fairness rule: apples-to-apples domain tags

`<domain:X>` tags appear as literal text in **both** the BPE baseline and MoT treatment corpora. BPE tokenizes the tag as an ordinary token with no special routing behavior; MoT uses it to route. This keeps the comparison apples-to-apples so any perplexity/compression gap between arms is attributable to tokenization architecture, not to MoT having access to extra information BPE never saw.

## Practical flags before downloading anything

1. Use `streaming=True` for all stage-2 datasets — several are hundreds of GB to multi-TB.
2. `the-stack-v2-dedup` requires explicit license acceptance per-use on Hugging Face before it will download.
3. Several dataset identifiers floating around in casual research (`hotpot_qa`, `edgar`, `fma_metadata`, legacy script-based loaders) may be stale under HF's `datasets` library — verify exact repo slugs on huggingface.co before hardcoding them into any script.
4. No dataset should be downloaded without confirming the exact file(s)/source/approximate size first.

## How to apply

When building dataset loaders or the training pipeline, hold the mix ratio and tagging policy identical across the BPE baseline and MoT treatment arms. Don't silently change either without flagging it — it breaks the ablation's ability to isolate what's actually causing a performance difference (same principle as the spec's own §7 2x2 axis-separation between domain-routing and syllable-tokenization).
