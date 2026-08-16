# generalization_through_tokenization

Mixture-of-Tokenizers (MoT): a domain-routed, multi-tokenizer LLM architecture, tested against unified BPE at matched small scale.

- Full architecture spec: [docs/architecture_spec.md](docs/architecture_spec.md)
- Dataset methodology (domain set, mix ratios, download sources): [docs/dataset_methodology.md](docs/dataset_methodology.md)

## Status

Stage 1 (CPU sanity check, ~1-10M params) scaffolding in progress. No data downloaded yet — see dataset methodology doc for exact sources before running `src/data/download.py`.

## Layout

```
docs/                  architecture spec + dataset methodology
data/{code,math,science,nlp}/   raw + processed per-domain corpora (gitignored)
src/tokenizers/         per-domain tokenizer training/encoding (code, math, science, nlp)
src/model/              disjoint embeddings, shared backbone, per-domain heads
src/data/               dataset download + domain-tagging scripts
scripts/                stage-1 / stage-2 run scripts
```

## Baselines vs. treatment (spec §9)

- Baseline A: unified byte-level BPE
- Baseline B: unified SentencePiece
- Treatment: MoT, disjoint per-domain tables, hard-commit routing (v0)
- Treatment+: MoT with mined intersection vocabulary (v1, spec §4)
