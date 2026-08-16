"""Tokens-per-word compression comparison (spec §10, first intrinsic metric): does each
domain's own tokenizer actually produce shorter sequences than the unified BPE baseline
for that domain's text? This needs no trained model - it's a property of the tokenizers
alone, so it's the fastest real signal available before any training finishes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.build_examples import TokenizerBundle
from src.data.prepare_corpus import DOMAIN_TAG, load_domain_texts


def main(data_dir: str = "data", sample_per_domain: int = 300) -> None:
    bundle = TokenizerBundle()
    print(f"{'domain':<10} {'words':>10} {'domain_tok':>12} {'baseline_tok':>13} {'domain_t/w':>11} {'baseline_t/w':>13} {'ratio':>7}")
    for domain in DOMAIN_TAG:
        texts = load_domain_texts(domain, data_dir)[:sample_per_domain]
        total_words = sum(len(t.split()) for t in texts)
        domain_tokens = 0
        for t in texts:
            ids, _ = bundle.encode_domain(domain, t, max_len=10**9)
            domain_tokens += len(ids)
        baseline_tokens = 0
        for t in texts:
            ids = bundle.encode_baseline(t, max_len=10**9)
            baseline_tokens += len(ids)

        dtw = domain_tokens / total_words
        btw = baseline_tokens / total_words
        ratio = dtw / btw
        print(
            f"{domain:<10} {total_words:>10} {domain_tokens:>12} {baseline_tokens:>13} "
            f"{dtw:>11.3f} {btw:>13.3f} {ratio:>7.3f}"
        )
    print("\nratio < 1.0 means the domain-specific tokenizer is MORE compressed (fewer tokens/word) than unified BPE.")
    print("Domain tokenizers: 8000 each; nlp hybrid combines surface+syllable+morpheme+byte; baseline: 16000 shared.")
    print("Baseline still has a 2x vocab budget over any single domain table - not perfectly matched, but proportionally")
    print("the same ratio as the v1 comparison (4000 vs 8000), so it's at least a consistent methodology across runs.")


if __name__ == "__main__":
    main()
