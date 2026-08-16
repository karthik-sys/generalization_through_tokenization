"""Train stage-2-sized tokenizers from the samples pulled by
src/data/stage2_sample_for_tokenizers.py. Run once before train_stage2.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import load_from_disk

from src.model.stage2_config import (
    BASELINE_VOCAB_SIZE,
    DOMAIN_TAG,
    DOMAIN_VOCAB_SIZES,
    NLP_MORPHEME_VOCAB,
    NLP_SURFACE_VOCAB,
    NLP_SYLLABLE_VOCAB,
)
from src.tokenizers.bpe_tokenizer import train_bpe
from src.tokenizers.nlp_hybrid_tokenizer import train_nlp_tokenizer


def _load_sample_texts(domain: str, sample_dir: str = "data/stage2_tokenizer_sample") -> list[str]:
    ds = load_from_disk(f"{sample_dir}/{domain}")
    return [r["text"] for r in ds]


def main(out_dir: str = "tokenizers_stage2", sample_dir: str = "data/stage2_tokenizer_sample") -> None:
    all_texts_by_domain = {}
    for domain, vocab_size in DOMAIN_VOCAB_SIZES.items():
        texts = _load_sample_texts(domain, sample_dir)
        all_texts_by_domain[domain] = texts
        print(f"training {domain} BPE tokenizer on {len(texts)} sampled docs, vocab={vocab_size}")
        train_bpe(texts, model_prefix=f"{out_dir}/{domain}/model", vocab_size=vocab_size)

    nlp_texts = _load_sample_texts("nlp", sample_dir)
    all_texts_by_domain["nlp"] = nlp_texts
    print(f"training nlp hybrid tokenizer on {len(nlp_texts)} sampled docs")
    train_nlp_tokenizer(
        nlp_texts, out_dir=f"{out_dir}/nlp",
        surface_vocab_size=NLP_SURFACE_VOCAB,
        syllable_vocab_size=NLP_SYLLABLE_VOCAB,
        morpheme_vocab_size=NLP_MORPHEME_VOCAB,
    )

    tagged_docs = []
    for domain, texts in all_texts_by_domain.items():
        tag = DOMAIN_TAG[domain]
        tagged_docs.extend(f"{tag}\n{t}" for t in texts)
    print(f"training unified BPE baseline on {len(tagged_docs)} tagged sampled docs, vocab={BASELINE_VOCAB_SIZE}")
    train_bpe(tagged_docs, model_prefix=f"{out_dir}/baseline_bpe/model", vocab_size=BASELINE_VOCAB_SIZE)


if __name__ == "__main__":
    main()
