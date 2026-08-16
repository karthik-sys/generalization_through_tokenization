"""Train every stage-1 tokenizer:
  - per-domain BPE for code/math/science (spec §6)
  - the NLP syllable/morpheme hybrid (spec §7)
  - the unified BPE baseline over the full tagged corpus (spec §9, Baseline A)

Run after src/data/prepare_corpus.py. Outputs go to tokenizers/<domain>/ and
tokenizers/baseline_bpe/, both gitignored (reproducible from data + this script).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.prepare_corpus import DOMAIN_TAG, load_domain_texts
from src.tokenizers.bpe_tokenizer import train_bpe
from src.tokenizers.nlp_hybrid_tokenizer import train_nlp_tokenizer

VOCAB_SIZES = {
    "code": 8000,
    "math": 8000,
    "science": 8000,
}
# Bumped from 4000 -> 8000 for the scaled-up (v2) run: corpora are now ~3x bigger per
# domain (15k real-world docs each vs 5k, plus a much larger math textbook side after
# fixing the real-world starvation - see docs/dataset_methodology.md). Keep all three
# BPE domains at the same vocab size unless a real reason to diverge shows up - it
# removes vocab budget as a confound in comparisons (math@800-vs-baseline@8000 already
# burned us once on this, see docs/architecture_spec.md decision log).


def train_domain_tokenizers(data_dir: str = "data", out_dir: str = "tokenizers") -> None:
    for domain, vocab_size in VOCAB_SIZES.items():
        texts = load_domain_texts(domain, data_dir)
        print(f"training {domain} BPE tokenizer on {len(texts)} docs, vocab={vocab_size}")
        train_bpe(texts, model_prefix=f"{out_dir}/{domain}/model", vocab_size=vocab_size)

    nlp_texts = load_domain_texts("nlp", data_dir)
    print(f"training nlp hybrid tokenizer on {len(nlp_texts)} docs")
    train_nlp_tokenizer(
        nlp_texts, out_dir=f"{out_dir}/nlp",
        surface_vocab_size=8000, syllable_vocab_size=4000, morpheme_vocab_size=4000,
    )


def train_baseline_tokenizer(
    corpus_path: str = "data/corpus/train.txt", out_dir: str = "tokenizers", vocab_size: int = 16000
) -> None:
    text = Path(corpus_path).read_text(encoding="utf-8")
    docs = text.split("\n<|endofdoc|>\n")
    print(f"training unified BPE baseline on {len(docs)} tagged docs, vocab={vocab_size}")
    train_bpe(docs, model_prefix=f"{out_dir}/baseline_bpe/model", vocab_size=vocab_size)


if __name__ == "__main__":
    train_domain_tokenizers()
    train_baseline_tokenizer()
