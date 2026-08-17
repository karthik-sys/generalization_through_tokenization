"""One-time step for arms="routed9"/"routed10": retrain the nlp tokenizer on PG-19 (Project
Gutenberg books) instead of FineWeb/OpenWebText, so it's fit to the corpus it'll be encoding.
Mirrors scripts/retrain_nlp_tokenizer_openwebtext.py exactly, swapping the source dataset -
see docs/handoff_optimized_89m_190m.md for why books: LAMBADA's passages are themselves
book-derived, and the long-range antecedent->target dependency the benchmark tests is a skill
short web pages rarely exercise, unlike continuous book narrative.

Writes to a separate tokenizers_stage2_books/ directory - does not touch tokenizers_stage2/
(shared) or tokenizers_stage2_owt/ (routed7/routed8's).

CPU-only (SentencePiece + Morfessor + pyphen, no GPU needed).

Usage:
  python3 scripts/retrain_nlp_tokenizer_books.py
  python3 scripts/retrain_nlp_tokenizer_books.py --out-dir tokenizers_stage2_books --n-sample 3000

Default --n-sample is 3000, not 50000 like the OpenWebText script - PG-19 rows are whole
books (tens to hundreds of thousands of words each), not short web pages, so 3000 of them is
already a far larger word-count corpus than 50000 OpenWebText pages. 50000 books would be
impractically slow/large to download and fit SentencePiece over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.stage2_sample_for_tokenizers import sample_domain
from src.model.stage2_config import NLP_MORPHEME_VOCAB, NLP_SURFACE_VOCAB, NLP_SYLLABLE_VOCAB, STREAM_SOURCES
from src.tokenizers.nlp_hybrid_tokenizer import train_nlp_tokenizer


def main(out_dir: str, n_sample: int) -> None:
    # Point ONLY the nlp source at PG-19 for this process - same override routed9/routed10's
    # training uses (_apply_books_nlp_source in train_stage2_pod.py). code/math/science are
    # never touched by this script.
    STREAM_SOURCES["nlp"] = {"path": "deepmind/pg19", "name": None, "gated": False}

    sample_dir = f"{out_dir}/_sample"
    print(f"sampling {n_sample} PG-19 docs for nlp tokenizer training -> {sample_dir}/nlp", flush=True)
    sample_domain("nlp", out_dir=sample_dir, n_docs=n_sample)

    from datasets import load_from_disk
    # PG-19 rows are whole books (avg ~400K-800K chars) - SentencePiece only needs enough text
    # per book to capture its subword/syllable/morpheme statistics, not the full text, and
    # fitting on all of it (3000 x ~400K+ chars = 1B+ chars) makes the unigram trainer's
    # suffix-array substring-extraction step impractically slow (confirmed: still running after
    # an hour on a real pod, vs. minutes for OpenWebText's naturally-short ~150-300M char
    # corpus). Capping each book's FITTING sample to 30K chars (~5K words, still clearly
    # book-length narrative, not a web snippet) brings total corpus size back into the same
    # regime OpenWebText's tokenizer fit quickly at. This only affects what the tokenizer is
    # FIT on - actual training still streams full, untruncated PG-19 books via
    # _apply_books_nlp_source in train_stage2_pod.py.
    MAX_CHARS_PER_DOC = 30_000
    texts = [r["text"][:MAX_CHARS_PER_DOC] for r in load_from_disk(f"{sample_dir}/nlp")]
    print(f"training nlp hybrid tokenizer on {len(texts)} PG-19 docs "
          f"(surface={NLP_SURFACE_VOCAB}, syllable={NLP_SYLLABLE_VOCAB}, morpheme={NLP_MORPHEME_VOCAB})",
          flush=True)
    # Same vocab-size constants as the original tokenizer, so embedding table dimensions stay
    # identical regardless of scale (base or large) - only the vocabulary CONTENT changes.
    train_nlp_tokenizer(
        texts, out_dir=f"{out_dir}/nlp",
        surface_vocab_size=NLP_SURFACE_VOCAB,
        syllable_vocab_size=NLP_SYLLABLE_VOCAB,
        morpheme_vocab_size=NLP_MORPHEME_VOCAB,
    )
    print(f"done - retrained nlp tokenizer at {out_dir}/nlp", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tokenizers_stage2_books")
    parser.add_argument("--n-sample", type=int, default=3000)
    args = parser.parse_args()
    main(args.out_dir, args.n_sample)
