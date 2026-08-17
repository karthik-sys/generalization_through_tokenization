"""One-time step for arm="routed7": retrain the nlp tokenizer on OpenWebText instead of
FineWeb, so it's actually fit to the corpus it'll be encoding, not just applied to it
out-of-distribution. Run ONCE before launching routed7's training - does not touch the
shared tokenizers_stage2/ directory every other arm depends on; writes to a separate
tokenizers_stage2_owt/ directory instead.

CPU-only (SentencePiece + Morfessor + pyphen, no GPU needed) - can run on the training pod
before the real run starts, or anywhere else with the repo's deps installed.

Usage:
  python3 scripts/retrain_nlp_tokenizer_openwebtext.py
  python3 scripts/retrain_nlp_tokenizer_openwebtext.py --out-dir tokenizers_stage2_owt --n-sample 50000
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
    # Point ONLY the nlp source at OpenWebText for this process - same override routed7's
    # training uses (_apply_openwebtext_nlp_source in train_stage2_pod.py), applied here so
    # sample_domain("nlp", ...) pulls from the right corpus. code/math/science are never
    # touched by this script at all.
    STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}

    sample_dir = f"{out_dir}/_sample"
    print(f"sampling {n_sample} OpenWebText docs for nlp tokenizer training -> {sample_dir}/nlp", flush=True)
    sample_domain("nlp", out_dir=sample_dir)

    from datasets import load_from_disk
    texts = [r["text"] for r in load_from_disk(f"{sample_dir}/nlp")]
    print(f"training nlp hybrid tokenizer on {len(texts)} OpenWebText docs "
          f"(surface={NLP_SURFACE_VOCAB}, syllable={NLP_SYLLABLE_VOCAB}, morpheme={NLP_MORPHEME_VOCAB})",
          flush=True)
    # Same vocab-size constants as the original FineWeb-trained tokenizer, so the resulting
    # embedding table dimensions match LARGE_MODEL_CFG's expectations exactly - no model
    # architecture changes needed, this only changes what the vocabulary IS, not its size.
    train_nlp_tokenizer(
        texts, out_dir=f"{out_dir}/nlp",
        surface_vocab_size=NLP_SURFACE_VOCAB,
        syllable_vocab_size=NLP_SYLLABLE_VOCAB,
        morpheme_vocab_size=NLP_MORPHEME_VOCAB,
    )
    print(f"done - retrained nlp tokenizer at {out_dir}/nlp", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tokenizers_stage2_owt")
    parser.add_argument("--n-sample", type=int, default=50000)
    args = parser.parse_args()
    main(args.out_dir, args.n_sample)
