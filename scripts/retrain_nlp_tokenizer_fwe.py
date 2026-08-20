"""One-time step for routed34/36 (the GPT-2-budget batch): retrain the nlp tokenizer on the
60% FineWeb-Edu / 40% OpenWebText blend specified for those arms, instead of applying the
plain-FineWeb or OWT-only tokenizer out of distribution. Same pattern as
retrain_nlp_tokenizer_openwebtext.py (routed7's precedent) - samples each source separately
(sample_domain only pulls from ONE STREAM_SOURCES override at a time) and concatenates before
training, rather than building general weighted-blend sampling just for this one-time step.

CPU-only (SentencePiece + Morfessor + pyphen, no GPU needed).

Usage:
  python3 scripts/retrain_nlp_tokenizer_fwe.py
  python3 scripts/retrain_nlp_tokenizer_fwe.py --out-dir tokenizers_stage2_fwe --n-sample 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.stage2_sample_for_tokenizers import sample_domain
from src.model.stage2_config import FWE_NLP_SOURCES, NLP_MORPHEME_VOCAB, NLP_SURFACE_VOCAB, NLP_SYLLABLE_VOCAB, STREAM_SOURCES
from src.tokenizers.nlp_hybrid_tokenizer import train_nlp_tokenizer


def main(out_dir: str, n_sample: int) -> None:
    sample_dir = f"{out_dir}/_sample"
    all_texts: list[str] = []
    for source_cfg, weight in FWE_NLP_SOURCES:
        n_docs = round(n_sample * weight)
        # Point STREAM_SOURCES["nlp"] at THIS blend member only, matching the override pattern
        # every existing nlp-source-swap script/arm uses (_apply_openwebtext_nlp_source etc.) -
        # sample_domain reads STREAM_SOURCES[domain] internally, one source per call.
        STREAM_SOURCES["nlp"] = {k: v for k, v in source_cfg.items() if k != "domain_filter"}
        tag = source_cfg["path"] + (f"[{source_cfg['name']}]" if source_cfg.get("name") else "")
        print(f"sampling {n_docs} docs ({weight:.0%} of {n_sample}) from {tag} -> {sample_dir}/nlp_{tag}", flush=True)
        this_sample_dir = f"{sample_dir}/{tag.replace('/', '_')}"
        sample_domain("nlp", out_dir=this_sample_dir, n_docs=n_docs)
        from datasets import load_from_disk
        all_texts.extend(r["text"] for r in load_from_disk(this_sample_dir))

    print(f"training nlp hybrid tokenizer on {len(all_texts)} blended docs "
          f"(surface={NLP_SURFACE_VOCAB}, syllable={NLP_SYLLABLE_VOCAB}, morpheme={NLP_MORPHEME_VOCAB})",
          flush=True)
    # Same vocab-size constants as every other nlp tokenizer variant, so embedding table
    # dimensions stay compatible with the shared model configs - only the vocabulary CONTENT
    # changes, not the architecture it has to fit into.
    train_nlp_tokenizer(
        all_texts, out_dir=f"{out_dir}/nlp",
        surface_vocab_size=NLP_SURFACE_VOCAB,
        syllable_vocab_size=NLP_SYLLABLE_VOCAB,
        morpheme_vocab_size=NLP_MORPHEME_VOCAB,
    )
    print(f"done - retrained nlp tokenizer at {out_dir}/nlp", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tokenizers_stage2_fwe")
    parser.add_argument("--n-sample", type=int, default=50000)
    args = parser.parse_args()
    main(args.out_dir, args.n_sample)
