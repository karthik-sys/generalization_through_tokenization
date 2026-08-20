"""Pre-tokenize a domain's weighted multi-source blend to a uint16 binary shard, so training
reads pre-tokenized ids from a memmap instead of live-streaming+tokenizing text every step.

Launch prerequisite for routed34/36 (not an optimization - see the session's own diagnosis:
nlpbranch's live natural-document streaming ran at ~23k tok/s vs routed35's ~295k tok/s on
identical batch/accum settings; the FWE/edu blend + long-doc fraction have the same slow
ingredients nlpbranch's data path did, and would sag toward that rate if streamed live).

Combined vocab size for every domain (nlp: surface+syllable+morpheme = 24000+6000+6000=36000;
code/math/science: ~30-40k each) stays well under uint16's 65535 ceiling, confirmed before
choosing this dtype - a real check, not an assumption.

Usage:
  python3 scripts/build_token_shards.py --domain nlp --target-tokens 6900000000 \
      --tokenizer-dir tokenizers_stage2_fwe --out data_shards/nlp_fwe.bin
  python3 scripts/build_token_shards.py --domain math --target-tokens 1000000000 \
      --tokenizer-dir tokenizers_stage2 --out data_shards/math_edu.bin
  python3 scripts/build_token_shards.py --domain science --target-tokens 1000000000 \
      --tokenizer-dir tokenizers_stage2 --out data_shards/science_edu.bin

Run the nlp tokenizer refit (scripts/retrain_nlp_tokenizer_fwe.py) BEFORE building the nlp
shard - it needs to exist at --tokenizer-dir/nlp first.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from datasets import load_dataset

from src.data.build_examples import TokenizerBundle
from src.data.domain_classifier import classify_domain
from src.model.stage2_config import EDU_MATH_SOURCES, EDU_SCIENCE_SOURCES, FWE_NLP_SOURCES

SOURCE_LISTS = {"nlp": FWE_NLP_SOURCES, "math": EDU_MATH_SOURCES, "science": EDU_SCIENCE_SOURCES}
LOG_EVERY_TOKENS = 10_000_000


def _stream_source(source_cfg: dict):
    """Yields text from one blend member forever (restarts on exhaustion, same multi-epoch-
    reuse behavior as the live training streams - a shard target of 1-7B tokens can exceed a
    single pass over a smaller source like a cosmopedia subset)."""
    domain_filter = source_cfg.get("domain_filter")
    while True:
        ds = load_dataset(source_cfg["path"], name=source_cfg.get("name"), split="train",
                           streaming=True, trust_remote_code=True)
        yielded_any = False
        for row in ds:
            text = row.get("text")
            if not text:
                continue
            if domain_filter and classify_domain(text) != domain_filter:
                continue
            yielded_any = True
            yield text
        if not yielded_any:
            raise RuntimeError(f"source {source_cfg['path']} [{source_cfg.get('name')}] yielded "
                                f"zero usable rows on a full pass - check the domain_filter isn't "
                                f"rejecting everything, or the row schema doesn't actually have 'text'")


def build_shard(domain: str, target_tokens: int, tokenizer_dir: str, out_path: str, seed: int = 0) -> None:
    sources = SOURCE_LISTS[domain]
    weights = [w for _, w in sources]
    iterators = [_stream_source(cfg) for cfg, _ in sources]
    rng = random.Random(seed)

    bundle = TokenizerBundle(
        tokenizer_dir=tokenizer_dir,
        nlp_tokenizer_dir=f"{tokenizer_dir}/nlp" if domain == "nlp" else None,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_source_tokens = [0] * len(sources)
    total = 0
    next_log = LOG_EVERY_TOKENS

    with open(out, "wb") as f:
        while total < target_tokens:
            idx = rng.choices(range(len(sources)), weights=weights, k=1)[0]
            text = next(iterators[idx])
            ids, _ = bundle.encode_domain(domain, text, max_len=10**9)
            ids_np = ids.numpy()
            if ids_np.size and ids_np.max() >= 65536:
                raise ValueError(f"token id {ids_np.max()} exceeds uint16 range - vocab is bigger "
                                  f"than assumed, do not silently truncate, fix the dtype instead")
            ids_np = ids_np.astype(np.uint16)
            f.write(ids_np.tobytes())
            total += ids_np.size
            per_source_tokens[idx] += ids_np.size
            if total >= next_log:
                print(f"{total:,}/{target_tokens:,} tokens ({100 * total / target_tokens:.1f}%)", flush=True)
                next_log += LOG_EVERY_TOKENS

    meta = {
        "domain": domain, "total_tokens": total, "dtype": "uint16", "seed": seed,
        "tokenizer_dir": tokenizer_dir,
        "sources": [
            {"path": cfg["path"], "name": cfg.get("name"), "domain_filter": cfg.get("domain_filter"),
             "target_weight": w, "actual_tokens": per_source_tokens[i],
             "actual_fraction": per_source_tokens[i] / max(total, 1)}
            for i, (cfg, w) in enumerate(sources)
        ],
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: {total:,} tokens -> {out}", flush=True)
    for s in meta["sources"]:
        print(f"  {s['path']} [{s['name']}]: target={s['target_weight']:.0%}  "
              f"actual={s['actual_fraction']:.1%}  ({s['actual_tokens']:,} tokens)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=list(SOURCE_LISTS))
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build_shard(args.domain, args.target_tokens, args.tokenizer_dir, args.out, args.seed)
