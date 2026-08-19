"""One-time local cache builder for a bounded-token-budget run (routed19).

Downloads each domain's text ONCE to a local JSONL file under data_cache/, so the training
loop never has to hit HF's live streaming API again for the rest of a long, unattended run -
removes the rate-limit dependency entirely (not just retried around it, see stage2_stream_
dataset.py's _load_dataset_with_retry) for whatever arm reads from the cache.
_raw_doc_stream/_raw_body_stream both check for this cache automatically and prefer it when
present, so nothing else needs to change to start using it.

Usage (on the pod, before launching a long training run):
  python3 scripts/build_domain_cache.py --domain nlp --target-gb 12 --source owt
  python3 scripts/build_domain_cache.py --domain code --target-gb 12
  python3 scripts/build_domain_cache.py --domain math --target-gb 12
  python3 scripts/build_domain_cache.py --domain science --target-gb 12

Or all four at once (nlp defaults to OpenWebText, matching routed19/routed8):
  python3 scripts/build_domain_cache.py --all --target-gb 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.stage2_stream_dataset import (  # noqa: E402
    DATA_CACHE_DIR, TEXT_EXTRACTORS, _cache_path, _load_dataset_with_retry,
)
from src.model.stage2_config import STREAM_SOURCES  # noqa: E402

DOMAINS = ("code", "math", "science", "nlp")

# routed33: WebText (GPT-2's actual training corpus) was curated by a >3-karma Reddit-link
# filter, which selects for long-form, human-written prose and screens out short pages and
# nav/boilerplate junk almost as a side effect. OpenWebText (our nlp source) replicates
# WebText's URL list but not that quality signal at read time - every scraped page is kept
# regardless of length or how much of it is site chrome. This is a cheap proxy for the same
# selection pressure: length bounds (WebText docs are long-form, ~500-5000 words is a rough
# proxy for "article", not "nav menu" or "entire book dump") plus a boilerplate-phrase count
# (repeated site-chrome phrases are a strong tell for scraped-not-authored content).
BOILERPLATE_PHRASES = (
    "click here", "subscribe now", "follow us", "all rights reserved", "sign up",
    "log in", "newsletter", "terms of service", "privacy policy", "cookie",
)


def aggressive_nlp_filter(text: str, min_words: int = 500, max_words: int = 5000) -> bool:
    words = text.split()
    n = len(words)
    if n < min_words or n > max_words:
        return False
    lowered = text.lower()
    return sum(1 for p in BOILERPLATE_PHRASES if p in lowered) < 2


def build_generalist_cache(target_bytes: int) -> None:
    """routed33's 5th domain: no external source of its own - pooled round-robin from the
    OTHER four domains' already-built local caches (must run this AFTER code/math/science/nlp
    are built), same pooling the generalist tokenizer itself was trained on. Keeps the
    generalist's training-time distribution consistent with what its vocab was fit to."""
    import random

    source_paths = {d: _cache_path(d) for d in ("code", "math", "science", "nlp")}
    missing = [d for d, p in source_paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"generalist cache needs {missing} built first - run those domains before this one")

    path = _cache_path("generalist")
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    readers = {d: open(p) for d, p in source_paths.items()}
    written_bytes, n_docs = 0, 0
    t0 = time.time()
    rng = random.Random(0)
    domains = list(readers)
    with open(path, "w") as out:
        while written_bytes < target_bytes and readers:
            d = rng.choice(list(readers))
            line = readers[d].readline()
            if not line:
                readers[d].close()
                del readers[d]
                continue
            out.write(line)
            written_bytes += len(line.encode("utf-8"))
            n_docs += 1
            if n_docs % 5000 == 0:
                print(f"[generalist] {n_docs:,} docs, {written_bytes/1e9:.2f}GB / "
                      f"{target_bytes/1e9:.2f}GB target, {time.time()-t0:.0f}s elapsed", flush=True)
    for r in readers.values():
        r.close()
    print(f"[generalist] DONE: {n_docs:,} docs, {written_bytes/1e9:.2f}GB written to {path} "
          f"in {time.time()-t0:.0f}s (pooled from {domains})", flush=True)


def build_cache(domain: str, target_bytes: int, source_override: dict | None = None,
                 nlp_filter: bool = False) -> None:
    cfg = source_override or STREAM_SOURCES[domain]
    extractor = TEXT_EXTRACTORS[domain]
    path = _cache_path(domain)
    DATA_CACHE_DIR.mkdir(exist_ok=True)

    written_bytes = 0
    n_docs = 0
    n_filtered = 0
    t0 = time.time()
    with open(path, "w") as out:
        stream = _load_dataset_with_retry(
            cfg["path"], name=cfg.get("name"), data_dir=cfg.get("data_dir"),
            revision=cfg.get("revision"), data_files=cfg.get("data_files"),
            split="train", streaming=True,
            **({"trust_remote_code": True} if domain != "nlp" else {}),
        )
        for row in stream:
            text = extractor(row)
            if not text:
                continue
            if domain == "nlp" and nlp_filter and not aggressive_nlp_filter(text):
                n_filtered += 1
                continue
            line = json.dumps({"text": text}) + "\n"
            out.write(line)
            written_bytes += len(line.encode("utf-8"))
            n_docs += 1
            if n_docs % 5000 == 0:
                elapsed = time.time() - t0
                print(f"[{domain}] {n_docs:,} docs, {written_bytes/1e9:.2f}GB / "
                      f"{target_bytes/1e9:.2f}GB target, {elapsed:.0f}s elapsed"
                      + (f", {n_filtered:,} filtered" if nlp_filter else ""), flush=True)
            if written_bytes >= target_bytes:
                break
    print(f"[{domain}] DONE: {n_docs:,} docs, {written_bytes/1e9:.2f}GB written to {path} "
          f"in {time.time()-t0:.0f}s" + (f" ({n_filtered:,} filtered out)" if nlp_filter else ""),
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=DOMAINS + ("generalist",))
    p.add_argument("--all", action="store_true")
    p.add_argument("--target-gb", type=float, default=12.0)
    p.add_argument("--source", choices=["owt", "fineweb"], default="owt",
                    help="nlp only: owt (OpenWebText, matches routed8/19) or fineweb (default STREAM_SOURCES)")
    p.add_argument("--nlp-filter", action="store_true",
                    help="nlp only: apply aggressive_nlp_filter (length bounds + boilerplate-phrase "
                         "screen) to mimic WebText's long-form, human-written selection pressure - "
                         "see routed33's docstring in build_cache")
    args = p.parse_args()

    if not args.domain and not args.all:
        raise SystemExit("pass --domain <name> or --all")

    domains = list(DOMAINS) if args.all else [args.domain]
    for d in domains:
        if d == "generalist":
            build_generalist_cache(int(args.target_gb * 1e9))
            continue
        override = None
        if d == "nlp" and args.source == "owt":
            override = {"path": "Skylion007/openwebtext", "name": None}
        build_cache(d, int(args.target_gb * 1e9), source_override=override, nlp_filter=args.nlp_filter)
