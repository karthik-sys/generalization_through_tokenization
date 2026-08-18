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


def build_cache(domain: str, target_bytes: int, source_override: dict | None = None) -> None:
    cfg = source_override or STREAM_SOURCES[domain]
    extractor = TEXT_EXTRACTORS[domain]
    path = _cache_path(domain)
    DATA_CACHE_DIR.mkdir(exist_ok=True)

    written_bytes = 0
    n_docs = 0
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
            line = json.dumps({"text": text}) + "\n"
            out.write(line)
            written_bytes += len(line.encode("utf-8"))
            n_docs += 1
            if n_docs % 5000 == 0:
                elapsed = time.time() - t0
                print(f"[{domain}] {n_docs:,} docs, {written_bytes/1e9:.2f}GB / "
                      f"{target_bytes/1e9:.2f}GB target, {elapsed:.0f}s elapsed", flush=True)
            if written_bytes >= target_bytes:
                break
    print(f"[{domain}] DONE: {n_docs:,} docs, {written_bytes/1e9:.2f}GB written to {path} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=DOMAINS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--target-gb", type=float, default=12.0)
    p.add_argument("--source", choices=["owt", "fineweb"], default="owt",
                    help="nlp only: owt (OpenWebText, matches routed8/19) or fineweb (default STREAM_SOURCES)")
    args = p.parse_args()

    if not args.domain and not args.all:
        raise SystemExit("pass --domain <name> or --all")

    domains = list(DOMAINS) if args.all else [args.domain]
    for d in domains:
        override = None
        if d == "nlp" and args.source == "owt":
            override = {"path": "Skylion007/openwebtext", "name": None}
        build_cache(d, int(args.target_gb * 1e9), source_override=override)
