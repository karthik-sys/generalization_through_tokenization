"""Synthetic multi-domain training documents (spec §2.2 / §10): every real document in
the corpus so far is single-domain, so there's nothing to teach the model mid-sequence
domain switching from. Build synthetic examples by concatenating snippets from 2-4
different domains into one document, with a <domain:X> tag at each switch point -
exactly the shape of a real cross-domain prompt/generation.

Switch positions are recorded in a metadata sidecar file so spec §10's boundary-adjacent
perplexity metric (loss on the 5-10 tokens right after a switch) can be computed later
without re-deriving where the switches are.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.prepare_corpus import DOMAIN_TAG, load_domain_texts

SNIPPET_WORDS = 250  # per-domain chunk length - short enough that switches happen within a 256-1024 token window
MIN_DOMAINS_PER_DOC = 2
MAX_DOMAINS_PER_DOC = 4
DOC_SEP = "\n<|endofdoc|>\n"


def _snippet(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def build_multidomain_corpus(
    n_docs: int, data_dir: str = "data", seed: int = 0,
) -> tuple[list[str], list[dict]]:
    rng = random.Random(seed)
    domains = list(DOMAIN_TAG)
    pools = {d: load_domain_texts(d, data_dir) for d in domains}
    for d, texts in pools.items():
        rng.shuffle(texts)

    docs, metadata = [], []
    cursors = {d: 0 for d in domains}
    for _ in range(n_docs):
        k = rng.randint(MIN_DOMAINS_PER_DOC, MAX_DOMAINS_PER_DOC)
        chosen = rng.sample(domains, k)
        parts, switches = [], []
        offset = 0
        for domain in chosen:
            pool = pools[domain]
            if cursors[domain] >= len(pool):
                cursors[domain] = 0
                rng.shuffle(pool)
            text = _snippet(pool[cursors[domain]], SNIPPET_WORDS)
            cursors[domain] += 1

            tag = DOMAIN_TAG[domain]
            span = f"{tag}\n{text}\n"
            switches.append({"domain": domain, "char_offset": offset, "tag_len": len(tag) + 1})
            parts.append(span)
            offset += len(span)

        doc = "".join(parts)
        docs.append(doc)
        metadata.append({"domains": chosen, "switches": switches, "doc_len": len(doc)})

    return docs, metadata


def write_multidomain_corpus(
    n_train: int = 2000, n_val: int = 200, out_dir: str = "data/corpus", data_dir: str = "data", seed: int = 0,
) -> None:
    train_docs, train_meta = build_multidomain_corpus(n_train, data_dir, seed)
    val_docs, val_meta = build_multidomain_corpus(n_val, data_dir, seed + 1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "multidomain_train.txt").write_text(DOC_SEP.join(train_docs), encoding="utf-8")
    (out / "multidomain_val.txt").write_text(DOC_SEP.join(val_docs), encoding="utf-8")
    (out / "multidomain_train_meta.json").write_text(json.dumps(train_meta))
    (out / "multidomain_val_meta.json").write_text(json.dumps(val_meta))

    domain_counts: dict[str, int] = {}
    switch_counts: dict[int, int] = {}
    for m in train_meta:
        for d in m["domains"]:
            domain_counts[d] = domain_counts.get(d, 0) + 1
        switch_counts[len(m["switches"])] = switch_counts.get(len(m["switches"]), 0) + 1

    print(f"train: {len(train_docs)} synthetic multi-domain docs  val: {len(val_docs)}")
    print(f"domain appearance counts (train): {domain_counts}")
    print(f"switches-per-doc distribution (train): {dict(sorted(switch_counts.items()))}")


if __name__ == "__main__":
    write_multidomain_corpus()
