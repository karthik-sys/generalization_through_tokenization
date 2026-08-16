"""Merge each domain's real-world + textbook sources at ~50/50 by word count (not row
count - row lengths differ wildly between e.g. short math problems and long textbook
pages), prefix every document with its literal domain tag, and write one shuffled,
tagged corpus used identically by both the BPE baseline and the MoT treatment (see
"Fairness rule" in docs/dataset_methodology.md - tags must be visible to both arms).

NLP has no stage-1 real-world counterpart by design (TinyStories is the textbook side;
the web/fineweb real-world side is deferred to stage 2 per dataset_methodology.md), so
it's the one domain that isn't actually balanced 50/50 here - that's expected, not a bug.
"""

import random
from pathlib import Path

from datasets import load_from_disk

DOMAIN_TAG = {
    "code": "<domain:code>",
    "math": "<domain:math>",
    "science": "<domain:science>",
    "nlp": "<domain:nlp>",
}

TEXT_EXTRACTORS = {
    "code": {
        "raw": lambda r: r["content"],
        "textbook": lambda r: r["content"] + "\n" + r["response"],
    },
    "math": {
        "raw": lambda r: r["text"],  # open-web-math (v2; v1 used hendrycks/competition_math's problem+solution)
        "textbook": lambda r: r["text"],
    },
    "science": {
        "raw": lambda r: r["title"] + "\n" + r["abstract"],
        "textbook": lambda r: r["text"],
    },
    "nlp": {
        "raw": lambda r: r["text"],  # fineweb (v2 - added, v1 had no nlp real-world side)
        "textbook": lambda r: r["text"],  # TinyStories
    },
}


def _load_side(domain: str, side: str, data_dir: str) -> list[str]:
    path = Path(data_dir) / domain / side
    if not path.exists():
        return []
    ds = load_from_disk(str(path))
    extractor = TEXT_EXTRACTORS[domain][side]
    return [extractor(r) for r in ds]


def _balance_by_word_count(a: list[str], b: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Downsample the larger list (by total word count) to roughly match the smaller."""
    rng = random.Random(seed)
    a, b = a[:], b[:]
    rng.shuffle(a)
    rng.shuffle(b)
    words_a = sum(len(t.split()) for t in a)
    words_b = sum(len(t.split()) for t in b)
    if words_a == 0 or words_b == 0:
        return a, b

    def trim(docs: list[str], target_words: int) -> list[str]:
        out, total = [], 0
        for t in docs:
            if total >= target_words:
                break
            out.append(t)
            total += len(t.split())
        return out

    if words_a > words_b:
        a = trim(a, words_b)
    else:
        b = trim(b, words_a)
    return a, b


def load_domain_texts(domain: str, data_dir: str = "data", seed: int = 0) -> list[str]:
    raw = _load_side(domain, "raw", data_dir)
    textbook = _load_side(domain, "textbook", data_dir)
    if raw and textbook:
        raw, textbook = _balance_by_word_count(raw, textbook, seed)
    return raw + textbook


def build_corpus(
    data_dir: str = "data", seed: int = 0, val_fraction: float = 0.05
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    tagged_docs = []
    for domain, tag in DOMAIN_TAG.items():
        texts = load_domain_texts(domain, data_dir, seed)
        tagged_docs.extend(f"{tag}\n{t}" for t in texts)
    rng.shuffle(tagged_docs)
    n_val = max(1, int(len(tagged_docs) * val_fraction))
    return tagged_docs[n_val:], tagged_docs[:n_val]


def write_corpus(out_dir: str = "data/corpus", data_dir: str = "data", seed: int = 0) -> None:
    train_docs, val_docs = build_corpus(data_dir=data_dir, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sep = "\n<|endofdoc|>\n"
    (out / "train.txt").write_text(sep.join(train_docs), encoding="utf-8")
    (out / "val.txt").write_text(sep.join(val_docs), encoding="utf-8")
    print(f"train docs: {len(train_docs)}  val docs: {len(val_docs)}")
    for domain in DOMAIN_TAG:
        n = len(load_domain_texts(domain, data_dir, seed))
        print(f"  {domain}: {n} docs (post-balancing)")


if __name__ == "__main__":
    write_corpus()
