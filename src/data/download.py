"""Dataset download for the MoT vs. baseline comparison.

Sources are locked in docs/dataset_methodology.md - do not swap datasets here without
updating that doc, since the baseline/treatment comparison depends on both arms seeing
the same underlying corpus.

v2 (scaled up from the original stage-1 sanity-check slices): the first pass starved
math badly - its real-world side (HuggingFaceH4/MATH-500) was only 500 rows, so the
word-count balancer in prepare_corpus.py trimmed math's much larger textbook side down
to match, leaving math with almost no data. Fixed by swapping math's real-world source
to a real slice of open-web-math. Also added nlp's real-world side (fineweb), which v1
deferred to stage 2 - doing it now since we're past the pure "does it run" sanity check.
"""

import itertools

from datasets import Dataset, load_dataset

N_REAL = 15000
N_TEXTBOOK_CODE = 15000
N_TEXTBOOK_MATH = 4000  # only ~4470 available in libretexts_filtered's math subset
N_TEXTBOOK_SCIENCE = 8000  # ~11519 available across chem/phys/bio/geo subsets

# "Real-world" side of the 50/50 mix (docs/dataset_methodology.md).
STAGE1_SOURCES = {
    "code": {"path": "bigcode/the-stack-smol", "name": "python", "split": f"train[:{N_REAL}]"},
    "science": {"path": "gfissore/arxiv-abstracts-2021", "split": f"train[:{N_REAL}]"},
}

# Large datasets pulled via streaming + take(), not split slicing, so we don't pull
# full shards (open-web-math and fineweb are hundreds of GB to multi-TB overall).
STREAMED_SOURCES = {
    "math": {"path": "open-web-math/open-web-math", "n": N_REAL},
    "nlp": {"path": "HuggingFaceFW/fineweb", "name": "sample-10BT", "n": N_REAL},
}

# "Textbook-quality" side of the 50/50 mix. common-pile/libretexts_filtered rows carry
# a source URL in `id` (e.g. "https://math.libretexts.org/...") - used here as a subject
# filter. This is a first-pass heuristic on subdomain naming, not a verified taxonomy.
TEXTBOOK_SOURCES = {
    "math": {
        "path": "common-pile/libretexts_filtered",
        "split": "train",
        "url_contains": ["math.libretexts.org"],
        "n": N_TEXTBOOK_MATH,
    },
    "science": {
        "path": "common-pile/libretexts_filtered",
        "split": "train",
        "url_contains": ["chem.libretexts.org", "phys.libretexts.org", "bio.libretexts.org", "geo.libretexts.org"],
        "n": N_TEXTBOOK_SCIENCE,
    },
    "code": {"path": "nampdn-ai/tiny-code-textbooks", "split": f"train[:{N_TEXTBOOK_CODE}]", "gated": True},
    "nlp": {"path": "roneneldan/TinyStories", "split": f"train[:{N_REAL}]"},
}

DOMAIN_TAG = {
    "code": "<domain:code>",
    "math": "<domain:math>",
    "science": "<domain:science>",
    "nlp": "<domain:nlp>",
}


def download_domain(domain: str, out_dir: str = "data") -> None:
    if domain in STREAMED_SOURCES:
        cfg = STREAMED_SOURCES[domain]
        stream = load_dataset(cfg["path"], name=cfg.get("name"), split="train", streaming=True)
        rows = list(itertools.islice(stream, cfg["n"]))
        ds = Dataset.from_list(rows)
    else:
        cfg = STAGE1_SOURCES[domain]
        ds = load_dataset(cfg["path"], name=cfg.get("name"), split=cfg["split"])
    ds.save_to_disk(f"{out_dir}/{domain}/raw")


def download_textbook(domain: str, out_dir: str = "data") -> None:
    cfg = TEXTBOOK_SOURCES[domain]
    if "url_contains" not in cfg:
        ds = load_dataset(cfg["path"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["path"], split=cfg["split"])
        ds = ds.filter(lambda row: any(u in row["id"] for u in cfg["url_contains"]))
        ds = ds.select(range(min(cfg["n"], len(ds))))
    ds.save_to_disk(f"{out_dir}/{domain}/textbook")


if __name__ == "__main__":
    for domain in DOMAIN_TAG:
        download_domain(domain)
    for domain in TEXTBOOK_SOURCES:
        if not TEXTBOOK_SOURCES[domain].get("gated"):
            download_textbook(domain)
