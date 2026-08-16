"""Pull a modest streamed sample per domain (not the full stage-2 corpus - that's what
train_stage2.py streams live during training) purely to train stage-2-sized tokenizers.
Tokenizer training needs a real corpus on disk; the training loop itself never needs the
full dataset materialized, just an iterator.

Run this once on whichever machine trains the tokenizers (doesn't need a GPU - SentencePiece
and Morfessor are CPU-only) before running train_stage2.py.
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import Dataset, load_dataset

from src.data.stage2_stream_dataset import TEXT_EXTRACTORS
from src.model.stage2_config import STREAM_SOURCES

N_SAMPLE = 50000  # docs per domain, enough for a stable 24k-32k BPE vocab without downloading full shards


def sample_domain(domain: str, out_dir: str = "data/stage2_tokenizer_sample") -> None:
    cfg = STREAM_SOURCES[domain]
    if cfg.get("gated"):
        print(f"{domain}: source is gated ({cfg['path']}) - accept its terms on huggingface.co and run "
              f"`huggingface-cli login` before this will work.")
    stream = load_dataset(cfg["path"], name=cfg.get("name"), data_dir=cfg.get("data_dir"), split="train", streaming=True)
    rows = []
    for row in itertools.islice(stream, N_SAMPLE):
        text = TEXT_EXTRACTORS[domain](row)
        if text:
            rows.append({"text": text})
    ds = Dataset.from_list(rows)
    ds.save_to_disk(f"{out_dir}/{domain}")
    print(f"{domain}: sampled {len(ds)} docs -> {out_dir}/{domain}")


if __name__ == "__main__":
    for domain in STREAM_SOURCES:
        sample_domain(domain)
