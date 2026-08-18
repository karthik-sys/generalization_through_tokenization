"""Streamed, packed-sequence datasets for stage-2 GPU training. Never materializes the
full corpus - pulls documents from the HF stream, tags them, tokenizes, and concatenates
token ids across document boundaries (standard LM-pretraining "packing", far more
GPU-efficient than stage 1's one-doc-per-example batch-size-1 loop) into fixed-length
max_seq_len+1 chunks (input/target via a shift by one).

MoT needs one domain per batch (routing is per-table, not mixable within a batch), so
each domain gets its own packed stream, cycled round-robin during training. The unified
baseline/SOTA arms mix all four domains into one combined packed stream, matching what
they'd see in a real deployment (mixed-domain data, no routing).
"""

from __future__ import annotations

import itertools
from typing import Iterator

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset, get_worker_info

from src.model.stage2_config import CODE_LICENSE_ALLOWLIST, DOMAIN_TAG, STREAM_SOURCES


def _extract_code(r: dict) -> str | None:
    """codeparrot/github-code spans every language and license in one stream - both need
    filtering here, not left to the caller (see CODE_LICENSE_ALLOWLIST)."""
    if r.get("language") != "Python":
        return None
    if r.get("license") not in CODE_LICENSE_ALLOWLIST:
        return None
    return r.get("code")


TEXT_EXTRACTORS = {
    "code": _extract_code,
    "math": lambda r: r.get("text"),
    "science": lambda r: (f"{r['title']}\n{r['abstract']}" if r.get("abstract") else None),
    "nlp": lambda r: r.get("text"),
}
DOC_SEP = "\n<|endofdoc|>\n"


def _raw_doc_stream(domain: str) -> Iterator[str]:
    """Yields tagged documents forever, restarting the source when it runs dry.

    Without the restart a smaller source (the-stack-smol is only ~10k python files, i.e.
    roughly one pass at stage-2 step counts) exhausts mid-run and `next(loader)` raises
    StopIteration, killing training partway through. Restarting turns that into ordinary
    multi-epoch reuse. Sources large enough to never exhaust are unaffected.
    """
    cfg = STREAM_SOURCES[domain]
    tag = DOMAIN_TAG[domain]
    extractor = TEXT_EXTRACTORS[domain]
    pass_num = 0
    # Resolved lazily (inside this generator's body, not at construction time) so it reflects
    # the actual worker this generator is running in - DataLoader(num_workers>1) calls
    # __iter__ once per worker PROCESS, and without sharding every worker would independently
    # replay the exact same stream from the same starting point (duplicate documents, not more
    # throughput) rather than each covering a distinct slice of the source.
    worker_info = get_worker_info()
    while True:
        stream = load_dataset(
            cfg["path"], name=cfg.get("name"), data_dir=cfg.get("data_dir"),
            revision=cfg.get("revision"), data_files=cfg.get("data_files"),
            split="train", streaming=True,
        )
        if worker_info is not None and worker_info.num_workers > 1:
            stream = split_dataset_by_node(stream, rank=worker_info.id, world_size=worker_info.num_workers)
        if pass_num:
            stream = stream.shuffle(seed=pass_num, buffer_size=1000)
            print(f"[stream] {domain}: source exhausted, restarting (pass {pass_num + 1})", flush=True)
        emitted = 0
        for row in stream:
            text = extractor(row)
            if text:
                emitted += 1
                yield f"{tag}\n{text}{DOC_SEP}"
        if emitted == 0:
            raise RuntimeError(f"{domain}: source {cfg['path']} yielded no usable rows")
        pass_num += 1


class PackedDomainStream(IterableDataset):
    """Single-domain packed stream for MoT - every chunk yielded is (domain, ids[, types])."""

    def __init__(self, domain: str, encode_domain_fn, seq_len: int):
        self.domain = domain
        self.encode_domain_fn = encode_domain_fn  # bundle.encode_domain-style: (domain, text, max_len) -> (ids, types|None)
        self.seq_len = seq_len

    def __iter__(self):
        buf_ids: list[int] = []
        buf_types: list[int] = []
        has_types = self.domain == "nlp"
        for text in _raw_doc_stream(self.domain):
            ids, types = self.encode_domain_fn(self.domain, text, max_len=10**9)
            buf_ids.extend(ids.tolist())
            # Non-nlp tokenizers only ever emit surface tokens (type id 0), so filling
            # zeros keeps every yielded chunk the same shape. Yielding None here instead
            # breaks DataLoader's default_collate, which can't batch None.
            buf_types.extend(types.tolist() if has_types else [0] * len(ids))
            while len(buf_ids) >= self.seq_len + 1:
                yield (
                    self.domain,
                    torch.tensor(buf_ids[: self.seq_len + 1], dtype=torch.long),
                    torch.tensor(buf_types[: self.seq_len + 1], dtype=torch.long),
                )
                buf_ids = buf_ids[self.seq_len + 1 :]
                buf_types = buf_types[self.seq_len + 1 :]


class PackedMixedStream(IterableDataset):
    """All-domain interleaved packed stream, for the unified baseline / SOTA-tokenizer arms."""

    def __init__(self, encode_fn, seq_len: int):
        self.encode_fn = encode_fn  # (text, max_len) -> ids
        self.seq_len = seq_len

    def __iter__(self):
        streams = [_raw_doc_stream(d) for d in STREAM_SOURCES]
        buf: list[int] = []
        for text in _round_robin(streams):
            ids = self.encode_fn(text, max_len=10**9)
            buf.extend(ids.tolist())
            while len(buf) >= self.seq_len + 1:
                yield torch.tensor(buf[: self.seq_len + 1], dtype=torch.long)
                buf = buf[self.seq_len + 1 :]


def _round_robin(iterables):
    iterators = [iter(it) for it in iterables]
    while iterators:
        for it in list(iterators):
            try:
                yield next(it)
            except StopIteration:
                iterators.remove(it)
