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
import json
import random
import time
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset, get_worker_info

from src.model.stage2_config import CODE_LICENSE_ALLOWLIST, DOMAIN_TAG, STREAM_SOURCES

# routed19 (and any future long, unattended run): a bounded token budget means the data
# requirement is FINITE and computable up front, so there's no reason to keep hitting HF's
# live API for the entire multi-hour run - a one-time local download (scripts/build_domain_
# cache.py) removes the rate-limit dependency entirely (not just retries around it) for
# whatever it runs. Falls back to the live path automatically when no cache exists, so every
# other arm's behavior is completely unaffected.
DATA_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


def _cache_path(domain: str) -> Path:
    return DATA_CACHE_DIR / f"{domain}.jsonl"


def _shard_id_and_count() -> tuple[int, int]:
    """(global_shard_id, global_num_shards), folding BOTH axes a run can be parallel across:
    DDP rank (multiple GPUs, torchrun) and DataLoader worker (multiple CPU processes per GPU).
    Without this, a DDP run would only shard by worker id WITHIN each GPU process - every GPU's
    worker-0 would replay the identical slice of data as every other GPU's worker-0, silently
    wasting the extra GPUs' unique-data potential (still numerically correct, since duplicate
    data isn't wrong, just wasted). No-ops (0, 1) outside both DDP and multi-worker contexts."""
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    num_workers = worker_info.num_workers if worker_info is not None else 1
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    except ImportError:
        rank, world_size = 0, 1
    return rank * num_workers + worker_id, world_size * num_workers


def _cached_doc_stream(domain: str) -> Iterator[str] | None:
    """Returns a forever-looping iterator over a local JSONL cache if one exists for this
    domain, else None (caller falls back to the live HF stream).

    Reads lazily (one line at a time via the file iterator), NOT loaded fully into memory -
    an earlier version materialized the whole file into a Python list per worker process, which
    OOM-killed a real run: PackedRoutedStream needs all 4 domains simultaneously, so every
    worker process loads every domain, and with multiple independently-constructed loaders
    (routed19's phase1/phase2/val streams each build their own worker pool) that's
    N_loaders * NUM_WORKERS * (sum of all 4 domains' cache sizes) held in memory at once -
    confirmed live at ~39GB/domain-set this was approaching hundreds of GB. The OS page cache
    still makes repeated reads of the same file fast (shared across processes, unlike a
    process-private Python list), so this isn't a speed regression, just a memory-safety one.

    Diversity across workers/ranks: each pass skips a bounded random number of lines before
    reading (reseeded every pass, seeded by a rank+worker-unique id) rather than a full
    shuffle - approximate, not a perfect shuffle, but avoids ever materializing the file."""
    path = _cache_path(domain)
    if not path.exists():
        return None

    shard_id, _ = _shard_id_and_count()

    def _gen():
        rng = random.Random(shard_id)
        while True:
            with open(path) as f:
                skip = rng.randint(0, 20000)  # bounded - just diversity, not a real shuffle
                for _ in range(skip):
                    if next(f, None) is None:
                        break
                yielded_any = False
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = json.loads(line)["text"]
                    except json.JSONDecodeError:
                        continue
                    yielded_any = True
                    yield text
                if not yielded_any and skip == 0:
                    raise RuntimeError(f"cache file {path} appears empty")

    return _gen()


def _load_dataset_with_retry(*args, max_retries: int = 6, **kwargs):
    """Real, recurring risk once NUM_WORKERS>1 (see train_stage2_pod.py): every worker
    process independently calls load_dataset() at stream construction AND on every source
    restart (multi-epoch reuse, _raw_doc_stream's `while True` loop) over a run that can span
    millions of steps. HF's public API rate-limits by IP (confirmed live: 4 concurrent workers
    tripped a 429 on first launch - 'Retry after 47 seconds... 0/500 requests remaining').
    Exponential backoff turns a transient 429 into a pause, not a crashed worker that takes the
    whole training process down with it (next(loader) has no exception handling upstream)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "rate limit" not in msg.lower():
                raise
            last_err = e
            wait = min(90, 5 * (2 ** attempt))
            print(f"[stream] HF API rate limit hit, retrying in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(wait)
    raise last_err


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
    tag = DOMAIN_TAG[domain]
    cached = _cached_doc_stream(domain)
    if cached is not None:
        for text in cached:
            yield f"{tag}\n{text}{DOC_SEP}"
        return

    cfg = STREAM_SOURCES[domain]
    extractor = TEXT_EXTRACTORS[domain]
    pass_num = 0
    # Resolved lazily (inside this generator's body, not at construction time) so it reflects
    # the actual worker/rank this generator is running in - both DataLoader(num_workers>1)
    # AND DDP (multiple GPU processes under torchrun) call __iter__ once per process, and
    # without 2D sharding every one of them would independently replay the exact same stream
    # from the same starting point (duplicate documents, not more throughput/unique data).
    shard_id, num_shards = _shard_id_and_count()
    while True:
        stream = load_dataset(
            cfg["path"], name=cfg.get("name"), data_dir=cfg.get("data_dir"),
            revision=cfg.get("revision"), data_files=cfg.get("data_files"),
            split="train", streaming=True,
        )
        if num_shards > 1:
            stream = split_dataset_by_node(stream, rank=shard_id, world_size=num_shards)
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
