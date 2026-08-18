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


def _line_offsets(path: Path) -> "np.ndarray":
    """Byte offset of the start of every line in path, built once and cached to
    {path}.idx.npy - a single sequential scan (~1min for a 12GB file), reused by every
    process/rank/worker/run from then on rather than rebuilt per worker. Enables mmap-based
    random access without ever loading file CONTENT into memory (only this offset array,
    tens of MB, one per process - shared read-only across workers via the OS page cache same
    as the mmap itself)."""
    import numpy as np
    idx_path = path.with_suffix(path.suffix + ".idx.npy")
    if idx_path.exists():
        return np.load(idx_path)
    offs = []
    with open(path, "rb") as f:
        off = 0
        for line in f:
            offs.append(off)
            off += len(line)
    arr = np.array(offs, dtype=np.uint64)
    np.save(idx_path, arr)
    return arr


def _cached_doc_stream(domain: str) -> Iterator[str] | None:
    """Returns a forever-looping iterator over a local JSONL cache if one exists for this
    domain, else None (caller falls back to the live HF stream).

    mmap + a shuffled line-offset index, NOT sequential reads from a random starting point -
    an earlier version (skip a random number of lines, then read sequentially to EOF) avoided
    the memory blowup of a prior full-materialization bug, but introduced real correlation:
    every worker reads a long CONTIGUOUS run of the source file per pass, and if the cache has
    any structural ordering (source/length/alphabetical - real risk, never audited), consecutive
    batches end up correlated rather than i.i.d., which is exactly what SGD assumes isn't
    happening. mmap gives memory-safe random access (shared read-only OS pages across every
    worker process, not a private copy) to a genuinely shuffled permutation of line offsets,
    with each (rank, worker) pair getting an EXACT, non-overlapping partition - both problems
    fixed by the same change, not two separate patches."""
    import mmap as mmap_module

    import numpy as np

    path = _cache_path(domain)
    if not path.exists():
        return None
    offsets = _line_offsets(path)
    if len(offsets) == 0:
        return None

    shard_id, num_shards = _shard_id_and_count()

    def _gen():
        with open(path, "rb") as f:
            mm = mmap_module.mmap(f.fileno(), 0, access=mmap_module.ACCESS_READ)
            pass_num = 0
            try:
                while True:
                    # Same base seed on every rank/worker (a real requirement, not a bug -
                    # the permutation must be identical everywhere so partitioning by
                    # [shard_id::num_shards] below gives an EXACT, non-overlapping split of
                    # the same shuffled order; only the SLICE differs per shard, not the
                    # underlying permutation). Reseeded per pass_num so multi-epoch reuse
                    # doesn't replay the identical order every time.
                    perm = np.random.RandomState(pass_num).permutation(len(offsets))
                    my_lines = perm[shard_id::num_shards]
                    for i in my_lines:
                        start = int(offsets[i])
                        end = int(offsets[i + 1]) if i + 1 < len(offsets) else len(mm)
                        line = mm[start:end].decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)["text"]
                        except json.JSONDecodeError:
                            continue
                    pass_num += 1
            finally:
                mm.close()

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
