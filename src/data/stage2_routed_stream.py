"""Packed, batched streaming data for MoTRoutedModel at stage-2 scale (spec §2.2).

The CPU version (src/data/build_multidomain_docs.py + encode_routed.py) builds a small
fixed local corpus and trains batch-size-1 - fine for an 8M-param CPU sanity check, far
too wasteful of GPU time at stage-2 scale. This streams real per-domain text live from
the same HF sources as the other 3 arms, composes it into synthetic multi-domain
documents on the fly (same 2-4-domains-per-doc, ~250-word-snippet shape as the CPU
version), and packs the result into fixed-length batched chunks - same packing strategy
as PackedDomainStream/PackedMixedStream.

Exploratory by design (per decision to keep this run's architecture as-is): each token
still commits to exactly one domain's table. A per-token multi-domain blend (discussed,
not built - see decision log) is real future work, not something to improvise here.
"""

from __future__ import annotations

import random
import re
from typing import Iterator

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset, get_worker_info

from src.data.stage2_stream_dataset import TEXT_EXTRACTORS, _cached_doc_stream, _shard_id_and_count
from src.model.stage2_config import DOMAIN_TAG, STREAM_SOURCES

SNIPPET_WORDS = 250
MIN_DOMAINS_PER_DOC = 2
MAX_DOMAINS_PER_DOC = 4


def _raw_body_stream(domain: str, pass_offset: int = 0) -> Iterator[str]:
    """Yields plain body text (no tag) for one domain, cycling forever.

    No domain-specific branch here anymore - code used to hardcode the-stack-smol with its
    own inline language filter, bypassing STREAM_SOURCES/TEXT_EXTRACTORS entirely. Now that
    TEXT_EXTRACTORS['code'] does its own language+license filtering (see stage2_stream_
    dataset.py), the generic path below works for every domain, code included.

    pass_offset: see _cached_doc_stream's docstring - lets two independently-constructed
    streams (e.g. routed19's phase1/phase2 loaders) read the SAME local cache from genuinely
    different starting points instead of silently colliding on the same pass_num=0 sequence.
    """
    cached = _cached_doc_stream(domain, pass_offset=pass_offset)
    if cached is not None:
        yield from cached
        return

    # Resolved lazily (inside this generator's body) so it reflects the worker/rank this
    # generator actually runs in - see the matching comment in stage2_stream_dataset.py's
    # _raw_doc_stream for why unsharded multi-worker/multi-GPU replay would just duplicate data.
    shard_id, num_shards = _shard_id_and_count()
    while True:
        cfg = STREAM_SOURCES[domain]
        stream = load_dataset(
            cfg["path"], name=cfg.get("name"), revision=cfg.get("revision"),
            data_files=cfg.get("data_files"), split="train", streaming=True,
            trust_remote_code=True,
        )
        if num_shards > 1:
            stream = split_dataset_by_node(stream, rank=shard_id, world_size=num_shards)
        extractor = TEXT_EXTRACTORS[domain]
        for row in stream:
            text = extractor(row)
            if text:
                yield text


def _snippet(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


# arm routed18 only: cheap heuristic proxy for "has LAMBADA-shaped structure" - not a
# linguistic parse, a fast word-recurrence check applied while streaming.
_STOPWORDS_FOR_MINING = frozenset("""
    the a an and or but if of to in on at by for with as is was were are be been being
    it its this that these those he she they we you i his her their our your my him
    them us me not no so up out about into over after before than then there here what
    which who when where how all any each other some such only own same too very just
    also one would could should will shall can may might must do does did have has had
    from
""".split())


def _is_copy_structured(text: str, min_word_len: int, min_gap: int) -> bool:
    """A document qualifies if some real content word (not a stopword, >= min_word_len
    chars) appears at least twice with >= min_gap words between the first and a later
    occurrence - i.e., a reader could plausibly retrieve the second occurrence from having
    seen the first, the same skill LAMBADA tests. Word-level, case/punctuation-insensitive."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < min_gap + 2:
        return False
    first_seen: dict[str, int] = {}
    for i, w in enumerate(words):
        if len(w) < min_word_len or w in _STOPWORDS_FOR_MINING:
            continue
        if w in first_seen:
            if i - first_seen[w] >= min_gap:
                return True
        else:
            first_seen[w] = i
    return False


def _filtered_body_stream(domain: str, min_word_len: int, min_gap: int, pass_offset: int = 0) -> Iterator[str]:
    """Like _raw_body_stream, but only yields documents passing _is_copy_structured -
    skips (not discards forever) documents that don't qualify, cycling the underlying
    source the same way _raw_body_stream does."""
    for text in _raw_body_stream(domain, pass_offset=pass_offset):
        if _is_copy_structured(text, min_word_len, min_gap):
            yield text


def synthetic_multidomain_doc_stream(
    seed: int = 0, min_domains: int = MIN_DOMAINS_PER_DOC, max_domains: int = MAX_DOMAINS_PER_DOC,
    snippet_words: int = SNIPPET_WORDS, force_domain: str | None = None,
    force_domain_snippet_words: int | None = None,
    force_domain_filter: tuple[int, int] | None = None,
    cache_pass_offset: int = 0,
) -> Iterator[str]:
    """Yields doc strings shaped like build_multidomain_docs.py's output, built live.

    min_domains/max_domains/snippet_words are parameterized (not just module constants) so
    arm "routed3" can train on denser cross-domain composition - always all 4 domains per
    doc, shorter snippets so more switches fit in a fixed 1024-token window - without a
    separate stream implementation. Defaults match the original routed/hybrid/routed2 shape.

    force_domain/force_domain_snippet_words (arms routed9/routed10/routed17): guarantees
    force_domain appears in EVERY doc (rather than the usual rng.sample() chance of being
    left out) and gives it its own, typically longer, snippet length - the mechanism for
    upweighting nlp's share of the token mixture well above its default ~1/4 without touching
    the other 3 domains at all. With force_domain_snippet_words=600 (vs the other domains'
    default 250) and min/max_domains left at 2/4, nlp lands at roughly 600/(600 + 2*250) ~=
    55% of tokens in expectation (E[additional domains] = E[k]-1 = 2 at k~Uniform{2,3,4});
    at 1200 words it's ~70% (routed17).

    force_domain_filter=(min_word_len, min_gap) (arm routed18 only): additionally restricts
    force_domain's source to documents passing _is_copy_structured - only documents with
    LAMBADA-shaped internal recurrence get used at all for that domain.

    cache_pass_offset (routed19's phase1/phase2 curriculum): passed straight through to the
    local-cache read path (see _cached_doc_stream's docstring) so two independently-constructed
    streams reading the SAME cache (e.g. phase1 vs phase2) start from genuinely different points
    in the shuffled order instead of silently colliding on the same pass_num=0 sequence.
    """
    rng = random.Random(seed)
    domains = list(STREAM_SOURCES)
    body_streams = {d: _raw_body_stream(d, pass_offset=cache_pass_offset) for d in domains}
    if force_domain and force_domain_filter is not None:
        min_word_len, min_gap = force_domain_filter
        body_streams[force_domain] = _filtered_body_stream(force_domain, min_word_len, min_gap, pass_offset=cache_pass_offset)
    other_domains = [d for d in domains if d != force_domain] if force_domain else domains
    while True:
        if force_domain:
            k = rng.randint(min_domains, min(max_domains, len(domains)))
            chosen = [force_domain] + rng.sample(other_domains, max(k - 1, 0))
        else:
            k = rng.randint(min_domains, min(max_domains, len(domains)))
            chosen = rng.sample(domains, k)
        parts = []
        for domain in chosen:
            words = force_domain_snippet_words if (domain == force_domain and force_domain_snippet_words) else snippet_words
            text = _snippet(next(body_streams[domain]), words)
            parts.append(f"{DOMAIN_TAG[domain]}\n{text}\n")
        yield "".join(parts)


class PackedRoutedStream(IterableDataset):
    """Yields (token_ids, domain_ids, is_control, type_ids, targets), each (seq_len,)."""

    def __init__(
        self, bundle, domain_index: dict[str, int], seq_len: int, seed: int = 0,
        min_domains: int = MIN_DOMAINS_PER_DOC, max_domains: int = MAX_DOMAINS_PER_DOC,
        snippet_words: int = SNIPPET_WORDS, force_domain: str | None = None,
        force_domain_snippet_words: int | None = None,
        force_domain_filter: tuple[int, int] | None = None,
        cache_pass_offset: int = 0,
    ):
        self.bundle = bundle
        self.domain_index = domain_index
        self.domains = list(domain_index)
        self.seq_len = seq_len
        self.seed = seed
        self.min_domains = min_domains
        self.max_domains = max_domains
        self.snippet_words = snippet_words
        self.force_domain = force_domain
        self.force_domain_snippet_words = force_domain_snippet_words
        self.force_domain_filter = force_domain_filter
        self.cache_pass_offset = cache_pass_offset

    def __iter__(self):
        buf_tok, buf_dom, buf_ctrl, buf_typ = [], [], [], []
        doc_stream = synthetic_multidomain_doc_stream(
            self.seed, self.min_domains, self.max_domains, self.snippet_words,
            self.force_domain, self.force_domain_snippet_words, self.force_domain_filter,
            self.cache_pass_offset)
        for doc in doc_stream:
            for domain, text in _split_spans(doc):
                if domain not in self.domain_index:
                    continue
                di = self.domain_index[domain]
                buf_tok.append(di)  # control token id is just the domain index into control_embedding
                buf_dom.append(di)
                buf_ctrl.append(1)
                buf_typ.append(0)

                ids, types = self.bundle.encode_domain(domain, text, max_len=10**9)
                buf_tok.extend(ids.tolist())
                buf_dom.extend([di] * len(ids))
                buf_ctrl.extend([0] * len(ids))
                buf_typ.extend(types.tolist() if types is not None else [0] * len(ids))

            window = self.seq_len + 1
            while len(buf_tok) >= window:
                c_tok, c_dom, c_ctrl, c_typ = buf_tok[:window], buf_dom[:window], buf_ctrl[:window], buf_typ[:window]
                targets = []
                for i in range(self.seq_len):
                    nxt = i + 1
                    if c_ctrl[nxt]:
                        from_domain = self.domains[c_dom[i]]
                        targets.append(self.bundle.domain_vocab_sizes[from_domain] + c_dom[nxt])
                    else:
                        targets.append(c_tok[nxt])
                yield (
                    torch.tensor(c_tok[: self.seq_len], dtype=torch.long),
                    torch.tensor(c_dom[: self.seq_len], dtype=torch.long),
                    torch.tensor(c_ctrl[: self.seq_len], dtype=torch.long),
                    torch.tensor(c_typ[: self.seq_len], dtype=torch.long),
                    torch.tensor(targets, dtype=torch.long),
                )
                buf_tok, buf_dom, buf_ctrl, buf_typ = buf_tok[window:], buf_dom[window:], buf_ctrl[window:], buf_typ[window:]


import re

_DOMAIN_SPAN_RE = re.compile(r"<domain:(\w+)>\n")


def _split_spans(doc: str) -> list[tuple[str, str]]:
    spans = []
    matches = list(_DOMAIN_SPAN_RE.finditer(doc))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        spans.append((m.group(1), doc[m.end():end]))
    return spans
