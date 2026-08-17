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
from torch.utils.data import IterableDataset

from src.data.stage2_stream_dataset import TEXT_EXTRACTORS
from src.model.stage2_config import DOMAIN_TAG, STREAM_SOURCES

SNIPPET_WORDS = 250
MIN_DOMAINS_PER_DOC = 2
MAX_DOMAINS_PER_DOC = 4


def _raw_body_stream(domain: str) -> Iterator[str]:
    """Yields plain body text (no tag) for one domain, cycling forever.

    No domain-specific branch here anymore - code used to hardcode the-stack-smol with its
    own inline language filter, bypassing STREAM_SOURCES/TEXT_EXTRACTORS entirely. Now that
    TEXT_EXTRACTORS['code'] does its own language+license filtering (see stage2_stream_
    dataset.py), the generic path below works for every domain, code included.
    """
    while True:
        cfg = STREAM_SOURCES[domain]
        stream = load_dataset(
            cfg["path"], name=cfg.get("name"), revision=cfg.get("revision"),
            data_files=cfg.get("data_files"), split="train", streaming=True,
        )
        extractor = TEXT_EXTRACTORS[domain]
        for row in stream:
            text = extractor(row)
            if text:
                yield text


def _snippet(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


def synthetic_multidomain_doc_stream(
    seed: int = 0, min_domains: int = MIN_DOMAINS_PER_DOC, max_domains: int = MAX_DOMAINS_PER_DOC,
    snippet_words: int = SNIPPET_WORDS, force_domain: str | None = None,
    force_domain_snippet_words: int | None = None,
) -> Iterator[str]:
    """Yields doc strings shaped like build_multidomain_docs.py's output, built live.

    min_domains/max_domains/snippet_words are parameterized (not just module constants) so
    arm "routed3" can train on denser cross-domain composition - always all 4 domains per
    doc, shorter snippets so more switches fit in a fixed 1024-token window - without a
    separate stream implementation. Defaults match the original routed/hybrid/routed2 shape.

    force_domain/force_domain_snippet_words (arms routed9/routed10): guarantees force_domain
    appears in EVERY doc (rather than the usual rng.sample() chance of being left out) and
    gives it its own, typically longer, snippet length - the mechanism for upweighting nlp's
    share of the token mixture well above its default ~1/4 without touching the other 3
    domains at all. With force_domain_snippet_words=600 (vs the other domains' default 250)
    and min/max_domains left at 2/4, nlp lands at roughly 600/(600 + 2*250) ~= 55% of tokens
    in expectation (E[additional domains] = E[k]-1 = 2 at k~Uniform{2,3,4}).
    """
    rng = random.Random(seed)
    domains = list(STREAM_SOURCES)
    body_streams = {d: _raw_body_stream(d) for d in domains}
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

    def __iter__(self):
        buf_tok, buf_dom, buf_ctrl, buf_typ = [], [], [], []
        doc_stream = synthetic_multidomain_doc_stream(
            self.seed, self.min_domains, self.max_domains, self.snippet_words,
            self.force_domain, self.force_domain_snippet_words)
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
