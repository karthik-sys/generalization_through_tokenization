"""Streaming data for the "routed on GPT-2's actual data" arms (routed-webtext,
routed-webtext-n): sources a single homogeneous corpus (OpenWebTextCorpus, the standard
open reconstruction of GPT-2's unreleased WebText - see docs/handoff_routed_webtext.md for
why) and classifies documents into domains on the fly, rather than relying on
STREAM_SOURCES' pre-curated per-domain sources the way every other arm does.

Two domain layouts, both built on the same classified-stream machinery:
  - 4-domain (routed-webtext): code/math/science/nlp, same domain set and tokenizers as
    every existing arm - isolates "same architecture, same domain COUNT, only the corpus
    changed" as cleanly as possible.
  - N-domain (routed-webtext-n): adds a further split of the nlp bucket (which absorbs
    the overwhelming majority of a general corpus like this one) into 4 register-based
    sub-buckets sharing the SAME nlp tokenizer but each getting its own disjoint
    embedding/head table - tests "does finer-grained routing help further" without
    needing new tokenizer training.

Known cost, stated plainly: each domain opens its OWN independent stream over the shared
corpus and discards non-matching documents (e.g. the "code" stream reads through mostly
non-code text before finding a match). Data loading isn't the bottleneck here (GPU-bound,
batch size 4 x grad-accum 16), so the redundant reads are an accepted, cheap trade for
reusing the existing PackedRoutedStream-style architecture instead of building shared-
buffer machinery this project doesn't otherwise need.
"""

from __future__ import annotations

import random
from typing import Callable, Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

from src.data.domain_classifier import NLP_SUBDOMAINS, classify_domain, classify_nlp_subdomain
from src.model.stage2_config import DOMAIN_TAG

OPENWEBTEXT_SOURCE = {"path": "Skylion007/openwebtext"}

# webtext-only tags for the nlp sub-buckets (the shared DOMAIN_TAG dict only has the base 4)
WEBTEXT_DOMAIN_TAG = dict(DOMAIN_TAG)
for _sub in NLP_SUBDOMAINS:
    WEBTEXT_DOMAIN_TAG[_sub] = f"<domain:{_sub}>"

SNIPPET_WORDS = 250
MIN_DOMAINS_PER_DOC = 2


def _classify_4domain(text: str) -> str:
    return classify_domain(text)


def _classify_n_domain(text: str) -> str:
    top = classify_domain(text)
    if top != "nlp":
        return top
    return classify_nlp_subdomain(text)


def _raw_webtext_body_stream(domain: str, classify_fn: Callable[[str], str]) -> Iterator[str]:
    """Yields plain body text for one domain label, filtered from the shared OpenWebText
    stream by classify_fn. Restarts (shuffled) on exhaustion, matching _raw_doc_stream's
    convention in stage2_stream_dataset.py."""
    pass_num = 0
    while True:
        stream = load_dataset(OPENWEBTEXT_SOURCE["path"], split="train", streaming=True)
        if pass_num:
            stream = stream.shuffle(seed=pass_num, buffer_size=1000)
        emitted = 0
        for row in stream:
            text = row.get("text")
            if not text:
                continue
            if classify_fn(text) == domain:
                emitted += 1
                yield text
        if emitted == 0:
            raise RuntimeError(f"webtext domain {domain}: zero matches in a full pass - "
                                f"classifier or domain name is likely wrong")
        pass_num += 1


def webtext_synthetic_multidomain_doc_stream(
    domains: list[str], classify_fn: Callable[[str], str],
    seed: int = 0, min_domains: int = MIN_DOMAINS_PER_DOC, snippet_words: int = SNIPPET_WORDS,
) -> Iterator[str]:
    """Same shape as stage2_routed_stream.synthetic_multidomain_doc_stream, sourced from
    the classified webtext streams instead of STREAM_SOURCES."""
    rng = random.Random(seed)
    max_domains = min(4, len(domains))  # cap composed docs at 4 domains/doc regardless of N
    body_streams = {d: _raw_webtext_body_stream(d, classify_fn) for d in domains}
    while True:
        k = rng.randint(min_domains, max_domains)
        chosen = rng.sample(domains, k)
        parts = []
        for domain in chosen:
            text = " ".join(next(body_streams[domain]).split()[:snippet_words])
            parts.append(f"{WEBTEXT_DOMAIN_TAG[domain]}\n{text}\n")
        yield "".join(parts)


def _split_spans(doc: str, domains: list[str]) -> list[tuple[str, str]]:
    import re
    tag_re = re.compile(r"<domain:(\w+)>\n")
    spans = []
    matches = list(tag_re.finditer(doc))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        spans.append((m.group(1), doc[m.end():end]))
    return spans


class WebTextPackedRoutedStream(IterableDataset):
    """Same interface/output contract as stage2_routed_stream.PackedRoutedStream:
    yields (token_ids, domain_ids, is_control, type_ids, targets), each (seq_len,).

    tokenize_fn(domain_label, text, max_len) -> (ids, types|None) lets the caller route
    nlp sub-bucket labels to the real "nlp" tokenizer (encode_domain only special-cases
    the literal string "nlp") while still using the sub-bucket label for domain_ids/table
    selection at the model level.
    """

    def __init__(
        self, domains: list[str], domain_index: dict[str, int], domain_vocab_sizes: dict[str, int],
        tokenize_fn: Callable[[str, str, int], tuple[torch.Tensor, torch.Tensor | None]],
        classify_fn: Callable[[str], str], seq_len: int, seed: int = 0,
    ):
        self.domains = domains
        self.domain_index = domain_index
        self.domain_vocab_sizes = domain_vocab_sizes
        self.tokenize_fn = tokenize_fn
        self.classify_fn = classify_fn
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        buf_tok, buf_dom, buf_ctrl, buf_typ = [], [], [], []
        doc_stream = webtext_synthetic_multidomain_doc_stream(self.domains, self.classify_fn, self.seed)
        for doc in doc_stream:
            for domain, text in _split_spans(doc, self.domains):
                if domain not in self.domain_index:
                    continue
                di = self.domain_index[domain]
                buf_tok.append(di)
                buf_dom.append(di)
                buf_ctrl.append(1)
                buf_typ.append(0)

                ids, types = self.tokenize_fn(domain, text, 10**9)
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
                        targets.append(self.domain_vocab_sizes[from_domain] + c_dom[nxt])
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
