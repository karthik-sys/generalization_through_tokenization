"""Encode multi-domain documents into the tensors MoTRoutedModel needs.

A doc like:

    <domain:science>
    ...science text...
    <domain:code>
    ...code text...

becomes a single token sequence where each span is tokenized by its own domain's
tokenizer, and the <domain:X> markers survive as control tokens the model can predict.

Target construction is the part that makes switching learnable: at the last position of
a span, the target is not a normal token but `vocab_d + domain_index(next_domain)` in
that span's head - i.e. "emit a switch to the next domain". Everywhere else the target
is the ordinary next token id.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.data.build_examples import TokenizerBundle

DOMAIN_SPAN_RE = re.compile(r"<domain:(\w+)>\n")
DOC_SEP = "\n<|endofdoc|>\n"


def split_domain_spans(doc: str) -> list[tuple[str, str]]:
    """Returns [(domain, text_without_tag), ...] in order."""
    spans = []
    matches = list(DOMAIN_SPAN_RE.finditer(doc))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        spans.append((m.group(1), doc[m.end():end]))
    return spans


def encode_routed_doc(
    bundle: TokenizerBundle, domain_index: dict[str, int], doc: str, max_len: int,
) -> dict[str, torch.Tensor] | None:
    """Returns input/target tensors, or None if the doc is too short to train on."""
    token_ids: list[int] = []
    domain_ids: list[int] = []
    is_control: list[int] = []
    type_ids: list[int] = []

    for domain, text in split_domain_spans(doc):
        if domain not in domain_index:
            continue
        # control marker announcing this span's domain
        token_ids.append(domain_index[domain])
        domain_ids.append(domain_index[domain])
        is_control.append(1)
        type_ids.append(0)

        ids, types = bundle.encode_domain(domain, text, max_len)
        token_ids.extend(ids.tolist())
        domain_ids.extend([domain_index[domain]] * len(ids))
        is_control.extend([0] * len(ids))
        type_ids.extend(types.tolist() if types is not None else [0] * len(ids))

        if len(token_ids) > max_len:
            break

    if len(token_ids) < 2:
        return None
    token_ids = token_ids[: max_len + 1]
    domain_ids = domain_ids[: max_len + 1]
    is_control = is_control[: max_len + 1]
    type_ids = type_ids[: max_len + 1]

    n = len(token_ids) - 1
    targets = []
    for i in range(n):
        nxt = i + 1
        if is_control[nxt]:
            # predicting a switch: index into THIS position's head, past its real vocab
            from_domain = [d for d, idx in domain_index.items() if idx == domain_ids[i]][0]
            targets.append(bundle.domain_vocab_sizes[from_domain] + domain_ids[nxt])
        else:
            targets.append(token_ids[nxt])

    return {
        "token_ids": torch.tensor(token_ids[:n], dtype=torch.long),
        "domain_ids": torch.tensor(domain_ids[:n], dtype=torch.long),
        "is_control": torch.tensor(is_control[:n], dtype=torch.long),
        "type_ids": torch.tensor(type_ids[:n], dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def load_routed_docs(path: str) -> list[str]:
    return [d for d in Path(path).read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
