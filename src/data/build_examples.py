"""Turn tagged documents (data/corpus/{train,val}.txt) into per-domain token id tensors.

Stage-1 simplification, noted explicitly: the domain tag is left as the literal text
prefix on each document (already done by prepare_corpus.py) and is encoded by whichever
tokenizer processes that document - the domain's own tokenizer for MoT, the shared
vocab for the baseline - rather than routed through a separate control vocabulary that
switches tables mid-sequence. Spec §2.2's self-emitted-tag mid-sequence routing needs
that control-vocab mechanism; it's not required for stage 1, where every document is
single-domain and the goal is just "does the pipeline run, does loss decrease" (spec §8).
Build it before relying on stage 2's boundary-adjacent perplexity metric (spec §10).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from src.tokenizers.bpe_tokenizer import BPETokenizer
from src.tokenizers.nlp_hybrid_tokenizer import NLPHybridTokenizer
from src.tokenizers.sota_tokenizer import SotaTokenizer

DOMAIN_TAG_RE = re.compile(r"^<domain:(\w+)>\n")


def load_tagged_docs(path: str) -> list[tuple[str, str]]:
    """Returns list of (domain, full_text_including_tag)."""
    text = Path(path).read_text(encoding="utf-8")
    docs = text.split("\n<|endofdoc|>\n")
    out = []
    for doc in docs:
        m = DOMAIN_TAG_RE.match(doc)
        if m:
            out.append((m.group(1), doc))
    return out


class TokenizerBundle:
    def __init__(self, tokenizer_dir: str = "tokenizers", nlp_tokenizer_dir: str | None = None,
                 generalist_tokenizer_dir: str | None = None):
        """nlp_tokenizer_dir overrides where the nlp tokenizer loads from, independent of
        tokenizer_dir (which still governs code/math/science/baseline/sota). Exists for
        arm="routed7": its nlp domain trains on a different corpus (OpenWebText, not
        FineWeb) and needs its OWN nlp tokenizer fit to that corpus - code/math/science
        keep using the shared, unmodified tokenizers under tokenizer_dir. Without this,
        there'd be no way to mix "most domains from the shared bundle, one domain from a
        separately-trained tokenizer" without either retraining everything or physically
        duplicating the untouched tokenizer files.

        generalist_tokenizer_dir: routed33 only. Adds a 5th plain-BPE domain ("generalist")
        trained on a pooled sample across all four existing domains, rather than special-
        casing it - encode_domain's `else` branch (self.bpe[domain].encode(...)) already
        handles any domain that isn't "nlp" identically, so no dispatch changes needed
        beyond registering it here."""
        self.bpe = {
            "code": BPETokenizer(f"{tokenizer_dir}/code/model.model"),
            "math": BPETokenizer(f"{tokenizer_dir}/math/model.model"),
            "science": BPETokenizer(f"{tokenizer_dir}/science/model.model"),
        }
        if generalist_tokenizer_dir:
            self.bpe["generalist"] = BPETokenizer(f"{generalist_tokenizer_dir}/model.model")
        self.nlp = NLPHybridTokenizer(nlp_tokenizer_dir or f"{tokenizer_dir}/nlp")
        self.baseline = BPETokenizer(f"{tokenizer_dir}/baseline_bpe/model.model")
        self.sota = SotaTokenizer("cl100k_base")

        sizes = self.nlp.vocab_sizes
        self.nlp_offsets = {
            "surface": 0,
            "syllable": sizes["surface"],
            "morpheme": sizes["surface"] + sizes["syllable"],
            "byte": sizes["surface"] + sizes["syllable"] + sizes["morpheme"],
        }
        self.nlp_vocab_size = sum(sizes.values())

        self.domain_vocab_sizes = {
            "code": self.bpe["code"].vocab_size,
            "math": self.bpe["math"].vocab_size,
            "science": self.bpe["science"].vocab_size,
            "nlp": self.nlp_vocab_size,
        }
        if generalist_tokenizer_dir:
            self.domain_vocab_sizes["generalist"] = self.bpe["generalist"].vocab_size
        self.baseline_vocab_size = self.baseline.vocab_size
        self.sota_vocab_size = self.sota.vocab_size

    def encode_domain(self, domain: str, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if domain == "nlp":
            type_names = ["surface", "syllable", "morpheme", "byte"]
            pairs = self.nlp.encode(text)[:max_len]
            ids = torch.tensor([tid + self.nlp_offsets[type_names[t]] for tid, t in pairs], dtype=torch.long)
            types = torch.tensor([t for _, t in pairs], dtype=torch.long)
            return ids, types
        ids = self.bpe[domain].encode(text)[:max_len]
        return torch.tensor(ids, dtype=torch.long), None

    def encode_baseline(self, text: str, max_len: int) -> torch.Tensor:
        ids = self.baseline.encode(text)[:max_len]
        return torch.tensor(ids, dtype=torch.long)

    def encode_sota(self, text: str, max_len: int) -> torch.Tensor:
        ids = self.sota.encode(text)[:max_len]
        return torch.tensor(ids, dtype=torch.long)
