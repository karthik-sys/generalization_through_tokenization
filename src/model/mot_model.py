"""Mixture-of-Tokenizers model (spec §3): one disjoint embedding table + projection
per domain, one shared transformer backbone, one disjoint unembedding head per domain.

Domain routing itself (which table/head a token uses) is resolved at the data layer,
not inside this module - spec §2.1 says training-time domain labels come from data
provenance (known ground truth), so each forward call processes one domain's sequence
at a time. Spec §2.2's self-emitted-tag routing is an inference-time behavior (the model
learns to *produce* the tag it was trained on), not something this training-time forward
pass needs to simulate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.backbone import Backbone

NUM_TYPES = 4  # SURFACE, SYLLABLE, MORPHEME, BYTE - see nlp_hybrid_tokenizer.py


class MoTModel(nn.Module):
    def __init__(
        self,
        domain_vocab_sizes: dict[str, int],
        emb_dim: int,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        max_seq_len: int,
        type_conditioned_domains: tuple[str, ...] = ("nlp",),
    ):
        super().__init__()
        self.domains = list(domain_vocab_sizes)
        self.type_conditioned_domains = set(type_conditioned_domains)

        # disjoint per-domain embedding tables (spec §3) - a shared integer index
        # across two domains' tables is bookkeeping only, never the same parameter.
        self.embeddings = nn.ModuleDict(
            {d: nn.Embedding(v, emb_dim) for d, v in domain_vocab_sizes.items()}
        )
        # type embeddings (spec §6: "each token carries a type id ... so the model
        # can condition on granularity, not just identity") - only meaningful where
        # a domain's tokenizer actually varies granularity (the NLP hybrid).
        self.type_embeddings = nn.ModuleDict(
            {d: nn.Embedding(NUM_TYPES, emb_dim) for d in type_conditioned_domains if d in domain_vocab_sizes}
        )
        # projection into the shared residual stream (spec §3)
        self.projections = nn.ModuleDict({d: nn.Linear(emb_dim, d_model) for d in domain_vocab_sizes})

        self.backbone = Backbone(d_model, n_heads, ffn_dim, n_layers, max_seq_len)

        # disjoint per-domain unembedding heads (spec §3)
        self.heads = nn.ModuleDict({d: nn.Linear(d_model, v) for d, v in domain_vocab_sizes.items()})

    def forward(self, domain: str, token_ids: torch.Tensor, type_ids: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embeddings[domain](token_ids)
        if domain in self.type_embeddings:
            assert type_ids is not None, f"{domain} is type-conditioned but no type_ids given"
            emb = emb + self.type_embeddings[domain](type_ids)
        x = self.projections[domain](emb)
        x = self.backbone(x)
        return self.heads[domain](x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_params_by_component(self) -> dict[str, int]:
        return {
            "embeddings": sum(p.numel() for p in self.embeddings.parameters()),
            "type_embeddings": sum(p.numel() for p in self.type_embeddings.parameters()),
            "projections": sum(p.numel() for p in self.projections.parameters()),
            "backbone": self.backbone.num_params(),
            "heads": sum(p.numel() for p in self.heads.parameters()),
        }
