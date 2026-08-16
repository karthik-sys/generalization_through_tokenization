"""Unified BPE baseline (spec §9, Baseline A): one embedding table, one shared backbone,
one unembedding head over a single vocab covering every domain (domain tags included as
literal text - spec's fairness rule in docs/dataset_methodology.md). Backbone config is
kept identical to MoTModel's for a matched-compute comparison; only the embedding/head
layer is single instead of disjoint-per-domain, which is precisely the variable under test.

No embedding->d_model projection here (unlike MoTModel): a single shared table has no
disjoint-table-into-shared-stream problem to solve, so adding one would just inflate this
model's parameter count for no architectural reason - emb_dim is set equal to d_model instead.
"""

import torch
import torch.nn as nn

from src.model.backbone import Backbone


class BaselineModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = Backbone(d_model, n_heads, ffn_dim, n_layers, max_seq_len)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = self.backbone(x)
        return self.head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
