"""Bet 2 ("deep experts"): per-domain LoRA adapters on the top N transformer layers.

Right now only the input embeddings, projections, and output heads are domain-specific -
the entire backbone (all n_layers blocks) is fully shared, per spec §3's "this is where
cross-domain generalization happens" design. That's a real hypothesis worth testing
directly: is the shared spine's representational capacity, contended across 4 domains,
actually limiting what the nlp head can do? Bet 2 tests it architecturally (give the top
layers their own per-domain transformation) rather than via data (Bet-2's sibling, the
100%-nlp "diet probe", tests the same question by removing the contention instead of
routing around it).

Implemented as LoRA (low-rank adapters), not full per-domain layer copies, specifically so
this can be launched as CONTINUED training from an already-trained checkpoint: each
adapter's down-projection is zero-initialized, so on step 1 every domain's forward pass is
IDENTICAL to the shared-backbone parent it's warm-started from (a LoRA no-op at init is the
standard trick - see Hu et al. 2021), and the model only diverges as training moves the
adapters away from zero. A full per-domain layer copy would need a from-scratch run to be a
fair comparison; this doesn't.

Only the FFN half of each expert layer is domain-adapted (not attention) - attention mixes
across positions, so a clean per-token domain routing there is a bigger change for
questionable extra payoff; FFN is already fully position-wise, so gathering-by-domain and
adding a delta is the same pattern embed_sequence() already uses per-domain.

Kill criterion: if LAMBADA at matched steps tracks routed8's own historical curve (see
results/metrics.json) within noise, top-layer contention wasn't the limiter and the shared
spine is already doing general-purpose work well - a legitimate negative result, not a bug.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.backbone import CausalSelfAttention
from src.model.mot_routed_model import MoTRoutedModel

DEFAULT_N_EXPERT_LAYERS = 2
DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 16.0


class DomainLoRAFFN(nn.Module):
    """Wraps a shared FFN with one low-rank adapter per domain, applied per-token by
    domain_id - same gather/apply/scatter shape as MoTRoutedModel.embed_sequence's
    per-domain embedding lookup. B is zero-initialized (standard LoRA convention) so every
    domain's delta is exactly zero at init - the wrapped block reproduces its parent
    checkpoint's behavior on step 1."""

    def __init__(self, ffn: nn.Sequential, d_model: int, domains: list[str],
                 rank: int = DEFAULT_LORA_RANK, alpha: float = DEFAULT_LORA_ALPHA):
        super().__init__()
        self.ffn = ffn
        self.domains = domains
        self.scale = alpha / rank
        self.lora_a = nn.ModuleDict({d: nn.Linear(d_model, rank, bias=False) for d in domains})
        self.lora_b = nn.ModuleDict({d: nn.Linear(rank, d_model, bias=False) for d in domains})
        for d in domains:
            nn.init.zeros_(self.lora_b[d].weight)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        out = self.ffn(x)
        for i, d in enumerate(self.domains):
            mask = domain_ids == i
            if not mask.any():
                continue
            delta = self.lora_b[d](self.lora_a[d](x[mask])) * self.scale
            out[mask] = out[mask] + delta.to(out.dtype)
        return out


class DomainLoRATransformerBlock(nn.Module):
    """Same shape as backbone.TransformerBlock (ln1/attn/ln2/ffn), except the FFN is
    domain-adapted. Named to keep attn/ln1/ln2 key-compatible with a plain TransformerBlock
    for warm-starting (see _warm_start_deep_expert in train_stage2_pod.py) - only the ffn
    submodule's internal path changes (ffn.ffn.* instead of ffn.*)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, domains: list[str],
                 rank: int = DEFAULT_LORA_RANK, alpha: float = DEFAULT_LORA_ALPHA):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        base_ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))
        self.ffn = DomainLoRAFFN(base_ffn, d_model, domains, rank, alpha)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x), domain_ids)
        return x


class DeepExpertBackbone(nn.Module):
    """First (n_layers - n_expert_layers) blocks are the plain, fully-shared
    backbone.TransformerBlock (unchanged). The last n_expert_layers are
    DomainLoRATransformerBlocks. forward() needs domain_ids, unlike the plain Backbone."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, n_layers: int, max_seq_len: int,
                 domains: list[str], n_expert_layers: int = DEFAULT_N_EXPERT_LAYERS,
                 lora_rank: int = DEFAULT_LORA_RANK, lora_alpha: float = DEFAULT_LORA_ALPHA):
        super().__init__()
        from src.model.backbone import TransformerBlock
        n_shared = n_layers - n_expert_layers
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.shared_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_shared)])
        self.expert_blocks = nn.ModuleList(
            [DomainLoRATransformerBlock(d_model, n_heads, ffn_dim, domains, lora_rank, lora_alpha)
             for _ in range(n_expert_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        pos = torch.arange(t, device=x.device)
        x = x + self.pos_emb(pos)[None, :, :]
        for block in self.shared_blocks:
            x = block(x)
        for block in self.expert_blocks:
            x = block(x, domain_ids)
        return self.ln_f(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MoTRoutedDeepExpertModel(MoTRoutedModel):
    def __init__(self, domain_vocab_sizes: dict[str, int], emb_dim: int, d_model: int, n_heads: int,
                 ffn_dim: int, n_layers: int, max_seq_len: int,
                 type_conditioned_domains: tuple[str, ...] = ("nlp",),
                 n_expert_layers: int = DEFAULT_N_EXPERT_LAYERS,
                 lora_rank: int = DEFAULT_LORA_RANK, lora_alpha: float = DEFAULT_LORA_ALPHA):
        super().__init__(domain_vocab_sizes, emb_dim, d_model, n_heads, ffn_dim, n_layers,
                          max_seq_len, type_conditioned_domains)
        self.backbone = DeepExpertBackbone(
            d_model, n_heads, ffn_dim, n_layers, max_seq_len, self.domains,
            n_expert_layers, lora_rank, lora_alpha)
        self.n_shared_layers = n_layers - n_expert_layers

    def forward(
        self, token_ids: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        targets: torch.Tensor | None = None, type_ids: torch.Tensor | None = None,
        switch_weight: float = 1.0,
    ):
        x = self.embed_sequence(token_ids, domain_ids, is_control, type_ids)
        h = self.backbone(x, domain_ids)
        return self.head_loss(h, domain_ids, targets, switch_weight)
