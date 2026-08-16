"""Shared transformer backbone (spec §3: "fully shared across domains - this is where
cross-domain generalization actually happens"). Used identically inside the MoT model
and, at matched size, as the baseline's backbone - only the embedding/head layers differ
between the two, per spec §9's matched-compute comparison.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, n_heads, t, head_dim)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (b, n_heads, t, head_dim)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Backbone(nn.Module):
    """Positional embedding + N transformer blocks + final norm. Operates on
    already-projected d_model vectors - embedding lookup happens outside, per-domain."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, n_layers: int, max_seq_len: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        pos = torch.arange(t, device=x.device)
        x = x + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
