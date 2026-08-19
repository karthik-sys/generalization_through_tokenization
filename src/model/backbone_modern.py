"""Modern-technique backbone variant, used by routed-D (RoPE + RMSNorm only, the "safe
improver") and routed-C (RoPE + RMSNorm + SwiGLU FFN + QK-norm, the aggressive stack).
Kept fully separate from backbone.py (used by every other arm tonight) so nothing there
is at risk - this is new, less-tested code.

Each technique here is well-established elsewhere (not novel research), just new to THIS
codebase:
  - RoPE (rotary position embeddings): rotates Q/K by a position-dependent angle instead
    of adding a learned absolute position vector. Zero parameters (vs backbone.py's
    524K-param pos_emb table at base scale), and documented to generalize better -
    standard in every model built since ~2022 (Llama, Mistral, Qwen, ...).
  - RMSNorm: rescale-only normalization (drops LayerNorm's mean-centering and bias term).
    Marginally cheaper, matches or slightly beats LayerNorm in most published ablations
    (Llama, T5, ...).
  - SwiGLU FFN (routed-C only): gated FFN (silu(Wgate(x)) * Wup(x)) -> Wdown, instead of
    plain Linear->GELU->Linear. Real structural change, not just fewer params - shown to
    improve quality at matched total FFN param count (PaLM, Llama, ...). Hidden dim scaled
    down (~2/3 of the plain-FFN hidden dim) to keep total FFN params roughly matched to the
    3-matrix layout below vs plain FFN's 2-matrix layout.
  - QK-norm (routed-C only): normalizes Q/K before the attention dot product (Gemma 2,
    Qwen2). Stabilizes attention logits, particularly relevant here since both D and C are
    2-4x deeper (14-23 layers) than anything this project has trained before - deeper
    stacks are exactly where attention-logit instability starts to matter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def _precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()  # each (max_seq_len, head_dim/2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (b, n_heads, t, head_dim); cos/sin: (t, head_dim/2), broadcast to (1,1,t,head_dim)
    cos = torch.cat([cos, cos], dim=-1)[None, None, :, :].to(q.dtype)
    sin = torch.cat([sin, sin], dim=-1)[None, None, :, :].to(q.dtype)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class RopeCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, use_qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        cos, sin = _precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, n_heads, t, head_dim)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = _apply_rope(q, k, self.rope_cos[:t].to(x.device), self.rope_sin[:t].to(x.device))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        hidden = int(ffn_dim * 2 / 3)  # keep total FFN params roughly matched to plain 2-matrix GELU FFN
        self.w_gate = nn.Linear(d_model, hidden)
        self.w_up = nn.Linear(d_model, hidden)
        self.w_down = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ModernTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, max_seq_len: int,
                 use_swiglu: bool = False, use_qk_norm: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = RopeCausalSelfAttention(d_model, n_heads, max_seq_len, use_qk_norm=use_qk_norm)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim) if use_swiglu else nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ModernBackbone(nn.Module):
    """Drop-in replacement for backbone.py's Backbone - same (b,t,d)->(b,t,d) interface,
    no positional embedding table (RoPE is injected inside attention instead)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, n_layers: int, max_seq_len: int,
                 use_swiglu: bool = False, use_qk_norm: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(d_model, n_heads, ffn_dim, max_seq_len,
                                    use_swiglu=use_swiglu, use_qk_norm=use_qk_norm)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
