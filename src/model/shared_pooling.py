"""Shared cross-domain pooling (PMA) + domain-adversarial alignment (DANN).

Why this exists: MoT's per-domain projections (mot_model.py) map each domain's embeddings
into a shared d_model space, but *nothing in the loss rewards those spaces for actually
aligning*. Intersection mining confirmed this empirically - cross-domain cosine similarity
came out at 0.327 against a 0.328 random control, i.e. no measurable shared structure at
all. Plain next-token cross-entropy has no reason to produce alignment, so it doesn't.

Two complementary pressures fix that, and they are deliberately different in sign:

  1. PMA (Pooled Multihead Attention, Set Transformer / Lee et al. 2019) is a *constructive*
     pressure. A small set of learnable seed vectors attend over the projected embeddings,
     producing a fixed-size summary. The seeds are shared parameters applied identically
     regardless of which domain the tokens came from, so whatever they learn to extract
     must be expressible across all domains - a forced common vocabulary.

  2. DANN (domain-adversarial, Ganin et al. 2015) is a *negative* pressure. A discriminator
     tries to name the source domain from the pooled summary; gradient reversal makes the
     pooling penalised for being guessable. On its own it only says "hide the domain", never
     "build something shared" - which is why it pairs with PMA rather than replacing it.

The discriminator attaches to the *pooled summary*, not to raw per-token projections. Domain
identity is trivially recoverable from a raw projection (it came from a domain-specific
table by construction), so a per-token discriminator wins instantly and reversal just fights
a solved classifier. The pooled summary is the one place where "can you tell which domains
contributed" is a meaningful question.

CAUSALITY (the subtle part). Pooling over a whole sequence and injecting the result at every
position leaks the future into the past: position t would carry information derived from
t+1, the token it must predict. That inflates results silently. Pooling here is therefore
chunk-causal - the summary injected into chunk j is pooled only from chunks 0..j-1, with a
learned null summary for chunk 0. One pool per chunk rather than per position keeps it cheap.

Everything here is opt-in and additive: shapes of existing parameters are untouched, so
checkpoints from the current stage-2 runs stay loadable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReversal(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward.

    This is the whole mechanism behind adversarial alignment: the discriminator below
    trains normally to *maximise* domain accuracy, but everything upstream of this node
    receives the negated gradient and is therefore pushed to *minimise* it. One module,
    two opposing objectives, no separate optimiser or alternating update schedule.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


class PMAPooling(nn.Module):
    """Learnable seed vectors attend over a set of embeddings -> fixed-size summary.

    Permutation-invariant by construction, which matters here because there is no natural
    1:1 correspondence between one domain's tokens and another's - the thing that made raw
    per-token cosine similarity a poor alignment measure in the first place.
    """

    def __init__(self, d_model: int, n_seeds: int = 16, n_heads: int = 8):
        super().__init__()
        self.n_seeds = n_seeds
        self.seeds = nn.Parameter(torch.randn(n_seeds, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) -> (batch, n_seeds, d_model)."""
        b = x.shape[0]
        q = self.seeds.unsqueeze(0).expand(b, -1, -1).to(x.dtype)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        return self.ln(pooled)


class ExpertRouter(nn.Module):
    """Soft softmax routing over a fixed bank of experts, with dormant experts.

    Dormancy without shape changes: a dormant expert's router logit carries a large negative
    bias, so exp(bias) is ~2e-9 and the softmax denominator is numerically unchanged - other
    experts' probabilities are preserved to ~9 decimal places. This is a *soft* gate rather
    than a hard mask: the bias is learnable, so the router can raise a dormant expert on its
    own if gradients say it is useful. Capacity discovery without ever reallocating tensors,
    which keeps checkpoints loadable and keeps parameter count fixed and declarable from step
    one (matched-compute parity against the baseline/SOTA arms depends on that).

    Experts are NOT provisioned per domain-subset. An expert is free to learn a pattern that
    cuts across domain boundaries; binding them to subsets would hard-code the assumption
    that domain identity is the interesting structure, which is the hypothesis under test.
    """

    def __init__(self, d_model: int, n_experts: int = 16, n_active: int = 8, dormant_bias: float = -20.0):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        bias = torch.zeros(n_experts)
        bias[n_active:] = dormant_bias
        self.expert_bias = nn.Parameter(bias)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
             for _ in range(n_experts)]
        )

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """summary: (batch, d_model) -> (routed output (batch, d_model), load-balance loss).

        The auxiliary load-balancing loss is the standard Switch-Transformer form: without
        it a softmax router reliably collapses onto one or two experts and the rest never
        receive gradient, which would defeat the point of having a bank at all.
        """
        probs = F.softmax(self.gate(summary) + self.expert_bias, dim=-1)  # (b, n_experts)
        stacked = torch.stack([e(summary) for e in self.experts], dim=1)  # (b, n_experts, d)
        routed = (probs.unsqueeze(-1) * stacked).sum(dim=1)

        mean_prob = probs.mean(dim=0)
        load_balance = self.n_experts * (mean_prob * mean_prob).sum()
        return routed, load_balance


class DomainDiscriminator(nn.Module):
    """Predicts which domain a pooled summary came from. Trained through gradient reversal,
    so upstream pooling is pushed toward summaries this cannot classify."""

    def __init__(self, d_model: int, num_domains: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, num_domains)
        )

    def forward(self, summary: torch.Tensor, lambda_: float) -> torch.Tensor:
        return self.net(gradient_reversal(summary, lambda_))


class SharedPoolingModule(nn.Module):
    """PMA + routing + adversarial head, injected between projections and the backbone.

    Injection is additive (a single global context vector added to every token's projected
    embedding) rather than prepending pooled vectors as extra sequence positions. Prepending
    would shift every downstream index: Backbone assigns positional embeddings by raw index,
    and MoTRoutedModel computes switch targets by raw position, so k extra leading tokens
    would silently misalign every target in the routed arm. Additive conditioning needs no
    mask changes and nothing else in the pipeline has to know this module exists.
    """

    def __init__(
        self,
        d_model: int,
        num_domains: int,
        n_seeds: int = 16,
        n_experts: int = 16,
        n_active: int = 8,
        chunk_size: int = 128,
        n_heads: int = 8,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.pma = PMAPooling(d_model, n_seeds=n_seeds, n_heads=n_heads)
        self.router = ExpertRouter(d_model, n_experts=n_experts, n_active=n_active)
        self.discriminator = DomainDiscriminator(d_model, num_domains)
        # chunk 0 has no history to pool from - a learned null summary rather than zeros,
        # so "no context yet" is itself a representation the backbone can condition on.
        self.null_summary = nn.Parameter(torch.zeros(d_model))

    def forward(
        self, x: torch.Tensor, domain_ids: torch.Tensor | None = None, lambda_: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (batch, seq, d_model) projected embeddings, pre-backbone.

        Returns (x + injected context, load-balance loss, adversarial loss).
        Adversarial loss is zero when domain_ids is None or lambda_ is 0.
        """
        b, t, d = x.shape
        n_chunks = (t + self.chunk_size - 1) // self.chunk_size

        summaries = []
        aux_losses = x.new_zeros(())
        adv_losses = x.new_zeros(())
        n_adv = 0

        for j in range(n_chunks):
            if j == 0:
                summaries.append(self.null_summary.to(x.dtype).expand(b, d))
                continue
            history = x[:, : j * self.chunk_size, :]
            pooled = self.pma(history)  # (b, n_seeds, d)
            summary = pooled.mean(dim=1)  # (b, d)

            routed, lb = self.router(summary)
            aux_losses = aux_losses + lb
            summaries.append(routed)

            if domain_ids is not None and lambda_ > 0.0:
                # supervise against the dominant domain of the pooled history
                hist_dom = domain_ids[:, : j * self.chunk_size]
                target = torch.mode(hist_dom, dim=1).values
                logits = self.discriminator(summary, lambda_)
                adv_losses = adv_losses + F.cross_entropy(logits, target)
                n_adv += 1

        ctx = torch.stack(summaries, dim=1)  # (b, n_chunks, d)
        ctx = ctx.repeat_interleave(self.chunk_size, dim=1)[:, :t, :]

        aux = aux_losses / max(n_chunks - 1, 1)
        adv = adv_losses / max(n_adv, 1)
        return x + ctx, aux, adv

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
