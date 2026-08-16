"""Pooled v2 (arm "pooled2"): same PMA + DANN cross-domain pooling as arm 5, kept
deliberately intact rather than stripped down - cross-domain BPB beat single-domain BPB for
BOTH routed (1.589 vs 2.057) and pooled (2.166 vs 2.397) on real held-out data, reproduced at
two different routed checkpoints. That's evidence the pooling itself is doing real work, so
this revision only touches things that were costing compute or gradient signal without
earning it back:

  1. GradNorm-lite loss balancing (src/model/gradnorm.py) replaces fixed/ramped weights
     (main + 0.01*load_balance + adversarial + 0.1*confidence, summed with no accounting for
     each term's actual scale) with per-term EMA normalization. Arm 5 plateaued hard (ema
     loss ~6.3-6.4 by 82k steps, 200+ controller plateau-cuts) - a real possibility is one
     term dominating the gradient by scale alone rather than by being genuinely more
     important, and fixed weights have no way to notice or correct that.

  2. Sparse top-2-of-16 expert routing (ExpertRouter.top_k, shared_pooling.py) instead of
     dense all-16. The original router computes every expert's forward pass every step
     regardless of routing weight - the single biggest avoidable compute cost in the arm.
     Sparse dispatch only runs the experts actually selected per example.

  3. Larger PMA chunk size (128 -> 256): half as many pooling calls per sequence, coarser
     but cheaper context.

  4. Confidence head dropped entirely. It's a calibration side-signal, orthogonal to the
     cross-domain compression result that's actually promising - safe to cut, unlike PMA/DANN.

Per-domain normalization and focal-loss reweighting (arm 5's other two changes) are kept
unchanged - neither was implicated in the plateau, and both are independently well-motivated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.gradnorm import GradNormBalancer
from src.model.mot_routed_model import MoTRoutedModel
from src.model.shared_pooling import SharedPoolingModule


class MoTPooled2Model(MoTRoutedModel):
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
        n_seeds: int = 16,
        n_experts: int = 16,
        n_active: int = 8,
        chunk_size: int = 256,
        top_k: int = 2,
        focal_gamma: float = 2.0,
    ):
        super().__init__(
            domain_vocab_sizes, emb_dim, d_model, n_heads, ffn_dim, n_layers,
            max_seq_len, type_conditioned_domains,
        )
        self.pooling = SharedPoolingModule(
            d_model, num_domains=self.num_domains, n_seeds=n_seeds,
            n_experts=n_experts, n_active=n_active, chunk_size=chunk_size,
            n_heads=n_heads, top_k=top_k,
        )
        self.focal_gamma = focal_gamma
        self.balancer = GradNormBalancer(["content", "load_balance", "adversarial"])

    def forward(
        self, token_ids: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        targets: torch.Tensor | None = None, type_ids: torch.Tensor | None = None,
        switch_weight: float = 1.0, adv_lambda: float = 0.0,
    ):
        x = self.embed_sequence(token_ids, domain_ids, is_control, type_ids)
        x, load_balance, adversarial = self.pooling(x, domain_ids, adv_lambda)
        h = self.backbone(x)

        if targets is None:
            return self.head_loss(h, domain_ids, None, switch_weight)

        content, per_domain = self._focal_loss(h, domain_ids, targets, switch_weight)
        total = self.balancer.normalize({
            "content": content, "load_balance": load_balance, "adversarial": adversarial,
        })
        per_domain = dict(per_domain)
        per_domain["_load_balance"] = float(load_balance.item())
        per_domain["_adversarial"] = float(adversarial.item())
        per_domain["_balancer_ema"] = self.balancer.state()
        return total, per_domain

    def _focal_loss(self, h, domain_ids, targets, switch_weight):
        """Per-domain-normalised focal CE - identical to MoTPooledModel's, minus the
        confidence-prediction auxiliary."""
        domain_mean_losses = []
        per_domain: dict[str, float] = {}

        for domain in self.domains:
            mask = domain_ids == self.domain_index[domain]
            if not mask.any():
                continue
            logits = self.heads[domain](h[mask])
            tgt = targets[mask]

            ce = F.cross_entropy(logits, tgt, reduction="none")
            if self.focal_gamma > 0.0:
                p = torch.exp(-ce.detach())
                ce = ((1.0 - p) ** self.focal_gamma) * ce
            if switch_weight != 1.0:
                is_switch = tgt >= self.domain_vocab_sizes[domain]
                w = torch.ones_like(ce)
                w[is_switch] = switch_weight
                ce = ce * w

            domain_mean_losses.append(ce.mean())
            per_domain[domain] = float(ce.mean().item())

        main = torch.stack(domain_mean_losses).mean()
        return main, per_domain

    def routing_stats(self) -> dict[str, float]:
        return {f"expert_{i}": float(b) for i, b in enumerate(self.pooling.router.expert_bias)}
