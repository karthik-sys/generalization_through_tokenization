"""MoTRoutedModel + shared cross-domain pooling + an architecture-fitting loss (arm 5).

This is deliberately a thin subclass of MoTRoutedModel: arm 4 (`routed`) and arm 5
(`pooled`) then differ by two things, both concentrated here -
  (a) the SharedPoolingModule inserted between projections and backbone (PMA + DANN), and
  (b) the loss function, which replaces plain per-position cross-entropy with three changes
      that each match something specific about this architecture.

It sits on the routed model (not plain MoT) because PMA pools over a sequence, and only the
routed arm's sequences contain multiple domains.

THE LOSS (the point of this rev). Plain cross-entropy summed over all positions is a poor
fit for a mixture-of-tokenizers model, for three concrete reasons:

  1. Per-domain normalisation. Predictions go into disjoint per-domain heads, and a batch
     rarely holds equal token counts per domain. Summed CE therefore lets whichever domain
     happens to contribute more positions dominate the gradient - a data-mix artefact, not
     a modelling choice. Averaging each domain's loss and then averaging across domains gives
     every domain equal say regardless of how many of its tokens landed in the batch. (This
     one is closer to a correctness fix than an enhancement.)

  2. Focal loss (Lin et al. 2017), gamma>0. Multiplies each position's CE by (1-p)^gamma,
     where p is the model's probability on the correct token. Tokens the model already nails
     stop consuming gradient; the hard ones - which for us are exactly the cross-domain
     boundary tokens and rare-vocab tokens - get the emphasis. The static 50x switch weight
     was a blunt approximation of this; focal does it dynamically per token.

  3. Confidence head. A small auxiliary head predicts, per position, whether the model will
     get the next token right (BCE against actual correctness). This is the "can the model
     predict how well it will do" idea - a calibration signal. It doesn't change what the
     model predicts, it asks the shared representation to also encode its own reliability,
     which is directly useful at inference for a router deciding whether to trust a domain.

Everything is opt-in via config weights; the pooling path is inert at adv_lambda=0, so early
ramp steps are a free correctness check that the base model still works.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mot_routed_model import MoTRoutedModel
from src.model.shared_pooling import SharedPoolingModule

LOAD_BALANCE_WEIGHT = 0.01  # Switch-Transformer's standard aux weight


class MoTPooledModel(MoTRoutedModel):
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
        chunk_size: int = 128,
        focal_gamma: float = 2.0,
        confidence_weight: float = 0.1,
    ):
        super().__init__(
            domain_vocab_sizes, emb_dim, d_model, n_heads, ffn_dim, n_layers,
            max_seq_len, type_conditioned_domains,
        )
        self.pooling = SharedPoolingModule(
            d_model, num_domains=self.num_domains, n_seeds=n_seeds,
            n_experts=n_experts, n_active=n_active, chunk_size=chunk_size, n_heads=n_heads,
        )
        self.focal_gamma = focal_gamma
        self.confidence_weight = confidence_weight
        # one shared scalar-confidence predictor over the backbone output (domain-agnostic:
        # "how sure am I about the next token here" is the same question in every domain).
        self.confidence_head = nn.Linear(d_model, 1)

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

        main, per_domain, confidence = self._fitting_loss(h, domain_ids, targets, switch_weight)
        total = (
            main
            + LOAD_BALANCE_WEIGHT * load_balance
            + adversarial
            + self.confidence_weight * confidence
        )
        per_domain = dict(per_domain)
        per_domain["_load_balance"] = float(load_balance.item())
        per_domain["_adversarial"] = float(adversarial.item())
        per_domain["_confidence"] = float(confidence.item())
        return total, per_domain

    def _fitting_loss(self, h, domain_ids, targets, switch_weight):
        """Per-domain-normalised focal CE + a confidence-prediction auxiliary.

        Mirrors MoTRoutedModel.head_loss's gather-by-domain structure (kept separate rather
        than parameterising the parent, so the routed arm's loss stays byte-for-byte
        unchanged and its running checkpoints are unaffected)."""
        domain_mean_losses = []
        per_domain: dict[str, float] = {}
        conf_loss = h.new_zeros(())
        conf_count = 0

        for domain in self.domains:
            mask = domain_ids == self.domain_index[domain]
            if not mask.any():
                continue
            hd = h[mask]
            logits = self.heads[domain](hd)
            tgt = targets[mask]

            ce = F.cross_entropy(logits, tgt, reduction="none")
            if self.focal_gamma > 0.0:
                # p = prob assigned to the correct token; detached so the focal weight
                # scales the gradient without differentiating through the weight itself.
                p = torch.exp(-ce.detach())
                ce = ((1.0 - p) ** self.focal_gamma) * ce
            if switch_weight != 1.0:
                is_switch = tgt >= self.domain_vocab_sizes[domain]
                w = torch.ones_like(ce)
                w[is_switch] = switch_weight
                ce = ce * w

            domain_mean_losses.append(ce.mean())
            per_domain[domain] = float(ce.mean().item())

            conf_logit = self.confidence_head(hd).squeeze(-1)
            with torch.no_grad():
                correct = (logits.argmax(dim=-1) == tgt).float()
            conf_loss = conf_loss + F.binary_cross_entropy_with_logits(
                conf_logit, correct, reduction="sum"
            )
            conf_count += int(mask.sum().item())

        # equal weight per active domain, not per position (see class docstring #1)
        main = torch.stack(domain_mean_losses).mean()
        confidence = conf_loss / max(conf_count, 1)
        return main, per_domain, confidence

    def routing_stats(self) -> dict[str, float]:
        """Per-expert bias values - the offline read on which dormant experts self-activated.
        A dormant expert whose bias has climbed well off its -20 init is one the router found
        a use for, which is the discovery signal this whole design exists to produce."""
        return {f"expert_{i}": float(b) for i, b in enumerate(self.pooling.router.expert_bias)}
