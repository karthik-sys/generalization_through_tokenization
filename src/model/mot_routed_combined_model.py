"""MoT-Routed with all three fixes stacked (arm "routed4"): gradient-decoupled switch head +
learned (Kendall-style) switch weight + 2x context (1024->2048), run together in one arm
rather than as three separate single-change ablations.

Rationale for stacking instead of isolating: each of these targets a DIFFERENT hypothesis
for why routed's switch signal taxes content quality (weighting mechanism, gradient-path
competition, and switch-density-per-window respectively), so there's no reason to expect
interference between them - and three separate overnight runs cost 3x the GPU-time of one.
If the combined arm wins, the individual contributions can be ripped apart in follow-up runs
(cheap, since we'd already know it's worth the spend); if it doesn't, we've spent one run
instead of three finding that out.

Mechanism (see mot_routed_decoupled_model.py and mot_routed_learned_model.py for the
individual writeups):
  1. Switch prediction moves to its own small per-domain classifier, fed the backbone's
     hidden state with .detach() - the shared backbone never receives gradient from switch
     loss, only from content loss.
  2. The switch loss's weight is a learned scalar (via log_sigma, Kendall et al. 2018
     homoscedastic-uncertainty form: switch_loss/(2*sigma^2) + log(sigma)), initialized to
     the same effective weight (50) plain routed uses, instead of held fixed forever.
  3. max_seq_len doubles to 2048 (LONGCTX_MODEL_CFG) - set at the training-arm/config level,
     not in this file (Backbone/embed_sequence are already max_seq_len-parameterized).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mot_routed_decoupled_model import MoTRoutedDecoupledModel

INITIAL_EFFECTIVE_WEIGHT = 50.0  # same anchor as plain routed's SWITCH_WEIGHT


class MoTRoutedCombinedModel(MoTRoutedDecoupledModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sigma_init = 1.0 / math.sqrt(2 * INITIAL_EFFECTIVE_WEIGHT)
        self.log_sigma_switch = nn.Parameter(torch.tensor(math.log(sigma_init)))

    def forward(
        self, token_ids: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        targets: torch.Tensor | None = None, type_ids: torch.Tensor | None = None,
    ):
        x = self.embed_sequence(token_ids, domain_ids, is_control, type_ids)
        h = self.backbone(x)
        return self.head_loss(h, domain_ids, targets)

    def head_loss(self, h: torch.Tensor, domain_ids: torch.Tensor, targets: torch.Tensor | None):
        if targets is None:
            # eval path: decoupled content-only heads, unwidened - same as the parent's,
            # switch positions' logits computed and then discarded downstream as usual.
            out = {}
            for domain in self.domains:
                mask = domain_ids == self.domain_index[domain]
                if mask.any():
                    out[domain] = (mask, self.heads[domain](h[mask]))
            return out

        content_sum = h.new_zeros(())
        switch_sum = h.new_zeros(())
        total_count = 0
        per_domain: dict[str, float] = {}
        for domain in self.domains:
            mask = domain_ids == self.domain_index[domain]
            if not mask.any():
                continue
            tgt = targets[mask]
            is_switch = tgt >= self.domain_vocab_sizes[domain]
            hd = h[mask]

            n = int(mask.sum().item())
            total_count += n

            if (~is_switch).any():
                content_logits = self.heads[domain](hd[~is_switch])
                content_tgt = tgt[~is_switch]
                cl = F.cross_entropy(content_logits, content_tgt, reduction="sum")
                content_sum = content_sum + cl
            else:
                cl = h.new_zeros(())
            if is_switch.any():
                # stop-gradient into the backbone, same as the decoupled-only arm.
                switch_logits = self.switch_heads[domain](hd[is_switch].detach())
                switch_tgt = tgt[is_switch] - self.domain_vocab_sizes[domain]
                sl = F.cross_entropy(switch_logits, switch_tgt, reduction="sum")
                switch_sum = switch_sum + sl
            else:
                sl = h.new_zeros(())

            per_domain[domain] = (cl.item() + sl.item()) / n

        sigma = torch.exp(self.log_sigma_switch)
        effective_weight = 1.0 / (2 * sigma * sigma)
        total_loss = (content_sum + effective_weight * switch_sum) / max(total_count, 1) + torch.log(sigma)

        per_domain["_content"] = (content_sum / max(total_count, 1)).item()
        per_domain["_switch_weight"] = effective_weight.item()
        per_domain["_sigma"] = sigma.item()
        return total_loss, per_domain
