"""MoT-Routed with a LEARNED switch weight (arm "routed4"): tests whether adapting
switch_weight helps, without GradNorm's separate content/switch normalization machinery in
the mix at all. Standard cross-entropy loss throughout - architecture, data, and everything
else identical to plain MoTRoutedModel. The only change is what multiplies switch-position
loss before it's summed with content loss.

Plain routed uses a flat switch_weight=50.0, hand-picked once and never revisited (see
stage2_config.py's SWITCH_WEIGHT comment - "measured... 50x upweighting fixed it", not a
principled derivation). GradNorm's variants (routed2/routed3/hybrid) replaced that with an
EMA-normalized 1:1 weighting between content and switch loss magnitudes - and every one of
them regressed hard on BPB/LAMBADA despite improving switch accuracy (see metrics.json:
routed2 LAMBADA ppl 1598 vs routed's 48, routed3 3199, hybrid 381). The likely mechanism:
equalizing magnitudes isn't the same as calibrating correctly - as switch loss shrinks with
training, GradNorm keeps re-inflating its relative weight to stay equal to content, pulling
gradient budget away from content on a sparse, noisy signal.

This arm asks a narrower question: can a weight that's free to move, but has to justify
moving away from a reasonable starting point, beat a value fixed forever at 50? Uses Kendall
et al. 2018's homoscedastic-uncertainty formulation (originally derived for Gaussian
regression, commonly adapted in practice to arbitrary loss terms including cross-entropy):
    weighted_switch_loss = switch_loss / (2 * sigma^2) + log(sigma)
sigma is a single learned scalar (via log_sigma for positivity), initialized so the initial
effective weight is exactly 50 (sigma_0 = 1/sqrt(100) = 0.1). The log(sigma) term is a real
regularizer, not decoration - it grows without bound as sigma -> infinity (weight -> 0), so
the optimizer can't cheat by just discounting the harder task to shrink observed loss, which
is the standard failure mode of an unconstrained learned multiplier.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mot_routed_model import MoTRoutedModel

INITIAL_EFFECTIVE_WEIGHT = 50.0  # matches SWITCH_WEIGHT - same anchor plain routed uses


class MoTRoutedLearnedModel(MoTRoutedModel):
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
            # eval/generation path, unchanged from the parent - the widened head shape is
            # identical, only the training-time loss combination differs here.
            return super().head_loss(h, domain_ids, targets, switch_weight=1.0)

        content_sum = h.new_zeros(())
        switch_sum = h.new_zeros(())
        total_count = 0
        per_domain: dict[str, float] = {}
        for domain in self.domains:
            mask = domain_ids == self.domain_index[domain]
            if not mask.any():
                continue
            logits = self.heads[domain](h[mask])
            tgt = targets[mask]
            per_pos_loss = F.cross_entropy(logits, tgt, reduction="none")
            is_switch = tgt >= self.domain_vocab_sizes[domain]

            n = int(mask.sum().item())
            total_count += n
            per_domain[domain] = per_pos_loss.sum().item() / n

            if is_switch.any():
                switch_sum = switch_sum + per_pos_loss[is_switch].sum()
            if (~is_switch).any():
                content_sum = content_sum + per_pos_loss[~is_switch].sum()

        sigma = torch.exp(self.log_sigma_switch)
        effective_weight = 1.0 / (2 * sigma * sigma)
        weighted_switch = effective_weight * switch_sum
        total_loss = (content_sum + weighted_switch) / max(total_count, 1) + torch.log(sigma)

        per_domain["_content"] = (content_sum / max(total_count, 1)).item()
        per_domain["_switch_weight"] = effective_weight.item()
        per_domain["_sigma"] = sigma.item()
        return total_loss, per_domain
