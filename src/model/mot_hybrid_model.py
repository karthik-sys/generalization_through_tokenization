"""MoT-Routed-Adaptive hybrid (arm "hybrid"): routed's mid-sequence switching mechanism,
retrained to remove the two taxes that made plain MoTRoutedModel (2.057 bpb) underperform
vanilla MoT (1.909 bpb) on raw content quality - all evaluated on the exact same held-out
data, so the gap isn't explained by what each arm is being asked to do.

Tax #1: switch_weight=50.0 is a flat, hand-picked multiplier that permanently reallocates
gradient budget toward switch-target positions, with no mechanism to notice if it's over- or
under-shooting. Fixed here with GradNormBalancer (src/model/gradnorm.py): content-loss and
switch-loss are each normalized by their own running-average magnitude before being summed,
so neither dominates purely from scale.

Tax #2: MoTRoutedModel trains exclusively on synthetic multi-domain documents chopped into
250-word snippets (see stage2_routed_stream.py) - it never sees a long, natural, continuous
single-domain passage, unlike MoT which sees nothing else. This model's forward() is
unchanged from MoTRoutedModel's, deliberately - it can already represent a pure single-domain
batch (domain_ids constant, is_control all zero) as a degenerate case. The fix lives in the
training loop (train_stage2_pod.py / stage2_modal.py): blend natural PackedDomainStream
batches in alongside PackedRoutedStream's switching batches, rather than training on
switching data exclusively.

The switching mechanism itself (control_embedding, widened per-domain heads) is untouched -
nothing here suggests it's broken, only that it's been trained under two avoidable handicaps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.gradnorm import GradNormBalancer
from src.model.mot_routed_model import MoTRoutedModel


class MoTHybridModel(MoTRoutedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.balancer = GradNormBalancer(["content", "switch"])

    def forward(
        self, token_ids: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        targets: torch.Tensor | None = None, type_ids: torch.Tensor | None = None,
        switch_weight: float = 1.0,
    ):
        """switch_weight is accepted and ignored - kept only so this model is a drop-in
        replacement at existing MoTRoutedModel call sites (train/calibrate loops pass it
        positionally); the actual switch/content balance comes from self.balancer instead."""
        x = self.embed_sequence(token_ids, domain_ids, is_control, type_ids)
        h = self.backbone(x)
        if targets is None:
            return self.head_loss(h, domain_ids, None)
        return self._balanced_loss(h, domain_ids, targets)

    def _balanced_loss(self, h: torch.Tensor, domain_ids: torch.Tensor, targets: torch.Tensor):
        content_losses, switch_losses = [], []
        per_domain: dict[str, float] = {}

        for domain in self.domains:
            mask = domain_ids == self.domain_index[domain]
            if not mask.any():
                continue
            logits = self.heads[domain](h[mask])
            tgt = targets[mask]
            is_switch = tgt >= self.domain_vocab_sizes[domain]
            ce = F.cross_entropy(logits, tgt, reduction="none")

            if (~is_switch).any():
                content_losses.append(ce[~is_switch].mean())
            if is_switch.any():
                switch_losses.append(ce[is_switch].mean())
            per_domain[domain] = float(ce.mean().item())

        content = torch.stack(content_losses).mean() if content_losses else h.new_zeros(())
        switch = torch.stack(switch_losses).mean() if switch_losses else h.new_zeros(())
        total = self.balancer.normalize({"content": content, "switch": switch})

        per_domain["_content"] = float(content.item())
        per_domain["_switch"] = float(switch.item())
        return total, per_domain
