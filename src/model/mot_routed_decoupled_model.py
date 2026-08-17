"""MoT-Routed with a gradient-DECOUPLED switch head (arm "routed5"): tests whether switch
prediction competing with content prediction for the shared backbone's gradient is what
actually taxes content quality, independent of how switch loss is weighted.

Plain MoTRoutedModel predicts switches from the SAME per-domain output head as regular
next-token content (widened by num_domains extra classes, one softmax, one weight matrix).
Every loss term computed on that head's logits sends gradient through the shared backbone -
so a heavily-upweighted switch signal (switch_weight=50) necessarily competes with content
for backbone capacity every step, structurally, regardless of how the weighting is chosen.
This is a different hypothesis from routed2/3/hybrid's (which varied the weighting
mechanism, not the gradient path) and from routed4's (which varies the weight but keeps it
on the same head).

Here the switch prediction gets its OWN small classifier per domain (num_domains-way, not
sharing the content head's weight matrix), fed the backbone's hidden state with
`.detach()` - so switch_head's own weights still learn from switch loss via backprop, but
NO gradient reaches the shared backbone/embeddings from that path. The backbone only ever
sees gradient from content loss. switch_weight stays fixed at the original 50 (unchanged
from plain routed) - this arm isolates gradient-path decoupling alone, not weighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mot_routed_model import MoTRoutedModel


class MoTRoutedDecoupledModel(MoTRoutedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.control_embedding.embedding_dim
        # unwiden the content heads back to plain vocab size (switch classes move out)
        self.heads = nn.ModuleDict(
            {d: nn.Linear(d_model, v) for d, v in self.domain_vocab_sizes.items()}
        )
        self.switch_heads = nn.ModuleDict(
            {d: nn.Linear(d_model, self.num_domains) for d in self.domain_vocab_sizes}
        )

    def head_loss(
        self, h: torch.Tensor, domain_ids: torch.Tensor, targets: torch.Tensor | None,
        switch_weight: float = 1.0,
    ):
        if targets is None:
            # content-only logits, unwidened - held-out eval already drops switch-target
            # positions before scoring (dt < domain_vocab_sizes[d]), so this needs no
            # special-casing: switch positions' logits are computed and then ignored,
            # same as they'd be discarded downstream either way.
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
            dom_loss_sum = h.new_zeros(())

            if (~is_switch).any():
                content_logits = self.heads[domain](hd[~is_switch])
                content_tgt = tgt[~is_switch]
                cl = F.cross_entropy(content_logits, content_tgt, reduction="sum")
                content_sum = content_sum + cl
                dom_loss_sum = dom_loss_sum + cl
            if is_switch.any():
                # stop-gradient into the backbone - switch_heads learns from switch_loss,
                # nothing upstream of hd.detach() does.
                switch_logits = self.switch_heads[domain](hd[is_switch].detach())
                switch_tgt = tgt[is_switch] - self.domain_vocab_sizes[domain]
                sl = F.cross_entropy(switch_logits, switch_tgt, reduction="sum")
                switch_sum = switch_sum + sl
                dom_loss_sum = dom_loss_sum + sl * switch_weight

            per_domain[domain] = dom_loss_sum.item() / n

        total_loss = (content_sum + switch_weight * switch_sum) / max(total_count, 1)
        per_domain["_content"] = (content_sum / max(total_count, 1)).item()
        return total_loss, per_domain
