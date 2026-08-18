"""Bet 1 ("copy gate"): a pointer/copy mechanism on the nlp head only.

LAMBADA's construction means the target word is very often mentioned earlier in the same
passage. A plain softmax head has to reconstruct that word from scratch, spreading mass
across plausible synonyms even when the exact word already appeared - this is a retrieval
problem, not a modelling-capacity problem, and pointer networks (See et al. 2017,
"Get To The Point") exist specifically for it.

At each nlp content position t, compute causal attention scores against all EARLIER
real-content nlp positions in the same sequence (switch-marker positions are excluded as
copy sources - they don't carry a meaningful vocab identity), softmax them, and scatter the
resulting mass onto the vocab ids those source positions actually hold. A learned per-
position gate blends that copy distribution with the normal vocab softmax:

    p_mix = (1 - gate) * softmax(vocab_logits) + gate * copy_distribution

Only ~525K new params (two d_model x d_model projections + a gate vector), and only the nlp
head's loss computation changes - code/math/science are untouched. Meant to be launched as
continued training from an already-trained checkpoint (e.g. routed8), not from scratch: the
copy-gate machinery is useless until the backbone already produces good representations to
attend with.

Kill criterion (see docs/handoff_lambada_bets.md): log the gate's mean value. If it collapses
toward 0, the model isn't finding the copy path useful and retrieval wasn't the limiting
error class - stop. If it rises and EM moves while ppl barely does, this is the lever.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mot_routed_model import MoTRoutedModel


class MoTRoutedCopyGateModel(MoTRoutedModel):
    def __init__(self, *args, gate_bias_init: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.control_embedding.embedding_dim
        self.copy_q = nn.Linear(d_model, d_model)
        self.copy_k = nn.Linear(d_model, d_model)
        self.copy_gate = nn.Linear(d_model, 1)
        if gate_bias_init != 0.0:
            # routed16 ("v2"): routed11/14 left this at PyTorch's default near-zero init, so
            # sigmoid(bias)~=0.5 - the gate started as an unbiased 50/50 coin flip rather than a
            # conservative default. A negative init means the gate only turns on where the
            # learned copy score actually justifies it, not by default everywhere.
            nn.init.constant_(self.copy_gate.bias, gate_bias_init)

    def forward(
        self, token_ids: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        targets: torch.Tensor | None = None, type_ids: torch.Tensor | None = None,
        switch_weight: float = 1.0,
    ):
        x = self.embed_sequence(token_ids, domain_ids, is_control, type_ids)
        h = self.backbone(x)
        return self.head_loss(h, domain_ids, targets, switch_weight, token_ids=token_ids, is_control=is_control)

    def head_loss(
        self, h: torch.Tensor, domain_ids: torch.Tensor, targets: torch.Tensor | None,
        switch_weight: float = 1.0, token_ids: torch.Tensor | None = None,
        is_control: torch.Tensor | None = None,
    ):
        if targets is None or token_ids is None or is_control is None:
            # inspection/generation path - copy gate isn't wired into it yet, fall back to
            # the plain vocab head so existing eval/generate code keeps working unmodified.
            return super().head_loss(h, domain_ids, targets, switch_weight)

        nlp = "nlp"
        total_loss = h.new_zeros(())
        total_count = 0
        per_domain: dict[str, float] = {}
        for domain in self.domains:
            mask = domain_ids == self.domain_index[domain]
            if not mask.any():
                continue
            if domain != nlp:
                logits = self.heads[domain](h[mask])
                tgt = targets[mask]
                per_pos_loss = F.cross_entropy(logits, tgt, reduction="none")
                if switch_weight != 1.0:
                    is_switch = tgt >= self.domain_vocab_sizes[domain]
                    weights = torch.ones_like(per_pos_loss)
                    weights[is_switch] = switch_weight
                    per_pos_loss = per_pos_loss * weights
                loss = per_pos_loss.sum()
                n = int(mask.sum().item())
            else:
                per_pos_loss, gate_mean = self._nlp_copy_gate_losses(
                    h, domain_ids, is_control, token_ids, targets, switch_weight)
                loss = per_pos_loss.sum()
                n = per_pos_loss.numel()
                per_domain["nlp_copy_gate_mean"] = gate_mean
            total_loss = total_loss + loss
            total_count += n
            per_domain[domain] = loss.item() / max(n, 1)
        return total_loss / max(total_count, 1), per_domain

    def _nlp_copy_gate_losses(
        self, h: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        token_ids: torch.Tensor, targets: torch.Tensor, switch_weight: float,
    ) -> tuple[torch.Tensor, float]:
        nlp = "nlp"
        idx = self.domain_index[nlp]
        vocab_size = self.domain_vocab_sizes[nlp] + self.num_domains
        ctrl = is_control.bool()
        b = h.shape[0]
        all_losses = []
        gate_vals = []
        for bi in range(b):
            query_mask = domain_ids[bi] == idx  # every nlp-head position, content + switch-marker
            query_pos = query_mask.nonzero(as_tuple=True)[0]
            if query_pos.numel() == 0:
                continue
            source_mask_full = query_mask & (~ctrl[bi])  # only real content tokens are copy sources
            is_source = source_mask_full[query_pos]

            h_q = h[bi, query_pos]
            tok_at_pos = token_ids[bi, query_pos]
            tgt_row = targets[bi, query_pos]

            q = self.copy_q(h_q)
            k = self.copy_k(h_q)
            gate = torch.sigmoid(self.copy_gate(h_q).squeeze(-1))
            vocab_logits = self.heads[nlp](h_q)

            n = query_pos.numel()
            strictly_before = query_pos.unsqueeze(1) > query_pos.unsqueeze(0)  # (n,n)
            valid_source = strictly_before & is_source.unsqueeze(0)
            scores = (q @ k.transpose(-1, -2)) / (h_q.shape[-1] ** 0.5)
            scores = scores.masked_fill(~valid_source, float("-inf"))
            row_has_source = valid_source.any(dim=-1)
            scores = torch.where(row_has_source.unsqueeze(-1), scores, torch.zeros_like(scores))
            p_copy_pos = F.softmax(scores, dim=-1) * row_has_source.unsqueeze(-1)

            copy_vocab = torch.zeros(n, vocab_size, device=h.device, dtype=p_copy_pos.dtype)
            copy_vocab.scatter_add_(1, tok_at_pos.unsqueeze(0).expand(n, -1), p_copy_pos)

            p_vocab = F.softmax(vocab_logits, dim=-1)
            p_mix = (1 - gate.unsqueeze(-1)) * p_vocab + gate.unsqueeze(-1) * copy_vocab
            p_mix = p_mix.clamp_min(1e-9)
            nll = -torch.log(p_mix.gather(1, tgt_row.unsqueeze(-1)).squeeze(-1))
            if switch_weight != 1.0:
                is_switch = tgt_row >= self.domain_vocab_sizes[nlp]
                w = torch.ones_like(nll)
                w[is_switch] = switch_weight
                nll = nll * w
            all_losses.append(nll)
            gate_vals.append(gate.mean().item())

        if not all_losses:
            return h.new_zeros(0), 0.0
        return torch.cat(all_losses), (sum(gate_vals) / len(gate_vals))
