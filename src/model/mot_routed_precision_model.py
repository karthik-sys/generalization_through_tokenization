"""Bet 3, exploratory ("precision head"): a margin-trained second pathway for the nlp head's
top-1 prediction, alongside the normal softmax.

NLL optimizes calibration (spread mass over plausible synonyms, don't be overconfident) -
EM optimizes argmax (pick the ONE right word). A single softmax head can't be told to do
both perfectly, and the standard training signal (cross-entropy) only ever pushes toward
the calibration objective. Bet 3 doesn't fight that - it adds a small second scoring
pathway whose only job is argmax separation among "narrative-common" words, trained with a
contrastive margin loss, and blends its scores additively into the main logits.

Simplifications vs. the original proposal, made explicit rather than silently applied:
  - "Curated frequent-word list" -> the nlp tokenizer's own first PRECISION_K_WORDS surface-
    tier ids. SentencePiece surface vocabs are fit on real text, so low ids already skew
    toward frequent, common pieces - this needs no external wordlist/dependency, and (a real
    bonus, not just convenience) makes "is target in the precision set" a single comparison
    (target < k) instead of a lookup table.
  - "Competitors mined every N steps" -> in-batch hard negatives, computed inline from the
    SAME forward pass's precision scores. The vocab logits (and now the precision scores)
    are already being computed every step; reusing them as the competitor pool is simpler
    than a periodic external mining loop and the competitors are always current, never stale.

Kill criterion: if the gate collapses to ~0, or EM doesn't beat Bet 1 at matched steps, the
NLL head's ranking was already the binding constraint and precision needs a data-side lever
(cooldown/annealing) instead of a new pathway.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mot_routed_model import MoTRoutedModel

PRECISION_K_WORDS = 16000
MARGIN = 0.5
MARGIN_LOSS_WEIGHT = 0.1
N_COMPETITORS = 8


class MoTRoutedPrecisionModel(MoTRoutedModel):
    def __init__(self, *args, precision_k_words: int = PRECISION_K_WORDS,
                 margin: float = MARGIN, margin_loss_weight: float = MARGIN_LOSS_WEIGHT,
                 n_competitors: int = N_COMPETITORS, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.control_embedding.embedding_dim
        emb_dim = self.embeddings["nlp"].embedding_dim
        self.precision_k_words = min(precision_k_words, self.domain_vocab_sizes["nlp"])
        self.margin = margin
        self.margin_loss_weight = margin_loss_weight
        self.n_competitors = n_competitors
        self.precision_proj = nn.Linear(d_model, emb_dim)
        self.precision_gate = nn.Linear(d_model, 1)

    def head_loss(
        self, h: torch.Tensor, domain_ids: torch.Tensor, targets: torch.Tensor | None,
        switch_weight: float = 1.0,
    ):
        if targets is None:
            return super().head_loss(h, domain_ids, targets, switch_weight)

        nlp = "nlp"
        k = self.precision_k_words
        total_loss = h.new_zeros(())
        total_count = 0
        per_domain: dict[str, float] = {}
        for domain in self.domains:
            mask = domain_ids == self.domain_index[domain]
            if not mask.any():
                continue
            h_masked = h[mask]
            logits = self.heads[domain](h_masked)
            tgt = targets[mask]

            margin_loss_val = 0.0
            if domain == nlp:
                gate = torch.sigmoid(self.precision_gate(h_masked)).squeeze(-1)  # (n,)
                nlp_emb = self.embeddings[nlp].weight[:k]  # (k, emb_dim), reused not duplicated
                precision_scores = self.precision_proj(h_masked) @ nlp_emb.T  # (n, k)
                bias = torch.zeros_like(logits)
                bias[:, :k] = gate.unsqueeze(-1) * precision_scores
                logits = logits + bias

                margin_loss = self._precision_margin_loss(precision_scores, tgt, k)
                margin_loss_val = margin_loss.item()
                per_domain["nlp_precision_gate_mean"] = gate.mean().item()

            per_pos_loss = F.cross_entropy(logits, tgt, reduction="none")
            if switch_weight != 1.0:
                is_switch = tgt >= self.domain_vocab_sizes[domain]
                weights = torch.ones_like(per_pos_loss)
                weights[is_switch] = switch_weight
                per_pos_loss = per_pos_loss * weights
            loss = per_pos_loss.sum()
            n = int(mask.sum().item())
            if domain == nlp:
                # per_pos_loss is summed (not mean-reduced) below, so scale the mean-
                # reduced margin loss by n to combine on the same footing.
                loss = loss + self.margin_loss_weight * margin_loss_val * n
                per_domain["nlp_margin_loss"] = margin_loss_val

            total_loss = total_loss + loss
            total_count += n
            per_domain[domain] = loss.item() / max(n, 1)
        return total_loss / max(total_count, 1), per_domain

    def _precision_margin_loss(self, precision_scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
        """In-batch margin loss: max(0, margin - score(true_target) + score(hardest_competitor)),
        only over positions whose true target is itself in the precision set (target < k) -
        elsewhere there's no "true precision score" to anchor the margin against."""
        is_precision_target = targets < k
        if not is_precision_target.any():
            return precision_scores.new_zeros(())
        scores = precision_scores[is_precision_target]
        tgt = targets[is_precision_target].unsqueeze(-1)
        true_score = scores.gather(1, tgt).squeeze(-1)
        masked = scores.scatter(1, tgt, float("-inf"))
        n_comp = min(self.n_competitors, k - 1)
        competitor_score = masked.topk(n_comp, dim=-1).values.max(dim=-1).values
        return torch.clamp(self.margin - true_score + competitor_score, min=0.0).mean()
