"""MoT with mid-sequence domain switching (spec §2.2's full version).

MoTModel (the v1 model) fixes one domain per forward pass - fine for training on
single-domain documents, but it can't represent a sequence that starts as prose and
switches to code partway through, which is what real inference traffic looks like.

This model routes *per position*: different spans of the same sequence go through
different domain embedding tables, and the domain switch itself is a token the model
predicts. Three pieces make that work:

  1. A shared control embedding for the <domain:X> tokens. These are not domain-specific
     (a switch marker means the same thing wherever it appears), so they get one small
     shared table rather than a row in every domain's table.

  2. Each domain's output head is widened by num_domains extra slots. Predicting slot
     `vocab_d + k` from inside domain d means "switch to domain k next". That's what
     makes a switch a normal next-token prediction instead of an external control
     signal - spec §2.2's "same mechanism as models learning to emit <tool_call>".

  3. A forward pass that gathers each domain's positions, embeds them with that domain's
     table, and scatters the results back into sequence order before the shared backbone
     runs once over the whole thing. Attention still sees the full mixed-domain context,
     which is where spec §3's cross-domain generalization is supposed to come from.

Loss is computed per-domain-group rather than as one flat tensor, because each domain
head emits a different number of logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.backbone import Backbone

NUM_TYPES = 4  # SURFACE, SYLLABLE, MORPHEME, BYTE - see nlp_hybrid_tokenizer.py


class MoTRoutedModel(nn.Module):
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
    ):
        super().__init__()
        self.domains = list(domain_vocab_sizes)
        self.domain_index = {d: i for i, d in enumerate(self.domains)}
        self.domain_vocab_sizes = dict(domain_vocab_sizes)
        self.num_domains = len(self.domains)

        self.embeddings = nn.ModuleDict(
            {d: nn.Embedding(v, emb_dim) for d, v in domain_vocab_sizes.items()}
        )
        self.type_embeddings = nn.ModuleDict(
            {d: nn.Embedding(NUM_TYPES, emb_dim) for d in type_conditioned_domains if d in domain_vocab_sizes}
        )
        self.projections = nn.ModuleDict({d: nn.Linear(emb_dim, d_model) for d in domain_vocab_sizes})

        # shared control vocabulary - one row per domain, meaning "<domain:X> appears here"
        self.control_embedding = nn.Embedding(self.num_domains, d_model)

        self.backbone = Backbone(d_model, n_heads, ffn_dim, n_layers, max_seq_len)

        # + num_domains slots per head: "switch to domain k" as a predictable token
        self.heads = nn.ModuleDict(
            {d: nn.Linear(d_model, v + self.num_domains) for d, v in domain_vocab_sizes.items()}
        )

    def switch_target_index(self, from_domain: str, to_domain: str) -> int:
        """Index in `from_domain`'s head that means 'switch to to_domain'."""
        return self.domain_vocab_sizes[from_domain] + self.domain_index[to_domain]

    def embed_sequence(
        self, token_ids: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """token_ids/domain_ids/is_control are all (batch, seq). Returns (batch, seq, d_model).

        The output tensor is allocated fp32 (matching parameter storage - autocast never
        changes that), but under torch.autocast the projection/embedding calls below
        return fp16/bf16. Index-assigning a lower-precision result into an fp32 tensor is
        a dtype mismatch, not an implicit cast - each assignment casts back to out.dtype
        explicitly, which loses nothing (autocast's whole point is compute in low
        precision, not storage), but was missing before and only shows up under real GPU
        mixed precision - the CPU-only stage-1 training never used autocast, so this bug
        was invisible there.
        """
        b, t = token_ids.shape
        out = torch.zeros(b, t, self.control_embedding.embedding_dim, device=token_ids.device, dtype=self.control_embedding.weight.dtype)

        control_mask = is_control.bool()
        if control_mask.any():
            out[control_mask] = self.control_embedding(domain_ids[control_mask]).to(out.dtype)

        for domain in self.domains:
            mask = (~control_mask) & (domain_ids == self.domain_index[domain])
            if not mask.any():
                continue
            ids = token_ids[mask]
            emb = self.embeddings[domain](ids)
            if domain in self.type_embeddings and type_ids is not None:
                emb = emb + self.type_embeddings[domain](type_ids[mask])
            out[mask] = self.projections[domain](emb).to(out.dtype)
        return out

    def forward(
        self, token_ids: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        targets: torch.Tensor | None = None, type_ids: torch.Tensor | None = None,
        switch_weight: float = 1.0,
    ):
        """domain_ids selects both the embedding table for a position and the head that
        predicts the next token from it. Those coincide: a control token <domain:X> is
        embedded from the shared control table but carries domain_ids = X, so it predicts
        into X's head - which is right, since the token after a switch marker is in X.

        switch_weight upweights the loss contribution of switch-target positions.
        Measured on the real corpus, switches are ~1 in 461 positions - with
        switch_weight=1.0 that signal is too sparse for unweighted cross-entropy to move
        argmax off the dominant regular-token prediction (confirmed: switch_accuracy
        stayed at 0.000 across 4 full epochs). Denominator stays the unweighted position
        count, so this changes the gradient's relative emphasis without changing what
        "average loss" means for comparison across runs.

        Returns (loss, per_domain_losses) when targets are given, else a dict of
        {domain: (positions_mask, logits)} for inspection/generation.
        """
        x = self.embed_sequence(token_ids, domain_ids, is_control, type_ids)
        h = self.backbone(x)
        return self.head_loss(h, domain_ids, targets, switch_weight)

    def head_loss(
        self, h: torch.Tensor, domain_ids: torch.Tensor, targets: torch.Tensor | None,
        switch_weight: float = 1.0,
    ):
        """Per-domain-group head application and loss. Split out of forward() so subclasses
        can insert work between embedding and the heads without duplicating this logic -
        pure refactor, identical behaviour, no state_dict change (so checkpoints from runs
        started before this split still load)."""
        if targets is None:
            out = {}
            for domain in self.domains:
                mask = domain_ids == self.domain_index[domain]
                if mask.any():
                    out[domain] = (mask, self.heads[domain](h[mask]))
            return out

        total_loss = h.new_zeros(())
        total_count = 0
        per_domain: dict[str, float] = {}
        for domain in self.domains:
            mask = domain_ids == self.domain_index[domain]
            if not mask.any():
                continue
            logits = self.heads[domain](h[mask])
            tgt = targets[mask]
            per_pos_loss = F.cross_entropy(logits, tgt, reduction="none")
            if switch_weight != 1.0:
                is_switch = tgt >= self.domain_vocab_sizes[domain]
                weights = torch.ones_like(per_pos_loss)
                weights[is_switch] = switch_weight
                per_pos_loss = per_pos_loss * weights
            loss = per_pos_loss.sum()
            total_loss = total_loss + loss
            n = int(mask.sum().item())
            total_count += n
            per_domain[domain] = loss.item() / n
        return total_loss / max(total_count, 1), per_domain

    def domain_embedding_alignment_loss(self) -> torch.Tensor:
        """CORAL-style distributional alignment across domain embedding TABLES (not
        activations) - Tier 1 fix for the cross-domain correlation gap identified during
        routed19's post-mortem: every domain gets a fully independent nn.Embedding with a
        disjoint vocab (separate tokenizers per domain), so nothing forces e.g. science's
        and nlp's tables into a shared region of R^emb_dim. The backbone has to bridge
        whatever gap opened up between them at every switch, with only the domain-tag
        control token as a hint.

        There's no token-level correspondence to align on (independently-tokenized vocabs
        share no subword inventory), so this matches DISTRIBUTIONS instead: for each pair
        of domains, penalize the squared difference between their embedding tables'
        (centered) covariance matrices and their mean vectors. Operates on the tables
        themselves (shape (V_d, emb_dim), fixed size regardless of batch), not per-batch
        activations - independent of batch composition, and cheap enough that the caller
        can compute this only every K steps rather than every step without losing anything
        (the tables move slowly relative to a single optimizer step).
        """
        domains = list(self.embeddings.keys())
        stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for d in domains:
            E = self.embeddings[d].weight  # (V_d, emb_dim)
            mu = E.mean(dim=0)
            centered = E - mu
            cov = (centered.T @ centered) / max(E.shape[0] - 1, 1)
            stats[d] = (mu, cov)
        loss = self.control_embedding.weight.new_zeros(())
        n_pairs = 0
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                mu_i, cov_i = stats[domains[i]]
                mu_j, cov_j = stats[domains[j]]
                loss = loss + (cov_i - cov_j).pow(2).sum() + (mu_i - mu_j).pow(2).sum()
                n_pairs += 1
        return loss / max(n_pairs, 1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_params_by_component(self) -> dict[str, int]:
        return {
            "embeddings": sum(p.numel() for p in self.embeddings.parameters()),
            "type_embeddings": sum(p.numel() for p in self.type_embeddings.parameters()),
            "control_embedding": sum(p.numel() for p in self.control_embedding.parameters()),
            "projections": sum(p.numel() for p in self.projections.parameters()),
            "backbone": self.backbone.num_params(),
            "heads": sum(p.numel() for p in self.heads.parameters()),
        }
