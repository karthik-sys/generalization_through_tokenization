"""Tied-head MoT (routed29): reallocates parameter budget rather than adding to it.

At base scale, MoTRoutedModel's 4 output heads (full Linear(d_model, vocab+num_domains)
per domain) cost ~55.5M params - 62% of the entire 89.1M model - with zero sharing against
the 4 input embedding tables (~13.9M) they're semantically the mirror of. GPT-2 avoids this
entirely via weight tying (Press & Wolf 2017: reuse the input embedding matrix as the output
projection too, shown to IMPROVE quality, not just save params) - but direct tying needs
emb_dim == d_model, which we don't have (emb_dim=128, d_model=512, bridged by a separate
`projections` layer on the way in).

This model ties anyway, via a small per-domain bridge Linear(d_model, emb_dim) on the way
OUT (mirroring `projections` on the way in) before reusing the domain's own embedding
matrix, transposed, as the vocab projection. The bridge (~66K params/domain) is the ONLY
new per-domain output cost, versus ~12M/domain for a full untied head - freeing roughly
48-50M at base scale. The "switch to domain k" vocab slots have no embedding row to tie to
(they're not real tokens), so they keep their own tiny separate head. That freed budget is
spent on backbone depth instead (see ROUTED29_MODEL_CFG in stage2_config.py) rather than
left on the table or spent growing total params the way routed28's large-scale run does.

Includes the copy-gate mechanism (nlp-only pointer/copy blend, see mot_routed_copygate_
model.py) unmodified in spirit - it mixes with the TIED vocab distribution the same way it
mixed with the untied one, since copy-gate only ever needed "some vocab distribution to
blend with", not a specific head implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.backbone import Backbone

NUM_TYPES = 4


class MoTRoutedTiedModel(nn.Module):
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
        gate_bias_init: float = 0.0,
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
        self.control_embedding = nn.Embedding(self.num_domains, d_model)
        self.backbone = Backbone(d_model, n_heads, ffn_dim, n_layers, max_seq_len)

        # tied output path - see module docstring
        self.head_bridges = nn.ModuleDict({d: nn.Linear(d_model, emb_dim) for d in domain_vocab_sizes})
        self.switch_heads = nn.ModuleDict({d: nn.Linear(d_model, self.num_domains) for d in domain_vocab_sizes})
        self.output_bias = nn.ParameterDict(
            {d: nn.Parameter(torch.zeros(v)) for d, v in domain_vocab_sizes.items()}
        )

        # copy-gate (routed11/20/21/24/25's mechanism), nlp-only - see mot_routed_copygate_model.py
        d_model_ = d_model
        self.copy_q = nn.Linear(d_model_, d_model_)
        self.copy_k = nn.Linear(d_model_, d_model_)
        self.copy_gate = nn.Linear(d_model_, 1)
        if gate_bias_init != 0.0:
            nn.init.constant_(self.copy_gate.bias, gate_bias_init)

    def switch_target_index(self, from_domain: str, to_domain: str) -> int:
        return self.domain_vocab_sizes[from_domain] + self.domain_index[to_domain]

    def embed_sequence(
        self, token_ids: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

    def _domain_logits(self, h_masked: torch.Tensor, domain: str) -> torch.Tensor:
        """Tied vocab logits + switch logits for one domain, concatenated to match the
        existing (vocab_size + num_domains) target-index convention every other arm uses."""
        bridged = self.head_bridges[domain](h_masked)
        vocab_logits = bridged @ self.embeddings[domain].weight.T + self.output_bias[domain]
        switch_logits = self.switch_heads[domain](h_masked)
        return torch.cat([vocab_logits, switch_logits], dim=-1)

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
        nlp = "nlp"
        if targets is None:
            out = {}
            for domain in self.domains:
                mask = domain_ids == self.domain_index[domain]
                if not mask.any():
                    continue
                if domain == nlp and token_ids is not None and is_control is not None:
                    p_mix = self._nlp_copy_gate_pmix(h, domain_ids, is_control, token_ids)
                    out[domain] = (mask, torch.log(p_mix.clamp_min(1e-9)))
                else:
                    out[domain] = (mask, self._domain_logits(h[mask], domain))
            return out

        if token_ids is None or is_control is None:
            raise ValueError("MoTRoutedTiedModel.head_loss needs token_ids/is_control (targets given)")

        total_loss = h.new_zeros(())
        total_count = 0
        per_domain: dict[str, float] = {}
        for domain in self.domains:
            mask = domain_ids == self.domain_index[domain]
            if not mask.any():
                continue
            if domain != nlp:
                logits = self._domain_logits(h[mask], domain)
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
                per_pos_loss, gate_mean = self._nlp_copy_gate_losses(h, domain_ids, is_control, token_ids, targets, switch_weight)
                loss = per_pos_loss.sum()
                n = per_pos_loss.numel()
                per_domain["nlp_copy_gate_mean"] = gate_mean
            total_loss = total_loss + loss
            total_count += n
            per_domain[domain] = loss.item() / max(n, 1)
        return total_loss / max(total_count, 1), per_domain

    def _nlp_copy_gate_pmix(
        self, h: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor, token_ids: torch.Tensor,
    ) -> torch.Tensor:
        nlp = "nlp"
        idx = self.domain_index[nlp]
        vocab_size = self.domain_vocab_sizes[nlp] + self.num_domains
        ctrl = is_control.bool()
        b = h.shape[0]
        all_pmix = []
        for bi in range(b):
            query_mask = domain_ids[bi] == idx
            query_pos = query_mask.nonzero(as_tuple=True)[0]
            if query_pos.numel() == 0:
                continue
            source_mask_full = query_mask & (~ctrl[bi])
            is_source = source_mask_full[query_pos]

            h_q = h[bi, query_pos]
            tok_at_pos = token_ids[bi, query_pos]

            q = self.copy_q(h_q)
            k = self.copy_k(h_q)
            gate = torch.sigmoid(self.copy_gate(h_q).squeeze(-1))
            vocab_logits = self._domain_logits(h_q, nlp)

            n = query_pos.numel()
            strictly_before = query_pos.unsqueeze(1) > query_pos.unsqueeze(0)
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
            all_pmix.append(p_mix)

        if not all_pmix:
            return h.new_zeros(0, vocab_size)
        return torch.cat(all_pmix, dim=0)

    def _nlp_copy_gate_losses(
        self, h: torch.Tensor, domain_ids: torch.Tensor, is_control: torch.Tensor,
        token_ids: torch.Tensor, targets: torch.Tensor, switch_weight: float,
    ) -> tuple[torch.Tensor, float]:
        nlp = "nlp"
        idx = self.domain_index[nlp]
        mask = domain_ids == idx
        if not mask.any():
            return h.new_zeros(0), 0.0
        p_mix = self._nlp_copy_gate_pmix(h, domain_ids, is_control, token_ids).clamp_min(1e-9)
        tgt = targets[mask]
        nll = -torch.log(p_mix.gather(1, tgt.unsqueeze(-1)).squeeze(-1))
        if switch_weight != 1.0:
            is_switch = tgt >= self.domain_vocab_sizes[nlp]
            w = torch.ones_like(nll)
            w[is_switch] = switch_weight
            nll = nll * w
        with torch.no_grad():
            gate_mean = torch.sigmoid(self.copy_gate(h[mask]).squeeze(-1)).mean().item()
        return nll, gate_mean

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_params_by_component(self) -> dict[str, int]:
        return {
            "embeddings": sum(p.numel() for p in self.embeddings.parameters()),
            "type_embeddings": sum(p.numel() for p in self.type_embeddings.parameters()),
            "control_embedding": sum(p.numel() for p in self.control_embedding.parameters()),
            "projections": sum(p.numel() for p in self.projections.parameters()),
            "backbone": self.backbone.num_params(),
            "head_bridges": sum(p.numel() for p in self.head_bridges.parameters()),
            "switch_heads": sum(p.numel() for p in self.switch_heads.parameters()),
            "output_bias": sum(p.numel() for p in self.output_bias.values()),
            "copy_gate_module": sum(p.numel() for n, p in self.named_parameters()
                                     if n.startswith(("copy_q.", "copy_k.", "copy_gate."))),
        }
