"""GradNorm-lite: EMA-based normalization for combining loss terms of unequal, unknown scale.

A fixed multiplier (e.g. routed's switch_weight=50.0) is a one-time guess at how much a
secondary loss term should matter relative to the main one. It doesn't adapt if the term's
actual magnitude drifts over training, and there's no principled way to have picked 50 over
20 or 100 in the first place - it was chosen because it moved switch_accuracy off zero, not
because it's balanced.

This does the cheap version of what GradNorm (Chen et al. 2018) targets: track each term's
own running-average magnitude, and divide by it before summing. A term that's naturally 10x
larger than another no longer dominates the combined gradient purely because of its scale -
every term contributes comparably regardless of what units/range it happens to live in.
"""

from __future__ import annotations

import torch


class GradNormBalancer:
    def __init__(self, names: list[str], beta: float = 0.98, eps: float = 1e-6):
        self.beta = beta
        self.eps = eps
        self.ema: dict[str, float] = {n: None for n in names}

    def normalize(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        """values: {name: scalar loss tensor}. Returns the summed, per-term-normalized loss.
        A term whose value is exactly 0 this step (e.g. no switch positions in this batch)
        is skipped for both the sum and the EMA update - nothing to normalize by."""
        total = None
        for name, val in values.items():
            v = float(val.detach().abs().item())
            if v < self.eps:
                continue
            self.ema[name] = v if self.ema[name] is None else self.beta * self.ema[name] + (1 - self.beta) * v
            weight = 1.0 / (self.ema[name] + self.eps)
            term = weight * val
            total = term if total is None else total + term
        return total if total is not None else next(iter(values.values())).new_zeros(())

    def state(self) -> dict[str, float | None]:
        return dict(self.ema)
