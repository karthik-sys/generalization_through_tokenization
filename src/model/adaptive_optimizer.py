"""Loss-trajectory-aware training controller (arm 5, exploratory).

Standard training treats the optimizer as memoryless w.r.t. the loss curve: Adam keeps
running averages of *gradients*, but nothing watches the scalar *loss trajectory* and reacts
to it. This wraps three mechanisms that do, each operating at a different timescale so they
compose rather than fight over the learning rate:

  #1 Spike guard (per-step, protective). If a step's loss is NaN/Inf or spikes far above the
     recent running mean, the update is skipped entirely - a single garbage batch never gets
     to corrupt the weights. This is the same instinct as banning a checkpoint that emits NaN
     embeddings: don't let one bad state poison the next. Cheapest and highest-value of the
     three.

  #2 Plateau rescue (coarse, ~hundreds of steps). If the smoothed loss stops improving for
     `patience` observations, cut the LR multiplier hard (x0.5). This is ReduceLROnPlateau -
     it "sees the last x losses and auto-corrects" when fine adaptation isn't enough.

  #3 Online LR adaptation (fine, per-step). Nudge the LR multiplier up when the loss is
     improving and down when it's worsening (RProp-sign / hypergradient-flavored). This is
     the *stable stand-in* for a fully meta-learned optimizer: a real neural-net optimizer
     trained online is powerful but unstable and would break matched-compute parity, so this
     captures the "optimizer adapts itself from loss history" idea in a bounded ~O(1) form.

All three modulate a single `lr_mult` on top of the existing cosine schedule, clamped to a
sane band so nothing can run away during an unattended overnight run.
"""

from __future__ import annotations

import math
from collections import deque


class AdaptiveController:
    def __init__(
        self,
        spike_factor: float = 3.0,       # skip if loss > spike_factor x running mean
        spike_window: int = 50,          # running-mean window for spike detection
        ema_beta: float = 0.98,          # smoothing for plateau/online signals (denoise)
        online_up: float = 1.02,         # LR nudge when improving (#3)
        online_down: float = 0.98,       # LR nudge when worsening (#3)
        plateau_patience: int = 200,     # steps without improvement before a hard cut (#2)
        plateau_factor: float = 0.5,     # LR multiplier cut on plateau (#2)
        plateau_min_delta: float = 1e-3, # improvement smaller than this doesn't count
        lr_mult_min: float = 0.1,
        lr_mult_max: float = 3.0,
    ):
        self.spike_factor = spike_factor
        self.recent = deque(maxlen=spike_window)
        self.ema_beta = ema_beta
        self.ema = None
        self.online_up = online_up
        self.online_down = online_down
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.plateau_min_delta = plateau_min_delta
        self.lr_mult_min = lr_mult_min
        self.lr_mult_max = lr_mult_max

        self.lr_mult = 1.0
        self.best_ema = math.inf
        self.plateau_counter = 0
        # telemetry the training loop can log/checkpoint
        self.n_skipped = 0
        self.n_plateau_cuts = 0

    def should_skip(self, loss: float) -> bool:
        """#1: True if this step's update should be discarded (bad batch / divergence)."""
        if not math.isfinite(loss):
            self.n_skipped += 1
            return True
        if len(self.recent) >= self.recent.maxlen // 2:
            mean = sum(self.recent) / len(self.recent)
            if mean > 0 and loss > self.spike_factor * mean:
                self.n_skipped += 1
                return True
        return False

    def observe(self, loss: float) -> float:
        """Feed a *good* (non-skipped) step's loss. Updates #2 and #3, returns lr_mult.

        Only good steps reach here - skipped steps neither pollute the spike window nor move
        the LR, which is what makes the guard and the controllers consistent with each other.
        """
        self.recent.append(loss)

        # EMA-smoothed loss so plateau/online logic reacts to the trend, not per-window noise
        self.ema = loss if self.ema is None else self.ema_beta * self.ema + (1 - self.ema_beta) * loss

        # #3 online adaptation: sign of the trend nudges the LR
        prev_best = self.best_ema
        if self.ema < prev_best - self.plateau_min_delta:
            self.lr_mult *= self.online_up
            self.best_ema = self.ema
            self.plateau_counter = 0
        else:
            self.lr_mult *= self.online_down
            self.plateau_counter += 1
            # #2 plateau rescue: sustained non-improvement -> hard cut
            if self.plateau_counter >= self.plateau_patience:
                self.lr_mult *= self.plateau_factor
                self.plateau_counter = 0
                self.n_plateau_cuts += 1

        self.lr_mult = max(self.lr_mult_min, min(self.lr_mult_max, self.lr_mult))
        return self.lr_mult

    def state(self) -> dict:
        return {
            "lr_mult": round(self.lr_mult, 4),
            "n_skipped": self.n_skipped,
            "n_plateau_cuts": self.n_plateau_cuts,
            "ema_loss": round(self.ema, 4) if self.ema is not None else None,
        }
