"""Single source of truth for stage-2 metrics.

Two layers:

  1. Pure metric math (no torch, no data) - bits_per_byte, perplexity, bits_per_token. These
     are the only definitions of each metric; every eval path (stage2_modal.py's evaluate /
     evaluate_lambada, the pod script, ad-hoc checks) should convert its accumulated nats +
     token/byte counts through THESE, so a metric can never mean two different things in two
     places (which is how the pre-2026-08-16 per-token-loss vs BPB confusion happened).

  2. A results store (results/metrics.json) - the canonical record of every arm's evaluated
     numbers, keyed by (arm, step). The dashboard generator reads ONLY this file, so "refresh
     the dashboard" means "regenerate from the store", never "hand-transcribe numbers into
     HTML". Eval runs append here via record().

Metric definitions (all lower = better):
  - BPB (bits per byte): total_nats / ln2 / total_bytes. Tokenizer-invariant - the only valid
    cross-arm compression comparison, since arms tokenize the same text into different token
    counts. Bytes are the shared denominator.
  - target-token perplexity (LAMBADA): exp(total_nats / total_target_tokens). Not byte-
    normalized because LAMBADA's unit is "one target word" of fixed identity across arms.
  - switch accuracy: fraction of switch-target positions whose argmax hit the true next
    domain. Chance ~= 1/num_domains.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

LN2 = math.log(2)
RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "metrics.json"


# --- layer 1: pure metric math --------------------------------------------------------------

def bits_per_byte(total_nats: float, total_bytes: float) -> float:
    return (total_nats / LN2) / max(total_bytes, 1)


def perplexity(total_nats: float, total_tokens: float) -> float:
    return math.exp(total_nats / max(total_tokens, 1))


def bits_per_token(total_nats: float, total_tokens: float) -> float:
    return (total_nats / LN2) / max(total_tokens, 1)


def bytes_per_token(total_bytes: float, total_tokens: float) -> float:
    """Measured on held-out text so BPB can convert per-token nats to per-byte without a
    lossy decode-back-to-bytes (the nlp hybrid tokenizer's decode is lossy)."""
    return total_bytes / max(total_tokens, 1)


# --- layer 2: the results store -------------------------------------------------------------

@dataclass
class ArmResult:
    arm: str
    step: int
    params: int | None = None
    bpb_single: float | None = None          # single-domain held-out BPB
    bpb_cross: float | None = None           # cross-domain held-out BPB (switching arms only)
    switch_accuracy: float | None = None     # switching arms only
    lambada_ppl: float | None = None         # target-token perplexity
    lambada_accuracy: float | None = None    # exact-match (near-0 at this scale, kept for record)
    per_domain_bpb: dict = field(default_factory=dict)  # {"code":..,"math":..,...}
    scale: str = "base"                      # "base" | "large"
    note: str = ""


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def record(result: ArmResult) -> None:
    """Upsert one arm's result, keyed 'arm@step' (or 'arm-large@step' for the scale runs) so a
    later checkpoint of the same arm doesn't clobber an earlier one - we want the trajectory."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = load_results()
    key = f"{result.arm}{'-large' if result.scale == 'large' else ''}@{result.step}"
    store[key] = {k: v for k, v in asdict(result).items() if v not in (None, {}, "")}
    RESULTS_PATH.write_text(json.dumps(store, indent=2, sort_keys=True))


def latest_per_arm(scale: str | None = None) -> dict:
    """Newest evaluated result per arm (highest step). scale=None returns both base and large."""
    store = load_results()
    best: dict[str, dict] = {}
    for row in store.values():
        if scale is not None and row.get("scale", "base") != scale:
            continue
        name = row["arm"] + ("-large" if row.get("scale") == "large" else "")
        if name not in best or row["step"] > best[name]["step"]:
            best[name] = row
    return best
