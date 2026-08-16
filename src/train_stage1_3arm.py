"""Three-way parity comparison: MoT vs. our own trained unified BPE baseline vs.
tiktoken cl100k_base (a real SOTA production tokenizer) - same backbone config on all
three (2 layers, 4 heads, d_model=128, ffn_dim=256), so the only thing that differs is
the tokenizer/embedding/head layer. cl100k_base's real vocab (100,277) is used as-is,
which makes that arm's embedding+head far larger than the other two - that's the honest
cost of testing against a real SOTA tokenizer, not something to shrink away.

Separate script from train_stage1.py (not an edit to it) so it can be run independently
without touching the still-running MoT-vs-baseline job.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.data.build_examples import TokenizerBundle, load_tagged_docs
from src.model.baseline_model import BaselineModel
from src.model.mot_model import MoTModel

MAX_SEQ_LEN = 256
MAX_TRAIN_DOCS = 15000
MAX_VAL_DOCS = 1500
EPOCHS = 8
LR = 3e-4
LOG_EVERY = 1000

BACKBONE_CFG = dict(d_model=128, n_heads=4, ffn_dim=256, n_layers=2, max_seq_len=MAX_SEQ_LEN)


def run_epoch(model_step, docs, train: bool, seed: int) -> tuple[float, int, dict[str, list[float]]]:
    rng = random.Random(seed)
    docs = docs[:]
    rng.shuffle(docs)
    total_loss, n = 0.0, 0
    by_domain: dict[str, list[float]] = {}
    for i, (domain, text) in enumerate(docs):
        loss = model_step(domain, text, train)
        if loss is None:
            continue
        total_loss += loss
        n += 1
        by_domain.setdefault(domain, []).append(loss)
        if train and (i + 1) % LOG_EVERY == 0:
            print(f"    step {i + 1}/{len(docs)}  running avg loss={total_loss / n:.4f}", flush=True)
    return total_loss / max(n, 1), n, by_domain


def make_encoder_step(model, opt, encode_fn):
    def step(domain: str, text: str, train: bool) -> float | None:
        ids = encode_fn(text, MAX_SEQ_LEN)
        if ids.shape[0] < 2:
            return None
        inp, tgt = ids[:-1].unsqueeze(0), ids[1:]
        model.train(train)
        with torch.set_grad_enabled(train):
            logits = model(inp)
            loss = F.cross_entropy(logits.squeeze(0), tgt)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

    return step


def main() -> None:
    torch.manual_seed(0)
    bundle = TokenizerBundle()
    print("domain vocab sizes:", bundle.domain_vocab_sizes)
    print("baseline (our BPE) vocab size:", bundle.baseline_vocab_size)
    print("sota (cl100k_base) vocab size:", bundle.sota_vocab_size)

    train_docs = load_tagged_docs("data/corpus/train.txt")[:MAX_TRAIN_DOCS]
    val_docs = load_tagged_docs("data/corpus/val.txt")[:MAX_VAL_DOCS]
    print(f"train docs: {len(train_docs)}  val docs: {len(val_docs)}")

    mot = MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, emb_dim=64, **BACKBONE_CFG)
    baseline = BaselineModel(vocab_size=bundle.baseline_vocab_size, **BACKBONE_CFG)
    sota = BaselineModel(vocab_size=bundle.sota_vocab_size, **BACKBONE_CFG)
    print(f"MoT params: {mot.num_params():,}")
    print(f"Baseline (our BPE) params: {baseline.num_params():,}")
    print(f"SOTA (cl100k_base) params: {sota.num_params():,}")

    mot_opt = torch.optim.AdamW(mot.parameters(), lr=LR)
    base_opt = torch.optim.AdamW(baseline.parameters(), lr=LR)
    sota_opt = torch.optim.AdamW(sota.parameters(), lr=LR)

    routed_domains_seen = set()

    def mot_step(domain: str, text: str, train: bool) -> float | None:
        ids, types = bundle.encode_domain(domain, text, MAX_SEQ_LEN)
        if ids.shape[0] < 2:
            return None
        vocab_size = bundle.domain_vocab_sizes[domain]
        assert ids.max().item() < vocab_size, f"{domain} token id out of range for its own table"
        routed_domains_seen.add(domain)

        inp, tgt = ids[:-1].unsqueeze(0), ids[1:]
        inp_types = types[:-1].unsqueeze(0) if types is not None else None
        mot.train(train)
        with torch.set_grad_enabled(train):
            logits = mot(domain, inp, inp_types)
            loss = F.cross_entropy(logits.squeeze(0), tgt)
        if train:
            mot_opt.zero_grad()
            loss.backward()
            mot_opt.step()
        return loss.item()

    baseline_step = make_encoder_step(baseline, base_opt, bundle.encode_baseline)
    sota_step = make_encoder_step(sota, sota_opt, bundle.encode_sota)

    arms = (("MoT", mot_step), ("Baseline (our BPE)", baseline_step), ("SOTA (cl100k_base)", sota_step))
    for tag, model_step in arms:
        print(f"\n=== training {tag} ===")
        t0 = time.time()
        for epoch in range(EPOCHS):
            train_loss, n_train, _ = run_epoch(model_step, train_docs, train=True, seed=epoch)
            val_loss, n_val, val_by_domain = run_epoch(model_step, val_docs, train=False, seed=1000 + epoch)
            print(
                f"  epoch {epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f} (n={n_train})  "
                f"val_loss={val_loss:.4f} (n={n_val})  elapsed={time.time() - t0:.0f}s"
            )
            if tag == "MoT":
                per_domain = {d: sum(v) / len(v) for d, v in val_by_domain.items()}
                print(f"    val loss by domain: {per_domain}")

    print(f"\nrouted domains seen during MoT training: {sorted(routed_domains_seen)}")
    assert routed_domains_seen == set(bundle.domain_vocab_sizes), "not all domains were routed through during training"
    print("sanity check: every domain's table was exercised - OK")


if __name__ == "__main__":
    main()
