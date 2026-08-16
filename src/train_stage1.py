"""Stage-1 CPU run. v1 (4000 docs/2 epochs, tiny per-domain slices) was a pure
mechanical sanity check - spec §8 doesn't expect it to show the real hybrid-vs-BPE
signal. v2 scales the corpus up substantially (15k real-world docs/domain, math's
real-world starvation fixed - see docs/dataset_methodology.md) and trains longer, to
get a more reliable MoT-vs-baseline comparison than pure noise, while being upfront
that this is still a CPU run at ~5M params, not spec §8's actual stage-2 GPU ablation.
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


def main() -> None:
    torch.manual_seed(0)
    bundle = TokenizerBundle()
    print("domain vocab sizes:", bundle.domain_vocab_sizes)
    print("baseline vocab size:", bundle.baseline_vocab_size)

    for domain, size in bundle.domain_vocab_sizes.items():
        assert size > 0, f"{domain} has empty vocab"
    print("sanity check: all per-domain tables have nonzero vocab size - OK")

    train_docs = load_tagged_docs("data/corpus/train.txt")[:MAX_TRAIN_DOCS]
    val_docs = load_tagged_docs("data/corpus/val.txt")[:MAX_VAL_DOCS]
    print(f"train docs: {len(train_docs)}  val docs: {len(val_docs)}")

    mot = MoTModel(
        domain_vocab_sizes=bundle.domain_vocab_sizes,
        emb_dim=64,
        d_model=128,
        n_heads=4,
        ffn_dim=256,
        n_layers=2,
        max_seq_len=MAX_SEQ_LEN,
    )
    baseline = BaselineModel(
        vocab_size=bundle.baseline_vocab_size,
        d_model=128,
        n_heads=4,
        ffn_dim=256,
        n_layers=2,
        max_seq_len=MAX_SEQ_LEN,
    )
    print(f"MoT params: {mot.num_params():,}   Baseline params: {baseline.num_params():,}")

    mot_opt = torch.optim.AdamW(mot.parameters(), lr=LR)
    base_opt = torch.optim.AdamW(baseline.parameters(), lr=LR)

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

    def baseline_step(domain: str, text: str, train: bool) -> float | None:
        ids = bundle.encode_baseline(text, MAX_SEQ_LEN)
        if ids.shape[0] < 2:
            return None
        inp, tgt = ids[:-1].unsqueeze(0), ids[1:]
        baseline.train(train)
        with torch.set_grad_enabled(train):
            logits = baseline(inp)
            loss = F.cross_entropy(logits.squeeze(0), tgt)
        if train:
            base_opt.zero_grad()
            loss.backward()
            base_opt.step()
        return loss.item()

    for tag, model_step, model in (("MoT", mot_step, mot), ("Baseline", baseline_step, baseline)):
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

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        {"model": mot.state_dict(), "domain_vocab_sizes": bundle.domain_vocab_sizes},
        "checkpoints/mot_stage1.pt",
    )
    print("\nsaved checkpoints/mot_stage1.pt")

    print(f"\nrouted domains seen during MoT training: {sorted(routed_domains_seen)}")
    assert routed_domains_seen == set(bundle.domain_vocab_sizes), "not all domains were routed through during training"
    print("sanity check: every domain's table was exercised - OK")


if __name__ == "__main__":
    main()
