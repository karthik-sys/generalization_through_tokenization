"""MoT-only training run, purely to produce a checkpoint for intersection-vocab mining
(spec §4) - the full v2 run (MoT val_loss=4.5438 final) never saved its weights, so
there's nothing to mine from yet. Skips training the baseline (not needed for mining)
to get a usable checkpoint faster. 4 epochs, not the full 8 - enough for real, non-random
embedding structure to mine from; can be re-run against a better checkpoint later
(spec explicitly frames intersection vocab as "a living artifact, not fixed at v1").
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.data.build_examples import TokenizerBundle, load_tagged_docs
from src.model.mot_model import MoTModel
from src.train_stage1 import LR, MAX_SEQ_LEN, MAX_TRAIN_DOCS, MAX_VAL_DOCS, run_epoch

CHECKPOINT_EPOCHS = 4


def main() -> None:
    torch.manual_seed(0)
    bundle = TokenizerBundle()
    print("domain vocab sizes:", bundle.domain_vocab_sizes)

    train_docs = load_tagged_docs("data/corpus/train.txt")[:MAX_TRAIN_DOCS]
    val_docs = load_tagged_docs("data/corpus/val.txt")[:MAX_VAL_DOCS]
    print(f"train docs: {len(train_docs)}  val docs: {len(val_docs)}")

    mot = MoTModel(
        domain_vocab_sizes=bundle.domain_vocab_sizes,
        emb_dim=64, d_model=128, n_heads=4, ffn_dim=256, n_layers=2, max_seq_len=MAX_SEQ_LEN,
    )
    print(f"MoT params: {mot.num_params():,}")
    mot_opt = torch.optim.AdamW(mot.parameters(), lr=LR)

    def mot_step(domain: str, text: str, train: bool):
        ids, types = bundle.encode_domain(domain, text, MAX_SEQ_LEN)
        if ids.shape[0] < 2:
            return None
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

    print(f"\n=== training MoT ({CHECKPOINT_EPOCHS} epochs, checkpoint-only run) ===")
    t0 = time.time()
    for epoch in range(CHECKPOINT_EPOCHS):
        train_loss, n_train, _ = run_epoch(mot_step, train_docs, train=True, seed=epoch)
        val_loss, n_val, _ = run_epoch(mot_step, val_docs, train=False, seed=1000 + epoch)
        print(
            f"  epoch {epoch + 1}/{CHECKPOINT_EPOCHS}  train_loss={train_loss:.4f} (n={n_train})  "
            f"val_loss={val_loss:.4f} (n={n_val})  elapsed={time.time() - t0:.0f}s"
        )

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        {"model": mot.state_dict(), "domain_vocab_sizes": bundle.domain_vocab_sizes},
        "checkpoints/mot_stage1.pt",
    )
    print("\nsaved checkpoints/mot_stage1.pt")


if __name__ == "__main__":
    main()
