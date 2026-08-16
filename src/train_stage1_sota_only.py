"""SOTA-tokenizer arm only (cl100k_base) - MoT and our-BPE baseline results already
exist from the v2 run (MoT val_loss=4.5438, baseline val_loss=4.6725, both epoch 8/8),
so there's no need to retrain them. Same corpus, same backbone config, same seeds as
that run, for a direct apples-to-apples comparison.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.data.build_examples import TokenizerBundle, load_tagged_docs
from src.model.baseline_model import BaselineModel
from src.train_stage1 import MAX_SEQ_LEN, MAX_TRAIN_DOCS, MAX_VAL_DOCS, EPOCHS, LR, LOG_EVERY, run_epoch


def main() -> None:
    torch.manual_seed(0)
    bundle = TokenizerBundle()
    print("sota (cl100k_base) vocab size:", bundle.sota_vocab_size)

    train_docs = load_tagged_docs("data/corpus/train.txt")[:MAX_TRAIN_DOCS]
    val_docs = load_tagged_docs("data/corpus/val.txt")[:MAX_VAL_DOCS]
    print(f"train docs: {len(train_docs)}  val docs: {len(val_docs)}")

    sota = BaselineModel(
        vocab_size=bundle.sota_vocab_size, d_model=128, n_heads=4, ffn_dim=256, n_layers=2, max_seq_len=MAX_SEQ_LEN,
    )
    print(f"SOTA (cl100k_base) params: {sota.num_params():,}  "
          f"(vs. MoT 8,103,244 / our-BPE baseline 4,409,984 from the v2 run)")
    sota_opt = torch.optim.AdamW(sota.parameters(), lr=LR)

    def sota_step(domain: str, text: str, train: bool) -> float | None:
        ids = bundle.encode_sota(text, MAX_SEQ_LEN)
        if ids.shape[0] < 2:
            return None
        inp, tgt = ids[:-1].unsqueeze(0), ids[1:]
        sota.train(train)
        with torch.set_grad_enabled(train):
            logits = sota(inp)
            loss = F.cross_entropy(logits.squeeze(0), tgt)
        if train:
            sota_opt.zero_grad()
            loss.backward()
            sota_opt.step()
        return loss.item()

    print("\n=== training SOTA (cl100k_base) ===")
    t0 = time.time()
    for epoch in range(EPOCHS):
        train_loss, n_train, _ = run_epoch(sota_step, train_docs, train=True, seed=epoch)
        val_loss, n_val, _ = run_epoch(sota_step, val_docs, train=False, seed=1000 + epoch)
        print(
            f"  epoch {epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f} (n={n_train})  "
            f"val_loss={val_loss:.4f} (n={n_val})  elapsed={time.time() - t0:.0f}s"
        )

    print("\n=== final 3-way comparison (val loss, epoch 8/8) ===")
    print("MoT:                4.5438")
    print("Baseline (our BPE): 4.6725")
    print(f"SOTA (cl100k_base): {val_loss:.4f}")


if __name__ == "__main__":
    main()
