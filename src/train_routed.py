"""Train MoTRoutedModel on synthetic multi-domain documents (spec §2.2).

Reports two things the single-domain runs couldn't measure:

  - switch accuracy: how often the model correctly predicts a domain switch at the
    position where one actually occurs. This is the direct test of spec §2.2's claim
    that self-emitted domain tokens can replace an external router.

  - boundary-adjacent loss (spec §10): loss on the N tokens immediately after a switch,
    compared to loss everywhere else. If the domain embedding spaces are incompatible in
    the shared backbone, there should be a visible jump right after a switch.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.data.build_examples import TokenizerBundle
from src.data.encode_routed import encode_routed_doc, load_routed_docs
from src.model.mot_routed_model import MoTRoutedModel

MAX_SEQ_LEN = 512
EPOCHS = 4
LR = 3e-4
LOG_EVERY = 500
BOUNDARY_WINDOW = 8  # tokens after a switch counted as "boundary-adjacent" (spec §10 says 5-10)
# Measured on the real corpus: switches are ~1 in 461 positions. switch_weight=1.0 left
# switch_accuracy at 0.000 across 4 full epochs - unweighted cross-entropy gives that
# class almost no gradient pull. 50x upweighting (not the full 461x inverse-frequency
# value, to avoid destabilizing the much more common regular-token loss).
SWITCH_WEIGHT = 50.0


def evaluate(model, bundle, domain_index, docs, domains) -> dict:
    """Returns overall loss plus the two switch-specific metrics."""
    model.eval()
    total_loss, total_n = 0.0, 0
    boundary_loss, boundary_n = 0.0, 0
    interior_loss, interior_n = 0.0, 0
    switch_correct, switch_total = 0, 0

    with torch.no_grad():
        for doc in docs:
            ex = encode_routed_doc(bundle, domain_index, doc, MAX_SEQ_LEN)
            if ex is None:
                continue
            out = model(
                ex["token_ids"].unsqueeze(0), ex["domain_ids"].unsqueeze(0),
                ex["is_control"].unsqueeze(0), type_ids=ex["type_ids"].unsqueeze(0),
            )
            n = ex["targets"].shape[0]
            per_pos_loss = torch.zeros(n)
            per_pos_pred = torch.zeros(n, dtype=torch.long)
            flat_domain = ex["domain_ids"]
            for domain, (mask, logits) in out.items():
                pos = mask.squeeze(0).nonzero().flatten()
                tgt = ex["targets"][pos]
                per_pos_loss[pos] = F.cross_entropy(logits, tgt, reduction="none")
                per_pos_pred[pos] = logits.argmax(dim=-1)

            total_loss += per_pos_loss.sum().item()
            total_n += n

            # switch positions: input index i whose target is a control token at i+1
            control_pos = ex["is_control"].nonzero().flatten().tolist()
            switch_inputs = [p - 1 for p in control_pos if p > 0]
            for i in switch_inputs:
                if i >= n:
                    continue
                switch_total += 1
                if per_pos_pred[i].item() == ex["targets"][i].item():
                    switch_correct += 1

            boundary_mask = torch.zeros(n, dtype=torch.bool)
            for p in control_pos:
                boundary_mask[p : min(p + BOUNDARY_WINDOW, n)] = True
            boundary_loss += per_pos_loss[boundary_mask].sum().item()
            boundary_n += int(boundary_mask.sum().item())
            interior_loss += per_pos_loss[~boundary_mask].sum().item()
            interior_n += int((~boundary_mask).sum().item())

    return {
        "loss": total_loss / max(total_n, 1),
        "boundary_loss": boundary_loss / max(boundary_n, 1),
        "interior_loss": interior_loss / max(interior_n, 1),
        "switch_accuracy": switch_correct / max(switch_total, 1),
        "switch_total": switch_total,
    }


def main() -> None:
    torch.manual_seed(0)
    bundle = TokenizerBundle()
    domains = list(bundle.domain_vocab_sizes)
    domain_index = {d: i for i, d in enumerate(domains)}

    train_docs = load_routed_docs("data/corpus/multidomain_train.txt")
    val_docs = load_routed_docs("data/corpus/multidomain_val.txt")
    print(f"multi-domain train docs: {len(train_docs)}  val docs: {len(val_docs)}")

    model = MoTRoutedModel(
        domain_vocab_sizes=bundle.domain_vocab_sizes,
        emb_dim=64, d_model=128, n_heads=4, ffn_dim=256, n_layers=2, max_seq_len=MAX_SEQ_LEN,
    )
    print(f"MoTRoutedModel params: {model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        rng = random.Random(epoch)
        docs = train_docs[:]
        rng.shuffle(docs)
        running, n_seen = 0.0, 0
        for i, doc in enumerate(docs, 1):
            ex = encode_routed_doc(bundle, domain_index, doc, MAX_SEQ_LEN)
            if ex is None:
                continue
            loss, _ = model(
                ex["token_ids"].unsqueeze(0), ex["domain_ids"].unsqueeze(0),
                ex["is_control"].unsqueeze(0), targets=ex["targets"].unsqueeze(0),
                type_ids=ex["type_ids"].unsqueeze(0), switch_weight=SWITCH_WEIGHT,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_seen += 1
            if i % LOG_EVERY == 0:
                print(f"    step {i}/{len(docs)}  running avg loss={running/n_seen:.4f}", flush=True)

        m = evaluate(model, bundle, domain_index, val_docs, domains)
        print(
            f"  epoch {epoch+1}/{EPOCHS}  train_loss={running/max(n_seen,1):.4f}  "
            f"val_loss={m['loss']:.4f}  switch_acc={m['switch_accuracy']:.3f} "
            f"({m['switch_total']} switches)  boundary_loss={m['boundary_loss']:.4f} "
            f"interior_loss={m['interior_loss']:.4f}  elapsed={time.time()-t0:.0f}s",
            flush=True,
        )

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "domain_vocab_sizes": bundle.domain_vocab_sizes},
        "checkpoints/mot_routed.pt",
    )
    print("\nsaved checkpoints/mot_routed.pt")


if __name__ == "__main__":
    main()
