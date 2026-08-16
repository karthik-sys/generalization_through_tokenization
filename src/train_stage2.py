"""Stage-2 GPU training (spec §8 stage 2): the actual ablation, not the CPU sanity
check. Requires a real GPU - built to run on Colab/Kaggle/Lightning AI's free tiers
(see stage2_colab.ipynb), not this local CPU-only machine.

Trains one arm per invocation (MoT / baseline / sota) so you can run them as separate
Colab sessions if your free-tier time budget doesn't cover all three back to back.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.build_examples import TokenizerBundle
from src.data.stage2_stream_dataset import PackedDomainStream, PackedMixedStream
from src.model.baseline_model import BaselineModel
from src.model.mot_model import MoTModel
from src.model.stage2_config import (
    BACKBONE_ONLY_CFG,
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    GRAD_ACCUM_STEPS,
    LOG_EVERY,
    LR,
    MAX_STEPS,
    MODEL_CFG,
    WARMUP_STEPS,
)


def lr_schedule(step: int) -> float:
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(model, opt, step: int, tag: str, out_dir: str = "checkpoints") -> None:
    Path(out_dir).mkdir(exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
        f"{out_dir}/{tag}_step{step}.pt",
    )
    print(f"checkpoint saved: {out_dir}/{tag}_step{step}.pt", flush=True)


def train_mot(bundle: TokenizerBundle, device: str) -> None:
    model = MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    print(f"MoT params: {model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    domains = list(bundle.domain_vocab_sizes)
    loaders = {
        d: iter(DataLoader(
            PackedDomainStream(d, bundle.encode_domain, MODEL_CFG["max_seq_len"]),
            batch_size=BATCH_SIZE,
        ))
        for d in domains
    }

    t0 = time.time()
    running_loss, running_n = 0.0, 0
    for step in range(1, MAX_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = LR * lr_schedule(step)

        domain = domains[step % len(domains)]
        batch = next(loaders[domain])
        domain_b, ids, types = batch
        ids = ids.to(device)
        types = types.to(device)  # always a tensor now; MoTModel ignores it for non-type-conditioned domains
        inp, tgt = ids[:, :-1], ids[:, 1:]
        inp_types = types[:, :-1] if types is not None else None

        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits = model(domain, inp, inp_types)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        running_loss += loss.item() * GRAD_ACCUM_STEPS
        running_n += 1

        if step % GRAD_ACCUM_STEPS == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        if step % LOG_EVERY == 0:
            print(f"step {step}/{MAX_STEPS}  domain={domain}  loss={running_loss/running_n:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
            running_loss, running_n = 0.0, 0

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, opt, step, "mot")


def train_single_vocab(tag: str, vocab_size: int, encode_fn, device: str) -> None:
    model = BaselineModel(vocab_size=vocab_size, **BACKBONE_ONLY_CFG).to(device)
    print(f"{tag} params: {model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    loader = iter(DataLoader(
        PackedMixedStream(encode_fn, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE,
    ))

    t0 = time.time()
    running_loss, running_n = 0.0, 0
    for step in range(1, MAX_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = LR * lr_schedule(step)

        ids = next(loader).to(device)
        inp, tgt = ids[:, :-1], ids[:, 1:]

        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        running_loss += loss.item() * GRAD_ACCUM_STEPS
        running_n += 1

        if step % GRAD_ACCUM_STEPS == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        if step % LOG_EVERY == 0:
            print(f"step {step}/{MAX_STEPS}  loss={running_loss/running_n:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
            running_loss, running_n = 0.0, 0

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, opt, step, tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["mot", "baseline", "sota"])
    parser.add_argument("--tokenizer-dir", default="tokenizers_stage2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. This script is built for stage-2 GPU scale - "
              "running on CPU will be extremely slow and defeats the point of stage 2.")

    bundle = TokenizerBundle(tokenizer_dir=args.tokenizer_dir)

    if args.arm == "mot":
        train_mot(bundle, device)
    elif args.arm == "baseline":
        train_single_vocab("baseline", bundle.baseline_vocab_size, bundle.encode_baseline, device)
    elif args.arm == "sota":
        train_single_vocab("sota", bundle.sota_vocab_size, bundle.encode_sota, device)
