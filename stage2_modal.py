"""Stage-2 GPU training on Modal (modal.com). Real alternative to the Colab notebook -
same underlying pipeline (src/train_stage2.py's logic), just wrapped as a Modal app so
it can be driven entirely from the CLI/Bash instead of a browser.

Usage:
  modal run stage2_modal.py --step sample-tokenizers   # pull tokenizer training samples (CPU)
  modal run stage2_modal.py --step train-tokenizers    # train stage-2 tokenizers (CPU)
  modal run stage2_modal.py --step calibrate --arm mot --steps 150   # short GPU timing run
  modal run stage2_modal.py --step train --arm mot                  # full run (uses MAX_STEPS from stage2_config)
"""

from __future__ import annotations

import time

import modal

app = modal.App("mot-stage2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.19",
        "tokenizers>=0.19",
        "sentencepiece>=0.2",
        "morfessor>=2.0",
        "pyphen>=0.14",
        "torch>=2.2",
        "numpy>=1.26",
        "huggingface_hub>=0.23",
        "tiktoken>=0.7",
        "zstandard>=0.22",
    )
    .add_local_dir(
        ".", remote_path="/root/repo",
        ignore=["data", "tokenizers", "tokenizers_stage2", "checkpoints", ".git", "__pycache__", "*.pyc"],
    )
)

volume = modal.Volume.from_name("mot-stage2-data", create_if_missing=True)
VOLUME_PATH = "/vol"


def _setup_paths():
    import sys

    sys.path.insert(0, "/root/repo")


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-token")])
def sample_tokenizers():
    _setup_paths()
    import os

    os.chdir("/root/repo")
    from datasets import Dataset, load_dataset

    from src.data.stage2_sample_for_tokenizers import N_SAMPLE, sample_domain
    from src.model.stage2_config import STREAM_SOURCES

    for domain in STREAM_SOURCES:
        if STREAM_SOURCES[domain].get("gated"):
            # the-stack-v2-dedup needs an accepted-terms HF token (not wired up yet) -
            # stand in with the-stack-smol (same as the local stress test) so the
            # pipeline runs end to end. Swap back once a Modal secret w/ HF_TOKEN is set.
            print(f"{domain}: real source ({STREAM_SOURCES[domain]['path']}) is gated - "
                  f"using bigcode/the-stack-smol as a stand-in for now")
            # newer `datasets` lib on Modal only exposes the "default" config (not the
            # older per-language "python" config) - filter by the lang column instead
            ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
            rows = []
            for row in ds:
                if row.get("lang", "").lower() == "python":
                    rows.append({"text": row["content"]})
                if len(rows) >= N_SAMPLE:
                    break
            Dataset.from_list(rows).save_to_disk(f"{VOLUME_PATH}/stage2_tokenizer_sample/{domain}")
            continue
        sample_domain(domain, out_dir=f"{VOLUME_PATH}/stage2_tokenizer_sample")
    volume.commit()


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=3600)
def train_tokenizers():
    _setup_paths()
    import os

    os.chdir("/root/repo")
    from src.tokenizers.train_all_stage2 import main as train_all_stage2_main

    train_all_stage2_main(
        out_dir=f"{VOLUME_PATH}/tokenizers_stage2",
        sample_dir=f"{VOLUME_PATH}/stage2_tokenizer_sample",
    )
    volume.commit()


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=1800,
              secrets=[modal.Secret.from_name("huggingface-token")])
def calibrate(arm: str = "mot", steps: int = 150):
    _setup_paths()
    import os

    os.chdir("/root/repo")
    import torch

    from src.data.build_examples import TokenizerBundle
    from src.data.stage2_routed_stream import PackedRoutedStream
    from src.data.stage2_stream_dataset import PackedDomainStream, PackedMixedStream
    from src.model.baseline_model import BaselineModel
    from src.model.mot_hybrid_model import MoTHybridModel
    from src.model.mot_model import MoTModel
    from src.model.mot_pooled2_model import MoTPooled2Model
    from src.model.mot_pooled_model import MoTPooledModel
    from src.model.mot_routed_model import MoTRoutedModel
    from src.model.stage2_config import (
        BACKBONE_ONLY_CFG, BATCH_SIZE, CONFIDENCE_WEIGHT, FOCAL_GAMMA, HYBRID_NATURAL_DATA_FRACTION,
        MODEL_CFG, SWITCH_WEIGHT,
    )
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    device = "cuda"
    print(f"CUDA available: {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(0)}")

    bundle = TokenizerBundle(tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2")
    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}

    if arm == "mot":
        model = MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "routed":
        model = MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled":
        model = MoTPooledModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                               focal_gamma=FOCAL_GAMMA, confidence_weight=CONFIDENCE_WEIGHT).to(device)
    elif arm == "hybrid":
        model = MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled2":
        model = MoTPooled2Model(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                focal_gamma=FOCAL_GAMMA).to(device)
    elif arm == "baseline":
        model = BaselineModel(vocab_size=bundle.baseline_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    else:
        model = BaselineModel(vocab_size=bundle.sota_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    print(f"{arm} params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    if arm in ("mot", "hybrid"):
        domains = list(bundle.domain_vocab_sizes)
        loaders = {
            d: iter(DataLoader(PackedDomainStream(d, bundle.encode_domain, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
            for d in domains
        }
    if arm in ("routed", "pooled", "pooled2", "hybrid"):
        loader = iter(DataLoader(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm not in ("mot",):
        encode_fn = bundle.encode_baseline if arm == "baseline" else bundle.encode_sota
        loader = iter(DataLoader(PackedMixedStream(encode_fn, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))

    def _hybrid_calib_batch(step: int):
        import random
        use_natural = random.Random(step).random() < HYBRID_NATURAL_DATA_FRACTION
        if use_natural:
            domain = domains[step % len(domains)]
            _, ids, types = next(loaders[domain])
            ids, types = ids.to(device), types.to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            typ = types[:, :-1]
            dom = torch.full_like(inp, domain_index[domain])
            ctrl = torch.zeros_like(inp)
            return inp, dom, ctrl, typ, tgt
        tok, dom, ctrl, typ, tgt = next(loader)
        return (t.to(device) for t in (tok, dom, ctrl, typ, tgt))

    t0 = time.time()
    for step in range(1, steps + 1):
        if arm == "mot":
            domain = domains[step % len(domains)]
            _, ids, types = next(loaders[domain])
            ids = ids.to(device)
            types = types.to(device)  # always a tensor now; MoTModel ignores it for non-type-conditioned domains
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda"):
                logits = model(domain, inp, types[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        elif arm == "hybrid":
            tok, dom, ctrl, typ, tgt = _hybrid_calib_batch(step)
            with torch.autocast("cuda"):
                loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
                if step == 1:
                    print(f"  loss parts: content={parts['_content']:.4f} switch={parts['_switch']:.4f}")
        elif arm in ("routed", "pooled", "pooled2"):
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
            with torch.autocast("cuda"):
                if arm in ("pooled", "pooled2"):
                    # calibrate at full adversarial strength: lambda=1 is the worst case for
                    # both speed and memory, so timing here bounds the real run rather than
                    # measuring the cheap early-ramp steps.
                    loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ,
                                        switch_weight=SWITCH_WEIGHT, adv_lambda=1.0)
                    if step == 1:
                        print(f"  loss parts: load_balance={parts['_load_balance']:.4f} "
                              f"adversarial={parts['_adversarial']:.4f}")
                else:
                    loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
        else:
            ids = next(loader).to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda"):
                logits = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

        # autocast here matches train() - without it, calibration measures fp32 and
        # under-reports throughput while over-reporting memory vs. the real run.
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1:
            print(f"peak GPU mem after first step: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        if step % 25 == 0:
            elapsed = time.time() - t0
            print(f"step {step}/{steps}  loss={loss.item():.4f}  elapsed={elapsed:.1f}s  "
                  f"sec/step={elapsed/step:.3f}")

    elapsed = time.time() - t0
    sec_per_step = elapsed / steps
    print(f"\nCALIBRATION RESULT: {sec_per_step:.3f} sec/step on {torch.cuda.get_device_name(0)}")
    print(f"Extrapolated to MAX_STEPS from stage2_config: see caller for cost estimate.")
    return sec_per_step


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=24 * 3600,
              secrets=[modal.Secret.from_name("huggingface-token")])
def train(arm: str = "mot", max_steps: int | None = None, resume_from: str | None = None):
    """Full stage-2 training for one arm. Checkpoints to the Modal Volume so a container
    restart doesn't lose the run - pass resume_from to pick a run back up."""
    _setup_paths()
    import os

    os.chdir("/root/repo")
    import math

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from src.data.build_examples import TokenizerBundle
    from src.data.stage2_routed_stream import PackedRoutedStream
    from src.data.stage2_stream_dataset import PackedDomainStream, PackedMixedStream
    from src.model.baseline_model import BaselineModel
    from src.model.mot_hybrid_model import MoTHybridModel
    from src.model.mot_model import MoTModel
    from src.model.mot_pooled2_model import MoTPooled2Model
    from src.model.mot_pooled_model import MoTPooledModel
    from src.model.mot_routed_model import MoTRoutedModel
    from src.model.stage2_config import (
        ADV_LAMBDA_RAMP_STEPS, ARM_LABELS, BACKBONE_ONLY_CFG, BATCH_SIZE, CHECKPOINT_EVERY,
        CONFIDENCE_WEIGHT, FOCAL_GAMMA, GRAD_ACCUM_STEPS, HYBRID_NATURAL_DATA_FRACTION, LOG_EVERY,
        LR, MAX_STEPS, MODEL_CFG, SWITCH_WEIGHT, WARMUP_STEPS,
    )

    total_steps = max_steps or MAX_STEPS
    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  arm: {arm}  steps: {total_steps}", flush=True)
    print(f"ARM: {arm}  =  {ARM_LABELS.get(arm, arm)}", flush=True)

    bundle = TokenizerBundle(tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2")
    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    if arm == "mot":
        model = MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "routed":
        model = MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled":
        model = MoTPooledModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                               focal_gamma=FOCAL_GAMMA, confidence_weight=CONFIDENCE_WEIGHT).to(device)
    elif arm == "hybrid":
        model = MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled2":
        model = MoTPooled2Model(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                focal_gamma=FOCAL_GAMMA).to(device)
    elif arm == "baseline":
        model = BaselineModel(vocab_size=bundle.baseline_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    else:
        model = BaselineModel(vocab_size=bundle.sota_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    print(f"{arm} params: {model.num_params():,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    ckpt_dir = f"{VOLUME_PATH}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    def _latest_checkpoint() -> str | None:
        """Newest checkpoint for this arm actually present on the volume.

        Load-bearing for preemption safety. Modal restarts a preempted container with the
        *same input*, so a fixed resume_from argument replays from whatever checkpoint was
        named at launch - silently discarding everything since. Observed for real: the routed
        arm was preempted at step ~32000 and restarted from its launch-time step-21000
        checkpoint, throwing away 11k steps. Worse, that failure repeats: if preemptions
        arrive closer together than the run takes to re-cover the lost ground, the arm
        thrashes forever without advancing. Discovering the newest checkpoint at container
        start makes restarts monotonic instead.
        """
        import glob
        import re

        volume.reload()  # pick up commits made by the previous (preempted) container
        paths = glob.glob(f"{ckpt_dir}/{arm}_step*.pt")
        # newest-first, so the caller can fall back down the list on a corrupt file
        return sorted(paths, key=lambda p: int(re.search(r"step(\d+)", p).group(1)), reverse=True)

    start_step = 1
    # An abrupt kill (the Starter-plan concurrency cap terminating a job mid-save) can leave
    # the newest checkpoint truncated - torch.load then raises a miniz/zip error. Observed for
    # real on mot. So walk checkpoints newest-first and use the first one that actually loads,
    # rather than trusting the newest blindly. A partially-written file costs at most the
    # ~1000 steps back to the previous good checkpoint.
    candidates = _latest_checkpoint()
    if not candidates and resume_from:
        candidates = [f"{VOLUME_PATH}/{resume_from}"]
    for cand in candidates:
        try:
            ckpt = torch.load(cand, map_location=device)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            start_step = ckpt["step"] + 1
            print(f"resumed from {cand} at step {start_step}", flush=True)
            break
        except Exception as e:
            print(f"checkpoint {cand} failed to load ({type(e).__name__}: {e}); trying older", flush=True)
            continue

    def lr_at(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    if arm in ("mot", "hybrid"):
        domains = list(bundle.domain_vocab_sizes)
        loaders = {
            d: iter(DataLoader(PackedDomainStream(d, bundle.encode_domain, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
            for d in domains
        }
    if arm in ("routed", "pooled", "pooled2", "hybrid"):
        loader = iter(DataLoader(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm not in ("mot",):
        encode_fn = bundle.encode_baseline if arm == "baseline" else bundle.encode_sota
        loader = iter(DataLoader(PackedMixedStream(encode_fn, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))

    def _hybrid_train_batch(step: int):
        import random
        use_natural = random.Random(step).random() < HYBRID_NATURAL_DATA_FRACTION
        if use_natural:
            domain = domains[step % len(domains)]
            _, ids, types = next(loaders[domain])
            ids, types = ids.to(device), types.to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            typ = types[:, :-1]
            dom = torch.full_like(inp, domain_index[domain])
            ctrl = torch.zeros_like(inp)
            return inp, dom, ctrl, typ, tgt
        tok, dom, ctrl, typ, tgt = next(loader)
        return (t.to(device) for t in (tok, dom, ctrl, typ, tgt))

    # Arms 5+: loss-trajectory controller (spike-skip + plateau rescue + online LR).
    # Kept off mot/baseline/sota/routed so it never confounds the matched-compute comparison
    # for the original 4-way ablation; every new complex-loss arm gets it by default.
    controller = None
    if arm in ("pooled", "pooled2", "hybrid"):
        from src.model.adaptive_optimizer import AdaptiveController
        controller = AdaptiveController()
        print("adaptive controller ON (spike-guard + plateau rescue + online LR)", flush=True)

    t0 = time.time()
    running, running_n = 0.0, 0
    history = []
    for step in range(start_step, total_steps + 1):
        lr_mult = controller.lr_mult if controller is not None else 1.0
        for g in opt.param_groups:
            g["lr"] = LR * lr_at(step) * lr_mult

        if arm == "mot":
            domain = domains[step % len(domains)]
            _, ids, types = next(loaders[domain])
            ids, types = ids.to(device), types.to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda"):
                logits = model(domain, inp, types[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        elif arm == "routed":
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
            with torch.autocast("cuda"):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
        elif arm == "hybrid":
            tok, dom, ctrl, typ, tgt = _hybrid_train_batch(step)
            with torch.autocast("cuda"):
                loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
            controller_loss = parts["_content"]
        elif arm in ("pooled", "pooled2"):
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
            # DANN's adversarial strength ramps 0 -> 1 rather than starting hot: full
            # pressure to hide domain identity before the model can predict anything just
            # starves the main objective. At lambda 0 this arm is the routed arm plus an
            # inert pooling path, so early steps are a free correctness check too.
            adv_lambda = min(1.0, step / max(1, ADV_LAMBDA_RAMP_STEPS))
            with torch.autocast("cuda"):
                loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ,
                                    switch_weight=SWITCH_WEIGHT, adv_lambda=adv_lambda)
            # The controller must watch the MAIN LM loss, not the total. The total includes
            # the adversarial term, which RISES by design as adv_lambda ramps 0->1 over the
            # first ADV_LAMBDA_RAMP_STEPS - so the total looks flat even while the model is
            # learning, and the plateau detector misreads that as a stall and chokes the LR.
            # (First pass did exactly this: 47 plateau cuts drove lr_mult to the 0.1 floor and
            # stalled the arm at ~9.2.) main = mean of the per-domain CE entries in `parts`.
            main_parts = [v for k, v in parts.items() if not k.startswith("_")]
            controller_loss = sum(main_parts) / max(len(main_parts), 1)
        else:
            ids = next(loader).to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda"):
                logits = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

        loss_val = loss.item()
        # controller_loss is the mean of the model's per-domain CE entries, which are already
        # Python floats (per_domain[domain] = float(...) in the model) - so it's a float, not a
        # tensor. The controller is None for arms that don't set it, so this is only consumed
        # for pooled/pooled2/hybrid.
        controller_loss_val = float(controller_loss) if arm in ("pooled", "pooled2", "hybrid") else loss_val
        # #1 spike guard (pooled only): a NaN/Inf or a loss spiking far above the running
        # mean is a bad batch - drop its gradient entirely rather than let one poisoned step
        # corrupt the weights (same instinct as banning a checkpoint that emits NaN). Spike
        # detection watches the TOTAL loss (a NaN anywhere is bad); the plateau/online LR
        # signals watch the MAIN loss (controller_loss_val) so the adversarial ramp doesn't
        # masquerade as a stall.
        if controller is not None and controller.should_skip(loss_val):
            opt.zero_grad()  # discard anything accumulated this micro-batch window
            if step % LOG_EVERY == 0:
                print(f"step {step}/{total_steps}  SKIPPED (loss={loss_val:.3f}, "
                      f"guard tripped)  {controller.state()}", flush=True)
            continue

        scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
        running += loss_val
        running_n += 1
        if step % GRAD_ACCUM_STEPS == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        if controller is not None:
            controller.observe(controller_loss_val)  # plateau/online LR from MAIN loss

        if step % LOG_EVERY == 0:
            avg = running / max(running_n, 1)
            entry = {"step": step, "loss": round(avg, 4), "elapsed": round(time.time() - t0)}
            if controller is not None:
                entry.update(controller.state())
            history.append(entry)
            ctl = f"  {controller.state()}" if controller is not None else ""
            print(f"step {step}/{total_steps}  loss={avg:.4f}  elapsed={time.time()-t0:.0f}s{ctl}", flush=True)
            running, running_n = 0.0, 0

        if step % CHECKPOINT_EVERY == 0 or step == total_steps:
            path = f"{ckpt_dir}/{arm}_step{step}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "domain_vocab_sizes": bundle.domain_vocab_sizes, "history": history}, path)
            volume.commit()
            print(f"checkpoint saved: {path}", flush=True)

    print(f"\nDONE {arm}: {total_steps} steps in {time.time()-t0:.0f}s", flush=True)
    return history


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=1800,
              secrets=[modal.Secret.from_name("huggingface-token")])
def evaluate(arm: str = "mot", checkpoint_step: int = 20000, eval_batches: int = 200, noisy: bool = False):
    """Held-out eval against a saved checkpoint. train()'s streams all start from the
    beginning of each HF source, so a fresh stream would just replay training data -
    this skips well past what 20-200k training steps could plausibly have consumed from
    the large sources (open-web-math, arxiv-abstracts, fineweb: millions of rows each).

    Code is a real exception, not a detail: the-stack-smol (its stand-in source) is only
    ~10k Python files, and PackedDomainStream reshuffles-and-restarts on exhaustion, so a
    20k+ step run almost certainly cycled the whole pool multiple times - there's no
    unseen Python left in that source. Eval reads a *different* language from the same
    dataset instead (genuinely unseen data, but a real distribution shift within "code",
    reported as such rather than passed off as clean).
    """
    _setup_paths()
    import os

    os.chdir("/root/repo")
    import itertools
    import math

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.utils.data import DataLoader, IterableDataset

    from src.data.build_examples import TokenizerBundle
    from src.model.baseline_model import BaselineModel
    from src.model.mot_model import MoTModel
    from src.model.mot_pooled_model import MoTPooledModel
    from src.model.mot_routed_model import MoTRoutedModel
    from src.model.stage2_config import (
        BACKBONE_ONLY_CFG, BATCH_SIZE, CONFIDENCE_WEIGHT, FOCAL_GAMMA, MODEL_CFG, STREAM_SOURCES,
    )

    device = "cuda"
    SKIP_DOCS = 300_000  # well past what 20k-200k training steps could consume from million-row sources

    def _corrupt(text: str, rate: float = 0.08) -> str:
        # char-level noise on the body only - tags/separators are left intact so corruption
        # tests robustness to noisy content (spec §10), not broken routing/tokenization
        import random
        out = []
        for c in text:
            r = random.random()
            if r >= rate or c.isspace():
                out.append(c)
            elif r < rate * 0.35:
                out.append(random.choice("abcdefghijklmnopqrstuvwxyz"))  # substitute
            elif r < rate * 0.55:
                pass  # delete
            elif r < rate * 0.8:
                out.append(c); out.append(random.choice("abcdefghijklmnopqrstuvwxyz"))  # insert
            else:
                out.append(c.swapcase())
        return "".join(out)

    def held_out_body_stream(domain: str, noisy: bool = False):
        """Plain body text (no tag, no DOC_SEP), for composing synthetic held-out
        multi-domain docs. codeparrot/github-code is large enough for a genuine
        `.skip()` split like every other domain now - the old the-stack-smol
        stand-in needed a different-language hack because it wasn't."""
        from src.data.stage2_stream_dataset import TEXT_EXTRACTORS

        cfg = STREAM_SOURCES[domain]
        stream = load_dataset(
            cfg["path"], name=cfg.get("name"), revision=cfg.get("revision"),
            data_files=cfg.get("data_files"), split="train", streaming=True,
        ).skip(SKIP_DOCS)
        extractor = TEXT_EXTRACTORS[domain]
        for row in stream:
            text = extractor(row)
            if text:
                yield _corrupt(text) if noisy else text

    def held_out_doc_stream(domain: str, noisy: bool = False):
        from src.data.stage2_stream_dataset import DOC_SEP
        from src.model.stage2_config import DOMAIN_TAG

        tag = DOMAIN_TAG[domain]
        for body in held_out_body_stream(domain, noisy):
            yield f"{tag}\n{body}{DOC_SEP}"

    def held_out_synthetic_multidomain_stream(noisy: bool = False, seed: int = 0):
        """Held-out mirror of stage2_routed_stream.synthetic_multidomain_doc_stream: composes
        the same 2-4-domain synthetic docs, but every snippet comes from held_out_body_stream
        (post-.skip()) instead of the training-time raw stream. This is what routed/pooled were
        actually built to handle - a single sequence spanning multiple domains with real
        switches - and it's what the eval used to skip entirely (see decision log:
        `evaluate()`'s domain-routed branch forced every eval sequence single-domain, so
        routed/pooled were never once tested on the capability they paid architectural cost
        for)."""
        import random

        from src.data.stage2_routed_stream import MAX_DOMAINS_PER_DOC, MIN_DOMAINS_PER_DOC, SNIPPET_WORDS
        from src.model.stage2_config import DOMAIN_TAG

        rng = random.Random(seed)
        domains = list(STREAM_SOURCES)
        body_streams = {d: held_out_body_stream(d, noisy) for d in domains}
        while True:
            k = rng.randint(MIN_DOMAINS_PER_DOC, MAX_DOMAINS_PER_DOC)
            chosen = rng.sample(domains, k)
            parts = []
            for domain in chosen:
                text = " ".join(next(body_streams[domain]).split()[:SNIPPET_WORDS])
                parts.append(f"{DOMAIN_TAG[domain]}\n{text}\n")
            yield "".join(parts)

    class HeldOutRoutedStream(IterableDataset):
        """Held-out mirror of PackedRoutedStream - same windowing/target logic, sourced from
        held_out_synthetic_multidomain_stream instead of the training-time stream."""

        def __init__(self, bundle, domain_index: dict[str, int], seq_len: int, noisy: bool = False, seed: int = 0):
            self.bundle, self.domain_index, self.seq_len = bundle, domain_index, seq_len
            self.domains = list(domain_index)
            self.noisy, self.seed = noisy, seed

        def __iter__(self):
            from src.data.stage2_routed_stream import _split_spans

            buf_tok, buf_dom, buf_ctrl, buf_typ = [], [], [], []
            for doc in held_out_synthetic_multidomain_stream(self.noisy, self.seed):
                for domain, text in _split_spans(doc):
                    if domain not in self.domain_index:
                        continue
                    di = self.domain_index[domain]
                    buf_tok.append(di); buf_dom.append(di); buf_ctrl.append(1); buf_typ.append(0)
                    ids, types = self.bundle.encode_domain(domain, text, max_len=10**9)
                    buf_tok.extend(ids.tolist())
                    buf_dom.extend([di] * len(ids))
                    buf_ctrl.extend([0] * len(ids))
                    buf_typ.extend(types.tolist() if types is not None else [0] * len(ids))

                window = self.seq_len + 1
                while len(buf_tok) >= window:
                    c_tok, c_dom, c_ctrl, c_typ = buf_tok[:window], buf_dom[:window], buf_ctrl[:window], buf_typ[:window]
                    targets = []
                    for i in range(self.seq_len):
                        nxt = i + 1
                        if c_ctrl[nxt]:
                            from_domain = self.domains[c_dom[i]]
                            targets.append(self.bundle.domain_vocab_sizes[from_domain] + c_dom[nxt])
                        else:
                            targets.append(c_tok[nxt])
                    yield (
                        torch.tensor(c_tok[:self.seq_len], dtype=torch.long),
                        torch.tensor(c_dom[:self.seq_len], dtype=torch.long),
                        torch.tensor(c_ctrl[:self.seq_len], dtype=torch.long),
                        torch.tensor(c_typ[:self.seq_len], dtype=torch.long),
                        torch.tensor(targets, dtype=torch.long),
                    )
                    buf_tok, buf_dom, buf_ctrl, buf_typ = buf_tok[window:], buf_dom[window:], buf_ctrl[window:], buf_typ[window:]

    class HeldOutDomainStream(IterableDataset):
        def __init__(self, domain, encode_domain_fn, seq_len, noisy=False):
            self.domain, self.encode_domain_fn, self.seq_len, self.noisy = domain, encode_domain_fn, seq_len, noisy

        def __iter__(self):
            buf_ids, buf_types = [], []
            has_types = self.domain == "nlp"
            for text in held_out_doc_stream(self.domain, self.noisy):
                ids, types = self.encode_domain_fn(self.domain, text, max_len=10**9)
                buf_ids.extend(ids.tolist())
                if has_types:
                    buf_types.extend(types.tolist())
                while len(buf_ids) >= self.seq_len + 1:
                    chunk_ids = torch.tensor(buf_ids[: self.seq_len + 1], dtype=torch.long)
                    chunk_types = (
                        torch.tensor(buf_types[: self.seq_len + 1], dtype=torch.long)
                        if has_types else torch.zeros(self.seq_len + 1, dtype=torch.long)
                    )
                    yield self.domain, chunk_ids, chunk_types
                    buf_ids = buf_ids[self.seq_len + 1:]
                    if has_types:
                        buf_types = buf_types[self.seq_len + 1:]

    class HeldOutMixedStream(IterableDataset):
        def __init__(self, encode_fn, seq_len, noisy=False):
            self.encode_fn, self.seq_len, self.noisy = encode_fn, seq_len, noisy

        def __iter__(self):
            streams = [held_out_doc_stream(d, self.noisy) for d in STREAM_SOURCES]
            buf = []
            iterators = [iter(s) for s in streams]
            while iterators:
                for it in list(iterators):
                    try:
                        text = next(it)
                    except StopIteration:
                        iterators.remove(it)
                        continue
                    ids = self.encode_fn(text, max_len=10**9)
                    buf.extend(ids.tolist())
                    while len(buf) >= self.seq_len + 1:
                        yield torch.tensor(buf[: self.seq_len + 1], dtype=torch.long)
                        buf = buf[self.seq_len + 1:]

    bundle = TokenizerBundle(tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2")
    ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{arm}_step{checkpoint_step}.pt", map_location=device)

    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    domain_routed = arm in ("mot", "routed", "pooled")  # arms with disjoint per-domain heads
    if arm == "mot":
        model = MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "routed":
        model = MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled":
        model = MoTPooledModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                               focal_gamma=FOCAL_GAMMA, confidence_weight=CONFIDENCE_WEIGHT).to(device)
    elif arm == "baseline":
        model = BaselineModel(vocab_size=bundle.baseline_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    else:
        model = BaselineModel(vocab_size=bundle.sota_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {arm} checkpoint at step {ckpt['step']}", flush=True)

    # --- bits-per-byte machinery ---------------------------------------------------------
    # Per-token loss isn't comparable across arms with different vocabularies. BPB normalises
    # the negative log-likelihood by the raw byte content of the text (identical for every
    # arm), so it measures how well each model compresses the SAME underlying bytes regardless
    # of how its tokenizer chunked them. We can't decode tokens back to bytes reliably (the
    # nlp hybrid's backoff is lossy), so instead measure each tokenizer's bytes-per-token
    # ratio directly on the held-out text and convert: BPB = (nats/ln2) / (tokens * bpt).
    LN2 = math.log(2)
    seq_len = MODEL_CFG["max_seq_len"]

    def bpt_domain(domain: str, n_docs: int = 150) -> float:
        gen = held_out_doc_stream(domain, noisy)
        tot_b, tot_t = 0, 0
        for _ in range(n_docs):
            text = next(gen)
            tot_b += len(text.encode("utf-8"))
            ids, _ = bundle.encode_domain(domain, text, max_len=10**9)
            tot_t += len(ids)
        return tot_b / max(tot_t, 1)

    def bpt_global(encode_fn, n_docs: int = 150) -> float:
        gens = [held_out_doc_stream(d, noisy) for d in STREAM_SOURCES]
        tot_b, tot_t = 0, 0
        for i in range(n_docs):
            text = next(gens[i % len(gens)])
            tot_b += len(text.encode("utf-8"))
            tot_t += len(encode_fn(text, max_len=10**9))
        return tot_b / max(tot_t, 1)

    mode = "NOISY (8% char-corruption)" if noisy else "clean"

    if domain_routed:
        domains = list(bundle.domain_vocab_sizes)
        # mot/routed/pooled share the same per-domain tokenizers, so bytes-per-token is shared
        bpt = {d: bpt_domain(d) for d in domains}
        nats = {d: 0.0 for d in domains}
        toks = {d: 0 for d in domains}
        loaders = {
            d: iter(DataLoader(HeldOutDomainStream(d, bundle.encode_domain, seq_len, noisy), batch_size=BATCH_SIZE))
            for d in domains
        }
        with torch.no_grad():
            for i in range(eval_batches):
                domain = domains[i % len(domains)]
                _, ids, types = next(loaders[domain])
                ids, types = ids.to(device), types.to(device)
                inp, tgt = ids[:, :-1], ids[:, 1:]
                with torch.autocast("cuda"):
                    if arm == "mot":
                        logits = model(domain, inp, types[:, :-1])
                        flat_logits, flat_tgt = logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)
                    else:
                        # routed/pooled: single-domain sequence, no switches. Ask for raw logits
                        # (targets=None) and compute PLAIN CE ourselves - the pooled arm's own
                        # loss is focal-weighted, which would distort NLL, so we bypass it.
                        di = domain_index[domain]
                        dom_ids = torch.full_like(inp, di)
                        ctrl = torch.zeros_like(inp)
                        out = model(inp, dom_ids, ctrl, targets=None, type_ids=types[:, :-1])
                        mask, flat_logits = out[domain]
                        flat_tgt = tgt[mask]
                    step_nats = F.cross_entropy(flat_logits, flat_tgt, reduction="sum")
                nats[domain] += step_nats.item()
                toks[domain] += flat_tgt.numel()
                if (i + 1) % 50 == 0:
                    seen = sum(nats.values()) / LN2 / max(sum(toks[d] * bpt[d] for d in domains), 1)
                    print(f"eval batch {i+1}/{eval_batches}  running BPB={seen:.4f}", flush=True)

        agg_bits = sum(nats[d] for d in domains) / LN2
        agg_bytes = sum(toks[d] * bpt[d] for d in domains)
        single_domain_bpb = agg_bits / max(agg_bytes, 1)
        print(f"\nHELD-OUT single-domain BPB for {arm}, {mode} (checkpoint step {ckpt['step']}): "
              f"{single_domain_bpb:.4f} bits/byte", flush=True)
        for d in domains:
            d_bpb = (nats[d] / LN2) / max(toks[d] * bpt[d], 1)
            d_ppl_tok = math.exp(nats[d] / max(toks[d], 1))
            print(f"  {d:8s} BPB={d_bpb:.4f}  (per-token ppl={d_ppl_tok:.1f}, bytes/tok={bpt[d]:.2f}, n_tok={toks[d]})", flush=True)

        cross_domain_bpb, switch_accuracy = None, None
        if arm in ("routed", "pooled"):
            # THE fix: routed/pooled were architected for mixed-domain sequences with real
            # switches, but the loop above forces every eval sequence single-domain. This
            # second pass is what they were actually built to be judged on. Switch-target
            # positions are excluded from the BPB accounting (predicting "switch to domain Y"
            # isn't compressing content bytes) and reported separately as switch_accuracy.
            print(f"\n--- cross-domain pass (real switches, {arm} only) ---", flush=True)
            cd_nats, cd_toks = {d: 0.0 for d in domains}, {d: 0 for d in domains}
            switch_correct, switch_total = 0, 0
            cd_loader = iter(DataLoader(
                HeldOutRoutedStream(bundle, domain_index, seq_len, noisy), batch_size=BATCH_SIZE))
            with torch.no_grad():
                for i in range(eval_batches):
                    tok, dom, ctrl, typ, tgt = next(cd_loader)
                    tok, dom, ctrl, typ, tgt = (t.to(device) for t in (tok, dom, ctrl, typ, tgt))
                    with torch.autocast("cuda"):
                        out = model(tok, dom, ctrl, targets=None, type_ids=typ)
                        for d in domains:
                            if d not in out:
                                continue
                            mask, logits = out[d]
                            d_tgt = tgt[mask]
                            is_switch = d_tgt >= bundle.domain_vocab_sizes[d]
                            content_logits, content_tgt = logits[~is_switch], d_tgt[~is_switch]
                            if content_tgt.numel():
                                cd_nats[d] += F.cross_entropy(content_logits, content_tgt, reduction="sum").item()
                                cd_toks[d] += content_tgt.numel()
                            if is_switch.any():
                                sw_logits, sw_tgt = logits[is_switch], d_tgt[is_switch]
                                switch_correct += (sw_logits.argmax(dim=-1) == sw_tgt).sum().item()
                                switch_total += sw_tgt.numel()
                    if (i + 1) % 50 == 0:
                        print(f"cross-domain eval batch {i+1}/{eval_batches}", flush=True)

            cd_bits = sum(cd_nats[d] for d in domains) / LN2
            cd_bytes = sum(cd_toks[d] * bpt[d] for d in domains)
            cross_domain_bpb = cd_bits / max(cd_bytes, 1)
            switch_accuracy = switch_correct / max(switch_total, 1)
            print(f"\nHELD-OUT cross-domain BPB for {arm}, {mode}: {cross_domain_bpb:.4f} bits/byte "
                  f"(vs single-domain {single_domain_bpb:.4f}); switch-prediction accuracy: "
                  f"{switch_accuracy:.4f} ({switch_correct}/{switch_total})", flush=True)

        result = {"single_domain_bpb": single_domain_bpb, "cross_domain_bpb": cross_domain_bpb,
                  "switch_accuracy": switch_accuracy}
    else:
        encode_fn = bundle.encode_baseline if arm == "baseline" else bundle.encode_sota
        bpt = bpt_global(encode_fn)
        total_nats, total_toks = 0.0, 0
        loader = iter(DataLoader(HeldOutMixedStream(encode_fn, seq_len, noisy), batch_size=BATCH_SIZE))
        with torch.no_grad():
            for i in range(eval_batches):
                ids = next(loader).to(device)
                inp, tgt = ids[:, :-1], ids[:, 1:]
                with torch.autocast("cuda"):
                    logits = model(inp)
                    step_nats = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
                total_nats += step_nats.item()
                total_toks += tgt.numel()
                if (i + 1) % 50 == 0:
                    print(f"eval batch {i+1}/{eval_batches}  running BPB={(total_nats/LN2)/max(total_toks*bpt,1):.4f}", flush=True)
        single_domain_bpb = (total_nats / LN2) / max(total_toks * bpt, 1)
        ppl_tok = math.exp(total_nats / max(total_toks, 1))
        print(f"\nHELD-OUT BPB for {arm}, {mode} (checkpoint step {ckpt['step']}): "
              f"{single_domain_bpb:.4f} bits/byte", flush=True)
        print(f"  (per-token ppl={ppl_tok:.1f}, bytes/tok={bpt:.2f}, n_tok={total_toks})", flush=True)
        # baseline/sota can't take mixed-domain input structurally (one shared vocab, no
        # per-position routing) - there's no separate "cross-domain capability" to measure.
        result = {"single_domain_bpb": single_domain_bpb, "cross_domain_bpb": None, "switch_accuracy": None}

    return result


@app.local_entrypoint()
def main(step: str = "calibrate", arm: str = "mot", steps: int = 0, resume_from: str = "", noisy: bool = False):
    """steps=0 means "use the default": 150 for calibrate, MAX_STEPS for train."""
    if step == "sample-tokenizers":
        sample_tokenizers.remote()
    elif step == "train-tokenizers":
        train_tokenizers.remote()
    elif step == "calibrate":
        sec_per_step = calibrate.remote(arm=arm, steps=steps or 150)
        from src.model.stage2_config import MAX_STEPS
        print(f"\n--- extrapolation ---")
        print(f"{sec_per_step:.3f} sec/step x {MAX_STEPS} steps = {sec_per_step*MAX_STEPS/3600:.2f} GPU-hours")
        print(f"at ~$0.59/hr (T4): ~${sec_per_step*MAX_STEPS/3600*0.59:.2f} for this arm")
    elif step == "evaluate":
        result = evaluate.remote(arm=arm, checkpoint_step=steps or 20000, noisy=noisy)
        mode = "noisy" if noisy else "clean"
        print(f"\nheld-out single-domain BPB for {arm} ({mode}): {result['single_domain_bpb']:.4f} bits/byte")
        if result["cross_domain_bpb"] is not None:
            print(f"held-out cross-domain BPB for {arm} ({mode}): {result['cross_domain_bpb']:.4f} bits/byte")
            print(f"switch-prediction accuracy: {result['switch_accuracy']:.4f}")
    elif step == "train":
        history = train.remote(arm=arm, max_steps=steps or None, resume_from=resume_from or None)
        print(f"\nfinal logged losses: {history[-5:] if history else '(none)'}")
    else:
        raise ValueError(f"unknown step: {step}")
