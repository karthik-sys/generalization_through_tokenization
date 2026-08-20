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
        # pinned exactly (not >=) - newer datasets releases dropped trust_remote_code support
        # entirely ("trust_remote_code is not supported anymore"), which pg19's legacy
        # loading-script format needs. Matches the training pods' installed version (verified:
        # pip show datasets -> 2.19.0), so eval reads the same data the way training did.
        "datasets==2.19.0",
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


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=1800)
def train_generalist_tokenizer(vocab_size: int = 32000):
    """routed33's 5th "generalist" domain: a statistical anchor trained on a MIX of all four
    existing domains' already-sampled corpora (dispatch's "mined intersection vocabulary"
    idea) rather than a new external source - zero new data to fetch. Reuses the exact same
    sampled texts already on the volume from the original tokenizer run, just pooled instead
    of kept per-domain, so this vocab reflects genuinely shared/general structure across
    code+math+science+nlp instead of specializing in any one of them."""
    _setup_paths()
    import os

    os.chdir("/root/repo")
    from src.tokenizers.train_all_stage2 import _load_sample_texts
    from src.tokenizers.bpe_tokenizer import train_bpe

    out_dir = f"{VOLUME_PATH}/tokenizers_stage2_generalist"
    sample_dir = f"{VOLUME_PATH}/stage2_tokenizer_sample"
    pooled = []
    for domain in ("code", "math", "science", "nlp"):
        texts = _load_sample_texts(domain, sample_dir)
        print(f"pooling {len(texts)} sampled docs from {domain}", flush=True)
        pooled.extend(texts)
    print(f"training generalist BPE tokenizer on {len(pooled)} pooled docs, vocab={vocab_size}", flush=True)
    train_bpe(pooled, model_prefix=f"{out_dir}/generalist/model", vocab_size=vocab_size)
    volume.commit()
    print("DONE: generalist tokenizer (pooled code+math+science+nlp) written to tokenizers_stage2_generalist", flush=True)


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=1800)
def train_shrunk_vocab_tokenizers(vocab_size: int = 10000):
    """routed-B: code/math/science are starved of tokens under the diet mixture (~70%+ nlp)
    yet still carry DOMAIN_VOCAB_SIZES' full 24k vocab each - real over-parameterization for
    how little data actually updates them. Retrains ONLY those three at a smaller vocab_size,
    reusing the ALREADY-SAMPLED corpus on the volume (no need to re-sample) and writing to a
    separate output dir so the existing tokenizers_stage2 (used by every other arm) is left
    untouched. nlp and the baseline/sota tokenizers are deliberately not retrained here."""
    _setup_paths()
    import os

    os.chdir("/root/repo")
    from src.tokenizers.train_all_stage2 import _load_sample_texts
    from src.tokenizers.bpe_tokenizer import train_bpe

    out_dir = f"{VOLUME_PATH}/tokenizers_stage2_shrunk"
    sample_dir = f"{VOLUME_PATH}/stage2_tokenizer_sample"
    for domain in ("code", "math", "science"):
        texts = _load_sample_texts(domain, sample_dir)
        print(f"retraining {domain} BPE tokenizer on {len(texts)} sampled docs, vocab={vocab_size}", flush=True)
        train_bpe(texts, model_prefix=f"{out_dir}/{domain}/model", vocab_size=vocab_size)
    volume.commit()
    print("DONE: shrunk code/math/science tokenizers written to tokenizers_stage2_shrunk", flush=True)


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
    elif arm in ("routed2", "routed3"):
        model = MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
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
    if arm in ("routed", "pooled", "pooled2", "hybrid", "routed2"):
        loader = iter(DataLoader(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm == "routed3":
        from src.model.stage2_config import ROUTED3_MAX_DOMAINS, ROUTED3_MIN_DOMAINS, ROUTED3_SNIPPET_WORDS
        loader = iter(DataLoader(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            min_domains=ROUTED3_MIN_DOMAINS, max_domains=ROUTED3_MAX_DOMAINS,
            snippet_words=ROUTED3_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
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
        elif arm in ("routed2", "routed3"):
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
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
    elif arm in ("routed2", "routed3"):
        model = MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
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
    if arm in ("routed", "pooled", "pooled2", "hybrid", "routed2"):
        loader = iter(DataLoader(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm == "routed3":
        from src.model.stage2_config import ROUTED3_MAX_DOMAINS, ROUTED3_MIN_DOMAINS, ROUTED3_SNIPPET_WORDS
        loader = iter(DataLoader(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            min_domains=ROUTED3_MIN_DOMAINS, max_domains=ROUTED3_MAX_DOMAINS,
            snippet_words=ROUTED3_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
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
    if arm in ("pooled", "pooled2", "hybrid", "routed2", "routed3"):
        from src.model.adaptive_optimizer import AdaptiveController
        controller = AdaptiveController()
        print("adaptive controller ON (spike-guard + plateau rescue + online LR)", flush=True)

    # Periodic held-out eval (idea 4a) - see train_stage2_pod.py for the rationale. Mirrored
    # here so the Modal entry point behaves identically. Switching arms only (all share
    # PackedRoutedStream); plain content CE on a held-out seed, distinct from training.
    from src.model.stage2_config import EVAL_EVERY, VAL_BATCHES, VAL_SEED
    from src.model.stage2_config import ROUTED3_MAX_DOMAINS, ROUTED3_MIN_DOMAINS, ROUTED3_SNIPPET_WORDS

    val_iter = None
    if arm in ("routed", "pooled", "pooled2", "hybrid", "routed2", "routed3"):
        rs3 = arm == "routed3"
        val_iter = iter(DataLoader(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"], seed=VAL_SEED,
            min_domains=ROUTED3_MIN_DOMAINS if rs3 else 2,
            max_domains=ROUTED3_MAX_DOMAINS if rs3 else 4,
            snippet_words=ROUTED3_SNIPPET_WORDS if rs3 else 250,
        ), batch_size=BATCH_SIZE))

    def _held_out_ce():
        model.eval()
        tot_nats, tot_tok = 0.0, 0
        with torch.no_grad():
            for _ in range(VAL_BATCHES):
                tok, dom, ctrl, typ, tgt = next(val_iter)
                tok, dom, ctrl, typ, tgt = (t.to(device) for t in (tok, dom, ctrl, typ, tgt))
                with torch.autocast("cuda"):
                    out = model(tok, dom, ctrl, targets=None, type_ids=typ)
                for d in model.domains:
                    if d not in out:
                        continue
                    mask, logits = out[d]
                    dt = tgt[mask]
                    keep = dt < bundle.domain_vocab_sizes[d]
                    if keep.any():
                        tot_nats += F.cross_entropy(logits[keep], dt[keep], reduction="sum").item()
                        tot_tok += int(keep.sum().item())
        model.train()
        return tot_nats / max(tot_tok, 1)

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
        elif arm in ("routed2", "routed3"):
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
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
        controller_loss_val = float(controller_loss) if arm in ("pooled", "pooled2", "hybrid", "routed2", "routed3") else loss_val
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

        if val_iter is not None and (step % EVAL_EVERY == 0 or step == total_steps):
            val_ce = _held_out_ce()
            history.append({"step": step, "val_ce": round(val_ce, 4), "elapsed": round(time.time() - t0)})
            print(f"  >>> held-out val CE @ step {step}: {val_ce:.4f} (over {VAL_BATCHES} val batches)", flush=True)

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


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=5400,
              secrets=[modal.Secret.from_name("huggingface-token")])
def evaluate(arm: str = "mot", checkpoint_step: int = 20000, eval_batches: int = 200, noisy: bool = False, scale: str = "base"):
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
    from src.model.backbone_modern import ModernBackbone
    from src.model.baseline_model import BaselineModel
    from src.model.mot_hybrid_model import MoTHybridModel
    from src.model.mot_model import MoTModel
    from src.model.mot_pooled2_model import MoTPooled2Model
    from src.model.mot_pooled_model import MoTPooledModel
    from src.model.mot_routed_combined_model import MoTRoutedCombinedModel
    from src.model.mot_routed_copygate_model import MoTRoutedCopyGateModel
    from src.model.mot_routed_deepexpert_model import MoTRoutedDeepExpertModel
    from src.model.mot_routed_model import MoTRoutedModel
    from src.model.mot_routed_precision_model import MoTRoutedPrecisionModel
    from src.model.mot_routed_tied_model import MoTRoutedTiedModel
    from src.model.stage2_config import (
        BACKBONE_ONLY_CFG, BATCH_SIZE, CONFIDENCE_WEIGHT, FOCAL_GAMMA, LARGE_BACKBONE_ONLY_CFG,
        LARGE_MODEL_CFG, LONGCTX_MODEL_CFG, MODEL_CFG, ROUTED30_MODEL_CFG, ROUTED_MODERN_MODEL_CFG,
        STREAM_SOURCES, TIED_MODEL_CFG,
    )

    device = "cuda"
    if arm == "routed4":
        MODEL_CFG = LONGCTX_MODEL_CFG  # routed4 is always 2x context, regardless of --scale
    elif arm == "routed7":
        MODEL_CFG = LARGE_MODEL_CFG  # routed7 is always large-scale, regardless of --scale
        scale = "large"
        # held-out eval must read from the SAME distribution routed7 trained on, or this
        # scores it against text it never saw (see _apply_openwebtext_nlp_source in
        # train_stage2_pod.py for the full rationale - code/math/science untouched).
        STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    elif arm in ("routed8", "routed11", "routed12", "routed13", "routed15", "routed16", "routed17", "routed18", "routed19"):
        # all base scale (unlike routed7) - routed11-18 are all continued-training runs from
        # routed8, so they reuse its exact nlp source override. routed19 is from scratch but
        # uses the same OWT nlp source (same tokenizer, same design as routed8).
        STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    elif arm == "routed14":
        MODEL_CFG = LARGE_MODEL_CFG  # routed14 is always large-scale (warm-started from routed7)
        scale = "large"
        STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    elif arm == "routed28":
        MODEL_CFG = LARGE_MODEL_CFG  # routed28 is always large-scale (warm-started from routed14)
        scale = "large"
        STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    elif arm == "routed10":
        MODEL_CFG = LARGE_MODEL_CFG  # routed10 is always large-scale, regardless of --scale
        scale = "large"
        STREAM_SOURCES["nlp"] = {"path": "deepmind/pg19", "name": None, "gated": False}
    elif arm == "routed9":
        # routed9 stays BASE scale (unlike routed10) - only the nlp source override applies.
        STREAM_SOURCES["nlp"] = {"path": "deepmind/pg19", "name": None, "gated": False}
    elif arm == "routed27":
        # base scale, but trained on books (see _apply_books_nlp_source in train_stage2_pod.py)
        # - held-out eval must match, or this scores it against a distribution it never saw.
        STREAM_SOURCES["nlp"] = {"path": "deepmind/pg19", "name": None, "gated": False}
    elif arm in ("routed20", "routed21", "routed22", "routed23", "routed24", "routed25", "routed26"):
        STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    elif arm == "routed29":
        MODEL_CFG = TIED_MODEL_CFG  # bridge-tied heads, 23 layers, from scratch
        STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    elif arm == "routed30":
        MODEL_CFG = ROUTED30_MODEL_CFG  # direct-tied (wide emb_dim=512), shrunk vocab, from scratch
        STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    elif arm in ("routed31", "routed32"):
        MODEL_CFG = ROUTED_MODERN_MODEL_CFG  # RoPE/RMSNorm(+SwiGLU/QK-norm for routed31 only)
        STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    elif arm == "routed33":
        MODEL_CFG = LARGE_MODEL_CFG  # 5-domain generalist, large scale, from scratch
        scale = "large"
        STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    elif scale == "large":
        MODEL_CFG = LARGE_MODEL_CFG
    BACKBONE_ONLY_CFG = LARGE_BACKBONE_ONLY_CFG if scale == "large" else BACKBONE_ONLY_CFG
    ckpt_prefix = f"large_{arm}" if scale == "large" else arm
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
        # PG-19 (routed9/routed10's nlp source) has only ~28.6k rows total, each a whole book -
        # SKIP_DOCS=300k (sized for the other, million-row sources) would force the streamer to
        # download and discard the ENTIRE corpus before ever yielding a row, which is exactly
        # what hung real evaluate() calls for tens of minutes (confirmed: an isolated
        # load_dataset+next() test on pg19 alone returns in under 10s - the hang was this skip,
        # not the dataset load). A small skip is just as "held-out" as a large one here anyway -
        # a corpus this size gets fully cycled by training many times over by step 20k+,
        # so no skip count restores a clean split; it only needs to not be larger than the
        # dataset.
        skip_docs = 500 if cfg["path"] == "deepmind/pg19" else SKIP_DOCS
        stream = load_dataset(
            cfg["path"], name=cfg.get("name"), revision=cfg.get("revision"),
            data_files=cfg.get("data_files"), split="train", streaming=True,
            trust_remote_code=True,
        ).skip(skip_docs)
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

    bundle = TokenizerBundle(
        tokenizer_dir=(f"{VOLUME_PATH}/tokenizers_stage2_shrunk" if arm == "routed30" else f"{VOLUME_PATH}/tokenizers_stage2"),
        nlp_tokenizer_dir=(None if arm == "routed30" else
                            (f"{VOLUME_PATH}/tokenizers_stage2_owt/nlp"
                            if arm in ("routed7", "routed8", "routed11", "routed12", "routed13", "routed14",
                                        "routed15", "routed16", "routed17", "routed18", "routed19",
                                        "routed20", "routed21", "routed22", "routed23", "routed24",
                                        "routed25", "routed26", "routed27", "routed28", "routed29",
                                        "routed31", "routed32", "routed33") else
                            (f"{VOLUME_PATH}/tokenizers_stage2_books/nlp" if arm in ("routed9", "routed10") else None))),
        generalist_tokenizer_dir=(f"{VOLUME_PATH}/tokenizers_stage2_generalist/generalist" if arm == "routed33" else None),
    )
    ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{ckpt_prefix}_step{checkpoint_step}.pt", map_location=device)

    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    domain_routed = arm in ("mot", "routed", "pooled", "hybrid", "pooled2", "routed2", "routed3", "routed4",
                             "routed7", "routed8", "routed9", "routed10", "routed11", "routed12", "routed13", "routed14",
                             "routed15", "routed16", "routed17", "routed18", "routed19",
                             "routed20", "routed21", "routed22", "routed23", "routed24",
                             "routed25", "routed26", "routed27", "routed28",
                             "routed29", "routed30", "routed31", "routed32", "routed33")
    if arm == "mot":
        model = MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm in ("routed", "routed7", "routed8", "routed9", "routed10", "routed15", "routed17", "routed18",
                 "routed19", "routed23"):
        model = MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm in ("routed11", "routed14", "routed16", "routed20", "routed21", "routed22", "routed24",
                 "routed25", "routed26", "routed27", "routed28", "routed33"):
        model = MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm in ("routed29", "routed30"):
        model = MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "routed31":
        model = MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                    backbone_cls=ModernBackbone,
                                    backbone_kwargs={"use_swiglu": True, "use_qk_norm": True}).to(device)
    elif arm == "routed32":
        model = MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                    backbone_cls=ModernBackbone,
                                    backbone_kwargs={"use_swiglu": False, "use_qk_norm": False}).to(device)
    elif arm == "routed12":
        model = MoTRoutedDeepExpertModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "routed13":
        model = MoTRoutedPrecisionModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled":
        model = MoTPooledModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                               focal_gamma=FOCAL_GAMMA, confidence_weight=CONFIDENCE_WEIGHT).to(device)
    elif arm in ("hybrid", "routed2", "routed3"):
        model = MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled2":
        model = MoTPooled2Model(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                focal_gamma=FOCAL_GAMMA).to(device)
    elif arm == "routed4":
        model = MoTRoutedCombinedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "baseline":
        model = BaselineModel(vocab_size=bundle.baseline_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    else:
        model = BaselineModel(vocab_size=bundle.sota_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_prefix} checkpoint at step {ckpt['step']}", flush=True)

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
        # routed33's "generalist" domain has no external source of its own (it's a synthetic
        # pool of the other 4 domains, see build_generalist_cache) - there's no genuinely
        # held-out generalist text to live-stream, so it's excluded from single-domain BPB
        # rather than KeyError on a STREAM_SOURCES lookup that was never meant to exist for it.
        domains = [d for d in bundle.domain_vocab_sizes if d in STREAM_SOURCES]
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
        if arm != "mot":
            # mot is the only domain_routed arm without switching capability at all - every
            # other one (routed, pooled, hybrid, pooled2, routed2, routed3) descends from
            # MoTRoutedModel and supports the control-token/switch mechanism, so this is
            # "domain_routed and switching-capable" rather than an explicit arm list that
            # needs updating every time a new switching arm is added.
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


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=1800,
              secrets=[modal.Secret.from_name("huggingface-token")])
def evaluate_lambada(arm: str = "mot", checkpoint_step: int = 150000, n_examples: int = 500, scale: str = "base"):
    """LAMBADA (Paperno et al. 2016): predict the final word of a passage, given the full
    preceding context. Deliberately NOT the same axis as BPB - BPB measures compression on
    held-out text drawn from the SAME domains/tokenizers the model trained on, so a model
    architected around domain-specific vocab has a structural advantage there almost by
    construction. LAMBADA is a fixed, external, architecture-agnostic completion benchmark -
    it doesn't care whose vocab is whose, it just asks "did you predict the right word."

    It's also one of the few standard benchmarks that stays meaningful at our scale (~70-120M
    params). MMLU/GSM8K/HumanEval are calibrated for billion-parameter, often instruction-
    tuned models and would put every arm here at or near random-chance floor - not because
    any arm failed, but because those benchmarks can't see anything at this size. LAMBADA has
    shown real, differentiated signal since the GPT-2-small era, which is our regime.

    All LAMBADA text is routed through the "nlp" domain for domain-routed arms (mot, routed,
    pooled, hybrid, pooled2) - it's natural English prose, the same kind of content nlp's
    tokenizer and embedding table were trained on. baseline/sota use their single shared
    vocab directly, no domain routing to speak of.

    Scoring: exact-match on the tokenized answer word, teacher-forced (the true context is
    given; the model doesn't need to have generated it correctly, it's scored on next-token
    predictions at each position of the answer). This is stricter than accuracy computed only
    on the answer's first token, but it's the fair choice here - different tokenizers split
    the same word into different numbers of pieces, so first-token-only would silently
    advantage whichever tokenizer happens to chunk more coarsely.
    """
    _setup_paths()
    import os

    os.chdir("/root/repo")
    import torch
    from datasets import load_dataset

    from src.data.build_examples import TokenizerBundle
    from src.model.backbone_modern import ModernBackbone
    from src.model.baseline_model import BaselineModel
    from src.model.mot_hybrid_model import MoTHybridModel
    from src.model.mot_model import MoTModel
    from src.model.mot_pooled2_model import MoTPooled2Model
    from src.model.mot_pooled_model import MoTPooledModel
    from src.model.mot_routed_combined_model import MoTRoutedCombinedModel
    from src.model.mot_routed_copygate_model import MoTRoutedCopyGateModel
    from src.model.mot_routed_deepexpert_model import MoTRoutedDeepExpertModel
    from src.model.mot_routed_model import MoTRoutedModel
    from src.model.mot_routed_precision_model import MoTRoutedPrecisionModel
    from src.model.mot_routed_tied_model import MoTRoutedTiedModel
    from src.model.stage2_config import (
        BACKBONE_ONLY_CFG, CONFIDENCE_WEIGHT, FOCAL_GAMMA, LARGE_BACKBONE_ONLY_CFG, LARGE_MODEL_CFG,
        LONGCTX_MODEL_CFG, MODEL_CFG, ROUTED30_MODEL_CFG, ROUTED_MODERN_MODEL_CFG, TIED_MODEL_CFG,
    )

    device = "cuda"
    if arm == "routed4":
        MODEL_CFG = LONGCTX_MODEL_CFG  # routed4 is always 2x context, regardless of --scale
    elif arm == "routed7":
        MODEL_CFG = LARGE_MODEL_CFG  # routed7 is always large-scale, regardless of --scale
        scale = "large"
        # (no STREAM_SOURCES override needed here, unlike evaluate() - LAMBADA always reads
        # from the fixed EleutherAI/lambada_openai benchmark, never from STREAM_SOURCES)
    elif arm in ("routed8", "routed11", "routed12", "routed13", "routed15", "routed16", "routed17", "routed18"):
        pass  # all BASE scale - only the nlp tokenizer dir below differs
    elif arm == "routed14":
        MODEL_CFG = LARGE_MODEL_CFG  # routed14 is always large-scale (warm-started from routed7)
        scale = "large"
    elif arm == "routed28":
        MODEL_CFG = LARGE_MODEL_CFG  # routed28 is always large-scale (warm-started from routed14)
        scale = "large"
    elif arm == "routed10":
        MODEL_CFG = LARGE_MODEL_CFG  # routed10 is always large-scale, regardless of --scale
        scale = "large"
    elif arm == "routed9":
        pass  # routed9 stays BASE scale - only the nlp tokenizer dir below differs
    elif arm == "routed29":
        MODEL_CFG = TIED_MODEL_CFG
    elif arm == "routed30":
        MODEL_CFG = ROUTED30_MODEL_CFG
    elif arm in ("routed31", "routed32"):
        MODEL_CFG = ROUTED_MODERN_MODEL_CFG
    elif arm == "routed33":
        MODEL_CFG = LARGE_MODEL_CFG
        scale = "large"
    elif scale == "large":
        MODEL_CFG = LARGE_MODEL_CFG
    BACKBONE_ONLY_CFG = LARGE_BACKBONE_ONLY_CFG if scale == "large" else BACKBONE_ONLY_CFG
    ckpt_prefix = f"large_{arm}" if scale == "large" else arm
    seq_len = MODEL_CFG["max_seq_len"]
    bundle = TokenizerBundle(
        tokenizer_dir=(f"{VOLUME_PATH}/tokenizers_stage2_shrunk" if arm == "routed30" else f"{VOLUME_PATH}/tokenizers_stage2"),
        nlp_tokenizer_dir=(None if arm == "routed30" else
                            (f"{VOLUME_PATH}/tokenizers_stage2_owt/nlp"
                            if arm in ("routed7", "routed8", "routed11", "routed12", "routed13", "routed14",
                                        "routed15", "routed16", "routed17", "routed18", "routed19",
                                        "routed20", "routed21", "routed22", "routed23", "routed24",
                                        "routed25", "routed26", "routed27", "routed28", "routed29",
                                        "routed31", "routed32", "routed33") else
                            (f"{VOLUME_PATH}/tokenizers_stage2_books/nlp" if arm in ("routed9", "routed10") else None))),
        generalist_tokenizer_dir=(f"{VOLUME_PATH}/tokenizers_stage2_generalist/generalist" if arm == "routed33" else None),
    )
    ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{ckpt_prefix}_step{checkpoint_step}.pt", map_location=device)

    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    domain_routed = arm in ("mot", "routed", "pooled", "hybrid", "pooled2", "routed2", "routed3", "routed4",
                             "routed7", "routed8", "routed9", "routed10", "routed11", "routed12", "routed13", "routed14",
                             "routed15", "routed16", "routed17", "routed18", "routed19",
                             "routed20", "routed21", "routed22", "routed23", "routed24",
                             "routed25", "routed26", "routed27", "routed28",
                             "routed29", "routed30", "routed31", "routed32", "routed33")
    if arm == "mot":
        model = MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm in ("routed", "routed7", "routed8", "routed9", "routed10", "routed15", "routed17", "routed18",
                 "routed19", "routed23"):
        model = MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm in ("routed11", "routed14", "routed16", "routed20", "routed21", "routed22", "routed24",
                 "routed25", "routed26", "routed27", "routed28", "routed33"):
        model = MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm in ("routed29", "routed30"):
        model = MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "routed31":
        model = MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                    backbone_cls=ModernBackbone,
                                    backbone_kwargs={"use_swiglu": True, "use_qk_norm": True}).to(device)
    elif arm == "routed32":
        model = MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                    backbone_cls=ModernBackbone,
                                    backbone_kwargs={"use_swiglu": False, "use_qk_norm": False}).to(device)
    elif arm == "routed12":
        model = MoTRoutedDeepExpertModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "routed13":
        model = MoTRoutedPrecisionModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled":
        model = MoTPooledModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                               focal_gamma=FOCAL_GAMMA, confidence_weight=CONFIDENCE_WEIGHT).to(device)
    elif arm == "hybrid":
        model = MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled2":
        model = MoTPooled2Model(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                focal_gamma=FOCAL_GAMMA).to(device)
    elif arm in ("routed2", "routed3"):
        model = MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "routed4":
        model = MoTRoutedCombinedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "baseline":
        model = BaselineModel(vocab_size=bundle.baseline_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    else:
        model = BaselineModel(vocab_size=bundle.sota_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_prefix} checkpoint at step {ckpt['step']}", flush=True)

    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test", streaming=True)
    encode_fn = None
    if not domain_routed:
        encode_fn = bundle.encode_baseline if arm == "baseline" else bundle.encode_sota

    import math
    LN2 = math.log(2)

    # GPT-2's published LAMBADA numbers (Radford et al. 2019) used a "stop-word filter":
    # common function words (the/a/and/of/...) are excluded from the model's candidate guess
    # for the target word, since LAMBADA's real targets are essentially never plain stopwords
    # - without the filter GPT-2's raw exact-match was 52.66%, with it 63.24% (their own
    # reported delta). Our primary `accuracy` above is the UNFILTERED, stricter number (every
    # sub-word piece of the answer must match exactly, no stopword exclusion) - it is NOT the
    # same thing GPT-2's headline figures measure. This computes the matched-methodology
    # number as a SEPARATE, additional metric so a comparison to their published numbers has
    # something to actually stand on, without touching the primary metric already recorded
    # for every other arm this session.
    #
    # This is our own reconstruction of their filter (the exact word list was never published
    # by OpenAI) - treat it as an approximation of their approximation, not a certified match.
    _STOPWORDS = [
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by",
        "for", "with", "as", "is", "was", "were", "are", "be", "been", "being", "it",
        "its", "this", "that", "these", "those", "he", "she", "they", "we", "you", "i",
        "his", "her", "their", "our", "your", "my", "him", "them", "us", "me", "not",
        "no", "so", "up", "out", "about", "into", "over", "after", "before", "than",
        "then", "there", "here", "what", "which", "who", "when", "where", "how", "all",
        "any", "each", "other", "some", "such", "only", "own", "same", "too", "very",
        "just", "also", "one", "would", "could", "should", "will", "shall", "can", "may",
        "might", "must", "do", "does", "did", "have", "has", "had", "from",
    ]

    def _stopword_token_ids(enc_fn) -> set[int]:
        """Single-token encodings only - a multi-piece "stopword" can't collide with a
        single-position first-token argmax the same way, and most true function words are
        single tokens in any reasonable BPE-family vocab anyway."""
        ids = set()
        for w in _STOPWORDS:
            for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                try:
                    enc = enc_fn(variant, max_len=10**9)
                    tok_ids = enc[0] if isinstance(enc, tuple) else enc
                    if len(tok_ids) == 1:
                        ids.add(int(tok_ids[0]))
                except Exception:
                    continue
        return ids

    if domain_routed:
        stopword_ids = _stopword_token_ids(lambda t, max_len: bundle.encode_domain("nlp", t, max_len=max_len))
    else:
        stopword_ids = _stopword_token_ids(encode_fn)
    stopword_id_tensor = torch.tensor(sorted(stopword_ids), dtype=torch.long, device=device) if stopword_ids else None
    print(f"stop-word filter: {len(stopword_ids)} single-token stopword ids identified for {arm}'s tokenizer", flush=True)

    n_correct, n_correct_filtered, n_scored, n_skipped = 0, 0, 0, 0
    total_nats, total_target_tokens = 0.0, 0
    with torch.no_grad():
        for i, row in enumerate(ds):
            if n_scored >= n_examples:
                break
            text = row["text"]
            if " " not in text.strip():
                n_skipped += 1
                continue
            context, target_word = text.rsplit(" ", 1)
            # Every arm trained exclusively on tag-prefixed documents (_raw_doc_stream
            # prepends "<domain:X>\n" to every document, baseline/sota's PackedMixedStream
            # included) - testing on naked untagged text is out-of-distribution formatting,
            # not a fair completion test. The tag matters far more for baseline/sota than
            # for domain-routed arms: mot/routed/pooled get their embedding-table/head
            # routing from the explicit encode_domain("nlp", ...) call below regardless of
            # what the text says, but baseline/sota have no external routing at all - the
            # tag is their ONLY signal for what kind of text follows.
            from src.model.stage2_config import DOMAIN_TAG
            context = f"{DOMAIN_TAG['nlp']}\n{context} "  # trailing space: target tokenizes as a natural continuation

            if domain_routed:
                ctx_ids, ctx_types = bundle.encode_domain("nlp", context, max_len=10**9)
                tgt_ids, tgt_types = bundle.encode_domain("nlp", target_word, max_len=10**9)
            else:
                ctx_ids, tgt_ids = encode_fn(context, max_len=10**9), encode_fn(target_word, max_len=10**9)
                ctx_types = tgt_types = None

            n_ctx, n_tgt = len(ctx_ids), len(tgt_ids)
            if n_tgt == 0 or n_ctx == 0 or n_ctx + n_tgt > seq_len:
                n_skipped += 1
                continue

            full_ids = torch.cat([ctx_ids, tgt_ids]).unsqueeze(0).to(device)
            full_types = None
            if ctx_types is not None:
                full_types = torch.cat([ctx_types, tgt_types]).unsqueeze(0).to(device)

            with torch.autocast("cuda"):
                if arm == "mot":
                    logits = model("nlp", full_ids[:, :-1], full_types[:, :-1] if full_types is not None else None)
                elif domain_routed:
                    di = domain_index["nlp"]
                    dom = torch.full_like(full_ids[:, :-1], di)
                    ctrl = torch.zeros_like(full_ids[:, :-1])
                    typ_in = full_types[:, :-1] if full_types is not None else None
                    out = model(full_ids[:, :-1], dom, ctrl, targets=None, type_ids=typ_in)
                    _, logits = out["nlp"]
                    logits = logits.unsqueeze(0)  # (1, L, vocab) to match the mot/baseline shape below
                else:
                    logits = model(full_ids[:, :-1])

            # input position i predicts full_ids position i+1; the first target token sits at
            # full_ids position n_ctx, so its prediction comes from input position n_ctx-1.
            target_logits = logits[0, n_ctx - 1:]
            pred_ids = target_logits.argmax(dim=-1)
            true_ids = full_ids[0, n_ctx:]
            correct = torch.equal(pred_ids[: len(true_ids)], true_ids)
            n_correct += int(correct)

            # Stop-word-filtered variant (see note above the loop): only the FIRST target
            # token's guess is filtered - that's the "which word" decision the stopword
            # heuristic targets. Any remaining sub-word pieces still use the plain (unfiltered)
            # argmax, same as the primary metric, since filtering isn't meaningful once the
            # first piece has already picked a specific word.
            if stopword_id_tensor is not None:
                first_logits = target_logits[0].clone()
                first_logits[stopword_id_tensor] = float("-inf")
                filtered_first = int(first_logits.argmax())
            else:
                filtered_first = int(pred_ids[0])
            filtered_pred_ids = pred_ids.clone()
            filtered_pred_ids[0] = filtered_first
            correct_filtered = torch.equal(filtered_pred_ids[: len(true_ids)], true_ids)
            n_correct_filtered += int(correct_filtered)
            n_scored += 1

            # Perplexity/BPB-style score on the target word's tokens - a continuous signal
            # that still differentiates arms even where strict exact-match saturates to 0
            # (LAMBADA is a notoriously hard, near-binary benchmark for small/undertrained
            # models; a floored accuracy metric would be exactly as uninformative here as
            # MMLU/GSM8K are at this scale - this is the fix for that, not a new problem).
            step_nats = torch.nn.functional.cross_entropy(
                target_logits[: len(true_ids)], true_ids, reduction="sum"
            )
            total_nats += step_nats.item()
            total_target_tokens += len(true_ids)

            if n_scored % 100 == 0:
                running_ppl = math.exp(total_nats / max(total_target_tokens, 1))
                print(f"  {n_scored}/{n_examples}  running accuracy={n_correct/n_scored:.4f}  "
                      f"target-token ppl={running_ppl:.1f}", flush=True)

    accuracy = n_correct / max(n_scored, 1)
    accuracy_stopword_filtered = n_correct_filtered / max(n_scored, 1)
    target_ppl = math.exp(total_nats / max(total_target_tokens, 1))
    target_bpb = (total_nats / LN2) / max(total_target_tokens, 1)  # nats->bits per target TOKEN, not byte -
    # LAMBADA targets are single words of varying byte length, so this is bits/token not a
    # true bits/byte; still directly comparable across arms since it's the same target-token
    # definition for everyone, same spirit as why plain BPB divides by bytes instead of tokens
    # for domain text - here the "unit" being predicted (one target word) is already fixed.
    print(f"\nLAMBADA for {arm} (checkpoint step {ckpt['step']}): "
          f"exact-match accuracy={accuracy:.4f}  stop-word-filtered accuracy={accuracy_stopword_filtered:.4f}  "
          f"({n_correct}/{n_scored} scored, {n_skipped} skipped)  "
          f"target-token perplexity={target_ppl:.2f}  bits/target-token={target_bpb:.3f}", flush=True)
    return {"accuracy": accuracy, "accuracy_stopword_filtered": accuracy_stopword_filtered,
            "target_token_ppl": target_ppl, "bits_per_target_token": target_bpb,
            "n_scored": n_scored, "n_skipped": n_skipped}


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=1800,
              secrets=[modal.Secret.from_name("huggingface-token")])
def diagnose_lambada(arm: str = "routed8", checkpoint_step: int = 575000, n_examples: int = 300):
    """The diagnostic Qwen-coder and this session both flagged as missing before committing
    real GPU-weeks to bets 1-3: classify routed8's LAMBADA errors instead of guessing which
    lever matters most by theory. Cheap (CPU-bound scoring on top of one eval pass, no
    training), and its output should reweight - not replace - the three bets already launched.

    Reuses evaluate_lambada's exact scoring loop (same context/target encoding, same
    teacher-forced first-token prediction), simplified to one hardcoded arm since this is
    specifically about routed8's error mass, not a general per-arm tool. For every WRONG
    first-token prediction, classifies into (in priority order, first match wins):
      - single_token_target: n_tgt==1 - informational tag, not an error class on its own,
        recorded so the coverage stat (see below) is directly checkable against the taxonomy.
      - copy_failure: the target word appears (case-insensitive substring) somewhere earlier
        in the context - the model had the word available to retrieve and didn't. This is
        the error class Bet 1 (copy gate) targets directly.
      - near_miss: the true first target token ranked in the model's own top-5 by logit, just
        not top-1 - a calibration/precision problem (mass spread across synonyms), the error
        class Bet 3 (precision head) targets.
      - other: neither - could be genuine lack of capacity/signal (Bet 2's territory) or a
        rare/unseen word.

    single_token_coverage is reported separately (fraction of ALL scored targets, correct or
    not, that are exactly 1 token under the nlp tokenizer) - this bounds Bet 1/3's ceiling:
    a multi-token target can still be a copy_failure/near_miss on its first piece, but exact-
    match requires every piece right, so low coverage caps how much any single-token-focused
    lever can move the headline number.
    """
    _setup_paths()
    import os

    os.chdir("/root/repo")
    import math

    import torch
    from datasets import load_dataset

    from src.data.build_examples import TokenizerBundle
    from src.model.mot_routed_model import MoTRoutedModel
    from src.model.stage2_config import DOMAIN_TAG, MODEL_CFG

    device = "cuda"
    bundle = TokenizerBundle(
        tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2",
        nlp_tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2_owt/nlp",
    )
    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{arm}_step{checkpoint_step}.pt", map_location=device)
    model = MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {arm} checkpoint at step {ckpt['step']}", flush=True)

    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test", streaming=True)
    counts = {"single_token_target": 0, "copy_failure": 0, "near_miss": 0, "other": 0, "correct": 0}
    n_scored, n_skipped = 0, 0
    examples: list[dict] = []  # a handful of concrete examples per class, for manual spot-checking
    with torch.no_grad():
        for row in ds:
            if n_scored >= n_examples:
                break
            text = row["text"]
            if " " not in text.strip():
                n_skipped += 1
                continue
            context, target_word = text.rsplit(" ", 1)
            tagged_context = f"{DOMAIN_TAG['nlp']}\n{context} "
            ctx_ids, ctx_types = bundle.encode_domain("nlp", tagged_context, max_len=10**9)
            tgt_ids, tgt_types = bundle.encode_domain("nlp", target_word, max_len=10**9)
            n_ctx, n_tgt = len(ctx_ids), len(tgt_ids)
            if n_tgt == 0 or n_ctx == 0 or n_ctx + n_tgt > MODEL_CFG["max_seq_len"]:
                n_skipped += 1
                continue

            full_ids = torch.cat([ctx_ids, tgt_ids]).unsqueeze(0).to(device)
            full_types = torch.cat([ctx_types, tgt_types]).unsqueeze(0).to(device)
            di = domain_index["nlp"]
            with torch.autocast("cuda"):
                dom = torch.full_like(full_ids[:, :-1], di)
                ctrl = torch.zeros_like(full_ids[:, :-1])
                out = model(full_ids[:, :-1], dom, ctrl, targets=None, type_ids=full_types[:, :-1])
                _, logits = out["nlp"]

            first_logits = logits[n_ctx - 1]
            true_first = int(full_ids[0, n_ctx])
            pred_first = int(first_logits.argmax())
            n_scored += 1

            if n_tgt == 1:
                counts["single_token_target"] += 1

            if pred_first == true_first:
                counts["correct"] += 1
                continue

            if target_word.strip(".,;:!?\"'").lower() in context.lower():
                cls = "copy_failure"
            elif true_first in first_logits.topk(5).indices.tolist():
                cls = "near_miss"
            else:
                cls = "other"
            counts[cls] += 1
            if sum(1 for e in examples if e["class"] == cls) < 5:
                examples.append({
                    "class": cls, "target": target_word,
                    "context_tail": context[-120:],
                })

            if n_scored % 50 == 0:
                print(f"  {n_scored}/{n_examples}  running: {counts}", flush=True)

    n_wrong = n_scored - counts["correct"]
    single_token_coverage = counts["single_token_target"] / max(n_scored, 1)
    print(f"\nDIAGNOSTIC for {arm} (checkpoint step {ckpt['step']}), {n_scored} scored ({n_skipped} skipped):",
          flush=True)
    print(f"  single_token_coverage: {single_token_coverage:.3f} "
          f"({counts['single_token_target']}/{n_scored} targets are exactly 1 nlp-tokenizer token)", flush=True)
    print(f"  correct: {counts['correct']}/{n_scored} ({counts['correct']/max(n_scored,1):.3f})", flush=True)
    for cls in ("copy_failure", "near_miss", "other"):
        pct_of_wrong = counts[cls] / max(n_wrong, 1)
        print(f"  {cls}: {counts[cls]}/{n_wrong} wrong ({pct_of_wrong:.3f}) -> "
              f"{'Bet 1 (copy gate)' if cls == 'copy_failure' else 'Bet 3 (precision head)' if cls == 'near_miss' else 'Bet 2 / capacity or rare-word'}",
              flush=True)
    print("\nspot-check examples:", flush=True)
    for e in examples:
        print(f"  [{e['class']}] target={e['target']!r}  ...{e['context_tail']!r}", flush=True)

    return {"n_scored": n_scored, "single_token_coverage": single_token_coverage, "counts": counts}


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=900,
              secrets=[modal.Secret.from_name("huggingface-token")])
def generate(arm: str = "mot", checkpoint_step: int = 150000, seed_domain: str = "science",
             max_new_tokens: int = 160, temperature: float = 0.8, n_samples: int = 3):
    """Autoregressive sampling - the "does it actually produce text, and does routed actually
    SWITCH domains" diagnostic. Not a demo: at 89M params on this corpus the text is
    GPT-2-small-rough at best. The signal we're after is behavioral - for switching arms,
    whether the model ever emits a "switch to domain k" token and changes register mid-stream
    (switch-prediction accuracy being at chance says it can't PREDICT switches; this shows
    whether it ever ACTS on the mechanism at all).

    Decode: code/math/science BPE and baseline/sota decode cleanly. The nlp hybrid's
    surface->syllable->morpheme->byte backoff is lossy, so nlp spans are shown as
    "<nlp: N tokens>" rather than decoded to garbage. seed_domain defaults to science (clean
    decode, and the domain that compresses best, so the least-incoherent starting point).
    """
    _setup_paths()
    import os

    os.chdir("/root/repo")
    import torch
    import torch.nn.functional as F

    from src.data.build_examples import TokenizerBundle
    from src.model.baseline_model import BaselineModel
    from src.model.mot_hybrid_model import MoTHybridModel
    from src.model.mot_model import MoTModel
    from src.model.mot_pooled2_model import MoTPooled2Model
    from src.model.mot_pooled_model import MoTPooledModel
    from src.model.mot_routed_copygate_model import MoTRoutedCopyGateModel
    from src.model.mot_routed_model import MoTRoutedModel
    from src.model.stage2_config import (
        BACKBONE_ONLY_CFG, CONFIDENCE_WEIGHT, COPYGATE_V2_BIAS_INIT, FOCAL_GAMMA, MODEL_CFG,
    )

    device = "cuda"
    seq_len = MODEL_CFG["max_seq_len"]
    # OWT_TOKENIZER_ARMS (see train_stage2_pod.py) - everything sourcing nlp from OpenWebText
    # with routed8's own OWT-fit tokenizer, not the default tokenizers_stage2 dir. Missing
    # this for any of these arms means the nlp vocab size in `bundle` doesn't match the
    # checkpoint's - load_state_dict would fail outright, not silently misbehave.
    owt_tok_arms = ("routed7", "routed8", "routed11", "routed12", "routed13", "routed14",
                     "routed15", "routed16", "routed17", "routed18", "routed19",
                     "routed20", "routed21", "routed22", "routed23")
    bundle = TokenizerBundle(
        tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2",
        nlp_tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2_owt/nlp" if arm in owt_tok_arms else None,
    )
    ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{arm}_step{checkpoint_step}.pt", map_location=device)
    domains = list(bundle.domain_vocab_sizes)
    domain_index = {d: i for i, d in enumerate(domains)}
    copygate_arms = ("routed11", "routed14", "routed16", "routed20", "routed21", "routed22")
    switching = arm in ("routed", "pooled", "hybrid", "pooled2", "routed2", "routed3", "routed7",
                         "routed8", "routed17", "routed18", "routed19", "routed23") + copygate_arms
    domain_routed = arm == "mot" or switching

    if arm in copygate_arms:
        model = MoTRoutedCopyGateModel(
            domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
            gate_bias_init=COPYGATE_V2_BIAS_INIT if arm == "routed16" else 0.0,
        ).to(device)
    elif arm == "mot":
        model = MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm in ("routed", "routed7", "routed8", "routed17", "routed18", "routed19", "routed23"):
        model = MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled":
        model = MoTPooledModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                               focal_gamma=FOCAL_GAMMA, confidence_weight=CONFIDENCE_WEIGHT).to(device)
    elif arm in ("hybrid", "routed2", "routed3"):
        model = MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    elif arm == "pooled2":
        model = MoTPooled2Model(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                focal_gamma=FOCAL_GAMMA).to(device)
    elif arm == "baseline":
        model = BaselineModel(vocab_size=bundle.baseline_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    else:
        model = BaselineModel(vocab_size=bundle.sota_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {arm} checkpoint at step {ckpt['step']}\n", flush=True)

    def decode_span(domain, ids):
        if domain == "nlp":
            return f"⟨nlp:{len(ids)}tok⟩"
        return bundle.bpe[domain].decode(ids)

    for s in range(n_samples):
        print(f"===== {arm} sample {s+1}/{n_samples} (seed domain: {seed_domain}) =====", flush=True)
        torch.manual_seed(1000 + s)
        if not domain_routed:
            # baseline/sota: seed with the domain tag they trained on, then free-run
            tag = f"<domain:{seed_domain}>\n"
            enc = bundle.encode_baseline if arm == "baseline" else bundle.encode_sota
            ids = enc(tag, max_len=10**9).tolist()
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    inp = torch.tensor(ids[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
                    with torch.autocast("cuda"):
                        logits = model(inp)[0, -1]
                    probs = F.softmax(logits / temperature, dim=-1)
                    ids.append(int(torch.multinomial(probs, 1)))
            dec = bundle.baseline.decode(ids) if arm == "baseline" else bundle.sota.decode(ids)
            print(dec.replace("\n", "\\n") + "\n", flush=True)
            continue

        # domain-routed: track (token, domain, is_control) as we go
        cur = seed_domain
        toks, doms, ctrls, typs = [], [], [], []
        # seed with one control token establishing the domain
        toks.append(domain_index[cur]); doms.append(domain_index[cur]); ctrls.append(1); typs.append(0)
        switch_events = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                t_tok = torch.tensor(toks[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
                t_dom = torch.tensor(doms[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
                t_ctrl = torch.tensor(ctrls[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
                t_typ = torch.tensor(typs[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
                with torch.autocast("cuda"):
                    if arm == "mot":
                        logits = model(cur, t_tok, t_typ)[0, -1]
                    else:
                        out = model(t_tok, t_dom, t_ctrl, targets=None, type_ids=t_typ)
                        logits = out[cur][1][-1]
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
                vocab_d = bundle.domain_vocab_sizes[cur]
                if arm != "mot" and nxt >= vocab_d:  # switch-to-domain-k token
                    k = nxt - vocab_d
                    if k < len(domains):
                        new_dom = domains[k]
                        switch_events.append((len(toks), cur, new_dom))
                        cur = new_dom
                        toks.append(domain_index[cur]); doms.append(domain_index[cur]); ctrls.append(1); typs.append(0)
                        continue
                    nxt = nxt % vocab_d  # out-of-range switch slot: clamp to a real token
                toks.append(nxt); doms.append(domain_index[cur]); ctrls.append(0); typs.append(0)

        # decode contiguous same-domain non-control spans
        pieces, span_ids, span_dom = [], [], domains[doms[0]]
        for tok, dom_i, ctrl in zip(toks, doms, ctrls):
            dom = domains[dom_i]
            if ctrl:
                if span_ids:
                    pieces.append(decode_span(span_dom, span_ids)); span_ids = []
                pieces.append(f" 〔switch→{dom}〕 ")
                span_dom = dom
                continue
            if dom != span_dom and span_ids:
                pieces.append(decode_span(span_dom, span_ids)); span_ids = []
            span_dom, _ = dom, span_ids.append(tok)
        if span_ids:
            pieces.append(decode_span(span_dom, span_ids))
        print("".join(pieces).replace("\n", "\\n"), flush=True)
        print(f"  [switches emitted: {len(switch_events)}]" +
              (f" {switch_events}" if switch_events else " — stayed in seed domain the whole time") + "\n", flush=True)

    return {"arm": arm, "switching": switching}


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=900,
              secrets=[modal.Secret.from_name("huggingface-token")])
def diagnose_gradient_conflict(arm: str = "routed25", checkpoint_step: int = 283000,
                                n_batches: int = 30, scale: str = "base"):
    """Cheap diagnostic (~30 held-out batches, no full eval sweep): does the switch-prediction
    auxiliary loss fight the main next-token LM loss for capacity in the shared backbone?
    Real, established technique (gradient-conflict detection, cf. PCGrad/Yu et al. 2020) -
    split every batch's per-position loss into "predict a real next token" vs "predict a
    domain switch" (same mask head_loss already uses internally: target id >= domain vocab
    size), backward each separately, and measure cosine similarity of the resulting gradients
    on shared BACKBONE params only (heads/embeddings are domain-specific, not shared, so
    conflict there is meaningless - every domain already has its own). Negative cosine means
    the two objectives are genuinely pulling backbone weights in opposite directions and
    gradient surgery (dropping the conflicting component) would likely help; near-zero means
    independent; positive means reinforcing."""
    _setup_paths()
    import os

    os.chdir("/root/repo")
    import statistics

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.utils.data import DataLoader, IterableDataset

    from src.data.build_examples import TokenizerBundle
    from src.model.mot_routed_copygate_model import MoTRoutedCopyGateModel
    from src.model.stage2_config import LARGE_MODEL_CFG, MODEL_CFG, STREAM_SOURCES

    device = "cuda"
    if arm == "routed28":
        MODEL_CFG = LARGE_MODEL_CFG
        scale = "large"
    elif scale == "large":
        MODEL_CFG = LARGE_MODEL_CFG
    STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    ckpt_prefix = f"large_{arm}" if scale == "large" else arm
    SKIP_DOCS = 300_000
    seq_len = MODEL_CFG["max_seq_len"]

    def held_out_body_stream(domain):
        from src.data.stage2_stream_dataset import TEXT_EXTRACTORS

        cfg = STREAM_SOURCES[domain]
        stream = load_dataset(cfg["path"], name=cfg.get("name"), revision=cfg.get("revision"),
                               data_files=cfg.get("data_files"), split="train", streaming=True,
                               trust_remote_code=True).skip(SKIP_DOCS)
        extractor = TEXT_EXTRACTORS[domain]
        for row in stream:
            text = extractor(row)
            if text:
                yield text

    def held_out_synthetic_multidomain_stream(seed=0):
        import random

        from src.data.stage2_routed_stream import MAX_DOMAINS_PER_DOC, MIN_DOMAINS_PER_DOC, SNIPPET_WORDS
        from src.model.stage2_config import DOMAIN_TAG

        rng = random.Random(seed)
        domains = list(STREAM_SOURCES)
        body_streams = {d: held_out_body_stream(d) for d in domains}
        while True:
            k = rng.randint(MIN_DOMAINS_PER_DOC, MAX_DOMAINS_PER_DOC)
            chosen = rng.sample(domains, k)
            parts = []
            for domain in chosen:
                text = " ".join(next(body_streams[domain]).split()[:SNIPPET_WORDS])
                parts.append(f"{DOMAIN_TAG[domain]}\n{text}\n")
            yield "".join(parts)

    class HeldOutRoutedStream(IterableDataset):
        def __init__(self, bundle, domain_index, seq_len):
            self.bundle, self.domain_index, self.seq_len = bundle, domain_index, seq_len
            self.domains = list(domain_index)

        def __iter__(self):
            from src.data.stage2_routed_stream import _split_spans

            buf_tok, buf_dom, buf_ctrl, buf_typ = [], [], [], []
            for doc in held_out_synthetic_multidomain_stream():
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
                    c_tok, c_dom, c_ctrl = buf_tok[:window], buf_dom[:window], buf_ctrl[:window]
                    c_typ = buf_typ[:window]
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
                    buf_tok = buf_tok[window:]; buf_dom = buf_dom[window:]
                    buf_ctrl = buf_ctrl[window:]; buf_typ = buf_typ[window:]

    bundle = TokenizerBundle(
        tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2",
        nlp_tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2_owt/nlp",
    )
    ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{ckpt_prefix}_step{checkpoint_step}.pt", map_location=device)
    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    model = MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_prefix} checkpoint at step {ckpt['step']}", flush=True)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    loader = iter(DataLoader(HeldOutRoutedStream(bundle, domain_index, seq_len), batch_size=8))

    cosines = []
    for b in range(n_batches):
        tok, dom, ctrl, typ, tgt = next(loader)
        tok, dom, ctrl, typ, tgt = (t.to(device) for t in (tok, dom, ctrl, typ, tgt))
        x = model.embed_sequence(tok, dom, ctrl, typ)
        h = model.backbone(x)

        main_losses, switch_losses = [], []
        for domain in model.domains:
            mask = dom == model.domain_index[domain]
            if not mask.any():
                continue
            dtgt = tgt[mask]
            if domain != "nlp":
                logits = model.heads[domain](h[mask])
                per_pos = F.cross_entropy(logits, dtgt, reduction="none")
            else:
                p_mix = model._nlp_copy_gate_pmix(h, dom, ctrl, tok).clamp_min(1e-9)
                per_pos = -torch.log(p_mix.gather(1, dtgt.unsqueeze(-1)).squeeze(-1))
            is_switch = dtgt >= model.domain_vocab_sizes[domain]
            if (~is_switch).any():
                main_losses.append(per_pos[~is_switch])
            if is_switch.any():
                switch_losses.append(per_pos[is_switch])

        if not main_losses or not switch_losses:
            continue
        main_loss = torch.cat(main_losses).mean()
        switch_loss = torch.cat(switch_losses).mean()

        model.zero_grad(set_to_none=True)
        main_loss.backward(retain_graph=True)
        g_main = torch.cat([p.grad.detach().flatten() for p in backbone_params if p.grad is not None]).clone()

        model.zero_grad(set_to_none=True)
        switch_loss.backward()
        g_switch = torch.cat([p.grad.detach().flatten() for p in backbone_params if p.grad is not None]).clone()

        cos = F.cosine_similarity(g_main.unsqueeze(0), g_switch.unsqueeze(0)).item()
        cosines.append(cos)
        print(f"  batch {b+1}/{n_batches}  main={main_loss.item():.3f}  switch={switch_loss.item():.3f}  "
              f"cos={cos:.4f}", flush=True)

    mean_cos = statistics.mean(cosines)
    frac_negative = sum(1 for c in cosines if c < 0) / len(cosines)
    print(f"\nGRADIENT CONFLICT for {arm} (checkpoint step {ckpt['step']}), {len(cosines)} batches:", flush=True)
    print(f"  mean cosine(main_grad, switch_grad) on shared backbone: {mean_cos:.4f}", flush=True)
    print(f"  fraction of batches with negative cosine (real conflict): {frac_negative:.3f}", flush=True)
    verdict = ("REAL CONFLICT - gradient surgery would likely help" if mean_cos < -0.05 else
               "MOSTLY INDEPENDENT - no strong conflict" if abs(mean_cos) <= 0.05 else
               "REINFORCING - switch task is not fighting the LM loss")
    print(f"  verdict: {verdict}", flush=True)
    return {"mean_cosine": mean_cos, "frac_negative": frac_negative, "n_batches": len(cosines)}


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=900,
              secrets=[modal.Secret.from_name("huggingface-token")])
def diagnose_calibration(arm: str = "routed25", checkpoint_step: int = 283000,
                          n_batches: int = 40, scale: str = "base"):
    """Cheap diagnostic: is model confidence calibrated? For every predicted token, bucket by
    the model's own max-softmax confidence and check whether the argmax was actually correct
    in that bucket. A well-calibrated model's 80%-confidence bucket is ~80% accurate - this is
    a genuinely different failure mode than "needs more scale," invisible to BPB/ppl/EM alone.
    Reports Expected Calibration Error (ECE) per domain, held-out, teacher-forced (same
    single-domain streams evaluate() uses, just with per-token confidence tracked instead of
    only aggregate loss)."""
    _setup_paths()
    import os

    os.chdir("/root/repo")
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.utils.data import DataLoader, IterableDataset

    from src.data.build_examples import TokenizerBundle
    from src.model.mot_routed_copygate_model import MoTRoutedCopyGateModel
    from src.model.stage2_config import LARGE_MODEL_CFG, MODEL_CFG, STREAM_SOURCES

    device = "cuda"
    if arm == "routed28":
        MODEL_CFG = LARGE_MODEL_CFG
        scale = "large"
    elif scale == "large":
        MODEL_CFG = LARGE_MODEL_CFG
    STREAM_SOURCES["nlp"] = {"path": "Skylion007/openwebtext", "name": None, "gated": False}
    ckpt_prefix = f"large_{arm}" if scale == "large" else arm
    SKIP_DOCS = 300_000
    seq_len = MODEL_CFG["max_seq_len"]
    n_bins = 10

    def held_out_doc_stream(domain):
        from src.data.stage2_stream_dataset import DOC_SEP, TEXT_EXTRACTORS
        from src.model.stage2_config import DOMAIN_TAG

        cfg = STREAM_SOURCES[domain]
        stream = load_dataset(cfg["path"], name=cfg.get("name"), revision=cfg.get("revision"),
                               data_files=cfg.get("data_files"), split="train", streaming=True,
                               trust_remote_code=True).skip(SKIP_DOCS)
        extractor = TEXT_EXTRACTORS[domain]
        tag = DOMAIN_TAG[domain]
        for row in stream:
            text = extractor(row)
            if text:
                yield f"{tag}\n{text}{DOC_SEP}"

    class HeldOutDomainStream(IterableDataset):
        def __init__(self, domain, encode_domain_fn, seq_len):
            self.domain, self.encode_domain_fn, self.seq_len = domain, encode_domain_fn, seq_len

        def __iter__(self):
            buf_ids, buf_types = [], []
            has_types = self.domain == "nlp"
            for text in held_out_doc_stream(self.domain):
                ids, types = self.encode_domain_fn(self.domain, text, max_len=10**9)
                buf_ids.extend(ids.tolist())
                if has_types:
                    buf_types.extend(types.tolist())
                while len(buf_ids) >= self.seq_len + 1:
                    chunk_ids = torch.tensor(buf_ids[: self.seq_len + 1], dtype=torch.long)
                    chunk_types = (torch.tensor(buf_types[: self.seq_len + 1], dtype=torch.long)
                                   if has_types else torch.zeros(self.seq_len + 1, dtype=torch.long))
                    yield chunk_ids, chunk_types
                    buf_ids = buf_ids[self.seq_len + 1:]
                    if has_types:
                        buf_types = buf_types[self.seq_len + 1:]

    bundle = TokenizerBundle(
        tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2",
        nlp_tokenizer_dir=f"{VOLUME_PATH}/tokenizers_stage2_owt/nlp",
    )
    ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{ckpt_prefix}_step{checkpoint_step}.pt", map_location=device)
    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    model = MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_prefix} checkpoint at step {ckpt['step']}", flush=True)

    results = {}
    per_domain_batches = max(1, n_batches // len(bundle.domain_vocab_sizes))
    with torch.no_grad():
        for domain in bundle.domain_vocab_sizes:
            di = domain_index[domain]
            loader = iter(DataLoader(HeldOutDomainStream(domain, bundle.encode_domain, seq_len), batch_size=8))
            bin_correct = [0] * n_bins
            bin_total = [0] * n_bins
            bin_conf_sum = [0.0] * n_bins
            for _ in range(per_domain_batches):
                ids, types = next(loader)
                ids, types = ids.to(device), types.to(device)
                inp, tgt = ids[:, :-1], ids[:, 1:]
                dom = torch.full_like(inp, di)
                ctrl = torch.zeros_like(inp)
                with torch.autocast("cuda"):
                    x = model.embed_sequence(inp, dom, ctrl, types[:, :-1])
                    h = model.backbone(x)
                    if domain != "nlp":
                        probs = F.softmax(model.heads[domain](h), dim=-1)
                    else:
                        p_mix = model._nlp_copy_gate_pmix(h, dom, ctrl, inp)
                        probs = p_mix.reshape(h.shape[0], h.shape[1], -1)
                    conf, pred = probs.max(dim=-1)
                conf_flat, pred_flat, tgt_flat = conf.reshape(-1), pred.reshape(-1), tgt.reshape(-1)
                correct = pred_flat == tgt_flat
                bins = (conf_flat.clamp(0, 0.9999) * n_bins).long()
                for bidx in range(n_bins):
                    m = bins == bidx
                    if m.any():
                        bin_total[bidx] += int(m.sum())
                        bin_correct[bidx] += int(correct[m].sum())
                        bin_conf_sum[bidx] += conf_flat[m].sum().item()

            total = sum(bin_total)
            ece = 0.0
            print(f"\n{domain} calibration ({total} predictions):", flush=True)
            for bidx in range(n_bins):
                if bin_total[bidx] == 0:
                    continue
                acc = bin_correct[bidx] / bin_total[bidx]
                avg_conf = bin_conf_sum[bidx] / bin_total[bidx]
                gap = abs(avg_conf - acc)
                ece += (bin_total[bidx] / total) * gap
                print(f"  conf[{bidx/n_bins:.1f}-{(bidx+1)/n_bins:.1f}]: n={bin_total[bidx]:5d}  "
                      f"avg_conf={avg_conf:.3f}  acc={acc:.3f}  gap={gap:.3f}", flush=True)
            print(f"  Expected Calibration Error (ECE): {ece:.4f}", flush=True)
            results[domain] = ece

    print(f"\nCALIBRATION SUMMARY for {arm} (checkpoint step {ckpt['step']}):", flush=True)
    for d, e in results.items():
        print(f"  {d}: ECE={e:.4f}", flush=True)
    return results


@app.local_entrypoint()
def main(step: str = "calibrate", arm: str = "mot", steps: int = 0, resume_from: str = "", noisy: bool = False, scale: str = "base", n_examples: int = 500):
    """steps=0 means "use the default": 150 for calibrate, MAX_STEPS for train."""
    if step == "sample-tokenizers":
        sample_tokenizers.remote()
    elif step == "train-tokenizers":
        train_tokenizers.remote()
    elif step == "train-shrunk-tokenizers":
        train_shrunk_vocab_tokenizers.remote(vocab_size=steps or 10000)
    elif step == "train-generalist-tokenizer":
        train_generalist_tokenizer.remote(vocab_size=steps or 32000)
    elif step == "calibrate":
        sec_per_step = calibrate.remote(arm=arm, steps=steps or 150)
        from src.model.stage2_config import MAX_STEPS
        print(f"\n--- extrapolation ---")
        print(f"{sec_per_step:.3f} sec/step x {MAX_STEPS} steps = {sec_per_step*MAX_STEPS/3600:.2f} GPU-hours")
        print(f"at ~$0.59/hr (T4): ~${sec_per_step*MAX_STEPS/3600*0.59:.2f} for this arm")
    elif step == "evaluate":
        result = evaluate.remote(arm=arm, checkpoint_step=steps or 20000, noisy=noisy, scale=scale)
        mode = "noisy" if noisy else "clean"
        print(f"\nheld-out single-domain BPB for {arm} ({mode}): {result['single_domain_bpb']:.4f} bits/byte")
        if result["cross_domain_bpb"] is not None:
            print(f"held-out cross-domain BPB for {arm} ({mode}): {result['cross_domain_bpb']:.4f} bits/byte")
            print(f"switch-prediction accuracy: {result['switch_accuracy']:.4f}")
    elif step == "train":
        history = train.remote(arm=arm, max_steps=steps or None, resume_from=resume_from or None)
        print(f"\nfinal logged losses: {history[-5:] if history else '(none)'}")
    elif step == "evaluate-lambada":
        result = evaluate_lambada.remote(arm=arm, checkpoint_step=steps or 150000, scale=scale, n_examples=n_examples)
        print(f"\nLAMBADA for {arm}: accuracy={result['accuracy']:.4f}  "
              f"stop-word-filtered accuracy={result['accuracy_stopword_filtered']:.4f}  "
              f"target-token ppl={result['target_token_ppl']:.2f}  "
              f"bits/target-token={result['bits_per_target_token']:.3f}  "
              f"(n_scored={result['n_scored']}, n_skipped={result['n_skipped']})")
    elif step == "generate":
        generate.remote(arm=arm, checkpoint_step=steps or 150000, seed_domain=resume_from or "science")
    elif step == "diagnose-lambada":
        diagnose_lambada.remote(arm=arm or "routed8", checkpoint_step=steps or 575000)
    elif step == "diagnose-gradient-conflict":
        diagnose_gradient_conflict.remote(arm=arm, checkpoint_step=steps or 283000, scale=scale)
    elif step == "diagnose-calibration":
        diagnose_calibration.remote(arm=arm, checkpoint_step=steps or 283000, scale=scale)
    else:
        raise ValueError(f"unknown step: {step}")
