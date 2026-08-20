"""Stage-2 GPU training, ported from stage2_modal.py for a plain RunPod pod (no Modal
wrapper). Same model dispatch / resume / adaptive-controller / BPB-eval logic as the
Modal version - only the storage layer changed (local disk under this repo instead of
a Modal Volume) and the @app.function/.remote() plumbing is gone.

Usage (run from the repo root on the pod):
  python3 scripts/train_stage2_pod.py train --arm mot --steps 150000
  python3 scripts/train_stage2_pod.py calibrate --arm mot --steps 150
  python3 scripts/train_stage2_pod.py evaluate --arm mot --checkpoint-step 121000
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.data.build_examples import TokenizerBundle
from src.data.stage2_routed_stream import PackedRoutedStream
from src.data.stage2_stream_dataset import PackedDomainStream, PackedMixedStream
from src.model.baseline_model import BaselineModel
from src.model.mot_hybrid_model import MoTHybridModel
from src.model.mot_model import MoTModel
from src.model.mot_pooled2_model import MoTPooled2Model
from src.model.mot_pooled_model import MoTPooledModel
from src.model.mot_routed_combined_model import MoTRoutedCombinedModel
from src.model.mot_routed_copygate_model import MoTRoutedCopyGateModel
from src.model.mot_routed_decoupled_model import MoTRoutedDecoupledModel
from src.model.mot_routed_deepexpert_model import DEFAULT_N_EXPERT_LAYERS, MoTRoutedDeepExpertModel
from src.model.mot_routed_model import MoTRoutedModel
from src.model.mot_routed_tied_model import MoTRoutedTiedModel
from src.model.backbone_modern import ModernBackbone
from src.model.mot_routed_precision_model import MoTRoutedPrecisionModel
from src.model.stage2_config import (
    ADV_LAMBDA_RAMP_STEPS, ALIGN_ARMS, ALIGN_LOSS_EVERY, ALIGN_LOSS_WEIGHT, ARM_LABELS,
    BACKBONE_ONLY_CFG, BATCH_SIZE, BET_BACKBONE_LR_SCALE,
    BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS, CHECKPOINT_EVERY, CONFIDENCE_WEIGHT, COOLDOWN_BACKBONE_LR_SCALE,
    COPY_MINE_MIN_GAP, COPY_MINE_MIN_WORD_LEN, COPYGATE_V2_BIAS_INIT, DIET_PHASE2_NLP_SNIPPET_WORDS,
    FOCAL_GAMMA, FROZEN_BACKBONE_ARMS, GRAD_ACCUM_STEPS, HYBRID_NATURAL_DATA_FRACTION, LOG_EVERY,
    LONGCTX_MODEL_CFG, LR, MAX_STEPS, MODEL_CFG, NEW_MODULE_MATCH, PER_ARM_BACKBONE_LR_SCALE,
    ROUTED3_MAX_DOMAINS,
    ROUTED3_MIN_DOMAINS, ROUTED3_SNIPPET_WORDS, ROUTED19_BATCH_SIZE, ROUTED19_GRAD_ACCUM_STEPS,
    ROUTED19_PHASE1_FRACTION, ROUTED19_PHASE1_SNIPPET_WORDS, ROUTED19_PHASE2_CACHE_PASS_OFFSET,
    ROUTED26_NLP_SNIPPET_WORDS,
    SWITCH_WEIGHT, WARM_START_PARENT,
    WARMUP_STEPS,
)

BET_ARMS = ("routed11", "routed12", "routed13", "routed14", "routed15", "routed16")
# copy-gate, deep-experts, precision-head, scaled-up copy-gate, control, copy-gate-v2-frozen -
# all use bare PackedRoutedStream + the NEW_MODULE_MATCH-driven differential-LR path.
LARGE_BET_ARMS = ("routed14", "routed28", "routed33", "routed35")  # subset that forces scale="large" (see _build_model)
DIET_ARMS = ("routed17", "routed18")  # round-2 data-lever arms: force_domain="nlp" upweighting,
# same mechanism as routed9/10 but no reinit (nlp-vs-rest differential LR via WARM_START_PARENT)
# routed20/21/22/23: the copy-gate-fix-night four-way ablation (see ALIGN_ARMS's comment block
# in stage2_config.py for the full rationale). RECIPE_ARMS is every one of them (for the
# generic step-loop dispatch, which is identical to BET_ARMS/DIET_ARMS's shape - plain
# model(tok, dom, ctrl, targets=tgt, ...) call); RECIPE_DIET_ARMS is the subset using the
# diet (nlp-upweighted) loader rather than the plain one.
RECIPE_ARMS = ("routed20", "routed21", "routed22", "routed23", "routed24",
                "routed25", "routed26", "routed27", "routed28", "routed29",
                "routed30", "routed31", "routed32", "routed33", "routed35")
RECIPE_DIET_ARMS = ("routed20", "routed21", "routed25", "routed26", "routed27", "routed28", "routed29",
                     "routed30", "routed31", "routed32", "routed33", "routed35")
OWT_TOKENIZER_ARMS = ("routed7", "routed8", "routed19") + BET_ARMS + DIET_ARMS + RECIPE_ARMS  # everything
# sourcing nlp from OpenWebText with routed8's own OWT-fit tokenizer (routed9/10 use PG-19 books instead)

# Every DataLoader in this file used to default to num_workers=0 - synchronous, single-
# process data loading, meaning the GPU sat idle every micro-step while Python fetched +
# tokenized the next document from a network-streamed HF dataset, with zero overlap between
# I/O and compute. NUM_WORKERS>0 background-prefetches while the GPU computes the current
# batch; PIN_MEMORY speeds up the host->device copy. Safe to apply everywhere (not just
# routed19) since the *_raw_doc_stream/_raw_body_stream sharding fix (get_worker_info-based,
# see stage2_stream_dataset.py/stage2_routed_stream.py) makes every existing single-worker
# call site (num_workers=0 elsewhere) behave identically to before - this only changes
# throughput, never what data is seen for a given num_workers value.
NUM_WORKERS = 4
PREFETCH_FACTOR = 4


def _ddp_setup() -> tuple[int, int, int, str]:
    """Returns (rank, world_size, local_rank, device). Reads torchrun's env vars (RANK/
    WORLD_SIZE/LOCAL_RANK) - the standard env://-based rendezvous, no manual multiprocessing
    spawn. No-ops (rank=0, world_size=1, device="cuda") when launched with plain `python3`
    (every launch this project has used before tonight), so this is purely additive - a
    single-GPU launch is byte-identical to before."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return 0, 1, 0, "cuda"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, f"cuda:{local_rank}"


def _loader(dataset, batch_size: int = None, world_size: int = 1):
    bs = batch_size or BATCH_SIZE
    if world_size > 1:
        # Divide, don't multiply: keeps the TOTAL effective batch (and therefore total steps
        # to reach the same token target, and the LR schedule computed against total_steps)
        # identical to the validated single-GPU recipe - DDP only parallelizes the SAME work
        # across GPUs, it doesn't change what's being trained. Each rank does 1/world_size of
        # the per-step batch; gradients all-reduce (averaged) across ranks on the sync step.
        assert bs % world_size == 0, (
            f"batch_size {bs} must be divisible by world_size {world_size} to keep the "
            f"effective batch unchanged - pick a batch size that divides cleanly."
        )
        bs = bs // world_size
    return DataLoader(
        dataset, batch_size=bs, num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=_worker_init if NUM_WORKERS > 0 else None,
    )


def _worker_init(_worker_id: int) -> None:
    # Each DataLoader worker is its own process; by default torch/numpy's BLAS backends may
    # spawn multiple threads PER WORKER for internal ops. With NUM_WORKERS(4) workers per GPU
    # process and up to world_size GPU processes on one host, that's real oversubscription risk
    # on shared CPU cores (this pipeline's workers do I/O + JSON/string parsing, not linear
    # algebra, so they gain nothing from multi-threading and only cost contention). Restricting
    # each worker to 1 thread is standard practice for CPU-bound-by-many-processes DataLoader
    # workloads like this one.
    torch.set_num_threads(1)


TOKENIZER_DIR = str(REPO_ROOT / "tokenizers_stage2")
# arm in ("routed7", "routed8"): nlp tokenizer retrained on OpenWebText (see
# scripts/retrain_nlp_tokenizer_openwebtext.py). code/math/science still load from
# TOKENIZER_DIR above, unchanged.
OWT_NLP_TOKENIZER_DIR = str(REPO_ROOT / "tokenizers_stage2_owt" / "nlp")
# arm in ("routed9", "routed10"): nlp tokenizer retrained on PG-19 books (see
# scripts/retrain_nlp_tokenizer_books.py).
BOOKS_NLP_TOKENIZER_DIR = str(REPO_ROOT / "tokenizers_stage2_books" / "nlp")
# arm routed30 only: code/math/science retrained at ROUTED30_SHRUNK_VOCAB (see
# stage2_modal.py's train_shrunk_vocab_tokenizers), nlp copied in unchanged from the OWT-fit
# tokenizer - a complete, self-contained tokenizer dir, no nlp_tokenizer_dir override needed.
SHRUNK_TOKENIZER_DIR = str(REPO_ROOT / "tokenizers_stage2_shrunk")
# arm routed33 only: the 5th "generalist" domain's own tokenizer, trained on a pooled sample
# across all four existing domains (see stage2_modal.py's train_generalist_tokenizer). Passed
# as TokenizerBundle's generalist_tokenizer_dir - code/math/science/nlp load from TOKENIZER_DIR
# (+ OWT_NLP_TOKENIZER_DIR for nlp) exactly as routed28 does, unchanged.
GENERALIST_TOKENIZER_DIR = str(REPO_ROOT / "tokenizers_stage2_generalist" / "generalist")
CKPT_DIR = REPO_ROOT / "checkpoints"
# arms routed9/routed10 only: state_dict key prefixes that are nlp-domain-specific and must
# be reinitialized (not warm-started) when switching to a differently-fit nlp tokenizer - see
# _warm_start_from_parent.
_NLP_PARAM_PREFIXES = ("embeddings.nlp.", "type_embeddings.nlp.", "projections.nlp.", "heads.nlp.")


def _hybrid_batch(step: int, domains: list[str], domain_index: dict[str, int],
                   natural_loaders: dict, routed_loader, device: str):
    """arm='hybrid' only: alternates natural single-domain batches (PackedDomainStream, long
    continuous context - what MoT trains on exclusively) with synthetic multi-domain switching
    batches (PackedRoutedStream), at HYBRID_NATURAL_DATA_FRACTION. A natural batch is recast
    into the same (tok, dom, ctrl, typ, tgt) shape MoTHybridModel.forward expects: dom_ids
    constant (one domain for the whole window), is_control all zero (no switches), targets are
    plain next-token ids. Deterministic on `step` so calibrate() and train() see the same mix
    for a given step count, which matters for comparing calibration numbers to the real run.
    """
    import random

    use_natural = random.Random(step).random() < HYBRID_NATURAL_DATA_FRACTION
    if use_natural:
        domain = domains[step % len(domains)]
        _, ids, types = next(natural_loaders[domain])
        ids, types = ids.to(device), types.to(device)
        inp, tgt = ids[:, :-1], ids[:, 1:]
        typ = types[:, :-1]
        dom = torch.full_like(inp, domain_index[domain])
        ctrl = torch.zeros_like(inp)
        return inp, dom, ctrl, typ, tgt
    tok, dom, ctrl, typ, tgt = next(routed_loader)
    return (t.to(device) for t in (tok, dom, ctrl, typ, tgt))


def _apply_openwebtext_nlp_source() -> None:
    """arm in ("routed7", "routed8"): re-point the shared nlp domain to OpenWebTextCorpus (the
    standard open reconstruction of GPT-2's unreleased WebText - see docs/handoff_routed7.md and
    docs/handoff_data_equivalent_runs.md) instead of FineWeb. code/math/science stay on their
    existing rich sources unchanged - OpenWebText
    was measured to contain almost no dense code/math/science content (code ~1-in-185 docs,
    math and science ~1-in-1250 each, over a real 5000-doc sample), so forcing those three
    domains out of it would starve their streams. nlp is the only domain LAMBADA is even
    routed through, so this is the minimal change that actually closes the GPT-2-comparison
    data-source gap.

    Mutates the module-level STREAM_SOURCES dict in place. Safe to do here: each arm is a
    separate process on its own pod, so this has zero effect on any other concurrently
    running arm. Must be called before any stream/DataLoader is constructed - _raw_doc_stream
    /_raw_body_stream read STREAM_SOURCES[domain] at iteration time, not import time, so the
    override only needs to land before the first `next()` call, but doing it immediately
    at startup is simplest and safest.

    OpenWebText's row schema uses the same "text" field FineWeb does (confirmed by direct
    test), so the existing TEXT_EXTRACTORS["nlp"] extractor needs no changes.
    """
    from src.model import stage2_config
    stage2_config.STREAM_SOURCES["nlp"] = {
        "path": "Skylion007/openwebtext", "name": None, "gated": False,
    }
    print("[routed7] nlp domain source overridden: FineWeb -> Skylion007/openwebtext", flush=True)


def _apply_generalist_domain_source() -> None:
    """arms routed33/routed35: registers "generalist" as a 5th domain in STREAM_SOURCES so
    synthetic_multidomain_doc_stream's `domains = list(STREAM_SOURCES)` includes it in the
    per-doc domain pool - without this, PackedRoutedStream's domain_index would list
    "generalist" (from the bundle) but the doc-generation step would never actually EMIT any
    generalist-tagged span, so its embedding/head would silently never see training signal
    despite looking correctly wired.

    Mutates the module-level STREAM_SOURCES dict in place - same safety argument as
    _apply_openwebtext_nlp_source (one process per arm, no cross-arm effect). The "path"
    value here is never actually read in practice: generalist has no HF source of its own,
    it's always served from data_cache/generalist.jsonl (built by build_domain_cache.py's
    build_generalist_cache, pooled from the other 4 domains' already-built local caches, which
    _cached_doc_stream/_raw_body_stream check and prefer before ever falling back to a live
    HF stream). Points at OpenWebText anyway (a real, valid dataset) purely so a fallback
    triggered by a missing cache fails informatively rather than on a bogus path.
    """
    from src.model import stage2_config
    stage2_config.STREAM_SOURCES["generalist"] = {
        "path": "Skylion007/openwebtext", "name": None, "gated": False,
    }
    print("[routed33/35] generalist domain registered (served from local pooled cache)", flush=True)


def _apply_books_nlp_source() -> None:
    """arm in ("routed9", "routed10"): re-point the shared nlp domain to PG-19 (Project
    Gutenberg books) instead of FineWeb/OpenWebText. LAMBADA's passages are themselves
    book-derived, and the long-range antecedent->target copying it tests is a skill short web
    pages rarely train, unlike continuous book narrative - see docs/handoff_optimized_89m_190m.md.
    code/math/science untouched, same rationale as _apply_openwebtext_nlp_source. PG-19's row
    schema uses the same "text" field (confirmed by direct test), so TEXT_EXTRACTORS["nlp"]
    needs no changes.
    """
    from src.model import stage2_config
    stage2_config.STREAM_SOURCES["nlp"] = {
        "path": "deepmind/pg19", "name": None, "gated": False,
    }
    print("[routed9/10] nlp domain source overridden: FineWeb -> deepmind/pg19", flush=True)


def _latest_parent_checkpoint(parent_ckpt_prefix: str):
    import glob
    import re
    paths = glob.glob(str(CKPT_DIR / f"{parent_ckpt_prefix}_step*.pt"))
    if not paths:
        return None
    return sorted(paths, key=lambda p: int(re.search(r"step(\d+)", p).group(1)), reverse=True)[0]


def _warm_start_from_parent(model, parent_ckpt_prefix: str, device: str,
                             skip_prefixes: tuple[str, ...] = _NLP_PARAM_PREFIXES) -> bool:
    """arms routed9/routed10/routed11/routed12/routed13: initialize from the parent arm's
    latest checkpoint instead of random init. skip_prefixes lists key prefixes to leave at
    fresh init rather than load - for routed9/10 that's the nlp domain's embedding/type_
    embedding/projection/head (they describe token ids under a DIFFERENT nlp tokenizer than
    the parent's, so loading them would be confidently wrong, not just stale, even though the
    tensor shapes match). routed11/12/13 reuse routed8's exact tokenizer, so they pass
    skip_prefixes=() - nothing needs skipping, and any genuinely NEW key (a bet's own new
    module, e.g. copy_q/copy_gate) simply isn't in the parent checkpoint at all, so it's
    never touched by this loop and stays at its own model's fresh init automatically.
    """
    latest = _latest_parent_checkpoint(parent_ckpt_prefix)
    if latest is None:
        print(f"WARNING: no parent checkpoint found for prefix '{parent_ckpt_prefix}' in "
              f"{CKPT_DIR} - starting from random init instead of a warm start", flush=True)
        return False
    parent_ckpt = torch.load(latest, map_location=device)
    parent_state = parent_ckpt["model"]
    own_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in parent_state.items():
        if skip_prefixes and k.startswith(skip_prefixes):
            skipped += 1
            continue
        if k in own_state and own_state[k].shape == v.shape:
            own_state[k] = v
            loaded += 1
        else:
            print(f"  warm-start: key '{k}' missing/shape-mismatched in child model, skipping", flush=True)
    model.load_state_dict(own_state)
    print(f"warm-started from {latest} (parent step {parent_ckpt['step']}): "
          f"{loaded} tensors loaded, {skipped} tensors left at fresh init", flush=True)
    return True


def _warm_start_deep_expert(model, parent_ckpt_prefix: str, device: str, n_shared: int) -> bool:
    """arm routed12 only: like _warm_start_from_parent, but the backbone's key NAMES changed
    shape (plain backbone.blocks.{i}.* -> backbone.shared_blocks.{i}.* for i<n_shared, or
    backbone.expert_blocks.{i-n_shared}.{...}* / .ffn.ffn.{...}* for i>=n_shared, since the
    expert layers' FFN got wrapped in DomainLoRAFFN - see mot_routed_deepexpert_model.py).
    Remaps parent keys to their child-equivalent names before loading, rather than relying on
    exact key match. Verified via a real state_dict test: 0 mismatched, 0 missing, only the
    new (fresh-init) lora_a/lora_b keys are absent from the remapped parent - see commit
    history for the standalone test this was checked against before wiring it in here.
    """
    latest = _latest_parent_checkpoint(parent_ckpt_prefix)
    if latest is None:
        print(f"WARNING: no parent checkpoint found for prefix '{parent_ckpt_prefix}' in "
              f"{CKPT_DIR} - starting from random init instead of a warm start", flush=True)
        return False
    parent_ckpt = torch.load(latest, map_location=device)
    parent_state = parent_ckpt["model"]
    remapped = {}
    for k, v in parent_state.items():
        if k.startswith("backbone.blocks."):
            rest = k[len("backbone.blocks."):]
            idx_str, tail = rest.split(".", 1)
            idx = int(idx_str)
            if idx < n_shared:
                remapped[f"backbone.shared_blocks.{idx}.{tail}"] = v
            else:
                eidx = idx - n_shared
                if tail.startswith("ffn."):
                    remapped[f"backbone.expert_blocks.{eidx}.ffn.ffn.{tail[len('ffn.'):]}"] = v
                else:
                    remapped[f"backbone.expert_blocks.{eidx}.{tail}"] = v
        else:
            remapped[k] = v
    own_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in remapped.items():
        if k in own_state and own_state[k].shape == v.shape:
            own_state[k] = v
            loaded += 1
        else:
            skipped += 1
            print(f"  warm-start: key '{k}' missing/shape-mismatched in child model, skipping", flush=True)
    model.load_state_dict(own_state)
    print(f"warm-started (deep-expert remap) from {latest} (parent step {parent_ckpt['step']}): "
          f"{loaded} tensors loaded, {skipped} skipped, "
          f"{len(own_state) - loaded} total child tensors left at fresh init (new LoRA adapters)", flush=True)
    return True


def _warm_start_routed33_generalist(model, parent_ckpt_prefix: str, device: str,
                                     exact_step: int | None = None) -> bool:
    """routed33: adds a 5th "generalist" domain to routed28's 4-domain copygate architecture.
    Every existing head's output is `vocab_size + num_domains` (see mot_routed_model.py's
    "+ num_domains slots per head: switch to domain k" comment) - going from num_domains=4 to
    5 widens EVERY domain's head by exactly one column (the new "switch to generalist" slot),
    and control_embedding grows by one row (generalist's own control token) the same way. A
    plain _warm_start_from_parent would see the shape mismatch on every single head and
    control_embedding, and SKIP all of them - silently discarding every domain's learned vocab
    logits, not just failing to warm-start the new domain. This does the surgical version:
    copy the OLD columns/rows (the parent's actual learned weights) into the new, wider
    tensor's matching slice, and leave only the genuinely new slice (the 5th switch column,
    the 5th control row, and generalist's own embedding/projection/head, which aren't in the
    parent checkpoint at all) at fresh init.

    exact_step (routed35): pins a SPECIFIC parent checkpoint instead of _latest_parent_
    checkpoint's always-pick-the-newest default. routed35 warm-starts from routed28@140000
    deliberately (its best-balance checkpoint per the audit - single BPB 1.5839, math/science
    still healthy) rather than routed28's final 300k (single BPB eroded to 1.8011, math ppl
    298->919) - picking "latest" here would silently inherit the erosion this arm exists to
    avoid propagating forward."""
    if exact_step is not None:
        candidate = CKPT_DIR / f"{parent_ckpt_prefix}_step{exact_step}.pt"
        latest = str(candidate) if candidate.exists() else None
    else:
        latest = _latest_parent_checkpoint(parent_ckpt_prefix)
    if latest is None:
        print(f"WARNING: no parent checkpoint found for prefix '{parent_ckpt_prefix}'"
              f"{f' at step {exact_step}' if exact_step is not None else ''} in "
              f"{CKPT_DIR} - starting from random init instead of a warm start", flush=True)
        return False
    parent_ckpt = torch.load(latest, map_location=device)
    parent_state = parent_ckpt["model"]
    own_state = model.state_dict()
    loaded, resized, fresh = 0, 0, 0
    for k, v in own_state.items():
        if k not in parent_state:
            fresh += 1  # generalist's own embedding/type_embedding/projection/head - new domain
            continue
        pv = parent_state[k]
        if v.shape == pv.shape:
            own_state[k] = pv
            loaded += 1
            continue
        is_head = k.startswith("heads.") and (k.endswith(".weight") or k.endswith(".bias"))
        is_control = k == "control_embedding.weight"
        if not (is_head or is_control):
            print(f"  warm-start: key '{k}' unexpected shape mismatch "
                  f"{tuple(pv.shape)} -> {tuple(v.shape)}, leaving at fresh init", flush=True)
            fresh += 1
            continue
        # surgical resize: copy every OLD row/column, leave the new one(s) at fresh init.
        # heads.*.weight is (vocab+num_domains, d_model) -> old rows are [:vocab+4].
        # heads.*.bias is (vocab+num_domains,) -> same slicing on dim 0.
        # control_embedding.weight is (num_domains, d_model) -> old rows are [:4].
        n_old = pv.shape[0]
        merged = own_state[k].clone()
        merged[:n_old] = pv
        own_state[k] = merged
        resized += 1
    model.load_state_dict(own_state)
    print(f"warm-started (generalist-domain resize) from {latest} (parent step {parent_ckpt['step']}): "
          f"{loaded} tensors loaded as-is, {resized} tensors surgically resized "
          f"(old weights preserved, only the new domain's slot is fresh), "
          f"{fresh} tensors fully fresh init (generalist's own new modules)", flush=True)
    return True


def _build_model(arm: str, bundle: TokenizerBundle, device: str, scale: str = "base"):
    # scale="large" (arms mot/baseline only, for the scale test) swaps the whole config for
    # LARGE_MODEL_CFG - ~3-4x params. Everything else about the run is identical, which is the
    # point: an apples-to-apples "does the advantage survive scale" pair.
    if scale == "large":
        from src.model.stage2_config import LARGE_BACKBONE_ONLY_CFG, LARGE_MODEL_CFG
        if arm == "mot":
            return MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LARGE_MODEL_CFG).to(device)
        if arm == "baseline":
            return BaselineModel(vocab_size=bundle.baseline_vocab_size, **LARGE_BACKBONE_ONLY_CFG).to(device)
        if arm == "routed":
            return MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LARGE_MODEL_CFG).to(device)
        if arm == "routed3":  # routed + GradNorm loss + densest cross-domain data (best direction so far)
            return MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LARGE_MODEL_CFG).to(device)
        if arm == "routed7":  # routed, same architecture, nlp domain sourced from OpenWebText (see _apply_openwebtext_nlp_source)
            return MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LARGE_MODEL_CFG).to(device)
        if arm == "routed10":  # routed7's pair - same architecture, warm-started + books + upweighted (see _warm_start_from_parent)
            return MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LARGE_MODEL_CFG).to(device)
        if arm == "routed14":  # routed11's copy-gate mechanism at large scale, warm-started from routed7
            return MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LARGE_MODEL_CFG).to(device)
        if arm == "routed28":  # routed25's recipe (gate+diet+full plasticity) at large scale,
            # warm-started from large_routed7 directly (no large_routed17 exists)
            return MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LARGE_MODEL_CFG).to(device)
        if arm == "routed33":  # 5-domain (code/math/science/nlp/generalist) copy-gate at large
            # scale, FROM SCRATCH (deliberately not in WARM_START_PARENT) - tests whether the
            # generalist-domain recipe has merit on its own before combining it with routed28's
            # already-good backbone. domain_vocab_sizes has 5 entries here (bundle built with
            # generalist_tokenizer_dir set), so every head/control_embedding is sized for 5
            # domains from the start - no resize needed since there's no parent to resize from.
            return MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LARGE_MODEL_CFG).to(device)
        if arm == "routed35":  # routed33's generalist-domain recipe merged onto routed28's
            # already-good backbone instead of from scratch - the exact follow-up routed33 was
            # built to de-risk. Same 5-domain shape as routed33 (generalist_tokenizer_dir set
            # below), same model class - the warm-start (see _warm_start_routed33_generalist,
            # called with exact_step=140000 in train()) is what differs, done via surgical
            # resize since routed28's checkpoint only has 4 domains' worth of head/control_
            # embedding width.
            return MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LARGE_MODEL_CFG).to(device)
        raise ValueError(
            f"scale=large supports mot/baseline/routed/routed3/routed7/routed10/routed14/routed28/routed33/routed35, not {arm}")
    if arm == "mot":
        return MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm in ("routed", "routed8", "routed9", "routed15", "routed17", "routed18", "routed19", "routed23"):
        # routed8/routed9: identical architecture to plain routed - only the nlp data source
        # (OpenWebText or PG-19 books, applied by _apply_openwebtext_nlp_source /
        # _apply_books_nlp_source in calibrate()/train()) and, for routed9, the warm-start +
        # mixture upweighting, differ. routed15 (control), routed17/18 (diet-phase data levers),
        # routed19 (curriculum + corrected token budget) are the same story - no architecture
        # change, only data/LR/step-count treatment differs. routed23: alignment-loss-only
        # ablation, plain architecture, from scratch - see ALIGN_ARMS in stage2_config.py.
        return MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm == "routed11":  # Bet 1: copy gate on the nlp head, warm-started from routed8
        return MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm in ("routed20", "routed21", "routed22", "routed24", "routed25", "routed26", "routed27"):
        # routed20/21: copy-gate + diet, warm-started from routed17 (routed11's exact recipe,
        # now proven to be the project's best LAMBADA result once actually measured - see
        # ALIGN_ARMS's comment block in stage2_config.py). routed22: same mechanism, but from
        # scratch, no diet - tests whether copy-gate needs a warm-started backbone to be
        # useful at all. routed24: routed11's exact recipe amped up (full backbone plasticity
        # instead of 0.3x, see PER_ARM_BACKBONE_LR_SCALE) - not a plain rerun, a test of
        # whether the "less throttle helped" trend continues further. routed25/26/27: the
        # second batch - routed20/21's combined win (gate+diet) plus routed24's plasticity
        # lever, run longer; routed26 pushes diet higher, routed27 swaps nlp source to books
        # (see PER_ARM_BACKBONE_LR_SCALE and ROUTED26_NLP_SNIPPET_WORDS in stage2_config.py).
        # Unbiased default gate_bias_init throughout (routed11's init, not
        # routed16's -4.0 "conservative" one - the conservative init underperformed once
        # correctly measured, so there's no reason to carry it into this set).
        return MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm == "routed29":
        # tied-head reallocation (see mot_routed_tied_model.py + TIED_MODEL_CFG in
        # stage2_config.py): same gate+diet recipe as routed25/26/27, but the ~50M normally
        # spent on 4 untied output heads is reinvested as backbone depth (23 layers at
        # MODEL_CFG's d_model=512, vs 6) instead. From scratch - no compatible warm-start
        # checkpoint exists for this architecture (head/embedding shapes differ structurally).
        from src.model.stage2_config import TIED_MODEL_CFG
        return MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **TIED_MODEL_CFG).to(device)
    if arm == "routed30":
        # vocab-shrink + direct tying (see ROUTED30_MODEL_CFG in stage2_config.py): code/
        # math/science retrained at ROUTED30_SHRUNK_VOCAB (10k, vs 24k - they're starved
        # under the diet mixture anyway), nlp untouched. emb_dim raised to d_model so tying
        # is direct (no bridge) - simpler than routed29 but leaves less for backbone depth
        # (16 layers here vs routed29's 23) - the alternative bet, not a strict upgrade.
        from src.model.stage2_config import ROUTED30_MODEL_CFG
        return MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **ROUTED30_MODEL_CFG).to(device)
    if arm == "routed31":
        # routed29's allocation (narrow tied embeddings, max depth) + the full modern-
        # technique stack: RoPE, RMSNorm, SwiGLU FFN, QK-norm (see backbone_modern.py). The
        # aggressive exploratory bet - do techniques proven elsewhere help here.
        from src.model.stage2_config import ROUTED_MODERN_MODEL_CFG
        return MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **ROUTED_MODERN_MODEL_CFG,
                                   backbone_cls=ModernBackbone,
                                   backbone_kwargs={"use_swiglu": True, "use_qk_norm": True}).to(device)
    if arm == "routed32":
        # routed29's allocation + only RoPE + RMSNorm (no SwiGLU/QK-norm) - the "safe
        # improver", the two changes closest to risk-free, meant to actually beat the
        # flagship rather than test a hypothesis.
        from src.model.stage2_config import ROUTED_MODERN_MODEL_CFG
        return MoTRoutedTiedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **ROUTED_MODERN_MODEL_CFG,
                                   backbone_cls=ModernBackbone,
                                   backbone_kwargs={"use_swiglu": False, "use_qk_norm": False}).to(device)
    if arm == "routed16":  # Bet 1 v2: same mechanism, conservative gate init (see routed11/14 post-mortem)
        return MoTRoutedCopyGateModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                       gate_bias_init=COPYGATE_V2_BIAS_INIT).to(device)
    if arm == "routed12":  # Bet 2: per-domain LoRA on the top layers, warm-started from routed8
        return MoTRoutedDeepExpertModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm == "routed13":  # Bet 3 (exploratory): precision/margin head, warm-started from routed8
        return MoTRoutedPrecisionModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm == "pooled":
        return MoTPooledModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                               focal_gamma=FOCAL_GAMMA, confidence_weight=CONFIDENCE_WEIGHT).to(device)
    if arm == "hybrid":
        return MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm == "pooled2":
        return MoTPooled2Model(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG,
                                focal_gamma=FOCAL_GAMMA).to(device)
    if arm in ("routed2", "routed3"):
        # same GradNorm-balanced architecture as hybrid - routed2/routed3 differ from hybrid
        # (and from each other) only in what data feeds them, not in model code.
        return MoTHybridModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm == "routed4":
        # all three fixes stacked: decoupled switch head + learned switch weight + 2x context.
        return MoTRoutedCombinedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LONGCTX_MODEL_CFG).to(device)
    if arm == "routed5":
        # decoupled-head-only, standard 1024 context - kept for a later ablation pass if
        # routed4's combined result is worth pulling apart, not launched tonight.
        return MoTRoutedDecoupledModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm == "routed6":
        # long-context-only, standard head/weight - same ablation-reserve status as routed5.
        return MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **LONGCTX_MODEL_CFG).to(device)
    if arm == "baseline":
        return BaselineModel(vocab_size=bundle.baseline_vocab_size, **BACKBONE_ONLY_CFG).to(device)
    return BaselineModel(vocab_size=bundle.sota_vocab_size, **BACKBONE_ONLY_CFG).to(device)


def calibrate(arm: str, steps: int, scale: str = "base") -> float:
    # Mirror train()'s TF32 setting so calibrate's timing is representative of a real run.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rank, world_size, local_rank, device = _ddp_setup()
    is_main = rank == 0
    _ld = partial(_loader, world_size=world_size)
    if is_main:
        print(f"CUDA available: {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(local_rank)}"
              f"  world_size: {world_size}", flush=True)
    # Mirror train()'s world_size-aware batch/accum split (see the detailed comment there) so
    # calibrate's memory profile matches what the real run will actually use - calibrate()
    # doesn't accumulate gradients at all (opt.step() every micro-step), so this only matters
    # for getting a representative peak-memory reading, not for timing per real optimizer update.
    if arm == "routed19":
        calib_grad_accum = 1 if world_size > 1 else ROUTED19_GRAD_ACCUM_STEPS
        routed19_batch_size = 64 // calib_grad_accum
    if arm in ("routed7", "routed10") + LARGE_BET_ARMS:
        scale = "large"  # always large-scale, regardless of what --scale was passed
    if arm in OWT_TOKENIZER_ARMS:
        _apply_openwebtext_nlp_source()  # everything reusing routed8's exact nlp source
    if arm in ("routed9", "routed10", "routed27"):
        # routed27 keeps the OWT-fit tokenizer (still in OWT_TOKENIZER_ARMS, unlike routed9/10
        # which reinit for a tokenizer swap) - only the raw text source changes, so it stays
        # warm-start-compatible with routed17's embedding tables. _apply_books_nlp_source is a
        # pure STREAM_SOURCES mutation, independent of tokenizer choice - see its docstring.
        _apply_books_nlp_source()
    if arm in ("routed33", "routed35"):
        _apply_generalist_domain_source()

    nlp_tok_dir = OWT_NLP_TOKENIZER_DIR if arm in OWT_TOKENIZER_ARMS else \
        (BOOKS_NLP_TOKENIZER_DIR if arm in ("routed9", "routed10") else None)
    tok_dir = SHRUNK_TOKENIZER_DIR if arm == "routed30" else TOKENIZER_DIR
    bundle = TokenizerBundle(tokenizer_dir=tok_dir, nlp_tokenizer_dir=None if arm == "routed30" else nlp_tok_dir,
                              generalist_tokenizer_dir=GENERALIST_TOKENIZER_DIR if arm in ("routed33", "routed35") else None)
    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    model = _build_model(arm, bundle, device, scale)
    if is_main:
        print(f"{arm} ({scale}) params: {model.num_params():,}", flush=True)
    # Compile the backbone only, before DDP wrap - it's pure static tensor ops (no data-
    # dependent branching) in both backbone classes, and is most of the FLOPs. The full routed
    # model is NOT compiled: per-domain masked gathers + keep.any() branches in head_loss cause
    # graph breaks/recompiles there. First 1-2 min of a run is compile time - judge throughput
    # from step 300+, not step 1. If an arm graph-breaks, drop this one line for that arm only.
    model.backbone = torch.compile(model.backbone)
    if world_size > 1:
        # DDP.__getattr__ does NOT forward arbitrary custom methods like num_params() to the
        # wrapped module the way it forwards forward() itself - only reference the DDP object
        # for the forward/backward pass from here on, never for model-specific attributes.
        model = DDP(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    if arm in ("mot", "hybrid"):
        domains = list(bundle.domain_vocab_sizes)
        loaders = {
            d: iter(_ld(PackedDomainStream(d, bundle.encode_domain, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
            for d in domains
        }
    if arm in ("routed9", "routed10"):
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in DIET_ARMS:
        # routed17: nlp upweighted to ~70%, no filter. routed18: same upweighting, PLUS only
        # copy-structured documents (see _is_copy_structured in stage2_routed_stream.py).
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=DIET_PHASE2_NLP_SNIPPET_WORDS,
            force_domain_filter=(COPY_MINE_MIN_WORD_LEN, COPY_MINE_MIN_GAP) if arm == "routed18" else None,
        ), batch_size=BATCH_SIZE))
    elif arm == "routed16":
        # Backbone is frozen (FROZEN_BACKBONE_ARMS) - only copy_q/copy_k/copy_gate get
        # gradient, and those only fire on nlp-domain tokens. A batch with zero nlp content
        # would have a loss fully disconnected from every trainable tensor, crashing
        # backward() ("does not require grad and does not have a grad_fn") - confirmed live,
        # not hypothetical. force_domain guarantees nlp's PRESENCE every batch (default
        # snippet_words, no upweighting - unlike routed17/18, routed16 isn't a mixture-share
        # bet, it just needs the copy-gate mechanism to always have something to learn from).
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"], force_domain="nlp",
        ), batch_size=BATCH_SIZE))
    elif arm == "routed26":
        # same recipe as routed25/27, but nlp diet pushed higher (~83% vs ~70%) - is 70%
        # actually the ceiling, or does more help further now that copy-gate is stacked in?
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=ROUTED26_NLP_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in RECIPE_DIET_ARMS:
        # routed20/21/25/27/28: copy-gate stacked on top of routed17's already-diet-adapted
        # mixture - same nlp upweighting as DIET_ARMS, kept through continuation (not reverted
        # to plain), so the model isn't asked to un-learn the mixture it's continuing from.
        # routed27's books-vs-OWT difference is a raw-text-source swap (_apply_books_nlp_source
        # above), not a change to this loader's own parameters.
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=DIET_PHASE2_NLP_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in ("routed22", "routed23", "routed24"):
        # plain (non-upweighted) mixture, matching routed11 exactly for routed24 - routed22
        # isolates whether copy-gate needs a warm-started backbone at all; routed23 isolates
        # the alignment loss alone; routed24 isolates backbone plasticity (see _build_model).
        loader = iter(_ld(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm in ("routed", "pooled", "pooled2", "hybrid", "routed2", "routed5", "routed7", "routed8") + BET_ARMS:
        loader = iter(_ld(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm == "routed3":
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            min_domains=ROUTED3_MIN_DOMAINS, max_domains=ROUTED3_MAX_DOMAINS,
            snippet_words=ROUTED3_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm == "routed19":
        phase1_loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            min_domains=1, max_domains=1, snippet_words=ROUTED19_PHASE1_SNIPPET_WORDS,
        ), batch_size=routed19_batch_size))
        phase2_loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            cache_pass_offset=ROUTED19_PHASE2_CACHE_PASS_OFFSET,
        ), batch_size=routed19_batch_size))
        routed19_phase1_steps = round(steps * ROUTED19_PHASE1_FRACTION)
    elif arm in ("routed4", "routed6"):  # both use 2x context
        loader = iter(_ld(PackedRoutedStream(bundle, domain_index, LONGCTX_MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm not in ("mot",):
        encode_fn = bundle.encode_baseline if arm == "baseline" else bundle.encode_sota
        loader = iter(_ld(PackedMixedStream(encode_fn, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))

    t0 = time.time()
    for step in range(1, steps + 1):
        if arm == "mot":
            domain = domains[step % len(domains)]
            _, ids, types = next(loaders[domain])
            ids, types = ids.to(device), types.to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(domain, inp, types[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        elif arm == "hybrid":
            tok, dom, ctrl, typ, tgt = _hybrid_batch(step, domains, domain_index, loaders, loader, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
        elif arm in ("routed2", "routed3", "routed4"):
            # routed4 (combined: decoupled head + learned weight) takes no switch_weight
            # kwarg - it's an internal learned parameter, same call shape as routed2/routed3.
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = (t.to(device, non_blocking=True) for t in (tok, dom, ctrl, typ, tgt))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
        elif arm in ("routed", "pooled", "pooled2", "routed5", "routed6", "routed7", "routed8", "routed9", "routed10") + BET_ARMS + DIET_ARMS + RECIPE_ARMS:
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = (t.to(device, non_blocking=True) for t in (tok, dom, ctrl, typ, tgt))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if arm in ("pooled", "pooled2"):
                    loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ,
                                     switch_weight=SWITCH_WEIGHT, adv_lambda=1.0)
                else:
                    loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
        elif arm == "routed19":
            src = phase1_loader if step <= routed19_phase1_steps else phase2_loader
            tok, dom, ctrl, typ, tgt = next(src)
            tok, dom, ctrl, typ, tgt = (t.to(device, non_blocking=True) for t in (tok, dom, ctrl, typ, tgt))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
        else:
            ids = next(loader).to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1:
            print(f"peak GPU mem after first step: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
        if step % 25 == 0:
            elapsed = time.time() - t0
            print(f"step {step}/{steps}  loss={loss.item():.4f}  elapsed={elapsed:.1f}s  "
                  f"sec/step={elapsed/step:.3f}", flush=True)

    elapsed = time.time() - t0
    sec_per_step = elapsed / steps
    if is_main:
        print(f"\nCALIBRATION RESULT: {sec_per_step:.3f} sec/step/rank on "
              f"{torch.cuda.get_device_name(local_rank)}  world_size={world_size}  "
              f"(~{sec_per_step / world_size:.3f} effective sec/step across all ranks)", flush=True)
    return sec_per_step


def train(arm: str, max_steps: int | None = None, scale: str = "base") -> list:
    # TF32 speeds up the remaining fp32 matmuls/convs (norm layers, optimizer math) outside
    # the bf16-autocast regions above - free on Ampere+, no accuracy-relevant effect at the
    # tolerances anything here is measured to.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rank, world_size, local_rank, device = _ddp_setup()
    is_main = rank == 0
    _ld = partial(_loader, world_size=world_size)
    total_steps = max_steps or MAX_STEPS
    # routed19's effective batch is a fixed invariant (64, matching every other arm's
    # BATCH_SIZE(4)*GRAD_ACCUM_STEPS(16)) - HOW it's reached differs by world_size, and the
    # split matters for real throughput, not just correctness:
    #  - world_size==1: grad_accum=2, micro-batch=32 - the validated single-GPU setting
    #    (calibrated live: 22GB/46GB peak memory, 0.442 sec/step). Kept exactly as measured.
    #  - world_size>1: grad_accum=1, micro-batch=64/world_size - DROPS accumulation entirely
    #    rather than keep dividing the single-GPU batch by world_size (which would have given
    #    a needlessly tiny batch, e.g. 32/4=8 at 4 GPUs, while still paying for 2 accumulation
    #    micro-steps and no_sync() bookkeeping per real update for no reason - once multiple
    #    GPUs are already splitting the effective batch, the accumulation dimension is no
    #    longer needed to keep any single device's batch small. At 4 GPUs this gives
    #    micro-batch=16, not 8 - genuinely bigger per-step work, simpler code (every step is a
    #    sync step, no no_sync() branch), same real optimizer-update count either way.
    if arm == "routed19":
        designed_effective_batch = 64
        grad_accum = 1 if world_size > 1 else ROUTED19_GRAD_ACCUM_STEPS
        routed19_batch_size = designed_effective_batch // grad_accum  # world_size division
        # happens inside _ld itself (see _loader) - passing this un-divided value through is
        # deliberate, not a bug: (64//grad_accum) // world_size == 64/(grad_accum*world_size).
        per_rank_micro = routed19_batch_size // max(world_size, 1)
        actual_effective_batch = per_rank_micro * grad_accum * world_size
        assert actual_effective_batch == designed_effective_batch, (
            f"effective batch mismatch: {per_rank_micro} (per-rank micro) * {grad_accum} "
            f"(grad_accum) * {world_size} (world_size) = {actual_effective_batch}, expected "
            f"{designed_effective_batch}."
        )
        if is_main:
            print(f"effective batch check OK: {per_rank_micro} x {grad_accum} x {world_size} "
                  f"= {actual_effective_batch}", flush=True)
    else:
        grad_accum = GRAD_ACCUM_STEPS
    if is_main:
        print(f"device: {torch.cuda.get_device_name(local_rank)}  world_size: {world_size}  "
              f"arm: {arm}  steps: {total_steps}", flush=True)
        print(f"ARM: {arm}  =  {ARM_LABELS.get(arm, arm)}", flush=True)
    if arm in ("routed7", "routed10") + LARGE_BET_ARMS:
        scale = "large"  # always large-scale, regardless of what --scale was passed
    if arm in OWT_TOKENIZER_ARMS:
        _apply_openwebtext_nlp_source()  # everything reusing routed8's exact nlp source
    if arm in ("routed9", "routed10", "routed27"):
        # routed27 keeps the OWT-fit tokenizer (still in OWT_TOKENIZER_ARMS, unlike routed9/10
        # which reinit for a tokenizer swap) - only the raw text source changes, so it stays
        # warm-start-compatible with routed17's embedding tables. _apply_books_nlp_source is a
        # pure STREAM_SOURCES mutation, independent of tokenizer choice - see its docstring.
        _apply_books_nlp_source()
    if arm in ("routed33", "routed35"):
        _apply_generalist_domain_source()

    nlp_tok_dir = OWT_NLP_TOKENIZER_DIR if arm in OWT_TOKENIZER_ARMS else \
        (BOOKS_NLP_TOKENIZER_DIR if arm in ("routed9", "routed10") else None)
    tok_dir = SHRUNK_TOKENIZER_DIR if arm == "routed30" else TOKENIZER_DIR
    bundle = TokenizerBundle(tokenizer_dir=tok_dir, nlp_tokenizer_dir=None if arm == "routed30" else nlp_tok_dir,
                              generalist_tokenizer_dir=GENERALIST_TOKENIZER_DIR if arm in ("routed33", "routed35") else None)
    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    model = _build_model(arm, bundle, device, scale)
    if is_main:
        print(f"{arm} ({scale}) params: {model.num_params():,}", flush=True)

    # routed9/routed10 ("cooldown" arms): the nlp branch is reinitialized fresh (see
    # _warm_start_from_parent) while everything else is warm-started, so it needs to catch up
    # fast without the warm-started backbone/other-domain tables being dragged along at the
    # same rate - COOLDOWN_BACKBONE_LR_SCALE (0.1) throttles the latter, applied per-step below
    # alongside the normal cosine schedule via each param group's "lr_scale".
    # routed11/12/13 ("bet" arms): nothing is reinitialized (same tokenizer as routed8), so
    # only each bet's OWN new module (matched via NEW_MODULE_MATCH) needs full LR - everything
    # else already trained gets BET_BACKBONE_LR_SCALE, a gentler throttle than the cooldown
    # arms' since there's no fresh-init branch racing to catch up here.
    if arm in WARM_START_PARENT and arm not in NEW_MODULE_MATCH:
        nlp_params = [p for n, p in model.named_parameters() if n.startswith(_NLP_PARAM_PREFIXES)]
        other_params = [p for n, p in model.named_parameters() if not n.startswith(_NLP_PARAM_PREFIXES)]
        opt = torch.optim.AdamW([
            {"params": nlp_params, "lr_scale": 1.0},
            {"params": other_params, "lr_scale": COOLDOWN_BACKBONE_LR_SCALE},
        ], lr=LR, fused=True)
    elif arm in FROZEN_BACKBONE_ARMS:
        # routed16 (copy-gate v2): routed11/14's post-mortem found the copy mechanism itself
        # never hurt, but the shared backbone/vocab head degraded under the joint objective
        # even at BET_BACKBONE_LR_SCALE (0.3x, not zero) - its gradient flows straight back
        # through the SAME hidden states the vocab head depends on. Fix: freeze everything
        # except the new module entirely (requires_grad=False, not just a low LR) so it
        # genuinely cannot move, then optimize only the trainable (new) params at full LR.
        match = NEW_MODULE_MATCH[arm]
        trainable = []
        for n, p in model.named_parameters():
            if any(s in n for s in match):
                trainable.append(p)
            else:
                p.requires_grad_(False)
        opt = torch.optim.AdamW(trainable, lr=LR, fused=True)
    elif arm in NEW_MODULE_MATCH:
        match = NEW_MODULE_MATCH[arm]
        if match:
            new_params = [p for n, p in model.named_parameters() if any(s in n for s in match)]
            old_params = [p for n, p in model.named_parameters() if not any(s in n for s in match)]
            # PER_ARM_BACKBONE_LR_SCALE overrides the default BET_BACKBONE_LR_SCALE for a
            # specific arm - routed24 uses 1.0 (no throttle at all) to test whether more
            # backbone plasticity than routed11's 0.3x helps further, per real evidence that
            # LESS restriction (routed11) beat MORE restriction (routed16, fully frozen).
            backbone_scale = PER_ARM_BACKBONE_LR_SCALE.get(arm, BET_BACKBONE_LR_SCALE)
            opt = torch.optim.AdamW([
                {"params": new_params, "lr_scale": 1.0},
                {"params": old_params, "lr_scale": backbone_scale},
            ], lr=LR, fused=True)
        else:
            # routed15 (control): no new module - every param gets the SAME BET_BACKBONE_LR_SCALE
            # throttle routed11/12/13's non-new-module params got, so this is a genuine matched
            # control (identical warm-start + LR treatment, minus whatever each bet added).
            opt = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
            for g in opt.param_groups:
                g["lr_scale"] = BET_BACKBONE_LR_SCALE
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
    # bf16 autocast (see the dtype= on every autocast block above) needs no loss scaling -
    # unlike fp16, its exponent range covers fp32's without over/underflowing, so there's no
    # GradScaler here (removed along with its scale/step/update calls below).

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_prefix = f"large_{arm}" if scale == "large" else arm  # keep large runs off base names

    def _latest_checkpoint() -> str | None:
        import glob
        import re
        paths = glob.glob(str(CKPT_DIR / f"{ckpt_prefix}_step*.pt"))
        return sorted(paths, key=lambda p: int(re.search(r"step(\d+)", p).group(1)), reverse=True)

    start_step = 1
    resumed_own = False
    resumed_history: list = []
    for cand in _latest_checkpoint():
        try:
            ckpt = torch.load(cand, map_location=device)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            start_step = ckpt["step"] + 1
            # Extend (never reset) the loss-curve history on resume - a bare `history = []`
            # here silently discarded every prior restart's data (real bug, confirmed: routed8's
            # checkpoint only had ~172k steps of history despite being at step 496k, all from
            # its most recent restart). Every pod restart (migrations, watchdog auto-restarts)
            # used to erase everything before it.
            resumed_history = ckpt.get("history", [])
            if is_main:
                print(f"resumed from {cand} at step {start_step} "
                      f"({len(resumed_history)} history entries carried forward)", flush=True)
            resumed_own = True
            break
        except Exception as e:
            if is_main:
                print(f"checkpoint {cand} failed to load ({type(e).__name__}: {e}); trying older", flush=True)
            continue
    if not resumed_own:
        warm_started = False
        if arm == "routed12":
            warm_started = _warm_start_deep_expert(
                model, WARM_START_PARENT[arm], device, n_shared=MODEL_CFG["n_layers"] - DEFAULT_N_EXPERT_LAYERS)
        elif arm == "routed35":
            # pinned to routed28's step-140000 checkpoint specifically, not "latest" - see
            # _warm_start_routed33_generalist's exact_step docstring for why. Prefix is
            # "large_routed28" (not "routed28") since routed28 is a scale="large" arm and
            # saves checkpoints under that prefix - see ckpt_prefix's construction above and
            # WARM_START_PARENT's own "large_routed14"/"large_routed7" entries for the same
            # convention on every other large-scale parent.
            warm_started = _warm_start_routed33_generalist(model, "large_routed28", device, exact_step=140000)
        elif arm in WARM_START_PARENT:
            skip = _NLP_PARAM_PREFIXES if arm in ("routed9", "routed10") else ()
            warm_started = _warm_start_from_parent(model, WARM_START_PARENT[arm], device, skip_prefixes=skip)
        if not warm_started and is_main:
            print(f"no checkpoint found for {arm} in {CKPT_DIR} - starting from step 1", flush=True)

    # Every requires_grad_(False) / optimizer param-group split above needs to have already
    # happened - DDP inspects which params need gradient sync at CONSTRUCTION time, so wrapping
    # first would sync frozen params too (or crash on include-in-optimizer-but-no-grad
    # mismatches). raw_model stays the reference for anything that isn't the forward/backward
    # pass itself (state_dict for checkpointing, custom attributes like .domains,
    # .num_params()) - DDP does NOT forward arbitrary custom attributes to the wrapped module
    # the way it forwards forward()/.train()/.eval(), so using the DDP object for those would
    # either crash or silently do the wrong thing.
    # Compile the backbone only, before raw_model/DDP - it's pure static tensor ops (no
    # data-dependent branching) in both backbone classes, and is most of the FLOPs. The full
    # routed model is NOT compiled: per-domain masked gathers + keep.any() branches in
    # head_loss cause graph breaks/recompiles there. First 1-2 min of a run is compile time -
    # judge throughput from step 300+, not step 1. If an arm graph-breaks, drop this one line
    # for that arm only.
    model.backbone = torch.compile(model.backbone)
    raw_model = model
    if world_size > 1:
        model = DDP(
            model, device_ids=[local_rank],
            gradient_as_bucket_view=True,  # fewer gradient-bucket memory copies
            broadcast_buffers=False,  # no BatchNorm-style running-stat buffers on a transformer
            static_graph=True,  # same forward path every step (just different data) - lets
            # DDP skip re-deriving its reducer/bucket structure each iteration
        )

    def lr_at(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    if arm in ("mot", "hybrid"):
        domains = list(bundle.domain_vocab_sizes)
        loaders = {
            d: iter(_ld(PackedDomainStream(d, bundle.encode_domain, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
            for d in domains
        }
    if arm in ("routed9", "routed10"):
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in DIET_ARMS:
        # routed17: nlp upweighted to ~70%, no filter. routed18: same upweighting, PLUS only
        # copy-structured documents (see _is_copy_structured in stage2_routed_stream.py).
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=DIET_PHASE2_NLP_SNIPPET_WORDS,
            force_domain_filter=(COPY_MINE_MIN_WORD_LEN, COPY_MINE_MIN_GAP) if arm == "routed18" else None,
        ), batch_size=BATCH_SIZE))
    elif arm == "routed16":
        # Backbone is frozen (FROZEN_BACKBONE_ARMS) - only copy_q/copy_k/copy_gate get
        # gradient, and those only fire on nlp-domain tokens. A batch with zero nlp content
        # would have a loss fully disconnected from every trainable tensor, crashing
        # backward() ("does not require grad and does not have a grad_fn") - confirmed live,
        # not hypothetical. force_domain guarantees nlp's PRESENCE every batch (default
        # snippet_words, no upweighting - unlike routed17/18, routed16 isn't a mixture-share
        # bet, it just needs the copy-gate mechanism to always have something to learn from).
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"], force_domain="nlp",
        ), batch_size=BATCH_SIZE))
    elif arm == "routed26":
        # same recipe as routed25/27, but nlp diet pushed higher (~83% vs ~70%) - is 70%
        # actually the ceiling, or does more help further now that copy-gate is stacked in?
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=ROUTED26_NLP_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in RECIPE_DIET_ARMS:
        # routed20/21/25/27/28: copy-gate stacked on top of routed17's already-diet-adapted
        # mixture - same nlp upweighting as DIET_ARMS, kept through continuation (not reverted
        # to plain), so the model isn't asked to un-learn the mixture it's continuing from.
        # routed27's books-vs-OWT difference is a raw-text-source swap (_apply_books_nlp_source
        # above), not a change to this loader's own parameters.
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=DIET_PHASE2_NLP_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in ("routed22", "routed23", "routed24"):
        # plain (non-upweighted) mixture, matching routed11 exactly for routed24 - routed22
        # isolates whether copy-gate needs a warm-started backbone at all; routed23 isolates
        # the alignment loss alone; routed24 isolates backbone plasticity (see _build_model).
        loader = iter(_ld(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm in ("routed", "pooled", "pooled2", "hybrid", "routed2", "routed5", "routed7", "routed8") + BET_ARMS:
        loader = iter(_ld(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm == "routed3":
        loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            min_domains=ROUTED3_MIN_DOMAINS, max_domains=ROUTED3_MAX_DOMAINS,
            snippet_words=ROUTED3_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm == "routed19":
        phase1_loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            min_domains=1, max_domains=1, snippet_words=ROUTED19_PHASE1_SNIPPET_WORDS,
        ), batch_size=routed19_batch_size))
        phase2_loader = iter(_ld(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            cache_pass_offset=ROUTED19_PHASE2_CACHE_PASS_OFFSET,
        ), batch_size=routed19_batch_size))
        routed19_phase1_steps = round(total_steps * ROUTED19_PHASE1_FRACTION)
        print(f"routed19 curriculum: phase 1 (single-domain) steps 1-{routed19_phase1_steps}, "
              f"phase 2 (switching) steps {routed19_phase1_steps + 1}-{total_steps}", flush=True)
    elif arm in ("routed4", "routed6"):  # both use 2x context
        loader = iter(_ld(PackedRoutedStream(bundle, domain_index, LONGCTX_MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm not in ("mot",):
        encode_fn = bundle.encode_baseline if arm == "baseline" else bundle.encode_sota
        loader = iter(_ld(PackedMixedStream(encode_fn, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))

    controller = None
    if arm in ("pooled", "pooled2", "hybrid", "routed2", "routed3", "routed4"):
        # routed4 gets the safety net too - its learned switch-weight is a dynamic parameter
        # (same destabilization risk profile as GradNorm's EMA reweighting), even though the
        # rest of its loss is standard CE. routed5/routed6 don't need it - fixed switch_weight,
        # no dynamic loss-shaping component, same as plain routed.
        from src.model.adaptive_optimizer import AdaptiveController
        controller = AdaptiveController()
        print("adaptive controller ON (spike-guard + plateau rescue + online LR)", flush=True)

    # Periodic held-out eval (idea 4a): a val stream on a DISTINCT seed so its synthetic
    # multi-domain composition never coincides with training's. Only wired for the switching
    # arms (every currently-live and next-round arm is one) - they all consume PackedRouted
    # Stream, so one held-out stream covers them. Plain next-token CE (via targets=None +
    # manual CE), not the arm's balanced/aux loss, so the number is comparable step-to-step
    # and across arms regardless of each arm's loss shaping.
    from src.model.stage2_config import EVAL_EVERY, VAL_BATCHES, VAL_SEED

    val_iter = None
    # is_main only: _held_out_ce() is already rank-0-guarded below, so building this loader
    # (its own NUM_WORKERS DataLoader worker pool) on every other rank was pure waste - workers
    # that spin up, hold memory, and are never actually read from.
    if is_main and arm in ("routed", "pooled", "pooled2", "hybrid", "routed2", "routed3", "routed4", "routed5",
               "routed6", "routed7", "routed8", "routed9", "routed10", "routed19") + BET_ARMS + DIET_ARMS:
        rs3 = arm == "routed3"
        books_upweighted = arm in ("routed9", "routed10")
        diet_upweighted = arm in DIET_ARMS
        seq_len = LONGCTX_MODEL_CFG["max_seq_len"] if arm in ("routed4", "routed6") else MODEL_CFG["max_seq_len"]
        val_stream = PackedRoutedStream(
            bundle, domain_index, seq_len, seed=VAL_SEED,
            min_domains=ROUTED3_MIN_DOMAINS if rs3 else 2,
            max_domains=ROUTED3_MAX_DOMAINS if rs3 else 4,
            snippet_words=ROUTED3_SNIPPET_WORDS if rs3 else 250,
            force_domain="nlp" if (books_upweighted or diet_upweighted) else None,
            force_domain_snippet_words=(BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS if books_upweighted else
                                         DIET_PHASE2_NLP_SNIPPET_WORDS if diet_upweighted else None),
            force_domain_filter=(COPY_MINE_MIN_WORD_LEN, COPY_MINE_MIN_GAP) if arm == "routed18" else None,
        )
        val_iter = iter(_ld(val_stream, batch_size=routed19_batch_size if arm == "routed19" else BATCH_SIZE))

    def _held_out_ce():
        model.eval()
        tot_nats, tot_tok = 0.0, 0
        with torch.no_grad():
            for _ in range(VAL_BATCHES):
                tok, dom, ctrl, typ, tgt = next(val_iter)
                tok, dom, ctrl, typ, tgt = (t.to(device) for t in (tok, dom, ctrl, typ, tgt))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(tok, dom, ctrl, targets=None, type_ids=typ)
                for d in raw_model.domains:
                    if d not in out:
                        continue
                    mask, logits = out[d]
                    dt = tgt[mask]
                    keep = dt < bundle.domain_vocab_sizes[d]  # content tokens only, drop switch targets
                    if keep.any():
                        tot_nats += F.cross_entropy(logits[keep], dt[keep], reduction="sum").item()
                        tot_tok += int(keep.sum().item())
        model.train()
        return tot_nats / max(tot_tok, 1)

    t0 = time.time()
    # GPU tensor, not a Python float - see the running += loss.detach() comment below for why.
    running, running_n = torch.zeros((), device=device), 0
    history = resumed_history
    for step in range(start_step, total_steps + 1):
        lr_mult = controller.lr_mult if controller is not None else 1.0
        for g in opt.param_groups:
            g["lr"] = LR * lr_at(step) * lr_mult * g.get("lr_scale", 1.0)

        if arm == "mot":
            domain = domains[step % len(domains)]
            _, ids, types = next(loaders[domain])
            ids, types = ids.to(device), types.to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(domain, inp, types[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        elif arm in ("routed", "routed7", "routed8", "routed9", "routed10") + BET_ARMS + DIET_ARMS + RECIPE_ARMS:
            tok, dom, ctrl, typ, tgt = next(loader)
            if arm in FROZEN_BACKBONE_ARMS and not (dom == domain_index["nlp"]).any():
                # force_domain guarantees nlp presence per synthetic DOC, but PackedRoutedStream
                # packs fixed windows from a buffer spanning multiple docs - a window can still
                # land entirely inside a non-nlp span from a different doc. For a frozen-backbone
                # arm that means zero path to any trainable tensor; skip rather than crash.
                if step % LOG_EVERY == 0:
                    print(f"step {step}/{total_steps}  SKIPPED (no nlp tokens in window)", flush=True)
                continue
            tok, dom, ctrl, typ, tgt = (t.to(device, non_blocking=True) for t in (tok, dom, ctrl, typ, tgt))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
            if arm in ALIGN_ARMS and step % ALIGN_LOSS_EVERY == 0:
                # CORAL-style cross-domain embedding-table alignment (see domain_embedding_
                # alignment_loss in mot_routed_model.py) - operates on the tables themselves,
                # not batch activations, so it's independent of what happened to be in THIS
                # batch and safe to add only periodically. raw_model, not model: DDP wraps
                # forward() but doesn't forward custom methods to the wrapped module.
                loss = loss + ALIGN_LOSS_WEIGHT * raw_model.domain_embedding_alignment_loss()
        elif arm == "routed19":
            # step, not an internal call counter, drives the phase switch - correct across
            # resumes (start_step is loaded from the checkpoint), unlike a loader-internal
            # counter that would reset to 0 and misjudge the phase on every restart.
            src = phase1_loader if step <= routed19_phase1_steps else phase2_loader
            tok, dom, ctrl, typ, tgt = next(src)
            tok, dom, ctrl, typ, tgt = (t.to(device, non_blocking=True) for t in (tok, dom, ctrl, typ, tgt))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
        elif arm == "hybrid":
            tok, dom, ctrl, typ, tgt = _hybrid_batch(step, domains, domain_index, loaders, loader, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
            controller_loss = parts["_content"]
        elif arm in ("routed2", "routed3", "routed4"):
            # routed4 (combined): no switch_weight kwarg, weight is an internal learned
            # parameter. Same call shape as routed2/routed3 otherwise.
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = (t.to(device, non_blocking=True) for t in (tok, dom, ctrl, typ, tgt))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
            controller_loss = parts["_content"]
        elif arm in ("routed5", "routed6"):
            # decoupled-head-only / long-context-only - fixed switch_weight, standard call
            # shape identical to plain routed. Not in the controller list, so no
            # controller_loss needed here.
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = (t.to(device, non_blocking=True) for t in (tok, dom, ctrl, typ, tgt))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
        elif arm in ("pooled", "pooled2"):
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = (t.to(device, non_blocking=True) for t in (tok, dom, ctrl, typ, tgt))
            adv_lambda = min(1.0, step / max(1, ADV_LAMBDA_RAMP_STEPS))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ,
                                     switch_weight=SWITCH_WEIGHT, adv_lambda=adv_lambda)
            main_parts = [v for k, v in parts.items() if not k.startswith("_")]
            controller_loss = sum(main_parts) / max(len(main_parts), 1)
        else:
            ids = next(loader).to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

        if world_size > 1:
            # Real, untested-until-now hazard for this architecture specifically: domain
            # composition per synthetic doc is randomized, so a given 1024-token micro-batch
            # can easily contain zero tokens for one or more domains (math/science especially).
            # That domain's head/embedding gets literally no gradient path that step - DDP's
            # default (find_unused_parameters=False, which static_graph=True above requires
            # anyway) expects every registered parameter to participate in every backward call,
            # so this would hang or crash mid-run, not fail loudly at launch. Forcing a
            # (numerically inert - the 0.0 multiplier means it changes nothing about what's
            # being optimized) dependency on every parameter is the standard, cheap fix -
            # restructuring PackedRoutedStream to guarantee per-domain quotas in every micro-
            # batch would also work but is a real data-pipeline change, not worth the risk this
            # close to launch when this is a one-line, correctness-only addition.
            loss = loss + 0.0 * sum(p.sum() for p in raw_model.parameters())

        # loss.item() forces a host<->device sync every micro-step - real, measurable cost at
        # this scale (llm.c-class throughput work flags exactly this pattern) and unnecessary
        # for every arm the project currently runs, none of which use a controller. Controller
        # arms (pooled/pooled2/hybrid/routed2/routed3/routed4 - all retired, none of the final
        # four) still need the per-step float for should_skip/observe's spike-guard logic, so
        # they keep the old unconditional sync; everything else defers the sync to the
        # LOG_EVERY block below (running accumulates the raw GPU tensor in the meantime).
        if controller is not None:
            loss_val = loss.item()
            controller_loss_val = float(controller_loss) if arm in ("pooled", "pooled2", "hybrid", "routed2", "routed3", "routed4") else loss_val
            if controller.should_skip(loss_val):
                opt.zero_grad()
                if step % LOG_EVERY == 0:
                    print(f"step {step}/{total_steps}  SKIPPED (loss={loss_val:.3f}, "
                          f"guard tripped)  {controller.state()}", flush=True)
                continue

        # DDP all-reduces gradients on every .backward() by default - correct but wasteful
        # during grad-accumulation micro-steps, since only the LAST one before opt.step()
        # actually needs synced gradients. no_sync() defers that all-reduce to the sync step.
        is_sync_step = (step % grad_accum == 0)
        sync_ctx = model.no_sync() if (world_size > 1 and not is_sync_step) else contextlib.nullcontext()
        try:
            with sync_ctx:
                (loss / grad_accum).backward()
        except RuntimeError as e:
            # Belt-and-braces for FROZEN_BACKBONE_ARMS: the upstream domain-presence guard
            # (above) reduces how often a batch's loss ends up disconnected from every
            # trainable tensor, but didn't eliminate it in practice (observed live on
            # routed16 - crashed again post-guard at a batch that DID contain nlp-domain
            # positions, so the exact internal trigger wasn't fully pinned down). Catch the
            # actual symptom directly rather than keep chasing the precise precondition.
            if "does not require grad" not in str(e):
                raise
            if step % LOG_EVERY == 0:
                print(f"step {step}/{total_steps}  SKIPPED backward (loss disconnected from "
                      f"every trainable tensor)", flush=True)
            opt.zero_grad()
            continue
        running += loss.detach()  # no .item() here - see the sync-avoidance comment above
        running_n += 1
        if is_sync_step:
            opt.step()
            opt.zero_grad()

        if controller is not None:
            controller.observe(controller_loss_val)

        # Forward-only (torch.no_grad()) inside _held_out_ce(), never a backward() call, so it
        # never touches DDP's gradient-sync hooks - safe to run on rank 0 alone without other
        # ranks needing to participate (no collective op gets triggered, no hang risk).
        if is_main and val_iter is not None and (step % EVAL_EVERY == 0 or step == total_steps):
            val_ce = _held_out_ce()
            history.append({"step": step, "val_ce": round(val_ce, 4), "elapsed": round(time.time() - t0)})
            print(f"  >>> held-out val CE @ step {step}: {val_ce:.4f} (over {VAL_BATCHES} val batches)", flush=True)

        if is_main and step % LOG_EVERY == 0:
            avg = (running / max(running_n, 1)).item()  # the one sync per LOG_EVERY steps, not per micro-step
            entry = {"step": step, "loss": round(avg, 4), "elapsed": round(time.time() - t0)}
            if controller is not None:
                entry.update(controller.state())
            history.append(entry)
            ctl = f"  {controller.state()}" if controller is not None else ""
            print(f"step {step}/{total_steps}  loss={avg:.4f}  elapsed={time.time()-t0:.0f}s{ctl}", flush=True)
            running, running_n = torch.zeros((), device=device), 0

        if is_main and (step % CHECKPOINT_EVERY == 0 or step == total_steps):
            # raw_model, not model - model.state_dict() on a DDP-wrapped module prefixes every
            # key with "module.", which would silently break every eval function (stage2_modal.
            # py's evaluate()/evaluate_lambada()) that loads a plain (non-DDP) model class.
            # Write to a .tmp path then os.replace (atomic on POSIX) rather than torch.save
            # directly to the final path - this pod has genuinely restarted mid-run multiple
            # times tonight (RunPod migrations, watchdog restarts); a crash mid-write to the
            # real filename would leave a truncated .pt that the NEXT launch's resume logic
            # would try to load, corrupting the run instead of just losing the latest interval.
            path = CKPT_DIR / f"{ckpt_prefix}_step{step}.pt"
            tmp_path = path.with_suffix(".pt.tmp")
            torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "domain_vocab_sizes": bundle.domain_vocab_sizes, "history": history}, tmp_path)
            os.replace(tmp_path, path)
            print(f"checkpoint saved: {path}", flush=True)
            # prune older checkpoints for this arm - keep only the newest, disk isn't infinite
            for old in _latest_checkpoint()[1:]:
                Path(old).unlink(missing_ok=True)

    print(f"\nDONE {arm}: {total_steps} steps in {time.time()-t0:.0f}s", flush=True)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["calibrate", "train"])
    parser.add_argument("--arm", required=True,
                         choices=["mot", "baseline", "sota", "routed", "pooled", "hybrid", "pooled2",
                                  "routed2", "routed3", "routed4", "routed5", "routed6", "routed7",
                                  "routed8", "routed9", "routed10", "routed11", "routed12", "routed13",
                                  "routed14", "routed15", "routed16", "routed17", "routed18", "routed19",
                                  "routed20", "routed21", "routed22", "routed23", "routed24",
                                  "routed25", "routed26", "routed27", "routed28", "routed29",
                                  "routed30", "routed31", "routed32", "routed33", "routed35"])
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--scale", choices=["base", "large"], default="base",
                         help="'large' (mot/baseline only) uses LARGE_MODEL_CFG for the scale test")
    parser.add_argument("--seed", type=int, default=0,
                         help="torch.manual_seed() for reproducible dropout/init/CUDA RNG - no arm "
                              "launch controlled this before, so every 'replicate' to date varied by "
                              "accident, not design. Same --seed across an old-vs-new settings pair is "
                              "what makes a loss-parity check meaningful; a genuine seed replicate should "
                              "pass a DIFFERENT --seed on purpose, not omit it.")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.mode == "calibrate":
        sec_per_step = calibrate(args.arm, args.steps or 150, scale=args.scale)
        print(f"\n--- extrapolation ---")
        print(f"{sec_per_step:.3f} sec/step x {MAX_STEPS} steps = {sec_per_step*MAX_STEPS/3600:.2f} GPU-hours (config MAX_STEPS)")
    else:
        train(args.arm, max_steps=args.steps or None, scale=args.scale)
