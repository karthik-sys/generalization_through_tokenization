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
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
from src.model.mot_routed_combined_model import MoTRoutedCombinedModel
from src.model.mot_routed_decoupled_model import MoTRoutedDecoupledModel
from src.model.mot_routed_model import MoTRoutedModel
from src.model.stage2_config import (
    ADV_LAMBDA_RAMP_STEPS, ARM_LABELS, BACKBONE_ONLY_CFG, BATCH_SIZE, BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS,
    CHECKPOINT_EVERY, CONFIDENCE_WEIGHT, COOLDOWN_BACKBONE_LR_SCALE, FOCAL_GAMMA, GRAD_ACCUM_STEPS,
    HYBRID_NATURAL_DATA_FRACTION, LOG_EVERY, LONGCTX_MODEL_CFG, LR, MAX_STEPS, MODEL_CFG,
    ROUTED3_MAX_DOMAINS, ROUTED3_MIN_DOMAINS, ROUTED3_SNIPPET_WORDS, SWITCH_WEIGHT, WARM_START_PARENT,
    WARMUP_STEPS,
)

TOKENIZER_DIR = str(REPO_ROOT / "tokenizers_stage2")
# arm in ("routed7", "routed8"): nlp tokenizer retrained on OpenWebText (see
# scripts/retrain_nlp_tokenizer_openwebtext.py). code/math/science still load from
# TOKENIZER_DIR above, unchanged.
OWT_NLP_TOKENIZER_DIR = str(REPO_ROOT / "tokenizers_stage2_owt" / "nlp")
# arm in ("routed9", "routed10"): nlp tokenizer retrained on PG-19 books (see
# scripts/retrain_nlp_tokenizer_books.py).
BOOKS_NLP_TOKENIZER_DIR = str(REPO_ROOT / "tokenizers_stage2_books" / "nlp")
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


def _warm_start_from_parent(model, parent_ckpt_prefix: str, device: str) -> bool:
    """arms routed9/routed10: initialize from the parent arm's latest checkpoint instead of
    random init, EXCEPT the nlp domain's embedding/type_embedding/projection/head. Those
    describe token ids under the PARENT's nlp tokenizer (FineWeb- or OpenWebText-fit) - a
    different vocabulary than routed9/routed10's PG-19-fit tokenizer, even though the tensor
    SHAPES match (same NLP_*_VOCAB constants everywhere). Loading them would be confidently
    wrong, not just stale, so they're left at fresh random init instead. The shared backbone
    and code/math/science tables are untouched by the nlp tokenizer swap and transfer directly.
    """
    import glob
    import re
    paths = glob.glob(str(CKPT_DIR / f"{parent_ckpt_prefix}_step*.pt"))
    if not paths:
        print(f"WARNING: no parent checkpoint found for prefix '{parent_ckpt_prefix}' in "
              f"{CKPT_DIR} - starting from random init instead of a warm start", flush=True)
        return False
    latest = sorted(paths, key=lambda p: int(re.search(r"step(\d+)", p).group(1)), reverse=True)[0]
    parent_ckpt = torch.load(latest, map_location=device)
    parent_state = parent_ckpt["model"]
    own_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in parent_state.items():
        if k.startswith(_NLP_PARAM_PREFIXES):
            skipped += 1
            continue
        if k in own_state and own_state[k].shape == v.shape:
            own_state[k] = v
            loaded += 1
        else:
            print(f"  warm-start: key '{k}' missing/shape-mismatched in child model, skipping", flush=True)
    model.load_state_dict(own_state)
    print(f"warm-started from {latest} (parent step {parent_ckpt['step']}): "
          f"{loaded} tensors loaded, {skipped} nlp-domain tensors left at fresh init", flush=True)
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
        raise ValueError(f"scale=large supports mot/baseline/routed/routed3/routed7/routed10, not {arm}")
    if arm == "mot":
        return MoTModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
    if arm in ("routed", "routed8", "routed9"):
        # routed8/routed9: identical architecture to plain routed - only the nlp data source
        # (OpenWebText or PG-19 books, applied by _apply_openwebtext_nlp_source /
        # _apply_books_nlp_source in calibrate()/train()) and, for routed9, the warm-start +
        # mixture upweighting, differ.
        return MoTRoutedModel(domain_vocab_sizes=bundle.domain_vocab_sizes, **MODEL_CFG).to(device)
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
    device = "cuda"
    print(f"CUDA available: {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(0)}", flush=True)
    if arm in ("routed7", "routed10"):
        scale = "large"  # always large-scale, regardless of what --scale was passed
    if arm in ("routed7", "routed8"):
        _apply_openwebtext_nlp_source()
    if arm in ("routed9", "routed10"):
        _apply_books_nlp_source()

    nlp_tok_dir = OWT_NLP_TOKENIZER_DIR if arm in ("routed7", "routed8") else \
        (BOOKS_NLP_TOKENIZER_DIR if arm in ("routed9", "routed10") else None)
    bundle = TokenizerBundle(tokenizer_dir=TOKENIZER_DIR, nlp_tokenizer_dir=nlp_tok_dir)
    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    model = _build_model(arm, bundle, device, scale)
    print(f"{arm} ({scale}) params: {model.num_params():,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    if arm in ("mot", "hybrid"):
        domains = list(bundle.domain_vocab_sizes)
        loaders = {
            d: iter(DataLoader(PackedDomainStream(d, bundle.encode_domain, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
            for d in domains
        }
    if arm in ("routed9", "routed10"):
        loader = iter(DataLoader(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in ("routed", "pooled", "pooled2", "hybrid", "routed2", "routed5", "routed7", "routed8"):
        loader = iter(DataLoader(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm == "routed3":
        loader = iter(DataLoader(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            min_domains=ROUTED3_MIN_DOMAINS, max_domains=ROUTED3_MAX_DOMAINS,
            snippet_words=ROUTED3_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in ("routed4", "routed6"):  # both use 2x context
        loader = iter(DataLoader(PackedRoutedStream(bundle, domain_index, LONGCTX_MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm not in ("mot",):
        encode_fn = bundle.encode_baseline if arm == "baseline" else bundle.encode_sota
        loader = iter(DataLoader(PackedMixedStream(encode_fn, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))

    t0 = time.time()
    for step in range(1, steps + 1):
        if arm == "mot":
            domain = domains[step % len(domains)]
            _, ids, types = next(loaders[domain])
            ids, types = ids.to(device), types.to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda"):
                logits = model(domain, inp, types[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        elif arm == "hybrid":
            tok, dom, ctrl, typ, tgt = _hybrid_batch(step, domains, domain_index, loaders, loader, device)
            with torch.autocast("cuda"):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
        elif arm in ("routed2", "routed3", "routed4"):
            # routed4 (combined: decoupled head + learned weight) takes no switch_weight
            # kwarg - it's an internal learned parameter, same call shape as routed2/routed3.
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
            with torch.autocast("cuda"):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
        elif arm in ("routed", "pooled", "pooled2", "routed5", "routed6", "routed7", "routed8", "routed9", "routed10"):
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
            with torch.autocast("cuda"):
                if arm in ("pooled", "pooled2"):
                    loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ,
                                     switch_weight=SWITCH_WEIGHT, adv_lambda=1.0)
                else:
                    loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
        else:
            ids = next(loader).to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda"):
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
    print(f"\nCALIBRATION RESULT: {sec_per_step:.3f} sec/step on {torch.cuda.get_device_name(0)}", flush=True)
    return sec_per_step


def train(arm: str, max_steps: int | None = None, scale: str = "base") -> list:
    device = "cuda"
    total_steps = max_steps or MAX_STEPS
    print(f"device: {torch.cuda.get_device_name(0)}  arm: {arm}  steps: {total_steps}", flush=True)
    print(f"ARM: {arm}  =  {ARM_LABELS.get(arm, arm)}", flush=True)
    if arm in ("routed7", "routed10"):
        scale = "large"  # always large-scale, regardless of what --scale was passed
    if arm in ("routed7", "routed8"):
        _apply_openwebtext_nlp_source()
    if arm in ("routed9", "routed10"):
        _apply_books_nlp_source()

    nlp_tok_dir = OWT_NLP_TOKENIZER_DIR if arm in ("routed7", "routed8") else \
        (BOOKS_NLP_TOKENIZER_DIR if arm in ("routed9", "routed10") else None)
    bundle = TokenizerBundle(tokenizer_dir=TOKENIZER_DIR, nlp_tokenizer_dir=nlp_tok_dir)
    domain_index = {d: i for i, d in enumerate(bundle.domain_vocab_sizes)}
    model = _build_model(arm, bundle, device, scale)
    print(f"{arm} ({scale}) params: {model.num_params():,}", flush=True)

    # routed9/routed10 ("cooldown" arms): the nlp branch is reinitialized fresh (see
    # _warm_start_from_parent) while everything else is warm-started, so it needs to catch up
    # fast without the warm-started backbone/other-domain tables being dragged along at the
    # same rate - COOLDOWN_BACKBONE_LR_SCALE (0.1) throttles the latter, applied per-step below
    # alongside the normal cosine schedule via each param group's "lr_scale".
    if arm in WARM_START_PARENT:
        nlp_params = [p for n, p in model.named_parameters() if n.startswith(_NLP_PARAM_PREFIXES)]
        other_params = [p for n, p in model.named_parameters() if not n.startswith(_NLP_PARAM_PREFIXES)]
        opt = torch.optim.AdamW([
            {"params": nlp_params, "lr_scale": 1.0},
            {"params": other_params, "lr_scale": COOLDOWN_BACKBONE_LR_SCALE},
        ], lr=LR)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_prefix = f"large_{arm}" if scale == "large" else arm  # keep large runs off base names

    def _latest_checkpoint() -> str | None:
        import glob
        import re
        paths = glob.glob(str(CKPT_DIR / f"{ckpt_prefix}_step*.pt"))
        return sorted(paths, key=lambda p: int(re.search(r"step(\d+)", p).group(1)), reverse=True)

    start_step = 1
    resumed_own = False
    for cand in _latest_checkpoint():
        try:
            ckpt = torch.load(cand, map_location=device)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            start_step = ckpt["step"] + 1
            print(f"resumed from {cand} at step {start_step}", flush=True)
            resumed_own = True
            break
        except Exception as e:
            print(f"checkpoint {cand} failed to load ({type(e).__name__}: {e}); trying older", flush=True)
            continue
    if not resumed_own:
        warm_started = arm in WARM_START_PARENT and _warm_start_from_parent(model, WARM_START_PARENT[arm], device)
        if not warm_started:
            print(f"no checkpoint found for {arm} in {CKPT_DIR} - starting from step 1", flush=True)

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
    if arm in ("routed9", "routed10"):
        loader = iter(DataLoader(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            force_domain="nlp", force_domain_snippet_words=BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in ("routed", "pooled", "pooled2", "hybrid", "routed2", "routed5", "routed7", "routed8"):
        loader = iter(DataLoader(PackedRoutedStream(bundle, domain_index, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm == "routed3":
        loader = iter(DataLoader(PackedRoutedStream(
            bundle, domain_index, MODEL_CFG["max_seq_len"],
            min_domains=ROUTED3_MIN_DOMAINS, max_domains=ROUTED3_MAX_DOMAINS,
            snippet_words=ROUTED3_SNIPPET_WORDS,
        ), batch_size=BATCH_SIZE))
    elif arm in ("routed4", "routed6"):  # both use 2x context
        loader = iter(DataLoader(PackedRoutedStream(bundle, domain_index, LONGCTX_MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))
    elif arm not in ("mot",):
        encode_fn = bundle.encode_baseline if arm == "baseline" else bundle.encode_sota
        loader = iter(DataLoader(PackedMixedStream(encode_fn, MODEL_CFG["max_seq_len"]), batch_size=BATCH_SIZE))

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
    if arm in ("routed", "pooled", "pooled2", "hybrid", "routed2", "routed3", "routed4", "routed5",
               "routed6", "routed7", "routed8", "routed9", "routed10"):
        rs3 = arm == "routed3"
        upweighted = arm in ("routed9", "routed10")
        seq_len = LONGCTX_MODEL_CFG["max_seq_len"] if arm in ("routed4", "routed6") else MODEL_CFG["max_seq_len"]
        val_stream = PackedRoutedStream(
            bundle, domain_index, seq_len, seed=VAL_SEED,
            min_domains=ROUTED3_MIN_DOMAINS if rs3 else 2,
            max_domains=ROUTED3_MAX_DOMAINS if rs3 else 4,
            snippet_words=ROUTED3_SNIPPET_WORDS if rs3 else 250,
            force_domain="nlp" if upweighted else None,
            force_domain_snippet_words=BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS if upweighted else None,
        )
        val_iter = iter(DataLoader(val_stream, batch_size=BATCH_SIZE))

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
                    keep = dt < bundle.domain_vocab_sizes[d]  # content tokens only, drop switch targets
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
            g["lr"] = LR * lr_at(step) * lr_mult * g.get("lr_scale", 1.0)

        if arm == "mot":
            domain = domains[step % len(domains)]
            _, ids, types = next(loaders[domain])
            ids, types = ids.to(device), types.to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda"):
                logits = model(domain, inp, types[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        elif arm in ("routed", "routed7", "routed8", "routed9", "routed10"):
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
            with torch.autocast("cuda"):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
        elif arm == "hybrid":
            tok, dom, ctrl, typ, tgt = _hybrid_batch(step, domains, domain_index, loaders, loader, device)
            with torch.autocast("cuda"):
                loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
            controller_loss = parts["_content"]
        elif arm in ("routed2", "routed3", "routed4"):
            # routed4 (combined): no switch_weight kwarg, weight is an internal learned
            # parameter. Same call shape as routed2/routed3 otherwise.
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
            with torch.autocast("cuda"):
                loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ)
            controller_loss = parts["_content"]
        elif arm in ("routed5", "routed6"):
            # decoupled-head-only / long-context-only - fixed switch_weight, standard call
            # shape identical to plain routed. Not in the controller list, so no
            # controller_loss needed here.
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
            with torch.autocast("cuda"):
                loss, _ = model(tok, dom, ctrl, targets=tgt, type_ids=typ, switch_weight=SWITCH_WEIGHT)
        elif arm in ("pooled", "pooled2"):
            tok, dom, ctrl, typ, tgt = next(loader)
            tok, dom, ctrl, typ, tgt = tok.to(device), dom.to(device), ctrl.to(device), typ.to(device), tgt.to(device)
            adv_lambda = min(1.0, step / max(1, ADV_LAMBDA_RAMP_STEPS))
            with torch.autocast("cuda"):
                loss, parts = model(tok, dom, ctrl, targets=tgt, type_ids=typ,
                                     switch_weight=SWITCH_WEIGHT, adv_lambda=adv_lambda)
            main_parts = [v for k, v in parts.items() if not k.startswith("_")]
            controller_loss = sum(main_parts) / max(len(main_parts), 1)
        else:
            ids = next(loader).to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            with torch.autocast("cuda"):
                logits = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

        loss_val = loss.item()
        controller_loss_val = float(controller_loss) if arm in ("pooled", "pooled2", "hybrid", "routed2", "routed3", "routed4") else loss_val
        if controller is not None and controller.should_skip(loss_val):
            opt.zero_grad()
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
            controller.observe(controller_loss_val)

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
            path = CKPT_DIR / f"{ckpt_prefix}_step{step}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "domain_vocab_sizes": bundle.domain_vocab_sizes, "history": history}, path)
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
                                  "routed8", "routed9", "routed10"])
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--scale", choices=["base", "large"], default="base",
                         help="'large' (mot/baseline only) uses LARGE_MODEL_CFG for the scale test")
    args = parser.parse_args()

    if args.mode == "calibrate":
        sec_per_step = calibrate(args.arm, args.steps or 150, scale=args.scale)
        print(f"\n--- extrapolation ---")
        print(f"{sec_per_step:.3f} sec/step x {MAX_STEPS} steps = {sec_per_step*MAX_STEPS/3600:.2f} GPU-hours (config MAX_STEPS)")
    else:
        train(args.arm, max_steps=args.steps or None, scale=args.scale)
