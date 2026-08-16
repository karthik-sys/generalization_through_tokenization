"""Stage-2 model/training config (spec §8 stage 2: ~50-100M params, real GPU compute).

Sizing check (see decision log): at these settings MoT lands at ~89.1M params, the
unified-BPE baseline at ~68.6M - both in spec's target range. Vocab sizes are 3x
stage-1's (24000 vs 8000 per BPE domain, NLP's syllable/morpheme caps at 6000 vs 4000)
to match the much larger stage-2 corpora; going further risks the same head-layer-
dominates-everything issue found at stage 1 (heads were 3.6M of MoT's 8.1M there,
55.5M of 89.1M here - proportionally similar, not worse).
"""

DOMAIN_VOCAB_SIZES = {
    "code": 24000,
    "math": 24000,
    "science": 24000,
}
NLP_SURFACE_VOCAB = 24000
NLP_SYLLABLE_VOCAB = 6000
NLP_MORPHEME_VOCAB = 6000

BASELINE_VOCAB_SIZE = 48000
SOTA_ENCODING = "cl100k_base"  # unchanged from stage 1 - already a real 100,277-token SOTA tokenizer, no reason to swap

MODEL_CFG = dict(emb_dim=128, d_model=512, n_heads=8, ffn_dim=2048, n_layers=6, max_seq_len=1024)
BACKBONE_ONLY_CFG = {k: v for k, v in MODEL_CFG.items() if k != "emb_dim"}

# BATCH_SIZE 16 OOM'd a real T4 (14.56GiB) during calibration - the logits tensor alone
# is batch*seq*vocab*4 bytes, which at batch 16 / seq 1024 is 2.4GB for nlp's 36k vocab
# and 6.6GB for cl100k_base's 100k vocab. Batch 4 keeps the largest arm's logits under
# 1GB; GRAD_ACCUM_STEPS raised to hold the effective batch size at 64.
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16  # effective batch size 64
LR = 3e-4
WARMUP_STEPS = 500
MAX_STEPS = 20000  # tune against your free-tier GPU's time budget (Colab T4: ~a few hours/day)
CHECKPOINT_EVERY = 1000
LOG_EVERY = 50

# MoTRoutedModel only. Measured on the real corpus (src/train_routed.py): switches are
# ~1-in-461 positions - unweighted cross-entropy left switch_accuracy at 0.000 across 4
# full CPU epochs. 50x upweighting fixed it (0.000 -> 0.34 at epoch 1, settling ~0.23-0.27).
# Same value carried over here; stage-2 scale hasn't been calibrated against this constant
# yet, so treat it as a starting point, not a re-derived optimum.
SWITCH_WEIGHT = 50.0

# MoTPooledModel (arm 5) only. Gradient-reversal strength ramps linearly 0 -> 1 over this
# many steps, standard DANN practice: adversarial pressure applied before the model can
# predict anything starves the main objective. Also a useful safety property - at lambda 0
# the pooled arm is exactly the routed arm plus an inert pooling path, so the ramp's early
# steps double as a correctness check that the new module didn't break the base model.
ADV_LAMBDA_RAMP_STEPS = 10000

# MoTPooledModel loss shaping (arm 5). See mot_pooled_model.py docstring.
#   FOCAL_GAMMA:      (1-p)^gamma reweighting of the main CE. 2.0 is the value from the
#                     focal-loss paper; 0.0 recovers plain CE.
#   CONFIDENCE_WEIGHT: weight on the calibration auxiliary (predict-your-own-correctness).
#                     Small - it's a side signal, not meant to dominate the main objective.
FOCAL_GAMMA = 2.0
CONFIDENCE_WEIGHT = 0.1

# Permissive-only, deliberately excludes copyleft (gpl-*, agpl-*, lgpl-*) - codeparrot/
# github-code includes a license column precisely because it does not filter for you.
CODE_LICENSE_ALLOWLIST = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc"}

# Full-size stage-2 sources (docs/dataset_methodology.md) - streamed, never fully downloaded.
STREAM_SOURCES = {
    # Code source history:
    #   the-stack-v2-dedup - gated with NO auto-approval, and stores only SWHID pointers;
    #     real content needs AWS creds + a Software Heritage/INRIA agreement. Unusable.
    #   the-stack-dedup (v1) - has inline `content`, but also gated with no access yet.
    #   the-stack-smol - used through 2026-08-16. Accessible, but only ~10k Python files -
    #     small enough that a 150k-step run cycles the whole pool many times over, and (a
    #     real methodological problem, not just a training-diversity one) leaves no genuinely
    #     unseen Python left for held-out eval, forcing a JS-as-stand-in workaround there.
    #   codeparrot/github-code - swapped in 2026-08-16. Ungated, large (real GitHub source
    #     across languages), confirmed streamable. Filtered to Python + permissive licenses
    #     (see CODE_LICENSE_ALLOWLIST) - the license column exists precisely because this
    #     dataset does NOT pre-filter by license, so filtering here is a real choice, not
    #     a formality. Fixes the held-out contamination problem too: large enough that a
    #     `.skip()`-based held-out split is genuinely unseen, same as the other domains.
    "code": {"path": "codeparrot/github-code", "name": None, "gated": False},
    "math": {"path": "open-web-math/open-web-math", "name": None, "gated": False},
    # EleutherAI/proof-pile-2's "arxiv" config uses a legacy loading script whose zstd
    # decompression path is broken with current zstandard tooling (reproducible
    # ZstdError: "Unknown frame descriptor", confirmed 2026-08-15, not transient).
    # Swapped for gfissore/arxiv-abstracts-2021, already proven reliable in stage 1
    # (2M rows available, plenty for stage-2 scale even though it's abstracts-only
    # rather than full papers).
    "science": {"path": "gfissore/arxiv-abstracts-2021", "name": None, "gated": False},
    "nlp": {"path": "HuggingFaceFW/fineweb", "name": "sample-10BT", "gated": False},
}
# Human-readable labels for the five training arms. The short keys (mot/baseline/sota/
# routed/pooled) are load-bearing - they're embedded in checkpoint filenames that the live
# runs write and auto-resume from, so they must NOT be renamed mid-flight. These labels are
# the display layer: printed at run start and used by the dashboard so it's unambiguous
# which arm is which without decoding the key.
ARM_LABELS = {
    "mot": "MoT (disjoint tokenizers, one domain per forward)",
    "baseline": "Unified-BPE baseline (single 48k tokenizer)",
    "sota": "SOTA tokenizer (cl100k_base, 100k vocab)",
    "routed": "MoT + mid-sequence routing (predicts domain switches)",
    "pooled": "MoT + routing + PMA/DANN + fitting-loss (focal, per-domain-norm, confidence)",
}
DOMAIN_TAG = {
    "code": "<domain:code>",
    "math": "<domain:math>",
    "science": "<domain:science>",
    "nlp": "<domain:nlp>",
}
