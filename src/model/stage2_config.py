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

# Scale test (arms "motlarge" / "baselinelarge"): ~3-4x the backbone to probe THE open
# question - does MoT's advantage over unified-BPE survive a real parameter increase, or is
# it a small-scale artifact? d_model 512->768, layers 6->12, heads 8->12, ffn 2048->3072,
# emb_dim 128->192; max_seq_len held at 1024 so the data pipeline is unchanged. Only a MATCHED
# pair is informative (scaling just the winner tells you nothing), so both arms use this.
LARGE_MODEL_CFG = dict(emb_dim=192, d_model=768, n_heads=12, ffn_dim=3072, n_layers=12, max_seq_len=1024)
LARGE_BACKBONE_ONLY_CFG = {k: v for k, v in LARGE_MODEL_CFG.items() if k != "emb_dim"}

# arm="routed6" only: plain MoTRoutedModel at 2x context (1024->2048), nothing else changed -
# more room between switches for the model to re-establish domain state, without touching
# loss weighting at all (unlike routed2/3/hybrid, which all failed by that route).
LONGCTX_MODEL_CFG = dict(emb_dim=128, d_model=512, n_heads=8, ffn_dim=2048, n_layers=6, max_seq_len=2048)

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

# Periodic held-out eval during training (idea 4a): every EVAL_EVERY steps, compute plain
# cross-entropy on VAL_BATCHES batches drawn from a held-out val stream (a distinct seed /
# skip offset from training), so we watch a *generalization* trajectory alongside the
# training-loss trajectory rather than only seeing held-out numbers after the run. Catches
# overfitting/plateau live and tells us the real "when to stop". Logged into `history` so it
# rides along in checkpoints and the dashboard. Lightweight plain-CE, not the full BPB/LAMBADA
# suite (those need byte-accounting / an external dataset and belong in the post-hoc eval).
EVAL_EVERY = 10000
VAL_BATCHES = 20
VAL_SEED = 9999  # held-out: distinct from training's seed=0 so synthetic composition differs

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
    #     across languages), filtered to Python + permissive licenses (see
    #     CODE_LICENSE_ALLOWLIST - the license column exists precisely because this dataset
    #     does NOT pre-filter by license). Fixes the held-out contamination problem too:
    #     large enough that a `.skip()`-based held-out split is genuinely unseen.
    #     NOTE: the dataset's own loading script is a legacy HF "dataset script", which
    #     current `datasets` versions refuse to run at all ("Dataset scripts are no longer
    #     supported") - confirmed broken on Modal's image even though it loaded fine on an
    #     older local `datasets` version, which is exactly the kind of environment mismatch
    #     that bit proof-pile-2 earlier. Fix: HF auto-converts most datasets to Parquet on
    #     a `refs/convert/parquet` branch; that conversion organises github-code into
    #     per-language folders, so `data_files` targets Python directly without downloading
    #     the other ~20 languages.
    "code": {"path": "codeparrot/github-code", "name": None, "gated": False,
             "revision": "refs/convert/parquet", "data_files": "Python-all/partial-train/*.parquet"},
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
    "hybrid": "MoT-Routed-Adaptive hybrid (GradNorm switch loss + blended natural/synthetic data)",
    "pooled2": "Pooled v2 (GradNorm loss, sparse top-2/16 routing, 256-chunk PMA, no confidence head)",
    "routed2": "Routed + GradNorm switch loss ONLY (data unchanged - isolates hybrid's loss-fix credit)",
    "routed3": "Routed + GradNorm + maximized cross-domain density (always 4 domains/doc, shorter snippets)",
    "routed4": "Routed + LEARNED switch weight (Kendall uncertainty weighting, starts at 50, standard loss - no GradNorm)",
    "routed5": "Routed + gradient-decoupled switch head (separate classifier, detached from backbone; switch_weight=50 fixed)",
    "routed6": "Routed + 2x context (1024->2048), otherwise unchanged",
    "routed7": "Routed, LARGE scale, nlp domain sourced from OpenWebTextCorpus instead of FineWeb "
               "(code/math/science unchanged) - closes the data-source gap vs GPT-2 for the domain "
               "LAMBADA actually routes through",
    "routed8": "Routed, BASE (champion) scale, nlp domain sourced from OpenWebTextCorpus - same "
               "mechanism as routed7, at champion scale instead of large. Paired with routed7 run "
               "to 600k steps: both target ~10B tokens/domain (llm.c's dedicated-GPT-2-124M-training "
               "budget), testing whether per-domain token-volume equivalence matters independent of "
               "model scale",
    "routed9": "Routed, BASE (champion) scale, warm-started from the routed champion checkpoint - "
               "nlp sourced from PG-19 books (not OWT), upweighted to ~55% of the token mixture, "
               "nlp embedding/head reinitialized (new vocab), backbone+other domains at 0.1x LR "
               "('cooldown', 50k steps) - targets LAMBADA specifically via book-narrative data and "
               "mixture share, not just data-source matching or raw volume",
    "routed10": "Routed, LARGE scale, warm-started from routed7's checkpoint - same books/upweight/"
               "cooldown treatment as routed9, at large scale. routed9's large-scale pair.",
    "routed11": "Bet 1 (copy gate): routed8 + a pointer/copy mechanism on the nlp head - "
               "warm-started, no reinit (same OWT tokenizer as routed8), continued training. "
               "Targets LAMBADA's retrieval failure mode directly (target word often appears "
               "earlier in the passage; a plain softmax head has to reconstruct it from scratch).",
    "routed12": "Bet 2 (deep experts): routed8 + per-domain LoRA adapters on the top 2 "
               "transformer layers (zero-init, no-op at warm-start) - tests whether the fully-"
               "shared backbone's capacity, contended across 4 domains, is limiting the nlp head.",
    "routed13": "Bet 3, exploratory (precision head): routed8 + a margin-trained second scoring "
               "pathway over the nlp tokenizer's ~16k most-frequent surface tokens, blended "
               "additively into the vocab logits via a learned gate - attacks the "
               "calibration-vs-argmax gap (NLL spreads mass over synonyms; EM only rewards top-1) "
               "directly, in-batch hard negatives instead of a periodic mining loop.",
    "routed14": "Scaled-up routed8: LARGE scale (190.5M) + the copy-gate mechanism (Bet 1's "
               "architecture, reused as-is - it's already generic over model size), warm-started "
               "from routed7@150k (already OWT-sourced + large-scale, so this saves the 150k "
               "steps routed8 needed to reach the same starting point) rather than routed8 itself "
               "(89M - wrong param count to warm-start a 190M model from). Tests whether scale "
               "and the data/volume lever compound, now that both individually helped.",
    "routed15": "Round-2 Recipe 1 (control): routed8 + 100k more steps, NO new module, same "
               "mixture - exact same warm-start + 0.3x backbone LR treatment routed11/12/13 got, "
               "minus whatever each of them added. Settles whether routed12/13's small gains were "
               "real mechanism wins or just continued training.",
    "routed16": "Round-2 Recipe 2: copy-gate v2, backbone genuinely frozen (not just throttled) - "
               "routed11/14's post-mortem found the copy MECHANISM never hurt (0/60 real "
               "regressions) but the underlying vocab head collapsed under joint training, even "
               "at 0.3x LR. This freezes everything except copy_q/copy_k/copy_gate entirely, plus "
               "a conservative (very negative) gate-bias init, to isolate the mechanism cleanly.",
    "routed17": "Round-2 Recipe 3: nlp-heavy diet phase, NO reinit - continues routed8 with nlp "
               "upweighted to ~70% of the mixture (vs the standard ~25%), full tables intact "
               "(unlike routed9/10's reinit), nlp at full LR / everything else at "
               "COOLDOWN_BACKBONE_LR_SCALE. Pure data-distribution lever - the one thing "
               "confirmed to work twice already (source-match, volume) - deliberately NOT another "
               "architecture bet.",
    "routed18": "Round-2 Recipe 4: copy-structure data mining - upweights nlp documents whose own "
               "structure matches LAMBADA's construction (a content word recurs verbatim later in "
               "the same document), via a stream-level filter, not a new loss/module. Attacks the "
               "same 82.8%-copy-failure error class as Bet 1, but through data instead of "
               "architecture - lower risk given what happened to routed11/14.",
}

# --- Round 2 (routed15-18): informed by routed11-14's post-mortem ---------------------------
# routed15 reuses BET_BACKBONE_LR_SCALE with an EMPTY NEW_MODULE_MATCH entry (see
# train_stage2_pod.py's optimizer construction) - same 0.3x-everything treatment routed11/12/13's
# non-new-module params got, just with nothing new added, so it's a genuine matched control.
ROUND2_STEPS = 100000

# routed16 only: routed11/14's forensics (see conversation/commit history around their eval)
# showed the copy MECHANISM itself never hurt (0/60 real regressions on a real LAMBADA check),
# but the underlying vocab head's accuracy collapsed under the joint objective even at 0.3x LR -
# gradient from the copy pathway flows straight back through the SAME shared hidden states the
# vocab head depends on, unlike routed12's LoRA (whose adapter gradient doesn't touch the base
# weights directly). Fix: freeze everything except the new copy-gate params entirely (not just
# throttle), and start the gate conservative (very negative bias -> near-0 sigmoid at init,
# rather than routed11/14's bias~0 -> ~0.5 default) so it only turns on where genuinely useful.
FROZEN_BACKBONE_ARMS = ("routed16",)
COPYGATE_V2_BIAS_INIT = -4.0

# routed17 only: nlp mixture share target. force_domain_snippet_words this large (vs
# BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS=600 for routed9/10's ~55%) pushes nlp toward ~70%:
# 1200 / (1200 + 2*250) ~= 0.706 (E[other domains]=2, same arithmetic as
# stage2_routed_stream.py's docstring). No reinit here (unlike routed9/10) since routed17 reuses
# routed8's own OWT-fit tokenizer - the mixture-share lever is being tested in isolation this
# time, not confounded with a tokenizer swap.
DIET_PHASE2_NLP_SNIPPET_WORDS = 1200

# routed18 only: minimum word length + minimum token-gap for the copy-structure document filter
# (see _is_copy_structured in stage2_routed_stream.py) - a document qualifies if some real content
# word (not a stopword, length >= COPY_MINE_MIN_WORD_LEN) appears at least twice with at least
# COPY_MINE_MIN_GAP words between the first and a later occurrence. Loose on purpose: this is a
# cheap proxy for "has LAMBADA-shaped structure", not a precise match - too strict and the filter
# starves the stream (most documents get rejected, training stalls waiting on OpenWebText).
COPY_MINE_MIN_WORD_LEN = 5
COPY_MINE_MIN_GAP = 15

# routed11/12/13 ("bet" arms, all warm-started from routed8@575k, no reinit - same OWT
# tokenizer throughout, unlike routed9/10's tokenizer swap). Each adds ONE new small module;
# everything already trained (backbone, embeddings, heads) trains at BET_BACKBONE_LR_SCALE so
# routed8's weights aren't disturbed while the new mechanism catches up, mirroring routed9/10's
# cooldown pattern but with a gentler backbone throttle since nothing here was reinitialized.
# Match strings are checked via substring-containment against each param's full dotted name
# (not startswith - routed12's LoRA keys are nested several levels deep, e.g.
# "backbone.expert_blocks.0.ffn.lora_a.nlp.weight").
BET_BACKBONE_LR_SCALE = 0.3
BET_STEPS = 100000
NEW_MODULE_MATCH = {
    "routed11": ("copy_q.", "copy_k.", "copy_gate."),
    "routed12": ("lora_a.", "lora_b."),
    "routed13": ("precision_proj.", "precision_gate."),
    "routed14": ("copy_q.", "copy_k.", "copy_gate."),  # same mechanism as routed11, at large scale
    "routed15": (),  # control - no new module, ALL params at BET_BACKBONE_LR_SCALE
    "routed16": ("copy_q.", "copy_k.", "copy_gate."),  # same mechanism, but see FROZEN_BACKBONE_ARMS
}

# routed7 (extended) and routed8 target ~600,000 steps rather than the usual 150,000. At
# BATCH_SIZE*GRAD_ACCUM_STEPS*max_seq_len = 64*1024 tokens/step, 150,000 steps split evenly
# across 4 domains (the doc-composition sampler treats all 4 symmetrically - see
# stage2_routed_stream.py) works out to ~2.46B tokens/domain. llm.c's GPT-2-124M reproduction
# (github.com/karpathy/llm.c) needed ~10B tokens dedicated to ONE domain to reach GPT-2's real
# reported performance - 4x closes that gap for every domain at once, not just nlp.
DATA_EQUIVALENT_STEPS = 600000

# routed9 (base/champion, 89M) and routed10 (large, 190M): warm-started continued-training
# ("cooldown") runs, not from-scratch. Both (a) source nlp from PG-19 books instead of
# OpenWebText/FineWeb - LAMBADA's passages are themselves book-derived, and the long-range
# antecedent->target copying it tests is a skill short web pages rarely exercise, unlike
# continuous book narrative; (b) upweight nlp's share of the mixture via force_domain in
# stage2_routed_stream.py - guaranteed presence + a longer snippet lands nlp at ~55% of
# tokens (see that module's docstring for the arithmetic), vs the ~25% every other arm gets;
# (c) warm-start from the existing champion/routed7 checkpoint rather than random init,
# reinitializing ONLY the nlp domain's embedding/type_embedding/projection/head (a new
# tokenizer vocabulary means those params' learned weights describe the WRONG content now,
# worse than random - see _warm_start_from_parent in train_stage2_pod.py), keeping the
# shared backbone and code/math/science tables exactly as trained.
# Values are checkpoint-file PREFIXES (not arm names) - routed10's parent (routed7) trains at
# large scale, so its checkpoints are named "large_routed7_step*.pt", not "routed7_step*.pt".
WARM_START_PARENT = {
    "routed9": "routed", "routed10": "large_routed7",
    "routed11": "routed8", "routed12": "routed8", "routed13": "routed8",
    "routed14": "large_routed7",  # NOT routed8 - wrong param count (89M) to warm-start a 190M model from
    "routed15": "routed8", "routed16": "routed8", "routed17": "routed8", "routed18": "routed8",
}
COOLDOWN_STEPS = 50000  # ~1/3 of a full run - cheap, and the backbone is already trained
COOLDOWN_BACKBONE_LR_SCALE = 0.1  # warm-started params (everything but nlp) train 10x slower
                                    # than the freshly-reinitialized nlp branch, so the cooldown
                                    # doesn't undo what the parent checkpoint already learned
BOOKS_NLP_UPWEIGHT_SNIPPET_WORDS = 600  # vs the other domains' default SNIPPET_WORDS=250

# arm="hybrid" only: fraction of training steps drawing a natural single-domain batch
# (PackedDomainStream, long continuous context - what MoT trains on exclusively) rather than
# a synthetic multi-domain switching batch (PackedRoutedStream). Addresses tax #2 in
# mot_hybrid_model.py's docstring: plain MoTRoutedModel never sees natural single-domain
# text, which is a plausible reason its raw content-modelling trails MoT even before
# accounting for the switch-prediction burden.
HYBRID_NATURAL_DATA_FRACTION = 0.6

# arm="routed3" only: hybrid moves routed TOWARD more single-domain data (60% natural,
# above). routed3 tests the opposite direction - LESS single-domain, MORE cross-domain
# density than routed/routed2 even see by default (MIN/MAX_DOMAINS_PER_DOC=2-4,
# SNIPPET_WORDS=250 in stage2_routed_stream.py). Every doc uses all 4 domains, and shorter
# snippets pack more switches into the same 1024-token window. Both use the same GradNorm
# switch-loss fix as routed2, so any difference between them isolates "does pushing further
# in the direction that's already winning (routed's cross-domain BPB/LAMBADA) help more than
# just fixing the loss".
ROUTED3_MIN_DOMAINS = 4
ROUTED3_MAX_DOMAINS = 4
ROUTED3_SNIPPET_WORDS = 100
DOMAIN_TAG = {
    "code": "<domain:code>",
    "math": "<domain:math>",
    "science": "<domain:science>",
    "nlp": "<domain:nlp>",
}
