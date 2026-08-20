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

# routed29: reallocates params rather than adding them (MoTRoutedTiedModel - see that file's
# docstring). At MODEL_CFG's dims, the 4 output heads cost ~55.5M/89.1M (62%) with zero
# sharing against the embeddings they mirror. Tying each head to its own domain's embedding
# (GPT-2's trick, bridged via a small per-domain Linear since our emb_dim != d_model) frees
# ~48-50M, reinvested here as backbone depth: 23 layers at the SAME d_model=512/ffn=2048 as
# MODEL_CFG (vs MODEL_CFG's 6 layers, and deeper than even LARGE_MODEL_CFG's 12) - lands at
# 88.06M total, within 1.2% of MODEL_CFG's 89.1M, with backbone now 82.9% of the model
# instead of 21.8%. From scratch only - no warm-start parent exists for this architecture
# (head/embedding shapes are structurally incompatible with every existing checkpoint).
TIED_MODEL_CFG = dict(emb_dim=128, d_model=512, n_heads=8, ffn_dim=2048, n_layers=23, max_seq_len=1024)

# routed30 (B), routed31 (C), routed32 (D): three more reallocations of the same 89M-ish
# budget, run alongside routed29, each testing a different lever rather than isolating one:
#   routed30 (B): vocab-shrink + direct tying. code/math/science are starved under the 70%
#     nlp diet (~30% of tokens split 3 ways) yet still carry DOMAIN_VOCAB_SIZES' full 24k
#     vocab each - retrained at 10k (tokenizers_stage2_shrunk, already built), nlp untouched.
#     emb_dim raised to d_model (512) so tying is DIRECT (no bridge needed, unlike routed29
#     which deliberately keeps emb_dim small) - simpler but leaves less budget for backbone
#     depth than routed29's approach, on purpose: this tests the alternative bet (wider,
#     cleaner tying + fixed starved vocabs) against routed29's (narrow embeddings, max depth).
#   routed31 (C): routed29's allocation (narrow tied embeddings, max depth) + the aggressive
#     modern-technique stack - RoPE, RMSNorm, SwiGLU FFN, QK-norm (see backbone_modern.py).
#     The exploratory bet: do techniques proven elsewhere help THIS specific setup.
#   routed32 (D): routed29's allocation + only RoPE + RMSNorm (no SwiGLU/QK-norm). The "safe
#     improver" - the two changes closest to risk-free, meant to actually beat the flagship
#     rather than test a hypothesis.
ROUTED30_MODEL_CFG = dict(emb_dim=512, d_model=512, n_heads=8, ffn_dim=2048, n_layers=16, max_seq_len=1024)
# (16 layers lands at 87.59M with code/math/science@10k+nlp@~36k, emb_dim=d_model=512 -
# verified by direct construction, not the ~12-14 first guessed before checking real numbers)
ROUTED30_SHRUNK_VOCAB = 10000  # code/math/science only - nlp stays at its normal size
ROUTED_MODERN_MODEL_CFG = dict(emb_dim=128, d_model=512, n_heads=8, ffn_dim=2048, n_layers=24, max_seq_len=1024)

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

# routed34/36 (and any future FWE/edu-data arm): weighted multi-source blends for the
# nlp/math/science domains, per the GPT-2-budget data spec. Each entry is (source_dict,
# weight) using the SAME {"path", "name", "gated", ...} shape as STREAM_SOURCES itself, so a
# blend list can be consumed by the same loading path with an extra weighted-choice step.
# All three domains keep code unchanged (codeparrot, above) - only nlp/math/science are
# reblended. Meant to be pre-tokenized to shards (scripts/build_token_shards.py), not streamed
# live - see that script's own docstring for why (nlpbranch's pipeline-bound throughput was
# the live warning sign).
FWE_NLP_SOURCES = [
    ({"path": "HuggingFaceFW/fineweb-edu", "name": "sample-100BT", "gated": False}, 0.60),
    ({"path": "Skylion007/openwebtext", "name": None, "gated": False}, 0.40),
]
EDU_MATH_SOURCES = [
    ({"path": "open-web-math/open-web-math", "name": None, "gated": False}, 0.45),
    ({"path": "HuggingFaceTB/cosmopedia", "name": "auto_math_text", "gated": False}, 0.35),
    ({"path": "HuggingFaceTB/cosmopedia", "name": "khanacademy", "gated": False}, 0.20),
]
EDU_SCIENCE_SOURCES = [
    ({"path": "HuggingFaceTB/cosmopedia", "name": "openstax", "gated": False}, 0.40),
    ({"path": "HuggingFaceTB/cosmopedia", "name": "stanford", "gated": False}, 0.25),
    # fineweb-edu filtered through src/data/domain_classifier.py -> science. Weight moves to
    # openstax (0.40 -> 0.60) if the classifier pass proves too slow to run once over the
    # target token budget - a launch-time fallback, not a design uncertainty.
    ({"path": "HuggingFaceFW/fineweb-edu", "name": "sample-100BT", "gated": False,
      "domain_filter": "science"}, 0.20),
    ({"path": "gfissore/arxiv-abstracts-2021", "name": None, "gated": False}, 0.15),
]

# Long-doc fraction (routed34/35/36/37, per routed19's own law: constant from step 1, no
# staging): this share of synthetic docs are pure single-domain at LONG_DOC_SNIPPET_WORDS
# instead of the normal spliced multi-domain composition. Decided at launch, not tuned mid-run.
LONG_DOC_FRACTION = 0.20
LONG_DOC_SNIPPET_WORDS = 2500
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
    "routed19": "Round-3: from-scratch, curriculum-ordered, corrected GPT-2-parity token budget. "
               "Fixes a verified 16x token-budget shortfall (routed8 delivered ~2.46B tokens against "
               "a ~9.83B/39.3B doc-intended target - a units mismatch between DATA_EQUIVALENT_STEPS' "
               "own comment and the training loop's actual step-counting). Phase 1 "
               "(ROUTED19_PHASE1_FRACTION of steps): single-domain-only data (PackedRoutedStream, "
               "min/max_domains=1, long snippets) - genuine coherent runs, no mid-sequence switching "
               "at all. Phase 2 (remainder): the standard splice-from-step-0 recipe every prior arm "
               "used for its entire run. Same architecture as routed8 (plain MoTRoutedModel, OWT nlp "
               "source, base scale) - the only variables are curriculum ordering and a corrected "
               "~9.83B-token budget, isolated from any architecture change.",
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
    "routed20": ("copy_q.", "copy_k.", "copy_gate."),  # same LR treatment as routed11 - see ALIGN_ARMS
    "routed21": ("copy_q.", "copy_k.", "copy_gate."),  # routed20's no-alignment control
    "routed24": ("copy_q.", "copy_k.", "copy_gate."),  # routed11's exact recipe, rerun fresh - see ALIGN_ARMS
    "routed25": ("copy_q.", "copy_k.", "copy_gate."),
    "routed26": ("copy_q.", "copy_k.", "copy_gate."),
    "routed27": ("copy_q.", "copy_k.", "copy_gate."),
    "routed28": ("copy_q.", "copy_k.", "copy_gate."),
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
    "routed20": "routed17", "routed21": "routed17",  # NOT routed8 - parent is the ALREADY diet-
    # adapted routed17@100000, not the pre-diet routed8@600000, so copy-gate is the only genuinely
    # new thing being learned during continuation (mixture stays 70% nlp throughout, it doesn't
    # have to be picked up from scratch at the same time as the new module). routed22/23
    # deliberately absent - from scratch, no parent at all.
    "routed24": "routed8",  # routed11's EXACT parent - see PER_ARM_BACKBONE_LR_SCALE below
    "routed25": "routed17", "routed26": "routed17", "routed27": "routed17",  # routed21's parent -
    # gate+diet stacking is the confirmed winner, all three continue from there.
    "routed28": "large_routed14",  # NOT large_routed7 - large_routed7 has neither diet nor
    # gate, so warm-starting from it would force the model to learn BOTH simultaneously at
    # 190M, exactly the "two new things at once" pattern routed21 avoided (and that routed21's
    # win over routed11/17 argues against). large_routed14 is already the 190M gate-trained
    # analog of routed11 (EM 9.49%/ppl 10.58) - routed28 only has to learn diet on top of an
    # already-gate-adapted parent, mirroring routed21's exact order-of-operations logic with
    # the two axes swapped (diet added second here, gate added second there).
}
COOLDOWN_STEPS = 50000  # ~1/3 of a full run - cheap, and the backbone is already trained
COOLDOWN_BACKBONE_LR_SCALE = 0.1  # warm-started params (everything but nlp) train 10x slower
                                    # than the freshly-reinitialized nlp branch, so the cooldown
                                    # doesn't undo what the parent checkpoint already learned

# routed24: the "amp it up" version of routed11, not a plain rerun. Real evidence from
# tonight's re-scoring: routed16 (FROZEN backbone, 0x plasticity) lost to routed11
# (throttled at BET_BACKBONE_LR_SCALE=0.3, i.e. NOT frozen) on every metric - so the trend
# so far is "more backbone plasticity helped, not hurt." routed24 tests the natural next
# point on that same axis: full, UNTHROTTLED backbone LR (1.0, no throttle at all) instead
# of 0.3x, same parent (routed8) and same unbiased gate init as routed11 - isolating
# backbone-plasticity as the one lever being pushed further, not conflated with diet or
# alignment (those are routed20/21/23's job). Launch longer than routed11's original
# BET_STEPS too (150k vs 100k) since routed11 showed no sign of saturating.
PER_ARM_BACKBONE_LR_SCALE = {
    "routed24": 1.0,
    "routed25": 1.0, "routed26": 1.0, "routed27": 1.0, "routed28": 1.0,  # all four
    # combine the two INDEPENDENTLY confirmed winners (diet+gate stacking from routed20/21,
    # full plasticity from routed24) - see the routed25-28 block below.
}  # overrides BET_BACKBONE_LR_SCALE for these arms only

# routed25/26/27/28: the second batch, launched the same night after routed20/21's result
# (gate+diet stacked on routed17, EM 14.75-14.79%) crushed every prior number, and routed24
# separately confirmed full backbone plasticity is a real (if smaller) additional lever.
# This batch combines both, plus two clean follow-up questions raised by the first batch's
# results and the review of them:
#
#   routed25: FLAGSHIP - routed21's exact recipe (gate + diet @ DIET_PHASE2_NLP_SNIPPET_WORDS,
#             warm-started from routed17@100000) + full backbone plasticity (PER_ARM_
#             BACKBONE_LR_SCALE=1.0 instead of routed21's 0.3x). Run to 300k steps (vs
#             routed21's 100k) since nothing in the first batch showed saturation.
#   routed26: identical to routed25 except nlp diet share pushed higher via
#             ROUTED26_NLP_SNIPPET_WORDS (~83% nlp vs routed25's ~70%) - is 70% actually
#             the ceiling, or does more help further now that copy-gate is stacked in?
#   routed27: identical to routed25 except nlp sourced from PG-19 books instead of
#             OpenWebText (routed9/10's source-match lever - LAMBADA passages are
#             book-derived, and the long-range antecedent->target copying it tests is a
#             skill short web pages rarely exercise). Tests whether source-matching stacks
#             with gate+diet, not a robustness/seed check - a genuinely different hypothesis,
#             reusing already-proven code (_apply_books_nlp_source).
#   routed28: the SAME recipe at 190M scale, warm-started from large_routed14 (NOT
#             large_routed7 - see WARM_START_PARENT's comment for why: large_routed14 is
#             already the 190M gate-trained analog of routed11, so routed28 only has to learn
#             diet as the one new thing, mirroring routed21's order-of-operations rather than
#             confounding diet+gate simultaneously the way a large_routed7 parent would have).
ROUTED26_NLP_SNIPPET_WORDS = 2400  # vs DIET_PHASE2_NLP_SNIPPET_WORDS=1200 -> ~82.8% nlp share
                                     # (2400 / (2400 + 2*250), same arithmetic as routed17's own)
ROUTED25_STEPS = 300000  # routed25/26/27's shared step budget (routed28 stays at BET_STEPS,
                           # cheaper to keep the large-scale run shorter for a first look)

# routed20/21/22/23: the four-way ablation launched the night evaluate_lambada's copy-gate
# measurement bug was found and fixed (see mot_routed_copygate_model.py). Two real, newly-
# CONFIRMED findings drive this set: (1) routed11's copy-gate mechanism - unbiased gate init,
# backbone throttled (not frozen) at BET_BACKBONE_LR_SCALE, warm-started from routed8 - is the
# best LAMBADA result in the project once actually measured (EM 9.86%/ppl 10.18, beating
# routed17's diet recipe AND routed16's "conservative init" follow-up that was motivated by
# routed11 looking unpromising under the SAME broken measurement); (2) routed19's from-scratch
# curriculum staging demonstrated real, sustained interference between single-domain
# competence and cross-domain switching when the shared backbone is retrained through a
# distribution shift - motivating a purely additive fix instead (domain_embedding_alignment_
# loss on MoTRoutedModel, CORAL-style mean+covariance matching across domain embedding
# TABLES, not activations - cheap enough to add every ALIGN_LOSS_EVERY steps rather than every
# step). Priority order for this set, per explicit instruction: best LAMBADA EM first, then
# single-domain BPB, then cross-domain BPB.
#
#   routed20: copy-gate (unbiased init) + diet (DIET_PHASE2_NLP_SNIPPET_WORDS, still 70% nlp
#             through continuation) + alignment loss, warm-started from routed17@100000 (NOT
#             routed8 - routed17 already did the diet adaptation, so copy-gate is the only
#             genuinely new thing this continuation has to learn - copy-gate and diet have
#             never been stacked before this). routed11's exact LR treatment (NEW_MODULE_MATCH,
#             BET_BACKBONE_LR_SCALE=0.3, not COOLDOWN's gentler 0.1 - v1 beat v2's more-frozen
#             treatment, so plasticity is plausibly part of why it works; don't over-protect).
#             Flagship candidate - every proven-or-plausible EM/cross-domain lever stacked.
#             Rough target given routed11 alone already reached EM 9.86%/ppl 10.18: this
#             should beat both, since it adds a second independently-real EM lever on top.
#   routed21: identical to routed20 MINUS the alignment loss - copy-gate-on-diet in isolation,
#             the single most important control, isolating what alignment actually contributes
#             on top of the (expected-strong-on-its-own) copy-gate+diet combination.
#   routed22: copy-gate (unbiased init) ALONE, from scratch (no warm start, no diet, no
#             alignment), uniform LR. Tests the copy-gate docstring's own claim that "the
#             mechanism is useless until the backbone already produces good representations
#             to attend with" - a real, previously untested hypothesis now that we know the
#             warm-started version works this well.
#   routed23: alignment loss ALONE, from scratch, plain MoTRoutedModel (no copy-gate, no
#             diet), uniform LR. Pure isolation of the alignment mechanism's own contribution
#             to cross-domain BPB/switch accuracy, uncontaminated by copy-gate or diet -
#             the direct test of the Priority-3 (cross-domain) architectural fix on its own.
ALIGN_ARMS = ("routed20", "routed23")
ALIGN_LOSS_WEIGHT = 0.05  # middle of the recommended 0.01-0.1 range
ALIGN_LOSS_EVERY = 50  # operates on embedding TABLES (fixed size), not batch activations -
                        # tables move slowly relative to one optimizer step, safe to skip most
FROM_SCRATCH_COPYGATE_ARMS = ("routed22",)  # routed22 only - routed20/21 are copygate+warm-started
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
# routed19 only: real fix for the DATA_EQUIVALENT_STEPS token-budget bug found this session -
# that constant's own comment defines "step" as one OPTIMIZER update (BATCH_SIZE*GRAD_ACCUM_
# STEPS*max_seq_len = 65,536 tokens), but train_stage2_pod.py's loop counts "step" as one
# MICRO-batch (BATCH_SIZE*max_seq_len = 4,096 tokens, GRAD_ACCUM_STEPS=16 micro-steps per real
# update). routed8 was launched with --steps 600000 under the doc's intended meaning (600,000
# real optimizer updates -> ~39.3B tokens) but the code delivered 600,000 micro-batches only
# (~2.46B tokens) - a verified 16x shortfall (2.46B x 16 = 39.3B). ROUTED19_TARGET_MICROSTEPS
# is DATA_EQUIVALENT_STEPS's ORIGINAL 150,000-optimizer-update baseline (~9.83B tokens, the
# doc's core GPT-2-parity target, not its extra 4x stretch tier) expressed correctly in the
# code's actual micro-step units - using ROUTED19_BATCH_SIZE/ROUTED19_GRAD_ACCUM_STEPS below
# (32/2), NOT the global BATCH_SIZE/GRAD_ACCUM_STEPS (4/16) other arms use: target_tokens
# (150,000 * 65,536 = 9.83B) / (ROUTED19_BATCH_SIZE(32) * max_seq_len(1024)) = 300,000.
# Getting this denominator wrong (i.e. using the global BATCH_SIZE instead) would silently
# produce an 8x-too-large step count - caught by hand-checking real_opt_updates == 150,000
# before ever launching a real run, not by the code itself, so verify this arithmetic again
# if ROUTED19_BATCH_SIZE ever changes.
ROUTED19_TARGET_MICROSTEPS = 300_000

# routed19 only: curriculum split (user-specified 60/40) - the first fraction of training
# uses PackedRoutedStream with min_domains=max_domains=1 (see ROUTED19_PHASE1_SNIPPET_WORDS
# below), giving the model genuine long single-domain runs before it ever has to handle
# mid-sequence switching, instead of splicing 2-4 domains together from step 1 the way every
# prior arm (routed8 included) has. The remainder uses the standard default PackedRoutedStream
# args - the exact same recipe routed8 always used, now applied as phase 2 rather than the
# entire run.
ROUTED19_PHASE1_FRACTION = 0.6

# routed19 phase 1 only: single-domain snippet length, vs the default SNIPPET_WORDS=250 used
# everywhere else. PackedRoutedStream packs a continuous token buffer across doc boundaries
# regardless of domain, so a long snippet doesn't GUARANTEE zero domain-mixing within a single
# 1024-token window - it just makes cross-domain window boundaries much rarer (roughly
# proportional to snippet_words/250, i.e. ~12x rarer at this setting), capped by whichever is
# shorter: this setting, or a given real source document's own natural length. An honest,
# real reduction in switch density, not a hard single-domain-per-window guarantee.
ROUTED19_PHASE1_SNIPPET_WORDS = 3000

# routed19 only: phase1_loader and phase2_loader are two INDEPENDENTLY-constructed
# PackedRoutedStream instances, each spawning their own DataLoader workers - but PyTorch's
# worker_id resets to 0..NUM_WORKERS-1 PER DataLoader instance, so phase1's worker 0 and
# phase2's worker 0 both compute the same shard_id and, without this offset, would both start
# reading the local cache's shuffled permutation from pass_num=0 - the SAME sequence, from the
# SAME starting point. Confirmed live on the actual routed19 run: phase 2 would have
# substantially re-read material phase 1 already consumed instead of getting fresh data.
# Any distinct, sufficiently-large offset works - reseeds the permutation into unrelated
# territory. 10,000 is arbitrary but far larger than any realistic pass_num phase 1 could reach
# on its own (each pass reshuffles the WHOLE cache, so even a few hundred passes is a lot).
ROUTED19_PHASE2_CACHE_PASS_OFFSET = 10_000

# routed19 only: BATCH_SIZE=4 (global) was deliberately kept small to bound sota's 100k-vocab
# logits tensor (batch*seq*vocab*4 bytes - see the comment above BATCH_SIZE). routed19 is a
# domain-routed arm with per-domain vocabs in the ~30-40k range (much smaller than sota's
# shared 100k), so it doesn't need that same headroom - scoped here rather than raising the
# global BATCH_SIZE, so sota/baseline's memory margin is untouched. GRAD_ACCUM_STEPS lowered
# proportionally to hold the same effective batch (64) and LR/schedule math routed8 used -
# this only changes wall-clock/cost, never what's mathematically being trained. Verify actual
# peak GPU memory via a real calibrate() run before trusting this blindly (done - see launch
# notes); reduce back toward BATCH_SIZE if it doesn't fit.
ROUTED19_BATCH_SIZE = 32
ROUTED19_GRAD_ACCUM_STEPS = 2  # 32 * 2 = 64, same effective batch as everywhere else

# nlpbranch + the final-four batch (routed34-37): same reasoning/precedent as routed19's own
# batch/accum override above - domain-routed arms with per-domain vocabs (~30-40k), not sota's
# 100k shared vocab, so they don't need BATCH_SIZE kept small for logits-tensor headroom.
# Reused (not duplicated) across every arm in this batch since they're all domain-routed at
# base or large scale. Effective batch stays 64, so LR schedule/token-per-step math is
# unchanged - this only changes wall-clock/cost, never what's mathematically being trained.
FAST_BATCH_SIZE = 32
FAST_GRAD_ACCUM_STEPS = 2  # 32 * 2 = 64

ROUTED3_MIN_DOMAINS = 4
ROUTED3_MAX_DOMAINS = 4
ROUTED3_SNIPPET_WORDS = 100
DOMAIN_TAG = {
    "code": "<domain:code>",
    "math": "<domain:math>",
    "science": "<domain:science>",
    "nlp": "<domain:nlp>",
    "generalist": "<domain:generalist>",
}
