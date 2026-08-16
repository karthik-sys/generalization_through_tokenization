"""Intersection-vocabulary mining (spec §4, steps 2-3): read a trained MoT checkpoint,
project each domain's embeddings into the shared d_model space (same projection layers
used at inference), find cross-domain token pairs whose projected embeddings are close
in cosine similarity, and report them as candidates - e.g. "gradient" in math vs. code
landing near each other despite living in disjoint tables.

This is mining + reporting only (spec's step 2-3). Splicing found candidates into a
v1 shared table and retraining (step 4) is a separate follow-up, done only after these
candidates have been eyeballed for whether they're real polysemy or noise (spec's own
sequencing - "this step is also the natural filter for false positives").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.data.build_examples import TokenizerBundle
from src.model.mot_model import MoTModel

MODEL_CFG = dict(emb_dim=64, d_model=128, n_heads=4, ffn_dim=256, n_layers=2, max_seq_len=256)


def load_model(checkpoint_path: str) -> tuple[MoTModel, dict[str, int]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    domain_vocab_sizes = ckpt["domain_vocab_sizes"]
    model = MoTModel(domain_vocab_sizes=domain_vocab_sizes, **MODEL_CFG)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, domain_vocab_sizes


def projected_embeddings(model: MoTModel, domain: str) -> torch.Tensor:
    with torch.no_grad():
        emb = model.embeddings[domain].weight  # (vocab, emb_dim)
        return model.projections[domain](emb)  # (vocab, d_model)


def id_to_text(bundle: TokenizerBundle, domain: str, token_id: int) -> str:
    if domain == "nlp":
        offsets = bundle.nlp_offsets
        type_names = ["surface", "syllable", "morpheme", "byte"]
        # find which sub-vocab this id falls in
        for name in reversed(type_names):
            if token_id >= offsets[name]:
                local_id = token_id - offsets[name]
                if name == "surface":
                    return bundle.nlp.surface.id_to_piece(local_id)
                elif name == "syllable":
                    idx = local_id - 4  # NUM_SPECIAL offset used at encode time
                    return bundle.nlp.syllable_vocab[idx] if 0 <= idx < len(bundle.nlp.syllable_vocab) else f"<special:{local_id}>"
                elif name == "morpheme":
                    idx = local_id - 4
                    return bundle.nlp.morpheme_vocab[idx] if 0 <= idx < len(bundle.nlp.morpheme_vocab) else f"<special:{local_id}>"
                else:  # byte
                    idx = local_id - 4
                    return f"<byte:{idx}>" if idx >= 0 else f"<special:{local_id}>"
        return f"<unk:{token_id}>"
    return bundle.bpe[domain].sp.id_to_piece(token_id)


def mine_pair(
    model: MoTModel, bundle: TokenizerBundle, domain_a: str, domain_b: str,
    top_k: int, min_similarity: float,
) -> list[dict]:
    emb_a = projected_embeddings(model, domain_a)
    emb_b = projected_embeddings(model, domain_b)
    norm_a = torch.nn.functional.normalize(emb_a, dim=-1)
    norm_b = torch.nn.functional.normalize(emb_b, dim=-1)

    sims = norm_a @ norm_b.T  # (vocab_a, vocab_b)
    top_vals, top_idx = sims.topk(k=min(top_k, sims.shape[1]), dim=-1)

    candidates = []
    for i in range(sims.shape[0]):
        for j in range(top_vals.shape[1]):
            sim = top_vals[i, j].item()
            if sim < min_similarity:
                continue
            b_id = top_idx[i, j].item()
            text_a = id_to_text(bundle, domain_a, i)
            text_b = id_to_text(bundle, domain_b, b_id)
            if text_a.strip("▁_ ").lower() == text_b.strip("▁_ ").lower():
                continue  # surface-identical spelling isn't an interesting "shared meaning" find on its own
            candidates.append({
                "domain_a": domain_a, "token_a": text_a, "id_a": i,
                "domain_b": domain_b, "token_b": text_b, "id_b": b_id,
                "similarity": round(sim, 4),
            })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/mot_stage1.pt")
    parser.add_argument("--top-k", type=int, default=3, help="nearest neighbors per token to consider")
    parser.add_argument("--min-similarity", type=float, default=0.5)
    parser.add_argument("--out", default="checkpoints/intersection_vocab_candidates.json")
    args = parser.parse_args()

    model, domain_vocab_sizes = load_model(args.checkpoint)
    bundle = TokenizerBundle()
    domains = list(domain_vocab_sizes)

    all_candidates = []
    for i, domain_a in enumerate(domains):
        for domain_b in domains[i + 1:]:
            print(f"mining {domain_a} <-> {domain_b} ...")
            pairs = mine_pair(model, bundle, domain_a, domain_b, args.top_k, args.min_similarity)
            print(f"  {len(pairs)} candidates above similarity {args.min_similarity}")
            all_candidates.extend(pairs)

    all_candidates.sort(key=lambda c: -c["similarity"])
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(all_candidates, indent=2))
    print(f"\n{len(all_candidates)} total candidates written to {args.out}")
    print("\ntop 20 by similarity:")
    for c in all_candidates[:20]:
        print(f"  {c['similarity']:.3f}  {c['domain_a']}:{c['token_a']!r}  <->  {c['domain_b']}:{c['token_b']!r}")


if __name__ == "__main__":
    main()
