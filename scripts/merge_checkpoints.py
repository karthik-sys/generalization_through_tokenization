"""Branch-train-merge weight averaging: M(alpha) = (1-alpha)*parent + alpha*branch.

Plain elementwise average over every tensor in the model state_dict - no optimizer state
(irrelevant post-training), no special-casing for norm gains/biases (they average fine, same
as every other tensor - there's nothing BatchNorm-style with running statistics in this
architecture that would need different treatment). Parent and branch MUST be the exact same
model class/config (verified by this script refusing to merge on any key or shape mismatch)
- that's what makes elementwise averaging meaningful at all.

Usage:
  python3 scripts/merge_checkpoints.py \
      --parent checkpoints/routed32_step300000.pt \
      --branch checkpoints/nlpbranch_step25000.pt \
      --alphas 0.25 0.5 0.75 \
      --out-prefix checkpoints/merged_a
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge_state_dicts(parent_sd: dict, branch_sd: dict, alpha: float) -> dict:
    if parent_sd.keys() != branch_sd.keys():
        only_parent = parent_sd.keys() - branch_sd.keys()
        only_branch = branch_sd.keys() - parent_sd.keys()
        raise ValueError(
            f"parent/branch key mismatch - not the same architecture, refusing to merge. "
            f"Only in parent: {sorted(only_parent)[:5]}{'...' if len(only_parent) > 5 else ''}  "
            f"Only in branch: {sorted(only_branch)[:5]}{'...' if len(only_branch) > 5 else ''}"
        )
    merged = {}
    for k, pv in parent_sd.items():
        bv = branch_sd[k]
        if pv.shape != bv.shape:
            raise ValueError(f"shape mismatch on '{k}': parent {tuple(pv.shape)} vs branch {tuple(bv.shape)}")
        merged[k] = ((1.0 - alpha) * pv.float() + alpha * bv.float()).to(pv.dtype)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, help="parent checkpoint .pt path (alpha=0 endpoint)")
    parser.add_argument("--branch", required=True, help="branch checkpoint .pt path (alpha=1 endpoint)")
    parser.add_argument("--alphas", type=float, nargs="+", required=True,
                         help="interior alpha values to merge at, e.g. 0.25 0.5 0.75")
    parser.add_argument("--out-prefix", required=True,
                         help="output path prefix - alpha 0.25 writes to '{prefix}25.pt', etc.")
    args = parser.parse_args()

    parent_ckpt = torch.load(args.parent, map_location="cpu")
    branch_ckpt = torch.load(args.branch, map_location="cpu")
    parent_sd, branch_sd = parent_ckpt["model"], branch_ckpt["model"]

    print(f"parent: {args.parent} (step {parent_ckpt.get('step', '?')}, {len(parent_sd)} tensors)")
    print(f"branch: {args.branch} (step {branch_ckpt.get('step', '?')}, {len(branch_sd)} tensors)")

    for alpha in args.alphas:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be strictly between 0 and 1 (0/1 are the parent/branch checkpoints "
                              f"themselves, no merge needed) - got {alpha}")
        merged_sd = merge_state_dicts(parent_sd, branch_sd, alpha)
        # Suffix encodes alpha as an integer percentage (25/50/75, not 0.25/0.5/0.75) to keep
        # filenames glob-friendly and consistent with this project's _step*.pt convention.
        out_path = Path(f"{args.out_prefix}{int(round(alpha * 100))}.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": merged_sd,
            "step": parent_ckpt.get("step"),
            "domain_vocab_sizes": parent_ckpt.get("domain_vocab_sizes"),
            "merge_alpha": alpha,
            "merge_parent": args.parent,
            "merge_branch": args.branch,
        }, out_path)
        print(f"  alpha={alpha:.2f} -> {out_path}")

    print("done")


if __name__ == "__main__":
    main()
