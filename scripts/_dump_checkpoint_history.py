"""Reconstruct full per-arm loss curves from checkpoint `history` fields on the volume.

Local logs have gaps (overwritten across relaunches), but every checkpoint stored the
step/loss history since its segment started. Loading a strided subset and merging the
histories (deduped by step) recovers the true continuous curve.
"""
import json
import modal

app = modal.App("mot-dump-history")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")
volume = modal.Volume.from_name("mot-stage2-data")
VOL = "/vol"


@app.function(image=image, volumes={VOL: volume}, timeout=1800)
def dump(arm: str, stride: int = 4000):
    import glob
    import re

    import torch

    volume.reload()
    paths = glob.glob(f"{VOL}/checkpoints/{arm}_step*.pt")
    step_of = lambda p: int(re.search(r"step(\d+)", p).group(1))
    paths = sorted(paths, key=step_of)
    if not paths:
        return []
    # strided subset for coverage, always including the newest
    chosen = [p for p in paths if step_of(p) % stride == 0]
    if paths[-1] not in chosen:
        chosen.append(paths[-1])
    if paths[0] not in chosen:
        chosen.insert(0, paths[0])

    merged = {}
    for p in chosen:
        try:
            ckpt = torch.load(p, map_location="cpu")
        except Exception as e:
            print(f"skip {p}: {e}", flush=True)
            continue
        for entry in ckpt.get("history", []):
            merged[int(entry["step"])] = float(entry["loss"])
    print(f"{arm}: {len(chosen)} checkpoints -> {len(merged)} unique steps "
          f"({min(merged) if merged else '-'}..{max(merged) if merged else '-'})", flush=True)
    return sorted(merged.items())


@app.local_entrypoint()
def main(arms: str = "mot,baseline,pooled", stride: int = 4000):
    out = {}
    for arm in arms.split(","):
        out[arm] = dump.remote(arm=arm.strip(), stride=stride)
    print("DUMP_JSON_START")
    print(json.dumps(out))
    print("DUMP_JSON_END")
