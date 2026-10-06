"""Rendered Gaussian count of the joint model over a split (read-only).

Per sample (batch 1): the number of Gaussians whose float32 opacity is >= 1/255 in forward_test's output - what
prune_invisible passes to the rasteriser - out of the unpruned total (1,008,074 at 256x512). Writes the mean / min /
max count and the mean share of the total to --out; for the joint model also the mean rendered pixel / volume split
(report-only); scripts/long_arm.sh writes it as count_<step>.json.

  CUDA_VISIBLE_DEVICES=<gpu> python tools/count_rendered_gaussians.py --py-config <config> --ckpt <checkpoint dir> \
      --out <count.json> [--split val|test] [--max-batches N]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import diagnose_gaussians as D  # noqa: E402

THRESHOLD = torch.tensor(1.0 / 255.0, dtype=torch.float32)  # the rasteriser's alpha cut (prune_invisible)


def rendered(gaussians):
    """[N, 14] Gaussians (opacity in channel 6) -> number the rasteriser would blend."""
    return int((gaussians[:, 6].float() >= THRESHOLD.to(gaussians.device)).sum())


def count(model, batches, max_batches=None, extras=None):
    """Per-sample rendered counts over batches of size 1, from forward_test's preds["gaussian"] (the tensor the
    renderer prunes). -> (counts, total). With a dict `extras` and a joint model (pixel_gs / volume_gs), appends per
    sample to extras["pixel"] / extras["volume"] the rendered split (the first n_pixel Gaussians are the pixel ones)."""
    hooks, seen = [], {}
    if extras is not None and hasattr(model, "pixel_gs") and hasattr(model, "volume_gs"):
        hooks.append(model.pixel_gs.register_forward_hook(
            lambda m, a, out: seen.__setitem__("n_pixel", out["gaussians"].shape[1])))
    try:
        return _count(model, batches, max_batches, extras, seen)
    finally:
        for h in hooks:
            h.remove()


def _count(model, batches, max_batches, extras, seen):
    counts, total = [], None
    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        with torch.no_grad():
            preds, _ = model.forward_test(batch)
        gaussians = preds["gaussian"]
        if gaussians.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {gaussians.shape[0]}")
        total = gaussians.shape[1]
        counts.append(rendered(gaussians[0]))
        if "n_pixel" in seen:
            n_pixel = seen["n_pixel"]
            pixel = rendered(gaussians[0, :n_pixel])
            extras.setdefault("pixel", []).append(pixel)
            extras.setdefault("volume", []).append(counts[-1] - pixel)
    return counts, total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--py-config", required=True)
    parser.add_argument("--ckpt", required=True, help="checkpoint directory or model.safetensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=sorted(D.SPLITS), default="val")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)
    _, model = D.build(args.py_config, args.ckpt)
    extras = {}
    counts, total = count(model, (D.to_cuda(b) for b in D.loader(args.split, 1)), args.max_batches, extras)
    if not counts:
        sys.exit("error: no batch was read")
    result = dict(py_config=args.py_config, ckpt=args.ckpt, split=args.split, samples=len(counts), total=total,
                  mean=sum(counts) / len(counts), min=min(counts), max=max(counts),
                  mean_share=sum(counts) / len(counts) / total)
    for key, values in extras.items():  # report-only
        result[f"mean_{key}"] = sum(values) / len(values)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
