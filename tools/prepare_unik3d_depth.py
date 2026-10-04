"""Generate the UniK3D depth prior (metric distance + confidence) used by the pixel branch.

Port of the author's generators (UniK3D fork, commit dc53883: model_mp3d.py /
depth_mp3d.py and model_360loc.py / depth_360loc.py). Inference settings are
unchanged: UniK3D ViT-L, resolution_level 9, Spherical camera
[500, 500, 320, 240, 1024, 512, pi, pi/2], normalize=True, RGB pixels whose
channel sum is below 16 are masked to 0, negative distances clamped to 0, and
(MP3D only) distances above 80 m set to 0.

Output layout (what the loaders read):
  MP3D     <scene>/<view>/depth_metric.npy, depth_conf.npy   next to rgb.png
  360Loc   <sequence>/depth_metric/<frame>_depth.npy, <frame>_conf.npy

Files are written under --out-root, mirroring the dataset tree, so the dataset
itself is never modified. Merge them into a copy of the dataset afterwards, or
pass --in-place to write next to the images of a dataset copy that is not a
protected tree.

Example:
  python tools/prepare_unik3d_depth.py --dataset mp3d --data-root /path/to/pano_grf \
      --out-root /path/to/pano_grf_depth --unik3d lpiccinelli/unik3d-vitl
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.write_guard import check_write_roots  # noqa: E402

MP3D_SETS = [("m3d", 0.1), ("m3d", 0.25), ("m3d", 0.5), ("m3d", 0.75), ("m3d", 1.0),
             ("residential", 0.15), ("replica", 0.5)]
LOC360_LOCATIONS = ["concourse", "hall", "piatrium", "atrium"]


def mp3d_images(data_root, stages):
    """rgb.png of every view, in the author's order (sorted scenes, sorted views)."""
    out = []
    for stage in stages:
        if stage == "test":
            roots = [data_root / f"png_render_test_1024x512_seq_len_3_{name}_dist_{dis}" for name, dis in MP3D_SETS]
        else:
            roots = [data_root / f"png_render_{stage}_1024x512_seq_len_3_m3d_dist_0.5"]
        for root in roots:
            if not root.exists():
                print(f"skip missing {root}")
                continue
            # The loaders skip .DS_Store entries; list exactly the views they read.
            for scene in sorted(f for f in os.listdir(root) if "DS_Store" not in f):
                for view in sorted(f for f in os.listdir(root / scene) if "DS_Store" not in f):
                    out.append(root / scene / view / "rgb.png")
    return out


def loc360_images(data_root):
    """Every panorama of every 360 sequence (mapping and query_360) of the four locations."""
    out = []
    for location in LOC360_LOCATIONS:
        seqs = [list((data_root / location / folder).glob("*360*/")) for folder in ("mapping", "query_360")]
        for seq in sum(seqs, []):
            with open(seq / "camera_pose.json") as f:
                frames = list(json.load(f).keys())
            out.extend(seq / "image" / frame for frame in frames)
    return out


def output_paths(dataset, image, data_root, out_root):
    rel = image.relative_to(data_root)
    base = out_root / rel
    if dataset == "mp3d":
        return base.with_name("depth_metric.npy"), base.with_name("depth_conf.npy")
    stem = image.stem
    depth_dir = base.parent.parent / "depth_metric"
    return depth_dir / f"{stem}_depth.npy", depth_dir / f"{stem}_conf.npy"


def load_rgb(dataset, path):
    image = Image.open(path)
    if dataset == "loc360":
        image = image.resize((1024, 512))
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)


def safe_save(path, array):
    """np.save without writing through links: the array goes to a new temp file in the target
    directory (O_EXCL | O_NOFOLLOW), and os.replace then swaps the directory entry. An existing
    output that is a hard link is replaced by a new inode, so its other names (e.g. in the
    original dataset) keep their contents; the directory is re-checked after mkdir."""
    directory = os.path.dirname(os.path.abspath(path))
    if os.path.realpath(directory) != directory:
        raise RuntimeError(f"output directory {directory} goes through a symlink")
    check_write_roots([directory])
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
        os.replace(tmp, path)
    except BaseException:
        if os.path.lexists(tmp):
            os.unlink(tmp)
        raise


def build_model(weights, device):
    from unik3d.models import UniK3D
    from unik3d.utils.camera import Spherical

    model = UniK3D.from_pretrained(weights)
    model.eval()
    model.to(device)
    model.resolution_level = 9
    H, W = 512, 1024
    params = torch.tensor([500, 500, 320, 240, W, H, 3.14159, 1.57079]).float()
    return model, Spherical(params=params)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mp3d", "loc360"], required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="dataset root (read only)")
    parser.add_argument("--out-root", type=Path, help="where the depth files are written (mirrors the dataset tree)")
    parser.add_argument("--in-place", action="store_true",
                        help="write next to the images; only for a dataset copy outside the protected trees")
    parser.add_argument("--stages", nargs="+", default=["train", "val", "test"], help="MP3D splits to process")
    parser.add_argument("--unik3d", default="lpiccinelli/unik3d-vitl", help="UniK3D weights (hub id or local dir)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if args.in_place == (args.out_root is not None):
        parser.error("pass exactly one of --out-root or --in-place")
    out_root = data_root if args.in_place else args.out_root.resolve()
    if not args.in_place and (out_root == data_root or data_root in out_root.parents):
        parser.error("--out-root must be outside --data-root (use --in-place on a dataset copy instead)")
    # The dataset trees the released models were trained on are protected: an
    # in-place run there is refused along with any other protected target.
    check_write_roots([out_root])

    images = mp3d_images(data_root, args.stages) if args.dataset == "mp3d" else loc360_images(data_root)
    print(f"{len(images)} panoramas")

    todo = [(img, *output_paths(args.dataset, img, data_root, out_root)) for img in images]
    # Check every output file, not just the root: a dataset "copy" made of symlinks would
    # otherwise write through to the original tree. realpath resolves symlinked directories
    # and files alike; symlinked outputs are refused outright.
    outputs = [p for _, d, c in todo for p in (d, c)]
    check_write_roots(outputs)
    linked = [p for p in outputs if os.path.realpath(p) != os.path.abspath(p)]
    if linked:
        parser.error(f"{len(linked)} output paths go through symlinks (e.g. {linked[0]}); "
                     "use a real copy of the dataset or --out-root")
    if args.skip_existing:
        todo = [t for t in todo if not (t[1].exists() and t[2].exists())]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, camera = build_model(args.unik3d, device)

    num_batches = math.ceil(len(todo) / args.batch_size)
    for i in range(num_batches):
        chunk = todo[i * args.batch_size:(i + 1) * args.batch_size]
        rgb = torch.stack([load_rgb(args.dataset, img) for img, _, _ in chunk], dim=0)
        with torch.no_grad():
            outputs = model.infer(rgb, camera=camera, normalize=True)
            depth = outputs["distance"]
            conf = outputs["confidence"]
            mask = (rgb.sum(axis=1) >= 16).unsqueeze(1).float()
            depth = torch.clamp(depth, min=0)
            depth = depth * mask.to(depth.device)
            if args.dataset == "mp3d":
                depth[depth > 80] = 0.0
        for b, (_, depth_path, conf_path) in enumerate(chunk):
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            safe_save(depth_path, depth[b].cpu().numpy())
            safe_save(conf_path, conf[b].cpu().numpy())
        if i % 10 == 9:
            print(f"{i + 1}/{num_batches} batches")
    print("done")


if __name__ == "__main__":
    main()
