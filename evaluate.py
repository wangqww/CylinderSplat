"""Evaluate a checkpoint on one row of the evaluation table (configs/eval_entries.py).

The row fixes the loader, split, scene keys and metrics; evaluate.py prints one
line per batch, one per scene and the Total line in the row's formats, and writes
metrics.json.

    CUDA_VISIBLE_DEVICES=1 python evaluate.py --dataset mp3d_double_256 \
        --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
        --ckpt <run>/checkpoint-48000 --run-id eval_all_256

The caller chooses the GPU; one process. The checkpoint is read in place and
never written. Everything is written under --out-dir (default
<runs root>/<run-id>): <timestamp>.log, the dumped config, metrics.json, with
--save-vis / --save-ply the PNG / PLY files in <out-dir>/<iteration>/, and the
models' own debug PNGs in <out-dir>/cwd. An --out-dir inside the checkpoint
directory, or one that contains it (the run directory of the checkpoint, or any
ancestor), is refused before anything is written.
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import argparse
import importlib
import importlib.util
import json
import logging
import os.path as osp
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from tools import switches as switch_registry
from tools.metrics import compute_psnr, compute_ssim, compute_ssim_gpu, compute_lpips, compute_pcc, WSPSNR
from tools.resume import name_allowed

REPO_ROOT = osp.dirname(osp.abspath(__file__))
# --run-id NAME writes to <runs root>/NAME (the same root as train.py).
RUNS_ROOT = os.environ.get("CYLINDERSPLAT_RUNS_ROOT", osp.join(REPO_ROOT, "workdirs"))
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Switches that only change the training loss or the run; sampling_align and prune_invisible change the forward and
# stay active here (a run is evaluated with the config it was trained with).
TRAIN_ONLY_SWITCHES = (
    "ddp_forward",
    "loc360_interleave",
    "depth_valid_mask",
    "lpips_input_range",
    "ws_loss",
    "volume_sparsity",
)


def load_eval_entries():
    # Loaded by path: `configs` is not a package and must not be shadowed by one on sys.path.
    path = osp.join(REPO_ROOT, "configs", "eval_entries.py")
    spec = importlib.util.spec_from_file_location("cylindersplat_eval_entries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EVAL_ENTRIES


EVAL_ENTRIES = load_eval_entries()
KNOWN_METRICS = (
    "psnr",
    "wspsnr",
    "ssim",
    "lpips",
    "pcc",
    "abs",
    "silog",
    "rmse",
    "delta1",
    "delta2",
    "delta3",
    "depthsim",
)
DEPTH_METRICS = ("abs", "silog", "rmse", "delta1", "delta2", "delta3", "depthsim")
_LABEL_RE = re.compile(r"(\w+): [%{]")


def check_entry(name, entry):
    """Every printed label must name the value printed under it."""
    for line_key, metrics_key, n_lead in (
        ("batch_line", "batch_metrics", 2),
        ("scene_line", "scene_metrics", 1),
        ("total_line", "total_metrics", 1),
    ):
        line, metrics = entry[line_key], entry[metrics_key]
        if line is None or metrics is None:
            if (line is None) != (metrics is None):
                raise ValueError(f"{name}: {line_key} and {metrics_key} must both be set or both be None")
            continue
        labels = tuple(_LABEL_RE.findall(line))
        n_fields = line.count("%") if line_key == "batch_line" else line.count("{")
        if labels != tuple(metrics) or n_fields != n_lead + len(metrics):
            raise ValueError(
                f"{name}: {line_key} prints {labels} in {n_fields} fields, "
                f"expected {tuple(metrics)} in {n_lead + len(metrics)}"
            )
        if not set(metrics) <= set(entry["batch_metrics"]):
            raise ValueError(f"{name}: {metrics_key} is not a subset of batch_metrics")
    if not set(entry["batch_metrics"]) <= set(KNOWN_METRICS):
        raise ValueError(f"{name}: unknown metric in batch_metrics")
    if not {"wspsnr", "ssim", "lpips", "pcc"} <= set(entry["batch_metrics"]):
        raise ValueError(f"{name}: wspsnr, ssim, lpips and pcc are computed for every row")
    if bool(set(DEPTH_METRICS) & set(entry["batch_metrics"])) != entry["depth_metrics"]:
        raise ValueError(f"{name}: depth_metrics flag disagrees with batch_metrics")
    targets = set(range(len(entry["target_views"])))
    context = set(entry["context_views"])
    if not context <= targets or set(entry["novel_views"]) != targets - context:
        raise ValueError(f"{name}: novel_views must be the non-context target positions")
    if (entry["scene_keys"] is None) != (entry["scene_metrics"] is None):
        raise ValueError(f"{name}: per-scene lines need scene keys")
    if not isinstance(entry["save_ply"], bool):
        raise ValueError(f"{name}: save_ply must be True or False")


# ----------------------------------------------------------------------------
# Metrics of one batch
# ----------------------------------------------------------------------------


def continuity(x, gt):
    # x, gt shape: [B, V, H, W]
    s = x[:, :, :, 0]  # left edge
    e = x[:, :, :, -1]  # right edge
    s_gt = gt[:, :, :, 0]
    e_gt = gt[:, :, :, -1]

    # difference of the edge differences
    diff = torch.abs((s - e) - (s_gt - e_gt)).mean(dim=-1)
    return diff


def depth_metrics(pred_depths_aligned, pred_depths, real_gt_depths, real_mask_gt):
    """AbsRel, SILog, RMSE and delta1..3 per view ([B, V]) under the GT mask.

    The default metrics pass the unaligned prediction as `pred_depths_aligned`
    (--align-depth passes the median-aligned one); SILog always uses `pred_depths`.
    """
    # abs_rel
    abs_diff = (real_gt_depths - pred_depths_aligned).abs()
    # relative error inside the mask, 0 elsewhere
    rel_err_map = (abs_diff / (real_gt_depths + 1e-8)) * real_mask_gt.float()
    error_sum = rel_err_map.sum(dim=(-2, -1))
    valid_count = real_mask_gt.float().sum(dim=(-2, -1))
    bv_abs = error_sum / (valid_count + 1e-8)

    # SILog
    log_diff = (
        torch.log(pred_depths.clamp(min=1e-8)) - torch.log(real_gt_depths.clamp(min=1e-8))
    ) * real_mask_gt.float()
    num_valid = real_mask_gt.float().sum(dim=(-2, -1)) + 1e-8
    term1 = (log_diff**2).sum(dim=(-2, -1)) / num_valid
    term2 = (log_diff.sum(dim=(-2, -1)) ** 2) / (num_valid**2)
    bv_silog = torch.sqrt((term1 - term2).abs()) * 100

    # rmse
    sq_diff = (real_gt_depths - pred_depths_aligned) ** 2
    masked_sq_diff = sq_diff * real_mask_gt.float()
    sum_sq_error = masked_sq_diff.sum(dim=(-2, -1))
    valid_count = real_mask_gt.float().sum(dim=(-2, -1))
    mse = sum_sq_error / (valid_count + 1e-8)
    bv_rmse = torch.sqrt(mse)

    # delta accuracies: thresh = max(gt / pred, pred / gt) inside the mask
    r1 = real_gt_depths / (pred_depths_aligned + 1e-8)
    r2 = pred_depths_aligned / (real_gt_depths + 1e-8)
    thresh = torch.max(r1, r2)
    valid_count = real_mask_gt.float().sum(dim=(-2, -1))
    valid_mask_bool = real_mask_gt.bool()
    delta1_correct = (thresh < 1.25) & valid_mask_bool
    delta1_count = delta1_correct.float().sum(dim=(-2, -1))
    bv_delta1 = delta1_count / (valid_count + 1e-8)
    delta2_correct = (thresh < 1.25**2) & valid_mask_bool
    delta2_count = delta2_correct.float().sum(dim=(-2, -1))
    bv_delta2 = delta2_count / (valid_count + 1e-8)
    delta3_correct = (thresh < 1.25**3) & valid_mask_bool
    delta3_count = delta3_correct.float().sum(dim=(-2, -1))
    bv_delta3 = delta3_count / (valid_count + 1e-8)
    return {
        "abs": bv_abs,
        "silog": bv_silog,
        "rmse": bv_rmse,
        "delta1": bv_delta1,
        "delta2": bv_delta2,
        "delta3": bv_delta3,
    }


def median_align_depths(pred_depths, real_gt_depths, real_mask_gt):
    """Per-view median scale alignment (--align-depth)."""
    pred_depths_aligned = pred_depths.clone()
    B, V, H, W = real_gt_depths.shape
    for b in range(B):
        for v in range(V):
            mask = real_mask_gt[b, v]  # [H, W]
            if mask.sum() < 1:
                continue
            valid_gt = real_gt_depths[b, v][mask]
            valid_pred = pred_depths[b, v][mask]
            gt_median = valid_gt.median()
            pred_median = valid_pred.median()
            if pred_median > 1e-8:
                scale = gt_median / pred_median
                pred_depths_aligned[b, v] = pred_depths[b, v] * scale
    return pred_depths_aligned


def compute_batch_metrics(entry, preds, gts, wspsnr_calculator, align_depth=False, fast_ssim=False):
    """Per-view metrics of one batch.

    Returns (bv, extra): {metric: [B, V] tensor} (pcc stays [B*V] for rows with
    pcc_per_view False), and the optional extra columns (--fast-ssim, --align-depth),
    which never feed bv.
    """
    bs = preds["img"].shape[0]
    pred_imgs = preds["img"]
    pred_depths = preds["depth"]
    gt_imgs = gts["img"]
    gt_depths = gts["depth"]
    bv = {}
    if entry["depth_metrics"]:
        real_gt_depths = gts["depth_gt"].squeeze(2)
        real_mask_gt = gts["mask_gt"].squeeze(2)
        # depthsim
        bv["depthsim"] = continuity(pred_depths, real_gt_depths).view(bs, -1)
    if "psnr" in entry["batch_metrics"]:
        bv["psnr"] = compute_psnr(
            rearrange(gt_imgs, "b v c h w -> (b v) c h w"), rearrange(pred_imgs, "b v c h w -> (b v) c h w")
        ).view(bs, -1)
    bv["wspsnr"] = wspsnr_calculator.ws_psnr(
        rearrange(gt_imgs, "b v c h w -> (b v) h w c"), rearrange(pred_imgs, "b v c h w -> (b v) h w c"), max_val=1.0
    ).view(bs, -1)
    bv["ssim"] = compute_ssim(
        rearrange(gt_imgs, "b v c h w -> (b v) c h w"), rearrange(pred_imgs, "b v c h w -> (b v) c h w")
    ).view(bs, -1)
    bv["lpips"] = compute_lpips(
        rearrange(gt_imgs, "b v c h w -> (b v) c h w"), rearrange(pred_imgs, "b v c h w -> (b v) c h w")
    ).view(bs, -1)
    bv_pcc = compute_pcc(
        rearrange(gt_depths, "b v c h w -> (b v c) h w"), rearrange(pred_depths, "b v h w -> (b v) h w")
    )
    bv["pcc"] = bv_pcc.view(bs, -1) if entry["pcc_per_view"] else bv_pcc
    if entry["depth_metrics"]:
        bv.update(depth_metrics(pred_depths, pred_depths, real_gt_depths, real_mask_gt))

    extra = {}
    if fast_ssim:
        extra["ssim_fast"] = compute_ssim_gpu(
            rearrange(gt_imgs, "b v c h w -> (b v) c h w"), rearrange(pred_imgs, "b v c h w -> (b v) c h w")
        ).view(bs, -1)
    if align_depth:
        aligned = median_align_depths(pred_depths, real_gt_depths, real_mask_gt)
        for name, value in depth_metrics(aligned, pred_depths, real_gt_depths, real_mask_gt).items():
            if name != "silog":  # scale invariant: equal to the unaligned value
                extra[name + "_aligned"] = value
    return bv, extra


def select_views(t, bs, views):
    """Columns `views` of a per-view metric ([B, V] or [B*V]) -> [B, len(views)]."""
    return t.reshape(bs, -1)[:, list(views)]


def _device_index(t):
    index = t.device.index
    return 0 if index is None else index


def _mean_records(res, names):
    # Per-scene aggregation: Python-float sums in dataset order / count.
    sums = {k: 0 for k in names}
    for m in res:
        for k in names:
            sums[k] = sums[k] + m[k].item()
    return {k: sums[k] / len(res) for k in names}


def _extra_line(prefix, values):
    return prefix + ", ".join("{}: {:.4f}".format(k, v) for k, v in values.items())


class EvalAggregator:
    """Batch totals and per-scene records of one evaluation.

    Per-scene records are collected for every sample whether or not images are
    saved. `views` (--novel-only) selects target views per sample before any
    averaging.
    """

    def __init__(self, entry, views=None, log=print):
        self.entry = entry
        self.views = None if views is None else tuple(views)
        self.log = log
        self.totals = {k: 0.0 for k in entry["batch_metrics"]}
        self.extra_totals = {}
        keys = entry["scene_keys"]
        self.scene_res = None if keys is None else {s: [] for s in keys}
        self.extra_scene_res = None if keys is None else {s: [] for s in keys}

    def add_batch(self, i_iter, bs, bv, extra=None, scenes=None, device_index=0):
        entry = self.entry
        extra = extra or {}
        if self.views is not None:
            bv = {k: select_views(t, bs, self.views) for k, t in bv.items()}
            extra = {k: select_views(t, bs, self.views) for k, t in extra.items()}
        means = {k: bv[k].mean() for k in entry["batch_metrics"]}
        for k in entry["batch_metrics"]:
            self.totals[k] += means[k]
        for k, t in extra.items():
            self.extra_totals[k] = self.extra_totals.get(k, 0.0) + t.mean()
        self.log(entry["batch_line"] % ((i_iter, device_index) + tuple(means[k] for k in entry["batch_metrics"])))
        if self.scene_res is None:
            return
        for b in range(bs):
            scene_name = scenes[b]
            if scene_name not in self.scene_res:
                raise KeyError(f"scene {scene_name!r} is not a scene key of this dataset: {sorted(self.scene_res)}")
            self.scene_res[scene_name].append({k: bv[k][b].mean() for k in entry["scene_metrics"]})
            if extra:
                self.extra_scene_res[scene_name].append({k: t[b].mean() for k, t in extra.items()})

    def log_scenes(self):
        """Log one line per scene key; return {scene: {num_samples, metrics, line[, extra]}}."""
        rows = {}
        if self.scene_res is None:
            return rows
        names = self.entry["scene_metrics"]
        for s in self.scene_res:
            res = self.scene_res[s]
            if len(res) == 0:
                self.log(" {} no samples".format(s))
                rows[s] = {"num_samples": 0}
                continue
            values = _mean_records(res, names)
            line = self.entry["scene_line"].format(s, *[values[k] for k in names])
            self.log(line)
            rows[s] = {"num_samples": len(res), "metrics": values, "line": line}
            if self.extra_scene_res[s]:
                extra = _mean_records(self.extra_scene_res[s], list(self.extra_scene_res[s][0]))
                self.log(_extra_line(" {} extra ".format(s), extra))
                rows[s]["extra"] = extra
        return rows

    def log_totals(self, gather, num_batches, time_s):
        """The Total line: batch means summed, gathered, divided by the batch count."""
        entry = self.entry
        totals = {k: gather(self.totals[k]).mean() for k in entry["batch_metrics"]}
        extra = {k: gather(v).mean() for k, v in self.extra_totals.items()}
        time_e = time.time()
        values = {k: totals[k].item() / num_batches for k in entry["batch_metrics"]}
        line = entry["total_line"].format(int(time_e - time_s), *[values[k] for k in entry["total_metrics"]])
        self.log(line)
        result = {"num_batches": num_batches, "metrics": values, "line": line}
        if extra:
            result["extra"] = {k: v.item() / num_batches for k, v in extra.items()}
            self.log(_extra_line("Extra totals: ", result["extra"]))
        return result


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def save_ply(gaussians, path, crop_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 12.0], compatible=True):
    # gaussians: [B, N, 14]
    # compatible: save pre-activated gaussians as in the original paper
    gaussians = torch.cat(
        [gaussians[:, 0:3], gaussians[:, 6:7], gaussians[:, 11:14], gaussians[:, 7:11], gaussians[:, 3:6]], dim=-1
    )

    from plyfile import PlyData, PlyElement

    means3D = gaussians[:, 0:3].contiguous().float()
    opacity = gaussians[:, 3:4].contiguous().float()
    scales = gaussians[:, 4:7].contiguous().float()
    rotations = gaussians[:, 7:11].contiguous().float()
    shs = gaussians[:, 11:].unsqueeze(1).contiguous().float()  # [N, 1, 3]

    if crop_range is not None:
        x_start, y_start, z_start, x_end, y_end, z_end = crop_range
        mask = (
            (means3D[:, 0] > x_start)
            & (means3D[:, 0] < x_end)
            & (means3D[:, 1] > y_start)
            & (means3D[:, 1] < y_end)
            & (means3D[:, 2] > z_start)
            & (means3D[:, 2] < z_end)
        )
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

    # prune by opacity
    mask = opacity.squeeze(-1) >= 0.005
    means3D = means3D[mask]
    opacity = opacity[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    shs = shs[mask]

    # invert activation to make it compatible with the original ply format
    if compatible:
        opacity = inverse_sigmoid(opacity)
        scales = torch.log(scales + 1e-8)
        shs = (shs - 0.5) / 0.28209479177387814

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ["x", "y", "z"]
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append("f_dc_{}".format(i))
    l.append("opacity")
    for i in range(scales.shape[1]):
        l.append("scale_{}".format(i))
    for i in range(rotations.shape[1]):
        l.append("rot_{}".format(i))

    dtype_full = [(attribute, "f4") for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")

    PlyData([el]).write(path)


def save_vis_png(path, pred_imgs, pred_depths, gt_imgs, gt_depths):
    """Visualisation of one sample: GT and rendered images, rendered and GT depth."""
    import imageio
    from tools.visualization import depths_to_colors

    v_pred_depths = pred_depths.clamp(0.0, 140.0)
    v_gt_depths = gt_depths.clamp(0.0, 140.0)
    cat_img_gt = rearrange(gt_imgs, "v c h w -> c h (v w)")
    cat_img_pred = rearrange(pred_imgs, "v c h w -> c h (v w)")
    grid_img = torch.cat([cat_img_gt, cat_img_pred], dim=1)
    grid_img = (grid_img.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    grid_depth = depths_to_colors(v_pred_depths)
    gt_depth = depths_to_colors(v_gt_depths.squeeze(1))
    grid_all = np.concatenate([grid_img, grid_depth, gt_depth], axis=0)
    imageio.imwrite(path, grid_all)


def save_outputs(entry, out_dir, i_iter, batch, preds, gts, save_vis, save_ply_files):
    bs = preds["img"].shape[0]
    if save_ply_files and preds["gaussian"].shape[0] != bs:
        # e.g. per-view pixel Gaussians laid out (b v) hw c: row b is not sample b.
        raise ValueError(
            f"preds['gaussian'] has {preds['gaussian'].shape[0]} rows for {bs} samples; "
            f"PLY files need one Gaussian set per sample ([B, N, 14])"
        )
    for b in range(bs):
        if entry["scene_keys"] is None:
            stem = entry["vis_name"].format(i_iter, b)
        else:
            stem = entry["vis_name"].format(i_iter, b, batch["scene"][b])
        if save_ply_files:
            save_ply(preds["gaussian"][b], osp.join(out_dir, stem + ".ply"), crop_range=None)
        if save_vis:
            save_vis_png(
                osp.join(out_dir, stem + ".png"), preds["img"][b], preds["depth"][b], gts["img"][b], gts["depth"][b]
            )


# ----------------------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------------------


def evaluate_loop(
    entry,
    my_model,
    val_dataloader,
    gather,
    log,
    views=None,
    save_vis=False,
    save_ply=False,
    vis_dir=None,
    align_depth=False,
    fast_ssim=False,
    num_batches=None,
):
    """Run forward_test over the loader and log the per-batch, per-scene and Total lines.

    `gather` is accelerator.gather_for_metrics; `num_batches` defaults to
    len(val_dataloader), the divisor of the Total line.
    """
    agg = EvalAggregator(entry, views=views, log=log)
    wspsnr_calculator = WSPSNR()
    time_s = time.time()
    with torch.no_grad():
        my_model.eval()
        for i_iter, batch in enumerate(val_dataloader):
            preds, gts = my_model.forward_test(batch)
            bs = preds["img"].shape[0]
            bv, extra = compute_batch_metrics(
                entry, preds, gts, wspsnr_calculator, align_depth=align_depth, fast_ssim=fast_ssim
            )
            batch_scenes = None if entry["scene_keys"] is None else [batch["scene"][b] for b in range(bs)]
            agg.add_batch(i_iter, bs, bv, extra, batch_scenes, device_index=_device_index(preds["img"]))
            if save_vis or save_ply:
                save_outputs(entry, vis_dir, i_iter, batch, preds, gts, save_vis, save_ply)

        torch.cuda.empty_cache()
        scene_rows = agg.log_scenes()
        total = agg.log_totals(gather, len(val_dataloader) if num_batches is None else num_batches, time_s)
        wall_time_s = time.time() - time_s

        timings = {}
        benchmarker = getattr(my_model, "benchmarker", None)
        if benchmarker is not None:
            for tag, times in benchmarker.execution_times.items():
                log(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
                timings[tag] = {"calls": len(times), "avg_s": float(np.mean(times))}
    return {"scenes": scene_rows, "total": total, "wall_time_s": wall_time_s, "benchmarker": timings}


# ----------------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------------

_CKPT_RE = re.compile(r"checkpoint-(\d+)$")


def resolve_checkpoint(ckpt):
    """--ckpt (checkpoint dir or model.safetensors) -> (dir, file, iteration or None)."""
    path = osp.abspath(osp.expanduser(ckpt))
    if osp.isdir(path):
        ckpt_dir, ckpt_file = path, osp.join(path, "model.safetensors")
    else:
        ckpt_dir, ckpt_file = osp.dirname(path), path
    if not osp.isfile(ckpt_file):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file} (--ckpt {ckpt})")
    match = _CKPT_RE.search(osp.basename(ckpt_dir))
    return ckpt_dir, ckpt_file, int(match.group(1)) if match else None


def _alias_key(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def compare_state_dicts(state_dict, model_dict):
    """Name/shape comparison of a checkpoint with the model under the name+shape filter.

    loaded: same name and shape; unused: checkpoint names the model lacks;
    shape: same name, other shape; missing: model tensors the checkpoint does not
    cover (they would keep their initial values), except names that alias a loaded
    tensor (shared storage, saved once).
    """
    loaded = [k for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape]
    unused = [k for k in state_dict if k not in model_dict]
    shape = [k for k, v in state_dict.items() if k in model_dict and v.shape != model_dict[k].shape]
    loaded_keys = {_alias_key(model_dict[k]) for k in loaded if model_dict[k].numel() > 0}
    missing, aliased = [], []
    for k, t in model_dict.items():
        if k not in state_dict:
            (aliased if t.numel() > 0 and _alias_key(t) in loaded_keys else missing).append(k)
    dtype = [k for k in loaded if state_dict[k].dtype != model_dict[k].dtype]
    return {"loaded": loaded, "unused": unused, "shape": shape, "missing": missing, "aliased": aliased, "dtype": dtype}


def allowed_extra_patterns(patterns):
    """The --allow-extra patterns without repeats.

    A pattern is an exact tensor name or 'prefix.*' (tools/resume.py name_allowed).
    ValueError for an empty pattern, a bare '*' (every name: that is --allow-partial)
    or a '*' anywhere but at the end.
    """
    merged = []
    for pattern in patterns:
        if pattern in ("", "*") or "*" in pattern[:-1]:
            raise ValueError(f"allowed-extra pattern {pattern!r}: give an exact tensor name or 'prefix.*'")
        if pattern not in merged:
            merged.append(pattern)
    return merged


def load_checkpoint(entry, my_model, accelerator, ckpt_dir, ckpt_file, strict, allowed_extra=()):
    """Load the weights after accelerator.prepare.

    Prints what matched; with `strict` (the default) exits if any checkpoint tensor
    is skipped or any model tensor is left uncovered. Checkpoint names that the model
    lacks and that match `allowed_extra` (--allow-extra) are skipped without failing
    (`unused_allowed`); other extra names, other shapes and missing names still exit.
    Zero loaded tensors always exit.
    """
    from safetensors.torch import load_file

    print(f"Loading checkpoint {ckpt_file} (read-only)")
    state_dict = load_file(ckpt_file, device="cpu")
    model_dict = my_model.state_dict()
    report = compare_state_dicts(state_dict, model_dict)
    report["unused_allowed"] = [k for k in report["unused"] if name_allowed(k, allowed_extra)]
    report["unused"] = [k for k in report["unused"] if not name_allowed(k, allowed_extra)]
    print(
        f"  loaded {len(report['loaded'])} tensors; skipped {len(report['unused'])} not in the model, "
        f"{len(report['shape'])} with another shape; {len(report['missing'])} model tensors not in the "
        f"checkpoint; {len(report['aliased'])} aliased; {len(report['dtype'])} cast to the model dtype"
    )
    if allowed_extra:
        per_pattern = ", ".join(
            f"{p} ({sum(name_allowed(k, (p,)) for k in report['unused_allowed'])})" for p in allowed_extra
        )
        print(f"  skipped {len(report['unused_allowed'])} more not in the model as allowed extra: {per_pattern}")
    for key in ("unused", "shape", "missing"):
        for name in report[key][:10]:
            print(f"    {key}: {name}")
        if len(report[key]) > 10:
            print(f"    {key}: ... {len(report[key]) - 10} more")
    if not report["loaded"]:
        sys.exit(f"error: no tensor of {ckpt_file} matches the model built from the config")
    if strict and (report["unused"] or report["shape"] or report["missing"]):
        sys.exit(
            "error: partial checkpoint load (see above); pass --allow-extra PATTERN for named checkpoint "
            "tensors the model does not build, or --allow-partial to evaluate anyway"
        )

    if entry["load"] == "accelerate_state":
        if report["shape"]:
            sys.exit("error: accelerator.load_state cannot skip tensors of another shape")
        # Also restores the RNG states saved with the checkpoint.
        accelerator.load_state(ckpt_dir, map_location="cpu", strict=False)
    else:
        # Every checkpoint tensor whose name and shape the model has.
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        my_model.load_state_dict(model_dict)
    return {k: len(v) for k, v in report.items()}


# ----------------------------------------------------------------------------
# Logging and summary
# ----------------------------------------------------------------------------


def get_model_summary(model: nn.Module) -> str:
    """Model summary in the PyTorch Lightning format."""
    total_params = 0
    trainable_params = 0
    train_mode_modules = 0
    eval_mode_modules = 0
    for module in model.modules():
        if module.training:
            train_mode_modules += 1
        else:
            eval_mode_modules += 1
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    non_trainable_params = total_params - trainable_params

    table_data = []
    for i, (name, module) in enumerate(model.named_children()):
        params = sum(p.numel() for p in module.parameters())
        mode = "train" if module.training else "eval"
        table_data.append((i, name, module.__class__.__name__, f"{params / 1e6:.1f} M" if params > 0 else "0", mode))

    col_widths = [3, 25, 25, 12, 8]
    separator = "-" * (sum(col_widths) + len(col_widths) - 1)
    summary_lines = []
    header = f"{' ':<{col_widths[0]}} | {'Name':<{col_widths[1]}} | {'Type':<{col_widths[2]}} | {'Params':>{col_widths[3]}} | {'Mode':<{col_widths[4]}}"
    summary_lines.append(header)
    summary_lines.append(separator)
    for row in table_data:
        line = f"{row[0]:<{col_widths[0]}} | {row[1]:<{col_widths[1]}} | {row[2]:<{col_widths[2]}} | {row[3]:>{col_widths[3]}} | {row[4]:<{col_widths[4]}}"
        summary_lines.append(line)
    summary_lines.append(separator)
    summary_lines.append(f"{trainable_params / 1e6:<.1f} M    Trainable params")
    summary_lines.append(f"{non_trainable_params / 1e6:<.1f} M    Non-trainable params")
    summary_lines.append(f"{total_params / 1e6:<.1f} M    Total params")
    memory_mb = total_params * 4 / 1e6  # fp32
    summary_lines.append(f"{memory_mb:<.3f}   Total estimated model params size (MB)")
    summary_lines.append(f"{train_mode_modules:<8}  Modules in train mode")
    summary_lines.append(f"{eval_mode_modules:<8}  Modules in eval mode")
    return "\n".join(summary_lines)


def create_logger(log_file=None, is_main_process=False, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def _is_within(path, root):
    path, root = osp.realpath(path), osp.realpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def default_run_dir(run_id):
    """<runs root>/<run_id>; the run id must be a plain directory name."""
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run id {run_id!r}: use letters, digits, '.', '_' or '-'")
    return osp.join(RUNS_ROOT, run_id)


def enter_scratch_cwd(scratch):
    """chdir into the run's scratch dir so relative writes by the models stay inside the run.

    The repository root goes on sys.path first, so imports keep working.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    os.chdir(scratch)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main(args):
    import warnings

    warnings.filterwarnings("ignore")
    entry = EVAL_ENTRIES[args.dataset]
    check_entry(args.dataset, entry)
    if args.align_depth and not entry["depth_metrics"]:
        sys.exit(f"error: --align-depth needs GT depth; dataset {args.dataset} has none")
    if args.save_ply and not entry["save_ply"]:
        sys.exit(
            f"error: --save-ply is not supported for {args.dataset}: its model returns per-view pixel "
            f"Gaussians, not one set per sample"
        )
    strict = args.strict_load and not args.allow_partial
    views = entry["novel_views"] if args.novel_only else None

    # Inputs (read-only) and output paths, resolved before any chdir.
    py_config = osp.abspath(osp.expanduser(args.py_config))
    if not osp.isfile(py_config):
        sys.exit(f"error: config not found: {py_config}")
    try:
        ckpt_dir, ckpt_file, global_iter = resolve_checkpoint(args.ckpt)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    if entry["load"] == "accelerate_state" and osp.basename(ckpt_file) != "model.safetensors":
        sys.exit(
            f"error: {args.dataset} loads through accelerator.load_state and needs a checkpoint directory "
            f"holding model.safetensors, got {ckpt_file}"
        )
    try:
        allowed_extra = allowed_extra_patterns(args.allow_extra)
    except ValueError as e:
        sys.exit(f"error: {e}")
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    try:
        # Absolute before the chdirs below (repo root for the model build, then the scratch cwd).
        out_dir = osp.abspath(
            osp.expanduser(args.out_dir or default_run_dir(args.run_id or f"eval_{args.dataset}_{timestamp}"))
        )
    except ValueError as e:
        sys.exit(f"error: {e}")
    # Not inside the checkpoint directory, and not its run directory or an ancestor of it (the dumped
    # config, metrics.json and <iteration>/ would land next to the source checkpoint).
    if _is_within(out_dir, ckpt_dir) or _is_within(ckpt_dir, out_dir):
        sys.exit(
            f"error: --out-dir {out_dir} is inside or contains the checkpoint directory {ckpt_dir}; "
            f"evaluation output never goes next to the source checkpoint"
        )
    vis_dir = osp.join(out_dir, str(global_iter) if global_iter is not None else "eval")
    log_file = osp.join(out_dir, f"{timestamp}.log")
    cfg_dump = osp.join(out_dir, osp.basename(py_config))
    metrics_path = osp.join(out_dir, "metrics.json")
    scratch = osp.join(out_dir, "cwd")
    os.makedirs(scratch, exist_ok=True)

    from mmengine import MMLogger
    from mmengine.config import Config
    from accelerate import Accelerator
    from accelerate.utils import set_seed, ProjectConfiguration

    # load config
    cfg = Config.fromfile(py_config)
    cfg.output_dir = out_dir
    cfg.eval_args = dict(save_vis=args.save_vis, save_ply=args.save_ply)
    try:
        switch_values = switch_registry.apply(cfg, args.switch)
    except (KeyError, ValueError) as e:
        sys.exit(f"error: {e}")
    MMLogger.get_instance("mmengine", log_level="WARNING")

    accelerator_project_config = ProjectConfiguration(project_dir=out_dir, logging_dir=None)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
    )
    if accelerator.num_processes != 1:
        sys.exit("error: evaluate.py runs on one process (per-scene results are not gathered across ranks)")

    # Seed before the model is built.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)

    dataset_config = cfg.dataset_params

    # configure logger
    cfg.dump(cfg_dump)
    logger = create_logger(log_file=log_file, is_main_process=accelerator.is_main_process)
    non_default = switch_registry.non_default(switch_values)
    logger.info(
        f"dataset {args.dataset}, config {py_config}, checkpoint {ckpt_file}, "
        f"views {'novel ' + str(list(views)) if views is not None else 'all targets'}, "
        f"switches {non_default or 'defaults'}"
    )
    for name in TRAIN_ONLY_SWITCHES:
        if name in non_default:
            logger.info(f"switch {name} only affects training; ignored here")

    # build model; the models read repo-relative files at construction
    # (model/losses.py: taming/modules/autoencoder/lpips/vgg.pth), so build from the repo root
    os.chdir(REPO_ROOT)
    from builder import builder as model_builder

    my_model = model_builder.build(cfg.model).to(accelerator.device)
    if entry["summary"] == "n_params":
        n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
        logger.info(f"Number of params: {n_parameters}")

    # generate datasets; process-wide OpenCV options, inherited by the loader workers
    import cv2

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    loader_module, loader_fn = entry["loader"]
    load_data = getattr(importlib.import_module(loader_module), loader_fn)
    val_dataloader = load_data(dataset_config[entry["batch_size_key"]], stage=entry["stage"])
    if len(val_dataloader) == 0:
        sys.exit(f"error: the {args.dataset} loader is empty (data roots missing?)")

    my_model, val_dataloader = accelerator.prepare(my_model, val_dataloader)

    load_counts = load_checkpoint(entry, my_model, accelerator, ckpt_dir, ckpt_file, strict, allowed_extra)
    print("work dir: ", out_dir)
    if entry["summary"] == "table":
        print(get_model_summary(my_model))

    # Relative writes by the models (debug PNGs) land in <out-dir>/cwd from here on.
    enter_scratch_cwd(scratch)
    if args.save_vis or args.save_ply:
        os.makedirs(vis_dir, exist_ok=True)

    # Evaluation
    results = evaluate_loop(
        entry,
        my_model,
        val_dataloader,
        accelerator.gather_for_metrics,
        logger.info,
        views=views,
        save_vis=args.save_vis,
        save_ply=args.save_ply,
        vis_dir=vis_dir,
        align_depth=args.align_depth,
        fast_ssim=args.fast_ssim,
    )

    record = {
        "dataset": args.dataset,
        "py_config": py_config,
        "checkpoint": {
            "path": ckpt_file,
            "iteration": global_iter,
            "load": load_counts,
            "strict": strict,
            "allowed_extra": allowed_extra,
        },
        "protocol": {
            "target_views": list(entry["target_views"]),
            "context_views": list(entry["context_views"]),
            "evaluated_views": list(views) if views is not None else list(entry["target_views"]),
            "novel_only": bool(args.novel_only),
        },
        "metric_labels": {"wspsnr": "WS-PSNR", "psnr": "PSNR (clipped)"},
        "switches": switch_values,
        "extras": {"fast_ssim": bool(args.fast_ssim), "align_depth": bool(args.align_depth)},
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "argv": sys.argv,
        **results,
    }
    with open(metrics_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"metrics written to {metrics_path}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a CylinderSplat checkpoint (see configs/eval_entries.py).")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(EVAL_ENTRIES),
        help="row of configs/eval_entries.py (loader, split, scene keys, metrics)",
    )
    parser.add_argument("--py-config", required=True, help="model config the checkpoint was trained with")
    parser.add_argument(
        "--ckpt",
        required=True,
        help="checkpoint directory (holding model.safetensors) or a model.safetensors file; read in place",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run name; output under <runs root>/<run-id> (runs root: $CYLINDERSPLAT_RUNS_ROOT, "
        "else workdirs/ in the repository; default run id eval_<dataset>_<time>)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: <runs root>/<run-id>); refused inside or containing the checkpoint directory",
    )
    parser.add_argument("--save-vis", action="store_true", help="write one PNG per sample (overrides cfg.eval_args)")
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="write one PLY per sample (overrides cfg.eval_args); refused for loc360_double_256",
    )
    parser.add_argument(
        "--novel-only",
        action="store_true",
        help="score only the target views that are not context views (per view, before averaging)",
    )
    parser.add_argument(
        "--align-depth",
        action="store_true",
        help="extra lines: depth metrics after per-view median scale alignment (MP3D rows)",
    )
    parser.add_argument(
        "--fast-ssim",
        action="store_true",
        help="extra column ssim_fast from a GPU SSIM (the reported ssim stays skimage)",
    )
    parser.add_argument(
        "--switch",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="behaviour switch (tools/switches.py), repeatable; overrides the config. sampling_align and "
        "prune_invisible change the forward: evaluate a run with the config it was trained with; the other switches "
        "only affect training",
    )
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exit if any checkpoint tensor is skipped or any model tensor is not covered (default on)",
    )
    parser.add_argument("--allow-partial", action="store_true", help="same as --no-strict-load")
    parser.add_argument(
        "--allow-extra",
        action="append",
        default=[],
        metavar="PATTERN",
        help="checkpoint tensors the model does not build that the strict load skips, repeatable: "
        "an exact name or 'prefix.*' (tools/resume.py name_allowed); other extra names, "
        "missing names and other shapes still exit; recorded in metrics.json",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
