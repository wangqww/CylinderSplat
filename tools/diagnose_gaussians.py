"""Where the joint model's Gaussians are redundant and where its error is (read-only diagnostics).

Runs the released joint model (OmniGaussianCylinderAll, e.g. mp3d_stage3_joint_256x512) over the MP3D
validation split (default) or a test split and writes <out-dir>/diagnose_<split>.json plus a short summary:

  a1  opacity histograms per branch: pixel scale x view, volume cylinder x gpv sibling; pixel scale sizes
  a2  volume occupancy: opacity per (r-bin, height-bin) and against the inputs' prior point cloud (above the
      ceiling, below the floor, beyond the farthest surface of the theta column, behind the prior surface)
  a3  contribution: share of Gaussians whose opacity gets no gradient from the summed alpha of the targets
  a4  zero-shot reductions: WS-PSNR / SSIM / LPIPS (all targets and the novel view) after removing a subset
  a5  error localisation: WS-MSE by latitude band, the seam columns, per target view, low-alpha pixels
  a6  prior consistency: disagreement of the two inputs' prior depths on co-visible pixels
  a7  one training step at batch 2: time per module and peak memory

  CUDA_VISIBLE_DEVICES=<gpu> python tools/diagnose_gaussians.py --ckpt <checkpoint dir> --out-dir <dir> \
      [--split val|test] [--parts a1,a2,...] [--max-batches N]

Test numbers describe the released checkpoint only; no change is chosen from them. Nothing is written outside
--out-dir; no weight is updated (a3 and a7 run backward passes whose gradients are dropped).
"""

import argparse
import importlib
import json
import math
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from einops import rearrange

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

PARTS = ("a1", "a2", "a3", "a4", "a5", "a6", "a7")
SPLITS = {"val": "mp3d_double_256_val", "test": "mp3d_double_256"}
OPACITY_EDGES = (1 / 255, 0.01, 0.05, 0.1, 0.5)
NOVEL_VIEW = 1  # target index of the middle frame (targets [0, 1, 2], inputs [0, 2])


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- model and data


def build(py_config, ckpt):
    from mmengine.config import Config
    from safetensors.torch import load_file

    from tools import switches

    cfg = Config.fromfile(py_config)
    switches.apply(cfg, [])
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the models read repo-relative files at construction (LPIPS weights)
    from builder import builder

    model = builder.build(cfg.model).cuda().eval()
    os.chdir(cwd)
    ckpt_file = osp.join(ckpt, "model.safetensors") if osp.isdir(ckpt) else ckpt
    state = load_file(ckpt_file, device="cpu")
    model.load_state_dict(state, strict=True)
    if type(model).__name__ != "OmniGaussianCylinderAll":
        sys.exit(f"error: this tool reads the joint model, got {type(model).__name__}")
    return cfg, model


def loader(split, batch_size):
    import evaluate

    entry = evaluate.EVAL_ENTRIES[SPLITS[split]]
    module, factory = entry["loader"]
    return getattr(importlib.import_module(module), factory)(batch_size, stage=entry["stage"])


def to_cuda(x):
    """The batch on the GPU (evaluate.py gets this from accelerate's prepared loader)."""
    if torch.is_tensor(x):
        return x.cuda(non_blocking=True)
    if isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cuda(v) for v in x)
    return x


class Capture:
    """Forward hooks on pixel_gs and volume_gs: the per-scale pixel Gaussians and the cylinder-frame volume
    Gaussians of the last forward_test."""

    def __init__(self, model):
        self.pixel, self.volume = None, None
        model.pixel_gs.register_forward_hook(self._pixel)
        model.volume_gs.register_forward_hook(self._volume)

    def _pixel(self, module, args, out):
        self.pixel = [s["gaussians"].detach() for s in out["stages"]]

    def _volume(self, module, args, out):
        self.volume = out.detach()


def predict(model, capture, batch):
    """forward_test plus the pieces: (preds, gts, data_dict, parts) with parts = dict(
    pixel_stages=[B, V*n_s, 14] world, volume_cyl=[B, V, Nv, 14] cylinder frame, n_pixel, n_volume)."""
    with torch.no_grad():
        preds, gts = model.forward_test(batch)
        data = model.get_data(batch)
    stages = capture.pixel
    n_pixel = sum(s.shape[1] for s in stages)
    b, v = data["imgs"].shape[:2]
    volume_cyl = capture.volume.view(b, v, -1, capture.volume.shape[-1])
    gaussians = preds["gaussian"]
    assert torch.equal(gaussians[:, :n_pixel], torch.cat(stages, dim=1)), "pixel Gaussians are not the hook's"
    assert gaussians.shape[1] == n_pixel + v * volume_cyl.shape[2]
    return preds, gts, data, dict(pixel_stages=stages, volume_cyl=volume_cyl, n_pixel=n_pixel, n_volume=volume_cyl.shape[2])


def render(model, gaussians, data, views=None):
    c2w, fx, fy = data["output_c2ws"], data["output_fovxs"], data["output_fovys"]
    if views is not None:
        c2w, fx, fy = c2w[:, views], fx[:, views], fy[:, views]
    with torch.no_grad():
        return model.renderer.render(gaussians=gaussians, c2w=c2w, fovx=fx, fovy=fy)


def input_points(data):
    """World points of the inputs' prior depth: [B, V, H, W, 3]."""
    depth = rearrange(data["depths"], "b v c h w -> b v h w c")
    return data["rays_o"] + data["rays_d"] * depth


def to_frame(points, c2w):
    """World points [..., 3] into the frame of camera c2w [4, 4]."""
    w2c = torch.inverse(c2w)
    return points @ w2c[:3, :3].T + w2c[:3, 3]


def erp_lookup(points_cam, image, h, w):
    """Value of image [C, h, w] at the ERP projection of camera-frame points [N, 3] (nearest pixel), and the
    points' ray distance."""
    dist = points_cam.norm(dim=-1).clamp(min=1e-6)
    theta = (torch.atan2(points_cam[:, 0], points_cam[:, 2]) + math.pi) / (2 * math.pi)
    phi = (torch.asin((points_cam[:, 1] / dist).clamp(-1, 1)) + math.pi / 2) / math.pi
    col = (theta * w).long().clamp(0, w - 1)
    row = (phi * h).long().clamp(0, h - 1)
    return image[:, row, col], dist


# ----------------------------------------------------------------------------- metrics


class Metrics:
    """WS-PSNR / SSIM / LPIPS of renders against the targets, kept per variant, per scene and per view."""

    def __init__(self):
        from tools.metrics import WSPSNR

        self.ws = WSPSNR()
        self.rows = defaultdict(list)  # (variant, scene) -> [(view, wspsnr, ssim, lpips)]

    def add(self, variant, scene, pred, gt):
        from tools.metrics import compute_lpips, compute_ssim

        p, g = pred.reshape(-1, *pred.shape[-3:]), gt.reshape(-1, *gt.shape[-3:])
        ws = self.ws.ws_psnr(rearrange(g, "n c h w -> n h w c"), rearrange(p, "n c h w -> n h w c"), max_val=1.0)
        ssim, lp = compute_ssim(g, p), compute_lpips(g, p)
        for view in range(p.shape[0]):
            self.rows[(variant, scene)].append((view, float(ws[view]), float(ssim[view]), float(lp[view])))

    def summary(self):
        out = defaultdict(dict)
        for (variant, scene), rows in self.rows.items():
            arr = np.array(rows)
            novel = arr[arr[:, 0] == NOVEL_VIEW]
            out[variant][scene] = dict(
                n_views=len(arr),
                all=dict(zip(("wspsnr", "ssim", "lpips"), arr[:, 1:].mean(0).round(5).tolist())),
                novel=dict(zip(("wspsnr", "ssim", "lpips"), novel[:, 1:].mean(0).round(5).tolist())),
            )
        return out


# ----------------------------------------------------------------------------- parts


class A1:
    """Opacity histograms and pixel scale sizes."""

    def __init__(self):
        self.counts = defaultdict(lambda: np.zeros(len(OPACITY_EDGES) + 1, dtype=np.int64))
        self.scales = defaultdict(list)

    def add(self, parts, v):
        for s, stage in enumerate(parts["pixel_stages"]):
            per_view = stage.view(stage.shape[0], v, -1, 14)
            for view in range(v):
                self._hist(f"pixel_s{s}_v{view}", per_view[:, view, :, 6])
                sample = per_view[:, view, :, 11:14].max(dim=-1).values.flatten()
                self.scales[f"pixel_s{s}"].append(sample[torch.randperm(sample.numel(), device=sample.device)[:20000]].cpu())
        vol = parts["volume_cyl"]
        for c in range(v):
            per_sibling = vol[:, c, :, 6].view(vol.shape[0], -1, 3)
            for g in range(3):
                self._hist(f"volume_c{c}_g{g}", per_sibling[..., g])

    def _hist(self, key, opacity):
        idx = torch.bucketize(opacity.flatten(), torch.tensor(OPACITY_EDGES, device=opacity.device))
        self.counts[key] += torch.bincount(idx, minlength=len(OPACITY_EDGES) + 1).cpu().numpy()

    def result(self):
        labels = ["<1/255", "1/255-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.5", ">=0.5"]
        out = {"opacity_fraction": {}, "pixel_scale_m": {}}
        for key, c in sorted(self.counts.items()):
            out["opacity_fraction"][key] = dict(zip(labels, (c / c.sum()).round(4).tolist()), n=int(c.sum()))
        for key, chunks in sorted(self.scales.items()):
            x = torch.cat(chunks).numpy()
            out["pixel_scale_m"][key] = {f"p{q}": float(np.percentile(x, q)) for q in (10, 50, 90, 99)}
        return out


class A2:
    """Volume occupancy against the prior point cloud of the inputs."""

    def __init__(self, cfg):
        self.pc_range = list(cfg.model.volume_gs.gs_decoder.pc_range)
        dec = cfg.model.volume_gs.gs_decoder
        self.grid = (dec.tpv_r, dec.tpv_theta, dec.tpv_z, dec.gpv)
        self.rz_sum = np.zeros((dec.tpv_r, dec.tpv_z))
        self.rz_n = 0
        self.mass = defaultdict(float)
        self.count = defaultdict(float)
        self.layout_checked = False

    def add(self, parts, data):
        vol = parts["volume_cyl"]
        b, v = vol.shape[:2]
        r_n, t_n, z_n, g_n = self.grid
        pts_world = input_points(data)  # [B, V, H, W, 3]
        for bi in range(b):
            all_in = pts_world[bi].reshape(-1, 3)
            for c in range(v):
                g = vol[bi, c]  # [Nv, 14] in cylinder c's frame (camera c, y down)
                xyz, opa = g[:, :3], g[:, 6]
                cells = opa.view(r_n, t_n, z_n, g_n)
                if not self.layout_checked:
                    self._check_layout(xyz.view(r_n, t_n, z_n, g_n, 3))
                self.rz_sum += cells.max(dim=-1).values.mean(dim=1).cpu().numpy()
                self.rz_n += 1
                prior = to_frame(all_in, data["c2ws"][bi, c])
                y_lo, y_hi = torch.quantile(prior[::7, 1], torch.tensor([0.01, 0.99], device=prior.device))
                r_xz = xyz[:, [0, 2]].norm(dim=-1)
                theta_g = torch.atan2(xyz[:, 0], xyz[:, 2])
                above = xyz[:, 1] < y_lo - 0.3  # y points down: smaller y = higher
                below = xyz[:, 1] > y_hi + 0.3
                # farthest prior surface per theta column (64 columns of 5.6 deg)
                cols = 64
                prior_theta = ((torch.atan2(prior[:, 0], prior[:, 2]) + math.pi) / (2 * math.pi) * cols).long() % cols
                prior_r = prior[:, [0, 2]].norm(dim=-1)
                r_far = torch.zeros(cols, device=prior.device).scatter_reduce(0, prior_theta, prior_r, "amax")
                g_col = ((theta_g + math.pi) / (2 * math.pi) * cols).long() % cols
                beyond = r_xz > r_far[g_col] + 0.5
                # behind the surface seen from camera c (and from both inputs)
                behind = []
                for cam in range(v):
                    p_cam = to_frame(transform(xyz, data["c2ws"][bi, c]), data["c2ws"][bi, cam])
                    prior_d, dist = erp_lookup(p_cam, data["depths"][bi, cam], *data["depths"].shape[-2:])
                    behind.append(dist > prior_d[0] + 0.3)
                behind_own, behind_all = behind[c], torch.stack(behind).all(dim=0)
                total = opa.sum().item()
                for name, mask in (("above_ceiling", above), ("below_floor", below), ("beyond_far_surface", beyond),
                                   ("behind_own_camera", behind_own), ("behind_both_inputs", behind_all),
                                   ("outside_room", above | below | beyond)):
                    self.mass[name] += opa[mask].sum().item() / max(total, 1e-12)
                    self.count[name] += mask.float().mean().item()
                self.mass["_n"] += 1
                self.count["_n"] += 1

    def _check_layout(self, xyz):
        r = xyz[..., [0, 2]].norm(dim=-1).mean(dim=(1, 2, 3))
        y = xyz[..., 1].mean(dim=(0, 1, 3))
        if not (torch.all(r[1:] > r[:-1]) and torch.all(y[1:] > y[:-1])):
            raise RuntimeError("volume Gaussians are not laid out as (r, theta, z, gpv); a2 binning would be wrong")
        self.layout_checked = True

    def result(self):
        n = max(self.mass.pop("_n", 1), 1)
        nc = max(self.count.pop("_n", 1), 1)
        z_edges = np.linspace(self.pc_range[2], self.pc_range[5], self.grid[2] + 1)
        return dict(
            fraction_of_opacity_mass={k: round(v / n, 4) for k, v in self.mass.items()},
            fraction_of_gaussians={k: round(v / nc, 4) for k, v in self.count.items()},
            mean_max_opacity_per_r_bin=(self.rz_sum / max(self.rz_n, 1)).mean(axis=1).round(4).tolist(),
            mean_max_opacity_per_height_bin=(self.rz_sum / max(self.rz_n, 1)).mean(axis=0).round(4).tolist(),
            height_bin_edges_y_down=z_edges.round(3).tolist(),
        )


def transform(points, c2w):
    """Camera-frame points [..., 3] of camera c2w into the world."""
    return points @ c2w[:3, :3].T + c2w[:3, 3]


class A3:
    """Gaussians that never reach a target pixel (zero opacity gradient of the summed target alpha)."""

    def __init__(self):
        self.frac = defaultdict(list)

    def add(self, model, preds, parts, data):
        g = preds["gaussian"].detach()
        opa = g[..., 6:7].clone().requires_grad_(True)
        out = model.renderer.render(
            gaussians=torch.cat([g[..., :6], opa, g[..., 7:]], dim=-1),
            c2w=data["output_c2ws"], fovx=data["output_fovxs"], fovy=data["output_fovys"],
        )
        out["alpha"].sum().backward()
        dead = (opa.grad[..., 0] == 0).float()
        n_pixel, v = parts["n_pixel"], data["imgs"].shape[1]
        self.frac["pixel"].append(dead[:, :n_pixel].mean().item())
        vol = dead[:, n_pixel:].view(dead.shape[0], v, -1)
        for c in range(v):
            self.frac[f"volume_c{c}"].append(vol[:, c].mean().item())
        self.frac["all"].append(dead.mean().item())

    def result(self):
        return {"never_contributing_fraction": {k: round(float(np.mean(v)), 4) for k, v in self.frac.items()}}


def variants(parts, data):
    """name -> (keep mask over the fused Gaussians [B, N]) for the zero-shot reductions."""
    stages, vol = parts["pixel_stages"], parts["volume_cyl"]
    b, v, nv = vol.shape[:3]
    n_pixel = parts["n_pixel"]
    dev = vol.device
    ones = torch.ones(b, n_pixel + v * nv, dtype=torch.bool, device=dev)
    out = {"full": ones}
    for c in range(v):
        m = ones.clone()
        m[:, n_pixel:] = False
        m[:, n_pixel + c * nv: n_pixel + (c + 1) * nv] = True
        out[f"pixels_plus_cylinder{c}"] = m  # the other cylinder removed; every pixel Gaussian kept
    # each world point of cylinder c is kept only when camera c is the nearest input camera
    m = ones.clone()
    centres = data["c2ws"][:, :, :3, 3]  # [B, V, 3]
    for c in range(v):
        world = torch.stack([transform(vol[bi, c, :, :3], data["c2ws"][bi, c]) for bi in range(b)])
        d = torch.stack([(world - centres[:, k, None]).norm(dim=-1) for k in range(v)], dim=-1)
        m[:, n_pixel + c * nv: n_pixel + (c + 1) * nv] = d.argmin(dim=-1) == c
    out["volume_nearest_camera"] = m
    # one sibling per cell: the most opaque
    m = ones.clone()
    for c in range(v):
        opa = vol[:, c, :, 6].view(b, -1, 3)
        keep = torch.zeros_like(opa, dtype=torch.bool).scatter_(-1, opa.argmax(dim=-1, keepdim=True), True)
        m[:, n_pixel + c * nv: n_pixel + (c + 1) * nv] = keep.view(b, -1)
    out["volume_gpv_max_sibling"] = m
    # pixel scales 0-1 dropped
    m = ones.clone()
    m[:, : stages[0].shape[1] + stages[1].shape[1]] = False
    out["pixel_drop_scales_0_1"] = m
    # cross-view pixel ownership: drop a view-j pixel Gaussian that view i (closer camera) sees at the same depth
    m = ones.clone()
    start = 0
    h, w = data["depths"].shape[-2:]
    for stage in stages:
        n_s = stage.shape[1] // v
        for bi in range(b):
            for j in range(v):
                xyz = stage[bi, j * n_s: (j + 1) * n_s, :3]
                drop = torch.zeros(n_s, dtype=torch.bool, device=dev)
                for i in range(v):
                    if i == j:
                        continue
                    p_i = to_frame(xyz, data["c2ws"][bi, i])
                    prior_i, dist_i = erp_lookup(p_i, data["depths"][bi, i], h, w)
                    dist_j = (xyz - data["c2ws"][bi, j, :3, 3]).norm(dim=-1)
                    drop |= ((dist_i - prior_i[0]).abs() < 0.05 * prior_i[0]) & (dist_i < dist_j)
                m[bi, start + j * n_s: start + (j + 1) * n_s] = ~drop
        start += stage.shape[1]
    out["pixel_cross_view_owner"] = m
    return out


class A4:
    def __init__(self):
        self.metrics = Metrics()
        self.kept = defaultdict(list)

    def add(self, model, preds, gts, parts, data, scene):
        g = preds["gaussian"]
        for name, keep in variants(parts, data).items():
            if name == "full":
                pred = preds["img"]
            else:
                sub = g[0, keep[0]][None]  # batch size 1
                pred = render(model, sub, data)["image"]
            self.metrics.add(name, scene, pred, gts["img"])
            self.kept[name].append(keep.float().mean().item())

    def result(self):
        return {"metrics": self.metrics.summary(), "kept_fraction": {k: round(float(np.mean(v)), 4) for k, v in self.kept.items()}}


SEAM_PROFILE = 32  # columns on each side of the seam in A5's profile (two tiles)


class A5:
    def __init__(self):
        self.acc = defaultdict(lambda: defaultdict(list))
        self.profiles = defaultdict(list)

    def add(self, preds, gts, data, scene):
        pred, gt = preds["img"][0], gts["img"][0]  # [T, 3, H, W]
        err = ((pred - gt) ** 2).mean(dim=1)  # [T, H, W]
        t, h, w = err.shape
        lat = (0.5 - (torch.arange(h, device=err.device) + 0.5) / h) * 180.0
        wts = torch.cos(lat * math.pi / 180.0)[:, None]
        a = self.acc[scene]
        for name, band in (("lat_lt30", lat.abs() < 30), ("lat_30_60", (lat.abs() >= 30) & (lat.abs() < 60)),
                           ("lat_ge60", lat.abs() >= 60)):
            a[f"wsmse_{name}"].append(((err[:, band] * wts[band]).sum() / (wts[band].sum() * t * w)).item())
        col = (err * wts).sum(dim=(0, 1)) / (wts.sum() * t)  # WS-weighted MSE per column
        a["seam_cols_mse_over_median"].append((torch.cat([col[:8], col[-8:]]).mean() / col.median()).item())
        # per-column profile around the seam: columns W-32..W-1 then 0..31, relative to the median column
        profile = torch.cat([col[-SEAM_PROFILE:], col[:SEAM_PROFILE]]) / col.median()
        self.profiles[scene].append(profile.cpu())
        for view in range(t):
            a[f"wsmse_view{view}"].append(((err[view] * wts).sum() / (wts.sum() * w)).item())
        alpha = render_alpha_cache.get(id(preds))
        if alpha is not None:
            low = alpha[0, :, 0] < 0.9
            a["low_alpha_pixel_fraction"].append(low.float().mean().item())
            a["low_alpha_error_share"].append(((err * low).sum() / err.sum().clamp(min=1e-12)).item())

    def result(self):
        out = {s: {k: round(float(np.mean(v)), 6) for k, v in d.items()} for s, d in self.acc.items()}
        for s, chunks in self.profiles.items():
            # column offsets -32..-1 (left of the seam = last image columns) then 0..31 (first image columns)
            out[s]["seam_column_profile_wsmse_over_median"] = torch.stack(chunks).mean(dim=0).round(decimals=3).tolist()
        return out


render_alpha_cache = {}


class A6:
    def __init__(self):
        self.acc = defaultdict(lambda: defaultdict(list))

    def add(self, data, scene):
        depth, c2w = data["depths"][0], data["c2ws"][0]  # [V, 1, H, W], [V, 4, 4]
        pts = input_points(data)[0]  # [V, H, W, 3]
        h, w = depth.shape[-2:]
        p01 = to_frame(pts[0].reshape(-1, 3)[::3], c2w[1])
        prior1, dist = erp_lookup(p01, depth[1], h, w)
        rel = (dist - prior1[0]).abs() / prior1[0].clamp(min=1e-3)
        covisible = rel < 0.2
        a = self.acc[scene]
        a["covisible_fraction"].append(covisible.float().mean().item())
        a["median_rel_disagreement_covisible"].append(rel[covisible].median().item() if covisible.any() else float("nan"))
        a["fraction_over_5pct_of_covisible"].append((rel[covisible] > 0.05).float().mean().item() if covisible.any() else float("nan"))
        # uncensored distribution (all projected pixels, and those below 0.5 = excluding gross occlusion)
        for name, sel in (("all", torch.ones_like(rel, dtype=torch.bool)), ("lt0.5", rel < 0.5)):
            r = rel[sel]
            if r.numel() == 0:
                continue
            for pct in (50, 75, 90):
                a[f"{name}_q{pct}"].append(torch.quantile(r[::7], pct / 100).item())
            for thr in (0.12, 0.15, 0.20, 0.30, 0.45):
                a[f"{name}_share_over_{thr:.2f}"].append((r > thr).float().mean().item())

    def result(self):
        return {s: {k: round(float(np.nanmean(v)), 4) for k, v in d.items()} for s, d in self.acc.items()}


def a7_profile(model, split):
    """One training forward + backward at batch 2: CUDA time of the top-level modules and peak memory."""
    batch = to_cuda(next(iter(loader(split, 2))))
    times = defaultdict(float)
    events = {}
    # use_checkpoint recomputes module forwards inside backward: keys carry the phase they were measured in
    phase = {"name": "forward"}

    def pre(name):
        def hook(module, args):
            events[name] = torch.cuda.Event(enable_timing=True)
            events[name].record()
        return hook

    def post(name):
        def hook(module, args, out):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            times[f"{phase['name']}: {name}"] += events[name].elapsed_time(end)
        return hook

    handles = []
    for name in ("backbone", "pixel_gs", "volume_gs", "perceptual_loss"):
        mod = getattr(model, name, None)
        if mod is not None:
            handles += [mod.register_forward_pre_hook(pre(name)), mod.register_forward_hook(post(name))]
    original_render = model.renderer.render

    def timed_render(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = original_render(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times[f"{phase['name']}: renderer.render"] += start.elapsed_time(end)
        return out

    model.renderer.render = timed_render
    decoder = model.volume_gs.gs_decoder
    original_color = decoder.get_panorama_color

    def timed_color(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = original_color(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times[f"{phase['name']}: get_panorama_color (incl. colour MLP)"] += start.elapsed_time(end)
        return out

    decoder.get_panorama_color = timed_color
    model.train()
    try:
        for step in range(2):  # the first step warms up
            times.clear()
            phase["name"] = "forward"
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            loss = model.forward(batch, "train")[0]
            torch.cuda.synchronize()
            t1 = time.time()
            phase["name"] = "backward recompute"
            loss.backward()
            torch.cuda.synchronize()
            t2 = time.time()
            model.zero_grad(set_to_none=True)
    finally:
        model.renderer.render = original_render
        decoder.get_panorama_color = original_color
        model.eval()
        for h in handles:
            h.remove()
    return dict(
        forward_s=round(t1 - t0, 3),
        backward_s=round(t2 - t1, 3),
        peak_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        module_ms={k: round(v, 1) for k, v in sorted(times.items())},
        note="module_ms 'forward: X' = X's forward in the forward pass; 'backward recompute: X' = X's forward "
        "recomputed by activation checkpointing during backward (the backward kernels themselves are not split)",
    )


# ----------------------------------------------------------------------------- main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--py-config", default="configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py")
    parser.add_argument("--ckpt", required=True, help="checkpoint directory or model.safetensors")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), default="val")
    parser.add_argument("--parts", default=",".join(PARTS))
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    if set(parts) - set(PARTS):
        sys.exit(f"error: unknown parts {sorted(set(parts) - set(PARTS))}; known {PARTS}")
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if len(visible) != 1:
        sys.exit("error: set CUDA_VISIBLE_DEVICES to exactly one GPU")
    py_config = osp.abspath(osp.join(REPO_ROOT, args.py_config) if not osp.isabs(args.py_config) else args.py_config)
    ckpt = osp.abspath(args.ckpt)
    out_dir = osp.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(osp.join(out_dir, "cwd"), exist_ok=True)
    torch.manual_seed(1111)

    cfg, model = build(py_config, ckpt)
    os.chdir(osp.join(out_dir, "cwd"))  # relative writes by the model stay in the output directory
    capture = Capture(model)
    log(f"model built and loaded from {ckpt}; split {args.split}; parts {parts}")

    a1, a2, a3, a4, a5, a6 = A1(), A2(cfg), A3(), A4(), A5(), A6()
    result = {"split": args.split, "dataset": SPLITS[args.split], "ckpt": ckpt, "config": py_config, "parts": parts}
    loop_parts = set(parts) - {"a7"}
    if loop_parts:
        for i, batch in enumerate(loader(args.split, 1)):
            if args.max_batches is not None and i >= args.max_batches:
                break
            batch = to_cuda(batch)
            preds, gts, data, pieces = predict(model, capture, batch)
            scene = batch["scene"][0] if "scene" in batch else args.split
            if args.split == "val":
                scene = "val_1.0m"
            if "a5" in loop_parts:
                render_alpha_cache[id(preds)] = render(model, preds["gaussian"], data)["alpha"]
            if "a1" in loop_parts:
                a1.add(pieces, data["imgs"].shape[1])
            if "a2" in loop_parts:
                a2.add(pieces, data)
            if "a3" in loop_parts:
                a3.add(model, preds, pieces, data)
            if "a4" in loop_parts:
                a4.add(model, preds, gts, pieces, data, scene)
            if "a5" in loop_parts:
                a5.add(preds, gts, data, scene)
                render_alpha_cache.clear()
            if "a6" in loop_parts:
                a6.add(data, scene)
            if i % 10 == 0:
                log(f"batch {i}: {pieces['n_pixel']} pixel + {data['imgs'].shape[1] * pieces['n_volume']} volume Gaussians")
            torch.cuda.empty_cache()
        result["counts"] = dict(pixel=pieces["n_pixel"], volume=data["imgs"].shape[1] * pieces["n_volume"],
                                pixel_per_scale=[s.shape[1] for s in pieces["pixel_stages"]])
    for name, part in (("a1", a1), ("a2", a2), ("a3", a3), ("a4", a4), ("a5", a5), ("a6", a6)):
        if name in loop_parts:
            result[name] = part.result()
    if "a7" in parts:
        result["a7"] = a7_profile(model, args.split)
    path = osp.join(out_dir, f"diagnose_{args.split}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    log(f"written {path}")


if __name__ == "__main__":
    main()
