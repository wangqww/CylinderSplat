"""Switch-off identity of the four stage models against their frozen f7b20b9 copies.

With every switch at its default (lpips_eval, freeze_frozen_bn, v1_identity_pose,
rotate_gaussians_to_world off; backbone_ckpt = the released path) each live model must
behave bitwise like tests/legacy_ref/models_ref.py: same modules and parameters, same
train()/eval() state, same camera metas for v=1 and v=2 with non-identity poses, same
losses, gradients, returned tensors and RNG consumption. The switch-on behaviour of D2a,
D2b and D3 is checked here as well; D4 (on) is in test_d4_composition.py and the render
removal (C1) in test_render_removal.py.

This module also holds the CPU harness the other two files import: stub backbone, pixel
head, volume head, renderer and LPIPS (monkeypatched over MODELS / GaussianRenderer /
LPIPS), synthetic batches, and comparison helpers. Nothing here needs CUDA.

The last section re-checks that the frozen copies models_ref.py, train_ref.py and
eval_ref.py are verbatim `git show f7b20b9:<path>` code (skipped without git or the commit).
"""

import inspect
import math
import re
from collections import OrderedDict

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import nn

import model.omni_gs_cylinder_all as all_mod
import model.omni_gs_cylinder_pixel as pixel_mod
import model.omni_gs_cylinder_volume as volume_mod
import model.omni_gs_cylinder_volume_360loc_pan2 as pan2_mod
from tests.legacy_ref import models_ref as ref
from tests.legacy_ref import verbatim

# ----------------------------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------------------------

H, W = 8, 16          # panorama size of inputs and targets
BS = 2                # batch size
N_OUT = 3             # target views
N_VOL = 10            # volume Gaussians per cylinder (stub)
# r_min, phi_min, z_min, r_max, phi_max, z_max: small enough that the masks are mixed
POINT_CLOUD_RANGE = [0.0, 0.0, -1.0, 2.0, 6.28, 1.0]
LEGACY_PANSPLAT_CKPT = '/home/qiwei/Nips25/PanSplat/logs/wwrerdvv/checkpoints/last.ckpt'

KINDS = ("all", "pan2", "volume", "pixel")
NEW = {
    "all": all_mod.OmniGaussianCylinderAll,
    "pan2": pan2_mod.OmniGaussianCylinderVolume360LocPan2,
    "volume": volume_mod.OmniGaussianCylinderVolume,
    "pixel": pixel_mod.OmniGaussianCylinderPixel,
}
LEGACY = {
    "all": ref.LegacyOmniGaussianCylinderAll,
    "pan2": ref.LegacyOmniGaussianCylinderVolume360LocPan2,
    "volume": ref.LegacyOmniGaussianCylinderVolume,
    "pixel": ref.LegacyOmniGaussianCylinderPixel,
}

# loss weights copied from the configs each class is trained with
_WEIGHTS = {
    # omni_gs_160x320_mp3d_cylinder_all_256.py
    "all": dict(weight_recon=1.0, weight_perceptual=0.05, weight_depth_abs=1.0, weight_recon_vol=0.1,
                weight_perceptual_vol=0.005, weight_depth_abs_vol=0.1, weight_volume_loss=0.0),
    # same with the (disabled in every config) alpha-entropy term on
    "all_entropy": dict(weight_recon=1.0, weight_perceptual=0.05, weight_depth_abs=1.0, weight_recon_vol=0.1,
                        weight_perceptual_vol=0.005, weight_depth_abs_vol=0.1, weight_volume_loss=0.1),
    # omni_gs_160x320_360Loc_cylinder_all_256.py (volume terms off)
    "pan2": dict(weight_recon=1.0, weight_perceptual=0.05, weight_depth_abs=0.1, weight_recon_vol=0.0,
                 weight_perceptual_vol=0.0, weight_depth_abs_vol=0.0, weight_volume_loss=0.0),
    # Pan2 with the volume terms on (then the volume render must stay)
    "pan2_vol": dict(weight_recon=1.0, weight_perceptual=0.05, weight_depth_abs=0.1, weight_recon_vol=0.1,
                     weight_perceptual_vol=0.005, weight_depth_abs_vol=0.1, weight_volume_loss=0.0),
    # omni_gs_160x320_mp3d_cylinder_volume_256.py
    "volume": dict(weight_recon=1.0, weight_perceptual=0.05, weight_depth_abs=0.1, weight_recon_vol=1.0,
                   weight_perceptual_vol=0.05, weight_depth_abs_vol=1.0, weight_volume_loss=0.0),
    # omni_gs_160x320_mp3d_cylinder_pixel_256.py
    "pixel": dict(weight_recon=1.0, weight_perceptual=0.05, weight_depth_abs=1.0, weight_recon_vol=1.0,
                  weight_perceptual_vol=0.05, weight_depth_abs_vol=1.0, weight_volume_loss=0.0),
}
# (kind, loss_kind) pairs the forward tests run
LOSS_CASES = [("all", "all"), ("all", "all_entropy"), ("pan2", "pan2"), ("pan2", "pan2_vol"),
              ("volume", "volume"), ("pixel", "pixel")]


def loss_args(loss_kind):
    return ConfigDict(dict(recon_loss_type="l2", recon_loss_vol_type="l2_mask", perceptual_loss_vol_type="mask",
                           depth_abs_loss_vol_type="mask", mask_dptm=True, perceptual_resolution=[H, W],
                           **_WEIGHTS[loss_kind]))


class PassThroughRegistry:
    """MODELS stand-in: the tests hand over ready-made stub modules instead of configs."""

    @staticmethod
    def build(cfg):
        return cfg


class StubBackbone(nn.Module):
    """Backbone stand-in with a BatchNorm, so train()/eval() changes its output and statistics."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.BatchNorm2d(3)

    def forward(self, img, depths_in, confs_in, pluckers, viewmats):
        b = img.shape[0]
        feat = self.norm(rearrange(img, "b v c h w -> (b v) c h w")) * self.scale
        return {"trans_features": [rearrange(feat, "(b v) c h w -> b v c h w", b=b)]}


def _pixel_gaussians(per_view, gain, img, img_feats, depths_in, origins_in, directions_in):
    depth = rearrange(depths_in, "b v c h w -> b v h w c")
    xyz = origins_in + directions_in * depth                              # world frame
    rgb = rearrange(img, "b v c h w -> b v h w c")
    feat = rearrange(img_feats["trans_features"][0], "b v c h w -> b v h w c")
    opacity = torch.sigmoid(rgb.mean(-1, keepdim=True) * gain)
    rotation = F.normalize(torch.cat([1.0 + rgb, depth], dim=-1), dim=-1)
    scale = 0.01 * (1.0 + depth.expand(*depth.shape[:-1], 3))
    gaussians = torch.cat([xyz, rgb, opacity, rotation, scale], dim=-1)  # 14 channels
    features = torch.cat([feat, depth], dim=-1)
    layout = "b v h w c -> b v (h w) c" if per_view else "b v h w c -> b (v h w) c"
    return rearrange(gaussians, layout), rearrange(features, layout)


class StubPixelGS(nn.Module):
    """PixelGaussian stand-in (PixelGaussian360Loc layout [b, v, hw, c] when per_view)."""

    def __init__(self, per_view=False):
        super().__init__()
        self.per_view = per_view
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, *args, **kwargs):
        # records how it was called and with which poses: PixelGaussian gets them as the 8th
        # positional argument (extrinsics_in when passed by keyword)
        self.calls.append({"n_args": len(args), "kwargs": sorted(kwargs),
                           "poses": args[7] if len(args) > 7 else kwargs.get("extrinsics_in")})
        return self._forward(*args, **kwargs)

    def _forward(self, img, img_feats, depths_in, confs_in, pluckers_in, origins_in, directions_in,
                 extrinsics_in, patch_idx=0, status="train"):
        gaussians, features = _pixel_gaussians(self.per_view, self.gain, img, img_feats, depths_in,
                                               origins_in, directions_in)
        return {"gaussians": gaussians, "features": features, "gaussians_raw": gaussians}


class StubPixelGS512(pixel_mod.PixelGaussian512):
    """PixelGaussian512 stand-in with the interface of plan D4 (i): extrinsics keyword-only."""

    def __init__(self):
        BaseModule.__init__(self)
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, img, img_feats, depths_in, confs_in, pluckers_in, origins_in, directions_in,
                patch_idx=0, status="train", *, extrinsics_in=None):
        self.calls.append({"patch_idx": patch_idx, "status": status, "extrinsics_in": extrinsics_in})
        gaussians, features = _pixel_gaussians(False, self.gain, img, img_feats, depths_in,
                                               origins_in, directions_in)
        return {"gaussians": gaussians, "features": features, "gaussians_raw": gaussians}


class StubVolumeGS(nn.Module):
    """VolumeGaussianCylinder stand-in: N_VOL Gaussians per cylinder in the reference camera
    frame, [xyz, rgb, opacity, rotation (wxyz), scale]. The output depends on the candidates
    (the cylinder mask), the features, the colours and the pose in the metas."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.rotation = None    # set a wxyz quaternion to give every Gaussian that rotation
        self.calls = []

    def forward(self, img_feats, candidate_gaussians, candidate_feats, img_color, img_depth, img_metas,
                status="train"):
        self.calls.append({"img_metas": img_metas, "status": status,
                           "n_candidates": [len(c) for c in candidate_gaussians]})
        feats = img_feats[0]                                           # (b v) vo c h w
        k = torch.arange(N_VOL, dtype=feats.dtype)
        out = []
        for i in range(feats.shape[0]):
            cand = candidate_gaussians[i]
            center = cand[:, :3].mean(0) if len(cand) else feats.new_zeros(3)
            cand_feat = candidate_feats[i].mean() if len(cand) else feats.new_zeros(())
            pose = img_metas[i]["lidar2img"].to(feats)                 # [vo, 4, 4]
            ring = torch.stack([torch.cos(k), 0.1 * k - 0.5, torch.sin(k)], dim=-1)
            xyz = center + 0.3 * ring * self.gain + 0.05 * pose[0, :3, 3]
            rgb = torch.sigmoid(feats[i].mean() + img_color[i].mean() + cand_feat + 0.1 * k)[:, None].expand(N_VOL, 3)
            opacity = torch.sigmoid(0.2 * k - 1.0)[:, None]
            if self.rotation is not None:
                rotation = self.rotation.to(feats).expand(N_VOL, 4)
            else:
                rotation = F.normalize(torch.stack([1.0 + 0.1 * k, torch.sin(k), torch.cos(k), 0.2 * k], -1), dim=-1)
            scale = (0.02 + 0.01 * k)[:, None] * torch.tensor([1.0, 2.0, 3.0], dtype=feats.dtype)
            out.append(torch.cat([xyz, rgb, opacity, rotation, scale], dim=-1))
        return torch.stack(out)


class StubRenderer:
    """GaussianRenderer stand-in: every pixel depends on every Gaussian channel and on the
    camera. Records each call with a copy of the Gaussians it was given."""

    MIX = torch.linspace(-1.0, 1.0, 14)

    def __init__(self, device, resolution=(H, W), znear=0.1, zfar=100.0, **kwargs):
        self.resolution = list(resolution)
        self.calls = []

    def render(self, gaussians, c2w, fovx=None, fovy=None, rays_o=None, rays_d=None, bg_color=None,
               scale_modifier=1.0):
        self.calls.append(("render", gaussians.detach().clone()))
        B, V = c2w.shape[:2]
        h, w = self.resolution
        mixed = (gaussians * self.MIX.to(gaussians)).sum(-1)                         # B N
        stats = torch.stack([mixed.mean(1), mixed.std(1), gaussians[..., 7:11].abs().mean((1, 2))], -1)
        cam = c2w[..., :3, :].reshape(B, V, 12).sum(-1)                               # B V
        grid = torch.linspace(0.0, 1.0, h * w, dtype=gaussians.dtype).view(1, 1, 1, h, w)
        base = stats[:, None, :, None, None] + 0.1 * cam[:, :, None, None, None] + grid  # B V 3 h w
        return {"image": torch.sigmoid(base),
                "alpha": torch.sigmoid(base.sum(2, keepdim=True)),
                "depth": 1.0 + F.softplus(base.mean(2, keepdim=True))}

    def render_orthographic(self, gaussians, width=30, height=30, **kwargs):
        self.calls.append(("orthographic", gaussians.detach().clone()))
        B = gaussians.shape[0]
        mixed = (gaussians * self.MIX.to(gaussians)).sum(-1).mean(1)
        image = torch.sigmoid(mixed[:, None, None, None] + torch.zeros(B, 3, 16, 16))
        return {"image": image, "alpha": image[:, :1], "depth": image[:, :1]}


class StubLPIPS(nn.Module):
    """LPIPS stand-in with the same dropout-in-train behaviour (use_dropout=True)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, input, target):
        return self.drop((input - target) ** 2).mean(dim=(1, 2, 3), keepdim=True)


def pansplat_state():
    """A PanSplat-style Lightning state dict for StubBackbone (plus keys the loader filters out)."""
    state = OrderedDict()
    for key, value in StubBackbone().state_dict().items():
        state["encoder.backbone." + key] = value + 0.25 if value.is_floating_point() else value.clone()
    state["encoder.depth_head.weight"] = torch.ones(2)
    state["encoder.backbone.norm.weight_unused"] = torch.ones(3)
    return state


class Harness:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.ckpt = tmp_path / "ckpt" / "pansplat.ckpt"

    def build(self, kind, legacy=False, loss_kind=None, pixel_gs=None, **switches):
        """A live (or legacy) model of `kind` around fresh, identically initialised stubs."""
        kwargs = dict(
            backbone=StubBackbone(),
            pixel_gs=pixel_gs if pixel_gs is not None else StubPixelGS(per_view=(kind == "pan2")),
            volume_gs=StubVolumeGS(),
            camera_args=ConfigDict(resolution=[H, W], znear=0.1, zfar=15.0),
            loss_args=loss_args(loss_kind or kind),
            dataset_params=ConfigDict(pc_range=list(POINT_CLOUD_RANGE)),
            use_checkpoint=False,
            point_cloud_range=list(POINT_CLOUD_RANGE),
        )
        if kind == "volume":
            kwargs["name"] = "ori"
        if kind == "pixel" and not legacy and "backbone_ckpt" not in switches:
            kwargs["backbone_ckpt"] = str(self.ckpt)
        kwargs.update(switches)
        return (LEGACY if legacy else NEW)[kind](**kwargs)

    def pngs(self):
        return sorted(p.name for p in self.tmp.glob("*.png"))


@pytest.fixture
def harness(monkeypatch, tmp_path):
    for mod in (all_mod, pan2_mod, volume_mod, pixel_mod, ref):
        monkeypatch.setattr(mod, "MODELS", PassThroughRegistry)
        monkeypatch.setattr(mod, "GaussianRenderer", StubRenderer)
        monkeypatch.setattr(mod, "LPIPS", StubLPIPS)
    h = Harness(tmp_path)
    h.ckpt.parent.mkdir()
    torch.save({"state_dict": pansplat_state()}, str(h.ckpt))
    real_load = torch.load

    def fake_load(f, *args, **kwargs):
        # the legacy Pixel model loads its hard-coded PanSplat path
        if f == LEGACY_PANSPLAT_CKPT:
            return {"state_dict": pansplat_state()}
        return real_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, "load", fake_load)
    monkeypatch.chdir(tmp_path)  # the legacy copies write debug PNGs into the cwd
    return h


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# exact 90 degree rotation about +y (x -> -z, z -> x)
ROT_Y_90 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def make_pose(R, t):
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor(t, dtype=torch.float32)
    return T


def equirect_dirs():
    lat = ((torch.arange(H, dtype=torch.float32) + 0.5) / H - 0.5) * math.pi
    lon = ((torch.arange(W, dtype=torch.float32) + 0.5) / W - 0.5) * 2 * math.pi
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    return torch.stack([torch.sin(lon) * torch.cos(lat), torch.sin(lat), torch.cos(lon) * torch.cos(lat)], -1)


def make_batch(v, seed=0, input_rotation=None):
    """A synthetic loader batch with v input views and N_OUT targets. Input poses are rotated
    and translated (non-identity c2w) unless input_rotation fixes every input rotation."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=g)

    def in_pose(b, i):
        R = input_rotation if input_rotation is not None else rot_y(0.4 + 0.7 * i + 0.1 * b) @ rot_x(0.2 - 0.15 * i)
        return make_pose(R, [0.3 * i - 0.1 * b, 0.05 * b - 0.02 * i, 0.2 - 0.25 * i])

    def out_pose(b, j):
        return make_pose(rot_y(-0.3 + 0.6 * j) @ rot_x(0.1 * j - 0.05 * b), [0.15 * j + 0.05 * b, 0.02 * j, 0.1 - 0.2 * j])

    c2w_in = torch.stack([torch.stack([in_pose(b, i) for i in range(v)]) for b in range(BS)])
    c2w_out = torch.stack([torch.stack([out_pose(b, j) for j in range(N_OUT)]) for b in range(BS)])
    dirs = equirect_dirs()

    def rays(c2w):
        d = torch.einsum("bnij,hwj->bnhwi", c2w[..., :3, :3], dirs).contiguous()
        o = c2w[:, :, None, None, :3, 3].expand_as(d).contiguous()
        return o, d

    rays_o_in, rays_d_in = rays(c2w_in)
    rays_o_out, rays_d_out = rays(c2w_out)
    return {
        "inputs": {"rgb": rand(BS, v, 3, H, W)},
        "inputs_pix": {
            "rays_o": rays_o_in, "rays_d": rays_d_in,
            "fx": rand(BS, v), "fy": rand(BS, v), "cx": rand(BS, v), "cy": rand(BS, v),
            "c2w": c2w_in, "ck": torch.zeros(BS, v, 3, 3),
            "depth_m": 0.5 + 2.0 * rand(BS, v, 1, H, W), "conf_m": rand(BS, v, 1, H, W),
        },
        "inputs_vol": {"w2i": torch.inverse(c2w_in)},
        "outputs": {
            "rgb": rand(BS, N_OUT, 3, H, W),
            "depth": 0.5 + 2.0 * rand(BS, N_OUT, 1, H, W),
            "depth_m": 0.5 + 2.0 * rand(BS, N_OUT, 1, H, W),
            "conf_m": rand(BS, N_OUT, 1, H, W),
            "rays_o": rays_o_out, "rays_d": rays_d_out, "c2w": c2w_out,
            "fovx": torch.full((BS, N_OUT), math.pi / 2), "fovy": torch.full((BS, N_OUT), math.pi / 2),
            "depth_gt": 0.5 + 2.0 * rand(BS, N_OUT, 1, H, W),
            "mask_gt": rand(BS, N_OUT, 1, H, W) > 0.3,
        },
    }


def run(model, batch, fn="forward", split="train", seed=1234):
    """Run forward / forward_test from a fixed RNG state; return (output, RNG state after)."""
    torch.manual_seed(seed)
    out = model.forward(batch, split) if fn == "forward" else model.forward_test(batch)
    return out, torch.get_rng_state()


def dropped_slots(kind, loss_kind, split):
    """forward() return slots filled by a render the live model no longer makes (None there)."""
    if split != "train":
        return set()
    if kind == "all":
        return {3}                                   # render_pkg_pixel
    if kind == "pan2":
        # render_pkg_pixel (slots 3 and 5); render_pkg_volume unless a volume loss reads it
        return {3, 5} if loss_kind == "pan2_vol" else {3, 4, 5}
    return set()


def assert_same(a, b, where="out"):
    """Bitwise equality of nested outputs (tensors, dicts, lists, tuples, scalars)."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), where
        assert a.dtype == b.dtype and a.shape == b.shape, f"{where}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"
        assert torch.equal(a, b), f"{where}: values differ (max abs {(a.double() - b.double()).abs().max().item()})"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{where}: keys {list(a)} vs {list(b) if isinstance(b, dict) else b}"
        for key in a:
            assert_same(a[key], b[key], f"{where}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), f"{where}: {type(a).__name__}/{len(a)} vs {b!r}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"


def training_flags(model):
    return {name: module.training for name, module in model.named_modules()}


# ----------------------------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------------------------

SWITCH_KWARGS = {
    "all": ("lpips_eval", "v1_identity_pose", "rotate_gaussians_to_world"),
    "pan2": ("lpips_eval", "v1_identity_pose", "rotate_gaussians_to_world"),
    "volume": ("lpips_eval", "freeze_frozen_bn", "rotate_gaussians_to_world"),
    "pixel": ("lpips_eval", "rotate_gaussians_to_world"),
}


@pytest.mark.parametrize("kind", KINDS)
def test_switch_kwargs_default_off(kind):
    params = inspect.signature(NEW[kind].__init__).parameters
    for name in SWITCH_KWARGS[kind]:
        assert params[name].default is False, (kind, name)


@pytest.mark.parametrize("kind", KINDS)
def test_constructor_matches_legacy(harness, kind):
    new, old = harness.build(kind), harness.build(kind, legacy=True)
    # The frozen copies are named Legacy<Class>; compare the class names without that prefix.
    assert [(n, type(m).__name__) for n, m in new.named_modules()] == \
           [(n, type(m).__name__.removeprefix("Legacy")) for n, m in old.named_modules()]
    assert [(n, p.shape, p.requires_grad) for n, p in new.named_parameters()] == \
           [(n, p.shape, p.requires_grad) for n, p in old.named_parameters()]
    assert_same(new.state_dict(), old.state_dict(), "state_dict")
    assert training_flags(new) == training_flags(old)


def test_pixel_backbone_ckpt_default_is_released_path():
    params = inspect.signature(pixel_mod.OmniGaussianCylinderPixel.__init__).parameters
    assert params["backbone_ckpt"].default == LEGACY_PANSPLAT_CKPT


def test_pixel_backbone_ckpt_loaded(harness):
    new = harness.build("pixel")
    expected = {k.replace("encoder.backbone.", ""): v for k, v in pansplat_state().items()
                if k.replace("encoder.backbone.", "") in StubBackbone().state_dict()}
    assert_same(dict(new.backbone.state_dict()), expected, "backbone")


def test_pixel_backbone_ckpt_missing_is_an_error(harness):
    missing = harness.tmp / "no_such_dir" / "last.ckpt"
    with pytest.raises(FileNotFoundError) as err:
        harness.build("pixel", backbone_ckpt=str(missing))
    assert str(missing) in str(err.value) and "backbone_ckpt" in str(err.value)
    with pytest.raises(FileNotFoundError):
        harness.build("pixel", backbone_ckpt=None)


# ----------------------------------------------------------------------------------------
# D2a / D2b: train() state
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_train_state_off_matches_legacy(harness, kind):
    new, old = harness.build(kind), harness.build(kind, legacy=True)
    for step in ("train", "eval", "train", "train_false", "train"):
        for m in (new, old):
            ret = m.train(False) if step == "train_false" else getattr(m, step)()
            assert ret is m
        assert training_flags(new) == training_flags(old), step
    # released behaviour: train() turns the LPIPS dropout back on
    assert new.perceptual_loss.training and new.perceptual_loss.drop.training


@pytest.mark.parametrize("kind", KINDS)
def test_lpips_eval_on(harness, kind):
    model = harness.build(kind, lpips_eval=True)
    reference = harness.build(kind)
    for _ in range(2):
        model.train()
        reference.train()
        flags, ref_flags = training_flags(model), training_flags(reference)
        for name, training in flags.items():
            if name == "perceptual_loss" or name.startswith("perceptual_loss."):
                assert training is False, name
            else:
                assert training == ref_flags[name], name
        model.eval()
        assert not any(training_flags(model).values())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("v", [1, 2])
def test_lpips_eval_on_consumes_no_dropout_rng(harness, kind, v):
    batch = make_batch(v)
    off = harness.build(kind)
    on = harness.build(kind, lpips_eval=True)
    off.train()
    on.train()
    torch.manual_seed(7)
    start = torch.get_rng_state()
    _, rng_on = run(on, batch, seed=7)
    _, rng_off = run(off, batch, seed=7)
    assert torch.equal(rng_on, start)          # LPIPS dropout inactive, nothing else draws
    assert not torch.equal(rng_off, start)     # released: dropout draws in the perceptual loss


def _frozen_prefixes(model):
    return tuple(p for p in ("backbone", "neck", "pixel_gs") if hasattr(model, p))


def test_freeze_frozen_bn_on(harness):
    model = harness.build("volume", freeze_frozen_bn=True)
    reference = harness.build("volume")
    prefixes = _frozen_prefixes(model)
    assert prefixes == ("backbone", "pixel_gs")
    for _ in range(2):
        model.train()
        reference.train()
        flags, ref_flags = training_flags(model), training_flags(reference)
        for name, training in flags.items():
            if name.split(".")[0] in prefixes:
                assert training is False, name
            else:
                assert training == ref_flags[name], name
    # the frozen BatchNorm keeps its statistics through a training forward
    before = {k: v.clone() for k, v in model.backbone.norm.state_dict().items()}
    run(model, make_batch(2))
    assert_same(dict(model.backbone.norm.state_dict()), before, "running stats")
    # released behaviour: they drift
    run(reference, make_batch(2))
    assert not torch.equal(reference.backbone.norm.running_mean, before["running_mean"])


def test_freeze_frozen_bn_and_lpips_eval_together(harness):
    model = harness.build("volume", freeze_frozen_bn=True, lpips_eval=True)
    model.train()
    assert model.volume_gs.training and model.training
    assert not model.pixel_gs.training and not model.backbone.training
    assert not model.perceptual_loss.training


# ----------------------------------------------------------------------------------------
# D3: camera metas
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("v", [1, 2])
def test_get_data_off_matches_legacy(harness, kind, v):
    new, old = harness.build(kind), harness.build(kind, legacy=True)
    batch = make_batch(v)
    assert_same(new.get_data(batch), old.get_data(batch), "data_dict")
    if v == 1 and kind in ("all", "pan2", "pixel"):
        # released: the absolute w2i of the single view
        for meta, w2i in zip(new.get_data(batch)["img_metas"], batch["inputs_vol"]["w2i"]):
            assert torch.equal(meta["lidar2img"], w2i)


@pytest.mark.parametrize("kind", ["all", "pan2"])
@pytest.mark.parametrize("v", [1, 2])
def test_v1_identity_pose_on(harness, kind, v):
    on, old = harness.build(kind, v1_identity_pose=True), harness.build(kind, legacy=True)
    batch = make_batch(v)
    metas_on = on.get_data(batch)["img_metas"]
    metas_old = old.get_data(batch)["img_metas"]
    if v == 2:
        assert_same(metas_on, metas_old, "img_metas")    # the switch only touches V=1
        return
    assert len(metas_on) == BS
    for meta, w2i in zip(metas_on, batch["inputs_vol"]["w2i"]):
        assert torch.equal(meta["lidar2img"], w2i @ w2i.inverse())
        assert torch.allclose(meta["lidar2img"], torch.eye(4).expand(1, 4, 4), atol=1e-5)
        assert meta["img_shape"] == [[H, W]]
    # as stage-2 OmniGaussianCylinderVolume already does
    volume_metas = harness.build("volume").get_data(batch)["img_metas"]
    assert_same(metas_on, volume_metas, "img_metas vs volume")


@pytest.mark.parametrize("kind", ["all", "pan2"])
def test_v1_identity_pose_reaches_volume_gs(harness, kind):
    on, off = harness.build(kind, v1_identity_pose=True), harness.build(kind)
    batch = make_batch(1)
    out_on, _ = run(on, batch)
    out_off, _ = run(off, batch)
    for meta in on.volume_gs.calls[-1]["img_metas"]:
        assert torch.allclose(meta["lidar2img"], torch.eye(4).expand(1, 4, 4), atol=1e-5)
    assert not torch.equal(out_on[7], out_off[7])


# ----------------------------------------------------------------------------------------
# forward / forward_test with every switch off (C1 + D3 off + D4 off)
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind,loss_kind", LOSS_CASES)
@pytest.mark.parametrize("v", [1, 2])
@pytest.mark.parametrize("split", ["train", "val"])
def test_forward_off_matches_legacy(harness, kind, loss_kind, v, split):
    new = harness.build(kind, loss_kind=loss_kind)
    old = harness.build(kind, legacy=True, loss_kind=loss_kind)
    new.train()
    old.train()
    out_new, rng_new = run(new, make_batch(v), split=split)
    out_old, rng_old = run(old, make_batch(v), split=split)

    assert torch.equal(rng_new, rng_old)         # same RNG draws (LPIPS dropout) in the same order
    dropped = dropped_slots(kind, loss_kind, split)
    for slot, (a, b) in enumerate(zip(out_new, out_old)):
        if slot in dropped:
            assert a is None and b is not None, slot
        else:
            assert_same(a, b, f"forward[{slot}]")
    # the kept path drew the same metas, candidates and pixel-head calls
    assert_same(new.volume_gs.calls, old.volume_gs.calls, "volume_gs calls")
    assert_same(new.pixel_gs.calls, old.pixel_gs.calls, "pixel_gs calls")
    assert_same(new.state_dict(), old.state_dict(), "state_dict after forward")  # BN statistics

    # same loss graph: identical gradients
    out_new[0].backward()
    out_old[0].backward()
    grads_new = {n: p.grad for n, p in new.named_parameters()}
    grads_old = {n: p.grad for n, p in old.named_parameters()}
    assert list(grads_new) == list(grads_old)
    for name in grads_new:
        if grads_old[name] is None:
            assert grads_new[name] is None, name
        else:
            assert_same(grads_new[name], grads_old[name], f"grad {name}")
    assert any(g is not None for g in grads_new.values())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("v", [1, 2])
def test_forward_test_off_matches_legacy(harness, kind, v):
    new, old = harness.build(kind), harness.build(kind, legacy=True)
    new.eval()
    old.eval()
    with torch.no_grad():
        out_new, rng_new = run(new, make_batch(v), fn="forward_test")
        out_old, rng_old = run(old, make_batch(v), fn="forward_test")
    assert torch.equal(rng_new, rng_old)
    assert_same(out_new, out_old, "forward_test")
    assert_same(new.volume_gs.calls, old.volume_gs.calls, "volume_gs calls")
    assert_same(new.pixel_gs.calls, old.pixel_gs.calls, "pixel_gs calls")
    assert {k: len(t) for k, t in new.benchmarker.execution_times.items()} == \
           {k: len(t) for k, t in old.benchmarker.execution_times.items()}


@pytest.mark.parametrize("fn", ["forward", "forward_test"])
def test_pixel_model_calls_pixel_gaussian_positionally(harness, fn):
    """PixelGaussian keeps the released call: the poses as the 8th positional argument."""
    new, old = harness.build("pixel"), harness.build("pixel", legacy=True)
    batch = make_batch(2)
    run(new, batch, fn=fn)
    run(old, batch, fn=fn)
    assert_same(new.pixel_gs.calls, old.pixel_gs.calls, "pixel_gs calls")
    assert new.pixel_gs.calls[-1]["n_args"] == 8
    assert torch.equal(new.pixel_gs.calls[-1]["poses"], batch["inputs_pix"]["c2w"])


# ----------------------------------------------------------------------------------------
# frozen copies are verbatim: models_ref, and the trainer and evaluator copies
# (train_ref, eval_ref), re-checked against `git show f7b20b9:<path>` when git and the
# commit are available
# ----------------------------------------------------------------------------------------

_MODELS_BLOCK = re.compile(r"    # ---- [\w /]+: (\S+) lines (\d+)-(\d+) @ f7b20b9 \(verbatim\) ----")
_MODELS_CLASS = re.compile(r'    """(\S+) @ f7b20b9 \(class body excerpts\)\."""')


def test_legacy_ref_blocks_are_verbatim():
    """models_ref.py: each class adds only its class line and a docstring naming the file all
    its blocks are copied from."""
    lines = verbatim.source_lines("models_ref")
    first_class = next(i for i, line in enumerate(lines) if line.startswith("class "))
    covered, source, n_blocks = set(), None, 0
    for i, line in enumerate(lines):
        if m := _MODELS_CLASS.fullmatch(line):
            source = m.group(1)
        elif m := _MODELS_BLOCK.fullmatch(line):
            path, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            assert path == source, (i + 1, path, source)
            covered.update(range(i, verbatim.assert_verbatim(lines, i + 1, path, a, b)))
            n_blocks += 1
    assert n_blocks == 4 * 7  # __init__, extract_img_feat, device/dtype, plucker_embedder, get_data, forward, forward_test
    allowed = (r"class Legacy\w+\(BaseModule\):", _MODELS_CLASS.pattern)
    assert verbatim.stray_lines(lines, covered, first_class, len(lines), allowed) == []


_EVAL_BLOCK = re.compile(r"# (\S+\.py):(\d+)-(\d+)(?: \(.*\))?")
_EVAL_LOOP = re.compile(r"def legacy_eval_\w+\(cfg, my_model, val_dataloader, accelerator, logger, global_iter\):")


def test_eval_ref_blocks_are_verbatim():
    """eval_ref.py: each block follows its source comment; the four evaluation loops add only
    their def line and `return locals()`."""
    lines = verbatim.source_lines("eval_ref")
    marks = [(i, m) for i, line in enumerate(lines) if (m := _EVAL_BLOCK.fullmatch(line))]
    assert len(marks) == 8
    covered, copies = set(), {}
    for i, m in marks:
        path, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        start = i + 2 if _EVAL_LOOP.fullmatch(lines[i + 1]) else i + 1
        end = verbatim.assert_verbatim(lines, start, path, a, b)
        copies[re.match(r"(?:def|class) (\w+)", lines[i + 1]).group(1)] = lines[start:end]
        if start == i + 2:
            assert lines[end] == "    return locals()", path
            end += 1
        covered.update(range(i, end))
    assert sum(name.startswith("legacy_eval_") for name in copies) == 4
    assert verbatim.stray_lines(lines, covered, marks[0][0], len(lines)) == []
    # the notes on two markers: save_ply is the same in the 512, 360Loc and VIGOR scripts; the
    # single-view script (run through the double_256 loop in test_eval_aggregation.py) differs
    # only in its loader import
    save_ply = copies["save_ply"]
    for script in ("evaluate_mp3d_double_512.py", "evaluate_360Loc_double_256.py", "evaluate_VIGOR.py"):
        src = verbatim.git_show(script)
        assert any(src[j:j + len(save_ply)] == save_ply for j in range(len(src))), script
    double = verbatim.git_show("evaluate_mp3d_double_256.py")
    single = verbatim.git_show("evaluate_mp3d_single_256.py")
    assert len(double) == len(single)
    assert [(x, y) for x, y in zip(double, single) if x != y] == [
        ("from data.mp3d_dataloader_double_256 import load_MP3D_data",
         "from data.mp3d_dataloader_single_256 import load_MP3D_data")]


_TRAIN_SOURCE = re.compile(r"(\S+\.py):(\d+)-(\d+)")
# resume(): "same block in all six trainers"; its header gives the other five by range only
_RESUME_ALSO = [("train_mp3d_cylinder_single_256.py", 175, 184),
                ("train_360Loc_cylinder_double_all_512.py", 175, 184),
                ("train_mp3d_cylinder_double.py", 160, 169), ("train_vigor_cylinder_double.py", 160, 169),
                ("train_mp3d_cylinder_double_512.py", 147, 156)]


def test_train_ref_blocks_are_verbatim():
    """train_ref.py: each function is its def line, a header comment naming the source (and the
    scripts with the same lines), the verbatim body and, for the schedulers, `return scheduler`."""
    lines = verbatim.source_lines("train_ref")
    defs = [i for i, line in enumerate(lines) if line.startswith("def ")]
    assert len(defs) == 3
    covered, starts = set(), {}
    for d in defs:
        start, header = d + 1, lines[d + 1]
        while header.count("(") > header.count(")"):  # the header runs to its closing parenthesis
            start += 1
            header += lines[start]
        start += 1
        (path, a, b), *same = _TRAIN_SOURCE.findall(header)
        end = verbatim.assert_verbatim(lines, start, path, int(a), int(b))
        for path, a, b in same:                       # "(same in <path>:<range> ...)"
            assert verbatim.assert_verbatim(lines, start, path, int(a), int(b)) == end, path
        covered.update(range(d, end))
        starts[re.match(r"def (\w+)", lines[d]).group(1)] = start
    for path, a, b in _RESUME_ALSO:
        verbatim.assert_verbatim(lines, starts["resume"], path, a, b)
    assert verbatim.stray_lines(lines, covered, defs[0], len(lines), (r"    return scheduler",)) == []
