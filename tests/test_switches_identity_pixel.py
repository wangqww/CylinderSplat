"""Switch-off identity of the three pixel heads (D4 rotate_gaussians_to_world,
D9 pixel_depth_sampling, PixelGaussian360Loc num_frames).

The live heads, built with their defaults, are compared bitwise with the frozen
f7b20b9 copies in tests/legacy_ref/pixel_ref.py:
  * the two edited statements of each head (the prior-depth grid_sample and the
    world-frame concatenation) on synthetic tensors, v=1 and v=2, non-identity c2w;
  * the whole forward of each head on CPU, v=1 and v=2 (PixelGaussian360Loc: v=2
    and v=3), non-identity c2w. The cost-volume UNet (xformers attention, forced
    bf16) is replaced by nn.Identity in the head instance, so the live and the
    legacy forward run the same modules.
PixelGaussian360Loc at v=1 (released code: RuntimeError) now runs PixelGaussian's
single-view fallback; its tests are at the end of this file.

The helpers below are also used by tests/test_pixel_switches_on.py.
"""

import math
import re

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from model.pixel.pixel_gs import PixelGaussian
from model.pixel.pixel_gs_360loc import PixelGaussian360Loc
from model.pixel.pixel_gs_512 import PixelGaussian512
from tests.legacy_ref import pixel_ref
from tests.legacy_ref.verbatim import git_show as _git_show

IMAGE_HEIGHT = 32
FEATURE_CHANNELS = [128, 96, 64, 32]  # trans_features, 1/8 .. 1/1 resolution

LEGACY_FORWARD = {
    PixelGaussian: pixel_ref.LegacyPixelGaussian.forward,
    PixelGaussian360Loc: pixel_ref.LegacyPixelGaussian360Loc.forward,
    PixelGaussian512: pixel_ref.LegacyPixelGaussian512.forward,
}


# ----------------------------------------------------------------------------- helpers

def rotation_matrix(axis, angle):
    """Rodrigues: unit axis (..., 3), angle (...) -> (..., 3, 3)."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, dtype=axis.dtype).expand(k.shape)
    s = torch.sin(angle)[..., None, None]
    c = torch.cos(angle)[..., None, None]
    return eye + s * k + (1 - c) * (k @ k)


def random_c2w(b, v, gen, identity_rotation=False):
    """[b, v, 4, 4] rigid camera-to-world poses; the rotations are far from identity."""
    axis = torch.nn.functional.normalize(torch.randn(b, v, 3, generator=gen), dim=-1)
    angle = 0.5 + 2.0 * torch.rand(b, v, generator=gen)
    c2w = torch.eye(4).repeat(b, v, 1, 1)
    if not identity_rotation:
        c2w[..., :3, :3] = rotation_matrix(axis, angle)
    c2w[..., :3, 3] = torch.randn(b, v, 3, generator=gen)
    return c2w


def erp_directions(h, w):
    """Camera-frame unit rays at ERP pixel centres [h, w, 3] (the loaders' convention)."""
    u = (torch.arange(w) + 0.5) / w
    t = (torch.arange(h) + 0.5) / h
    phi = (u * 2 * math.pi - math.pi)[None, :].expand(h, w)
    theta = (t * math.pi - math.pi / 2)[:, None].expand(h, w)
    return torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta),
                        torch.cos(theta) * torch.cos(phi)], dim=-1)


def make_inputs(b, v, seed=0, identity_rotation=False, height=IMAGE_HEIGHT):
    """Synthetic forward inputs of the pixel heads at height x 2*height."""
    gen = torch.Generator().manual_seed(seed)
    h, w = height, 2 * height
    c2w = random_c2w(b, v, gen, identity_rotation=identity_rotation)
    dirs = erp_directions(h, w)
    rays_d = torch.einsum("bvij,hwj->bvhwi", c2w[..., :3, :3], dirs).contiguous()
    rays_o = c2w[:, :, None, None, :3, 3].expand(b, v, h, w, 3).contiguous()
    feats = [torch.randn(b, v, c, h // s, w // s, generator=gen)
             for c, s in zip(FEATURE_CHANNELS, (8, 4, 2, 1))]
    return dict(
        img=torch.rand(b, v, 3, h, w, generator=gen),
        img_feats={"trans_features": feats},
        depths=0.5 + 4.5 * torch.rand(b, v, 1, h, w, generator=gen),
        confs=torch.rand(b, v, 1, h, w, generator=gen),
        pluckers=torch.randn(b, v, 6, h, w, generator=gen),
        rays_o=rays_o,
        rays_d=rays_d,
        c2w=c2w,
    )


def build_head(cls, seed=0, **kwargs):
    """A small head on CPU; the cost-volume UNet (if any) is replaced by nn.Identity."""
    torch.manual_seed(seed)
    head = cls(image_height=IMAGE_HEIGHT, patchs_height=1, patchs_width=1, gh_cnn_layers=3, **kwargs)
    if hasattr(head, "corr_refine_nets"):
        assert type(head.corr_refine_nets[3]).__name__ == "UNetModel"
        head.corr_refine_nets[3] = torch.nn.Identity()
    return head.eval()


def positional_args(inputs, with_extrinsics):
    args = [inputs["img"], inputs["img_feats"], inputs["depths"], inputs["confs"],
            inputs["pluckers"], inputs["rays_o"], inputs["rays_d"]]
    if with_extrinsics:
        args.append(inputs["c2w"])
    return args


def run_live(head, inputs, **kwargs):
    """The live forward, called the way the top-level models call each head."""
    with torch.no_grad():
        if isinstance(head, PixelGaussian512):
            return head(*positional_args(inputs, False), **kwargs)
        return head(*positional_args(inputs, True), **kwargs)


def run_legacy(head, inputs):
    """The frozen f7b20b9 forward, run on the same head instance (same weights)."""
    forward = LEGACY_FORWARD[type(head)]
    with torch.no_grad():
        return forward(head, *positional_args(inputs, not isinstance(head, PixelGaussian512)))


def assert_outputs_identical(a, b):
    assert set(a) == set(b)
    assert len(a["stages"]) == len(b["stages"])
    for key in a:
        if key == "stages":
            for sa, sb in zip(a["stages"], b["stages"]):
                assert set(sa) == set(sb)
                for k in sa:
                    assert sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]), (key, k)
        else:
            assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), key


# ------------------------------------------------------------- frozen copies are verbatim

_BLOCK = re.compile(r"^[ \t]*# >>> BEGIN (\w+) (\S+):(\d+)-(\d+)[^\n]*\n(.*?)^[ \t]*# <<< END \1[ \t]*$",
                    re.S | re.M)


def test_legacy_ref_blocks_are_verbatim():
    src = open(pixel_ref.__file__).read()
    blocks = _BLOCK.findall(src)
    assert len(blocks) == 11
    for kind, path, a, b, body in blocks:
        legacy = _git_show(path)[int(a) - 1:int(b)]
        if kind == "stmt":
            assert [l.strip() for l in body.rstrip("\n").split("\n")] == [l.strip() for l in legacy], path
        else:
            assert body == "\n".join(legacy) + "\n", (path, a, b)
    # the helpers are shared by the two cost-volume heads
    assert _git_show("model/pixel/pixel_gs.py")[23:112] == _git_show("model/pixel/pixel_gs_360loc.py")[23:112]


# ------------------------------------------------------------------ constructor defaults

@pytest.mark.parametrize("cls", [PixelGaussian, PixelGaussian360Loc, PixelGaussian512])
def test_defaults_are_released_behaviour(cls):
    head = build_head(cls)
    assert head.rotate_gaussians_to_world is False
    assert head.pixel_depth_sampling == "bilinear"


@pytest.mark.parametrize("cls", [PixelGaussian, PixelGaussian360Loc, PixelGaussian512])
def test_switches_add_no_parameters(cls):
    """Same parameter/buffer names, shapes and init values (same RNG use) with switches on."""
    torch.manual_seed(0)
    off = cls(image_height=IMAGE_HEIGHT).state_dict()
    torch.manual_seed(0)
    on = cls(image_height=IMAGE_HEIGHT, rotate_gaussians_to_world=True, pixel_depth_sampling="nearest").state_dict()
    assert list(off) == list(on)
    for k in off:
        assert torch.equal(off[k], on[k]), k


def test_pixel_gaussian_360loc_num_frames_default_and_argument():
    torch.manual_seed(0)
    default = PixelGaussian360Loc(image_height=IMAGE_HEIGHT)
    frames = [m.n_frames for m in default.modules() if hasattr(m, "n_frames")]
    assert frames and all(n == 2 for n in frames)
    torch.manual_seed(0)
    three = PixelGaussian360Loc(image_height=IMAGE_HEIGHT, num_frames=3)
    assert all(m.n_frames == 3 for m in three.modules() if hasattr(m, "n_frames"))
    sd2, sd3 = default.state_dict(), three.state_dict()
    assert list(sd2) == list(sd3)
    for k in sd2:
        assert torch.equal(sd2[k], sd3[k]), k


# ------------------------------------------------------- edited statements, synthetic data

@pytest.fixture(scope="module")
def heads():
    return {cls: build_head(cls) for cls in (PixelGaussian, PixelGaussian360Loc, PixelGaussian512)}


LEGACY_DEPTH_SAMPLE = {
    PixelGaussian: pixel_ref.pixel_gs_depth_sample,
    PixelGaussian360Loc: pixel_ref.pixel_gs_360loc_depth_sample,
    PixelGaussian512: pixel_ref.pixel_gs_512_depth_sample,
}
LEGACY_CONCAT = {
    PixelGaussian: pixel_ref.pixel_gs_world_concat,
    PixelGaussian360Loc: pixel_ref.pixel_gs_360loc_world_concat,
    PixelGaussian512: pixel_ref.pixel_gs_512_world_concat,
}


@pytest.mark.parametrize("v", [1, 2])
@pytest.mark.parametrize("cls", [PixelGaussian, PixelGaussian360Loc, PixelGaussian512])
def test_depth_sample_step_off_matches_legacy(heads, cls, v):
    gen = torch.Generator().manual_seed(1)
    b, n = 2, 57
    depths_fullres = 0.5 + 4.5 * torch.rand(b * v, 1, 16, 32, generator=gen)
    grid = torch.rand(b * v, n, 1, 2, generator=gen) * 2.2 - 1.1  # includes border clamping
    live = heads[cls]._sample_prior_depth(depths_fullres, grid)
    legacy = LEGACY_DEPTH_SAMPLE[cls](depths_fullres, grid)
    assert torch.equal(live, legacy)


def synthetic_gaussian_parts(cls, b, v, m, gen):
    """means/rgbs/opacities/rotations/scales in the layout the head concatenates."""
    shape = (b, v, m) if cls is PixelGaussian360Loc else (b, v * m)
    rot = torch.nn.functional.normalize(torch.randn(*shape, 4, generator=gen), dim=-1)
    return (torch.randn(*shape, 3, generator=gen), torch.rand(*shape, 3, generator=gen),
            torch.rand(*shape, 1, generator=gen), rot, torch.rand(*shape, 3, generator=gen))


@pytest.mark.parametrize("v", [1, 2])
@pytest.mark.parametrize("cls", [PixelGaussian, PixelGaussian360Loc, PixelGaussian512])
def test_world_concat_step_off_matches_legacy(heads, cls, v):
    gen = torch.Generator().manual_seed(2)
    b, m = 2, 23
    parts = synthetic_gaussian_parts(cls, b, v, m, gen)
    c2w = random_c2w(b, v, gen)
    live = heads[cls]._assemble_gaussians(*parts, c2w)
    legacy = LEGACY_CONCAT[cls](*parts)
    assert live.dtype == legacy.dtype and torch.equal(live, legacy)
    if cls is PixelGaussian512:
        # default call sites pass no extrinsics
        assert torch.equal(heads[cls]._assemble_gaussians(*parts, None), legacy)


# -------------------------------------------------------------- whole forward vs legacy

@pytest.mark.parametrize("g", [1, 3])
@pytest.mark.parametrize("v", [1, 2])
def test_pixel_gaussian_forward_off_matches_legacy(v, g):
    head = build_head(PixelGaussian, gaussians_per_pixel=g)
    inputs = make_inputs(b=2, v=v, seed=10 + v)
    assert_outputs_identical(run_live(head, inputs), run_legacy(head, inputs))


@pytest.mark.parametrize("g", [1, 2])
@pytest.mark.parametrize("v", [2, 3])
def test_pixel_gaussian_360loc_forward_off_matches_legacy(v, g):
    head = build_head(PixelGaussian360Loc, gaussians_per_pixel=g)
    inputs = make_inputs(b=2, v=v, seed=20 + v)
    assert_outputs_identical(run_live(head, inputs), run_legacy(head, inputs))


@pytest.mark.parametrize("v", [1, 2])
def test_pixel_gaussian_512_forward_off_matches_legacy(v):
    head = build_head(PixelGaussian512)
    inputs = make_inputs(b=2, v=v, seed=40 + v)
    legacy = run_legacy(head, inputs)
    assert_outputs_identical(run_live(head, inputs), legacy)
    # the new keyword-only extrinsics_in is ignored while the switch is off
    assert_outputs_identical(run_live(head, inputs, extrinsics_in=inputs["c2w"]), legacy)


# ------------------------------------------- PixelGaussian360Loc single view (v=1, new path)
#
# The released PixelGaussian360Loc stops at torch.stack([]) when v == 1. The live head runs
# PixelGaussian's single-view fallback there (the feature self-correlation), inside a block
# that only v == 1 enters; the v >= 2 forwards above stay bitwise equal to the legacy copy.

GAUSSIAN_WIDTHS = {"gaussians": 14, "features": 128, "gaussians_raw": 14}


def run_live_with_cost_volume_input(head, inputs):
    """The live forward and the input it feeds the cost-volume refine UNet:
    cat(correlation, stage-2 features, nearest prior depth) [(v b), 1 + c + 1, h, w]."""
    seen = []
    hook = head.corr_refine_nets.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    try:
        out = run_live(head, inputs)
    finally:
        hook.remove()
    assert len(seen) == 1
    return out, seen[0]


@pytest.mark.parametrize("g", [1, 2])
def test_pixel_gaussian_360loc_single_view_runs(g):
    """v=1 has no released behaviour (legacy raises); the live head gives finite Gaussians
    in its per-view layout [B, 1, M, C]."""
    b = 2
    head = build_head(PixelGaussian360Loc, num_frames=1, gaussians_per_pixel=g)
    inputs = make_inputs(b=b, v=1, seed=30)
    with pytest.raises(RuntimeError):
        run_legacy(head, inputs)
    out = run_live(head, inputs)
    per_stage = [getattr(head, f"gs_xy_{s}_0").shape[0] * g for s in range(head.gh_stages)]
    assert len(out["stages"]) == head.gh_stages
    for stage, m in zip(out["stages"], per_stage):
        for key, c in GAUSSIAN_WIDTHS.items():
            assert stage[key].shape == (b, 1, m, c), key
    for key, c in GAUSSIAN_WIDTHS.items():
        assert out[key].shape == (b, 1, sum(per_stage), c), key
        assert out[key].dtype == torch.float32 and torch.isfinite(out[key]).all(), key


@pytest.mark.parametrize("g", [1, 2])
def test_pixel_gaussian_360loc_single_view_equals_pixel_gaussian(g):
    """At v=1 the cost-volume input equals PixelGaussian's bitwise for the same features; with
    the same weights the whole forward equals PixelGaussian's up to the per-view layout."""
    loc = build_head(PixelGaussian360Loc, num_frames=1, gaussians_per_pixel=g)
    pg = build_head(PixelGaussian, num_frames=1, gaussians_per_pixel=g)
    pg.load_state_dict(loc.state_dict())
    inputs = make_inputs(b=2, v=1, seed=31)
    out_loc, cv_loc = run_live_with_cost_volume_input(loc, inputs)
    out_pg, cv_pg = run_live_with_cost_volume_input(pg, inputs)
    assert cv_loc.dtype == cv_pg.dtype and torch.equal(cv_loc, cv_pg)

    # [|f|^2 / sqrt(c), f, prior depth] of the single view's stage-2 features f
    feat = inputs["img_feats"]["trans_features"][2][:, 0]
    c, (h, w) = feat.shape[1], feat.shape[-2:]
    assert cv_loc.shape == (2, 1 + c + 1, h, w)
    torch.testing.assert_close(cv_loc[:, :1], (feat * feat).sum(1, keepdim=True) / c**0.5)
    assert torch.equal(cv_loc[:, 1:1 + c], feat)
    assert torch.equal(cv_loc[:, 1 + c:], F.interpolate(inputs["depths"][:, 0], size=(h, w), mode="nearest"))

    for key in GAUSSIAN_WIDTHS:
        assert torch.equal(rearrange(out_loc[key], "b v m c -> b (v m) c"), out_pg[key]), key


@pytest.mark.parametrize("v", [2, 3])
def test_pixel_gaussian_360loc_single_view_block_not_entered_for_multi_view(monkeypatch, v):
    head = build_head(PixelGaussian360Loc)

    def entered():
        raise AssertionError("single-view block entered with v >= 2")

    monkeypatch.setattr(head, "_check_single_view_frames", entered)
    inputs = make_inputs(b=2, v=v, seed=20 + v)
    assert_outputs_identical(run_live(head, inputs), run_legacy(head, inputs))


def test_pixel_gaussian_360loc_single_view_needs_num_frames_1():
    """A cost-volume UNet built for 2 views would fold a v=1 batch of 2 into 2 views."""
    torch.manual_seed(0)
    two = PixelGaussian360Loc(image_height=IMAGE_HEIGHT).eval()  # real UNet, num_frames=2
    with pytest.raises(ValueError, match="num_frames=1"):
        run_live(two, make_inputs(b=2, v=1, seed=32))
    torch.manual_seed(0)
    PixelGaussian360Loc(image_height=IMAGE_HEIGHT, num_frames=1)._check_single_view_frames()


def inputs_to(inputs, device):
    out = {k: t.to(device) for k, t in inputs.items() if torch.is_tensor(t)}
    out["img_feats"] = {"trans_features": [f.to(device) for f in inputs["img_feats"]["trans_features"]]}
    return out


@pytest.mark.gpu
def test_pixel_gaussian_360loc_single_view_runs_cuda():
    """v=1 through the real cost-volume UNet (xformers, bf16 attention) built with num_frames=1."""
    torch.manual_seed(0)
    head = PixelGaussian360Loc(image_height=IMAGE_HEIGHT, num_frames=1).cuda().eval()
    out = run_live(head, inputs_to(make_inputs(b=2, v=1, seed=33), "cuda"))
    for key, c in GAUSSIAN_WIDTHS.items():
        assert out[key].shape[:2] == (2, 1) and out[key].shape[-1] == c, key
        assert torch.isfinite(out[key]).all(), key
