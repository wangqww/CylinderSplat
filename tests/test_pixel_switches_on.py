"""Pixel-head switches turned ON (CPU; the cost-volume UNet is stubbed as in
tests/test_switches_identity_pixel.py).

D4 rotate_gaussians_to_world: identity c2w leaves the quaternions unchanged; a 90 deg
    rotation about y gives the hand-computed wxyz result; the composed rotation matrix
    is R_c2w @ R(q_local); through the whole forward only the rotation channels change.
D9 pixel_depth_sampling="nearest": only the prior-depth grid_sample of each stage
    switches to mode="nearest"; with that call forced back to bilinear the forward
    equals the bilinear head bitwise.
Both are also run through PixelGaussian360Loc's single-view (v=1) path.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from model.pixel.pixel_gs import PixelGaussian
from model.pixel.pixel_gs_360loc import PixelGaussian360Loc
from model.pixel.pixel_gs_512 import PixelGaussian512
from model.utils.quaternion import compose_quaternion_c2w
from tests.test_switches_identity_pixel import (
    build_head, make_inputs, random_c2w, run_live, synthetic_gaussian_parts)

HEADS = [PixelGaussian, PixelGaussian360Loc, PixelGaussian512]
S = math.sqrt(0.5)
R_Y90 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def quat_to_matrix(q):
    """wxyz -> rotation matrix, the rasterizer's formula (gaussian.build_rotation)."""
    q = F.normalize(q, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)], -1),
        torch.stack([2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)], -1),
        torch.stack([2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def by_view(t, cls, v):
    """Head layout -> [B, V, M, C] (PixelGaussian360Loc already is per view)."""
    return t if cls is PixelGaussian360Loc else rearrange(t, "b (v m) c -> b v m c", v=v)


def head_layout(t, cls):
    """[B, V, M, C] -> the layout the head concatenates."""
    return t if cls is PixelGaussian360Loc else rearrange(t, "b v m c -> b (v m) c")


@pytest.fixture(scope="module")
def step_heads():
    return {cls: (build_head(cls), build_head(cls, rotate_gaussians_to_world=True)) for cls in HEADS}


# ------------------------------------------------------------ D4 on the concat step

@pytest.mark.parametrize("v", [1, 2])
@pytest.mark.parametrize("cls", HEADS)
def test_d4_identity_c2w_leaves_quaternions_unchanged(step_heads, cls, v):
    off, on = step_heads[cls]
    gen = torch.Generator().manual_seed(3)
    parts = synthetic_gaussian_parts(cls, 2, v, 17, gen)
    c2w = random_c2w(2, v, gen, identity_rotation=True)  # translation only
    assert torch.equal(on._assemble_gaussians(*parts, c2w), off._assemble_gaussians(*parts, c2w))


@pytest.mark.parametrize("v", [1, 2])
@pytest.mark.parametrize("cls", HEADS)
def test_d4_rotation_90_about_y_hand_computed(step_heads, cls, v):
    off, on = step_heads[cls]
    b, m = 2, 4
    # local quaternions: identity, 90 deg about x, alternating
    q_local = torch.tensor([[1.0, 0.0, 0.0, 0.0], [S, S, 0.0, 0.0]]).repeat(m // 2, 1)
    q_local = q_local.expand(b, v, m, 4).contiguous()
    # view 0 is rotated 90 deg about y, the other views are not rotated
    c2w = torch.eye(4).repeat(b, v, 1, 1)
    c2w[:, 0, :3, :3] = R_Y90
    c2w[..., :3, 3] = torch.tensor([0.3, -1.0, 2.0])
    gen = torch.Generator().manual_seed(4)
    means, rgbs, opacities, _, scales = synthetic_gaussian_parts(cls, b, v, m, gen)
    parts = (means, rgbs, opacities, head_layout(q_local, cls), scales)

    out_on = on._assemble_gaussians(*parts, c2w)
    out_off = off._assemble_gaussians(*parts, c2w)

    expected = q_local.clone()
    expected[:, 0, 0::2] = torch.tensor([S, 0.0, S, 0.0])       # q_y90 (x) identity
    expected[:, 0, 1::2] = torch.tensor([0.5, 0.5, 0.5, -0.5])  # q_y90 (x) q_x90
    torch.testing.assert_close(by_view(out_on[..., 7:11], cls, v), expected, rtol=0, atol=1e-6)
    # R(q_world) = R_y90 @ R(q_local)
    torch.testing.assert_close(quat_to_matrix(by_view(out_on[..., 7:11], cls, v)[:, 0]),
                               R_Y90 @ quat_to_matrix(q_local[:, 0]), rtol=0, atol=1e-6)
    # means, colours, opacities and scales are untouched
    assert torch.equal(out_on[..., :7], out_off[..., :7])
    assert torch.equal(out_on[..., 11:], out_off[..., 11:])


@pytest.mark.parametrize("cls", HEADS)
def test_d4_composition_matches_rotation_matrices(step_heads, cls):
    _, on = step_heads[cls]
    gen = torch.Generator().manual_seed(5)
    b, v, m = 2, 3, 11
    parts = synthetic_gaussian_parts(cls, b, v, m, gen)
    c2w = random_c2w(b, v, gen)
    out = by_view(on._assemble_gaussians(*parts, c2w)[..., 7:11], cls, v)
    q_local = by_view(parts[3], cls, v)
    torch.testing.assert_close(quat_to_matrix(out), c2w[:, :, None, :3, :3] @ quat_to_matrix(q_local),
                               rtol=0, atol=1e-5)
    torch.testing.assert_close(out.norm(dim=-1), q_local.norm(dim=-1), rtol=0, atol=1e-6)


# ------------------------------------------------------------- D4 through the forward

def forward_cases():
    return ([(PixelGaussian, v, g) for v in (1, 2) for g in (1, 3)]
            + [(PixelGaussian360Loc, 1, 1), (PixelGaussian360Loc, 2, 1), (PixelGaussian360Loc, 3, 2)]
            + [(PixelGaussian512, v, None) for v in (1, 2)])


def build(cls, g, **kwargs):
    if g is not None:
        kwargs["gaussians_per_pixel"] = g
    return build_head(cls, **kwargs)


def live(head, inputs):
    if isinstance(head, PixelGaussian512):
        return run_live(head, inputs, extrinsics_in=inputs["c2w"])
    return run_live(head, inputs)


@pytest.mark.parametrize("cls,v,g", forward_cases())
def test_d4_forward_changes_only_rotations(cls, v, g):
    inputs = make_inputs(b=2, v=v, seed=50 + v)
    out_off = live(build(cls, g), inputs)
    out_on = live(build(cls, g, rotate_gaussians_to_world=True), inputs)
    rot = inputs["c2w"][..., :3, :3]
    for s_off, s_on in zip(out_off["stages"], out_on["stages"]):
        g_off, g_on = s_off["gaussians"], s_on["gaussians"]
        assert torch.equal(g_on[..., :7], g_off[..., :7])      # xyz, rgb, opacity
        assert torch.equal(g_on[..., 11:], g_off[..., 11:])    # scales
        expected = compose_quaternion_c2w(by_view(g_off[..., 7:11], cls, v), rot)
        torch.testing.assert_close(by_view(g_on[..., 7:11], cls, v), expected)
        assert not torch.allclose(g_on[..., 7:11], g_off[..., 7:11])
        for k in s_off:
            if k != "gaussians":
                assert torch.equal(s_on[k], s_off[k]), k
    for k in out_off:
        if k not in ("stages", "gaussians"):
            assert torch.equal(out_on[k], out_off[k]), k


@pytest.mark.parametrize("cls,v,g", forward_cases())
def test_d4_forward_identity_c2w_is_bitwise_off(cls, v, g):
    inputs = make_inputs(b=2, v=v, seed=60 + v, identity_rotation=True)
    out_off = live(build(cls, g), inputs)
    out_on = live(build(cls, g, rotate_gaussians_to_world=True), inputs)
    for k in out_off:
        if k != "stages":
            assert torch.equal(out_on[k], out_off[k]), k


def test_d4_pixel_gaussian_512_needs_extrinsics():
    head = build_head(PixelGaussian512, rotate_gaussians_to_world=True)
    inputs = make_inputs(b=1, v=1, seed=70)
    with pytest.raises(ValueError, match="extrinsics_in"):
        run_live(head, inputs)


# ------------------------------------------------------------------------------- D9

@pytest.mark.parametrize("cls", HEADS)
def test_d9_invalid_mode_rejected(cls):
    with pytest.raises(ValueError, match="pixel_depth_sampling"):
        cls(image_height=16, pixel_depth_sampling="bicubic")


@pytest.mark.parametrize("cls", HEADS)
def test_d9_sampler_nearest(cls):
    head = build_head(cls, pixel_depth_sampling="nearest")
    gen = torch.Generator().manual_seed(6)
    depths = 0.5 + 4.5 * torch.rand(4, 1, 16, 32, generator=gen)
    grid = torch.rand(4, 101, 1, 2, generator=gen) * 2 - 1
    out = head._sample_prior_depth(depths, grid)
    assert torch.equal(out, F.grid_sample(depths, grid, mode="nearest", padding_mode="border"))
    assert not torch.equal(out, F.grid_sample(depths, grid, padding_mode="border"))


def spy_grid_sample(monkeypatch):
    calls = []
    real = F.grid_sample

    def spy(input, grid, *args, **kwargs):
        calls.append((kwargs.get("mode", args[0] if args else "bilinear"), input))
        return real(input, grid, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "grid_sample", spy)
    return calls


@pytest.mark.parametrize("v", [1, 2])
@pytest.mark.parametrize("cls", HEADS)
def test_d9_only_prior_depth_sampling_is_nearest(monkeypatch, cls, v):
    inputs = make_inputs(b=2, v=v, seed=80 + v)
    depths_fullres = rearrange(inputs["depths"], "b v ... -> (b v) ...")

    calls = spy_grid_sample(monkeypatch)
    live(build_head(cls), inputs)
    n_bilinear = len(calls)
    assert all(mode == "bilinear" for mode, _ in calls)

    calls.clear()
    live(build_head(cls, pixel_depth_sampling="nearest"), inputs)
    assert len(calls) == n_bilinear
    nearest = [inp for mode, inp in calls if mode == "nearest"]
    assert len(nearest) == 4  # one prior-depth read per Gaussian stage
    assert all(torch.equal(inp, depths_fullres) for inp in nearest)
    assert all(mode in ("bilinear", "nearest") for mode, _ in calls)


@pytest.mark.parametrize("cls,v,g", forward_cases())
def test_d9_nearest_differs_only_at_depth_sampling(monkeypatch, cls, v, g):
    inputs = make_inputs(b=2, v=v, seed=90 + v)
    out_bilinear = live(build(cls, g), inputs)
    near = build(cls, g, pixel_depth_sampling="nearest")
    out_nearest = live(near, inputs)

    # depth only moves the means and the depth-scaled scales
    for s_b, s_n in zip(out_bilinear["stages"], out_nearest["stages"]):
        assert torch.equal(s_n["gaussians"][..., 3:11], s_b["gaussians"][..., 3:11])
        for k in s_b:
            if k != "gaussians":
                assert torch.equal(s_n[k], s_b[k]), k
    assert not torch.equal(out_nearest["gaussians"][..., :3], out_bilinear["gaussians"][..., :3])

    # forcing that one call back to bilinear gives the bilinear head bitwise
    monkeypatch.setattr(near, "_sample_prior_depth",
                        lambda d, grid: F.grid_sample(d, grid, padding_mode="border"))
    forced = live(near, inputs)
    for k in out_bilinear:
        if k != "stages":
            assert torch.equal(forced[k], out_bilinear[k]), k
