"""D4 rotate_gaussians_to_world, switch on, on the volume tail of All, Pan2 and Volume.

The volume head predicts Gaussians in the reference camera frame and the models move
xyz to the world with c2w. With the switch on, the rotation channels 7:11 (wxyz, the
renderer's order) are composed with the same c2w rotation, q_world = q(R_c2w) (x) q_local,
in forward and forward_test; xyz, colour, opacity and the scales stay bitwise the
switch-off values. Runs on CPU with the stubs of tests/test_switches_identity_models.py.

Also covers the Pixel model's side of D4 (i): PixelGaussian512 gets the poses by keyword.
"""

import inspect
import math

import pytest
import torch
from einops import rearrange

from model.pixel import PixelGaussian512
from tests.test_switches_identity_models import (  # noqa: F401  (harness is a fixture)
    H, W, ROT_Y_90, StubPixelGS512, assert_same, harness, make_batch, run,
)

C = math.sqrt(0.5)
KINDS = ("all", "pan2", "volume")
FNS = ("forward", "forward_test")


def volume_gaussians(kind, fn, model, out, v):
    """The volume Gaussians the model produced, as [b, v, n, 14] in the world frame."""
    if fn == "forward":
        g = out[7]                                         # gaussians_volume
        layout = "(b v) n c -> b v n c" if kind == "pan2" else "b (v n) c -> b v n c"
        return rearrange(g, layout, v=v)
    if kind == "all":                                      # preds["gaussian"] = [pixel, volume]
        return rearrange(out[0]["gaussian"][:, v * H * W:], "b (v n) c -> b v n c", v=v)
    if kind == "volume":                                   # preds["gaussian"] = volume
        return rearrange(out[0]["gaussian"], "b (v n) c -> b v n c", v=v)
    # Pan2 returns only its pixel Gaussians: read the fused render's input [(b v), hw + n, 14]
    kind_, gaussians_all = model.renderer.calls[-1]
    assert kind_ == "render"
    return rearrange(gaussians_all[:, H * W:], "(b v) n c -> b v n c", v=v)


def quat_to_matrix(q):
    """wxyz -> rotation matrix, as the rasterizer builds it (pano_gaussian forward.cu computeCov3D)."""
    q = q / q.norm(dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def run_pair(harness, kind, fn, v, input_rotation=None, rotation=None):
    """Volume Gaussians with the switch off and on, same stubs and batch."""
    batch = make_batch(v, input_rotation=input_rotation)
    result = {}
    for on in (False, True):
        model = harness.build(kind, rotate_gaussians_to_world=on)
        model.volume_gs.rotation = rotation
        if fn == "forward":
            model.train()
            out, _ = run(model, batch)
        else:
            model.eval()
            with torch.no_grad():
                out, _ = run(model, batch, fn="forward_test")
        result[on] = volume_gaussians(kind, fn, model, out, v).detach()
    return batch, result[False], result[True]


def assert_only_rotations_change(off, on):
    assert torch.equal(on[..., :7], off[..., :7])      # xyz (same c2w transform), colour, opacity
    assert torch.equal(on[..., 11:], off[..., 11:])    # scales


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fn", FNS)
@pytest.mark.parametrize("v", [1, 2])
def test_identity_c2w_leaves_rotation_unchanged(harness, kind, fn, v):
    # identity rotations, non-zero translations
    _, off, on = run_pair(harness, kind, fn, v, input_rotation=torch.eye(3))
    assert_only_rotations_change(off, on)
    assert torch.equal(on[..., 7:11], off[..., 7:11])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fn", FNS)
@pytest.mark.parametrize("v", [1, 2])
@pytest.mark.parametrize("q_local,expected", [
    # identity in the camera frame -> the camera rotation itself: 90 deg about y = (cos45, 0, sin45, 0)
    ((1.0, 0.0, 0.0, 0.0), (C, 0.0, C, 0.0)),
    # 90 deg about x in the camera frame, then the camera's 90 deg about y:
    # R_y(90) R_x(90) = [[0,1,0],[0,0,-1],[-1,0,0]] -> (1/2, 1/2, 1/2, -1/2)
    ((C, C, 0.0, 0.0), (0.5, 0.5, 0.5, -0.5)),
])
def test_rot_y_90_hand_computed(harness, kind, fn, v, q_local, expected):
    q_local = torch.tensor(q_local)
    _, off, on = run_pair(harness, kind, fn, v, input_rotation=ROT_Y_90, rotation=q_local)
    assert_only_rotations_change(off, on)
    assert torch.equal(off[..., 7:11], q_local.expand_as(off[..., 7:11]))   # released: camera frame
    assert torch.allclose(on[..., 7:11], torch.tensor(expected).expand_as(on[..., 7:11]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fn", FNS)
@pytest.mark.parametrize("v", [1, 2])
def test_general_rotation_follows_c2w(harness, kind, fn, v):
    # rotated and translated input cameras, a different quaternion per Gaussian
    batch, off, on = run_pair(harness, kind, fn, v)
    assert_only_rotations_change(off, on)
    R_c2w = batch["inputs_pix"]["c2w"][:, :, :3, :3]                 # [b, v, 3, 3]
    expected = R_c2w[:, :, None] @ quat_to_matrix(off[..., 7:11])
    assert torch.allclose(quat_to_matrix(on[..., 7:11]), expected, atol=1e-5)
    assert torch.allclose(on[..., 7:11].norm(dim=-1), off[..., 7:11].norm(dim=-1), atol=1e-6)
    assert not torch.allclose(on[..., 7:11], off[..., 7:11])


def test_pixel_model_accepts_switch(harness):
    # no volume tail in the Pixel model: its pixel head gets its own copy of the switch
    on, off = harness.build("pixel", rotate_gaussians_to_world=True), harness.build("pixel")
    batch = make_batch(2)
    out_on, _ = run(on, batch)
    out_off, _ = run(off, batch)
    assert_same(out_on, out_off, "pixel forward")


@pytest.mark.parametrize("fn", FNS)
def test_pixel_model_passes_poses_to_pixel_gaussian_512_by_keyword(harness, fn):
    head = StubPixelGS512()
    model = harness.build("pixel", pixel_gs=head)
    batch = make_batch(2)
    run(model, batch, fn=fn)
    call = head.calls[-1]
    assert torch.equal(call["extrinsics_in"], batch["inputs_pix"]["c2w"])
    assert call["patch_idx"] == 0                   # released callers put c2ws into this slot
    assert call["status"] == ("test" if fn == "forward_test" else "train")


def test_pixel_gaussian_512_takes_extrinsics_keyword_only():
    """The PixelGaussian512 interface the Pixel model relies on (plan D4 (i))."""
    params = inspect.signature(PixelGaussian512.forward).parameters
    assert params["extrinsics_in"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["extrinsics_in"].default is None
    assert list(params)[8] == "patch_idx"
