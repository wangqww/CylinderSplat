"""The LS switches on CPU: sampling_align (pixel-branch warp and volume retrieval), the latitude weights of ws_loss,
and the row-weighted LPIPS average."""

import math
import types

import pytest

torch = pytest.importorskip("torch")


def _pixel_gs():
    pytest.importorskip("mmengine")
    from model.pixel import pixel_gs

    return pixel_gs


def _decoder_cls():
    pytest.importorskip("mmengine")
    from model.volume.volume_gs_decoder_cylinder import VolumeGaussianDecoderCylinder

    return VolumeGaussianDecoderCylinder


# ----------------------------------------------------------------------------- pixel-branch warp


@pytest.mark.parametrize("h, w", [(8, 16), (16, 32)])
def test_aligned_warp_with_identity_pose_returns_the_feature_map(h, w):
    pixel_gs = _pixel_gs()
    gen = torch.Generator().manual_seed(0)
    feature = torch.rand(2, 5, h, w, generator=gen)
    pose = torch.eye(4).repeat(2, 1, 1)
    depth = 1.0 + torch.rand(2, 1, h, w, generator=gen)
    aligned = pixel_gs.warp_with_pose_depth_candidates(feature, pose, depth, sampling_align=True)
    assert aligned.shape == (2, 5, 1, h, w)
    assert torch.allclose(aligned[:, :, 0], feature, atol=1e-5)
    # The released warp (align_corners=True on pixel-centre coordinates) is off by up to half a pixel.
    released = pixel_gs.warp_with_pose_depth_candidates(feature, pose, depth)
    assert (released[:, :, 0] - feature).abs().max() > 1e-2


def test_aligned_warp_wraps_in_longitude():
    pixel_gs = _pixel_gs()
    h, w = 4, 8
    feature = torch.rand(1, 3, h, w, generator=torch.Generator().manual_seed(4)) + 0.5
    angle = 2 * math.pi / w  # a yaw of exactly one pixel
    pose = torch.eye(4)[None]
    pose[0, 0, 0], pose[0, 0, 2], pose[0, 2, 0], pose[0, 2, 2] = (
        math.cos(angle), math.sin(angle), -math.sin(angle), math.cos(angle)
    )
    depth = torch.ones(1, 1, h, w)
    out = pixel_gs.warp_with_pose_depth_candidates(feature, pose, depth, sampling_align=True)[:, :, 0]
    # R_y(+angle) maps longitude phi to phi + angle, so pixel j samples column j + 1: a pure roll by -1, including
    # the last column, which reads column 0 through the wrap (not zero padding)
    assert torch.allclose(out, torch.roll(feature, -1, dims=-1), atol=1e-4)


# ----------------------------------------------------------------------------- volume retrieval


def _sampler(sampling_align):
    cls = _decoder_cls()
    stub = types.SimpleNamespace(sampling_align=sampling_align)
    stub.normalize = types.MethodType(cls.normalize, stub)
    return types.MethodType(cls.sample_source, stub)


def test_aligned_retrieval_reads_pixel_centres_exactly():
    h, w = 6, 12
    src = torch.rand(1, 3, h, w, generator=torch.Generator().manual_seed(1))
    rows, cols = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    locs = torch.stack([cols, rows], dim=-1).view(1, 1, -1, 2)
    out = _sampler(True)(src, locs, h, w, pad=2).view(1, 3, h, w)
    assert torch.allclose(out, src, atol=1e-6)
    released = _sampler(False)(src, locs, h, w, pad=2).view(1, 3, h, w)
    assert (released - src).abs().max() > 1e-2  # u / (w - 1) drifts by up to a pixel


def test_aligned_retrieval_wraps_at_the_seam():
    h, w = 4, 8
    src = torch.arange(w, dtype=torch.float32).repeat(1, 1, h, 1)
    locs = torch.tensor([[w - 0.25, 1.5], [0.25, 1.5], [-0.5, 1.5]]).view(1, 1, 3, 2)
    out = _sampler(True)(src, locs, h, w, pad=2).flatten()
    # u = w - 0.25: 3/4 of the last column, 1/4 of column 0; u = 0.25: the mirror; u = -0.5: the last column
    assert torch.allclose(out, torch.tensor([0.75 * (w - 1), 0.25 * (w - 1), w - 1.0]), atol=1e-5)


# ----------------------------------------------------------------------------- latitude weights


def test_row_weighted_lpips_average():
    pytest.importorskip("torchvision")
    from model.losses import row_weighted_average, spatial_average

    const = torch.full((2, 1, 8, 16), 3.0)
    assert torch.allclose(row_weighted_average(const), spatial_average(const))
    x = torch.rand(1, 1, 8, 4, generator=torch.Generator().manual_seed(2))
    w = torch.sin((torch.arange(8) + 0.5) * math.pi / 8).view(1, 1, 8, 1)
    expected = (x * w).sum() / (w.expand_as(x)).sum()
    assert torch.allclose(row_weighted_average(x).flatten(), expected.flatten(), atol=1e-6)


def test_ws_mean_matches_the_metric_weights():
    pytest.importorskip("mmengine")
    import numpy as np

    from model.omni_gs_cylinder_all import OmniGaussianCylinderAll
    from tools.metrics import WSPSNR

    stub = types.SimpleNamespace(_row_weights=OmniGaussianCylinderAll._row_weights)
    ws_mean = types.MethodType(OmniGaussianCylinderAll._ws_mean, stub)
    err = torch.rand(2, 3, 3, 16, 32, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    w = torch.tensor(WSPSNR().get_weights(16, 32))
    expected = (err * w).sum() / (w.expand_as(err)).sum()
    assert torch.allclose(ws_mean(err), expected)
    assert np.isclose(float(ws_mean(torch.full((1, 4, 8), 2.0))), 2.0)
