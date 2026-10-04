"""D8 render.prune_opacity (model/gaussian.py) and the B5 lazy imports of the
renderer modules and of the legacy nuScenes dataset.

The CUDA rasteriser and the .cuda()-only camera helper are replaced by CPU
stand-ins that record every argument. The frozen f7b20b9 GaussianRenderer.render
(tests/legacy_ref/pixel_ref.py) runs against the same stand-ins, so tau = 0 is
checked to hand the rasteriser exactly the legacy inputs.
"""

import importlib
import sys

import pytest
import torch

import model.gaussian as gaussian_mod
from tests.legacy_ref import pixel_ref

PER_GAUSSIAN = ("means3D", "means2D", "colors_precomp", "opacities", "scales", "rotations")
B, V, N = 2, 3, 64
RESOLUTION = (4, 8)


def fake_settings(**kwargs):
    return dict(kwargs)


def fake_cam_info(c2w, fovx, fovy, znear, zfar):
    w2c = torch.inverse(c2w)
    return w2c, w2c * (fovx + fovy + znear + zfar), c2w[:3, 3]


def make_rasterizer(calls):
    class FakeRasterizer:
        def __init__(self, raster_settings):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
            calls.append((self.raster_settings, kwargs))
            h, w = self.raster_settings["image_height"], self.raster_settings["image_width"]
            n = kwargs["means3D"].shape[0]
            total = sum(kwargs[k].sum() for k in PER_GAUSSIAN)
            image = torch.sigmoid(total) * torch.ones(3, h, w)
            alpha = torch.full((1, h, w), n / 1000.0)
            depth = total * torch.ones(1, h, w)
            return image, None, None, alpha, depth, torch.zeros(n, dtype=torch.int32)

    return FakeRasterizer


@pytest.fixture
def calls(monkeypatch):
    recorded = []
    rasterizer = make_rasterizer(recorded)
    for mod in (gaussian_mod, pixel_ref):
        monkeypatch.setattr(mod, "GaussianRasterizer", rasterizer)
        monkeypatch.setattr(mod, "GaussianRasterizationSettings", fake_settings)
        monkeypatch.setattr(mod, "get_cam_info_gaussian", fake_cam_info)
    monkeypatch.setattr(gaussian_mod, "_prune_opacity", 0.0)  # restored after the test
    return recorded


def make_renderer():
    """A panorama GaussianRenderer without __init__ (which allocates on CUDA)."""
    r = gaussian_mod.GaussianRenderer.__new__(gaussian_mod.GaussianRenderer)
    r.renderer_type = "panorama"
    r.resolution = list(RESOLUTION)
    r.znear, r.zfar = 0.1, 100.0
    r.bg_color = torch.zeros(3)
    return r


def make_scene(seed=0, tau=None):
    gen = torch.Generator().manual_seed(seed)
    gaussians = torch.cat([
        torch.randn(B, N, 3, generator=gen),                                      # xyz
        torch.rand(B, N, 3, generator=gen),                                       # rgb
        torch.rand(B, N, 1, generator=gen),                                       # opacity
        torch.nn.functional.normalize(torch.randn(B, N, 4, generator=gen), dim=-1),  # rotation
        0.1 * torch.rand(B, N, 3, generator=gen),                                 # scale
    ], dim=-1)
    if tau is not None:
        gaussians[:, 0, 6] = tau                                  # on the threshold: kept
        gaussians[:, 1, 6] = torch.nextafter(torch.tensor(tau), torch.tensor(0.0))  # just below: dropped
    c2w = torch.eye(4).repeat(B, V, 1, 1)
    c2w[..., :3, 3] = torch.randn(B, V, 3, generator=gen)
    fovx = 1.0 + torch.rand(B, V, generator=gen)
    return gaussians, c2w, fovx


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    else:
        assert a == b


def test_tau_zero_gives_the_rasteriser_the_legacy_inputs(calls, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prune_by_opacity must not run when prune_opacity == 0")

    monkeypatch.setattr(gaussian_mod, "prune_by_opacity", forbidden)
    assert gaussian_mod.get_prune_opacity() == 0.0
    renderer = make_renderer()
    gaussians, c2w, fovx = make_scene()

    out_live = renderer.render(gaussians, c2w, fovx=fovx)
    live_calls = list(calls)
    calls.clear()
    out_legacy = pixel_ref.LegacyGaussianRenderer.render(renderer, gaussians, c2w, fovx=fovx)

    assert len(live_calls) == len(calls) == B * V
    for (s_live, k_live), (s_legacy, k_legacy) in zip(live_calls, calls):
        assert set(s_live) == set(s_legacy)
        for key in s_live:
            assert_same(s_live[key], s_legacy[key])
        assert set(k_live) == set(k_legacy)
        for key in k_live:
            assert_same(k_live[key], k_legacy[key])
    assert set(out_live) == set(out_legacy)
    for key in out_live:
        assert torch.equal(out_live[key], out_legacy[key]), key


def test_tau_positive_masks_every_per_gaussian_tensor(calls, monkeypatch):
    # The threshold is a process global; restore it so later tests render with the default.
    monkeypatch.setattr(gaussian_mod, "_prune_opacity", gaussian_mod.get_prune_opacity())
    tau = 0.4
    renderer = make_renderer()
    gaussians, c2w, fovx = make_scene(seed=1, tau=tau)
    pixel_ref.LegacyGaussianRenderer.render(renderer, gaussians, c2w, fovx=fovx)
    legacy_calls = list(calls)
    calls.clear()

    gaussian_mod.set_prune_opacity(tau)
    assert gaussian_mod.get_prune_opacity() == tau
    renderer.render(gaussians, c2w, fovx=fovx)

    assert len(calls) == len(legacy_calls) == B * V
    for i, ((s_live, k_live), (s_legacy, k_legacy)) in enumerate(zip(calls, legacy_calls)):
        b = i // V
        keep = gaussians[b, :, 6].float() >= tau
        assert keep[0] and not keep[1] and 0 < int(keep.sum()) < N
        for key in s_live:  # camera settings unchanged
            assert_same(s_live[key], s_legacy[key])
        for key in PER_GAUSSIAN:
            assert k_live[key].shape[0] == int(keep.sum()), key
            assert k_live[key].is_contiguous(), key
            assert torch.equal(k_live[key], k_legacy[key][keep]), key
        assert k_live["shs"] is None and k_live["cov3D_precomp"] is None


def test_prune_by_opacity_uses_one_mask():
    gen = torch.Generator().manual_seed(2)
    opacity = torch.rand(20, 1, generator=gen)
    tensors = [torch.randn(20, c, generator=gen) for c in (3, 3)] + [opacity] + \
              [torch.randn(20, c, generator=gen) for c in (4, 3)]
    keep = opacity[:, 0] >= 0.5
    for got, full in zip(gaussian_mod.prune_by_opacity(0.5, *tensors), tensors):
        assert torch.equal(got, full[keep])


def test_set_prune_opacity_validates(monkeypatch):
    monkeypatch.setattr(gaussian_mod, "_prune_opacity", 0.0)
    for bad in (-0.1, 1.0, float("nan")):
        with pytest.raises(ValueError):
            gaussian_mod.set_prune_opacity(bad)
    assert gaussian_mod.get_prune_opacity() == 0.0
    gaussian_mod.set_prune_opacity(0.005)
    assert gaussian_mod.get_prune_opacity() == 0.005
    gaussian_mod.set_prune_opacity(0)
    assert gaussian_mod.get_prune_opacity() == 0.0


# ------------------------------------------------------------------ B5 lazy imports

def test_twodgaussian_imports_without_the_optional_rasterizers(monkeypatch):
    for name in ("diff_surfel_rasterization", "diff_gaussian_rasterization"):
        monkeypatch.setitem(sys.modules, name, None)  # any import of them now fails
    monkeypatch.delitem(sys.modules, "model.twodgaussian", raising=False)
    mod = importlib.import_module("model.twodgaussian")
    assert mod.GaussianRasterizer is None and mod.ThreeDGaussianRasterizer is None
    with pytest.raises(ImportError):
        mod._load_rasterizers()


def test_nuscenes_dataset_module_imports_without_nuscenes(monkeypatch):
    for name in ("nuscenes", "nuscenes.utils", "nuscenes.utils.geometry_utils"):
        monkeypatch.setitem(sys.modules, name, None)
    for name in ("data.dataloader", "data.transforms.loading"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    mod = importlib.import_module("data.dataloader")
    assert hasattr(mod, "nuScenesDataset")
    assert "data.transforms.loading" not in sys.modules
