"""prune_invisible on the CUDA panorama rasteriser (skipped without CUDA / pano_gaussian): dropping the Gaussians with
opacity < 1/255 leaves image, alpha and depth bit-identical, and the kept Gaussians' gradients equal up to the order of
the backward's atomic adds; the dropped ones never get a gradient."""

import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("pano_gaussian")

H, W = 128, 256


def scene(n=20000, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    direction = torch.nn.functional.normalize(torch.randn(n, 3, device="cuda", generator=gen), dim=-1)
    xyz = direction * (1.0 + 4.0 * torch.rand(n, 1, device="cuda", generator=gen))
    rgb = torch.rand(n, 3, device="cuda", generator=gen)
    opacity = torch.rand(n, 1, device="cuda", generator=gen)
    opacity[: n // 3] *= 1.0 / 255.0 * 0.999  # a third below the threshold
    opacity[n // 3 : n // 3 + 50] = 1.0 / 255.0  # exactly at it (kept)
    quat = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda", generator=gen), dim=-1)
    scale = 0.01 + 0.08 * torch.rand(n, 3, device="cuda", generator=gen)
    return torch.cat([xyz, rgb, opacity, quat, scale], dim=-1)[None]


def render(prune, gaussians):
    from model.gaussian import GaussianRenderer

    r = GaussianRenderer("cuda", resolution=[H, W], znear=0.1, zfar=15.0, prune_invisible=prune)
    fov = torch.full((1, 1), math.pi / 2, device="cuda")
    return r.render(gaussians=gaussians, c2w=torch.eye(4, device="cuda")[None, None], fovx=fov, fovy=fov)


def test_renders_are_bit_identical_and_fewer_gaussians_are_rasterised(monkeypatch):
    import model.gaussian as gaussian_module

    counts = []
    original = gaussian_module.GaussianRasterizer.forward

    def counting(self, means3D, *args, **kwargs):
        counts.append(means3D.shape[0])
        return original(self, means3D, *args, **kwargs)

    monkeypatch.setattr(gaussian_module.GaussianRasterizer, "forward", counting)
    g = scene()
    plain, pruned = render(False, g), render(True, g)
    for key in ("image", "alpha", "depth"):
        assert torch.equal(plain[key], pruned[key]), key
    visible = int((g[0, :, 6] >= torch.tensor(1.0 / 255.0, dtype=torch.float32, device="cuda")).sum())
    assert counts == [g.shape[1], visible] and visible < g.shape[1]


def test_gradients_of_kept_gaussians_match_and_dropped_ones_are_zero():
    # The backward accumulates with atomicAdd, so two identical runs already differ (measured on 4090: max abs 1e-2 on
    # a few of ~2e5 entries). Pruned vs plain must stay within that run-to-run spread; dropped Gaussians get exactly 0.
    g0 = scene(seed=1)
    grads = []
    for prune in (False, False, True):
        g = g0.clone().requires_grad_(True)
        out = render(prune, g)
        (out["image"].sum() + out["depth"].sum() + out["alpha"].sum()).backward()
        grads.append(g.grad[0])
    dropped = g0[0, :, 6] < torch.tensor(1.0 / 255.0, dtype=torch.float32, device="cuda")
    assert dropped.sum() > 0
    assert all(torch.count_nonzero(gr[dropped]) == 0 for gr in grads)
    run_to_run = (grads[0] - grads[1])[~dropped].abs()
    pruned = (grads[0] - grads[2])[~dropped].abs()
    assert pruned.max() <= 4 * run_to_run.max() + 1e-6
    loose = 1e-4 * grads[0][~dropped].abs() + 1e-7
    assert (pruned > loose).sum() <= 4 * (run_to_run > loose).sum() + 20
