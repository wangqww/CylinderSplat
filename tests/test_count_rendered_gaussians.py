"""tools/count_rendered_gaussians.py: the count is taken from forward_test's output at the renderer's threshold (CPU)."""

import pytest

torch = pytest.importorskip("torch")

from tools import count_rendered_gaussians as crg  # noqa: E402


def gaussians_with_opacity(values):
    g = torch.zeros(len(values), 14)
    g[:, 6] = torch.tensor(values, dtype=torch.float32)
    return g


def test_threshold_is_the_renderers_float32_comparison():
    cut = torch.tensor(1.0 / 255.0, dtype=torch.float32)
    below = torch.nextafter(cut, torch.tensor(0.0)).item()
    assert crg.rendered(gaussians_with_opacity([0.0, below, cut.item(), 0.5])) == 2
    # a float16 opacity is compared after the cast to float32, like the renderer
    assert crg.rendered(gaussians_with_opacity([0.5, 0.0]).half()) == 1


class FakeModel:
    """forward_test returns the given Gaussians of each batch."""

    def __init__(self, gaussians):
        self.gaussians = gaussians

    def forward_test(self, batch):
        return {"gaussian": self.gaussians[batch]}, None


def test_counts_come_from_the_forward_test_output():
    a = gaussians_with_opacity([0.5, 0.0, 0.0, 0.2]).unsqueeze(0)  # two below 1/255
    b = gaussians_with_opacity([0.5, 0.5, 0.5, 0.0]).unsqueeze(0)
    counts, total = crg.count(FakeModel([a, b]), [0, 1])
    assert counts == [2, 3] and total == 4
    assert crg.count(FakeModel([a, b]), [0, 1], max_batches=1)[0] == [2]


def test_batches_of_more_than_one_sample_are_refused():
    two = gaussians_with_opacity([0.5, 0.5]).unsqueeze(0).repeat(2, 1, 1)
    with pytest.raises(ValueError, match="batch size 1"):
        crg.count(FakeModel([two]), [0])


class FakeJoint(torch.nn.Module):
    """pixel_gs / volume_gs submodules; forward_test concatenates the pixel and the volume Gaussians."""

    def __init__(self, pixel, volume):
        super().__init__()
        self.pixel_gs = _Const({"gaussians": pixel})
        self.volume_gs = _Const(volume)

    def forward_test(self, batch):
        pixel = self.pixel_gs()["gaussians"]
        volume = self.volume_gs()
        return {"gaussian": torch.cat([pixel, volume.reshape(1, -1, 14)], dim=1)}, None


class _Const(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        return self.value


def test_extras_split_pixel_and_volume():
    pixel = gaussians_with_opacity([0.5, 0.0, 0.5]).unsqueeze(0)  # 2 rendered pixel Gaussians
    volume = torch.zeros(2, 4, 14)  # two cylinders x 4 slots
    volume[..., 6] = torch.tensor([[0.5, 0.0, 0.5, 0.0], [0.5, 0.5, 0.5, 0.0]])  # 5 of 8 rendered
    extras = {}
    counts, total = crg.count(FakeJoint(pixel, volume), [0], extras=extras)
    assert counts == [7] and total == 11
    assert extras == {"pixel": [2], "volume": [5]}
    assert crg.count(FakeJoint(pixel, volume), [0])[0] == [7]  # no extras dict: the plain count, no hooks left behind
