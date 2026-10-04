"""evaluate.py vs the frozen legacy evaluation loops (tests/legacy_ref/eval_ref.py), on CPU.

A synthetic fixture of per-view predictions and targets (tiny images, every scene
key, batches that mix scenes, a short last batch) goes through the verbatim legacy
loop with save_vis ON and through evaluate.evaluate_loop with save_vis OFF. The
per-batch and per-scene lines must be identical (WS-PSNR relabelled `wspsnr` where
the legacy script called it `psnr` / `ws_psnr`), the per-scene values and totals
bitwise equal, and the Total line correctly labelled. LPIPS is stubbed (no VGG
download); SSIM (skimage), WS-PSNR, PCC and the depth metrics are the real ones.
The legacy loops use the frozen pre-cache WSPSNR, the new path the cached one.
"""

import functools
import importlib.util
import logging
import os
import re
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.legacy_ref import eval_ref  # noqa: E402
from tools import metrics  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H, W = 16, 32
GLOBAL_ITER = 48000


def _load_evaluate():
    # By path: a site-packages `evaluate` (Hugging Face) must not shadow the repo script.
    spec = importlib.util.spec_from_file_location("cylindersplat_evaluate", os.path.join(REPO_ROOT, "evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load_evaluate()


class _Cuda0(torch.Tensor):
    """CPU tensor that reports cuda:0, so the verbatim '[Eval] Batch %d-%d' line
    (which prints `tensor.device.index`, None on CPU) runs in the legacy copy."""

    @property
    def device(self):
        return torch.device("cuda", 0)


def _fake_lpips(ground_truth, predicted):
    return (ground_truth - predicted).abs().mean(dim=(1, 2, 3))


@pytest.fixture(autouse=True)
def _patches(monkeypatch):
    monkeypatch.setattr(eval_ref, "compute_lpips", _fake_lpips)
    monkeypatch.setattr(ev, "compute_lpips", _fake_lpips)
    monkeypatch.setattr(eval_ref, "compute_psnr",
                        lambda gt, pred: metrics.compute_psnr(gt, pred).as_subclass(_Cuda0))

    class _LegacyWSPSNR(eval_ref.WSPSNR):
        def ws_psnr(self, y_pred, y_true, max_val=1.0):
            return super().ws_psnr(y_pred, y_true, max_val=max_val).as_subclass(_Cuda0)

    monkeypatch.setattr(eval_ref, "WSPSNR", _LegacyWSPSNR)
    import matplotlib
    import matplotlib.cm
    if not hasattr(matplotlib.cm, "get_cmap"):  # removed in matplotlib 3.9; legacy vis uses it
        monkeypatch.setattr(matplotlib.cm, "get_cmap", matplotlib.colormaps.get_cmap, raising=False)


# ----------------------------------------------------------------------------
# fixture
# ----------------------------------------------------------------------------

def _scene_list(entry):
    keys = entry["scene_keys"]
    if keys is None:
        return [None] * 7
    if len(keys) == 1:
        return [keys[0]] * 5
    counts = [3, 2, 2, 3, 1, 2, 2]  # 15 samples: batches of 2 mix scenes, the last has 1
    return [key for key, n in zip(keys, counts) for _ in range(n)]


def make_batches(entry, seed=0, bs=2):
    g = torch.Generator().manual_seed(seed)
    n_views = len(entry["target_views"])
    scenes = _scene_list(entry)
    batches, outputs = [], []
    for i, start in enumerate(range(0, len(scenes), bs)):
        names = scenes[start:start + bs]
        b = len(names)
        gt_img = torch.rand(b, n_views, 3, H, W, generator=g)
        pred_img = gt_img + 0.08 * torch.randn(b, n_views, 3, H, W, generator=g)  # partly outside [0, 1]
        prior = 0.5 + 4.0 * torch.rand(b, n_views, 1, H, W, generator=g)
        pred_depth = (prior[:, :, 0] * (1.0 + 0.2 * torch.randn(b, n_views, H, W, generator=g))).abs() + 0.05
        gts = {"img": gt_img, "depth": prior, "depth_m": prior}
        if entry["depth_metrics"]:
            depth_gt = 0.2 + 6.0 * torch.rand(b, n_views, 1, H, W, generator=g)
            for j, name in enumerate(names):
                if not name.startswith("m3d"):  # the loaders give zeros without GT depth
                    depth_gt[j] = 0.0
            gts["depth_gt"] = depth_gt
            gts["mask_gt"] = (depth_gt > 0.45) & (depth_gt < 10.0)
        preds = {"img": pred_img, "depth": pred_depth, "gaussian": torch.rand(b, 20, 14, generator=g)}
        batch = {"_i": i}
        if entry["scene_keys"] is not None:
            batch["scene"] = list(names)
        batches.append(batch)
        outputs.append((preds, gts))
    return batches, outputs


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs
        self.benchmarker = types.SimpleNamespace(execution_times={"render": [0.25, 0.5]})

    def eval(self):
        return self

    def forward_test(self, batch):
        return self.outputs[batch["_i"]]


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def run_legacy(legacy_fn, batches, outputs, out_dir, save_vis=True):
    logger = logging.getLogger(f"legacy_eval_{id(batches)}")
    logger.propagate = False
    handler = ListHandler()
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    cfg = types.SimpleNamespace(print_freq=100, output_dir=str(out_dir),
                                eval_args=types.SimpleNamespace(save_vis=save_vis, save_ply=False))
    accelerator = types.SimpleNamespace(gather_for_metrics=lambda x: x)
    ns = legacy_fn(cfg, FakeModel(outputs), batches, accelerator, logger, GLOBAL_ITER)
    return handler.messages, ns


def run_new(entry, batches, outputs, **kwargs):
    messages = []
    results = ev.evaluate_loop(entry, FakeModel(outputs), batches, lambda x: x, messages.append, **kwargs)
    return messages, results


def relabel(line, legacy_labels):
    for new, old in legacy_labels.items():
        line = re.sub(r"(?<![\w])" + re.escape(old) + r":", new + ":", line)
    return line


def _kind(message, entry):
    if message.startswith("[Eval] Batch"):
        return "batch"
    if message.startswith("Finish evluation"):
        return "total"
    if entry["scene_keys"] is not None and any(message.startswith(f" {s} ") for s in entry["scene_keys"]):
        return "extra" if message.split(" ")[2] == "extra" else "scene"
    if " calls, avg. " in message:
        return "timing"
    return "other"


def _by_kind(messages, entry, legacy_labels=None):
    out = {"batch": [], "scene": [], "extra": [], "total": [], "timing": [], "other": []}
    for message in messages:
        out[_kind(message, entry)].append(relabel(message, legacy_labels) if legacy_labels else message)
    return out


def _legacy_scene_means(scene_res, legacy_names):
    # The legacy per-scene formula applied to the legacy per-sample records.
    means = {}
    for scene, res in scene_res.items():
        sums = {k: 0 for k in legacy_names}
        for m in res:
            for k in legacy_names:
                sums[k] = sums[k] + m[k].item()
        means[scene] = {k: sums[k] / len(res) for k in legacy_names}
    return means


# new metric name -> (legacy per-scene record key, legacy total variable)
LEGACY_NAMES = {
    "mp3d_double_256": {k: (k, "total_" + k) for k in ev.KNOWN_METRICS},
    "mp3d_single_256": {k: (k, "total_" + k) for k in ev.KNOWN_METRICS},
    "mp3d_double_512": {"wspsnr": ("psnr", "total_psnr"), "ssim": ("ssim", "total_ssim"),
                        "lpips": ("lpips", "total_lpips"), "pcc": (None, "total_pcc")},
    "loc360_double_256": {"psnr": (None, "total_psnr"), "wspsnr": (None, "total_ws_psnr"),
                          "ssim": (None, "total_ssim"), "lpips": (None, "total_lpips"), "pcc": (None, "total_pcc")},
    "vigor_double": {k: (k, "total_" + k) for k in ("psnr", "wspsnr", "ssim", "lpips", "pcc")},
}
LEGACY_LOOPS = {
    "mp3d_double_256": eval_ref.legacy_eval_mp3d_double_256,
    "mp3d_single_256": eval_ref.legacy_eval_mp3d_double_256,  # the single script differs only in its loader
    "mp3d_double_512": eval_ref.legacy_eval_mp3d_double_512,
    "loc360_double_256": eval_ref.legacy_eval_360Loc_double_256,
    "vigor_double": eval_ref.legacy_eval_VIGOR,
}


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", sorted(LEGACY_LOOPS))
def test_new_loop_matches_legacy(dataset, tmp_path):
    entry = ev.EVAL_ENTRIES[dataset]
    ev.check_entry(dataset, entry)
    batches, outputs = make_batches(entry)
    legacy_msgs, ns = run_legacy(LEGACY_LOOPS[dataset], batches, outputs, tmp_path / "legacy")
    new_msgs, results = run_new(entry, batches, outputs)  # save_vis off
    legacy = _by_kind(legacy_msgs, entry, entry["legacy_labels"])
    new = _by_kind(new_msgs, entry)

    # the legacy copy really ran its save_vis branch; the new path wrote nothing
    assert len(list((tmp_path / "legacy" / str(GLOBAL_ITER)).glob("*.png"))) == sum(
        o[0]["img"].shape[0] for o in outputs)

    # per-batch lines
    assert len(new["batch"]) == len(batches)
    assert new["batch"] == legacy["batch"]
    assert new["timing"] == legacy["timing"]

    names = LEGACY_NAMES[dataset]
    # per-scene lines: one non-empty row per scene key, identical to legacy
    if entry["scene_keys"] is not None:
        assert [line.split(" ")[1] for line in new["scene"]] == list(entry["scene_keys"])
        assert new["scene"] == legacy["scene"]
        legacy_means = _legacy_scene_means(ns["scene_res"], [names[k][0] for k in entry["scene_metrics"]])
        for scene in entry["scene_keys"]:
            row = results["scenes"][scene]
            assert row["num_samples"] > 0
            assert row["line"] in new["scene"]
            for k in entry["scene_metrics"]:
                assert row["metrics"][k] == legacy_means[scene][names[k][0]]  # bitwise
    else:
        assert results["scenes"] == {} and new["scene"] == []

    # totals: bitwise equal to the legacy totals
    n = len(batches)
    for k in entry["batch_metrics"]:
        legacy_total = ns[names[k][1]]
        assert results["total"]["metrics"][k] == legacy_total.item() / n
    # the Total line prints every total under its own label
    tail = new["total"][0].split(" s). ", 1)[1]
    pairs = re.findall(r"(\w+): (-?\d+\.\d+|nan|inf)", tail)
    assert [k for k, _ in pairs] == list(entry["total_metrics"])
    for k, printed in pairs:
        precision = 3 if k in ("psnr", "wspsnr") else 4
        assert printed == f"{results['total']['metrics'][k]:.{precision}f}"
    if entry["total_line"] == ev.EVAL_ENTRIES["mp3d_double_256"]["total_line"]:
        # legacy bug: every value from wspsnr on under the previous label, depthsim dropped
        legacy_tail = legacy["total"][0].split(" s). ", 1)[1]
        assert f"ssim: {ns['total_wspsnr'].item() / n:.4f}," in legacy_tail
        assert "wspsnr" not in legacy_tail
    else:
        assert tail == legacy["total"][0].split(" s). ", 1)[1]


@pytest.mark.parametrize("dataset", ["mp3d_double_256", "vigor_double"])
def test_save_vis_does_not_change_numbers(dataset, tmp_path):
    entry = ev.EVAL_ENTRIES[dataset]
    batches, outputs = make_batches(entry, seed=1)
    off_msgs, off = run_new(entry, batches, outputs)
    on_msgs, on = run_new(entry, batches, outputs, save_vis=True, vis_dir=str(tmp_path))
    assert _by_kind(off_msgs, entry)["scene"] == _by_kind(on_msgs, entry)["scene"]
    assert off["scenes"] == on["scenes"]
    assert off["total"]["metrics"] == on["total"]["metrics"]
    assert len(list(tmp_path.glob("Batch_*_Sampe_*_Scene_*.png"))) == sum(o[0]["img"].shape[0] for o in outputs)


def test_novel_views_per_row():
    expected = {"mp3d_double_256": (1,), "mp3d_double_256_val": (1,), "mp3d_single_256": (0, 2),
                "mp3d_double_512": (1,), "mp3d_double_512_full": (1,), "mp3d_double_512_full_val": (1,),
                "loc360_double_256": (1, 2), "loc360_double_256_da": (1, 2), "vigor_double": (1,)}
    assert {k: tuple(v["novel_views"]) for k, v in ev.EVAL_ENTRIES.items()} == expected
    assert ev.EVAL_ENTRIES["mp3d_double_256"]["context_views"] == (0, 2)
    assert ev.EVAL_ENTRIES["mp3d_single_256"]["context_views"] == (1,)
    assert ev.EVAL_ENTRIES["loc360_double_256"]["context_views"] == (0, 3)


def test_select_views():
    t = torch.arange(12.0).view(4, 3)
    assert ev.select_views(t, 4, (1,)).tolist() == [[1.0], [4.0], [7.0], [10.0]]
    assert ev.select_views(t, 4, (0, 2)).tolist() == [[0.0, 2.0], [3.0, 5.0], [6.0, 8.0], [9.0, 11.0]]
    # un-reshaped [B*V] metrics (legacy pcc of the 512 / 360Loc scripts)
    assert ev.select_views(torch.arange(8.0), 2, (1, 2)).tolist() == [[1.0, 2.0], [5.0, 6.0]]


def _slice_outputs(outputs, views):
    sliced = []
    for preds, gts in outputs:
        p = {"img": preds["img"][:, views], "depth": preds["depth"][:, views], "gaussian": preds["gaussian"]}
        g = {k: v[:, views] for k, v in gts.items()}
        sliced.append((p, g))
    return sliced


@pytest.mark.parametrize("dataset", sorted(LEGACY_LOOPS))
def test_novel_only_scores_only_the_novel_views(dataset):
    entry = ev.EVAL_ENTRIES[dataset]
    views = list(entry["novel_views"])
    batches, outputs = make_batches(entry, seed=2)
    _, novel = run_new(entry, batches, outputs, views=entry["novel_views"])
    _, sliced = run_new(entry, batches, _slice_outputs(outputs, views))
    for k in entry["batch_metrics"]:
        assert novel["total"]["metrics"][k] == pytest.approx(sliced["total"]["metrics"][k], rel=1e-6, abs=1e-9)
    for scene, row in sliced["scenes"].items():
        for k, value in row["metrics"].items():
            assert novel["scenes"][scene]["metrics"][k] == pytest.approx(value, rel=1e-6, abs=1e-9)
    _, full = run_new(entry, batches, outputs)
    assert novel["total"]["metrics"]["wspsnr"] != full["total"]["metrics"]["wspsnr"]


def test_extras_leave_default_lines_unchanged():
    entry = ev.EVAL_ENTRIES["mp3d_double_256"]
    batches, outputs = make_batches(entry, seed=3)
    base_msgs, base = run_new(entry, batches, outputs)
    msgs, res = run_new(entry, batches, outputs, align_depth=True, fast_ssim=True)
    assert _by_kind(msgs, entry)["scene"] == _by_kind(base_msgs, entry)["scene"]
    assert _by_kind(msgs, entry)["batch"] == _by_kind(base_msgs, entry)["batch"]
    assert res["total"]["metrics"] == base["total"]["metrics"]
    extra = res["total"]["extra"]
    assert set(extra) == {"ssim_fast", "abs_aligned", "rmse_aligned", "delta1_aligned", "delta2_aligned",
                          "delta3_aligned"}
    assert extra["ssim_fast"] == pytest.approx(res["total"]["metrics"]["ssim"], abs=1e-4)
    assert "extra" in res["scenes"]["m3d_0.1"]
    assert any(m.startswith(" m3d_0.1 extra ssim_fast: ") for m in msgs)


def test_wspsnr_weight_cache_is_bitwise_identical():
    legacy, cached = eval_ref.WSPSNR(), metrics.WSPSNR()
    g = torch.Generator().manual_seed(4)
    for dtype in (torch.float32, torch.float64):
        for n, h, w in ((6, 16, 32), (2, 16, 32), (3, 8, 24), (6, 16, 32)):
            a = torch.rand(n, h, w, 3, generator=g).to(dtype)
            b = (a + 0.05 * torch.randn(n, h, w, 3, generator=g).to(dtype))
            assert torch.equal(cached.ws_psnr(a, b, max_val=1.0), legacy.ws_psnr(a, b, max_val=1.0))
    assert len(cached.tensor_cache) == 4  # (device, H, W, dtype)
    # The device is part of the key: the same size and dtype on another device is a separate entry.
    on_cpu = cached.get_weight_tensor(16, 32, torch.device("cpu"), torch.float32)
    on_meta = cached.get_weight_tensor(16, 32, torch.device("meta"), torch.float32)
    assert on_cpu.device.type == "cpu" and on_meta.device.type == "meta"
    assert len(cached.tensor_cache) == 5


def test_wspsnr_weights_are_cos_latitude():
    # Row r's centre lies at polar angle (r + 0.5) * pi / H from the north pole; the weight is
    # sin(polar angle) = cos(latitude): largest at the equator, smallest at the poles.
    height = 8
    weights = metrics.WSPSNR().get_weights(height, 4)
    latitude = np.pi / 2 - (np.arange(height) + 0.5) * np.pi / height
    assert np.allclose(weights, np.cos(latitude)[:, None])
    assert np.isclose(weights[0, 0], weights[-1, 0]) and weights[0, 0] < weights[height // 2, 0]
    spec = importlib.util.spec_from_file_location("eval_entries_doc",
                                                  os.path.join(REPO_ROOT, "configs", "eval_entries.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "cos(latitude) weights" in module.__doc__ and "sin(latitude)" not in module.__doc__


def test_gpu_ssim_agrees_with_skimage():
    g = torch.Generator().manual_seed(5)
    a = torch.rand(4, 3, 24, 40, generator=g)
    b = (a + 0.1 * torch.randn(4, 3, 24, 40, generator=g)).clamp(0, 1)
    fast = metrics.compute_ssim_gpu(a, b)
    assert fast.dtype == b.dtype and fast.shape == (4,)
    assert torch.allclose(fast, metrics.compute_ssim(a, b), atol=1e-4)


def test_check_entry_rejects_the_legacy_total_line():
    entry = dict(ev.EVAL_ENTRIES["mp3d_double_256"])
    entry["total_line"] = ("Finish evluation ({:d} s). Total psnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}, "
                           "pcc: {:.4f}, abs: {:.4f}, silog: {:.4f}, rmse: {:.4f}, delta1: {:.4f}, "
                           "delta2: {:.4f}, delta3: {:.4f}, depthsim: {:.4f}.")
    with pytest.raises(ValueError):
        ev.check_entry("legacy", entry)


def test_save_ply_flag_per_row():
    assert {k: v["save_ply"] for k, v in ev.EVAL_ENTRIES.items()} == {
        "mp3d_double_256": True, "mp3d_double_256_val": True, "mp3d_single_256": True,
        "mp3d_double_512": True, "mp3d_double_512_full": True, "mp3d_double_512_full_val": True,
        "loc360_double_256": False, "loc360_double_256_da": False, "vigor_double": True}
    entry = dict(ev.EVAL_ENTRIES["vigor_double"], save_ply=None)
    with pytest.raises(ValueError, match="save_ply"):
        ev.check_entry("vigor_double", entry)


def test_save_outputs_refuses_gaussians_that_are_not_per_sample(tmp_path):
    # The Pan2 layout: per-view pixel Gaussians, (b v) hw c, so row b is not sample b.
    entry = ev.EVAL_ENTRIES["loc360_double_256"]
    batches, outputs = make_batches(entry)
    preds, gts = outputs[0]
    bs, n_views = preds["img"].shape[:2]
    per_view = dict(preds, gaussian=torch.rand(bs * n_views, 20, 14))
    with pytest.raises(ValueError, match="one Gaussian set per sample"):
        ev.save_outputs(entry, str(tmp_path), 0, batches[0], per_view, gts, save_vis=True, save_ply_files=True)
    assert list(tmp_path.iterdir()) == []  # refused before the first file
    # PNGs alone do not read the Gaussians.
    ev.save_outputs(entry, str(tmp_path), 0, batches[0], per_view, gts, save_vis=True, save_ply_files=False)
    assert len(list(tmp_path.glob("Batch_0_Sampe_*.png"))) == bs


def test_save_outputs_writes_one_ply_per_sample(tmp_path):
    pytest.importorskip("plyfile")
    entry = ev.EVAL_ENTRIES["mp3d_double_256"]
    batches, outputs = make_batches(entry)
    preds, gts = outputs[0]
    ev.save_outputs(entry, str(tmp_path), 0, batches[0], preds, gts, save_vis=False, save_ply_files=True)
    assert len(list(tmp_path.glob("Batch_0_Sampe_*_Scene_*.ply"))) == preds["img"].shape[0]


def test_main_refuses_save_ply_for_loc360_before_any_work(tmp_path):
    config = os.path.join(REPO_ROOT, "configs", "OmniScene", "omni_gs_160x320_360Loc_cylinder_all_256.py")
    ckpt_dir = tmp_path / "checkpoint-24000"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").write_bytes(b"")
    cwd = os.getcwd()
    try:
        with pytest.raises(SystemExit) as exc:
            ev.main(ev.parse_args(["--dataset", "loc360_double_256", "--py-config", config, "--ckpt", str(ckpt_dir),
                                   "--out-dir", str(tmp_path / "out"), "--save-ply"]))
        assert "--save-ply is not supported for loc360_double_256" in str(exc.value.code)
        assert not (tmp_path / "out").exists()
    finally:
        os.chdir(cwd)


def test_compare_state_dicts():
    class Net(torch.nn.Module):
        def __init__(self, n=3):
            super().__init__()
            self.a = torch.nn.Linear(n, 2)
            self.b = torch.nn.Linear(2, 2)
            self.c = self.b  # the same module under two names
    model_dict = Net().state_dict()
    ckpt = {"a.weight": torch.zeros(2, 4), "a.bias": torch.zeros(2), "b.weight": torch.zeros(2, 2),
            "b.bias": torch.zeros(2), "old.weight": torch.zeros(1)}
    report = ev.compare_state_dicts(ckpt, model_dict)
    assert report["loaded"] == ["a.bias", "b.weight", "b.bias"]
    assert report["shape"] == ["a.weight"]
    assert report["unused"] == ["old.weight"]
    assert report["missing"] == []
    assert sorted(report["aliased"]) == ["c.bias", "c.weight"]
    del ckpt["b.bias"]
    assert ev.compare_state_dicts(ckpt, model_dict)["missing"] == ["b.bias", "c.bias"]


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 2)


def test_load_checkpoint_allow_extra_skips_only_the_matching_names(tmp_path, capsys):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    entry = ev.EVAL_ENTRIES["mp3d_double_256"]  # load="filter"; the accelerator is not used
    ckpt_file = tmp_path / "model.safetensors"
    source = {k: torch.randn(v.shape) for k, v in _Head().state_dict().items()}
    mono = {"pixel_gs.mono_depth.encoder.weight": torch.ones(4, 2), "pixel_gs.mono_depth.encoder.bias": torch.ones(4)}

    def load(tensors, allowed_extra=(), strict=True):
        safetensors_torch.save_file(tensors, str(ckpt_file))
        model = _Head()
        counts = ev.load_checkpoint(entry, model, None, str(tmp_path), str(ckpt_file), strict, list(allowed_extra))
        return model, counts

    # strict and no pattern: the extra names stop the run (the REF-T4-pixel failure)
    with pytest.raises(SystemExit) as exc:
        load({**source, **mono})
    assert "--allow-extra" in str(exc.value.code)

    model, counts = load({**source, **mono}, ["pixel_gs.mono_depth.*"])
    assert counts["loaded"] == 2 and counts["unused"] == 0 and counts["unused_allowed"] == 2
    assert counts["missing"] == 0 and counts["shape"] == 0
    for k, v in model.state_dict().items():
        assert torch.equal(v, source[k])
    assert "skipped 2 more not in the model as allowed extra: pixel_gs.mono_depth.* (2)" in capsys.readouterr().out
    # the same weights as the legacy filtered load (--allow-partial)
    legacy, _ = load({**source, **mono}, strict=False)
    for k, v in legacy.state_dict().items():
        assert torch.equal(v, model.state_dict()[k])
    # exact names tolerate only themselves
    load({**source, **mono}, list(mono))
    with pytest.raises(SystemExit):
        load({**source, **mono}, ["pixel_gs.mono_depth.encoder.weight"])
    # another extra name, a missing name or another shape still exits
    for tensors in ({**source, **mono, "pixel_gs.other.weight": torch.ones(1)},
                    {"head.weight": source["head.weight"], **mono},
                    {**source, "head.bias": torch.ones(3), **mono}):
        with pytest.raises(SystemExit) as exc:
            load(tensors, ["pixel_gs.mono_depth.*"])
        assert "partial checkpoint load" in str(exc.value.code)


def test_resolve_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "checkpoint-48000"
    ckpt_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        ev.resolve_checkpoint(str(ckpt_dir))
    (ckpt_dir / "model.safetensors").write_bytes(b"")
    assert ev.resolve_checkpoint(str(ckpt_dir)) == (str(ckpt_dir), str(ckpt_dir / "model.safetensors"), 48000)
    assert ev.resolve_checkpoint(str(ckpt_dir / "model.safetensors"))[2] == 48000
    other = tmp_path / "weights.safetensors"
    other.write_bytes(b"")
    assert ev.resolve_checkpoint(str(other)) == (str(tmp_path), str(other), None)


def _main_args(tmp_path, ckpt, out_dir, extra=()):
    config = os.path.join(REPO_ROOT, "configs", "OmniScene", "omni_gs_160x320_mp3d_cylinder_all_256.py")
    return ev.parse_args(["--dataset", "mp3d_double_256", "--py-config", config, "--ckpt", str(ckpt),
                          "--out-dir", str(out_dir), *extra])


def test_main_refuses_missing_checkpoint_and_unsafe_out_dirs(tmp_path):
    cwd = os.getcwd()
    try:
        with pytest.raises(SystemExit) as exc:
            ev.main(_main_args(tmp_path, tmp_path / "checkpoint-1", tmp_path / "out"))
        assert exc.value.code not in (0, None)
        assert not (tmp_path / "out").exists()

        ckpt_dir = tmp_path / "checkpoint-2"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.safetensors").write_bytes(b"")
        with pytest.raises(SystemExit) as exc:  # inside the source checkpoint
            ev.main(_main_args(tmp_path, ckpt_dir, ckpt_dir / "eval"))
        assert exc.value.code not in (0, None)
        assert not (ckpt_dir / "eval").exists()

        with pytest.raises(SystemExit) as exc:  # protected tree (a `workdirs` component)
            ev.main(_main_args(tmp_path, ckpt_dir, tmp_path / "workdirs" / "eval"))
        assert exc.value.code not in (0, None)
        assert not (tmp_path / "workdirs").exists()
    finally:
        os.chdir(cwd)


def _runs_with_training_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "runs" / "mp3d_all_256" / "checkpoint-6000"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model.safetensors").write_bytes(b"")
    return tmp_path / "runs", ckpt_dir


def test_main_refuses_an_out_dir_that_contains_the_checkpoint(tmp_path, monkeypatch):
    runs, ckpt_dir = _runs_with_training_checkpoint(tmp_path)
    monkeypatch.setattr(ev, "default_run_dir", functools.partial(ev.default_run_dir, root=str(runs)))
    config = os.path.join(REPO_ROOT, "configs", "OmniScene", "omni_gs_160x320_mp3d_cylinder_all_256.py")
    cases = [
        ["--out-dir", str(runs / "mp3d_all_256")],  # the training run's directory (--out-dir D --ckpt D/checkpoint-N)
        ["--run-id", "mp3d_all_256"],               # the training run's id under the runs root
        ["--out-dir", str(runs)],                   # any ancestor
    ]
    cwd = os.getcwd()
    try:
        for extra in cases:
            with pytest.raises(SystemExit) as exc:
                ev.main(ev.parse_args(["--dataset", "mp3d_double_256", "--py-config", config,
                                       "--ckpt", str(ckpt_dir), *extra]))
            assert "inside or contains the checkpoint directory" in str(exc.value.code)
            # nothing written: no scratch cwd, log, config dump or metrics.json next to the checkpoint
            assert sorted(os.listdir(runs)) == ["mp3d_all_256"]
            assert sorted(os.listdir(runs / "mp3d_all_256")) == ["checkpoint-6000"]
    finally:
        os.chdir(cwd)


def test_main_resolves_relative_out_dirs_before_any_chdir(tmp_path, monkeypatch):
    # A relative CYLINDERSPLAT_RUNS_ROOT (default_run_dir) or --out-dir is resolved against the launch
    # directory before evaluate.py changes directory; the refusal names the absolute path.
    runs, ckpt_dir = _runs_with_training_checkpoint(tmp_path)
    monkeypatch.setattr(ev, "default_run_dir", functools.partial(ev.default_run_dir, root="runs"))
    monkeypatch.chdir(tmp_path)
    config = os.path.join(REPO_ROOT, "configs", "OmniScene", "omni_gs_160x320_mp3d_cylinder_all_256.py")
    for extra in (["--run-id", "mp3d_all_256"], ["--out-dir", os.path.join("runs", "mp3d_all_256")]):
        with pytest.raises(SystemExit) as exc:
            ev.main(ev.parse_args(["--dataset", "mp3d_double_256", "--py-config", config,
                                   "--ckpt", str(ckpt_dir), *extra]))
        assert f"--out-dir {runs / 'mp3d_all_256'} is inside or contains" in str(exc.value.code)
        assert sorted(os.listdir(runs / "mp3d_all_256")) == ["checkpoint-6000"]


def test_cli_defaults():
    args = ev.parse_args(["--dataset", "mp3d_double_256", "--py-config", "c.py", "--ckpt", "x"])
    assert args.save_vis is False and args.save_ply is False and args.novel_only is False
    assert args.strict_load is True and args.allow_partial is False and args.allow_extra == []
    assert args.switch == [] and args.out_dir is None
    args = ev.parse_args(["--dataset", "mp3d_double_256", "--py-config", "c.py", "--ckpt", "x",
                          "--allow-extra", "a.*", "--allow-extra", "b.w"])
    assert args.allow_extra == ["a.*", "b.w"]


def test_apply_render_switches_resets_the_prune_threshold(monkeypatch):
    # The threshold is a process global: a default evaluation after a pruned one must render unpruned.
    from model import gaussian as gaussian_mod
    monkeypatch.setattr(gaussian_mod, "_prune_opacity", gaussian_mod.get_prune_opacity())
    ev.apply_render_switches({"prune_opacity": 0.4})
    assert gaussian_mod.get_prune_opacity() == 0.4
    ev.apply_render_switches({"prune_opacity": 0.0})
    assert gaussian_mod.get_prune_opacity() == 0.0


def test_validation_row_is_the_test_row_on_the_val_split():
    test_row, val_row = ev.EVAL_ENTRIES["mp3d_double_256"], ev.EVAL_ENTRIES["mp3d_double_256_val"]
    assert val_row["stage"] == "val" and test_row["stage"] == "test"
    # the loader labels every validation scene m3d_0.1 (its single root is zipped with the first test set)
    assert val_row["scene_keys"] == ("m3d_0.1",)
    differing = {k for k in test_row if test_row[k] != val_row[k]}
    assert differing == {"stage", "scene_keys"}
    ev.check_entry("mp3d_double_256_val", val_row)


@pytest.mark.parametrize("row,base", [("mp3d_double_512_full", "mp3d_double_256"),
                                      ("mp3d_double_512_full_val", "mp3d_double_256_val")])
def test_stage4_rows_are_the_256_rows_on_the_512_loader(row, base):
    # Same metric code, protocol, batch-size key, scene keys and split as the 256 row; only the
    # loader differs, and no legacy script is claimed (the 512 script computes a reduced set).
    new, old = ev.EVAL_ENTRIES[row], ev.EVAL_ENTRIES[base]
    assert set(new) == set(old)
    assert {k for k in new if new[k] != old[k]} == {"loader", "legacy_script"}
    assert new["loader"] == ("data.mp3d_dataloader_double_512", "load_MP3D_data")
    assert new["legacy_script"] is None
    assert new["batch_metrics"] == new["scene_metrics"] == new["total_metrics"] == ev.KNOWN_METRICS
    assert new["depth_metrics"] is True and new["pcc_per_view"] is True
    assert (new["context_views"], new["target_views"], new["novel_views"]) == ((0, 2), (0, 1, 2), (1,))
    ev.check_entry(row, new)


def test_loc360_da_row_is_the_360loc_row_with_the_depth_anywhere_loader():
    # Same split, samples, views, metric code and lines; only the loader (its outputs['depth'],
    # the PCC reference) differs, and no legacy script is claimed.
    new, old = ev.EVAL_ENTRIES["loc360_double_256_da"], ev.EVAL_ENTRIES["loc360_double_256"]
    assert set(new) == set(old)
    assert {k for k in new if new[k] != old[k]} == {"loader", "legacy_script"}
    assert new["loader"] == ("data.loc360_dataloader_da", "load_360Loc_data_da")
    assert new["legacy_script"] is None and old["legacy_script"] == "legacy/evaluate_360Loc_double_256.py"
    ev.check_entry("loc360_double_256_da", new)


def test_stage4_rows_split_and_scene_keys():
    test_row, val_row = ev.EVAL_ENTRIES["mp3d_double_512_full"], ev.EVAL_ENTRIES["mp3d_double_512_full_val"]
    assert (test_row["stage"], val_row["stage"]) == ("test", "val")
    assert val_row["scene_keys"] == ("m3d_0.1",)
    assert test_row["scene_keys"] == ev.EVAL_ENTRIES["mp3d_double_256"]["scene_keys"]
    assert {k for k in test_row if test_row[k] != val_row[k]} == {"stage", "scene_keys"}
    # The legacy 512 row is unchanged: its script, its reduced metric set, its own labels.
    legacy = ev.EVAL_ENTRIES["mp3d_double_512"]
    assert legacy["legacy_script"] == "legacy/evaluate_mp3d_double_512.py"
    assert legacy["batch_metrics"] == ("wspsnr", "ssim", "lpips", "pcc") and legacy["depth_metrics"] is False


def _loader_scene_keys(rel):
    # The loader's test_datasets literal -> the scene keys it labels its samples with (name_dis).
    import ast
    with open(os.path.join(REPO_ROOT, rel)) as f:
        tree = ast.parse(f.read())
    (node,) = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "test_datasets"]
    return tuple(f"{d['name']}_{d['dis']}" for d in ast.literal_eval(node.value))


def test_512_loader_labels_scenes_like_the_256_loader():
    # Both loaders zip their roots with the same test_datasets list, so the stage-4 rows can use the
    # 256 rows' scene keys (and the validation split is labelled m3d_0.1 by both).
    keys_256 = _loader_scene_keys("data/mp3d_dataloader_double_256.py")
    keys_512 = _loader_scene_keys("data/mp3d_dataloader_double_512.py")
    assert keys_512 == keys_256
    assert set(keys_512) == set(ev.EVAL_ENTRIES["mp3d_double_512_full"]["scene_keys"])
    assert keys_512[0] == "m3d_0.1"
