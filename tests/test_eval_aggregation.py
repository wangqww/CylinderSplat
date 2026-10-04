"""evaluate.py and its rows (configs/eval_entries.py), on CPU.

The rows: views, loaders, batch-size and scene keys, line formats. The aggregation: evaluate_loop over a fake
model with synthetic per-view predictions (tiny images, every scene key, batches that mix scenes, a short last
batch) must print batch, scene and Total lines whose values equal a direct computation from the same per-view
metrics. LPIPS is stubbed (no VGG download); SSIM, WS-PSNR, PCC and the depth metrics are the real ones.
Also: --novel-only, the extra columns, PNG / PLY output, checkpoint loading and the refusals of main().
"""

import ast
import os
import re

import pytest

from tests.conftest import REPO_ROOT, load_by_path

torch = pytest.importorskip("torch")
ev = load_by_path("evaluate.py", "cylindersplat_evaluate")
from tools import metrics  # noqa: E402

H, W = 16, 32
ROWS = sorted(ev.EVAL_ENTRIES)
FORMAT_ROWS = ["loc360_double_256", "mp3d_double_256", "mp3d_double_512", "vigor_double"]  # one per line format
ALL_256 = os.path.join(REPO_ROOT, "configs", "OmniScene", "omni_gs_160x320_mp3d_cylinder_all_256.py")


@pytest.fixture(autouse=True)
def fake_lpips(monkeypatch):
    monkeypatch.setattr(ev, "compute_lpips", lambda gt, pred: (gt - pred).abs().mean(dim=(1, 2, 3)))


def parse(rel):
    with open(os.path.join(REPO_ROOT, rel)) as f:
        return ast.parse(f.read())


# ----------------------------------------------------------------------------- the rows


def test_views_per_row():
    two_view = ((0, 2), (0, 1, 2), (1,))  # (context, targets, novel)
    loc360 = ((0, 3), (0, 1, 2, 3), (1, 2))
    got = {row: (e["context_views"], e["target_views"], e["novel_views"]) for row, e in ev.EVAL_ENTRIES.items()}
    assert got == {
        "mp3d_double_256": two_view,
        "mp3d_double_256_val": two_view,
        "mp3d_single_256": ((1,), (0, 1, 2), (0, 2)),
        "mp3d_double_512_full": two_view,
        "mp3d_double_512_full_val": two_view,
        "mp3d_double_512": two_view,
        "loc360_double_256": loc360,
        "loc360_double_256_da": loc360,
        "vigor_double": two_view,
    }


@pytest.mark.parametrize("row", ROWS)
def test_row_is_consistent(row):
    entry = ev.EVAL_ENTRIES[row]
    ev.check_entry(row, entry)  # every printed label names its value; novel views = targets - context; ...
    module, function = entry["loader"]
    tree = parse(module.replace(".", "/") + ".py")
    assert function in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if entry["scene_keys"] is not None:
        # The per-scene lines group the samples by batch["scene"], which the loader must provide.
        keys = {
            k.value for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys if isinstance(k, ast.Constant)
        }
        assert "scene" in keys
    config = pytest.importorskip("mmengine.config")
    cfg = config.Config.fromfile(os.path.join(REPO_ROOT, entry["example_config"]))
    assert entry["batch_size_key"] in cfg.dataset_params


@pytest.mark.parametrize(
    "row,base,differing",
    [
        ("mp3d_double_256_val", "mp3d_double_256", {"stage", "scene_keys"}),
        ("mp3d_double_512_full", "mp3d_double_256", {"loader", "example_config"}),
        ("mp3d_double_512_full_val", "mp3d_double_256_val", {"loader", "example_config"}),
        ("loc360_double_256_da", "loc360_double_256", {"loader"}),
    ],
)
def test_derived_rows_change_only_their_own_fields(row, base, differing):
    new, old = ev.EVAL_ENTRIES[row], ev.EVAL_ENTRIES[base]
    assert set(new) == set(old) and {k for k in new if new[k] != old[k]} == differing
    if row.endswith("_val"):
        # The MP3D loaders label every validation scene with the first test set's key.
        assert new["stage"] == "val" and new["scene_keys"] == ("m3d_0.1",)


@pytest.mark.parametrize(
    "rel",
    ["data/mp3d_dataloader_double_256.py", "data/mp3d_dataloader_single_256.py", "data/mp3d_dataloader_double_512.py"],
)
def test_mp3d_loaders_label_scenes_with_the_row_scene_keys(rel):
    (node,) = [n for n in parse(rel).body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "test_datasets"]
    labels = tuple(f"{d['name']}_{d['dis']}" for d in ast.literal_eval(node.value))
    assert labels == ev.EVAL_ENTRIES["mp3d_double_256"]["scene_keys"]


def test_save_ply_per_row():
    assert {row for row, e in ev.EVAL_ENTRIES.items() if not e["save_ply"]} == {
        "loc360_double_256",
        "loc360_double_256_da",
    }
    with pytest.raises(ValueError, match="save_ply"):
        ev.check_entry("vigor_double", dict(ev.EVAL_ENTRIES["vigor_double"], save_ply=None))


def test_check_entry_refuses_a_mislabelled_line():
    # A Total line without wspsnr prints every later value under the previous label.
    line = ev.EVAL_ENTRIES["mp3d_double_256"]["total_line"].replace("wspsnr: {:.3f}, ", "")
    with pytest.raises(ValueError, match="total_line"):
        ev.check_entry("mislabelled", dict(ev.EVAL_ENTRIES["mp3d_double_256"], total_line=line))


# ----------------------------------------------------------------------------- aggregation


def make_batches(entry, seed=0, bs=2):
    """Synthetic (batches, outputs): every scene key, batches that mix scenes, a short last batch."""
    g = torch.Generator().manual_seed(seed)
    n_views = len(entry["target_views"])
    keys = entry["scene_keys"]
    if keys is None:
        scenes = [None] * 7
    elif len(keys) == 1:
        scenes = [keys[0]] * 5
    else:
        scenes = [key for key, n in zip(keys, [3, 2, 2, 3, 1, 2, 2]) for _ in range(n)]
    batches, outputs = [], []
    for i, start in enumerate(range(0, len(scenes), bs)):
        names = scenes[start : start + bs]
        b = len(names)
        gt_img = torch.rand(b, n_views, 3, H, W, generator=g)
        pred_img = gt_img + 0.08 * torch.randn(b, n_views, 3, H, W, generator=g)  # partly outside [0, 1]
        prior = 0.5 + 4.0 * torch.rand(b, n_views, 1, H, W, generator=g)
        pred_depth = (prior[:, :, 0] * (1.0 + 0.2 * torch.randn(b, n_views, H, W, generator=g))).abs() + 0.05
        gts = {"img": gt_img, "depth": prior}
        if entry["depth_metrics"]:
            depth_gt = 0.2 + 6.0 * torch.rand(b, n_views, 1, H, W, generator=g)
            for j, name in enumerate(names):
                if not name.startswith("m3d"):  # the loaders give zeros where there is no GT depth
                    depth_gt[j] = 0.0
            gts.update(depth_gt=depth_gt, mask_gt=(depth_gt > 0.45) & (depth_gt < 10.0))
        preds = {"img": pred_img, "depth": pred_depth, "gaussian": torch.rand(b, 20, 14, generator=g)}
        batches.append({"i": i} if keys is None else {"i": i, "scene": list(names)})
        outputs.append((preds, gts))
    return batches, outputs


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def eval(self):
        return self

    def forward_test(self, batch):
        return self.outputs[batch["i"]]


def run(entry, batches, outputs, **kwargs):
    lines = []
    results = ev.evaluate_loop(entry, FakeModel(outputs), batches, lambda x: x, lines.append, **kwargs)
    return lines, results


def direct(entry, batches, outputs):
    """Batch means, totals and per-scene means computed straight from compute_batch_metrics."""
    names = entry["batch_metrics"]
    wspsnr = metrics.WSPSNR()
    batch_means, totals, samples = [], dict.fromkeys(names, 0.0), {}
    for batch, (preds, gts) in zip(batches, outputs):
        bv, _ = ev.compute_batch_metrics(entry, preds, gts, wspsnr)
        means = {k: bv[k].mean() for k in names}
        batch_means.append(means)
        totals = {k: totals[k] + means[k] for k in names}
        for b in range(preds["img"].shape[0]):
            if entry["scene_keys"] is not None:
                record = {k: bv[k][b].mean().item() for k in entry["scene_metrics"]}
                samples.setdefault(batch["scene"][b], []).append(record)
    totals = {k: v.item() / len(batches) for k, v in totals.items()}
    scenes = {s: {k: sum(r[k] for r in rs) / len(rs) for k in entry["scene_metrics"]} for s, rs in samples.items()}
    return batch_means, totals, scenes


@pytest.mark.parametrize("row", FORMAT_ROWS)
def test_lines_and_values(row):
    entry = ev.EVAL_ENTRIES[row]
    batches, outputs = make_batches(entry)
    lines, results = run(entry, batches, outputs)
    batch_means, totals, scenes = direct(entry, batches, outputs)
    n = len(batches)
    # One line per batch (device index 0 on CPU), one per scene key in order, then the Total line.
    assert lines[:n] == [
        entry["batch_line"] % ((i, 0) + tuple(m[k] for k in entry["batch_metrics"])) for i, m in enumerate(batch_means)
    ]
    if entry["scene_keys"] is None:
        assert lines[n:-1] == [] and results["scenes"] == {}
    else:
        assert lines[n:-1] == [
            entry["scene_line"].format(s, *[scenes[s][k] for k in entry["scene_metrics"]]) for s in entry["scene_keys"]
        ]
        assert {s: r["metrics"] for s, r in results["scenes"].items()} == scenes
    assert results["total"]["metrics"] == totals  # bitwise
    # The Total line prints each total under its own label.
    pairs = re.findall(r"(\w+): (-?\d+\.\d+|nan|inf)", lines[-1].split(" s). ", 1)[1])
    assert [k for k, _ in pairs] == list(entry["total_metrics"])
    for k, printed in pairs:
        assert printed == f"{totals[k]:.{3 if 'psnr' in k else 4}f}"


def test_a_scene_outside_the_row_keys_is_an_error():
    entry = ev.EVAL_ENTRIES["vigor_double"]
    batches, outputs = make_batches(entry)
    batches[1]["scene"][0] = "Seattle"
    with pytest.raises(KeyError, match="not a scene key"):
        run(entry, batches, outputs)


def test_select_views():
    t = torch.arange(12.0).view(4, 3)
    assert ev.select_views(t, 4, (1,)).tolist() == [[1.0], [4.0], [7.0], [10.0]]
    assert ev.select_views(t, 4, (0, 2)).tolist() == [[0.0, 2.0], [3.0, 5.0], [6.0, 8.0], [9.0, 11.0]]
    # A [B*V] metric (pcc of the rows with pcc_per_view False)
    assert ev.select_views(torch.arange(8.0), 2, (1, 2)).tolist() == [[1.0, 2.0], [5.0, 6.0]]


@pytest.mark.parametrize("row", FORMAT_ROWS)
def test_novel_only_scores_only_the_novel_views(row):
    entry = ev.EVAL_ENTRIES[row]
    views = list(entry["novel_views"])
    batches, outputs = make_batches(entry, seed=2)
    sliced = [
        (
            {"img": p["img"][:, views], "depth": p["depth"][:, views], "gaussian": p["gaussian"]},
            {k: v[:, views] for k, v in g.items()},
        )
        for p, g in outputs
    ]
    _, novel = run(entry, batches, outputs, views=entry["novel_views"])
    _, reference = run(entry, batches, sliced)
    for k, value in reference["total"]["metrics"].items():
        assert novel["total"]["metrics"][k] == pytest.approx(value, rel=1e-6, abs=1e-9)
    for scene, scene_row in reference["scenes"].items():
        for k, value in scene_row["metrics"].items():
            assert novel["scenes"][scene]["metrics"][k] == pytest.approx(value, rel=1e-6, abs=1e-9)
    _, full = run(entry, batches, outputs)
    assert novel["total"]["metrics"]["wspsnr"] != full["total"]["metrics"]["wspsnr"]


def test_extra_columns_leave_the_default_lines_unchanged():
    entry = ev.EVAL_ENTRIES["mp3d_double_256"]
    batches, outputs = make_batches(entry, seed=3)
    base_lines, base = run(entry, batches, outputs)
    lines, results = run(entry, batches, outputs, align_depth=True, fast_ssim=True)
    default = [line for line in lines if " extra " not in line and not line.startswith("Extra totals: ")]
    assert default[:-1] == base_lines[:-1]  # batch and scene lines (the Total line carries the run time)
    assert results["total"]["metrics"] == base["total"]["metrics"]
    extra = results["total"]["extra"]
    assert set(extra) == {
        "ssim_fast",
        "abs_aligned",
        "rmse_aligned",
        "delta1_aligned",
        "delta2_aligned",
        "delta3_aligned",
    }
    assert extra["ssim_fast"] == pytest.approx(results["total"]["metrics"]["ssim"], abs=1e-4)
    assert "extra" in results["scenes"]["m3d_0.1"]


def test_wspsnr_weights_are_cos_latitude():
    # Row r's centre lies at polar angle (r + 0.5) * pi / H; its weight is sin(polar angle) = cos(latitude).
    np = pytest.importorskip("numpy")
    height = 8
    weights = metrics.WSPSNR().get_weights(height, 4)
    latitude = np.pi / 2 - (np.arange(height) + 0.5) * np.pi / height
    assert np.allclose(weights, np.cos(latitude)[:, None])


def test_gpu_ssim_agrees_with_skimage():
    g = torch.Generator().manual_seed(5)
    a = torch.rand(4, 3, 24, 40, generator=g)
    b = (a + 0.1 * torch.randn(4, 3, 24, 40, generator=g)).clamp(0, 1)
    fast = metrics.compute_ssim_gpu(a, b)
    assert fast.dtype == b.dtype and fast.shape == (4,)
    assert torch.allclose(fast, metrics.compute_ssim(a, b), atol=1e-4)


# ----------------------------------------------------------------------------- outputs


def test_save_outputs_writes_one_png_and_one_ply_per_sample(tmp_path, monkeypatch):
    pytest.importorskip("plyfile")
    pytest.importorskip("imageio")
    import matplotlib
    import matplotlib.cm

    if not hasattr(matplotlib.cm, "get_cmap"):  # removed in matplotlib 3.9; tools/visualization.py calls it
        monkeypatch.setattr(matplotlib.cm, "get_cmap", matplotlib.colormaps.get_cmap, raising=False)
    entry = ev.EVAL_ENTRIES["mp3d_double_256"]
    batches, outputs = make_batches(entry)
    preds, gts = outputs[0]
    ev.save_outputs(entry, str(tmp_path), 0, batches[0], preds, gts, save_vis=True, save_ply_files=True)
    stems = [f"Batch_0_Sampe_{b}_Scene_{scene}" for b, scene in enumerate(batches[0]["scene"])]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(s + ext for s in stems for ext in (".png", ".ply"))


def test_save_outputs_refuses_gaussians_that_are_not_per_sample(tmp_path):
    # The 360Loc model's layout: per-view pixel Gaussians, (b v) hw c, so row b is not sample b.
    entry = ev.EVAL_ENTRIES["loc360_double_256"]
    batches, outputs = make_batches(entry)
    preds, gts = outputs[0]
    per_view = dict(preds, gaussian=torch.rand(preds["img"].shape[0] * 4, 20, 14))
    with pytest.raises(ValueError, match="one Gaussian set per sample"):
        ev.save_outputs(entry, str(tmp_path), 0, batches[0], per_view, gts, save_vis=False, save_ply_files=True)
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------------- checkpoints


def test_compare_state_dicts():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(3, 2)
            self.b = torch.nn.Linear(2, 2)
            self.c = self.b  # the same module under two names

    model_dict = Net().state_dict()
    ckpt = {
        "a.weight": torch.zeros(2, 4),
        "a.bias": torch.zeros(2),
        "b.weight": torch.zeros(2, 2),
        "b.bias": torch.zeros(2),
        "old.weight": torch.zeros(1),
    }
    report = ev.compare_state_dicts(ckpt, model_dict)
    assert report["loaded"] == ["a.bias", "b.weight", "b.bias"]
    assert report["shape"] == ["a.weight"] and report["unused"] == ["old.weight"] and report["missing"] == []
    assert sorted(report["aliased"]) == ["c.bias", "c.weight"]
    del ckpt["b.bias"]
    assert ev.compare_state_dicts(ckpt, model_dict)["missing"] == ["b.bias", "c.bias"]


def test_load_checkpoint_strict_and_allowed_extra(tmp_path, capsys):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    entry = ev.EVAL_ENTRIES["mp3d_double_256"]  # load="filter": the accelerator is not used
    ckpt_file = tmp_path / "model.safetensors"
    source = {"head.weight": torch.randn(2, 3), "head.bias": torch.randn(2)}
    mono = {"pixel_gs.mono_depth.encoder.weight": torch.ones(4, 2), "pixel_gs.mono_depth.encoder.bias": torch.ones(4)}

    def load(tensors, allowed_extra=(), strict=True):
        safetensors_torch.save_file(tensors, str(ckpt_file))
        model = torch.nn.Module()
        model.head = torch.nn.Linear(3, 2)
        counts = ev.load_checkpoint(entry, model, None, str(tmp_path), str(ckpt_file), strict, list(allowed_extra))
        return model, counts

    with pytest.raises(SystemExit, match="--allow-extra"):  # strict, no pattern: extra names stop the run
        load({**source, **mono})
    model, counts = load({**source, **mono}, ["pixel_gs.mono_depth.*"])
    assert (counts["loaded"], counts["unused"], counts["unused_allowed"], counts["missing"]) == (2, 0, 2, 0)
    assert all(torch.equal(v, source[k]) for k, v in model.state_dict().items())
    assert "skipped 2 more not in the model as allowed extra: pixel_gs.mono_depth.* (2)" in capsys.readouterr().out
    partial, _ = load({**source, **mono}, strict=False)  # --allow-partial: the same weights
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in partial.state_dict().items())
    # Another extra name, a missing name or another shape still stops the run.
    for tensors in (
        {**source, **mono, "pixel_gs.other.weight": torch.ones(1)},
        {"head.weight": source["head.weight"], **mono},
        {**source, "head.bias": torch.ones(3), **mono},
    ):
        with pytest.raises(SystemExit, match="partial checkpoint load"):
            load(tensors, ["pixel_gs.mono_depth.*"])


def test_resolve_checkpoint_and_allowed_extra_patterns(tmp_path):
    ckpt_dir = tmp_path / "checkpoint-48000"
    ckpt_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        ev.resolve_checkpoint(str(ckpt_dir))
    (ckpt_dir / "model.safetensors").write_bytes(b"")
    assert ev.resolve_checkpoint(str(ckpt_dir)) == (str(ckpt_dir), str(ckpt_dir / "model.safetensors"), 48000)
    assert ev.resolve_checkpoint(str(ckpt_dir / "model.safetensors"))[2] == 48000
    (tmp_path / "weights.safetensors").write_bytes(b"")
    assert ev.resolve_checkpoint(str(tmp_path / "weights.safetensors")) == (
        str(tmp_path),
        str(tmp_path / "weights.safetensors"),
        None,
    )
    assert ev.allowed_extra_patterns(["a.*", "b.w", "a.*"]) == ["a.*", "b.w"]
    for bad in ("", "*", "a*.b"):
        with pytest.raises(ValueError):
            ev.allowed_extra_patterns([bad])


# ----------------------------------------------------------------------------- main(): refusals before any write


def test_main_refusals_write_nothing(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    ckpt = runs / "all_256" / "checkpoint-6000"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(ev, "RUNS_ROOT", str(runs))
    monkeypatch.chdir(tmp_path)
    loc360_config = os.path.join(REPO_ROOT, "configs", "OmniScene", "omni_gs_160x320_360Loc_cylinder_all_256.py")
    mp3d = ["--dataset", "mp3d_double_256", "--py-config", ALL_256]
    loc360 = ["--dataset", "loc360_double_256", "--py-config", loc360_config, "--ckpt", str(ckpt)]
    inside = f"--out-dir {runs / 'all_256'} is inside or contains the checkpoint directory"
    cases = [
        (loc360 + ["--save-ply"], "--save-ply is not supported for loc360_double_256"),
        (loc360 + ["--align-depth"], "--align-depth needs GT depth"),
        (mp3d + ["--ckpt", str(tmp_path / "missing")], "checkpoint not found"),
        (mp3d + ["--ckpt", str(ckpt), "--allow-extra", "*"], "allowed-extra pattern"),
        (mp3d + ["--ckpt", str(ckpt), "--out-dir", str(ckpt / "eval")], "is inside or contains the checkpoint"),
        (mp3d + ["--ckpt", str(ckpt), "--out-dir", str(runs)], "is inside or contains the checkpoint"),
        (mp3d + ["--ckpt", str(ckpt), "--run-id", "all_256"], inside),  # the training run's own directory
        (mp3d + ["--ckpt", str(ckpt), "--out-dir", os.path.join("runs", "all_256")], inside),  # relative
    ]
    for argv, message in cases:
        with pytest.raises(SystemExit) as exc:
            ev.main(ev.parse_args(argv))
        assert message in str(exc.value.code), argv
        assert sorted(os.listdir(tmp_path)) == ["runs"] and os.listdir(runs / "all_256") == ["checkpoint-6000"]
        assert os.listdir(ckpt) == ["model.safetensors"]


def test_cli_defaults():
    args = ev.parse_args(["--dataset", "mp3d_double_256", "--py-config", "c.py", "--ckpt", "x"])
    assert not (args.save_vis or args.save_ply or args.novel_only or args.align_depth or args.fast_ssim)
    assert args.strict_load is True and args.allow_partial is False and args.allow_extra == []
    assert args.switch == [] and args.out_dir is None and args.run_id is None
