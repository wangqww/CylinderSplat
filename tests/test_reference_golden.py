"""repro/reference_metrics.json holds the frozen references exactly (plan INV-1, V1),
and `evaluate.py --check-ref` compares a run with them exactly.

REF-T1 is copied from the 2026-09-29 capture on 4090
(.claude-implement/reviews/20260929T1530Z-cylindersplat-dev-cleanup/attempt-3/
evidence-remote-4090.txt, section 6). REF-T2, REF-T3, REF-T4-pixel and REF-T4-volume
are copied from the legacy-script runs of the 2026-09-29/30 checkpoint sweep
(.claude-implement/work/sweep/ref_lines.txt, logs under /data/qiwei/cylindersplat_repro
and /data/qiwei/cylindersplat_repro2 on 4090). The reference file tests are pure
Python; the --check-ref tests load evaluate.py (torch) and run no model: they feed
synthetic metrics.json content to the comparison.
"""

import importlib.util
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_FILE = os.path.join(REPO_ROOT, "repro", "reference_metrics.json")
ALLOWED_KEYS_FILE = os.path.join(REPO_ROOT, "tests", "fixtures", "allowed_keys.json")

REF_T1_LINES = [
    "m3d_0.1 psnr: 29.309, wspsnr: 28.321, ssim: 0.9093, lpips: 0.0702, pcc: 0.9402, abs: 0.2041, silog: 12.5354, rmse: 0.4725, delta1: 0.6408, delta2: 0.9825, delta3: 0.9980, depthsim: 0.0735",
    "m3d_0.25 psnr: 28.866, wspsnr: 27.845, ssim: 0.9059, lpips: 0.0790, pcc: 0.9414, abs: 0.1909, silog: 12.4760, rmse: 0.4586, delta1: 0.7004, delta2: 0.9821, delta3: 0.9976, depthsim: 0.0733",
    "m3d_0.5 psnr: 29.587, wspsnr: 28.590, ssim: 0.9195, lpips: 0.0809, pcc: 0.9211, abs: 0.1933, silog: 13.7999, rmse: 0.4591, delta1: 0.7091, delta2: 0.9689, delta3: 0.9959, depthsim: 0.0831",
    "m3d_0.75 psnr: 26.878, wspsnr: 25.912, ssim: 0.8773, lpips: 0.1292, pcc: 0.8931, abs: 0.2253, silog: 18.9647, rmse: 0.5210, delta1: 0.6706, delta2: 0.9241, delta3: 0.9755, depthsim: 0.1317",
    "m3d_1.0 psnr: 24.598, wspsnr: 23.653, ssim: 0.8346, lpips: 0.1801, pcc: 0.8528, abs: 0.2611, silog: 25.0637, rmse: 0.5775, delta1: 0.6125, delta2: 0.8643, delta3: 0.9428, depthsim: 0.1425",
    "residential_0.15 psnr: 28.745, wspsnr: 28.172, ssim: 0.8668, lpips: 0.1567, pcc: 0.8162, abs: 0.0000, silog: 0.0000, rmse: 0.0000, delta1: 0.0000, delta2: 0.0000, delta3: 0.0000, depthsim: 0.2276",
    "replica_0.5 psnr: 31.284, wspsnr: 30.290, ssim: 0.9577, lpips: 0.0641, pcc: 0.8594, abs: 0.0000, silog: 0.0000, rmse: 0.0000, delta1: 0.0000, delta2: 0.0000, delta3: 0.0000, depthsim: 0.0511",
]
# T2_D_single_pixel_256 checkpoint-33000 (/data/qiwei/cylindersplat_repro2/logs/T2_D_single_pixel_256_ck33000.out)
REF_T2_LINES = [
    "m3d_0.1 psnr: 31.978, wspsnr: 31.030, ssim: 0.9497, lpips: 0.0538, pcc: 0.9328, abs: 0.2063, silog: 13.4253, rmse: 0.4840, delta1: 0.6536, delta2: 0.9799, delta3: 0.9980, depthsim: 0.1054",
    "m3d_0.25 psnr: 29.550, wspsnr: 28.689, ssim: 0.9056, lpips: 0.0806, pcc: 0.9298, abs: 0.2104, silog: 14.8150, rmse: 0.4877, delta1: 0.6552, delta2: 0.9706, delta3: 0.9959, depthsim: 0.1047",
    "m3d_0.5 psnr: 27.543, wspsnr: 26.782, ssim: 0.8464, lpips: 0.1299, pcc: 0.9020, abs: 0.2174, silog: 16.7055, rmse: 0.4975, delta1: 0.6599, delta2: 0.9528, delta3: 0.9889, depthsim: 0.1095",
    "m3d_0.75 psnr: 25.615, wspsnr: 25.075, ssim: 0.7944, lpips: 0.1948, pcc: 0.8659, abs: 0.2379, silog: 20.0229, rmse: 0.5517, delta1: 0.6430, delta2: 0.9156, delta3: 0.9579, depthsim: 0.1299",
    "m3d_1.0 psnr: 24.447, wspsnr: 23.910, ssim: 0.7513, lpips: 0.2450, pcc: 0.8423, abs: 0.2600, silog: 24.7591, rmse: 0.6113, delta1: 0.6167, delta2: 0.8673, delta3: 0.9184, depthsim: 0.1546",
    "residential_0.15 psnr: 29.974, wspsnr: 29.593, ssim: 0.8585, lpips: 0.1425, pcc: 0.8173, abs: 0.0000, silog: 0.0000, rmse: 0.0000, delta1: 0.0000, delta2: 0.0000, delta3: 0.0000, depthsim: 0.2345",
    "replica_0.5 psnr: 27.454, wspsnr: 26.746, ssim: 0.8948, lpips: 0.1190, pcc: 0.8312, abs: 0.0000, silog: 0.0000, rmse: 0.0000, delta1: 0.0000, delta2: 0.0000, delta3: 0.0000, depthsim: 0.0912",
]
# T3_D_360Loc_all_256 checkpoint-24000 (/data/qiwei/cylindersplat_repro/logs/T3_D_360Loc_all_256_ck24000.out):
# the legacy Total line after its wall-time prefix "Finish evluation (771 s). ".
REF_T3_LEGACY_TOTAL = "Total psnr: 27.244, ws_psnr: 28.205, ssim: 0.8708, lpips: 0.1134, pcc: 0.9897."
REF_T3_EVALUATE_PY_TOTAL = "Total psnr: 27.244, wspsnr: 28.205, ssim: 0.8708, lpips: 0.1134, pcc: 0.9897."
# T4_H_pixel_256 checkpoint-36000 (/data/qiwei/cylindersplat_repro/logs/T4_H_pixel_256_ck36000.out)
REF_T4_PIXEL_LINES = [
    "m3d_0.1 psnr: 29.159, wspsnr: 28.235, ssim: 0.9134, lpips: 0.0695, pcc: 0.9376, abs: 0.2120, silog: 12.2868, rmse: 0.4866, delta1: 0.6246, delta2: 0.9829, delta3: 0.9981, depthsim: 0.0782",
    "m3d_0.25 psnr: 28.987, wspsnr: 28.008, ssim: 0.9114, lpips: 0.0765, pcc: 0.9387, abs: 0.2004, silog: 12.5033, rmse: 0.4755, delta1: 0.6741, delta2: 0.9801, delta3: 0.9974, depthsim: 0.0852",
    "m3d_0.5 psnr: 29.211, wspsnr: 28.176, ssim: 0.9162, lpips: 0.0826, pcc: 0.9184, abs: 0.2063, silog: 14.4819, rmse: 0.4787, delta1: 0.6803, delta2: 0.9618, delta3: 0.9947, depthsim: 0.0939",
    "m3d_0.75 psnr: 26.157, wspsnr: 25.297, ssim: 0.8680, lpips: 0.1379, pcc: 0.8777, abs: 0.2408, silog: 21.6039, rmse: 0.5358, delta1: 0.6430, delta2: 0.9062, delta3: 0.9697, depthsim: 0.1501",
    "m3d_1.0 psnr: 23.951, wspsnr: 23.224, ssim: 0.8309, lpips: 0.1852, pcc: 0.8402, abs: 0.2802, silog: 32.6251, rmse: 0.6156, delta1: 0.6047, delta2: 0.8525, delta3: 0.9290, depthsim: 0.1656",
    "residential_0.15 psnr: 28.354, wspsnr: 27.793, ssim: 0.8621, lpips: 0.1590, pcc: 0.8054, abs: 0.0000, silog: 0.0000, rmse: 0.0000, delta1: 0.0000, delta2: 0.0000, delta3: 0.0000, depthsim: 0.2234",
    "replica_0.5 psnr: 30.969, wspsnr: 29.976, ssim: 0.9558, lpips: 0.0658, pcc: 0.8568, abs: 0.0000, silog: 0.0000, rmse: 0.0000, delta1: 0.0000, delta2: 0.0000, delta3: 0.0000, depthsim: 0.0601",
]
# T4_H_volume_256 checkpoint-36000 (/data/qiwei/cylindersplat_repro/logs/T4_H_volume_256_ck36000.out)
REF_T4_VOLUME_LINES = [
    "m3d_0.1 psnr: 26.901, wspsnr: 26.202, ssim: 0.8430, lpips: 0.1793, pcc: 0.9253, abs: 0.2530, silog: 14.6011, rmse: 0.5657, delta1: 0.5645, delta2: 0.9464, delta3: 0.9901, depthsim: 0.1651",
    "m3d_0.25 psnr: 26.016, wspsnr: 25.471, ssim: 0.8292, lpips: 0.2013, pcc: 0.9335, abs: 0.2174, silog: 14.5265, rmse: 0.5085, delta1: 0.6597, delta2: 0.9542, delta3: 0.9898, depthsim: 0.1331",
    "m3d_0.5 psnr: 25.415, wspsnr: 24.832, ssim: 0.8127, lpips: 0.2260, pcc: 0.9044, abs: 0.2024, silog: 16.6309, rmse: 0.4832, delta1: 0.7075, delta2: 0.9426, delta3: 0.9866, depthsim: 0.1336",
    "m3d_0.75 psnr: 23.408, wspsnr: 22.904, ssim: 0.7560, lpips: 0.2886, pcc: 0.8646, abs: 0.2433, silog: 22.9057, rmse: 0.5493, delta1: 0.6572, delta2: 0.8878, delta3: 0.9585, depthsim: 0.1689",
    "m3d_1.0 psnr: 21.797, wspsnr: 21.428, ssim: 0.7042, lpips: 0.3401, pcc: 0.8234, abs: 0.3255, silog: 30.9966, rmse: 0.6922, delta1: 0.5660, delta2: 0.7943, delta3: 0.8832, depthsim: 0.1930",
    "residential_0.15 psnr: 26.711, wspsnr: 27.360, ssim: 0.8398, lpips: 0.2930, pcc: 0.7987, abs: 0.0000, silog: 0.0000, rmse: 0.0000, delta1: 0.0000, delta2: 0.0000, delta3: 0.0000, depthsim: 0.2306",
    "replica_0.5 psnr: 24.937, wspsnr: 24.248, ssim: 0.8405, lpips: 0.2174, pcc: 0.8837, abs: 0.0000, silog: 0.0000, rmse: 0.0000, delta1: 0.0000, delta2: 0.0000, delta3: 0.0000, depthsim: 0.1543",
]

COLUMNS = ["psnr", "wspsnr", "ssim", "lpips", "pcc", "abs", "silog", "rmse",
           "delta1", "delta2", "delta3", "depthsim"]
TOTAL_COLUMNS = ["psnr", "wspsnr", "ssim", "lpips", "pcc"]
# Plan INV-1: the WS-PSNR of each scene (the paper's PSNR column).
WSPSNR_BY_SCENE = {
    "m3d_0.1": "28.321", "m3d_0.25": "27.845", "m3d_0.5": "28.590", "m3d_0.75": "25.912",
    "m3d_1.0": "23.653", "residential_0.15": "28.172", "replica_0.5": "30.290",
}

# name -> (evaluate.py dataset row, config, checkpoint, legacy script)
PROVENANCE = {
    "REF-T1": ("mp3d_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py",
               "/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256/checkpoint-48000",
               "evaluate_mp3d_double_256.py"),
    "REF-T2": ("mp3d_single_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_single.py",
               "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_single_pixel_256/checkpoint-33000",
               "evaluate_mp3d_single_256.py"),
    "REF-T3": ("loc360_double_256", "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py",
               "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_cylinder_double_all_256/checkpoint-24000",
               "evaluate_360Loc_double_256.py"),
    "REF-T4-pixel": ("mp3d_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py",
                     "/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256/checkpoint-36000",
                     "evaluate_mp3d_double_256.py"),
    "REF-T4-volume": ("mp3d_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py",
                      "/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256/checkpoint-36000",
                      "evaluate_mp3d_double_256.py"),
}
PER_SCENE_REFS = {
    "REF-T1": REF_T1_LINES,
    "REF-T2": REF_T2_LINES,
    "REF-T4-pixel": REF_T4_PIXEL_LINES,
    "REF-T4-volume": REF_T4_VOLUME_LINES,
}


def _load():
    with open(REFERENCE_FILE, encoding="utf-8") as f:
        return json.load(f)


def _split(line):
    scene, rest = line.split(" ", 1)
    return scene, [tuple(pair.split(": ")) for pair in rest.split(", ")]


def _eval_entries():
    spec = importlib.util.spec_from_file_location(
        "eval_entries_for_golden", os.path.join(REPO_ROOT, "configs", "eval_entries.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EVAL_ENTRIES


# ----------------------------------------------------------------------------
# The frozen references
# ----------------------------------------------------------------------------

def test_every_reference_is_frozen_and_none_pending():
    refs = _load()
    frozen = [k for k, v in refs.items() if isinstance(v, dict) and v.get("status") == "frozen"]
    assert frozen == ["REF-T1", "REF-T2", "REF-T3", "REF-T4-pixel", "REF-T4-volume"]
    assert "pending" not in refs
    assert set(refs) == {"schema", "description", *frozen}


def test_ref_t1_rows_exactly_as_captured():
    ref = _load()["REF-T1"]
    assert ref["status"] == "frozen"
    assert ref["columns"] == COLUMNS
    scenes = [line.split(" ", 1)[0] for line in REF_T1_LINES]
    assert ref["scene_order"] == scenes
    assert list(ref["rows"]) == scenes
    for line in REF_T1_LINES:
        scene, pairs = _split(line)
        row = ref["rows"][scene]
        assert row["printed"] == line
        assert list(row["columns"].items()) == pairs
        assert [k for k, _ in pairs] == COLUMNS
        assert row["metrics"] == {k: float(v) for k, v in pairs}
        assert list(row["metrics"]) == COLUMNS
        assert row["columns"]["wspsnr"] == WSPSNR_BY_SCENE[scene]
    for scene in ("residential_0.15", "replica_0.5"):
        for k in ("abs", "silog", "rmse", "delta1", "delta2", "delta3"):
            assert ref["rows"][scene]["columns"][k] == "0.0000"


@pytest.mark.parametrize("name", sorted(PER_SCENE_REFS))
def test_per_scene_rows_exactly_as_captured(name):
    ref = _load()[name]
    lines = PER_SCENE_REFS[name]
    assert ref["status"] == "frozen"
    assert ref["columns"] == COLUMNS
    scenes = [line.split(" ", 1)[0] for line in lines]
    assert scenes == list(WSPSNR_BY_SCENE)
    assert ref["scene_order"] == scenes
    assert list(ref["rows"]) == scenes
    assert "total" not in ref
    for line in lines:
        scene, pairs = _split(line)
        row = ref["rows"][scene]
        assert row["printed"] == line
        assert list(row["columns"].items()) == pairs
        assert [k for k, _ in pairs] == COLUMNS
        assert row["metrics"] == {k: float(v) for k, v in pairs}
        assert list(row["metrics"]) == COLUMNS
    for scene in ("residential_0.15", "replica_0.5"):  # no GT depth there
        for k in ("abs", "silog", "rmse", "delta1", "delta2", "delta3"):
            assert ref["rows"][scene]["columns"][k] == "0.0000"


@pytest.mark.parametrize("name", sorted(PER_SCENE_REFS))
def test_line_format_reprints_every_row(name):
    ref = _load()[name]
    for scene, row in ref["rows"].items():
        values = [row["metrics"][k] for k in COLUMNS]
        assert ref["line_format"].format(scene, *values).strip() == row["printed"]


@pytest.mark.parametrize("name", sorted(PER_SCENE_REFS))
def test_line_format_is_the_evaluate_py_scene_line(name):
    ref = _load()[name]
    entry = _eval_entries()[ref["eval_entry"]["dataset"]]
    assert entry["scene_line"] == ref["line_format"]
    assert tuple(entry["scene_keys"]) == tuple(ref["scene_order"])
    assert tuple(entry["scene_metrics"]) == tuple(COLUMNS)


def test_ref_t3_total_exactly_as_captured():
    ref = _load()["REF-T3"]
    assert ref["status"] == "frozen"
    assert "rows" not in ref and "scene_order" not in ref
    total = ref["total"]
    assert total["printed"] == REF_T3_LEGACY_TOTAL
    assert total["evaluate_py_line"] == REF_T3_EVALUATE_PY_TOTAL
    # The two differ only in the WS-PSNR label (legacy `ws_psnr`, evaluate.py `wspsnr`).
    assert REF_T3_LEGACY_TOTAL.replace(" ws_psnr: ", " wspsnr: ") == REF_T3_EVALUATE_PY_TOTAL
    assert ref["legacy_labels"] == {"wspsnr": "ws_psnr"}
    assert ref["columns"] == TOTAL_COLUMNS
    pairs = [tuple(p.split(": ")) for p in REF_T3_EVALUATE_PY_TOTAL[len("Total "):].rstrip(".").split(", ")]
    assert list(total["columns"].items()) == pairs
    assert total["metrics"] == {k: float(v) for k, v in pairs}
    assert list(total["metrics"]) == TOTAL_COLUMNS
    assert total["columns"]["wspsnr"] == "28.205"


def test_ref_t3_total_is_the_line_evaluate_py_prints():
    ref = _load()["REF-T3"]
    entry = _eval_entries()["loc360_double_256"]
    assert entry["scene_keys"] is None and entry["scene_line"] is None  # totals only, as legacy
    assert tuple(entry["total_metrics"]) == tuple(TOTAL_COLUMNS)
    assert entry["legacy_labels"] == ref["legacy_labels"]
    assert entry["total_line"] == "Finish evluation ({:d} s). " + ref["line_format"]
    values = [ref["total"]["metrics"][k] for k in TOTAL_COLUMNS]
    printed = entry["total_line"].format(771, *values)
    assert printed == "Finish evluation (771 s). " + REF_T3_EVALUATE_PY_TOTAL
    assert ref["legacy_line_format"].format(*values) == REF_T3_LEGACY_TOTAL


def test_ref_t1_provenance():
    ref = _load()["REF-T1"]
    prov = ref["provenance"]
    assert ref["eval_entry"] == {"dataset": "mp3d_double_256", "novel_only": False}
    assert prov["checkpoint"] == \
        "/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256/checkpoint-48000"
    assert prov["config"] == "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py"
    assert prov["script"].startswith("evaluate_mp3d_double_256.py")
    assert prov["date"] == "2026-09-29"
    assert prov["code_commit"].startswith("f7b20b9")
    assert "all three frames" in prov["protocol"]


@pytest.mark.parametrize("name", sorted(PROVENANCE))
def test_provenance(name):
    dataset, config, checkpoint, script = PROVENANCE[name]
    ref = _load()[name]
    prov = ref["provenance"]
    assert ref["eval_entry"] == {"dataset": dataset, "novel_only": False}
    assert prov["config"] == config
    assert prov["checkpoint"] == checkpoint
    assert prov["script"] == f"{script} (legacy/{script})"
    assert prov["code_commit"] == "f7b20b9950dd9577a321edef642179f65a67d910"
    assert os.path.isfile(os.path.join(REPO_ROOT, config))
    assert os.path.isfile(os.path.join(REPO_ROOT, "legacy", script))
    assert _eval_entries()[dataset]["legacy_script"] == f"legacy/{script}"


def test_reference_allowed_extra():
    # REF-T4-pixel's checkpoint holds the 333 pixel_gs.mono_depth.* tensors of the frozen
    # stage1_to_stage2 transfer. REF-T2's checkpoint holds none (safetensors header read on
    # 4090, 2026-09-30: 580 tensors, 0 mono_depth), so it needs no allowed_extra.
    with open(ALLOWED_KEYS_FILE, encoding="utf-8") as f:
        transfers = json.load(f)["transfers"]
    mono = transfers["stage1_to_stage2"]["allowed_extra"]
    assert mono == ["pixel_gs.mono_depth.*"] == transfers["double_pixel_to_single_pixel"]["allowed_extra"]
    refs = _load()
    allowed = {name: refs[name]["provenance"].get("allowed_extra") for name in PROVENANCE}
    assert allowed == {"REF-T1": None, "REF-T2": None, "REF-T3": None, "REF-T4-pixel": mono,
                       "REF-T4-volume": None}
    assert "333 pixel_gs.mono_depth.*" in refs["REF-T4-pixel"]["provenance"]["allowed_extra_note"]


# ----------------------------------------------------------------------------
# evaluate.py --check-ref on synthetic metrics.json content (no model)
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ev():
    pytest.importorskip("torch")
    # By path: a site-packages `evaluate` (Hugging Face) must not shadow the repo script.
    spec = importlib.util.spec_from_file_location("cylindersplat_evaluate_refcheck",
                                                  os.path.join(REPO_ROOT, "evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics_json(tmp_path, name, scenes=None, total_line=None, dataset=None, config=None, iteration=None,
                  novel_only=False):
    """A metrics.json as evaluate.py writes it, for the pairing of reference `name` unless overridden."""
    ref_dataset, ref_config, ref_checkpoint, _ = PROVENANCE[name]
    ref_iteration = int(ref_checkpoint.rsplit("checkpoint-", 1)[1])
    record = {
        "dataset": dataset or ref_dataset,
        "py_config": os.path.join("/somewhere", "configs", "OmniScene", config or os.path.basename(ref_config)),
        "checkpoint": {"path": f"/ckpts/checkpoint-{iteration or ref_iteration}/model.safetensors",
                       "iteration": iteration or ref_iteration},
        "protocol": {"novel_only": novel_only},
        "scenes": scenes or {},
        "total": {"num_batches": 3, "line": total_line} if total_line is not None else {},
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(record, indent=2))
    return json.loads(path.read_text())


def _scene_rows(lines):
    # evaluate.py keeps the logger message's leading space in `line`.
    return {line.split(" ", 1)[0]: {"num_samples": 5, "line": " " + line} for line in lines}


def _bump_last_digit(line, label):
    head, rest = line.split(f"{label}: ", 1)
    value, tail = rest.split(",", 1) if "," in rest else (rest.rstrip("."), "")
    digit = str((int(value[-1]) + 1) % 10)
    return f"{head}{label}: {value[:-1]}{digit}" + (f",{tail}" if tail else ".")


def _check(ev, name, record):
    logs = []
    result = ev.check_reference(name, record, logs.append)
    return result, "\n".join(logs)


@pytest.mark.parametrize("name", sorted(PER_SCENE_REFS))
def test_check_ref_per_scene_match(ev, tmp_path, name):
    record = _metrics_json(tmp_path, name, scenes=_scene_rows(PER_SCENE_REFS[name]),
                           total_line="Finish evluation (48 s). Total psnr: 1.000")  # Total not compared here
    result, log = _check(ev, name, record)
    assert result == {"reference": name, "ok": True, "compared": 7, "mismatches": [], "pairing": []}
    assert "7/7 lines identical" in log
    ev.finish_reference_check(result)  # no exit


@pytest.mark.parametrize("name", sorted(PER_SCENE_REFS))
def test_check_ref_per_scene_mismatch_exits_non_zero(ev, tmp_path, name):
    lines = list(PER_SCENE_REFS[name])
    lines[4] = _bump_last_digit(lines[4], "lpips")  # m3d_1.0, one printed digit
    scenes = _scene_rows(lines)
    del scenes["replica_0.5"]  # a scene the run did not report
    record = _metrics_json(tmp_path, name, scenes=scenes)
    result, log = _check(ev, name, record)
    assert result["ok"] is False
    assert result["mismatches"] == ["m3d_1.0", "replica_0.5"]
    assert result["pairing"] == []
    assert "m3d_1.0: MISMATCH" in log and "5/7 lines identical" in log
    with pytest.raises(SystemExit) as exc:
        ev.finish_reference_check(result)
    assert exc.value.code not in (0, None)


def test_check_ref_per_scene_needs_every_digit(ev, tmp_path):
    # wspsnr at printed precision: 23.653 vs 23.654 is a mismatch, no tolerance.
    lines = list(REF_T1_LINES)
    lines[4] = lines[4].replace("wspsnr: 23.653", "wspsnr: 23.654")
    result, _ = _check(ev, "REF-T1", _metrics_json(tmp_path, "REF-T1", scenes=_scene_rows(lines)))
    assert result["mismatches"] == ["m3d_1.0"] and result["ok"] is False


@pytest.mark.parametrize("seconds", [771, 5])
def test_check_ref_totals_only_match(ev, tmp_path, seconds):
    line = f"Finish evluation ({seconds} s). " + REF_T3_EVALUATE_PY_TOTAL
    record = _metrics_json(tmp_path, "REF-T3", total_line=line)  # scenes {} as for loc360_double_256
    result, log = _check(ev, "REF-T3", record)
    assert result == {"reference": "REF-T3", "ok": True, "compared": 1, "mismatches": [], "pairing": []}
    assert "Total: match" in log
    ev.finish_reference_check(result)


def test_check_ref_totals_only_uses_the_line_evaluate_py_prints(ev, tmp_path):
    ref = ev.load_reference("REF-T3")
    entry = ev.EVAL_ENTRIES["loc360_double_256"]
    values = [ref["total"]["metrics"][k] for k in entry["total_metrics"]]
    line = entry["total_line"].format(771, *values)
    result, _ = _check(ev, "REF-T3", _metrics_json(tmp_path, "REF-T3", total_line=line))
    assert result["ok"] is True


@pytest.mark.parametrize("line", [
    "Finish evluation (771 s). " + REF_T3_LEGACY_TOTAL,  # the legacy label is not what evaluate.py prints
    "Finish evluation (771 s). " + _bump_last_digit(REF_T3_EVALUATE_PY_TOTAL, "pcc"),
    "Finish evluation (771 s). " + REF_T3_EVALUATE_PY_TOTAL.replace("wspsnr: 28.205", "wspsnr: 28.204"),
    None,  # no Total line
])
def test_check_ref_totals_only_mismatch_exits_non_zero(ev, tmp_path, line):
    record = _metrics_json(tmp_path, "REF-T3", total_line=line)
    result, log = _check(ev, "REF-T3", record)
    assert result["ok"] is False
    assert result["mismatches"] == ["Total"]
    assert "Total: MISMATCH" in log
    with pytest.raises(SystemExit) as exc:
        ev.finish_reference_check(result)
    assert exc.value.code not in (0, None)


@pytest.mark.parametrize("override, fragment", [
    ({"dataset": "mp3d_single_256"}, "dataset mp3d_single_256, reference mp3d_double_256"),
    ({"config": "omni_gs_160x320_mp3d_cylinder_pixel_256.py"}, "config omni_gs_160x320_mp3d_cylinder_pixel_256.py"),
    ({"iteration": 45000}, "checkpoint iteration 45000, reference 48000"),
    ({"novel_only": True}, "novel_only True, reference False"),
])
def test_check_ref_wrong_pairing_fails_even_with_identical_lines(ev, tmp_path, override, fragment):
    record = _metrics_json(tmp_path, "REF-T1", scenes=_scene_rows(REF_T1_LINES), **override)
    result, log = _check(ev, "REF-T1", record)
    assert result["mismatches"] == []
    assert result["ok"] is False
    assert len(result["pairing"]) == 1 and fragment in result["pairing"][0]
    assert "WARNING" in log
    with pytest.raises(SystemExit) as exc:
        ev.finish_reference_check(result)
    assert exc.value.code not in (0, None)


def test_check_ref_pairing_of_each_reference(ev):
    refs = _load()
    for name, (dataset, config, checkpoint, _) in PROVENANCE.items():
        iteration = int(checkpoint.rsplit("checkpoint-", 1)[1])
        assert ev.reference_pairing_mismatches(refs[name], dataset, False, "/x/" + os.path.basename(config),
                                               iteration) == []
    # REF-T4-pixel and REF-T4-volume share row and iteration; only the config tells them apart.
    problems = ev.reference_pairing_mismatches(
        refs["REF-T4-volume"], "mp3d_double_256", False,
        "/x/omni_gs_160x320_mp3d_cylinder_pixel_256.py", 36000)
    assert problems == ["config omni_gs_160x320_mp3d_cylinder_pixel_256.py, "
                        "reference omni_gs_160x320_mp3d_cylinder_volume_256.py"]
    # A checkpoint not named checkpoint-N has no iteration and cannot pass.
    assert ev.reference_pairing_mismatches(refs["REF-T3"], "loc360_double_256", False,
                                           "/x/omni_gs_160x320_360Loc_cylinder_all_256.py", None)


def test_load_reference_rejects_unknown_and_unfrozen(ev, tmp_path):
    for name in ("REF-T9", "schema", "description", "pending", "REF-T4"):
        with pytest.raises(ValueError):
            ev.load_reference(name)
    path = tmp_path / "refs.json"
    path.write_text(json.dumps({"REF-X": {"status": "pending"}, "REF-Y": {"status": "frozen"}}))
    with pytest.raises(ValueError):
        ev.load_reference("REF-X", path=str(path))
    with pytest.raises(ValueError):  # frozen but neither rows nor a Total line
        ev.load_reference("REF-Y", path=str(path))
    result, _ = _check(ev, "REF-T9", _metrics_json(tmp_path, "REF-T1"))
    assert result["ok"] is False and result["mismatches"] == ["no frozen reference"]


def test_total_line_body(ev):
    assert ev.total_line_body("Finish evluation (12 s). Total psnr: 1.000.") == "Total psnr: 1.000."
    assert ev.total_line_body(" Total psnr: 1.000. ") == "Total psnr: 1.000."


def _main_args(ev, dataset, config, ckpt, out_dir, ref_name, extra=()):
    return ev.parse_args(["--dataset", dataset, "--py-config", os.path.join(REPO_ROOT, config),
                          "--ckpt", str(ckpt), "--out-dir", str(out_dir), "--check-ref", ref_name, *extra])


@pytest.mark.parametrize("ref_name, dataset, config, iteration, extra", [
    ("REF-T2", "mp3d_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py", 48000, ()),
    ("REF-T1", "mp3d_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py", 47000, ()),
    ("REF-T1", "mp3d_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py", 48000,
     ("--novel-only",)),
    ("REF-T4-volume", "mp3d_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py", 36000,
     ()),
    ("REF-T3", "loc360_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py", 24000, ()),
    ("REF-T9", "mp3d_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py", 48000, ()),
])
def test_main_refuses_a_wrong_pairing_before_any_work(ev, tmp_path, ref_name, dataset, config, iteration, extra):
    ckpt_dir = tmp_path / f"checkpoint-{iteration}"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").write_bytes(b"")
    out_dir = tmp_path / "out"
    cwd = os.getcwd()
    try:
        with pytest.raises(SystemExit) as exc:
            ev.main(_main_args(ev, dataset, config, ckpt_dir, out_dir, ref_name, extra))
        assert exc.value.code not in (0, None)
        assert "--check-ref" in str(exc.value.code)
        assert not out_dir.exists()
    finally:
        os.chdir(cwd)


def test_allowed_extra_patterns(ev):
    refs = {name: ev.load_reference(name) for name in PROVENANCE}
    mono = ["pixel_gs.mono_depth.*"]
    assert ev.allowed_extra_patterns([], refs["REF-T4-pixel"]) == mono
    for name in ("REF-T1", "REF-T2", "REF-T3", "REF-T4-volume"):
        assert ev.allowed_extra_patterns([], refs[name]) == []
    assert ev.allowed_extra_patterns([]) == []
    # the command-line patterns first, then the reference's, without repeats
    assert ev.allowed_extra_patterns(["a.w", "pixel_gs.mono_depth.*", "a.w"], refs["REF-T4-pixel"]) == ["a.w", *mono]
    for bad in ("", "*", "**", "pixel_gs.*.weight"):
        with pytest.raises(ValueError):
            ev.allowed_extra_patterns([bad])


def _pixel_checkpoint(parent):
    ckpt_dir = parent / "checkpoint-36000"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model.safetensors").write_bytes(b"")
    return ckpt_dir


def test_main_applies_the_reference_allowed_extra(ev, tmp_path, monkeypatch):
    # The documented REF-T4-pixel command (no load flag) reaches the load with the reference's patterns.
    real = ev.allowed_extra_patterns
    calls = []

    def spy(patterns, ref=None):
        calls.append(real(patterns, ref))
        return calls[-1]

    def stop(*args, **kwargs):  # prepare_run_dir: stop before any directory is created
        raise ev.ProtectedPathError("stop here")

    monkeypatch.setattr(ev, "allowed_extra_patterns", spy)
    monkeypatch.setattr(ev, "prepare_run_dir", stop)
    config = "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py"
    cwd = os.getcwd()
    try:
        for i, (extra, expected) in enumerate((((), ["pixel_gs.mono_depth.*"]),
                                               (("--allow-extra", "x.w"), ["x.w", "pixel_gs.mono_depth.*"]))):
            with pytest.raises(SystemExit) as exc:
                ev.main(_main_args(ev, "mp3d_double_256", config, _pixel_checkpoint(tmp_path / f"run{i}"),
                                   tmp_path / "out", "REF-T4-pixel", extra))
            assert "stop here" in str(exc.value.code)
            assert len(calls) == i + 1 and calls[-1] == expected
        assert not (tmp_path / "out").exists()
    finally:
        os.chdir(cwd)


def test_main_refuses_a_bad_allow_extra_before_any_work(ev, tmp_path):
    out_dir = tmp_path / "out"
    cwd = os.getcwd()
    try:
        with pytest.raises(SystemExit) as exc:
            ev.main(_main_args(ev, "mp3d_double_256", "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py",
                               _pixel_checkpoint(tmp_path), out_dir, "REF-T4-pixel", ("--allow-extra", "*")))
        assert "allowed-extra pattern '*'" in str(exc.value.code)
        assert not out_dir.exists()
    finally:
        os.chdir(cwd)
