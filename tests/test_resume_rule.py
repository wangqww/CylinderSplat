"""Resume rule (plan A1): weights-only load of model.safetensors under frozen allowed lists.

The runtime lists live in configs/entries.py (TRANSFERS); tests/fixtures/allowed_keys.json
is the frozen copy they must equal. Synthetic safetensors files cover a missing file,
zero matching tensors, extra names, missing names, dtype and shape mismatches, tied
weights, and equality with the legacy name+shape filter when the rule passes.
"""

import importlib.util
import json
import os
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "tests", "fixtures", "allowed_keys.json")
EMPTY = {"allowed_missing": [], "allowed_extra": []}
MONO_DEPTH = {"allowed_missing": [], "allowed_extra": ["pixel_gs.mono_depth.*"]}


def _transfers():
    spec = importlib.util.spec_from_file_location("cylindersplat_entries_resume_test",
                                                  os.path.join(REPO, "configs", "entries.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TRANSFERS


def test_runtime_lists_equal_frozen_fixture():
    with open(FIXTURE) as f:
        frozen = json.load(f)["transfers"]
    assert _transfers() == frozen


def test_frozen_lists_are_the_plan_lists():
    t = _transfers()
    assert t["exact"] == EMPTY
    assert t["stage1_to_stage2"] == MONO_DEPTH
    assert t["double_pixel_to_single_pixel"] == MONO_DEPTH
    for name in ("stage2_to_stage3", "stage3_to_stage4_512", "mp3d_all_256_to_loc360_pan2", "d3_single_view",
                 "pan2_single_view"):
        assert t[name] == EMPTY, name
    # D7: extra is empty; allowed-missing is frozen by the integrator from the implemented module.
    assert t["d7_visibility_softmax"]["allowed_extra"] == []
    assert all(n.startswith("volume_gs.") and "gaussian_to_color_vis." in n
               for n in t["d7_visibility_softmax"]["allowed_missing"])


def test_runtime_does_not_read_tests():
    with open(os.path.join(REPO, "tools", "resume.py")) as f:
        assert "tests/" not in f.read()
    with open(os.path.join(REPO, "train.py")) as f:
        assert "allowed_keys.json" not in f.read()


def test_name_allowed_patterns():
    from tools.resume import name_allowed
    assert name_allowed("pixel_gs.mono_depth.a.weight", ["pixel_gs.mono_depth.*"])
    assert not name_allowed("pixel_gs.mono_depthX.weight", ["pixel_gs.mono_depth.*"])
    assert not name_allowed("pixel_gs.head.weight", ["pixel_gs.mono_depth.*"])
    assert name_allowed("a.b", ["a.b"]) and not name_allowed("a.bc", ["a.b"])
    assert not name_allowed("anything", [])


# ----------------------------------------------------------------------------- synthetic checkpoints

@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _tiny(torch, seed):
    torch.manual_seed(seed)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pixel_gs = torch.nn.Linear(3, 4)
            self.volume_gs = torch.nn.Linear(4, 2)
            self.register_buffer("scale", torch.rand(2))

    return Tiny()


def _save(torch, path, tensors):
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)
    return path


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_exact_checkpoint_loads(torch, tmp_path):
    from tools.resume import load_weights_only
    src, dst = _tiny(torch, 0), _tiny(torch, 1)
    ckpt_dir = tmp_path / "checkpoint-36000"
    _save(torch, str(ckpt_dir / "model.safetensors"), src.state_dict())
    for path in (str(ckpt_dir), str(ckpt_dir / "model.safetensors")):  # dir or file
        dst = _tiny(torch, 1)
        report = load_weights_only(dst, path, **EMPTY)
        assert report["file"] == str(ckpt_dir / "model.safetensors")
        assert sorted(report["matched"]) == sorted(src.state_dict())
        assert report["extra"] == report["missing"] == report["aliased"] == []
        for k, v in src.state_dict().items():
            assert torch.equal(dst.state_dict()[k], v)


def test_missing_file(torch, tmp_path):
    from tools.resume import ResumeError, load_weights_only
    with pytest.raises(ResumeError, match="not found"):
        load_weights_only(_tiny(torch, 0), str(tmp_path / "nope" / "model.safetensors"))
    (tmp_path / "empty_ckpt").mkdir()
    with pytest.raises(ResumeError, match="not found"):
        load_weights_only(_tiny(torch, 0), str(tmp_path / "empty_ckpt"))


def test_zero_match(torch, tmp_path):
    from tools.resume import ResumeError, load_weights_only
    path = _save(torch, str(tmp_path / "c" / "model.safetensors"), {"other.weight": torch.zeros(2)})
    model = _tiny(torch, 0)
    before = _state(model)
    with pytest.raises(ResumeError, match="zero"):
        load_weights_only(model, path, **MONO_DEPTH)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())


def test_extra_names(torch, tmp_path):
    from tools.resume import ResumeError, load_weights_only
    src = _tiny(torch, 0)
    tensors = dict(src.state_dict())
    tensors["pixel_gs.mono_depth.encoder.weight"] = torch.ones(3)
    tensors["pixel_gs.mono_depth.encoder.bias"] = torch.ones(1)
    path = _save(torch, str(tmp_path / "c" / "model.safetensors"), tensors)
    model = _tiny(torch, 1)
    before = _state(model)
    with pytest.raises(ResumeError, match="extra names") as err:
        load_weights_only(model, path, **EMPTY)
    assert "pixel_gs.mono_depth.encoder.weight" in str(err.value)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())  # nothing half-loaded
    # stage 1 -> 2 / double -> single pixel: the mono_depth tensors are allowed extra.
    report = load_weights_only(model, path, **_transfers()["stage1_to_stage2"])
    assert report["extra"] == ["pixel_gs.mono_depth.encoder.bias", "pixel_gs.mono_depth.encoder.weight"]
    assert all(torch.equal(model.state_dict()[k], v) for k, v in src.state_dict().items())
    # Any other extra name is still refused under that transfer.
    tensors["volume_gs.unknown"] = torch.ones(1)
    path = _save(torch, str(tmp_path / "d" / "model.safetensors"), tensors)
    with pytest.raises(ResumeError, match="volume_gs.unknown"):
        load_weights_only(_tiny(torch, 1), path, **_transfers()["stage1_to_stage2"])


def test_missing_names(torch, tmp_path):
    from tools.resume import ResumeError, load_weights_only
    src = _tiny(torch, 0)
    tensors = {k: v for k, v in src.state_dict().items() if k != "volume_gs.bias"}
    path = _save(torch, str(tmp_path / "c" / "model.safetensors"), tensors)
    with pytest.raises(ResumeError, match="missing") as err:
        load_weights_only(_tiny(torch, 1), path, **EMPTY)
    assert "volume_gs.bias" in str(err.value)
    # An allowed-missing name keeps the model's own initial value; everything else is loaded.
    model = _tiny(torch, 1)
    own_bias = model.volume_gs.bias.detach().clone()
    report = load_weights_only(model, path, allowed_missing=["volume_gs.bias"], allowed_extra=[])
    assert report["missing"] == ["volume_gs.bias"]
    assert torch.equal(model.volume_gs.bias, own_bias)
    for k, v in tensors.items():
        assert torch.equal(model.state_dict()[k], v)


def test_dtype_mismatch(torch, tmp_path):
    from tools.resume import ResumeError, load_weights_only
    tensors = dict(_tiny(torch, 0).state_dict())
    tensors["pixel_gs.weight"] = tensors["pixel_gs.weight"].double()
    path = _save(torch, str(tmp_path / "c" / "model.safetensors"), tensors)
    with pytest.raises(ResumeError, match="dtype") as err:
        load_weights_only(_tiny(torch, 1), path, **EMPTY)
    assert "pixel_gs.weight" in str(err.value)


def test_shape_mismatch(torch, tmp_path):
    from tools.resume import ResumeError, load_weights_only
    tensors = dict(_tiny(torch, 0).state_dict())
    tensors["volume_gs.weight"] = torch.zeros(2, 5)
    path = _save(torch, str(tmp_path / "c" / "model.safetensors"), tensors)
    model = _tiny(torch, 1)
    before = _state(model)
    with pytest.raises(ResumeError, match="shape") as err:
        load_weights_only(model, path, **MONO_DEPTH)
    assert "volume_gs.weight" in str(err.value)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())


def test_tied_weights_are_not_missing(torch, tmp_path):
    from tools.resume import load_weights_only

    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(2, 2)
            self.b = torch.nn.Linear(2, 2)
            self.b.weight = self.a.weight

    torch.manual_seed(0)
    src = Tied()
    # Accelerate's safetensors writer keeps one name per storage.
    tensors = {k: v for k, v in src.state_dict().items() if k != "b.weight"}
    path = _save(torch, str(tmp_path / "c" / "model.safetensors"), tensors)
    torch.manual_seed(1)
    dst = Tied()
    report = load_weights_only(dst, path, **EMPTY)
    assert report["aliased"] == ["b.weight"] and report["missing"] == []
    assert torch.equal(dst.a.weight, src.a.weight) and torch.equal(dst.b.weight, src.a.weight)


def test_same_state_as_legacy_filter_when_the_rule_passes(torch, tmp_path):
    from tests.legacy_ref import train_ref as legacy
    from tools.resume import load_weights_only
    src = _tiny(torch, 0)
    tensors = dict(src.state_dict())
    tensors["pixel_gs.mono_depth.w"] = torch.ones(4)
    path = _save(torch, str(tmp_path / "c" / "model.safetensors"), tensors)
    new, old = _tiny(torch, 1), _tiny(torch, 1)
    load_weights_only(new, path, **_transfers()["stage1_to_stage2"])
    legacy.resume(old, types.SimpleNamespace(resume_from=path), types.SimpleNamespace(print=lambda *a: None))
    assert new.state_dict().keys() == old.state_dict().keys()
    for k in new.state_dict():
        assert torch.equal(new.state_dict()[k], old.state_dict()[k]), k


def test_source_and_work_dir_overlap_is_refused(tmp_path):
    from tools.resume import ResumeError, refuse_source_inside
    work = tmp_path / "runs" / "r1"
    (work / "checkpoint-3000").mkdir(parents=True)
    with pytest.raises(ResumeError, match="overlap"):
        refuse_source_inside(str(work / "checkpoint-3000" / "model.safetensors"), str(work))
    # The work dir equal to, or nested inside, the checkpoint dir overlaps it too.
    with pytest.raises(ResumeError, match="overlap"):
        refuse_source_inside(str(work / "checkpoint-3000"), str(work / "checkpoint-3000"))
    with pytest.raises(ResumeError, match="overlap"):
        refuse_source_inside(str(work / "checkpoint-3000" / "model.safetensors"), str(work / "checkpoint-3000"))
    with pytest.raises(ResumeError, match="overlap"):
        refuse_source_inside(str(work / "checkpoint-3000"), str(work / "checkpoint-3000" / "sub"))
    # Through a symlinked work dir as well.
    link = tmp_path / "link_run"
    link.symlink_to(work, target_is_directory=True)
    with pytest.raises(ResumeError):
        refuse_source_inside(str(work / "checkpoint-3000"), str(link))
    # A sibling run (or a protected tree) is a read-only input and fine.
    refuse_source_inside(str(tmp_path / "runs" / "r0" / "checkpoint-3000"), str(work))
    refuse_source_inside("/home/qiwei/nips25/workdirs/x/checkpoint-36000/model.safetensors", str(work))
    refuse_source_inside(str(tmp_path / "runs" / "r1b" / "model.safetensors"), str(work))


def test_transfer_is_explicit_and_defaults_to_exact():
    pytest.importorskip("torch")
    import train
    args = train.parse_args(["--entry", "mp3d_double_256", "--py-config", "x.py", "--run-id", "r1"])
    assert args.transfer == "exact"
    assert train.TRANSFERS == _transfers()
    args = train.parse_args(["--entry", "mp3d_double_256", "--py-config", "x.py", "--run-id", "r1",
                             "--transfer", "stage1_to_stage2"])
    assert args.transfer == "stage1_to_stage2"
    with pytest.raises(SystemExit):
        train.parse_args(["--entry", "mp3d_double_256", "--py-config", "x.py", "--run-id", "r1", "--transfer", "nope"])
    with pytest.raises(SystemExit):
        train.parse_args(["--entry", "mp3d_double_256", "--py-config", "x.py", "--run-id", "r1",
                          "--resume-from", "a", "--no-resume"])
