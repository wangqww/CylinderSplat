"""Weights-only resume (tools/resume.py) under the transfer lists of configs/entries.py.

Synthetic safetensors files cover a missing file, zero matching tensors, extra and missing names, dtype and
shape mismatches and tied weights; a refused checkpoint leaves the model untouched. Also: a run never saves
next to, inside or around the checkpoint it starts from.
"""

import os

import pytest

from tests.conftest import load_by_path
from tools.resume import ResumeError, name_allowed, refuse_source_inside

TRANSFERS = load_by_path("configs/entries.py", "cylindersplat_entries_resume_test").TRANSFERS
EMPTY = dict(allowed_missing=[], allowed_extra=[])
MONO_DEPTH = dict(allowed_missing=[], allowed_extra=["pixel_gs.mono_depth.*"])


def test_transfer_lists():
    # The pixel models' depth network (pixel_gs.mono_depth.*, 333 tensors) is the only allowed difference.
    assert TRANSFERS == {
        "exact": EMPTY,
        "stage1_to_stage2": MONO_DEPTH,
        "double_pixel_to_single_pixel": MONO_DEPTH,
        "stage2_to_stage3": EMPTY,
        "stage3_to_stage4_512": EMPTY,
        "mp3d_all_256_to_loc360_pan2": EMPTY,
    }


def test_name_allowed():
    assert name_allowed("pixel_gs.mono_depth.a.weight", ["pixel_gs.mono_depth.*"])
    assert not name_allowed("pixel_gs.mono_depthX.weight", ["pixel_gs.mono_depth.*"])
    assert not name_allowed("pixel_gs.head.weight", ["pixel_gs.mono_depth.*"])
    assert name_allowed("a.b", ["a.b"]) and not name_allowed("a.bc", ["a.b"])
    assert not name_allowed("anything", [])


def test_a_run_never_saves_around_its_source_checkpoint(tmp_path):
    work = tmp_path / "runs" / "r1"
    ckpt = work / "checkpoint-3000"
    ckpt.mkdir(parents=True)
    for source, work_dir in [
        (ckpt / "model.safetensors", work),  # inside the work dir
        (ckpt, ckpt),  # the work dir is the checkpoint dir
        (ckpt / "model.safetensors", ckpt),
        (ckpt, ckpt / "sub"),  # the work dir is inside the checkpoint dir
    ]:
        with pytest.raises(ResumeError, match="overlap"):
            refuse_source_inside(str(source), str(work_dir))
    link = tmp_path / "link_run"
    link.symlink_to(work, target_is_directory=True)
    with pytest.raises(ResumeError, match="overlap"):
        refuse_source_inside(str(ckpt), str(link))  # through a symlinked work dir
    # A sibling run is a read-only input.
    refuse_source_inside(str(tmp_path / "runs" / "r0" / "checkpoint-3000"), str(work))
    refuse_source_inside(str(tmp_path / "runs" / "r1b" / "model.safetensors"), str(work))


# ----------------------------------------------------------------------------- synthetic checkpoints


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def tiny(torch, seed):
    torch.manual_seed(seed)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pixel_gs = torch.nn.Linear(3, 4)
            self.volume_gs = torch.nn.Linear(4, 2)
            self.register_buffer("scale", torch.rand(2))

    return Tiny()


def save(path, tensors):
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)
    return path


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_state(torch, model, expected):
    state = model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in expected.items())


def test_an_exact_checkpoint_loads_from_its_dir_or_file(torch, tmp_path):
    from tools.resume import load_weights_only

    src = tiny(torch, 0)
    ckpt = tmp_path / "checkpoint-36000"
    save(str(ckpt / "model.safetensors"), src.state_dict())
    for path in (ckpt, ckpt / "model.safetensors"):
        dst = tiny(torch, 1)
        report = load_weights_only(dst, str(path), **EMPTY)
        assert report["file"] == str(ckpt / "model.safetensors")
        assert sorted(report["matched"]) == sorted(src.state_dict())
        assert report["extra"] == report["missing"] == report["aliased"] == []
        assert_state(torch, dst, src.state_dict())


def test_a_missing_file_or_zero_matches_is_refused(torch, tmp_path):
    from tools.resume import load_weights_only

    (tmp_path / "empty_ckpt").mkdir()
    for path in (tmp_path / "nope" / "model.safetensors", tmp_path / "empty_ckpt"):
        with pytest.raises(ResumeError, match="not found"):
            load_weights_only(tiny(torch, 0), str(path))
    path = save(str(tmp_path / "c" / "model.safetensors"), {"other.weight": torch.zeros(2)})
    model = tiny(torch, 0)
    before = snapshot(model)
    with pytest.raises(ResumeError, match="zero"):
        load_weights_only(model, path, **MONO_DEPTH)
    assert_state(torch, model, before)


def test_extra_names_are_refused_unless_the_transfer_allows_them(torch, tmp_path):
    from tools.resume import load_weights_only

    src = tiny(torch, 0)
    tensors = dict(src.state_dict())
    tensors["pixel_gs.mono_depth.encoder.weight"] = torch.ones(3)
    tensors["pixel_gs.mono_depth.encoder.bias"] = torch.ones(1)
    path = save(str(tmp_path / "c" / "model.safetensors"), tensors)
    model = tiny(torch, 1)
    before = snapshot(model)
    with pytest.raises(ResumeError, match="extra names") as err:
        load_weights_only(model, path, **EMPTY)
    assert "pixel_gs.mono_depth.encoder.weight" in str(err.value)
    assert_state(torch, model, before)  # nothing half-loaded
    report = load_weights_only(model, path, **TRANSFERS["stage1_to_stage2"])
    assert report["extra"] == ["pixel_gs.mono_depth.encoder.bias", "pixel_gs.mono_depth.encoder.weight"]
    assert_state(torch, model, src.state_dict())
    # Any other extra name is still refused under that transfer.
    tensors["volume_gs.unknown"] = torch.ones(1)
    path = save(str(tmp_path / "d" / "model.safetensors"), tensors)
    with pytest.raises(ResumeError, match="volume_gs.unknown"):
        load_weights_only(tiny(torch, 1), path, **TRANSFERS["stage1_to_stage2"])


def test_missing_names_are_refused_unless_allowed(torch, tmp_path):
    from tools.resume import load_weights_only

    tensors = {k: v for k, v in tiny(torch, 0).state_dict().items() if k != "volume_gs.bias"}
    path = save(str(tmp_path / "c" / "model.safetensors"), tensors)
    with pytest.raises(ResumeError, match="missing") as err:
        load_weights_only(tiny(torch, 1), path, **EMPTY)
    assert "volume_gs.bias" in str(err.value)
    # An allowed-missing name keeps the model's own value; everything else is loaded.
    model = tiny(torch, 1)
    own_bias = model.volume_gs.bias.detach().clone()
    report = load_weights_only(model, path, allowed_missing=["volume_gs.bias"], allowed_extra=[])
    assert report["missing"] == ["volume_gs.bias"]
    assert torch.equal(model.volume_gs.bias, own_bias)
    assert_state(torch, model, tensors)


@pytest.mark.parametrize("name,kind", [("pixel_gs.weight", "dtype"), ("volume_gs.weight", "shape")])
def test_dtype_and_shape_mismatches_are_refused(name, kind, torch, tmp_path):
    from tools.resume import load_weights_only

    tensors = dict(tiny(torch, 0).state_dict())
    tensors[name] = tensors[name].double() if kind == "dtype" else torch.zeros(2, 5)
    path = save(str(tmp_path / "c" / "model.safetensors"), tensors)
    model = tiny(torch, 1)
    before = snapshot(model)
    with pytest.raises(ResumeError, match=kind) as err:
        load_weights_only(model, path, **MONO_DEPTH)
    assert name in str(err.value)
    assert_state(torch, model, before)


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
    path = save(
        str(tmp_path / "c" / "model.safetensors"), {k: v for k, v in src.state_dict().items() if k != "b.weight"}
    )
    torch.manual_seed(1)
    dst = Tied()
    report = load_weights_only(dst, path, **EMPTY)
    assert report["aliased"] == ["b.weight"] and report["missing"] == []
    assert torch.equal(dst.a.weight, src.a.weight) and torch.equal(dst.b.weight, src.a.weight)
