"""data/loc360_dataloader_double_all_512.py options on a tiny fake 360Loc tree (CPU):

- loc360_interleave: every (sequence, i) pair of an epoch exactly once, the sequences mixed,
  and each sample bit-identical to the one the released per-sequence stream builds for the same
  frames (the uint8 frame cache reproduces to_tensor exactly);
- pcc_reference='depth_anywhere' (eval row loc360_double_256_da): only outputs['depth'] changes;
- the constructor refuses options outside their split; convert_images(strict=True) raises.
The model side of the 360Loc fixes (depth_valid_mask) is checked at the end.
"""

import collections
import json
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL.Image")
L = pytest.importorskip("data.loc360_dataloader_double_all_512")

N_FRAMES = 8
TIMES = 5


def make_sequence(root, scene, name, seed, depth_anywhere=False):
    rng = np.random.RandomState(seed)
    seq = root / scene / "mapping" / name
    (seq / "image").mkdir(parents=True)
    (seq / "depth_metric").mkdir()
    if depth_anywhere:
        (seq / "depthanywhere").mkdir()
    poses = {}
    for k in range(N_FRAMES):
        frame = f"{k:04d}.jpg"
        PIL.fromarray((rng.rand(32, 64, 3) * 255).astype(np.uint8)).save(seq / "image" / frame, quality=95)
        np.save(
            seq / "depth_metric" / frame.replace(".jpg", "_depth.npy"), (rng.rand(1, 16, 32) * 10).astype(np.float32)
        )
        np.save(seq / "depth_metric" / frame.replace(".jpg", "_conf.npy"), rng.rand(1, 16, 32).astype(np.float32))
        if depth_anywhere:
            PIL.fromarray((rng.rand(16, 32) * 255).astype(np.uint8), mode="L").save(
                seq / "depthanywhere" / frame.replace(".jpg", "_depth_anywhere.png")
            )
        pose = np.eye(4)
        pose[0, 3] = 0.5 * k
        poses[frame] = pose.tolist()
    with open(seq / "camera_pose.json", "w") as f:
        json.dump(poses, f)
    return seq


@pytest.fixture
def train_sequences(tmp_path):
    return [
        make_sequence(tmp_path, scene, "daytime_360_0", seed)
        for seed, scene in enumerate(("concourse", "hall", "piatrium"))
    ]


def dataset(stage, sequences, **kwargs):
    ds = L.Dataset360Loc(stage=stage, **kwargs)
    ds.data = list(sequences)  # the real roots are absent here
    ds.times_per_scene = TIMES
    return ds


def recording_two_sample(monkeypatch, fixed=False):
    """Wrap two_sample; record (scene, i) per call. fixed: context [j, j+3] with j = i % (N - 3)."""
    calls = []
    orig = L.two_sample

    def wrapped(scene, extrinsics, times_per_scene, stage="train", i=0):
        calls.append((scene, i))
        if fixed:
            j = i % (extrinsics.shape[0] - 3)
            return torch.tensor((j, j + 3)), torch.arange(j, j + 4)
        return orig(scene, extrinsics, times_per_scene, stage=stage, i=i)

    monkeypatch.setattr(L, "two_sample", wrapped)
    return calls


def collect(ds, calls):
    """[(scene, i), sample] in the order the stream yields them."""
    out = []
    for sample in ds:
        out.append((calls[-1], sample))
    return out


def assert_same(a, b, path="sample"):
    assert type(a) is type(b), path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
    else:
        assert a == b, path


def test_interleave_yields_every_pair_once_and_mixes_the_sequences(monkeypatch, train_sequences):
    calls = recording_two_sample(monkeypatch)
    ds = dataset("train", train_sequences, interleave=True)
    torch.manual_seed(0)
    order = [key for key, _ in collect(ds, calls)]
    assert len(order) == len(ds) == 3 * TIMES
    assert sorted(order) == sorted((f"{s.parts[-3]}-{s.parts[-1]}", i) for s in train_sequences for i in range(TIMES))
    assert set(collections.Counter(scene for scene, _ in order).values()) == {TIMES}
    # The released stream gives TIMES consecutive samples per sequence; here the first TIMES mix them.
    assert len({scene for scene, _ in order[:TIMES]}) > 1


def test_released_stream_is_one_sequence_after_another(monkeypatch, train_sequences):
    calls = recording_two_sample(monkeypatch)
    torch.manual_seed(0)
    order = [key for key, _ in collect(dataset("train", train_sequences), calls)]
    blocks = [order[k : k + TIMES] for k in range(0, len(order), TIMES)]
    assert all(len({scene for scene, _ in block}) == 1 for block in blocks)


def test_interleaved_samples_equal_the_released_samples(monkeypatch, train_sequences):
    calls = recording_two_sample(monkeypatch, fixed=True)
    released = dict(collect(dataset("train", train_sequences), calls))
    interleaved = dict(collect(dataset("train", train_sequences, interleave=True), calls))
    assert released.keys() == interleaved.keys() and len(released) == 3 * TIMES
    for key in released:
        assert_same(interleaved[key], released[key], str(key))


def test_frame_cache_reproduces_to_tensor(train_sequences):
    ds = dataset("train", train_sequences, interleave=True)
    path = train_sequences[1] / "image" / "0003.jpg"
    cached = ds.frame_uint8(path)
    assert cached.dtype == torch.uint8 and cached.shape == (3, ds.height, ds.width)
    assert torch.equal(cached.float().div(255), ds.convert_images([path])[0])
    assert ds.frame_uint8(path) is cached  # decoded once


def test_depth_anywhere_reference_changes_only_outputs_depth(monkeypatch, tmp_path):
    seq = make_sequence(tmp_path, "atrium", "daytime_360_0", 7, depth_anywhere=True)
    calls = recording_two_sample(monkeypatch, fixed=True)
    default = collect(dataset("val", [seq]), calls)
    da = collect(dataset("val", [seq], pcc_reference="depth_anywhere"), calls)
    assert [k for k, _ in default] == [k for k, _ in da]
    ds = dataset("val", [seq])
    for (key, a), (_, b) in zip(default, da):
        j = key[1] % (N_FRAMES - 3)
        pngs = [seq / "depthanywhere" / f"{k:04d}_depth_anywhere.png" for k in range(j, j + 4)]
        expected = ds.convert_images(pngs).clamp(min=0.0)
        assert torch.equal(b["outputs"]["depth"], expected)
        assert not torch.equal(a["outputs"]["depth"], b["outputs"]["depth"])
        a["outputs"].pop("depth"), b["outputs"].pop("depth")
        assert_same(b, a, str(key))


def test_depth_anywhere_loader_factory(monkeypatch):
    da = pytest.importorskip("data.loc360_dataloader_da")
    built = []
    fake = lambda **kwargs: built.append(kwargs) or []
    monkeypatch.setattr(L, "Dataset360Loc", fake)
    monkeypatch.setattr(da, "Dataset360Loc", fake)
    loader = da.load_360Loc_data_da(2, stage="val")
    assert built == [{"stage": "val", "pcc_reference": "depth_anywhere"}]
    reference = L.load_360Loc_data(2, stage="val")
    assert (loader.batch_size, loader.num_workers, loader.persistent_workers) == (
        reference.batch_size,
        reference.num_workers,
        reference.persistent_workers,
    )
    assert loader.generator.initial_seed() == reference.generator.initial_seed() == 3456
    with pytest.raises(ValueError, match="evaluation loader"):
        da.load_360Loc_data_da(2, stage="train")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(stage="val", interleave=True), "train split only"),
        (dict(stage="train", pcc_reference="depth_anywhere"), "evaluation stages only"),
        (dict(stage="val", pcc_reference="gt"), "pcc_reference must be"),
    ],
)
def test_constructor_refuses_options_outside_their_split(kwargs, match):
    with pytest.raises(ValueError, match=match):
        L.Dataset360Loc(**kwargs)


def test_convert_images_strict(train_sequences, capsys):
    ds = dataset("train", train_sequences)
    good = train_sequences[0] / "image" / "0000.jpg"
    missing = train_sequences[0] / "image" / "missing.jpg"
    assert ds.convert_images([good, missing]).shape[0] == 1  # released: printed and skipped
    assert "Error" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        ds.convert_images([good, missing], strict=True)


def test_prior_depth_valid_mask_bounds():
    pan2 = pytest.importorskip("model.omni_gs_cylinder_volume_360loc_pan2")
    cls = pan2.OmniGaussianCylinderVolume360LocPan2
    holder = types.SimpleNamespace(depth_valid_near=0.45, depth_valid_far=50.0)
    depth = torch.tensor([0.0, 0.45, 0.46, 49.9, 50.0, 60.0])
    assert cls._prior_depth_valid(holder, depth).tolist() == [False, False, True, True, False, False]
    import inspect

    params = inspect.signature(cls.__init__).parameters
    assert params["depth_valid_mask"].default is False
    assert (params["depth_valid_near"].default, params["depth_valid_far"].default) == (0.45, 50.0)
