"""The dev-release recipe configs (configs/OmniScene/release/ and the stage-4 512x1024 config) differ from
the configs they build on only in the values the README documents (CPU, mmengine)."""

import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMNI = "configs/OmniScene"

# release config -> (base config, {dotted key: value} that differ from the base)
RELEASE = {
    "release/stage1_pixel_256_b4.py": ("omni_gs_160x320_mp3d_cylinder_pixel_256.py", {
        "lr": 4e-4, "save_freq": 1500, "val_freq": 1500, "resume_from": False,
        "dataset_params.batch_size_train": 4, "model.dataset_params.batch_size_train": 4}),
    "release/stage2_volume_256_b4.py": ("omni_gs_160x320_mp3d_cylinder_volume_256.py", {
        "lr": 4e-4, "save_freq": 1500, "val_freq": 1500, "resume_from": False,
        "dataset_params.batch_size_train": 4, "model.dataset_params.batch_size_train": 4}),
    "release/stage3_all_256.py": ("omni_gs_160x320_mp3d_cylinder_all_256.py", {"resume_from": False}),
    "release/loc360_finetune_256.py": ("omni_gs_160x320_360Loc_cylinder_all_256.py", {
        "lr": 2e-4, "max_epochs": 10, "seed": 1111, "resume_from": False, "dataset_params.seed": 1111}),
}
STAGE4 = "omni_gs_160x320_mp3d_cylinder_all_512x1024.py"


def _load(rel):
    pytest.importorskip("torch")
    config = pytest.importorskip("mmengine.config")
    return config.Config.fromfile(os.path.join(REPO, OMNI, rel))


def _leaves(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out.update(_leaves(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: node}


def _diff(a, b):
    la, lb = _leaves(a.to_dict()), _leaves(b.to_dict())
    return {k for k in set(la) | set(lb) if la.get(k, "<missing>") != lb.get(k, "<missing>")}


@pytest.mark.parametrize("rel", sorted(RELEASE))
def test_release_config_changes_only_the_documented_values(rel):
    base_rel, changed = RELEASE[rel]
    cfg, base = _load(rel), _load(base_rel)
    assert _diff(cfg, base) == {k for k, v in changed.items() if _leaves(base.to_dict()).get(k) != v}
    leaves = _leaves(cfg.to_dict())
    for key, value in changed.items():
        assert leaves[key] == value, key


def test_stage4_config_is_all_256_at_512x1024():
    cfg, base = _load(STAGE4), _load("omni_gs_160x320_mp3d_cylinder_all_256.py")
    assert cfg.resolution == [512, 1024] and cfg.model.pixel_gs.image_height == 512
    assert cfg.model.camera_args.resolution == [512, 1024]
    assert cfg.model.loss_args.perceptual_resolution == [512, 1024]
    assert (cfg.lr, cfg.max_epochs, cfg.resume_from) == (2e-4, 10, False)
    assert cfg.dataset_params.batch_size_train == cfg.dataset_params.batch_size_test == 1
    # resolution and what is built from it, the batch sizes, max_epochs, resume_from, exp_name
    assert _diff(cfg, base) == {
        "resolution", "dataset_params.resolution", "model.dataset_params.resolution",
        "camera_args.resolution", "model.camera_args.resolution",
        "loss_args.perceptual_resolution", "model.loss_args.perceptual_resolution", "model.pixel_gs.image_height",
        "dataset_params.batch_size_train", "dataset_params.batch_size_test",
        "model.dataset_params.batch_size_train", "model.dataset_params.batch_size_test",
        "max_epochs", "resume_from", "exp_name"}
