"""Configs that build on another config differ from it only in the values their headers document (mmengine).

The release/ recipes, the stage-4 512x1024 config and the single-view joint config.
"""

import os

import pytest

from tests.conftest import REPO_ROOT

OMNI = os.path.join("configs", "OmniScene")
ALL_256 = "omni_gs_160x320_mp3d_cylinder_all_256.py"

# release config -> (base config, {dotted key: value} that may differ from the base)
RELEASE = {
    "release/stage1_pixel_256_b4.py": (
        "omni_gs_160x320_mp3d_cylinder_pixel_256.py",
        {
            "lr": 4e-4,
            "save_freq": 1500,
            "val_freq": 1500,
            "resume_from": False,
            "dataset_params.batch_size_train": 4,
            "model.dataset_params.batch_size_train": 4,
        },
    ),
    "release/stage2_volume_256_b4.py": (
        "omni_gs_160x320_mp3d_cylinder_volume_256.py",
        {
            "lr": 4e-4,
            "save_freq": 1500,
            "val_freq": 1500,
            "resume_from": False,
            "dataset_params.batch_size_train": 4,
            "model.dataset_params.batch_size_train": 4,
        },
    ),
    "release/stage3_all_256.py": (ALL_256, {"resume_from": False}),
    "release/loc360_finetune_256.py": (
        "omni_gs_160x320_360Loc_cylinder_all_256.py",
        {"lr": 2e-4, "max_epochs": 10, "seed": 1111, "resume_from": False, "dataset_params.seed": 1111},
    ),
}


def load(rel):
    config = pytest.importorskip("mmengine.config")
    return config.Config.fromfile(os.path.join(REPO_ROOT, OMNI, rel))


def leaves(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out.update(leaves(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: node}


def diff(a, b):
    la, lb = leaves(a.to_dict()), leaves(b.to_dict())
    return {k for k in set(la) | set(lb) if la.get(k, "<missing>") != lb.get(k, "<missing>")}


@pytest.mark.parametrize("rel", sorted(RELEASE))
def test_release_config_changes_only_the_documented_values(rel):
    base_rel, changed = RELEASE[rel]
    cfg, base = load(rel), load(base_rel)
    assert diff(cfg, base) == {k for k, v in changed.items() if leaves(base.to_dict()).get(k) != v}
    assert all(leaves(cfg.to_dict())[k] == v for k, v in changed.items())


def test_stage4_config_is_all_256_at_512x1024():
    cfg, base = load("omni_gs_160x320_mp3d_cylinder_all_512x1024.py"), load(ALL_256)
    assert cfg.resolution == [512, 1024] and cfg.model.pixel_gs.image_height == 512
    assert cfg.model.camera_args.resolution == cfg.model.loss_args.perceptual_resolution == [512, 1024]
    assert (cfg.lr, cfg.max_epochs, cfg.resume_from) == (2e-4, 10, False)
    assert cfg.dataset_params.batch_size_train == cfg.dataset_params.batch_size_test == 1
    # resolution and what is built from it, the batch sizes, max_epochs, resume_from, exp_name
    assert diff(cfg, base) == {
        "resolution",
        "dataset_params.resolution",
        "model.dataset_params.resolution",
        "camera_args.resolution",
        "model.camera_args.resolution",
        "loss_args.perceptual_resolution",
        "model.loss_args.perceptual_resolution",
        "model.pixel_gs.image_height",
        "dataset_params.batch_size_train",
        "dataset_params.batch_size_test",
        "model.dataset_params.batch_size_train",
        "model.dataset_params.batch_size_test",
        "max_epochs",
        "resume_from",
        "exp_name",
    }


def test_single_view_config_is_all_256_with_one_frame():
    # The pixel branch's UNet attends over one view; no parameter shape depends on it.
    cfg, base = load("omni_gs_160x320_mp3d_cylinder_all_256_single.py"), load(ALL_256)
    assert cfg.model.pixel_gs.num_frames == 1 and "num_frames" not in base.model.pixel_gs
    assert diff(cfg, base) == {"exp_name", "resume_from", "model.pixel_gs.num_frames"}
