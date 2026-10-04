"""tools/prepare_unik3d_depth.py: output layout, view listing and the link-safe writes (no UniK3D needed)."""

import importlib.util
import os

import numpy as np
import pytest

from tests.conftest import REPO_ROOT


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "prepare_unik3d_depth", os.path.join(REPO_ROOT, "tools", "prepare_unik3d_depth.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def test_safe_save_replaces_a_hard_link_without_touching_the_other_name(tmp_path):
    original = tmp_path / "dataset" / "depth_metric.npy"
    original.parent.mkdir()
    np.save(original, np.zeros(3))
    out = tmp_path / "copy" / "depth_metric.npy"
    out.parent.mkdir()
    os.link(original, out)  # e.g. a `cp -al` copy of the dataset
    tool.safe_save(str(out), np.ones(3))
    assert np.load(original).tolist() == [0.0, 0.0, 0.0]
    assert np.load(out).tolist() == [1.0, 1.0, 1.0]
    assert os.stat(out).st_nlink == 1
    assert not [p for p in os.listdir(out.parent) if p.endswith(".tmp")]


def test_safe_save_refuses_a_symlinked_directory(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        tool.safe_save(str(link / "depth_metric.npy"), np.ones(3))
    assert not os.listdir(target)


def test_safe_save_refuses_a_protected_tree():
    from tools.write_guard import ProtectedPathError
    with pytest.raises(ProtectedPathError):
        tool.safe_save("/data/qiwei/nips25/pano_grf/x/depth_metric.npy", np.ones(3))


def test_mp3d_listing_skips_ds_store_like_the_loaders(tmp_path):
    root = tmp_path / "png_render_test_1024x512_seq_len_3_m3d_dist_0.5"
    for scene in ("00", "01"):
        for view in ("00", "01", "02"):
            (root / scene / view).mkdir(parents=True)
    (root / ".DS_Store").write_bytes(b"")
    (root / "00" / ".DS_Store").write_bytes(b"")
    images = tool.mp3d_images(tmp_path, ["test"])
    assert [str(p.relative_to(root)) for p in images] == [
        f"{s}/{v}/rgb.png" for s in ("00", "01") for v in ("00", "01", "02")]


def test_output_layout_matches_the_loaders(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    rgb = data / "png_render_test_1024x512_seq_len_3_m3d_dist_0.5" / "00" / "01" / "rgb.png"
    depth, conf = tool.output_paths("mp3d", rgb, data, out)
    assert depth == out / "png_render_test_1024x512_seq_len_3_m3d_dist_0.5" / "00" / "01" / "depth_metric.npy"
    assert conf == depth.with_name("depth_conf.npy")
    frame = data / "atrium" / "mapping" / "daytime_360_0" / "image" / "0002.jpg"
    depth, conf = tool.output_paths("loc360", frame, data, out)
    assert depth == out / "atrium" / "mapping" / "daytime_360_0" / "depth_metric" / "0002_depth.npy"
    assert conf == depth.with_name("0002_conf.npy")
