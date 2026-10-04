"""data/paths.py: the dataset roots follow CYLINDERSPLAT_PANO_GRF / CYLINDERSPLAT_360LOC, and the loaders
read them (scripts/reproduce.sh relies on both)."""

import importlib
from pathlib import Path

import pytest

import data.paths as paths


def test_roots_follow_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CYLINDERSPLAT_PANO_GRF", str(tmp_path / "pano_grf"))
    monkeypatch.setenv("CYLINDERSPLAT_360LOC", str(tmp_path / "360Loc"))
    try:
        importlib.reload(paths)
        assert paths.PANO_GRF_ROOT == tmp_path / "pano_grf"
        assert paths.LOC360_ROOT == tmp_path / "360Loc"
    finally:
        monkeypatch.undo()
        importlib.reload(paths)


def test_defaults_without_environment(monkeypatch):
    monkeypatch.delenv("CYLINDERSPLAT_PANO_GRF", raising=False)
    monkeypatch.delenv("CYLINDERSPLAT_360LOC", raising=False)
    try:
        importlib.reload(paths)
        assert paths.PANO_GRF_ROOT == Path("/data/qiwei/nips25/pano_grf")
        assert paths.LOC360_ROOT == Path("/data/qiwei/nips25/360Loc")
    finally:
        monkeypatch.undo()
        importlib.reload(paths)


@pytest.mark.parametrize(
    "module", ["mp3d_dataloader_double_256", "mp3d_dataloader_single_256", "mp3d_dataloader_double_512"]
)
def test_mp3d_loaders_use_pano_grf_root(module):
    pytest.importorskip("torch")
    loader = importlib.import_module(f"data.{module}")
    assert loader.roots == [paths.PANO_GRF_ROOT]


def test_loc360_loader_scans_loc360_root(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    L = importlib.import_module("data.loc360_dataloader_double_all_512")
    for folder in ("atrium/mapping/daytime_360_0", "atrium/query_360/night_360_1", "atrium/mapping/pinhole_0"):
        (tmp_path / folder).mkdir(parents=True)
    monkeypatch.setattr(L, "LOC360_ROOT", tmp_path)
    ds = L.Dataset360Loc(stage="val")
    assert sorted(p.relative_to(tmp_path).as_posix() for p in ds.data) == [
        "atrium/mapping/daytime_360_0",
        "atrium/query_360/night_360_1",
    ]
