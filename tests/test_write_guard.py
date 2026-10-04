"""Write guard (INV-3) as train.py uses it.

Every write root of a training run (work dir = Accelerate project_dir, logs, dumped
config, switches.json, `latest`, validation output, checkpoint-N dirs, the timestamped
log, the scratch cwd) must be refused when it resolves into a protected tree, including
through a symlink inside the run dir and through a run dir that is itself a symlink
into a workdirs tree; relative writes must land in <run dir>/cwd.
"""

import os

import pytest

from tools.write_guard import (ProtectedPathError, check_write_roots, default_run_dir, enter_scratch_cwd,
                               guard_save_path, is_protected, prepare_run_dir)

ROOT_NAMES = {"work_dir", "logging_dir", "config_dump", "switches", "latest", "validation", "scratch_cwd"}


@pytest.fixture
def protected(tmp_path, monkeypatch):
    """A real directory registered as protected for this test (CYLINDERSPLAT_PROTECTED)."""
    tree = tmp_path / "protected_tree"
    (tree / "checkpoint-36000").mkdir(parents=True)
    monkeypatch.setenv("CYLINDERSPLAT_PROTECTED", str(tree))
    return tree


def _train():
    pytest.importorskip("torch")
    import train
    return train


def _roots(work, cfg="configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py"):
    return _train().write_roots(str(work), cfg)


# ----------------------------------------------------------------------------- the roots train.py declares

def test_write_roots_cover_every_output(tmp_path):
    work = str(tmp_path / "runs" / "r1")
    roots = _roots(work)
    assert set(roots) == ROOT_NAMES
    assert roots["work_dir"] == work
    assert roots["logging_dir"] == os.path.join(work, "logs")  # Accelerate logging_dir
    assert roots["config_dump"] == os.path.join(work, "omni_gs_160x320_mp3d_cylinder_all_256.py")
    assert roots["latest"] == os.path.join(work, "latest")
    assert roots["validation"] == os.path.join(work, "validation")  # not cfg.output_dir/exp_name/validation
    assert roots["scratch_cwd"] == os.path.join(work, "cwd")
    for path in roots.values():
        assert path == work or path.startswith(work + os.sep)


def test_default_work_dir_is_under_the_dev_runs_root():
    from tools import write_guard
    train = _train()
    args = train.parse_args(["--entry", "mp3d_double_160", "--py-config", "c.py", "--run-id", "abc_1"])
    runs_root = os.path.abspath(os.path.expanduser(write_guard.DEFAULT_RUNS_ROOT))
    assert args.work_dir == os.path.join(runs_root, "abc_1") == default_run_dir("abc_1")
    assert not is_protected(args.work_dir)
    with pytest.raises(SystemExit):
        train.parse_args(["--entry", "mp3d_double_160", "--py-config", "c.py"])  # neither --run-id nor --work-dir
    with pytest.raises(ValueError):
        train.parse_args(["--entry", "mp3d_double_160", "--py-config", "c.py", "--run-id", "../escape"])


def test_default_run_dir_is_absolute_with_a_relative_runs_root(tmp_path, monkeypatch):
    from tools import write_guard
    monkeypatch.chdir(tmp_path)
    here = os.getcwd()
    assert default_run_dir("abc_1", root="runs") == os.path.join(here, "runs", "abc_1")
    # CYLINDERSPLAT_RUNS_ROOT is read once, at import, into DEFAULT_RUNS_ROOT.
    monkeypatch.setattr(write_guard, "DEFAULT_RUNS_ROOT", "runs")
    run_dir = default_run_dir("abc_1")
    assert os.path.isabs(run_dir) and run_dir == os.path.join(here, "runs", "abc_1")
    # Writes made after the chdir into <run dir>/cwd still land in the run dir
    # (a relative one would put metrics.json in runs/abc_1/cwd/runs/abc_1/).
    enter_scratch_cwd(prepare_run_dir(run_dir))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        f.write("{}")
    assert os.path.isfile(os.path.join(here, "runs", "abc_1", "metrics.json"))
    assert not os.path.exists(os.path.join(here, "runs", "abc_1", "cwd", "runs"))
    monkeypatch.setattr(write_guard, "DEFAULT_RUNS_ROOT", os.path.join("~", "runs"))
    assert default_run_dir("abc_1") == os.path.join(os.path.expanduser("~"), "runs", "abc_1")


def test_relative_runs_root_gives_an_absolute_train_work_dir(tmp_path, monkeypatch):
    from tools import write_guard
    train = _train()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(write_guard, "DEFAULT_RUNS_ROOT", "runs")
    args = train.parse_args(["--entry", "mp3d_double_160", "--py-config", "c.py", "--run-id", "abc_1"])
    assert args.work_dir == os.path.join(os.getcwd(), "runs", "abc_1")


def test_clean_run_dir_passes_and_scratch_cwd_takes_relative_writes(tmp_path, monkeypatch):
    work = tmp_path / "runs" / "r1"
    roots = _roots(work)
    check_write_roots(roots.values())
    monkeypatch.chdir(tmp_path)
    scratch = prepare_run_dir(str(work), extra_write_roots=list(roots.values()))
    assert scratch == roots["scratch_cwd"] and os.path.isdir(scratch)
    enter_scratch_cwd(scratch)
    with open("debug_render.png", "w") as f:  # what the models' relative PNG writes do
        f.write("x")
    assert os.path.isfile(os.path.join(scratch, "debug_render.png"))
    assert not os.path.exists(tmp_path / "debug_render.png")


# ----------------------------------------------------------------------------- protected trees

@pytest.mark.parametrize("path", [
    "/home/qiwei/program/cylinderSplat/out",
    "/data/qiwei/home_archive/program/cylinderSplat",
    "/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256",
    "/data/qiwei/nips25/workdirs/x",
    "/home/qiwei/ICLR25/workdirs/x",
    "/data/qiwei/ICLR25/workdirs/x",
    "/data/qiwei/nips25/pano_grf/run",
    "/data/qiwei/nips25/360Loc/run",
    "/data/qiwei/nips25/Kansas/run",
    "/data/qiwei/nips25/Kansas",
    "/data/dataset/VIGOR/run",
    "/data/qiwei/cylindersplat_repro/run",
    "/somewhere/else/workdirs/run",
])
def test_protected_work_dir_is_refused(path):
    with pytest.raises(ProtectedPathError):
        check_write_roots(_roots(path).values())


def test_the_old_repro_tree_is_protected_but_repro2_is_not():
    assert is_protected("/data/qiwei/cylindersplat_repro/a")
    assert not is_protected("/data/qiwei/cylindersplat_repro2/a")
    assert not is_protected("/data/qiwei/cylindersplat_dev/runs/a")


def test_the_vigor_dataset_trees_are_protected():
    # vigor_dataloader_double reads root_dir /data/qiwei/nips25/ + city Kansas; the legacy
    # VIGOR loaders read /data/dataset/VIGOR.
    for path in ("/data/qiwei/nips25/Kansas", "/data/qiwei/nips25/Kansas/eval", "/data/dataset/VIGOR",
                 "/data/dataset/VIGOR/Kansas/panorama"):
        assert is_protected(path), path
    assert not is_protected("/data/qiwei/nips25/KansasX/run")  # a prefix only matches whole components


def test_workdirs_component_is_refused(tmp_path):
    with pytest.raises(ProtectedPathError):
        check_write_roots(_roots(tmp_path / "workdirs" / "run").values())


def test_run_dir_symlinked_into_workdirs_is_refused(tmp_path):
    real = tmp_path / "old" / "workdirs" / "omni_gs_all_256"
    real.mkdir(parents=True)
    link = tmp_path / "runs" / "r1"
    link.parent.mkdir()
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ProtectedPathError):
        check_write_roots(_roots(link).values())
    with pytest.raises(ProtectedPathError):
        prepare_run_dir(str(link))
    assert not (real / "cwd").exists()


def test_run_dir_symlinked_into_a_protected_tree_is_refused(tmp_path, protected):
    link = tmp_path / "runs" / "r1"
    link.parent.mkdir()
    link.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ProtectedPathError):
        check_write_roots(_roots(link).values())


@pytest.mark.parametrize("name", sorted(ROOT_NAMES - {"work_dir"}))
def test_symlink_inside_the_run_dir_into_a_protected_tree_is_refused(name, tmp_path, protected):
    work = tmp_path / "runs" / "r1"
    work.mkdir(parents=True)
    roots = _roots(work)
    target = protected / "checkpoint-36000"
    os.symlink(str(target), roots[name])
    with pytest.raises(ProtectedPathError):
        check_write_roots(roots.values())
    with pytest.raises(ProtectedPathError):
        prepare_run_dir(str(work), extra_write_roots=list(roots.values()))


def test_symlink_already_in_the_scratch_dir_is_refused(tmp_path):
    # The models write fixed names (render_*.png) into the cwd; a planted link must not redirect them.
    work = tmp_path / "runs" / "r1"
    scratch = work / "cwd"
    scratch.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere.png"
    os.symlink(str(elsewhere), str(scratch / "render_gt_mp3d_all.png"))
    with pytest.raises(ProtectedPathError, match="symlinks"):
        prepare_run_dir(str(work))
    (scratch / "render_gt_mp3d_all.png").unlink()
    (scratch / "render_gt_mp3d_all.png").write_bytes(b"")  # a plain file left by an earlier run is fine
    assert prepare_run_dir(str(work)) == str(scratch)
    # A hard link would be truncated in place when the model rewrites that name.
    original = tmp_path / "original.png"
    original.write_bytes(b"keep")
    os.link(str(original), str(scratch / "render_fuse_mp3d_all.png"))
    with pytest.raises(ProtectedPathError, match="hard links"):
        prepare_run_dir(str(work))


def test_dangling_symlink_into_workdirs_is_refused(tmp_path):
    work = tmp_path / "runs" / "r1"
    work.mkdir(parents=True)
    roots = _roots(work)
    os.symlink("/data/qiwei/nips25/workdirs/omni_gs_all/latest_target", roots["latest"])
    with pytest.raises(ProtectedPathError):
        check_write_roots(roots.values())


def test_checkpoint_and_log_paths_are_rechecked_at_write_time(tmp_path, protected):
    work = tmp_path / "runs" / "r1"
    work.mkdir(parents=True)
    ok = guard_save_path(str(work / "checkpoint-3000"))
    assert ok == str(work / "checkpoint-3000")
    os.symlink(str(protected / "checkpoint-36000"), str(work / "checkpoint-6000"))
    with pytest.raises(ProtectedPathError):
        guard_save_path(str(work / "checkpoint-6000"))
    os.symlink(str(protected), str(work / "20260930_120000.log"))
    with pytest.raises(ProtectedPathError):
        guard_save_path(str(work / "20260930_120000.log"))



def test_existing_links_are_not_written_through(tmp_path):
    # A reused out/work dir whose metrics.json or log is a hard link (or a symlink) to another file
    # would write into that file when opened in place.
    work = tmp_path / "runs" / "r1"
    work.mkdir(parents=True)
    other = tmp_path / "other.json"
    other.write_text("keep")
    os.link(str(other), str(work / "metrics.json"))
    with pytest.raises(ProtectedPathError, match="hard links"):
        guard_save_path(str(work / "metrics.json"))
    os.symlink(str(other), str(work / "20260930_120000.log"))
    with pytest.raises(ProtectedPathError, match="symlink"):
        guard_save_path(str(work / "20260930_120000.log"))
    # `latest` is a link the trainer replaces (mmengine.utils.symlink removes it first).
    os.symlink(str(work / "checkpoint-3000"), str(work / "latest"))
    assert guard_save_path(str(work / "latest"), allow_symlink=True) == str(work / "latest")
    # A plain file or a directory is fine.
    (work / "config.py").write_text("")
    assert guard_save_path(str(work / "config.py")) == str(work / "config.py")
    assert guard_save_path(str(work)) == str(work)
    assert other.read_text() == "keep"

# ----------------------------------------------------------------------------- through train.main (stubbed)

def test_fake_run_writes_only_under_the_run_dir(tmp_path, monkeypatch):
    from tests.test_entries_table import run_fake
    for row in ("mp3d_double_256", "mp3d_double_160"):
        events, work, _ = run_fake(monkeypatch, tmp_path, row)
        cfg_dir = os.path.realpath(str(tmp_path / "cfg"))
        runs = os.path.realpath(str(tmp_path / "runs"))
        for dirpath, dirnames, filenames in os.walk(str(tmp_path)):
            for name in filenames + dirnames:
                real = os.path.realpath(os.path.join(dirpath, name))
                assert real.startswith(runs + os.sep) or real.startswith(cfg_dir) or real == runs, real
        real_work = os.path.realpath(work)
        for ev in events:
            if ev[0] in ("save_state", "validation_step"):
                assert os.path.realpath(ev[1]).startswith(real_work + os.sep)
            if ev[0] == "forward":
                assert os.path.realpath(ev[4]) == os.path.join(real_work, "cwd")
        assert os.path.realpath(os.path.join(work, "latest")).startswith(real_work + os.sep)


def test_fake_run_refuses_a_protected_root_before_anything_is_written(tmp_path, monkeypatch, protected):
    from tests.test_entries_table import run_fake
    work = tmp_path / "runs" / "r1"
    work.mkdir(parents=True)
    os.symlink(str(protected), str(work / "validation"))
    with pytest.raises(ProtectedPathError):
        run_fake(monkeypatch, tmp_path, "mp3d_double_160", work_dir=work)
    assert sorted(os.listdir(work)) == ["validation"]  # no cwd, no config dump, no log
    assert sorted(os.listdir(protected)) == ["checkpoint-36000"]


def test_fake_run_refuses_a_work_dir_symlinked_into_workdirs(tmp_path, monkeypatch):
    from tests.test_entries_table import run_fake
    real = tmp_path / "old" / "workdirs" / "run"
    real.mkdir(parents=True)
    link = tmp_path / "runs" / "r1"
    link.parent.mkdir()
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ProtectedPathError):
        run_fake(monkeypatch, tmp_path, "mp3d_double_256", work_dir=link)
    assert os.listdir(real) == []


def test_fake_run_refuses_resume_source_inside_the_run_dir(tmp_path, monkeypatch):
    from tests.test_entries_table import run_fake
    from tools.resume import ResumeError
    work = tmp_path / "runs" / "r1"
    (work / "checkpoint-3000").mkdir(parents=True)
    with pytest.raises(ResumeError, match="overlap"):
        run_fake(monkeypatch, tmp_path, "mp3d_double_160",
                 ["--resume-from", str(work / "checkpoint-3000")], work_dir=work)
    assert sorted(os.listdir(work)) == ["checkpoint-3000"]
