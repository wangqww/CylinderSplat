"""Run directories and the write guard used by train.py and evaluate.py.

Every file a run writes (checkpoints, the `latest` link, logs, the dumped config,
validation images, PLY files, evaluation output, and the models' own relative
debug images) lands under one run directory. The guard refuses to start when
any write root resolves, through symlinks, into a protected tree: the
author's original code and work directories, the datasets, or another
experiment's output.
"""

import os
import re
import sys

# Default roots on the author's machine; both can be overridden per run.
DEFAULT_RUNS_ROOT = os.environ.get("CYLINDERSPLAT_RUNS_ROOT", "/data/qiwei/cylindersplat_dev/runs")
DEFAULT_REPRO_ROOT = os.environ.get("CYLINDERSPLAT_REPRO_ROOT", "/data/qiwei/cylindersplat_repro2")

# Read-only trees: inputs may be read from here, nothing may be written.
PROTECTED_PREFIXES = (
    "/home/qiwei/program/cylinderSplat",
    "/data/qiwei/home_archive/program/cylinderSplat",
    "/home/qiwei/nips25/workdirs",
    "/data/qiwei/nips25/workdirs",
    "/home/qiwei/ICLR25/workdirs",
    "/data/qiwei/ICLR25/workdirs",
    "/data/qiwei/nips25/pano_grf",
    "/data/qiwei/nips25/360Loc",
    "/data/qiwei/nips25/Kansas",  # vigor_dataloader_double: root_dir /data/qiwei/nips25/ + city Kansas
    "/data/dataset/VIGOR",        # legacy vigor_dataloader / vigor_dataloader_cube root
    "/data/qiwei/cylindersplat_repro",
)
# Any path with a `workdirs` component is somebody's experiment output.
PROTECTED_COMPONENTS = ("workdirs",)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ProtectedPathError(RuntimeError):
    pass


def _extra_protected():
    extra = os.environ.get("CYLINDERSPLAT_PROTECTED", "")
    return tuple(p for p in extra.split(os.pathsep) if p)


def _resolve(path):
    # realpath follows every symlink that exists, including a final link such as
    # `latest`, and normalises the parts that do not exist yet.
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def is_protected(path):
    real = _resolve(path)
    for prefix in PROTECTED_PREFIXES + _extra_protected():
        for candidate in {prefix, _resolve(prefix)}:
            if real == candidate or real.startswith(candidate.rstrip("/") + "/"):
                return True
    parts = real.split(os.sep)
    return any(component in parts for component in PROTECTED_COMPONENTS)


def check_write_roots(paths):
    """Exit-worthy check: raise ProtectedPathError if any path resolves into a protected tree.

    `paths` may contain None entries (unused roots); they are skipped.
    """
    bad = []
    for path in paths:
        if path is None:
            continue
        if is_protected(path):
            bad.append(f"{path} -> {_resolve(path)}")
    if bad:
        raise ProtectedPathError(
            "refusing to write into a protected tree:\n  " + "\n  ".join(bad)
            + "\nPass a --work-dir / --out-dir outside these trees."
        )


def validate_run_id(run_id):
    if not _RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"invalid run id {run_id!r}: use letters, digits, '.', '_' or '-'")
    return run_id


def default_run_dir(run_id, root=None):
    # Absolute, so a relative CYLINDERSPLAT_RUNS_ROOT does not move after the chdir into the scratch cwd.
    return os.path.abspath(os.path.expanduser(os.path.join(root or DEFAULT_RUNS_ROOT, validate_run_id(run_id))))


def prepare_run_dir(run_dir, extra_write_roots=()):
    """Check every write root, create the run directory, and return its scratch cwd path."""
    scratch = os.path.join(run_dir, "cwd")
    check_write_roots([run_dir, scratch, *extra_write_roots])
    os.makedirs(scratch, exist_ok=True)
    # Re-check after creation: a pre-existing symlink inside the run dir could point elsewhere.
    check_write_roots([run_dir, scratch, *extra_write_roots])
    # The models write fixed file names into the cwd; a symlink already sitting in the scratch
    # dir would redirect such a write, so a reused scratch dir must not contain any.
    # A hard link would be opened in place and truncate the shared file, so refuse those as well.
    linked = [name for name in os.listdir(scratch)
              if os.path.islink(os.path.join(scratch, name))
              or (os.path.isfile(os.path.join(scratch, name)) and os.stat(os.path.join(scratch, name)).st_nlink > 1)]
    if linked:
        raise ProtectedPathError(f"scratch dir {scratch} contains symlinks or hard links {sorted(linked)[:5]}; "
                                 "remove them or use a fresh run directory")
    return scratch


def enter_scratch_cwd(scratch):
    """chdir into the run's scratch dir so relative writes by the models stay inside the run.

    The repository root is put on sys.path first, so imports keep working.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.chdir(scratch)
    return repo_root


def guard_save_path(path, allow_symlink=False):
    """Check a single file or directory about to be written (checkpoint, link, PLY).

    Besides the protected-prefix check, an existing symlink (unless the caller replaces the
    link itself, as for `latest`) or a regular file with more than one hard link is refused:
    opening either in place would write into another file.
    """
    check_write_roots([path])
    if os.path.islink(path):
        if not allow_symlink:
            raise ProtectedPathError(f"refusing to write through the symlink {path}")
    elif os.path.isfile(path) and os.stat(path).st_nlink > 1:
        raise ProtectedPathError(f"refusing to write into {path}: it has {os.stat(path).st_nlink} hard links")
    return path
