"""Checks that the frozen copies in this package are verbatim f7b20b9 code.

Not a frozen copy itself: the helpers the identity tests use to compare every copied
block with `git show f7b20b9:<path>`. The tests skip when git or the commit is not
available; a path that does not exist at the commit fails.
"""

import functools
import os
import re
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMIT = "f7b20b9"


@functools.lru_cache(maxsize=None)
def _skip_reason():
    if shutil.which("git") is None:
        return "git not available"
    res = subprocess.run(["git", "-C", REPO_ROOT, "cat-file", "-e", f"{COMMIT}^{{commit}}"],
                         capture_output=True)
    return None if res.returncode == 0 else f"commit {COMMIT} not available in this clone"


@functools.lru_cache(maxsize=None)
def _show(path):
    res = subprocess.run(["git", "-C", REPO_ROOT, "show", f"{COMMIT}:{path}"],
                         capture_output=True, text=True)
    return tuple(res.stdout.split("\n")) if res.returncode == 0 else None


def git_show(path):
    """The lines of `git show f7b20b9:<path>` (split on '\\n'); skips without git or the commit."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    lines = _show(path)
    assert lines is not None, f"{path} does not exist at {COMMIT}"
    return list(lines)


def source_lines(name):
    """The lines of the frozen-copy module tests/legacy_ref/<name>.py, split like git_show
    (read as text: the module is not imported)."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")) as f:
        return f.read().split("\n")


def assert_verbatim(lines, start, path, a, b):
    """lines[start:] begins with lines a-b (1-based, inclusive) of f7b20b9:<path>.
    Returns the index of the first line after the copy."""
    legacy = git_show(path)
    n_lines = len(legacy) - (legacy[-1] == "")  # a final newline leaves an empty last item
    assert 1 <= a <= b <= n_lines, f"{path}:{a}-{b} is outside the file at {COMMIT} ({n_lines} lines)"
    legacy = legacy[a - 1:b]
    copy = lines[start:start + len(legacy)]
    for k, (x, y) in enumerate(zip(copy, legacy)):
        assert x == y, f"line {start + k + 1} differs from {path}:{a + k}\n  copy:   {x!r}\n  legacy: {y!r}"
    assert len(copy) == len(legacy), f"the copy of {path}:{a}-{b} is cut short"
    return start + len(legacy)


def stray_lines(lines, covered, start, stop, allowed=()):
    """The non-blank lines in [start, stop) that are neither in `covered` (indices of markers
    and copied lines) nor match one of the `allowed` regexes, as (line number, text)."""
    return [(i + 1, lines[i]) for i in range(start, stop)
            if i not in covered and lines[i].strip() and not any(re.fullmatch(p, lines[i]) for p in allowed)]
