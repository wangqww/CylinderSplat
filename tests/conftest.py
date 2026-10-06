"""Shared test setup: the repository root on sys.path and a few helpers.

The suite runs on CPU from the repository root, in the training environment:

    CUDA_VISIBLE_DEVICES= python -m pytest tests/

Tests that need torch, mmengine, mmdet3d or the compiled CUDA extensions skip when they are missing.
"""

import glob
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_by_path(rel, name):
    """Import a repository file by path (configs/ is not a package; a site-packages `evaluate` must not shadow ours)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kept_configs():
    """Every model config, repository-relative: configs/OmniScene/*.py, the release/ recipes and the screen/ arms."""
    root = os.path.join(REPO_ROOT, "configs", "OmniScene")
    paths = [path for sub in ("", "release", "screen") for path in glob.glob(os.path.join(root, sub, "*.py"))]
    return sorted(os.path.relpath(path, REPO_ROOT) for path in paths)


def import_models():
    """builder.builder, which registers every model class; skips the test when the model package cannot be
    imported here (torch, mmdet3d or a compiled CUDA extension missing)."""
    try:
        from builder import builder
    except ImportError as e:
        pytest.skip(f"model package not importable: {e}")
    return builder
