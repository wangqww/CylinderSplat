"""Shared pytest setup.

Tests run from the repository root in the `omniscene` environment. CPU-only by
default (CUDA_VISIBLE_DEVICES=""); tests marked `gpu` are skipped without CUDA.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device (skipped on CPU-only runs)")


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
