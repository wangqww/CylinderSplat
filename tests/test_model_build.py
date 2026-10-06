"""Every model config builds its model on CPU, as train.py and evaluate.py build it (the config's own switches
applied - none for the released configs - from the repository root).

Some constructors allocate on 'cuda' directly (the renderer's background colour, triplane reference points) and
torch.load() checkpoints saved on a GPU; here both stay on the CPU. A config whose pretrained weights are not on
this machine (the PanSplat backbone checkpoint, checkpoints/ of the ablation models, torchvision downloads) is
skipped with the missing path.
"""

import os

import pytest

from tests.conftest import REPO_ROOT, import_models, kept_configs

torch = pytest.importorskip("torch")
from torch.overrides import TorchFunctionMode  # noqa: E402


def is_cuda(device):
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return isinstance(device, str) and device.startswith("cuda")


class CudaOnCpu(TorchFunctionMode):
    """Tensors created with device='cuda', and .cuda() / .to('cuda') moves, stay on the CPU."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if is_cuda(kwargs.get("device")):
            kwargs["device"] = "cpu"
        name = getattr(func, "__name__", None)
        if name == "cuda":
            return args[0]
        if name == "to":
            args = tuple("cpu" if is_cuda(arg) else arg for arg in args)
        return func(*args, **kwargs)


def load_config(rel):
    """The config as train.py / evaluate.py build it: its switches block applied (defaults for every other switch)."""
    config = pytest.importorskip("mmengine.config")
    from tools import switches

    cfg = config.Config.fromfile(os.path.join(REPO_ROOT, rel))
    switches.apply(cfg, [])
    return cfg


@pytest.mark.parametrize("rel", kept_configs())
def test_model_builds_on_cpu(rel, monkeypatch):
    builder = import_models()  # imported outside the mode below, as in a real run
    cfg = load_config(rel)
    load = torch.load

    def load_on_cpu(f, *args, **kwargs):
        if not args:
            kwargs.setdefault("map_location", "cpu")
        return load(f, *args, **kwargs)

    monkeypatch.setattr(torch, "load", load_on_cpu)
    monkeypatch.chdir(REPO_ROOT)  # the ablation models read checkpoints/ relative to the working directory
    try:
        with CudaOnCpu():
            model = builder.build(cfg.model)
    except OSError as e:  # FileNotFoundError, or a download without network
        pytest.skip(f"pretrained weights not available here: {e}")
    assert type(model).__name__ == cfg.model.type
    parameters = list(model.parameters())
    assert parameters and all(p.device.type == "cpu" for p in parameters)
