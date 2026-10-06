"""Volume sparsity budget (switch volume_sparsity, model/volume/sparsity.py) on CPU: the budget's value and gradient,
the ramp, the model mixin in the joint model, train.py's SparsityMonitor, and the sparsity configs (s3k_*) and their
weight transfer. The real forward (CUDA rasteriser) is not exercised here."""

import inspect
import json
import os
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mmengine")


def _sparsity():
    from tests.conftest import import_models

    import_models()
    from model.volume import sparsity

    return sparsity


def _shell(cls_name, sparsity_on=True, sparsity_args=None):
    """A model shell carrying only what the mixin reads, initialised through the mixin's init."""
    from tests.conftest import import_models

    import_models()
    import model as models

    cls = getattr(models, cls_name)
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    obj._init_volume_sparsity(sparsity_on, sparsity_args)
    return obj


def _opacity_volume(alive_logits, dead_logits, rows=2):
    """[(rows), N, 14] Gaussians whose opacity = sigmoid(x) of a leaf x (alive ones above 1/255, dead ones below)."""
    x = torch.tensor(alive_logits + dead_logits, dtype=torch.float32).repeat(rows, 1).requires_grad_(True)
    other = torch.rand(rows, x.shape[1], 14, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
    g = torch.cat([other[..., :6], torch.sigmoid(x)[..., None], other[..., 7:]], dim=-1)
    return g, x, other


# ----------------------------------------------------------------------------- the budget


def test_the_budget_is_zero_while_the_share_is_under_rho():
    sp = _sparsity()
    g, x, _ = _opacity_volume([0.0, 1.0, 2.0], [-8.0])
    loss, stats = sp.volume_sparsity_budget(g, "train", 0, dict(sp.SPARSITY_DEFAULTS), rho_override=0.80)  # B = 0.75
    assert stats == dict(volume_share=0.75, volume_share_target=0.80)
    loss.backward()
    assert float(loss) == 0.0 and torch.count_nonzero(x.grad) == 0


@pytest.mark.parametrize("weight", [0.02, 1.0, 25.0])
def test_the_gradient_is_uniform_on_the_rendered_logits_and_nothing_else(weight):
    sp = _sparsity()
    g, x, other = _opacity_volume([-4.0, 0.0, 1.5, 6.0, 12.0], [-6.0, -9.0], rows=3)
    args = dict(sp.SPARSITY_DEFAULTS, budget_weight=weight)
    slots = 3 * 7
    share = 15 / slots
    loss, stats = sp.volume_sparsity_budget(g, "train", 10, args, rho_override=0.40)
    assert stats["volume_share"] == pytest.approx(share)
    assert float(loss) == pytest.approx(weight * (share - 0.40) ** 2, rel=1e-5)
    loss.backward()
    expected = 2 * weight * (share - 0.40) / slots
    assert torch.allclose(x.grad[:, :5], torch.full_like(x.grad[:, :5], expected), rtol=1e-4)
    assert torch.count_nonzero(x.grad[:, 5:]) == 0  # opacity below 1/255: not rendered, not pushed
    assert torch.count_nonzero(other.grad) == 0  # every other channel untouched


def test_no_term_off_the_training_split_and_the_ramp():
    sp = _sparsity()
    g, _, _ = _opacity_volume([0.0, 1.0], [-9.0])
    with pytest.raises(ValueError, match="budget_start"):  # no default start: each config sets its init's ceiling
        sp.volume_sparsity_budget(g, "train", 0, dict(sp.SPARSITY_DEFAULTS))
    args = dict(sp.SPARSITY_DEFAULTS, budget_start=0.85)
    for split in ("val", "test"):
        loss, stats = sp.volume_sparsity_budget(g, split, 5000, args)
        assert loss is None and stats == dict(volume_share=pytest.approx(2 / 3))
    assert sp.volume_sparsity_budget(g, "train", 0, args)[1]["volume_share_target"] == 0.85
    assert sp.volume_sparsity_budget(g, "train", 1800, args)[1]["volume_share_target"] == pytest.approx(0.60)
    assert sp.volume_sparsity_budget(g, "train", 9000, args)[1]["volume_share_target"] == 0.35
    held = dict(args, budget_start=0.40, ramp_steps=1000)
    assert sp.sparsity_rho(500, held) == pytest.approx(0.375) and sp.sparsity_rho(7000, held) == 0.35


# ----------------------------------------------------------------------------- the joint model


def test_the_joint_model_declares_and_checks_the_switch():
    sp = _sparsity()
    import model as models

    cls_name = "OmniGaussianCylinderAll"
    cls = getattr(models, cls_name)
    assert issubclass(cls, sp.VolumeSparsityMixin)
    params = inspect.signature(cls.__init__).parameters
    assert params["volume_sparsity"].default is False and params["sparsity_args"].default is None
    with pytest.raises(ValueError, match="sparsity_args: unknown keys"):
        _shell(cls_name, sparsity_args=dict(budgett=0.3))


def test_the_stage2_volume_model_has_no_budget():
    import model as models

    sp = _sparsity()
    assert not issubclass(models.OmniGaussianCylinderVolume, sp.VolumeSparsityMixin)
    assert "volume_sparsity" not in inspect.signature(models.OmniGaussianCylinderVolume.__init__).parameters


def test_off_adds_nothing_and_on_passes_the_model_args():
    cls_name = "OmniGaussianCylinderAll"
    g, _, _ = _opacity_volume([0.0, 1.0], [-9.0])
    assert _shell(cls_name, sparsity_on=False)._apply_sparsity(g, "train", 0) == (None, {})
    obj = _shell(cls_name, sparsity_args=dict(budget=0.2, budget_weight=3.0))
    obj._sparsity_rho_override = 0.5
    loss, stats = obj._apply_sparsity(g, "train", 0)
    assert stats["volume_share_target"] == 0.5 and float(loss) == pytest.approx(3.0 * (2 / 3 - 0.5) ** 2, rel=1e-5)


# ----------------------------------------------------------------------------- train.py SparsityMonitor


def _monitor(tmp_path, num_processes=1, **attrs):
    from tests.conftest import import_models

    import_models()
    import train

    return train.SparsityMonitor(types.SimpleNamespace(**attrs), str(tmp_path), num_processes)


def _args(**overrides):
    return dict(_sparsity().SPARSITY_DEFAULTS, **dict(dict(budget_start=0.85), **overrides))


def _run(mon, steps, share, target=0.35):
    for step in steps:
        mon.check(step, torch.tensor(1.0), {"train/volume_share": share(step), "train/volume_share_target": target})


def test_the_monitor_refuses_a_run_without_a_budget_start(tmp_path):
    with pytest.raises(SystemExit, match="budget_start"):
        _monitor(tmp_path, volume_sparsity=True, sparsity_args=_args(budget_start=None))
    assert not _monitor(tmp_path, volume_sparsity=False, sparsity_args=_args(budget_start=None)).active


def test_the_monitor_refuses_more_than_one_process(tmp_path):
    with pytest.raises(SystemExit, match="one process"):
        _monitor(tmp_path, num_processes=3, volume_sparsity=True, sparsity_args=_args())
    assert not _monitor(tmp_path, num_processes=3, volume_sparsity=False, sparsity_args=_args()).active


def test_the_monitor_is_active_with_the_switch_only(tmp_path):
    assert _monitor(tmp_path, volume_sparsity=True, sparsity_args=_args()).active
    assert not _monitor(tmp_path, volume_sparsity=False).active
    assert not _monitor(tmp_path).active
    _monitor(tmp_path).check(5, torch.tensor(float("nan")), {})  # inactive: nothing happens
    assert not (tmp_path / "arm_failed.json").exists()


def test_a_non_finite_loss_stops_the_run(tmp_path):
    mon = _monitor(tmp_path, volume_sparsity=True, sparsity_args=_args())
    with pytest.raises(SystemExit) as stop:
        mon.check(3, torch.tensor(float("inf")), {})
    assert stop.value.code == 7
    assert json.loads((tmp_path / "arm_failed.json").read_text())["reason"] == "non-finite loss"


@pytest.mark.parametrize("share, fails", [(0.49 * 0.35, True), (0.51 * 0.35, False)])
def test_the_collapse_floor_fires_below_half_the_target_after_collapse_from(tmp_path, share, fails):
    mon = _monitor(tmp_path, volume_sparsity=True, sparsity_args=_args(collapse_floor=0.5, collapse_from=1000))
    if fails:
        _run(mon, range(800, 1000), lambda s: share)  # collapsed before collapse_from: tolerated
        assert not (tmp_path / "arm_failed.json").exists()
        with pytest.raises(SystemExit):
            _run(mon, [1000], lambda s: share)
        report = json.loads((tmp_path / "arm_failed.json").read_text())
        assert report["reason"] == "volume share collapsed" and report["step"] == 1000 and report["steps"] == 100
    else:
        _run(mon, range(800, 3000), lambda s: share)
        assert not (tmp_path / "arm_failed.json").exists()


def test_the_floor_uses_the_trailing_mean_and_its_own_window(tmp_path):
    mon = _monitor(tmp_path, volume_sparsity=True, sparsity_args=_args(collapse_floor=0.5, collapse_from=1000))
    _run(mon, range(900, 1200), lambda s: 0.01 if s == 1100 else 0.30)  # one collapsed batch in a healthy window
    assert not (tmp_path / "arm_failed.json").exists()
    mon = _monitor(tmp_path, volume_sparsity=True, sparsity_args=_args(collapse_floor=0.5, collapse_window=300,
                                                                       collapse_from=3600))
    _run(mon, range(3400, 3600), lambda s: 0.1)
    with pytest.raises(SystemExit):
        _run(mon, range(3600, 3700), lambda s: 0.1)
    assert json.loads((tmp_path / "arm_failed.json").read_text())["steps"] == 300


def test_without_a_floor_only_non_finite_losses_stop(tmp_path):
    mon = _monitor(tmp_path, volume_sparsity=True, sparsity_args=_args(collapse_floor=None))
    _run(mon, range(0, 5000, 50), lambda s: 0.0)
    assert not (tmp_path / "arm_failed.json").exists()


# ----------------------------------------------------------------------------- configs and transfers

SCREEN = "configs/OmniScene/screen"
LS_SWITCHES = {"prune_invisible", "lpips_input_range", "sampling_align", "ws_loss"}
STUDY_CONFIGS = {
    "s3k_base.py": ("OmniGaussianCylinderAll", LS_SWITCHES),
    "s3k_ls.py": ("OmniGaussianCylinderAll", LS_SWITCHES),
    "s3k_sp3d.py": ("OmniGaussianCylinderAll", LS_SWITCHES | {"volume_sparsity"}),
    "s3k_ls_sp3.py": ("OmniGaussianCylinderAll", LS_SWITCHES | {"volume_sparsity"}),
}


def _cfg(name):
    from mmengine.config import Config

    from tests.conftest import REPO_ROOT, import_models
    from tools import switches

    import_models()  # switches.apply imports the model registry for a non-default switch

    cfg = Config.fromfile(os.path.join(REPO_ROOT, SCREEN, name))
    switches.apply(cfg, [])
    return cfg


@pytest.mark.parametrize("name", sorted(STUDY_CONFIGS))
def test_study_configs(name):
    from tools import switches

    cfg = _cfg(name)
    model_type, expected = STUDY_CONFIGS[name]
    assert cfg.model.type == model_type
    assert set(switches.non_default(switches.resolve(cfg))) == expected
    assert cfg.seed == 1111
    assert (cfg.lr, cfg.onecycle_total_steps, cfg.save_freq) == (5e-5, 8100, 4000)
    assert cfg.model.loss_args.weight_depth_abs == 0.1


def test_the_sparsity_schedule():
    sp3d = dict(budget=0.35, ramp_steps=3600, budget_weight=0.05, collapse_floor=0.5, collapse_from=3600)
    assert dict(_cfg("s3k_sp3d.py").model.sparsity_args) == sp3d
    # budget_start is the init's ceiling: LS's largest training-batch volume share 0.808 -> 0.85
    assert dict(_cfg("s3k_ls_sp3.py").model.sparsity_args) == dict(sp3d, budget_start=0.85)
    # the control differs from the recipe only by the budget
    ls, sp = _cfg("s3k_ls.py"), _cfg("s3k_ls_sp3.py")
    assert sp.switches == dict(ls.switches, volume_sparsity=True)


def _state_keys(name):
    from tests.conftest import import_models
    from tests.test_model_build import CudaOnCpu

    import_models()
    from mmdet3d.registry import MODELS

    try:
        with CudaOnCpu():
            model = MODELS.build(_cfg(name).model)
    except OSError as e:  # pretrained weights (LPIPS / VGG) not on this machine
        pytest.skip(f"pretrained weights not available here: {e}")
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def test_every_study_transfer_is_exact():
    """LS and its sparsity fine-tunes have one state dict (names and shapes): the budget adds no parameter, so the
    transfer from LS is `exact`."""
    ls = _state_keys("long_s.py")
    for name in ("s3k_ls.py", "s3k_ls_sp3.py"):
        assert _state_keys(name) == ls, name


def test_the_defaults_are_the_studied_schedule():
    sp = _sparsity()
    sp3d = _cfg("s3k_sp3d.py").model.sparsity_args
    assert {k: v for k, v in sp.SPARSITY_DEFAULTS.items() if k in sp3d} == dict(sp3d)


def test_a_saturated_opacity_gets_no_budget_gradient():
    sp = _sparsity()
    g, x, _ = _opacity_volume([0.0, 15.0], [-9.0])  # sigmoid(15) > 1 - 1e-6: the clamped logit has no gradient
    loss, _ = sp.volume_sparsity_budget(g, "train", 0, dict(sp.SPARSITY_DEFAULTS, budget_weight=1.0), rho_override=0.1)
    loss.backward()
    assert x.grad[:, 0].abs().min() > 0 and torch.count_nonzero(x.grad[:, 1]) == 0
