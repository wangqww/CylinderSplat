"""tools/switches.py: value parsing (incl. the prune_opacity range), config / CLI resolution,
where non-default values are written, how a default overrides a key already in the config
(a dumped config), and the refusal of a switch whose target class does not declare it.

The unit tests use a minimal stand-in for mmengine's Config and a stubbed class
lookup, so they need neither torch nor mmengine. The real-config tests load the
repo's configs with mmengine and look the classes up in the mmdet3d MODELS registry
after importing builder.builder (CPU; they import the model package like the other
model tests).
"""

import copy
import os

import pytest

from tools import switches

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_256 = "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py"
VOLUME_256 = "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py"
LOC360_ALL_256 = "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py"

# The documented table (plan §5 WS-D): name -> (default, config paths written when non-default,
# or when the key is already in the config).
DOCUMENTED = {
    "ddp_forward": (False, []),
    "shuffle_train": (False, []),
    "lpips_eval": (False, ["model.lpips_eval"]),
    "freeze_frozen_bn": (False, ["model.freeze_frozen_bn"]),
    "v1_identity_pose": (False, ["model.v1_identity_pose"]),
    "rotate_gaussians_to_world": (False, ["model.rotate_gaussians_to_world",
                                          "model.pixel_gs.rotate_gaussians_to_world"]),
    "theta_periodic": (False, ["model.volume_gs.encoder.theta_periodic",
                               "model.volume_gs.gs_decoder.theta_periodic"]),
    "cell_center_anchor": (False, ["model.volume_gs.gs_decoder.cell_center_anchor"]),
    "rgb_retrieval": ("concat", ["model.volume_gs.gs_decoder.rgb_retrieval"]),
    "prune_opacity": (0.0, []),
    "pixel_depth_sampling": ("bilinear", ["model.pixel_gs.pixel_depth_sampling"]),
    "loc360_interleave": (False, []),
    "depth_valid_mask": (False, ["model.depth_valid_mask"]),
}
# One non-default value per switch: (CLI text, parsed value).
NON_DEFAULT = {
    "ddp_forward": ("true", True),
    "shuffle_train": ("on", True),
    "lpips_eval": ("1", True),
    "freeze_frozen_bn": ("yes", True),
    "v1_identity_pose": ("True", True),
    "rotate_gaussians_to_world": ("true", True),
    "theta_periodic": ("true", True),
    "cell_center_anchor": ("true", True),
    "rgb_retrieval": ("visibility_softmax", "visibility_softmax"),
    "prune_opacity": ("0.005", 0.005),
    "pixel_depth_sampling": ("nearest", "nearest"),
    "loc360_interleave": ("true", True),
    "depth_valid_mask": ("on", True),
}
DEFAULTS = {name: spec[0] for name, spec in DOCUMENTED.items()}
NAMES = sorted(DOCUMENTED)
MODEL_SWITCHES = [name for name in NAMES if DOCUMENTED[name][1]]


def _cli_text(value):
    return "true" if value is True else "false" if value is False else str(value)


class FakeConfig(dict):
    """What tools/switches.py uses of mmengine's Config: item access, .get() and attribute assignment."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def fake_cfg(switch_block=None, model=None):
    cfg = FakeConfig(model=model if model is not None else dict(
        type="FakeTop",
        pixel_gs=dict(type="FakePixel"),
        volume_gs=dict(type="FakeVolume", encoder=dict(type="FakeEncoder"), gs_decoder=dict(type="FakeDecoder")),
    ))
    if switch_block is not None:
        cfg["switches"] = switch_block
    return cfg


def _get(node, dotted):
    for key in dotted.split("."):
        node = node[key]
    return node


def _set(node, dotted, value):
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def _plain(node):
    """A plain-dict deep copy (works for dict, FakeConfig and mmengine's ConfigDict)."""
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return type(node)(_plain(v) for v in node)
    return copy.deepcopy(node)


@pytest.fixture
def unregistered(monkeypatch):
    """Every type counts as unregistered, so apply() checks nothing (records the lookups)."""
    looked_up = []

    def lookup(type_name):
        looked_up.append(type_name)
        return None

    monkeypatch.setattr(switches, "_registered_class", lookup)
    return looked_up


# ----------------------------------------------------------------------------- the table

def test_switch_table_is_the_documented_one():
    assert {name: (spec[0], spec[2]) for name, spec in switches.SWITCHES.items()} == DOCUMENTED
    for name, (default, kind, _, help_text) in switches.SWITCHES.items():
        assert isinstance(help_text, str) and help_text
        if isinstance(kind, tuple):
            assert default in kind
        else:
            assert isinstance(default, kind)
    assert switches.SWITCHES["prune_opacity"][1] is float
    assert switches.SWITCHES["rgb_retrieval"][1] == ("concat", "visibility_softmax")
    assert switches.SWITCHES["pixel_depth_sampling"][1] == ("bilinear", "nearest")


# ----------------------------------------------------------------------------- parse_cli / _parse_value

def test_parse_cli():
    assert switches.parse_cli(None) == {}
    assert switches.parse_cli([]) == {}
    # The name is stripped, the value is passed on as written (and stripped when parsed).
    assert switches.parse_cli(["a=1", " b = x ", "c=d=e", "a=2"]) == {"a": "2", "b": " x ", "c": "d=e"}
    with pytest.raises(ValueError, match="name=value"):
        switches.parse_cli(["theta_periodic"])


@pytest.mark.parametrize("text", ["1", "true", "True", "YES", "on", " true "])
def test_bool_true_spellings(text):
    assert switches._parse_value("ddp_forward", text) is True


@pytest.mark.parametrize("text", ["0", "false", "False", "no", "OFF", " off "])
def test_bool_false_spellings(text):
    assert switches._parse_value("theta_periodic", text) is False


def test_bool_rejects_other_values():
    with pytest.raises(ValueError, match="expected true/false"):
        switches._parse_value("lpips_eval", "maybe")
    # A config value must be a real bool (or a string spelled as above).
    assert switches._parse_value("lpips_eval", True) is True
    assert switches._parse_value("lpips_eval", False) is False
    for raw in (1, 0, None, 1.0):
        with pytest.raises(ValueError, match="expected a bool"):
            switches._parse_value("lpips_eval", raw)


def test_float_values():
    assert switches._parse_value("prune_opacity", "0.25") == 0.25
    assert switches._parse_value("prune_opacity", " 1e-3 ") == 1e-3
    value = switches._parse_value("prune_opacity", 0)  # an int in a config becomes a float
    assert value == 0.0 and isinstance(value, float)
    assert switches._parse_value("prune_opacity", 0.5) == 0.5
    with pytest.raises(ValueError):
        switches._parse_value("prune_opacity", "abc")


@pytest.mark.parametrize("text", ["1", "1.0", "1.5", "-0.01", "nan", "inf", "-inf"])
def test_prune_opacity_outside_zero_one_is_refused_at_resolve(text, unregistered):
    # An opacity threshold in [0, 1): refused while the switches are resolved, i.e. before
    # train.py / evaluate.py write a run dir or build the model.
    with pytest.raises(ValueError, match=r"prune_opacity must be in \[0, 1\)"):
        switches.resolve(fake_cfg(), [f"prune_opacity={text}"])
    cfg = fake_cfg(switch_block={"prune_opacity": float(text)})  # the same value from a config
    before = _plain(cfg)
    with pytest.raises(ValueError, match=r"prune_opacity must be in \[0, 1\)"):
        switches.apply(cfg)
    assert _plain(cfg) == before  # nothing written, not even cfg.switches


@pytest.mark.parametrize("raw,value", [("0", 0.0), (0, 0.0), (0.0, 0.0), ("0.25", 0.25), ("1e-3", 1e-3),
                                       ("0.999", 0.999)])
def test_prune_opacity_inside_zero_one_is_accepted(raw, value):
    cli = [f"prune_opacity={raw}"] if isinstance(raw, str) else []
    cfg = fake_cfg(switch_block=None if cli else {"prune_opacity": raw})
    got = switches.resolve(cfg, cli)["prune_opacity"]
    assert got == value and isinstance(got, float)


@pytest.mark.parametrize("name,good,bad", [("rgb_retrieval", "visibility_softmax", "softmax"),
                                           ("pixel_depth_sampling", "nearest", "bicubic")])
def test_choice_values(name, good, bad):
    assert switches._parse_value(name, good) == good
    assert switches._parse_value(name, f" {good} ") == good
    with pytest.raises(ValueError, match="expected one of"):
        switches._parse_value(name, bad)
    with pytest.raises(ValueError, match="expected one of"):
        switches._parse_value(name, 3)  # a config value outside the choices


def test_other_kinds_are_literal_evaluated(monkeypatch):
    monkeypatch.setitem(switches.SWITCHES, "fake_int", (0, int, [], "test only"))
    assert switches._parse_value("fake_int", "7") == 7
    assert switches._parse_value("fake_int", " [1, 2] ") == [1, 2]
    assert switches._parse_value("fake_int", 5) == 5


# ----------------------------------------------------------------------------- resolve

def test_resolve_defaults():
    for cfg in (fake_cfg(), fake_cfg(switch_block={}), fake_cfg(switch_block=None)):
        assert switches.resolve(cfg) == DEFAULTS
    cfg = fake_cfg()
    cfg["switches"] = None
    assert switches.resolve(cfg, []) == DEFAULTS


@pytest.mark.parametrize("name", NAMES)
def test_resolve_each_switch_from_cli_and_config(name):
    text, value = NON_DEFAULT[name]
    assert switches.resolve(fake_cfg(), [f"{name}={text}"]) == dict(DEFAULTS, **{name: value})
    assert switches.resolve(fake_cfg(switch_block={name: text})) == dict(DEFAULTS, **{name: value})


def test_cli_overrides_config():
    cfg = fake_cfg(switch_block=dict(theta_periodic=True, rgb_retrieval="visibility_softmax", prune_opacity=0.1))
    snapshot = _plain(cfg)
    values = switches.resolve(cfg, ["theta_periodic=false", "prune_opacity=0.2"])
    assert values == dict(DEFAULTS, rgb_retrieval="visibility_softmax", prune_opacity=0.2)
    assert _plain(cfg) == snapshot  # resolve() writes nothing


@pytest.mark.parametrize("where", ["config", "cli"])
def test_unknown_switch_is_an_error(where):
    cfg = fake_cfg(switch_block={"no_such_switch": True} if where == "config" else None)
    cli = ["no_such_switch=true"] if where == "cli" else []
    with pytest.raises(ValueError, match=r"unknown switch\(es\) \['no_such_switch'\]"):
        switches.resolve(cfg, cli)
    with pytest.raises(ValueError, match="no_such_switch"):
        switches.apply(fake_cfg(switch_block={"no_such_switch": True} if where == "config" else None), cli)


def test_non_default():
    values = dict(DEFAULTS, theta_periodic=True, prune_opacity=0.01)
    assert switches.non_default(values) == {"theta_periodic": True, "prune_opacity": 0.01}
    assert switches.non_default(DEFAULTS) == {}


# ----------------------------------------------------------------------------- apply: what is written where

def test_apply_all_defaults_writes_nothing(unregistered):
    cfg = fake_cfg()
    before = _plain(cfg["model"])
    assert switches.apply(cfg) == DEFAULTS
    assert _plain(cfg["model"]) == before
    assert cfg["switches"] == DEFAULTS
    assert unregistered == []  # defaults are never checked either


def test_apply_explicit_defaults_writes_nothing(unregistered):
    # Every switch set to its default, in the config and again on the command line.
    cfg = fake_cfg(switch_block=dict(DEFAULTS))
    cli = [f"{name}={_cli_text(value)}" for name, value in DEFAULTS.items()]
    before = _plain(cfg["model"])
    assert switches.apply(cfg, cli) == DEFAULTS
    assert _plain(cfg["model"]) == before
    assert unregistered == []


def test_defaults_need_no_target_in_the_config(unregistered):
    # A model without pixel_gs / volume_gs: nothing is written, so nothing is missing.
    cfg = fake_cfg(switch_block=dict(DEFAULTS), model=dict(type="FakeTop"))
    assert switches.apply(cfg) == DEFAULTS
    assert cfg["model"] == dict(type="FakeTop")


@pytest.mark.parametrize("name", NAMES)
def test_apply_writes_exactly_the_documented_paths(name, unregistered):
    text, value = NON_DEFAULT[name]
    cfg = fake_cfg()
    expected = _plain(cfg)
    for path in DOCUMENTED[name][1]:
        _set(expected, path, value)
    values = switches.apply(cfg, [f"{name}={text}"])
    assert values == dict(DEFAULTS, **{name: value})
    assert switches.non_default(values) == {name: value}
    expected["switches"] = values
    assert _plain(cfg) == expected
    for path in DOCUMENTED[name][1]:
        assert _get(cfg, path) == value


def test_apply_all_non_default_together(unregistered):
    cfg = fake_cfg(switch_block={name: NON_DEFAULT[name][1] for name in NAMES})
    expected = _plain(cfg)
    for name in NAMES:
        for path in DOCUMENTED[name][1]:
            _set(expected, path, NON_DEFAULT[name][1])
    values = switches.apply(cfg)
    expected["switches"] = values
    assert _plain(cfg) == expected
    # Each written key's owner was looked up by its own type.
    assert sorted(set(unregistered)) == ["FakeDecoder", "FakeEncoder", "FakePixel", "FakeTop"]


@pytest.mark.parametrize("model,missing", [
    (dict(type="FakeTop"), "volume_gs"),
    (dict(type="FakeTop", volume_gs=None), "volume_gs"),
    (dict(type="FakeTop", volume_gs=dict(type="FakeVolume")), "encoder"),
])
def test_non_default_without_its_target_is_an_error(model, missing, unregistered):
    with pytest.raises(KeyError, match=f"config has no '{missing}'"):
        switches.apply(fake_cfg(model=model), ["theta_periodic=true"])


# ----------------------------------------------------------------------------- apply: a config dumped by an earlier run

def _dumped_cfg(**on):
    """A config as train.py dumps it: the non-default values in cfg.model and the resolved `switches`."""
    cfg = fake_cfg()
    for name, value in on.items():
        for path in DOCUMENTED[name][1]:
            _set(cfg, path, value)
    cfg["switches"] = dict(DEFAULTS, **on)
    return cfg


@pytest.mark.parametrize("name", MODEL_SWITCHES)
def test_explicit_default_overrides_a_switch_baked_into_the_config(name, unregistered):
    # e.g. evaluate.py --py-config <run>/<config>.py --switch theta_periodic=false on a D5 run:
    # the model is built with the default, as switches.json and the log say.
    cfg = _dumped_cfg(**{name: NON_DEFAULT[name][1]})
    expected = _plain(cfg)
    for path in DOCUMENTED[name][1]:
        _set(expected, path, DEFAULTS[name])
    values = switches.apply(cfg, [f"{name}={_cli_text(DEFAULTS[name])}"])
    assert values == DEFAULTS
    expected["switches"] = DEFAULTS
    assert _plain(cfg) == expected  # only the baked keys change; absent ones stay absent
    for path in DOCUMENTED[name][1]:
        assert _get(cfg, path) == DEFAULTS[name]
    assert unregistered == []  # a default is never checked against the class


def test_dumped_config_without_an_override_keeps_its_switches(unregistered):
    cfg = _dumped_cfg(theta_periodic=True, rgb_retrieval="visibility_softmax")
    expected = _plain(cfg)
    values = switches.apply(cfg)
    assert switches.non_default(values) == {"theta_periodic": True, "rgb_retrieval": "visibility_softmax"}
    expected["switches"] = values
    assert _plain(cfg) == expected


def test_default_changes_only_the_keys_already_present(unregistered):
    # Only the encoder carries theta_periodic, and pixel_gs is None: a default adds no key and needs no target.
    cfg = fake_cfg()
    cfg["model"]["volume_gs"]["encoder"]["theta_periodic"] = True
    cfg["model"]["pixel_gs"] = None
    switches.apply(cfg, ["theta_periodic=off", "pixel_depth_sampling=bilinear"])
    assert cfg["model"]["volume_gs"]["encoder"]["theta_periodic"] is False
    assert "theta_periodic" not in cfg["model"]["volume_gs"]["gs_decoder"]
    assert cfg["model"]["pixel_gs"] is None
    assert "rotate_gaussians_to_world" not in cfg["model"] and "lpips_eval" not in cfg["model"]


# ----------------------------------------------------------------------------- apply: supported-keyword check (stubbed lookup)

class _Declares:
    def __init__(self, backbone=None, lpips_eval=False, theta_periodic=False, **kwargs):
        pass


class _SwallowsKwargs:
    def __init__(self, backbone=None, **kwargs):
        pass


def _classes(monkeypatch, table):
    looked_up = []

    def lookup(type_name):
        looked_up.append(type_name)
        return table.get(type_name)

    monkeypatch.setattr(switches, "_registered_class", lookup)
    return looked_up


def test_declared_keyword_is_accepted(monkeypatch):
    looked_up = _classes(monkeypatch, {"FakeTop": _Declares})
    cfg = fake_cfg()
    switches.apply(cfg, ["lpips_eval=true"])
    assert cfg["model"]["lpips_eval"] is True
    assert looked_up == ["FakeTop"]


def test_keyword_swallowed_by_kwargs_is_refused(monkeypatch):
    _classes(monkeypatch, {"FakeTop": _SwallowsKwargs})
    with pytest.raises(ValueError, match="switch target model.lpips_eval: FakeTop does not support 'lpips_eval'"):
        switches.apply(fake_cfg(), ["lpips_eval=true"])


def test_the_owner_of_each_path_is_checked(monkeypatch):
    # theta_periodic goes to the encoder and the decoder; the decoder here does not declare it.
    looked_up = _classes(monkeypatch, {"FakeEncoder": _Declares, "FakeDecoder": _SwallowsKwargs,
                                       "FakeVolume": _SwallowsKwargs})
    with pytest.raises(ValueError, match="model.volume_gs.gs_decoder.theta_periodic: FakeDecoder"):
        switches.apply(fake_cfg(), ["theta_periodic=true"])
    assert looked_up == ["FakeEncoder", "FakeDecoder"]  # never the intermediate FakeVolume


def test_unregistered_type_is_left_to_the_builder(monkeypatch):
    looked_up = _classes(monkeypatch, {})
    cfg = fake_cfg()
    switches.apply(cfg, ["freeze_frozen_bn=true", "pixel_depth_sampling=nearest"])
    assert cfg["model"]["freeze_frozen_bn"] is True
    assert cfg["model"]["pixel_gs"]["pixel_depth_sampling"] == "nearest"
    assert sorted(looked_up) == ["FakePixel", "FakeTop"]


@pytest.mark.parametrize("model", [dict(pixel_gs=dict()), dict(type=_SwallowsKwargs, pixel_gs=dict(type=None))])
def test_nodes_without_a_type_name_are_not_checked(model, monkeypatch):
    def lookup(type_name):
        raise AssertionError(f"looked up {type_name!r}")

    monkeypatch.setattr(switches, "_registered_class", lookup)
    cfg = fake_cfg(model=model)
    switches.apply(cfg, ["lpips_eval=true", "pixel_depth_sampling=nearest"])
    assert cfg["model"]["lpips_eval"] is True
    assert cfg["model"]["pixel_gs"]["pixel_depth_sampling"] == "nearest"


# ----------------------------------------------------------------------------- real configs, real registry

def _real_cfg(rel):
    pytest.importorskip("torch")
    pytest.importorskip("mmdet3d")
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(os.path.join(REPO, rel))


def _cli(names):
    return [f"{name}={NON_DEFAULT[name][0]}" for name in names]


# Every switch the config's classes declare (the entry-point switches have no model path).
SUPPORTED = {
    # OmniGaussianCylinderAll + PixelGaussian + TPVFormerEncoderCylinder + VolumeGaussianDecoderCylinder
    ALL_256: ["cell_center_anchor", "ddp_forward", "loc360_interleave", "lpips_eval", "pixel_depth_sampling",
              "prune_opacity", "rgb_retrieval", "rotate_gaussians_to_world", "shuffle_train", "theta_periodic",
              "v1_identity_pose"],
    # OmniGaussianCylinderVolume (stage 2) + the same heads
    VOLUME_256: ["cell_center_anchor", "freeze_frozen_bn", "lpips_eval", "pixel_depth_sampling",
                 "rgb_retrieval", "rotate_gaussians_to_world", "theta_periodic"],
    # OmniGaussianCylinderVolume360LocPan2 + PixelGaussian360Loc + the same volume branch
    LOC360_ALL_256: ["cell_center_anchor", "depth_valid_mask", "lpips_eval", "pixel_depth_sampling", "rgb_retrieval",
                     "rotate_gaussians_to_world", "theta_periodic", "v1_identity_pose"],
}
REFUSED = [
    (ALL_256, "freeze_frozen_bn", "OmniGaussianCylinderAll"),
    (ALL_256, "depth_valid_mask", "OmniGaussianCylinderAll"),
    (VOLUME_256, "depth_valid_mask", "OmniGaussianCylinderVolume"),
    (VOLUME_256, "v1_identity_pose", "OmniGaussianCylinderVolume"),
    (LOC360_ALL_256, "freeze_frozen_bn", "OmniGaussianCylinderVolume360LocPan2"),
]


def test_all_256_supports_every_switch_but_freeze_frozen_bn_and_depth_valid_mask():
    assert sorted(SUPPORTED[ALL_256] + ["depth_valid_mask", "freeze_frozen_bn"]) == NAMES


@pytest.mark.parametrize("rel", sorted(SUPPORTED))
def test_real_config_accepts_its_supported_switches(rel):
    cfg = _real_cfg(rel)
    expected = _plain(cfg.model)
    for name in SUPPORTED[rel]:
        for path in DOCUMENTED[name][1]:
            _set(expected, path[len("model."):], NON_DEFAULT[name][1])
    values = switches.apply(cfg, _cli(SUPPORTED[rel]))
    assert switches.non_default(values) == {name: NON_DEFAULT[name][1] for name in SUPPORTED[rel]}
    assert _plain(cfg.model) == expected
    assert cfg.switches == values


@pytest.mark.parametrize("rel", sorted(SUPPORTED))
def test_real_config_accepts_each_supported_switch_alone(rel):
    for name in SUPPORTED[rel]:
        cfg = _real_cfg(rel)
        switches.apply(cfg, _cli([name]))
        for path in DOCUMENTED[name][1]:
            assert _get(cfg, path) == NON_DEFAULT[name][1], (name, path)


@pytest.mark.parametrize("rel,name,cls", REFUSED)
def test_real_config_refuses_an_undeclared_switch(rel, name, cls):
    cfg = _real_cfg(rel)
    assert cfg.model.type == cls
    with pytest.raises(ValueError, match=f"switch target model.{name}: {cls} does not support '{name}'"):
        switches.apply(cfg, _cli([name]))


def test_real_config_defaults_leave_the_model_untouched():
    cfg = _real_cfg(ALL_256)
    before = _plain(cfg.model)
    assert switches.apply(cfg, []) == DEFAULTS
    assert _plain(cfg.model) == before


def test_real_dumped_config_switch_off_overrides_the_baked_value(tmp_path):
    # train.py dumps the config after apply(); a later run on that dump turns the switch off again.
    cfg = _real_cfg(ALL_256)
    released = _plain(cfg.model)
    switches.apply(cfg, ["theta_periodic=true"])
    dumped = str(tmp_path / os.path.basename(ALL_256))
    cfg.dump(dumped)
    again = type(cfg).fromfile(dumped)
    for path in DOCUMENTED["theta_periodic"][1]:
        assert _get(again, path) is True
    assert switches.apply(again, ["theta_periodic=false"]) == DEFAULTS
    for path in DOCUMENTED["theta_periodic"][1]:
        assert _get(again, path) is False
        _set(released, path[len("model."):], False)
    assert _plain(again.model) == released


def test_real_registry_leaves_an_unregistered_type_to_the_builder():
    cfg = _real_cfg(ALL_256)
    assert switches._registered_class("OmniGaussianCylinderAll").__name__ == "OmniGaussianCylinderAll"
    assert switches._registered_class("UnregisteredModelForSwitchTest") is None
    # freeze_frozen_bn is refused on OmniGaussianCylinderAll (above) but not checked on an unknown type.
    cfg.model.type = "UnregisteredModelForSwitchTest"
    switches.apply(cfg, ["freeze_frozen_bn=true"])
    assert cfg.model.freeze_frozen_bn is True
