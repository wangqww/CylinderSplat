"""tools/switches.py: the three switches, value parsing, config / CLI resolution, where a model switch is
written, and the refusal of unknown switches and of a switch the target class does not declare.

The unit tests use a small stand-in for mmengine's Config and a stubbed registry lookup (no torch). The last
test resolves depth_valid_mask on every model config against the real registry.
"""

import os

import pytest

from tests.conftest import REPO_ROOT, import_models, kept_configs
from tools import switches

# name -> (default, config paths written when the value is not the default)
TABLE = {
    "ddp_forward": (False, []),
    "loc360_interleave": (False, []),
    "depth_valid_mask": (False, ["model.depth_valid_mask"]),
}
DEFAULTS = {name: default for name, (default, _) in TABLE.items()}


class FakeConfig(dict):
    """What tools/switches.py uses of mmengine's Config: item access, .get() and attribute assignment."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def fake_cfg(switch_block=None, **model):
    cfg = FakeConfig(model=dict(type="FakeModel", **model))
    if switch_block is not None:
        cfg["switches"] = switch_block
    return cfg


def plain(node):
    """A plain-dict copy of a FakeConfig or mmengine ConfigDict tree."""
    return {k: plain(v) for k, v in node.items()} if isinstance(node, dict) else node


@pytest.fixture
def registry(monkeypatch):
    """Stubbed class lookup: fill `classes` ({type name: class}); `looked_up` records every lookup."""
    classes, looked_up = {}, []

    def registered_class(type_name):
        looked_up.append(type_name)
        return classes.get(type_name)

    monkeypatch.setattr(switches, "_registered_class", registered_class)
    return classes, looked_up


class Declares:
    def __init__(self, backbone=None, depth_valid_mask=False, **kwargs):
        pass


class SwallowsKwargs:
    def __init__(self, backbone=None, **kwargs):
        pass


# ----------------------------------------------------------------------------- table, parsing, resolution


def test_the_switch_table():
    assert {name: (spec[0], spec[2]) for name, spec in switches.SWITCHES.items()} == TABLE
    for default, kind, _, help_text in switches.SWITCHES.values():
        assert kind is bool and isinstance(default, bool) and help_text


def test_parse_cli():
    assert switches.parse_cli(None) == switches.parse_cli([]) == {}
    # The name is stripped; the value is kept as written (it is stripped when parsed); the last one wins.
    assert switches.parse_cli(["a=1", " b = x ", "c=d=e", "a=2"]) == {"a": "2", "b": " x ", "c": "d=e"}
    with pytest.raises(ValueError, match="name=value"):
        switches.parse_cli(["ddp_forward"])


TRUE_SPELLINGS = ["1", "true", "YES", " on ", True]
FALSE_SPELLINGS = ["0", "False", "no", "OFF", False]


@pytest.mark.parametrize("raw", TRUE_SPELLINGS + FALSE_SPELLINGS)
def test_bool_values(raw):
    assert switches._parse_value("ddp_forward", raw) is (raw in TRUE_SPELLINGS)


@pytest.mark.parametrize("raw", ["maybe", "", 1, 0, None, 1.0])
def test_other_values_are_refused(raw):
    with pytest.raises(ValueError, match="switch depth_valid_mask: expected"):
        switches._parse_value("depth_valid_mask", raw)


def test_resolve():
    assert switches.resolve(fake_cfg()) == switches.resolve(fake_cfg(switch_block={})) == DEFAULTS
    cfg = fake_cfg(switch_block={"loc360_interleave": True, "depth_valid_mask": "on"})
    before = plain(cfg)
    # The command line wins over the config; resolve() writes nothing.
    values = switches.resolve(cfg, ["depth_valid_mask=false", "ddp_forward=1"])
    assert values == dict(ddp_forward=True, loc360_interleave=True, depth_valid_mask=False)
    assert plain(cfg) == before
    assert switches.non_default(values) == dict(ddp_forward=True, loc360_interleave=True)
    assert switches.non_default(DEFAULTS) == {}


def test_unknown_switches_are_refused():
    with pytest.raises(ValueError, match=r"unknown switch\(es\) \['no_such_switch'\]"):
        switches.resolve(fake_cfg(), ["no_such_switch=true"])
    with pytest.raises(ValueError, match="no_such_switch"):
        switches.apply(fake_cfg(switch_block={"no_such_switch": True}))


# ----------------------------------------------------------------------------- apply: what is written where


def test_defaults_write_nothing(registry):
    cfg = fake_cfg(switch_block=dict(DEFAULTS))
    assert switches.apply(cfg, [f"{name}=false" for name in TABLE]) == DEFAULTS
    assert cfg["model"] == dict(type="FakeModel")  # no key is added
    assert cfg["switches"] == DEFAULTS
    assert registry[1] == []  # a default is never checked against the class


@pytest.mark.parametrize("name", sorted(TABLE))
def test_a_switch_is_written_to_its_paths_only(name, registry):
    cfg = fake_cfg()
    values = switches.apply(cfg, [f"{name}=true"])
    assert switches.non_default(values) == {name: True} and cfg["switches"] == values
    expected = dict(type="FakeModel")
    for path in TABLE[name][1]:
        expected[path[len("model.") :]] = True
    assert cfg["model"] == expected


def test_a_default_overrides_the_value_of_a_dumped_config(registry):
    # train.py dumps the config after apply(); a run on that dump can turn the switch off again.
    dumped = dict(switch_block=dict(DEFAULTS, depth_valid_mask=True), depth_valid_mask=True)
    cfg = fake_cfg(**dumped)
    assert switches.apply(cfg, ["depth_valid_mask=false"]) == DEFAULTS
    assert cfg["model"]["depth_valid_mask"] is False
    cfg = fake_cfg(**dumped)
    assert switches.non_default(switches.apply(cfg)) == {"depth_valid_mask": True}
    assert cfg["model"]["depth_valid_mask"] is True


def test_a_missing_target_is_an_error(registry):
    with pytest.raises(KeyError, match="config has no 'model'"):
        switches.apply(FakeConfig(model=None), ["depth_valid_mask=true"])


# ----------------------------------------------------------------------------- apply: the declared-keyword check


def test_a_declared_keyword_is_accepted(registry):
    classes, looked_up = registry
    classes["FakeModel"] = Declares
    cfg = fake_cfg()
    switches.apply(cfg, ["depth_valid_mask=true"])
    assert cfg["model"]["depth_valid_mask"] is True and looked_up == ["FakeModel"]


def test_a_keyword_only_swallowed_by_kwargs_is_refused(registry):
    registry[0]["FakeModel"] = SwallowsKwargs
    with pytest.raises(ValueError, match="model.depth_valid_mask: FakeModel does not support 'depth_valid_mask'"):
        switches.apply(fake_cfg(), ["depth_valid_mask=true"])


def test_an_unregistered_or_untyped_model_is_left_to_the_builder(registry):
    cfg = fake_cfg()
    switches.apply(cfg, ["depth_valid_mask=true"])
    assert cfg["model"]["depth_valid_mask"] is True and registry[1] == ["FakeModel"]
    cfg = FakeConfig(model=dict(type=SwallowsKwargs))  # not a registry name: not looked up
    switches.apply(cfg, ["depth_valid_mask=true"])
    assert cfg["model"]["depth_valid_mask"] is True and registry[1] == ["FakeModel"]


# ----------------------------------------------------------------------------- real configs, real registry


@pytest.mark.parametrize("rel", kept_configs())
def test_depth_valid_mask_is_accepted_only_by_the_360loc_model(rel):
    config = pytest.importorskip("mmengine.config")
    import_models()
    cfg = config.Config.fromfile(os.path.join(REPO_ROOT, rel))
    before = plain(cfg.model)
    assert switches.apply(cfg, []) == DEFAULTS
    assert plain(cfg.model) == before
    model_type = cfg.model.type
    if model_type == "OmniGaussianCylinderVolume360LocPan2":
        switches.apply(cfg, ["depth_valid_mask=true"])
        assert cfg.model.depth_valid_mask is True
    else:
        with pytest.raises(ValueError, match=f"{model_type} does not support 'depth_valid_mask'"):
            switches.apply(cfg, ["depth_valid_mask=true"])
