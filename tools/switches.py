"""Opt-in behaviour switches.

Every switch defaults to the behaviour of the released code, so a config or
command line that sets none of them builds, trains and evaluates exactly as
before. A switch is set in a config file

    switches = dict(ddp_forward=True, loc360_interleave=True)

or on the command line (`--switch loc360_interleave=true`, repeatable; the
command line wins). Model switches are written into the model config only when
their value differs from the default, so model classes that do not know a
switch are built unchanged; a switch key already present in the config (e.g. a
config dumped by an earlier run) is overwritten with the resolved value, so
`--switch name=<default>` really turns it off. Setting a switch on a model class
that does not declare that keyword argument is refused before the model is
built (the model classes take **kwargs, so the constructor alone would silently
swallow it).
"""

import inspect

# name -> (default, type, config paths it is written to, help)
# A path is a dotted key under cfg; no path means the switch is read by the
# entry point itself (train.py) and is not written into the model.
SWITCHES = {
    # Call the DDP-wrapped model, so gradients are all-reduced across ranks.
    "ddp_forward": (False, bool, [], "train: forward through the DDP wrapper (gradient sync)"),
    # 360Loc train split: one globally shuffled stream of (sequence, sample) pairs instead of
    # whole sequences one after another (~167 consecutive steps per sequence); each sequence still
    # gives times_per_scene samples per epoch, and every frame is decoded once per worker.
    "loc360_interleave": (False, bool, [], "data: interleave the 360Loc training sequences"),
    # 360Loc: the depth loss ignores prior depth outside the loader's (near, far) range.
    "depth_valid_mask": (False, bool, ["model.depth_valid_mask"], "train: mask invalid prior depth in the depth loss"),
}

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _parse_value(name, raw):
    """A config or command-line value -> bool (every switch is a bool)."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f"switch {name}: expected true/false, got {raw!r}")
    raise ValueError(f"switch {name}: expected a bool, got {raw!r}")


def parse_cli(pairs):
    """['a=1', 'b=x'] -> {'a': '1', 'b': 'x'}"""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--switch expects name=value, got {pair!r}")
        name, value = pair.split("=", 1)
        out[name.strip()] = value
    return out


def resolve(cfg, cli_pairs=None):
    """Merge config `switches` and CLI overrides; return {name: value} for every switch."""
    requested = dict(cfg.get("switches", None) or {})
    requested.update(parse_cli(cli_pairs))
    unknown = sorted(set(requested) - set(SWITCHES))
    if unknown:
        raise ValueError(f"unknown switch(es) {unknown}; known: {sorted(SWITCHES)}")
    values = {name: spec[0] for name, spec in SWITCHES.items()}
    for name, raw in requested.items():
        values[name] = _parse_value(name, raw)
    return values


def _set_path(cfg, dotted, value):
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        if key not in node or node[key] is None:
            raise KeyError(f"switch target {dotted}: config has no {key!r}")
        node = node[key]
    node[keys[-1]] = value


def _has_leaf(cfg, dotted):
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        if not hasattr(node, "get") or node.get(key, None) is None:
            return False
        node = node[key]
    return hasattr(node, "get") and keys[-1] in node


def apply(cfg, cli_pairs=None):
    """Resolve switches and write every non-default model switch into cfg.model.

    Returns the resolved {name: value} dict (log it with the run).
    """
    values = resolve(cfg, cli_pairs)
    for name, value in values.items():
        default, _, paths, _ = SWITCHES[name]
        for path in paths:
            if value == default:
                # A config dumped by an earlier run may already carry the switch; make it agree
                # with the resolved value. Absent keys stay absent (released configs are unchanged).
                if _has_leaf(cfg, path):
                    _set_path(cfg, path, value)
                continue
            _set_path(cfg, path, value)
            _check_supported(cfg, path)
    cfg.switches = values
    return values


def _registered_class(type_name):
    """The class registered under `type_name`, or None if the registry does not know it."""
    import builder.builder  # noqa: F401  (registers every model class)
    from mmdet3d.registry import MODELS

    return MODELS.get(type_name)


def _check_supported(cfg, dotted):
    """Refuse a switch whose target class does not declare the keyword argument.

    Types that are not registered are left to the builder, which fails on them anyway.
    """
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node[key]
    type_name = node.get("type", None) if hasattr(node, "get") else None
    if not isinstance(type_name, str):
        return
    cls = _registered_class(type_name)
    if cls is None:
        return
    if keys[-1] not in inspect.signature(cls.__init__).parameters:
        raise ValueError(f"switch target {dotted}: {type_name} does not support {keys[-1]!r}")


def non_default(values):
    return {k: v for k, v in values.items() if v != SWITCHES[k][0]}
