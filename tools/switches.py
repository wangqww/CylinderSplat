"""Opt-in behaviour switches.

Every switch defaults to the behaviour of the released code, so a config or
command line that sets none of them builds, trains and evaluates exactly as
before. A switch is set in a config file

    switches = dict(ddp_forward=True, theta_periodic=True)

or on the command line (`--switch theta_periodic=true`, repeatable; the command
line wins). Model switches are written into the model config only when their
value differs from the default, so model classes that do not know a switch
are built unchanged; a switch key already present in the config (e.g. a config dumped
by an earlier run) is overwritten with the resolved value, so `--switch name=<default>`
really turns it off. Setting a switch on a model class that does not declare
that keyword argument is refused before the model is built (the model classes
take **kwargs, so the constructor alone would silently swallow it).
"""

import ast
import inspect

# name -> (default, allowed values or type, config paths it is written to, help)
# A path is a dotted key under cfg; "" means the switch is read by the entry
# point itself (train.py / evaluate.py) and is not written into the model.
SWITCHES = {
    # D1: call the DDP-wrapped model, so gradients are all-reduced across ranks.
    "ddp_forward": (False, bool, [], "train: forward through the DDP wrapper (gradient sync)"),
    # D1b: seeded RandomSampler (the loader's train generator) for map-style training loaders.
    # Accelerate shards its batches and keeps the generator in sync across ranks; a
    # DistributedSampler would be sharded a second time.
    "shuffle_train": (False, bool, [], "data: shuffle map-style training data every epoch"),
    # D2a: keep the LPIPS network in eval mode (no dropout) while training.
    "lpips_eval": (False, bool, ["model.lpips_eval"], "train: LPIPS module stays in eval()"),
    # D2b: keep the frozen backbone / pixel branch in eval mode in stage 2.
    "freeze_frozen_bn": (False, bool, ["model.freeze_frozen_bn"], "train: frozen modules stay in eval()"),
    # D3: single-view camera metas use the identity pose (w2i @ inv(w2i)).
    "v1_identity_pose": (False, bool, ["model.v1_identity_pose"], "model: V=1 relative pose"),
    # D4: rotate Gaussian quaternions from the camera frame to the world frame.
    "rotate_gaussians_to_world": (False, bool,
                                  ["model.rotate_gaussians_to_world", "model.pixel_gs.rotate_gaussians_to_world"],
                                  "model: world-frame Gaussian rotations"),
    # D5: treat the cylinder angle theta as periodic.
    "theta_periodic": (False, bool,
                       ["model.volume_gs.encoder.theta_periodic", "model.volume_gs.gs_decoder.theta_periodic"],
                       "model: circular theta padding / sampling"),
    # D6: anchors at cell centres with symmetric offsets.
    "cell_center_anchor": (False, bool, ["model.volume_gs.gs_decoder.cell_center_anchor"],
                           "model: cell-centred volume anchors"),
    # D7: colour head for volume Gaussians.
    "rgb_retrieval": ("concat", ("concat", "visibility_softmax"), ["model.volume_gs.gs_decoder.rgb_retrieval"],
                      "model: volume colour retrieval head"),
    # D8: drop Gaussians with opacity below tau before rasterisation (0 = off).
    "prune_opacity": (0.0, float, [], "render: opacity pruning threshold"),
    # D9: interpolation used to read the depth prior at pixel-branch sample points.
    "pixel_depth_sampling": ("bilinear", ("bilinear", "nearest"), ["model.pixel_gs.pixel_depth_sampling"],
                             "model: prior-depth sampling mode"),
    # D10 (360Loc train split): one globally shuffled stream of (sequence, sample) pairs instead of
    # whole sequences one after another (~167 consecutive steps per sequence); each sequence still
    # gives times_per_scene samples per epoch, and every frame is decoded once per worker.
    "loc360_interleave": (False, bool, [], "data: interleave the 360Loc training sequences"),
    # D11 (360Loc Pan2): the depth loss ignores prior depth outside the loader's (near, far) range.
    "depth_valid_mask": (False, bool, ["model.depth_valid_mask"], "train: mask invalid prior depth in the depth loss"),
}


def _parse_value(name, raw):
    default, kind, _, _ = SWITCHES[name]
    if isinstance(raw, str):
        text = raw.strip()
        if kind is bool:
            low = text.lower()
            if low in ("1", "true", "yes", "on"):
                return True
            if low in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"switch {name}: expected true/false, got {raw!r}")
        if kind is float:
            return _check_range(name, float(text), raw)
        if isinstance(kind, tuple):
            if text not in kind:
                raise ValueError(f"switch {name}: expected one of {kind}, got {raw!r}")
            return text
        return ast.literal_eval(text)
    if kind is bool and not isinstance(raw, bool):
        raise ValueError(f"switch {name}: expected a bool, got {raw!r}")
    if kind is float:
        return _check_range(name, float(raw), raw)
    if isinstance(kind, tuple) and raw not in kind:
        raise ValueError(f"switch {name}: expected one of {kind}, got {raw!r}")
    return raw


def _check_range(name, value, raw):
    # prune_opacity is an opacity threshold; refuse bad values before any run directory is written.
    if name == "prune_opacity" and not (0.0 <= value < 1.0):  # also rejects NaN and inf
        raise ValueError(f"switch prune_opacity must be in [0, 1), got {raw!r}")
    return value


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


def _has_leaf(cfg, dotted):
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        if not hasattr(node, "get") or node.get(key, None) is None:
            return False
        node = node[key]
    return hasattr(node, "get") and keys[-1] in node


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
