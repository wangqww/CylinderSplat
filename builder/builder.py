"""Build a model from its config through the mmdet3d MODELS registry."""

from model import *  # noqa: F401,F403  (registers every model class)
from mmengine import build_from_cfg
from mmdet3d.registry import MODELS


def build(model_config):
    net = build_from_cfg(model_config, MODELS)
    net.init_weights()
    return net
