from model import *
from mmengine import Registry, build_from_cfg
from mmdet3d.registry import MODELS

def build(model_config):
    net = build_from_cfg(model_config, MODELS)
    net.init_weights()
    return net

