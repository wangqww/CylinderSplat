"""Image backbones; importing the package registers them in the MODELS registry."""

from .backbone_resnet import BackboneResnet

__all__ = ["BackboneResnet"]
