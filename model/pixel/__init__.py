"""Pixel-branch Gaussian heads; importing the package registers them in the MODELS registry."""

from .pixel_gs import PixelGaussian
from .pixel_gs_360loc import PixelGaussian360Loc
from .pixel_gs_512 import PixelGaussian512

__all__ = ["PixelGaussian", "PixelGaussian360Loc", "PixelGaussian512"]
