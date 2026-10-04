"""Model classes; importing the package registers them in the mmdet3d MODELS registry."""

from .pixel import *
from .volume import *
from .backbone import *

# CylinderSplat: pixel branch, volume branch, joint model, 360Loc model
from .omni_gs_cylinder_pixel import OmniGaussianCylinderPixel
from .omni_gs_cylinder_volume import OmniGaussianCylinderVolume
from .omni_gs_cylinder_all import OmniGaussianCylinderAll
from .omni_gs_cylinder_volume_360loc_pan2 import OmniGaussianCylinderVolume360LocPan2

# Kansas City: the pixel model with a UniFuse depth prior (omni_gs_160x320_VIGOR_cylinder_pixel_unifuse.py)
from .omni_gs_cylinder_pixel_unifuse import OmniGaussianCylinderPixelUniFuse
