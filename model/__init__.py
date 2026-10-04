"""Model classes; importing the package registers them in the mmdet3d MODELS registry."""

from .pixel import *
from .volume import *
from .backbone import *

# CylinderSplat: pixel branch, volume branch, joint model, 360Loc model
from .omni_gs_cylinder_pixel import OmniGaussianCylinderPixel
from .omni_gs_cylinder_volume import OmniGaussianCylinderVolume
from .omni_gs_cylinder_all import OmniGaussianCylinderAll
from .omni_gs_cylinder_volume_360loc_pan2 import OmniGaussianCylinderVolume360LocPan2

# ablations: other depth priors and triplane coordinates
from .omni_gs_cylinder_pixel_unifuse import OmniGaussianCylinderPixelUniFuse
from .omni_gs_cylinder_volume_unifuse import OmniGaussianCylinderVolumeUniFuse
from .omni_gs_cylinder_all_unifuse import OmniGaussianCylinderAllUniFuse
from .omni_gs_cylinder_pixel_depthanywhere import OmniGaussianCylinderPixelDepthanywhere
from .omni_gs_decare_volume import OmniGaussianDecareVolume
from .omni_gs_spherical_volume import OmniGaussianSphericalVolume
