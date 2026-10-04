"""Volume-branch (triplane) modules; importing the package registers them in the MODELS registry."""

from .cross_view_hybrid_attention import TPVCrossViewHybridAttention
from .image_cross_attention import TPVImageCrossAttention
from .positional_encoding import TPVFormerPositionalEncoding
from .tpvformer_layer import TPVFormerLayer

# cylindrical triplanes (CylinderSplat)
from .tpvformer_encoder_cylinder import TPVFormerEncoderCylinder
from .volume_gs_cylinder import VolumeGaussianCylinder
from .volume_gs_decoder_cylinder import VolumeGaussianDecoderCylinder

# ablations: Cartesian and spherical triplanes
from .tpvformer_encoder_decare import TPVFormerEncoderDecare
from .volume_gs_decare import VolumeGaussianDecare
from .volume_gs_decoder_decare import VolumeGaussianDecoderDecare
from .tpvformer_encoder_spherical import TPVFormerEncoderSpherical
from .volume_gs_spherical import VolumeGaussianSpherical
from .volume_gs_decoder_spherical import VolumeGaussianDecoderSpherical

__all__ = [
    "TPVCrossViewHybridAttention",
    "TPVImageCrossAttention",
    "TPVFormerPositionalEncoding",
    "TPVFormerLayer",
    "TPVFormerEncoderCylinder",
    "VolumeGaussianCylinder",
    "VolumeGaussianDecoderCylinder",
    "TPVFormerEncoderDecare",
    "VolumeGaussianDecare",
    "VolumeGaussianDecoderDecare",
    "TPVFormerEncoderSpherical",
    "VolumeGaussianSpherical",
    "VolumeGaussianDecoderSpherical",
]
