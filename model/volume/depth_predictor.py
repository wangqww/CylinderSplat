import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .conversions import relative_disparity_to_depth
from .distribution_sampler import DistributionSampler


class DepthPredictor(nn.Module):
    projection: nn.Sequential
    sampler: DistributionSampler
    num_samples: int
    num_surfaces: int

    def __init__(
        self,
        d_in: int,
        num_samples: int,
        num_surfaces: int,
        use_transmittance: bool,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_in, 2 * num_samples * num_surfaces),
        )
        self.sampler = DistributionSampler()
        self.num_samples = num_samples
        self.num_surfaces = num_surfaces
        self.use_transmittance = use_transmittance

        # This exists for hooks to latch onto.
        self.to_pdf = nn.Softmax(dim=-1)
        self.to_offset = nn.Sigmoid()

    def forward(
        self,
        features: Float[Tensor, "batch ray channel"],
        near: Float[Tensor, "batch"],
        far: Float[Tensor, "batch"],
        deterministic: bool,
        gaussians_per_pixel: int,
    ) -> tuple[
        Float[Tensor, "batch ray sample"],  # depth
        Float[Tensor, "batch ray sample"],  # pdf
    ]:
        s = self.num_samples

        # Convert the features into a depth distribution plus intra-bucket offsets.
        features = self.projection(features)
        pdf_raw, offset_raw = rearrange(
            features, "... (dpt c) -> c ... dpt", c=2
        )
        pdf = self.to_pdf(pdf_raw)
        offset = self.to_offset(offset_raw)

        # Sample from the depth distribution.
        index, pdf_i = self.sampler.sample(pdf, deterministic, gaussians_per_pixel)
        offset = self.sampler.gather(index, offset)

        # Convert the sampled bucket and offset to a depth.
        relative_disparity = (index + offset) / s
        depth = relative_disparity_to_depth(
            relative_disparity,
            rearrange(near, "b -> b () ()"),
            rearrange(far, "b -> b () ()"),
        )

        # Compute opacity from PDF.
        if self.use_transmittance:
            partial = pdf.cumsum(dim=-1)
            partial = torch.cat(
                (torch.zeros_like(partial[..., :1]), partial[..., :-1]), dim=-1
            )
            opacity = pdf / (1 - partial + 1e-10)
            opacity = self.sampler.gather(index, opacity)
        else:
            opacity = pdf_i

        return depth, opacity
