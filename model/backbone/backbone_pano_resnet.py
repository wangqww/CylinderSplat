import functools
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.models import ResNet

from .backbone import Backbone


@dataclass
class BackboneResnetCfg:
    name: Literal["resnet"]
    model: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"
    ]
    num_layers: int
    use_first_pool: bool
    d_out: int


class BackboneResnet(Backbone[BackboneResnetCfg]):
    model: ResNet

    def __init__(self, d_in: int = 3) -> None:
        super().__init__()

        assert d_in == 3
        self.num_layers = 4
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50", trust_repo=True)

        # Set up projections
        self.projections = nn.ModuleDict({})
        for index in range(1, self.num_layers):
            key = f"layer{index}"
            block = getattr(self.model, key)
            conv_index = 1
            try:
                while True:
                    d_layer_out = getattr(block[-1], f"conv{conv_index}").out_channels
                    conv_index += 1
            except AttributeError:
                pass
            self.projections[key] = nn.Conv2d(d_layer_out, 512, 1)

        # Add a projection for the first layer.
        self.projections["layer0"] = nn.Conv2d(
            self.model.conv1.out_channels, 512, 1
        )

    def forward(
        self,
        img,
    ) -> Float[Tensor, "batch view d_out height width"]:
        # Merge the batch dimensions.
        b, c, h, w = img.shape
        x = img
        # Run the images through the resnet.
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        features = [self.projections["layer0"](x)]

        # Propagate the input through the resnet's layers.
        for index in range(1, self.num_layers):
            key = f"layer{index}"
            x = getattr(self.model, key)(x)
            features.append(self.projections[key](x))

        # Upscale the features.
        features = [
            F.interpolate(f, (h, w), mode="bilinear", align_corners=True)
            for f in features
        ]
        features = torch.stack(features).sum(dim=0)

        # Separate batch dimensions.
        return features

    @property
    def d_out(self) -> int:
        return 512