import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import imageio
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import warnings
from einops import rearrange, einsum, repeat
from plyfile import PlyData, PlyElement
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor
from ..utils.ops import get_ray_directions, get_rays
from torch.nn.init import normal_
from .geometry import sample_image_grid, fibonacci_sphere_grid, pad_pano, unpad_pano

@MODELS.register_module()
class PixelGaussian512(BaseModule):

    def __init__(self,
                 image_height=160,
                 patchs_height=1,
                 patchs_width=1,
                 gh_cnn_layers=3,
                 **kwargs,
                 ):

        super().__init__()

        feature_channels_list = [128, 96, 64, 32]

        # gs_channels = 1 + 1 + 3 + 4 + 3 # offset, opacity, scale, rotation, rgb
        # self.gs_channels = gs_channels
        # self.to_gaussians = nn.Sequential(
        #     nn.GELU(),
        #     nn.Conv2d(out_embed_dims[0], gs_channels, 1),
        # )

        self.opt_act = torch.sigmoid
        # self.scale_act = lambda x: F.softplus(x) * 0.01
        self.scale_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid

        self.scale_min = 0.5
        self.scale_max = 15.0

        self.to_gaussians_list = nn.ModuleList()
        self.gaussians_mlp_list = nn.ModuleList()
        self.plucker_to_embed_list = nn.ModuleList()
        self.cams_embeds_list = nn.ParameterList()

        self.full_shape = []
        self.gh_stages = len(feature_channels_list)
        self.padded_cache = [{} for _ in range(self.gh_stages)]
        self.padding = gh_cnn_layers
        self.gau_out = 1 + 1 + 3 + 4 + 3 # offset, opacity, scale, rotation, rgb

        for stage_idx, feature_channels in enumerate(feature_channels_list):
            # Stage shape
            scale = 2**(self.gh_stages - stage_idx - 1)
            stage_height = image_height // scale
            self.full_shape.append((stage_height, stage_height * 2))

            # Gaussians xy and patch range
            patch_info = self.stage_patch_info(stage_idx, patchs_height, patchs_width)
            for patch_idx, (gs_xy_patch, range_xy_patch, range_hw_patch) in enumerate(zip(*patch_info)):
                key = f"{stage_idx}_{patch_idx}"
                self.register_buffer(f"gs_xy_{key}", gs_xy_patch, persistent=False)
                self.register_buffer(f"range_hw_{key}", range_hw_patch, persistent=False)
                self.register_buffer(f"range_xy_{key}", range_xy_patch, persistent=False)

            # Gaussians prediction: covariance, color
            gau_in = 3 + 1 + 1 + feature_channels  ## rgb+depth+conf+feature
            gau_hid = 128
            if stage_idx > 0:
                gau_in += gau_hid
            self.to_gaussians_list.append(self.gaussians_cnn(
                gau_in, gau_hid, gau_hid, gh_cnn_layers
            ))
            self.gaussians_mlp_list.append(self.fibo_mlp(
                gau_hid, gau_hid, self.gau_out, 1
            ))
            self.plucker_to_embed_list.append(
                nn.Linear(6, feature_channels)
            )
            cams_embeds = nn.Parameter(torch.Tensor(6, feature_channels)) # 使用 torch.empty 更标准
            nn.init.normal_(cams_embeds, mean=0.0, std=0.02)
            self.cams_embeds_list.append(cams_embeds)

    def gaussians_cnn(self, in_channels, hidden_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels, hidden_channels, 3, 1, 1))
            layers.append(nn.GELU())
            in_channels = hidden_channels
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def fibo_mlp(self, in_channels, hidden_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_channels, hidden_channels))
            layers.append(nn.GELU())
            in_channels = hidden_channels
        layers.append(nn.Linear(in_channels, out_channels))
        return nn.Sequential(*layers)

    def stage_patch_info(self, stage_idx, patchs_height, patchs_width):
        h, w = self.full_shape[stage_idx]
        patch_h, patch_w = h // patchs_height, w // patchs_width

        range_hw = []
        for i in range(patchs_height):
            for j in range(patchs_width):
                h_start = i * patch_h
                h_end = (i + 1) * patch_h
                w_start = j * patch_w
                w_end = (j + 1) * patch_w
                patch_hw = torch.tensor([[h_start, h_end], [w_start, w_end]])
                range_hw.append(patch_hw)

        gs_xy = []

        lonlat = fibonacci_sphere_grid(c=w)
        xy = lonlat / lonlat.new_tensor([np.pi, np.pi / 2])
        for patch_hw in range_hw:
            patch_xy = patch_hw.flip(0).float()
            patch_xy = patch_xy / patch_xy.new_tensor([w, h]).unsqueeze(-1) * 2 - 1
            start, end = patch_xy.unbind(1)
            eps = 1e-6
            start[start <= -1 + eps] = -1 - eps
            end[end >= 1 - eps] = 1 + eps
            xy_patch = xy[(xy[:, 0] >= start[0]) & (xy[:, 0] < end[0]) & (xy[:, 1] >= start[1]) & (xy[:, 1] < end[1])]
            gs_xy.append(xy_patch)

        range_xy = []
        for patch_hw in range_hw:
            patch_hw[:, 1] += self.padding
            patch_hw[:, 0] -= self.padding
            patch_xy = patch_hw.flip(0).float()
            patch_xy = patch_xy / patch_xy.new_tensor([w, h]).unsqueeze(-1) * 2 - 1
            range_xy.append(patch_xy)

        return gs_xy, range_xy, range_hw

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def map_patch_xy(self, xy, stage_idx, patch_idx):
        range_xy = getattr(self, f"range_xy_{stage_idx}_{patch_idx}")
        patch_size = range_xy[:, 1] - range_xy[:, 0]
        xy = (xy - range_xy[:, 0]) / patch_size * 2 - 1
        return xy

    def crop_patch(self, f, stage_idx, patch_idx, key):
        full = self.cache_padding(f, stage_idx, key)
        range_hw = getattr(self, f"range_hw_{stage_idx}_{patch_idx}")
        range_hw = range_hw + self.padding
        patch = full[..., range_hw[0, 0]:range_hw[0, 1], range_hw[1, 0]:range_hw[1, 1]]
        return patch

    def cache_padding(self, f, stage_idx, key):
        if not hasattr(self, "padded_cache"):
            self.clean_padded_cache()
        stage = self.padded_cache[stage_idx]
        if key in stage:
            return stage[key]
        full = F.interpolate(f, size=self.full_shape[stage_idx], mode="bilinear")
        full = pad_pano(full, self.padding)
        stage[key] = full
        return full
    
    def clean_padded_cache(self):
        # must be called after forward
        self.padded_cache = [{} for _ in range(self.gh_stages)]

    def forward(self, img, img_feats, depths_in, confs_in, pluckers_in, origins_in, directions_in, patch_idx=0, status="train"):
        """Forward training function."""
        bs, v, _, _, _ = img.shape

        images_fullres = rearrange(img, "b v c h w -> (b v) c h w")
        confs_in_fullres = rearrange(confs_in, "b v ... -> (b v) ...")
        depths_in_fullres = rearrange(depths_in, "b v ... -> (b v) ...")
        origins_fullres = rearrange(origins_in, "b v h w c -> (b v) c h w")
        directions_fullres = rearrange(directions_in, "b v h w c -> (b v) c h w")
        pluckers_fullres = rearrange(pluckers_in, "b v ... -> (b v) ...")

        gaussians_all = {}
        gaussians_all["stages"] = []
        self.clean_padded_cache()
        for stage_idx, stage in enumerate(img_feats['trans_features']):
            _, _, _, h, w = stage.shape
            features = rearrange(stage, "b v ... -> (b v) ...")
            
            # feature refine
            features = self.crop_patch(features, stage_idx, patch_idx, "features")
            images = self.crop_patch(images_fullres, stage_idx, patch_idx, "images")
            confs = self.crop_patch(confs_in_fullres, stage_idx, patch_idx, "confs")
            depths = self.crop_patch(depths_in_fullres, stage_idx, patch_idx, "depths")
            pluckers = self.crop_patch(pluckers_fullres, stage_idx, patch_idx, "pluckers")
            origins = self.crop_patch(origins_fullres, stage_idx, patch_idx, "origins")
            directions = self.crop_patch(directions_fullres, stage_idx, patch_idx, "directions")
            
            # pluckers = rearrange(pluckers, "bv c h w -> bv h w c")
            # plucker_embeds = self.plucker_to_embed_list[stage_idx](pluckers)
            # plucker_embeds = rearrange(plucker_embeds, "bv h w c -> bv c h w")

            # cams_embeds = self.cams_embeds_list[stage_idx][None, :v, :, None, None].repeat(bs, 1, 1, images.shape[2], images.shape[3])
            # cams_embeds = rearrange(cams_embeds, "b v c h w -> (b v) c h w", v=v)
            
            # features = features + cams_embeds + plucker_embeds

            raw_gaussians_in = torch.cat((images, confs, depths / 20.0, features), dim=1)

            # fibonnaci sphere grid
            xy = getattr(self, f"gs_xy_{stage_idx}_{patch_idx}")
            full_grid = repeat(xy, "n xy -> bv n 1 xy", bv=bs * v)

            # add residual
            if stage_idx > 0:
                last_raw_gaussians = F.interpolate(
                    last_raw_gaussians, scale_factor=2, mode="bilinear")
                last_raw_gaussians = unpad_pano(last_raw_gaussians, self.padding)
                raw_gaussians_in = torch.cat([raw_gaussians_in, last_raw_gaussians], dim=1)

            delta_raw_gaussians = self.to_gaussians_list[stage_idx](raw_gaussians_in)

            # add residual
            if stage_idx == 0:
                raw_gaussians = delta_raw_gaussians
            else:
                raw_gaussians = last_raw_gaussians + delta_raw_gaussians

            last_raw_gaussians = raw_gaussians

            patch_grid = repeat(self.map_patch_xy(xy, stage_idx, patch_idx), "n xy -> bv n 1 xy", bv=bs*v)
            
            depths_in_curr = F.grid_sample(depths_in_fullres, full_grid, padding_mode="border")
            depths_in_curr = rearrange(depths_in_curr, "(b v) c n 1 -> b v n c", v=v, b=bs)
            origins_curr = F.grid_sample(origins_fullres, full_grid, padding_mode="border")
            origins_curr = rearrange(origins_curr, "(b v) c n 1 -> b v n c", v=v, b=bs)
            directions_curr = F.grid_sample(directions_fullres, full_grid, padding_mode="border")
            directions_curr = rearrange(directions_curr, "(b v) c n 1 -> b v n c", v=v, b=bs)

            # depths_in_curr = F.grid_sample(depths, patch_grid, padding_mode="border")
            # depths_in_curr = rearrange(depths_in_curr, "(b v) c n 1 -> b v n c", v=v, b=bs)
            # origins_curr = F.grid_sample(origins, patch_grid, padding_mode="border")
            # origins_curr = rearrange(origins_curr, "(b v) c n 1 -> b v n c", v=v, b=bs)
            # directions_curr = F.grid_sample(directions, patch_grid, padding_mode="border")
            # directions_curr = rearrange(directions_curr, "(b v) c n 1 -> b v n c", v=v, b=bs)

            raw_gaussians = F.grid_sample(raw_gaussians, patch_grid, padding_mode="border")
            raw_gaussians = rearrange(raw_gaussians, "(b v) c n 1 -> b v n c", v=v, b=bs)

            raw_gaussians_final = self.gaussians_mlp_list[stage_idx](raw_gaussians)
            gaussians = rearrange(raw_gaussians_final, "b v n c -> b (v n) c",
                                b=bs, v=v, c=self.gau_out)
            
            offsets = gaussians[..., :1]
            opacities = self.opt_act(gaussians[..., 1:2])
            scales = self.scale_act(gaussians[..., 2:5])
            rotations = self.rot_act(gaussians[..., 5:9])
            rgbs = self.rgb_act(gaussians[..., 9:12])

            depths_in_curr = rearrange(depths_in_curr, "b v n c-> b (v n) c", b=bs, v=v)
            origins_curr = rearrange(origins_curr, "b v n c -> b (v n) c")
            origins_curr = origins_curr.unsqueeze(-2)
            directions_curr = rearrange(directions_curr, "b v n c -> b (v n) c")
            directions_curr = directions_curr.unsqueeze(-2)
            depth_pred = (depths_in_curr + offsets).clamp(min=0.0)
            means = origins_curr + directions_curr * depth_pred[..., None]
            means = rearrange(means, "b r n c -> b (r n) c")
            # means = means + offsets

            # new scale
            scales_new = self.scale_min + (self.scale_max - self.scale_min) * scales
            pixel_size = 1 / torch.tensor((w, h), dtype=scales_new.dtype, device=scales_new.device)
            multiplier = self.get_scale_multiplier(pixel_size)
            scales_new = scales_new * depth_pred * multiplier[..., None]

            gaussians_final = torch.cat([means, rgbs, opacities, rotations, scales_new], dim=-1)
            gaussians_stage = {
                "gaussians": gaussians_final,
                "features": rearrange(raw_gaussians, "b v n c -> b (v n) c", b=bs, v=v).contiguous(),
            }
            gaussians_all["stages"].append(gaussians_stage)
        
        # gaussians_all.update(gaussians_stage)
        gaussians_all['gaussians'] = torch.cat([g["gaussians"] for g in gaussians_all["stages"]], dim=1)
        gaussians_all['features'] = torch.cat([g["features"] for g in gaussians_all["stages"]], dim=1)
        
        return gaussians_all

    def get_scale_multiplier(
        self,
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * pixel_size.new_tensor([2 * np.pi, np.pi]) * pixel_size
        return xy_multipliers.sum(dim=-1)