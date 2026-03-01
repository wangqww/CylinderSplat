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
from .geometry import sample_image_grid, fibonacci_sphere_grid, pad_pano, unpad_pano, get_world_rays_erp
from .ldm_unet.unet import UNetModel
from ..backbone.unimatch.geometry import points_grid
from .unifuse.networks import UniFuse
from .unifuse.networks.convert_module import erp_convert

def prepare_feat_proj_data_lists(
    features: Float[Tensor, "b v c h w"],
    extrinsics: Float[Tensor, "b v 4 4"],
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        cur_ref_pose_to_v0_list = []
        for v0, v1 in zip(init_view_order, cur_view_order):
            cur_ref_pose_to_v0_list.append(
                extrinsics[:, v1].clone().detach().float().inverse().type_as(extrinsics)
                @ extrinsics[:, v0].clone().detach()
            )
        cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
        pose_curr_lists.append(cur_ref_pose_to_v0s)

    return feat_lists, pose_curr_lists

def warp_with_pose_depth_candidates(
    feature1,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        points = points_grid(
            b, h, w, device=depth.device
        ).to(pose.dtype)  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = points.view(b, 3, -1)  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = points / points.norm(p=2, dim=1, keepdim=True).clamp(
            min=clamp_min_depth
        )  # normalize
        phi = torch.atan2(points[:, 0], points[:, 2])
        theta = torch.asin(points[:, 1])
        u = (phi + np.pi) / (2 * np.pi)
        v = (theta + np.pi / 2) / np.pi

        # normalize to [-1, 1]
        x_grid = 2 * u - 1
        y_grid = 2 * v - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature


@MODELS.register_module()
class PixelGaussian360Loc(BaseModule):

    def __init__(self,
                 image_height=160,
                 patchs_height=1,
                 patchs_width=1,
                 gh_cnn_layers=3,
                 gaussians_per_pixel=1,
                 **kwargs,
                 ):

        super().__init__()

        self.gaussians_per_pixel = gaussians_per_pixel
        feature_channels_list = [128, 96, 64, 32]
        self.costvolume_unet_feat_dims_list = [128, 64, 32]
        # gs_channels = 1 + 1 + 3 + 4 + 3 # offset, opacity, scale, rotation, rgb
        # self.gs_channels = gs_channels
        # self.to_gaussians = nn.Sequential(
        #     nn.GELU(),
        #     nn.Conv2d(out_embed_dims[0], gs_channels, 1),
        # )

        self.opt_act = torch.sigmoid
        self.scale_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid
        
        self.xy_act = torch.sigmoid
        self.offset_act = torch.tanh
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
        # Single gaussian parameters: offset(1) + opacity(1) + scale(3) + rotation(4) + rgb(3) + xy_offset(2) = 14
        self.gau_out_single = 1 + 1 + 3 + 4 + 3 + 2 # offset, opacity, scale, rotation, rgb, xy
        self.gau_out = self.gau_out_single * gaussians_per_pixel

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
            gau_in = 3 + feature_channels + 32  ## rgb+feature+raw_correlation
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

        # Cost volume refinement network and depth head
        corr_stage = 2
        feature_channels = feature_channels_list[corr_stage]
        input_channels = 1 + feature_channels # num_depth candidate + feature
        input_channels += 1  # add 1 for the previous depth
        # input_channels += 32 # add 32 for the mono depth feature
        channels = self.costvolume_unet_feat_dims_list[corr_stage]
        modules = [
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=(4,),
                channel_mult=(1, 1, 1),
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=2,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(channels, channels, 3, 1, 1)
        ]
        self.corr_refine_nets = nn.Sequential(*modules)
        # cost volume u-net skip connection
        self.regressor_residuals = nn.Conv2d(
            input_channels, channels, 1, 1, 0
        )

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_heads = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, 32, 3, 1, 1),
        )


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

    def forward(self, img, img_feats, depths_in, confs_in, pluckers_in, origins_in, directions_in, extrinsics_in, patch_idx=0, status="train"):
        """Forward training function."""
        bs, v, _, img_h, img_w = img.shape

        images_fullres = rearrange(img, "b v c h w -> (b v) c h w")
        confs_in_fullres = rearrange(confs_in, "b v ... -> (b v) ...")
        depths_in_fullres = rearrange(depths_in, "b v ... -> (b v) ...")
        origins_fullres = rearrange(origins_in, "b v h w c -> (b v) c h w")
        directions_fullres = rearrange(directions_in, "b v h w c -> (b v) c h w")
        pluckers_fullres = rearrange(pluckers_in, "b v ... -> (b v) ...")

        gaussians_all = {}
        gaussians_all["stages"] = []
        self.clean_padded_cache()
        # mono_erp_inputs = rearrange(mono_image, "b v c h w -> (b v) c h w")
        # mono_cube_inputs = rearrange(cube_image, "b v c h (f w) -> (b v) c h (f w)", f=2)
        # mono_depth = self.mono_depth(mono_erp_inputs, mono_cube_inputs)
        # mono_feat = mono_depth["mono_feat"] # (b v) c h w

        for stage_idx in range(2, 3):
            features = img_feats['trans_features'][stage_idx]
            corr_refine_net = self.corr_refine_nets
            regressor_residual = self.regressor_residuals
            depth_head = self.depth_heads
            b, v, c, h, w = features.shape
            feat_comb_lists, pose_curr_lists = prepare_feat_proj_data_lists(
                features, extrinsics_in
            )
            # cost volume constructions
            feat01 = feat_comb_lists[0]
            raw_correlation_in_lists = []
            disp_candi_curr = rearrange(depths_in, 'b v ... -> (v b) ...', v=v, b=bs)
            disp_candi_curr = F.interpolate(disp_candi_curr, size=(h, w), mode="nearest")
            for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                # sample feat01 from feat10 via camera projection
                feat01_warped = warp_with_pose_depth_candidates(
                    feat10,
                    pose_curr,
                    disp_candi_curr,
                    warp_padding_mode="zeros",
                )  # [vB, C, D, H, W]
                # calculate similarity
                raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                    1
                ) / (
                    c**0.5
                )  # [vB, D, H, W]
                raw_correlation_in_lists.append(raw_correlation_in)
            # average all cost volumes
            raw_correlation_in = torch.mean(
                torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
            )  # [vxb d, h, w]
            # mono_features = F.interpolate(mono_feat, size=raw_correlation_in.shape[-2:], mode="bilinear")
            raw_correlation_in = torch.cat((raw_correlation_in, feat01, disp_candi_curr), dim=1)
            # refine cost volume via 2D u-net
            raw_correlation = corr_refine_net(raw_correlation_in)  # (vb d h w)
            # apply skip connection
            raw_correlation = raw_correlation + regressor_residual(
                raw_correlation_in
            )
            raw_correlation = depth_head(raw_correlation)  # (vb 1 h w)
            raw_correlation_fullres = rearrange(raw_correlation, "(v b) ... -> (b v) ...", v=v, b=bs)


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
            raw_correlation = self.crop_patch(raw_correlation_fullres, stage_idx, patch_idx, "raw_correlation")
            # pluckers = rearrange(pluckers, "bv c h w -> bv h w c")
            # plucker_embeds = self.plucker_to_embed_list[stage_idx](pluckers)
            # plucker_embeds = rearrange(plucker_embeds, "bv h w c -> bv c h w")

            # cams_embeds = self.cams_embeds_list[stage_idx][None, :v, :, None, None].repeat(bs, 1, 1, images.shape[2], images.shape[3])
            # cams_embeds = rearrange(cams_embeds, "b v c h w -> (b v) c h w", v=v)
            
            # features = features + cams_embeds + plucker_embeds

            raw_gaussians_in = torch.cat((images, features, raw_correlation), dim=1)

            # fibonnaci sphere grid
            xy = getattr(self, f"gs_xy_{stage_idx}_{patch_idx}")
            full_grid = repeat(xy, "n xy -> bv n 1 xy", bv=bs * v)

            xy_ray = rearrange(xy, "n xy -> 1 1 n xy") # [1, 1, N, 2]
            xy_ray = xy_ray.repeat(bs, v, 1, 1)
            xy_ray = xy_ray / 2 + 0.5

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
            
            # Expand the sample grid to handle multiple gaussians per pixel
            if self.gaussians_per_pixel > 1:
                # full_grid shape: [bv, c, n, 2], we want to repeat along the n dimension
                full_grid_expanded = repeat(full_grid, "bv h w d -> bv (h g) w d", g=self.gaussians_per_pixel)
                patch_grid_expanded = repeat(patch_grid, "bv h w d -> bv (h g) w d", g=self.gaussians_per_pixel)
            else:
                full_grid_expanded = full_grid
                patch_grid_expanded = patch_grid
            depths_in_curr = F.grid_sample(depths_in_fullres, full_grid_expanded, padding_mode="border")
            origins_curr = F.grid_sample(origins_fullres, full_grid_expanded, padding_mode="border")
            directions_curr = F.grid_sample(directions_fullres, full_grid_expanded, padding_mode="border")
            raw_gaussians = F.grid_sample(raw_gaussians, patch_grid, padding_mode="border")

            if self.gaussians_per_pixel > 1:
                depths_in_curr = rearrange(depths_in_curr, "(b v) c (n g) 1 -> b v n g c 1", v=v, b=bs, g=self.gaussians_per_pixel).squeeze(-1)
                origins_curr = rearrange(origins_curr, "(b v) c (n g) 1 -> b v n g c 1", v=v, b=bs, g=self.gaussians_per_pixel).squeeze(-1)
                directions_curr = rearrange(directions_curr, "(b v) c (n g) 1 -> b v n g c 1", v=v, b=bs, g=self.gaussians_per_pixel).squeeze(-1)
            else:
                depths_in_curr = rearrange(depths_in_curr, "(b v) c n 1 -> b v n 1 c 1", v=v, b=bs).squeeze(-1)
                origins_curr = rearrange(origins_curr, "(b v) c n 1 -> b v n 1 c 1", v=v, b=bs).squeeze(-1)
                directions_curr = rearrange(directions_curr, "(b v) c n 1 -> b v n 1 c 1", v=v, b=bs).squeeze(-1)
            raw_gaussians = rearrange(raw_gaussians, "(b v) c n 1 -> b v n c", v=v, b=bs)
            # depths_in_curr = F.grid_sample(depths, patch_grid, padding_mode="border")
            # depths_in_curr = rearrange(depths_in_curr, "(b v) c n 1 -> b v n c", v=v, b=bs)
            # origins_curr = F.grid_sample(origins, patch_grid, padding_mode="border")
            # origins_curr = rearrange(origins_curr, "(b v) c n 1 -> b v n c", v=v, b=bs)
            # directions_curr = F.grid_sample(directions, patch_grid, padding_mode="border")
            # directions_curr = rearrange(directions_curr, "(b v) c n 1 -> b v n c", v=v, b=bs)

            raw_gaussians_final = self.gaussians_mlp_list[stage_idx](raw_gaussians)
            if self.gaussians_per_pixel > 1:
                gaussians = rearrange(raw_gaussians_final, "b v n (g c) -> b v n g c",
                                    b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel, c=self.gau_out_single)
            else:
                gaussians = rearrange(raw_gaussians_final, "b v n c -> b v n 1 c",
                                    b=bs, v=v, n=xy.shape[0])

            # Extract parameters for each gaussian
            offsets = gaussians[..., :, 0:1]      # [B, V, N, G, 1]
            opacities = self.opt_act(gaussians[..., :, 1:2])     # [B, V, N, G, 1]
            scales = self.scale_act(gaussians[..., :, 2:5])      # [B, V, N, G, 3]
            rotations = self.rot_act(gaussians[..., :, 5:9])     # [B, V, N, G, 4]
            rgbs = self.rgb_act(gaussians[..., :, 9:12])         # [B, V, N, G, 3]
            xy_offset = self.xy_act(gaussians[..., :, 12:14])     # [B, V, N, G, 2]

            # Reshape xy_offset and xy_ray for coordinate calculation
            if self.gaussians_per_pixel > 1:
                # Expand xy_ray to match multiple gaussians per pixel first
                xy_ray_expanded = repeat(xy_ray, "b v n xy -> b v n g xy", g=self.gaussians_per_pixel)
                xy_ray_expanded = rearrange(xy_ray_expanded, "b v n g xy -> b (v n g) xy")
                # Then reshape xy_offset to match
                xy_offset = rearrange(xy_offset, "b v n g c -> b (v n g) c")
            else:
                xy_offset = rearrange(xy_offset, "b v n 1 c -> b (v n) c")
                xy_ray_expanded = rearrange(xy_ray, "b v n xy -> b (v n) xy")

            pixel_size = 1 / torch.tensor((w, h), device=xy.device).type_as(xy) # [2]
            lat = full_grid_expanded[..., 0, 1] * np.pi / 2
            r = torch.cos(lat)
            r[r < 1e-2] = 1e-2
            pixel_width = 1 / r
            pixel_height = pixel_width.new_ones(pixel_width.shape)
            pixel_size = torch.stack((pixel_width, pixel_height), dim=-1) * pixel_size
            pixel_size = rearrange(pixel_size, "(b v) n xy -> b (v n) xy", v=v, b=bs)

            coordinates = xy_ray_expanded + (xy_offset - 0.5) * pixel_size # [B, V*N*G, 2]
            coordinates = rearrange(coordinates, "b (v n g) xy -> b v (n g) xy", v=v, n=xy.shape[0], g=self.gaussians_per_pixel)

            # Expand extrinsics to match multiple gaussians
            if self.gaussians_per_pixel > 1:
                extrinsics_expanded = repeat(extrinsics_in, "b v c1 c2 -> b v (n g) c1 c2", n=xy.shape[0], g=self.gaussians_per_pixel)
            else:
                extrinsics_expanded = repeat(extrinsics_in, "b v c1 c2 -> b v (n) c1 c2", n=xy.shape[0])

            origins, directions = get_world_rays_erp(coordinates, extrinsics_expanded)
            if self.gaussians_per_pixel > 1:
                origins = origins.view(bs, v, xy.shape[0], self.gaussians_per_pixel, 3)
                directions = directions.view(bs, v, xy.shape[0], self.gaussians_per_pixel, 3)
                origins = rearrange(origins, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                directions = rearrange(directions, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
            else:
                origins = origins.view(bs, v, xy.shape[0], 3)
                directions = directions.view(bs, v, xy.shape[0], 3)
                origins = rearrange(origins, "b v n c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                directions = rearrange(directions, "b v n c -> b (v n) c", b=bs, v=v, n=xy.shape[0])

            # Reshape parameters to handle multiple gaussians per pixel
            if self.gaussians_per_pixel > 1:
                offsets = rearrange(offsets, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                opacities = rearrange(opacities, "b v n g c -> b v (n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                scales = rearrange(scales, "b v n g c -> b v (n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                rotations = rearrange(rotations, "b v n g c -> b v (n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                rgbs = rearrange(rgbs, "b v n g c -> b v (n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                depths_in_curr = rearrange(depths_in_curr, "b v n g c-> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                origins_curr = rearrange(origins_curr, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                origins_curr = origins_curr
                directions_curr = rearrange(directions_curr, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                directions_curr = directions_curr
            else:
                offsets = rearrange(offsets, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                opacities = rearrange(opacities, "b v n 1 c -> b v n c", b=bs, v=v, n=xy.shape[0])
                scales = rearrange(scales, "b v n 1 c -> b v n c", b=bs, v=v, n=xy.shape[0])
                rotations = rearrange(rotations, "b v n 1 c -> b v n c", b=bs, v=v, n=xy.shape[0])
                rgbs = rearrange(rgbs, "b v n 1 c -> b v n c", b=bs, v=v, n=xy.shape[0])

                depths_in_curr = rearrange(depths_in_curr, "b v n 1 c-> b (v n) c", b=bs, v=v, n=xy.shape[0])
                origins_curr = rearrange(origins_curr, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                origins_curr = origins_curr
                directions_curr = rearrange(directions_curr, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                directions_curr = directions_curr
            depth_pred = (depths_in_curr + offsets).clamp(min=0.0)
            # means = origins_curr + directions_curr * depth_pred[..., None]
            means = origins + directions * depth_pred # [B, V*N*G, 3]
            means = rearrange(means, "b (v n g) c -> b v (n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
            depth_pred = rearrange(depth_pred, "b (v n g) c -> b v (n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
            # means = means + offsets

            # new scale
            scales_new = self.scale_min + (self.scale_max - self.scale_min) * scales
            pixel_size = 1 / torch.tensor((w, h), dtype=scales_new.dtype, device=scales_new.device)
            multiplier = self.get_scale_multiplier(pixel_size)
            scales_new = scales_new * depth_pred * multiplier[..., None]

            gaussians_final = torch.cat([means, rgbs, opacities, rotations, scales_new], dim=-1)

            # Handle features for multiple gaussians per pixel
            if self.gaussians_per_pixel > 1:
                features_expanded = repeat(raw_gaussians, "b v n c -> b v (n g) c", g=self.gaussians_per_pixel)
                gaussians_raw = rearrange(raw_gaussians_final, "b v n (g c) -> b v (n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel, c=self.gau_out_single)
            else:
                features_expanded = raw_gaussians
                gaussians_raw = raw_gaussians_final

            gaussians_stage = {
                "gaussians": gaussians_final,
                "features": features_expanded,
                "gaussians_raw": gaussians_raw,
            }
            gaussians_all["stages"].append(gaussians_stage)
        
        # gaussians_all.update(gaussians_stage)
        gaussians_all['gaussians'] = torch.cat([g["gaussians"] for g in gaussians_all["stages"]], dim=2)
        gaussians_all['features'] = torch.cat([g["features"] for g in gaussians_all["stages"]], dim=2)
        
        gaussians_all['gaussians_raw'] = torch.cat([g["gaussians_raw"] for g in gaussians_all["stages"]], dim=2)
        return gaussians_all
    
    def get_scale_multiplier(
        self,
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * pixel_size.new_tensor([2 * np.pi, np.pi]) * pixel_size
        return xy_multipliers.sum(dim=-1)