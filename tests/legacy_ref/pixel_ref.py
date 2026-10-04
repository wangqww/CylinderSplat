"""Frozen legacy copies (commit f7b20b9) for the pixel-head and renderer switch tests.

Every block between a "# >>> BEGIN <name>" and "# <<< END <name>" marker is copied
verbatim from `git show f7b20b9:<path>` (path and line range on the BEGIN line;
test_switches_identity_pixel.py re-checks this when git and the commit are
available). Do not edit the blocks: the tests compare the live code, with every
switch off, against them. Only the imports and the three stand-in names below are
not part of the legacy code.
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from model.backbone.unimatch.geometry import points_grid
from model.pixel.geometry import get_world_rays_erp, unpad_pano

# The legacy GaussianRenderer.render below reads these module globals; the tests
# bind them (monkeypatch) to CPU stand-ins of the CUDA rasteriser and of the
# .cuda()-only camera helper.
GaussianRasterizationSettings = None
GaussianRasterizer = None
get_cam_info_gaussian = None


# >>> BEGIN helpers model/pixel/pixel_gs.py:24-112 (byte-identical to model/pixel/pixel_gs_360loc.py:24-112)
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
# <<< END helpers


# Single legacy statements, wrapped in functions (compared after stripping indentation).
def pixel_gs_depth_sample(depths_in_fullres, full_grid_expanded):
    # >>> BEGIN stmt model/pixel/pixel_gs.py:460-460
    depths_in_curr = F.grid_sample(depths_in_fullres, full_grid_expanded, padding_mode="border")
    # <<< END stmt
    return depths_in_curr


def pixel_gs_360loc_depth_sample(depths_in_fullres, full_grid_expanded):
    # >>> BEGIN stmt model/pixel/pixel_gs_360loc.py:450-450
    depths_in_curr = F.grid_sample(depths_in_fullres, full_grid_expanded, padding_mode="border")
    # <<< END stmt
    return depths_in_curr


def pixel_gs_512_depth_sample(depths_in_fullres, full_grid):
    # >>> BEGIN stmt model/pixel/pixel_gs_512.py:247-247
    depths_in_curr = F.grid_sample(depths_in_fullres, full_grid, padding_mode="border")
    # <<< END stmt
    return depths_in_curr


def pixel_gs_world_concat(means, rgbs, opacities, rotations, scales_new):
    # >>> BEGIN stmt model/pixel/pixel_gs.py:574-574
    gaussians_final = torch.cat([means, rgbs, opacities, rotations, scales_new], dim=-1)
    # <<< END stmt
    return gaussians_final


def pixel_gs_360loc_world_concat(means, rgbs, opacities, rotations, scales_new):
    # >>> BEGIN stmt model/pixel/pixel_gs_360loc.py:565-565
    gaussians_final = torch.cat([means, rgbs, opacities, rotations, scales_new], dim=-1)
    # <<< END stmt
    return gaussians_final


def pixel_gs_512_world_concat(means, rgbs, opacities, rotations, scales_new):
    # >>> BEGIN stmt model/pixel/pixel_gs_512.py:290-290
    gaussians_final = torch.cat([means, rgbs, opacities, rotations, scales_new], dim=-1)
    # <<< END stmt
    return gaussians_final


class LegacyPixelGaussian:
    """Holder of the legacy PixelGaussian.forward; call it as LegacyPixelGaussian.forward(head, ...)."""

    # >>> BEGIN method model/pixel/pixel_gs.py:328-598
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
            
            if len(raw_correlation_in_lists) == 0:
                raw_correlation_in = (feat01.unsqueeze(2) * feat01.unsqueeze(2)).sum(
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
                opacities = rearrange(opacities, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                scales = rearrange(scales, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                rotations = rearrange(rotations, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                rgbs = rearrange(rgbs, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)

                depths_in_curr = rearrange(depths_in_curr, "b v n g c-> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                origins_curr = rearrange(origins_curr, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                origins_curr = origins_curr
                directions_curr = rearrange(directions_curr, "b v n g c -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel)
                directions_curr = directions_curr
            else:
                offsets = rearrange(offsets, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                opacities = rearrange(opacities, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                scales = rearrange(scales, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                rotations = rearrange(rotations, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                rgbs = rearrange(rgbs, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])

                depths_in_curr = rearrange(depths_in_curr, "b v n 1 c-> b (v n) c", b=bs, v=v, n=xy.shape[0])
                origins_curr = rearrange(origins_curr, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                origins_curr = origins_curr
                directions_curr = rearrange(directions_curr, "b v n 1 c -> b (v n) c", b=bs, v=v, n=xy.shape[0])
                directions_curr = directions_curr
            depth_pred = (depths_in_curr + offsets).clamp(min=0.0)
            # means = origins_curr + directions_curr * depth_pred[..., None]
            means = origins + directions * depth_pred
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
                features_expanded = rearrange(features_expanded, "b v n c -> b (v n) c", b=bs, v=v, n=xy.shape[0]*self.gaussians_per_pixel).contiguous()
                gaussians_raw = rearrange(raw_gaussians_final, "b v n (g c) -> b (v n g) c", b=bs, v=v, n=xy.shape[0], g=self.gaussians_per_pixel, c=self.gau_out_single)
            else:
                features_expanded = raw_gaussians
                features_expanded = rearrange(features_expanded, "b v n c -> b (v n) c", b=bs, v=v, n=xy.shape[0]).contiguous()
                gaussians_raw = rearrange(raw_gaussians_final, "b v n c -> b (v n) c", b=bs, v=v, n=xy.shape[0], c=self.gau_out_single)

            gaussians_stage = {
                "gaussians": gaussians_final,
                "features": features_expanded,
                "gaussians_raw": gaussians_raw,
            }
            gaussians_all["stages"].append(gaussians_stage)
        
        # gaussians_all.update(gaussians_stage)
        gaussians_all['gaussians'] = torch.cat([g["gaussians"] for g in gaussians_all["stages"]], dim=1)
        gaussians_all['features'] = torch.cat([g["features"] for g in gaussians_all["stages"]], dim=1)
        
        gaussians_all['gaussians_raw'] = torch.cat([g["gaussians_raw"] for g in gaussians_all["stages"]], dim=1)
        return gaussians_all
    # <<< END method


class LegacyPixelGaussian360Loc:
    """Holder of the legacy PixelGaussian360Loc.forward."""

    # >>> BEGIN method model/pixel/pixel_gs_360loc.py:327-587
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
    # <<< END method


class LegacyPixelGaussian512:
    """Holder of the legacy PixelGaussian512.forward."""

    # >>> BEGIN method model/pixel/pixel_gs_512.py:186-301
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
    # <<< END method


class LegacyGaussianRenderer:
    """Holder of the legacy GaussianRenderer.render (panorama path)."""

    # >>> BEGIN method model/gaussian.py:210-344
    def render(
        self, 
        gaussians: Float[Tensor, "B N F"], 
        c2w: Float[Tensor, "B V 4 4"],
        fovx: Float[Tensor, "B V"] = None,
        fovy: Float[Tensor, "B V"] = None,
        rays_o: Float[Tensor, "B V H W 3"] = None,
        rays_d: Float[Tensor, "B V H W 3"] = None,
        bg_color: Float[Tensor, "... 3"] = None, 
        scale_modifier: float = 1.,
    ):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        # at least one of fovx and fovy is not none
        assert fovx is not None or fovy is not None
        if fovx is None:
            fovx = fovy
        if fovy is None:
            fovy = fovx

        device = gaussians.device

        if self.renderer_type == "vanilla":
            c2b = torch.inverse(self.extrinsics).to(c2w.device)
            c2w = c2w[:, :, None, :, :] @ c2b[None, None, :, :, :] # B V 6 4 4
            c2w = c2w.reshape(c2w.shape[0], -1, 4, 4)
            fovx = fovx.repeat(1,6)
            fovy = fovy.repeat(1,6)
        B, V = c2w.shape[:2]

        # loop of loop...
        images = []
        alphas = []
        depths = []
        for b in range(B):

            means3D = gaussians[b, :, 0:3].contiguous().float()
            rgbs = gaussians[b, :, 3:6].contiguous().float() # [N, 3]
            opacity = gaussians[b, :, 6:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            scales = gaussians[b, :, 11:].contiguous().float()
            means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device=device)

            for v in range(V):
                fovx_ = fovx[b, v].clone()
                fovy_ = fovy[b, v].clone()
                c2w_ = c2w[b, v].clone()
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w_, fovx=fovx_, fovy=fovy_, znear=self.znear, zfar=self.zfar
                )
                # render novel views
                tan_half_fovx = torch.tan(fovx_ * 0.5)
                tan_half_fovy = torch.tan(fovy_ * 0.5)

                if self.renderer_type == "vanilla":
                    raster_settings = GaussianRasterizationSettings(
                        image_height=self.resolution[0],
                        image_width=self.resolution[1],
                        tanfovx=tan_half_fovx,
                        tanfovy=tan_half_fovy,
                        bg=self.bg_color if bg_color is None else bg_color,
                        scale_modifier=scale_modifier,
                        viewmatrix=w2c,
                        projmatrix=proj,
                        sh_degree=0,
                        campos=cam_p,
                        prefiltered=False,
                        debug=False,
                    )
                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                elif self.renderer_type == "panorama":
                    raster_settings = GaussianRasterizationSettings(
                        image_height=self.resolution[0],
                        image_width=self.resolution[1],
                        tanfovx=tan_half_fovx,
                        tanfovy=tan_half_fovy,
                        bg=self.bg_color if bg_color is None else bg_color,
                        scale_modifier=1.0,
                        viewmatrix=w2c,
                        projmatrix=proj,
                        sh_degree=0,
                        campos=cam_p,
                        prefiltered=False,  # This matches the original usage.
                        debug=False,
                    )
                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                else:
                    raise NotImplementedError

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                if self.renderer_type == "vanilla":
                    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None,
                    )
                    rendered_normal = None
                elif self.renderer_type == "panorama":
                    rendered_image, feature_map, confidence_map, rendered_alpha, rendered_depth, rendered_radii = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None,
                    )
                else:
                    raise NotImplementedError

                rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)
                images.append(rendered_image)
                alphas.append(rendered_alpha)
                depths.append(rendered_depth)
        if self.renderer_type == "panorama":
            images = torch.stack(images, dim=0).view(B, V, 3, self.resolution[0], self.resolution[1])
            alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])
            depths = torch.stack(depths, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])
        else:
            images = self.C2E(torch.stack(images, dim=0)).view(B, -1, 3, self.resolution[0] * 2, self.resolution[1] * 4)
            alphas = self.C2E(torch.stack(alphas, dim=0)).view(B, -1, 1, self.resolution[0] * 2, self.resolution[1] * 4)
            depths = self.C2E(torch.stack(depths, dim=0)).view(B, -1, 1, self.resolution[0] * 2, self.resolution[1] * 4)
        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
            "depth": depths
        }
    # <<< END method
