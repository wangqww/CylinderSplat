import os
import numpy as np
import torch
import imageio
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import warnings
from einops import rearrange
from vis_feat import single_features_to_RGB, visualize_counts_as_heatmap, features_to_blocky_heatmap
import math

@MODELS.register_module()
class VolumeGaussianSpherical(BaseModule):

    def __init__(self,
                 encoder=None,
                 gs_decoder=None,
                 use_checkpoint=False,
                 **kwargs,
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint

        if encoder:
            self.encoder = MODELS.build(encoder)
        if gs_decoder:
            self.gs_decoder = MODELS.build(gs_decoder)

        self.tpv_theta = self.encoder.tpv_theta  # theta
        self.tpv_phi = self.encoder.tpv_phi  # phi
        self.tpv_r = self.encoder.tpv_r  # r
        self.pc_range = self.encoder.pc_range
        self.pc_rrange = self.pc_range[5] - self.pc_range[2]
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img_feats, candidate_gaussians, candidate_feats, img_color, img_depth, img_metas, status="train"):
        """Forward training function.
        """
        if candidate_gaussians is not None and candidate_feats is not None:
            bs = len(candidate_feats)
            _, c = candidate_feats[0].shape
            project_feats_thetaphi = candidate_feats[0].new_zeros((bs, self.tpv_theta, self.tpv_phi, c))
            project_feats_rtheta = candidate_feats[0].new_zeros((bs, self.tpv_r, self.tpv_theta, c))
            project_feats_phir = candidate_feats[0].new_zeros((bs, self.tpv_phi, self.tpv_r, c))

            for i in range(bs):
                candidate_xyzs_i = candidate_gaussians[i][..., :3]
                
                # Decare
                # candidate_hs_i = (self.tpv_h * (candidate_xyzs_i[..., 1] - self.pc_range[1]) / self.pc_yrange - 0.5).int()
                # candidate_ws_i = (self.tpv_w * (candidate_xyzs_i[..., 0] - self.pc_range[0]) / self.pc_xrange - 0.5).int()
                # candidate_zs_i = (self.tpv_z * (candidate_xyzs_i[..., 2] - self.pc_range[2]) / self.pc_zrange - 0.5).int()
                
                # Spherical
                eps = 1e-5
                x = candidate_xyzs_i[..., 0]
                y = candidate_xyzs_i[..., 1]
                z = candidate_xyzs_i[..., 2]
                candidate_thetas_i = (self.tpv_theta * (torch.atan2(x, z + eps) + torch.pi)/(2 * torch.pi) - eps).int()
                candidate_phis_i = (self.tpv_phi * (torch.atan2(y, torch.sqrt(x**2 + z**2 + eps))  + torch.pi/2) / torch.pi - eps).int()
                candidate_rs_i = (self.tpv_r * torch.sqrt(x**2 + y**2 + z**2 + eps) / self.pc_rrange - eps).int()
                
                # original
                # candidate_hs_i = candidate_uv_map[i][:, 1]
                # candidate_ws_i = candidate_uv_map[i][:, 0]
                # candidate_zs_i = (self.tpv_z * candidate_depth_pred[i][:, 0] / self.pc_zrange - 0.5).int()
                # n, c
                #candidate_feats_i = candidate_feats[[i, valid_mask]]
                candidate_feats_i = candidate_feats[i]
                # thetar: n, 2
                candidate_coords_thetaphi_i = torch.stack([candidate_thetas_i, candidate_phis_i], dim=-1)
                linear_inds_thetaphi_i = (candidate_coords_thetaphi_i[..., 0] * self.tpv_phi + candidate_coords_thetaphi_i[..., 1]).to(dtype=torch.int64)
                project_feats_thetaphi_i = project_feats_thetaphi[i].view(-1, c)
                project_feats_thetaphi_i.scatter_add_(0, linear_inds_thetaphi_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_thetaphi_i = project_feats_thetaphi_i.new_zeros((self.tpv_theta * self.tpv_phi, c), dtype=torch.float32)
                ones_thetaphi_i = torch.ones_like(candidate_feats_i)
                count_thetaphi_i.scatter_add_(0, linear_inds_thetaphi_i.unsqueeze(-1).expand(-1, c), ones_thetaphi_i)
                count_thetaphi_i = torch.where(count_thetaphi_i == 0, torch.ones_like(count_thetaphi_i), count_thetaphi_i)
                project_feats_thetaphi_i = (project_feats_thetaphi_i / count_thetaphi_i).view(self.tpv_theta, self.tpv_phi, c)
                project_feats_thetaphi[i] = project_feats_thetaphi_i

                # rtheta: n, 2
                candidate_coords_rtheta_i = torch.stack([candidate_rs_i, candidate_thetas_i], dim=-1)
                linear_inds_rtheta_i = (candidate_coords_rtheta_i[..., 0] * self.tpv_theta + candidate_coords_rtheta_i[..., 1]).to(dtype=torch.int64)
                project_feats_rtheta_i = project_feats_rtheta[i].view(-1, c)
                project_feats_rtheta_i.scatter_add_(0, linear_inds_rtheta_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_rtheta_i = project_feats_rtheta_i.new_zeros((self.tpv_r * self.tpv_theta, c), dtype=torch.float32)
                ones_rtheta_i = torch.ones_like(candidate_feats_i)
                count_rtheta_i.scatter_add_(0, linear_inds_rtheta_i.unsqueeze(-1).expand(-1, c), ones_rtheta_i)
                count_rtheta_i = torch.where(count_rtheta_i == 0, torch.ones_like(count_rtheta_i), count_rtheta_i)
                project_feats_rtheta_i = (project_feats_rtheta_i / count_rtheta_i).view(self.tpv_r, self.tpv_theta, c)
                project_feats_rtheta[i] = project_feats_rtheta_i

                # phir: n, 2
                candidate_coords_phir_i = torch.stack([candidate_phis_i, candidate_rs_i], dim=-1)
                linear_inds_phir_i = (candidate_coords_phir_i[..., 0] * self.tpv_r + candidate_coords_phir_i[..., 1]).to(dtype=torch.int64)
                project_feats_phir_i = project_feats_phir[i].view(-1, c)
                project_feats_phir_i.scatter_add_(0, linear_inds_phir_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_phir_i = project_feats_phir_i.new_zeros((self.tpv_phi * self.tpv_r, c), dtype=torch.float32)
                ones_phir_i = torch.ones_like(candidate_feats_i)
                count_phir_i.scatter_add_(0, linear_inds_phir_i.unsqueeze(-1).expand(-1, c), ones_phir_i)
                count_phir_i = torch.where(count_phir_i == 0, torch.ones_like(count_phir_i), count_phir_i)
                project_feats_phir_i = (project_feats_phir_i / count_phir_i).view(self.tpv_phi, self.tpv_r, c)
                project_feats_phir[i] = project_feats_phir_i
            
            project_feats_thetaphi = rearrange(project_feats_thetaphi, "b h w c -> b c h w")
            project_feats_rtheta = rearrange(project_feats_rtheta, "b h w c -> b c h w")
            project_feats_phir = rearrange(project_feats_phir, "b h w c -> b c h w")
            project_feats = [project_feats_thetaphi, project_feats_rtheta, project_feats_phir]
        else:
            project_feats = [None, None, None]

        # TODO: visualize the project feats
        # single_features_to_RGB(project_feats_hw, img_name='feat_hw.png')
        # single_features_to_RGB(project_feats_zh, img_name='feat_zh.png')
        # single_features_to_RGB(project_feats_wz, img_name='feat_wz.png')

        # visualize_counts_as_heatmap(count_hw_i,
        #                             self.tpv_h, 
        #                             self.tpv_w, 
        #                             'count_hw.png', 
        #                             cmap_name='Blues'
        #                             )
        # visualize_counts_as_heatmap(count_zh_i,
        #                             self.tpv_z, 
        #                             self.tpv_h, 
        #                             'count_zh.png', 
        #                             cmap_name='Blues'
        #                             )
        # visualize_counts_as_heatmap(count_wz_i,
        #                             self.tpv_w, 
        #                             self.tpv_z, 
        #                             'count_wz.png', 
        #                             cmap_name='Blues'
        #                             )

        # features_to_blocky_heatmap(project_feats_thetaphi, img_name='feat_thetaphi.png', cmap_name='Blues', final_w=128, idx=1)
        # features_to_blocky_heatmap(project_feats_rtheta, img_name='feat_rtheta.png', cmap_name='Greens', final_w=512, idx=1)
        # features_to_blocky_heatmap(project_feats_phir, img_name='feat_phir.png', cmap_name='Reds', final_w=512, idx=1)

        if self.use_checkpoint and status != "test":
            input_vars_enc = (img_feats, project_feats, img_metas, img_color)
            outs = torch.utils.checkpoint.checkpoint(
                self.encoder, *input_vars_enc, use_reentrant=False
            )
            gaussians = torch.utils.checkpoint.checkpoint(
                self.gs_decoder, 
                outs,
                img_color, 
                img_depth, 
                img_metas, 
                use_reentrant=False,
            )
        else:
            outs = self.encoder(img_feats, project_feats, img_metas, img_color)
            gaussians = self.gs_decoder(
                outs, 
                img_color, 
                img_depth, 
                img_metas,
            )
        bs = gaussians.shape[0]
        n_feature = gaussians.shape[-1]
        gaussians = gaussians.reshape(bs, -1, n_feature)
        return gaussians
