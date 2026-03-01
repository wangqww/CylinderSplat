import os
import numpy as np
import torch
import imageio
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import warnings
from einops import rearrange
from vis_feat import single_features_to_RGB, visualize_counts_as_heatmap, visualize_counts_as_polar_heatmap, features_to_blocky_heatmap

@MODELS.register_module()
class VolumeGaussianCylinder(BaseModule):

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

        self.tpv_theta = self.encoder.tpv_theta  #theta
        self.tpv_r = self.encoder.tpv_r  #r
        self.tpv_z = self.encoder.tpv_z  #z
        self.pc_range = self.encoder.pc_range
        self.pc_rrange = self.pc_range[3] - self.pc_range[0]
        self.pc_thetarange = self.pc_range[4] - self.pc_range[1]
        self.pc_zrange = self.pc_range[5] - self.pc_range[2]

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img_feats, candidate_gaussians, candidate_feats, img_color, img_depth, img_metas, status="train"):
        """Forward training function.
        """
        if candidate_gaussians is not None and candidate_feats is not None:
            bs = len(candidate_feats)
            _, c = candidate_feats[0].shape
            project_feats_thetar = candidate_feats[0].new_zeros((bs, self.tpv_theta, self.tpv_r, c))
            project_feats_ztheta = candidate_feats[0].new_zeros((bs, self.tpv_z, self.tpv_theta, c))
            project_feats_rz = candidate_feats[0].new_zeros((bs, self.tpv_r, self.tpv_z, c))

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
                candidate_rs_i = (self.tpv_r * (torch.sqrt(x**2 + z**2) - self.pc_range[0]) / self.pc_rrange - eps).int() ## r
                candidate_thetas_i = (self.tpv_theta * (torch.atan2(x, z + eps) + torch.pi)/(2 * torch.pi) - eps).int() ## theta
                candidate_zs_i = (self.tpv_z * (y - self.pc_range[2]) / self.pc_zrange - eps).int() ## z
                
                candidate_feats_i = candidate_feats[i]
                # thetar: n, 2
                candidate_coords_thetar_i = torch.stack([candidate_thetas_i, candidate_rs_i], dim=-1)
                linear_inds_thetar_i = (candidate_coords_thetar_i[..., 0] * self.tpv_r + candidate_coords_thetar_i[..., 1]).to(dtype=torch.int64)
                project_feats_thetar_i = project_feats_thetar[i].view(-1, c)
                project_feats_thetar_i.scatter_add_(0, linear_inds_thetar_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_thetar_i = project_feats_thetar_i.new_zeros((self.tpv_theta * self.tpv_r, c), dtype=torch.float32)
                ones_thetar_i = torch.ones_like(candidate_feats_i)
                count_thetar_i.scatter_add_(0, linear_inds_thetar_i.unsqueeze(-1).expand(-1, c), ones_thetar_i)
                count_thetar_i = torch.where(count_thetar_i == 0, torch.ones_like(count_thetar_i), count_thetar_i)
                project_feats_thetar_i = (project_feats_thetar_i / count_thetar_i).view(self.tpv_theta, self.tpv_r, c)
                project_feats_thetar[i] = project_feats_thetar_i

                # ztheta: n, 2
                candidate_coords_ztheta_i = torch.stack([candidate_zs_i, candidate_thetas_i], dim=-1)
                linear_inds_ztheta_i = (candidate_coords_ztheta_i[..., 0] * self.tpv_theta + candidate_coords_ztheta_i[..., 1]).to(dtype=torch.int64)
                project_feats_ztheta_i = project_feats_ztheta[i].view(-1, c)
                project_feats_ztheta_i.scatter_add_(0, linear_inds_ztheta_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_ztheta_i = project_feats_ztheta_i.new_zeros((self.tpv_z * self.tpv_theta, c), dtype=torch.float32)
                ones_ztheta_i = torch.ones_like(candidate_feats_i)
                count_ztheta_i.scatter_add_(0, linear_inds_ztheta_i.unsqueeze(-1).expand(-1, c), ones_ztheta_i)
                count_ztheta_i = torch.where(count_ztheta_i == 0, torch.ones_like(count_ztheta_i), count_ztheta_i)
                project_feats_ztheta_i = (project_feats_ztheta_i / count_ztheta_i).view(self.tpv_z, self.tpv_theta, c)
                project_feats_ztheta[i] = project_feats_ztheta_i

                # rz: n, 2
                candidate_coords_rz_i = torch.stack([candidate_rs_i, candidate_zs_i], dim=-1)
                linear_inds_rz_i = (candidate_coords_rz_i[..., 0] * self.tpv_z + candidate_coords_rz_i[..., 1]).to(dtype=torch.int64)
                project_feats_rz_i = project_feats_rz[i].view(-1, c)
                project_feats_rz_i.scatter_add_(0, linear_inds_rz_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_rz_i = project_feats_rz_i.new_zeros((self.tpv_r * self.tpv_z, c), dtype=torch.float32)
                ones_rz_i = torch.ones_like(candidate_feats_i)
                count_rz_i.scatter_add_(0, linear_inds_rz_i.unsqueeze(-1).expand(-1, c), ones_rz_i)
                count_rz_i = torch.where(count_rz_i == 0, torch.ones_like(count_rz_i), count_rz_i)
                project_feats_rz_i = (project_feats_rz_i / count_rz_i).view(self.tpv_r, self.tpv_z, c)
                project_feats_rz[i] = project_feats_rz_i
            
            project_feats_thetar = rearrange(project_feats_thetar, "b h w c -> b c h w")
            project_feats_ztheta = rearrange(project_feats_ztheta, "b h w c -> b c h w")
            project_feats_rz = rearrange(project_feats_rz, "b h w c -> b c h w")
            project_feats = [project_feats_thetar, project_feats_ztheta, project_feats_rz]
        else:
            project_feats = [None, None, None]

        # features_to_blocky_heatmap(project_feats_thetar, img_name='feat_thetar.png', cmap_name='Greens', final_w=128, idx=1)
        # features_to_blocky_heatmap(project_feats_ztheta, img_name='feat_ztheta.png', cmap_name='Blues', final_w=512, idx=1)
        # features_to_blocky_heatmap(project_feats_rz, img_name='feat_rz.png', cmap_name='Reds', final_w=512, idx=1)

        # # vis count_rtheta
        # linear_inds_rtheta_i = (candidate_coords_thetar_i[..., 1] * self.tpv_theta + candidate_coords_thetar_i[..., 0]).to(dtype=torch.int64)
        # count_rtheta_i = project_feats_thetar_i.new_zeros((self.tpv_theta * self.tpv_r, c), dtype=torch.float32)
        # ones_rtheta_i = torch.ones_like(candidate_feats_i)
        # count_rtheta_i.scatter_add_(0, linear_inds_rtheta_i.unsqueeze(-1).expand(-1, c), ones_rtheta_i)
        # count_rtheta_i = torch.where(count_rtheta_i == 0, torch.ones_like(count_rtheta_i), count_rtheta_i)

        # visualize_counts_as_polar_heatmap(count_rtheta_i,
        #                             self.tpv_r, 
        #                             self.tpv_theta, 
        #                             'count_rtheta.png', 
        #                             cmap_name='Blues'
        #                             )
        
        # # vis count_thetaz
        # linear_inds_thetaz_i = (candidate_coords_ztheta_i[..., 1] * self.tpv_z + candidate_coords_ztheta_i[..., 0]).to(dtype=torch.int64)
        # count_thetaz_i = project_feats_ztheta_i.new_zeros((self.tpv_z * self.tpv_theta, c), dtype=torch.float32)
        # ones_thetaz_i = torch.ones_like(candidate_feats_i)
        # count_thetaz_i.scatter_add_(0, linear_inds_thetaz_i.unsqueeze(-1).expand(-1, c), ones_thetaz_i)
        # count_thetaz_i = torch.where(count_thetaz_i == 0, torch.ones_like(count_thetaz_i), count_thetaz_i)

        # visualize_counts_as_heatmap(count_thetaz_i,
        #                             self.tpv_theta, 
        #                             self.tpv_z, 
        #                             'count_thetaz.png', 
        #                             cmap_name='Blues'
        #                             )
        
        # # vis count_zr
        # linear_inds_zr_i = (candidate_coords_rz_i[..., 1] * self.tpv_r + candidate_coords_rz_i[..., 0]).to(dtype=torch.int64)
        # count_zr_i = project_feats_rz_i.new_zeros((self.tpv_r * self.tpv_z, c), dtype=torch.float32)
        # ones_rz_i = torch.ones_like(candidate_feats_i)
        # count_zr_i.scatter_add_(0, linear_inds_zr_i.unsqueeze(-1).expand(-1, c), ones_rz_i)
        # count_zr_i = torch.where(count_rz_i == 0, torch.ones_like(count_zr_i), count_zr_i)
        # visualize_counts_as_heatmap(count_zr_i,
        #                             self.tpv_z, 
        #                             self.tpv_r, 
        #                             'count_zr.png', 
        #                             cmap_name='Blues'
        #                             )

        if self.use_checkpoint and status != "test":
            input_vars_enc = (img_feats, project_feats, img_metas, img_color)
            outs = torch.utils.checkpoint.checkpoint(
                self.encoder, *input_vars_enc, use_reentrant=False
            )
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, 
                                                          outs,
                                                          img_color, 
                                                          img_depth, 
                                                          img_metas, 
                                                          use_reentrant=False,
                                                          )
        else:
            outs = self.encoder(img_feats, project_feats, img_metas, img_color)
            gaussians = self.gs_decoder(outs, 
                                        img_color, 
                                        img_depth, 
                                        img_metas,
                                        )
        bs = gaussians.shape[0]
        n_feature = gaussians.shape[-1]
        gaussians = gaussians.reshape(bs, -1, n_feature)

        return gaussians
