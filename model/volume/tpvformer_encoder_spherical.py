import numpy as np
import torch
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmengine.registry import MODELS
from torch import nn
from torch.nn.init import normal_

from .cross_view_hybrid_attention import TPVCrossViewHybridAttention
from .image_cross_attention import TPVMSDeformableAttention3D
from einops import rearrange
from vis_feat import single_features_to_RGB, show_vis_points

@MODELS.register_module()
class TPVFormerEncoderSpherical(TransformerLayerSequence):

    def __init__(self,
                 tpv_theta=200,
                 tpv_phi=200,
                 tpv_r=16,
                 tpv_only=False,
                 pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                 num_feature_levels=4,
                 num_cams=6,
                 embed_dims=256,
                 num_points_in_pillar=[4, 32, 32],
                 num_points_in_pillar_cross_view=[32, 32, 32],
                 num_layers=5,
                 transformerlayers=None,
                 positional_encoding=None,
                 return_intermediate=False):
        super().__init__(transformerlayers, num_layers)

        self.tpv_theta = tpv_theta
        self.tpv_phi = tpv_phi
        self.tpv_r = tpv_r
        self.pc_range = pc_range
        self.real_theta = pc_range[3] - pc_range[0]
        self.real_phi = pc_range[4] - pc_range[1]
        self.real_r = pc_range[5] - pc_range[2]

        self.level_embeds = nn.Parameter(
            torch.Tensor(num_feature_levels, embed_dims))
        self.tpv_embedding_thetaphi = nn.Embedding(tpv_theta * tpv_phi, embed_dims)
        self.tpv_embedding_rtheta = nn.Embedding(tpv_r * tpv_theta, embed_dims)
        self.tpv_embedding_phir = nn.Embedding(tpv_phi * tpv_r, embed_dims)
        if not tpv_only:
            self.project_transform_thetaphi = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1)
            self.project_transform_rtheta = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1)
            self.project_transform_phir = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1)
        ref_3d_thetaphi = self.get_reference_points(tpv_theta, tpv_phi, self.real_r,
                                              num_points_in_pillar[0])
        ref_3d_rtheta = self.get_reference_points(tpv_r, tpv_theta, self.real_phi,
                                              num_points_in_pillar[1])
        ref_3d_rtheta = ref_3d_rtheta.permute(3, 0, 1, 2)[[1, 2, 0]]  # change to theta,phi,r
        ref_3d_rtheta = ref_3d_rtheta.permute(1, 2, 3, 0)
        ref_3d_phir = self.get_reference_points(tpv_phi, tpv_r, self.real_theta,
                                              num_points_in_pillar[2])
        ref_3d_phir = ref_3d_phir.permute(3, 0, 1, 2)[[2, 0, 1]]  # change totheta,phi,r
        ref_3d_phir = ref_3d_phir.permute(1, 2, 3, 0)
        self.register_buffer('ref_3d_thetaphi', ref_3d_thetaphi)
        self.register_buffer('ref_3d_rtheta', ref_3d_rtheta)
        self.register_buffer('ref_3d_phir', ref_3d_phir)

        cross_view_ref_points = self.get_cross_view_ref_points(
            tpv_theta, tpv_phi, tpv_r, num_points_in_pillar_cross_view)
        self.register_buffer('cross_view_ref_points', cross_view_ref_points)

        # positional encoding
        self.positional_encoding = MODELS.build(positional_encoding)
        self.return_intermediate = return_intermediate
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, TPVMSDeformableAttention3D) or isinstance(
                    m, TPVCrossViewHybridAttention):
                m.init_weights()
        normal_(self.level_embeds)

    @staticmethod
    def get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar):
        # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
        # generate points for hw and level 1
        h_ranges = torch.linspace(0.5, tpv_h - 0.5, tpv_h) / tpv_h
        w_ranges = torch.linspace(0.5, tpv_w - 0.5, tpv_w) / tpv_w
        h_ranges = h_ranges.unsqueeze(-1).expand(-1, tpv_w).flatten()
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_h, -1).flatten()
        hw_hw = torch.stack([w_ranges, h_ranges], dim=-1)  # hw, 2
        hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2],
                                          -1)  # hw, #p, 2
        # generate points for hw and level 2
        z_ranges = torch.linspace(0.5, tpv_z - 0.5,
                                  num_points_in_pillar[2]) / tpv_z  # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h * tpv_w, -1)  # hw, #p
        h_ranges = torch.linspace(0.5, tpv_h - 0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_w, num_points_in_pillar[2]).flatten(0, 1)
        hw_zh = torch.stack([h_ranges, z_ranges], dim=-1)  # hw, #p, 2
        # generate points for hw and level 3
        z_ranges = torch.linspace(0.5, tpv_z - 0.5,
                                  num_points_in_pillar[2]) / tpv_z  # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h * tpv_w, -1)  # hw, #p
        w_ranges = torch.linspace(0.5, tpv_w - 0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(1, -1, 1).expand(
            tpv_h, -1, num_points_in_pillar[2]).flatten(0, 1)
        hw_wz = torch.stack([z_ranges, w_ranges], dim=-1)  # hw, #p, 2

        # generate points for zh and level 1
        w_ranges = torch.linspace(0.5, tpv_w - 0.5,
                                  num_points_in_pillar[1]) / tpv_w
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z * tpv_h, -1)
        h_ranges = torch.linspace(0.5, tpv_h - 0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(
            tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for zh and level 2
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        h_ranges = torch.linspace(0.5, tpv_h - 0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(
            tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_zh = torch.stack([h_ranges, z_ranges], dim=-1)  # zh, #p, 2
        # generate points for zh and level 3
        w_ranges = torch.linspace(0.5, tpv_w - 0.5,
                                  num_points_in_pillar[1]) / tpv_w
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z * tpv_h, -1)
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        # generate points for wz and level 1
        h_ranges = torch.linspace(0.5, tpv_h - 0.5,
                                  num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w * tpv_z, -1)
        w_ranges = torch.linspace(0.5, tpv_w - 0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for wz and level 2
        h_ranges = torch.linspace(0.5, tpv_h - 0.5,
                                  num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w * tpv_z, -1)
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(
            tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)
        # generate points for wz and level 3
        w_ranges = torch.linspace(0.5, tpv_w - 0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(
            tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        reference_points = torch.cat([
            torch.stack([hw_hw, hw_zh, hw_wz], dim=1),
            torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
            torch.stack([wz_hw, wz_zh, wz_wz], dim=1)
        ],
                                     dim=0)  # hw+zh+wz, 3, #p, 2

        return reference_points

    @staticmethod
    def get_reference_points(T,
                             P,
                             R=8,
                             num_points_in_pillar=4,
                             dim='3d',
                             bs=1,
                             device='cuda',
                             dtype=torch.float):
        """Get the reference points used in SCA and TSA.

        Args:
            H, W: spatial shape of tpv.
            Z: height of pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        thetas = torch.linspace(
            0.5, T - 0.5, T, dtype=dtype, device=device).view(1, -1, 1).expand(
                num_points_in_pillar, T, P) / T
        phis = torch.linspace(
            0.5, P - 0.5, P, dtype=dtype, device=device).view(1, 1, -1).expand(
                num_points_in_pillar, T, P) / P
        rs = torch.linspace(
            0.5, R - 0.5, num_points_in_pillar,
            dtype=dtype, device=device).view(-1, 1, 1).expand(
                num_points_in_pillar, T, P) / R       
        ref_3d = torch.stack((thetas, phis, rs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1) # [num_points_in_pillar, T*P, 3]
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    def point_sampling(self, reference_points, pc_range, img_metas):
        h, w = list(img_metas[0]["img_shape"])[0]
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"].cpu())
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B_ref, num_query = reference_points.size()[:3]
        B = len(img_metas)
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B_ref, 1, num_query, 4).repeat(
            1, B // B_ref, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4,
                                   4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32),
            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        tpv_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= w
        reference_points_cam[..., 1] /= h

        tpv_mask = (
            tpv_mask & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0))

        tpv_mask = torch.nan_to_num(tpv_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        tpv_mask = tpv_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, tpv_mask

    def pano_point_sampling(self, reference_points, pc_range, img_metas, img_ori):
        B = len(img_metas)
        # init reference_points
        # ori_reference_points = reference_points.clone().permute(0,2,1,3).repeat(B, 1, 1, 1).unsqueeze(0)
        # ori_reference_points_cam = ori_reference_points[..., :2]
        
        eps = 1e-5
        tri_reference_points = reference_points.clone()
        spherical_reference_points = torch.ones_like(tri_reference_points, device=tri_reference_points.device)
        spherical_theta = tri_reference_points[..., 0:1]*2*torch.pi
        spherical_phi = tri_reference_points[..., 1:2]*torch.pi
        spherical_r = tri_reference_points[..., 2:3]
        # vis_sample_points(spherical_phi, spherical_theta)
        spherical_reference_points[..., 0:1] = -spherical_r * torch.sin(spherical_phi) * torch.sin(spherical_theta)
        spherical_reference_points[..., 1:2] = -spherical_r * torch.cos(spherical_phi)
        spherical_reference_points[..., 2:3] = -spherical_r * torch.sin(spherical_phi) * torch.cos(spherical_theta)
        B_ref, D, num_query, _ = spherical_reference_points.shape
        
        # init lidar2img
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = torch.stack(lidar2img)
        lidar2img = lidar2img[:,:,None,None,:,:].repeat(1,1,D,num_query,1,1)
        num_cams = lidar2img.shape[1]

        # get reference_points
        spherical_reference_points = spherical_reference_points.unsqueeze(0).repeat(B // B_ref, num_cams, 1, 1, 1)
        ones = torch.ones_like(spherical_reference_points[..., :1], device=spherical_reference_points.device, dtype=reference_points.dtype)
        reference_points_homogeneous = torch.cat((spherical_reference_points, ones), dim=-1)
        P_cam_homogeneous = torch.matmul(lidar2img, reference_points_homogeneous.unsqueeze(-1))
        P_cam_homogeneous = P_cam_homogeneous.squeeze(-1)
        w_prime = P_cam_homogeneous[..., 3:]
        spherical_reference_points = P_cam_homogeneous[..., :3] / (w_prime + eps)

        x = spherical_reference_points[...,0:1]
        y = spherical_reference_points[...,1:2]
        z = spherical_reference_points[...,2:3]
        theta = (torch.atan2(x, z + eps) + torch.pi)/(2 * torch.pi)
        phi = (torch.atan2(y, torch.sqrt(x**2 + z**2 + eps)) + torch.pi/2)/torch.pi
        reference_points_cam = torch.cat((theta, phi), dim=-1).permute(1,0,3,2,4)

        # if reference_points_cam.shape[3] == 8:
        #     for i in range(8):
        #         gap = reference_points_cam.shape[3] / 8
        #         sample_idx = int(i * gap)
        #         show_vis_points(reference_points_cam, sample_idx, background_image=img_ori[0,0], output_filename=f'vis/spherical/feat_thetaphi{sample_idx}')

        tpv_mask = (
            (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0))

        tpv_mask = torch.nan_to_num(tpv_mask).squeeze(-1)

        return reference_points_cam, tpv_mask

    def forward(self, mlvl_feats, project_feats, img_metas, img_ori):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        # tpv queries and pos embeds
        tpv_queries_thetaphi = self.tpv_embedding_thetaphi.weight.to(dtype)
        tpv_queries_rtheta = self.tpv_embedding_rtheta.weight.to(dtype)
        tpv_queries_phir = self.tpv_embedding_phir.weight.to(dtype)
        tpv_queries_thetaphi = tpv_queries_thetaphi.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_rtheta = tpv_queries_rtheta.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_phir = tpv_queries_phir.unsqueeze(0).repeat(bs, 1, 1)
        # add projected feats to tpv queries
        if project_feats[0] is not None and project_feats[1] is not None and project_feats[2] is not None:
            project_feats_thetaphi, project_feats_rtheta, project_feats_phir = project_feats
            project_feats_thetaphi = rearrange(self.project_transform_thetaphi(project_feats_thetaphi), "b c h w -> b (h w) c")
            project_feats_rtheta = rearrange(self.project_transform_rtheta(project_feats_rtheta), "b c z h -> b (z h) c")
            project_feats_phir = rearrange(self.project_transform_phir(project_feats_phir), "b c w z -> b (w z) c")
            tpv_queries_thetaphi = tpv_queries_thetaphi + project_feats_thetaphi
            tpv_queries_rtheta = tpv_queries_rtheta + project_feats_rtheta
            tpv_queries_phir = tpv_queries_phir + project_feats_phir

        tpv_query = [tpv_queries_thetaphi, tpv_queries_rtheta, tpv_queries_phir]

        tpv_pos_thetaphi = self.positional_encoding(bs, device, 'z')
        tpv_pos_rtheta = self.positional_encoding(bs, device, 'w')
        tpv_pos_phir = self.positional_encoding(bs, device, 'h')
        tpv_pos = [tpv_pos_thetaphi, tpv_pos_rtheta, tpv_pos_phir]

        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # num_cam, bs, hw, c
            feat = feat + self.level_embeds[None, None,
                                            lvl:lvl + 1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)  # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        reference_points_cams, tpv_masks = [], []
        ref_3ds = [self.ref_3d_thetaphi, self.ref_3d_rtheta, self.ref_3d_phir]
        for ref_3d in ref_3ds:
            reference_points_cam, tpv_mask = self.pano_point_sampling(
                ref_3d, self.pc_range,
                img_metas, img_ori)  # num_cam, bs, hw++, #p, 2
            reference_points_cams.append(reference_points_cam)
            tpv_masks.append(tpv_mask)

        ref_cross_view = self.cross_view_ref_points.clone().unsqueeze(
            0).expand(bs, -1, -1, -1, -1)

        intermediate = []
        for layer in self.layers:
            output = layer(
                tpv_query,
                feat_flatten,
                feat_flatten,
                tpv_pos=tpv_pos,
                ref_2d=ref_cross_view,
                tpv_h=self.tpv_theta,
                tpv_w=self.tpv_phi,
                tpv_z=self.tpv_r,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                tpv_masks=tpv_masks)
            tpv_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output