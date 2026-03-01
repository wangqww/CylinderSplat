import os
import numpy as np
import torch
import imageio
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import warnings
from einops import rearrange
from vis_feat import single_features_to_RGB
import time # 用于计时比较 (可选)
import matplotlib.pyplot as plt

opacity_threshold = 0.05
distance_threshold = 0.3

# --- 检查并导入 PyTorch3D ---
try:
    from pytorch3d.ops import knn_points
    pytorch3d_available = True
    print("PyTorch3D found.")
except ImportError:
    pytorch3d_available = False
    print("Warning: PyTorch3D not found. This implementation requires PyTorch3D.")
    print("Please install it following the instructions at:")
    print("https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")


def calculate_confidence_nn(points_gt, points_pred, scale=1.0):
    """
    为 points_pred 中的每个点计算置信度，基于其到 points_gt 的最近邻距离。

    Args:
        points_gt (torch.Tensor): Ground truth point cloud, shape [N, 3] or [B, N, 3].
        points_pred (torch.Tensor): Predicted point cloud, shape [M, 3] or [B, M, 3].
        scale (float): 控制指数衰减的尺度参数 k in exp(-k * dist^2).
                       值越大，衰减越快，对距离越敏感。

    Returns:
        torch.Tensor: 置信度分数，形状与 points_pred 的前缀匹配 (e.g., [M] or [B, M]).
                      范围 [0, 1].
    """
    # 确保输入是 3D 张量 (B, N, D) 或 (B, M, D)
    was_2d_gt = points_gt.dim() == 2
    was_2d_pred = points_pred.dim() == 2
    if was_2d_gt:
        points_gt = points_gt.unsqueeze(0) # Add batch dimension
    if was_2d_pred:
        points_pred = points_pred.unsqueeze(0) # Add batch dimension

    if points_gt.shape[0] != points_pred.shape[0]:
        # 如果批次大小不匹配，但其中一个是 1，尝试广播
        if points_gt.shape[0] == 1:
            points_gt = points_gt.expand(points_pred.shape[0], -1, -1)
        elif points_pred.shape[0] == 1:
            # 这个函数是为 pred 的每个点算置信度，通常 pred 会有批次
            # GT 可能是固定的，所以 GT 批次为 1 更常见
             raise ValueError("Batch size mismatch and pred batch size is 1, cannot easily broadcast GT.")
        else:
             raise ValueError(f"Batch size mismatch: GT {points_gt.shape[0]}, Pred {points_pred.shape[0]}")


    # --- 1. 计算 B 中每个点到 A 的最近邻 ---
    # knn_points(query, points, K) finds K nearest neighbors in points for each query point.
    # We want nearest neighbor in A (points_gt) for each point in B (points_pred).
    # Returns:
    #   dists: Squared distances, shape [B, M, K]
    #   idx: Indices of neighbors in points_gt, shape [B, M, K]
    #   nn: Neighbor points coordinates (optional), shape [B, M, K, 3]
    # start_time = time.time()
    knn_dists_sq, knn_idx, _ = knn_points(points_pred, points_gt, K=1, return_nn=False)
    # knn_dists_sq shape is [B, M, 1]
    # end_time = time.time()
    # print(f"KNN calculation took: {end_time - start_time:.4f} seconds")

    # 取 K=1 的结果，并移除 K 维度
    nearest_dist_sq = knn_dists_sq[:, :, :1] # Shape: [B, M, 1]

    # --- 2. 将平方距离转换为置信度 ---
    # 使用指数衰减: conf = exp(-scale * distance^2)
    confidences = torch.exp(-scale * nearest_dist_sq) # Shape: [B, M, 1]

    # 如果原始输入是 2D，恢复输出为 1D
    if was_2d_pred: # Check based on the input we are calculating confidence for
        confidences = confidences.squeeze(0)

    return confidences

def vis_dist_conf(dist_conf):
    confidence = dist_conf[:,0].clone()
    # --- 可选：可视化置信度分布 ---
    plt.figure(figsize=(8, 4))
    plt.hist(confidence.detach().cpu().numpy(), bins=50, density=True)
    plt.xlabel("Confidence Score")
    plt.ylabel("Density")
    plt.title("Histogram of Calculated Confidences")
    plt.grid(True)
    plt.savefig('conf.png')

@MODELS.register_module()
class VolumeGaussianConf(BaseModule):

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

        self.tpv_h = self.encoder.tpv_h  #y
        self.tpv_w = self.encoder.tpv_w  #x
        self.tpv_z = self.encoder.tpv_z  #z
        self.pc_range = self.encoder.pc_range
        self.pc_xrange = self.pc_range[3] - self.pc_range[0]
        self.pc_yrange = self.pc_range[4] - self.pc_range[1]
        self.pc_zrange = self.pc_range[5] - self.pc_range[2]

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img_feats, candidate_gaussians, candidate_feats, candidate_depth_pred, img_metas=None, status="train"):
        """Forward training function.
        """
        if candidate_gaussians is not None and candidate_feats is not None:
            bs = len(candidate_feats)
            _, c = candidate_feats[0].shape
            project_feats_hw = candidate_feats[0].new_zeros((bs, self.tpv_h, self.tpv_w, c))
            project_feats_zh = candidate_feats[0].new_zeros((bs, self.tpv_z, self.tpv_h, c))
            project_feats_wz = candidate_feats[0].new_zeros((bs, self.tpv_w, self.tpv_z, c))

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
                candidate_hs_i = (self.tpv_h * (torch.atan2(y, torch.sqrt(x**2 + z**2 + eps))  + torch.pi/2) / torch.pi - eps).int()
                candidate_ws_i = (self.tpv_w * (torch.atan2(x, z) + torch.pi)/(2 * torch.pi) - eps).int()
                candidate_zs_i = (self.tpv_z * candidate_depth_pred[i][:, 0] / self.pc_zrange - eps).int()
                
                # original
                # candidate_hs_i = candidate_uv_map[i][:, 1]
                # candidate_ws_i = candidate_uv_map[i][:, 0]
                # candidate_zs_i = (self.tpv_z * candidate_depth_pred[i][:, 0] / self.pc_zrange - 0.5).int()
                # n, c
                #candidate_feats_i = candidate_feats[[i, valid_mask]]
                candidate_feats_i = candidate_feats[i]
                # hw: n, 2
                candidate_coords_hw_i = torch.stack([candidate_hs_i, candidate_ws_i], dim=-1)
                linear_inds_hw_i = (candidate_coords_hw_i[..., 0] * self.tpv_w + candidate_coords_hw_i[..., 1]).to(dtype=torch.int64)
                project_feats_hw_i = project_feats_hw[i].view(-1, c)
                project_feats_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_hw_i = project_feats_hw_i.new_zeros((self.tpv_h * self.tpv_w, c), dtype=torch.float32)
                ones_hw_i = torch.ones_like(candidate_feats_i)
                count_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), ones_hw_i)
                count_hw_i = torch.where(count_hw_i == 0, torch.ones_like(count_hw_i), count_hw_i)
                project_feats_hw_i = (project_feats_hw_i / count_hw_i).view(self.tpv_h, self.tpv_w, c)
                project_feats_hw[i] = project_feats_hw_i

                # zh: n, 2
                candidate_coords_zh_i = torch.stack([candidate_zs_i, candidate_hs_i], dim=-1)
                linear_inds_zh_i = (candidate_coords_zh_i[..., 0] * self.tpv_h + candidate_coords_zh_i[..., 1]).to(dtype=torch.int64)
                project_feats_zh_i = project_feats_zh[i].view(-1, c)
                project_feats_zh_i.scatter_add_(0, linear_inds_zh_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_zh_i = project_feats_zh_i.new_zeros((self.tpv_z * self.tpv_h, c), dtype=torch.float32)
                ones_zh_i = torch.ones_like(candidate_feats_i)
                count_zh_i.scatter_add_(0, linear_inds_zh_i.unsqueeze(-1).expand(-1, c), ones_zh_i)
                count_zh_i = torch.where(count_zh_i == 0, torch.ones_like(count_zh_i), count_zh_i)
                project_feats_zh_i = (project_feats_zh_i / count_zh_i).view(self.tpv_z, self.tpv_h, c)
                project_feats_zh[i] = project_feats_zh_i

                # wz: n, 2
                candidate_coords_wz_i = torch.stack([candidate_ws_i, candidate_zs_i], dim=-1)
                linear_inds_wz_i = (candidate_coords_wz_i[..., 0] * self.tpv_z + candidate_coords_wz_i[..., 1]).to(dtype=torch.int64)
                project_feats_wz_i = project_feats_wz[i].view(-1, c)
                project_feats_wz_i.scatter_add_(0, linear_inds_wz_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_wz_i = project_feats_wz_i.new_zeros((self.tpv_w * self.tpv_z, c), dtype=torch.float32)
                ones_wz_i = torch.ones_like(candidate_feats_i)
                count_wz_i.scatter_add_(0, linear_inds_wz_i.unsqueeze(-1).expand(-1, c), ones_wz_i)
                count_wz_i = torch.where(count_wz_i == 0, torch.ones_like(count_wz_i), count_wz_i)
                project_feats_wz_i = (project_feats_wz_i / count_wz_i).view(self.tpv_w, self.tpv_z, c)
                project_feats_wz[i] = project_feats_wz_i
            
            project_feats_hw = rearrange(project_feats_hw, "b h w c -> b c h w")
            project_feats_zh = rearrange(project_feats_zh, "b h w c -> b c h w")
            project_feats_wz = rearrange(project_feats_wz, "b h w c -> b c h w")
            project_feats = [project_feats_hw, project_feats_zh, project_feats_wz]
        else:
            project_feats = [None, None, None]

        # single_features_to_RGB(project_feats_hw, img_name='feat_hw.png')
        # single_features_to_RGB(project_feats_zh, img_name='feat_zh.png')
        # single_features_to_RGB(project_feats_wz, img_name='feat_wz.png')

        if self.use_checkpoint and status != "test":
            input_vars_enc = (img_feats, project_feats, img_metas)
            outs = torch.utils.checkpoint.checkpoint(
                self.encoder, *input_vars_enc, use_reentrant=False
            )
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, outs, use_reentrant=False)
        else:
            outs = self.encoder(img_feats, project_feats, img_metas)
            gaussians = self.gs_decoder(outs)
        bs = gaussians.shape[0]
        n_feature = gaussians.shape[-1]
        gaussians = gaussians.reshape(bs, -1, n_feature)
        dist_confs = []
        for b in range(bs):
            volume_xyzs = gaussians[b][..., :3]
            pixel_xyz = candidate_gaussians[b][..., :3]
            dist_conf = calculate_confidence_nn(pixel_xyz, volume_xyzs, scale=0.1)
            # vis_dist_conf(dist_conf)
            dist_confs.append(dist_conf)
        dist_confs = torch.stack(dist_confs, dim=0)
        dist_confs = 1.0 - dist_confs
        opacities = gaussians[:, :, 6:7]
        # confidences = torch.ones_like(opacities, device=opacities.device)
        # # mask_opa = ranking_feature > opacity_threshold
        # mask_dist = (dist_confs > 0.1)
        # filtered_gaussians = gaussians * mask_dist.float()

        # _, top_indices = torch.topk(ranking_feature, k=K, dim=1) # dim=1 是高斯球数量的维度
        # indices_for_gather = top_indices.unsqueeze(-1).expand(-1, -1, 14) # Shape: [6, 10000, 14]
        # filtered_gaussians = torch.gather(gaussians, dim=1, index=indices_for_gather) # Shape: [6, 10000, 14]

        # filtered_gaussians = gaussians
        return gaussians, opacities, dist_confs

        # return gaussians