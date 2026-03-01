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
from einops import rearrange, einsum
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor
from .gaussian import GaussianRenderer
from .losses import LPIPS, LossDepthTV
from .utils.image import maybe_resize
from .utils.benchmarker import Benchmarker
from .utils.interpolation import interpolate_extrinsics

from pano2cube import Equirec2Cube, Cube2Equirec
from vis_feat import single_features_to_RGB
import torchvision.transforms as transforms
to_pil_image = transforms.ToPILImage()
import matplotlib.cm as cm
import cv2
import matplotlib.pyplot as plt

def vis_depth(depth_xyz):
    B, N, C = depth_xyz.shape
    depth_values_flat = depth_xyz.squeeze(-1).flatten().cpu().detach().numpy()
    # 选项 1: 可视化所有批次合并后的深度分布
    plt.figure(figsize=(10, 6))
    plt.hist(depth_values_flat, bins=50, color='skyblue', edgecolor='black') # bins 控制柱子的数量
    plt.title(f'Overall Depth Distribution (All Batches, {B*N} points)')
    plt.xlabel('Depth (meters)')
    plt.ylabel('Number of Points')
    plt.xlim(0, 90) # 根据你的实际范围调整
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('depth.png')


def onlyDepth(depth, save_name):
    cmap = cm.Spectral
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().detach().numpy()
    depth = depth.astype(np.uint8)
    
    c_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(save_name, c_depth)

@MODELS.register_module()
class OmniGaussian360Loc(BaseModule):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 pixel_gs=None,
                 near_volume_gs=None,
                 far_volume_gs=None,
                 camera_args=None,
                 loss_args=None,
                 dataset_params=None,
                 use_checkpoint=False,
                 task=None,
                 near_point_cloud_range=None,
                 far_point_cloud_range=None,
                 **kwargs,
                 ):

        super().__init__()

        assert pixel_gs is not None and near_volume_gs is not None and far_volume_gs is not None
        self.use_checkpoint = use_checkpoint
        if backbone:
            self.backbone = MODELS.build(backbone)
        if neck:
            self.neck = MODELS.build(neck)
        self.pixel_gs = MODELS.build(pixel_gs)
        self.near_volume_gs = MODELS.build(near_volume_gs)
        self.far_volume_gs = MODELS.build(far_volume_gs)
        self.task = task
        self.dataset_params = dataset_params
        self.camera_args = camera_args
        self.loss_args = loss_args

        self.near_point_cloud_range = near_point_cloud_range
        self.far_point_cloud_range = far_point_cloud_range

        self.renderer = GaussianRenderer(self.device, **camera_args)

        # Perceptual loss
        if self.loss_args.weight_perceptual > 0:
            # self.perceptual_loss = LPIPS(net="vgg")
            self.perceptual_loss = LPIPS().eval()
        else:
            self.perceptual_loss = None

        # record runtime
        self.benchmarker = Benchmarker()

        self.E2C = Equirec2Cube(equ_h=320, equ_w=640, cube_length=self.camera_args['resolution'][0])
        self.C2E = Cube2Equirec(cube_length=40, equ_h=80)

    def extract_img_feat(self, img, status="train"):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        if self.use_checkpoint and status != "test":
            img_feats = torch.utils.checkpoint.checkpoint(
                            self.backbone, img, use_reentrant=False)
        else:
            img_feats = self.backbone(img)
        img_feats = self.neck(img_feats) # BV, C, H, W
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            # single_features_to_RGB(img_feat)
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def cube_extract_img_feat(self, img, status="train"):
        # TODO: transform panorama to cubemap
        cube_rgb = self.E2C(img.squeeze(1))
        """Extract features of images."""
        B, N, C, H, W = cube_rgb.size()
        cube_rgb = cube_rgb.view(B * N, C, H, W)

        if self.use_checkpoint and status != "test":
            img_feats = torch.utils.checkpoint.checkpoint(
                            self.backbone, cube_rgb, use_reentrant=False)
        else:
            img_feats = self.backbone(cube_rgb)
        img_feats = self.neck(img_feats) # BV, C, H, W
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            # TODO: transform cubemap to panorama
            panorama_feat = self.C2E(img_feat)
            single_features_to_RGB(panorama_feat)
            img_feats_reshaped.append(panorama_feat.unsqueeze(1))
        return img_feats_reshaped

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def plucker_embedder(
        self, 
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker
    
    def get_data(self, batch):

        # ================== batch data process ================== #
        device_id = self.device
        data_dict = {}
        # for img feature extraction
        data_dict["imgs"] = batch["inputs"]["rgb"].to(device_id, dtype=self.dtype)
        # for pixel-gs
        rays_o = batch["inputs_pix"]["rays_o"].to(device_id, dtype=self.dtype)
        rays_d = batch["inputs_pix"]["rays_d"].to(device_id, dtype=self.dtype)
        data_dict["rays_o"] = rays_o
        data_dict["rays_d"] = rays_d
        # TODO Panorama direction
        data_dict["pluckers"] = self.plucker_embedder(rays_o, rays_d)
        data_dict["fxs"] = batch["inputs_pix"]["fx"].to(device_id, dtype=self.dtype)
        data_dict["fys"] = batch["inputs_pix"]["fy"].to(device_id, dtype=self.dtype)
        data_dict["cxs"] = batch["inputs_pix"]["cx"].to(device_id, dtype=self.dtype)
        data_dict["cys"] = batch["inputs_pix"]["cy"].to(device_id, dtype=self.dtype)
        data_dict["c2ws"] = batch["inputs_pix"]["c2w"].to(device_id, dtype=self.dtype)
        data_dict["cks"] = batch["inputs_pix"]["ck"].to(device_id, dtype=self.dtype)
        data_dict["depths"] = batch["inputs_pix"]["depth_m"].to(device_id, dtype=self.dtype)
        data_dict["confs"] = batch["inputs_pix"]["conf_m"].to(device_id, dtype=self.dtype)
        # for volume-gs
        img_metas = []
        bs, v, c, h, w = batch["inputs"]["rgb"].shape
        for w2i in batch["inputs_vol"]["w2i"]:
            img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
        data_dict["img_metas"] = img_metas
        # for render and loss and eval
        data_dict["output_imgs"] = batch["outputs"]["rgb"].to(device_id, dtype=self.dtype)
        data_dict["output_depths"] = batch["outputs"]["depth"].to(device_id, dtype=self.dtype)
        data_dict["output_depths_m"] = batch["outputs"]["depth_m"].to(device_id, dtype=self.dtype)
        data_dict["output_confs_m"] = batch["outputs"]["conf_m"].to(device_id, dtype=self.dtype)
        depth_m = rearrange(batch["outputs"]["depth_m"], "b v c h w -> b v h w c")
        data_dict["output_positions"] = (batch["outputs"]["rays_o"] + batch["outputs"]["rays_d"] * \
                            depth_m).to(device_id, dtype=self.dtype)
        data_dict["output_rays_o"] = batch["outputs"]["rays_o"].to(device_id, dtype=self.dtype)
        data_dict["output_rays_d"] = batch["outputs"]["rays_d"].to(device_id, dtype=self.dtype)
        data_dict["output_c2ws"] = batch["outputs"]["c2w"].to(device_id, dtype=self.dtype)
        data_dict["output_fovxs"] = batch["outputs"]["fovx"].to(device_id, dtype=self.dtype)
        data_dict["output_fovys"] = batch["outputs"]["fovy"].to(device_id, dtype=self.dtype)

        data_dict["bin_token"] = 'test'

        return data_dict
    
    def configure_optimizers(self, lr):
        backbone_layers = torch.nn.ModuleList([self.backbone])
        backbone_layers_params = list(map(id, backbone_layers.parameters()))
        base_params = list(filter(lambda p: id(p) not in backbone_layers_params, self.parameters()))
        
        opt = torch.optim.AdamW(
            [{'params': base_params}, {'params': backbone_layers.parameters(), 'lr': lr*0.1}],
            lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
        return [opt]
    
    def forward(self, batch, split="train", iter=0, iter_end=100000):
        """Forward training function.
        """
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        # test_img = to_pil_image(img[0,0].clip(min=0, max=1))    
        # test_img.save('input_img.png')

        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img)

        # pixel-gs prediction
        gaussians_pixel, gaussians_feat, depth_pred, near_uv_map, far_uv_map = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        near_gaussians_pixel_mask, near_gaussians_feat_mask, near_depth_pred_mask = [], [], []
        far_gaussians_pixel_mask, far_gaussians_feat_mask, far_depth_pred_mask = [], [], []
        near_uv_map_mask, far_uv_map_mask = [], []

        depth_xyz = torch.sqrt(gaussians_pixel[..., 0:1]**2 + gaussians_pixel[..., 1:2]**2 + gaussians_pixel[..., 2:3]**2 + 1e-5)
        
        # vis_depth(depth_xyz)
        
        # decare
        if self.task == 'square':
            for b in range(bs):
                mask_pixel_i = (gaussians_pixel[b, :, 0] > x_start) & (gaussians_pixel[b, :, 0] < x_end) & \
                            (gaussians_pixel[b, :, 1] > y_start) & (gaussians_pixel[b, :, 1] < y_end) & \
                            (gaussians_pixel[b, :, 2] > z_start) & (gaussians_pixel[b, :, 2] < z_end)
                gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
                gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
                far_uv_map_i = far_uv_map[b][mask_pixel_i]
                depth_pred_i = depth_pred[b][mask_pixel_i]

                far_gaussians_pixel_mask.append(gaussians_pixel_mask_i)
                far_gaussians_feat_mask.append(gaussians_feat_mask_i)
                far_uv_map_mask.append(far_uv_map_i)
                far_depth_pred_mask.append(depth_pred_i)
        else:
            # Spherical
            for b in range(bs):
                mask_pixel_i = (depth_xyz[b, :, 0] > self.near_point_cloud_range[2]) & (depth_xyz[b, :, 0] <= self.near_point_cloud_range[5])
                # fix tab
                gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
                gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
                near_uv_map_i = near_uv_map[b][mask_pixel_i]
                depth_pred_i = depth_xyz[b][mask_pixel_i] - self.near_point_cloud_range[2]

                near_gaussians_pixel_mask.append(gaussians_pixel_mask_i)
                near_gaussians_feat_mask.append(gaussians_feat_mask_i)
                near_uv_map_mask.append(near_uv_map_i)
                near_depth_pred_mask.append(depth_pred_i)
            
            for b in range(bs):
                mask_pixel_i = (depth_xyz[b, :, 0] > self.far_point_cloud_range[2]) & (depth_xyz[b, :, 0] <= self.far_point_cloud_range[5])
                # fix tab
                gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
                gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
                far_uv_map_i = far_uv_map[b][mask_pixel_i]
                depth_pred_i = depth_xyz[b][mask_pixel_i] - self.far_point_cloud_range[2]

                far_gaussians_pixel_mask.append(gaussians_pixel_mask_i)
                far_gaussians_feat_mask.append(gaussians_feat_mask_i)
                far_uv_map_mask.append(far_uv_map_i)
                far_depth_pred_mask.append(depth_pred_i)
        
        # single_features_to_RGB(img_feats[0].squeeze(1), img_name='input_feat.png')
        gaussians_volume_near, gaussians_volume_near_filtered, gaussians_confidence_near = self.near_volume_gs(
                [img_feats[0]],
                near_gaussians_pixel_mask,
                near_gaussians_feat_mask,
                near_uv_map_mask,
                near_depth_pred_mask,
                data_dict["img_metas"])

        gaussians_volume_far, gaussians_volume_far_filtered, gaussians_confidence_far = self.far_volume_gs(
                [img_feats[0]],
                far_gaussians_pixel_mask,
                far_gaussians_feat_mask,
                far_uv_map_mask,
                far_depth_pred_mask,
                data_dict["img_metas"])        
        
        gaussians_volume_filtered = torch.cat([gaussians_volume_near_filtered, gaussians_volume_far_filtered], dim=1)
        gaussians_volume_filtered_target = torch.zeros_like(gaussians_volume_filtered, device=gaussians_volume_filtered.device)
        gaussians_confidence = torch.cat([gaussians_confidence_near, gaussians_confidence_far], dim=1)

        gaussians_volume = torch.cat([gaussians_volume_near, gaussians_volume_far], dim=1)
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)

        bs = gaussians_pixel.shape[0]
        render_c2w = data_dict["output_c2ws"]
        render_fovxs = data_dict["output_fovxs"]
        render_fovys = data_dict["output_fovys"]
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )
        # render_pkg_pixel = self.renderer.render(
        #     gaussians=gaussians_pixel,
        #     c2w=render_c2w,
        #     fovx=render_fovxs,
        #     fovy=render_fovys,
        #     rays_o=None,
        #     rays_d=None
        # )
        render_pkg_pixel_bev = self.renderer.render_orthographic(
            gaussians=gaussians_all,
            width=100,
            height=100, #mp3d 15 vigor 35
        )
        if split == "train" or split == "val":
            render_pkg_pixel = self.renderer.render(
                gaussians=gaussians_pixel,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )
            render_pkg_volume = self.renderer.render(
                gaussians=gaussians_volume,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )
        else:
            render_pkg_pixel, render_pkg_volume = None, None
        
        # render_pkg_fuse["image"] = 0.3 * render_pkg_fuse["image"] + 0.7 * render_pkg_volume["image"]
        # ======================== losses ======================== #
        loss = 0.0
        loss_terms = {}
        def set_loss(key, split, loss_value, loss_weight=1.0):
            loss_terms[f"{split}/loss_{key}"] = loss_value.item()
            loss_terms[f"{split}/loss_{key}_w"] = loss_value.item() * loss_weight

        # =================== Data preparation =================== #        
        rgb_gt = data_dict["output_imgs"]
        # rgb_gt = self.E2C(rgb_gt)
        data_dict["rgb_gt"] = rgb_gt
        depth_m_gt = data_dict["output_depths_m"]
        conf_m_gt = data_dict["output_confs_m"]
        data_dict["depth_m_gt"] = depth_m_gt
        data_dict["conf_m_gt"] = conf_m_gt
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        if self.task == 'square':
            output_positions = data_dict["output_positions"]
            mask_dptm = (output_positions[..., 0] > x_start) & (output_positions[..., 0] < x_end) & \
                        (output_positions[..., 1] > y_start) & (output_positions[..., 1] < y_end) & \
                        (output_positions[..., 2] > z_start) & (output_positions[..., 2] < z_end)
        else:
            output_positions = data_dict["output_positions"]
            output_depth =  torch.sqrt(output_positions[..., 0]**2 + output_positions[..., 1]**2 + output_positions[..., 2]**2 + 1e-5)
            mask_dptm = (output_depth > z_start) & (output_depth <= z_end)
        
        mask_dptm = mask_dptm.float()
        # mask_dptm = torch.ones_like(data_dict["output_depths_m"].squeeze(2), device=rgb_gt.device).float()
        # mask_dptm = self.E2C(mask_dptm).squeeze(2)
        data_dict["mask_dptm"] = mask_dptm

        test_img = to_pil_image(render_pkg_pixel["image"][0,0].clip(min=0, max=1))    
        test_img.save('render_pixel_360loc.png')
        test_img = to_pil_image(render_pkg_fuse["image"][0,0].clip(min=0, max=1))    
        test_img.save('render_fuse_360loc.png')
        test_img = to_pil_image(render_pkg_volume["image"][0,0].clip(min=0, max=1))    
        test_img.save('render_volume_360loc.png')
        test_img = to_pil_image(rgb_gt[0,0].clip(min=0, max=1))    
        test_img.save('render_gt_360loc.png')
        test_img = to_pil_image(render_pkg_pixel_bev["image"][0].clip(min=0, max=1))
        test_img.save('render_bev_360loc.png')
        # onlyDepth(render_pkg_volume["depth"][0,0,0], save_name='render_depth_mp3d_double.png')
        # ======================== RGB loss ======================== #
        if self.loss_args.weight_recon > 0:
            # RGB loss for omni-gs
            if self.loss_args.recon_loss_type == "l1":
                rec_loss = torch.abs(rgb_gt - render_pkg_fuse["image"])
            elif self.loss_args.recon_loss_type == "l2":
                rec_loss = (rgb_gt - render_pkg_fuse["image"]) ** 2
            loss = loss + (rec_loss.mean() * self.loss_args.weight_recon)
            set_loss("recon", split, rec_loss.mean(), self.loss_args.weight_recon)
        if self.loss_args.weight_recon_vol > 0 and iter < iter_end - 1000:
            # RGB loss for volume-gs
            if self.loss_args.recon_loss_vol_type == "l1":
                rec_loss_vol = torch.abs(rgb_gt - render_pkg_volume["image"])
            elif self.loss_args.recon_loss_vol_type == "l2":
                rec_loss_vol = (rgb_gt - render_pkg_volume["image"]) ** 2
            elif self.loss_args.recon_loss_vol_type == "l2_mask" or self.loss_args.recon_loss_vol_type == "l2_mask_self":
                rec_loss_vol = (rgb_gt * mask_dptm.unsqueeze(2) - render_pkg_volume["image"] * mask_dptm.unsqueeze(2)) ** 2
            loss = loss + (rec_loss_vol.mean() * self.loss_args.weight_recon_vol)
            set_loss("recon_vol", split, rec_loss_vol.mean(), self.loss_args.weight_recon_vol)

        # ==================== Perceptual loss ===================== #
        if self.loss_args.weight_perceptual > 0:
            # Perceptual loss for omni-gs
            ## resize images to smaller size to save memory
            p_inp_pred = maybe_resize(
                render_pkg_fuse["image"].reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]),
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_inp_gt = maybe_resize(
                rgb_gt.reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]), 
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_loss = self.perceptual_loss(p_inp_pred, p_inp_gt)
            p_loss = rearrange(p_loss, "(b v) c h w -> b v c h w", b=bs)
            p_loss = p_loss.mean()
            loss = loss + (p_loss * self.loss_args.weight_perceptual)
            set_loss("perceptual", split, p_loss, self.loss_args.weight_perceptual)
        if self.loss_args.weight_perceptual_vol > 0 and iter < iter_end - 1000:
            # Perceptual loss for volume-gs
            p_inp_pred_vol = maybe_resize(
                render_pkg_volume["image"].reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]),
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_inp_gt = maybe_resize(
                rgb_gt.reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]), 
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_inp_mask_vol = maybe_resize(
                mask_dptm.reshape(-1, 1, self.camera_args.resolution[0], self.camera_args.resolution[1]), 
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_loss_vol = self.perceptual_loss(p_inp_pred_vol * p_inp_mask_vol, p_inp_gt * p_inp_mask_vol)
            p_loss_vol = rearrange(p_loss_vol, "(b v) c h w -> b v c h w", b=bs)
            p_loss_vol = p_loss_vol.mean()
            loss = loss + (p_loss_vol * self.loss_args.weight_perceptual_vol)
            set_loss("perceptual_vol", split, p_loss_vol, self.loss_args.weight_perceptual_vol)

        # ==================== Depth loss ===================== #
        ## Depth loss for omni-gs. For regularization use.
        # depth_m_gt = self.E2C(depth_m_gt).squeeze(2)
        # conf_m_gt = self.E2C(conf_m_gt).squeeze(2)
        if self.loss_args.weight_depth_abs > 0:
            depth_abs_loss = torch.abs(render_pkg_fuse["depth"] - depth_m_gt)
            depth_abs_loss = depth_abs_loss * conf_m_gt
            depth_abs_loss = depth_abs_loss.mean()
            loss = loss + self.loss_args.weight_depth_abs * depth_abs_loss
            set_loss("depth_abs", split, depth_abs_loss, self.loss_args.weight_depth_abs)
        ## Depth loss for volume-gs
        if self.loss_args.weight_depth_abs_vol > 0 and iter < iter_end - 1000:
            depth_abs_loss_vol = torch.abs(render_pkg_volume["depth"] * mask_dptm.unsqueeze(2) - depth_m_gt * mask_dptm.unsqueeze(2))
            depth_abs_loss_vol = depth_abs_loss_vol * conf_m_gt
            depth_abs_loss_vol = depth_abs_loss_vol.mean()
            loss = loss + self.loss_args.weight_depth_abs_vol * depth_abs_loss_vol
            set_loss("depth_abs_vol", split, depth_abs_loss_vol, self.loss_args.weight_depth_abs_vol)
        
        # ====================Volume loss ===================== #
        # if self.loss_args.weight_volume_loss > 0  and iter < iter_end - 1000:
        #     volume_loss = F.mse_loss(gaussians_confidence * gaussians_volume_filtered, gaussians_volume_filtered_target, reduction='mean')
        #     loss = loss + self.loss_args.weight_volume_loss * volume_loss
        #     set_loss("volume", split, volume_loss, self.loss_args.weight_volume_loss)

        return loss, loss_terms, render_pkg_fuse, render_pkg_pixel, render_pkg_volume, gaussians_all, gaussians_pixel, gaussians_volume, data_dict
    
    def validation_step(self, batch, val_result_savedir):
        (loss_val, loss_term_val, render_pkg_fuse,
         render_pkg_pixel, render_pkg_volume, gaussians_all,
         gaussians_pixel, gaussians_volume, batch_data) = \
            self.forward(batch, "val")
        self.save_val_results(batch_data, render_pkg_fuse, render_pkg_pixel, render_pkg_volume,
                                gaussians_all, gaussians_pixel, gaussians_volume, val_result_savedir)
        return loss_term_val
    
    def forward_test(self, batch):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img, status="test")

        # pixel-gs prediction
        gaussians_pixel, gaussians_feat, depth_pred, near_uv_map, far_uv_map = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"], status='test')

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        near_gaussians_pixel_mask, near_gaussians_feat_mask, near_depth_pred_mask = [], [], []
        far_gaussians_pixel_mask, far_gaussians_feat_mask, far_depth_pred_mask = [], [], []
        near_uv_map_mask, far_uv_map_mask = [], []

        depth_xyz = torch.sqrt(gaussians_pixel[..., 0:1]**2 + gaussians_pixel[..., 1:2]**2 + gaussians_pixel[..., 2:3]**2 + 1e-5)
        # decare
        if self.task == 'square':
            for b in range(bs):
                mask_pixel_i = (gaussians_pixel[b, :, 0] > x_start) & (gaussians_pixel[b, :, 0] < x_end) & \
                            (gaussians_pixel[b, :, 1] > y_start) & (gaussians_pixel[b, :, 1] < y_end) & \
                            (gaussians_pixel[b, :, 2] > z_start) & (gaussians_pixel[b, :, 2] < z_end)
                gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
                gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
                far_uv_map_i = far_uv_map[b][mask_pixel_i]
                depth_pred_i = depth_pred[b][mask_pixel_i]

                far_gaussians_pixel_mask.append(gaussians_pixel_mask_i)
                far_gaussians_feat_mask.append(gaussians_feat_mask_i)
                far_uv_map_mask.append(far_uv_map_i)
                far_depth_pred_mask.append(depth_pred_i)
        else:
            # Spherical
            for b in range(bs):
                mask_pixel_i = (depth_xyz[b, :, 0] > self.near_point_cloud_range[2]) & (depth_xyz[b, :, 0] <= self.near_point_cloud_range[5])
                # fix tab
                gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
                gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
                near_uv_map_i = near_uv_map[b][mask_pixel_i]
                depth_pred_i = depth_xyz[b][mask_pixel_i] - self.near_point_cloud_range[2]

                near_gaussians_pixel_mask.append(gaussians_pixel_mask_i)
                near_gaussians_feat_mask.append(gaussians_feat_mask_i)
                near_uv_map_mask.append(near_uv_map_i)
                near_depth_pred_mask.append(depth_pred_i)
            
            for b in range(bs):
                mask_pixel_i = (depth_xyz[b, :, 0] > self.far_point_cloud_range[2]) & (depth_xyz[b, :, 0] <= self.far_point_cloud_range[5])
                # fix tab
                gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
                gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
                far_uv_map_i = far_uv_map[b][mask_pixel_i]
                depth_pred_i = depth_xyz[b][mask_pixel_i] - self.far_point_cloud_range[2]

                far_gaussians_pixel_mask.append(gaussians_pixel_mask_i)
                far_gaussians_feat_mask.append(gaussians_feat_mask_i)
                far_uv_map_mask.append(far_uv_map_i)
                far_depth_pred_mask.append(depth_pred_i)
        
        with self.benchmarker.time("volume_gs"):
            gaussians_volume_near, _, _ = self.near_volume_gs(
                    [img_feats[0]],
                    near_gaussians_pixel_mask,
                    near_gaussians_feat_mask,
                    near_uv_map_mask,
                    near_depth_pred_mask,
                    data_dict["img_metas"], status='test')

            gaussians_volume_far, _, _ = self.far_volume_gs(
                    [img_feats[0]],
                    far_gaussians_pixel_mask,
                    far_gaussians_feat_mask,
                    far_uv_map_mask,
                    far_depth_pred_mask,
                    data_dict["img_metas"], status='test')        
            
            gaussians_volume = torch.cat([gaussians_volume_near, gaussians_volume_far], dim=1)
        
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)
        # gaussians_all = gaussians_volume
        bs = gaussians_all.shape[0]
        render_c2w = data_dict["output_c2ws"]
        render_fovxs = data_dict["output_fovxs"]
        render_fovys = data_dict["output_fovys"]
        
        with self.benchmarker.time("render", num_calls=render_c2w.shape[1]):
            render_pkg_fuse = self.renderer.render(
                gaussians=gaussians_all,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )

        output_imgs = render_pkg_fuse["image"] # b v 3 h w
        output_depths = render_pkg_fuse["depth"].squeeze(2) # b v h w

        target_imgs = data_dict["output_imgs"] # b v 3 h w
        target_depths = data_dict["output_depths"] # b v h w
        target_depths_m = data_dict["output_depths_m"] # b v h w

        preds = {"img": output_imgs, "depth": output_depths, "gaussian": gaussians_all}
        gts = {"img": target_imgs, "depth": target_depths, "depth_m": target_depths_m}

        return preds, gts
    
    def forward_demo(self, batch):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img, status="test")

        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])
        
        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)

        bs = gaussians_all.shape[0]     
        # forward 3 meters, return, and then rotate. backward 3 meters, return, and then rotate.
        c2w_cf = data_dict["output_c2ws"][:, -6]
        c2w_cf_forward = c2w_cf.clone()
        c2w_cf_forward[..., 1, 3] = c2w_cf_forward[..., 1, 3] + 3
        c2w_cfr = data_dict["output_c2ws"][:, -5]
        c2w_cfl = data_dict["output_c2ws"][:, -4]
        c2w_cb = data_dict["output_c2ws"][:, -3]
        c2w_cb[..., 1, 3] = c2w_cb[..., 1, 3] + 1.5
        c2w_cb_backward = c2w_cb.clone()
        c2w_cb_backward[..., 1, 3] = c2w_cb_backward[..., 1, 3] - 3
        c2w_cbl = data_dict["output_c2ws"][:, -2]
        c2w_cbr = data_dict["output_c2ws"][:, -1]
        # cf -> cfr -> cbr -> cb -> cbl -> cfl -> cf
        num_frames_short = 60
        num_frames_long = 120
        num_frames_all = 60 * 4 + 120 * 6
        t_short = torch.linspace(0, 1, num_frames_short, dtype=torch.float32, device=self.device)
        t_long = torch.linspace(0, 1 - 1 / (num_frames_long + 1), num_frames_long, dtype=torch.float32, device=self.device)
        # obtain camera trajectories for each clip
        c2w_interp_forward0 = interpolate_extrinsics(c2w_cf, c2w_cf_forward, t_short)
        c2w_interp_forward1 = interpolate_extrinsics(c2w_cf_forward, c2w_cf, t_short)
        c2w_interp_0 = interpolate_extrinsics(c2w_cf, c2w_cfr, t_long)
        c2w_interp_1 = interpolate_extrinsics(c2w_cfr, c2w_cbr, t_long)
        c2w_interp_2 = interpolate_extrinsics(c2w_cbr, c2w_cb, t_long)
        c2w_interp_backward0 = interpolate_extrinsics(c2w_cb, c2w_cb_backward, t_short)
        c2w_interp_backward1 = interpolate_extrinsics(c2w_cb_backward, c2w_cb, t_short)
        c2w_interp_3 = interpolate_extrinsics(c2w_cb, c2w_cbl, t_long)
        c2w_interp_4 = interpolate_extrinsics(c2w_cbl, c2w_cfl, t_long)
        c2w_interp_5 = interpolate_extrinsics(c2w_cfl, c2w_cf, t_long)
        c2w_interp = torch.cat([c2w_interp_forward0, c2w_interp_forward1,
                                c2w_interp_0, c2w_interp_1, c2w_interp_2,
                                c2w_interp_backward0, c2w_interp_backward1,
                                c2w_interp_3, c2w_interp_4, c2w_interp_5], dim=1)
        fovxs_interp = data_dict["output_fovxs"][:, -6:-5].repeat(1, num_frames_all)
        fovys_interp = data_dict["output_fovys"][:, -6:-5].repeat(1, num_frames_all)
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=c2w_interp,
            fovx=fovxs_interp,
            fovy=fovys_interp,
            rays_o=None,
            rays_d=None
        )

        output_imgs = render_pkg_fuse["image"] # b v 3 h w
        output_depths = render_pkg_fuse["depth"].squeeze(2) # b v h w

        preds = {"img": output_imgs, "depth": output_depths}

        return preds, data_dict["bin_token"]

    def save_val_results(self, batch_gt, render_pkg_fuse, render_pkg_pixel, render_pkg_volume,
                         gaussians_all, gaussians_pixel, gaussians_volume, save_dir):
        # os.makedirs(save_dir, exist_ok=True)
        batch_size = render_pkg_fuse["image"].shape[0]
        n_rand_view = render_pkg_fuse["image"].shape[1]

        rgbs_gt = batch_gt["output_imgs"].cpu()
        depths_gt = batch_gt["output_depths"]
        depths_gt = (depths_gt / depths_gt.max()).repeat(1, 1, 3, 1, 1).cpu()
        depths_m_gt = batch_gt["output_depths_m"]
        depths_m_gt = (depths_m_gt / depths_m_gt.max()).repeat(1, 1, 3, 1, 1).cpu()
        confs_m_gt = batch_gt["output_confs_m"]
        confs_m_gt = confs_m_gt.repeat(1, 1, 3, 1, 1).cpu()
        mask_dptm = batch_gt["mask_dptm"].unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()

        def save_vis(prefix, i, save_dir, n_rand_view, render_pkg, gaussians, rgbs_gt, depths_m_gt, mask_dptm, renderer):
            
            sample_save_dir = "/".join(save_dir.split('/')[:-1])
            os.makedirs(sample_save_dir, exist_ok=True)

            for v in range(n_rand_view):
                rgb = render_pkg["image"][i, v].cpu()
                depth = render_pkg["depth"][i, v]
                h, w = depth.shape[1:]
                depth_abs = (depth / depth.max()).repeat(3, 1, 1).cpu()
                cat_gt = torch.cat(
                        [rgbs_gt[i, v], depths_m_gt[i, v], mask_dptm[i, v]],
                        dim=-1
                    )
                cat_pred = torch.cat(
                        [rgb, depth_abs, mask_dptm[i, v]], dim=-1
                    )
                grid = torch.cat(
                    [cat_gt, cat_pred], dim=1
                )
                grid = (grid.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                imageio.imwrite(os.path.join(sample_save_dir, f"{save_dir.split('/')[-1]}-sample-{i}-{prefix}-{v}.png"), grid)
            # if gaussians is not None:
            #     gs_save_path = os.path.join(sample_save_dir, f"sample-{i}-{prefix}.ply")
            #     gaussians_reformat = torch.cat([gaussians[i:i+1, :, 0:3],
            #                                     gaussians[i:i+1, :, 6:7],
            #                                     gaussians[i:i+1, :, 11:14],
            #                                     gaussians[i:i+1, :, 7:11],
            #                                     gaussians[i:i+1, :, 3:6]], dim=-1)
            #     renderer.save_ply(gaussians_reformat, gs_save_path)
            
        for i in range(batch_size):
            save_vis("omni", i, save_dir, n_rand_view, render_pkg_fuse, gaussians_all, rgbs_gt, depths_m_gt, mask_dptm, self.renderer)
        
        if render_pkg_pixel is not None:
            for i in range(batch_size):
                save_vis("pixel", i, save_dir, n_rand_view, render_pkg_pixel, None, rgbs_gt, depths_m_gt, mask_dptm, self.renderer)
        
        if render_pkg_volume is not None:
            for i in range(batch_size):
                save_vis("volume", i, save_dir, n_rand_view, render_pkg_volume, None, rgbs_gt, depths_m_gt, mask_dptm, self.renderer)
