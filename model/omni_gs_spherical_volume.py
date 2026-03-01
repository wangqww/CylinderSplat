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
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor
from .gaussian import GaussianRenderer
from .losses import LPIPS, LossDepthTV
from .utils.image import maybe_resize
from .utils.benchmarker import Benchmarker
from .utils.interpolation import interpolate_extrinsics

from vis_feat import single_features_to_RGB, reduce_gaussian_features_to_rgb, save_point_cloud, point_features_to_rgb_colormap

from pano2cube import Equirec2Cube, Cube2Equirec
from vis_feat import single_features_to_RGB
import torchvision.transforms as transforms
to_pil_image = transforms.ToPILImage()
import matplotlib.cm as cm
import cv2
from sample_anchors import transform_points

def onlyDepth(depth, save_name):
    cmap = cm.Spectral
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().detach().numpy()
    depth = depth.astype(np.uint8)
    
    c_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(save_name, c_depth)

@MODELS.register_module()
class OmniGaussianSphericalVolume(BaseModule):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 pixel_gs=None,
                 volume_gs=None,
                 camera_args=None,
                 loss_args=None,
                 dataset_params=None,
                 use_checkpoint=False,
                 point_cloud_range=None,
                 **kwargs,
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        if backbone:
            self.backbone = MODELS.build(backbone)
        if neck:
            self.neck = MODELS.build(neck)
        self.pixel_gs = MODELS.build(pixel_gs)
        for param in self.pixel_gs.parameters():
            param.requires_grad = False
        self.pixel_gs.eval()
        if backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        if neck:
            for param in self.neck.parameters():
                param.requires_grad = False
            self.neck.eval()

        self.volume_gs = MODELS.build(volume_gs)
        # self.volume_gs = MODELS.build(volume_gs)
        self.dataset_params = dataset_params
        self.camera_args = camera_args
        self.loss_args = loss_args

        self.point_cloud_range = point_cloud_range
        self.renderer = GaussianRenderer(self.device, **camera_args)

        # Perceptual loss
        if self.loss_args.weight_perceptual > 0:
            # self.perceptual_loss = LPIPS(net="vgg")
            self.perceptual_loss = LPIPS().eval()
        else:
            self.perceptual_loss = None

        # record runtime
        self.benchmarker = Benchmarker()

        self.E2C = Equirec2Cube(equ_h=160, equ_w=320, cube_length=self.camera_args['resolution'][0])
        self.C2E = Cube2Equirec(cube_length=40, equ_h=80)

    def extract_img_feat(self, img, depths_in, confs_in, pluckers, viewmats, status="train"):
        """Extract features of images."""
        # B, N, C, H, W = img.size()
        # img = img.view(B * N, C, H, W)

        if self.use_checkpoint and status != "test":
            img_feats = torch.utils.checkpoint.checkpoint(
                            self.backbone, 
                            img,
                            depths_in,
                            confs_in,
                            pluckers,
                            viewmats, 
                            use_reentrant=False)
        else:
            img_feats = self.backbone(img,depths_in,confs_in,pluckers,viewmats)
        # img_feats = self.neck(img_feats) # BV, C, H, W
        # img_feats_reshaped = []
        # for img_feat in img_feats:
        #     _, C, H, W = img_feat.size()
        #     # single_features_to_RGB(img_feat)
        #     img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats

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
            # 1. 動態獲取當前樣本的視圖數量 v
            v = w2i.shape[0]
            if v < 2: # 如果視圖少於2個，無法計算相對姿態，跳過或只用絕對姿態
                img_metas.append({"lidar2img": w2i @ w2i.inverse(), "img_shape": [[h, w]] * v})
                continue

            # 2. 循環遍歷每一個視圖，將其輪流作為參考視圖 (reference camera)
            for i in range(v):
                # 複製一份原始姿態，以防修改原數據
                w2i_relative = w2i.clone()
                
                # 選取第 i 個視圖作為參考相機
                ref_cam = w2i[i]
                
                # 計算參考相機的逆矩陣，用於將世界坐標轉換到該相機的坐標系
                ref_cam_inv = ref_cam.inverse()
                
                # 3. 使用向量化操作，將所有視圖的姿態都轉換為相對於 ref_cam 的姿態
                # 這裡的矩陣乘法 @ 會自動進行廣播 (broadcasting)
                # w2i 的形狀是 [v, 4, 4], ref_cam_inv 的形狀是 [4, 4]
                # PyTorch 會將 ref_cam_inv 與 w2i 中的每一個 4x4 矩陣相乘
                w2i_relative = w2i @ ref_cam_inv
                
                # 此時，w2i_relative[i] 將會是一個單位矩陣，因為它是 ref_cam @ ref_cam_inv
                
                # 4. 將這一組增強後的相對姿態添加到 meta 列表中
                img_metas.append({"lidar2img": w2i_relative, "img_shape": [[h, w]] * v})

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

        bs, v, _, h, w = img.shape

        # pixel-gs prediction
        with torch.no_grad():
            img_feats = self.extract_img_feat(img=img,
                                            depths_in=data_dict["depths"], 
                                            confs_in=data_dict["confs"], 
                                            pluckers=data_dict["pluckers"],
                                            viewmats=data_dict["c2ws"]
                                            )
            # pixel-gs prediction
            gaussians = self.pixel_gs(
                    img, img_feats,
                    data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                    data_dict["rays_o"], data_dict["rays_d"])

            gaussians_pixel = gaussians["gaussians"]
            gaussians_feat = gaussians["features"]

            # volume-pixel-gs prediction
            tmp_gaussians_pixel = repeat(gaussians_pixel[:,None,:,:], 'b vo n c -> b (vo v) n c', v=v).contiguous()
            tmp_gaussians_pixel = rearrange(tmp_gaussians_pixel, 'b v n c -> (b v) n c').contiguous()
            tmp_gaussians_feat = repeat(gaussians_feat[:,None,:,:], 'b vo n c -> b (vo v) n c', v=v).contiguous()
            volume_gaussians_feat = rearrange(tmp_gaussians_feat, 'b v n c -> (b v) n c').contiguous()
            tmp_gaussians_points = transform_points(tmp_gaussians_pixel[..., :3], rearrange(torch.inverse(data_dict["c2ws"]), "b v h w -> (b v) h w"))
            volume_gaussians_pixel = torch.cat([tmp_gaussians_points, tmp_gaussians_pixel[..., 3:]], dim=-1)
        
            # original
            # volume_gaussians_pixel = gaussians_pixel
            # volume_gaussians_feat = gaussians_feat

            # volume-gs prediction
            gaussians_pixel_mask, gaussians_feat_mask = [], []
            spherical_r = torch.sqrt(volume_gaussians_pixel[..., 0]**2 + volume_gaussians_pixel[..., 1]**2 + volume_gaussians_pixel[..., 2]**2 + 1e-5)
            # Spherical
            for b in range(bs * v):
                mask_pixel_i = (spherical_r[b] > self.point_cloud_range[2]) & (spherical_r[b] < self.point_cloud_range[5])
                # fix tab
                gaussians_pixel_mask_i = volume_gaussians_pixel[b][mask_pixel_i]
                gaussians_feat_mask_i = volume_gaussians_feat[b][mask_pixel_i]

                gaussians_pixel_mask.append(gaussians_pixel_mask_i)
                gaussians_feat_mask.append(gaussians_feat_mask_i)

        # single_features_to_RGB(img_feats[0].squeeze(1), img_name='input_feat.png')
        
        gaussians_volume = self.volume_gs(
            [repeat(img_feats['trans_features'][0], "b vo c h w -> (b v) vo c h w", v=v)],
            gaussians_pixel_mask,
            gaussians_feat_mask,
            repeat(data_dict["imgs"], "b vo c h w -> (b v) vo c h w", v=v),
            repeat(data_dict["depths"], "b vo c h w -> (b v) vo c h w", v=v),
            data_dict["img_metas"]
        )

        new_gaussian_points = transform_points(gaussians_volume[..., :3], rearrange(data_dict["c2ws"], "b v h w -> (b v) h w"))
        gaussians_volume = torch.cat([new_gaussian_points, gaussians_volume[..., 3:]], dim=-1)
        gaussians_volume = rearrange(gaussians_volume, '(b v) n c -> b (v n) c', v=v)
                
        # original
        # gaussians_volume = self.volume_gs(
        #     [img_feats],
        #     gaussians_pixel_mask,
        #     gaussians_feat_mask,
        #     data_dict["imgs"],
        #     data_dict["depths"],
        #     data_dict["img_metas"]
        # )

        render_c2w = data_dict["output_c2ws"]
        render_fovxs = data_dict["output_fovxs"]
        render_fovys = data_dict["output_fovys"]
        
        # ======================== render ======================== #
        render_pkg_pixel_bev = self.renderer.render_orthographic(
            gaussians=gaussians_volume,
            width=30,
            height=30, #mp3d 15 vigor 35
        )
        if split == "train" or split == "val":
            render_pkg_volume = self.renderer.render(
                gaussians=gaussians_volume,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )
            render_pkg_pixel = self.renderer.render(
                gaussians=gaussians_pixel,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )
        else:
            render_pkg_pixel, render_pkg_volume = None, None
        
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

        output_positions = data_dict["output_positions"].view(bs, -1, 3)  # [B,v,h,w,xyz] -> [b,np,xyz]
        positions_expanded = output_positions.unsqueeze(1).expand(-1, v, -1, -1)
        positions_batched = positions_expanded.reshape(bs * v, -1, 3)

        transformed_positions = transform_points(positions_batched, rearrange(torch.inverse(data_dict["c2ws"]), "b v h w -> (b v) h w"))
        output_spherical_r = torch.sqrt(transformed_positions[..., 0]**2 + transformed_positions[..., 1]**2 + transformed_positions[..., 2]**2 + 1e-5)
        mask_inside = (output_spherical_r  > self.point_cloud_range[2]) & (output_spherical_r  > self.point_cloud_range[5])
        mask_dptm = mask_inside.view(bs, v, render_c2w.shape[1], h, w).any(dim=1).float()
        data_dict["mask_dptm"] = mask_dptm

        test_img = to_pil_image(render_pkg_pixel["image"][0,0].clip(min=0, max=1))    
        test_img.save('render_pixel_mp3d_volume_S.png')
        test_img = to_pil_image(render_pkg_volume["image"][0,0].clip(min=0, max=1))    
        test_img.save('render_volume_mp3d_volume_S.png')
        test_img = to_pil_image(rgb_gt[0,0].clip(min=0, max=1))    
        test_img.save('render_gt_mp3d_volume_S.png')
        test_img = to_pil_image(render_pkg_pixel_bev["image"][0].clip(min=0, max=1))
        test_img.save('render_bev_mp3d_volume_S.png')


        # vis rgb points
        # idx = 4
        # opactity = gaussians_volume[..., 6:7]
        # opactity_mask = (opactity > 0.9).squeeze(-1)
        # gaussians_volume_save = gaussians_volume[idx][opactity_mask[idx]]
        # points_xyz = gaussians_volume_save[..., :3].detach().cpu().numpy()
        # points_rgb = gaussians_volume_save[..., 3:6].detach().cpu().numpy()
        # save_point_cloud(points_xyz, points_rgb, filename="point_cloud.ply")


        # onlyDepth(render_pkg_volume["depth"][0,0,0], save_name='render_depth_mp3d_double.png')
        # ======================== RGB loss ======================== #
        if self.loss_args.weight_recon_vol > 0:
            # RGB loss for volume-gs
            if self.loss_args.recon_loss_vol_type == "l1":
                rec_loss_vol = torch.abs(rgb_gt - render_pkg_volume["image"])
            elif self.loss_args.recon_loss_vol_type == "l2":
                rec_loss_vol = (rgb_gt - render_pkg_volume["image"]) ** 2
            elif self.loss_args.recon_loss_vol_type == "l2_mask" or self.loss_args.recon_loss_vol_type == "l2_mask_self":
                rec_loss_vol = (rgb_gt - render_pkg_volume["image"]) ** 2
            loss = loss + (rec_loss_vol.mean() * self.loss_args.weight_recon_vol)
            set_loss("recon_vol", split, rec_loss_vol.mean(), self.loss_args.weight_recon_vol)

        if self.loss_args.weight_perceptual_vol > 0:
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
            p_loss_vol = self.perceptual_loss(p_inp_pred_vol, p_inp_gt)
            p_loss_vol = rearrange(p_loss_vol, "(b v) c h w -> b v c h w", b=bs)
            p_loss_vol = p_loss_vol.mean()
            loss = loss + (p_loss_vol * self.loss_args.weight_perceptual_vol)
            set_loss("perceptual_vol", split, p_loss_vol, self.loss_args.weight_perceptual_vol)

        # ==================== Depth loss ===================== #
        if self.loss_args.weight_depth_abs_vol > 0:
            depth_abs_loss_vol = torch.abs(render_pkg_volume["depth"] - depth_m_gt)
            depth_abs_loss_vol = depth_abs_loss_vol * conf_m_gt
            depth_abs_loss_vol = depth_abs_loss_vol.mean()
            loss = loss + self.loss_args.weight_depth_abs_vol * depth_abs_loss_vol
            set_loss("depth_abs_vol", split, depth_abs_loss_vol, self.loss_args.weight_depth_abs_vol)        
        # ====================Volume loss ===================== #
        if self.loss_args.weight_volume_loss > 0:
            volume_loss = (- render_pkg_volume["alpha"] * torch.log(render_pkg_volume["alpha"] + 1e-8)
                           - (1 - render_pkg_volume["alpha"]) * torch.log(1 - render_pkg_volume["alpha"] + 1e-8)).mean()
            loss = loss + self.loss_args.weight_volume_loss * volume_loss
            set_loss("volume", split, volume_loss, self.loss_args.weight_volume_loss)        

        return loss, loss_terms, render_pkg_volume, render_pkg_pixel, render_pkg_volume, gaussians_volume, gaussians_pixel, gaussians_volume, data_dict
    
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
        bs, v, _, _, _ = img.shape
        img_feats = self.extract_img_feat(img=img,
                                          depths_in=data_dict["depths"], 
                                          confs_in=data_dict["confs"], 
                                          pluckers=data_dict["pluckers"],
                                          viewmats=data_dict["c2ws"],
                                          status="test"
                                        )
        # pixel-gs prediction
        gaussians = self.pixel_gs(
                img, img_feats,
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"], status="test")
        
        gaussians_pixel = gaussians["gaussians"]
        gaussians_feat = gaussians["features"]
        
        # volume-pixel-gs prediction
        tmp_gaussians_pixel = repeat(gaussians_pixel[:,None,:,:], 'b vo n c -> b (vo v) n c', v=v).contiguous()
        tmp_gaussians_pixel = rearrange(tmp_gaussians_pixel, 'b v n c -> (b v) n c').contiguous()
        tmp_gaussians_feat = repeat(gaussians_feat[:,None,:,:], 'b vo n c -> b (vo v) n c', v=v).contiguous()
        volume_gaussians_feat = rearrange(tmp_gaussians_feat, 'b v n c -> (b v) n c').contiguous()
        tmp_gaussians_points = transform_points(tmp_gaussians_pixel[..., :3], rearrange(torch.inverse(data_dict["c2ws"]), "b v h w -> (b v) h w"))
        volume_gaussians_pixel = torch.cat([tmp_gaussians_points, tmp_gaussians_pixel[..., 3:]], dim=-1)

        # original
        # volume_gaussians_pixel = gaussians_pixel
        # volume_gaussians_feat = gaussians_feat

        # volume-gs prediction
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        spherical_r = torch.sqrt(volume_gaussians_pixel[..., 0]**2 + volume_gaussians_pixel[..., 1]**2 + volume_gaussians_pixel[..., 2]**2 + 1e-5)
        # Spherical
        for b in range(bs*v):
            mask_pixel_i = (spherical_r[b] > self.point_cloud_range[2]) & (spherical_r[b] < self.point_cloud_range[5])
            # fix tab
            gaussians_pixel_mask_i = volume_gaussians_pixel[b][mask_pixel_i]
            gaussians_feat_mask_i = volume_gaussians_feat[b][mask_pixel_i]

            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        
        with self.benchmarker.time("volume_gs"):
            gaussians_volume = self.volume_gs(
                [repeat(img_feats['trans_features'][0], "b vo c h w -> (b v) vo c h w", v=v)],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                repeat(data_dict["imgs"], "b vo c h w -> (b v) vo c h w", v=v),
                repeat(data_dict["depths"], "b vo c h w -> (b v) vo c h w", v=v),
                data_dict["img_metas"]
            )

            new_gaussian_points = transform_points(gaussians_volume[..., :3], rearrange(data_dict["c2ws"], "b v h w -> (b v) h w"))
            gaussians_volume = torch.cat([new_gaussian_points, gaussians_volume[..., 3:]], dim=-1)
            gaussians_volume = rearrange(gaussians_volume, '(b v) n c -> b (v n) c', v=v)

            # original
            # gaussians_volume = self.volume_gs(
            #     [img_feats],
            #     gaussians_pixel_mask,
            #     gaussians_feat_mask,
            #     data_dict["imgs"],
            #     data_dict["depths"],
            #     data_dict["img_metas"]
            # )

        gaussians_all = gaussians_volume
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
        target_depths = data_dict["output_depths"] # b v 1 h w
        target_depths_m = data_dict["output_depths_m"] # b v 1 h w
        
        test_img = to_pil_image(target_imgs[0,0])    
        test_img.save('render_gt_mp3d_volume_t.png')
        test_img = to_pil_image(output_imgs[0,0])    
        test_img.save('render_volume_mp3d_volume_t.png')

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
