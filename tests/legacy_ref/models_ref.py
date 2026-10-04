"""Frozen copies of the four stage models as released (commit f7b20b9).

Verbatim method bodies, copied with `git show f7b20b9:<path>`; each block names its
source file and line range. Only the class statements are new: the classes derive from
BaseModule directly and are not registered in MODELS (their names differ from the live
ones). Methods the tests never call (configure_optimizers, validation_step, forward_demo,
save_val_results, voxelizaton_with_fusion, pad_tensor_list) are left out.

The live models must match these copies bitwise whenever every switch is at its default
(tests/test_switches_identity_models.py, tests/test_render_removal.py). Do not edit the
copied blocks.

Module-level names below are the ones the copied code uses; the tests monkeypatch
MODELS, GaussianRenderer, LPIPS (and torch.load for the Pixel checkpoint) with CPU stubs.
"""

from collections import OrderedDict

import torch
from einops import rearrange, repeat
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import torchvision.transforms as transforms

from model.gaussian import GaussianRenderer
from model.losses import LPIPS
from model.utils.image import maybe_resize
from model.utils.benchmarker import Benchmarker
from sample_anchors import transform_points

to_pil_image = transforms.ToPILImage()



class LegacyOmniGaussianCylinderAll(BaseModule):
    """model/omni_gs_cylinder_all.py @ f7b20b9 (class body excerpts)."""

    # ---- __init__: model/omni_gs_cylinder_all.py lines 45-80 @ f7b20b9 (verbatim) ----
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
        
        self.backbone = MODELS.build(backbone)
        self.pixel_gs = MODELS.build(pixel_gs)
        self.volume_gs = MODELS.build(volume_gs)
        
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

    # ---- extract_img_feat: model/omni_gs_cylinder_all.py lines 82-103 @ f7b20b9 (verbatim) ----
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
        # img_feats_reshaped = []       
        # for img_feat in img_feats:
        #     _, C, H, W = img_feat.size()
        #     # single_features_to_RGB(img_feat)
        #     img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats

    # ---- device / dtype: model/omni_gs_cylinder_all.py lines 105-111 @ f7b20b9 (verbatim) ----
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # ---- plucker_embedder: model/omni_gs_cylinder_all.py lines 113-121 @ f7b20b9 (verbatim) ----
    def plucker_embedder(
        self, 
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker

    # ---- get_data: model/omni_gs_cylinder_all.py lines 123-200 @ f7b20b9 (verbatim) ----
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
                img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
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
        # original
        # for w2i in batch["inputs_vol"]["w2i"]:    
        #     img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
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

        # for real depth
        data_dict['depth_gt'] = batch["outputs"]['depth_gt'].to(device_id, dtype=self.dtype)
        data_dict['mask_gt'] = batch["outputs"]['mask_gt'].to(device_id, dtype=torch.bool)

        data_dict["bin_token"] = 'test'

        return data_dict

    # ---- forward: model/omni_gs_cylinder_all.py lines 263-590 @ f7b20b9 (verbatim) ----
    def forward(self, batch, split="train", iter=0, iter_end=100000):
        """Forward training function.
        """
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        # test_img = to_pil_image(img[0,0].clip(min=0, max=1))    
        # test_img.save('input_img.png')

        bs, v, _, h, w = img.shape
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
                data_dict["rays_o"], data_dict["rays_d"], data_dict["c2ws"])
        
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
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []

        cylinder_r = torch.sqrt(volume_gaussians_pixel[..., 0]**2 + volume_gaussians_pixel[..., 2]**2 + 1e-5)
        
        # Cylinder
        mask_pixel = (cylinder_r <= self.point_cloud_range[3]) & \
                    (volume_gaussians_pixel[..., 1] >= self.point_cloud_range[2]) & \
                    (volume_gaussians_pixel[..., 1] <= self.point_cloud_range[5])
        gaussians_pixel_mask = [volume_gaussians_pixel[b][mask_pixel[b]] for b in range(bs * v)]
        gaussians_feat_mask = [volume_gaussians_feat[b][mask_pixel[b]] for b in range(bs * v)]

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

        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)

        # ======================== fuse ======================== #
        # pts_all = gaussians_all[..., :3]
        # feats_all = gaussians_all[..., 3:]
        # conf_all = gaussians_all[..., 6]
        # neural_feats_list, neural_pts_list = [], []

        # for b_i in range(bs):
        #     neural_pts, neural_feats = self.voxelizaton_with_fusion(
        #         feats_all[b_i],
        #         pts_all[b_i],
        #         conf=conf_all[b_i],
        #         voxel_size=0.02
        #     )
        #     neural_feats_list.append(neural_feats)
        #     neural_pts_list.append(neural_pts)

        # max_voxels = max(f.shape[0] for f in neural_feats_list)
        # neural_feats = self.pad_tensor_list(
        #     neural_feats_list, (max_voxels,), value=-1e10
        # )

        # neural_pts = self.pad_tensor_list(
        #     neural_pts_list, (max_voxels,), -1e4
        # )  # -1 == invalid voxel

        # gaussians_all = torch.cat([neural_pts, neural_feats], dim=-1)

        # ======================== render ======================== #
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
        # panorama_ray_d = torch.cat((data_dict["output_rays_o"], data_dict["output_rays_d"]), dim=-1)
        # panorama_ray_d = rearrange(panorama_ray_d, "b v h w c -> (b v) c h w").contiguous()
        # panorama_ray_d = self.E2C(panorama_ray_d)
        # panorama_ray_d = rearrange(panorama_ray_d, "(b n) v c h w -> b (n v) h w c", b=bs, n=img.shape[2]).contiguous()
        # render_pkg_volume = self.renderer.render(
        #     gaussians=gaussians_volume,
        #     c2w=render_c2w,
        #     fovx=render_fovxs,
        #     fovy=render_fovys,
        #     rays_o=None,
        #     rays_d=panorama_ray_d,
        # )
        render_pkg_pixel_bev = self.renderer.render_orthographic(
            gaussians=gaussians_all,
            width=30,
            height=30, #mp3d 15 vigor 35
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
        output_cylinder_r = torch.sqrt(transformed_positions[..., 0]**2 + transformed_positions[..., 2]**2 + 1e-5)
        mask_inside = (output_cylinder_r < x_end) & (transformed_positions[..., 1] > z_start) & (transformed_positions[..., 1] < z_end)
        mask_dptm = mask_inside.view(bs, v, render_c2w.shape[1], h, w).any(dim=1).float()
        data_dict["mask_dptm"] = mask_dptm

        # test_img = to_pil_image(mask_dptm[0])    
        # test_img.save('mask_dptm.png')

        test_img = to_pil_image(render_pkg_fuse["image"][0,1].clip(min=0, max=1))    
        test_img.save('render_fuse_mp3d_all.png')
        test_img = to_pil_image(render_pkg_pixel["image"][0,1].clip(min=0, max=1))    
        test_img.save('render_pixel_mp3d_all.png')
        test_img = to_pil_image(render_pkg_volume["image"][0,1].clip(min=0, max=1))    
        test_img.save('render_volume_mp3d_all.png')
        test_img = to_pil_image(render_pkg_pixel_bev["image"][0].clip(min=0, max=1))
        test_img.save('render_bev_mp3d_all.png')
        test_img = to_pil_image(rgb_gt[0,1].clip(min=0, max=1))    
        test_img.save('render_gt_mp3d_all.png')

        # vis rgb points
        # idx = 4
        # opactity = gaussians_all[..., 6:7]
        # opactity_mask = (opactity > 0.1).squeeze(-1)
        # gaussians_all_save = gaussians_all[idx][opactity_mask[idx]]
        # points_xyz = gaussians_all_save[..., :3].detach().cpu().numpy()
        # points_rgb = gaussians_all_save[..., 3:6].detach().cpu().numpy()
        # save_point_cloud(points_xyz, points_rgb, filename="point_cloud.ply")


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

        # if self.loss_args.weight_recon_vol > 0:
        #     # RGB loss for pixel-gs
        #     if self.loss_args.recon_loss_type == "l1":
        #         rec_loss_vol = torch.abs(rgb_gt - render_pkg_pixel["image"])
        #     elif self.loss_args.recon_loss_type == "l2":
        #         rec_loss_vol = (rgb_gt - render_pkg_pixel["image"]) ** 2
        #     loss = loss + (rec_loss_vol.mean() * self.loss_args.weight_recon_vol)
        #     set_loss("recon_vol", split, rec_loss_vol.mean(), self.loss_args.weight_recon_vol)

        if self.loss_args.weight_recon_vol > 0:
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

        # if self.loss_args.weight_perceptual_vol > 0:
        #     # Perceptual loss for pixel-gs
        #     ## resize images to smaller size to save memory
        #     p_inp_pred = maybe_resize(
        #         render_pkg_pixel["image"].reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]),
        #         tgt_reso=self.loss_args.perceptual_resolution
        #     )
        #     p_inp_gt = maybe_resize(
        #         rgb_gt.reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]), 
        #         tgt_reso=self.loss_args.perceptual_resolution
        #     )
        #     p_loss = self.perceptual_loss(p_inp_pred, p_inp_gt)
        #     p_loss = rearrange(p_loss, "(b v) c h w -> b v c h w", b=bs)
        #     p_loss = p_loss.mean()
        #     loss = loss + (p_loss * self.loss_args.weight_perceptual_vol)
        #     set_loss("perceptual_pixel", split, p_loss, self.loss_args.weight_perceptual_vol)

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
            p_loss_vol = self.perceptual_loss(p_inp_pred_vol * p_inp_mask_vol, p_inp_gt * p_inp_mask_vol)
            p_loss_vol = rearrange(p_loss_vol, "(b v) c h w -> b v c h w", b=bs)
            p_loss_vol = p_loss_vol.mean()
            loss = loss + (p_loss_vol * self.loss_args.weight_perceptual_vol)
            set_loss("perceptual_vol", split, p_loss_vol, self.loss_args.weight_perceptual_vol)

        # ==================== Depth loss ===================== #
        # Depth loss for omni-gs. For regularization use.
        # depth_m_gt = self.E2C(depth_m_gt.squeeze(2)).squeeze(2)
        # conf_m_gt = self.E2C(conf_m_gt.squeeze(2)).squeeze(2)
        if self.loss_args.weight_depth_abs > 0:
            depth_abs_loss = torch.abs(render_pkg_fuse["depth"] - depth_m_gt)
            depth_abs_loss = depth_abs_loss * conf_m_gt
            valid_mask = (render_pkg_fuse["depth"] > 0)
            depth_abs_loss = depth_abs_loss[valid_mask].mean()
            loss = loss + self.loss_args.weight_depth_abs * depth_abs_loss
            set_loss("depth_abs", split, depth_abs_loss, self.loss_args.weight_depth_abs)
        
        # Depth loss for pixel-gs
        # if self.loss_args.weight_depth_abs_vol > 0:
        #     depth_abs_loss = torch.abs(render_pkg_pixel["depth"] - depth_m_gt)
        #     depth_abs_loss = depth_abs_loss * conf_m_gt
        #     valid_mask = (render_pkg_pixel["depth"] > 0)
        #     depth_abs_loss = depth_abs_loss[valid_mask].mean()
        #     loss = loss + self.loss_args.weight_depth_abs_vol * depth_abs_loss
        #     set_loss("depth_abs_pixel", split, depth_abs_loss, self.loss_args.weight_depth_abs_vol)

        # Depth loss for volume-gs
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

        return loss, loss_terms, render_pkg_fuse, render_pkg_pixel, render_pkg_volume, gaussians_all, gaussians_pixel, gaussians_volume, data_dict

    # ---- forward_test: model/omni_gs_cylinder_all.py lines 601-708 @ f7b20b9 (verbatim) ----
    def forward_test(self, batch):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs, v, _, h, w = img.shape
        with self.benchmarker.time("pixel_gs"):
            img_feats = self.extract_img_feat(img=img,
                                            depths_in=data_dict["depths"], 
                                            confs_in=data_dict["confs"], 
                                            pluckers=data_dict["pluckers"],
                                            viewmats=data_dict["c2ws"],
                                            status='test'
                                            )
            # pixel-gs prediction
            gaussians = self.pixel_gs(
                    img, img_feats,
                    data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                    data_dict["rays_o"], data_dict["rays_d"], data_dict["c2ws"], status='test')
        
        gaussians_pixel = gaussians["gaussians"]
        gaussians_feat = gaussians["features"]
        tmp_gaussians_pixel = repeat(gaussians_pixel[:,None,:,:], 'b vo n c -> b (vo v) n c', v=v).contiguous()
        tmp_gaussians_pixel = rearrange(tmp_gaussians_pixel, 'b v n c -> (b v) n c').contiguous()
        tmp_gaussians_feat = repeat(gaussians_feat[:,None,:,:], 'b vo n c -> b (vo v) n c', v=v).contiguous()
        volume_gaussians_feat = rearrange(tmp_gaussians_feat, 'b v n c -> (b v) n c').contiguous()
        tmp_gaussians_points = transform_points(tmp_gaussians_pixel[..., :3], rearrange(torch.inverse(data_dict["c2ws"]), "b v h w -> (b v) h w"))
        volume_gaussians_pixel = torch.cat([tmp_gaussians_points, tmp_gaussians_pixel[..., 3:]], dim=-1)
        
        # volume_gaussians_pixel = gaussians_pixel
        # volume_gaussians_feat = gaussians_feat

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []

        cylinder_r = torch.sqrt(volume_gaussians_pixel[..., 0]**2 + volume_gaussians_pixel[..., 2]**2 + 1e-5)

        # Cylinder
        mask_pixel = (cylinder_r <= x_end) & \
                    (volume_gaussians_pixel[..., 1] >= z_start) & \
                    (volume_gaussians_pixel[..., 1] <= z_end)
        gaussians_pixel_mask = [volume_gaussians_pixel[b][mask_pixel[b]] for b in range(bs * v)]
        gaussians_feat_mask = [volume_gaussians_feat[b][mask_pixel[b]] for b in range(bs * v)]
        
        with self.benchmarker.time("volume_gs"):
            gaussians_volume = self.volume_gs(
                [repeat(img_feats['trans_features'][0], "b vo c h w -> (b v) vo c h w", v=v)],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                repeat(data_dict["imgs"], "b vo c h w -> (b v) vo c h w", v=v),
                repeat(data_dict["depths"], "b vo c h w -> (b v) vo c h w", v=v),
                data_dict["img_metas"],
                status='test'
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

        test_img = to_pil_image(render_pkg_fuse["image"][0,1].clip(min=0, max=1))    
        test_img.save('render_fuse_mp3d_all.png')
        test_img = to_pil_image(data_dict["output_imgs"][0,1].clip(min=0, max=1))    
        test_img.save('render_gt_mp3d_all.png')


        target_imgs = data_dict["output_imgs"] # b v 3 h w
        target_depths = data_dict["output_depths"] # b v h w
        target_depths_m = data_dict["output_depths_m"] # b v h w

        preds = {"img": output_imgs, "depth": output_depths, "gaussian": gaussians_all}
        gts = {"img": target_imgs, 
               "depth": target_depths, 
               "depth_m": target_depths_m,
               "depth_gt": data_dict['depth_gt'],
               "mask_gt": data_dict['mask_gt'],
               }

        return preds, gts


class LegacyOmniGaussianCylinderVolume360LocPan2(BaseModule):
    """model/omni_gs_cylinder_volume_360loc_pan2.py @ f7b20b9 (class body excerpts)."""

    # ---- __init__: model/omni_gs_cylinder_volume_360loc_pan2.py lines 42-80 @ f7b20b9 (verbatim) ----
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
        
        self.backbone = MODELS.build(backbone)
        self.pixel_gs = MODELS.build(pixel_gs)
        self.volume_gs = MODELS.build(volume_gs)
        
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

        # self.E2C = Equirec2Cube(equ_h=160, equ_w=320, cube_length=self.camera_args['resolution'][0])
        # self.C2E = Cube2Equirec(cube_length=40, equ_h=80)

    # ---- extract_img_feat: model/omni_gs_cylinder_volume_360loc_pan2.py lines 82-103 @ f7b20b9 (verbatim) ----
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
        # img_feats_reshaped = []       
        # for img_feat in img_feats:
        #     _, C, H, W = img_feat.size()
        #     # single_features_to_RGB(img_feat)
        #     img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats

    # ---- device / dtype: model/omni_gs_cylinder_volume_360loc_pan2.py lines 105-111 @ f7b20b9 (verbatim) ----
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # ---- plucker_embedder: model/omni_gs_cylinder_volume_360loc_pan2.py lines 113-121 @ f7b20b9 (verbatim) ----
    def plucker_embedder(
        self, 
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker

    # ---- get_data: model/omni_gs_cylinder_volume_360loc_pan2.py lines 123-196 @ f7b20b9 (verbatim) ----
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
        # data_dict["depths"] = batch["inputs_pix"]["depth"].to(device_id, dtype=self.dtype) * 80.0
        data_dict["confs"] = batch["inputs_pix"]["conf_m"].to(device_id, dtype=self.dtype)
        # for volume-gs
        img_metas = []
        bs, v, c, h, w = batch["inputs"]["rgb"].shape
        for w2i in batch["inputs_vol"]["w2i"]:            
            # 1. 動態獲取當前樣本的視圖數量 v
            v = w2i.shape[0]
            if v < 2: # 如果視圖少於2個，無法計算相對姿態，跳過或只用絕對姿態
                img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
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
        # data_dict["output_depths_m"] = batch["outputs"]["depth"].to(device_id, dtype=self.dtype) * 80.0
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

    # ---- forward: model/omni_gs_cylinder_volume_360loc_pan2.py lines 208-464 @ f7b20b9 (verbatim) ----
    def forward(self, batch, split="train", iter=0, iter_end=100000):
        """Forward training function.
        """
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        # test_img = to_pil_image(img[0,0].clip(min=0, max=1))    
        # test_img.save('input_img.png')

        bs, v, _, h, w = img.shape
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
                data_dict["rays_o"], data_dict["rays_d"], data_dict["c2ws"])
        
        gaussians_pixel = gaussians["gaussians"]
        gaussians_feat = gaussians["features"]

        # volume-pixel-gs prediction
        tmp_gaussians_pixel = rearrange(gaussians_pixel, "b v hw c -> b (v hw) c").unsqueeze(1).repeat(1,v,1,1).contiguous()
        tmp_gaussians_pixel = rearrange(tmp_gaussians_pixel, 'b v n c -> (b v) n c').contiguous()
        tmp_gaussians_feat = rearrange(gaussians_feat, "b v hw c -> b (v hw) c").unsqueeze(1).repeat(1,v,1,1).contiguous()
        volume_gaussians_feat = rearrange(tmp_gaussians_feat, 'b v n c -> (b v) n c').contiguous()
        tmp_gaussians_points = transform_points(tmp_gaussians_pixel[..., :3], rearrange(torch.inverse(data_dict["c2ws"]), "b v h w -> (b v) h w"))
        volume_gaussians_pixel = torch.cat([tmp_gaussians_points, tmp_gaussians_pixel[..., 3:]], dim=-1)

        gaussians_pixel = rearrange(gaussians_pixel, "b v hw c -> (b v) hw c")
        gaussians_feat = rearrange(gaussians_feat, "b v hw c -> (b v) hw c")

        # original
        # volume_gaussians_pixel = gaussians_pixel
        # volume_gaussians_feat = gaussians_feat

        # volume-gs prediction
        cylinder_r = torch.sqrt(volume_gaussians_pixel[..., 0]**2 + volume_gaussians_pixel[..., 2]**2 + 1e-5)

        # Cylinder
        mask_pixel = (cylinder_r < self.point_cloud_range[3]) & \
                    (volume_gaussians_pixel[..., 1] > self.point_cloud_range[2]) & \
                    (volume_gaussians_pixel[..., 1] < self.point_cloud_range[5])
        gaussians_pixel_mask = [volume_gaussians_pixel[b][mask_pixel[b]] for b in range(volume_gaussians_pixel.shape[0])]
        gaussians_feat_mask = [volume_gaussians_feat[b][mask_pixel[b]] for b in range(volume_gaussians_feat.shape[0])]

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

        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)

        render_c2w = data_dict["output_c2ws"]
        render_c2w = repeat(render_c2w, "b vc h w -> (b v) vc h w", v=v)
        render_fovxs = data_dict["output_fovxs"]
        render_fovxs = repeat(render_fovxs, "b vc -> (b v) vc", v=v)
        render_fovys = data_dict["output_fovys"]
        render_fovys = repeat(render_fovys, "b vc -> (b v) vc", v=v)

        # ======================== render ======================== #

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
        

        render_pkg_pixel_bev = self.renderer.render_orthographic(
            gaussians=gaussians_all,
            width=30,
            height=30, #mp3d 15 vigor 35
        )
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )
        # fuse
        tmp_pixel_img = rearrange(render_pkg_fuse["image"], "(b v) vc c h w -> b vc v c h w", b=bs, v=v) # b v vc 3 h w
        tmp_pixel_depth = rearrange(render_pkg_fuse["depth"], "(b v) vc c h w -> b vc v c h w", b=bs, v=v) # b v vc 1 h w

        target = repeat(data_dict["output_c2ws"][:, :, :3, 3], "b v d -> b v vc d", vc=data_dict["c2ws"].shape[1])
        context = repeat(data_dict["c2ws"][:, :, :3, 3], "b vc d -> b v vc d", v=data_dict["output_c2ws"].shape[1])
        dist = torch.norm(target - context, dim=-1)

        eps = 1e-8
        inv_dist = 1.0 / (dist + eps)
        weights = inv_dist / inv_dist.sum(-1, keepdim=True)

        # weights = 0.5 * torch.ones_like(weights)
        tmp_pixel_img = tmp_pixel_img * weights[..., None, None, None]
        render_pkg_fuse["image"] = tmp_pixel_img.sum(dim=2, keepdim=False) # b v 3 h w
        tmp_pixel_depth = tmp_pixel_depth * weights[..., None, None, None]
        render_pkg_fuse["depth"] = tmp_pixel_depth.sum(dim=2, keepdim=False) # b v 1 h w

        # pixel
        tmp_pixel_img = rearrange(render_pkg_pixel["image"], "(b v) vc c h w -> b vc v c h w", b=bs, v=v) # b v vc 3 h w
        tmp_pixel_depth = rearrange(render_pkg_pixel["depth"], "(b v) vc c h w -> b vc v c h w", b=bs, v=v) # b v vc 1 h w
        
        tmp_pixel_img = tmp_pixel_img * weights[..., None, None, None]
        render_pkg_pixel["image"] = tmp_pixel_img.sum(dim=2, keepdim=False) # b v 3 h w
        tmp_pixel_depth = tmp_pixel_depth * weights[..., None, None, None]
        render_pkg_pixel["depth"] = tmp_pixel_depth.sum(dim=2, keepdim=False) # b v 1 h w
        # volume
        tmp_volume_img = rearrange(render_pkg_volume["image"], "(b v) vc c h w -> b vc v c h w", b=bs, v=v) # b v vc 3 h w
        tmp_volume_depth = rearrange(render_pkg_volume["depth"], "(b v) vc c h w -> b vc v c h w", b=bs, v=v) # b v vc 1 h w
        
        tmp_volume_img = tmp_volume_img * weights[..., None, None, None]
        render_pkg_volume["image"] = tmp_volume_img.sum(dim=2, keepdim=False) # b v 3 h w
        tmp_volume_depth = tmp_volume_depth * weights[..., None, None, None]
        render_pkg_volume["depth"] = tmp_volume_depth.sum(dim=2, keepdim=False) # b v 1 h w

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
        output_cylinder_r = torch.sqrt(transformed_positions[..., 0]**2 + transformed_positions[..., 2]**2 + 1e-5)
        mask_inside = (output_cylinder_r < self.point_cloud_range[3]) & (transformed_positions[..., 1] > self.point_cloud_range[2]) & (transformed_positions[..., 1] < self.point_cloud_range[5])
        mask_dptm = mask_inside.view(bs, v, render_c2w.shape[1], h, w).any(dim=1).float()
        data_dict["mask_dptm"] = mask_dptm

        test_img = to_pil_image(render_pkg_fuse["image"][0,1].clip(min=0, max=1))    
        test_img.save('render_fuse_360Loc_all.png')
        test_img = to_pil_image(render_pkg_pixel["image"][0,1].clip(min=0, max=1))    
        test_img.save('render_pixel_360Loc_all.png')
        test_img = to_pil_image(render_pkg_volume["image"][0,1].clip(min=0, max=1))    
        test_img.save('render_volume_360Loc_all.png')
        test_img = to_pil_image(render_pkg_pixel_bev["image"][0].clip(min=0, max=1))
        test_img.save('render_bev_360Loc_all.png')
        test_img = to_pil_image(rgb_gt[0,1].clip(min=0, max=1))    
        test_img.save('render_gt_360Loc_all.png') 

        # vis rgb points
        # points_xyz = gaussians_pixel[..., :3][4].detach().cpu().numpy()
        # points_rgb = gaussians_pixel[..., 3:6][4].detach().cpu().numpy()
        # save_point_cloud(points_xyz, points_rgb, filename="point_cloud.ply")
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
        if self.loss_args.weight_recon_vol > 0 and iter < iter_end:
            # RGB loss for volume-gs
            if self.loss_args.recon_loss_vol_type == "l1":
                rec_loss_vol = torch.abs(rgb_gt - render_pkg_volume["image"])
            elif self.loss_args.recon_loss_vol_type == "l2":
                rec_loss_vol = (rgb_gt - render_pkg_volume["image"]) ** 2
            elif self.loss_args.recon_loss_vol_type == "l2_mask" or self.loss_args.recon_loss_vol_type == "l2_mask_self":
                rec_loss_vol = (rgb_gt - render_pkg_volume["image"]) ** 2
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
        if self.loss_args.weight_perceptual_vol > 0 and iter < iter_end:
            # Perceptual loss for volume-gs
            p_inp_pred_vol = maybe_resize(
                render_pkg_volume["image"].reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]),
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_inp_gt = maybe_resize(
                rgb_gt.reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]), 
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_loss_vol = self.perceptual_loss(p_inp_pred_vol, p_inp_gt)
            p_loss_vol = rearrange(p_loss_vol, "(b v) c h w -> b v c h w", b=bs)
            p_loss_vol = p_loss_vol.mean()
            loss = loss + (p_loss_vol * self.loss_args.weight_perceptual_vol)
            set_loss("perceptual_vol", split, p_loss_vol, self.loss_args.weight_perceptual_vol)

        # ==================== Depth loss ===================== #
        # Depth loss for omni-gs. For regularization use.
        # depth_m_gt = self.E2C(depth_m_gt.squeeze(2)).squeeze(2)
        # conf_m_gt = self.E2C(conf_m_gt.squeeze(2)).squeeze(2)
        if self.loss_args.weight_depth_abs > 0:
            depth_abs_loss = torch.abs(render_pkg_fuse["depth"] - depth_m_gt)
            depth_abs_loss = depth_abs_loss * conf_m_gt
            valid_mask = (render_pkg_fuse["depth"] > 0)
            depth_abs_loss = depth_abs_loss[valid_mask].mean()
            loss = loss + self.loss_args.weight_depth_abs * depth_abs_loss
            set_loss("depth_abs", split, depth_abs_loss, self.loss_args.weight_depth_abs)
        # Depth loss for volume-gs
        if self.loss_args.weight_depth_abs_vol > 0 and iter < iter_end:
            depth_abs_loss_vol = torch.abs(render_pkg_volume["depth"] - depth_m_gt)
            depth_abs_loss_vol = depth_abs_loss_vol * conf_m_gt
            depth_abs_loss_vol = depth_abs_loss_vol.mean()
            loss = loss + self.loss_args.weight_depth_abs_vol * depth_abs_loss_vol
            set_loss("depth_abs_vol", split, depth_abs_loss_vol, self.loss_args.weight_depth_abs_vol)          

        return loss, loss_terms, render_pkg_fuse, render_pkg_pixel, render_pkg_volume, render_pkg_pixel, gaussians_pixel, gaussians_volume, data_dict

    # ---- forward_test: model/omni_gs_cylinder_volume_360loc_pan2.py lines 475-584 @ f7b20b9 (verbatim) ----
    def forward_test(self, batch):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs, v, _, h, w = img.shape
        img_feats = self.extract_img_feat(img=img,
                                          depths_in=data_dict["depths"], 
                                          confs_in=data_dict["confs"], 
                                          pluckers=data_dict["pluckers"],
                                          viewmats=data_dict["c2ws"]
                                        )

        # pixel-gs prediction
        with self.benchmarker.time("pixel_gs"):
            gaussians = self.pixel_gs(
                    img, img_feats,
                    data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                    data_dict["rays_o"], data_dict["rays_d"], data_dict["c2ws"])
            
            gaussians_pixel = gaussians["gaussians"]
            gaussians_feat = gaussians["features"]

        # vis feature points
        # points_xyz = gaussians_pixel[..., :3][0].detach().cpu().numpy()
        # points_rgb = point_features_to_rgb_colormap(gaussians_feat, cmap_name='rainbow')[0]
        # save_point_cloud(points_xyz, points_rgb, filename="point_cloud.ply")

        # volume-pixel-gs prediction
        tmp_gaussians_pixel = rearrange(gaussians_pixel, "b v hw c -> b (v hw) c").unsqueeze(1).repeat(1,v,1,1).contiguous()
        tmp_gaussians_pixel = rearrange(tmp_gaussians_pixel, 'b v n c -> (b v) n c').contiguous()
        tmp_gaussians_feat = rearrange(gaussians_feat, "b v hw c -> b (v hw) c").unsqueeze(1).repeat(1,v,1,1).contiguous()
        volume_gaussians_feat = rearrange(tmp_gaussians_feat, 'b v n c -> (b v) n c').contiguous()
        tmp_gaussians_points = transform_points(tmp_gaussians_pixel[..., :3], rearrange(torch.inverse(data_dict["c2ws"]), "b v h w -> (b v) h w"))
        volume_gaussians_pixel = torch.cat([tmp_gaussians_points, tmp_gaussians_pixel[..., 3:]], dim=-1)

        gaussians_pixel = rearrange(gaussians_pixel, "b v hw c -> (b v) hw c")
        gaussians_feat = rearrange(gaussians_feat, "b v hw c -> (b v) hw c")

        # tmp_gaussians_pixel = gaussians_pixel
        cylinder_r = torch.sqrt(volume_gaussians_pixel[..., 0]**2 + volume_gaussians_pixel[..., 2]**2 + 1e-5)

        # Cylinder
        mask_pixel = (cylinder_r < self.point_cloud_range[3]) & \
                    (volume_gaussians_pixel[..., 1] > self.point_cloud_range[2]) & \
                    (volume_gaussians_pixel[..., 1] < self.point_cloud_range[5])
        gaussians_pixel_mask = [volume_gaussians_pixel[b][mask_pixel[b]] for b in range(volume_gaussians_pixel.shape[0])]
        gaussians_feat_mask = [volume_gaussians_feat[b][mask_pixel[b]] for b in range(volume_gaussians_feat.shape[0])]

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

        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)

        render_c2w = data_dict["output_c2ws"]
        render_c2w = repeat(render_c2w, "b vc h w -> (b v) vc h w", v=v)
        render_fovxs = data_dict["output_fovxs"]
        render_fovxs = repeat(render_fovxs, "b vc -> (b v) vc", v=v)
        render_fovys = data_dict["output_fovys"]
        render_fovys = repeat(render_fovys, "b vc -> (b v) vc", v=v)

        with self.benchmarker.time("render", num_calls=render_c2w.shape[1]):
            render_pkg_fuse = self.renderer.render(
                gaussians=gaussians_all,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )
            # test_img = to_pil_image(render_pkg_fuse["image"][0,1].clip(min=0, max=1))    
            # test_img.save('render_fuse_360Loc_all.png')
            # fuse
            tmp_pixel_img = rearrange(render_pkg_fuse["image"], "(b v) vc c h w -> b vc v c h w", b=bs, v=v) # b v vc 3 h w
            tmp_pixel_depth = rearrange(render_pkg_fuse["depth"], "(b v) vc c h w -> b vc v c h w", b=bs, v=v) # b v vc 1 h w

            target = repeat(data_dict["output_c2ws"][:, :, :3, 3], "b v d -> b v vc d", vc=data_dict["c2ws"].shape[1])
            context = repeat(data_dict["c2ws"][:, :, :3, 3], "b vc d -> b v vc d", v=data_dict["output_c2ws"].shape[1])
            dist = torch.norm(target - context, dim=-1)

            eps = 1e-8
            inv_dist = 1.0 / (dist + eps)
            weights = inv_dist / inv_dist.sum(-1, keepdim=True)
            # total = dist.sum(-1, keepdim=True)
            # weights = 1 - dist / total # b, v, vc
            # weights = 0.5 * torch.ones_like(weights)
            tmp_pixel_img = tmp_pixel_img * weights[..., None, None, None]
            render_pkg_fuse["image"] = tmp_pixel_img.sum(dim=2, keepdim=False) # b v 3 h w
            tmp_pixel_depth = tmp_pixel_depth * weights[..., None, None, None]
            render_pkg_fuse["depth"] = tmp_pixel_depth.sum(dim=2, keepdim=False) # b v 1 h w
            
            output_imgs = render_pkg_fuse["image"] # b v 3 h w
            output_depths = render_pkg_fuse["depth"].squeeze(2) # b v h w

            target_imgs = data_dict["output_imgs"] # b v 3 h w
            target_depths = data_dict["output_depths"]# b v 1 h w
            target_depths_m = data_dict["output_depths_m"] # b 1 v h w

        preds = {"img": output_imgs, "depth": output_depths, "gaussian": gaussians_pixel}
        gts = {"img": target_imgs, "depth": target_depths, "depth_m": target_depths_m}

        return preds, gts


class LegacyOmniGaussianCylinderVolume(BaseModule):
    """model/omni_gs_cylinder_volume.py @ f7b20b9 (class body excerpts)."""

    # ---- __init__: model/omni_gs_cylinder_volume.py lines 44-93 @ f7b20b9 (verbatim) ----
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
                 name='',
                 **kwargs,
                 ):

        super().__init__()
        self.name = name
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

    # ---- extract_img_feat: model/omni_gs_cylinder_volume.py lines 95-116 @ f7b20b9 (verbatim) ----
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
        # img_feats_reshaped = []
        # for img_feat in img_feats:
        #     _, C, H, W = img_feat.size()
        #     # single_features_to_RGB(img_feat)
        #     img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats

    # ---- device / dtype: model/omni_gs_cylinder_volume.py lines 118-124 @ f7b20b9 (verbatim) ----
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # ---- plucker_embedder: model/omni_gs_cylinder_volume.py lines 126-134 @ f7b20b9 (verbatim) ----
    def plucker_embedder(
        self, 
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker

    # ---- get_data: model/omni_gs_cylinder_volume.py lines 136-213 @ f7b20b9 (verbatim) ----
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
        # original
        # for w2i in batch["inputs_vol"]["w2i"]:    
        #     img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
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

        # for real depth
        data_dict['depth_gt'] = batch["outputs"]['depth_gt'].to(device_id, dtype=self.dtype)
        data_dict['mask_gt'] = batch["outputs"]['mask_gt'].to(device_id, dtype=torch.bool)

        data_dict["bin_token"] = 'test'

        return data_dict

    # ---- forward: model/omni_gs_cylinder_volume.py lines 225-433 @ f7b20b9 (verbatim) ----
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
                    data_dict["rays_o"], data_dict["rays_d"], data_dict["c2ws"])
            
            gaussians_pixel = gaussians["gaussians"]
            gaussians_feat = gaussians["features"]
            gaussians_pixel_raw = gaussians["gaussians_raw"]
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
            pc_range = self.dataset_params.pc_range
            x_start, y_start, z_start, x_end, y_end, z_end = pc_range
            gaussians_pixel_mask, gaussians_feat_mask = [], []

            cylinder_r = torch.sqrt(volume_gaussians_pixel[..., 0]**2 + volume_gaussians_pixel[..., 2]**2 + 1e-5)

            # Cylinder
            mask_pixel = (cylinder_r <= self.point_cloud_range[3]) & \
                        (volume_gaussians_pixel[..., 1] >= self.point_cloud_range[2]) & \
                        (volume_gaussians_pixel[..., 1] <= self.point_cloud_range[5])
            gaussians_pixel_mask = [volume_gaussians_pixel[b][mask_pixel[b]] for b in range(bs * v)]
            gaussians_feat_mask = [volume_gaussians_feat[b][mask_pixel[b]] for b in range(bs * v)]

        # single_features_to_RGB(img_feats[0].squeeze(1), img_name='input_feat.png')
        
        gaussians_volume = self.volume_gs(
            [repeat(img_feats['trans_features'][0], "b vo c h w -> (b v) vo c h w", v=v)],
            gaussians_pixel_mask,
            gaussians_feat_mask,
            repeat(data_dict["imgs"], "b vo c h w -> (b v) vo c h w", v=v),
            repeat(data_dict["depths"], "b vo c h w -> (b v) vo c h w", v=v),
            data_dict["img_metas"]
        )

        new_gaussian_points = transform_points(gaussians_volume[..., :3], rearrange(data_dict["c2ws"], "b v h w -> (b v) h w")) # [B*v, N, 3]
        gaussians_volume = torch.cat([new_gaussian_points, gaussians_volume[..., 3:]], dim=-1)
        gaussians_volume = rearrange(gaussians_volume, '(b v) n c -> b (v n) c', v=v)
        
        # original
        # gaussians_volume = self.volume_gs(
        #         [img_feats],
        #         gaussians_pixel_mask,
        #         gaussians_feat_mask,
        #         data_dict["imgs"],
        #         data_dict["depths"],
        #         data_dict["img_metas"]
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
        output_cylinder_r = torch.sqrt(transformed_positions[..., 0]**2 + transformed_positions[..., 2]**2 + 1e-5)
        mask_inside = (output_cylinder_r < x_end) & (transformed_positions[..., 1] > z_start) & (transformed_positions[..., 1] < z_end)
        mask_dptm = mask_inside.view(bs, v, render_c2w.shape[1], h, w).any(dim=1).float()
        data_dict["mask_dptm"] = mask_dptm

        # test_img = to_pil_image(mask_dptm[0])    
        # test_img.save('mask_dptm.png')

        test_img = to_pil_image(render_pkg_volume["image"][0,0].clip(min=0, max=1))    
        test_img.save(f'render_volume_mp3d_volume_{self.name}.png')
        test_img = to_pil_image(rgb_gt[0,0].clip(min=0, max=1))    
        test_img.save(f'render_gt_mp3d_volume_{self.name}.png')
        test_img = to_pil_image(render_pkg_pixel["image"][0,0].clip(min=0, max=1))    
        test_img.save(f'render_pixel_mp3d_volume_{self.name}.png')
        test_img = to_pil_image(render_pkg_pixel_bev["image"][0].clip(min=0, max=1))
        test_img.save(f'render_bev_mp3d_volume_{self.name}.png')


        # vis rgb points
        # idx = 4
        # opactity = gaussians_volume[..., 6:7]
        # opactity_mask = (opactity > 0.95).squeeze(-1)
        # gaussians_volume_save = gaussians_volume[idx][opactity_mask[idx]]
        # points_xyz = gaussians_volume_save[..., :3].detach().cpu().numpy()
        # points_rgb = gaussians_volume_save[..., 3:6].detach().cpu().numpy()
        # save_point_cloud(points_xyz, points_rgb, filename="point_cloud.ply")


        # onlyDepth(render_pkg_volume["depth"][0,0,0], save_name='render_depth_mp3d_double.png')

        if self.loss_args.weight_recon_vol > 0:
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
            p_loss_vol = self.perceptual_loss(p_inp_pred_vol * p_inp_mask_vol, p_inp_gt * p_inp_mask_vol)
            p_loss_vol = rearrange(p_loss_vol, "(b v) c h w -> b v c h w", b=bs)
            p_loss_vol = p_loss_vol.mean()
            loss = loss + (p_loss_vol * self.loss_args.weight_perceptual_vol)
            set_loss("perceptual_vol", split, p_loss_vol, self.loss_args.weight_perceptual_vol)

        # ==================== Depth loss ===================== #
        ## Depth loss for omni-gs. For regularization use.
        ## Depth loss for volume-gs
        if self.loss_args.weight_depth_abs_vol > 0:
            depth_abs_loss_vol = torch.abs(render_pkg_volume["depth"] * mask_dptm.unsqueeze(2) - depth_m_gt * mask_dptm.unsqueeze(2))
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

        return loss, loss_terms, render_pkg_volume, render_pkg_volume, render_pkg_volume, gaussians_volume, gaussians_volume, gaussians_volume, data_dict

    # ---- forward_test: model/omni_gs_cylinder_volume.py lines 444-561 @ f7b20b9 (verbatim) ----
    def forward_test(self, batch):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs, v, _, h, w = img.shape
        with self.benchmarker.time("pixel_gs"):
            img_feats = self.extract_img_feat(img=img,
                                            depths_in=data_dict["depths"], 
                                            confs_in=data_dict["confs"], 
                                            pluckers=data_dict["pluckers"],
                                            viewmats=data_dict["c2ws"],
                                            status='test'
                                            )
            # pixel-gs prediction
            gaussians = self.pixel_gs(
                    img, img_feats,
                    data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                    data_dict["rays_o"], data_dict["rays_d"], data_dict["c2ws"], status='test')
        
        gaussians_pixel = gaussians["gaussians"]
        gaussians_feat = gaussians["features"]
        # volume_pixel prediction
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
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []

        cylinder_r = torch.sqrt(volume_gaussians_pixel[..., 0]**2 + volume_gaussians_pixel[..., 2]**2 + 1e-5)

        # Cylinder
        mask_pixel = (cylinder_r <= x_end) & \
                    (volume_gaussians_pixel[..., 1] >= z_start) & \
                    (volume_gaussians_pixel[..., 1] <= z_end)
        gaussians_pixel_mask = [volume_gaussians_pixel[b][mask_pixel[b]] for b in range(bs * v)]
        gaussians_feat_mask = [volume_gaussians_feat[b][mask_pixel[b]] for b in range(bs * v)]
        
        with self.benchmarker.time("volume_gs"):
            gaussians_volume = self.volume_gs(
                [repeat(img_feats['trans_features'][0], "b vo c h w -> (b v) vo c h w", v=v)],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                repeat(data_dict["imgs"], "b vo c h w -> (b v) vo c h w", v=v),
                repeat(data_dict["depths"], "b vo c h w -> (b v) vo c h w", v=v),
                data_dict["img_metas"],
                status='test'
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

        output_positions = data_dict["output_positions"].view(bs, -1, 3)  # [B,v,h,w,xyz] -> [b,np,xyz]
        positions_expanded = output_positions.unsqueeze(1).expand(-1, v, -1, -1)
        positions_batched = positions_expanded.reshape(bs * v, -1, 3)

        transformed_positions = transform_points(positions_batched, rearrange(torch.inverse(data_dict["c2ws"]), "b v h w -> (b v) h w"))
        output_cylinder_r = torch.sqrt(transformed_positions[..., 0]**2 + transformed_positions[..., 2]**2 + 1e-5)
        mask_inside = (output_cylinder_r < x_end) & (transformed_positions[..., 1] > z_start) & (transformed_positions[..., 1] < z_end)
        mask_dptm = mask_inside.view(bs, v, render_c2w.shape[1], h, w).any(dim=1).float()
        data_dict["mask_dptm"] = mask_dptm

        output_imgs = render_pkg_fuse["image"] * mask_dptm.unsqueeze(2) # b v 3 h w
        output_depths = render_pkg_fuse["depth"].squeeze(2) # b v h w

        target_imgs = data_dict["output_imgs"] * mask_dptm.unsqueeze(2) # b v 3 h w
        target_depths = data_dict["output_depths"] # b v 1 h w
        target_depths_m = data_dict["output_depths_m"] # b v 1 h w
        
        test_img = to_pil_image(target_imgs[0,0])    
        test_img.save('render_gt_mp3d_volume_t.png')
        test_img = to_pil_image(output_imgs[0,0])    
        test_img.save('render_volume_mp3d_volume_t.png')

        preds = {"img": output_imgs, "depth": output_depths, "gaussian": gaussians_all}
        gts = { "img": target_imgs, 
                "depth": target_depths, 
                "depth_m": target_depths_m,
                "depth_gt": data_dict['depth_gt'],
                "mask_gt": data_dict['mask_gt'],
            }

        return preds, gts


class LegacyOmniGaussianCylinderPixel(BaseModule):
    """model/omni_gs_cylinder_pixel.py @ f7b20b9 (class body excerpts)."""

    # ---- __init__: model/omni_gs_cylinder_pixel.py lines 42-90 @ f7b20b9 (verbatim) ----
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
            ckpt_path = '/home/qiwei/Nips25/PanSplat/logs/wwrerdvv/checkpoints/last.ckpt'
            unimatch_pretrained_model = torch.load(ckpt_path)["state_dict"]
            updated_state_dict = OrderedDict(
                {
                    k.replace('encoder.backbone.', ''): v
                    for k, v in unimatch_pretrained_model.items()
                    if k.replace('encoder.backbone.', '') in self.backbone.state_dict() and v.shape == self.backbone.state_dict()[k.replace('encoder.backbone.', '')].shape
                }
            )
            # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
            self.backbone.load_state_dict(updated_state_dict, strict=True)
            print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)

        self.pixel_gs = MODELS.build(pixel_gs)
        self.volume_gs = MODELS.build(volume_gs)
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

    # ---- extract_img_feat: model/omni_gs_cylinder_pixel.py lines 92-114 @ f7b20b9 (verbatim) ----
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

    # ---- device / dtype: model/omni_gs_cylinder_pixel.py lines 116-122 @ f7b20b9 (verbatim) ----
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # ---- plucker_embedder: model/omni_gs_cylinder_pixel.py lines 124-132 @ f7b20b9 (verbatim) ----
    def plucker_embedder(
        self, 
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker

    # ---- get_data: model/omni_gs_cylinder_pixel.py lines 134-182 @ f7b20b9 (verbatim) ----
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

        # for real depth
        data_dict['depth_gt'] = batch["outputs"]['depth_gt'].to(device_id, dtype=self.dtype)
        data_dict['mask_gt'] = batch["outputs"]['mask_gt'].to(device_id, dtype=torch.bool)

        data_dict["bin_token"] = 'test'

        return data_dict

    # ---- forward: model/omni_gs_cylinder_pixel.py lines 195-326 @ f7b20b9 (verbatim) ----
    def forward(self, batch, split="train", iter=0, iter_end=100000):
        """Forward training function.
        """
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        # test_img = to_pil_image(img[0,0].clip(min=0, max=1))    
        # test_img.save('input_img.png')

        bs, v, _, _, _ = img.shape
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
                data_dict["rays_o"], data_dict["rays_d"], data_dict["c2ws"])

        gaussians_all = gaussians['gaussians']

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

        render_pkg_pixel_bev = self.renderer.render_orthographic(
            gaussians=gaussians_all,
            width=30,
            height=30, #mp3d 15 vigor 35
        )
        if split == "train" or split == "val":
            render_pkg_pixel = render_pkg_fuse
            render_pkg_volume = render_pkg_pixel
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
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range

        output_positions = data_dict["output_positions"]
        output_cylinder_r = torch.sqrt(output_positions[..., 0]**2 + output_positions[..., 2]**2 + 1e-5)
        mask_dptm = (output_cylinder_r < x_end) & \
                    (output_positions[..., 1] > z_start) & (output_positions[..., 1] < z_end)
        
        mask_dptm = mask_dptm.float()
        # mask_dptm = self.E2C(mask_dptm).squeeze(2)
        data_dict["mask_dptm"] = mask_dptm

        test_img = to_pil_image(render_pkg_pixel["image"][0,1].clip(min=0, max=1))    
        test_img.save('render_pred_mp3d_pixel.png')
        test_img = to_pil_image(rgb_gt[0,1].clip(min=0, max=1))    
        test_img.save('render_gt_mp3d_pixel.png')
        test_img = to_pil_image(render_pkg_pixel_bev["image"][0].clip(min=0, max=1))
        test_img.save('render_bev_mp3d_pixel.png')

        # vis rgb points
        # idx = 4
        # opactity = gaussians_pixel[..., 6:7]
        # opactity_mask = (opactity > 0.1).squeeze(-1)
        # gaussians_pixel_save = gaussians_pixel[idx][opactity_mask[idx]]
        # points_xyz = gaussians_pixel_save[..., :3].detach().cpu().numpy()
        # points_rgb = gaussians_pixel_save[..., 3:6].detach().cpu().numpy()
        # save_point_cloud(points_xyz, points_rgb, filename="point_cloud.ply")
        # onlyDepth(render_pkg_volume["depth"][0,0,0], save_name='render_depth_mp3d_double.png')
        # ======================== RGB loss ======================== #
        if self.loss_args.weight_recon > 0:
            # RGB loss for omni-gs
            if self.loss_args.recon_loss_type == "l1":
                rec_loss = torch.abs(rgb_gt - render_pkg_pixel["image"])
            elif self.loss_args.recon_loss_type == "l2":
                rec_loss = (rgb_gt - render_pkg_pixel["image"]) ** 2
            loss = loss + (rec_loss.mean() * self.loss_args.weight_recon)
            set_loss("recon", split, rec_loss.mean(), self.loss_args.weight_recon)

        # ==================== Perceptual loss ===================== #
        if self.loss_args.weight_perceptual > 0:
            # Perceptual loss for omni-gs
            ## resize images to smaller size to save memory
            p_inp_pred = maybe_resize(
                render_pkg_pixel["image"].reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]),
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

        # ==================== Depth loss ===================== #
        ## Depth loss for omni-gs. For regularization use.
        # depth_m_gt = self.E2C(depth_m_gt.squeeze(2)).squeeze(2)
        # conf_m_gt = self.E2C(conf_m_gt.squeeze(2)).squeeze(2)
        if self.loss_args.weight_depth_abs > 0:
            depth_abs_loss = torch.abs(render_pkg_pixel["depth"] - depth_m_gt)
            depth_abs_loss = depth_abs_loss * conf_m_gt
            valid_mask = (render_pkg_pixel["depth"] > 0)
            depth_abs_loss = depth_abs_loss[valid_mask].mean()
            loss = loss + self.loss_args.weight_depth_abs * depth_abs_loss
            set_loss("depth_abs", split, depth_abs_loss, self.loss_args.weight_depth_abs)    
      
        return loss, loss_terms, render_pkg_pixel, render_pkg_pixel, render_pkg_pixel, gaussians_all, gaussians_all, gaussians_all, data_dict

    # ---- forward_test: model/omni_gs_cylinder_pixel.py lines 337-390 @ f7b20b9 (verbatim) ----
    def forward_test(self, batch):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs = img.shape[0]
        render_c2w = data_dict["output_c2ws"]
        render_fovxs = data_dict["output_fovxs"]
        render_fovys = data_dict["output_fovys"]

        with self.benchmarker.time("render", num_calls=render_c2w.shape[1]):
            img_feats = self.extract_img_feat(img=img,
                                    depths_in=data_dict["depths"], 
                                    confs_in=data_dict["confs"], 
                                    pluckers=data_dict["pluckers"],
                                    viewmats=data_dict["c2ws"],
                                    status='test'
                                )

            # pixel-gs prediction
            gaussians = self.pixel_gs(
                    img, img_feats,
                    data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                    data_dict["rays_o"], data_dict["rays_d"], data_dict["c2ws"], status='test')

            gaussians_all = gaussians['gaussians']
            # vis feature points
            # points_xyz = gaussians_pixel[..., :3][0].detach().cpu().numpy()
            # points_rgb = point_features_to_rgb_colormap(gaussians_feat, cmap_name='rainbow')[0]
            # save_point_cloud(points_xyz, points_rgb, filename="point_cloud.ply")

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
        target_depths = data_dict["output_depths"]# b v 1 h w
        target_depths_m = data_dict["output_depths_m"] # b 1 v h w

        preds = {"img": output_imgs, "depth": output_depths, "gaussian": gaussians_all}
        gts = { "img": target_imgs, 
                "depth": target_depths, 
                "depth_m": target_depths_m,
                "depth_gt": data_dict['depth_gt'],
                "mask_gt": data_dict['mask_gt'],
            }

        return preds, gts
