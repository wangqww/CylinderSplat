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
from .unifuse.networks import UniFuse

from pano2cube import Equirec2Cube, Cube2Equirec, Equirec2CubeUniFuse
from vis_feat import single_features_to_RGB, reduce_gaussian_features_to_rgb, save_point_cloud, point_features_to_rgb_colormap
import torchvision.transforms as transforms
to_pil_image = transforms.ToPILImage()
import matplotlib.cm as cm
import cv2
from collections import OrderedDict
from .unifuse.networks.convert_module import erp_convert


def onlyDepth(depth, save_name):
    cmap = cm.Spectral
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().detach().numpy()
    depth = depth.astype(np.uint8)
    
    c_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(save_name, c_depth)

@MODELS.register_module()
class OmniGaussianCylinderPixelUniFuse(BaseModule):

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
            ckpt_path = 'checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth'
            unimatch_pretrained_model = torch.load(ckpt_path)["model"]
            updated_state_dict = OrderedDict(
                {
                    k: v
                    for k, v in unimatch_pretrained_model.items()
                    if k in self.backbone.state_dict() and v.shape == self.backbone.state_dict()[k].shape
                }
            )
            # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
            self.backbone.load_state_dict(updated_state_dict, strict=False)
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


        mono_depth = UniFuse(**{'num_layers': 18, 
                                'equi_h': 512, 
                                'equi_w': 1024, 
                                'pretrained': False, 
                                'max_depth': 10.0, 
                                'fusion_type': 'cee', 
                                'se_in_fusion': False, 
                                'mono_uncertainty': False, 
                                'mono_lowres_pred': False}
                            )

        mono_depth.equi_encoder = erp_convert(mono_depth.equi_encoder)
        mono_depth.equi_decoder = erp_convert(mono_depth.equi_decoder)

        ckpt_path = 'checkpoints/unifuse_pretrained.pth'
        if os.path.isfile(ckpt_path):
            print("==> Load pretrained unifuse checkpoint: %s" % ckpt_path)
            unifuse_pretrained_weight = torch.load(ckpt_path)
            replace_dict = {}
            for i, k in enumerate(mono_depth.equi_decoder.keys()):
                replace_dict[f"equi_decoder.{i}."] = f"equi_decoder.{k}."
            for i, k in enumerate(mono_depth.c2e.keys()):
                replace_dict[f"projectors.{i}."] = f"c2e.{k}."
            new_unifuse_pretrained_weight = {}
            for k, v in unifuse_pretrained_weight.items():
                for old_k, new_k in replace_dict.items():
                    k = k.replace(old_k, new_k)
                new_unifuse_pretrained_weight[k] = v
            mono_depth.load_state_dict(new_unifuse_pretrained_weight, False)
        else:
            print(f"==> No pretrained unifuse checkpoint found at {ckpt_path}")

        # There is a bug in the PanoGRF codebase, where the non erp part of the model has wrong weights
        # So we have to load the weights from the habitat monodepth model
        # The erp part of the model is not overwritten due to disconnect references after erp_convert
        ckpt_path = 'checkpoints/habitat_monodepth.pt'
        if os.path.isfile(ckpt_path):
            print("==> Load finetuned unifuse checkpoint: %s" % ckpt_path)
            habitat_monodepth_weight = torch.load(ckpt_path)["model_state_dict"]
            new_habitat_monodepth_weight = {}
            for k, v in habitat_monodepth_weight.items():
                for old_k, new_k in replace_dict.items():
                    k = k.replace(old_k, new_k)
                if 'depthconv_0' in k:
                    continue
                new_habitat_monodepth_weight[k] = v
            mono_depth.load_state_dict(new_habitat_monodepth_weight, False)
        else:
            print(f"==> No finetuned unifuse checkpoint found at {ckpt_path}")

        mono_depth.requires_grad_(False)
        mono_depth.eval()
        self.mono_depth = mono_depth
        self.e2c_mono = Equirec2CubeUniFuse(512, 1024, 256)


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
    
    def configure_optimizers(self, lr):
        # 直接将所有需要训练的参数作为一个整体传递
        # 过滤掉 requires_grad=False 的参数是一个好习惯
        all_trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        opt = torch.optim.AdamW(
            # 不再是复杂的列表，直接传入所有参数
            all_trainable_params,
            lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
        return [opt]
    
    def forward(self, batch, split="train", iter=0, iter_end=100000):
        """Forward training function.
        """
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        # test_img = to_pil_image(img[0,0].clip(min=0, max=1))    
        # test_img.save('input_img.png')

        bs, v, _, _, _ = img.shape

        mono_erp_inputs = rearrange(data_dict["output_imgs"], "b v c h w -> (b v) c h w")
        mono_erp_inputs = F.interpolate(mono_erp_inputs, size=(512, 1024), mode='bilinear', align_corners=False)
        # Project the images to the cube.
        cube_image = []
        for img_mono in mono_erp_inputs:
            img_mono = img_mono.cpu().detach().numpy()
            img_mono = rearrange(img_mono, "c h w -> h w c")
            img_mono = self.e2c_mono.run(img_mono)
            cube_image.append(img_mono)
        cube_image = np.stack(cube_image)
        mono_cube_inputs = torch.tensor(cube_image, dtype=torch.float32).to(self.device)
        mono_cube_inputs = mono_cube_inputs.permute(0, 3, 1, 2)  # (B*V, C, H, W)
        mono_depth = self.mono_depth(mono_erp_inputs, mono_cube_inputs)
        depth_uni = rearrange(mono_depth["pred_depth"], "(b v) 1 h w -> b v h w", b=bs)
        depth_uni = F.interpolate(depth_uni, size=(160, 320), mode='bilinear', align_corners=False).unsqueeze(2)
        depths_in = depth_uni[:, [0,2], ...]  # only use two views for feature extraction

        img_feats = self.extract_img_feat(img=img,
                                          depths_in=depths_in, 
                                          confs_in=data_dict["confs"], 
                                          pluckers=data_dict["pluckers"],
                                          viewmats=data_dict["c2ws"]
                                        )

        # pixel-gs prediction
        gaussians = self.pixel_gs(
                img, img_feats,
                depths_in, data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])

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
        test_img.save('render_pred_mp3d_pixel_uni.png')
        test_img = to_pil_image(rgb_gt[0,1].clip(min=0, max=1))    
        test_img.save('render_gt_mp3d_pixel_uni.png')
        test_img = to_pil_image(render_pkg_pixel_bev["image"][0].clip(min=0, max=1))
        test_img.save('render_bev_mp3d_pixel_uni.png')

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

        # extract depth
        mono_erp_inputs = rearrange(data_dict["output_imgs"], "b v c h w -> (b v) c h w")
        mono_erp_inputs = F.interpolate(mono_erp_inputs, size=(512, 1024), mode='bilinear', align_corners=False)
        # Project the images to the cube.
        cube_image = []
        for img_mono in mono_erp_inputs:
            img_mono = img_mono.cpu().detach().numpy()
            img_mono = rearrange(img_mono, "c h w -> h w c")
            img_mono = self.e2c_mono.run(img_mono)
            cube_image.append(img_mono)
        cube_image = np.stack(cube_image)
        mono_cube_inputs = torch.tensor(cube_image, dtype=torch.float32).to(self.device)
        mono_cube_inputs = mono_cube_inputs.permute(0, 3, 1, 2)  # (B*V, C, H, W)
        mono_depth = self.mono_depth(mono_erp_inputs, mono_cube_inputs)
        depth_uni = rearrange(mono_depth["pred_depth"], "(b v) 1 h w -> b v h w", b=bs)
        depth_uni = F.interpolate(depth_uni, size=(160, 320), mode='bilinear', align_corners=False).unsqueeze(2)
        depths_in = depth_uni[:, [0,2], ...]  # only use two views for feature extraction        

        render_c2w = data_dict["output_c2ws"]
        render_fovxs = data_dict["output_fovxs"]
        render_fovys = data_dict["output_fovys"]

        with self.benchmarker.time("render", num_calls=render_c2w.shape[1]):
            img_feats = self.extract_img_feat(img=img,
                                    depths_in=depths_in, 
                                    confs_in=data_dict["confs"], 
                                    pluckers=data_dict["pluckers"],
                                    viewmats=data_dict["c2ws"],
                                    status='test'
                                )

            # pixel-gs prediction
            gaussians = self.pixel_gs(
                    img, img_feats,
                    depths_in, data_dict["confs"], data_dict["pluckers"],
                    data_dict["rays_o"], data_dict["rays_d"], status='test')

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

        test_img = to_pil_image(output_imgs[0,1].clip(min=0, max=1))    
        test_img.save('render_pred_mp3d_uni.png')
        test_img = to_pil_image(data_dict["output_imgs"][0,1].clip(min=0, max=1))    
        test_img.save('render_gt_mp3d_uni.png')

        target_imgs = data_dict["output_imgs"] # b v 3 h w
        target_depths = data_dict["output_depths"]# b v 1 h w
        target_depths_m = data_dict["output_depths_m"] # b 1 v h w

        preds = {"img": output_imgs, "depth": output_depths, "gaussian": gaussians_all}
        gts = {"img": target_imgs, 
               "depth": target_depths, 
               "depth_m": target_depths_m,
               "depth_gt": data_dict['depth_gt'],
               "mask_gt": data_dict['mask_gt'],
               }

        return preds, gts

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
