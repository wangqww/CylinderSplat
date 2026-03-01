import os
import math

import torch
from torch import nn

import torch.nn.functional as F
import numpy as np

renderer_type = 'vanilla' # "vanilla" or "panorama"

from diff_surfel_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

if renderer_type == 'panorama':
    from pano_gaussian import GaussianRasterizationSettings as ThreeDGaussianRasterizationSettings
    from pano_gaussian import GaussianRasterizer as ThreeDGaussianRasterizer

else:
    from diff_gaussian_rasterization import GaussianRasterizationSettings as ThreeDGaussianRasterizationSettings
    from diff_gaussian_rasterization import GaussianRasterizer as ThreeDGaussianRasterizer

from pano2cube import Equirec2Cube, Cube2Equirec
from .cam_utils import MiniCam

from .utils.ops import get_cam_info_gaussian
from .utils.typing import *


C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def covariance_from_scaling_rotation(scaling, rotation, c2ws):
    L = build_scaling_rotation(scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def depths_to_points(rays, depthmap):
    points = rays[...,:3].view(-1,3)  + depthmap.view(-1, 1) * rays[...,3:].view(-1,3)
    return points

def depth_to_normal(rays, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(rays, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

def compute_equal_aabb_with_margin(
    minima: Float[Tensor, "*#batch 3"],
    maxima: Float[Tensor, "*#batch 3"],
    margin: float = 0.1,
) -> tuple[
    Float[Tensor, "*batch 3"],  # minima of the scene
    Float[Tensor, "*batch 3"],  # maxima of the scene
]:
    midpoint = (maxima + minima) * 0.5
    span = (maxima - minima).max() * (1 + margin)
    scene_minima = midpoint - 0.5 * span
    scene_maxima = midpoint + 0.5 * span
    return scene_minima, scene_maxima

class Renderer(nn.Module):
    def __init__(self, sh_degree=3, white_background=True, radius=1):
        super(Renderer, self).__init__()
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.setup_functions()
        
        self.bg_color = torch.tensor(
            [0, 0, 0],
            dtype=torch.float32,
        )

    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def set_bg_color(self, bg):
        self.bg_color = bg
        
    def set_rasterizer(self, viewpoint_camera, scaling_modifier=1.0, device="cuda"):
        # Set up rasterization configuration

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=viewpoint_camera.tan_half_fovx,
            tanfovy=viewpoint_camera.tan_half_fovy,
            bg=self.bg_color.to(device),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        return GaussianRasterizer(raster_settings=raster_settings)


    def get_params(self, position_lr_init=0.00016,feature_lr=0.0025,opacity_lr=0.05,scaling_lr=0.005,rotation_lr=0.001):
        l = [
            {'params': [self._xyz], 'lr': position_lr_init, "name": "xyz"},
            {'params': [self._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"}
        ]
        return l


    
    def get_scaling(self, _scaling):
        return self.scaling_activation(_scaling)
    
    def get_rotation(self, _rotation):
        return self.rotation_activation(_rotation)
    
    def get_opacity(self, _opacity):
        return self.opacity_activation(_opacity)

    def get_covariance(self, _scaling, _rotation, c2ws):
        return covariance_from_scaling_rotation(self.get_scaling(_scaling), self.get_rotation(_rotation), c2ws)
    
    def render_img(
            self,
            cam,
            rays,
            centers,
            shs,
            colors_precomp,
            opacity,
            scales,
            rotations,
            device,
            cov3D_precomp=None,
            prex='',
            depth_ratio=0.0
            ):
        
        rasterizer = self.set_rasterizer(cam, device=device)

        centers = centers
        shs = shs
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                centers,
                dtype=centers.dtype,
                requires_grad=True,
                device=device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass
  
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, allmap = rasterizer(
            means3D=centers,
            means2D=screenspace_points,
            shs=shs,
            opacities=opacity,
            scales=scales[...,:2],
            rotations=rotations,
            colors_precomp=colors_precomp,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)
        if rays is None:
            return rendered_image


        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (cam.world_view_transform[:3,:3].T)).permute(2,0,1)
        
        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # get depth distortion map
        render_dist = allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        surf_depth = render_depth_expected * (1-depth_ratio) + depth_ratio * render_depth_median
        
        # generate psudo surface normal for regularizations
        
        surf_normal, surf_point = depth_to_normal(rays, surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        surf_point = surf_point.permute(2,0,1)
        # remember to add alphas
        surf_normal = surf_normal * (render_alpha).detach()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            f"image{prex}": rendered_image,
            f"depth{prex}": surf_depth,
            f"acc_map{prex}": render_alpha,
            f"rend_normal{prex}": render_normal,
            f"depth_normal{prex}": surf_normal,
            f"rend_dist{prex}": render_dist,
            # "viewspace_points": screenspace_points,
            # "visibility_filter": radii > 0,
            # "radii": radii,
        }
    



class GaussianRenderer:
    def __init__(
        self, 
        device,
        resolution: list = [512, 512],
        znear: float = 0.1,
        zfar: float = 100.0, 
        **kwargs,
    ):  
        self.renderer_type = renderer_type

        self.resolution = resolution
        self.znear = znear
        self.zfar = zfar
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        self.setup_functions()
        self.C2E = Cube2Equirec(cube_length=80, equ_h=160)
        self.gs_render = Renderer(sh_degree=0, white_background=False, radius=1)

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def render(
        self, 
        gaussians, 
        c2w,
        fovx = None,
        fovy = None,
        rays_o = None,
        rays_d = None,
        bg_color = None, 
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
        B, V = c2w.shape[:2]

        # loop of loop...
        images = []
        rend_dists = []
        rend_normals = []
        depth_normals = []
        depths = []
        acc_maps = []
        for b in range(B):

            means3D = gaussians[b, :, 0:3].contiguous().float()
            rgbs = gaussians[b, :, 3:6].contiguous().float() # [N, 3]
            opacity = gaussians[b, :, 6:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            scales = gaussians[b, :, 11:].contiguous().float()

            for v in range(V):
                fovx_ = fovx[b, v].clone()
                fovy_ = fovy[b, v].clone()
                c2w_ = c2w[b, v].clone()
                rays_d_ = rays_d[b, v].clone()
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w_, fovx=fovx_, fovy=fovy_, znear=self.znear, zfar=self.zfar
                )
                # render novel views
                tan_half_fovx = torch.tan(fovx_ * 0.5)
                tan_half_fovy = torch.tan(fovy_ * 0.5)

                if self.renderer_type == "vanilla":
                    self.gs_render.set_bg_color(torch.tensor([0., 0., 0.], device=c2w.device))
                    cam = MiniCam(
                        image_height=self.resolution[0],
                        image_width=self.resolution[1],
                        tan_half_fovx=tan_half_fovx, 
                        tan_half_fovy=tan_half_fovy, 
                        world_view_transform=w2c, 
                        full_proj_transform=proj, 
                        camera_center=cam_p, 
                        device=c2w.device
                    )

                    frame = self.gs_render.render_img(cam,
                                                      rays_d_, 
                                                      means3D, 
                                                      None, 
                                                      rgbs, 
                                                      opacity, 
                                                      scales, 
                                                      rotations, 
                                                      c2w.device
                                                      )

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

                images.append(frame['image'])
                rend_dists.append(frame['rend_dist'])
                rend_normals.append(frame['rend_normal'])
                depth_normals.append(frame['depth_normal'])
                depths.append(frame['depth'])
                acc_maps.append(frame['acc_map'])

        if self.renderer_type == "panorama":
            images = torch.stack(images, dim=0).view(B, V, 3, self.resolution[0], self.resolution[1])
            depths = torch.stack(depths, dim=0).view(B, V, 3, self.resolution[0], self.resolution[1])
            rend_dists = torch.stack(rend_dists, dim=0).view(B, V, self.resolution[0], self.resolution[1])
            rend_normals = torch.stack(rend_normals, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])
            depth_normals = torch.stack(depth_normals, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])
            acc_maps = torch.stack(acc_maps, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])
        else:
            images = self.C2E(torch.stack(images, dim=0)).view(B, -1, 3, self.resolution[0] * 2, self.resolution[1] * 4)
            depths = self.C2E(torch.stack(depths, dim=0)).view(B, -1, 1, self.resolution[0] * 2, self.resolution[1] * 4)
            rend_dists = self.C2E(torch.stack(rend_dists, dim=0)).view(B, -1, 1, self.resolution[0] * 2, self.resolution[1] * 4)
            rend_normals = self.C2E(torch.stack(rend_normals, dim=0)).view(B, -1, 3, self.resolution[0] * 2, self.resolution[1] * 4)
            depth_normals = self.C2E(torch.stack(depth_normals, dim=0)).view(B, -1, 3, self.resolution[0] * 2, self.resolution[1] * 4)
            acc_maps = self.C2E(torch.stack(acc_maps, dim=0)).view(B, -1, 1, self.resolution[0] * 2, self.resolution[1] * 4)

        return {
            "image": images, # [B, V, 3, H, W]
            "depth": depths, # [B, V, 1, H, W]
            "rend_dist": rend_dists, # [B, V, 1, H, W]
            "rend_normal": rend_normals, # [B, V, 3, H, W]
            "depth_normal": depth_normals, # [B, V, 3, H, W]
            "acc_map": acc_maps, # [B, V, 1, H, W] 
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
    
    def render_orthographic(
        self, 
        gaussians,
        bg_color = None, 
        scale_modifier: float = 1.,
        width: float = 30,
        height: float = 30,
        fov_degrees: float = 0.1,
        margin: float = 0.1,
        look_axis: int = 1,
        bev_width: int = 256,
    ):
        device = gaussians.device
        B = gaussians.shape[0]
        images = []
        alphas = []
        depths = []
        for b in range(B):

            means3D = gaussians[b, :, 0:3].contiguous().float()
            means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device=device)
            rgbs = gaussians[b, :, 3:6].contiguous().float() # [N, 3]
            opacity = gaussians[b, :, 6:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            scales = gaussians[b, :, 11:].contiguous().float()

            minima = means3D.min(dim=0).values
            maxima = means3D.max(dim=0).values
            scene_minima, scene_maxima = compute_equal_aabb_with_margin(
                minima, maxima, margin=margin / 2
            )

            right_axis = (look_axis + 1) % 3
            down_axis = (look_axis + 2) % 3

            extrinsics = torch.zeros((4, 4), dtype=torch.float32, device=device)
            extrinsics[right_axis, 0] = 1
            extrinsics[down_axis, 1] = 1
            extrinsics[look_axis, 2] = 1

            extrinsics[look_axis, 3] = scene_minima[look_axis]
            extrinsics[3, 3] = 1

            extents = scene_maxima - scene_minima
            far = extents[look_axis]
            near = torch.zeros_like(far)

            fovx = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
            tan_fov_x = (0.5 * fovx).tan()
            distance_to_near = (0.5 * width) / tan_fov_x
            tan_fov_y = 0.5 * height / distance_to_near
            fovy = (2 * tan_fov_y).atan()
            near = near + distance_to_near
            far = far + distance_to_near
            move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
            move_back[2, 3] = -distance_to_near
            extrinsics = extrinsics @ move_back

            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=extrinsics, fovx=fovx, fovy=fovy, znear=self.znear, zfar=self.zfar
            )
            # render novel views
            tan_half_fovx = torch.tan(fovx * 0.5)
            tan_half_fovy = torch.tan(fovy * 0.5)

            if self.renderer_type == "vanilla":
                raster_settings = ThreeDGaussianRasterizationSettings(
                    image_height=bev_width,
                    image_width=bev_width,
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
                rasterizer = ThreeDGaussianRasterizer(raster_settings=raster_settings)
            elif self.renderer_type == "panorama":
                raster_settings = ThreeDGaussianRasterizationSettings(
                    image_height=bev_width,
                    image_width=bev_width,
                    tanfovx=tan_half_fovx,
                    tanfovy=tan_half_fovy,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=w2c,
                    projmatrix=proj,
                    sh_degree=0,
                    campos=cam_p,
                    prefiltered=False,  # This matches the original usage.
                    debug=False,
                )
                rasterizer = ThreeDGaussianRasterizer(raster_settings=raster_settings)
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
            elif self.renderer_type == "panorama":
                rendered_image, feature_map, confidence_map, mask, rendered_depth, rendered_alpha = rasterizer(
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
            alphas.append(rendered_depth)
            depths.append(rendered_depth)

        images = torch.stack(images, dim=0).view(B, 3, bev_width, bev_width)
        alphas = torch.stack(alphas, dim=0).view(B, 1, bev_width, bev_width)
        depths = torch.stack(depths, dim=0).view(B, 1, bev_width, bev_width)

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
            "depth": depths
        }