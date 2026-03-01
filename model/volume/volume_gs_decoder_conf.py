
import torch, torch.nn as nn, torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from sample_anchors import sample_concentrating_sphere, project_onto_planes
import math
from vis_feat import single_features_to_RGB
from simple_knn._C import distCUDA2

def sigmoid_scaling(scaling:torch.Tensor, lower_bound=0.005, upper_bound=0.02):
    sig = torch.sigmoid(scaling)
    return lower_bound * (1 - sig) + upper_bound * sig

@MODELS.register_module()
class VolumeGaussianDecoderConf(BaseModule):
    def __init__(
        self, tpv_theta, tpv_r, tpv_z, pc_range, gs_dim=14,
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_theta=2, scale_r=2, scale_z=2, gpv=4, offset_max=None, scale_max=None,
        use_checkpoint=False
    ):
        super().__init__()
        self.tpv_theta = tpv_theta
        self.tpv_r = tpv_r
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.scale_theta = scale_theta
        self.scale_r = scale_r
        self.scale_z = scale_z
        self.gpv = gpv
        self.pc_depth = math.sqrt(pc_range[0]**2 + pc_range[1]**2 + pc_range[2]**2)
        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.gs_decoder = nn.Linear(out_dims, gs_dim*gpv)
        self.use_checkpoint = use_checkpoint

        # set activations
        # TODO check if optimal
        self.pos_act = lambda x: torch.tanh(x)
        if offset_max is None:
            self.offset_max = [1.0] * 3 # meters
        else:
            self.offset_max = offset_max
        # self.offset_max = [1.0] * 3
        #self.scale_act = lambda x: sigmoid_scaling(x, lower_bound=0.005, upper_bound=0.02)
        if scale_max is None:
            self.scale_max = [1.0] * 3 # meters
        else:
            self.scale_max = scale_max
        # self.scale_max = [1.0] * 3

        self.scale_act = lambda x: torch.sigmoid(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)
        self.gaussian_to_color = nn.Sequential(
            nn.Linear(72, 128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(128, 128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(128, 3, bias=True),
        )


        # obtain anchor points for gaussians        
        # r = torch.linspace(0.5, self.tpv_z-0.5, self.tpv_z, device='cuda')
        # anchors_coordinates = sample_concentrating_sphere(r, 2000, threshold=3.0, device='cuda') # [N_radii * n_samples, 3]
        gs_anchors = self.get_panorama_reference_points(tpv_r, tpv_theta, tpv_z, pc_range) # 1, w, h, z, 3
        # gs_anchors = self.get_sample_reference_points(anchors_coordinates, pc_range[5] - pc_range[2]) # [num_points, 1, 3]
        # mask_lower = gs_anchors[:, 0, 1] >= -2.5
        # mask_upper = gs_anchors[:, 0, 1] <= 2.5
        # combined_mask = torch.logical_and(mask_lower, mask_upper)
        # anchors_dist = torch.clamp_min(distCUDA2(gs_anchors.view(-1,3)).float().cuda(), 0.0000001)
        # self.offset_max = [anchors_dist.max()] * 3
        # self.scale_max = [anchors_dist.max()] * 3

        self.register_buffer('gs_anchors', gs_anchors)
        # self.register_buffer('anchors_coordinates', anchors_coordinates[combined_mask])

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def generate_window_grid(self, h_min, h_max, w_min, w_max, len_h, len_w, device=None):
        assert device is not None

        x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                            torch.linspace(h_min, h_max, len_h, device=device)],
                            )
        grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

        return grid


    def get_panorama_color(
            self,
            xyz: torch.Tensor,  # [bs, num_points, 3]
            source_imgs: torch.Tensor, #[bs, view, c, h, w]
            source_depths: torch.Tensor, # [bs, view, h, w]
            img_metas: list, # list of dicts, each dict contains 'lidar2img' key
            local_radius: int = 1,
    ):
        eps = 1e-5
        b,v,_,h,w = source_imgs.shape
        # init lidar2img
        source_cams = []
        for img_meta in img_metas:
            source_cams.append(img_meta["lidar2img"])
        source_cams = torch.stack(source_cams, dim=0) # [bs, view, 4, 4]

        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1

        window_grid = self.generate_window_grid(-local_radius, local_radius,
                                                -local_radius, local_radius,
                                                local_h, local_w, device=xyz.device)  # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(b, v, 1, 1)

        ones = torch.ones_like(xyz[..., :1], device=xyz.device, dtype=xyz.dtype)
        reference_points_homogeneous = torch.cat((xyz, ones), dim=-1) # [bs, num_points, 4]
        P_cam_homogeneous = torch.matmul(source_cams[:,:,None,:,:], reference_points_homogeneous[:,None,:,:,None]) # [bs, view, num_points, 4, 1]
        P_cam_homogeneous = P_cam_homogeneous.squeeze(-1) # [bs, view, num_points, 4]        
        w_prime = P_cam_homogeneous[..., 3:]
        reference_points = P_cam_homogeneous[..., :3] / (w_prime + eps) # [bs, view, num_points, 3]
        x = reference_points[...,0:1]
        y = reference_points[...,1:2]
        z = reference_points[...,2:3]

        project_depth = torch.sqrt(x**2 + y**2 + z**2 + eps).squeeze(-1) # [bs, view, num_points]
        theta = w * (torch.atan2(x, z) + torch.pi)/(2 * torch.pi) # [bs, view, num_points, 1]
        phi = h * (torch.atan2(y, torch.sqrt(x**2 + z**2 + eps)) + torch.pi/2)/torch.pi # [bs, view, num_points, 1]
        pixel_locations = torch.cat((theta, phi), dim=-1) # [bs, view, num_points, 2]
        # vis_sample_points(pixel_locations[0,0], project_depth[0,0], W=w, H=h)
        mask_in_front = (
              (pixel_locations[..., 1] > 0.0)
            & (pixel_locations[..., 1] < h)
            & (pixel_locations[..., 0] < w)
            & (pixel_locations[..., 0] > 0.0)
            & (project_depth > 0.0)
        ) # [bs, view, num_points]

        depths_sampled = F.grid_sample(
            source_depths.view(b*v, 1, h, w), 
            self.normalize(pixel_locations.view(b*v, 1, -1, 2), h, w), 
            align_corners=False
        )

        depths_sampled = depths_sampled.squeeze().view(b, v, -1) # [bs, view, num_points]
        retrived_depth = depths_sampled.masked_fill(mask_in_front==0, 0)
        projected_depth = project_depth*mask_in_front
        
        visibility_map = projected_depth - retrived_depth
        visibility_map = visibility_map.unsqueeze(-1).repeat(1, 1, 1, local_h*local_w).contiguous() # [bs, view, num_points, local_h*local_w]
        visibility_map = visibility_map.permute(0,2,1,3).unsqueeze(-1) # [bs, num_points, view, local_h*local_w, 1]

        # bradcast pixel locations and mask_in_front to match the shape of window grid
        pixel_locations = pixel_locations.unsqueeze(dim=3) + window_grid.unsqueeze(dim=2) # [bs, view, num_points, local_h*local_w, 2]
        pixel_locations = pixel_locations.view(b, v, -1, 2) # [bs, view, num_points*local_h*local_w, 2]
        normalized_pixel_locations = self.normalize(pixel_locations, h, w) # [bs, view, num_points*local_h*local_w, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(2) # [bs, view, 1, num_points*local_h*local_w, 2]
        mask_in_front = mask_in_front.unsqueeze(dim=3).repeat(1, 1, 1, local_h*local_w).contiguous() # [bs, view, num_points, local_h*local_w]
        mask_in_front = mask_in_front.view(b, v, -1) # [bs, view, num_points*local_h*local_w]

        rgbs_sampled = F.grid_sample(source_imgs.view(b*v,3,h,w), 
                                     normalized_pixel_locations.view(b*v,1,-1,2), 
                                     align_corners=False
        ) # [bs*v, 3, num_points*local_h*local_w]
        
        rgb_sampled = rgbs_sampled.view(b, v, 3, -1) # [bs, view, 3, num_points*local_h*local_w]
        rgb_sampled = rgb_sampled.permute(0, 1, 3, 2) # [bs, view, num_points*local_h*local_w, 3]
        rgb = rgb_sampled.masked_fill(mask_in_front.unsqueeze(-1)==0, 0) # [bs, view, num_points*local_h*local_w, 3]
        rgb = rgb.view(b,v,-1,local_h*local_w,3).permute(0,2,1,3,4) # [bs, num_points, view, local_h*local_w, 3]

        # cam_pos = torch.inverse(source_cams)[..., :3, 3] # [bs, view, 3]
        # ob_view = xyz.unsqueeze(1) - cam_pos.unsqueeze(2) # [bs, view, num_points, 3]
        # ob_view = ob_view.permute(0, 2, 1, 3) # [bs, num_points, view, 3]
        # ob_dist = ob_view.norm(dim=-1, keepdim=True)
        # ob_view = ob_view / ob_dist
        # ob_view = ob_view.unsqueeze(-2).repeat(1, 1, 1, local_h*local_w, 1) # [bs, num_points, view, local_h*local_w, 3]
        sampled_feat = torch.concat([rgb, visibility_map],dim=-1).view(b, -1, v*local_h*local_w*4) # [bs, num_points, view*local_h*local_w*4]
        color = self.gaussian_to_color(sampled_feat) # [bs, num_points, 3]

        return color
    

    @staticmethod
    def get_panorama_reference_points(R, Theta, Phi, pc_range, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            H, W: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space
        rs = (pc_range[3] - pc_range[0]) * torch.linspace(0, R, R+1, dtype=dtype,
                            device=device)[:-1].view(-1, 1, 1).expand(R, Theta, Phi) / R  + pc_range[0]
        thetas = 2 * torch.pi * torch.linspace(0, Theta, Theta+1, dtype=dtype,
                            device=device)[:-1].view(1, -1, 1).expand(R, Theta, Phi) / Theta
        phis = torch.pi * torch.linspace(0, Phi, Phi+1, dtype=dtype,
                            device=device)[:-1].view(1, 1, -1).expand(R, Theta, Phi) / Phi
        
        xs = -torch.sin(phis) * torch.sin(thetas) * rs
        ys = -torch.cos(phis) * rs
        zs = -torch.sin(phis) * torch.cos(thetas) * rs

        ref_3d = torch.stack((xs, ys, zs), -1)
        # ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        # ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        # ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1, 1) # b, w, h, z, 3
        return ref_3d
    
    @staticmethod
    def get_sample_reference_points(anchors_coordinates, range_r, dtype=torch.float):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            H, W: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space
        thetas = 2 * torch.pi * (anchors_coordinates[..., 0:1] + 1.0) / 2
        phis = torch.pi * (anchors_coordinates[..., 1:2] + 1.0) / 2
        rs = range_r * (anchors_coordinates[..., 2:3] + 1.0) / 2
        xs = -torch.sin(phis) * torch.sin(thetas) * rs
        ys = -torch.cos(phis) * rs
        zs = -torch.sin(phis) * torch.cos(thetas) * rs

        ref_3d = torch.stack((xs, ys, zs), -1) # [num_points, 1, 3]

        return ref_3d
    
    def forward(self, tpv_list, img_color, img_depth, img_metas, debug=False):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_thetar, tpv_ztheta, tpv_rz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_thetar.shape

        tpv_thetar = tpv_thetar.permute(0, 2, 1).reshape(bs, c, self.tpv_theta, self.tpv_r) # [theta, phi]
        tpv_ztheta = tpv_ztheta.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_theta) # [phi, r]
        tpv_rz = tpv_rz.permute(0, 2, 1).reshape(bs, c, self.tpv_r, self.tpv_z) # [r, theta]

        # #print("before voxelize:{}".format(torch.cuda.memory_allocated(0)))
        tpv_thetar = tpv_thetar.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
        tpv_ztheta = tpv_ztheta.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_r*self.tpv_r, -1, -1)
        tpv_rz = tpv_rz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_theta*self.tpv_theta, -1)

        gaussians = tpv_thetar + tpv_ztheta + tpv_rz
        #print("after voxelize:{}".format(torch.cuda.memory_allocated(0)))
        gaussians = gaussians.permute(0, 2, 3, 4, 1) # bs, w, h, z, c
        bs, w, h, z, _ = gaussians.shape

        if self.use_checkpoint:
            gaussians = torch.utils.checkpoint.checkpoint(self.decoder, gaussians, use_reentrant=False)
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, gaussians, use_reentrant=False)
            # gaussians = gaussians.view(bs, num_points, self.gpv, -1)
            gaussians = gaussians.view(bs, w, h, z, self.gpv, -1)
        else:
            gaussians = self.decoder(gaussians)
            gaussians = self.gs_decoder(gaussians)
            # gaussians = gaussians.view(bs, num_points, self.gpv, -1)
            gaussians = gaussians.view(bs, w, h, z, self.gpv, -1)
        #print("after decode:{}".format(torch.cuda.memory_allocated(0)))
        gs_offsets_x = self.pos_act(gaussians[..., :1]) * self.offset_max[0] # bs, w, h, z, 3
        gs_offsets_y = self.pos_act(gaussians[..., 1:2]) * self.offset_max[1] # bs, w, h, z, 3
        gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.offset_max[2] # bs, w, h, z, 3
        #gs_offsets = gaussians[..., :3]
        gs_positions = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1) + self.gs_anchors[:, :, :, :, None, :]
        color = self.get_panorama_color(
            gs_positions.view(bs, -1, 3),
            img_color,
            img_depth,
            img_metas
        )
        rgbs = color.view(bs, w, h, z, self.gpv, 3) # bs, w, h, z, gpv, 3
        x = torch.cat([gs_positions, rgbs, gaussians[..., 6:]], dim=-1)
        opacity = self.opacity_act(x[..., 6:7])
        rotation = self.rot_act(x[..., 7:11])
        scale_x = self.scale_act(x[..., 11:12]) * self.scale_max[0]
        scale_y = self.scale_act(x[..., 12:13]) * self.scale_max[1]
        scale_z = self.scale_act(x[..., 13:14]) * self.scale_max[2]

        gaussians = torch.cat([gs_positions, rgbs, opacity, rotation, scale_x, scale_y, scale_z], dim=-1) # bs, w, h, z, gpv, 14
        return gaussians
