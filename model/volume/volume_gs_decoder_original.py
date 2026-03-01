
import torch, torch.nn as nn, torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from vis_feat import single_features_to_RGB

def sigmoid_scaling(scaling:torch.Tensor, lower_bound=0.005, upper_bound=0.02):
    sig = torch.sigmoid(scaling)
    return lower_bound * (1 - sig) + upper_bound * sig

@MODELS.register_module()
class VolumeGaussianDecoderOriginal(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, pc_range, gs_dim=14,
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, gpv=4, offset_max=None, scale_max=None,
        use_checkpoint=False
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.gpv = gpv

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
        # if offset_max is None:
        #     self.offset_max = [1.0] * 3 # meters
        # else:
        #     self.offset_max = offset_max
        self.offset_max = [1.0] * 3
        #self.scale_act = lambda x: sigmoid_scaling(x, lower_bound=0.005, upper_bound=0.02)
        # if scale_max is None:
        #     self.scale_max = [1.0] * 3 # meters
        # else:
        #     self.scale_max = scale_max
        self.scale_max = [1.0] * 3 
        self.scale_act = lambda x: torch.sigmoid(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)

        # obtain anchor points for gaussians
        gs_anchors = self.get_reference_points(tpv_h * scale_h, tpv_w * scale_w, tpv_z * scale_z, pc_range) # 1, w, h, z, 3
        # gs_anchors = self.get_panorama_reference_points(tpv_h * scale_h, tpv_w * scale_w, tpv_z * scale_z, pc_range) # 1, w, h, z, 3

        self.register_buffer('gs_anchors', gs_anchors)
    
    @staticmethod
    def get_reference_points(H, W, Z, pc_range, dim='3d', bs=1, device='cuda', dtype=torch.float):
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
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, -1).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, -1, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(2, 1, 0, 3) # w, h, z, 3
        ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1, 1) # b, w, h, z, 3
        return ref_3d
    
    @staticmethod
    def get_panorama_reference_points(H, W, Z, pc_range, dim='3d', bs=1, device='cuda', dtype=torch.float):
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
        rs = (pc_range[5] - pc_range[2]) * torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(Z, H, W) / Z
        thetas = 2 * torch.pi * torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, -1).expand(Z, H, W) / W
        phis = torch.pi * torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, -1, 1).expand(Z, H, W) / H
        
        xs = -torch.sin(phis) * torch.sin(thetas) * rs
        ys = -torch.cos(phis) * rs
        zs = -torch.sin(phis) * torch.cos(thetas) * rs

        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(2, 1, 0, 3) # w, h, z, 3
        # ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        # ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        # ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1, 1) # b, w, h, z, 3
        return ref_3d
    
    def forward(self, tpv_list, debug=False):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw, 
                size=(self.tpv_h*self.scale_h, self.tpv_w*self.scale_w),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh, 
                size=(self.tpv_z*self.scale_z, self.tpv_h*self.scale_h),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz, 
                size=(self.tpv_w*self.scale_w, self.tpv_z*self.scale_z),
                mode='bilinear'
            )

        # single_features_to_RGB(tpv_hw, img_name='feat_hw.png')
        # single_features_to_RGB(tpv_zh, img_name='feat_zh.png')
        # single_features_to_RGB(tpv_wz, img_name='feat_wz.png')

        #print("before voxelize:{}".format(torch.cuda.memory_allocated(0)))
        tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
        tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
        tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1)

        gaussians = tpv_hw + tpv_zh + tpv_wz
        #print("after voxelize:{}".format(torch.cuda.memory_allocated(0)))
        gaussians = gaussians.permute(0, 2, 3, 4, 1) # bs, w, h, z, c
        bs, w, h, z, _ = gaussians.shape
        if self.use_checkpoint:
            gaussians = torch.utils.checkpoint.checkpoint(self.decoder, gaussians, use_reentrant=False)
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, gaussians, use_reentrant=False)
            gaussians = gaussians.view(bs, w, h, z, self.gpv, -1)
        else:
            gaussians = self.decoder(gaussians)
            gaussians = self.gs_decoder(gaussians)
            gaussians = gaussians.view(bs, w, h, z, self.gpv, -1)
        #print("after decode:{}".format(torch.cuda.memory_allocated(0)))
        gs_offsets_x = self.pos_act(gaussians[..., :1]) * self.offset_max[0] # bs, w, h, z, 3
        gs_offsets_y = self.pos_act(gaussians[..., 1:2]) * self.offset_max[1] # bs, w, h, z, 3
        gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.offset_max[2] # bs, w, h, z, 3
        #gs_offsets = gaussians[..., :3]
        gs_positions = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1) + self.gs_anchors[:, :, :, :, None, :]
        x = torch.cat([gs_positions, gaussians[..., 3:]], dim=-1)
        rgbs = self.rgb_act(x[..., 3:6])
        opacity = self.opacity_act(x[..., 6:7])
        rotation = self.rot_act(x[..., 7:11])
        scale_x = self.scale_act(x[..., 11:12]) * self.scale_max[0]
        scale_y = self.scale_act(x[..., 12:13]) * self.scale_max[1]
        scale_z = self.scale_act(x[..., 13:14]) * self.scale_max[2]

        if debug:
            opacity[:] = 1.0
            scale_x[:] = 0.5
            scale_y[:] = 0.5
            scale_z[:] = 0.5
            rgbs[..., 0] = 1.0
            rgbs[..., 1] = 0.0
            rgbs[..., 2] = 0.0

        gaussians = torch.cat([gs_positions, rgbs, opacity, rotation, scale_x, scale_y, scale_z], dim=-1) # bs, w, h, z, gpv, 14
    
        return gaussians
