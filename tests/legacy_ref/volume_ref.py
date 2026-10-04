"""Frozen legacy copies of the cylindrical volume-branch code touched by D5/D6/D7.

Each class below is a verbatim copy from the released code at f7b20b9
(`git show f7b20b9:<path>`), with only its `@MODELS.register_module()`
decorator dropped so it does not clash with the live class. The two attention
classes are registered under `LegacyRef*` names at the end of this file, so a
legacy encoder can be built from a config whose attention `type`s use those
names. Do not edit the copied blocks.
"""

import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmengine.model import BaseModule, constant_init, xavier_init
from mmengine.registry import MODELS
from torch import Tensor
from torch.nn.init import normal_


# ---- verbatim: model/volume/cross_view_hybrid_attention.py lines 16-213 at f7b20b9 (decorator line 15 dropped) ----
class TPVCrossViewHybridAttention(BaseModule):
    """TPVFormer Cross-view Hybrid Attention Module."""

    def __init__(self,
                 tpv_h: int,
                 tpv_w: int,
                 tpv_z: int,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_points: int = 4,
                 num_anchors: int = 2,
                 init_mode: int = 0,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = 3
        self.num_points = num_points
        self.num_anchors = num_anchors
        self.init_mode = init_mode
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])
        self.output_proj = nn.ModuleList(
            [nn.Linear(embed_dims, embed_dims) for _ in range(3)])
        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * 3 * num_points * 2)
            for _ in range(3)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * 3 * (num_points + 1))
            for _ in range(3)
        ])
        self.value_proj = nn.ModuleList(
            [nn.Linear(embed_dims, embed_dims) for _ in range(3)])

        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        device = next(self.parameters()).device
        # self plane
        theta_self = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_self = torch.stack(
            [theta_self.cos(), theta_self.sin()], -1)  # H, 2
        grid_self = grid_self.view(self.num_heads, 1,
                                   2).repeat(1, self.num_points, 1)
        for j in range(self.num_points):
            grid_self[:, j, :] *= (j + 1) / 2

        if self.init_mode == 0:
            # num_phi = 4
            phi = torch.arange(
                4, dtype=torch.float32, device=device) * (2.0 * math.pi / 4)
            assert self.num_heads % 4 == 0
            num_theta = int(self.num_heads / 4)
            theta = torch.arange(
                num_theta, dtype=torch.float32, device=device) * (
                    math.pi / num_theta) + (math.pi / num_theta / 2)  # 3
            x = torch.matmul(theta.sin().unsqueeze(-1),
                             phi.cos().unsqueeze(0)).flatten()
            y = torch.matmul(theta.sin().unsqueeze(-1),
                             phi.sin().unsqueeze(0)).flatten()
            z = theta.cos().unsqueeze(-1).repeat(1, 4).flatten()
            xyz = torch.stack([x, y, z], dim=-1)  # H, 3

        elif self.init_mode == 1:

            xyz = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0],
                   [-1, 0, 0]]
            xyz = torch.tensor(xyz, dtype=torch.float32, device=device)

        grid_hw = xyz[:, [0, 1]]  # H, 2
        grid_zh = xyz[:, [2, 0]]
        grid_wz = xyz[:, [1, 2]]

        for i in range(3):
            grid = torch.stack([grid_hw, grid_zh, grid_wz], dim=1)  # H, 3, 2
            grid = grid.unsqueeze(2).repeat(1, 1, self.num_points, 1)

            grid = grid.reshape(self.num_heads, self.num_levels,
                                self.num_anchors, -1, 2)
            for j in range(self.num_points // self.num_anchors):
                grid[:, :, :, j, :] *= 2 * (j + 1)
            grid = grid.flatten(2, 3)
            grid[:, i, :, :] = grid_self

            constant_init(self.sampling_offsets[i], 0.)
            self.sampling_offsets[i].bias.data = grid.view(-1)

            constant_init(self.attention_weights[i], val=0., bias=0.)
            attn_bias = torch.zeros(
                self.num_heads, 3, self.num_points + 1, device=device)
            attn_bias[:, i, -1] = 10
            self.attention_weights[i].bias.data = attn_bias.flatten()
            xavier_init(self.value_proj[i], distribution='uniform', bias=0.)
            xavier_init(self.output_proj[i], distribution='uniform', bias=0.)

    def get_sampling_offsets_and_attention(
            self, queries: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(
                zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape

            offset = fc(query).reshape(bs, l, self.num_heads, self.num_levels,
                                       self.num_points, 2)
            offsets.append(offset)

            attention = attn(query).reshape(bs, l, self.num_heads, 3, -1)
            level_attention = attention[:, :, :, :,
                                        -1:].softmax(-2)  # bs, l, H, 3, 1
            attention = attention[:, :, :, :, :-1]
            attention = attention.softmax(-1)  # bs, l, H, 3, p
            attention = attention * level_attention
            attns.append(attention)

        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns

    def reshape_output(self, output: Tensor, lens: List[int]) -> List[Tensor]:
        outputs = torch.split(output, [lens[0], lens[1], lens[2]], dim=1)
        return outputs

    def forward(self,
                query: List[Tensor],
                identity: Optional[List[Tensor]] = None,
                query_pos: Optional[List[Tensor]] = None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None):
        identity = query if identity is None else identity
        if query_pos is not None:
            query = [q + p for q, p in zip(query, query_pos)]

        # value proj
        query_lens = [q.shape[1] for q in query]
        value = [layer(q) for layer, q in zip(self.value_proj, query)]
        value = torch.cat(value, dim=1)
        bs, num_value, _ = value.shape
        value = value.view(bs, num_value, self.num_heads, -1)

        # sampling offsets and weights
        sampling_offsets, attention_weights = \
            self.get_sampling_offsets_and_attention(query)

        if reference_points.shape[-1] == 2:
            """For each tpv query, it owns `num_Z_anchors` in 3D space that
            having different heights. After projecting, each tpv query has
            `num_Z_anchors` reference points in each 2D image. For each
            referent point, we sample `num_points` sampling points.

            For `num_Z_anchors` reference points,
            it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, _, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, :, :, None, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape  # noqa
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_Z_anchors,
                num_all_points // num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape  # noqa

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, 64)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # output = multi_scale_deformable_attn_pytorch(
        #                 value, spatial_shapes, sampling_locations, attention_weights)

        outputs = self.reshape_output(output, query_lens)

        results = []
        for out, layer, drop, residual in zip(outputs, self.output_proj,
                                              self.dropout, identity):
            results.append(residual + drop(layer(out)))

        return results


# ---- verbatim: model/volume/image_cross_attention.py lines 186-469 at f7b20b9 (decorator line 185 dropped) ----
class TPVMSDeformableAttention3D(BaseModule):
    """An attention module used in tpvFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.

    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=[8, 64, 64],
        num_z_anchors=[4, 32, 32],
        pc_range=None,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
        floor_sampling_offset=True,
        tpv_h=None,
        tpv_w=None,
        tpv_z=None,
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_z_anchors = num_z_anchors
        self.base_num_points = num_points[0]
        self.base_z_anchors = min(num_z_anchors)
        self.points_multiplier = [
            points // self.base_z_anchors for points in num_z_anchors
        ]
        self.pc_range = pc_range
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i] * 2)
            for i in range(3)
        ])
        self.floor_sampling_offset = floor_sampling_offset
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i])
            for i in range(3)
        ])
        self.value_proj = nn.Linear(embed_dims, embed_dims)

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        device = next(self.parameters()).device
        for i in range(3):
            constant_init(self.sampling_offsets[i], 0.)
            thetas = torch.arange(
                self.num_heads, dtype=torch.float32,
                device=device) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                         grid_init.abs().max(-1, keepdim=True)[0]).view(
                             self.num_heads, 1, 1,
                             2).repeat(1, self.num_levels, self.num_points[i],
                                       1)
            grid_init = grid_init.reshape(self.num_heads, self.num_levels,
                                          self.num_z_anchors[i], -1, 2)
            for j in range(self.num_points[i] // self.num_z_anchors[i]):
                grid_init[:, :, :, j, :] *= j + 1

            self.sampling_offsets[i].bias.data = grid_init.view(-1)
            constant_init(self.attention_weights[i], val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_sampling_offsets_and_attention(self, queries):
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(
                zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape

            offset = fc(query).reshape(bs, l, self.num_heads, self.num_levels,
                                       self.points_multiplier[i], -1, 2)
            offset = offset.permute(0, 1, 4, 2, 3, 5, 6).flatten(1, 2)
            offsets.append(offset)

            attention = attn(query).reshape(bs, l, self.num_heads, -1)
            attention = attention.softmax(-1)
            attention = attention.view(bs, l, self.num_heads, self.num_levels,
                                       self.points_multiplier[i], -1)
            attention = attention.permute(0, 1, 4, 2, 3, 5).flatten(1, 2)
            attns.append(attention)

        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns

    def reshape_reference_points(self, reference_points):
        reference_point_list = []
        for i, reference_point in enumerate(reference_points):
            bs, l, z_anchors, _ = reference_point.shape
            reference_point = reference_point.reshape(
                bs, l, self.points_multiplier[i], -1, 2)
            reference_point = reference_point.flatten(1, 2)
            reference_point_list.append(reference_point)
        return torch.cat(reference_point_list, dim=1)

    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        outputs = torch.split(
            output, [
                lens[0] * self.points_multiplier[0], lens[1] *
                self.points_multiplier[1], lens[2] * self.points_multiplier[2]
            ],
            dim=1)

        outputs = [
            o.reshape(bs, -1, self.points_multiplier[i], d).sum(dim=2)
            for i, o in enumerate(outputs)
        ]
        return outputs

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = [q.permute(1, 0, 2) for q in query]
            value = value.permute(1, 0, 2)

        # bs, num_query, _ = query.shape
        query_lens = [q.shape[1] for q in query]
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets, attention_weights = \
            self.get_sampling_offsets_and_attention(query)

        reference_points = self.reshape_reference_points(reference_points)

        if reference_points.shape[-1] == 2:
            """For each tpv query, it owns `num_Z_anchors` in 3D space that
            having different heights. After projecting, each tpv query has
            `num_Z_anchors` reference points in each 2D image. For each
            referent point, we sample `num_points` sampling points.

            For `num_Z_anchors` reference points,
            it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, :, None, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = \
                sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_Z_anchors,
                num_all_points // num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, \
                xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

            if self.floor_sampling_offset:
                sampling_locations = sampling_locations - torch.floor(
                    sampling_locations)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # output = multi_scale_deformable_attn_pytorch(
        #         value, spatial_shapes, sampling_locations, attention_weights)

        output = self.reshape_output(output, query_lens)
        if not self.batch_first:
            output = [o.permute(1, 0, 2) for o in output]

        return output


# ---- verbatim: model/volume/tpvformer_encoder_cylinder.py lines 15-362 at f7b20b9 (decorator line 14 dropped) ----
class TPVFormerEncoderCylinder(TransformerLayerSequence):

    def __init__(self,
                 tpv_theta=200,
                 tpv_r=200,
                 tpv_z=16,
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
        self.tpv_r = tpv_r
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]
        self.real_z = pc_range[5] - pc_range[2]

        self.level_embeds = nn.Parameter(
            torch.Tensor(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims))
        self.tpv_embedding_thetar = nn.Embedding(tpv_theta * tpv_r, embed_dims)
        self.tpv_embedding_ztheta = nn.Embedding(tpv_z * tpv_theta, embed_dims)
        self.tpv_embedding_rz = nn.Embedding(tpv_r * tpv_z, embed_dims)
        if not tpv_only:
            self.project_transform_thetar = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1)
            self.project_transform_ztheta = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1)
            self.project_transform_rz = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1)

        ref_3d_thetar = self.get_reference_points(tpv_theta, tpv_r, self.real_z,
                                              num_points_in_pillar[0])
        ref_3d_ztheta = self.get_reference_points(tpv_z, tpv_theta, self.real_w,
                                              num_points_in_pillar[1])
        ref_3d_ztheta = ref_3d_ztheta.permute(3, 0, 1, 2)[[2, 0, 1]]  # change to x,y,z
        ref_3d_ztheta = ref_3d_ztheta.permute(1, 2, 3, 0)
        ref_3d_rz = self.get_reference_points(tpv_r, tpv_z, self.real_h,
                                              num_points_in_pillar[2])
        ref_3d_rz = ref_3d_rz.permute(3, 0, 1, 2)[[1, 2, 0]]  # change to x,y,z
        ref_3d_rz = ref_3d_rz.permute(1, 2, 3, 0)
        self.register_buffer('ref_3d_thetar', ref_3d_thetar)
        self.register_buffer('ref_3d_ztheta', ref_3d_ztheta)
        self.register_buffer('ref_3d_rz', ref_3d_rz)

        cross_view_ref_points = self.get_cross_view_ref_points(
            tpv_theta, tpv_r, tpv_z, num_points_in_pillar_cross_view)
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
        normal_(self.cams_embeds)

    @staticmethod
    def get_cross_view_ref_points(tpv_theta, tpv_r, tpv_z, num_points_in_pillar):
        # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
        # generate points for hw and level 1
        theta_ranges = torch.linspace(0.5, tpv_theta - 0.5, tpv_theta) / tpv_theta
        w_ranges = torch.linspace(0.5, tpv_r - 0.5, tpv_r) / tpv_r
        theta_ranges = theta_ranges.unsqueeze(-1).expand(-1, tpv_r).flatten()
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_theta, -1).flatten()
        thetar_thetar = torch.stack([w_ranges, theta_ranges], dim=-1)  # hw, 2
        thetar_thetar = thetar_thetar.unsqueeze(1).expand(-1, num_points_in_pillar[2],
                                          -1)  # hw, #p, 2
        # generate points for hw and level 2
        z_ranges = torch.linspace(0.5, tpv_z - 0.5,
                                  num_points_in_pillar[2]) / tpv_z  # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_theta * tpv_r, -1)  # hw, #p
        theta_ranges = torch.linspace(0.5, tpv_theta - 0.5, tpv_theta) / tpv_theta
        theta_ranges = theta_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_r, num_points_in_pillar[2]).flatten(0, 1)
        thetar_ztheta = torch.stack([theta_ranges, z_ranges], dim=-1)  # hw, #p, 2
        # generate points for hw and level 3
        z_ranges = torch.linspace(0.5, tpv_z - 0.5,
                                  num_points_in_pillar[2]) / tpv_z  # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_theta * tpv_r, -1)  # hw, #p
        w_ranges = torch.linspace(0.5, tpv_r - 0.5, tpv_r) / tpv_r
        w_ranges = w_ranges.reshape(1, -1, 1).expand(
            tpv_theta, -1, num_points_in_pillar[2]).flatten(0, 1)
        thetar_rz = torch.stack([z_ranges, w_ranges], dim=-1)  # hw, #p, 2

        # generate points for zh and level 1
        w_ranges = torch.linspace(0.5, tpv_r - 0.5,
                                  num_points_in_pillar[1]) / tpv_r
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z * tpv_theta, -1)
        theta_ranges = torch.linspace(0.5, tpv_theta - 0.5, tpv_theta) / tpv_theta
        theta_ranges = theta_ranges.reshape(1, -1, 1).expand(
            tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        ztheta_thetar = torch.stack([w_ranges, theta_ranges], dim=-1)
        # generate points for zh and level 2
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_theta, num_points_in_pillar[1]).flatten(0, 1)
        theta_ranges = torch.linspace(0.5, tpv_theta - 0.5, tpv_theta) / tpv_theta
        theta_ranges = theta_ranges.reshape(1, -1, 1).expand(
            tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        ztheta_ztheta = torch.stack([theta_ranges, z_ranges], dim=-1)  # zh, #p, 2
        # generate points for zh and level 3
        w_ranges = torch.linspace(0.5, tpv_r - 0.5,
                                  num_points_in_pillar[1]) / tpv_r
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z * tpv_theta, -1)
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_theta, num_points_in_pillar[1]).flatten(0, 1)
        ztheta_rz = torch.stack([z_ranges, w_ranges], dim=-1)

        # generate points for wz and level 1
        theta_ranges = torch.linspace(0.5, tpv_theta - 0.5,
                                  num_points_in_pillar[0]) / tpv_theta
        theta_ranges = theta_ranges.unsqueeze(0).expand(tpv_r * tpv_z, -1)
        w_ranges = torch.linspace(0.5, tpv_r - 0.5, tpv_r) / tpv_r
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        rz_thetar = torch.stack([w_ranges, theta_ranges], dim=-1)
        # generate points for wz and level 2
        theta_ranges = torch.linspace(0.5, tpv_theta - 0.5,
                                  num_points_in_pillar[0]) / tpv_theta
        theta_ranges = theta_ranges.unsqueeze(0).expand(tpv_r * tpv_z, -1)
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(
            tpv_r, -1, num_points_in_pillar[0]).flatten(0, 1)
        rz_ztheta = torch.stack([theta_ranges, z_ranges], dim=-1)
        # generate points for wz and level 3
        w_ranges = torch.linspace(0.5, tpv_r - 0.5, tpv_r) / tpv_r
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(
            tpv_r, -1, num_points_in_pillar[0]).flatten(0, 1)
        rz_rz = torch.stack([z_ranges, w_ranges], dim=-1)

        reference_points = torch.cat([
            torch.stack([thetar_thetar, thetar_ztheta, thetar_rz], dim=1),
            torch.stack([ztheta_thetar, ztheta_ztheta, ztheta_rz], dim=1),
            torch.stack([rz_thetar, rz_ztheta, rz_rz], dim=1)
        ],
                                     dim=0)  # hw+zh+wz, 3, #p, 2

        return reference_points

    @staticmethod
    def get_reference_points(H,
                             W,
                             Z=8,
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
        zs = torch.linspace(
            0.5, Z - 0.5, num_points_in_pillar,
            dtype=dtype, device=device).view(-1, 1, 1).expand(
                num_points_in_pillar, H, W) / Z
        xs = torch.linspace(
            0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, -1).expand(
                num_points_in_pillar, H, W) / W
        ys = torch.linspace(
            0.5, H - 0.5, H, dtype=dtype, device=device).view(1, -1, 1).expand(
                num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    def pano_point_sampling_cylinder(self, reference_points, pc_range, img_metas, img_ori):
        B = len(img_metas)
        # init reference_points
        # ori_reference_points = reference_points.clone().permute(0,2,1,3).repeat(B, 1, 1, 1).unsqueeze(0)
        # ori_reference_points_cam = ori_reference_points[..., :2]
        
        eps = 1e-5
        tri_reference_points = reference_points.clone()
        cylinder_reference_points = torch.ones_like(tri_reference_points, device=tri_reference_points.device)
        cylinder_r = tri_reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        cylinder_reference_points[..., 0:1] = -cylinder_r * torch.sin(tri_reference_points[..., 1:2]*2*torch.pi)
        cylinder_reference_points[..., 1:2] = tri_reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        cylinder_reference_points[..., 2:3] = -cylinder_r * torch.cos(tri_reference_points[..., 1:2]*2*torch.pi)
        B_ref, D, num_query, _ = cylinder_reference_points.shape
        
        # init lidar2img
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = torch.stack(lidar2img)
        lidar2img = lidar2img[:,:,None,None,:,:].repeat(1,1,D,num_query,1,1)
        num_cams = lidar2img.shape[1]

        # get reference_points
        cylinder_reference_points = cylinder_reference_points.unsqueeze(0).repeat(B // B_ref, num_cams, 1, 1, 1)
        ones = torch.ones_like(cylinder_reference_points[..., :1], device=cylinder_reference_points.device, dtype=reference_points.dtype)
        reference_points_homogeneous = torch.cat((cylinder_reference_points, ones), dim=-1)
        P_cam_homogeneous = torch.matmul(lidar2img, reference_points_homogeneous.unsqueeze(-1))
        P_cam_homogeneous = P_cam_homogeneous.squeeze(-1)
        w_prime = P_cam_homogeneous[..., 3:]
        cylinder_reference_points = P_cam_homogeneous[..., :3] / (w_prime + eps)

        x = cylinder_reference_points[...,0:1]
        y = cylinder_reference_points[...,1:2]
        z = cylinder_reference_points[...,2:3]
        theta = (torch.atan2(x, z + eps) + torch.pi)/(2 * torch.pi)
        phi = (torch.atan2(y, torch.sqrt(x**2 + z**2 + eps)) + torch.pi/2)/torch.pi
        reference_points_cam = torch.cat((theta, phi), dim=-1).permute(1,0,3,2,4).contiguous()

        # if reference_points_cam.shape[3] == 8:
        #     for i in range(8):
        #         gap = reference_points_cam.shape[3] / 8
        #         sample_idx = int(i * gap)
        #         show_vis_points(reference_points_cam, sample_idx, background_image=img_ori[0,0], output_filename=f'vis/cylinder/feat_ztheta{sample_idx}')
        
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
        tpv_queries_thetar = self.tpv_embedding_thetar.weight.to(dtype)
        tpv_queries_ztheta = self.tpv_embedding_ztheta.weight.to(dtype)
        tpv_queries_rz = self.tpv_embedding_rz.weight.to(dtype)
        tpv_queries_thetar = tpv_queries_thetar.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_ztheta = tpv_queries_ztheta.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_rz = tpv_queries_rz.unsqueeze(0).repeat(bs, 1, 1)
        # add projected feats to tpv queries
        if project_feats[0] is not None and project_feats[1] is not None and project_feats[2] is not None:
            project_feats_thetar, project_feats_ztheta, project_feats_rz = project_feats
            project_feats_thetar = rearrange(self.project_transform_thetar(project_feats_thetar), "b c h w -> b (h w) c")
            project_feats_ztheta = rearrange(self.project_transform_ztheta(project_feats_ztheta), "b c z h -> b (z h) c")
            project_feats_rz = rearrange(self.project_transform_rz(project_feats_rz), "b c w z -> b (w z) c")
            tpv_queries_thetar = tpv_queries_thetar + project_feats_thetar
            tpv_queries_ztheta = tpv_queries_ztheta + project_feats_ztheta
            tpv_queries_rz = tpv_queries_rz + project_feats_rz

        tpv_query = [tpv_queries_thetar, tpv_queries_ztheta, tpv_queries_rz]

        tpv_pos_thetar = self.positional_encoding(bs, device, 'z')
        tpv_pos_ztheta = self.positional_encoding(bs, device, 'w')
        tpv_pos_rz = self.positional_encoding(bs, device, 'h')
        tpv_pos = [tpv_pos_thetar, tpv_pos_ztheta, tpv_pos_rz]

        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:num_cam, None, None, :].to(dtype)
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
        ref_3ds = [self.ref_3d_thetar, self.ref_3d_ztheta, self.ref_3d_rz]
        for ref_3d in ref_3ds:
            reference_points_cam, tpv_mask = self.pano_point_sampling_cylinder(
                ref_3d, self.pc_range,
                img_metas, img_ori
            )  # num_cam, bs, hw++, #p, 2
            # reference_points_cam, tpv_mask = self.pano_point_sampling(
            #     ref_3d, self.pc_range,
            #     img_metas)  # num_cam, bs, hw++, #p, 2
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
                tpv_w=self.tpv_r,
                tpv_z=self.tpv_z,
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


# ---- verbatim: model/volume/volume_gs_decoder_cylinder.py lines 44-386 at f7b20b9 (decorator line 43 dropped) ----
class VolumeGaussianDecoderCylinder(BaseModule):
    def __init__(
        self, tpv_theta, tpv_r, tpv_z, pc_range, gs_dim=14,
        in_dims=64, hidden_dims=128, out_dims=None, num_cams=6,
        scale_theta=2, scale_r=2, scale_z=2, gpv=4, offset_max=None, scale_max=None,
        use_checkpoint=False
    ):
        super().__init__()
        self.tpv_theta = tpv_theta
        self.tpv_r = tpv_r
        self.tpv_z = tpv_z
        self.pc_range = pc_range
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
        # if offset_max is None:
        #     self.offset_max = [1.0] * 3 # meters
        # else:
        #     self.offset_max = offset_max
        self.offset_max = [(pc_range[3] - pc_range[0])/tpv_r, 2*torch.pi/tpv_theta, (pc_range[5] - pc_range[2])/tpv_z] # r, theta, z
        self.scale_act = lambda x: torch.sigmoid(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)
        self.sampled_feat_length = 36 * num_cams
        self.gaussian_to_color = nn.Sequential(
            nn.Linear(self.sampled_feat_length, 128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(128, 128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(128, 3, bias=True),
            nn.Sigmoid()
        )

        # obtain anchor points for gaussians        
        # r = torch.linspace(0.5, self.tpv_z-0.5, self.tpv_z, device='cuda')
        # anchors_coordinates = sample_concentrating_sphere(r, 2000, threshold=3.0, device='cuda') # [N_radii * n_samples, 3]
        scale_cylinder = self.get_scale_cylinder(tpv_theta * scale_theta, tpv_r * scale_r, tpv_z * scale_z, pc_range[0], pc_range[3], pc_range[2], pc_range[5])
        self.register_buffer('scale_cylinder', scale_cylinder)
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

    @staticmethod
    def get_scale_cylinder(
        theta_res: int, 
        r_res: int,     # 修改: phi_res -> r_res
        z_res: int,     # 修改: 新增 z_res
        r_min: float, 
        r_max: float,
        z_min: float,   # 修改: 新增 z_min
        z_max: float    # 修改: 新增 z_max
    ) -> torch.Tensor:
        """
        为柱坐标系下均匀分布的点云计算初始的3DGS对数尺度。

        返回:
            一个形状为 [theta_res, r_res, z_res, 3] 的张量，存储了每个点的 (scale_x, scale_y, scale_z)。
        """
        # 1. 计算步长 (弧度)
        delta_r = (r_max - r_min) / r_res
        delta_z = (z_max - z_min) / z_res  # 修改: 计算 delta_z
        delta_theta = 2 * np.pi / theta_res
        # delta_phi 不再需要

        # 2. 创建柱坐标网格
        # 我们取每个格子的中心点作为高斯球的中心
        theta_vals = torch.linspace(0, 2 * np.pi, theta_res + 1)[:-1] + delta_theta / 2
        r_vals = torch.linspace(r_min, r_max, r_res + 1)[:-1] + delta_r / 2
        z_vals = torch.linspace(z_min, z_max, z_res + 1)[:-1] + delta_z / 2 # 修改: 创建 z_vals
        
        # 修改: 网格现在是 theta, r, z
        grid_r, grid_theta, grid_z = torch.meshgrid(r_vals, theta_vals, z_vals, indexing='ij')

        # 3. 为每个点计算其格子的笛卡尔尺寸 (Δx, Δy, Δz)
        cos_theta, sin_theta = torch.cos(grid_theta), torch.sin(grid_theta)
        # cos_phi, sin_phi 不再需要

        # 修改: Z方向的尺寸现在是一个常数，与位置无关
        # 因为z轴是独立的，所以格子的“高度”就是delta_z
        delta_y_values = torch.full_like(grid_z, delta_z)
        
        # 修改: X和Y方向的尺寸计算变得更简单，只和r, theta有关
        # 这描述了一个2D极坐标网格单元在x,y方向上的尺寸
        delta_z = torch.abs(cos_theta) * delta_r + torch.abs(grid_r * sin_theta) * delta_theta
                    
        delta_x = torch.abs(sin_theta) * delta_r + torch.abs(grid_r * cos_theta) * delta_theta

        # 4. 计算初始Scale (格子尺寸的一半)
        scales = torch.stack([delta_x / 2, delta_y_values / 2, delta_z / 2], dim=-1) * 1.01
        
        return scales
    
    def get_offsets_reference_points(self, 
                                     THETA, 
                                     R, 
                                     Z, 
                                     offset_T, 
                                     offset_R, 
                                     offset_Z,
                                     delta_T,
                                     delta_R,
                                     delta_Z, 
                                     pc_range, 
                                     dim='3d', 
                                     bs=1, 
                                     device='cuda', 
                                     dtype=torch.float,
                                     ):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            THETA, R: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # 1. 计算步长 (弧度)
        delta_r = (pc_range[3] - pc_range[0]) * delta_R / R
        delta_z = (pc_range[5] - pc_range[2]) * delta_Z / Z  # 修改: 计算 delta_z
        delta_theta = 2 * np.pi * delta_T / THETA

        # reference points in 3D space
        rs = (pc_range[3] - pc_range[0]) * torch.linspace(0, R, R+1, dtype=dtype,
                            device=device)[:-1].view(-1, 1, 1).expand(R, THETA, Z)[None,:,:,:,None,None] / R + offset_R 
        thetas = 2 * torch.pi * torch.linspace(0, THETA, THETA+1, dtype=dtype,
                            device=device)[:-1].view(1, -1, 1).expand(R, THETA, Z)[None,:,:,:,None,None] / THETA + offset_T
        zs = (pc_range[5] - pc_range[2]) * torch.linspace(0, Z, Z+1, dtype=dtype,
                            device=device)[:-1].view(1, 1, -1).expand(R, THETA, Z)[None,:,:,:,None,None] / Z + offset_Z
        # rs = torch.clamp(rs, min=0)
        xs = -torch.sin(thetas) * rs
        ys = zs + pc_range[2] 
        zs = -torch.cos(thetas) * rs

        # 2. 计算每个点的笛卡尔尺寸 (Δx, Δy, Δz)
        cos_theta, sin_theta = torch.cos(thetas), torch.sin(thetas)
        delta_y = delta_z
        delta_z_values = torch.abs(cos_theta) * delta_r + torch.abs(rs * sin_theta) * delta_theta                    
        delta_x_values = torch.abs(sin_theta) * delta_r + torch.abs(rs * cos_theta) * delta_theta

        ref_3d = torch.cat((xs, ys, zs), -1)
        # scale_3d = torch.cat((delta_x_values / 2, delta_y / 2, delta_z_values / 2), -1) * 1.01
        scale_3d = torch.cat((delta_x_values, delta_y, delta_z_values), -1) * 1.01
        # ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        # ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        # ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        return ref_3d, scale_3d

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
        theta = w * (torch.atan2(x, z + eps) + torch.pi)/(2 * torch.pi) # [bs, view, num_points, 1]
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
            source_depths.reshape(b*v, 1, h, w), 
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

        rgbs_sampled = F.grid_sample(source_imgs.reshape(b*v,3,h,w), 
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
        padding_needed = self.sampled_feat_length - v*local_h*local_w*4
        if padding_needed > 0:
            # F.pad 的參數格式是一個元組 (pad_left, pad_right, pad_top, pad_bottom, ...)
            # 我們只想在最後一個維度（特徵維度 N）的右邊補 0
            # 所以參數是 (0, padding_needed)
            padded_feat = F.pad(sampled_feat, (0, padding_needed), "constant", 0)
        else:
            padded_feat = sampled_feat

        color = self.gaussian_to_color(padded_feat) # [bs, num_points, 3]

        return color

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
        tpv_thetar = tpv_thetar.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.tpv_z)
        tpv_ztheta = tpv_ztheta.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.tpv_r, -1, -1)
        tpv_rz = tpv_rz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.tpv_theta, -1)

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

        gs_offsets_r = self.pos_act(gaussians[..., :1]) * self.offset_max[0] # r
        gs_offsets_theta = self.pos_act(gaussians[..., 1:2]) * self.offset_max[1] # theta
        gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.offset_max[2] # z

        scale_x = self.scale_act(gaussians[..., 3:4])
        scale_y = self.scale_act(gaussians[..., 4:5])
        scale_z = self.scale_act(gaussians[..., 5:6])

        #gs_offsets = gaussians[..., :3]
        gs_positions, scale_3d = self.get_offsets_reference_points(
            self.tpv_theta, 
            self.tpv_r, 
            self.tpv_z, 
            gs_offsets_theta, gs_offsets_r, gs_offsets_z,
            scale_x, scale_y, scale_z, 
            self.pc_range, bs=bs, device=gaussians.device
        )
        color = self.get_panorama_color(
            gs_positions.view(bs, -1, 3),
            img_color,
            img_depth,
            img_metas
        )
        rgbs = color.view(bs, w, h, z, self.gpv, 3) # bs, w, h, z, gpv, 3
        x = torch.cat([gs_positions, scale_3d, rgbs, gaussians[..., 9:]], dim=-1)
        # rgbs = self.rgb_act(x[..., 6:9])
        opacity = self.opacity_act(x[..., 9:10])
        rotation = self.rot_act(x[..., 10:14])

        gaussians = torch.cat([gs_positions, rgbs, opacity, rotation, scale_3d], dim=-1) # bs, w, h, z, gpv, 14
    
        return gaussians


# ---- registry names for building a legacy encoder from a config (not part of the copies) ----
LEGACY_HYBRID_ATTENTION = 'LegacyRefTPVCrossViewHybridAttention'
LEGACY_DEFORMABLE_ATTENTION = 'LegacyRefTPVMSDeformableAttention3D'
MODELS.register_module(name=LEGACY_HYBRID_ATTENTION, module=TPVCrossViewHybridAttention, force=True)
MODELS.register_module(name=LEGACY_DEFORMABLE_ATTENTION, module=TPVMSDeformableAttention3D, force=True)
