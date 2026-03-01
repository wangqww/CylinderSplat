"""
Copied from diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D
Add `patch_size` and `sample_size` attributes
"""


import torch
import torch.nn as nn
from mmengine.registry import MODELS

from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import apply_freeu
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.attention_processor import Attention

from .attention_processor import KVCompressCrossViewAttnProcessor
from ..utils.typing import *

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, is_torch_version
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle


@MODELS.register_module()
class MVDownsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 8,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        # for attention processor
        kv_compress_ratio: int = 2,
        num_views: int = 4,
    ):
        super().__init__()
        resnets = []
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attn_procs = KVCompressCrossViewAttnProcessor(
                out_channels, sr_ratio=kv_compress_ratio, num_views=num_views)
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                    processor=attn_procs,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb=None, scale=scale)
            cross_attention_kwargs = {"scale": scale}
            hidden_states = attn(hidden_states, **cross_attention_kwargs)
            output_states = output_states + (hidden_states, )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale)

        return hidden_states, output_states

@MODELS.register_module()
class MVUpsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 8,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
        # for attention processor
        kv_compress_ratio: int = 2,
        num_views: int = 4,
    ):
        super().__init__()
        resnets = []
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `out_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            input_channels = res_skip_channels + resnet_in_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attn_procs = KVCompressCrossViewAttnProcessor(
                out_channels, sr_ratio=kv_compress_ratio, num_views=num_views)
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups if resnet_time_scale_shift != "spatial" else None,
                    spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                    processor=attn_procs,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(
        self, hidden_states: torch.FloatTensor,
        res_samples_tuple: tuple,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0
    ) -> torch.FloatTensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_samples = res_samples_tuple[-1]
            res_samples_tuple = res_samples_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_samples], dim=1)
            hidden_states = resnet(hidden_states, temb=temb, scale=scale)
            cross_attention_kwargs = {"scale": scale}
            hidden_states = attn(hidden_states, temb=temb, **cross_attention_kwargs)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, scale=scale)

        return hidden_states
    
@MODELS.register_module()
class MVMiddle2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 8,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        # for attention processor
        kv_compress_ratio: int = 2,
        num_views: int = 4,
    ):
        super().__init__()
        resnets = []
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attn_procs = KVCompressCrossViewAttnProcessor(
                out_channels, sr_ratio=kv_compress_ratio, num_views=num_views)
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                    processor=attn_procs,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)


    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb=None, scale=scale)
            cross_attention_kwargs = {"scale": scale}
            hidden_states = attn(hidden_states, **cross_attention_kwargs)

        return hidden_states