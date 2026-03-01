import torch
import torch.nn as nn
from einops import rearrange

from .unimatch.utils import split_feature, merge_splits
import torch.nn.functional as F
from .rope import RotaryEmbedding2D, apply_2d_rotary_pos_emb
from .prope import PropeDotProductAttention 

class TransformerLayer(nn.Module):
    def __init__(
        self,
        patches_x,
        patches_y,
        image_width,
        image_height,
        d_model=256,
        nhead=1,
        no_ffn=False,
        ffn_dim_expansion=4,
    ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn = PropeDotProductAttention(
            head_dim=d_model // nhead,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
        )

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        source,
        target,
        # <<< NEW >>>: 接收相機幾何參數
        viewmats_q,
        Ks_q,
        viewmats_kv,
        Ks_kv,
        is_cross_attention: bool,
        **kwargs,
    ):
        # source: [B, L, C], target: [B, L, C] 或 [B, M, L, C]
        query, key, value = source, target, target

        # 投影 Q, K, V
        # 假設 B, L, C = 批次, 序列長度, 特徵維度
        # 假設 M = 其他視角數量
        b, l, c = source.shape

        # source, target: [B, L, C] for 2-view
        # for multi-view cross-attention, source: [B, L, C], target: [B, N-1, L, C]
        query = self.q_proj(query).view(b, l, self.nhead, c // self.nhead).permute(0, 2, 1, 3) # [B, nhead, L, C/nhead]

        # 處理 K, V 的維度
        if is_cross_attention:
            m = key.shape[1]
            key = key.view(b, m, l, c)
            value = value.view(b, m, l, c)
            # 合併 B 和 M 維度以進行批次投影
            key = self.k_proj(key.view(b * m, l, c)).view(b, m, l, self.nhead, c // self.nhead).permute(0, 3, 1, 2, 4) # [B, nhead, M, L, C/nhead]
            value = self.v_proj(value.view(b * m, l, c)).view(b, m, l, self.nhead, c // self.nhead).permute(0, 3, 1, 2, 4) # [B, nhead, M, L, C/nhead]
            # 合併 M 和 L 維度以匹配 attention 輸入
            key = key.reshape(b, self.nhead, m * l, c // self.nhead)
            value = value.reshape(b, self.nhead, m * l, c // self.nhead)
        else: # self-attention
            key = self.k_proj(key).view(b, l, self.nhead, c // self.nhead).permute(0, 2, 1, 3) # [B, nhead, L, C/nhead]
            value = self.v_proj(value).view(b, l, self.nhead, c // self.nhead).permute(0, 2, 1, 3) # [B, nhead, L, C/nhead]

        # <<< MODIFIED >>>: 核心修改，使用 PrOPE Attention
        if is_cross_attention:
            # 使用 PrOPE 的靈活模式進行 cross-attention
            self.attn._precompute_and_cache_apply_fns(viewmats_q, Ks_q)
            q_proped = self.attn._apply_to_q(query)
            
            # 對每個 target view 獨立計算
            # 這裡簡化處理，假設所有 target view 共享變換，實際可擴展
            self.attn._precompute_and_cache_apply_fns(viewmats_kv, Ks_kv)
            k_proped = self.attn._apply_to_kv(key)
            v_proped = self.attn._apply_to_kv(value)
            
            message = F.scaled_dot_product_attention(q_proped, k_proped, v_proped)
            message = self.attn._apply_to_o(message)
        else:
            # 使用 PrOPE 的簡單模式進行 self-attention
            message = self.attn(query, key, value, viewmats_q, Ks_q)

        message = message.permute(0, 2, 1, 3).contiguous().view(b, l, c) # 恢復形狀
        message = self.merge(message)
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(
        self,
        **kwargs,
    ):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(no_ffn=True, **kwargs)
        self.cross_attn_ffn = TransformerLayer(**kwargs)

    def forward(self, source, target, 
                viewmats_q, Ks_q, 
                viewmats_kv, Ks_kv,
                **kwargs):
        # self attention
        source = self.self_attn(
            source, source,
            viewmats_q=viewmats_q, Ks_q=Ks_q,
            viewmats_kv=viewmats_q, Ks_kv=Ks_q, # self-attn, Q/K/V的相機都一樣
            is_cross_attention=False,
            **kwargs,
        )

        # cross attention and ffn
        source = self.cross_attn_ffn(
            source, target,
            viewmats_q=viewmats_q, Ks_q=Ks_q,
            viewmats_kv=viewmats_kv, Ks_kv=Ks_kv,
            is_cross_attention=True,
            **kwargs,
        )

        return source


def batch_features(features):
    # construct inputs to multi-view transformer in batch
    # features: list of [B, C, H, W] or [B, H*W, C]

    # query, key and value for transformer
    q = []
    kv = []

    num_views = len(features)

    for i in range(num_views):
        x = features.copy()
        q.append(x.pop(i))  # [B, C, H, W] or [B, H*W, C]

        # [B, N-1, C, H, W] or [B, N-1, H*W, C]
        kv.append(torch.stack(x, dim=1))

    q = torch.cat(q, dim=0)  # [N*B, C, H, W] or [N*B, H*W, C]
    kv = torch.cat(kv, dim=0)  # [N*B, N-1, C, H, W] or [N*B, N-1, H*W, C]

    return q, kv

def batch_cameras(viewmats, Ks):
    # viewmats: list of [B, 4, 4]
    # Ks: list of [B, 3, 3] or list of None
    q_viewmats, q_Ks = [], []
    kv_viewmats, kv_Ks = [], []
    
    num_views = len(viewmats)

    # 檢查 Ks 是否為 None
    has_intrinsics = Ks[0] is not None

    for i in range(num_views):
        v_copy = viewmats.copy()
        q_viewmats.append(v_copy.pop(i))
        kv_viewmats.append(torch.stack(v_copy, dim=1))

        if has_intrinsics:
            k_copy = Ks.copy()
            q_Ks.append(k_copy.pop(i))
            kv_Ks.append(torch.stack(k_copy, dim=1))

    # [N*B, 1, 4, 4]
    q_viewmats = torch.cat(q_viewmats, dim=0)[:,None,:,:]
    # [N*B, N-1, 4, 4]
    kv_viewmats = torch.cat(kv_viewmats, dim=0)

    if has_intrinsics:
        # [N*B, 1, 3, 3]
        q_Ks = torch.cat(q_Ks, dim=0)[:,None,:,:]
        # [N*B, N-1, 3, 3]
        kv_Ks = torch.cat(kv_Ks, dim=0)
    else:
        # 如果沒有內參，直接返回 None
        q_Ks = None
        kv_Ks = None

    return q_viewmats, q_Ks, kv_viewmats, kv_Ks


class MultiViewFeatureTransformer(nn.Module):
    def __init__(
        self,
        num_layers=1,
        d_model=128,
        nhead=4,
        image_height=None, # <<< NEW >>>
        image_width=None,  # <<< NEW >>>
        patch_size=4, # <<< NEW >>>
        **kwargs,
    ):
        super(MultiViewFeatureTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # <<< NEW >>>: 計算 patch 數量
        assert image_height is not None and image_width is not None
        # 假設 patch size 是 4x4，這需要根據您的特徵提取器來確定
        self.patches_y = image_height // patch_size
        self.patches_x = image_width // patch_size

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    patches_x=self.patches_x,
                    patches_y=self.patches_y,
                    image_width=image_width,
                    image_height=image_height,
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # zero init layers beyond 6
        if num_layers > 6:
            for i in range(6, num_layers):
                self.layers[i].self_attn.norm1.weight.data.zero_()
                self.layers[i].self_attn.norm1.bias.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.weight.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.bias.data.zero_()

    def forward(
        self,
        multi_view_features,
        multi_view_viewmats,
        multi_view_Ks, # <<< 現在可以接收一個由 None 組成的 list
        **kwargs,
    ):
        b, c, h, w = multi_view_features[0].shape
        num_views = len(multi_view_features)

        if num_views == 1:
            feature = multi_view_features[0]
            viewmat = multi_view_viewmats[0]
            K = multi_view_Ks[0] # K 現在可能是 None

            feature_seq = feature.reshape(b, c, -1).permute(0, 2, 1)
            viewmat = viewmat.unsqueeze(1)
            
            # 如果 K 不是 None，才增加維度
            if K is not None:
                K = K.unsqueeze(1)

            for layer in self.layers:
                feature_seq = layer(
                    source=feature_seq,
                    target=feature_seq.unsqueeze(1),
                    viewmats_q=viewmat,
                    Ks_q=K,
                    viewmats_kv=viewmat,
                    Ks_kv=K,
                )
            
            output_feature = feature_seq.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return [output_feature]
        else:
            # 多視圖邏輯幾乎不變，因為 batch_cameras 已經處理了 None
            q_feat, kv_feat = batch_features(multi_view_features)

            q_feat = q_feat.reshape(num_views * b, c, -1).permute(
                0, 2, 1
            )  # [N*B, H*W, C]
            kv_feat = kv_feat.reshape(num_views * b, num_views - 1, c, -1).permute(
                0, 1, 3, 2
            )  # [N*B, N-1, H*W, C]

            q_viewmats, q_Ks, kv_viewmats, kv_Ks = batch_cameras(multi_view_viewmats, multi_view_Ks)
            
            for i, layer in enumerate(self.layers):
                q_feat = layer(
                    q_feat,
                    kv_feat,
                    viewmats_q=q_viewmats,
                    Ks_q=q_Ks,
                    viewmats_kv=kv_viewmats,
                    Ks_kv=kv_Ks,
                )
                if i < len(self.layers) - 1:
                    features = list(q_feat.chunk(chunks=num_views, dim=0))
                    q_feat, kv_feat = batch_features(features)

            features = q_feat.chunk(chunks=num_views, dim=0)
            features = [
                f.view(b, h, w, c).permute(0, 3, 1, 2).contiguous() for f in features
            ]
            return features