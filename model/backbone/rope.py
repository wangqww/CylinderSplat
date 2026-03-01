import torch
import torch.nn as nn

class RotaryEmbedding2D(nn.Module):
    """
    为二维图像特征生成旋转位置编码（RoPE）。
    """
    def __init__(self, dim, temperature=10000):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("维度必须能被4整除。")
        
        half_dim = dim // 2
        self.inv_freq = 1.0 / (temperature ** (torch.arange(0, half_dim, 2).float() / half_dim))

    def forward(self, h, w, device):
        self.inv_freq = self.inv_freq.to(device)
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        
        freqs_x = x.unsqueeze(-1) * self.inv_freq
        freqs_y = y.unsqueeze(-1) * self.inv_freq

        cos_x = freqs_x.cos().unsqueeze(0).unsqueeze(3)
        sin_x = freqs_x.sin().unsqueeze(0).unsqueeze(3)
        cos_y = freqs_y.cos().unsqueeze(0).unsqueeze(3)
        sin_y = freqs_y.sin().unsqueeze(0).unsqueeze(3)

        return cos_x, sin_x, cos_y, sin_y

def apply_2d_rotary_pos_emb(x, cos_x, sin_x, cos_y, sin_y):
    """
    将2D RoPE应用到输入的Q或K张量上。
    此版本经过修正，支持任意数量的前导维度 (batch, multi-view, etc.)。

    参数:
        x (torch.Tensor): 输入张量，形状如 [..., n_heads, head_dim]
        cos_x, sin_x, cos_y, sin_y: RotaryEmbedding2D模块的输出，
                                    需要被正确 unsqueeze 以匹配 x 的维度。
    返回:
        旋转后的张量，形状与 x 相同。
    """
    # 使用 ... 省略号来表示所有不参与计算的前导维度
    d_half = x.shape[-1] // 2
    d_quarter = x.shape[-1] // 4
    
    # 将特征维度d分成4部分：x1_x, x2_x, x1_y, x2_y
    x_part = x[..., :d_half]
    y_part = x[..., d_half:]
    
    x1_x, x2_x = x_part.split(d_quarter, dim=-1)
    x1_y, x2_y = y_part.split(d_quarter, dim=-1)
    
    # 应用x轴旋转 (cos_x 和 sin_x 会自动广播到 x 的形状)
    rotated_x_part = torch.cat(
        [x1_x * cos_x - x2_x * sin_x, x1_x * sin_x + x2_x * cos_x],
        dim=-1
    )
    
    # 应用y轴旋转
    rotated_y_part = torch.cat(
        [x1_y * cos_y - x2_y * sin_y, x1_y * sin_y + x2_y * cos_y],
        dim=-1
    )
    
    # 重新组合
    return torch.cat([rotated_x_part, rotated_y_part], dim=-1)