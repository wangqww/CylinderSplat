import torch

def equirectangular_to_xyz(width, height, device):
    """Convert equirectangular coordinates to spherical 3D coordinates in OpenCV convention and rotate 90° around Y-axis using PyTorch."""
    # 创建 theta 和 phi 为 1D 张量
    theta = torch.linspace(0, 2 * torch.pi, width, device=device)  # 方位角 [0, 2π]
    phi = torch.linspace(0, torch.pi, height, device=device)       # 仰角 [0, π]
    
    # 生成网格，调整 indexing='ij' 确保符合 PyTorch 约定
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')

    # 计算 OpenCV 形式的 X, Y, Z 坐标
    x = -torch.sin(phi) * torch.sin(theta)   # OpenCV X: 右
    y = -torch.cos(phi)                     # OpenCV Y: 下
    z = -torch.sin(phi) * torch.cos(theta)  # OpenCV Z: 前

    # 将 x, y, z 堆叠在一起，并调整维度 (height, width, 3)
    xyz = torch.stack((x, y, z), dim=-1)  # (B, V, H, W, 3)

    return xyz

xyz_coords = equirectangular_to_xyz(64, 32, 'cuda')

x = xyz_coords[..., :1]
y = xyz_coords[..., 1:2]
z = xyz_coords[..., 2:]

r = x**2 + y**2 + z**2
eps = 1e-5

theta = (torch.atan2(x, z) + torch.pi)/(2 * torch.pi)
phi = (torch.atan2(y, torch.sqrt(x**2 + z**2 + eps)) + torch.pi/2)/torch.pi


