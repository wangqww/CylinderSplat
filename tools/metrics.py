"""Image and depth metrics used by evaluate.py (see configs/eval_entries.py for the metric names)."""

from functools import cache

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor


class WSPSNR:
    """Weighted to spherical PSNR"""

    def __init__(self):
        self.weight_cache = {}
        self.tensor_cache = {}

    def get_weight_tensor(self, height, width, device, dtype):
        """get_weights(height, width) as a tensor, cached per (device, H, W, dtype).

        Built by the same torch.tensor(...) call as before, once instead of per
        batch (a host-to-device copy of an H x W array); values are identical.
        """
        key = (device, height, width, dtype)
        if key not in self.tensor_cache:
            self.tensor_cache[key] = torch.tensor(self.get_weights(height, width), device=device, dtype=dtype)
        return self.tensor_cache[key]

    def get_weights(self, height=1080, width=1920):
        """Gets cached weights.

        Args:
            height: Height.
            width: Width.

        Returns:
        Weights as H, W tensor.

        """
        key = str(height) + ";" + str(width)
        if key not in self.weight_cache:
            v = (np.arange(0, height) + 0.5) * (np.pi / height)
            v = np.sin(v).reshape(height, 1)
            v = np.broadcast_to(v, (height, width))
            self.weight_cache[key] = v.copy()
        return self.weight_cache[key]

    def calculate_wsmse(self, reconstructed, reference):
        """Calculates weighted mse for a single channel.

        Args:
            reconstructed: Image as B, H, W, C tensor.
            reference: Image as B, H, W, C tensor.

        Returns:
            wsmse
        """
        batch_size, height, width, channels = reconstructed.shape
        weights = self.get_weight_tensor(height, width, reconstructed.device, reconstructed.dtype)
        weights = weights.view(1, height, width, 1).expand(batch_size, -1, -1, channels)
        squared_error = torch.pow((reconstructed - reference), 2.0)
        wmse = torch.sum(weights * squared_error, dim=(1, 2, 3)) / torch.sum(weights, dim=(1, 2, 3))
        return wmse

    def ws_psnr(self, y_pred, y_true, max_val=1.0):
        """Weighted to spherical PSNR.

        Args:
        y_pred: First image as B, H, W, C tensor.
        y_true: Second image.
        max: Maximum value.

        Returns:
        Tensor.

        """
        wmse = self.calculate_wsmse(y_pred, y_true)
        ws_psnr = 10 * torch.log10(max_val * max_val / wmse)
        return ws_psnr


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_pcc(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height width"],
) -> Float[Tensor, " batch"]:
    b, h, w = ground_truth.shape

    # Flatten each image individually
    gt_flat = ground_truth.view(b, -1)
    pred_flat = predicted.view(b, -1)

    # Subtract the mean
    gt_centered = gt_flat - gt_flat.mean(dim=1, keepdim=True)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)

    # Compute covariance
    covariance = (gt_centered * pred_centered).mean(dim=1)

    # Compute standard deviations
    gt_std = gt_centered.std(dim=1)
    pred_std = pred_centered.std(dim=1)

    # Add a small epsilon for numerical stability
    epsilon = 1e-6

    # Calculate PCC
    pcc = covariance / (gt_std * pred_std + epsilon)
    return pcc


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


@cache
def get_ssim_kernel(device: torch.device, sigma: float = 1.5, truncate: float = 3.5) -> Tensor:
    # scipy.ndimage.gaussian_filter's 1-D kernel as used by skimage (radius 5 -> 11 taps).
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 / (sigma * sigma) * x**2)
    return (kernel / kernel.sum()).to(device)


@torch.no_grad()
def compute_ssim_gpu(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    data_range: float = 1.0,
) -> Float[Tensor, " batch"]:
    """SSIM on the input's device, for the optional `evaluate.py --fast-ssim` column.

    Same settings as compute_ssim (skimage: Gaussian window sigma 1.5 / 11 taps,
    sample covariance, K1=0.01, K2=0.03, the 5-pixel border excluded, mean over
    channels), computed in float64. It agrees with compute_ssim to rounding, not
    bitwise; compute_ssim stays the reported SSIM.
    """
    b, c, h, w = ground_truth.shape
    x = ground_truth.to(torch.float64).reshape(b * c, 1, h, w)
    y = predicted.to(torch.float64).reshape(b * c, 1, h, w)
    kernel = get_ssim_kernel(x.device)
    taps = kernel.numel()
    # Separable filter without padding: the output is exactly skimage's cropped region.
    maps = torch.cat([x, y, x * x, y * y, x * y], dim=0)
    maps = F.conv2d(maps, kernel.view(1, 1, 1, taps))
    maps = F.conv2d(maps, kernel.view(1, 1, taps, 1))
    ux, uy, uxx, uyy, uxy = maps.chunk(5, dim=0)
    cov_norm = taps * taps / (taps * taps - 1)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2))
    ssim = s.reshape(b, c, -1).mean(dim=-1).mean(dim=-1)
    return ssim.to(predicted.dtype)
