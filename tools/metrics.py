from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor
from torchmetrics import PearsonCorrCoef
import numpy as np

class WSPSNR:
    """Weighted to spherical PSNR"""

    def __init__(self):
        self.weight_cache = {}

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
        weights = torch.tensor(
            self.get_weights(height, width),
            device=reconstructed.device,
            dtype=reconstructed.dtype
        )
        weights = weights.view(1, height, width, 1).expand(
            batch_size, -1, -1, channels)
        squared_error = torch.pow((reconstructed - reference), 2.0)
        wmse = torch.sum(weights * squared_error, dim=(1, 2, 3)) / torch.sum(
            weights, dim=(1, 2, 3))
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


@cache
def get_pcc(device: torch.device):
    return PearsonCorrCoef().to(device)

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
def compute_absrel(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height width"],
) -> Float[Tensor, " batch"]:
    results_absrel = []
    results_rmse = []
    for i in range(ground_truth.shape[0]):
        gt_depth = ground_truth[i]
        pred_depth = predicted[i]
        mask = gt_depth > 0
        gt_depth = gt_depth[mask]
        pred_depth = pred_depth[mask]
        gt_depth[pred_depth < 1e-3] = 1e-3
        gt_depth[pred_depth > 80] = 80
        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80] = 80
        abs_rel = torch.mean(torch.abs(gt_depth - pred_depth) / gt_depth).unsqueeze(0)
        rmse = (gt_depth - pred_depth) ** 2
        rmse = torch.sqrt(rmse.mean()).unsqueeze(0)
        if torch.isnan(abs_rel).sum() != 0 or torch.isnan(rmse).sum() != 0:
            abs_rel[:] = 0.
            rmse[:] = 0.
        results_absrel.append(abs_rel)
        results_rmse.append(rmse)
    results_absrel = torch.cat(results_absrel, dim=0)
    results_rmse = torch.cat(results_rmse, dim=0)
    return results_absrel, results_rmse

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
