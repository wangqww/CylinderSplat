"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

import math
import os

# The LPIPS weights ship with the repo; resolve them from the repo root, not the cwd
# (runs chdir into a scratch dir so that debug images never land in the repo).
_LPIPS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "taming/modules/autoencoder/lpips"
)
_LPIPS_FILES = {"vgg_lpips": "vgg.pth"}


def get_ckpt_path(name, root=_LPIPS_DIR):
    """Path of the shipped LPIPS weights (replaces taming.util.get_ckpt_path, which returned
    the same path when the file exists and otherwise downloaded it)."""
    path = os.path.join(root, _LPIPS_FILES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"LPIPS weights not found: {path} (they ship with the repo under taming/)")
    return path


class LossDepthTV(nn.Module):
    def __init__(self, use_second_derivative=False, sigma_image=False, near=0.1, far=1000.0):
        super().__init__()
        self.use_second_derivative = use_second_derivative
        self.sigma_image = sigma_image
        self.near = near
        self.far = far

    def forward(self, prediction):
        # Scale the depth between the near and far planes.
        near = math.log(self.near)
        far = math.log(self.far)
        depth = prediction.clamp(near, far)
        depth = (depth - near) / (far - near)

        # Compute the difference between neighboring pixels in each direction.
        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        # If desired, compute a 2nd derivative.
        if self.use_second_derivative:
            depth_dx = depth_dx.diff(dim=-1)
            depth_dy = depth_dy.diff(dim=-2)

        return depth_dx.abs().mean() + depth_dy.abs().mean()


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, _LPIPS_DIR)
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target, row_weighted=False):
        """row_weighted: replace each layer's spatial mean by a mean weighted per row with the WS-PSNR weights
        sin((row + 0.5) * pi / H_layer) (equirectangular inputs)."""
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        average = row_weighted_average if row_weighted else spatial_average
        res = [average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def row_weighted_average(x, keepdim=True):
    """Mean over H and W with the equirectangular row weights sin((row + 0.5) * pi / H), normalised."""
    height = x.shape[2]
    rows = torch.arange(height, device=x.device, dtype=x.dtype)
    w = torch.sin((rows + 0.5) * (math.pi / height)).view(1, 1, height, 1)
    return (x * w).mean([2, 3], keepdim=keepdim) / w.mean()


import torch
import torch.nn as nn


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()
