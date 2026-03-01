from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os

import torch
from torch import Generator, nn
import torchvision.transforms as tf
from einops import repeat
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from model.utils.ops import get_panorama_ray_directions, get_rays

test_datasets = [{'name': 'm3d', 'dis': 0.1}, {'name': 'm3d', 'dis': 0.25}, {'name': 'm3d', 'dis': 0.5}, {'name': 'm3d', 'dis': 0.75}, {'name': 'm3d', 'dis': 1.0}, {'name': 'residential', 'dis': 0.15}, {'name': 'replica', 'dis': 0.5}]
roots = [Path('/data/qiwei/nips25/pano_grf')]
pano_width = 320
pano_height = 160

class DatasetMP3D(Dataset):
    def __init__(
        self,
        stage = 'train',
    ):
        super().__init__()
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder

        self.near = 0.45
        self.far = 10.0
        self.width = pano_width
        self.height = pano_height

        # scan folders in cfg.roots[0]
        if stage == "predict":
            stage = "test"

        height = 512
        height = max(height, 512)
        resolution = (height * 2, height)
        resolution = 'x'.join(map(str, resolution))
        if stage == "test":
            self.roots = []
            for test_dataset in test_datasets:
                name = test_dataset["name"]
                dis = test_dataset["dis"]
                self.roots.append(
                    roots[0] / f"png_render_{stage}_{resolution}_seq_len_3_{name}_dist_{dis}"
                )
        else:
            self.roots = [r / f"png_render_{stage}_{resolution}_seq_len_3_m3d_dist_0.5" for r in roots]

        data = []
        for root, test_dataset in zip(self.roots, test_datasets):
            if not os.path.exists(root):
                continue
            scenes =  [f for f in os.listdir(root) if "DS_Store" not in f]
            scenes.sort()
            for s in scenes:
                data.append({
                    'root': root,
                    'scene_id': s,
                    'name': test_dataset["name"],
                    'dis': test_dataset["dis"],
                    'baseline': test_dataset["dis"] * 2,
                })
        self.data = data
        self.direction = get_panorama_ray_directions(self.height, self.width)

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        scene = data['scene_id']
        scene_path = data['root'] / scene
        views = [f for f in os.listdir(scene_path) if "DS_Store" not in f]
        views.sort()

        # Load the images.
        rgbs_path = [str(scene_path / v / 'rgb.png') for v in views]
        context_indices = torch.tensor([1])
        target_indices = torch.tensor([0, 1, 2])
        context_images = [rgbs_path[i] for i in context_indices]
        target_images = [rgbs_path[i] for i in target_indices]
        context_images = self.convert_images(context_images)
        target_images = self.convert_images(target_images)

        # Load the depth.
        # relative depth path
        depths_path = [str(scene_path / v / 'depth_anywhere.png') for v in views]
        # depths_path = [str(scene_path / v / 'depth.png') for v in views]
        context_depths = [depths_path[i] for i in context_indices]
        target_depths = [depths_path[i] for i in target_indices]
        context_depths = self.convert_images(context_depths)
        target_depths = self.convert_images(target_depths)

        # metric depth path
        depths_m_path = [str(scene_path / v / 'depth_metric.npy') for v in views]
        confs_m_path = [str(scene_path / v / 'depth_conf.npy') for v in views]
        # depths_path = [str(scene_path / v / 'depth.png') for v in views]
        context_m_depths = [depths_m_path[i] for i in context_indices]
        target_m_depths = [depths_m_path[i] for i in target_indices]
        context_m_depths = self.convert_depths(context_m_depths)
        target_m_depths = self.convert_depths(target_m_depths)

        context_m_confs = [confs_m_path[i] for i in context_indices]
        target_m_confs = [confs_m_path[i] for i in target_indices]
        context_m_confs = self.convert_depths(context_m_confs)
        target_m_confs = self.convert_depths(target_m_confs)

        # context_depths = context_depths.float() / 1000
        # target_depths = target_depths.float() / 1000
        context_depths = context_depths.clamp(min=0.)
        target_depths = target_depths.clamp(min=0.)
        context_mask = (context_m_depths > self.near) & (context_m_depths < self.far)
        target_mask = (target_m_depths > self.near) & (target_m_depths < self.far)

        # load camera
        trans_path = [scene_path / v / 'tran.txt' for v in views]
        rots_path = [scene_path / v / 'rot.txt' for v in views]
        trans = []
        rots = []
        for tran_path, rot_path in zip(trans_path, rots_path):
            trans.append(np.loadtxt(tran_path))
            rots.append(np.loadtxt(rot_path))
        trans = torch.tensor(np.array(trans))
        rots = torch.tensor(np.array(rots))
        extrinsics = self.convert_poses(trans, rots)

        intrinsics = torch.eye(3, dtype=torch.float32)
        fx, fy, cx, cy = 0.25, 0.5, 0.5, 0.5
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        intrinsics = repeat(intrinsics, "h w -> b h w", b=len(extrinsics)).clone()

        input_dict = {"rgb": context_images}

        # process rays
        output_fovxs = torch.deg2rad(torch.tensor([90], dtype=torch.float32)).repeat(len(target_indices))
        output_fovys = torch.deg2rad(torch.tensor([90], dtype=torch.float32)).repeat(len(target_indices))
        input_directions = output_directions = self.direction.unsqueeze(0)
        input_rays_o, input_rays_d = get_rays(
            input_directions, extrinsics[context_indices], keepdim=True, normalize=False)
        output_rays_o, output_rays_d = get_rays(
            output_directions, extrinsics[target_indices], keepdim=True, normalize=False)
        
        input_dict_pix = {
            "depth_m": context_m_depths, 
            "conf_m": context_m_confs,
            "ck": torch.zeros(1,3,3), 
            "c2w": extrinsics[context_indices],
            "cx": torch.tensor([cx]), 
            "cy": torch.tensor([cy]), 
            "fx": torch.tensor([fx]), 
            "fy": torch.tensor([fy]),
            "rays_o": input_rays_o, 
            "rays_d": input_rays_d
        }

        input_dict_vol = {"w2i": torch.inverse(extrinsics[context_indices])}

        output_dict = {
            "rgb": target_images, 
            "depth": target_depths,
            "depth_m": target_m_depths, 
            "conf_m": target_m_confs,
            "c2w": extrinsics[target_indices], 
            "fovx": output_fovxs, 
            "fovy": output_fovys, 
            "rays_o": output_rays_o,
            "rays_d": output_rays_d, 
        }

        scene_name = str(data['name']) + "_" + str(data['dis'])
        return {
            "outputs": output_dict,
            "inputs": input_dict,
            "inputs_pix": input_dict_pix,
            "inputs_vol": input_dict_vol,
            "scene": scene_name,
        }

    def convert_poses(
        self,
        trans,
        rots,
    ):  # extrinsics
        b, _ = trans.shape

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        c2w = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        c2w[:, :3, :3] = rots
        c2w[:, :3, 3] = -trans
        # w2w = torch.tensor([  # X -> X, -Z -> Y, upY -> Z
        #     [1, 0, 0, 0],
        #     [0, 0, -1, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 0, 1],
        # ]).float()
        # c2c = torch.tensor([  # rightx -> rightx, upy -> -downy, backz -> -forwardz
        #     [1, 0, 0, 0],
        #     [0, -1, 0, 0],
        #     [0, 0, -1, 0],
        #     [0, 0, 0, 1],
        # ]).float()
        # c2w = w2w @ c2w @ c2c
        return c2w

    def convert_images(
        self,
        images,
    ):
        torch_images = []
        for image in images:
            image = Image.open(image)
            image = image.resize([self.width, self.height], Image.LANCZOS)
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    def convert_depths(
        self,
        depths,
    ):
        torch_depths = []
        for depth in depths:
            depth = np.load(depth)
            depth = torch.tensor(depth, dtype=torch.float32)
            torch_depths.append(depth)
        return F.interpolate(torch.stack(torch_depths),
                             size=(self.height, self.width), 
                             mode='bilinear', 
                             align_corners=False
                             )

    def get_bound(
        self,
        bound,
        num_views,
    ):
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self) -> int:
        return len(self.data)


def get_generator(seed):
    generator = Generator()
    generator.manual_seed(seed)
    return generator

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))

def load_MP3D_data(batch_size, stage='train'):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    MP3D = DatasetMP3D(stage = stage)

    if stage == 'train':
        seed = 1234
        shuffle = True
        persistent_workers = True
    elif stage == 'val':
        seed = 3456
        shuffle = False
        persistent_workers = True
    elif stage == 'test':
        seed = 2345
        shuffle = False
        persistent_workers = False
    else:
        seed = 6789
        shuffle = False
        persistent_workers = True

    dataloader = DataLoader(
        MP3D, 
        batch_size=batch_size,
        num_workers=32,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        shuffle=False
    )
    # val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return dataloader