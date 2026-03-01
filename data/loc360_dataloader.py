from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

import random
from torch import Generator, nn

from einops import rearrange
import torch.nn.functional as F
import json
from functools import cached_property
from model.utils.ops import get_panorama_ray_directions, get_rays
import matplotlib.pyplot as plt

pano_width = 320
pano_height = 160

w2w = torch.tensor([  #  X -> X, Z -> Y, Y -> -Z
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
]).float()

def two_sample(scene, extrinsics, stage="train", i=0):
    num_views, _, _ = extrinsics.shape
    # Compute the context view spacing based on the current global step.
    if stage == "val":
        # When testing, always use the full gap.
        min_gap = max_gap = 3
    else:
        min_gap = max_gap = 2
    max_gap = min(num_views - 1, min_gap)

    # Pick the gap between the context views.
    # NOTE: we keep the bug untouched to follow initial pixelsplat cfgs

    context_gap = torch.randint(
        min_gap,
        max_gap + 1,
        size=tuple(),
        device='cpu',
    ).item()

    # Pick the left and right context indices.

    if stage == "val":
        index_context_left = (num_views - context_gap - 1) * i / max((100 - 1), 1)
        index_context_left = int(index_context_left)
    else:
        index_context_left = torch.randint(
            num_views - context_gap,
            size=tuple(),
            device='cpu',
        ).item()
    index_context_right = index_context_left + context_gap

    # Pick the target view indices.
    if stage == "val":
        # When testing, pick all.
        index_target = torch.arange(
            index_context_left,
            index_context_right + 1,
            device='cpu',
        )
    else:
        # When training or validating (visualizing), pick at random.
        index_target = torch.randint(
            index_context_left,
            index_context_right + 1,
            size=(1,),
            device='cpu',
        )

    return (
        torch.tensor((index_context_left, index_context_right)),
        index_target,
    )

def one_sample(scene, extrinsics, stage="train", i=0):
    num_views, _, _ = extrinsics.shape
    context_gap = 4

    # Pick the left and right context indices.

    if stage == "val":
        index_context_left = (num_views - context_gap - 1) * (i+1) / max((100 + 1), 1)
        index_context_left = int(index_context_left)
    else:
        index_context_left = torch.randint(context_gap, (num_views - context_gap - 1), (1,)).item()

    return (
        torch.tensor([index_context_left]),
        torch.tensor((index_context_left - context_gap//2, index_context_left, index_context_left + context_gap//2)),
    )

class Dataset360Loc(IterableDataset):
    def __init__(
        self,
        stage,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        self.near = 0.45
        self.far = 50
        self.width = pano_width
        self.height = pano_height

        if stage == "train":
            locations = ['concourse', 'hall', 'piatrium']
            # locations = ['hall']
        else:
            locations = ['atrium']
        root = Path('/data/qiwei/nips25/360Loc')
        self.data = []
        for location in locations:
            seqs = [list((root / location / folder).glob('daytime_360*/')) for folder in ('mapping', 'query_360')]
            seqs = sum(seqs, [])
            self.data.extend(seqs)

        self.times_per_scene = 1000 if self.stage == "train" else 20
        self.load_images = True
        self.direction = get_panorama_ray_directions(self.height, self.width)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def load_extrinsics(self, example_path):
        example = example_path / 'camera_pose.json'
        with open(example) as f:
            example = json.load(f)
        frames, extrinsics_orig = list(example.keys()), list(example.values())
        extrinsics_orig = torch.tensor(extrinsics_orig)
        return frames, extrinsics_orig

    @cached_property
    def total_frames(self):
        extrinsics = [self.load_extrinsics(example)[1] for example in self.data]
        return sum(len(e) for e in extrinsics)

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train"):
            self.data = self.shuffle(self.data)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage != "train" and worker_info is not None:
            self.data = [
                example
                for data_idx, example in enumerate(self.data)
                if data_idx % worker_info.num_workers == worker_info.id
            ]

        for example_path in self.data:
            frames, extrinsics_orig = self.load_extrinsics(example_path)
            scene = f"{example_path.parts[-3]}-{example_path.parts[-1]}"

            if self.stage == "train":
                images_path = [example_path / 'image' / frame for frame in frames]
                images = self.convert_images(images_path)

            for i in range(self.times_per_scene):
                context_indices, target_indices = one_sample(
                    scene,
                    extrinsics_orig,
                    stage=self.stage,
                    i=i,
                )
                if context_indices is None:
                    break


                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics_orig[context_indices]
                target_extrinsics = extrinsics_orig[target_indices]
                target_extrinsics_relative = torch.inverse(context_extrinsics) @ target_extrinsics
                # target_extrinsics_relative = w2w.T @ target_extrinsics_relative @ w2w

                # Load the images.
                if self.stage == "train":
                    context_images = images[context_indices]
                    target_images = images[target_indices]
                else:
                    context_images_path = [example_path / 'image' / frames[i] for i in context_indices]
                    context_images = self.convert_images(context_images_path)
                    target_images_path = [example_path / 'image' / frames[i] for i in target_indices]
                    target_images = self.convert_images(target_images_path)
                
                input_dict = {"rgb": context_images}

                # Load the depth.
                # relative depth path
                index = torch.cat((context_indices, target_indices))
                depths_path = []
                depths_m_path = []
                confs_m_path = []
                if self.stage == "train":
                    for i in index:
                        depths_path.append(str(images_path[i]).replace('image', 'depth_metric').replace('.jpg', '_depth.npy'))
                        depths_m_path.append(str(images_path[i]).replace('image', 'depth_metric').replace('.jpg', '_depth.npy'))
                        confs_m_path.append(str(images_path[i]).replace('image', 'depth_metric').replace('.jpg', '_conf.npy'))
                else:
                    depths_path = [example_path / 'depth_metric' / frames[i].replace('.jpg', '_depth.npy') for i in index]
                    depths_m_path = [example_path / 'depth_metric' / frames[i].replace('.jpg', '_depth.npy') for i in index]
                    confs_m_path = [example_path / 'depth_metric' / frames[i].replace('.jpg', '_conf.npy') for i in index]

                target_index = len(context_indices)

                context_depths = self.convert_depths(depths_path[:target_index])
                target_depths = self.convert_depths(depths_path[target_index:])
                # metric depth path
                # depths_path = [str(scene_path / v / 'depth.png') for v in views]
                context_m_depths = self.convert_depths(depths_m_path[:target_index])
                target_m_depths = self.convert_depths(depths_m_path[target_index:])

                context_m_confs = self.convert_depths(confs_m_path[:target_index])
                target_m_confs = self.convert_depths(confs_m_path[target_index:])

                # context_depths = context_depths.float() / 1000
                # target_depths = target_depths.float() / 1000
                context_depths = context_depths.clamp(min=0.)
                target_depths = target_depths.clamp(min=0.)
                context_mask = (context_m_depths > self.near) & (context_m_depths < self.far)
                target_mask = (target_m_depths > self.near) & (target_m_depths < self.far)

                # process rays
                output_fovxs = torch.deg2rad(torch.tensor([90], dtype=torch.float32)).repeat(len(target_indices))
                output_fovys = torch.deg2rad(torch.tensor([90], dtype=torch.float32)).repeat(len(target_indices))
                input_directions = output_directions = self.direction.unsqueeze(0)
                
                input_rays_o, input_rays_d = get_rays(
                    input_directions, torch.eye(4,4)[None,:,:], keepdim=True, normalize=False)
                output_rays_o, output_rays_d = get_rays(
                                    output_directions, target_extrinsics_relative, keepdim=True, normalize=False)
                fx, fy, cx, cy = 0.25, 0.5, 0.5, 0.5
                
                input_dict_pix = {
                    "depth_m": context_m_depths, 
                    "conf_m": context_m_confs,
                    "ck": torch.zeros(1,3,3), 
                    "c2w": torch.eye(4,4)[None,:,:],
                    "cx": torch.tensor([cx]), 
                    "cy": torch.tensor([cy]), 
                    "fx": torch.tensor([fx]), 
                    "fy": torch.tensor([fy]),
                    "rays_o": input_rays_o, 
                    "rays_d": input_rays_d
                }

                input_dict_vol = {"w2i": torch.eye(4,4)[None,:,:]}

                output_dict = {
                    "rgb": target_images, 
                    "depth": target_depths,
                    "depth_m": target_m_depths, 
                    "conf_m": target_m_confs,
                    "c2w": target_extrinsics_relative, 
                    "fovx": output_fovxs, 
                    "fovy": output_fovys, 
                    "rays_o": output_rays_o,
                    "rays_d": output_rays_d, 
                }

                yield {
                    "outputs": output_dict,
                    "inputs": input_dict,
                    "inputs_pix": input_dict_pix,
                    "inputs_vol": input_dict_vol,
                }
                
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

    def convert_images(
        self,
        images,
    ):
        torch_images = []
        for image in images:
            try:
                image = Image.open(image)
                image = image.resize([self.width, self.height], Image.LANCZOS)
                torch_images.append(self.to_tensor(image))
            except Exception as e:
                print(f"Error: {e}")
        return torch.stack(torch_images)
    
    def convert_poses(
        self,
        context_extrinsics,
        target_extrinsics,
    ):  # extrinsics

        # w2c = context_extrinsics @ torch.inverse(target_extrinsics)
        c2w = torch.inverse(context_extrinsics) @ target_extrinsics
        c2w = w2w @ c2w
        return c2w
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self) -> int:
        return len(self.data) * self.times_per_scene

def get_generator(seed):
    generator = Generator()
    generator.manual_seed(seed)
    return generator

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))

def load_360Loc_data(batch_size, stage='train'):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    Loc360 = Dataset360Loc(stage = stage)

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
        Loc360, 
        batch_size=batch_size,
        num_workers=1,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        shuffle=False
    )
    # val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return dataloader

def ply_post_opencv(data):
    # 提取每一张图片的平移向量（最后一列）
    camera_positions = data[:, :, 3].numpy()

    # 提取 X 和 Y 坐标
    x = camera_positions[:, 0]
    y = camera_positions[:, 2]

    # 创建一个 2D 绘图
    plt.figure()

    # 绘制相机轨迹
    plt.plot(x, y, marker='o', color='b', label='Camera Trajectory')
    # 设置 X 和 Y 坐标轴比例一致
    plt.axis('equal')
    # 添加标签和标题
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Camera Movement in XY Plane')
    plt.legend()

    # 如果需要保存图形，可以使用以下命令
    plt.savefig("camera_trajectory_xy1.png", dpi=300)
    plt.close()

def ply_post(data):
    # 提取每一张图片的平移向量（最后一列）
    camera_positions = data[:, :, 3].numpy()

    # 提取 X 和 Y 坐标
    x = camera_positions[:, 0]
    y = camera_positions[:, 1]

    # 创建一个 2D 绘图
    plt.figure()

    # 绘制相机轨迹
    plt.plot(x, y, marker='o', color='b', label='Camera Trajectory')
    # 设置 X 和 Y 坐标轴比例一致
    plt.axis('equal')
    # 添加标签和标题
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Camera Movement in XY Plane')
    plt.legend()

    # 如果需要保存图形，可以使用以下命令
    plt.savefig("camera_trajectory_xy2.png", dpi=300)
    plt.close()