"""360Loc two-view panorama loader at 256x512 (despite the file name) for the 360Loc train and eval rows."""

from typing import Literal

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

import random
from torch import Generator

import torch.nn.functional as F
import json
from functools import cached_property
from model.utils.ops import get_panorama_ray_directions, get_rays
from .paths import LOC360_ROOT

pano_width = 512
pano_height = 256

w2w = torch.tensor(
    [  #  X -> X, Z -> Y, Y -> -Z
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]
).float()


def one_sample(scene, extrinsics, stage="train", i=0):
    num_views, _, _ = extrinsics.shape
    context_gap = 4

    # Pick the left and right context indices.

    if stage == "val":
        index_context_left = (num_views - context_gap - 1) * (i + 1) / max((100 + 1), 1)
        index_context_left = int(index_context_left)
    else:
        index_context_left = torch.randint(context_gap, (num_views - context_gap - 1), (1,)).item()

    return (
        torch.tensor([index_context_left]),
        torch.tensor(
            (index_context_left - context_gap // 2, index_context_left, index_context_left + context_gap // 2)
        ),
    )


def two_sample(scene, extrinsics, times_per_scene, stage="train", i=0):
    num_views, _, _ = extrinsics.shape
    # Compute the context view spacing based on the current global step.
    if stage == "val":
        # When testing, always use the full gap.
        min_gap = max_gap = 3
    else:
        min_gap = max_gap = 3
    max_gap = min(num_views - 1, min_gap)

    # Pick the gap between the context views.
    # NOTE: we keep the bug untouched to follow initial pixelsplat cfgs

    context_gap = torch.randint(
        min_gap,
        max_gap + 1,
        size=tuple(),
        device="cpu",
    ).item()

    # Pick the left and right context indices.

    if stage == "val":
        index_context_left = (num_views - context_gap - 1) * i / max((times_per_scene - 1), 1)
        index_context_left = int(index_context_left)
    else:
        index_context_left = torch.randint(
            num_views - context_gap,
            size=tuple(),
            device="cpu",
        ).item()
    index_context_right = index_context_left + context_gap

    index_target = torch.arange(
        index_context_left,
        index_context_right + 1,
        device="cpu",
    )

    return (
        torch.tensor((index_context_left, index_context_right)),
        index_target,
    )


class Dataset360Loc(IterableDataset):
    def __init__(
        self,
        stage,
        interleave=False,
        pcc_reference="depth_metric",
    ) -> None:
        """stage: 'train' (concourse, hall, piatrium) or another stage ('val' = the held-out atrium).

        Options (the defaults reproduce the released loader):
            interleave (train only; switch loc360_interleave): yield one globally shuffled stream of
                (sequence, sample) pairs instead of the sequences one after another; each sequence still
                gives times_per_scene samples per epoch. Frames are decoded on first use and kept as
                uint8 (to_tensor's exact values), so a persistent worker decodes every frame once.
            pcc_reference (evaluation stages only): 'depth_metric' (the UniK3D prior, as released) or
                'depth_anywhere' (the Depth Anywhere pseudo-GT the paper's PCC uses) as outputs['depth'].
        """
        super().__init__()
        if interleave and stage != "train":
            raise ValueError(f"interleave applies to the train split only, not stage {stage!r}")
        if pcc_reference not in ("depth_metric", "depth_anywhere"):
            raise ValueError(f"pcc_reference must be 'depth_metric' or 'depth_anywhere', got {pcc_reference!r}")
        if pcc_reference != "depth_metric" and stage == "train":
            raise ValueError("pcc_reference applies to evaluation stages only")
        self.interleave = interleave
        self.pcc_reference = pcc_reference
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        self.near = 0.45
        self.far = 50
        self.width = pano_width
        self.height = pano_height

        if stage == "train":
            locations = ["concourse", "hall", "piatrium"]
        else:
            locations = ["atrium"]
        root = LOC360_ROOT
        self.data = []
        for location in locations:
            seqs = [list((root / location / folder).glob("*360*/")) for folder in ("mapping", "query_360")]
            seqs = sum(seqs, [])
            self.data.extend(seqs)

        self.times_per_scene = 1000 if self.stage == "train" else 100
        self.load_images = True
        self.direction = get_panorama_ray_directions(self.height, self.width)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def load_extrinsics(self, example_path):
        example = example_path / "camera_pose.json"
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
        if self.interleave:
            yield from self._iter_interleaved()
            return
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train"):
            self.data = self.shuffle(self.data)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.data = [
                example
                for data_idx, example in enumerate(self.data)
                if data_idx % worker_info.num_workers == worker_info.id
            ]

        for example_path in self.data:
            frames, extrinsics_orig = self.load_extrinsics(example_path)
            scene = f"{example_path.parts[-3]}-{example_path.parts[-1]}"

            if self.stage == "train":
                images_path = [example_path / "image" / frame for frame in frames]
                images = self.convert_images(images_path)

            for i in range(self.times_per_scene):
                context_indices, target_indices = two_sample(
                    scene,
                    extrinsics_orig,
                    self.times_per_scene,
                    stage=self.stage,
                    i=i,
                )
                if context_indices is None:
                    break

                yield self._make_sample(
                    example_path,
                    frames,
                    extrinsics_orig,
                    context_indices,
                    target_indices,
                    images=images if self.stage == "train" else None,
                    images_path=images_path if self.stage == "train" else None,
                )

    def _iter_interleaved(self):
        """loc360_interleave: every (sequence, i) pair of the epoch in one random order.

        The released stream (above) shuffles only the sequence order and then yields all
        times_per_scene samples of one sequence before the next, so ~times_per_scene / (ranks x batch)
        consecutive optimizer steps (and the BatchNorm running statistics a checkpoint saves) come
        from a single sequence. Here the pairs of all sequences are permuted together (torch's RNG of
        the worker, like the sequence shuffle above); the per-sample code is the same.
        """
        data = list(self.data)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            data = [
                example for data_idx, example in enumerate(data) if data_idx % worker_info.num_workers == worker_info.id
            ]
        sequences = []
        for example_path in data:
            frames, extrinsics_orig = self.load_extrinsics(example_path)
            images_path = [example_path / "image" / frame for frame in frames]
            sequences.append((example_path, frames, extrinsics_orig, images_path))
        pairs = [(s, i) for s in range(len(sequences)) for i in range(self.times_per_scene)]
        for k in torch.randperm(len(pairs)).tolist():
            s, i = pairs[k]
            example_path, frames, extrinsics_orig, images_path = sequences[s]
            scene = f"{example_path.parts[-3]}-{example_path.parts[-1]}"
            context_indices, target_indices = two_sample(
                scene,
                extrinsics_orig,
                self.times_per_scene,
                stage=self.stage,
                i=i,
            )
            yield self._make_sample(
                example_path,
                frames,
                extrinsics_orig,
                context_indices,
                target_indices,
                images=_FrameCache(self, images_path),
                images_path=images_path,
            )

    def _make_sample(
        self, example_path, frames, extrinsics_orig, context_indices, target_indices, images=None, images_path=None
    ):
        """One sample. Train: `images` indexes the sequence's frames (the stacked sequence, or the
        loc360_interleave frame cache) and `images_path` lists their files; other stages read files."""
        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics_orig[context_indices]
        target_extrinsics = extrinsics_orig[target_indices]
        ref_extrinsics = target_extrinsics[:1]
        target_extrinsics_relative = torch.inverse(ref_extrinsics) @ target_extrinsics
        context_extrinsics_relative = torch.inverse(ref_extrinsics) @ context_extrinsics

        # Load the images.
        if self.stage == "train":
            context_images = images[context_indices]
            target_images = images[target_indices]
        else:
            context_images_path = [example_path / "image" / frames[i] for i in context_indices]
            context_images = self.convert_images(context_images_path)
            target_images_path = [example_path / "image" / frames[i] for i in target_indices]
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
                depths_path.append(str(images_path[i]).replace("image", "depth_metric").replace(".jpg", "_depth.npy"))
                depths_m_path.append(str(images_path[i]).replace("image", "depth_metric").replace(".jpg", "_depth.npy"))
                confs_m_path.append(str(images_path[i]).replace("image", "depth_metric").replace(".jpg", "_conf.npy"))
        else:
            depths_path = [example_path / "depth_metric" / frames[i].replace(".jpg", "_depth.npy") for i in index]
            depths_m_path = [example_path / "depth_metric" / frames[i].replace(".jpg", "_depth.npy") for i in index]
            confs_m_path = [example_path / "depth_metric" / frames[i].replace(".jpg", "_conf.npy") for i in index]

        target_index = len(context_indices)

        if self.pcc_reference == "depth_anywhere":
            # PCC reference = Depth Anywhere pseudo-GT, read like the 160x320 loader
            # (data/loc360_dataloader_double_all.py: the png through convert_images, then clamp)
            depths_path = [
                example_path / "depthanywhere" / frames[i].replace(".jpg", "_depth_anywhere.png") for i in index
            ]
            context_depths = self.convert_images(depths_path[:target_index], strict=True)
            target_depths = self.convert_images(depths_path[target_index:], strict=True)
        else:
            context_depths = self.convert_depths(depths_path[:target_index])
            target_depths = self.convert_depths(depths_path[target_index:])
        # metric depth path
        context_m_depths = self.convert_depths(depths_m_path[:target_index])
        target_m_depths = self.convert_depths(depths_m_path[target_index:])

        context_m_confs = self.convert_depths(confs_m_path[:target_index])
        target_m_confs = self.convert_depths(confs_m_path[target_index:])

        context_depths = context_depths.clamp(min=0.0)
        target_depths = target_depths.clamp(min=0.0)

        # process rays
        output_fovxs = torch.deg2rad(torch.tensor([90], dtype=torch.float32)).repeat(len(target_indices))
        output_fovys = torch.deg2rad(torch.tensor([90], dtype=torch.float32)).repeat(len(target_indices))
        input_directions = output_directions = self.direction.unsqueeze(0)

        input_rays_o, input_rays_d = get_rays(
            input_directions, context_extrinsics_relative, keepdim=True, normalize=False
        )
        output_rays_o, output_rays_d = get_rays(
            output_directions, target_extrinsics_relative, keepdim=True, normalize=False
        )
        fx, fy, cx, cy = 0.25, 0.5, 0.5, 0.5

        input_dict_pix = {
            "depth_m": context_m_depths,
            "conf_m": context_m_confs,
            "ck": torch.zeros(1, 3, 3),
            "c2w": context_extrinsics_relative,
            "cx": torch.tensor([cx]),
            "cy": torch.tensor([cy]),
            "fx": torch.tensor([fx]),
            "fy": torch.tensor([fy]),
            "rays_o": input_rays_o,
            "rays_d": input_rays_d,
        }

        input_dict_vol = {"w2i": torch.inverse(context_extrinsics_relative)}

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

        return {
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
        for depth_path in depths:
            depth = np.load(depth_path, allow_pickle=False)
            depth = torch.tensor(depth, dtype=torch.float32)
            torch_depths.append(depth)
        return F.interpolate(torch.stack(torch_depths), size=(self.height, self.width), mode="nearest")

    def convert_images(
        self,
        images,
        strict=False,
    ):
        # strict: raise on an unreadable file (the default prints and skips it, as released, which
        # shifts every later frame of a stacked sequence against its pose)
        torch_images = []
        for image in images:
            try:
                image = Image.open(image)
                image = image.resize([self.width, self.height], Image.LANCZOS)
                torch_images.append(self.to_tensor(image))
            except Exception as e:
                if strict:
                    raise
                print(f"Error: {e}")
        return torch.stack(torch_images)

    def frame_uint8(self, path):
        """loc360_interleave: one frame resized like convert_images, cached as uint8 CHW.

        to_tensor turns a uint8 PIL image into uint8 / 255 in float32, so `.float().div(255)` of the
        cached tensor is bit-identical to convert_images([path])[0].
        """
        cache = self.__dict__.setdefault("_frame_cache", {})
        key = str(path)
        if key not in cache:
            image = Image.open(path)
            image = image.resize([self.width, self.height], Image.LANCZOS)
            if image.mode != "RGB":
                raise ValueError(f"{path}: expected an RGB image, got mode {image.mode}")
            cache[key] = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).contiguous()
        return cache[key]

    def convert_poses(
        self,
        context_extrinsics,
        target_extrinsics,
    ):  # extrinsics

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


class _FrameCache:
    """images[index] for one sequence, served from Dataset360Loc.frame_uint8 (loc360_interleave)."""

    def __init__(self, dataset, images_path):
        self.dataset = dataset
        self.images_path = images_path

    def __getitem__(self, indices):
        return torch.stack([self.dataset.frame_uint8(self.images_path[int(i)]).float().div(255) for i in indices])


def get_generator(seed):
    generator = Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


def load_360Loc_data(batch_size, stage="train"):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    Loc360 = Dataset360Loc(stage=stage)

    if stage == "train":
        seed = 1234
        persistent_workers = True
    elif stage == "val":
        seed = 3456
        persistent_workers = True
    elif stage == "test":
        seed = 2345
        persistent_workers = False
    else:
        seed = 6789
        persistent_workers = True

    dataloader = DataLoader(
        Loc360,
        batch_size=batch_size,
        num_workers=1,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        shuffle=False,
    )

    return dataloader
