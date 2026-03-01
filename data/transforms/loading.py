import numpy as np
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from model.utils.image import resize_image, HWC3
from typing import List, Optional, Union
import cv2
import os.path as osp
import json
import copy

def load_info(info):
    img_path = info["data_path"]
    # use lidar coordinate of the key frame as the world coordinate
    c2w = info["sensor2lidar_transform"]
    # opencv cam -> opengl cam, maybe not necessary!
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = c2w@flip_yz

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
    
    return img_path, c2w, w2c

def load_conditions(img_paths, reso):
    
    def maybe_resize(img, tgt_reso, ck):
        if not isinstance(img, PIL.Image.Image):
            img = Image.fromarray(img)
        resize_flag = False
        if img.height != tgt_reso[0] or img.width != tgt_reso[1]:
            # img.resize((w, h))
            fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
            scale_h, scale_w = tgt_reso[0] / img.height, tgt_reso[1] / img.width
            fx_scaled, fy_scaled, cx_scaled, cy_scaled = fx * scale_w, fy * scale_h, cx * scale_w, cy * scale_h
            ck = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])
            img = img.resize((tgt_reso[1], tgt_reso[0]))
            resize_flag = True
        return np.array(img), ck, resize_flag
    
    imgs, cks = [], []
    depths = []
    depths_m = []
    confs_m = []
    for img_path in img_paths:      
        # param
        param_path = img_path.replace("samples", "samples_param_small") # 224x400 resolution
        param_path = param_path.replace("sweeps", "sweeps_param_small")
        param_path = param_path.replace(".jpg", ".json")
        param = json.load(open(param_path))
        ck = np.array(param["camera_intrinsic"])

        # img
        img_path = img_path.replace("samples", "samples_small")
        img_path = img_path.replace("sweeps", "sweeps_small")
        img = Image.open(img_path)
        h, w = img.height, img.width
        img, ck, resize_flag = maybe_resize(img, reso, ck)
        img = HWC3(img)
        imgs.append(img)
        cks.append(ck)

        # relative depth from DepthAnything-v2
        depth_path = img_path.replace("sweeps_small", "sweeps_dpt_small")
        depth_path = depth_path.replace("samples_small", "samples_dpt_small")
        depth_path = depth_path.replace(".jpg", ".npy")
        disp = np.load(depth_path).astype(np.float32)
        if resize_flag:
            disp = Image.fromarray(disp)
            disp = disp.resize((reso[1], reso[0]), Image.BILINEAR)
            disp = np.array(disp)
        # inverse disparity to relative depth
        # clamping the farthest depth to 50x of the nearest
        range = np.minimum(disp.max() / (disp.min() + 0.001), 50.0)
        max = disp.max()
        min = max / range
        depth = 1 / np.maximum(disp, min)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depths.append(depth)
        # metric depth from Metric3D-v2
        depthm_path = img_path.replace("sweeps_small", "sweeps_dptm_small")
        depthm_path = depthm_path.replace("samples_small", "samples_dptm_small")
        depthm_path = depthm_path.replace(".jpg", "_dpt.npy")
        conf_path = depthm_path.replace("_dpt.npy", "_conf.npy")
        dptm = np.load(depthm_path).astype(np.float32)
        conf = np.load(conf_path).astype(np.float32)
        if resize_flag:
            dptm = Image.fromarray(dptm)
            dptm = dptm.resize((reso[1], reso[0]), Image.BILINEAR)
            dptm = np.array(dptm)
            conf = Image.fromarray(conf)
            conf = conf.resize((reso[1], reso[0]), Image.BILINEAR)
            conf = np.array(conf)
        depths_m.append(dptm)
        confs_m.append(conf)

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0  # [v c h w]
    depths = torch.from_numpy(np.stack(depths, axis=0)).float()  # [v h w]
    depths_m = torch.from_numpy(np.stack(depths_m, axis=0)).float()  # [v h w]
    confs_m = torch.from_numpy(np.stack(confs_m, axis=0)).float()  # [v h w]
    cks = torch.as_tensor(cks, dtype=torch.float32)

    return imgs, depths, depths_m, confs_m, cks

def load_lidar_info(info):
    pcd_path = info["data_path"]
    lidar2sensor = np.eye(4)
    rot = info["sensor2lidar_rotation"]
    trans = info["sensor2lidar_translation"]
    lidar2sensor[:3, :3] = copy.deepcopy(rot.T)
    lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
    return pcd_path, lidar2sensor

from nuscenes.utils.geometry_utils import view_points
def load_sparse_depths(points, lidar2sensor, w2c, ck):
    # transform lidar points to camera coordinate
    points[:, :3] = points[:, :3] @ lidar2sensor[:3, :3]
    points[:, :3] -= lidar2sensor[:3, 3]
    points_new = np.ones((points.shape[0], 4))
    points_new[:, :3] = points
    points_new = np.matmul(np.repeat(w2c.T[None], points.shape[0], axis=0), points_new[..., None])[:, :, 0]
    points_new /= points_new[:, -1:]
    points_new = points_new[:, :3]
    depths = points_new[:, 2]
    # transform 3d points to 2d images
    points_2d = view_points(points_new.transpose(), ck, normalize=True).transpose()
    width, height = int(ck[0, 2] * 2), int(ck[1, 2] * 2)
    # filter invalid points in 2d
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points_2d[:, 0] > 1)
    mask = np.logical_and(mask, points_2d[:, 0] < width - 1)
    mask = np.logical_and(mask, points_2d[:, 1] > 1)
    mask = np.logical_and(mask, points_2d[:, 1] < height - 1)
    points_2d = points_2d[mask]
    points_2d_int = np.array(points_2d, dtype=int)
    depths = depths[mask]
    depth_map = np.zeros((width, height))
    depth_map[points_2d_int[:, 0], points_2d_int[:, 1]] = depths
    return np.transpose(depth_map, (1, 0))
