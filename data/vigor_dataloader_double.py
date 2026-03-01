import random

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
from torch import Generator, nn
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from einops import repeat
from model.utils.ops import get_panorama_ray_directions, get_rays

sat_range = 200

GrdImg_H = 160  # 256
GrdImg_W = 320  # 1024

pano_width = 320
pano_height = 160

depth_scale = 10.0  # from meter to decameter

# 坐标系转换：从(x东, y北, z上)转换为(x南, y下, z东)
# x_new = -y_old, y_new = -z_old, z_new = x_old
OpenCV_Transform = torch.tensor([
    [0, -1, 0, 0],  # x_new = -y_old
    [0, 0, -1, 0],  # y_new = -z_old
    [1, 0, 0, 0],   # z_new = x_old
    [0, 0, 0, 1]
], dtype=torch.float32)

SatMap_end_sidelength = 256
class SatGrdDataset(Dataset):
    def __init__(self, root_dir, T_in=1, T_out=3, is_train=True):
        self.root = root_dir
        self.T_in = T_in
        self.T_out = T_out
        self.is_train = is_train
        
        self.city_list = ["Kansas"]
        # self.city_list = ["chicago"]
        # self.city_list = ["newyork","Orlando","Phoenix","SanFrancisco","seattle"]
        self.grd_in_sat_path = "GrdInSat_dist_dir_month_new"
        self.grd_dir = "Ground"
        self.sat_dir = "Satellite"
        self.depth_dir = 'depth_metric'

        self.SatMap_length = SatMap_end_sidelength
        self.satmap_transform = transforms.Compose([
            transforms.Resize(size=[self.SatMap_length, self.SatMap_length]),#sat_d*sat_d
            transforms.ToTensor(),
        ])

        Grd_h = GrdImg_H
        Grd_w = GrdImg_W

        self.grdimage_transform = transforms.Compose([
            transforms.Resize(size=[Grd_h, Grd_w]),#grd_H*grd_W
            transforms.ToTensor(),
        ])

        # #读取规定的
        # with open('dataloader/selected_test_files_2.txt', 'r') as f:
        #     # 去除每行的换行符和首尾空格
        #     self.file_name = [line.strip() for line in f if line.strip()]

        self.width = pano_width
        self.height = pano_height

        # #原始代码
        self.file_name = []
        for city in self.city_list:
            if is_train:
                data_list_path = os.path.join(self.root, city, 'train_list.txt')
            else:
                data_list_path = os.path.join(self.root, city, 'test_list_om.txt')
            with open(data_list_path, 'r') as f:
                file_name = f.readlines()
            for file in file_name:
                self.file_name.append(city + '/' + file[:-1])
        
        # selected_files = random.sample(self.file_name, 500)
        # # 保存到 txt 文件
        # with open('selected_files.txt', 'w') as f:
        #     for file in selected_files:
        #         f.write(file + '\n')
        # print(f"Dataset length: {len(self.file_name)}")

        self.direction = get_panorama_ray_directions(self.height, self.width)
        self.R_ENU_to_SDE = np.array([
            [0, -1, 0],  # x_new = -y_old
            [0, 0, -1],  # y_new = -z_old
            [1, 0, 0]    # z_new = x_old
        ], dtype=np.float32)

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        R_pitch = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        R_yaw = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        return R_yaw @ R_pitch @ R_roll

    def create_transformation_matrix(self, x, y, z, R):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T
    
    def get_point(self, all_id, sample_num):
        id_list = []
        if self.is_train:
            now_id = random.randint(0, max(all_id - sample_num, 0))
        else:
            now_id = 0
        for i in range(sample_num):
            id_list.append(now_id)
            now_id = min(now_id + 1, all_id-1)
        return id_list
        
        

    def __getitem__(self, idx):
        city, road, file_name = self.file_name[idx].split('/')
        input_ims = []
        target_Ts = []  

        context_indices = torch.tensor([0, 2])
        target_indices = torch.tensor([0, 1, 2])
        if not file_name.endswith('.png'):
            file_name = file_name + 'g'
        SatMap_name = os.path.join(self.root, city, road, self.sat_dir, file_name)

        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            orin_sat_S = sat_map.size[0]
            sat_map_tensor = self.satmap_transform(sat_map)
        input_ims.append(sat_map_tensor)
        orin_meter_per_pixel = sat_range/orin_sat_S
         # =================== read correspond grd ============================
        GrdInSat_file_path = os.path.join(self.root, city, road, self.grd_in_sat_path, file_name.replace('.png', '.txt'))
        with open(GrdInSat_file_path, 'r') as GrdInSat_f:
            sat2grd_name = GrdInSat_f.readlines()

        targets_indexes = np.array(self.get_point(len(sat2grd_name), self.T_out))
        # targets_indexes = np.array([3, 4, 5])  # 固定采样后三个视角
        draw_camera_pose = []
        grd_img_name = []
        depths = []
        confs = []

        for i in targets_indexes:
            #target_ims
            # print(sat2grd_name[i])
            grd_name, u, v, yaw, alt, o_alt = sat2grd_name[i].split(' ')
            u, v, yaw, alt, o_alt = float(u), float(v), float(yaw), float(alt), float(o_alt)
            draw_camera_pose.append([u, v, yaw, alt])
            grd_name = grd_name.split('/')[-1]

            left_img_name = os.path.join(self.root, city, road, self.grd_dir, grd_name)
            delta_E = (float(u) - orin_sat_S/2)*orin_meter_per_pixel
            delta_E = delta_E / depth_scale  # scale to decameter
            delta_N = -(float(v) - orin_sat_S/2)*orin_meter_per_pixel
            delta_N = delta_N / depth_scale  # scale to decameter
            delta_alt = alt - o_alt
            delta_alt = delta_alt / depth_scale  # scale to decameter

            # gemini code
            # R_enu = self.euler_to_rotation_matrix(0, 0, yaw) 
            # t_enu = np.array([delta_E, delta_N, delta_alt])
            # R_M = self.R_ENU_to_SDE
            # R_sde = R_M @ R_enu
            # t_sde = R_M @ t_enu
            # target_Ts_RT = self.create_transformation_matrix(t_sde[0], t_sde[1], t_sde[2], R_sde)
            
            # xinaghui code
            target_Ts_R = self.euler_to_rotation_matrix(0, -(yaw + 90), 0)

            # real_heading = np.deg2rad(yaw)
            # cos = np.cos(-real_heading)
            # sin = np.sin(-real_heading)
            # zeros = np.zeros_like(cos)
            # ones = np.ones_like(cos)
            # R = np.stack([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], axis=-1)  # shape = [B,9]
            # target_Ts_R = R.reshape(3, 3)
            
            target_Ts_RT = self.create_transformation_matrix(-delta_N, -delta_alt, delta_E, target_Ts_R) # x是南，y是下 z是东

            target_Ts.append(target_Ts_RT)
            grd_img_name.append(left_img_name)

            depth_name = left_img_name.replace(self.grd_dir, self.depth_dir).replace('.png', '_depth.npy')
            depths.append(depth_name)

            conf_name = depth_name.replace('_depth.npy', '_conf.npy')  # in meter
            confs.append(conf_name)

        context_images = self.convert_images([grd_img_name[i] for i in context_indices])
        target_images = self.convert_images([grd_img_name[i] for i in target_indices])

        context_m_depths = [depths[i] for i in context_indices]
        target_m_depths = [depths[i] for i in target_indices]
        context_m_depths = self.convert_depths(context_m_depths) / depth_scale  # to decimeter
        target_m_depths = self.convert_depths(target_m_depths) / depth_scale  # to decimeter

        context_m_confs = [confs[i] for i in context_indices]
        target_m_confs = [confs[i] for i in target_indices]
        context_m_confs = self.convert_depths(context_m_confs)
        target_m_confs = self.convert_depths(target_m_confs)


        extrinsics = torch.stack([torch.from_numpy(T).float() for T in target_Ts])       # x是东，y是北 z是上
        # extrinsics = torch.bmm(OpenCV_Transform.unsqueeze(0).expand(len(extrinsics), -1, -1), extrinsics) # [B,4,4] # x是南，y是下，z是东

        ref_cam = extrinsics[1:2]
        # 将所有相机转换到中间相机的坐标系下
        ref_cam_inv = torch.inverse(ref_cam)  # [1,4,4]
        extrinsics = torch.einsum("bij,bjk->bik", ref_cam_inv, extrinsics)  # [B,4,4]
    
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
            "depth": context_m_depths,
            "depth_m": context_m_depths, 
            "conf_m": context_m_confs,
            "ck": torch.zeros(1,3,3), 
            "c2w": extrinsics[context_indices],
            "cx": torch.tensor([cx]), 
            "cy": torch.tensor([cy]), 
            "fx": torch.tensor([fx]), 
            "fy": torch.tensor([fy]),
            "rays_o": input_rays_o, 
            "rays_d": input_rays_d,
            "depth_gt": torch.zeros_like(context_m_depths),  # [B, 1, H, W]
            "mask_gt": torch.zeros_like(context_m_depths, dtype=torch.bool),  # [B, 1, H, W] - 确保布尔类型一致
        }

        input_dict_vol = {"w2i": torch.inverse(extrinsics[context_indices])}

        output_dict = {
            "rgb": target_images, 
            "depth": target_m_depths,
            "depth_m": target_m_depths, 
            "conf_m": target_m_confs,
            "c2w": extrinsics[target_indices], 
            "fovx": output_fovxs, 
            "fovy": output_fovys, 
            "rays_o": output_rays_o,
            "rays_d": output_rays_d,
            "depth_gt": torch.zeros_like(target_m_depths),  # [B, 1, H, W]
            "mask_gt": torch.zeros_like(target_m_depths, dtype=torch.bool),  # [B, 1, H, W] - 确保布尔类型一致
        }

        scene_name = 'VIGOR'

        return {
            "outputs": output_dict,
            "inputs": input_dict,
            "inputs_pix": input_dict_pix,
            "inputs_vol": input_dict_vol,
            "scene": scene_name,
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
            image = Image.open(image)
            image = image.convert('RGB')  # 确保图像是RGB格式，而不是RGBA
            image = image.resize([self.width, self.height], Image.LANCZOS)
            torch_images.append(transforms.ToTensor()(image))
        return torch.stack(torch_images)
    

def get_generator(seed):
    generator = Generator()
    generator.manual_seed(seed)
    return generator

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


def load_VIGOR_data(batch_size, stage='train'):
    """

    Args:
        batch_size: B
        area: same | cross
    """
    if stage == 'train':
        seed = 1234
        shuffle = True
        persistent_workers = True
        is_Train = True
    elif stage == 'val':
        seed = 3456
        shuffle = False
        persistent_workers = True
        is_Train = False
    elif stage == 'test':
        seed = 2345
        shuffle = False
        persistent_workers = False
        is_Train = False
    else:
        seed = 6789
        shuffle = False
        persistent_workers = True
        is_Train = False

    Loc360 = SatGrdDataset(root_dir= '/data/qiwei/nips25/', is_train=is_Train)

    dataloader = DataLoader(
        Loc360, 
        batch_size=batch_size,
        num_workers=32,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        shuffle=False
    )
    # val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return dataloader