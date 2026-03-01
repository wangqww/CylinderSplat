import random

import numpy as np
import os
from PIL import Image
import PIL
from torch.utils.data import Dataset, Subset

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from model.utils.ops import get_panorama_ray_directions, get_rays


num_thread_workers = 32
pano_width = 320
pano_height = 160
root = '/data/dataset/VIGOR'

class VIGORDataset(Dataset):
    def __init__(self, root, label_root='splits__corrected', split='same', train=True, transform=None, pos_only=True, amount=1.):
        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.pos_only = pos_only

        self.width = pano_width
        self.height = pano_height

        if transform != None:
            self.grdimage_transform = transform[0]
            self.satimage_transform = transform[1]

        if self.split == 'same':
            self.city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        elif self.split == 'cross':
            if self.train:
                self.city_list = ['NewYork', 'Seattle']
            else:
                self.city_list = ['SanFrancisco', 'Chicago']

        self.meter_per_pixel_dict = {'NewYork': 0.113248 * 640 / 512,
                                     'Seattle': 0.100817 * 640 / 512,
                                     'SanFrancisco': 0.118141 * 640 / 512,
                                     'Chicago': 0.111262 * 640 / 512}

        # load grd list
        self.grd_list = []
        self.depth_list = []
        idx = 0
        for city in self.city_list:
            # load grd panorama list
            if self.split == 'same':
                if self.train:
                    label_fname = os.path.join(self.root, self.label_root, city, 'same_area_balanced_train__corrected.txt')
                else:
                    label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test__corrected.txt')
            elif self.split == 'cross':
                label_fname = os.path.join(self.root, self.label_root, city, 'pano_label_balanced__corrected.txt')

            with open(label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    self.grd_list.append(os.path.join(self.root, city, 'pano_mask_sky', data[0]))
                    self.depth_list.append(os.path.join(self.root, city, 'depth_anywhere_same', data[0].replace('.jpg', '_depth.png')))
                    idx += 1
            print('InputData::__init__: load ', label_fname, idx)

        # TODO reopen
        # from sklearn.utils import shuffle
        # for rand_state in range(20):
        #     self.grd_list, self.label, self.delta = shuffle(self.grd_list, self.label, self.delta, random_state=rand_state)

        self.data_size = int(len(self.grd_list) * amount)
        self.grd_list = self.grd_list[: self.data_size]

        print('Grd loaded, data size:{}'.format(self.data_size))

        # deirection shape: [6, 224, 400, 3]
        self.direction = get_panorama_ray_directions(self.height, self.width)
        self.extrinsics = torch.tensor([[[ 1.,  0.,  0.,  0.],
                                        [ 0.,  1.,  0.,  0.],
                                        [ 0.,  0.,  1.,  0.],
                                        [ 0.,  0.,  0.,  1.]],

                                        [[ 0.,  0., -1.,  0.],
                                        [ 0.,  1.,  0.,  0.],
                                        [ 1.,  0.,  0.,  0.],
                                        [ 0.,  0.,  0.,  1.]],

                                        [[-1.,  0.,  0.,  0.],
                                        [ 0.,  1.,  0.,  0.],
                                        [-0.,  0., -1.,  0.],
                                        [ 0.,  0.,  0.,  1.]],

                                        [[ 0.,  0.,  1.,  0.],
                                        [ 0.,  1.,  0.,  0.],
                                        [-1.,  0.,  0.,  0.],
                                        [ 0.,  0.,  0.,  1.]],

                                        [[ 1.,  0.,  0.,  0.],
                                        [ 0.,  0.,  1.,  0.],
                                        [ 0., -1.,  0.,  0.],
                                        [ 0.,  0.,  0.,  1.]],

                                        [[ 1.,  0.,  0.,  0.],
                                        [ 0.,  0., -1.,  0.],
                                        [ 0.,  1.,  0.,  0.],
                                        [ 0.,  0.,  0.,  1.]]]
                                    )

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        # full ground panorama
        try:
            grd = PIL.Image.open(os.path.join(self.grd_list[idx]))
            grd = grd.convert('RGB')
        except:
            print('unreadable image')
            grd = PIL.Image.new('RGB', (self.height, self.width))  # if the image is unreadable, use a blank image
        grd = self.grdimage_transform(grd)

        try:
            depth = PIL.Image.open(os.path.join(self.depth_list[idx]))
            depth = depth.convert('L')
        except:
            print('unreadable image')
            depth = PIL.Image.new('L', (self.height, self.width))

        depth_img = self.grdimage_transform(depth).unsqueeze(0) * 35
        confs_img = torch.ones_like(depth_img, dtype=torch.float32)

        # grd.shape: [6, 3, 224, 400]
        grd = grd.unsqueeze(0)
        input_dict = {"rgb": grd}
        input_c2ws = torch.eye(4, 4, dtype=torch.float32).unsqueeze(0)

        input_rays_d = output_rays_d = self.direction.unsqueeze(0)
        input_rays_o = output_rays_o = torch.zeros(self.height, self.width, 3).unsqueeze(0)

        output_fovxs = torch.deg2rad(torch.tensor([90], dtype=torch.float32)).repeat(6)
        output_fovys = torch.deg2rad(torch.tensor([90], dtype=torch.float32)).repeat(6)
        
        ### input_dict_pix
        # depths_m.shape [6, 224, 400]
        # confs_m.shape [6, 224, 400]
        # ck.shape [6, 3, 3]
        # c2w.shape [6, 4, 4]
        # cx.shape [6] 200
        # cy.shape [6] 112
        # fx.shape [6]
        # fy.shape [6]
        # rays_o.shape [6, 224, 400, 3]
        # rays_d.shape [6, 224, 400, 3]

        input_dict_pix = {
            "depth_m": depth_img, 
            "conf_m": confs_img,
            "ck": torch.zeros(1,3,3), 
            "c2w": input_c2ws,
            "cx": torch.zeros(1), 
            "cy": torch.zeros(1), 
            "fx": torch.zeros(1), 
            "fy": torch.zeros(1),
            "rays_o": input_rays_o, 
            "rays_d": input_rays_d
        }
        
        ### input_dict_vol
        # w2i [6, 4, 4]
        # input_dict_vol = {"w2i": input_c2ws}
        input_dict_vol = {"w2i": self.extrinsics}
        ### output_dict
        # rgb [18, 3, 224, 400]
        # depth [18, 224, 400]
        # depths_m.shape [18, 224, 400]
        # conf_m.shape [18, 224, 400]
        # c2w.shape [18, 4, 4]
        # fx.shape [18]
        # fy.shape [18]
        # rays_o.shape [18, 224, 400, 3]
        # rays_d.shape [18, 224, 400, 3]
             
        # output_dict = {
        #     "rgb": grd, 
        #     "depth": depth_img,
        #     "depth_m": depth_img, 
        #     "conf_m": confs_img,
        #     "c2w": input_c2ws, 
        #     "fovx": output_fovxs, 
        #     "fovy": output_fovys, 
        #     "rays_o": output_rays_o,
        #     "rays_d": output_rays_d, 
        # }

        output_dict = {
            "rgb": grd, 
            "depth": depth_img,
            "depth_m": depth_img, 
            "conf_m": confs_img,
            "c2w": torch.inverse(self.extrinsics), 
            "fovx": output_fovxs, 
            "fovy": output_fovys, 
            "rays_o": output_rays_o,
            "rays_d": output_rays_d, 
        }
        

        return {
            "outputs": output_dict,
            "inputs": input_dict,
            "inputs_pix": input_dict_pix,
            "inputs_vol": input_dict_vol
        }

# ---------------------------------------------------------------------------------
class DistanceBatchSampler:
    def __init__(self, sampler, batch_size, drop_last, train_label):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.backup = []
        self.train_label = train_label

    def check_add(self, id_list, idx):
        '''
        id_list: a list containing grd image indexes we currently have in a batch
        idx: the grd image index to be determined where or not add to the current batch
        '''

        sat_idx = self.train_label[idx]
        for id in id_list:
            sat_id = self.train_label[id]
            for i in sat_id:
                if i in sat_idx:
                    return False

        return True

    def __iter__(self):
        batch = []

        for idx in self.sampler:

            if self.check_add(batch, idx):
                # add to batch
                batch.append(idx)

            else:
                # add to back up
                self.backup.append(idx)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

                remove = []
                for i in range(len(self.backup)):
                    idx = self.backup[i]

                    if self.check_add(batch, idx):
                        batch.append(idx)
                        remove.append(i)

                for i in sorted(remove, reverse=True):
                    self.backup.remove(self.backup[i])

        if len(batch) > 0 and not self.drop_last:
            yield batch
            print('batched all, left in backup:', len(self.backup))

    def __len__(self):

        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size



def load_vigor_data(batch_size, area="same", train=True, weak_supervise=True, amount=1.):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    transform_grd = transforms.Compose([
        transforms.Resize([pano_height, pano_width]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    transform_sat = transforms.Compose([
        # resize
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    vigor = VIGORDataset(root, split=area, train=train, transform=(transform_grd, transform_sat),
                         amount=amount)

    if train is True:
        index_list = np.arange(vigor.__len__())
        # np.random.shuffle(index_list)
        train_indices = index_list[0: int(len(index_list) * 0.98)]
        val_indices = index_list[int(len(index_list) * 0.98):]
        training_set = Subset(vigor, train_indices)
        val_set = Subset(vigor, val_indices)

        train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=False, num_workers=num_thread_workers)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_thread_workers)

        return train_dataloader, val_dataloader

    else:
        test_dataloader = DataLoader(vigor, batch_size=batch_size, shuffle=False, num_workers=num_thread_workers)

        return test_dataloader
