import torch
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
import random
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image

class SmartphoneLoader(Dataset):

    def __init__(self, root):
        self.input_size = (504, 378)
        self.center_crop = (336, 252)
        self.rand_crop = (224, 224)
        self.cropping = (
            self.center_crop[0] - self.rand_crop[0], self.center_crop[1] - self.rand_crop[1])
        self.indexes = np.rint(np.linspace(
            0, 48, 10, endpoint=True)).astype(int)
        self.focus_dists = []
        # https://storage.googleapis.com/cvpr2020-af-data/LearnAF%20Dataset%20Readme.pdf
        focus_dists = [3910.92, 2289.27, 1508.71, 1185.83, 935.91, 801.09, 700.37, 605.39, 546.23, 486.87, 447.99, 407.40, 379.91, 350.41, 329.95, 307.54,
                       291.72, 274.13, 261.53, 247.35, 237.08, 225.41, 216.88, 207.10, 198.18, 191.60, 183.96, 178.29, 171.69, 165.57, 160.99, 155.61, 150.59, 146.81,
                       142.35, 138.98, 134.99, 131.23, 127.69, 124.99, 121.77, 118.73, 116.40, 113.63, 110.99, 108.47, 106.54, 104.23, 102.01]
        for index in self.indexes:
            self.focus_dists.append(focus_dists[index])
        self.focus_dists = np.expand_dims(self.focus_dists, axis=1)
        self.focus_dists = np.expand_dims(
            self.focus_dists, axis=2).astype(np.float32)
        self.focus_dists = self.focus_dists*0.001
        self.focus_dists = torch.Tensor(
            np.tile(self.focus_dists, [1, self.center_crop[0]+16, self.center_crop[1]+4]))
        self.focus_dists = 1/self.focus_dists
        self.max_depth = 1/0.10201
        self.min_depth = 1/3.91092
        self.root = root
        self.depths = []
        self.confids = []
        self.FS = []
        
        self.mean_input= [0.485, 0.456, 0.406]
        self.std_input=[0.229, 0.224, 0.225]

        path = self.root + 'test' + '/'
        scenes = os.listdir(path+'scaled_images/')
        for scene in scenes:
            self.depths.append(path + 'merged_depth/' +
                               scene + '/' + 'result_merged_depth_center.png')
            self.confids.append(path + 'merged_conf/' +
                                scene + '/' + 'result_merged_conf_center.exr')
            FS_imgs = []
            for j in self.indexes:
                FS_imgs.append(path + 'scaled_images/' + scene +
                               '/' + str(j) + '/result_scaled_image_center.jpg')
            self.FS.append(FS_imgs)

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        # Create sample dict
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        # base_focal_length = self.FS_focal_length[idx][48]
        FS = np.zeros(
            (self.center_crop[0], self.center_crop[1], 10, 3), dtype=np.float32)
        for i in range(0, 10):
            img = cv2.imread(self.FS[idx][i]).astype(np.float32)[:, :, :]
            img = img/255.0
            FS[:, :, i, :] = img[84:-84, 63:-63, :].astype(np.float32)
        img = cv2.imread(self.FS[idx][9]).astype(np.float32)[:, :, :]
        img = img/255.0
        FS[:, :, 9, :] = img[84:-84, 63:-63, :].astype(np.float32)

        gt = cv2.imread(self.depths[idx], cv2.IMREAD_UNCHANGED).astype(
            np.float32)[84:-84, 63:-63]
        gt = gt/255.0
        gt = (20)/(100-(100-0.2)*gt)
        gt = 1/gt
        conf = cv2.imread(self.confids[idx], cv2.IMREAD_UNCHANGED)[
            84:-84, 63:-63, -1]
        conf[conf > 1.0] = 1.0
        # FS = FS/127.5 - 1.0
        gt[gt < self.min_depth] = 0.0
        gt[gt > self.max_depth] = 0.0
        mask = torch.from_numpy(np.where(gt == 0.0, 0., 1.).astype(np.bool_))

        FS = torch.from_numpy(np.transpose(FS, (3, 2, 0, 1)))

        N, C, H, W = FS.shape
        if H % 32 != 0:
            pad_h = 32 - (H % 32)
        else:
            pad_h = 0
        if W % 32 != 0:
            pad_w = 32 - (W % 32)
        else:
            pad_w = 0
        FS = F.pad(torch.Tensor(FS), (0, pad_w, 0, pad_h),
                   'reflect')  # top 4 padding
        mask = F.pad(mask, (0, pad_w, 0, pad_h), 'constant', 0)
        gt = F.pad(torch.Tensor(gt), (0, pad_w, 0, pad_h), 'constant', 0)
        
        # FS torch.Size([3, 10, 352, 256]) to torch.Size([10, 3, 352, 256])
        FS = FS.permute(1, 0, 2, 3)
        
        normalize = transforms.Normalize(self.mean_input, self.std_input)
        FS = normalize(FS)

        # select one as the rgb image out of FS
        rgb = FS[0, :, :, :]

        # convert focus_dists to list from torch.Size([10, 352, 256]) to torch.Size([10])
        if self.focus_dists.shape == torch.Size([10, 352, 256]):
            self.focus_dists = self.focus_dists[:, 0, 0]

        # gt = torch.from_numpy(gt)
        
        # rgb = rgb / 255.0
        gt = gt.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return rgb, FS, gt, self.focus_dists, mask, conf

    def get_seeds(self):
        return (random.randint(0, self.cropping[0]-1), random.randint(0, self.cropping[1]-1), random.uniform(0.4, 1.6), random.uniform(-0.1, 0.1), random.uniform(0.5, 2.0), random.uniform(0, 1.0), random.uniform(0, 1.0), random.randint(0, 3))
