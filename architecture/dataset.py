# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor

import numpy as np
import cv2
import glob
import random
from PIL import Image
from tqdm import tqdm


# from degradation.degradation_main import degredate_process, preparation
from opt import opt


class ImageDataset(Dataset):
    @torch.no_grad()
    def __init__(self, train_lr_paths, degrade_hr_paths, train_hr_paths):
        # print("low_res path sample is ", train_lr_paths[0])
        # print(train_hr_paths[0])
        # hr_height, hr_width = hr_shape
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.files_lr = train_lr_paths
        self.files_degrade_hr = degrade_hr_paths
        self.files_hr = train_hr_paths

        assert(len(self.files_lr) == len(self.files_hr))
        assert(len(self.files_lr) == len(self.files_degrade_hr))


    def augment(self, imgs, hflip=True, rotation=True):
        """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

        All the images in the list use the same augmentation.

        Args:
            imgs (list[ndarray] | ndarray): Images to be augmented. If the input
                is an ndarray, it will be transformed to a list.
            hflip (bool): Horizontal flip. Default: True.
            rotation (bool): Rotation. Default: True.

        Returns:
            imgs (list[ndarray] | ndarray): Augmented images and flows. If returned
                results only have one element, just return ndarray.

        """
        hflip = hflip and random.random() < 0.5
        vflip = rotation and random.random() < 0.5
        rot90 = rotation and random.random() < 0.5

        def _augment(img):
            if hflip:  # horizontal
                cv2.flip(img, 1, img)
            if vflip:  # vertical
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0, 2)
            return img


        if not isinstance(imgs, list):
            imgs = [imgs]
        
        imgs = [_augment(img) for img in imgs]
        if len(imgs) == 1:
            imgs = imgs[0]


        return imgs
            

    def __getitem__(self, index):
        
        # Read File
        img_lr = cv2.imread(self.files_lr[index % len(self.files_lr)]) # Should be BGR
        img_degrade_hr = cv2.imread(self.files_degrade_hr[index % len(self.files_degrade_hr)]) 
        img_hr = cv2.imread(self.files_hr[index % len(self.files_hr)])

        # Augmentation
        if random.random() < opt["augment_prob"]:
            img_lr, img_degrade_hr, img_hr = self.augment([img_lr, img_degrade_hr, img_hr])
        
        # Transform to Tensor
        img_lr = self.transform(img_lr)
        img_degrade_hr = self.transform(img_degrade_hr)
        img_hr = self.transform(img_hr)  # ToTensor() is already in the range [0, 1]


        return {"lr": img_lr, "degrade_hr": img_degrade_hr, "hr": img_hr}
    
    def __len__(self):
        assert(len(self.files_hr) == len(self.files_lr))
        return len(self.files_hr)
