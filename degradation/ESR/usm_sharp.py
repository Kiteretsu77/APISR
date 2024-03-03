# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import os, sys
root_path = os.path.abspath('.')
sys.path.append(root_path)
from degradation.ESR.utils import filter2D, np2tensor, tensor2np


def usm_sharp_func(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img



class USMSharp(torch.nn.Module):

    def __init__(self, type, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0).cuda()
        self.register_buffer('kernel', kernel)

        self.type = type


    def forward(self, img, weight=0.5, threshold=10, store=False):

        if self.type == "cv2":
            # pre-process cv2 type
            img = np2tensor(img)

        blur = filter2D(img, self.kernel.cuda())
        if store:
            cv2.imwrite("blur.png", tensor2np(blur))

        residual = img - blur
        if store:
            cv2.imwrite("residual.png", tensor2np(residual))

        mask = torch.abs(residual) * 255 > threshold
        if store:
            cv2.imwrite("mask.png", tensor2np(mask))


        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel.cuda())
        if store:
            cv2.imwrite("soft_mask.png", tensor2np(soft_mask))

        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        if store:
            cv2.imwrite("sharp.png", tensor2np(sharp))

        output =  soft_mask * sharp + (1 - soft_mask) * img
        if self.type == "cv2":
            output = tensor2np(output)
        
        return output
    


if __name__ == "__main__":

    usm_sharper = USMSharp(type="cv2")
    img = cv2.imread("sample3.png")
    print(img.shape)
    sharp_output = usm_sharper(img, store=False, threshold=10)
    cv2.imwrite(os.path.join("output.png"), sharp_output)


    # dir = r"C:\Users\HikariDawn\Desktop\Real-CUGAN\datasets\sample"
    # output_dir = r"C:\Users\HikariDawn\Desktop\Real-CUGAN\datasets\sharp_regular"
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    # for file_name in sorted(os.listdir(dir)):
    #     print(file_name)
    #     file = os.path.join(dir, file_name)
    #     img = cv2.imread(file)
    #     sharp_output = usm_sharper(img)
    #     cv2.imwrite(os.path.join(output_dir, file_name), sharp_output)
