# -*- coding: utf-8 -*-

'''
    From ESRGAN
'''


import os, sys
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from scipy import special
import random
import math
from torchvision.utils import make_grid

from degradation.ESR.degradations_functionality import *

root_path = os.path.abspath('.')
sys.path.append(root_path)


def np2tensor(np_frame):
    return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).cuda().float()/255

def tensor2np(tensor):
    # tensor should be batch size1 and cannot be grayscale input
    return (np.transpose(tensor.detach().squeeze(0).cpu().numpy(), (1, 2, 0))) * 255

def mass_tensor2np(tensor):
    ''' The input tensor is massive tensor
    '''
    return (np.transpose(tensor.detach().squeeze(0).cpu().numpy(), (0, 2, 3, 1))) * 255

def save_img(tensor, save_name):
    np_img = tensor2np(tensor)[:,:,16]
    # np_img = np.expand_dims(np_img, axis=2)
    cv2.imwrite(save_name, np_img)


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)
    

def generate_kernels(opt):

    kernel_range = [2 * v + 1 for v in range(opt["kernel_range"][0], opt["kernel_range"][1])] 

    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < opt['sinc_prob']:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            opt['kernel_list'],
            opt['kernel_prob'],
            kernel_size,
            opt['blur_sigma'],
            opt['blur_sigma'], [-math.pi, math.pi],
            opt['betag_range'],
            opt['betap_range'],
            noise_range=None)
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    
    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < opt['sinc_prob2']:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            opt['kernel_list2'],
            opt['kernel_prob2'],
            kernel_size,
            opt['blur_sigma2'],
            opt['blur_sigma2'], [-math.pi, math.pi],
            opt['betag_range2'],
            opt['betap_range2'],
            noise_range=None)

    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
    
    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)
    return (kernel, kernel2)


