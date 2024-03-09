# -*- coding: utf-8 -*-

import argparse
import cv2
import torch
import numpy as np
import os, shutil, time
import sys, random
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from math import log10, sqrt
import torch.nn.functional as F

root_path = os.path.abspath('.')
sys.path.append(root_path)
from degradation.ESR.degradations_functionality import *
from degradation.ESR.diffjpeg import *
from degradation.ESR.utils import filter2D
from degradation.image_compression.jpeg import JPEG
from degradation.image_compression.webp import WEBP
from degradation.image_compression.heif import HEIF
from degradation.image_compression.avif import AVIF
from opt import opt


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr



def downsample_1st(out, opt):
    # Resize with different mode
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(opt['resize_options'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)

    return out


def downsample_2nd(out, opt, ori_h, ori_w):
    # Second Resize for 4x scaling
    if opt['scale'] == 4:
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(opt['resize_options'])
        # Resize这边改回来原来的版本，不用连续的resize了
        # out = F.interpolate(out, scale_factor=scale, mode=mode)
        out = F.interpolate(
            out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode
        )
    
    return out


def common_degradation(out, opt, kernels, process_id, verbose = False):
    jpeger = DiffJPEG(differentiable=False).cuda()
    kernel1, kernel2 = kernels


    downsample_1st_position = random.choices([0, 1, 2])[0]
    if opt['scale'] == 4:
        # Only do the second downsample at 4x scale
        downsample_2nd_position = random.choices([0, 1, 2])[0]
    else:
        # print("We don't use the second resize")
        downsample_2nd_position = -1

        
    ####---------------------------- Frist Degradation ----------------------------------####
    batch_size, _, ori_h, ori_w = out.size()

    if downsample_1st_position == 0:
        out = downsample_1st(out, opt)

    # Bluring kernel
    out = filter2D(out, kernel1)
    if verbose: print(f"(1st) blur noise")


    if downsample_1st_position == 1:
        out = downsample_1st(out, opt)


    # Noise effect (gaussian / poisson)
    gray_noise_prob = opt['gray_noise_prob']
    if np.random.uniform() < opt['gaussian_noise_prob']:
        # Gaussian noise
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        name = "gaussian_noise"
    else:
        # Poisson noise
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
        name = "poisson_noise"
    if verbose: print("(1st) " + str(name))


    if downsample_1st_position == 2:
        out = downsample_1st(out, opt)


    # Choose an image compression codec (All degradation batch use the same codec)
    image_codec = random.choices(opt['compression_codec1'], opt['compression_codec_prob1'])[0]     # All lower case
    if image_codec == "jpeg":
        out = JPEG.compress_tensor(out)
    elif image_codec == "webp":
        try:
            out = WEBP.compress_tensor(out, idx=process_id)
        except Exception:
            print("There is exception again in webp!")
            out = WEBP.compress_tensor(out, idx=process_id)
    elif image_codec == "heif":
        out = HEIF.compress_tensor(out, idx=process_id)
    elif image_codec == "avif":
        out = AVIF.compress_tensor(out, idx=process_id)
    else:
        raise NotImplementedError("We don't have such image compression designed!")
    # ##########################################################################################


    # ####---------------------------- Second Degradation ----------------------------------####
    if downsample_2nd_position == 0:
        out = downsample_2nd(out, opt, ori_h, ori_w)


    # Add blur 2nd time
    if np.random.uniform() < opt['second_blur_prob']:
        # 这个bluring不是必定触发的
        if verbose: print("(2nd) blur noise")
        out = filter2D(out, kernel2)


    if downsample_2nd_position == 1:
        out = downsample_2nd(out, opt, ori_h, ori_w)


    # Add noise 2nd time
    gray_noise_prob = opt['gray_noise_prob2']
    if np.random.uniform() < opt['gaussian_noise_prob2']:
        # gaussian noise
        if verbose: print("(2nd) gaussian noise")
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        name = "gaussian_noise"
    else:
        # poisson noise
        if verbose: print("(2nd) poisson noise")
        out = random_add_poisson_noise_pt(
            out, scale_range=opt['poisson_scale_range2'], gray_prob=gray_noise_prob, clip=True, rounds=False)
        name = "poisson_noise"


    if downsample_2nd_position == 2:
        out = downsample_2nd(out, opt, ori_h, ori_w)


    return out