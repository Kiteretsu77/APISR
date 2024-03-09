# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import os, shutil
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import random
from collections import namedtuple

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from degradation.ESR.usm_sharp import USMSharp


class worker:
    def __init__(self, start_index=1):
        # The index you want to start with 
        self.output_index = start_index 
            
    def process(self, path, opt, usm_sharper):
        ''' crop the image here (also do usm here)
        Args:
            path (str):     path of the image
            opt (dict):     all setting in a dictionary
            usm_sharper (class): usm sharpener
        
        Returns:
            cropped_num (int): how many cropped images you have for this path
        '''

        crop_size = opt['crop_size'] # usually 400

        # read image 
        img = cv2.imread(path)
        height, width = img.shape[0:2]

        res_store = []
        crop_num = (height//crop_size)*(width//crop_size)
        random_num = opt['crop_num_per_img']

        # Use shift offset to make image more cover origional image size
        shift_offset_h, shift_offset_w = 0, 0 

        if random_num == -1:
            # We should select all sub-frames order by order (not randomly select here)
            choices = [i for i in range(crop_num)]
            shift_offset_h = 0  #random.randint(0, height - crop_size * (height//crop_size))
            shift_offset_w = 0  #random.randint(0, width - crop_size * (width//crop_size))
        else:
            # Divide imgs by crop_size x crop_size and choose opt['crop_num_per_img'] num of them to avoid overlap
            num = min(random_num, crop_num)  
            choices = random.sample(range(crop_num), num)

        for choice in choices:
            row_num = (width//crop_size)
            x, y = crop_size * (choice // row_num), crop_size * (choice % row_num)
            # add offset
            res_store.append((x, y))

        
        # Sharp the image before selection
        if opt['usm_save_folder'] != None:
            sharpened_img = usm_sharper(img)
            

        for (h, w) in res_store:
            cropped_img = img[h+shift_offset_h : h+crop_size+shift_offset_h, w+shift_offset_w : w+crop_size+shift_offset_w, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(osp.join(opt['save_folder'], f'img_{self.output_index:06d}.png'), cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])     # Save in lossless mode

            # store the sharpened cropped image
            if opt['usm_save_folder'] != None:
                cropped_sharpened_img = sharpened_img[h+shift_offset_h : h+crop_size+shift_offset_h, w+shift_offset_w : w+crop_size+shift_offset_w, ...]
                cropped_sharpened_img = np.ascontiguousarray(cropped_sharpened_img)
                cv2.imwrite(osp.join(opt['usm_save_folder'], f'img_{self.output_index:06d}.png'), cropped_sharpened_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            self.output_index += 1


        cropped_num = len(res_store)
        return cropped_num


def extract_subimages(opt):

    # Input
    input_folders = opt['input_folders']

    # Make folders
    save_folder = opt['save_folder']
    usm_save_folder = opt['usm_save_folder']

    if osp.exists(save_folder):
        print(f'Folder {save_folder} already exists. Program will delete this folder!')
        shutil.rmtree(save_folder)

    os.makedirs(save_folder)
    if usm_save_folder != None:
        if osp.exists(usm_save_folder):
            print(f'Folder {usm_save_folder} already exists. Program will delete this folder!')
            shutil.rmtree(usm_save_folder)

        print("Use usm sharp")
        os.makedirs(usm_save_folder)

    # USM
    usm_sharper = USMSharp(type="cv2")

    # Iterate all datasets' folders
    start_index = 1
    for input_folder in input_folders:
        print(input_folder, start_index)

        # Scan all images
        img_list = []
        for file in sorted(os.listdir(input_folder)):
            if file.split(".")[-1] in ["png", "jpg"]:
                img_list.append(osp.join(input_folder, file))

        # Iterate can crop
        obj = worker(start_index=start_index)     # The start_index determines where you will start your naming your image (usually start from 0)
        for path in img_list:
            if random.random() < opt['select_rate']:
                cropped_num = obj.process(path, opt, usm_sharper)
                start_index += cropped_num
                print(start_index, path)
            else:
                print("SKIP")


    print('All processes done.')


def main(args):
    opt = {}

    input_folders = []
    if type(args.input_folder) == str:
        input_folders.append(args.input_folder)
    else:
        for input_folder in args.input_folder:
            input_folders.append(input_folder)
    print("input folders have ", input_folders)


    opt['input_folders'] = input_folders
    opt['save_folder'] = args.save_folder
    opt['usm_save_folder'] = args.output_usm
    opt['crop_size'] = args.crop_size
    opt['crop_num_per_img'] = args.crop_num_per_img
    opt['select_rate'] = args.select_rate

    # Extract subimages
    extract_subimages(opt)


if __name__ == '__main__':
    random.seed(777) # We setup a random seed such that all program get the same cropped images 

    parser = argparse.ArgumentParser()
    # Try to split image after default
    parser.add_argument('-i', '--input_folder', nargs='+', type=str, default='datasets/all_Anime_hq_frames_resize', help='Input folder') # TODO: support multiple image input
    parser.add_argument('-o', '--save_folder', type=str, default='datasets/train_hr', help='Output folder')
    parser.add_argument('--output_usm', type=str, help='usm sharpened hr folder')
    parser.add_argument('--crop_size', type=int, default=360, help='Crop size')
    parser.add_argument('--select_rate', type=float, default=1, help='(0-1): Proportion to keep; 1 means to keep them all')
    parser.add_argument('--crop_num_per_img', type=int, default=-1, help='Crop size (int); -1 means use all possible sub-frames')  
    args = parser.parse_args()

    main(args)