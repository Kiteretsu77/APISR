'''
    目前这个只是temporarly代替usm而已, 等这个方案确定以后再跟crop_images.py结合一下
'''

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


def crop_usm_subimages(input_dir, store_dir, crop_size: int):
    '''
        Crop sub-images for usm result only
    '''

    # Prepare
    output_index = 1

    # Iterate all frames under the folder
    for img_name in sorted(os.listdir(input_dir)):
        print(img_name)
        input_path = os.path.join(input_dir, img_name)

        # Read image 
        img = cv2.imread(input_path)
        height, width = img.shape[0:2]
        crop_num = (height//crop_size)*(width//crop_size)
        batch_num_each_row = (width//crop_size)

        # Prepare orders we need
        res_store = []
        for choice in range(crop_num):
            x, y = crop_size * (choice // batch_num_each_row), crop_size * (choice % batch_num_each_row)
            res_store.append((x, y))

        # Crop images now
        for (h, w) in res_store:
            cropped_img = img[h : h+crop_size,  w : w+crop_size,  ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(osp.join(store_dir, f'img_{output_index:06d}.png'), cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])     # Save in lossless mode

            output_index += 1





if __name__ == '__main__':
    random.seed(777) # We setup a random seed such that all program get the same cropped images 


    parser = argparse.ArgumentParser()
    # Try to split image after default
    parser.add_argument('-i', '--input_folder', type=str, default='datasets/all_Anime_hq_frames_resize', help='Input folder') # TODO: support multiple image input
    parser.add_argument('-o', '--save_folder', type=str, default='datasets/train_hr', help='Output folder')
    parser.add_argument('--size', type=int, default=256, help='Crop size')
    args = parser.parse_args()

    input_dir = args.input_folder   #"sharpen2_no_dilate_16_threshold/"
    store_dir = args.save_folder    #"datasets/train_hr_anime_GEASR_usm_16_threshold"
    crop_size = args.size           #256
    print(input_dir, store_dir, crop_size)
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir)


    crop_usm_subimages(input_dir, store_dir, crop_size)
    