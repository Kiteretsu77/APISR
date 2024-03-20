# -*- coding: utf-8 -*-
import argparse
import cv2
import torch
import os, shutil, time
import sys
from multiprocessing import Process, Queue
from os import path as osp
from tqdm import tqdm
import copy
import warnings
import gc

warnings.filterwarnings("ignore")

# import same folder files #
root_path = os.path.abspath('.')
sys.path.append(root_path)
from degradation.ESR.utils import np2tensor
from degradation.ESR.degradations_functionality import *
from degradation.degradation_esr import degradation_v1
from opt import opt
os.environ['CUDA_VISIBLE_DEVICES'] = opt['CUDA_VISIBLE_DEVICES']  #'0,1'



def crop_process(path, crop_size, lr_dataset_path, output_index):
    ''' crop the image here (also do usm here)
    Args:
        path (str):             Path of the image
        crop_size (int):        Crop size
        lr_dataset_path (str):  LR dataset path folder name
        output_index (int):     The index we used to store images
    Returns:
        output_index (int):     The next index we need to use to store images
    '''

    # read image 
    img = cv2.imread(path)
    height, width = img.shape[0:2]

    res_store = []
    crop_num = (height//crop_size)*(width//crop_size)

    # Use shift offset to make image more cover origional image size
    shift_offset_h, shift_offset_w = 0, 0 


    # Select all sub-frames order by order (not randomly select here)
    choices = [i for i in range(crop_num)]
    shift_offset_h = 0 #random.randint(0, height - crop_size * (height//crop_size))
    shift_offset_w = 0 #random.randint(0, width - crop_size * (width//crop_size))


    for choice in choices:
        row_num = (width//crop_size)
        x, y = crop_size * (choice // row_num), crop_size * (choice % row_num)
        # add offset
        res_store.append((x, y))

        

    for (h, w) in res_store:
        cropped_img = img[h+shift_offset_h : h+crop_size+shift_offset_h, w+shift_offset_w : w+crop_size+shift_offset_w, ...]
        cropped_img = np.ascontiguousarray(cropped_img)
        cv2.imwrite(osp.join(lr_dataset_path, f'img_{output_index:06d}.png'), cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])     # Save in lossless mode

        output_index += 1

    return output_index



def single_process(queue, opt, process_id):
    ''' Multi Process instance
    Args:
        queue (multiprocessing.Queue):      The input queue
        opt (dict):                         The setting we need to use
        process_id (int):                   The id we used to store temporary file
    '''

    # Initialization
    obj_img = degradation_v1()

    while True:
        items = queue.get()
        if items == None:
            break
        input_path, store_path = items

        # Reset kernels in every degradation batch for ESR
        obj_img.reset_kernels(opt)
        
        # Read all images and transform them to tensor
        img_bgr = cv2.imread(input_path)

        out = np2tensor(img_bgr) # tensor

        # ESR Degradation execution
        obj_img.degradate_process(out, opt, store_path, process_id, verbose = False)



@torch.no_grad()
def generate_low_res_esr(org_opt, verbose=False):
    ''' Generate LR dataset from HR ones by ESR degradation
    Args:
        org_opt (dict):     The setting we will use
        verbose (bool): Whether we print out some information
    '''

    # Prepare folders and files
    input_folder = org_opt['input_folder']
    save_folder = org_opt['save_folder']
    if osp.exists(save_folder):
        shutil.rmtree(save_folder)
    if osp.exists("tmp"):
        shutil.rmtree("tmp")
    os.makedirs(save_folder)
    os.makedirs("tmp")
    if os.path.exists("datasets/degradation_log.txt"):
        os.remove("datasets/degradation_log.txt")


    # Scan all images
    input_img_lists, output_img_lists = [], []
    for file in sorted(os.listdir(input_folder)):       
        input_img_lists.append(osp.join(input_folder, file))
        output_img_lists.append(osp.join("tmp", file))
    assert(len(input_img_lists) == len(output_img_lists))


    # Multi-Process Preparation
    parallel_num = opt['parallel_num']
    queue = Queue()


    # Save all files in the Queue
    for idx in range(len(input_img_lists)):     
        # Find the needed img lists
        queue.put((input_img_lists[idx], output_img_lists[idx]))


    # Start the process
    Processes = []
    for process_id in range(parallel_num):
        p1 = Process(target=single_process, args =(queue, opt, process_id, ))
        p1.start()
        Processes.append(p1)
    for _ in range(parallel_num):
        queue.put(None) # Used to end the process
    # print("All Process starts")

    # tqdm wait progress
    for idx in tqdm(range(0, len(output_img_lists)), desc ="Degradation"):
        while True:
            if os.path.exists(output_img_lists[idx]):
                break
            time.sleep(0.1)

    # Merge all processes
    for process in Processes:
        process.join()



    # Crop images under folder "tmp"
    output_index = 1
    for img_name in sorted(os.listdir("tmp")):
        path = os.path.join("tmp", img_name)
        output_index = crop_process(path, opt['hr_size']//opt['scale'], opt['save_folder'], output_index)
        

        
def main(args):
    opt['input_folder'] = args.input
    opt['save_folder'] = args.output

    generate_low_res_esr(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default = opt["full_patch_source"], help='Input folder')
    parser.add_argument('--output', type=str, default = opt["lr_dataset_path"], help='Output folder')
    args = parser.parse_args()

    main(args)