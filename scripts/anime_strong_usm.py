import cv2
import argparse
import numpy as np
import copy
import os, sys, copy, shutil
from kornia import morphology as morph
import math
import gc, time
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F
from multiprocessing import set_start_method

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from degradation.ESR.utils import filter2D, np2tensor, tensor2np



# This config is found by the author
# modify if not the desired output
XDoG_config = dict(
    size=0,
    sigma=0.6,
    eps=-15,
    phi=10e8,
    k=2.5,
    gamma=0.97
)

# I wanted the gamma between [0.97, 0.98], but it depends on the image so I made it move randomly comment out if this is not needed
# In our case, black means background information; white means hand-drawn line
XDoG_config['gamma'] += 0.01 * np.random.rand(1)
dilation_kernel = torch.tensor([[1, 1, 1],[1, 1, 1],[1, 1, 1]]).cuda()
white_color_value = 1       # In binary map, 0 stands for black and 1 stands for white



def DoG(image, size, sigma, k=1.6, gamma=1.):
    g1 = cv2.GaussianBlur(image, (size, size), sigma)
    g2 = cv2.GaussianBlur(image, (size, size), sigma*k)
    return g1 - gamma * g2


def XDoG(image, size, sigma, eps, phi, k=1.6, gamma=1.):
    eps /= 255
    d = DoG(image, size, sigma, k, gamma)
    d /= d.max()
    e = 1 + np.tanh(phi * (d - eps))
    e[e >= 1] = 1
    return e



class USMSharp(torch.nn.Module):
    '''
        Basically, the same as Real-ESRGAN
    '''

    def __init__(self, type, radius=50, sigma=0):
        # 感觉radius有点大
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0).cuda()
        self.register_buffer('kernel', kernel)

        self.type = type


    def forward(self, img, weight=0.5, threshold=10, store=False):
        #  weight=0.5, threshold=10

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



def get_xdog_sketch_map(img_bgr, outlier_threshold):
        
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sketch_map = gen_xdog_image(gray, outlier_threshold)
    sketch_map = np.stack((sketch_map, sketch_map, sketch_map), axis=2) # concatenate to 3 dim

    return np.uint8(sketch_map)


def process_single_img(queue, usm_sharper, extra_sharpen_time, outlier_threshold):

    counter = 0
    while True:
        counter += 1
        if counter == 10:
            counter = 0
            gc.collect()
            print("We will sleep here to clear memory")
            time.sleep(5)
        info = queue[0]
        queue = queue[1:]
        if info == None:
            break

        img_dir, store_path = info
        print("We are processing ", img_dir)
        img = cv2.imread(img_dir)

        img = usm_sharper(img, store=False, threshold=10)
        first_sharpened_img = copy.deepcopy(img)

        for _ in range(extra_sharpen_time): 
            # sketch_map = get_xdog_sketch_map(img_temp) 
            img = usm_sharper(img, store=False, threshold=10)
            # img = (sharpened_img * sketch_map) + (org_img * (1-sketch_map))

        sketch_map = get_xdog_sketch_map(img, outlier_threshold)
        img = (img * sketch_map) + (first_sharpened_img * (1-sketch_map))  

        
        cv2.imwrite(store_path, img)

    print("Finish all program")



def outlier_removal(img, outlier_threshold):
    ''' Remove outlier pixel after finding the sketch
    Here, black(0) means background information; white(1) means hand-drawn line
    '''

    global_list = set()
    h,w = img.shape

    def dfs(i, j):
        '''
            Using Depth First Search to find the full area of mapping
        '''
        if (i,j) in visited:
            # If this is an already visited pixel, return
            return
        
        if (i,j) in global_list:
            # If it is already existed in the global list, return 
            return

        if i >= h or j >= w or i < 0 or j < 0:
            # If it is out of boundary, return
            return
        
        if img[i][j] == white_color_value:
            visited.add((i,j))

            # If it is over threshold, we won't remove them
            if len(visited) >= 100:
                return
        
            dfs(i+1, j) 
            dfs(i, j+1) 
            dfs(i-1, j) 
            dfs(i, j-1) 
            dfs(i-1, j-1) 
            dfs(i+1, j+1) 
            dfs(i-1, j+1) 
            dfs(i+1, j-1) 

        return
    
    def bfs(i, j):
        '''
            Using Breadth First Search to find the full area of mapping
        '''
        if (i,j) in visited:
            # If this is an already visited pixel, return
            return
        
        if (i,j) in global_list:
            # If it is already existed in the global list, return 
            return

        visited.add((i,j))
        if img[i][j] != white_color_value:
            return
        
        queue = [(i, j)]
        while queue:
            base_row, base_col = queue.pop(0)

            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                row, col = base_row+dx, base_col+dy

                if (row, col) in visited:
                    # If this is an already visited pixel, continue
                    continue
                
                if (row, col) in global_list:
                    # If it is already existed in the global list, continue 
                    continue

                if row >= h or col >= w or row < 0 or col < 0:
                    # If it is out of boundary, continue
                    continue
                
                if img[row][col] == white_color_value:
                    visited.add((row, col))
                    queue.append((row, col))

        
    temp = np.copy(img)
    for i in range(h):
        for j in range(w):
            if (i,j) in global_list:
                continue
            if temp[i][j] != white_color_value:
                # We only consider white color (hand-drawn line) situation
                continue

            global visited
            visited = set()
            
            # USE depth/breadth first search to find neighbor white value
            bfs(i, j)

            if len(visited) < outlier_threshold:
                # If the number of white pixels counting all neighbors are less than the outlier_threshold, paint the whole region to black (0：background symbol)
                for u, v in visited:
                    temp[u][v] = 0
            
            # Add those searched line to global_list to speed up
            for u, v in visited:
                global_list.add((u, v))

    return temp


def active_dilate(img):
    def np2tensor(np_frame):
        return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).cuda().float()/255
    def tensor2np(tensor):
        # tensor should be batch size1 and cannot be grayscale input
        return (np.transpose(tensor.detach().cpu().numpy(), (1, 2, 0))) * 255
    
    dilated_edge_map = morph.dilation(np2tensor(np.expand_dims(img, 2)), dilation_kernel)

    return tensor2np(dilated_edge_map[0]).squeeze(2)


def passive_dilate(img):
    # IF there is 3 white pixel in 9 block, we will fill in
    h,w = img.shape

    def detect_fill(i, j):
        if img[i][j] == white_color_value:
            return False
        
        def sanity_check(i, j):
            if i >= h or j >= w or i < 0 or j < 0:
                return False
            
            if img[i][j] == white_color_value:
                return True
            return False


        num_white = sanity_check(i-1,j-1) + sanity_check(i-1,j) + sanity_check(i-1,j+1) + sanity_check(i,j-1) + sanity_check(i,j+1) + sanity_check(i+1,j-1) + sanity_check(i+1,j) + sanity_check(i+1,j+1)
        if num_white >= 3:
            return True
        

    temp = np.copy(img)
    for i in range(h):
        for j in range(w):
            global visited
            visited = set()
            
            should_fill = detect_fill(i, j)
            if should_fill:
                temp[i][j] = 1

    # return True to say that we need to remove it; else, we don't need to remove it
    return temp


def gen_xdog_image(gray, outlier_threshold):
    '''
    Returns:
        dogged (numpy):     binary map in range (1 stands for white pixel)
    '''
    
    dogged = XDoG(gray, **XDoG_config)
    dogged = 1 - dogged   # black white transform


    # Remove unnecessary outlier
    dogged = outlier_removal(dogged, outlier_threshold)

    # Dilate the image
    dogged = passive_dilate(dogged)

    
    return dogged



if __name__ == "__main__":


    # Parse variables available
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type = str)
    parser.add_argument('-o', '--store_dir', type = str)
    parser.add_argument('--outlier_threshold', type = int, default=32)
    parser.add_argument('--num_workers', type = int, default=6)
    args = parser.parse_args()

    input_dir = args.input_dir
    store_dir = args.store_dir
    outlier_threshold = args.outlier_threshold
    num_workers = args.num_workers

    print("We are handling Strong USM sharpening on hand-drawn line for Anime images!")


    extra_sharpen_time = 2


    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir)


    dir_list = []
    for img_name in sorted(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(store_dir, img_name)
        dir_list.append((input_path, output_path))
    
    length = len(dir_list)
    

    # USM sharpener preparation
    usm_sharper = USMSharp(type="cv2").cuda()
    usm_sharper.share_memory()

    for idx in range(num_workers):
        set_start_method('spawn', force=True)

        num = math.ceil(length / num_workers)
        request_list = dir_list[:num]
        request_list.append(None)
        dir_list = dir_list[num:]

        # process_single_img(request_list, usm_sharper, extra_sharpen_time)   # This is for debug purpose
        p = mp.Process(target=process_single_img, args=(request_list, usm_sharper, extra_sharpen_time, outlier_threshold))
        p.start()

    print("Submitted all jobs!")