import torch, sys, os, random
import torch.nn.functional as F
import numpy as np
import cv2
from multiprocessing import Process, Queue
from PIL import Image

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder
from opt import opt
from degradation.ESR.utils import tensor2np, np2tensor




class WEBP():
    def __init__(self) -> None:
        # Choose an image compression degradation
        pass

    def compress_and_store(self, np_frames, store_path, idx):
        ''' Compress and Store the whole batch as WebP (~ VP8)
        Args:
            np_frames (numpy):      The numpy format of the data (Shape:?)
            store_path (str):       The store path      
        Return:
            None
        '''
        single_frame = np_frames

        # Choose the quality
        quality = random.randint(*opt['webp_quality_range2'])
        method = random.randint(*opt['webp_encode_speed2'])

        # Transform to PIL and then compress
        PIL_image = Image.fromarray(np.uint8(single_frame[...,::-1])).convert('RGB')
        PIL_image.save(store_path, 'webp', quality=quality, method=method)

            
    @staticmethod
    def compress_tensor(tensor_frames, idx = 0):
        ''' Compress tensor input to WEBP and then return it
        Args:
            tensor_frame (tensor):  Tensor inputs    
        Returns:
            result (tensor):        Tensor outputs (same shape as input)
        '''
        single_frame = tensor2np(tensor_frames)

        # Choose the quality
        quality = random.randint(*opt['webp_quality_range1'])
        method = random.randint(*opt['webp_encode_speed1'])

        # Transform to PIL and then compress
        PIL_image = Image.fromarray(np.uint8(single_frame[...,::-1])).convert('RGB')
        store_path = os.path.join("tmp", "temp_"+str(idx)+".webp")
        PIL_image.save(store_path, 'webp', quality=quality, method=method)

        # Read back
        decimg = cv2.imread(store_path)
        result = np2tensor(decimg)
        os.remove(store_path)

        return result