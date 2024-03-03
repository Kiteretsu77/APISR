import torch, sys, os, random
import torch.nn.functional as F
import numpy as np
import cv2
from multiprocessing import Process, Queue
from PIL import Image
from pillow_heif import register_heif_opener
import pillow_heif

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder
from opt import opt
from degradation.ESR.utils import tensor2np, np2tensor




class HEIF():
    def __init__(self) -> None:
        # Choose an image compression degradation
        pass

    def compress_and_store(self, np_frames, store_path):
        ''' Compress and Store the whole batch as HEIF (~ HEVC)
        Args:
            np_frames (numpy):      The numpy format of the data (Shape:?)
            store_path (str):       The store path    
        Return:
            None
        ''' 
        # Init call for heif
        register_heif_opener()

        single_frame = np_frames

        # Prepare
        essential_name = store_path.split('.')[0]

        # Choose the quality
        quality = random.randint(*opt['heif_quality_range1'])
        method = random.randint(*opt['heif_encode_speed1'])

        # Transform to PIL and then compress
        PIL_image = Image.fromarray(np.uint8(single_frame[...,::-1])).convert('RGB')
        PIL_image.save(essential_name+'.heic', quality=quality, method=method)

        # Transform as png format
        heif_file = pillow_heif.open_heif(essential_name+'.heic', convert_hdr_to_8bit=False, bgr_mode=True)
        np_array = np.asarray(heif_file)
        cv2.imwrite(store_path, np_array)

        os.remove(essential_name+'.heic')


    @staticmethod
    def compress_tensor(tensor_frames, idx=0):
        ''' Compress tensor input to HEIF and then return it
        Args:
            tensor_frame (tensor):  Tensor inputs    
        Returns:
            result (tensor):        Tensor outputs (same shape as input)
        '''

        # Init call for heif
        register_heif_opener()

        # Prepare
        single_frame = tensor2np(tensor_frames)
        essential_name = "tmp/temp_"+str(idx)

        # Choose the quality
        quality = random.randint(*opt['heif_quality_range1'])
        method = random.randint(*opt['heif_encode_speed1'])

        # Transform to PIL and then compress
        PIL_image = Image.fromarray(np.uint8(single_frame[...,::-1])).convert('RGB')
        PIL_image.save(essential_name+'.heic', quality=quality, method=method)

        # Transform as png format
        heif_file = pillow_heif.open_heif(essential_name+'.heic', convert_hdr_to_8bit=False, bgr_mode=True)
        decimg = np.asarray(heif_file)
        os.remove(essential_name+'.heic')

        # Read back
        result = np2tensor(decimg)
        
        return result
            

