import torch, sys, os, random
import torch.nn.functional as F
import numpy as np
import cv2
from multiprocessing import Process, Queue
from PIL import Image
import pillow_heif

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder
from opt import opt
from degradation.ESR.utils import tensor2np, np2tensor



class AVIF():
    def __init__(self) -> None:
        # Choose an image compression degradation
        pass

    def compress_and_store(self, np_frames, store_path, idx):
        ''' Compress and Store the whole batch as AVIF (~ AV1)
        Args:
            np_frames (numpy):      The numpy format of the data (Shape:?)
            store_path (str):       The store path   
        Return:
            None
        '''
        # Init call for avif
        pillow_heif.register_avif_opener()


        single_frame = np_frames

        # Prepare
        essential_name = "tmp/temp_"+str(idx)

        # Choose the quality
        quality = random.randint(*opt['avif_quality_range2'])
        method = random.randint(*opt['avif_encode_speed2'])

        # Transform to PIL and then compress
        PIL_image = Image.fromarray(np.uint8(single_frame[...,::-1])).convert('RGB')
        PIL_image.save(essential_name+'.avif', quality=quality, method=method)

        # Read as png
        avif_file = pillow_heif.open_heif(essential_name+'.avif', convert_hdr_to_8bit=False, bgr_mode=True)
        np_array = np.asarray(avif_file)
        cv2.imwrite(store_path, np_array)

        os.remove(essential_name+'.avif')



    @staticmethod
    def compress_tensor(tensor_frames, idx=0):
        ''' Compress tensor input to AVIF and then return it
        Args:
            tensor_frame (tensor):  Tensor inputs    
        Returns:
            result (tensor):        Tensor outputs (same shape as input)
        '''
        # Init call for avif
        pillow_heif.register_avif_opener()

        # Prepare
        single_frame = tensor2np(tensor_frames)
        essential_name = "tmp/temp_"+str(idx)

        # Choose the quality
        quality = random.randint(*opt['avif_quality_range1'])
        method = random.randint(*opt['avif_encode_speed1'])

        # Transform to PIL and then compress
        PIL_image = Image.fromarray(np.uint8(single_frame[...,::-1])).convert('RGB')
        PIL_image.save(essential_name+'.avif', quality=quality, method=method)

        # Transform as png format
        avif_file = pillow_heif.open_heif(essential_name+'.avif', convert_hdr_to_8bit=False, bgr_mode=True)
        decimg = np.asarray(avif_file)
        os.remove(essential_name+'.avif')

        # Read back
        result = np2tensor(decimg)


        return result