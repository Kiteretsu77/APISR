import sys, os, random
import cv2, torch
from multiprocessing import Process, Queue

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder
from opt import opt
from degradation.ESR.utils import tensor2np, np2tensor



class JPEG():
    def __init__(self) -> None:
        # Choose an image compression degradation
        pass

    def compress_and_store(self, np_frames, store_path, idx):
        ''' Compress and Store the whole batch as JPEG
        Args:
            np_frames (numpy):      The numpy format of the data (Shape:?)
            store_path (str):       The store path    
        Return:
            None
        '''

        # Preparation
        single_frame = np_frames

        # Compress as JPEG
        jpeg_quality = random.randint(*opt['jpeg_quality_range2'])

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode('.jpg', single_frame, encode_param)       
        decimg = cv2.imdecode(encimg, 1)

        # Store the image with quality
        cv2.imwrite(store_path, decimg)  
    
    

    @staticmethod
    def compress_tensor(tensor_frames):
        ''' Compress tensor input to JPEG and then return it
        Args:
            tensor_frame (tensor):  Tensor inputs   
        Returns:
            result (tensor):        Tensor outputs (same shape as input) 
        '''

        single_frame = tensor2np(tensor_frames)

        # Compress as JPEG
        jpeg_quality = random.randint(*opt['jpeg_quality_range1'])

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode('.jpg', single_frame, encode_param)       
        decimg = cv2.imdecode(encimg, 1)

        # Store the image with quality
        # cv2.imwrite(store_name, decimg)  
        result = np2tensor(decimg)

        return result


            