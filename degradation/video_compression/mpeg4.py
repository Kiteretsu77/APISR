import torch, sys, os, random
import cv2
import shutil

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder
from opt import opt



class MPEG4():
    def __init__(self) -> None:
        # Choose an image compression degradation
        pass

    def compress_and_store(self, single_frame, store_path, idx):
        ''' Compress and Store the whole batch as MPEG-4 (for 2nd stage)
        Args:
            single_frame (numpy):      The numpy format of the data (Shape:?)
            store_path (str):       The store path   
            idx (int):              A unique process idx
        Return:
            None
        '''

        # Prepare
        temp_input_path = "tmp/input_"+str(idx)
        video_store_dir = "tmp/encoded_"+str(idx)+".mp4"
        temp_store_path = "tmp/output_"+str(idx)
        os.makedirs(temp_input_path)
        os.makedirs(temp_store_path)
        
        # Move frame 
        cv2.imwrite(os.path.join(temp_input_path, "1.png"), single_frame)


        # Decide the quality
        quality = str(random.randint(*opt['mpeg4_quality2']))
        preset = random.choices(opt['mpeg4_preset_mode2'], opt['mpeg4_preset_prob2'])[0]

        # Encode
        ffmpeg_encode_cmd = "ffmpeg -i " + temp_input_path + "/%d.png -vcodec libxvid -qscale:v " + quality + " -preset " + preset + " -pix_fmt yuv420p " + video_store_dir + " -loglevel 0"
        os.system(ffmpeg_encode_cmd)
        

        # Decode
        ffmpeg_decode_cmd = "ffmpeg -i " + video_store_dir + " " + temp_store_path + "/%d.png -loglevel 0"
        os.system(ffmpeg_decode_cmd)
        assert(len(os.listdir(temp_store_path)) == 1)

        # Move frame to the target places
        shutil.copy(os.path.join(temp_store_path, "1.png"), store_path)

        # Clean temp files
        os.remove(video_store_dir)
        shutil.rmtree(temp_input_path)
        shutil.rmtree(temp_store_path)



    @staticmethod
    def compress_tensor(tensor_frames, idx=0):
        ''' Compress tensor input to MPEG4 and then return it (for 1st stage)
        Args:
            tensor_frame (tensor):  Tensor inputs    
        Returns:
            result (tensor):        Tensor outputs (same shape as input)
        '''

        pass