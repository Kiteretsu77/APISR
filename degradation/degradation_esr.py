# -*- coding: utf-8 -*-
import torch
import os
import sys
import torch.nn.functional as F

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder
from opt import opt
from degradation.ESR.utils import generate_kernels, mass_tensor2np, tensor2np
from degradation.ESR.degradations_functionality import *
from degradation.ESR.degradation_esr_shared import common_degradation as regular_common_degradation
from degradation.image_compression.jpeg import JPEG   
from degradation.image_compression.webp import WEBP
from degradation.image_compression.heif import HEIF
from degradation.image_compression.avif import AVIF
from degradation.video_compression.h264 import H264
from degradation.video_compression.h265 import H265
from degradation.video_compression.mpeg2 import MPEG2
from degradation.video_compression.mpeg4 import MPEG4


class degradation_v1:
    def __init__(self):
        self.kernel1, self.kernel2, self.sinc_kernel = None, None, None
        self.queue_size = 160

        # Init the compression instance
        self.jpeg_instance = JPEG()
        self.webp_instance = WEBP()
        # self.heif_instance = HEIF()
        self.avif_instance = AVIF()
        self.H264_instance = H264()
        self.H265_instance = H265()
        self.MPEG2_instance = MPEG2()
        self.MPEG4_instance = MPEG4()


    def reset_kernels(self, opt):
        kernel1, kernel2 = generate_kernels(opt)
        self.kernel1 = kernel1.unsqueeze(0).cuda()
        self.kernel2 = kernel2.unsqueeze(0).cuda()
        

    @torch.no_grad()
    def degradate_process(self, out, opt, store_path, process_id, verbose = False):
        ''' ESR Degradation V1 mode (Same as the original paper)
        Args:
            out (tensor):           BxCxHxW All input images as tensor
            opt (dict):             All configuration we need to process 
            store_path (str):       Store Directory
            process_id (int):       The id we used to store temporary file
            verbose (bool):         Whether print some information for auxiliary log (default: False)
        '''

        batch_size, _, ori_h, ori_w = out.size()

        # Shared degradation until the last step
        resize_mode = random.choice(opt['resize_options'])
        out = regular_common_degradation(out, opt, [self.kernel1, self.kernel2], process_id, verbose=verbose)


        # Resize back
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode = resize_mode)
        out = torch.clamp(out, 0, 1)

        # Tensor2np
        np_frame = tensor2np(out)

        # Choose an image compression codec (All degradation batch use the same codec)
        compression_codec = random.choices(opt['compression_codec2'], opt['compression_codec_prob2'])[0]     # All lower case
        
        if compression_codec == "jpeg":
            self.jpeg_instance.compress_and_store(np_frame, store_path, process_id)
        
        elif compression_codec == "webp":
            try:
                self.webp_instance.compress_and_store(np_frame, store_path, process_id)
            except Exception:
                print("There appears to be exception in webp again!")
                if os.path.exists(store_path):
                    os.remove(store_path)
                self.webp_instance.compress_and_store(np_frame, store_path, process_id)
        
        elif compression_codec == "avif":
            self.avif_instance.compress_and_store(np_frame, store_path, process_id)

        elif compression_codec == "h264":
            self.H264_instance.compress_and_store(np_frame, store_path, process_id)

        elif compression_codec == "h265":
            self.H265_instance.compress_and_store(np_frame, store_path, process_id)

        elif compression_codec == "mpeg2":
            self.MPEG2_instance.compress_and_store(np_frame, store_path, process_id)

        elif compression_codec == "mpeg4":
            self.MPEG4_instance.compress_and_store(np_frame, store_path, process_id)

        else:
            raise NotImplementedError("This compression codec is not supported! Please check the implementation!")


             
        



