'''
    This is file is to execute the inference for a single image or a folder input
'''
import argparse
import time
import numpy as np
import os, sys, cv2, shutil, warnings
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import ffmpegcv
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.editor import VideoFileClip
warnings.simplefilter("default")
os.environ["PYTHONWARNINGS"] = "default"


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from test_code.test_utils import load_grl, load_rrdb, load_dat, load_cunet



@torch.no_grad      # You must add these time, else it will have Out of Memory
def super_resolve_img(generator, input_path, output_path=None, weight_dtype=torch.float32, downsample_threshold=-1, crop_for_4x=True):
    ''' Super Resolve a low resolution image
    Args:
        generator (torch):              the generator class that is already loaded
        input_path (str):               the path to the input lr images
        output_path (str):              the directory to store the generated images
        weight_dtype (bool):            the weight type (float32/float16)
        downsample_threshold (int):     the threshold of height/width (short side) to downsample the input
        crop_for_4x (bool):             whether we crop the lr images to match 4x scale (needed for some situation)
    '''
    print("Processing image {}".format(input_path))
    
    # Read the image and do preprocess
    img_lr = cv2.imread(input_path)
    h, w, c = img_lr.shape


    # Downsample if needed
    short_side = min(h, w)
    if downsample_threshold != -1 and short_side > downsample_threshold:
        resize_ratio = short_side / downsample_threshold
        img_lr = cv2.resize(img_lr, (int(w/resize_ratio), int(h/resize_ratio)), interpolation = cv2.INTER_LINEAR)


    # Crop if needed
    if crop_for_4x:
        h, w, _ = img_lr.shape
        if h % 4 != 0:
            img_lr = img_lr[:4*(h//4),:,:]
        if w % 4 != 0:
            img_lr = img_lr[:,:4*(w//4),:]


    # Transform to tensor
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    img_lr = ToTensor()(img_lr).unsqueeze(0).cuda()     # Use tensor format
    img_lr = img_lr.to(dtype=weight_dtype)
    
    
    # Model inference
    # print("lr shape is ", img_lr.shape)
    super_resolved_img = generator(img_lr)

    # Store the generated result
    with torch.cuda.amp.autocast():
        if output_path is not None:
            save_image(super_resolved_img, output_path)

    # Empty the cache everytime you finish processing one image
    torch.cuda.empty_cache() 
    
    return super_resolved_img



@torch.no_grad
def super_resolve_video(generator, input_path, output_path, scale, weight_dtype=torch.float32, downsample_threshold=-1, crop_for_4x=True):

    # Default setting 
    encode_params = ['-crf', '32', '-preset', 'medium']   # This is one of the best setting I used to use
    
    # Read the video path
    objVideoReader = VideoFileClip(filename=input_path)

    
    # Obtain basic video information
    width, height = objVideoReader.reader.size
    original_fps = objVideoReader.reader.fps
    nframes = objVideoReader.reader.nframes
    has_audio = objVideoReader.audio


    # Handle the rescale
    short_side = min(height, width)
    if downsample_threshold != -1 and short_side > downsample_threshold:
        rescale_factor = short_side / downsample_threshold
    else:
        rescale_factor = 1
    

    # Create a tmp file
    temp_file_name = "inference_tmp"
    if os.path.exists(temp_file_name):
        shutil.rmtree(temp_file_name)
    os.makedirs(temp_file_name)
    if os.path.exists(output_path):
        os.remove(output_path)
    

    
    # Create a video writer
    output_size = (int(width * scale / rescale_factor), int(height * scale / rescale_factor))
    if has_audio:
        objVideoReader.audio.write_audiofile(temp_file_name+"/output_audio.mp3")    # Hopefully, mp3 format is supported for all input video 
        writer = FFMPEG_VideoWriter(output_path, output_size, original_fps, ffmpeg_params=encode_params, audiofile=temp_file_name+"/output_audio.mp3")
    else:
        writer = FFMPEG_VideoWriter(output_path, size=output_size, fps=original_fps, ffmpeg_params=encode_params)
    
    
    # Setup Progress bar
    progress_bar = tqdm(range(0, nframes), initial=0, desc="Frame",)
    
    
    # Iterate frames from the video and super-resolve individually
    for frame_idx, img_lr in enumerate(objVideoReader.iter_frames(fps=original_fps)):
        
        # Downsample if needed
        if rescale_factor != 1:
            img_lr = cv2.resize(img_lr, (int(width/rescale_factor), int(height/rescale_factor)), interpolation = cv2.INTER_LINEAR)

        # Crop if needed
        if crop_for_4x:
            h, w, _ = img_lr.shape
            if h % 4 != 0:
                img_lr = img_lr[:4*(h//4),:,:]
            if w % 4 != 0:
                img_lr = img_lr[:,:4*(w//4),:]


        # Transform to tensor
        # img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        img_lr = ToTensor()(img_lr).unsqueeze(0).cuda()     # Use tensor format
        img_lr = img_lr.to(dtype=weight_dtype)
        
        
        # Model inference
        super_resolved_img = generator(img_lr)

        # Post process
        super_resolved_img = np.uint8(np.clip(torch.permute(super_resolved_img[0]*255.0, (1, 2, 0)).cpu().detach().numpy(), 0, 255))
        # cv2.imwrite("sr_"+str(frame_idx)+".png", super_resolved_img)


        # Write into the frame
        writer.write_frame(super_resolved_img)
        
        progress_bar.update(1)

    writer.close()


    # Clean the temp file
    shutil.rmtree(temp_file_name)




if __name__ == "__main__":
    
    # Fundamental setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type = str, default = '__assets__/lr_inputs', help="Can be either single image input or a folder input")
    parser.add_argument('--scale', type = int, default = 4, help="Upscaler factor")
    parser.add_argument('--store_dir', type = str, default = 'sample_outputs', help="The folder to store the super-resolved images")
    parser.add_argument('--model', type = str, default = 'GRL', help=" 'GRL' || 'RRDB' (for ESRNET & ESRGAN) || 'DAT' || 'CUNET' (for Real-ESRGAN) ")
    parser.add_argument('--weight_path', type = str, default = 'pretrained/4x_APISR_GRL_GAN_generator.pth', help="Weight path directory, usually under saved_models folder")
    parser.add_argument('--downsample_threshold', type = int, default = -1, help="Downsample with same aspect ratio if the height/width (short side) is over the threshold limit, recommend to set as 720")
    parser.add_argument('--float16_inference', type = bool, default = False, help="The folder to store the super-resolved images")      # Currently, this is only supported in RRDB, there is some bug with GRL model
    args = parser.parse_args()
    

    # Sample Command:
    # 4x GRL (Default):     python test_code/inference.py --model GRL --scale 4 --downsample_threshold 1080 --weight_path pretrained/4x_APISR_GRL_GAN_generator.pth
    # 4X DAT:               python test_code/inference.py --model DAT --scale 4 --downsample_threshold 720 --weight_path pretrained/4x_APISR_DAT_GAN_generator.pth
    # 4x RRDB:              python test_code/inference.py --model RRDB --scale 4 --downsample_threshold 1080 --weight_path pretrained/4x_APISR_RRDB_GAN_generator.pth
    # 2x RRDB:              python test_code/inference.py --model RRDB --scale 2 --downsample_threshold 1080 --weight_path pretrained/2x_APISR_RRDB_GAN_generator.pth
    


    # Read argument and prepare the folder needed
    input_dir = args.input_dir
    model = args.model
    weight_path = args.weight_path
    store_dir = args.store_dir
    scale = args.scale
    downsample_threshold = args.downsample_threshold
    float16_inference = args.float16_inference

    # Some other setting
    supported_img_extension = ['jpg', 'png', 'webp', 'jpeg']
    supported_video_extension = ['mp4', 'mkv']
    
    
    # Check the path of the weight
    if not os.path.exists(weight_path):
        print("we cannot locate weight path ", weight_path) 
        # TODO: I am not sure if I should automatically download weight from github release based on the upscale factor and model name.
        os._exit(0)
    
    
    # Prepare the store folder
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir)



    # Define the weight type
    if float16_inference:
        torch.backends.cudnn.benchmark = True
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
        

    # Load the model
    if model == "GRL":
        generator = load_grl(weight_path, scale=scale)  # GRL for Real-World SR only support 4x upscaling

    elif model == "DAT":
        generator = load_dat(weight_path, scale=scale)  # GRL for Real-World SR only support 4x upscaling

    elif model == "RRDB":
        generator = load_rrdb(weight_path, scale=scale)  # Can be any size
        
    generator = generator.to(dtype=weight_dtype)


    # Should have a for loop here
    def inner_loop(process_dir):
        # First, check whether this single file is image or video
        filename = os.path.split(process_dir)[-1].split('.')[0]     # Extract the code name if the file length is too long.
        input_extension = process_dir.split('.')[-1]

        if input_extension in supported_img_extension: # If the input path is single image
            output_path = os.path.join(store_dir, filename+"_"+str(scale)+"x.png")      # Output fixed to be png
            # In default, we will automatically use crop to match 4x size
            super_resolve_img(generator, process_dir, output_path, weight_dtype, downsample_threshold, crop_for_4x=True)
        
        elif input_extension in supported_video_extension: # If the input path is single video
            output_path = os.path.join(store_dir, filename+"_"+str(scale)+"x.mp4")      # Output fixed to be mp4
            super_resolve_video(generator, process_dir, output_path, scale, weight_dtype, downsample_threshold, crop_for_4x=True)

        else:
            raise NotImplementedError("This single file input format is not what we support!")




    start = time.time()

    # Take the input path and do inference
    if os.path.isdir(input_dir):        # If the input is a directory, we will iterate it
        for filename in sorted(os.listdir(input_dir)):
            inner_loop(os.path.join(input_dir, filename))
            
    else:   # If the input is a single file (img/video), we will process it directly and write on the same folder
        inner_loop(input_dir)
        
        
    end = time.time()
    print("Total inference time spent is ", end-start)

    







        