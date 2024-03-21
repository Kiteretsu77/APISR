'''
    This is file is to execute the inference for a single image or a folder input
'''
import argparse
import os, sys, cv2, shutil, warnings
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
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
        img_lr = cv2.resize(img_lr, (int(h*resize_ratio), int(w*resize_ratio)), interpolation = cv2.INTER_LINEAR)


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
    print("lr shape is ", img_lr.shape)
    super_resolved_img = generator(img_lr)

    # Store the generated result
    with torch.cuda.amp.autocast():
        if output_path is not None:
            save_image(super_resolved_img, output_path)

    # Empty the cache everytime you finish processing one image
    torch.cuda.empty_cache() 
    
    return super_resolved_img




if __name__ == "__main__":
    
    # Fundamental setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type = str, default = '__assets__/lr_inputs', help="Can be either single image input or a folder input")
    parser.add_argument('--store_dir', type = str, default = 'sample_outputs', help="The folder to store the super-resolved images")
    parser.add_argument('--model', type = str, default = 'GRL', help=" 'GRL' || 'RRDB' (for ESRNET & ESRGAN) || 'DAT' || 'CUNET' (for Real-ESRGAN) ")
    parser.add_argument('--scale', type = int, default = 4, help="Up scaler factor")
    parser.add_argument('--weight_path', type = str, default = 'pretrained/4x_APISR_GRL_GAN_generator.pth', help="Weight path directory, usually under saved_models folder")
    parser.add_argument('--downsample_threshold', type = int, default = -1, help="Downsample with same aspect ratio if the height/width (short side) is over the threshold limit, recomemnd to set as 720")
    parser.add_argument('--float16_inference', type = bool, default = False, help="The folder to store the super-resolved images")      # Currently, this is only supported in RRDB, there is some bug with GRL model
    args = parser.parse_args()
    
    # Sample Command:
    # 4X DAT:               python test_code/inference.py --model DAT --scale 4 --downsample_threshold 1080 --weight_path saved_models/dat_best_generator.pth
    # 4x GRL (Default):     python test_code/inference.py --model GRL --scale 4 --downsample_threshold 1080 --weight_path pretrained/4x_APISR_GRL_GAN_generator.pth
    # 2x RRDB:              python test_code/inference.py --model RRDB --scale 2 --downsample_threshold 1080 --weight_path pretrained/2x_APISR_RRDB_GAN_generator.pth


    # Read argument and prepare the folder needed
    input_dir = args.input_dir
    model = args.model
    weight_path = args.weight_path
    store_dir = args.store_dir
    scale = args.scale
    downsample_threshold = args.downsample_threshold
    float16_inference = args.float16_inference
    
    
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
    

    # Take the input path and do inference
    if os.path.isdir(store_dir):    # If the input is a directory, we will iterate it
        for filename in sorted(os.listdir(input_dir)):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(store_dir, "".join(filename.split('.')[:-1])+".png")
            # In default, we will automatically use crop to match 4x size
            super_resolve_img(generator, input_path, output_path, weight_dtype, downsample_threshold, crop_for_4x=True)
            
    else:   # If the input is a single image, we will process it directly and write on the same folder
        filename = os.path.split(input_dir)[-1].split('.')[0]
        output_path = os.path.join(store_dir, filename+"_"+str(scale)+"x.png")
        # In default, we will automatically use crop to match 4x size
        super_resolve_img(generator, input_dir, output_path, weight_dtype, downsample_threshold, crop_for_4x=True)

    







        