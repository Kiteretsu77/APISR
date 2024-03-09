'''
    This is file is to execute the inference for a single image or a folder input
'''
import argparse
import os, sys, cv2, shutil, json, warnings, collections, time
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
warnings.simplefilter("default")
os.environ["PYTHONWARNINGS"] = "default"


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from test_code.test_utils import load_grl, load_rrdb, load_cunet


@torch.no_grad      # You must add these time, else it will have Out of Memory
def super_resolve_img(generator, input_path, output_path, crop_for_4x = False):
    ''' Super Resolve a low resolution image
    Args:
        generator (torch):              the generator class that is already loaded
        input_path (str):               the path to the input lr images
        output_path (str):              the directory to store the generated images
        crop_for_4x (bool):             whether we crop the lr images to match 4x scale (needed for some situation)
    '''
    print("Processing image {}".format(input_path))
    
    # Read the image and do preprocess
    img_lr = cv2.imread(input_path)
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
    
    
    # Model inference
    print("lr shape is ", img_lr.shape)
    super_resolved_img = generator(img_lr)

    # Store the generated result
    save_image(super_resolved_img, output_path)

    # Empty the cache everytime you finish processing one image
    torch.cuda.empty_cache() 




if __name__ == "__main__":
    # Fundamental setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type = str, default='__assets__/lr_inputs', help="Can be either single image input or a folder input")
    parser.add_argument('--model', type = str, default='GRL', help=" 'GRL' || 'RRDB' (for ESRNet & ESRGAN) || 'CUNET' (for Real-ESRGAN) ")
    parser.add_argument('--scale', type = int, default=4, help="Up scaler factor")
    parser.add_argument('--weight_path', type = str, default='pretrained/4x_APISR_GRL_GAN_generator.pth', help="Weight path directory, usually uner saved_models folder")
    parser.add_argument('--store_dir', type = str, default='sample_outputs', help="The folder to store the super-resolved images")
    args  = parser.parse_args()
    
    # Some Command
    # python test_code/inference.py  --model RRDB --scale 2 --weight_path pretrained/2x_RRDB_GAN_generator.pth

    # Read argument and prepare the folder needed
    input_dir = args.input_dir
    model = args.model
    weight_path = args.weight_path
    store_dir = args.store_dir
    scale = args.scale
    

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir)


    # Load the model
    if model == "GRL":
        generator = load_grl(weight_path, scale=scale)  # GRL for Real-World SR only support 4x upscaling
    elif model == "RRDB":
        generator = load_rrdb(weight_path, scale=scale)  # Can be any size


    # Iterate the input
    if os.path.isdir(store_dir):
        for filename in sorted(os.listdir(input_dir)):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(store_dir, filename)

            super_resolve_img(generator, input_path, output_path)
    else:
        filename = os.path.split(input_dir)[-1].split('.')[0]
        output_path = os.path.join(store_dir, filename+"_4x.png")
        super_resolve_img(generator, input_dir, output_path)

    







        