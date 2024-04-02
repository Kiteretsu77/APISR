'''
    This file is to modify model architecture namign into the pth file based on the pth file you download from the github release. 
    This is for the convenience of ncnn and other deployment.
'''

import os, sys
import argparse
import torch

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from architecture.rrdb import RRDBNet



def load_rrdb(generator_weight_PATH, scale, print_options=False):  
    ''' A simpler API to load RRDB model from Real-ESRGAN
    Args:
        generator_weight_PATH (str): The path to the weight
        scale (int): the scaling factor
        print_options (bool): whether to print options to show what kinds of setting is used
    Returns:
        generator (torch): the generator instance of the model
    '''  

    # Load the checkpoint
    checkpoint_g = torch.load(generator_weight_PATH)

    # Find the generator weight
    if 'model_state_dict' in checkpoint_g:
        # For my personal trained weight
        weight = checkpoint_g['model_state_dict']
        generator = RRDBNet(3, 3, scale=scale)          

    else:
        print("This weight is not supported")
        os._exit(0)


    generator.load_state_dict(weight)
    generator = generator.eval().cuda()


    return generator


if __name__ == "__main__":
    
    # Fundamental setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type = str, default = '4x_APISR_RRDB_GAN_generator.pth', help = "This is the directory to the weight path")
    parser.add_argument('--architecture', type = str, default = 'RRDB', help = " 'GRL' || 'RRDB' (for ESRNET & ESRGAN) || 'CUNET' (for Real-ESRGAN) ")
    parser.add_argument('--scale', type = int, default = 4, help="Upscaler factor")
    parser.add_argument('--store_path', type = str, default = '4x_APISR_RRDB_GAN_generator_.pth', help = " Define the store path of the newly edited weights ")
    args = parser.parse_args()
    

    # Change
    weight_path = args.weight_path
    architecture = args.architecture
    scale = args.scale
    store_path = args.store_path

    
    # Load model
    if architecture == "RRDB":
        model = load_rrdb(weight_path, scale = scale)
    

    # Save the model with model architecture information
    torch.save({
                "params_ema": model.state_dict(),
                }, store_path)


