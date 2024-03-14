import os, sys
import torch

# Import files from same folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from architecture.rrdb import RRDBNet
from architecture.grl import GRL
from architecture.swinir import SwinIR
from architecture.cunet import UNet_Full


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
    if 'params_ema' in checkpoint_g:
        # For official ESRNET/ESRGAN weight
        weight = checkpoint_g['params_ema']
        generator = RRDBNet(3, 3, scale=scale)    # Default blocks num is 6     

    elif 'params' in checkpoint_g:
        # For official ESRNET/ESRGAN weight
        weight = checkpoint_g['params']
        generator = RRDBNet(3, 3, scale=scale)          

    elif 'model_state_dict' in checkpoint_g:
        # For my personal trained weight
        weight = checkpoint_g['model_state_dict']
        generator = RRDBNet(3, 3, scale=scale)          

    else:
        print("This weight is not supported")
        os._exit(0)


    # Handle torch.compile weight key rename
    old_keys = [key for key in weight]
    for old_key in old_keys:
        if old_key[:10] == "_orig_mod.":
            new_key = old_key[10:]
            weight[new_key] = weight[old_key]
            del weight[old_key]

    generator.load_state_dict(weight)
    generator = generator.eval().cuda()


    # Print options to show what kinds of setting is used
    if print_options:
        if 'opt' in checkpoint_g:
            for key in checkpoint_g['opt']:
                value = checkpoint_g['opt'][key]
                print(f'{key} : {value}')

    return generator


def load_cunet(generator_weight_PATH, scale, print_options=False):
    ''' A simpler API to load CUNET model from Real-CUGAN
    Args:
        generator_weight_PATH (str): The path to the weight
        scale (int): the scaling factor
        print_options (bool): whether to print options to show what kinds of setting is used
    Returns:
        generator (torch): the generator instance of the model
    '''  
    # This func is deprecated now
    
    if scale != 2:
        raise NotImplementedError("We only support 2x in CUNET")

    # Load the checkpoint
    checkpoint_g = torch.load(generator_weight_PATH)

    # Find the generator weight
    if 'model_state_dict' in checkpoint_g:
        # For my personal trained weight
        weight = checkpoint_g['model_state_dict']
        loss = checkpoint_g["lowest_generator_weight"]
        if "iteration" in checkpoint_g:
            iteration = checkpoint_g["iteration"]
        else:
            iteration = "NAN"
        generator = UNet_Full()          
        # generator = torch.compile(generator)# torch.compile
        print(f"the generator weight is {loss} at iteration {iteration}")

    else:
        print("This weight is not supported")
        os._exit(0)


    # Handle torch.compile weight key rename
    old_keys = [key for key in weight]
    for old_key in old_keys:
        if old_key[:10] == "_orig_mod.":
            new_key = old_key[10:]
            weight[new_key] = weight[old_key]
            del weight[old_key]

    generator.load_state_dict(weight)
    generator = generator.eval().cuda()


    # Print options to show what kinds of setting is used
    if print_options:
        if 'opt' in checkpoint_g:
            for key in checkpoint_g['opt']:
                value = checkpoint_g['opt'][key]
                print(f'{key} : {value}')

    return generator

def load_grl(generator_weight_PATH, scale=4):
    ''' A simpler API to load GRL model
    Args:
        generator_weight_PATH (str): The path to the weight
        scale (int):        Scale Factor (Usually Set as 4)
    Returns:
        generator (torch): the generator instance of the model
    '''

    # Load the checkpoint
    checkpoint_g = torch.load(generator_weight_PATH)

     # Find the generator weight
    if 'model_state_dict' in checkpoint_g:
        weight = checkpoint_g['model_state_dict']

        # GRL tiny model (Note: tiny2 version)
        generator = GRL(
            upscale = scale,
            img_size = 64,
            window_size = 8,
            depths = [4, 4, 4, 4],
            embed_dim = 64,
            num_heads_window = [2, 2, 2, 2],
            num_heads_stripe = [2, 2, 2, 2],
            mlp_ratio = 2,
            qkv_proj_type = "linear",
            anchor_proj_type = "avgpool",
            anchor_window_down_factor = 2,
            out_proj_type = "linear",
            conv_type = "1conv",
            upsampler = "nearest+conv",     # Change
        ).cuda()

    else:
        print("This weight is not supported")
        os._exit(0)


    generator.load_state_dict(weight)
    generator = generator.eval().cuda()


    num_params = 0
    for p in generator.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"Number of parameters {num_params / 10 ** 6: 0.2f}")


    return generator
