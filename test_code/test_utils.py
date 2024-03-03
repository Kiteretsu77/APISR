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
