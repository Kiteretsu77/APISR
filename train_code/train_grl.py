# -*- coding: utf-8 -*-
import sys
import os
import torch


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from architecture.grl import GRL               # This place need to adjust for different models
from train_code.train_master import train_master



# Mixed precision training
scaler = torch.cuda.amp.GradScaler()


class train_grl(train_master):
    def __init__(self, options, args) -> None:
        super().__init__(options, args, "grl") # 这里要传递一个universal的model name


    def loss_init(self):
        # Prepare pixel loss
        self.pixel_loss_load()
        

    def call_model(self):
        patch_size = 144
        window_size = 8

        if opt['model_size'] == "small":
            # GRL small model 
            self.generator = GRL(
                upscale = opt['scale'],              
                img_size = patch_size,
                window_size = 8,
                depths = [4, 4, 4, 4],
                embed_dim = 128,
                num_heads_window = [2, 2, 2, 2],
                num_heads_stripe = [2, 2, 2, 2],
                mlp_ratio = 2,
                qkv_proj_type = "linear",
                anchor_proj_type = "avgpool",
                anchor_window_down_factor = 2,
                out_proj_type = "linear",
                conv_type = "1conv",
                upsampler = "pixelshuffle",
            ).cuda()

        elif opt['model_size'] == "tiny":
            # GRL tiny model
            self.generator = GRL(
                upscale = opt['scale'],
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
                upsampler = "pixelshuffledirect",
            ).cuda()


        elif opt['model_size'] == "tiny2":
            # GRL tiny model
            self.generator = GRL(
                upscale = opt['scale'],
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


        elif opt['model_size'] == "edit1":
            print("We are trining with edit1 version")
            self.generator = GRL(
                upscale = opt['scale'],
                img_size = 144,                 # Change
                window_size = 8,        
                depths = [4, 4, 4, 4],
                embed_dim = 80,                 # Change
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
            raise NotImplementedError("We don't support such model size in GRL model")
        
        # self.generator = torch.compile(self.generator).cuda()     # Don't use this for 3090Ti
        self.generator.train()

    
    def run(self):
        self.master_run()
                        
        # TODO: 这边还少了一个ema，论文说是为了better training and performance

    
    def calculate_loss(self, gen_hr, imgs_hr):
        # 这里就是各种自定义化需要的loss function

        # Generator pixel loss (l1 loss):  generated vs. GT
        l_g_pix = self.cri_pix(gen_hr, imgs_hr, self.batch_idx)
        self.weight_store["pixel_loss"] = l_g_pix
        self.generator_loss += l_g_pix


    def tensorboard_report(self, iteration):
        # self.writer.add_scalar('Loss/train-Generator_Loss-Iteration', self.generator_loss, iteration)
        self.writer.add_scalar('Loss/train-Pixel_Loss-Iteration', self.weight_store["pixel_loss"], iteration)
