# -*- coding: utf-8 -*-
import sys
import os
import torch


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from architecture.dat import DAT               
from train_code.train_master import train_master



# Mixed precision training
scaler = torch.cuda.amp.GradScaler()


class train_dat(train_master):
    def __init__(self, options, args) -> None:
        super().__init__(options, args, "dat")      # Pass a model name unique code 


    def loss_init(self):
        # Prepare pixel loss
        self.pixel_loss_load()
        

    def call_model(self):

        # Generator: DAT light
        if opt['model_size'] == "light":
            # DAT light model 762K param
            self.generator = DAT(
                upscale=4,
                in_chans=3,
                img_size=64,
                img_range=1.,
                depth=[18],
                embed_dim=60,
                num_heads=[6],
                expansion_factor=2,
                resi_connection='3conv',
                split_size=[8,32],
                upsampler='pixelshuffledirect',
            ).cuda()
        
        elif opt['model_size'] == "small":
            # DAT small model 11.21M param
            self.generator = DAT(
                upscale=4,
                in_chans=3,
                img_size=64,
                img_range=1.,
                depth=[6,6,6,6,6,6],
                embed_dim=180,
                num_heads=[6,6,6,6,6,6],
                expansion_factor=2,
                resi_connection='1conv',
                split_size=[8,16],
                upsampler='pixelshuffledirect',
            ).cuda()

        else:
            raise NotImplementedError("We don't support such model size in DAT model")
        
        self.generator.train()

    
    def run(self):
        self.master_run()
                        
    
    def calculate_loss(self, gen_hr, imgs_hr):
        # Define the loss function here

        # Generator pixel loss (l1 loss):  generated vs. GT
        l_g_pix = self.cri_pix(gen_hr, imgs_hr, self.batch_idx)
        self.weight_store["pixel_loss"] = l_g_pix
        self.generator_loss += l_g_pix


    def tensorboard_report(self, iteration):
        # self.writer.add_scalar('Loss/train-Generator_Loss-Iteration', self.generator_loss, iteration)
        self.writer.add_scalar('Loss/train-Pixel_Loss-Iteration', self.weight_store["pixel_loss"], iteration)
